# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned specifications for deterministic, camera-ready avatar motions.

The pose-video workflow intentionally keeps *what* a behavior should do separate
from both ARDY inference and rendering.  A :class:`BehaviorSpec` can therefore
be reviewed, diffed, regenerated, and rendered without relying on an imprecise
natural-language prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class JointTransform(StrictModel):
    """A local-space rotation delta and optional root translation.

    Euler angles are expressed in degrees in ARDY's local XYZ joint frame.  The
    values are deltas from the selected base pose, not absolute world angles.
    """

    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)


class PoseKeyframe(StrictModel):
    """Sparse joint targets at one point on the behavior timeline."""

    time_seconds: float = Field(ge=0.0)
    joints: dict[str, JointTransform]
    label: str = ""

    @field_validator("joints")
    @classmethod
    def require_joints(cls, value: dict[str, JointTransform]) -> dict[str, JointTransform]:
        if not value:
            raise ValueError("a pose keyframe must contain at least one joint")
        return value


class AmbientMotion(StrictModel):
    """Small deterministic motion layered underneath authored keyframes."""

    enabled: bool = True
    breathing_cycles: float = Field(default=2.0, gt=0.0)
    breathing_degrees: float = Field(default=0.8, ge=0.0, le=5.0)
    sway_cycles: float = Field(default=1.0, gt=0.0)
    sway_degrees: float = Field(default=0.65, ge=0.0, le=5.0)
    head_sway_degrees: float = Field(default=0.35, ge=0.0, le=3.0)


class BasePoseSpec(StrictModel):
    """How to obtain the neutral pose on which behavior deltas are applied."""

    mode: Literal["authored_neutral", "motion_frame"] = "authored_neutral"
    motion_path: str | None = None
    frame: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def require_motion_path(self) -> "BasePoseSpec":
        if self.mode == "motion_frame" and not self.motion_path:
            raise ValueError("base_pose.motion_path is required when mode='motion_frame'")
        return self


class FaceIntent(StrictModel):
    """Metadata for Kling/a future facial layer; ARDY does not consume this."""

    gaze: str = "camera"
    expression: str = "neutral"
    blink: str = "natural"
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)


class CameraSpec(StrictModel):
    """Fixed camera shared by every asset in a compatible library."""

    resolution: tuple[int, int] = (1080, 1920)
    render_resolution: tuple[int, int] = (540, 960)
    position: tuple[float, float, float] = (0.0, 0.56, 2.3)
    look_at: tuple[float, float, float] = (0.0, 0.54, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Keep the serialized behavior contract aligned with the renderer's
    # ``CameraConfig`` so a valid spec cannot fail only at render time.
    vertical_fov_degrees: float = Field(default=25.0, gt=1.0, le=120.0)
    background_rgb: tuple[int, int, int] = (238, 241, 246)
    body_rgb: tuple[int, int, int] = (76, 112, 176)

    @field_validator("resolution", "render_resolution")
    @classmethod
    def valid_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("camera resolutions must be positive")
        if value[0] % 2 or value[1] % 2:
            raise ValueError("camera resolutions must be even for H.264 yuv420p")
        return value

    @field_validator("background_rgb", "body_rgb")
    @classmethod
    def valid_rgb(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("RGB channels must be in [0, 255]")
        return value

    @model_validator(mode="after")
    def valid_view(self) -> "CameraSpec":
        view = tuple(target - eye for target, eye in zip(self.look_at, self.position))
        view_length = math.sqrt(sum(component * component for component in view))
        up_length = math.sqrt(sum(component * component for component in self.up))
        if view_length <= 1e-9:
            raise ValueError("camera position and look_at must be different")
        if up_length <= 1e-9:
            raise ValueError("camera up vector must be non-zero")
        cosine = sum(a * b for a, b in zip(view, self.up)) / (view_length * up_length)
        if abs(cosine) >= 1.0 - 1e-7:
            raise ValueError("camera up vector must not be parallel to its viewing direction")
        return self


class BehaviorSpec(StrictModel):
    """Complete deterministic definition of a reusable avatar behavior."""

    schema_version: Literal[1] = SCHEMA_VERSION
    behavior_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    behavior_type: Literal["loop", "one_shot"]
    speaking_mode: Literal["speaking", "listening", "either"] = "either"
    fps: int = Field(default=30, ge=1, le=120)
    duration_seconds: float = Field(default=10.0, gt=0.0, le=120.0)
    boundary_seconds: float = Field(default=2.0, gt=0.0)
    base_pose: BasePoseSpec = Field(default_factory=BasePoseSpec)
    locks: list[str] = Field(default_factory=lambda: ["Hips"])
    keyframes: list[PoseKeyframe] = Field(default_factory=list)
    ambient_motion: AmbientMotion = Field(default_factory=AmbientMotion)
    face_intent: FaceIntent = Field(default_factory=FaceIntent)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    source_reference: str | None = None

    @field_validator("locks")
    @classmethod
    def unique_locks(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("locked joints must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_timeline(self) -> "BehaviorSpec":
        if 2 * self.boundary_seconds >= self.duration_seconds:
            raise ValueError("two boundary blocks must leave a non-empty behavior interior")
        if self.boundary_frames < 2:
            raise ValueError("boundary_seconds must span at least two frames")
        if 2 * self.boundary_frames >= self.num_frames:
            raise ValueError("rounded boundary blocks must leave at least one interior frame")
        previous = -1.0
        for keyframe in self.keyframes:
            if keyframe.time_seconds <= previous:
                raise ValueError("keyframes must be strictly ordered by time_seconds")
            if keyframe.time_seconds >= self.duration_seconds:
                raise ValueError("keyframe time must be before duration_seconds")
            if not self.boundary_seconds < keyframe.time_seconds < self.duration_seconds - self.boundary_seconds:
                raise ValueError("keyframes must lie strictly between the two canonical boundary blocks")
            previous = keyframe.time_seconds
        return self

    @property
    def num_frames(self) -> int:
        return int(round(self.duration_seconds * self.fps))

    @property
    def boundary_frames(self) -> int:
        return int(round(self.boundary_seconds * self.fps))

    def resolve_relative_paths(self, spec_path: str | Path) -> "BehaviorSpec":
        """Return a copy whose file references are absolute.

        Paths inside a pose specification are resolved relative to that JSON
        file, which makes behavior libraries portable as a directory.
        """

        base_dir = Path(spec_path).resolve().parent
        updates: dict[str, object] = {}
        if self.base_pose.motion_path and not Path(self.base_pose.motion_path).is_absolute():
            updates["base_pose"] = self.base_pose.model_copy(
                update={"motion_path": str((base_dir / self.base_pose.motion_path).resolve())}
            )
        if self.source_reference and not Path(self.source_reference).is_absolute():
            updates["source_reference"] = str((base_dir / self.source_reference).resolve())
        return self.model_copy(update=updates)


def load_behavior_spec(path: str | Path) -> BehaviorSpec:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BehaviorSpec.model_validate(payload).resolve_relative_paths(path)


def save_behavior_spec(spec: BehaviorSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.model_dump(mode="json"), handle, indent=2)
        handle.write("\n")


def behavior_spec_fingerprint(spec: BehaviorSpec) -> str:
    """Return a canonical SHA-256 provenance digest for an effective spec.

    Resolve relative file references before calling this helper. JSON key order
    and formatting do not influence the digest, while every versioned spec
    field does.
    """

    canonical = json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
