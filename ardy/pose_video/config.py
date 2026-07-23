# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable fixed-camera and rendering-style configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _finite_tuple(name: str, value: Iterable[float], length: int) -> tuple[float, ...]:
    result = tuple(float(component) for component in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class CameraConfig:
    """A camera contract that remains fixed for every frame of an asset.

    Reuse the same serialized instance for every behavior that needs to cut
    seamlessly.  Width and height must be even because the exported MP4 uses
    yuv420p chroma subsampling.
    """

    width: int = 1080
    height: int = 1920
    eye: tuple[float, float, float] = (0.0, 0.56, 2.3)
    target: tuple[float, float, float] = (0.0, 0.54, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vertical_fov_degrees: float = 25.0
    near: float = 0.05
    far: float = 100.0

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise TypeError("camera width and height must be integers")
        if self.width < 16 or self.height < 16:
            raise ValueError("camera width and height must both be at least 16 pixels")
        if self.width % 2 or self.height % 2:
            raise ValueError("camera width and height must be even for H.264 yuv420p output")

        object.__setattr__(self, "eye", _finite_tuple("eye", self.eye, 3))
        object.__setattr__(self, "target", _finite_tuple("target", self.target, 3))
        object.__setattr__(self, "up", _finite_tuple("up", self.up, 3))

        if not 1.0 <= self.vertical_fov_degrees <= 120.0:
            raise ValueError("vertical_fov_degrees must be in [1, 120]")
        if not (math.isfinite(self.near) and math.isfinite(self.far) and 0.0 < self.near < self.far):
            raise ValueError("camera clipping planes must satisfy 0 < near < far")

        eye = np.asarray(self.eye)
        target = np.asarray(self.target)
        up = np.asarray(self.up)
        view = target - eye
        if np.linalg.norm(view) < 1e-8:
            raise ValueError("camera eye and target cannot be the same point")
        if np.linalg.norm(up) < 1e-8:
            raise ValueError("camera up vector cannot be zero")
        if np.linalg.norm(np.cross(view, up)) < 1e-8:
            raise ValueError("camera up vector cannot be parallel to the viewing direction")

    @classmethod
    def video_call_portrait(cls, width: int = 1080, height: int = 1920) -> "CameraConfig":
        """Waist-up portrait framing for the canonical Core avatar."""

        return cls(width=width, height=height)

    @classmethod
    def full_body_portrait(cls, width: int = 1080, height: int = 1920) -> "CameraConfig":
        """Fixed full-body portrait framing, useful while authoring motion."""

        return cls(
            width=width,
            height=height,
            eye=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            vertical_fov_degrees=28.0,
        )

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def with_resolution(self, width: int, height: int) -> "CameraConfig":
        return replace(self, width=width, height=height)

    def pose_matrix(self) -> np.ndarray:
        """Return the OpenGL camera-to-world transform.

        OpenGL cameras look along local ``-Z`` and use local ``+Y`` as up.
        """

        eye = np.asarray(self.eye, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        z_axis = eye - target
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        pose[:3, 3] = eye
        return pose

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("camera configuration must be a JSON object")
        return cls(**payload)


@dataclass(frozen=True)
class RenderStyle:
    """Stable material, background, and light settings for motion references."""

    background_rgba: tuple[float, float, float, float] = (0.945, 0.957, 0.976, 1.0)
    body_rgba: tuple[float, float, float, float] = (0.25, 0.61, 0.88, 1.0)
    ambient_light: float = 0.32
    key_light_intensity: float = 3.0
    fill_light_intensity: float = 0.75
    metallic: float = 0.0
    roughness: float = 0.72
    smooth_shading: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "background_rgba", _finite_tuple("background_rgba", self.background_rgba, 4))
        object.__setattr__(self, "body_rgba", _finite_tuple("body_rgba", self.body_rgba, 4))
        for name in ("background_rgba", "body_rgba"):
            if not all(0.0 <= value <= 1.0 for value in getattr(self, name)):
                raise ValueError(f"{name} values must be in [0, 1]")
        for name in ("ambient_light", "key_light_intensity", "fill_light_intensity", "metallic", "roughness"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.metallic > 1.0 or self.roughness > 1.0:
            raise ValueError("metallic and roughness must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
