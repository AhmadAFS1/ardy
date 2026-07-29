# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EGL-backed offscreen renderer for the canonical Core skin."""

from __future__ import annotations

from collections.abc import Iterator
import os
import sys
from typing import Any

import numpy as np
import torch

from ardy.skeleton import CoreSkeleton27
from ardy.viz.core_skin import (
    CoreSkin,
    MATERIAL_BODY,
    MATERIAL_BROW,
    MATERIAL_EYE_BACKING,
    MATERIAL_IRIS,
    MATERIAL_MOUTH,
    MATERIAL_PUPIL,
    MATERIAL_SCLERA,
    MATERIAL_SKIN,
)

from .config import CameraConfig, RenderStyle
from .motion import CoreMotion

BLINK_PROTECTED_SECONDS = 2.0
# Three irregularly spaced events avoid a metronomic loop. The middle blink
# reopens one frame more slowly for subtle timing variation.
_BASE_BLINK_EVENTS = (
    (0.10, 2, 3),
    (0.47, 2, 4),
    (0.83, 2, 3),
)
# The first slow-blink pass was 1.5x the baseline. Stretching that result by
# another 50% makes the current animation 2.25x the original duration.
BLINK_DURATION_SCALE = 2.25
BLINK_EVENTS = tuple(
    (phase, close_frames * BLINK_DURATION_SCALE, open_frames * BLINK_DURATION_SCALE)
    for phase, close_frames, open_frames in _BASE_BLINK_EVENTS
)
MIDDLE_BLINK_EVENT = ((0.52, BLINK_EVENTS[0][1], BLINK_EVENTS[0][2]),)
BLINK_MINIMUM_SCALE = 0.70
# Kept as a compatibility alias for callers that explicitly inspect the
# neutral-resting light-blink profile.
NEUTRAL_RESTING_BLINK_MINIMUM_SCALE = BLINK_MINIMUM_SCALE
FACIAL_EXPRESSION_BOUNDARY_SECONDS = 0.5
SMILE_CORNER_LIFT_METERS = 0.006
SMILE_CORNER_WIDEN_METERS = 0.002
BROW_RAISE_METERS = 0.003
GAZE_HORIZONTAL_METERS = 0.0025
GAZE_VERTICAL_METERS = 0.0012


class RenderingBackendError(RuntimeError):
    """Raised when the optional headless OpenGL stack cannot initialize."""


def _load_rendering_modules() -> tuple[Any, Any]:
    # EGL is available without X11 on the supported Linux/NVIDIA servers.  The
    # environment variable must be set before importing PyOpenGL/pyrender.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
        import trimesh
    except Exception as exc:  # pragma: no cover - exact import failure is platform-specific
        raise RenderingBackendError(
            "Offline video rendering requires the 'video' extra. "
            "Install it with `pip install -e '.[video]'`."
        ) from exc
    return pyrender, trimesh


def blink_scale_for_frame(
    frame_index: int,
    frame_count: int,
    fps: float,
    minimum_scale: float = BLINK_MINIMUM_SCALE,
    *,
    behavior_id: str | None = None,
) -> float:
    """Return deterministic local eye height for one natural blink schedule.

    The optional behavior identifier applies the certified MVP policy: three
    light 30%-closure blinks for the neutral idle and no baked blinks for any
    moving pose. Calls without a behavior identifier retain the duration-based
    compatibility schedule for low-level animation tests and legacy callers.
    """

    if frame_count <= 0 or fps <= 0.0 or frame_index < 0 or frame_index >= frame_count:
        raise ValueError("invalid blink frame coordinates")
    if not 0.0 < minimum_scale <= 1.0:
        raise ValueError("minimum_scale must be in (0, 1]")
    protected_frames = int(round(BLINK_PROTECTED_SECONDS * fps))
    if frame_index < protected_frames or frame_index >= frame_count - protected_frames:
        return 1.0

    # The schedule margins intentionally use the original timing so changing
    # blink speed does not move the three established peak frames.
    max_close_frames = max(event[1] for event in _BASE_BLINK_EVENTS)
    max_open_frames = max(event[2] for event in _BASE_BLINK_EVENTS)
    action_start = protected_frames + max_close_frames
    action_end = frame_count - protected_frames - max_open_frames - 1
    if action_end <= action_start:
        return 1.0
    action_span = action_end - action_start

    normalized_behavior_id = (behavior_id or "").lower()
    if normalized_behavior_id == "neutral_resting":
        events = BLINK_EVENTS
    elif normalized_behavior_id:
        events = ()
    else:
        duration_seconds = frame_count / fps
        if duration_seconds >= 9.0:
            events = BLINK_EVENTS
        elif duration_seconds >= 7.0:
            events = (
                (0.28, BLINK_EVENTS[0][1], BLINK_EVENTS[0][2]),
                (0.72, BLINK_EVENTS[1][1], BLINK_EVENTS[1][2]),
            )
        elif duration_seconds >= 5.5:
            events = MIDDLE_BLINK_EVENT
        else:
            events = ()

    scale = 1.0
    for phase, close_frames, open_frames in events:
        center = int(round(action_start + phase * action_span))
        delta = frame_index - center
        if -close_frames <= delta <= 0:
            progress = (delta + close_frames) / close_frames
            eased = 0.5 - 0.5 * float(np.cos(np.pi * progress))
            candidate = 1.0 - (1.0 - minimum_scale) * eased
            scale = min(scale, candidate)
        elif 0 < delta <= open_frames:
            progress = delta / open_frames
            eased = 0.5 - 0.5 * float(np.cos(np.pi * progress))
            candidate = minimum_scale + (1.0 - minimum_scale) * eased
            scale = min(scale, candidate)
    return scale


def blink_minimum_scale_for_motion(motion: CoreMotion) -> float:
    """Return the approved light-blink depth when a behavior schedules blinks."""

    return BLINK_MINIMUM_SCALE


def _behavior_id(motion: CoreMotion) -> str:
    if motion.source_path is None:
        return ""
    name = motion.source_path.name.lower()
    for suffix in (".motion.npz", ".npz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return motion.source_path.stem.lower()


def facial_cue_for_frame(
    behavior_id: str,
    frame_index: int,
    frame_count: int,
    fps: float,
) -> dict[str, float]:
    """Return reset-safe procedural face cues for one behavior frame."""

    if frame_count <= 0 or fps <= 0.0 or frame_index < 0 or frame_index >= frame_count:
        raise ValueError("invalid facial-cue frame coordinates")
    boundary_frames = int(round(FACIAL_EXPRESSION_BOUNDARY_SECONDS * fps))
    interior_start = boundary_frames
    interior_end = frame_count - boundary_frames - 1
    if frame_index <= interior_start or frame_index >= interior_end or interior_end <= interior_start:
        envelope = 0.0
        phase = 0.0
    else:
        phase = (frame_index - interior_start) / (interior_end - interior_start)
        envelope = float(np.sin(np.pi * phase) ** 2)

    cue = {
        "smile": 0.0,
        "left_brow": 0.0,
        "right_brow": 0.0,
        "gaze_horizontal": 0.0,
        "gaze_vertical": 0.0,
    }
    behavior_id = behavior_id.lower()
    if behavior_id == "light_smile":
        # A readable closed-mouth smile: ease up from neutral, hold naturally,
        # then release before the canonical closing boundary.
        rise = float(np.clip((phase - 0.05) / 0.25, 0.0, 1.0))
        release = float(np.clip((0.95 - phase) / 0.25, 0.0, 1.0))
        rise = rise * rise * (3.0 - 2.0 * rise)
        release = release * release * (3.0 - 2.0 * release)
        cue["smile"] = min(rise, release)
    elif behavior_id == "amused_laugh":
        chuckle = 0.88 + 0.12 * float(np.sin(2.0 * np.pi * phase) ** 2)
        cue["smile"] = envelope * chuckle
    elif behavior_id == "curious_eyebrow_or_nod":
        cue["left_brow"] = 0.25 * envelope
        cue["right_brow"] = envelope
    elif behavior_id == "empathetic_head_tilt":
        cue["left_brow"] = 0.18 * envelope
        cue["right_brow"] = 0.18 * envelope
    elif behavior_id == "active_listening":
        cue["left_brow"] = 0.08 * envelope
        cue["right_brow"] = 0.08 * envelope
    elif behavior_id == "active_listening_empathetic_v1":
        cue["left_brow"] = 0.12 * envelope
        cue["right_brow"] = 0.12 * envelope
        glance = envelope * float(np.exp(-0.5 * ((phase - 0.48) / 0.10) ** 2))
        cue["gaze_horizontal"] = -0.12 * glance
        cue["gaze_vertical"] = -0.04 * glance

    if behavior_id == "thinking_glance":
        cue["gaze_horizontal"] = 0.85 * envelope
        cue["gaze_vertical"] = -0.70 * envelope
    elif behavior_id == "look_away_reset":
        cue["gaze_horizontal"] = -0.35 * envelope
    elif behavior_id == "speaking_direct_v2":
        # A speaker does not keep the upper face perfectly frozen.  These are
        # brief, low-amplitude eye-led checks and brow impulses aligned to
        # phrase stresses; the mouth remains completely available to MuseTalk.
        early_glance = float(np.exp(-0.5 * ((phase - 0.17) / 0.055) ** 2))
        center_glance = float(np.exp(-0.5 * ((phase - 0.64) / 0.070) ** 2))
        late_glance = float(np.exp(-0.5 * ((phase - 0.82) / 0.050) ** 2))
        cue["gaze_horizontal"] = envelope * (
            -0.08 * early_glance + 0.18 * center_glance - 0.10 * late_glance
        )
        cue["gaze_vertical"] = envelope * (
            0.015 * early_glance - 0.055 * center_glance - 0.025 * late_glance
        )

        emphasis = sum(
            amplitude * float(np.exp(-0.5 * ((phase - center) / width) ** 2))
            for center, width, amplitude in (
                (0.08, 0.035, 0.11),
                (0.30, 0.040, 0.18),
                (0.57, 0.045, 0.15),
                (0.84, 0.040, 0.13),
            )
        )
        cue["left_brow"] = envelope * 0.72 * emphasis
        cue["right_brow"] = envelope * emphasis
    return cue


def _direction_pose(origin: tuple[float, float, float], target: tuple[float, float, float]) -> np.ndarray:
    """Return a light/camera pose aimed from ``origin`` at ``target``."""

    origin_array = np.asarray(origin, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    z_axis = origin_array - target_array
    z_axis /= np.linalg.norm(z_axis)
    candidate_up = np.array((0.0, 1.0, 0.0))
    if abs(float(np.dot(z_axis, candidate_up))) > 0.99:
        candidate_up = np.array((1.0, 0.0, 0.0))
    x_axis = np.cross(candidate_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = origin_array
    return pose


class CoreMeshRenderer:
    """Render Core skin vertices from one immutable portrait camera.

    This class owns an EGL context and should be used as a context manager.
    It contains no viser server, browser state, overlays, grids, or gizmos.
    """

    def __init__(self, camera: CameraConfig | None = None, style: RenderStyle | None = None) -> None:
        self.camera = camera or CameraConfig.video_call_portrait()
        self.style = style or RenderStyle()
        self._pyrender: Any | None = None
        self._trimesh: Any | None = None
        self._scene: Any | None = None
        self._renderer: Any | None = None
        self._mesh_node: Any | None = None
        self._material: Any | None = None

        self.skeleton = CoreSkeleton27()
        self.skin = CoreSkin(self.skeleton)
        self.faces = np.ascontiguousarray(self.skin.faces.cpu().numpy(), dtype=np.int32)
        self.material_ids = np.ascontiguousarray(
            self.skin.vertex_material_ids.cpu().numpy(),
            dtype=np.int32,
        )
        self.blink_vertical_offsets = np.ascontiguousarray(
            self.skin.blink_vertical_offsets.cpu().numpy(),
            dtype=np.float32,
        )
        self.blink_weights = np.ascontiguousarray(
            self.skin.blink_weights.cpu().numpy(),
            dtype=np.float32,
        )
        bind_vertices = self.skin.bind_vertices.cpu().numpy()
        mouth = self.material_ids == MATERIAL_MOUTH
        brow = self.material_ids == MATERIAL_BROW
        gaze = (self.material_ids == MATERIAL_IRIS) | (self.material_ids == MATERIAL_PUPIL)
        self.mouth_corner_weights = np.zeros(len(self.material_ids), dtype=np.float32)
        self.mouth_corner_weights[mouth] = np.clip(
            np.abs(bind_vertices[mouth, 0]) / 0.038,
            0.0,
            1.0,
        ) ** 1.5
        self.mouth_corner_directions = (
            np.sign(bind_vertices[:, 0]).astype(np.float32)
            * self.mouth_corner_weights
        )
        self.left_brow_weights = (brow & (bind_vertices[:, 0] < 0.0)).astype(np.float32)
        self.right_brow_weights = (brow & (bind_vertices[:, 0] > 0.0)).astype(np.float32)
        self.gaze_weights = gaze.astype(np.float32)
        self.head_joint_index = self.skeleton.bone_order_names.index("Head")

    def __enter__(self) -> "CoreMeshRenderer":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._renderer is not None

    def open(self) -> None:
        if self.is_open:
            return
        pyrender, trimesh = _load_rendering_modules()
        self._pyrender = pyrender
        self._trimesh = trimesh

        background = np.round(np.asarray(self.style.background_rgba) * 255.0).astype(np.uint8)
        ambient = np.full(3, self.style.ambient_light, dtype=np.float32)
        scene = pyrender.Scene(bg_color=background, ambient_light=ambient)

        camera = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(self.camera.vertical_fov_degrees),
            aspectRatio=self.camera.aspect_ratio,
            znear=self.camera.near,
            zfar=self.camera.far,
        )
        scene.add(camera, pose=self.camera.pose_matrix())

        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.style.key_light_intensity)
        scene.add(key_light, pose=_direction_pose((-3.0, 4.5, 4.0), self.camera.target))
        if self.style.fill_light_intensity > 0.0:
            fill_light = pyrender.DirectionalLight(
                color=np.array((0.82, 0.9, 1.0)),
                intensity=self.style.fill_light_intensity,
            )
            scene.add(fill_light, pose=_direction_pose((3.5, 2.0, 2.0), self.camera.target))

        self._material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(1.0, 1.0, 1.0, 1.0),
            metallicFactor=self.style.metallic,
            roughnessFactor=self.style.roughness,
        )
        try:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=self.camera.width,
                viewport_height=self.camera.height,
            )
        except Exception as exc:
            self._pyrender = None
            self._trimesh = None
            raise RenderingBackendError(
                "Could not create a headless OpenGL context. On Linux, ensure EGL is installed "
                "and PYOPENGL_PLATFORM is unset or set to 'egl'."
            ) from exc

        self._scene = scene
        self._renderer = renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.delete()
        self._renderer = None
        self._scene = None
        self._mesh_node = None
        self._material = None
        self._pyrender = None
        self._trimesh = None

    def skin_vertices(self, motion: CoreMotion, chunk_size: int = 32) -> Iterator[np.ndarray]:
        """Yield skinned vertices in deterministic CPU chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        rotations = torch.from_numpy(motion.global_rot_mats)
        positions = torch.from_numpy(motion.posed_joints)
        minimum_blink_scale = blink_minimum_scale_for_motion(motion)
        behavior_id = _behavior_id(motion)
        with torch.inference_mode():
            for start in range(0, motion.num_frames, chunk_size):
                end = min(start + chunk_size, motion.num_frames)
                vertices = self.skin.skin(rotations[start:end], positions[start:end], rot_is_global=True)
                skinned_vertices = vertices.cpu().numpy()
                blink_scales = np.asarray(
                    [
                        blink_scale_for_frame(
                            index,
                            motion.num_frames,
                            motion.fps,
                            minimum_scale=minimum_blink_scale,
                            behavior_id=behavior_id,
                        )
                        for index in range(start, end)
                    ],
                    dtype=np.float32,
                )
                blink_amount = (
                    (blink_scales - 1.0)[:, None]
                    * self.blink_vertical_offsets[None, :]
                    * self.blink_weights[None, :]
                )
                head_up = motion.global_rot_mats[
                    start:end,
                    self.head_joint_index,
                    :,
                    1,
                ]
                head_right = motion.global_rot_mats[
                    start:end,
                    self.head_joint_index,
                    :,
                    0,
                ]
                facial_cues = [
                    facial_cue_for_frame(
                        behavior_id,
                        index,
                        motion.num_frames,
                        motion.fps,
                    )
                    for index in range(start, end)
                ]
                expression_up = np.stack(
                    [
                        (
                            cue["smile"]
                            * SMILE_CORNER_LIFT_METERS
                            * self.mouth_corner_weights
                            + cue["left_brow"]
                            * BROW_RAISE_METERS
                            * self.left_brow_weights
                            + cue["right_brow"]
                            * BROW_RAISE_METERS
                            * self.right_brow_weights
                            + cue["gaze_vertical"]
                            * GAZE_VERTICAL_METERS
                            * self.gaze_weights
                        )
                        for cue in facial_cues
                    ],
                    axis=0,
                )
                expression_right = np.stack(
                    [
                        (
                            cue["gaze_horizontal"]
                            * GAZE_HORIZONTAL_METERS
                            * self.gaze_weights
                            + cue["smile"]
                            * SMILE_CORNER_WIDEN_METERS
                            * self.mouth_corner_directions
                        )
                        for cue in facial_cues
                    ],
                    axis=0,
                )
                skinned_vertices = (
                    skinned_vertices
                    + blink_amount[:, :, None] * head_up[:, None, :]
                    + expression_up[:, :, None] * head_up[:, None, :]
                    + expression_right[:, :, None] * head_right[:, None, :]
                )
                for frame_vertices in skinned_vertices:
                    yield np.ascontiguousarray(frame_vertices, dtype=np.float32)

    def render_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Render one ``[V, 3]`` Core-skin vertex array to RGB uint8."""

        if not self.is_open:
            self.open()
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.shape != (self.skin.bind_vertices.shape[0], 3):
            raise ValueError(
                f"vertices must have shape ({self.skin.bind_vertices.shape[0]}, 3), got {vertices.shape}"
            )
        if not np.isfinite(vertices).all():
            raise ValueError("vertices contain NaN or infinite values")

        assert self._scene is not None
        assert self._renderer is not None
        assert self._pyrender is not None
        assert self._trimesh is not None
        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)

        palette = np.zeros((MATERIAL_EYE_BACKING + 1, 4), dtype=np.uint8)
        palette[MATERIAL_BODY] = np.round(np.asarray(self.style.body_rgba) * 255.0).astype(np.uint8)
        palette[MATERIAL_SKIN] = (183, 126, 96, 255)
        palette[MATERIAL_SCLERA] = (242, 239, 229, 255)
        palette[MATERIAL_IRIS] = (83, 49, 30, 255)
        palette[MATERIAL_PUPIL] = (18, 16, 15, 255)
        palette[MATERIAL_BROW] = (52, 35, 28, 255)
        palette[MATERIAL_MOUTH] = (124, 61, 62, 255)
        palette[MATERIAL_EYE_BACKING] = palette[MATERIAL_SKIN]
        vertex_colors = palette[self.material_ids]
        triangle_mesh = self._trimesh.Trimesh(
            vertices=vertices,
            faces=self.faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        render_mesh = self._pyrender.Mesh.from_trimesh(
            triangle_mesh,
            material=None,
            smooth=self.style.smooth_shading,
        )
        self._mesh_node = self._scene.add(render_mesh)
        color, _depth = self._renderer.render(self._scene, flags=self._pyrender.RenderFlags.NONE)
        color = np.asarray(color, dtype=np.uint8)
        expected_shape = (self.camera.height, self.camera.width, 3)
        if color.shape != expected_shape:
            raise RenderingBackendError(f"renderer returned {color.shape}, expected {expected_shape}")
        return np.ascontiguousarray(color)

    def render_motion(self, motion: CoreMotion, chunk_size: int = 32) -> Iterator[np.ndarray]:
        """Yield clean RGB frames for an already-resampled motion."""

        for vertices in self.skin_vertices(motion, chunk_size=chunk_size):
            yield self.render_vertices(vertices)
