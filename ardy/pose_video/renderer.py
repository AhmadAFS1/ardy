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
from ardy.viz.core_skin import CoreSkin

from .config import CameraConfig, RenderStyle
from .motion import CoreMotion


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
            baseColorFactor=self.style.body_rgba,
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
        with torch.inference_mode():
            for start in range(0, motion.num_frames, chunk_size):
                end = min(start + chunk_size, motion.num_frames)
                vertices = self.skin.skin(rotations[start:end], positions[start:end], rot_is_global=True)
                for frame_vertices in vertices.cpu().numpy():
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

        triangle_mesh = self._trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        render_mesh = self._pyrender.Mesh.from_trimesh(
            triangle_mesh,
            material=self._material,
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
