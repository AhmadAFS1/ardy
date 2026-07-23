# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading, validation, and frame-rate conversion for Core motion archives."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ardy.skeleton import CoreSkeleton27


CORE_JOINT_COUNT = 27
DEFAULT_SOURCE_FPS = 20.0


@dataclass(frozen=True)
class CoreMotion:
    """Render-ready global Core skeleton transforms."""

    global_rot_mats: np.ndarray
    posed_joints: np.ndarray
    fps: float
    text: str = ""
    source_path: Path | None = None

    def __post_init__(self) -> None:
        rotations = np.asarray(self.global_rot_mats, dtype=np.float32)
        positions = np.asarray(self.posed_joints, dtype=np.float32)
        if rotations.ndim != 4 or rotations.shape[1:] != (CORE_JOINT_COUNT, 3, 3):
            raise ValueError(
                "global_rot_mats must have shape "
                f"[T, {CORE_JOINT_COUNT}, 3, 3], got {rotations.shape}"
            )
        if positions.ndim != 3 or positions.shape[1:] != (CORE_JOINT_COUNT, 3):
            raise ValueError(
                f"posed_joints must have shape [T, {CORE_JOINT_COUNT}, 3], got {positions.shape}"
            )
        if rotations.shape[0] != positions.shape[0]:
            raise ValueError("rotation and position arrays have different frame counts")
        if rotations.shape[0] == 0:
            raise ValueError("motion must contain at least one frame")
        if not np.isfinite(rotations).all() or not np.isfinite(positions).all():
            raise ValueError("motion arrays contain NaN or infinite values")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise ValueError("motion fps must be a finite positive number")

        # Reject corrupted transforms while allowing the small numerical drift
        # found in generated float32 archives.
        identity = np.eye(3, dtype=np.float32)
        orthogonality_error = np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - identity))
        determinants = np.linalg.det(rotations)
        if orthogonality_error > 5e-2 or np.min(determinants) <= 0.0:
            raise ValueError("global_rot_mats contains invalid rotation matrices")

        object.__setattr__(self, "global_rot_mats", np.ascontiguousarray(rotations))
        object.__setattr__(self, "posed_joints", np.ascontiguousarray(positions))
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "text", str(self.text))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def num_frames(self) -> int:
        return int(self.posed_joints.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Nominal media duration, treating every source sample as one frame."""

        return self.num_frames / self.fps


def _read_scalar_fps(data: np.lib.npyio.NpzFile, path: Path) -> float:
    if "fps" not in data:
        return DEFAULT_SOURCE_FPS
    fps_array = np.asarray(data["fps"])
    if fps_array.size != 1:
        raise ValueError(f"{path}: fps must be a scalar")
    return float(fps_array.reshape(()))


def load_core_motion(path: str | Path) -> CoreMotion:
    """Load a generated Core ``.npz`` without enabling pickle.

    Generated archives normally contain global transforms directly.  Archives
    with only ``local_rot_mats`` and ``root_positions`` are reconstructed with
    the canonical :class:`CoreSkeleton27` forward kinematics.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".npz":
        raise ValueError(f"expected a .npz motion archive, got {source}")

    with np.load(source, allow_pickle=False) as data:
        global_rot_mats = np.asarray(data["global_rot_mats"], dtype=np.float32) if "global_rot_mats" in data else None
        posed_joints = np.asarray(data["posed_joints"], dtype=np.float32) if "posed_joints" in data else None

        if global_rot_mats is None or posed_joints is None:
            if "local_rot_mats" not in data or "root_positions" not in data:
                raise ValueError(
                    f"{source}: expected global_rot_mats + posed_joints, or local_rot_mats + root_positions"
                )
            local_rot_mats = torch.from_numpy(np.asarray(data["local_rot_mats"], dtype=np.float32))
            root_positions = torch.from_numpy(np.asarray(data["root_positions"], dtype=np.float32))
            if local_rot_mats.ndim != 4 or local_rot_mats.shape[1:] != (CORE_JOINT_COUNT, 3, 3):
                raise ValueError(
                    f"{source}: local_rot_mats must have shape [T, {CORE_JOINT_COUNT}, 3, 3]"
                )
            if root_positions.shape != (local_rot_mats.shape[0], 3):
                raise ValueError(f"{source}: root_positions must have shape [T, 3]")
            with torch.inference_mode():
                fk_rotations, fk_positions, _ = CoreSkeleton27().fk(local_rot_mats, root_positions)
            if global_rot_mats is None:
                global_rot_mats = fk_rotations.cpu().numpy()
            if posed_joints is None:
                posed_joints = fk_positions.cpu().numpy()

        fps = _read_scalar_fps(data, source)
        text = str(np.asarray(data["text"]).reshape(())) if "text" in data else ""

    return CoreMotion(
        global_rot_mats=global_rot_mats,
        posed_joints=posed_joints,
        fps=fps,
        text=text,
        source_path=source.resolve(),
    )


def _slerp_quaternions(q0: np.ndarray, q1: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Vectorized shortest-arc quaternion interpolation (SciPy xyzw order)."""

    dots = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dots < 0.0, -q1, q1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    linear = dots > 0.9995
    theta = np.arccos(dots)
    sin_theta = np.sin(theta)
    safe_denominator = np.where(np.abs(sin_theta) < 1e-8, 1.0, sin_theta)
    weight0 = np.sin((1.0 - amount) * theta) / safe_denominator
    weight1 = np.sin(amount * theta) / safe_denominator
    spherical = weight0 * q0 + weight1 * q1
    lerped = (1.0 - amount) * q0 + amount * q1
    result = np.where(linear, lerped, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def resample_core_motion(motion: CoreMotion, target_fps: float) -> CoreMotion:
    """Resample positions linearly and global rotations with quaternion SLERP.

    The output frame count is ``round(T * target_fps / source_fps)``.  Sampling
    includes both original endpoints, which is important for canonical boundary
    frames shared by multiple behavior assets.
    """

    target_fps = float(target_fps)
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be a finite positive number")
    if math.isclose(motion.fps, target_fps, rel_tol=0.0, abs_tol=1e-9):
        return replace(motion, fps=target_fps)

    output_frames = max(1, int(round(motion.num_frames * target_fps / motion.fps)))
    source_coordinates = np.linspace(0.0, motion.num_frames - 1, output_frames, dtype=np.float64)
    lower = np.floor(source_coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, motion.num_frames - 1)
    amount = (source_coordinates - lower).astype(np.float32)

    positions = (
        (1.0 - amount[:, None, None]) * motion.posed_joints[lower]
        + amount[:, None, None] * motion.posed_joints[upper]
    )

    source_quaternions = Rotation.from_matrix(motion.global_rot_mats.reshape(-1, 3, 3)).as_quat()
    source_quaternions = source_quaternions.reshape(motion.num_frames, CORE_JOINT_COUNT, 4)
    quaternions = _slerp_quaternions(
        source_quaternions[lower],
        source_quaternions[upper],
        amount[:, None, None],
    )
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4)).as_matrix()
    rotations = rotations.reshape(output_frames, CORE_JOINT_COUNT, 3, 3).astype(np.float32)

    return CoreMotion(
        global_rot_mats=rotations,
        posed_joints=positions.astype(np.float32),
        fps=target_fps,
        text=motion.text,
        source_path=motion.source_path,
    )
