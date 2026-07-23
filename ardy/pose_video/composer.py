# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic high-precision motion composition on ARDY skeletons.

This module is deliberately independent of diffusion inference.  Authored
keyframes are exact, repeatable, and safe to splice.  A generated ARDY motion
can still be selected as the base pose, while optional diffusion naturalizing
can happen later without being trusted with canonical boundary frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from ardy.skeleton.registry import build_skeleton

from .spec import BehaviorSpec, behavior_spec_fingerprint


@dataclass(frozen=True)
class ComposedMotion:
    behavior_id: str
    fps: int
    local_rot_mats: np.ndarray
    global_rot_mats: np.ndarray
    root_positions: np.ndarray
    posed_joints: np.ndarray
    boundary_frames: int
    metadata: dict

    @property
    def num_frames(self) -> int:
        return int(self.local_rot_mats.shape[0])

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            behavior_id=np.asarray(self.behavior_id),
            fps=np.asarray(self.fps),
            boundary_frames=np.asarray(self.boundary_frames),
            local_rot_mats=self.local_rot_mats,
            global_rot_mats=self.global_rot_mats,
            root_positions=self.root_positions,
            smooth_root_pos=self.root_positions,
            posed_joints=self.posed_joints,
            foot_contacts=np.zeros((self.num_frames, 4), dtype=np.float32),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )


def _rotation_matrix_xyz(degrees: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(degrees, dtype=np.float64), degrees=True).as_matrix()


def _authored_neutral(skeleton) -> tuple[np.ndarray, np.ndarray]:
    """Return a relaxed upper-body base pose and a fixed root position.

    Core's bind pose is a T-pose.  The shoulder rotations below lower the arms
    alongside the torso so portrait motion references read as a calm person,
    while retaining an entirely deterministic and inspectable starting pose.
    """

    local = np.repeat(np.eye(3, dtype=np.float32)[None], skeleton.nbjoints, axis=0)
    authored = {
        # Keep the clavicles broad and rotate at the anatomical upper-arm
        # joints. The earlier pose lowered the whole shoulder chains, pulling
        # the arm sockets into the torso. Small mirrored local-Y rotations
        # bring the arms toward the camera; local-Z lowers them alongside the
        # body with a restrained elbow angle.
        "RightShoulder": (2.0, 2.0, 5.0),
        "LeftShoulder": (2.0, -2.0, -5.0),
        "RightArm": (0.0, 6.0, 75.0),
        "LeftArm": (0.0, -6.0, -75.0),
        "RightForeArm": (0.0, 2.0, 5.0),
        "LeftForeArm": (0.0, -2.0, -5.0),
        "Spine": (-2.0, 0.0, 0.0),
        "Spine1": (1.0, 0.0, 0.0),
        "Neck": (1.0, 0.0, 0.0),
    }
    for joint_name, degrees in authored.items():
        if joint_name in skeleton.bone_index:
            local[skeleton.bone_index[joint_name]] = _rotation_matrix_xyz(degrees).astype(np.float32)
    return local, np.zeros(3, dtype=np.float32)


def _load_base_pose(spec: BehaviorSpec, skeleton) -> tuple[np.ndarray, np.ndarray]:
    if spec.base_pose.mode == "authored_neutral":
        return _authored_neutral(skeleton)

    motion_path = Path(spec.base_pose.motion_path or "")
    if not motion_path.is_file():
        raise FileNotFoundError(f"base motion does not exist: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as motion:
        if "local_rot_mats" not in motion or "root_positions" not in motion:
            raise ValueError(f"{motion_path} must contain local_rot_mats and root_positions")
        local = np.asarray(motion["local_rot_mats"])
        root = np.asarray(motion["root_positions"])
        # Some model outputs retain a sample dimension.
        if local.ndim == 5:
            if local.shape[0] == 0:
                raise ValueError(f"base motion has an empty sample dimension: {motion_path}")
            local = local[0]
        if root.ndim == 3:
            if root.shape[0] == 0:
                raise ValueError(f"base motion has an empty sample dimension: {motion_path}")
            root = root[0]
        if local.ndim != 4 or local.shape[1:] != (skeleton.nbjoints, 3, 3):
            raise ValueError(
                f"base motion has rotation shape {local.shape}; "
                f"expected [frames, {skeleton.nbjoints}, 3, 3]"
            )
        if root.ndim != 2 or root.shape != (len(local), 3):
            raise ValueError(
                f"base motion has root shape {root.shape}; expected [{len(local)}, 3]"
            )
        if not np.isfinite(local).all() or not np.isfinite(root).all():
            raise ValueError(f"base motion contains NaN or infinite values: {motion_path}")
        frame = spec.base_pose.frame
        if frame >= len(local):
            raise ValueError(f"base-pose frame {frame} is outside {motion_path} ({len(local)} frames)")
        identity = np.eye(3, dtype=np.float64)
        orthogonality_error = float(
            np.max(np.abs(np.swapaxes(local[frame], -1, -2) @ local[frame] - identity))
        )
        determinant_error = float(np.max(np.abs(np.linalg.det(local[frame]) - 1.0)))
        if orthogonality_error > 2e-3 or determinant_error > 2e-3:
            raise ValueError(f"base motion frame {frame} does not contain valid SO(3) rotations")
        return local[frame].astype(np.float32), root[frame].astype(np.float32)


def _quintic_smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def _interpolate_joint_deltas(
    times: np.ndarray,
    key_times: list[float],
    key_degrees: list[tuple[float, float, float]],
) -> np.ndarray:
    """Piecewise eased spherical interpolation through rotation deltas."""

    matrices = np.empty((len(times), 3, 3), dtype=np.float64)
    key_mats = [_rotation_matrix_xyz(value) for value in key_degrees]
    for segment in range(len(key_times) - 1):
        start, end = key_times[segment], key_times[segment + 1]
        if segment == len(key_times) - 2:
            mask = (times >= start) & (times <= end)
        else:
            mask = (times >= start) & (times < end)
        if not mask.any():
            continue
        if end <= start:
            raise ValueError("joint keyframe times must be strictly increasing")
        alpha = _quintic_smoothstep((times[mask] - start) / (end - start))
        rotations = Rotation.from_matrix(np.stack([key_mats[segment], key_mats[segment + 1]]))
        matrices[mask] = Slerp([0.0, 1.0], rotations)(alpha).as_matrix()
    before = times < key_times[0]
    after = times > key_times[-1]
    matrices[before] = key_mats[0]
    matrices[after] = key_mats[-1]
    return matrices


def _tapered_ambient(spec: BehaviorSpec, times: np.ndarray) -> dict[str, np.ndarray]:
    """Generate deterministic breathing/sway that is zero at boundary joins."""

    ambient = spec.ambient_motion
    zeros = np.zeros((len(times), 3), dtype=np.float64)
    if not ambient.enabled:
        return {}

    interior_start = spec.boundary_seconds
    interior_end = spec.duration_seconds - spec.boundary_seconds
    fade = min(0.75, (interior_end - interior_start) / 4.0)
    envelope = np.ones(len(times), dtype=np.float64)
    envelope[times <= interior_start] = 0.0
    envelope[times >= interior_end] = 0.0
    enter = (times > interior_start) & (times < interior_start + fade)
    leave = (times > interior_end - fade) & (times < interior_end)
    envelope[enter] = _quintic_smoothstep((times[enter] - interior_start) / fade)
    envelope[leave] = _quintic_smoothstep((interior_end - times[leave]) / fade)

    interior_duration = interior_end - interior_start
    normalized = (times - interior_start) / max(interior_duration, 1e-6)
    breath = np.sin(2.0 * np.pi * ambient.breathing_cycles * normalized) * envelope
    sway = np.sin(2.0 * np.pi * ambient.sway_cycles * normalized) * envelope

    tracks: dict[str, np.ndarray] = {}
    spine2 = zeros.copy()
    spine2[:, 0] = -ambient.breathing_degrees * breath
    spine2[:, 2] = ambient.sway_degrees * sway
    tracks["Spine2"] = spine2

    spine3 = zeros.copy()
    spine3[:, 0] = 0.55 * ambient.breathing_degrees * breath
    spine3[:, 2] = 0.45 * ambient.sway_degrees * sway
    tracks["Spine3"] = spine3

    head = zeros.copy()
    head[:, 1] = ambient.head_sway_degrees * np.sin(
        2.0 * np.pi * (ambient.sway_cycles * 0.5) * normalized + np.pi / 3.0
    ) * envelope
    tracks["Head"] = head
    return tracks


def _canonical_boundary(fps: int, boundary_frames: int, skeleton) -> np.ndarray:
    """A shared zero-velocity neutral motion block.

    ``sin(pi*t)^2`` starts and ends at the same pose with zero analytical
    velocity.  Every behavior receives this exact array at both ends.
    """

    local_base, _ = _authored_neutral(skeleton)
    local = np.repeat(local_base[None], boundary_frames, axis=0).astype(np.float64)
    if boundary_frames < 2:
        return local.astype(np.float32)
    phase = np.linspace(0.0, 1.0, boundary_frames)
    lobe = np.sin(np.pi * phase) ** 2
    deltas = {
        "Spine2": np.column_stack((-0.45 * lobe, np.zeros_like(lobe), 0.18 * lobe)),
        "Spine3": np.column_stack((0.25 * lobe, np.zeros_like(lobe), 0.08 * lobe)),
        "Neck": np.column_stack((0.08 * lobe, np.zeros_like(lobe), -0.06 * lobe)),
    }
    for joint_name, degrees in deltas.items():
        if joint_name not in skeleton.bone_index:
            continue
        joint_idx = skeleton.bone_index[joint_name]
        delta_mats = Rotation.from_euler("xyz", degrees, degrees=True).as_matrix()
        local[:, joint_idx] = local[:, joint_idx] @ delta_mats
    return local.astype(np.float32)


def compose_behavior(spec: BehaviorSpec) -> ComposedMotion:
    """Compile a behavior specification into exact ARDY Core motion arrays."""

    skeleton = build_skeleton(27)
    valid_names = set(skeleton.bone_order_names)
    unknown_locks = sorted(set(spec.locks) - valid_names)
    if unknown_locks:
        raise ValueError(f"unknown locked joints: {unknown_locks}")

    base_local, base_root = _load_base_pose(spec, skeleton)
    num_frames = spec.num_frames
    boundary_frames = spec.boundary_frames
    if boundary_frames < 2:
        raise ValueError("boundary block must contain at least two frames")
    times = np.arange(num_frames, dtype=np.float64) / spec.fps
    local = np.repeat(base_local[None], num_frames, axis=0).astype(np.float64)
    root = np.repeat(base_root[None], num_frames, axis=0).astype(np.float32)

    # Gather sparse authored deltas by joint.  Implicit neutral targets at the
    # two interior edges guarantee a return before canonical blocks are copied.
    authored: dict[str, list[tuple[float, tuple[float, float, float]]]] = {}
    for keyframe in spec.keyframes:
        for joint_name, transform in keyframe.joints.items():
            if joint_name not in valid_names:
                raise ValueError(f"unknown joint in keyframe: {joint_name}")
            if joint_name in spec.locks:
                raise ValueError(f"keyframe attempts to move locked joint: {joint_name}")
            authored.setdefault(joint_name, []).append((keyframe.time_seconds, transform.rotation_degrees))

    interior_start = spec.boundary_seconds
    interior_end = spec.duration_seconds - spec.boundary_seconds
    for joint_name, targets in authored.items():
        by_time = {interior_start: (0.0, 0.0, 0.0), interior_end: (0.0, 0.0, 0.0)}
        for target_time, degrees in targets:
            if target_time <= interior_start or target_time >= interior_end:
                raise ValueError(
                    f"{joint_name} keyframe at {target_time}s lies inside a canonical boundary block"
                )
            by_time[target_time] = degrees
        ordered = sorted(by_time.items())
        delta = _interpolate_joint_deltas(times, [x[0] for x in ordered], [x[1] for x in ordered])
        joint_idx = skeleton.bone_index[joint_name]
        local[:, joint_idx] = local[:, joint_idx] @ delta

    for joint_name, degrees in _tapered_ambient(spec, times).items():
        if joint_name not in valid_names or joint_name in spec.locks:
            continue
        joint_idx = skeleton.bone_index[joint_name]
        delta = Rotation.from_euler("xyz", degrees, degrees=True).as_matrix()
        local[:, joint_idx] = local[:, joint_idx] @ delta

    canonical = _canonical_boundary(spec.fps, boundary_frames, skeleton)
    # Canonical blocks always use the same authored base.  A custom base pose
    # is allowed for experimentation but cannot silently weaken seam identity.
    if spec.base_pose.mode != "authored_neutral":
        canonical = np.repeat(base_local[None], boundary_frames, axis=0).astype(np.float64)
        canonical_delta = _canonical_boundary(spec.fps, boundary_frames, skeleton)
        neutral, _ = _authored_neutral(skeleton)
        for joint_idx in range(skeleton.nbjoints):
            relative = np.swapaxes(neutral[joint_idx], -1, -2) @ canonical_delta[:, joint_idx]
            canonical[:, joint_idx] = canonical[:, joint_idx] @ relative
        canonical = canonical.astype(np.float32)
    local[:boundary_frames] = canonical
    local[-boundary_frames:] = canonical

    local_tensor = torch.from_numpy(local.astype(np.float32))
    root_tensor = torch.from_numpy(root)
    with torch.no_grad():
        global_rots, posed_joints, _ = skeleton.fk(local_tensor, root_tensor)

    metadata = {
        "schema_version": spec.schema_version,
        "behavior_id": spec.behavior_id,
        "behavior_type": spec.behavior_type,
        "speaking_mode": spec.speaking_mode,
        "duration_seconds": spec.duration_seconds,
        "boundary_seconds": spec.boundary_seconds,
        "face_intent": spec.face_intent.model_dump(mode="json"),
        "source_reference": spec.source_reference,
        "composer": "ardy.pose_video.precision/v2",
        "spec_fingerprint": behavior_spec_fingerprint(spec),
    }
    return ComposedMotion(
        behavior_id=spec.behavior_id,
        fps=spec.fps,
        local_rot_mats=local.astype(np.float32),
        global_rot_mats=global_rots.cpu().numpy().astype(np.float32),
        root_positions=root,
        posed_joints=posed_joints.cpu().numpy().astype(np.float32),
        boundary_frames=boundary_frames,
        metadata=metadata,
    )
