# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion- and pixel-level QA for transferable avatar behavior assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ardy.skeleton.registry import build_skeleton


@dataclass(frozen=True)
class VideoValidationExpectations:
    """Optional media contract layered on top of decoded seam validation.

    General-purpose validation intentionally leaves these fields unset: a
    post-processing service may return a perfectly switchable video with a
    different delivery codec.  A library build, however, knows the exact
    encoder contract it requested and should certify that ffmpeg actually
    produced it.
    """

    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None
    codec_name: str | None = None
    pixel_format: str | None = None
    color_space: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    audio_stream_count: int | None = None
    all_intra: bool | None = None
    allowed_profiles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("expected fps must be a finite positive number")
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"expected {name} must be positive")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0
        ):
            raise ValueError("expected duration_seconds must be a finite positive number")
        if self.audio_stream_count is not None and self.audio_stream_count < 0:
            raise ValueError("expected audio_stream_count cannot be negative")
        if self.allowed_profiles is not None and not self.allowed_profiles:
            raise ValueError("allowed_profiles cannot be empty")

    @classmethod
    def master_reference(
        cls,
        fps: float,
        *,
        width: int | None = None,
        height: int | None = None,
        frame_count: int | None = None,
        duration_seconds: float | None = None,
    ) -> "VideoValidationExpectations":
        """The lossless ARDY motion-reference encoder contract.

        The observed H.264 profile is recorded but deliberately not pinned:
        lossless x264 profile labels can vary by ffmpeg/x264 build, and these
        files are reference masters rather than browser-delivery encodes.
        """

        return cls(
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
            codec_name="h264",
            pixel_format="yuv420p",
            color_space="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
            audio_stream_count=0,
            all_intra=True,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.allowed_profiles is not None:
            payload["allowed_profiles"] = list(self.allowed_profiles)
        return payload


def _rotation_angle_degrees(matrices: np.ndarray) -> np.ndarray:
    flat = np.asarray(matrices).reshape(-1, 3, 3)
    return np.degrees(Rotation.from_matrix(flat).magnitude()).reshape(np.asarray(matrices).shape[:-2])


def _relative_rotations(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.swapaxes(reference, -1, -2) @ values


def validate_motion_npz(path: str | Path) -> dict:
    """Validate a composed motion and return a JSON-serializable report."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as motion:
        required = {"local_rot_mats", "global_rot_mats", "root_positions", "posed_joints", "fps"}
        missing = sorted(required - set(motion.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        local = np.asarray(motion["local_rot_mats"], dtype=np.float64)
        global_rots = np.asarray(motion["global_rot_mats"], dtype=np.float64)
        root = np.asarray(motion["root_positions"], dtype=np.float64)
        joints = np.asarray(motion["posed_joints"], dtype=np.float64)
        fps_value = np.asarray(motion["fps"])
        if fps_value.size != 1:
            raise ValueError("fps must be a scalar")
        fps_float = float(fps_value.reshape(()))
        if not np.isfinite(fps_float) or fps_float <= 0 or not fps_float.is_integer():
            raise ValueError("fps must be a positive integer")
        fps = int(fps_float)
        boundary_value = np.asarray(
            motion["boundary_frames"] if "boundary_frames" in motion else max(2, round(2 * fps))
        )
        if boundary_value.size != 1:
            raise ValueError("boundary_frames must be a scalar")
        boundary_float = float(boundary_value.reshape(()))
        if not np.isfinite(boundary_float) or not boundary_float.is_integer():
            raise ValueError("boundary_frames must be an integer")
        boundary_frames = int(boundary_float)
        behavior_id = str(np.asarray(motion.get("behavior_id", path.stem)).item())

    if local.ndim != 4 or local.shape[-2:] != (3, 3):
        raise ValueError(f"invalid local rotation shape: {local.shape}")
    if local.shape[1] != 27:
        raise ValueError(f"local_rot_mats must use the 27-joint Core skeleton, got {local.shape[1]} joints")
    if boundary_frames < 2:
        raise ValueError("boundary_frames must be at least two")
    if len(local) <= 2 * boundary_frames:
        raise ValueError("motion must contain an interior between its two canonical boundary blocks")
    if global_rots.shape != local.shape:
        raise ValueError("global_rot_mats must match local_rot_mats shape")
    if root.shape != (len(local), 3):
        raise ValueError("root_positions must have shape [frames, 3]")
    if joints.shape != (len(local), local.shape[1], 3):
        raise ValueError("posed_joints must have shape [frames, joints, 3]")
    if not all(np.isfinite(array).all() for array in (local, global_rots, root, joints)):
        raise ValueError("motion contains NaN or infinite values")

    identity = np.eye(3)
    local_orthogonality = np.max(np.abs(np.swapaxes(local, -1, -2) @ local - identity))
    local_determinant_error = np.max(np.abs(np.linalg.det(local) - 1.0))
    global_orthogonality = np.max(np.abs(np.swapaxes(global_rots, -1, -2) @ global_rots - identity))
    global_determinant_error = np.max(np.abs(np.linalg.det(global_rots) - 1.0))
    endpoint_angles = _rotation_angle_degrees(_relative_rotations(local[0], local[-1]))
    endpoint_root_cm = float(np.linalg.norm(root[-1] - root[0]) * 100.0)
    block_abs_error = float(np.max(np.abs(local[:boundary_frames] - local[-boundary_frames:])))
    block_root_error_cm = float(
        np.max(np.linalg.norm(root[:boundary_frames] - root[-boundary_frames:], axis=-1)) * 100.0
    )

    # Seam speed is evaluated on both sides as well as on the actual last->first
    # edge. Canonical zero-velocity boundaries should keep all three tiny.
    start_step = _rotation_angle_degrees(_relative_rotations(local[0], local[1]))
    end_step = _rotation_angle_degrees(_relative_rotations(local[-2], local[-1]))
    seam_step = _rotation_angle_degrees(_relative_rotations(local[-1], local[0]))

    skeleton = build_skeleton(local.shape[1])
    with torch.inference_mode():
        expected_global, expected_joints, _ = skeleton.fk(
            torch.from_numpy(local.astype(np.float32)),
            torch.from_numpy(root.astype(np.float32)),
        )
    expected_global_np = expected_global.cpu().numpy().astype(np.float64)
    expected_joints_np = expected_joints.cpu().numpy().astype(np.float64)
    fk_rotation_abs_error = float(np.max(np.abs(global_rots - expected_global_np)))
    fk_joint_error_cm = float(
        np.max(np.linalg.norm(joints - expected_joints_np, axis=-1)) * 100.0
    )
    head_idx = skeleton.bone_index.get("Head")
    head_excursion = None
    if head_idx is not None:
        relative_head = _relative_rotations(global_rots[0, head_idx], global_rots[:, head_idx])
        head_excursion = float(_rotation_angle_degrees(relative_head).max())

    tolerances = {
        "rotation_endpoint_degrees": 1e-4,
        "root_endpoint_cm": 1e-4,
        "boundary_abs": 1e-7,
        "so3_abs": 2e-4,
        "fk_rotation_abs": 2e-5,
        "fk_joint_cm": 1e-3,
        "seam_step_degrees": 0.05,
    }
    checks = {
        "finite": True,
        "valid_so3": bool(
            local_orthogonality <= tolerances["so3_abs"]
            and local_determinant_error <= tolerances["so3_abs"]
            and global_orthogonality <= tolerances["so3_abs"]
            and global_determinant_error <= tolerances["so3_abs"]
        ),
        "consistent_forward_kinematics": bool(
            fk_rotation_abs_error <= tolerances["fk_rotation_abs"]
            and fk_joint_error_cm <= tolerances["fk_joint_cm"]
        ),
        "matching_endpoint_pose": bool(endpoint_angles.max() <= tolerances["rotation_endpoint_degrees"]),
        "matching_endpoint_root": bool(endpoint_root_cm <= tolerances["root_endpoint_cm"]),
        "identical_boundary_motion": bool(
            block_abs_error <= tolerances["boundary_abs"] and block_root_error_cm <= tolerances["root_endpoint_cm"]
        ),
        "zero_velocity_loop_seam": bool(
            max(float(start_step.max()), float(end_step.max()), float(seam_step.max()))
            <= tolerances["seam_step_degrees"]
        ),
    }
    return {
        "schema_version": 1,
        "asset": str(path),
        "behavior_id": behavior_id,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "frames": int(len(local)),
            "fps": fps,
            "duration_seconds": len(local) / fps,
            "boundary_frames": boundary_frames,
            "max_so3_orthogonality_error": float(max(local_orthogonality, global_orthogonality)),
            "max_so3_determinant_error": float(max(local_determinant_error, global_determinant_error)),
            "max_local_so3_orthogonality_error": float(local_orthogonality),
            "max_local_so3_determinant_error": float(local_determinant_error),
            "max_global_so3_orthogonality_error": float(global_orthogonality),
            "max_global_so3_determinant_error": float(global_determinant_error),
            "fk_rotation_abs_error": fk_rotation_abs_error,
            "fk_joint_error_cm": fk_joint_error_cm,
            "max_endpoint_rotation_error_degrees": float(endpoint_angles.max()),
            "endpoint_root_error_cm": endpoint_root_cm,
            "boundary_rotation_abs_error": block_abs_error,
            "boundary_root_error_cm": block_root_error_cm,
            "max_start_step_degrees": float(start_step.max()),
            "max_end_step_degrees": float(end_step.max()),
            "loop_seam_step_degrees": float(seam_step.max()),
            "max_head_excursion_degrees": head_excursion,
            "max_root_drift_cm": float(np.linalg.norm(root - root[0], axis=-1).max() * 100.0),
        },
        "tolerances": tolerances,
    }


def _probe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,codec_long_name,profile,level,"
            "width,height,avg_frame_rate,nb_frames,pix_fmt,has_b_frames,"
            "color_range,color_space,color_transfer,color_primaries,"
            "sample_fmt,sample_rate,channels,channel_layout"
        ),
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _probe_frame_types(path: Path) -> dict:
    """Return decoded-picture facts needed to prove an all-intra stream."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=key_frame,pict_type",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    frames = payload.get("frames", [])
    picture_types: dict[str, int] = {}
    key_frame_count = 0
    for frame in frames:
        key_frame_count += int(frame.get("key_frame", 0))
        picture_type = str(frame.get("pict_type", "unknown"))
        picture_types[picture_type] = picture_types.get(picture_type, 0) + 1
    frame_count = len(frames)
    return {
        "probed_frame_count": frame_count,
        "key_frame_count": key_frame_count,
        "picture_types": dict(sorted(picture_types.items())),
        "all_intra": bool(
            frame_count > 0
            and key_frame_count == frame_count
            and set(picture_types).issubset({"I"})
        ),
    }


def _ratio(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def video_frame_hashes(path: str | Path, boundary_frames: int) -> dict:
    """Hash decoded RGB frames without retaining a whole video in memory."""

    path = Path(path)
    if boundary_frames < 1:
        raise ValueError("boundary_frames must be positive")
    probe = _probe_video(path)
    streams = probe.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise ValueError(f"video has no video stream: {path}")
    stream = video_streams[0]
    frame_facts = _probe_frame_types(path)
    frame_facts["all_intra"] = bool(
        frame_facts["all_intra"] and int(stream.get("has_b_frames", 0)) == 0
    )
    width, height = int(stream["width"]), int(stream["height"])
    reported_frames = stream.get("nb_frames")
    expected_frames = int(reported_frames) if reported_frames not in (None, "", "N/A") else 0
    frame_bytes = width * height * 3
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("failed to open ffmpeg frame pipe")

    first_frame_hash = None
    last_frame_hash = None
    first_block = hashlib.sha256()
    tail: list[bytes] = []
    count = 0
    stderr = ""
    return_code = -1
    try:
        while True:
            frame = process.stdout.read(frame_bytes)
            if not frame:
                break
            if len(frame) != frame_bytes:
                process.kill()
                raise RuntimeError(f"ffmpeg returned a partial frame for {path}")
            frame_digest = hashlib.sha256(frame).digest()
            digest = frame_digest.hex()
            if first_frame_hash is None:
                first_frame_hash = digest
            last_frame_hash = digest
            if count < boundary_frames:
                first_block.update(frame_digest)
            tail.append(frame_digest)
            if len(tail) > boundary_frames:
                tail.pop(0)
            count += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    finally:
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed while decoding {path}: {stderr.strip()}")
    if count == 0:
        raise ValueError(f"video has no decoded frames: {path}")
    if expected_frames and count != expected_frames:
        raise ValueError(f"decoded {count} frames but ffprobe reported {expected_frames}: {path}")
    if len(tail) != boundary_frames:
        raise ValueError(f"video has fewer than {boundary_frames} boundary frames")
    last_block = hashlib.sha256()
    for frame_digest in tail:
        last_block.update(frame_digest)
    return {
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": _ratio(stream["avg_frame_rate"]),
        "duration_seconds": float(probe["format"]["duration"]),
        "codec_name": stream.get("codec_name"),
        "codec_long_name": stream.get("codec_long_name"),
        "profile": stream.get("profile"),
        "level": stream.get("level"),
        "pixel_format": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "has_b_frames": int(stream.get("has_b_frames", 0)),
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "audio_streams": [
            {
                "index": item.get("index"),
                "codec_name": item.get("codec_name"),
                "profile": item.get("profile"),
                "sample_format": item.get("sample_fmt"),
                "sample_rate": item.get("sample_rate"),
                "channels": item.get("channels"),
                "channel_layout": item.get("channel_layout"),
            }
            for item in audio_streams
        ],
        **frame_facts,
        "first_frame_sha256": first_frame_hash,
        "last_frame_sha256": last_frame_hash,
        "first_block_sha256": first_block.hexdigest(),
        "last_block_sha256": last_block.hexdigest(),
    }


def validate_video(
    path: str | Path,
    boundary_frames: int,
    *,
    expectations: VideoValidationExpectations | None = None,
) -> dict:
    hashes = video_frame_hashes(path, boundary_frames)
    checks = {
        "matching_first_last_frame": hashes["first_frame_sha256"] == hashes["last_frame_sha256"],
        "matching_first_last_block": hashes["first_block_sha256"] == hashes["last_block_sha256"],
        "h264_compatible_dimensions": hashes["width"] % 2 == 0 and hashes["height"] % 2 == 0,
        "frame_type_count_matches_decoded": hashes["probed_frame_count"] == hashes["frame_count"],
    }
    if expectations is not None:
        if expectations.width is not None:
            checks["width_matches_contract"] = hashes["width"] == expectations.width
        if expectations.height is not None:
            checks["height_matches_contract"] = hashes["height"] == expectations.height
        if expectations.frame_count is not None:
            checks["frame_count_matches_contract"] = (
                hashes["frame_count"] == expectations.frame_count
            )
        if expectations.duration_seconds is not None:
            checks["duration_matches_contract"] = math.isclose(
                hashes["duration_seconds"],
                expectations.duration_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        if expectations.fps is not None:
            checks["fps_matches_contract"] = math.isclose(
                hashes["fps"], expectations.fps, rel_tol=0.0, abs_tol=1e-9
            )
        if expectations.codec_name is not None:
            checks["codec_matches_contract"] = hashes["codec_name"] == expectations.codec_name
        if expectations.pixel_format is not None:
            checks["pixel_format_matches_contract"] = (
                hashes["pixel_format"] == expectations.pixel_format
            )
        if expectations.color_space is not None:
            checks["color_space_matches_contract"] = (
                hashes["color_space"] == expectations.color_space
            )
        if expectations.color_primaries is not None:
            checks["color_primaries_match_contract"] = (
                hashes["color_primaries"] == expectations.color_primaries
            )
        if expectations.color_transfer is not None:
            checks["color_transfer_matches_contract"] = (
                hashes["color_transfer"] == expectations.color_transfer
            )
        if expectations.audio_stream_count is not None:
            checks["audio_stream_count_matches_contract"] = (
                hashes["audio_stream_count"] == expectations.audio_stream_count
            )
        if expectations.all_intra is not None:
            checks["frame_types_match_contract"] = hashes["all_intra"] is expectations.all_intra
        if expectations.allowed_profiles is not None:
            checks["profile_matches_contract"] = hashes["profile"] in expectations.allowed_profiles
    return {
        "schema_version": 1,
        "asset": str(path),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": hashes,
        "expectations": expectations.to_dict() if expectations is not None else None,
    }


def validate_library(
    video_paths: list[str | Path],
    boundary_frames: int,
    *,
    expectations: (
        VideoValidationExpectations
        | Sequence[VideoValidationExpectations]
        | None
    ) = None,
) -> dict:
    if not video_paths:
        raise ValueError("video_paths must contain at least one asset")
    if expectations is None or isinstance(expectations, VideoValidationExpectations):
        asset_expectations = [expectations] * len(video_paths)
        serialized_expectations: dict | list[dict] | None = (
            expectations.to_dict() if expectations is not None else None
        )
    else:
        asset_expectations = list(expectations)
        if len(asset_expectations) != len(video_paths):
            raise ValueError(
                "per-asset expectations must have the same length as video_paths"
            )
        if not all(
            isinstance(item, VideoValidationExpectations)
            for item in asset_expectations
        ):
            raise TypeError(
                "per-asset expectations must contain VideoValidationExpectations"
            )
        serialized_expectations = [
            item.to_dict() for item in asset_expectations
        ]
    reports = [
        validate_video(path, boundary_frames, expectations=expected)
        for path, expected in zip(video_paths, asset_expectations)
    ]
    first_blocks = {report["metrics"]["first_block_sha256"] for report in reports}
    last_blocks = {report["metrics"]["last_block_sha256"] for report in reports}
    all_ordered_pairs_match = all(
        outgoing["metrics"]["last_block_sha256"]
        == incoming["metrics"]["first_block_sha256"]
        for outgoing in reports
        for incoming in reports
    )
    formats = {
        (
            report["metrics"]["width"],
            report["metrics"]["height"],
            report["metrics"]["fps"],
            report["metrics"]["codec_name"],
            report["metrics"]["profile"],
            report["metrics"]["level"],
            report["metrics"]["pixel_format"],
            report["metrics"]["color_range"],
            report["metrics"]["color_space"],
            report["metrics"]["color_transfer"],
            report["metrics"]["color_primaries"],
            report["metrics"]["video_stream_count"],
            report["metrics"]["audio_stream_count"],
            tuple(
                tuple(sorted(stream.items()))
                for stream in report["metrics"]["audio_streams"]
            ),
            report["metrics"]["has_b_frames"],
            report["metrics"]["all_intra"],
            tuple(sorted(report["metrics"]["picture_types"])),
        )
        for report in reports
    }
    checks = {
        "all_assets_pass_individually": all(report["passed"] for report in reports),
        "shared_opening_block": len(first_blocks) == 1,
        "shared_ending_block": len(last_blocks) == 1,
        "opening_equals_ending": first_blocks == last_blocks,
        "all_ordered_pairs_switch_compatible": all_ordered_pairs_match,
        "matching_video_formats": len(formats) == 1,
    }
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "assets": reports,
        "expectations": serialized_expectations,
    }


def save_validation_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
