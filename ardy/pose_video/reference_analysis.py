# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, model-free analysis of motion-reference videos.

This module deliberately avoids pose-estimation dependencies.  It measures
decoded luma changes in fixed regions of interest (ROIs), identifies coarse
activity extrema, and emits a timing-only behavior template.  The output is a
useful seed for ARDY pose authoring, but it must not be interpreted as measured
head yaw/pitch/roll or as a semantic action classifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


REPORT_SCHEMA_VERSION = "ardy.reference-analysis.v1"
_BEHAVIORS = {"auto", "idle", "nod", "look_away", "generic"}


@dataclass(frozen=True)
class ReferenceAnalysisConfig:
    """Configuration for deterministic reference-video analysis.

    ROI coordinates use ``(x_min, y_min, x_max, y_max)`` normalized to the
    decoded analysis frame.  Defaults target a centered, portrait video-call
    composition while remaining useful for synthetic and landscape clips.
    """

    analysis_width: int = 192
    boundary_seconds: float = 2.0
    smoothing_seconds: float = 0.12
    spatial_blur_sigma: float = 0.8
    head_roi: tuple[float, float, float, float] = (0.16, 0.04, 0.84, 0.47)
    upper_body_roi: tuple[float, float, float, float] = (0.07, 0.08, 0.93, 0.78)
    active_pixel_percentile: float = 70.0
    event_threshold_fraction: float = 0.15
    hold_threshold_fraction: float = 0.82
    max_extrema: int = 8

    def validate(self) -> None:
        if self.analysis_width < 32:
            raise ValueError("analysis_width must be at least 32 pixels")
        if self.boundary_seconds < 0:
            raise ValueError("boundary_seconds must be non-negative")
        if self.smoothing_seconds < 0:
            raise ValueError("smoothing_seconds must be non-negative")
        if self.spatial_blur_sigma < 0:
            raise ValueError("spatial_blur_sigma must be non-negative")
        if not 0 <= self.active_pixel_percentile <= 100:
            raise ValueError("active_pixel_percentile must be in [0, 100]")
        if not 0 < self.event_threshold_fraction < 1:
            raise ValueError("event_threshold_fraction must be in (0, 1)")
        if not 0 < self.hold_threshold_fraction <= 1:
            raise ValueError("hold_threshold_fraction must be in (0, 1]")
        if self.max_extrema < 1:
            raise ValueError("max_extrema must be positive")
        for name, roi in (("head_roi", self.head_roi), ("upper_body_roi", self.upper_body_roi)):
            if len(roi) != 4:
                raise ValueError(f"{name} must have four values")
            x0, y0, x1, y1 = roi
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"{name} must be normalized and non-empty")


@dataclass(frozen=True)
class VideoProbe:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    declared_frame_count: int | None
    codec_name: str | None
    pixel_format: str | None
    has_audio: bool


def infer_behavior(source_name: str) -> str:
    """Infer a coarse behavior label from a filename, never from pixels."""

    normalized = source_name.lower().replace("-", "_").replace(" ", "_")
    if "nod" in normalized:
        return "nod"
    if "look" in normalized or "away" in normalized or "glance" in normalized:
        return "look_away"
    if "idle" in normalized or "neutral" in normalized or "rest" in normalized:
        return "idle"
    return "generic"


def _parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return float(value)


def probe_video(path: str | Path) -> VideoProbe:
    """Read video metadata with ffprobe."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reference video does not exist: {source}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for reference-video analysis") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"Could not probe {source}: {detail}") from exc
    payload = json.loads(completed.stdout)
    video_stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {source}")
    fps = _parse_rate(video_stream.get("avg_frame_rate")) or _parse_rate(video_stream.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError(f"Could not determine a positive frame rate for {source}")
    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    declared_count_raw = video_stream.get("nb_frames")
    declared_count = int(declared_count_raw) if declared_count_raw not in (None, "N/A") else None
    return VideoProbe(
        path=str(source),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        duration_seconds=duration,
        declared_frame_count=declared_count,
        codec_name=video_stream.get("codec_name"),
        pixel_format=video_stream.get("pix_fmt"),
        has_audio=any(item.get("codec_type") == "audio" for item in payload.get("streams", [])),
    )


def _analysis_dimensions(width: int, height: int, target_width: int) -> tuple[int, int]:
    scaled_width = min(width, target_width)
    scaled_height = max(2, int(round(height * scaled_width / width)))
    # Fixed even dimensions make raw-video decoding and future encoder reuse predictable.
    scaled_width -= scaled_width % 2
    scaled_height -= scaled_height % 2
    return max(2, scaled_width), max(2, scaled_height)


def decode_luma_frames(path: str | Path, config: ReferenceAnalysisConfig) -> tuple[np.ndarray, VideoProbe]:
    """Decode all frames to a deterministic, reduced-resolution luma array."""

    config.validate()
    probe = probe_video(path)
    width, height = _analysis_dimensions(probe.width, probe.height, config.analysis_width)
    vf = f"scale={width}:{height}:flags=bicubic,format=gray"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        probe.path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        vf,
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for reference-video analysis") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Could not decode {probe.path}: {detail}") from exc
    frame_bytes = width * height
    if not completed.stdout or len(completed.stdout) % frame_bytes:
        raise RuntimeError(
            f"Unexpected decoded byte count for {probe.path}: {len(completed.stdout)} is not divisible by {frame_bytes}"
        )
    frames = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, height, width).copy()
    if len(frames) < 2:
        raise ValueError(f"Reference video must contain at least two frames: {probe.path}")
    return frames, probe


def _roi_slices(roi: Sequence[float], height: int, width: int) -> tuple[slice, slice]:
    x0, y0, x1, y1 = roi
    left = min(width - 1, max(0, int(np.floor(x0 * width))))
    top = min(height - 1, max(0, int(np.floor(y0 * height))))
    right = min(width, max(left + 1, int(np.ceil(x1 * width))))
    bottom = min(height, max(top + 1, int(np.ceil(y1 * height))))
    return slice(top, bottom), slice(left, right)


def _roi_description(roi: Sequence[float], height: int, width: int) -> dict[str, Any]:
    rows, cols = _roi_slices(roi, height, width)
    return {
        "normalized_xyxy": [round(float(value), 6) for value in roi],
        "analysis_pixels_xyxy": [cols.start, rows.start, cols.stop, rows.stop],
    }


def _mean_abs_over_roi(values: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    rows, cols = _roi_slices(roi, values.shape[-2], values.shape[-1])
    return np.mean(np.abs(values[..., rows, cols]), axis=(-2, -1), dtype=np.float64)


def _round_float(value: float, places: int = 8) -> float:
    return round(float(value), places)


def _series(values: np.ndarray, places: int = 8) -> list[float | None]:
    return [None if not np.isfinite(value) else round(float(value), places) for value in values]


def _trace_summary(values: np.ndarray, fps: float) -> dict[str, Any]:
    maximum = int(np.argmax(values))
    return {
        "mean": _round_float(np.mean(values)),
        "median": _round_float(np.median(values)),
        "p95": _round_float(np.percentile(values, 95)),
        "max": _round_float(values[maximum]),
        "max_frame": maximum,
        "max_time_seconds": _round_float(maximum / fps, 6),
    }


def _effective_boundary_frames(frame_count: int, fps: float, requested_seconds: float) -> int:
    requested = int(round(requested_seconds * fps))
    # Preserve at least 60% of a short clip as analyzable interior.
    maximum = max(1, frame_count // 5)
    return min(maximum, max(1, requested)) if requested_seconds > 0 else 1


def _active_pixel_mask(temporal_abs: np.ndarray, roi: Sequence[float], percentile: float) -> np.ndarray:
    """Build a heuristic activity mask; this is not semantic foreground segmentation."""

    # RMS retains short movements that a temporal percentile can entirely miss
    # (for example, a feature that changes on only four frames in a long clip).
    activity = np.sqrt(np.mean(np.square(temporal_abs), axis=0, dtype=np.float64))
    rows, cols = _roi_slices(roi, activity.shape[0], activity.shape[1])
    roi_activity = activity[rows, cols]
    threshold = float(np.percentile(roi_activity, percentile))
    mask = np.zeros_like(activity, dtype=bool)
    selected = roi_activity >= threshold
    # Uniform synthetic/static clips can otherwise mark the entire ROI as foreground.
    if threshold <= np.finfo(np.float32).eps:
        selected = roi_activity > 0
    mask[rows, cols] = selected
    return mask


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = int(mask.sum())
    if count == 0:
        return np.zeros(values.shape[0], dtype=np.float64)
    return np.mean(np.abs(values[:, mask]), axis=1, dtype=np.float64)


def _change_centroid(deviation_abs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    weights = deviation_abs * mask[None, ...]
    mass = np.sum(weights, axis=(1, 2), dtype=np.float64)
    x = np.full(len(weights), np.nan, dtype=np.float64)
    y = np.full(len(weights), np.nan, dtype=np.float64)
    valid = mass > np.finfo(np.float32).eps
    x[valid] = np.sum(weights[valid] * x_grid, axis=(1, 2), dtype=np.float64) / mass[valid] / max(width - 1, 1)
    y[valid] = np.sum(weights[valid] * y_grid, axis=(1, 2), dtype=np.float64) / mass[valid] / max(height - 1, 1)
    return x, y, mass / max(int(mask.sum()), 1)


def _find_run_around_peak(values: np.ndarray, peak: int, threshold: float, low: int, high: int) -> tuple[int, int]:
    start = peak
    while start > low and values[start - 1] >= threshold:
        start -= 1
    end = peak
    while end + 1 < high and values[end + 1] >= threshold:
        end += 1
    return start, end


def _extract_event(
    smoothed_deviation: np.ndarray,
    smoothed_motion: np.ndarray,
    fps: float,
    boundary_frames: int,
    config: ReferenceAnalysisConfig,
) -> dict[str, Any]:
    frame_count = len(smoothed_deviation)
    low = min(boundary_frames, frame_count - 1)
    high = max(low + 1, frame_count - boundary_frames)
    interior = smoothed_deviation[low:high]
    floor = float(np.percentile(interior, 10))
    peak = low + int(np.argmax(interior))
    amplitude = max(0.0, float(smoothed_deviation[peak]) - floor)
    threshold = floor + config.event_threshold_fraction * amplitude
    start, end = _find_run_around_peak(smoothed_deviation, peak, threshold, low, high)
    hold_threshold = floor + config.hold_threshold_fraction * amplitude
    hold_start, hold_end = _find_run_around_peak(smoothed_deviation, peak, hold_threshold, start, end + 1)

    boundary_values = np.concatenate(
        (smoothed_deviation[:boundary_frames], smoothed_deviation[-boundary_frames:]),
    )
    noise = 1.4826 * float(np.median(np.abs(boundary_values - np.median(boundary_values))))
    motion_interior = smoothed_motion[low:high]
    motion_noise = 1.4826 * float(np.median(np.abs(motion_interior - np.median(motion_interior))))
    prominence_threshold = max(motion_noise * 0.75, float(np.ptp(motion_interior)) * 0.04, 1e-8)
    distance = max(1, int(round(0.18 * fps)))
    peak_indices, _ = find_peaks(smoothed_motion[low:high], distance=distance, prominence=prominence_threshold)
    peak_indices = peak_indices + low
    if len(peak_indices):
        prominences = peak_prominences(smoothed_motion, peak_indices)[0]
        chosen = np.argsort(prominences)[-config.max_extrema :]
        extrema = sorted(
            (
                {
                    "frame": int(peak_indices[index]),
                    "time_seconds": _round_float(peak_indices[index] / fps, 6),
                    "motion_value": _round_float(smoothed_motion[peak_indices[index]]),
                    "prominence": _round_float(prominences[index]),
                }
                for index in chosen
            ),
            key=lambda item: item["frame"],
        )
    else:
        extrema = []

    confidence_ratio = amplitude / max(noise, 1e-8)
    if confidence_ratio >= 8:
        timing_confidence = "high"
    elif confidence_ratio >= 3:
        timing_confidence = "medium"
    else:
        timing_confidence = "low"
    return {
        "analysis_window": {
            "start_frame": low,
            "end_frame_inclusive": high - 1,
            "start_time_seconds": _round_float(low / fps, 6),
            "end_time_seconds": _round_float((high - 1) / fps, 6),
        },
        "activity_start": {"frame": start, "time_seconds": _round_float(start / fps, 6)},
        "peak": {
            "frame": peak,
            "time_seconds": _round_float(peak / fps, 6),
            "baseline_deviation": _round_float(smoothed_deviation[peak]),
        },
        "hold": {
            "start_frame": hold_start,
            "end_frame_inclusive": hold_end,
            "start_time_seconds": _round_float(hold_start / fps, 6),
            "end_time_seconds": _round_float(hold_end / fps, 6),
        },
        "activity_end": {"frame": end, "time_seconds": _round_float(end / fps, 6)},
        "deviation_floor": _round_float(floor),
        "deviation_amplitude": _round_float(amplitude),
        "canonical_boundary_noise_estimate": _round_float(noise),
        "motion_prominence_threshold": _round_float(prominence_threshold),
        "timing_confidence": timing_confidence,
        "motion_extrema": extrema,
    }


def _template_for_behavior(
    behavior: str,
    frame_count: int,
    fps: float,
    boundary_frames: int,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    entry_end = max(0, boundary_frames - 1)
    exit_start = min(frame_count - 1, frame_count - boundary_frames)
    peak = int(event["peak"]["frame"])
    start = max(entry_end, int(event["activity_start"]["frame"]))
    end = min(exit_start, int(event["activity_end"]["frame"]))

    if behavior == "idle":
        behavior_id = "neutral_resting"
        behavior_type = "loop"
        target = {"kind": "micro_motion", "joint_angles": None}
        face = {"blink": "author_separately", "expression": "neutral"}
    elif behavior == "nod":
        behavior_id = "nod_agree"
        behavior_type = "one_shot"
        target = {"kind": "head_neck_rotation", "primary_axis": "pitch", "joint_angles": None}
        face = {"expression": "neutral_or_agreeing"}
    elif behavior == "look_away":
        behavior_id = "look_away_reset"
        behavior_type = "one_shot"
        target = {"kind": "head_neck_rotation", "primary_axis": "yaw", "joint_angles": None}
        face = {"gaze": "author_separately_before_head_turn", "expression": "neutral"}
    else:
        behavior_id = "reference_behavior"
        behavior_type = "one_shot"
        target = {"kind": "unclassified_motion", "joint_angles": None}
        face = {"author_separately": True}

    if behavior == "idle":
        interior_length = max(0, exit_start - entry_end)
        roles = [
            ("canonical_entry_end", entry_end),
            ("loop_phase_quarter", entry_end + round(interior_length * 0.25)),
            ("loop_phase_half", entry_end + round(interior_length * 0.50)),
            ("loop_phase_three_quarter", entry_end + round(interior_length * 0.75)),
            ("canonical_exit_start", exit_start),
        ]
    else:
        roles = [
            ("canonical_entry_end", entry_end),
            ("activity_start", start),
            ("peak_pose", peak),
            ("activity_end", end),
            ("canonical_exit_start", exit_start),
        ]
    keyframes: list[dict[str, Any]] = []
    for role, frame in roles:
        if keyframes and frame < keyframes[-1]["frame"]:
            frame = keyframes[-1]["frame"]
        keyframes.append({"role": role, "frame": frame, "time_seconds": _round_float(frame / fps, 6)})

    return {
        "behavior_id": behavior_id,
        "behavior_type": behavior_type,
        "measurement_scope": "timing_seed_only",
        "target": target,
        "face_intent": face,
        "canonical_boundary": {
            "entry_frames_inclusive": [0, entry_end],
            "exit_frames_inclusive": [exit_start, frame_count - 1],
            "copy_exact_motion_arrays_during_asset_build": True,
        },
        "suggested_keyframes": keyframes,
        "authoring_requirements": [
            "Calibrate anatomical joint rotations in the ARDY pose editor; pixels do not provide joint angles here.",
            "Lock the seated root, lower body, camera, and resting limbs unless the behavior explicitly changes them.",
            "Copy canonical motion frames exactly, then validate pose and velocity continuity after final rendering.",
        ],
    }


def analyze_frames(
    frames: np.ndarray,
    fps: float,
    *,
    behavior: str = "generic",
    source_name: str = "in_memory",
    config: ReferenceAnalysisConfig | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze decoded grayscale or RGB frames and return a JSON-safe report.

    Args:
        frames: ``(frames, height, width)`` luma or ``(frames, height, width,
            channels)`` RGB-like uint8/float frames.
        fps: Constant analysis frame rate in frames per second.
        behavior: One of ``idle``, ``nod``, ``look_away``, ``generic``, or
            ``auto``. ``auto`` only inspects ``source_name``.
    """

    config = config or ReferenceAnalysisConfig()
    config.validate()
    if behavior not in _BEHAVIORS:
        raise ValueError(f"behavior must be one of {sorted(_BEHAVIORS)}")
    if behavior == "auto":
        behavior = infer_behavior(source_name)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite value")
    values = np.asarray(frames)
    if values.ndim == 4:
        if values.shape[-1] < 3:
            raise ValueError("4-D frame arrays must have at least three color channels")
        values = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    if values.ndim != 3:
        raise ValueError("frames must have shape (frames, height, width) or (frames, height, width, channels)")
    if values.shape[0] < 2 or values.shape[1] < 4 or values.shape[2] < 4:
        raise ValueError("at least two frames of size 4x4 are required")
    values = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("frames contain non-finite values")
    scale = 255.0 if float(np.max(values)) > 1.5 else 1.0
    luma = np.clip(values / scale, 0.0, 1.0)
    if config.spatial_blur_sigma:
        luma = gaussian_filter(luma, sigma=(0.0, config.spatial_blur_sigma, config.spatial_blur_sigma), mode="nearest")

    frame_count, height, width = luma.shape
    boundary_frames = _effective_boundary_frames(frame_count, fps, config.boundary_seconds)
    canonical_samples = np.concatenate((luma[:boundary_frames], luma[-boundary_frames:]), axis=0)
    baseline = np.median(canonical_samples, axis=0).astype(np.float32)
    deviation_abs = np.abs(luma - baseline[None, ...])
    temporal_delta = np.empty_like(luma)
    temporal_delta[0] = 0.0
    temporal_delta[1:] = luma[1:] - luma[:-1]
    temporal_abs = np.abs(temporal_delta)

    full_roi = (0.0, 0.0, 1.0, 1.0)
    global_motion = _mean_abs_over_roi(temporal_delta, full_roi)
    head_motion = _mean_abs_over_roi(temporal_delta, config.head_roi)
    upper_motion = _mean_abs_over_roi(temporal_delta, config.upper_body_roi)
    head_deviation = _mean_abs_over_roi(deviation_abs, config.head_roi)
    upper_deviation = _mean_abs_over_roi(deviation_abs, config.upper_body_roi)
    active_mask = _active_pixel_mask(temporal_abs, config.upper_body_roi, config.active_pixel_percentile)
    active_motion = _masked_mean(temporal_delta, active_mask)
    centroid_x, centroid_y, centroid_mass = _change_centroid(deviation_abs, active_mask)

    sigma_frames = config.smoothing_seconds * fps
    if sigma_frames > 0:
        smooth_head_deviation = gaussian_filter1d(head_deviation, sigma=sigma_frames, mode="nearest")
        smooth_head_motion = gaussian_filter1d(head_motion, sigma=sigma_frames, mode="nearest")
    else:
        smooth_head_deviation = head_deviation.copy()
        smooth_head_motion = head_motion.copy()
    event = _extract_event(smooth_head_deviation, smooth_head_motion, fps, boundary_frames, config)

    first_last_delta = luma[0] - luma[-1]
    incoming_velocity = luma[-1] - luma[-2]
    outgoing_velocity = luma[1] - luma[0]
    velocity_cut_error = np.mean(np.abs(outgoing_velocity - incoming_velocity), dtype=np.float64)
    peak_frame = int(event["peak"]["frame"])
    boundary = {
        "scope": "analysis_resolution_luma",
        "effective_boundary_frames": boundary_frames,
        "effective_boundary_seconds": _round_float(boundary_frames / fps, 6),
        "first_last_mean_absolute_error": _round_float(np.mean(np.abs(first_last_delta), dtype=np.float64)),
        "first_last_max_absolute_error": _round_float(np.max(np.abs(first_last_delta))),
        "first_last_exact_at_analysis_resolution": bool(np.array_equal(luma[0], luma[-1])),
        "cut_velocity_mean_absolute_error": _round_float(velocity_cut_error),
        "first_frame_sha256": hashlib.sha256(np.ascontiguousarray(frames[0]).tobytes()).hexdigest(),
        "last_frame_sha256": hashlib.sha256(np.ascontiguousarray(frames[-1]).tobytes()).hexdigest(),
        "note": "Exact production validation must be repeated on full-resolution decoded final assets.",
    }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "name": source_name,
            "metadata": dict(source_metadata or {}),
        },
        "analysis": {
            "behavior_hint": behavior,
            "behavior_hint_source": "filename_or_cli_not_pixel_classification",
            "fps": _round_float(fps, 8),
            "frame_count": frame_count,
            "duration_seconds": _round_float(frame_count / fps, 6),
            "analysis_width": width,
            "analysis_height": height,
            "config": asdict(config),
            "regions": {
                "head": _roi_description(config.head_roi, height, width),
                "upper_body": _roi_description(config.upper_body_roi, height, width),
                "activity_mask": {
                    "method": "upper_body_temporal_activity_percentile",
                    "active_pixel_count": int(active_mask.sum()),
                    "active_pixel_fraction_of_frame": _round_float(np.mean(active_mask)),
                    "is_semantic_foreground_segmentation": False,
                },
            },
        },
        "boundary_continuity": boundary,
        "motion_summary": {
            "global_temporal_energy": _trace_summary(global_motion, fps),
            "head_temporal_energy": _trace_summary(head_motion, fps),
            "upper_body_temporal_energy": _trace_summary(upper_motion, fps),
            "heuristic_active_region_temporal_energy": _trace_summary(active_motion, fps),
            "head_canonical_deviation": _trace_summary(head_deviation, fps),
            "upper_body_canonical_deviation": _trace_summary(upper_deviation, fps),
            "peak_change_centroid_normalized_xy": [
                None if not np.isfinite(centroid_x[peak_frame]) else _round_float(centroid_x[peak_frame], 6),
                None if not np.isfinite(centroid_y[peak_frame]) else _round_float(centroid_y[peak_frame], 6),
            ],
            "change_centroid_definition": (
                "centroid of absolute luma change from the canonical median, masked by activity; not a head center"
            ),
        },
        "detected_event": event,
        "behavior_template": _template_for_behavior(behavior, frame_count, fps, boundary_frames, event),
        "timeseries": {
            "frame": list(range(frame_count)),
            "time_seconds": [_round_float(index / fps, 6) for index in range(frame_count)],
            "global_temporal_energy": _series(global_motion),
            "head_temporal_energy": _series(head_motion),
            "upper_body_temporal_energy": _series(upper_motion),
            "active_region_temporal_energy": _series(active_motion),
            "head_canonical_deviation": _series(head_deviation),
            "head_canonical_deviation_smoothed": _series(smooth_head_deviation),
            "change_centroid_x_normalized": _series(centroid_x, 6),
            "change_centroid_y_normalized": _series(centroid_y, 6),
            "change_mass_per_active_pixel": _series(centroid_mass),
        },
        "limitations": [
            "This model-free analysis estimates visual-change timing, not anatomical joint rotations or 3D pose.",
            (
                "The fixed portrait ROIs must be adjusted if the subject is not centered or the crop differs "
                "substantially."
            ),
            (
                "Camera movement, lighting changes, compression artifacts, and facial motion can be mistaken for "
                "body motion."
            ),
            (
                "Eye gaze, blinks, smiles, eyebrow motion, and expression intensity require a facial tracker or "
                "manual authoring."
            ),
            "The behavior label comes from the CLI or filename; it is not inferred semantically from pixels.",
        ],
    }
    return report


def analyze_video(
    path: str | Path,
    *,
    behavior: str = "auto",
    config: ReferenceAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Decode and analyze one reference video."""

    config = config or ReferenceAnalysisConfig()
    frames, probe = decode_luma_frames(path, config)
    metadata = asdict(probe)
    report = analyze_frames(
        frames,
        probe.fps,
        behavior=behavior,
        source_name=Path(probe.path).name,
        config=config,
        source_metadata=metadata,
    )
    report["source"]["sha256"] = _file_sha256(Path(probe.path))
    report["source"]["decoded_frame_count_matches_probe"] = (
        None if probe.declared_frame_count is None else len(frames) == probe.declared_frame_count
    )
    return report


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary_signature(frames: np.ndarray, boundary_frames: int) -> dict[str, str]:
    count = min(boundary_frames, len(frames))
    return {
        "first_frame_sha256": hashlib.sha256(np.ascontiguousarray(frames[0]).tobytes()).hexdigest(),
        "last_frame_sha256": hashlib.sha256(np.ascontiguousarray(frames[-1]).tobytes()).hexdigest(),
        "entry_block_sha256": hashlib.sha256(np.ascontiguousarray(frames[:count]).tobytes()).hexdigest(),
        "exit_block_sha256": hashlib.sha256(np.ascontiguousarray(frames[-count:]).tobytes()).hexdigest(),
    }


def analyze_reference_set(
    paths: Iterable[str | Path],
    *,
    behavior_overrides: Mapping[str, str] | None = None,
    config: ReferenceAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyze videos and compare their reduced-resolution canonical blocks."""

    config = config or ReferenceAnalysisConfig()
    resolved_paths = [Path(path).expanduser().resolve() for path in paths]
    if not resolved_paths:
        raise ValueError("At least one reference video is required")
    overrides = dict(behavior_overrides or {})
    reports: list[dict[str, Any]] = []
    signatures: dict[str, dict[str, str]] = {}
    for path in resolved_paths:
        frames, probe = decode_luma_frames(path, config)
        behavior = overrides.get(str(path), overrides.get(path.name, "auto"))
        report = analyze_frames(
            frames,
            probe.fps,
            behavior=behavior,
            source_name=path.name,
            config=config,
            source_metadata=asdict(probe),
        )
        report["source"]["sha256"] = _file_sha256(path)
        report["source"]["decoded_frame_count_matches_probe"] = (
            None if probe.declared_frame_count is None else len(frames) == probe.declared_frame_count
        )
        reports.append(report)
        signatures[path.name] = _boundary_signature(
            frames,
            int(report["boundary_continuity"]["effective_boundary_frames"]),
        )

    fields = ("first_frame_sha256", "last_frame_sha256", "entry_block_sha256", "exit_block_sha256")
    all_match = {field: len({signature[field] for signature in signatures.values()}) == 1 for field in fields}
    cross_cut_frame_match = len(
        {signature[side] for signature in signatures.values() for side in ("first_frame_sha256", "last_frame_sha256")}
    ) == 1
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "reference_set",
        "reports": reports,
        "canonical_comparison": {
            "scope": "reduced_resolution_decoded_luma",
            "signatures": signatures,
            "all_videos_match": all_match,
            "all_first_and_last_frames_cross_match": cross_cut_frame_match,
            "production_note": "Re-run full-resolution decoded-frame hashing on every final Kling/MuseTalk asset.",
        },
    }


def write_json_report(report: Mapping[str, Any], output: str | Path) -> Path:
    """Atomically write a report without partially replacing an existing file."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
