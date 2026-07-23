# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Core NPZ to portrait MP4 pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Sequence

import numpy as np

from .config import CameraConfig, RenderStyle
from .encoder import FFmpegH264Encoder, sha256_file
from .motion import load_core_motion, resample_core_motion
from .renderer import CoreMeshRenderer


ProgressCallback = Callable[[int, int], None]


def _require_exact_motion_boundaries(motion, boundary_frames: int) -> None:
    if boundary_frames <= 0:
        return
    if motion.num_frames <= boundary_frames * 2:
        raise ValueError(
            f"motion has {motion.num_frames} frames; an exact {boundary_frames}-frame "
            "boundary requires a non-empty interior"
        )
    for name, values in (
        ("global rotations", motion.global_rot_mats),
        ("posed joints", motion.posed_joints),
    ):
        if not np.array_equal(values[:boundary_frames], values[-boundary_frames:]):
            raise ValueError(f"motion {name} do not have exact matching boundary blocks")


def _validate_canonical_rgb_frames(
    frames: Sequence[np.ndarray], camera: CameraConfig, boundary_frames: int
) -> tuple[np.ndarray, ...]:
    if len(frames) != boundary_frames:
        raise ValueError(
            f"canonical RGB cache has {len(frames)} frames; expected {boundary_frames}"
        )
    expected_shape = (camera.height, camera.width, 3)
    validated: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        value = np.asarray(frame)
        if value.shape != expected_shape or value.dtype != np.uint8:
            raise ValueError(
                f"canonical RGB frame {index} must be uint8 {expected_shape}, "
                f"got {value.dtype} {value.shape}"
            )
        validated.append(np.ascontiguousarray(value))
    if validated and not np.array_equal(validated[0], validated[-1]):
        raise ValueError("canonical RGB cache must have identical first and last frames")
    return tuple(validated)


def render_canonical_boundary_frames(
    input_path: str | Path,
    *,
    spec: "VideoExportSpec | None" = None,
    camera: CameraConfig | None = None,
    renderer: CoreMeshRenderer | None = None,
) -> tuple[np.ndarray, ...]:
    """Render one canonical RGB block for reuse across an entire library.

    EGL can vary a few edge pixels by one least-significant bit when an
    identical mesh is rasterized twice.  Rendering the canonical block once
    and reusing its raw frames makes the stronger cross-asset decoded-pixel
    guarantee possible without weakening image quality or tolerances.
    """

    resolved = spec or VideoExportSpec()
    resolved_camera = camera or resolved.camera
    if resolved.verify_boundary_frames <= 0:
        raise ValueError("spec.verify_boundary_frames must be positive")
    motion = resample_core_motion(load_core_motion(Path(input_path).resolve()), resolved.target_fps)
    boundary_frames = resolved.verify_boundary_frames
    _require_exact_motion_boundaries(motion, boundary_frames)
    if renderer is not None and (
        renderer.camera != resolved_camera or renderer.style != resolved.style
    ):
        raise ValueError("shared renderer camera/style do not match the export specification")
    frames: list[np.ndarray] = []
    renderer_scope = (
        nullcontext(renderer)
        if renderer is not None
        else CoreMeshRenderer(camera=resolved_camera, style=resolved.style)
    )
    with renderer_scope as active_renderer:
        for index, vertices in enumerate(
            active_renderer.skin_vertices(motion, chunk_size=resolved.skinning_chunk_size)
        ):
            if index >= boundary_frames:
                break
            if index == boundary_frames - 1:
                # The motion block starts and ends at the same exact pose. Use
                # its first raster again because some EGL drivers can vary a
                # few edge pixels by one LSB between identical draws.
                frames.append(frames[0])
            else:
                frames.append(active_renderer.render_vertices(vertices).copy())
    return tuple(frames)


@dataclass(frozen=True)
class VideoExportSpec:
    """Reusable master-asset rendering contract.

    The default verifies the first/last two seconds (60 frames at 30 fps) on
    decoded pixels. Use :func:`render_motion_to_mp4` directly for short
    diagnostic clips that do not yet carry canonical boundary blocks.
    """

    target_fps: float = 30.0
    camera: CameraConfig = field(default_factory=CameraConfig.video_call_portrait)
    style: RenderStyle = field(default_factory=RenderStyle)
    # All-intra CRF rate control can still choose different quantizers for
    # identical frames at different offsets. Lossless x264 preserves identical
    # decoded YUV pixels and is therefore the master-asset default.
    crf: int = 0
    preset: str = "medium"
    all_intra: bool = True
    verify_boundary_frames: int = 60
    skinning_chunk_size: int = 32

    def __post_init__(self) -> None:
        if self.verify_boundary_frames < 0:
            raise ValueError("verify_boundary_frames cannot be negative")
        if self.skinning_chunk_size <= 0:
            raise ValueError("skinning_chunk_size must be positive")


@dataclass(frozen=True)
class RenderResult:
    input_path: Path
    output_path: Path
    source_frames: int
    source_fps: float
    output_frames: int
    output_fps: float
    duration_seconds: float
    input_sha256: str
    raw_rgb_sha256: str
    mp4_sha256: str
    verified_boundary_frames: int
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("input_path", "output_path", "manifest_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def render_motion_to_mp4(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: float = 30.0,
    camera: CameraConfig | None = None,
    style: RenderStyle | None = None,
    overwrite: bool = False,
    manifest_path: str | Path | None = None,
    ffmpeg_binary: str | Path | None = None,
    crf: int = 18,
    preset: str = "medium",
    all_intra: bool = True,
    verify_boundary_frames: int = 0,
    skinning_chunk_size: int = 32,
    canonical_boundary_rgb_frames: Sequence[np.ndarray] | None = None,
    renderer: CoreMeshRenderer | None = None,
    progress: ProgressCallback | None = None,
) -> RenderResult:
    """Render one Core motion archive into a clean H.264/yuv420p MP4.

    Camera, lighting, and framing remain immutable throughout the clip.  Pass
    the same ``CameraConfig`` to every behavior to preserve the visual contract
    required for seamless downstream switching.
    """

    source_path = Path(input_path).resolve()
    destination = Path(output_path)
    camera = camera or CameraConfig.video_call_portrait()
    style = style or RenderStyle()
    source_motion = load_core_motion(source_path)
    render_motion = resample_core_motion(source_motion, target_fps)
    _require_exact_motion_boundaries(render_motion, verify_boundary_frames)
    shared_boundary = None
    if canonical_boundary_rgb_frames is not None:
        if verify_boundary_frames <= 0:
            raise ValueError(
                "canonical_boundary_rgb_frames requires verify_boundary_frames > 0"
            )
        shared_boundary = _validate_canonical_rgb_frames(
            canonical_boundary_rgb_frames, camera, verify_boundary_frames
        )
    if renderer is not None and (renderer.camera != camera or renderer.style != style):
        raise ValueError("shared renderer camera/style do not match the export specification")

    encoder = FFmpegH264Encoder(
        destination,
        width=camera.width,
        height=camera.height,
        fps=render_motion.fps,
        overwrite=overwrite,
        ffmpeg_binary=ffmpeg_binary,
        crf=crf,
        preset=preset,
        all_intra=all_intra,
        verify_boundary_frames=verify_boundary_frames,
    )
    renderer_scope = (
        nullcontext(renderer)
        if renderer is not None
        else CoreMeshRenderer(camera=camera, style=style)
    )
    with renderer_scope as active_renderer:
        encoder.open()
        try:
            captured_boundary: list[np.ndarray] = []
            for zero_index, vertices in enumerate(
                active_renderer.skin_vertices(render_motion, chunk_size=skinning_chunk_size)
            ):
                if verify_boundary_frames and zero_index < verify_boundary_frames:
                    if shared_boundary is not None:
                        frame = shared_boundary[zero_index]
                    elif zero_index == verify_boundary_frames - 1:
                        frame = captured_boundary[0]
                        captured_boundary.append(frame)
                    else:
                        frame = active_renderer.render_vertices(vertices)
                        captured_boundary.append(frame.copy())
                elif verify_boundary_frames and zero_index >= render_motion.num_frames - verify_boundary_frames:
                    boundary_index = zero_index - (render_motion.num_frames - verify_boundary_frames)
                    boundary = shared_boundary if shared_boundary is not None else captured_boundary
                    frame = boundary[boundary_index]
                else:
                    frame = active_renderer.render_vertices(vertices)
                encoder.write(frame)
                index = zero_index + 1
                if progress is not None:
                    progress(index, render_motion.num_frames)
            encoding = encoder.finish()
        except Exception:
            encoder.abort()
            raise

    resolved_manifest = Path(manifest_path).resolve() if manifest_path is not None else None
    result = RenderResult(
        input_path=source_path,
        output_path=encoding.output_path,
        source_frames=source_motion.num_frames,
        source_fps=source_motion.fps,
        output_frames=encoding.frame_count,
        output_fps=render_motion.fps,
        duration_seconds=encoding.frame_count / render_motion.fps,
        input_sha256=sha256_file(source_path),
        raw_rgb_sha256=encoding.raw_rgb_sha256,
        mp4_sha256=encoding.mp4_sha256,
        verified_boundary_frames=encoding.verified_boundary_frames,
        manifest_path=resolved_manifest,
    )
    if resolved_manifest is not None:
        manifest = result.to_dict()
        manifest["camera"] = camera.to_dict()
        manifest["render_style"] = style.to_dict()
        manifest["codec"] = {
            "video": "H.264/libx264",
            "pixel_format": "yuv420p",
            "color_space": "bt709",
            "crf": crf,
            "preset": preset,
            "all_intra": all_intra,
            "verified_boundary_frames": encoding.verified_boundary_frames,
        }
        _write_json_atomic(resolved_manifest, manifest)
    return result


def render_motion_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    spec: VideoExportSpec | None = None,
    camera: CameraConfig | None = None,
    overwrite: bool = False,
    manifest_path: str | Path | None = None,
    ffmpeg_binary: str | Path | None = None,
    canonical_boundary_rgb_frames: Sequence[np.ndarray] | None = None,
    renderer: CoreMeshRenderer | None = None,
    progress: ProgressCallback | None = None,
) -> RenderResult:
    """Render a boundary-safe master asset from a spec or explicit camera.

    ``camera`` is an intentional convenience override for callers that already
    manage their camera contract separately. All other rendering choices come
    from ``spec``.
    """

    resolved = spec or VideoExportSpec()
    return render_motion_to_mp4(
        input_path,
        output_path,
        target_fps=resolved.target_fps,
        camera=camera or resolved.camera,
        style=resolved.style,
        overwrite=overwrite,
        manifest_path=manifest_path,
        ffmpeg_binary=ffmpeg_binary,
        crf=resolved.crf,
        preset=resolved.preset,
        all_intra=resolved.all_intra,
        verify_boundary_frames=resolved.verify_boundary_frames,
        skinning_chunk_size=resolved.skinning_chunk_size,
        canonical_boundary_rgb_frames=canonical_boundary_rgb_frames,
        renderer=renderer,
        progress=progress,
    )
