# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standards-friendly H.264 delivery proxies for certified pose masters.

The lossless ARDY reference masters intentionally use x264 CRF 0, which is
signaled as High 4:4:4 Intra even when the stored pixel format is yuv420p.
That is a strong archival format but a poor compatibility boundary for upload
services and hardware-backed browser decoders.  This module derives separate
High Profile, Level 4.0 MP4 proxies without replacing the certified masters.

Constant-QP and all-IDR encoding are deliberate.  Unlike independent CRF or
bitrate-controlled encodes, identical canonical source frames then receive the
same reconstruction regardless of each clip's different interior.  A complete
proxy batch is decoded and hash-validated before its manifest is approved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence

from .validation import VideoValidationExpectations, validate_library


_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)


@dataclass(frozen=True)
class DeliveryProxySpec:
    """Immutable encoding and validation contract for one proxy batch."""

    qp: int = 12
    preset: str = "medium"
    boundary_frames: int = 60
    profile: str = "high"
    level: str = "4.0"
    pixel_format: str = "yuv420p"
    source_fps: float = 30.0
    source_width: int = 1080
    source_height: int = 1920
    source_frame_count: int = 300
    source_duration_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.qp, int) or isinstance(self.qp, bool):
            raise TypeError("qp must be an integer")
        if self.qp == 0:
            raise ValueError("QP 0 selects x264's High 4:4:4 lossless profile; use QP 1-51")
        if not 1 <= self.qp <= 51:
            raise ValueError("qp must be in [1, 51]")
        if self.preset not in _PRESETS:
            raise ValueError(f"preset must be one of {_PRESETS}")
        if not isinstance(self.boundary_frames, int) or self.boundary_frames < 1:
            raise ValueError("boundary_frames must be a positive integer")
        if self.profile != "high":
            raise ValueError("delivery profile is fixed to 'high'")
        if self.level != "4.0":
            raise ValueError("delivery level is fixed to '4.0' for 1080p30 compatibility")
        if self.pixel_format != "yuv420p":
            raise ValueError("delivery pixel_format is fixed to 'yuv420p'")
        if not math.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps must be a finite positive number")
        for name in ("source_width", "source_height", "source_frame_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.source_duration_seconds) or self.source_duration_seconds <= 0:
            raise ValueError("source_duration_seconds must be a finite positive number")

    def to_dict(self) -> dict:
        return {
            "codec": "H.264/libx264",
            "container": "MP4",
            "codec_tag": "avc1",
            "profile": "High",
            "level": self.level,
            "pixel_format": self.pixel_format,
            "rate_control": "constant_qp",
            "qp": self.qp,
            "preset": self.preset,
            "all_intra": True,
            "b_frames": 0,
            "audio_streams": 0,
            "color_range": "tv",
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "boundary_frames": self.boundary_frames,
            "frame_rate_policy": "passthrough_from_certified_source",
            "video_track_timescale": 30000,
            "source_contract": {
                "fps": self.source_fps,
                "width": self.source_width,
                "height": self.source_height,
                "frame_count": self.source_frame_count,
                "duration_seconds": self.source_duration_seconds,
            },
        }


@dataclass(frozen=True)
class DeliveryProxyAsset:
    source_path: Path
    output_path: Path
    source_sha256: str
    output_sha256: str
    source_bytes: int
    output_bytes: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["output_path"] = str(self.output_path)
        return payload


@dataclass(frozen=True)
class DeliveryProxyBuildResult:
    output_dir: Path
    manifest_path: Path
    source_validation_path: Path
    delivery_validation_path: Path
    source_manifest_path: Path | None
    assets: tuple[DeliveryProxyAsset, ...]
    passed: bool

    def to_dict(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "source_validation_path": str(self.source_validation_path),
            "delivery_validation_path": str(self.delivery_validation_path),
            "source_manifest_path": (
                str(self.source_manifest_path) if self.source_manifest_path else None
            ),
            "assets": [asset.to_dict() for asset in self.assets],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class _ManifestAssetTiming:
    """Per-source timing recovered from a certified pose-library manifest."""

    behavior_id: str
    source_path: Path
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "behavior_id": self.behavior_id,
            "source_path": str(self.source_path),
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
        }


def _resolve_ffmpeg(binary: str | Path | None) -> str:
    if binary is None:
        resolved = shutil.which("ffmpeg")
    else:
        candidate = Path(binary).expanduser()
        resolved = str(candidate.resolve()) if candidate.is_file() else shutil.which(str(binary))
    if resolved is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _transcode_command(
    ffmpeg_binary: str,
    source: Path,
    temporary_output: Path,
    spec: DeliveryProxySpec,
) -> list[str]:
    x264_params = (
        "threads=1:lookahead_threads=1:sliced_threads=0:"
        "keyint=1:min-keyint=1:scenecut=0:bframes=0:open-gop=0"
    )
    return [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map_metadata",
        "-1",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        spec.preset,
        "-qp:v",
        str(spec.qp),
        "-pix_fmt",
        spec.pixel_format,
        "-profile:v",
        spec.profile,
        "-level:v",
        spec.level,
        "-tag:v",
        "avc1",
        "-g",
        "1",
        "-keyint_min",
        "1",
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-threads",
        "1",
        "-x264-params",
        x264_params,
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-fps_mode",
        "passthrough",
        "-video_track_timescale",
        "30000",
        "-movflags",
        "+faststart",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        str(temporary_output),
    ]


def transcode_delivery_proxy(
    source_path: str | Path,
    output_path: str | Path,
    *,
    spec: DeliveryProxySpec | None = None,
    overwrite: bool = False,
    ffmpeg_binary: str | Path | None = None,
) -> Path:
    """Atomically transcode one certified master into a delivery proxy."""

    contract = spec or DeliveryProxySpec()
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source master does not exist: {source}")
    if source == destination:
        raise ValueError("source master and delivery output must be different files")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"delivery output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_ffmpeg = _resolve_ffmpeg(ffmpeg_binary)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".partial.mp4", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        command = _transcode_command(resolved_ffmpeg, source, temporary, contract)
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg disappeared before the delivery encode started") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown ffmpeg error").strip()
            raise RuntimeError(f"delivery proxy encode failed: {detail}") from exc
        del completed
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("ffmpeg completed without producing a delivery proxy")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"delivery output already exists: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _master_expectations(
    spec: DeliveryProxySpec,
    *,
    frame_count: int | None = None,
    duration_seconds: float | None = None,
) -> VideoValidationExpectations:
    return VideoValidationExpectations.master_reference(
        spec.source_fps,
        width=spec.source_width,
        height=spec.source_height,
        frame_count=(
            spec.source_frame_count if frame_count is None else frame_count
        ),
        duration_seconds=(
            spec.source_duration_seconds
            if duration_seconds is None
            else duration_seconds
        ),
    )


def _delivery_expectations(
    spec: DeliveryProxySpec,
    *,
    frame_count: int | None = None,
    duration_seconds: float | None = None,
) -> VideoValidationExpectations:
    return VideoValidationExpectations(
        fps=spec.source_fps,
        width=spec.source_width,
        height=spec.source_height,
        frame_count=(
            spec.source_frame_count if frame_count is None else frame_count
        ),
        duration_seconds=(
            spec.source_duration_seconds
            if duration_seconds is None
            else duration_seconds
        ),
        codec_name="h264",
        pixel_format="yuv420p",
        color_space="bt709",
        color_primaries="bt709",
        color_transfer="bt709",
        audio_stream_count=0,
        all_intra=True,
        allowed_profiles=("High",),
    )


def _strengthen_delivery_report(delivery_report: dict, source_report: dict) -> None:
    """Add source parity and High@4.0 requirements absent from generic QA."""

    for output_asset, source_asset in zip(
        delivery_report["assets"], source_report["assets"], strict=True
    ):
        output_metrics = output_asset["metrics"]
        source_metrics = source_asset["metrics"]
        checks = output_asset["checks"]
        checks.update(
            {
                "level_matches_delivery_contract": output_metrics["level"] == 40,
                "limited_range_matches_delivery_contract": output_metrics["color_range"] == "tv",
                "single_video_stream": output_metrics["video_stream_count"] == 1,
                "dimensions_match_source": (
                    output_metrics["width"], output_metrics["height"]
                )
                == (source_metrics["width"], source_metrics["height"]),
                "frame_count_matches_source": (
                    output_metrics["frame_count"] == source_metrics["frame_count"]
                ),
                "duration_matches_source": math.isclose(
                    output_metrics["duration_seconds"],
                    source_metrics["duration_seconds"],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
            }
        )
        output_asset["passed"] = all(checks.values())
    delivery_report["checks"]["all_assets_pass_individually"] = all(
        asset["passed"] for asset in delivery_report["assets"]
    )
    delivery_report["passed"] = all(delivery_report["checks"].values())


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_manifest(path: Path, sources: tuple[Path, ...]) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"source manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "ardy_pose_video_library":
        raise ValueError("source manifest must be an ARDY pose-video library manifest")
    behaviors = payload.get("behaviors")
    if not isinstance(behaviors, list) or not behaviors:
        raise ValueError("source manifest must contain a non-empty behaviors list")
    manifested: list[Path] = []
    for behavior in behaviors:
        if not isinstance(behavior, dict) or not behavior.get("video_path"):
            raise ValueError("every source-manifest behavior must identify a video_path")
        video_path = Path(behavior["video_path"])
        if not video_path.is_absolute():
            video_path = path.parent / video_path
        manifested.append(video_path.resolve())
    if len(set(manifested)) != len(manifested):
        raise ValueError("source manifest contains duplicate video paths")
    if set(manifested) != set(sources):
        raise ValueError("source masters must exactly match the videos in the source manifest")
    if payload.get("passed") is not True:
        raise ValueError("source manifest is not marked as passed")
    return payload


def _positive_manifest_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _positive_manifest_integer(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive integer")
    return value


def _manifest_asset_timings(
    payload: dict,
    manifest_path: Path,
    sources: tuple[Path, ...],
    spec: DeliveryProxySpec,
) -> tuple[_ManifestAssetTiming, ...] | None:
    """Return manifest timing in source input order and verify its provenance.

    New manifests expose the same timing twice: the complete behavior
    ``spec_snapshot`` and the compact shared-contract ``behavior_timing`` map.
    Checking both prevents a stale or hand-edited manifest from silently
    certifying a video against the wrong duration. Older manifests that only
    contain the spec snapshot remain supported.
    """

    behaviors = payload["behaviors"]
    contract_payload = payload.get("contract")
    contract_payload = contract_payload if isinstance(contract_payload, dict) else {}
    compact_timings = contract_payload.get("behavior_timing")
    if compact_timings is not None and not isinstance(compact_timings, dict):
        raise ValueError("source manifest contract.behavior_timing must be an object")
    if compact_timings is None and all(
        not isinstance(behavior.get("spec_snapshot"), dict)
        for behavior in behaviors
    ):
        # Legacy source manifests did not promise per-behavior timing. Retain
        # the caller-supplied uniform DeliveryProxySpec contract for them.
        return None

    by_source: dict[Path, _ManifestAssetTiming] = {}
    behavior_ids: set[str] = set()
    for index, behavior in enumerate(behaviors):
        assert isinstance(behavior, dict)
        source_path = Path(behavior["video_path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()

        behavior_id = behavior.get("behavior_id")
        if not isinstance(behavior_id, str) or not behavior_id:
            raise ValueError(
                f"source manifest behavior {index} must identify a behavior_id"
            )
        if behavior_id in behavior_ids:
            raise ValueError("source manifest contains duplicate behavior ids")
        behavior_ids.add(behavior_id)

        snapshot = behavior.get("spec_snapshot")
        compact = (
            compact_timings.get(behavior_id)
            if compact_timings is not None
            else None
        )
        if not isinstance(snapshot, dict) and not isinstance(compact, dict):
            raise ValueError(
                f"source manifest behavior {behavior_id} has no timing metadata"
            )
        if isinstance(snapshot, dict):
            snapshot_id = snapshot.get("behavior_id")
            if snapshot_id is not None and snapshot_id != behavior_id:
                raise ValueError(
                    f"source manifest behavior {behavior_id} has a mismatched "
                    "spec_snapshot"
                )
            fps = _positive_manifest_number(
                snapshot.get("fps"),
                f"source manifest behavior {behavior_id} fps",
            )
            if not math.isclose(
                fps,
                spec.source_fps,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"source manifest behavior {behavior_id} fps does not "
                    "match the delivery source contract"
                )
            duration_seconds = _positive_manifest_number(
                snapshot.get("duration_seconds"),
                f"source manifest behavior {behavior_id} duration_seconds",
            )
            frame_count = int(round(duration_seconds * fps))

            snapshot_boundary = snapshot.get("boundary_seconds")
            if snapshot_boundary is not None:
                boundary_seconds = _positive_manifest_number(
                    snapshot_boundary,
                    f"source manifest behavior {behavior_id} boundary_seconds",
                )
                if int(round(boundary_seconds * fps)) != spec.boundary_frames:
                    raise ValueError(
                        f"source manifest behavior {behavior_id} boundary does "
                        "not match the delivery source contract"
                    )
            camera = snapshot.get("camera")
            if isinstance(camera, dict) and camera.get("resolution") is not None:
                resolution = camera["resolution"]
                if (
                    not isinstance(resolution, (list, tuple))
                    or len(resolution) != 2
                    or tuple(resolution)
                    != (spec.source_width, spec.source_height)
                ):
                    raise ValueError(
                        f"source manifest behavior {behavior_id} resolution does "
                        "not match the delivery source contract"
                    )
        else:
            fps = spec.source_fps
            assert isinstance(compact, dict)
            duration_seconds = _positive_manifest_number(
                compact.get("duration_seconds"),
                f"source manifest behavior_timing {behavior_id} duration_seconds",
            )
            frame_count = _positive_manifest_integer(
                compact.get("frames"),
                f"source manifest behavior_timing {behavior_id} frames",
            )

        if compact_timings is not None:
            if not isinstance(compact, dict):
                raise ValueError(
                    f"source manifest behavior_timing is missing {behavior_id}"
                )
            compact_frames = _positive_manifest_integer(
                compact.get("frames"),
                f"source manifest behavior_timing {behavior_id} frames",
            )
            compact_duration = _positive_manifest_number(
                compact.get("duration_seconds"),
                f"source manifest behavior_timing {behavior_id} duration_seconds",
            )
            if compact_frames != frame_count or not math.isclose(
                compact_duration,
                duration_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"source manifest timing disagrees for behavior {behavior_id}"
                )
            frame_count = compact_frames
            duration_seconds = compact_duration

        if frame_count != int(round(duration_seconds * fps)):
            raise ValueError(
                f"source manifest timing disagrees for behavior {behavior_id}"
            )
        if 2 * spec.boundary_frames >= frame_count:
            raise ValueError(
                f"source manifest behavior {behavior_id} is too short for two "
                "canonical boundary blocks"
            )
        by_source[source_path] = _ManifestAssetTiming(
            behavior_id=behavior_id,
            source_path=source_path,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )

    if compact_timings is not None and set(compact_timings) != behavior_ids:
        raise ValueError(
            "source manifest behavior_timing keys must exactly match its behaviors"
        )
    return tuple(by_source[source] for source in sources)


def _expectations_for_timings(
    spec: DeliveryProxySpec,
    timings: tuple[_ManifestAssetTiming, ...] | None,
    *,
    delivery: bool,
) -> (
    VideoValidationExpectations
    | Sequence[VideoValidationExpectations]
):
    factory = _delivery_expectations if delivery else _master_expectations
    if timings is None:
        return factory(spec)
    expectations = tuple(
        factory(
            spec,
            frame_count=timing.frame_count,
            duration_seconds=timing.duration_seconds,
        )
        for timing in timings
    )
    if len(
        {
            (expectation.frame_count, expectation.duration_seconds)
            for expectation in expectations
        }
    ) == 1:
        # Keep the historical validation-report shape for homogeneous batches.
        return expectations[0]
    return expectations


def _delivery_contract_payload(
    spec: DeliveryProxySpec,
    timings: tuple[_ManifestAssetTiming, ...] | None,
) -> dict:
    payload = spec.to_dict()
    if timings is None or len(
        {(timing.frame_count, timing.duration_seconds) for timing in timings}
    ) == 1:
        return payload
    source_contract = payload["source_contract"]
    source_contract.pop("frame_count")
    source_contract.pop("duration_seconds")
    source_contract["timing_policy"] = "per_asset_from_source_manifest"
    source_contract["per_asset_timing"] = [
        timing.to_dict() for timing in timings
    ]
    return payload


def _promote_staged_files(
    staged_to_final: tuple[tuple[Path, Path], ...],
    *,
    overwrite: bool,
    staging_dir: Path,
) -> None:
    """Promote a validated batch with rollback if any filesystem step fails."""

    final_paths = [final for _, final in staged_to_final]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("delivery promotion targets must be unique")
    for staged, final in staged_to_final:
        if not staged.is_file():
            raise FileNotFoundError(f"staged delivery artifact is missing: {staged}")
        if final.exists() and not final.is_file():
            raise IsADirectoryError(f"delivery target is not a regular file: {final}")
        if final.exists() and not overwrite:
            raise FileExistsError(f"delivery output already exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)

    backup_dir = staging_dir / ".rollback"
    backup_dir.mkdir()
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for index, (_, final) in enumerate(staged_to_final):
            if final.exists():
                backup = backup_dir / f"{index:04d}-{final.name}"
                os.replace(final, backup)
                backups.append((final, backup))
        for staged, final in staged_to_final:
            os.replace(staged, final)
            promoted.append(final)
    except Exception:
        for final in reversed(promoted):
            final.unlink(missing_ok=True)
        for final, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, final)
        raise


def _stage_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage.", dir=destination.parent)
    )


def build_delivery_proxy_library(
    source_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    spec: DeliveryProxySpec | None = None,
    overwrite: bool = False,
    ffmpeg_binary: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> DeliveryProxyBuildResult:
    """Build and certify a complete High@4.0 proxy library.

    Sources are strictly validated as ARDY masters before staging begins.
    Every proxy, report, and manifest is built in a temporary sibling
    directory; final paths are promoted only after batch validation succeeds.
    Promotion rolls back preexisting files if a filesystem operation fails.
    """

    contract = spec or DeliveryProxySpec()
    sources = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if not sources:
        raise ValueError("at least one source master is required")
    if len(set(sources)) != len(sources):
        raise ValueError("source master paths must be unique")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source masters do not exist: {missing}")
    if any(path.suffix.lower() != ".mp4" for path in sources):
        raise ValueError("every source master must be an MP4 file")
    names = [path.name for path in sources]
    if len(set(names)) != len(names):
        raise ValueError("source master filenames must be unique within a proxy batch")

    destination = Path(output_dir).expanduser().resolve()
    targets = tuple(destination / name for name in names)
    source_validation_path = destination / "source.library.validation.json"
    delivery_validation_path = destination / "delivery.library.validation.json"
    manifest_path = destination / "delivery.manifest.json"
    planned = (*targets, source_validation_path, delivery_validation_path, manifest_path)
    resolved_source_manifest = (
        Path(source_manifest_path).expanduser().resolve()
        if source_manifest_path is not None
        else None
    )
    source_manifest_payload = None
    manifest_timings: tuple[_ManifestAssetTiming, ...] | None = None
    if resolved_source_manifest is not None:
        if not resolved_source_manifest.is_file():
            raise FileNotFoundError(f"source manifest does not exist: {resolved_source_manifest}")
        if resolved_source_manifest in {path.resolve() for path in planned}:
            raise ValueError("source manifest must not collide with a delivery output")
        source_manifest_payload = _validate_source_manifest(resolved_source_manifest, sources)
        manifest_timings = _manifest_asset_timings(
            source_manifest_payload,
            resolved_source_manifest,
            sources,
            contract,
        )
        timing_values = (
            {
                (timing.frame_count, timing.duration_seconds)
                for timing in manifest_timings
            }
            if manifest_timings is not None
            else set()
        )
        if len(timing_values) == 1:
            only_frame_count, only_duration = next(iter(timing_values))
            if (
                only_frame_count != contract.source_frame_count
                or not math.isclose(
                    only_duration,
                    contract.source_duration_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "homogeneous source-manifest timing must match the "
                    "DeliveryProxySpec source timing"
                )
            # Preserve the established scalar expectations and manifest shape
            # for a fixed-duration delivery batch.
            manifest_timings = None
    source_set = set(sources)
    collisions = sorted(source_set.intersection(path.resolve() for path in planned))
    if collisions:
        raise ValueError(
            "delivery outputs must not collide with source masters: "
            + ", ".join(str(path) for path in collisions)
        )
    if not overwrite:
        existing = [path for path in planned if path.exists()]
        if existing:
            raise FileExistsError(
                "delivery outputs already exist: " + ", ".join(str(path) for path in existing)
            )

    source_report = validate_library(
        list(sources),
        contract.boundary_frames,
        expectations=_expectations_for_timings(
            contract,
            manifest_timings,
            delivery=False,
        ),
    )
    if not source_report["passed"]:
        raise RuntimeError(
            "source library does not satisfy the strict master and shared-boundary contract: "
            f"{source_report['checks']}"
        )

    resolved_ffmpeg = _resolve_ffmpeg(ffmpeg_binary)
    staging = _stage_directory(destination)
    try:
        staged_targets = tuple(staging / target.name for target in targets)
        staged_source_validation = staging / source_validation_path.name
        staged_delivery_validation = staging / delivery_validation_path.name
        staged_manifest = staging / manifest_path.name

        assets: list[DeliveryProxyAsset] = []
        for source, staged_target, final_target in zip(
            sources, staged_targets, targets, strict=True
        ):
            output = transcode_delivery_proxy(
                source,
                staged_target,
                spec=contract,
                overwrite=False,
                ffmpeg_binary=resolved_ffmpeg,
            )
            assets.append(
                DeliveryProxyAsset(
                    source_path=source,
                    output_path=final_target,
                    source_sha256=_sha256_file(source),
                    output_sha256=_sha256_file(output),
                    source_bytes=source.stat().st_size,
                    output_bytes=output.stat().st_size,
                )
            )

        delivery_report = validate_library(
            list(staged_targets),
            contract.boundary_frames,
            expectations=_expectations_for_timings(
                contract,
                manifest_timings,
                delivery=True,
            ),
        )
        _strengthen_delivery_report(delivery_report, source_report)
        for asset_report, final_target in zip(
            delivery_report["assets"], targets, strict=True
        ):
            asset_report["asset"] = str(final_target)
        if not delivery_report["passed"]:
            raise RuntimeError(
                "delivery proxy library failed exact media certification: "
                f"{delivery_report['checks']}"
            )

        _write_json_atomic(staged_source_validation, source_report)
        _write_json_atomic(staged_delivery_validation, delivery_report)
        contract_payload = _delivery_contract_payload(contract, manifest_timings)
        artifact_hashes = {
            "source_validation": _sha256_file(staged_source_validation),
            "delivery_validation": _sha256_file(staged_delivery_validation),
            "contract": _canonical_json_sha256(contract_payload),
            "ordered_sources": _canonical_json_sha256(
                [
                    {
                        "path": str(asset.source_path),
                        "sha256": asset.source_sha256,
                    }
                    for asset in assets
                ]
            ),
        }
        if resolved_source_manifest is not None:
            artifact_hashes["source_manifest"] = _sha256_file(resolved_source_manifest)

        manifest_assets = [asset.to_dict() for asset in assets]
        for asset_payload, asset_report in zip(
            manifest_assets,
            delivery_report["assets"],
            strict=True,
        ):
            metrics = asset_report["metrics"]
            asset_payload["decoded_video"] = {
                "frame_count": metrics["frame_count"],
                "duration_seconds": metrics["duration_seconds"],
                "fps": metrics["fps"],
                "width": metrics["width"],
                "height": metrics["height"],
                "first_frame_sha256": metrics["first_frame_sha256"],
                "last_frame_sha256": metrics["last_frame_sha256"],
                "opening_handle_sha256": metrics["first_block_sha256"],
                "ending_handle_sha256": metrics["last_block_sha256"],
            }
        if manifest_timings is not None:
            for asset_payload, timing in zip(
                manifest_assets,
                manifest_timings,
                strict=True,
            ):
                asset_payload["source_timing"] = timing.to_dict()

        manifest = {
            "schema_version": 1,
            "kind": "ardy_pose_video_delivery_library",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "contract": contract_payload,
            "ffmpeg_binary": resolved_ffmpeg,
            "source_manifest_path": (
                str(resolved_source_manifest) if resolved_source_manifest else None
            ),
            "source_manifest_kind": (
                source_manifest_payload.get("kind") if source_manifest_payload else None
            ),
            "source_validation_path": str(source_validation_path),
            "delivery_validation_path": str(delivery_validation_path),
            "artifact_sha256": artifact_hashes,
            "assets": manifest_assets,
            "certification": {
                "boundary_frames": contract.boundary_frames,
                "ordered_nonself_transition_count": len(assets) * (len(assets) - 1),
                "all_assets_pass_individually": delivery_report["checks"][
                    "all_assets_pass_individually"
                ],
                "all_ordered_pairs_switch_compatible": delivery_report["checks"][
                    "all_ordered_pairs_switch_compatible"
                ],
                "opening_equals_ending": delivery_report["checks"][
                    "opening_equals_ending"
                ],
                "shared_decoded_handle_sha256": delivery_report["assets"][0][
                    "metrics"
                ]["first_block_sha256"],
            },
            "usage": {
                "archive": "retain the lossless source masters and their original certification",
                "upload_or_browser": "use these High Profile delivery proxies",
                "webrtc": "decode to frames, then let the persistent WebRTC track negotiate its RTP codec",
            },
        }
        _write_json_atomic(staged_manifest, manifest)

        staged_to_final = tuple(zip(staged_targets, targets, strict=True)) + (
            (staged_source_validation, source_validation_path),
            (staged_delivery_validation, delivery_validation_path),
            (staged_manifest, manifest_path),
        )
        _promote_staged_files(
            staged_to_final,
            overwrite=overwrite,
            staging_dir=staging,
        )
        return DeliveryProxyBuildResult(
            output_dir=destination,
            manifest_path=manifest_path,
            source_validation_path=source_validation_path,
            delivery_validation_path=delivery_validation_path,
            source_manifest_path=resolved_source_manifest,
            assets=tuple(assets),
            passed=True,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "DeliveryProxyAsset",
    "DeliveryProxyBuildResult",
    "DeliveryProxySpec",
    "build_delivery_proxy_library",
    "transcode_delivery_proxy",
]
