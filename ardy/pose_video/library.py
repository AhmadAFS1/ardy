# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible builds for a seamless avatar behavior library.

The important unit in this module is the *library*, not an isolated video.  A
video-call switch can only be invisible when every behavior shares the same
motion boundary, frame rate, camera, render style, resolution, and encoder
contract.  A behavior's duration may vary because switching happens only at
the identical canonical boundary blocks.  Building and certifying those files
together prevents an individually plausible clip from silently breaking a
later transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

if TYPE_CHECKING:
    import numpy as np

    from .renderer import CoreMeshRenderer

from .composer import compose_behavior
from .encoder import sha256_file
from .spec import BehaviorSpec, load_behavior_spec
from .validation import (
    VideoValidationExpectations,
    save_validation_report,
    validate_library,
    validate_motion_npz,
    validate_video,
)


ProgressCallback = Callable[[str], None]

# Lossless all-intra frames make a cached canonical RGB block decode to the
# same pixels regardless of the neighboring behavior frames.
_MASTER_CRF = 0
_MASTER_PRESET = "medium"
_MASTER_ENCODER_CONTRACT = {
    "codec": "H.264/libx264",
    "codec_name": "h264",
    "pixel_format": "yuv420p",
    "color_space": "bt709",
    "color_primaries": "bt709",
    "color_transfer": "bt709",
    "audio_stream_count": 0,
    "all_intra": True,
    "profile_policy": "recorded_not_constrained_reference_master",
    "crf": _MASTER_CRF,
    "preset": _MASTER_PRESET,
}


@dataclass(frozen=True)
class BuiltBehavior:
    behavior_id: str
    spec_path: Path
    spec_sha256: str
    motion_path: Path
    video_path: Path | None
    motion_validation_path: Path
    video_validation_path: Path | None
    render_manifest_path: Path | None

    def to_dict(self) -> dict:
        return {
            "behavior_id": self.behavior_id,
            "spec_path": str(self.spec_path),
            "spec_sha256": self.spec_sha256,
            "motion_path": str(self.motion_path),
            "video_path": str(self.video_path) if self.video_path else None,
            "motion_validation_path": str(self.motion_validation_path),
            "video_validation_path": (
                str(self.video_validation_path) if self.video_validation_path else None
            ),
            "render_manifest_path": (
                str(self.render_manifest_path) if self.render_manifest_path else None
            ),
        }


@dataclass(frozen=True)
class LibraryBuildResult:
    output_dir: Path
    manifest_path: Path
    library_validation_path: Path | None
    switch_test_path: Path | None
    behaviors: tuple[BuiltBehavior, ...]
    passed: bool

    def to_dict(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "library_validation_path": (
                str(self.library_validation_path) if self.library_validation_path else None
            ),
            "switch_test_path": str(self.switch_test_path) if self.switch_test_path else None,
            "behaviors": [behavior.to_dict() for behavior in self.behaviors],
            "passed": self.passed,
        }


def discover_behavior_specs(path: str | Path) -> list[Path]:
    """Return behavior JSON files in stable order.

    ``path`` may be one JSON file or a directory.  Non-behavior JSON files are
    ignored after schema parsing, which permits a future ``library.json`` to
    live beside behavior specs without weakening strict behavior validation.
    """

    source = Path(path)
    candidates = [source] if source.is_file() else sorted(source.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no behavior JSON files found at {source}")
    discovered: list[Path] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            load_behavior_spec(candidate)
        except Exception as exc:
            if source.is_file():
                raise
            errors.append(f"{candidate.name}: {exc}")
            continue
        discovered.append(candidate.resolve())
    if not discovered:
        detail = "; ".join(errors)
        raise ValueError(f"no valid behavior specs found at {source}: {detail}")
    return discovered


def _camera_contract(spec: BehaviorSpec) -> dict:
    # ``render_resolution`` is an authoring-only draft/preview setting.  Master
    # assets are rendered at ``resolution`` (see ``renderer_camera_and_style``),
    # so two otherwise identical behaviors must not become incompatible merely
    # because their editors used different preview sizes.
    contract = spec.camera.model_dump(mode="json")
    contract.pop("render_resolution", None)
    return contract


def _canonical_base_contract(spec: BehaviorSpec) -> dict:
    """Describe the pose on which the shared canonical boundary is based."""

    if spec.base_pose.mode == "authored_neutral":
        return {"mode": "authored_neutral"}
    return spec.base_pose.model_dump(mode="json")


def validate_shared_contract(specs: Sequence[BehaviorSpec]) -> dict:
    """Fail early when behavior specs cannot form one switch-safe library."""

    if not specs:
        raise ValueError("a behavior library requires at least one spec")
    behavior_ids = [spec.behavior_id for spec in specs]
    if len(set(behavior_ids)) != len(behavior_ids):
        raise ValueError(f"duplicate behavior ids: {behavior_ids}")

    first = specs[0]
    expected = {
        "fps": first.fps,
        "boundary_seconds": first.boundary_seconds,
        "boundary_frames": first.boundary_frames,
        "camera": _camera_contract(first),
        "canonical_base_pose": _canonical_base_contract(first),
        "encoder": dict(_MASTER_ENCODER_CONTRACT),
    }
    mismatches: list[str] = []
    for spec in specs[1:]:
        actual = {
            "fps": spec.fps,
            "boundary_seconds": spec.boundary_seconds,
            "boundary_frames": spec.boundary_frames,
            "camera": _camera_contract(spec),
            "canonical_base_pose": _canonical_base_contract(spec),
            "encoder": dict(_MASTER_ENCODER_CONTRACT),
        }
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                mismatches.append(
                    f"{spec.behavior_id}.{field}={actual[field]!r}, expected {expected_value!r}"
                )
    if mismatches:
        raise ValueError("incompatible behavior library contract: " + "; ".join(mismatches))
    expected["behavior_timing"] = {
        spec.behavior_id: {
            "duration_seconds": spec.duration_seconds,
            "frames": spec.num_frames,
            "opening_boundary_start_frame": 0,
            "closing_boundary_start_frame": spec.num_frames - spec.boundary_frames,
        }
        for spec in specs
    }
    return expected


def renderer_camera_and_style(spec: BehaviorSpec):
    """Translate the versioned behavior camera into the renderer contract.

    Kept here rather than in ``spec.py`` so loading/editing a JSON pose spec
    never imports optional OpenGL rendering dependencies.
    """

    from .config import CameraConfig, RenderStyle

    camera = CameraConfig(
        width=spec.camera.resolution[0],
        height=spec.camera.resolution[1],
        eye=spec.camera.position,
        target=spec.camera.look_at,
        up=spec.camera.up,
        vertical_fov_degrees=spec.camera.vertical_fov_degrees,
    )
    style = RenderStyle(
        background_rgba=tuple(channel / 255.0 for channel in spec.camera.background_rgb) + (1.0,),
        body_rgba=tuple(channel / 255.0 for channel in spec.camera.body_rgb) + (1.0,),
    )
    return camera, style


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


def _behavior_manifest_entry(behavior: BuiltBehavior, spec: BehaviorSpec) -> dict:
    """Snapshot source intent and bind every emitted artifact by content hash."""

    payload = behavior.to_dict()
    artifacts: dict[str, str] = {
        "spec": behavior.spec_sha256,
        "motion": sha256_file(behavior.motion_path),
        "motion_validation": sha256_file(behavior.motion_validation_path),
    }
    for name, path in (
        ("video", behavior.video_path),
        ("video_validation", behavior.video_validation_path),
        ("render_manifest", behavior.render_manifest_path),
    ):
        if path is not None:
            artifacts[name] = sha256_file(path)
    payload["artifact_sha256"] = artifacts
    payload["spec_snapshot"] = spec.model_dump(mode="json")
    return payload


def _observed_media_facts(report: dict | None) -> dict | None:
    """Summarize probed stream facts without hiding per-asset differences."""

    if report is None:
        return None
    metrics = [asset["metrics"] for asset in report.get("assets", [])]
    if not metrics:
        return None

    def values(field: str) -> list:
        return sorted({item.get(field) for item in metrics}, key=lambda value: str(value))

    return {
        "codec_names": values("codec_name"),
        "profiles": values("profile"),
        "levels": values("level"),
        "pixel_formats": values("pixel_format"),
        "color_ranges": values("color_range"),
        "color_spaces": values("color_space"),
        "color_primaries": values("color_primaries"),
        "color_transfers": values("color_transfer"),
        "audio_stream_counts": values("audio_stream_count"),
        "all_intra_values": values("all_intra"),
        "picture_type_sets": [
            dict(items)
            for items in sorted(
                {tuple(sorted(item.get("picture_types", {}).items())) for item in metrics}
            )
        ],
    }


def create_switch_stress_test(
    videos: Sequence[str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Concatenate complete all-intra library assets without re-encoding.

    The resulting sequence makes every library cut easy to inspect in a normal
    player.  It is a diagnostic preview, not the WebRTC scheduling logic.
    """

    resolved = [Path(video).resolve() for video in videos]
    if not resolved:
        raise ValueError("at least one video is required for a switch test")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"switch-test videos do not exist: {missing}")
    destination = Path(output_path).resolve()
    if destination in resolved:
        raise ValueError("switch-test output must be different from every input video")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"switch test already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # The concat demuxer accepts shell-like quoting.  Escaping single quotes is
    # required even though subprocess itself never invokes a shell.
    def quote(path: Path) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"

    descriptor, list_name = tempfile.mkstemp(prefix="ardy-switch-", suffix=".txt")
    output_descriptor, temporary_output_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".mp4", dir=destination.parent
    )
    os.close(output_descriptor)
    Path(temporary_output_name).unlink()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for video in resolved:
                stream.write(f"file {quote(video)}\n")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_name,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            temporary_output_name,
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to create a switch stress test") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "unknown ffmpeg error").strip()
            raise RuntimeError(f"could not create switch stress test: {detail}") from exc
        del completed
        # Preserve no-overwrite semantics if another process created the
        # destination while ffmpeg was running.
        if destination.exists() and not overwrite:
            raise FileExistsError(f"switch test already exists: {destination}")
        os.replace(temporary_output_name, destination)
    finally:
        Path(list_name).unlink(missing_ok=True)
        Path(temporary_output_name).unlink(missing_ok=True)
    return destination


def _planned_output_paths(
    destination: Path,
    specs: Sequence[BehaviorSpec],
    *,
    render: bool,
    create_switch_test: bool,
) -> tuple[Path, ...]:
    """Return every file a library build may create, before doing any work."""

    planned: list[Path] = [destination / "library.manifest.json"]
    if render:
        planned.append(destination / "library.validation.json")
        if create_switch_test:
            planned.append(destination / "switch_stress_test.mp4")
    for spec in specs:
        behavior_dir = destination / spec.behavior_id
        planned.extend(
            (
                behavior_dir / f"{spec.behavior_id}.motion.npz",
                behavior_dir / f"{spec.behavior_id}.motion.validation.json",
            )
        )
        if render:
            planned.extend(
                (
                    behavior_dir / f"{spec.behavior_id}.mp4",
                    behavior_dir / f"{spec.behavior_id}.video.validation.json",
                    behavior_dir / f"{spec.behavior_id}.render.json",
                )
            )
    return tuple(planned)


def _render_motion(
    motion_path: Path,
    video_path: Path,
    spec: BehaviorSpec,
    *,
    overwrite: bool,
    render_manifest_path: Path,
    canonical_boundary_rgb_frames: Sequence[np.ndarray],
    renderer: CoreMeshRenderer,
    progress: ProgressCallback | None,
) -> None:
    # Imported lazily so composition, JSON editing, and tests work without the
    # optional EGL renderer installed.
    from .pipeline import VideoExportSpec, render_motion_npz

    camera, style = renderer_camera_and_style(spec)

    def frame_progress(frame: int, total: int) -> None:
        if progress and (frame == total or frame == 1 or frame % max(1, total // 10) == 0):
            progress(f"render {spec.behavior_id}: {frame}/{total}")

    export_spec = VideoExportSpec(
        target_fps=spec.fps,
        camera=camera,
        style=style,
        crf=_MASTER_CRF,
        preset=_MASTER_PRESET,
        all_intra=True,
        verify_boundary_frames=spec.boundary_frames,
    )
    render_motion_npz(
        motion_path,
        video_path,
        spec=export_spec,
        overwrite=overwrite,
        manifest_path=render_manifest_path,
        canonical_boundary_rgb_frames=canonical_boundary_rgb_frames,
        renderer=renderer,
        progress=frame_progress,
    )


def build_pose_library(
    spec_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    render: bool = True,
    overwrite: bool = False,
    create_switch_test: bool = True,
    progress: ProgressCallback | None = None,
) -> LibraryBuildResult:
    """Compose, render, and certify a compatible set of behavior specs."""

    paths = tuple(Path(path).resolve() for path in spec_paths)
    if not paths:
        raise ValueError("no behavior specs were provided")
    specs = tuple(load_behavior_spec(path) for path in paths)
    spec_hashes = tuple(sha256_file(path) for path in paths)
    contract = validate_shared_contract(specs)
    master_expectations = tuple(
        VideoValidationExpectations.master_reference(
            spec.fps,
            width=spec.camera.resolution[0],
            height=spec.camera.resolution[1],
            frame_count=spec.num_frames,
            duration_seconds=spec.duration_seconds,
        )
        for spec in specs
    )
    library_expectations: (
        VideoValidationExpectations
        | tuple[VideoValidationExpectations, ...]
    )
    if len(
        {
            (expectation.frame_count, expectation.duration_seconds)
            for expectation in master_expectations
        }
    ) == 1:
        # Preserve the historical report shape for fixed-duration libraries.
        library_expectations = master_expectations[0]
    else:
        library_expectations = master_expectations
    destination = Path(output_dir).resolve()
    if not overwrite:
        planned = _planned_output_paths(
            destination,
            specs,
            render=render,
            create_switch_test=create_switch_test,
        )
        existing = [path for path in planned if path.exists()]
        if existing:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"library outputs already exist: {formatted}")
    destination.mkdir(parents=True, exist_ok=True)

    built: list[BuiltBehavior] = []
    all_motion_pass = True
    for path, spec, spec_sha256 in zip(paths, specs, spec_hashes):
        behavior_dir = destination / spec.behavior_id
        behavior_dir.mkdir(parents=True, exist_ok=True)
        motion_path = behavior_dir / f"{spec.behavior_id}.motion.npz"
        motion_report_path = behavior_dir / f"{spec.behavior_id}.motion.validation.json"
        video_path = behavior_dir / f"{spec.behavior_id}.mp4" if render else None
        video_report_path = (
            behavior_dir / f"{spec.behavior_id}.video.validation.json" if render else None
        )
        render_manifest_path = (
            behavior_dir / f"{spec.behavior_id}.render.json" if render else None
        )

        if progress:
            progress(f"compose {spec.behavior_id}")
        composed = compose_behavior(spec)
        composed.save_npz(motion_path)
        motion_report = validate_motion_npz(motion_path)
        save_validation_report(motion_report, motion_report_path)
        all_motion_pass = all_motion_pass and bool(motion_report["passed"])
        if not motion_report["passed"]:
            raise RuntimeError(f"motion certification failed for {spec.behavior_id}: {motion_report['checks']}")

        built.append(
            BuiltBehavior(
                behavior_id=spec.behavior_id,
                spec_path=path,
                spec_sha256=spec_sha256,
                motion_path=motion_path,
                video_path=video_path,
                motion_validation_path=motion_report_path,
                video_validation_path=video_report_path,
                render_manifest_path=render_manifest_path,
            )
        )

    # Render only after every motion has been composed and certified.  The
    # first motion's canonical block is rasterized once and fed verbatim to
    # every encoder.  This avoids nondeterministic one-LSB edge differences
    # observed when EGL independently rasterizes an identical mesh.
    if render:
        from .pipeline import VideoExportSpec, render_canonical_boundary_frames
        from .renderer import CoreMeshRenderer

        camera, style = renderer_camera_and_style(specs[0])
        shared_export = VideoExportSpec(
            target_fps=specs[0].fps,
            camera=camera,
            style=style,
            crf=_MASTER_CRF,
            preset=_MASTER_PRESET,
            all_intra=True,
            verify_boundary_frames=specs[0].boundary_frames,
        )
        if progress:
            progress("render shared canonical boundary")
        with CoreMeshRenderer(camera=camera, style=style) as shared_renderer:
            canonical_rgb_frames = render_canonical_boundary_frames(
                built[0].motion_path,
                spec=shared_export,
                renderer=shared_renderer,
            )
            for behavior, spec, expectations in zip(
                built, specs, master_expectations
            ):
                assert behavior.video_path is not None
                assert behavior.video_validation_path is not None
                assert behavior.render_manifest_path is not None
                if progress:
                    progress(f"render {spec.behavior_id}")
                _render_motion(
                    behavior.motion_path,
                    behavior.video_path,
                    spec,
                    overwrite=overwrite,
                    render_manifest_path=behavior.render_manifest_path,
                    canonical_boundary_rgb_frames=canonical_rgb_frames,
                    renderer=shared_renderer,
                    progress=progress,
                )
                video_report = validate_video(
                    behavior.video_path,
                    spec.boundary_frames,
                    expectations=expectations,
                )
                save_validation_report(video_report, behavior.video_validation_path)
                if not video_report["passed"]:
                    raise RuntimeError(
                        f"decoded-pixel certification failed for {spec.behavior_id}: "
                        f"{video_report['checks']}"
                    )

    library_report_path: Path | None = None
    library_report: dict | None = None
    switch_path: Path | None = None
    if render:
        video_paths = [behavior.video_path for behavior in built]
        assert all(path is not None for path in video_paths)
        library_report = validate_library(
            [path for path in video_paths if path is not None],
            specs[0].boundary_frames,
            expectations=library_expectations,
        )
        library_report_path = destination / "library.validation.json"
        save_validation_report(library_report, library_report_path)
        if not library_report["passed"]:
            raise RuntimeError(f"library certification failed: {library_report['checks']}")
        if create_switch_test:
            # Idle is placed between actions when present, matching the intended
            # runtime state machine and making every action reset easy to judge.
            idle = next(
                (behavior.video_path for behavior in built if behavior.behavior_id == "neutral_resting"),
                None,
            )
            non_idle = [
                behavior.video_path
                for behavior in built
                if behavior.video_path is not None and behavior.video_path != idle
            ]
            if idle is None:
                sequence = [
                    behavior.video_path
                    for behavior in built
                    if behavior.video_path is not None
                ]
            elif not non_idle:
                sequence = [idle]
            else:
                # Always exercise both idle->action and action->idle cuts and
                # avoid duplicate idle clips when specs are supplied in an
                # arbitrary order.
                sequence = [idle]
                for action in non_idle:
                    sequence.extend((action, idle))
            switch_path = create_switch_stress_test(
                sequence,
                destination / "switch_stress_test.mp4",
                overwrite=overwrite,
            )

    passed = all_motion_pass and (library_report is None or bool(library_report["passed"]))
    manifest_path = destination / "library.manifest.json"
    manifest_artifact_hashes: dict[str, str] = {}
    if library_report_path is not None:
        manifest_artifact_hashes["library_validation"] = sha256_file(library_report_path)
    if switch_path is not None:
        manifest_artifact_hashes["switch_test"] = sha256_file(switch_path)
    manifest = {
        "schema_version": 1,
        "kind": "ardy_pose_video_library",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "contract": contract,
        "runtime_switch_protocol": {
            "transition_out": "play the current asset through its final canonical block",
            "transition_in": "start the next asset at frame zero",
            "safe_switch_window_frames": specs[0].boundary_frames,
            "do_not": "jump between arbitrary interior frames",
        },
        "behaviors": [
            _behavior_manifest_entry(behavior, spec)
            for behavior, spec in zip(built, specs)
        ],
        "observed_media": _observed_media_facts(library_report),
        "artifact_sha256": manifest_artifact_hashes,
        "library_validation_path": str(library_report_path) if library_report_path else None,
        "switch_test_path": str(switch_path) if switch_path else None,
    }
    _write_json_atomic(manifest_path, manifest)
    return LibraryBuildResult(
        output_dir=destination,
        manifest_path=manifest_path,
        library_validation_path=library_report_path,
        switch_test_path=switch_path,
        behaviors=tuple(built),
        passed=passed,
    )
