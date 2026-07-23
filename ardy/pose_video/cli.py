# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified command line for precision pose authoring and video builds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from pydantic import ValidationError

from .composer import compose_behavior
from .library import (
    build_pose_library,
    discover_behavior_specs,
    renderer_camera_and_style,
)
from .reference_analysis import (
    ReferenceAnalysisConfig,
    analyze_reference_set,
    analyze_video,
    write_json_report,
)
from .spec import load_behavior_spec
from .validation import (
    save_validation_report,
    validate_library,
    validate_motion_npz,
    validate_video,
)


def _add_output_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "-o", type=Path, help="JSON output; omit to print to stdout")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ardy.pose_video",
        description="Author, render, and certify seamless ARDY avatar motion-reference videos.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze",
        help="measure coarse motion timing in existing reference MP4s",
    )
    analyze.add_argument("videos", type=Path, nargs="+", help="one or more reference videos")
    analyze.add_argument(
        "--behavior",
        choices=("auto", "idle", "nod", "look_away", "generic"),
        default="auto",
        help="single-video timing hint; auto only uses the filename",
    )
    analyze.add_argument("--analysis-width", type=int, default=192)
    analyze.add_argument("--boundary-seconds", type=float, default=2.0)
    analyze.add_argument("--smoothing-seconds", type=float, default=0.12)
    _add_output_json(analyze)

    compose = commands.add_parser(
        "compose",
        help="compile exact joint keyframes into deterministic ARDY motion NPZs",
    )
    compose.add_argument("specs", type=Path, nargs="+", help="behavior JSON specs")
    compose.add_argument("--output-dir", "-o", type=Path, required=True)
    compose.add_argument("--force", action="store_true", help="replace existing motion files")

    render = commands.add_parser(
        "render",
        help="render a Core motion NPZ as a fixed-camera all-intra H.264 MP4",
    )
    render.add_argument("input", type=Path, help="Core motion NPZ")
    render.add_argument("--output", "-o", type=Path, required=True, help="destination MP4")
    render.add_argument("--behavior-spec", type=Path, help="reuse exact camera/FPS/boundary contract")
    render.add_argument("--fps", type=float, default=30.0)
    render.add_argument("--boundary-frames", type=int, default=60)
    render.add_argument(
        "--camera", choices=("video-call", "full-body"), default="video-call"
    )
    render.add_argument("--camera-json", type=Path)
    render.add_argument("--save-camera-json", type=Path)
    render.add_argument("--width", type=int)
    render.add_argument("--height", type=int)
    render.add_argument("--manifest", type=Path)
    render.add_argument(
        "--crf",
        type=int,
        default=0,
        help="x264 quality (default 0/lossless, required for exact decoded seams)",
    )
    render.add_argument("--preset", default="medium")
    render.add_argument(
        "--no-boundary-verification",
        action="store_true",
        help="diagnostic-only: do not require decoded first/last block equality",
    )
    render.add_argument("--force", action="store_true")

    validate = commands.add_parser("validate", help="certify motion and decoded video seams")
    validation_types = validate.add_subparsers(dest="validation_type", required=True)
    motion = validation_types.add_parser("motion", help="validate one or more motion NPZs")
    motion.add_argument("inputs", type=Path, nargs="+")
    _add_output_json(motion)
    video = validation_types.add_parser(
        "video", help="validate one video or a compatible video library"
    )
    video.add_argument("inputs", type=Path, nargs="+")
    video.add_argument("--boundary-frames", type=int, default=60)
    _add_output_json(video)

    build = commands.add_parser(
        "build",
        help="compose, render, and certify a complete switch-safe behavior library",
    )
    build.add_argument(
        "spec_source", type=Path, help="behavior JSON file or directory containing specs"
    )
    build.add_argument("--output-dir", "-o", type=Path, required=True)
    build.add_argument("--compose-only", action="store_true", help="skip OpenGL rendering")
    build.add_argument("--no-switch-test", action="store_true")
    build.add_argument("--force", action="store_true")

    proxy = commands.add_parser(
        "proxy",
        help="derive and certify H.264 High upload/browser copies from approved masters",
    )
    proxy.add_argument("inputs", type=Path, nargs="+", help="certified master MP4 files")
    proxy.add_argument("--output-dir", "-o", type=Path, required=True)
    proxy.add_argument(
        "--source-manifest",
        type=Path,
        help="optional library.manifest.json that must identify exactly these masters",
    )
    proxy.add_argument("--qp", type=int, default=12, help="fixed x264 quantizer (default: 12)")
    proxy.add_argument("--preset", default="medium")
    proxy.add_argument("--boundary-frames", type=int, default=60)
    proxy.add_argument("--source-fps", type=float, default=30.0)
    proxy.add_argument("--source-width", type=int, default=1080)
    proxy.add_argument("--source-height", type=int, default=1920)
    proxy.add_argument("--source-frame-count", type=int, default=300)
    proxy.add_argument("--source-duration-seconds", type=float, default=10.0)
    proxy.add_argument("--force", action="store_true")
    return parser


def _emit_json(payload: dict | list, destination: Path | None) -> None:
    if destination is None:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        save_validation_report(payload, destination)
        print(destination.resolve())


def _run_analyze(args: argparse.Namespace) -> int:
    if args.output is not None:
        destination = args.output.resolve()
        if destination in {path.resolve() for path in args.videos}:
            raise ValueError("analysis output must be different from every input video")
    config = ReferenceAnalysisConfig(
        analysis_width=args.analysis_width,
        boundary_seconds=args.boundary_seconds,
        smoothing_seconds=args.smoothing_seconds,
    )
    if len(args.videos) == 1:
        report = analyze_video(args.videos[0], behavior=args.behavior, config=config)
    else:
        if args.behavior != "auto":
            raise ValueError("--behavior applies only to one input")
        report = analyze_reference_set(args.videos, config=config)
    if args.output:
        print(write_json_report(report, args.output))
    else:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


def _run_compose(args: argparse.Namespace) -> int:
    loaded = [(spec_path, load_behavior_spec(spec_path)) for spec_path in args.specs]
    behavior_ids = [spec.behavior_id for _, spec in loaded]
    if len(set(behavior_ids)) != len(behavior_ids):
        raise ValueError(f"duplicate behavior ids: {behavior_ids}")

    planned = []
    for _, spec in loaded:
        planned.extend(
            (
                args.output_dir / f"{spec.behavior_id}.motion.npz",
                args.output_dir / f"{spec.behavior_id}.motion.validation.json",
            )
        )
    if not args.force:
        existing = [path for path in planned if path.exists()]
        if existing:
            raise FileExistsError(
                "outputs already exist: " + ", ".join(str(path) for path in existing)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for _, spec in loaded:
        destination = args.output_dir / f"{spec.behavior_id}.motion.npz"
        motion = compose_behavior(spec)
        motion.save_npz(destination)
        report = validate_motion_npz(destination)
        report_path = args.output_dir / f"{spec.behavior_id}.motion.validation.json"
        save_validation_report(report, report_path)
        if not report["passed"]:
            raise RuntimeError(f"motion validation failed: {report['checks']}")
        results.append(
            {
                "behavior_id": spec.behavior_id,
                "motion": str(destination.resolve()),
                "validation": str(report_path.resolve()),
                "passed": True,
            }
        )
    _emit_json({"passed": True, "motions": results}, None)
    return 0


def _resolve_render_spec(args: argparse.Namespace):
    from .config import CameraConfig, RenderStyle
    from .pipeline import VideoExportSpec

    if args.behavior_spec:
        behavior = load_behavior_spec(args.behavior_spec)
        camera, style = renderer_camera_and_style(behavior)
        export = VideoExportSpec(
            target_fps=behavior.fps,
            camera=camera,
            style=style,
            crf=args.crf,
            preset=args.preset,
            all_intra=True,
            verify_boundary_frames=(0 if args.no_boundary_verification else behavior.boundary_frames),
        )
    else:
        if args.camera_json:
            camera = CameraConfig.load(args.camera_json)
        elif args.camera == "full-body":
            camera = CameraConfig.full_body_portrait()
        else:
            camera = CameraConfig.video_call_portrait()
        if args.width is not None or args.height is not None:
            camera = replace(
                camera,
                width=args.width if args.width is not None else camera.width,
                height=args.height if args.height is not None else camera.height,
            )
        export = VideoExportSpec(
            target_fps=args.fps,
            camera=camera,
            style=RenderStyle(),
            crf=args.crf,
            preset=args.preset,
            all_intra=True,
            verify_boundary_frames=(0 if args.no_boundary_verification else args.boundary_frames),
        )
    return export


def _run_render(args: argparse.Namespace) -> int:
    from .pipeline import render_motion_npz

    input_path = args.input.resolve()
    artifact_paths = [args.output.resolve()]
    if args.manifest is not None:
        artifact_paths.append(args.manifest.resolve())
    if args.save_camera_json is not None:
        artifact_paths.append(args.save_camera_json.resolve())
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("render output, manifest, and saved camera JSON must use different paths")
    protected_inputs = {input_path}
    for source in (args.behavior_spec, args.camera_json):
        if source is not None:
            protected_inputs.add(source.resolve())
    overlap = protected_inputs.intersection(artifact_paths)
    if overlap:
        raise ValueError(
            "render artifacts must not overwrite inputs: "
            + ", ".join(str(path) for path in sorted(overlap))
        )
    if not args.force:
        existing = [path for path in artifact_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "render outputs already exist: " + ", ".join(str(path) for path in existing)
            )

    export = _resolve_render_spec(args)
    last_percent = -1

    def progress(frame: int, total: int) -> None:
        nonlocal last_percent
        percent = int(frame * 100 / total)
        if percent != last_percent and (percent % 5 == 0 or frame == total):
            print(f"render: {frame}/{total} ({percent}%)", file=sys.stderr)
            last_percent = percent

    result = render_motion_npz(
        args.input,
        args.output,
        spec=export,
        overwrite=args.force,
        manifest_path=args.manifest,
        progress=progress,
    )
    if args.save_camera_json:
        export.camera.save(args.save_camera_json)
    _emit_json(result.to_dict(), None)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    if args.output is not None and args.output.resolve() in {
        path.resolve() for path in args.inputs
    }:
        raise ValueError("validation output must be different from every validated asset")
    if args.validation_type == "motion":
        reports = [validate_motion_npz(path) for path in args.inputs]
        payload = {
            "schema_version": 1,
            "passed": all(report["passed"] for report in reports),
            "assets": reports,
        }
    elif len(args.inputs) == 1:
        payload = validate_video(args.inputs[0], args.boundary_frames)
    else:
        payload = validate_library(args.inputs, args.boundary_frames)
    _emit_json(payload, args.output)
    return 0 if payload["passed"] else 3


def _run_build(args: argparse.Namespace) -> int:
    spec_paths = discover_behavior_specs(args.spec_source)

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    result = build_pose_library(
        spec_paths,
        args.output_dir,
        render=not args.compose_only,
        overwrite=args.force,
        create_switch_test=not args.no_switch_test,
        progress=progress,
    )
    _emit_json(result.to_dict(), None)
    return 0 if result.passed else 3


def _run_proxy(args: argparse.Namespace) -> int:
    from .delivery import DeliveryProxySpec, build_delivery_proxy_library

    contract = DeliveryProxySpec(
        qp=args.qp,
        preset=args.preset,
        boundary_frames=args.boundary_frames,
        source_fps=args.source_fps,
        source_width=args.source_width,
        source_height=args.source_height,
        source_frame_count=args.source_frame_count,
        source_duration_seconds=args.source_duration_seconds,
    )
    result = build_delivery_proxy_library(
        args.inputs,
        args.output_dir,
        spec=contract,
        overwrite=args.force,
        source_manifest_path=args.source_manifest,
    )
    _emit_json(result.to_dict(), None)
    return 0 if result.passed else 3


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            return _run_analyze(args)
        if args.command == "compose":
            return _run_compose(args)
        if args.command == "render":
            return _run_render(args)
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "build":
            return _run_build(args)
        if args.command == "proxy":
            return _run_proxy(args)
        raise AssertionError(args.command)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValidationError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
