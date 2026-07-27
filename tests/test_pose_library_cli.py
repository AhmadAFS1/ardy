# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-focused tests for pose-library orchestration and its CLI."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from ardy.pose_video.cli import main
from ardy.pose_video.library import (
    build_pose_library,
    create_switch_stress_test,
    discover_behavior_specs,
    renderer_camera_and_style,
    validate_shared_contract,
)
from ardy.pose_video.renderer import RenderingBackendError
from ardy.pose_video.spec import BehaviorSpec, save_behavior_spec
from ardy.pose_video.validation import (
    VideoValidationExpectations,
    validate_library,
)


def _spec(
    behavior_id: str,
    *,
    fps: int = 10,
    duration_seconds: float = 1.0,
    boundary_seconds: float = 0.2,
    resolution: tuple[int, int] = (64, 96),
    render_resolution: tuple[int, int] = (32, 48),
) -> BehaviorSpec:
    keyframes = []
    if behavior_id == "nod_agree":
        keyframes = [
            {
                "time_seconds": 0.5,
                "label": "restrained downbeat",
                "joints": {"Head": {"rotation_degrees": (10.0, 0.0, 0.0)}},
            }
        ]
    elif behavior_id == "look_away_reset":
        keyframes = [
            {
                "time_seconds": 0.5,
                "label": "brief side glance",
                "joints": {"Head": {"rotation_degrees": (0.0, 12.0, 0.0)}},
            }
        ]
    return BehaviorSpec.model_validate(
        {
            "behavior_id": behavior_id,
            "behavior_type": "loop" if behavior_id == "neutral_resting" else "one_shot",
            "fps": fps,
            "duration_seconds": duration_seconds,
            "boundary_seconds": boundary_seconds,
            "locks": ["Hips"],
            "keyframes": keyframes,
            "ambient_motion": {"enabled": False},
            "camera": {
                "resolution": resolution,
                "render_resolution": render_resolution,
                "position": (0.1, 0.6, 2.7),
                "look_at": (0.0, 0.5, 0.0),
                "up": (0.0, 1.0, 0.0),
                "vertical_fov_degrees": 31.0,
                "background_rgb": (10, 20, 30),
                "body_rgb": (40, 80, 120),
            },
        }
    )


def _write_specs(directory: Path, *behavior_ids: str) -> list[Path]:
    paths = []
    for behavior_id in behavior_ids:
        path = directory / f"{behavior_id}.json"
        save_behavior_spec(_spec(behavior_id), path)
        paths.append(path)
    return paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        return_code = main(arguments)
    return return_code, stdout.getvalue(), stderr.getvalue()


class DiscoveryAndContractTests(unittest.TestCase):
    def test_directory_discovery_is_sorted_and_ignores_non_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            beta, alpha = _write_specs(directory, "beta", "alpha")
            (directory / "library.json").write_text('{"kind": "not-a-behavior"}\n')
            discovered = discover_behavior_specs(directory)
        self.assertEqual(discovered, [alpha.resolve(), beta.resolve()])

    def test_single_invalid_file_is_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "bad.json"
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                discover_behavior_specs(path)

    def test_shared_contract_rejects_output_changes_but_not_draft_resolution(self) -> None:
        first = _spec("alpha", render_resolution=(32, 48))
        second = _spec(
            "beta",
            duration_seconds=1.4,
            render_resolution=(48, 64),
        )
        contract = validate_shared_contract((first, second))
        self.assertNotIn("render_resolution", contract["camera"])
        self.assertNotIn("duration_seconds", contract)
        self.assertNotIn("frames", contract)
        self.assertEqual(
            contract["behavior_timing"],
            {
                "alpha": {
                    "duration_seconds": 1.0,
                    "frames": 10,
                    "opening_boundary_start_frame": 0,
                    "closing_boundary_start_frame": 8,
                },
                "beta": {
                    "duration_seconds": 1.4,
                    "frames": 14,
                    "opening_boundary_start_frame": 0,
                    "closing_boundary_start_frame": 12,
                },
            },
        )
        self.assertEqual(contract["encoder"]["crf"], 0)
        self.assertTrue(contract["encoder"]["all_intra"])
        self.assertEqual(contract["encoder"]["codec_name"], "h264")
        self.assertEqual(contract["encoder"]["pixel_format"], "yuv420p")
        self.assertEqual(contract["encoder"]["color_primaries"], "bt709")
        self.assertEqual(contract["encoder"]["color_transfer"], "bt709")
        self.assertEqual(contract["encoder"]["audio_stream_count"], 0)

        incompatible = _spec("gamma", resolution=(96, 128))
        with self.assertRaisesRegex(ValueError, r"gamma\.camera"):
            validate_shared_contract((first, incompatible))

    def test_shared_contract_rejects_duplicate_ids_and_timing_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate behavior ids"):
            validate_shared_contract((_spec("alpha"), _spec("alpha")))
        with self.assertRaisesRegex(ValueError, r"beta\.fps"):
            validate_shared_contract((_spec("alpha"), _spec("beta", fps=12)))

    def test_renderer_uses_master_resolution_and_maps_colors_exactly(self) -> None:
        spec = _spec("alpha", resolution=(80, 120), render_resolution=(32, 48))
        camera, style = renderer_camera_and_style(spec)
        self.assertEqual((camera.width, camera.height), (80, 120))
        self.assertEqual(camera.eye, spec.camera.position)
        self.assertEqual(camera.target, spec.camera.look_at)
        self.assertEqual(camera.vertical_fov_degrees, spec.camera.vertical_fov_degrees)
        self.assertEqual(style.background_rgba, (10 / 255, 20 / 255, 30 / 255, 1.0))
        self.assertEqual(style.body_rgba, (40 / 255, 80 / 255, 120 / 255, 1.0))


class ComposeOnlyLibraryTests(unittest.TestCase):
    def test_multi_spec_build_is_certified_serializable_and_forceable(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = _write_specs(directory, "neutral_resting", "nod_agree")
            output = directory / "library"

            result = build_pose_library(paths, output, render=False)
            self.assertTrue(result.passed)
            self.assertIsNone(result.library_validation_path)
            self.assertIsNone(result.switch_test_path)
            self.assertEqual(
                [item.behavior_id for item in result.behaviors],
                ["neutral_resting", "nod_agree"],
            )
            json.dumps(result.to_dict())
            manifest = json.loads(result.manifest_path.read_text())
            self.assertTrue(manifest["passed"])
            self.assertNotIn("render_resolution", manifest["contract"]["camera"])
            self.assertIsNone(manifest["observed_media"])
            manifest_behaviors = {
                item["behavior_id"]: item for item in manifest["behaviors"]
            }
            for behavior in result.behaviors:
                self.assertTrue(behavior.motion_path.is_file())
                report = json.loads(behavior.motion_validation_path.read_text())
                self.assertTrue(report["passed"])
                provenance = manifest_behaviors[behavior.behavior_id]
                self.assertEqual(provenance["spec_sha256"], _sha256(behavior.spec_path))
                self.assertEqual(
                    provenance["artifact_sha256"]["spec"], _sha256(behavior.spec_path)
                )
                self.assertEqual(
                    provenance["artifact_sha256"]["motion"], _sha256(behavior.motion_path)
                )
                self.assertEqual(
                    provenance["artifact_sha256"]["motion_validation"],
                    _sha256(behavior.motion_validation_path),
                )
                self.assertEqual(
                    provenance["spec_snapshot"]["behavior_id"], behavior.behavior_id
                )

            original_hash = _sha256(result.behaviors[0].motion_path)
            with self.assertRaises(FileExistsError):
                build_pose_library(paths, output, render=False)
            self.assertEqual(_sha256(result.behaviors[0].motion_path), original_hash)
            rebuilt = build_pose_library(paths, output, render=False, overwrite=True)
            self.assertTrue(rebuilt.passed)

    def test_preflight_prevents_partial_multi_spec_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = _write_specs(directory, "alpha", "beta")
            output = directory / "library"
            blocked = output / "beta" / "beta.motion.validation.json"
            blocked.parent.mkdir(parents=True)
            blocked.write_text("keep me\n")

            with self.assertRaises(FileExistsError):
                build_pose_library(paths, output, render=False)
            self.assertFalse((output / "alpha" / "alpha.motion.npz").exists())
            self.assertEqual(blocked.read_text(), "keep me\n")

    def test_render_build_reuses_one_rgb_boundary_and_builds_idle_switch_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            # Intentionally place idle in the middle; sequencing must not depend
            # on input order.
            paths = _write_specs(directory, "nod_agree", "neutral_resting", "look_away_reset")
            save_behavior_spec(
                _spec("look_away_reset", duration_seconds=1.4),
                paths[-1],
            )
            canonical = (object(), object())
            passing_video = {"passed": True, "checks": {}, "metrics": {}}
            passing_library = {"passed": True, "checks": {}, "assets": []}
            switch_path = directory / "library" / "switch_stress_test.mp4"
            shared_renderer = mock.Mock(name="shared_renderer")
            renderer_context = mock.MagicMock(name="renderer_context")
            renderer_context.__enter__.return_value = shared_renderer

            def fake_render_motion(
                _motion_path: Path,
                video_path: Path,
                _spec: BehaviorSpec,
                *,
                render_manifest_path: Path,
                **_kwargs,
            ) -> None:
                video_path.write_bytes(b"mock video")
                render_manifest_path.write_text("{}\n")

            def fake_switch_test(_videos, output_path, *, overwrite=False):
                del overwrite
                output_path.write_bytes(b"mock switch")
                return output_path

            with (
                mock.patch(
                    "ardy.pose_video.renderer.CoreMeshRenderer",
                    return_value=renderer_context,
                ) as renderer_factory,
                mock.patch(
                    "ardy.pose_video.pipeline.render_canonical_boundary_frames",
                    return_value=canonical,
                ) as render_boundary,
                mock.patch(
                    "ardy.pose_video.library._render_motion",
                    side_effect=fake_render_motion,
                ) as render_motion,
                mock.patch(
                    "ardy.pose_video.library.validate_video",
                    return_value=passing_video,
                ) as validate_video_mock,
                mock.patch(
                    "ardy.pose_video.library.validate_library",
                    return_value=passing_library,
                ) as validate_library_mock,
                mock.patch(
                    "ardy.pose_video.library.create_switch_stress_test",
                    side_effect=fake_switch_test,
                ) as switch,
            ):
                result = build_pose_library(paths, directory / "library", render=True)

            self.assertTrue(result.passed)
            renderer_factory.assert_called_once()
            render_boundary.assert_called_once()
            self.assertIs(render_boundary.call_args.kwargs["renderer"], shared_renderer)
            self.assertEqual(render_motion.call_count, 3)
            for call in render_motion.call_args_list:
                self.assertIs(call.kwargs["canonical_boundary_rgb_frames"], canonical)
                self.assertIs(call.kwargs["renderer"], shared_renderer)
            self.assertEqual(
                [
                    call.kwargs["expectations"].frame_count
                    for call in validate_video_mock.call_args_list
                ],
                [10, 10, 14],
            )
            library_expectations = validate_library_mock.call_args.kwargs[
                "expectations"
            ]
            self.assertEqual(
                [expectation.frame_count for expectation in library_expectations],
                [10, 10, 14],
            )
            self.assertEqual(
                [
                    expectation.duration_seconds
                    for expectation in library_expectations
                ],
                [1.0, 1.0, 1.4],
            )
            sequence = switch.call_args.args[0]
            by_id = {item.behavior_id: item.video_path for item in result.behaviors}
            self.assertEqual(
                sequence,
                [
                    by_id["neutral_resting"],
                    by_id["nod_agree"],
                    by_id["neutral_resting"],
                    by_id["look_away_reset"],
                    by_id["neutral_resting"],
                ],
            )


class CliTests(unittest.TestCase):
    def test_compose_and_motion_validate_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = _write_specs(directory, "alpha", "beta")
            output = directory / "motions"
            return_code, stdout, stderr = _run_cli(
                ["compose", *(str(path) for path in paths), "-o", str(output)]
            )
            self.assertEqual(return_code, 0, stderr)
            payload = json.loads(stdout)
            self.assertTrue(payload["passed"])
            self.assertEqual(len(payload["motions"]), 2)

            report_path = directory / "motion-report.json"
            return_code, stdout, stderr = _run_cli(
                [
                    "validate",
                    "motion",
                    str(output / "alpha.motion.npz"),
                    str(output / "beta.motion.npz"),
                    "-o",
                    str(report_path),
                ]
            )
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(stdout.strip(), str(report_path.resolve()))
            self.assertTrue(json.loads(report_path.read_text())["passed"])

            return_code, _, stderr = _run_cli(
                ["compose", *(str(path) for path in paths), "-o", str(output)]
            )
            self.assertEqual(return_code, 2)
            self.assertIn("already exist", stderr)

    def test_compose_duplicate_ids_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = directory / "one.json"
            second = directory / "two.json"
            save_behavior_spec(_spec("same"), first)
            save_behavior_spec(_spec("same"), second)
            output = directory / "motions"
            return_code, _, stderr = _run_cli(
                ["compose", str(first), str(second), "-o", str(output)]
            )
            self.assertEqual(return_code, 2)
            self.assertIn("duplicate behavior ids", stderr)
            self.assertFalse((output / "same.motion.npz").exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe unavailable")
class VideoCliAndSwitchTests(unittest.TestCase):
    @staticmethod
    def _make_video(path: Path, color: str, *, frames: int = 4) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=32x32:r=10",
                "-frames:v",
                str(frames),
                "-an",
                "-c:v",
                "libx264",
                "-g",
                "1",
                "-keyint_min",
                "1",
                "-sc_threshold",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(path),
            ],
            check=True,
        )

    def test_library_accepts_variable_lengths_with_per_asset_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            short = directory / "short.mp4"
            long = directory / "long.mp4"
            self._make_video(short, "red", frames=10)
            self._make_video(long, "red", frames=14)
            common = {
                "fps": 10,
                "width": 32,
                "height": 32,
                "codec_name": "h264",
                "pixel_format": "yuv420p",
                "audio_stream_count": 0,
                "all_intra": True,
            }
            report = validate_library(
                [short, long],
                boundary_frames=2,
                expectations=[
                    VideoValidationExpectations(
                        **common,
                        frame_count=10,
                        duration_seconds=1.0,
                    ),
                    VideoValidationExpectations(
                        **common,
                        frame_count=14,
                        duration_seconds=1.4,
                    ),
                ],
            )

            self.assertTrue(report["passed"])
            self.assertTrue(report["checks"]["matching_video_formats"])
            self.assertTrue(report["checks"]["shared_opening_block"])
            self.assertTrue(report["checks"]["shared_ending_block"])
            self.assertTrue(report["checks"]["all_ordered_pairs_switch_compatible"])
            self.assertEqual(
                [asset["metrics"]["frame_count"] for asset in report["assets"]],
                [10, 14],
            )
            self.assertIsInstance(report["expectations"], list)
            self.assertEqual(
                [item["frame_count"] for item in report["expectations"]],
                [10, 14],
            )

    def test_library_rejects_misaligned_per_asset_expectations(self) -> None:
        expectation = VideoValidationExpectations(fps=10)
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_library(
                ["one.mp4", "two.mp4"],
                boundary_frames=2,
                expectations=[expectation],
            )

    def test_analyze_and_video_validate_cli_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            video = directory / "idle.mp4"
            self._make_video(video, "red", frames=10)
            analysis = directory / "analysis.json"
            return_code, stdout, stderr = _run_cli(
                [
                    "analyze",
                    str(video),
                    "--behavior",
                    "idle",
                    "--analysis-width",
                    "32",
                    "--boundary-seconds",
                    "0.2",
                    "-o",
                    str(analysis),
                ]
            )
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(stdout.strip(), str(analysis.resolve()))
            self.assertEqual(json.loads(analysis.read_text())["analysis"]["behavior_hint"], "idle")

            validation = directory / "video-validation.json"
            return_code, stdout, stderr = _run_cli(
                [
                    "validate",
                    "video",
                    str(video),
                    "--boundary-frames",
                    "2",
                    "-o",
                    str(validation),
                ]
            )
            self.assertEqual(return_code, 0, stderr)
            self.assertEqual(stdout.strip(), str(validation.resolve()))
            self.assertTrue(json.loads(validation.read_text())["passed"])

    def test_tiny_rendered_library_has_cross_asset_pixel_identical_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            paths = _write_specs(directory, "neutral_resting", "nod_agree")
            try:
                result = build_pose_library(
                    paths,
                    directory / "rendered-library",
                    render=True,
                    create_switch_test=False,
                )
            except RenderingBackendError as exc:
                self.skipTest(str(exc))
            self.assertTrue(result.passed)
            report = json.loads(result.library_validation_path.read_text())
            self.assertTrue(report["checks"]["shared_opening_block"])
            self.assertTrue(report["checks"]["shared_ending_block"])
            self.assertTrue(report["checks"]["opening_equals_ending"])
            self.assertEqual(report["expectations"]["codec_name"], "h264")
            self.assertEqual(report["expectations"]["pixel_format"], "yuv420p")
            self.assertTrue(report["expectations"]["all_intra"])
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["observed_media"]["codec_names"], ["h264"])
            self.assertEqual(manifest["observed_media"]["pixel_formats"], ["yuv420p"])
            self.assertEqual(manifest["observed_media"]["audio_stream_counts"], [0])
            self.assertEqual(manifest["observed_media"]["all_intra_values"], [True])
            self.assertEqual(
                manifest["artifact_sha256"]["library_validation"],
                _sha256(result.library_validation_path),
            )
            for item in manifest["behaviors"]:
                self.assertEqual(item["spec_sha256"], _sha256(Path(item["spec_path"])))
                self.assertEqual(item["artifact_sha256"]["video"], _sha256(Path(item["video_path"])))
            self.assertNotEqual(
                _sha256(result.behaviors[0].video_path),
                _sha256(result.behaviors[1].video_path),
            )

    def test_switch_concat_handles_quoted_paths_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            red = directory / "red's clip.mp4"
            blue = directory / "blue.mp4"
            output = directory / "switch.mp4"
            self._make_video(red, "red")
            self._make_video(blue, "blue")

            created = create_switch_stress_test([red, blue, red], output)
            self.assertEqual(created, output.resolve())
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(int(probe.stdout.strip()), 12)
            first_hash = _sha256(output)
            with self.assertRaises(FileExistsError):
                create_switch_stress_test([red, blue], output)
            self.assertEqual(_sha256(output), first_hash)
            create_switch_stress_test([blue, red], output, overwrite=True)
            self.assertNotEqual(_sha256(output), first_hash)
            with self.assertRaisesRegex(ValueError, "different"):
                create_switch_stress_test([red], red, overwrite=True)

    def test_cli_refuses_to_overwrite_an_input_with_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            video = Path(directory_name) / "idle.mp4"
            self._make_video(video, "black")
            return_code, _, stderr = _run_cli(
                ["validate", "video", str(video), "-o", str(video)]
            )
            self.assertEqual(return_code, 2)
            self.assertIn("must be different", stderr)
            # The rejection happens before ffmpeg and, critically, preserves
            # the source as a video rather than replacing it with JSON.
            self.assertGreater(video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
