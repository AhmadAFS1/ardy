# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for standards-friendly, seam-safe H.264 delivery proxies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from ardy.pose_video.delivery import (
    DeliveryProxySpec,
    build_delivery_proxy_library,
    transcode_delivery_proxy,
)
from ardy.pose_video.encoder import FFmpegH264Encoder


def _solid(value: tuple[int, int, int]) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def _write_master(
    path: Path,
    interior: tuple[int, int, int],
    *,
    frame_count: int = 8,
) -> None:
    if frame_count < 5:
        raise ValueError("test master needs room for two boundary blocks")
    anchor_a = _solid((40, 80, 120))
    # The canonical block is exact and its own first/last frames match, just as
    # the production breathing lobe starts and ends at the same neutral pose.
    anchor_b = anchor_a
    action = _solid(interior)
    frames = (
        anchor_a,
        anchor_b,
        *((action,) * (frame_count - 4)),
        anchor_a,
        anchor_b,
    )
    encoder = FFmpegH264Encoder(
        path,
        width=32,
        height=32,
        fps=10.0,
        crf=0,
        all_intra=True,
        verify_boundary_frames=2,
    )
    encoder.open()
    for frame in frames:
        encoder.write(frame)
    encoder.finish()


def _tiny_spec(**updates) -> DeliveryProxySpec:
    values = {
        "qp": 12,
        "preset": "medium",
        "boundary_frames": 2,
        "source_fps": 10.0,
        "source_width": 32,
        "source_height": 32,
        "source_frame_count": 8,
        "source_duration_seconds": 0.8,
    }
    values.update(updates)
    return DeliveryProxySpec(**values)


def _write_source_manifest(
    path: Path,
    behaviors: list[tuple[str, Path, int]],
) -> None:
    timing = {
        behavior_id: {
            "frames": frame_count,
            "duration_seconds": frame_count / 10.0,
        }
        for behavior_id, _, frame_count in behaviors
    }
    payload = {
        "schema_version": 1,
        "kind": "ardy_pose_video_library",
        "passed": True,
        "contract": {
            "fps": 10,
            "boundary_frames": 2,
            "behavior_timing": timing,
        },
        "behaviors": [
            {
                "behavior_id": behavior_id,
                "video_path": str(video_path),
                "spec_snapshot": {
                    "behavior_id": behavior_id,
                    "fps": 10,
                    "duration_seconds": frame_count / 10.0,
                    "boundary_seconds": 0.2,
                    "camera": {"resolution": [32, 32]},
                },
            }
            for behavior_id, video_path, frame_count in behaviors
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe unavailable")
class DeliveryProxyIntegrationTests(unittest.TestCase):
    def test_batch_proxy_is_high_profile_atomic_and_cross_asset_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source_dir = directory / "masters"
            source_dir.mkdir()
            alpha = source_dir / "alpha.mp4"
            beta = source_dir / "beta.mp4"
            _write_master(alpha, (180, 50, 30))
            _write_master(beta, (30, 170, 70))

            result = build_delivery_proxy_library(
                [alpha, beta],
                directory / "delivery",
                spec=_tiny_spec(),
            )

            self.assertTrue(result.passed)
            self.assertEqual([asset.source_path for asset in result.assets], [alpha.resolve(), beta.resolve()])
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.source_validation_path.is_file())
            self.assertTrue(result.delivery_validation_path.is_file())
            json.dumps(result.to_dict())

            report = json.loads(result.delivery_validation_path.read_text())
            self.assertTrue(report["passed"])
            self.assertTrue(report["checks"]["shared_opening_block"])
            self.assertTrue(report["checks"]["shared_ending_block"])
            self.assertTrue(report["checks"]["opening_equals_ending"])
            for asset_report in report["assets"]:
                metrics = asset_report["metrics"]
                self.assertEqual(metrics["profile"], "High")
                self.assertEqual(metrics["level"], 40)
                self.assertEqual(metrics["pixel_format"], "yuv420p")
                self.assertEqual(metrics["audio_stream_count"], 0)
                self.assertEqual(metrics["key_frame_count"], metrics["frame_count"])
                self.assertEqual(metrics["picture_types"], {"I": 8})

            manifest = json.loads(result.manifest_path.read_text())
            self.assertTrue(manifest["passed"])
            self.assertEqual(manifest["contract"]["rate_control"], "constant_qp")
            self.assertEqual(manifest["contract"]["qp"], 12)
            self.assertEqual(manifest["contract"]["profile"], "High")
            self.assertEqual(manifest["contract"]["level"], "4.0")
            self.assertEqual(len(manifest["assets"]), 2)
            self.assertFalse(list((directory / "delivery").glob("*.partial.mp4")))

    def test_manifest_aligns_variable_timing_to_each_input_and_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source_dir = directory / "masters"
            source_dir.mkdir()
            short = source_dir / "short.mp4"
            longer = source_dir / "longer.mp4"
            _write_master(short, (180, 50, 30), frame_count=8)
            _write_master(longer, (30, 170, 70), frame_count=12)
            source_manifest = source_dir / "library.manifest.json"
            # Deliberately order the manifest differently from the proxy input
            # list. Timing must be joined by resolved video path, not position.
            _write_source_manifest(
                source_manifest,
                [
                    ("short", short, 8),
                    ("longer", longer, 12),
                ],
            )

            result = build_delivery_proxy_library(
                [longer, short],
                directory / "delivery",
                spec=_tiny_spec(),
                source_manifest_path=source_manifest,
            )

            self.assertTrue(result.passed)
            source_report = json.loads(result.source_validation_path.read_text())
            delivery_report = json.loads(result.delivery_validation_path.read_text())
            self.assertEqual(
                [item["expectations"]["frame_count"] for item in source_report["assets"]],
                [12, 8],
            )
            self.assertEqual(
                [
                    item["expectations"]["duration_seconds"]
                    for item in delivery_report["assets"]
                ],
                [1.2, 0.8],
            )
            self.assertTrue(delivery_report["checks"]["shared_opening_block"])
            self.assertTrue(delivery_report["checks"]["shared_ending_block"])
            self.assertTrue(delivery_report["checks"]["opening_equals_ending"])

            manifest = json.loads(result.manifest_path.read_text())
            source_contract = manifest["contract"]["source_contract"]
            self.assertNotIn("frame_count", source_contract)
            self.assertNotIn("duration_seconds", source_contract)
            self.assertEqual(
                source_contract["timing_policy"],
                "per_asset_from_source_manifest",
            )
            self.assertEqual(
                [item["frame_count"] for item in source_contract["per_asset_timing"]],
                [12, 8],
            )
            self.assertEqual(
                [item["source_timing"]["behavior_id"] for item in manifest["assets"]],
                ["longer", "short"],
            )
            self.assertEqual(
                [item["source_timing"]["frame_count"] for item in manifest["assets"]],
                [12, 8],
            )

    def test_homogeneous_manifest_keeps_the_uniform_contract_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "master.mp4"
            _write_master(source, (180, 50, 30))
            source_manifest = directory / "library.manifest.json"
            _write_source_manifest(
                source_manifest,
                [("neutral_resting", source, 8)],
            )

            result = build_delivery_proxy_library(
                [source],
                directory / "delivery",
                spec=_tiny_spec(),
                source_manifest_path=source_manifest,
            )

            source_report = json.loads(result.source_validation_path.read_text())
            manifest = json.loads(result.manifest_path.read_text())
            self.assertIsInstance(source_report["expectations"], dict)
            self.assertEqual(
                manifest["contract"]["source_contract"]["frame_count"],
                8,
            )
            self.assertEqual(
                manifest["contract"]["source_contract"]["duration_seconds"],
                0.8,
            )
            self.assertNotIn("source_timing", manifest["assets"][0])

    def test_existing_output_requires_overwrite_and_force_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "master.mp4"
            target = directory / "proxy.mp4"
            _write_master(source, (180, 50, 30))
            spec = _tiny_spec()
            transcode_delivery_proxy(source, target, spec=spec)
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                transcode_delivery_proxy(source, target, spec=spec)
            self.assertEqual(target.read_bytes(), original)
            transcode_delivery_proxy(source, target, spec=spec, overwrite=True)
            self.assertEqual(target.read_bytes(), original)

    def test_source_library_mismatch_is_rejected_before_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            alpha = directory / "alpha.mp4"
            beta = directory / "beta.mp4"
            _write_master(alpha, (180, 50, 30))
            _write_master(beta, (30, 170, 70))
            # Make beta's canonical block genuinely different.
            changed = directory / "changed.mp4"
            anchor = _solid((220, 220, 20))
            action = _solid((30, 170, 70))
            encoder = FFmpegH264Encoder(
                changed,
                width=32,
                height=32,
                fps=10.0,
                crf=0,
                verify_boundary_frames=2,
            )
            encoder.open()
            for frame in (anchor, anchor, action, action, action, action, anchor, anchor):
                encoder.write(frame)
            encoder.finish()

            output = directory / "delivery"
            with self.assertRaisesRegex(RuntimeError, "source library"):
                build_delivery_proxy_library(
                    [alpha, changed],
                    output,
                    spec=_tiny_spec(),
                )
            self.assertFalse(output.exists())

    def test_second_encode_failure_leaves_entire_existing_batch_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source_dir = directory / "masters"
            source_dir.mkdir()
            alpha = source_dir / "alpha.mp4"
            beta = source_dir / "beta.mp4"
            _write_master(alpha, (180, 50, 30))
            _write_master(beta, (30, 170, 70))
            output = directory / "delivery"
            output.mkdir()
            existing = {
                output / "alpha.mp4": b"old-alpha",
                output / "beta.mp4": b"old-beta",
                output / "source.library.validation.json": b"old-source-report",
                output / "delivery.library.validation.json": b"old-delivery-report",
                output / "delivery.manifest.json": b"old-manifest",
            }
            for path, content in existing.items():
                path.write_bytes(content)

            calls = 0

            def fail_second(source, target, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("synthetic second encode failure")
                Path(target).write_bytes(b"staged-first-proxy")
                return Path(target).resolve()

            with mock.patch(
                "ardy.pose_video.delivery.transcode_delivery_proxy",
                side_effect=fail_second,
            ):
                with self.assertRaisesRegex(RuntimeError, "second encode"):
                    build_delivery_proxy_library(
                        [alpha, beta],
                        output,
                        spec=_tiny_spec(),
                        overwrite=True,
                    )

            for path, content in existing.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertFalse(list(directory.glob(".delivery.stage.*")))


class DeliveryProxySafetyTests(unittest.TestCase):
    def test_spec_rejects_lossless_qp_and_unknown_presets(self) -> None:
        with self.assertRaisesRegex(ValueError, "QP 0"):
            DeliveryProxySpec(qp=0)
        with self.assertRaisesRegex(ValueError, "preset"):
            DeliveryProxySpec(preset="made-up")
        with self.assertRaisesRegex(ValueError, "boundary_frames"):
            DeliveryProxySpec(boundary_frames=0)

    def test_source_target_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            source = Path(directory_name) / "master.mp4"
            source.write_bytes(b"not a real video")
            with self.assertRaisesRegex(ValueError, "different"):
                transcode_delivery_proxy(source, source, ffmpeg_binary=shutil.which("false") or "/bin/false")
            self.assertEqual(source.read_bytes(), b"not a real video")

    def test_failed_ffmpeg_preserves_existing_target_and_removes_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "master.mp4"
            target = directory / "proxy.mp4"
            source.write_bytes(b"source")
            target.write_bytes(b"keep-existing")
            failure = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="synthetic failure")
            with mock.patch("ardy.pose_video.delivery.subprocess.run", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    transcode_delivery_proxy(
                        source,
                        target,
                        overwrite=True,
                        ffmpeg_binary=shutil.which("false") or "/bin/false",
                    )
            self.assertEqual(target.read_bytes(), b"keep-existing")
            self.assertFalse(list(directory.glob("*.partial.mp4")))

    def test_transcode_command_contains_the_certified_delivery_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "master.mp4"
            target = directory / "proxy.mp4"
            source.write_bytes(b"source")
            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"encoded")
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch("ardy.pose_video.delivery.subprocess.run", side_effect=fake_run):
                transcode_delivery_proxy(
                    source,
                    target,
                    ffmpeg_binary=shutil.which("false") or "/bin/false",
                )

            command = commands[0]
            for expected in (
                "-an",
                "-map_metadata",
                "-qp:v",
                "12",
                "-profile:v",
                "high",
                "-level:v",
                "4.0",
                "-pix_fmt",
                "yuv420p",
                "-bf",
                "0",
                "+bitexact",
            ):
                self.assertIn(expected, command)
            self.assertEqual(target.read_bytes(), b"encoded")

    def test_batch_duplicate_filenames_and_preflight_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = directory / "one" / "same.mp4"
            second = directory / "two" / "same.mp4"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "unique"):
                build_delivery_proxy_library([first, second], directory / "delivery")

    def test_manifest_rejects_disagreeing_timing_before_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = directory / "master.mp4"
            source.write_bytes(b"not decoded because manifest preflight fails")
            source_manifest = directory / "library.manifest.json"
            _write_source_manifest(
                source_manifest,
                [("neutral_resting", source, 8)],
            )
            payload = json.loads(source_manifest.read_text())
            payload["contract"]["behavior_timing"]["neutral_resting"]["frames"] = 9
            source_manifest.write_text(json.dumps(payload), encoding="utf-8")
            output = directory / "delivery"

            with self.assertRaisesRegex(ValueError, "timing disagrees"):
                build_delivery_proxy_library(
                    [source],
                    output,
                    spec=_tiny_spec(),
                    source_manifest_path=source_manifest,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
