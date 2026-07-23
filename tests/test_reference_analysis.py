# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

import numpy as np

from ardy.pose_video.reference_analysis import (
    ReferenceAnalysisConfig,
    analyze_frames,
    analyze_video,
    infer_behavior,
)


def _moving_square_frames(*, axis: str, frame_count: int = 100, peak_frame: int = 50) -> np.ndarray:
    height, width = 64, 48
    frames = np.full((frame_count, height, width), 180, dtype=np.uint8)
    base_x, base_y = 19, 13
    for frame in range(frame_count):
        if frame < 20 or frame >= 80:
            progress = 0.0
        elif frame <= peak_frame:
            progress = (frame - 20) / (peak_frame - 20)
        else:
            progress = (80 - frame) / (80 - peak_frame)
        x = base_x + (int(round(9 * progress)) if axis == "x" else 0)
        y = base_y + (int(round(9 * progress)) if axis == "y" else 0)
        frames[frame, y : y + 14, x : x + 10] = 25
    return frames


class ReferenceAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ReferenceAnalysisConfig(
            analysis_width=48,
            boundary_seconds=1.0,
            smoothing_seconds=0.05,
            spatial_blur_sigma=0.0,
        )

    def test_behavior_inference_only_uses_name(self) -> None:
        self.assertEqual(infer_behavior("simple-NOD.MP4"), "nod")
        self.assertEqual(infer_behavior("thinking_glance.mov"), "look_away")
        self.assertEqual(infer_behavior("neutral_resting.mp4"), "idle")
        self.assertEqual(infer_behavior("take_004.mp4"), "generic")

    def test_nod_like_motion_yields_timing_template(self) -> None:
        report = analyze_frames(
            _moving_square_frames(axis="y"),
            10.0,
            behavior="nod",
            source_name="synthetic_nod.mp4",
            config=self.config,
        )
        peak = report["detected_event"]["peak"]["frame"]
        self.assertLessEqual(abs(peak - 50), 3)
        self.assertEqual(report["behavior_template"]["behavior_id"], "nod_agree")
        self.assertEqual(report["behavior_template"]["target"]["primary_axis"], "pitch")
        self.assertEqual(report["behavior_template"]["measurement_scope"], "timing_seed_only")
        frames = [item["frame"] for item in report["behavior_template"]["suggested_keyframes"]]
        self.assertEqual(frames, sorted(frames))

    def test_look_away_template_and_change_centroid(self) -> None:
        report = analyze_frames(
            _moving_square_frames(axis="x"),
            10.0,
            behavior="look_away",
            source_name="synthetic_look_away.mp4",
            config=self.config,
        )
        self.assertEqual(report["behavior_template"]["behavior_id"], "look_away_reset")
        self.assertEqual(report["behavior_template"]["target"]["primary_axis"], "yaw")
        centroid = report["motion_summary"]["peak_change_centroid_normalized_xy"]
        self.assertIsNotNone(centroid[0])
        self.assertGreater(centroid[0], 0.4)

    def test_exact_loop_boundary_and_velocity(self) -> None:
        frames = np.full((40, 32, 32), 127, dtype=np.uint8)
        report = analyze_frames(frames, 10.0, behavior="idle", config=self.config)
        boundary = report["boundary_continuity"]
        self.assertTrue(boundary["first_last_exact_at_analysis_resolution"])
        self.assertEqual(boundary["first_last_mean_absolute_error"], 0.0)
        self.assertEqual(boundary["cut_velocity_mean_absolute_error"], 0.0)
        self.assertEqual(report["behavior_template"]["behavior_type"], "loop")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    def test_ffmpeg_video_decode_path(self) -> None:
        frames = _moving_square_frames(axis="y", frame_count=30, peak_frame=15)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nod_roundtrip.mp4"
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                "48x64",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "mpeg4",
                "-y",
                str(path),
            ]
            subprocess.run(command, input=frames.tobytes(), check=True)
            report = analyze_video(path, behavior="nod", config=self.config)
        self.assertEqual(report["analysis"]["frame_count"], 30)
        self.assertAlmostEqual(report["analysis"]["fps"], 10.0)
        self.assertEqual(report["source"]["metadata"]["codec_name"], "mpeg4")


if __name__ == "__main__":
    unittest.main()
