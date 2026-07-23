# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for deterministic Core motion video rendering."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ardy.pose_video.composer import compose_behavior
from ardy.pose_video.config import CameraConfig
from ardy.pose_video.encoder import FFmpegH264Encoder, decoded_frame_hashes
from ardy.pose_video.motion import CoreMotion, load_core_motion, resample_core_motion
from ardy.pose_video.renderer import CoreMeshRenderer, RenderingBackendError
from ardy.pose_video.spec import BehaviorSpec
from ardy.skeleton import CoreSkeleton27


JOINTS = 27


def _identity_motion(frames: int = 2, fps: float = 20.0) -> CoreMotion:
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, JOINTS, 3, 3)).copy()
    neutral = CoreSkeleton27().neutral_joints.float().numpy()
    positions = np.broadcast_to(neutral, (frames, JOINTS, 3)).copy()
    return CoreMotion(rotations, positions, fps)


class CameraConfigTests(unittest.TestCase):
    def test_video_call_defaults_are_master_portrait_contract(self) -> None:
        config = CameraConfig.video_call_portrait()
        self.assertEqual((config.width, config.height), (1080, 1920))
        self.assertEqual(config.eye, (0.0, 0.56, 2.3))
        self.assertEqual(config.target, (0.0, 0.54, 0.0))
        self.assertEqual(config.vertical_fov_degrees, 25.0)

    def test_pose_is_orthonormal_and_json_round_trips(self) -> None:
        config = CameraConfig.full_body_portrait(width=64, height=96)
        pose = config.pose_matrix()
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(pose[:3, 3], config.eye)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "camera.json"
            config.save(path)
            self.assertEqual(CameraConfig.load(path), config)

    def test_rejects_odd_yuv420_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            CameraConfig(width=65, height=96)


class MotionTests(unittest.TestCase):
    def test_resampling_uses_linear_positions_and_rotation_slerp(self) -> None:
        motion = _identity_motion(frames=2, fps=2.0)
        positions = motion.posed_joints.copy()
        positions[1, :, 0] += 1.0
        rotations = motion.global_rot_mats.copy()
        rotations[1] = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()
        motion = CoreMotion(rotations, positions, fps=2.0)

        output = resample_core_motion(motion, target_fps=3.0)
        self.assertEqual(output.num_frames, 3)
        np.testing.assert_allclose(output.posed_joints[0], positions[0])
        np.testing.assert_allclose(output.posed_joints[-1], positions[-1])
        np.testing.assert_allclose(output.posed_joints[1, :, 0], positions[0, :, 0] + 0.5, atol=1e-6)
        midpoint = Rotation.from_matrix(output.global_rot_mats[1, 0]).as_euler("xyz", degrees=True)
        self.assertAlmostEqual(midpoint[1], 45.0, places=4)

    def test_loader_reconstructs_global_data_from_local_transforms(self) -> None:
        frames = 2
        local = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, JOINTS, 3, 3)).copy()
        roots = np.zeros((frames, 3), dtype=np.float32)
        roots[1, 0] = 0.25
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.npz"
            np.savez(path, local_rot_mats=local, root_positions=roots, fps=np.array(20), text=np.array("test"))
            loaded = load_core_motion(path)
        self.assertEqual(loaded.num_frames, frames)
        self.assertEqual(loaded.text, "test")
        np.testing.assert_allclose(loaded.posed_joints[:, 0], roots, atol=1e-6)


class EncodingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe unavailable")
    def test_encoder_outputs_h264_yuv420p_and_is_reproducible(self) -> None:
        frames = []
        for value in (32, 224):
            frame = np.full((96, 64, 3), value, dtype=np.uint8)
            frame[:, :32, 1] = 128
            frames.append(frame)
        hashes = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for iteration in range(2):
                path = Path(temp_dir) / f"output-{iteration}.mp4"
                encoder = FFmpegH264Encoder(path, width=64, height=96, fps=30.0)
                encoder.open()
                for frame in frames:
                    encoder.write(frame)
                result = encoder.finish()
                hashes.append(result.mp4_sha256)
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt,width,height,nb_frames",
                        "-of",
                        "json",
                        str(path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                stream = json.loads(probe.stdout)["streams"][0]
                self.assertEqual(stream["codec_name"], "h264")
                self.assertEqual(stream["pix_fmt"], "yuv420p")
                self.assertEqual((stream["width"], stream["height"]), (64, 96))
                self.assertEqual(int(stream["nb_frames"]), 2)
        self.assertEqual(hashes[0], hashes[1])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_all_intra_preserves_identical_decoded_boundary_blocks(self) -> None:
        anchor_a = np.full((96, 64, 3), (50, 110, 170), dtype=np.uint8)
        anchor_b = np.full((96, 64, 3), (65, 125, 185), dtype=np.uint8)
        interior = np.full((96, 64, 3), (180, 70, 45), dtype=np.uint8)
        frames = (anchor_a, anchor_b, interior, interior[:, ::-1], anchor_a, anchor_b)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "boundaries.mp4"
            encoder = FFmpegH264Encoder(
                path,
                width=64,
                height=96,
                fps=30.0,
                all_intra=True,
                verify_boundary_frames=2,
            )
            encoder.open()
            for frame in frames:
                encoder.write(frame)
            result = encoder.finish()
            self.assertEqual(result.verified_boundary_frames, 2)
            hashes = decoded_frame_hashes(path)
        self.assertEqual(hashes[:2], hashes[-2:])


class RenderingTests(unittest.TestCase):
    def test_headless_core_skin_frame_has_fixed_pixels(self) -> None:
        motion = _identity_motion(frames=1)
        camera = CameraConfig.full_body_portrait(width=64, height=96)
        try:
            with CoreMeshRenderer(camera=camera) as renderer:
                vertices = next(renderer.skin_vertices(motion))
                first = renderer.render_vertices(vertices)
                second = renderer.render_vertices(vertices)
        except RenderingBackendError as exc:
            self.skipTest(str(exc))
        self.assertEqual(first.shape, (96, 64, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))
        background = first[0, 0]
        self.assertGreater(np.count_nonzero(np.any(first != background, axis=-1)), 100)

    def test_video_call_neutral_visibly_separates_both_arms_from_torso(self) -> None:
        spec = BehaviorSpec.model_validate(
            {
                "behavior_id": "rendered_arm_contract",
                "behavior_type": "loop",
                "fps": 30,
                "duration_seconds": 10.0,
                "boundary_seconds": 2.0,
            }
        )
        composed = compose_behavior(spec)
        motion = CoreMotion(
            composed.global_rot_mats[:1],
            composed.posed_joints[:1],
            composed.fps,
        )
        camera = CameraConfig.video_call_portrait(width=180, height=320)
        try:
            with CoreMeshRenderer(camera=camera) as renderer:
                frame = next(renderer.render_motion(motion))
        except RenderingBackendError as exc:
            self.skipTest(str(exc))

        background = frame[0, 0]
        foreground = np.any(frame != background, axis=-1)
        # At lower-chest height, a correct webcam silhouette has three spans:
        # left arm, torso, and right arm, with visible background between them.
        row = foreground[round(camera.height * 0.72)]
        starts = np.flatnonzero(row & np.r_[True, ~row[:-1]])
        ends = np.flatnonzero(row & np.r_[~row[1:], True])
        spans = list(zip(starts.tolist(), ends.tolist()))
        self.assertEqual(len(spans), 3, spans)
        left_arm, torso, right_arm = spans
        self.assertGreaterEqual(left_arm[1] - left_arm[0] + 1, 20)
        self.assertGreaterEqual(right_arm[1] - right_arm[0] + 1, 20)
        self.assertGreaterEqual(torso[0] - left_arm[1] - 1, 3)
        self.assertGreaterEqual(right_arm[0] - torso[1] - 1, 3)
        self.assertLessEqual(
            abs((left_arm[1] - left_arm[0]) - (right_arm[1] - right_arm[0])),
            2,
        )


if __name__ == "__main__":
    unittest.main()
