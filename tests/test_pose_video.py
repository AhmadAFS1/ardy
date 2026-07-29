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
import torch

from ardy.pose_video.composer import compose_behavior
from ardy.pose_video.config import CameraConfig
from ardy.pose_video.encoder import FFmpegH264Encoder, decoded_frame_hashes
from ardy.pose_video.motion import CoreMotion, load_core_motion, resample_core_motion
from ardy.pose_video.renderer import (
    BLINK_DURATION_SCALE,
    BLINK_EVENTS,
    BLINK_MINIMUM_SCALE,
    NEUTRAL_RESTING_BLINK_MINIMUM_SCALE,
    CoreMeshRenderer,
    RenderingBackendError,
    blink_minimum_scale_for_motion,
    blink_scale_for_frame,
    facial_cue_for_frame,
)
from ardy.pose_video.spec import BehaviorSpec
from ardy.skeleton import CoreSkeleton27
from ardy.viz.core_skin import (
    CoreSkin,
    MATERIAL_BODY,
    MATERIAL_BROW,
    MATERIAL_EYE_BACKING,
    MATERIAL_IRIS,
    MATERIAL_MOUTH,
    MATERIAL_PUPIL,
    MATERIAL_SCLERA,
    MATERIAL_SKIN,
    neutralize_core_eyelids,
    neutralize_core_mouth,
)


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


class CoreSkinNeutralFaceTests(unittest.TestCase):
    @staticmethod
    def _raw_vertices() -> np.ndarray:
        return np.asarray(
            np.load(
                Path(CoreSkeleton27().folder) / "skin_standard.npz",
                allow_pickle=False,
            )["bind_vertices"],
            dtype=np.float32,
        )

    def test_eyelid_correction_is_symmetric_and_localized(self) -> None:
        vertices = self._raw_vertices()
        corrected = neutralize_core_eyelids(torch.tensor(vertices, dtype=torch.float32)).numpy()
        delta = corrected - vertices

        self.assertGreater(np.max(np.abs(delta[:, 0])), 0.0007)
        self.assertLessEqual(np.max(np.abs(delta[:, 0])), 0.00101)
        self.assertGreater(np.max(np.abs(delta[:, 1])), 0.003)
        self.assertLessEqual(np.max(np.abs(delta[:, 1])), 0.00401)
        np.testing.assert_array_equal(delta[:, 2], np.zeros_like(delta[:, 2]))

        back_of_head = vertices[:, 2] < 0.0
        self.assertLess(np.max(np.abs(delta[back_of_head, 0])), 1e-5)
        self.assertLess(np.max(np.abs(delta[back_of_head, 1])), 1e-5)

    def test_mouth_correction_lifts_only_forward_corner_region(self) -> None:
        vertices = self._raw_vertices()
        corrected = neutralize_core_mouth(torch.tensor(vertices, dtype=torch.float32)).numpy()
        delta = corrected - vertices

        self.assertGreater(np.max(delta[:, 1]), 0.002)
        self.assertLessEqual(np.max(delta[:, 1]), 0.00301)
        self.assertGreaterEqual(np.min(delta[:, 1]), 0.0)
        np.testing.assert_array_equal(delta[:, 0], np.zeros_like(delta[:, 0]))
        np.testing.assert_array_equal(delta[:, 2], np.zeros_like(delta[:, 2]))

        back_of_head = vertices[:, 2] < 0.0
        self.assertLess(np.max(np.abs(delta[back_of_head, 1])), 1e-5)

    def test_colored_guidance_face_is_rigidly_bound_to_head(self) -> None:
        raw_vertex_count = self._raw_vertices().shape[0]
        skeleton = CoreSkeleton27()
        skin = CoreSkin(skeleton)
        material_ids = skin.vertex_material_ids.cpu().numpy()

        self.assertGreater(skin.bind_vertices.shape[0], raw_vertex_count)
        self.assertEqual(skin.bind_vertices.shape[0], skin.lbs_indices.shape[0])
        self.assertEqual(skin.bind_vertices.shape[0], skin.lbs_weights.shape[0])
        self.assertEqual(skin.bind_vertices.shape[0], material_ids.shape[0])
        self.assertEqual(
            set(np.unique(material_ids)),
            {
                MATERIAL_BODY,
                MATERIAL_SKIN,
                MATERIAL_SCLERA,
                MATERIAL_IRIS,
                MATERIAL_PUPIL,
                MATERIAL_BROW,
                MATERIAL_MOUTH,
                MATERIAL_EYE_BACKING,
            },
        )

        guidance = np.arange(skin.bind_vertices.shape[0]) >= raw_vertex_count
        head_index = skeleton.bone_order_names.index("Head")
        indices = skin.lbs_indices.cpu().numpy()[guidance]
        weights = skin.lbs_weights.cpu().numpy()[guidance]
        self.assertTrue(np.all(indices == head_index))
        np.testing.assert_array_equal(weights[:, 0], np.ones(weights.shape[0], dtype=weights.dtype))
        np.testing.assert_array_equal(weights[:, 1:], np.zeros_like(weights[:, 1:]))


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
    def test_blink_animation_is_another_fifty_percent_slower(self) -> None:
        self.assertEqual(BLINK_DURATION_SCALE, 2.25)
        self.assertEqual(
            tuple((close_frames, open_frames) for _, close_frames, open_frames in BLINK_EVENTS),
            ((4.5, 6.75), (4.5, 9.0), (4.5, 6.75)),
        )

    def test_blinks_are_natural_and_excluded_from_shared_boundaries(self) -> None:
        scales = np.asarray(
            [blink_scale_for_frame(index, 300, 30.0) for index in range(300)]
        )
        np.testing.assert_array_equal(scales[:60], np.ones(60))
        np.testing.assert_array_equal(scales[-60:], np.ones(60))

        blink_peaks = np.flatnonzero(np.isclose(scales, BLINK_MINIMUM_SCALE))
        self.assertEqual(blink_peaks.shape[0], 3)
        for peak in blink_peaks:
            self.assertGreater(scales[peak - 1], scales[peak])
            self.assertGreater(scales[peak + 1], scales[peak])
            self.assertLess(scales[peak + 1], scales[peak - 1])

    def test_blink_count_scales_with_video_duration(self) -> None:
        expected_counts = {
            10.0: 3,
            8.0: 2,
            6.0: 1,
            4.8: 0,
            2.8: 0,
        }
        for duration, expected_count in expected_counts.items():
            frame_count = int(round(duration * 30.0))
            scales = np.asarray(
                [blink_scale_for_frame(index, frame_count, 30.0) for index in range(frame_count)]
            )
            self.assertEqual(
                np.count_nonzero(np.isclose(scales, BLINK_MINIMUM_SCALE)),
                expected_count,
                msg=f"duration={duration}",
            )

    def test_colored_eye_geometry_closes_and_reopens_in_head_space(self) -> None:
        motion = _identity_motion(frames=300, fps=30.0)
        renderer = CoreMeshRenderer()
        scales = np.asarray(
            [blink_scale_for_frame(index, motion.num_frames, motion.fps) for index in range(300)]
        )
        peak = int(np.argmin(scales))
        frames = {}
        for index, vertices in enumerate(renderer.skin_vertices(motion, chunk_size=300)):
            if index in (0, peak, 299):
                frames[index] = vertices

        sclera = renderer.material_ids == MATERIAL_SCLERA
        open_height = float(np.ptp(frames[0][sclera, 1]))
        closed_height = float(np.ptp(frames[peak][sclera, 1]))
        self.assertGreater(closed_height, open_height * 0.65)
        self.assertLess(closed_height, open_height * 0.75)
        np.testing.assert_array_equal(frames[0], frames[299])

    def test_mvp_behaviors_use_the_approved_blink_counts_and_depth(self) -> None:
        base = _identity_motion(frames=300, fps=30.0)
        expected_counts = {
            "neutral_resting": 3,
            "speaking_direct_v2": 0,
            "active_listening_empathetic_v1": 0,
            "light_smile": 0,
        }
        for behavior_id, expected_count in expected_counts.items():
            motion = CoreMotion(
                base.global_rot_mats,
                base.posed_joints,
                base.fps,
                source_path=Path(f"/candidate/{behavior_id}.motion.npz"),
            )
            minimum = blink_minimum_scale_for_motion(motion)
            self.assertEqual(minimum, NEUTRAL_RESTING_BLINK_MINIMUM_SCALE)
            self.assertEqual(minimum, 0.70)
            scales = np.asarray(
                [
                    blink_scale_for_frame(
                        index,
                        motion.num_frames,
                        motion.fps,
                        minimum,
                        behavior_id=behavior_id,
                    )
                    for index in range(motion.num_frames)
                ]
            )
            np.testing.assert_array_equal(scales[:60], np.ones(60))
            np.testing.assert_array_equal(scales[-60:], np.ones(60))
            self.assertEqual(
                np.count_nonzero(np.isclose(scales, minimum)),
                expected_count,
                msg=behavior_id,
            )

    def test_expression_cues_reset_at_boundaries_and_match_behavior_intent(self) -> None:
        for behavior_id in (
            "light_smile",
            "amused_laugh",
            "curious_eyebrow_or_nod",
            "thinking_glance",
        ):
            self.assertTrue(
                all(
                    value == 0.0
                    for value in facial_cue_for_frame(behavior_id, 0, 120, 30.0).values()
                )
            )
            self.assertTrue(
                all(
                    value == 0.0
                    for value in facial_cue_for_frame(behavior_id, 119, 120, 30.0).values()
                )
            )

        smile = facial_cue_for_frame("light_smile", 60, 120, 30.0)
        smile_onset = facial_cue_for_frame("light_smile", 20, 120, 30.0)
        smile_hold = facial_cue_for_frame("light_smile", 75, 120, 30.0)
        smile_release = facial_cue_for_frame("light_smile", 105, 120, 30.0)
        curious = facial_cue_for_frame("curious_eyebrow_or_nod", 60, 120, 30.0)
        thinking = facial_cue_for_frame("thinking_glance", 60, 120, 30.0)
        merged_listening = facial_cue_for_frame(
            "active_listening_empathetic_v1",
            60,
            240,
            30.0,
        )
        speaking_v2 = facial_cue_for_frame("speaking_direct_v2", 192, 300, 30.0)
        self.assertEqual(smile["smile"], 1.0)
        self.assertLess(smile_onset["smile"], smile["smile"])
        self.assertEqual(smile_hold["smile"], 1.0)
        self.assertLess(smile_release["smile"], smile_hold["smile"])
        self.assertGreater(curious["right_brow"], curious["left_brow"])
        self.assertGreater(thinking["gaze_horizontal"], 0.8)
        self.assertLess(thinking["gaze_vertical"], -0.6)
        self.assertGreater(merged_listening["left_brow"], 0.0)
        self.assertEqual(
            merged_listening["left_brow"],
            merged_listening["right_brow"],
        )
        self.assertLess(merged_listening["gaze_horizontal"], 0.0)
        self.assertGreater(speaking_v2["gaze_horizontal"], 0.0)
        self.assertLess(speaking_v2["gaze_vertical"], 0.0)
        self.assertGreater(speaking_v2["right_brow"], speaking_v2["left_brow"])
        self.assertGreater(speaking_v2["left_brow"], 0.0)

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
