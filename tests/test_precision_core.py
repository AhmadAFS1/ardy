"""Regression tests for deterministic pose-video specifications and composition."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from ardy.pose_video.composer import compose_behavior
from ardy.pose_video.spec import (
    BehaviorSpec,
    CameraSpec,
    JointTransform,
    PoseKeyframe,
    load_behavior_spec,
    save_behavior_spec,
)
from ardy.pose_video.validation import (
    VideoValidationExpectations,
    save_validation_report,
    validate_library,
    validate_motion_npz,
    validate_video,
    video_frame_hashes,
)
from ardy.skeleton.registry import build_skeleton


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATHS = tuple(sorted((REPOSITORY_ROOT / "pose_specs").glob("*.json")))
ALL_SPEC_PATHS = tuple(sorted((REPOSITORY_ROOT / "pose_specs").rglob("*.json")))
CANDIDATE_SPEC_PATHS = tuple(
    sorted((REPOSITORY_ROOT / "pose_candidate_specs" / "speaking_listening_v2").glob("*.json"))
)


def _minimal_spec(**updates: object) -> BehaviorSpec:
    payload: dict[str, object] = {
        "behavior_id": "test_behavior",
        "behavior_type": "one_shot",
        "fps": 30,
        "duration_seconds": 10.0,
        "boundary_seconds": 2.0,
    }
    payload.update(updates)
    return BehaviorSpec.model_validate(payload)


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


class BehaviorSpecTests(unittest.TestCase):
    def test_checked_in_specs_load_and_have_the_expected_contract(self) -> None:
        self.assertEqual(len(SPEC_PATHS), 10)
        specs = [load_behavior_spec(path) for path in SPEC_PATHS]
        self.assertEqual(
            {spec.behavior_id for spec in specs},
            {
                "neutral_resting",
                "nod_agree",
                "look_away_reset",
                "active_listening",
                "speaking_direct",
                "thinking_glance",
                "light_smile",
                "empathetic_head_tilt",
                "curious_eyebrow_or_nod",
                "amused_laugh",
            },
        )
        for spec in specs:
            self.assertEqual(spec.schema_version, 1)
            self.assertEqual(spec.fps, 30)
            self.assertEqual(spec.boundary_frames, 15)
            if spec.source_reference is not None:
                self.assertTrue(Path(spec.source_reference).is_absolute())

    def test_active_specs_share_the_switch_contract_and_variable_timing(self) -> None:
        self.assertEqual(len(ALL_SPEC_PATHS), 10)
        expected_durations = {
            "neutral_resting": 10.0,
            "nod_agree": 2.8,
            "look_away_reset": 4.8,
            "active_listening": 8.0,
            "speaking_direct": 10.0,
            "thinking_glance": 4.2,
            "light_smile": 4.0,
            "empathetic_head_tilt": 4.8,
            "curious_eyebrow_or_nod": 3.6,
            "amused_laugh": 4.2,
        }
        expected_camera = CameraSpec()
        for path in ALL_SPEC_PATHS:
            with self.subTest(path=path.relative_to(REPOSITORY_ROOT).as_posix()):
                spec = load_behavior_spec(path)
                self.assertEqual(spec.fps, 30)
                self.assertEqual(spec.duration_seconds, expected_durations[spec.behavior_id])
                self.assertEqual(spec.boundary_seconds, 0.5)
                self.assertEqual(spec.base_pose.mode, "authored_neutral")
                self.assertEqual(spec.camera, expected_camera)

    def test_nested_defaults_and_lists_are_not_shared(self) -> None:
        first = _minimal_spec(behavior_id="first")
        second = _minimal_spec(behavior_id="second")
        self.assertIsNot(first.base_pose, second.base_pose)
        self.assertIsNot(first.ambient_motion, second.ambient_motion)
        self.assertIsNot(first.face_intent, second.face_intent)
        self.assertIsNot(first.camera, second.camera)
        first.locks.append("Head")
        self.assertEqual(second.locks, ["Hips"])

    def test_round_trip_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            spec = _minimal_spec(source_reference="reference.mp4")
            path = directory / "behavior.json"
            save_behavior_spec(spec, path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["source_reference"], "reference.mp4")
            loaded = load_behavior_spec(path)
            self.assertEqual(loaded.source_reference, str((directory / "reference.mp4").resolve()))

    def test_rejects_unknown_fields_nonfinite_values_and_bad_identifiers(self) -> None:
        with self.assertRaises(ValidationError):
            _minimal_spec(unknown=True)
        with self.assertRaises(ValidationError):
            _minimal_spec(behavior_id="Not Valid")
        with self.assertRaises(ValidationError):
            JointTransform(rotation_degrees=(float("nan"), 0.0, 0.0))
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            _minimal_spec(locks=["Hips", "Hips"])

    def test_rejects_unordered_and_boundary_keyframes(self) -> None:
        joint = {"Head": {"rotation_degrees": [1.0, 0.0, 0.0]}}
        with self.assertRaisesRegex(ValidationError, "strictly ordered"):
            _minimal_spec(
                keyframes=[
                    {"time_seconds": 4.0, "joints": joint},
                    {"time_seconds": 3.0, "joints": joint},
                ]
            )
        for boundary_time in (2.0, 8.0):
            with self.subTest(boundary_time=boundary_time):
                with self.assertRaisesRegex(ValidationError, "strictly between"):
                    _minimal_spec(keyframes=[{"time_seconds": boundary_time, "joints": joint}])

    def test_rejects_frame_rounded_boundary_collisions(self) -> None:
        # The continuous intervals do not overlap, but both round to two frames
        # in a three-frame clip.  The frame-domain guard must catch this.
        with self.assertRaisesRegex(ValidationError, "rounded boundary blocks"):
            _minimal_spec(fps=2, duration_seconds=1.6, boundary_seconds=0.79)
        with self.assertRaisesRegex(ValidationError, "at least two frames"):
            _minimal_spec(fps=2, duration_seconds=5.0, boundary_seconds=0.25)

    def test_camera_rejects_degenerate_views(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must be different"):
            CameraSpec(position=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValidationError, "non-zero"):
            CameraSpec(up=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValidationError, "parallel"):
            CameraSpec(
                position=(0.0, 0.0, 1.0),
                look_at=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 2.0),
            )


class DeterministicCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = [load_behavior_spec(path) for path in SPEC_PATHS]
        cls.motions = [compose_behavior(spec) for spec in cls.specs]

    def test_composition_is_bitwise_deterministic(self) -> None:
        for spec, expected in zip(self.specs, self.motions):
            with self.subTest(behavior_id=spec.behavior_id):
                actual = compose_behavior(spec)
                self.assertTrue(np.array_equal(actual.local_rot_mats, expected.local_rot_mats))
                self.assertTrue(np.array_equal(actual.global_rot_mats, expected.global_rot_mats))
                self.assertTrue(np.array_equal(actual.root_positions, expected.root_positions))
                self.assertTrue(np.array_equal(actual.posed_joints, expected.posed_joints))
                self.assertEqual(actual.metadata, expected.metadata)

    def test_speaking_and_listening_v2_candidates_share_the_release_boundary(self) -> None:
        self.assertEqual(len(CANDIDATE_SPEC_PATHS), 2)
        candidates = [compose_behavior(load_behavior_spec(path)) for path in CANDIDATE_SPEC_PATHS]
        reference = self.motions[0]
        boundary = reference.boundary_frames
        self.assertEqual(
            {motion.behavior_id for motion in candidates},
            {"speaking_direct_v2", "active_listening_empathetic_v1"},
        )
        for motion in candidates:
            with self.subTest(behavior_id=motion.behavior_id):
                self.assertEqual(motion.fps, 30)
                self.assertEqual(motion.boundary_frames, 15)
                self.assertTrue(
                    np.array_equal(reference.local_rot_mats[:boundary], motion.local_rot_mats[:boundary])
                )
                self.assertTrue(
                    np.array_equal(reference.local_rot_mats[:boundary], motion.local_rot_mats[-boundary:])
                )
                self.assertTrue(
                    np.array_equal(reference.posed_joints[:boundary], motion.posed_joints[:boundary])
                )
                self.assertTrue(
                    np.array_equal(reference.posed_joints[:boundary], motion.posed_joints[-boundary:])
                )

    def test_speaking_v2_uses_restrained_diagonal_targets_and_shoulder_countermotion(self) -> None:
        path = next(path for path in CANDIDATE_SPEC_PATHS if path.stem == "speaking_direct_v2")
        spec = load_behavior_spec(path)
        intervals = np.diff([keyframe.time_seconds for keyframe in spec.keyframes])
        self.assertGreater(float(np.ptp(intervals)), 0.5)
        self.assertGreater(len(spec.keyframes), 36)
        self.assertNotIn("RightShoulder", spec.locks)
        self.assertNotIn("LeftShoulder", spec.locks)
        self.assertNotIn("RightArm", spec.locks)
        self.assertNotIn("LeftArm", spec.locks)
        head_keyframes = [keyframe for keyframe in spec.keyframes if "Head" in keyframe.joints]
        self.assertGreaterEqual(head_keyframes[0].time_seconds, 0.9)
        head_targets = np.asarray(
            [keyframe.joints["Head"].rotation_degrees for keyframe in head_keyframes]
        )
        self.assertTrue(np.any(np.abs(head_targets[:, 0]) > 0.5))
        self.assertTrue(np.any(np.abs(head_targets[:, 1]) > 0.3))
        self.assertTrue(np.any(np.abs(head_targets[:, 2]) > 0.1))
        self.assertGreater(float(np.max(np.abs(head_targets[:, 0]))), 0.7)
        motion = compose_behavior(spec)
        head_index = build_skeleton(27).bone_index["Head"]
        head_global = motion.global_rot_mats[:, head_index]
        frame_deltas = np.swapaxes(head_global[:-1], -1, -2) @ head_global[1:]
        head_steps = np.degrees(Rotation.from_matrix(frame_deltas).magnitude())
        self.assertGreater(float(np.max(head_steps)), 0.4)
        self.assertLessEqual(float(np.max(head_steps)), 0.5)
        self.assertTrue(
            any(
                "RightArm" in keyframe.joints or "LeftArm" in keyframe.joints
                for keyframe in spec.keyframes
            )
        )

    def test_all_assets_share_exact_opening_and_ending_motion_blocks(self) -> None:
        reference = self.motions[0]
        boundary = reference.boundary_frames
        for motion in self.motions:
            with self.subTest(behavior_id=motion.behavior_id):
                self.assertTrue(np.array_equal(motion.local_rot_mats[:boundary], motion.local_rot_mats[-boundary:]))
                self.assertTrue(np.array_equal(motion.global_rot_mats[:boundary], motion.global_rot_mats[-boundary:]))
                self.assertTrue(np.array_equal(motion.root_positions[:boundary], motion.root_positions[-boundary:]))
                self.assertTrue(np.array_equal(motion.posed_joints[:boundary], motion.posed_joints[-boundary:]))
                self.assertTrue(np.array_equal(reference.local_rot_mats[:boundary], motion.local_rot_mats[:boundary]))
                self.assertTrue(np.array_equal(reference.global_rot_mats[:boundary], motion.global_rot_mats[:boundary]))
                self.assertTrue(np.array_equal(reference.posed_joints[:boundary], motion.posed_joints[:boundary]))

    def test_all_local_and_global_transforms_are_so3(self) -> None:
        identity = np.eye(3)
        for motion in self.motions:
            for label, matrices in (
                ("local", motion.local_rot_mats),
                ("global", motion.global_rot_mats),
            ):
                with self.subTest(behavior_id=motion.behavior_id, transform=label):
                    gram = matrices.swapaxes(-1, -2) @ matrices
                    self.assertLessEqual(float(np.max(np.abs(gram - identity))), 2e-6)
                    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=2e-6)

    def test_neutral_pose_keeps_both_arm_chains_relaxed_beside_the_torso(self) -> None:
        """Guard the body pose against folded/hidden arm regressions."""

        motion = compose_behavior(_minimal_spec(behavior_id="neutral_arm_contract"))
        skeleton = build_skeleton(27)
        joints = {
            name: motion.posed_joints[0, skeleton.bone_index[name]]
            for name in (
                "Spine3",
                "RightShoulder",
                "RightArm",
                "RightForeArm",
                "RightHand",
                "LeftShoulder",
                "LeftArm",
                "LeftForeArm",
                "LeftHand",
            )
        }

        # Elbows and wrists stay laterally outside the torso and advance toward
        # the front-facing camera instead of folding behind the chest.
        for side in ("Right", "Left"):
            arm = joints[f"{side}Arm"]
            elbow = joints[f"{side}ForeArm"]
            hand = joints[f"{side}Hand"]
            shoulder = joints[f"{side}Shoulder"]
            self.assertGreaterEqual(abs(float(elbow[0])), 0.22)
            self.assertLessEqual(abs(float(elbow[0])), 0.27)
            self.assertGreaterEqual(abs(float(hand[0])), 0.24)
            self.assertLessEqual(abs(float(hand[0])), 0.29)
            self.assertLess(abs(float(shoulder[1] - arm[1])), 0.04)
            self.assertGreater(float(elbow[2]), float(joints["Spine3"][2]))
            self.assertGreater(float(hand[2]), float(elbow[2]))

        # The canonical pose is deliberately bilateral, so both sides will
        # occupy the same visible area when rendered by the fixed camera.
        for right_name, left_name in (
            ("RightShoulder", "LeftShoulder"),
            ("RightArm", "LeftArm"),
            ("RightForeArm", "LeftForeArm"),
            ("RightHand", "LeftHand"),
        ):
            right = joints[right_name]
            left = joints[left_name]
            self.assertAlmostEqual(float(right[0]), -float(left[0]), places=5)
            self.assertAlmostEqual(float(right[1]), float(left[1]), places=5)
            self.assertAlmostEqual(float(right[2]), float(left[2]), places=5)

    def test_authored_behaviors_have_distinct_head_excursions(self) -> None:
        skeleton = build_skeleton(27)
        head = skeleton.bone_index["Head"]
        excursions: dict[str, float] = {}
        for motion in self.motions:
            relative = motion.global_rot_mats[0, head].T @ motion.global_rot_mats[:, head]
            excursions[motion.behavior_id] = float(
                np.degrees(Rotation.from_matrix(relative).magnitude()).max()
            )
        self.assertLess(excursions["neutral_resting"], 3.0)
        self.assertGreater(excursions["nod_agree"], 8.0)
        self.assertGreater(excursions["look_away_reset"], 30.0)

    def test_unknown_and_locked_authored_joints_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown locked joints"):
            compose_behavior(_minimal_spec(locks=["NoSuchJoint"]))
        locked = PoseKeyframe(
            time_seconds=4.0,
            joints={"Head": JointTransform(rotation_degrees=(1.0, 0.0, 0.0))},
        )
        with self.assertRaisesRegex(ValueError, "locked joint"):
            compose_behavior(_minimal_spec(locks=["Head"], keyframes=[locked]))
        unknown = PoseKeyframe(
            time_seconds=4.0,
            joints={"NoSuchJoint": JointTransform(rotation_degrees=(1.0, 0.0, 0.0))},
        )
        with self.assertRaisesRegex(ValueError, "unknown joint"):
            compose_behavior(_minimal_spec(locks=[], keyframes=[unknown]))

    def test_invalid_motion_frame_base_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad-base.npz"
            local = np.broadcast_to(np.eye(3), (3, 27, 3, 3)).copy()
            roots = np.zeros((2, 3), dtype=np.float32)
            np.savez(path, local_rot_mats=local, root_positions=roots)
            spec = _minimal_spec(
                base_pose={"mode": "motion_frame", "motion_path": str(path), "frame": 0}
            )
            with self.assertRaisesRegex(ValueError, "root shape"):
                compose_behavior(spec)

            roots = np.zeros((3, 3), dtype=np.float32)
            local[0, 0, 0, 0] = 2.0
            np.savez(path, local_rot_mats=local, root_positions=roots)
            with self.assertRaisesRegex(ValueError, r"SO\(3\)"):
                compose_behavior(spec)


class MotionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_behavior_spec(REPOSITORY_ROOT / "pose_specs" / "nod_agree.json")
        self.motion = compose_behavior(self.spec)

    def test_composed_archive_passes_and_report_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            motion_path = directory / "motion.npz"
            report_path = directory / "report.json"
            self.motion.save_npz(motion_path)
            report = validate_motion_npz(motion_path)
            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["metrics"]["frames"], self.spec.num_frames)
            self.assertEqual(report["metrics"]["boundary_frames"], self.spec.boundary_frames)
            self.assertEqual(report["metrics"]["boundary_rotation_abs_error"], 0.0)
            self.assertEqual(report["metrics"]["fk_rotation_abs_error"], 0.0)
            self.assertEqual(report["metrics"]["fk_joint_error_cm"], 0.0)
            save_validation_report(report, report_path)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)

    def test_corrupted_boundary_and_fk_data_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.npz"
            self.motion.save_npz(path)
            payload = _load_npz_payload(path)
            payload["local_rot_mats"] = payload["local_rot_mats"].copy()
            payload["local_rot_mats"][0, 26] = Rotation.from_euler(
                "x", 1.0, degrees=True
            ).as_matrix()
            np.savez(path, **payload)
            report = validate_motion_npz(path)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["matching_endpoint_pose"])
            self.assertFalse(report["checks"]["identical_boundary_motion"])
            self.assertFalse(report["checks"]["consistent_forward_kinematics"])

    def test_valid_but_inconsistent_global_rotation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.npz"
            self.motion.save_npz(path)
            payload = _load_npz_payload(path)
            payload["global_rot_mats"] = payload["global_rot_mats"].copy()
            interior_frame = self.spec.boundary_frames + 1
            payload["global_rot_mats"][interior_frame, 26] = Rotation.from_euler(
                "y", 4.0, degrees=True
            ).as_matrix()
            np.savez(path, **payload)
            report = validate_motion_npz(path)
            self.assertTrue(report["checks"]["valid_so3"])
            self.assertFalse(report["checks"]["consistent_forward_kinematics"])
            self.assertFalse(report["passed"])

    def test_invalid_scalar_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.npz"
            self.motion.save_npz(path)
            payload = _load_npz_payload(path)
            payload["fps"] = np.asarray(29.5)
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                validate_motion_npz(path)
            payload["fps"] = np.asarray(30)
            payload["boundary_frames"] = np.asarray(0)
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "at least two"):
                validate_motion_npz(path)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe unavailable")
class VideoHashValidationTests(unittest.TestCase):
    width = 32
    height = 24
    fps = 10
    boundary_frames = 3

    @classmethod
    def _canonical_block(cls) -> list[np.ndarray]:
        yy, xx = np.mgrid[: cls.height, : cls.width]
        neutral = np.empty((cls.height, cls.width, 3), dtype=np.uint8)
        neutral[..., 0] = xx * 4
        neutral[..., 1] = yy * 6
        neutral[..., 2] = 72
        accent = neutral.copy()
        accent[8:16, 10:22] = (210, 80, 40)
        return [neutral, accent, neutral.copy()]

    @classmethod
    def _encode_all_intra(
        cls,
        path: Path,
        middle_value: int,
        corrupt_end: bool = False,
        *,
        all_intra: bool = True,
    ) -> None:
        boundary = cls._canonical_block()
        ending = [frame.copy() for frame in boundary]
        if corrupt_end:
            ending[-1][0:4, 0:4] = (255, 255, 255)
        middle = [
            np.full((cls.height, cls.width, 3), middle_value + index, dtype=np.uint8)
            for index in range(4)
        ]
        frames = boundary + middle + ending
        command = [
            shutil.which("ffmpeg") or "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{cls.width}x{cls.height}",
            "-r",
            str(cls.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-threads",
            "1",
        ]
        if all_intra:
            command.extend(("-g", "1", "-keyint_min", "1", "-sc_threshold", "0"))
        command.append(str(path))
        subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames), check=True)

    def test_exact_decoded_blocks_and_cross_asset_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.mp4"
            second = Path(temp_dir) / "second.mp4"
            self._encode_all_intra(first, 40)
            self._encode_all_intra(second, 120)
            report = validate_library([first, second], self.boundary_frames)
            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            for asset in report["assets"]:
                self.assertEqual(asset["metrics"]["frame_count"], 10)
                self.assertEqual(asset["metrics"]["first_block_sha256"], asset["metrics"]["last_block_sha256"])
                self.assertEqual(asset["metrics"]["first_frame_sha256"], asset["metrics"]["last_frame_sha256"])

    def test_changed_ending_pixels_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.mp4"
            self._encode_all_intra(path, 40, corrupt_end=True)
            report = validate_video(path, self.boundary_frames)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["matching_first_last_frame"])
            self.assertFalse(report["checks"]["matching_first_last_block"])

    def test_master_contract_is_strict_but_general_validation_accepts_delivery_media(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            master = directory / "master.mp4"
            inter_coded = directory / "inter-coded.mp4"
            with_audio = directory / "with-audio.mp4"
            self._encode_all_intra(master, 40)
            self._encode_all_intra(inter_coded, 40, all_intra=False)
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(master),
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=mono",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(with_audio),
                ],
                check=True,
            )

            expectations = VideoValidationExpectations.master_reference(
                self.fps,
                width=self.width,
                height=self.height,
                frame_count=10,
                duration_seconds=1.0,
            )
            strict_master = validate_video(
                master, self.boundary_frames, expectations=expectations
            )
            self.assertTrue(strict_master["passed"])
            self.assertEqual(strict_master["metrics"]["codec_name"], "h264")
            self.assertEqual(strict_master["metrics"]["pixel_format"], "yuv420p")
            self.assertEqual(strict_master["metrics"]["color_space"], "bt709")
            self.assertEqual(strict_master["metrics"]["audio_stream_count"], 0)
            self.assertTrue(strict_master["metrics"]["all_intra"])
            self.assertIsNotNone(strict_master["metrics"]["profile"])

            general_inter = validate_video(inter_coded, self.boundary_frames)
            self.assertTrue(general_inter["passed"])
            self.assertFalse(general_inter["metrics"]["all_intra"])
            strict_inter = validate_video(
                inter_coded, self.boundary_frames, expectations=expectations
            )
            self.assertFalse(strict_inter["passed"])
            self.assertFalse(strict_inter["checks"]["frame_types_match_contract"])

            general_audio = validate_video(with_audio, self.boundary_frames)
            self.assertTrue(general_audio["passed"])
            self.assertEqual(general_audio["metrics"]["audio_stream_count"], 1)
            strict_audio = validate_video(
                with_audio, self.boundary_frames, expectations=expectations
            )
            self.assertFalse(strict_audio["passed"])
            self.assertFalse(strict_audio["checks"]["audio_stream_count_matches_contract"])

            mixed_library = validate_library(
                [master, with_audio], self.boundary_frames
            )
            self.assertTrue(mixed_library["checks"]["all_assets_pass_individually"])
            self.assertFalse(mixed_library["checks"]["matching_video_formats"])
            self.assertFalse(mixed_library["passed"])

    def test_invalid_boundary_counts_and_empty_libraries_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            video_frame_hashes("unused.mp4", 0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_library([], self.boundary_frames)


if __name__ == "__main__":
    unittest.main()
