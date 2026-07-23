# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch

from ardy.constraints import (
    PoseConstraintSet,
    load_constraints_lst,
    save_constraints_lst,
)
from ardy.geometry import axis_angle_to_matrix, matrix_to_cont6d
from ardy.motion_rep.conditioning import build_condition_dicts
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
from ardy.skeleton import CoreSkeleton27


class PoseConstraintSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton = CoreSkeleton27()

    def _make_sparse_pose(self):
        root_positions = torch.tensor(
            [
                [0.25, 0.95, -0.10],
                [0.50, 0.96, -0.05],
            ],
            dtype=torch.float32,
        )
        head_positions = torch.tensor(
            [
                [0.30, 1.72, -0.08],
                [0.48, 1.70, -0.01],
            ],
            dtype=torch.float32,
        )
        rotations = axis_angle_to_matrix(
            torch.tensor(
                [
                    [0.10, 0.20, -0.05],
                    [-0.15, 0.05, 0.08],
                ],
                dtype=torch.float32,
            )
        )
        return PoseConstraintSet(
            self.skeleton,
            position_frame_indices=[2, 2, 7, 7],
            position_joint_names=["Hips", "Head", "Hips", "Head"],
            global_joints_positions=torch.stack(
                [root_positions[0], head_positions[0], root_positions[1], head_positions[1]]
            ),
            rotation_frame_indices=[2, 7],
            rotation_joint_names=["Head", "Neck"],
            global_joints_rots=rotations,
        )

    def test_sparse_channels_feed_existing_conditioning_contract(self):
        pose = self._make_sparse_pose()
        index_dict, data_dict = build_condition_dicts([pose])

        expected_position_indices = torch.tensor(
            [
                [2, self.skeleton.bone_index["Hips"]],
                [2, self.skeleton.bone_index["Head"]],
                [7, self.skeleton.bone_index["Hips"]],
                [7, self.skeleton.bone_index["Head"]],
            ]
        )
        expected_rotation_indices = torch.tensor(
            [
                [2, self.skeleton.bone_index["Head"]],
                [7, self.skeleton.bone_index["Neck"]],
            ]
        )
        torch.testing.assert_close(index_dict["global_joints_positions"][0], expected_position_indices)
        torch.testing.assert_close(index_dict["global_joints_rots"][0], expected_rotation_indices)
        torch.testing.assert_close(index_dict["root_2d"][0], torch.tensor([2, 7]))
        torch.testing.assert_close(index_dict["root_y_pos"][0], torch.tensor([2, 7]))
        torch.testing.assert_close(
            data_dict["root_2d"][0], pose.global_joints_positions[[0, 2]][:, [0, 2]]
        )

    def test_sparse_pose_populates_motion_rep_position_and_rotation_masks(self):
        pose = self._make_sparse_pose()
        motion_rep = ArdyMotionRep(self.skeleton, fps=20)
        observed, mask = motion_rep.create_conditions_from_constraints(
            [pose], length=10, to_normalize=False, device="cpu"
        )

        root = observed[:, motion_rep.slice_dict["root_pos"]]
        root_mask = mask[:, motion_rep.slice_dict["root_pos"]]
        torch.testing.assert_close(root[2], pose.global_joints_positions[0])
        torch.testing.assert_close(root[7], pose.global_joints_positions[2])
        self.assertTrue(root_mask[2].bool().all())
        self.assertTrue(root_mask[7].bool().all())

        local_positions = observed[:, motion_rep.slice_dict["local_joints_positions"]].reshape(
            10, self.skeleton.nbjoints - 1, 3
        )
        local_position_mask = mask[:, motion_rep.slice_dict["local_joints_positions"]].reshape(
            10, self.skeleton.nbjoints - 1, 3
        )
        head_feature_idx = self.skeleton.bone_index["Head"] - 1
        expected_head = pose.global_joints_positions[1] - pose.global_joints_positions[0]
        expected_head[1] += pose.global_joints_positions[0, 1]
        torch.testing.assert_close(local_positions[2, head_feature_idx], expected_head)
        self.assertTrue(local_position_mask[2, head_feature_idx].bool().all())
        self.assertEqual(int(local_position_mask.sum()), 2 * 3)

        global_rotations = observed[:, motion_rep.slice_dict["global_rot_data"]].reshape(
            10, self.skeleton.nbjoints, 6
        )
        global_rotation_mask = mask[:, motion_rep.slice_dict["global_rot_data"]].reshape(
            10, self.skeleton.nbjoints, 6
        )
        head_idx = self.skeleton.bone_index["Head"]
        neck_idx = self.skeleton.bone_index["Neck"]
        torch.testing.assert_close(global_rotations[2, head_idx], matrix_to_cont6d(pose.global_joints_rots[0]))
        torch.testing.assert_close(global_rotations[7, neck_idx], matrix_to_cont6d(pose.global_joints_rots[1]))
        self.assertTrue(global_rotation_mask[2, head_idx].bool().all())
        self.assertTrue(global_rotation_mask[7, neck_idx].bool().all())
        self.assertEqual(int(global_rotation_mask.sum()), 2 * 6)

    def test_rotation_only_pose_is_valid_and_root_rotation_sets_heading(self):
        root_rotation = axis_angle_to_matrix(torch.tensor([[0.0, 0.4, 0.0]], dtype=torch.float32))
        pose = PoseConstraintSet(
            self.skeleton,
            rotation_frame_indices=[4],
            rotation_joint_names=["Hips"],
            global_joints_rots=root_rotation,
        )
        index_dict, data_dict = build_condition_dicts([pose])

        torch.testing.assert_close(index_dict["global_root_heading"][0], torch.tensor([4]))
        heading = data_dict["global_root_heading"][0]
        self.assertEqual(tuple(heading.shape), (1, 2))
        torch.testing.assert_close(torch.linalg.vector_norm(heading, dim=-1), torch.ones(1))

    def test_json_round_trip_uses_versioned_joint_name_schema(self):
        pose = self._make_sparse_pose()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pose.json"
            saved = save_constraints_lst(path, [pose])
            loaded = load_constraints_lst(path, self.skeleton)
            on_disk = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved, on_disk)
        self.assertEqual(on_disk[0]["type"], "pose")
        self.assertEqual(on_disk[0]["schema_version"], 1)
        self.assertEqual(on_disk[0]["coordinate_space"], "global")
        self.assertEqual([item["frame"] for item in on_disk[0]["keyframes"]], [2, 7])
        self.assertEqual(list(on_disk[0]["keyframes"][0]["joints"]), ["Hips", "Head"])
        self.assertIn("global_position", on_disk[0]["keyframes"][0]["joints"]["Head"])
        self.assertIn("global_rotation", on_disk[0]["keyframes"][0]["joints"]["Head"])

        self.assertEqual(len(loaded), 1)
        restored = loaded[0]
        self.assertIsInstance(restored, PoseConstraintSet)
        self.assertEqual(restored.position_joint_names, pose.position_joint_names)
        self.assertEqual(restored.rotation_joint_names, pose.rotation_joint_names)
        torch.testing.assert_close(restored.position_frame_indices, pose.position_frame_indices)
        torch.testing.assert_close(restored.rotation_frame_indices, pose.rotation_frame_indices)
        torch.testing.assert_close(restored.global_joints_positions, pose.global_joints_positions)
        torch.testing.assert_close(restored.global_joints_rots, pose.global_joints_rots)

    def test_crop_moves_sparse_observations_and_allows_empty_result(self):
        pose = self._make_sparse_pose()
        cropped = pose.crop_move(5, 9)
        torch.testing.assert_close(cropped.frame_indices, torch.tensor([2]))
        torch.testing.assert_close(cropped.position_frame_indices, torch.tensor([2, 2]))
        torch.testing.assert_close(cropped.rotation_frame_indices, torch.tensor([2]))
        self.assertEqual(cropped.position_joint_names, ("Hips", "Head"))
        self.assertEqual(cropped.rotation_joint_names, ("Neck",))

        empty = pose.crop_move(8, 10)
        self.assertEqual(empty.frame_indices.numel(), 0)
        index_dict, data_dict = defaultdict(list), defaultdict(list)
        empty.update_constraints(data_dict, index_dict)
        self.assertFalse(index_dict)
        self.assertFalse(data_dict)

    def test_rejects_invalid_sparse_observations(self):
        identity = torch.eye(3).unsqueeze(0)
        invalid_cases = [
            (
                "empty",
                {},
                ValueError,
                "at least one",
            ),
            (
                "partial channel",
                {"rotation_frame_indices": [1], "rotation_joint_names": ["Head"]},
                ValueError,
                "together",
            ),
            (
                "floating frame",
                {
                    "rotation_frame_indices": [1.0],
                    "rotation_joint_names": ["Head"],
                    "global_joints_rots": identity,
                },
                TypeError,
                "integers",
            ),
            (
                "unknown joint",
                {
                    "rotation_frame_indices": [1],
                    "rotation_joint_names": ["NotARealJoint"],
                    "global_joints_rots": identity,
                },
                ValueError,
                "Unknown",
            ),
            (
                "duplicate",
                {
                    "rotation_frame_indices": [1, 1],
                    "rotation_joint_names": ["Head", "Head"],
                    "global_joints_rots": identity.repeat(2, 1, 1),
                },
                ValueError,
                "Duplicate",
            ),
            (
                "bad rotation",
                {
                    "rotation_frame_indices": [1],
                    "rotation_joint_names": ["Head"],
                    "global_joints_rots": (torch.eye(3) * 2).unsqueeze(0),
                },
                ValueError,
                "valid rotation",
            ),
            (
                "missing root anchor",
                {
                    "position_frame_indices": [1],
                    "position_joint_names": ["Head"],
                    "global_joints_positions": [[0.0, 1.7, 0.0]],
                },
                ValueError,
                "root-position anchor",
            ),
        ]

        for label, kwargs, error_type, message in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(error_type, message):
                    PoseConstraintSet(self.skeleton, **kwargs)

    def test_rejects_unsupported_or_ambiguous_json(self):
        base = {
            "type": "pose",
            "schema_version": 1,
            "coordinate_space": "global",
            "keyframes": [
                {
                    "frame": 3,
                    "joints": {"Head": {"global_rotation": torch.eye(3).tolist()}},
                }
            ],
        }

        invalid_version = dict(base, schema_version=99)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            PoseConstraintSet.from_dict(self.skeleton, invalid_version)

        duplicate_frame = dict(base)
        duplicate_frame["keyframes"] = base["keyframes"] * 2
        with self.assertRaisesRegex(ValueError, "more than one keyframe"):
            PoseConstraintSet.from_dict(self.skeleton, duplicate_frame)

        unknown_field = dict(base)
        unknown_field["keyframes"] = [
            {"frame": 3, "joints": {"Head": {"rotation_degrees": [0, 10, 0]}}}
        ]
        with self.assertRaisesRegex(ValueError, "Unknown target field"):
            PoseConstraintSet.from_dict(self.skeleton, unknown_field)


if __name__ == "__main__":
    unittest.main()
