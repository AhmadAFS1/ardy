# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Optional

import torch
from torch import Tensor

from ardy.motion_rep.tools import compute_heading_angle
from ardy.skeleton import SkeletonBase

from .geometry import axis_angle_to_matrix, matrix_to_axis_angle


def create_pairs(tensor_A, tensor_B):
    pairs = torch.stack(
        (
            tensor_A[:, None].expand(-1, len(tensor_B)),
            tensor_B.expand(len(tensor_A), -1),
        ),
        dim=-1,
    ).reshape(-1, 2)
    return pairs


def compute_global_heading(global_joints_positions: Tensor, skeleton: SkeletonBase):
    root_heading_angle = compute_heading_angle(global_joints_positions, skeleton)
    global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)
    return global_root_heading


class PoseConstraintSet:
    """Sparse, precision constraints for arbitrary joints in global space.

    Positions and rotations are stored as independent ``(frame, joint)``
    observations.  This is intentionally different from
    :class:`FullBodyConstraintSet`: a pose constraint can lock only ``Head`` and
    ``Neck`` rotations at one frame, for example, without constraining the rest
    of the skeleton.

    Global positions require the root position at every affected frame.  ARDY's
    motion representation stores non-root positions relative to the root, so a
    world-space position is otherwise underdetermined.  Root anchors are used to
    populate the root x/z and y conditioning channels automatically.

    The JSON representation is versioned and uses joint names rather than
    skeleton-specific integer indices::

        {
          "type": "pose",
          "schema_version": 1,
          "coordinate_space": "global",
          "keyframes": [
            {
              "frame": 12,
              "joints": {
                "Hips": {"global_position": [0.0, 1.0, 0.0]},
                "Head": {"global_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
              }
            }
          ]
        }

    Args:
        skeleton: Skeleton that defines valid joint names and indices.
        position_frame_indices: One frame index per position observation.
        position_joint_names: One joint name per position observation.
        global_joints_positions: Sparse global xyz values, shape ``(P, 3)``.
        rotation_frame_indices: One frame index per rotation observation.
        rotation_joint_names: One joint name per rotation observation.
        global_joints_rots: Sparse global rotation matrices, shape
            ``(R, 3, 3)``.

    Note:
        Rotation matrices are validated as members of SO(3).  Duplicate
        observations for the same ``(frame, joint)`` and channel are rejected so
        conditioning never depends on implicit last-write-wins behavior.
    """

    name = "pose"
    schema_version = 1
    coordinate_space = "global"
    _rotation_atol = 1e-4

    def __init__(
        self,
        skeleton: SkeletonBase,
        *,
        position_frame_indices: Optional[Tensor | Sequence[int]] = None,
        position_joint_names: Optional[Sequence[str]] = None,
        global_joints_positions: Optional[Tensor | Sequence[Sequence[float]]] = None,
        rotation_frame_indices: Optional[Tensor | Sequence[int]] = None,
        rotation_joint_names: Optional[Sequence[str]] = None,
        global_joints_rots: Optional[Tensor | Sequence[Sequence[Sequence[float]]]] = None,
        _allow_empty: bool = False,
    ) -> None:
        self.skeleton = skeleton

        (
            self.position_frame_indices,
            self.position_joint_names,
            self.position_joint_indices,
            self.global_joints_positions,
        ) = self._validate_channel(
            channel="position",
            frame_indices=position_frame_indices,
            joint_names=position_joint_names,
            values=global_joints_positions,
            value_shape=(3,),
        )
        (
            self.rotation_frame_indices,
            self.rotation_joint_names,
            self.rotation_joint_indices,
            self.global_joints_rots,
        ) = self._validate_channel(
            channel="rotation",
            frame_indices=rotation_frame_indices,
            joint_names=rotation_joint_names,
            values=global_joints_rots,
            value_shape=(3, 3),
        )

        if not _allow_empty and not (len(self.position_frame_indices) or len(self.rotation_frame_indices)):
            raise ValueError("A pose constraint must contain at least one position or rotation observation.")

        self._validate_rotation_matrices()
        self._validate_position_root_anchors()

        all_frames = torch.cat([self.position_frame_indices, self.rotation_frame_indices])
        self.frame_indices = torch.unique(all_frames, sorted=True)

    @property
    def _value_device(self):
        return self.skeleton.device

    def _validate_channel(
        self,
        *,
        channel: str,
        frame_indices,
        joint_names,
        values,
        value_shape: tuple[int, ...],
    ):
        provided = (frame_indices is not None, joint_names is not None, values is not None)
        if any(provided) and not all(provided):
            raise ValueError(
                f"{channel} observations require frame indices, joint names, and values together."
            )

        if not any(provided):
            empty_values = torch.empty((0, *value_shape), device=self._value_device)
            return torch.empty(0, dtype=torch.long), tuple(), torch.empty(0, dtype=torch.long), empty_values

        raw_frames = torch.as_tensor(frame_indices)
        if raw_frames.ndim != 1:
            raise ValueError(f"{channel}_frame_indices must be one-dimensional, got shape {tuple(raw_frames.shape)}.")
        if raw_frames.dtype == torch.bool or raw_frames.is_floating_point() or raw_frames.is_complex():
            raise TypeError(f"{channel}_frame_indices must contain integers.")
        frames = raw_frames.to(device="cpu", dtype=torch.long).clone()
        if (frames < 0).any():
            raise ValueError(f"{channel}_frame_indices cannot contain negative frames.")

        if isinstance(joint_names, (str, bytes)) or not isinstance(joint_names, Sequence):
            raise TypeError(f"{channel}_joint_names must be a sequence of joint-name strings.")
        names = tuple(joint_names)
        if any(not isinstance(name, str) for name in names):
            raise TypeError(f"{channel}_joint_names must contain only strings.")
        unknown_names = sorted(set(names).difference(self.skeleton.bone_index))
        if unknown_names:
            raise ValueError(
                f"Unknown {channel} joint name(s) for {self.skeleton.name}: {', '.join(unknown_names)}."
            )

        value_tensor = torch.as_tensor(values, device=self._value_device)
        if not value_tensor.is_floating_point():
            value_tensor = value_tensor.to(torch.get_default_dtype())
        value_tensor = value_tensor.clone()
        expected_shape = (len(frames), *value_shape)
        if tuple(value_tensor.shape) != expected_shape:
            raise ValueError(
                f"global_joints_{'positions' if channel == 'position' else 'rots'} must have shape "
                f"{expected_shape}, got {tuple(value_tensor.shape)}."
            )
        if len(names) != len(frames):
            raise ValueError(
                f"{channel}_joint_names must have {len(frames)} entries, got {len(names)}."
            )
        if not torch.isfinite(value_tensor).all():
            raise ValueError(f"{channel} values must contain only finite numbers.")

        joint_indices = torch.tensor([self.skeleton.bone_index[name] for name in names], dtype=torch.long)
        observations = list(zip(frames.tolist(), joint_indices.tolist()))
        if len(set(observations)) != len(observations):
            raise ValueError(f"Duplicate {channel} observation for the same frame and joint.")

        return frames, names, joint_indices, value_tensor

    def _validate_rotation_matrices(self) -> None:
        if not len(self.global_joints_rots):
            return
        rotations = self.global_joints_rots
        # CPU linalg does not implement det for half/bfloat16.  Validation in
        # float32 is still substantially tighter than the accepted tolerance.
        if rotations.dtype in (torch.float16, torch.bfloat16):
            rotations = rotations.float()
        identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device).expand_as(rotations)
        gram = rotations.transpose(-1, -2) @ rotations
        orthogonal = torch.isclose(gram, identity, atol=self._rotation_atol, rtol=self._rotation_atol).all(
            dim=(-2, -1)
        )
        determinants = torch.linalg.det(rotations)
        proper = torch.isclose(
            determinants,
            torch.ones_like(determinants),
            atol=self._rotation_atol,
            rtol=self._rotation_atol,
        )
        invalid = torch.where(~(orthogonal & proper))[0]
        if len(invalid):
            bad = int(invalid[0])
            frame = int(self.rotation_frame_indices[bad])
            joint = self.rotation_joint_names[bad]
            raise ValueError(
                f"Global rotation for joint {joint!r} at frame {frame} is not a valid rotation matrix."
            )

    def _validate_position_root_anchors(self) -> None:
        if not len(self.position_frame_indices):
            return
        root_idx = self.skeleton.root_idx
        position_frames = set(self.position_frame_indices.tolist())
        root_frames = set(self.position_frame_indices[self.position_joint_indices == root_idx].tolist())
        missing_root_frames = sorted(position_frames.difference(root_frames))
        if missing_root_frames:
            root_name = self.skeleton.bone_order_names[root_idx]
            frames = ", ".join(str(frame) for frame in missing_root_frames)
            raise ValueError(
                "Global joint positions require a root-position anchor at every affected frame; "
                f"add {root_name!r} at frame(s): {frames}."
            )

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        if len(self.position_frame_indices):
            position_indices = torch.stack(
                [self.position_frame_indices, self.position_joint_indices], dim=-1
            )
            data_dict["global_joints_positions"].append(self.global_joints_positions)
            index_dict["global_joints_positions"].append(position_indices)

            root_mask = self.position_joint_indices == self.skeleton.root_idx
            root_frames = self.position_frame_indices[root_mask]
            root_positions = self.global_joints_positions[root_mask.to(self.global_joints_positions.device)]
            data_dict["root_2d"].append(root_positions[:, [0, 2]])
            index_dict["root_2d"].append(root_frames)
            data_dict["root_y_pos"].append(root_positions[:, 1])
            index_dict["root_y_pos"].append(root_frames)

        if len(self.rotation_frame_indices):
            rotation_indices = torch.stack(
                [self.rotation_frame_indices, self.rotation_joint_indices], dim=-1
            )
            data_dict["global_joints_rots"].append(self.global_joints_rots)
            index_dict["global_joints_rots"].append(rotation_indices)

            # Root heading is a duplicated channel in ArdyMotionRep.  Keep it
            # consistent whenever the sparse pose explicitly locks root rotation.
            root_mask = self.rotation_joint_indices == self.skeleton.root_idx
            if root_mask.any():
                root_frames = self.rotation_frame_indices[root_mask]
                root_rotations = self.global_joints_rots[root_mask.to(self.global_joints_rots.device)]
                neutral = self.skeleton.neutral_joints.to(
                    device=root_rotations.device, dtype=root_rotations.dtype
                )
                right_hip, left_hip = self.skeleton.hip_joint_idx
                neutral_hip_vector = neutral[right_hip] - neutral[left_hip]
                rotated_hip_vector = torch.einsum("nij,j->ni", root_rotations, neutral_hip_vector)
                heading_angle = torch.atan2(rotated_hip_vector[:, 2], -rotated_hip_vector[:, 0])
                heading = torch.stack([torch.cos(heading_angle), torch.sin(heading_angle)], dim=-1)
                data_dict["global_root_heading"].append(heading)
                index_dict["global_root_heading"].append(root_frames)

    def crop_move(self, start: int, end: int):
        """Crop observations to ``[start, end)`` and move them to frame zero."""
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("Crop bounds must be integers.")
        if start < 0 or end < start:
            raise ValueError(f"Invalid crop range [{start}, {end}).")

        position_mask = (self.position_frame_indices >= start) & (self.position_frame_indices < end)
        rotation_mask = (self.rotation_frame_indices >= start) & (self.rotation_frame_indices < end)
        position_names = [
            name for name, keep in zip(self.position_joint_names, position_mask.tolist()) if keep
        ]
        rotation_names = [
            name for name, keep in zip(self.rotation_joint_names, rotation_mask.tolist()) if keep
        ]
        position_value_mask = position_mask.to(self.global_joints_positions.device)
        rotation_value_mask = rotation_mask.to(self.global_joints_rots.device)
        return PoseConstraintSet(
            self.skeleton,
            position_frame_indices=self.position_frame_indices[position_mask] - start,
            position_joint_names=position_names,
            global_joints_positions=self.global_joints_positions[position_value_mask],
            rotation_frame_indices=self.rotation_frame_indices[rotation_mask] - start,
            rotation_joint_names=rotation_names,
            global_joints_rots=self.global_joints_rots[rotation_value_mask],
            _allow_empty=True,
        )

    def get_save_info(self) -> dict:
        """Return the canonical, versioned JSON-ready pose schema."""
        keyframes: dict[int, dict[str, dict[str, Tensor]]] = {}

        for frame, joint_name, position in zip(
            self.position_frame_indices.tolist(),
            self.position_joint_names,
            self.global_joints_positions,
        ):
            joint_targets = keyframes.setdefault(frame, {}).setdefault(joint_name, {})
            joint_targets["global_position"] = position

        for frame, joint_name, rotation in zip(
            self.rotation_frame_indices.tolist(),
            self.rotation_joint_names,
            self.global_joints_rots,
        ):
            joint_targets = keyframes.setdefault(frame, {}).setdefault(joint_name, {})
            joint_targets["global_rotation"] = rotation

        ordered_keyframes = []
        for frame in sorted(keyframes):
            joints = keyframes[frame]
            ordered_joints = {
                name: joints[name]
                for name in self.skeleton.bone_order_names
                if name in joints
            }
            ordered_keyframes.append({"frame": frame, "joints": ordered_joints})

        return {
            "type": self.name,
            "schema_version": self.schema_version,
            "coordinate_space": self.coordinate_space,
            "keyframes": ordered_keyframes,
        }

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: Mapping):
        """Build a sparse pose constraint from its versioned JSON schema."""
        if not isinstance(dico, Mapping):
            raise TypeError("A pose constraint must be loaded from a JSON object.")
        version = dico.get("schema_version")
        if version != cls.schema_version:
            raise ValueError(
                f"Unsupported pose schema_version {version!r}; expected {cls.schema_version}."
            )
        coordinate_space = dico.get("coordinate_space")
        if coordinate_space != cls.coordinate_space:
            raise ValueError(
                f"Unsupported pose coordinate_space {coordinate_space!r}; expected {cls.coordinate_space!r}."
            )

        keyframes = dico.get("keyframes")
        if not isinstance(keyframes, list) or not keyframes:
            raise ValueError("Pose keyframes must be a non-empty list.")

        position_frames = []
        position_names = []
        positions = []
        rotation_frames = []
        rotation_names = []
        rotations = []
        seen_frames = set()

        for keyframe in keyframes:
            if not isinstance(keyframe, Mapping):
                raise TypeError("Each pose keyframe must be a JSON object.")
            frame = keyframe.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise TypeError("Each pose keyframe frame must be an integer.")
            if frame < 0:
                raise ValueError("Pose keyframe frames cannot be negative.")
            if frame in seen_frames:
                raise ValueError(f"Pose schema contains more than one keyframe object for frame {frame}.")
            seen_frames.add(frame)

            joints = keyframe.get("joints")
            if not isinstance(joints, Mapping) or not joints:
                raise ValueError(f"Pose keyframe at frame {frame} must contain a non-empty joints object.")

            for joint_name, targets in joints.items():
                if not isinstance(joint_name, str):
                    raise TypeError("Pose joint names must be strings.")
                if not isinstance(targets, Mapping) or not targets:
                    raise ValueError(
                        f"Pose target for joint {joint_name!r} at frame {frame} must be a non-empty object."
                    )
                unknown_targets = set(targets).difference({"global_position", "global_rotation"})
                if unknown_targets:
                    fields = ", ".join(sorted(unknown_targets))
                    raise ValueError(
                        f"Unknown target field(s) for joint {joint_name!r} at frame {frame}: {fields}."
                    )
                if "global_position" in targets:
                    position_frames.append(frame)
                    position_names.append(joint_name)
                    positions.append(targets["global_position"])
                if "global_rotation" in targets:
                    rotation_frames.append(frame)
                    rotation_names.append(joint_name)
                    rotations.append(targets["global_rotation"])

        return cls(
            skeleton,
            position_frame_indices=position_frames or None,
            position_joint_names=position_names or None,
            global_joints_positions=positions or None,
            rotation_frame_indices=rotation_frames or None,
            rotation_joint_names=rotation_names or None,
            global_joints_rots=rotations or None,
        )


class Root2DConstraintSet:
    name = "root2d"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        root_2d: Tensor,
        global_root_heading: Optional[Tensor] = None,
        to_crop: bool = False,
    ) -> None:
        self.skeleton = skeleton
        if to_crop:
            root_2d = root_2d[frame_indices]
            if global_root_heading is not None:
                global_root_heading = global_root_heading[frame_indices]
        else:
            assert len(root_2d) == len(frame_indices), "The number of root 2d should be match the number of frames"
            if global_root_heading is not None:
                assert len(global_root_heading) == len(frame_indices), (
                    "The number of global root heading should match the number of frames"
                )
        self.root_2d = root_2d
        self.global_root_heading = global_root_heading
        self.frame_indices = frame_indices

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        data_dict["root_2d"].append(self.root_2d)
        index_dict["root_2d"].append(self.frame_indices)
        if self.global_root_heading is not None:
            # Convert heading angles to [cos, sin] format
            # self.global_root_heading contains angles in radians
            heading_cos_sin = torch.stack(
                [
                    torch.cos(self.global_root_heading),
                    torch.sin(self.global_root_heading),
                ],
                dim=-1,
            )
            data_dict["global_root_heading"].append(heading_cos_sin)
            index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int):
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return (
            Root2DConstraintSet(
                self.skeleton,
                self.frame_indices[mask] - start,
                self.root_2d[mask],
                self.global_root_heading[mask],
            )
            if self.global_root_heading is not None
            else Root2DConstraintSet(self.skeleton, self.frame_indices[mask] - start, self.root_2d[mask])
        )

    def get_save_info(self):
        info = {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "root_2d": self.root_2d,
        }
        if self.global_root_heading is not None:
            info["global_root_heading"] = self.global_root_heading
        return info

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict):
        device = skeleton.device
        root_2d_key = "root_2d" if "root_2d" in dico else "smooth_root_2d"
        return cls(
            skeleton,
            frame_indices=torch.tensor(dico["frame_indices"]),
            root_2d=torch.tensor(dico[root_2d_key], device=device),
            global_root_heading=torch.tensor(dico["global_root_heading"]) if "global_root_heading" in dico else None,
        )


class FullBodyConstraintSet:
    name = "fullbody"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        root_2d: Optional[Tensor] = None,
        to_crop: bool = False,
    ):
        self.skeleton = skeleton
        self.frame_indices = frame_indices

        if to_crop:
            global_joints_positions = global_joints_positions[frame_indices]
            global_joints_rots = global_joints_rots[frame_indices]
            if root_2d is not None:
                root_2d = root_2d[frame_indices]
        else:
            assert len(global_joints_positions) == len(frame_indices), (
                "The number of global positions should be match the number of frames"
            )
            assert len(global_joints_rots) == len(frame_indices), (
                "The number of global joint rotations should be match the number of frames"
            )

            if root_2d is not None:
                assert len(root_2d) == len(frame_indices), (
                    "The number of root 2d (if specified) should be match the number of frames"
                )

        if root_2d is None:
            # substitute root 2d with the real root
            root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]

        # root y: from smooth or pelvis is the same
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]

        self.global_joints_positions = global_joints_positions
        self.global_joints_rots = global_joints_rots
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self.root_2d = root_2d

    def update_constraints(self, data_dict, index_dict):
        nbjoints = self.skeleton.nbjoints
        indices_lst = create_pairs(
            self.frame_indices,
            torch.arange(nbjoints),
        )
        data_dict["global_joints_positions"].append(
            self.global_joints_positions.reshape(-1, 3)
        )  # flatten the global positions
        index_dict["global_joints_positions"].append(indices_lst)

        # global rotations are not used here

        # also constraint root 2d to get the same full body
        # maybe keep storing the hips offset, if we smooth it ourselves
        data_dict["root_2d"].append(self.root_2d)
        index_dict["root_2d"].append(self.frame_indices)

        # constraint the y pos of the root
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

        # constraint the global heading
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int):
        mask = (self.frame_indices >= start) & (self.frame_indices < end)
        return FullBodyConstraintSet(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.root_2d[mask],
        )

    def get_save_info(self):
        local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
        local_joints_rot = matrix_to_axis_angle(local_joints_rot)

        root_positions = self.global_joints_positions[:, self.skeleton.root_idx]
        return {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "root_2d": self.root_2d,
        }

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict):
        frame_indices = torch.tensor(dico["frame_indices"])
        device = skeleton.device
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            axis_angle_to_matrix(torch.tensor(dico["local_joints_rot"], device=device)),
            torch.tensor(dico["root_positions"], device=device),
        )
        root_2d = None
        if "root_2d" in dico:
            root_2d = torch.tensor(dico["root_2d"], device=device)
        elif "smooth_root_2d" in dico:
            root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            root_2d=root_2d,
        )


class EndEffectorConstraintSet:
    name = "end-effector"

    def __init__(
        self,
        skeleton: SkeletonBase,
        frame_indices: Tensor,
        global_joints_positions: Tensor,
        global_joints_rots: Tensor,
        root_2d: Optional[Tensor],
        *,
        joint_names: list[str],
        to_crop: bool = False,
    ) -> None:
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.joint_names = joint_names

        # joint_names are constant for all the frames
        rot_joint_names, pos_joint_names = self.skeleton.expand_joint_names(self.joint_names)
        # indexing works for motion_rep with smooth root only (contains pelvis index)
        self.pos_indices = torch.tensor([self.skeleton.bone_index[jname] for jname in pos_joint_names])
        self.rot_indices = torch.tensor([self.skeleton.bone_index[jname] for jname in rot_joint_names])

        if to_crop:
            global_joints_positions = global_joints_positions[frame_indices]
            global_joints_rots = global_joints_rots[frame_indices]
            if root_2d is not None:
                root_2d = root_2d[frame_indices]
        else:
            assert len(global_joints_positions) == len(frame_indices), (
                "The number of global positions should be match the number of frames"
            )
            assert len(global_joints_rots) == len(frame_indices), (
                "The number of global joint rotations should be match the number of frames"
            )
            if root_2d is not None:
                assert len(root_2d) == len(frame_indices), (
                    "The number of root 2d (if specified) should be match the number of frames"
                )

        if root_2d is None:
            # substitute root 2d with the real root
            root_2d = global_joints_positions[:, skeleton.root_idx, [0, 2]]

        # root y: from smooth or pelvis is the same
        self.root_y_pos = global_joints_positions[:, skeleton.root_idx, 1]

        self.global_joints_positions = global_joints_positions
        self.global_root_heading = compute_global_heading(global_joints_positions, skeleton)
        self.global_joints_rots = global_joints_rots
        self.root_2d = root_2d

    def update_constraints(self, data_dict, index_dict):
        crop_frames_indexing = torch.arange(len(self.frame_indices))

        # constraint positions
        pos_indices_real = create_pairs(
            self.frame_indices,
            self.pos_indices,
        )
        pos_indices_crop = create_pairs(
            crop_frames_indexing,
            self.pos_indices,
        )
        data_dict["global_joints_positions"].append(self.global_joints_positions[tuple(pos_indices_crop.T)])
        index_dict["global_joints_positions"].append(pos_indices_real)

        # constraint rotations
        rot_indices_real = create_pairs(
            self.frame_indices,
            self.rot_indices,
        )
        rot_indices_crop = create_pairs(
            crop_frames_indexing,
            self.rot_indices,
        )
        data_dict["global_joints_rots"].append(self.global_joints_rots[tuple(rot_indices_crop.T)])
        index_dict["global_joints_rots"].append(rot_indices_real)

        # also constraint root 2d to get the same full body
        # maybe keep storing the hips offset, if we smooth it ourselves
        data_dict["root_2d"].append(self.root_2d)
        index_dict["root_2d"].append(self.frame_indices)

        # constraint the y pos of the root
        data_dict["root_y_pos"].append(self.root_y_pos)
        index_dict["root_y_pos"].append(self.frame_indices)

        # constraint the global heading
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)

    def crop_move(self, start: int, end: int):
        mask = (self.frame_indices >= start) & (self.frame_indices < end)

        cls = type(self)
        kwargs = {}
        if not hasattr(cls, "joint_names"):
            kwargs["joint_names"] = self.joint_names

        return cls(
            self.skeleton,
            self.frame_indices[mask] - start,
            self.global_joints_positions[mask],
            self.global_joints_rots[mask],
            self.root_2d[mask],
            **kwargs,
        )

    def get_save_info(self):
        local_joints_rot = self.skeleton.global_rots_to_local_rots(self.global_joints_rots)
        local_joints_rot = matrix_to_axis_angle(local_joints_rot)

        root_positions = self.global_joints_positions[:, self.skeleton.root_idx]
        output = {
            "type": self.name,
            "frame_indices": self.frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "root_2d": self.root_2d,
        }
        if not hasattr(self.__class__, "joint_names"):
            # save the joint_names for this base class
            # but not for children
            output["joint_names"] = self.joint_names
        return output

    @classmethod
    def from_dict(cls, skeleton: SkeletonBase, dico: dict):
        frame_indices = torch.tensor(dico["frame_indices"])
        device = skeleton.device
        global_joints_rots, global_joints_positions, _ = skeleton.fk(
            axis_angle_to_matrix(torch.tensor(dico["local_joints_rot"], device=device)),
            torch.tensor(dico["root_positions"], device=device),
        )
        root_2d = None
        if "root_2d" in dico:
            root_2d = torch.tensor(dico["root_2d"], device=device)
        elif "smooth_root_2d" in dico:
            root_2d = torch.tensor(dico["smooth_root_2d"], device=device)

        kwargs = {}
        if not hasattr(cls, "joint_names"):
            kwargs["joint_names"] = dico["joint_names"]

        return cls(
            skeleton,
            frame_indices=frame_indices,
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            root_2d=root_2d,
            **kwargs,
        )


class LeftHandConstraintSet(EndEffectorConstraintSet):
    name = "left-hand"
    joint_names: list[str] = ["LeftHand", "Hips"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class RightHandConstraintSet(EndEffectorConstraintSet):
    name = "right-hand"
    joint_names: list[str] = ["RightHand", "Hips"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class LeftFootConstraintSet(EndEffectorConstraintSet):
    name = "left-foot"
    joint_names: list[str] = ["LeftFoot", "Hips"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


class RightFootConstraintSet(EndEffectorConstraintSet):
    name = "right-foot"
    joint_names: list[str] = ["RightFoot", "Hips"]

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args, joint_names=self.joint_names, **kwargs)


TYPE_TO_CLASS = {
    "pose": PoseConstraintSet,
    "root2d": Root2DConstraintSet,
    "fullbody": FullBodyConstraintSet,
    "left-hand": LeftHandConstraintSet,
    "right-hand": RightHandConstraintSet,
    "left-foot": LeftFootConstraintSet,
    "right-foot": RightFootConstraintSet,
    "end-effector": EndEffectorConstraintSet,
}


def load_constraints_lst(path_or_data: str | PathLike | list, skeleton: SkeletonBase):
    from ardy.tools import load_json

    if isinstance(path_or_data, (str, PathLike)):
        saved = load_json(path_or_data)
    else:
        saved = path_or_data

    if not isinstance(saved, list):
        raise TypeError("A constraints file must contain a JSON list.")

    constraints_lst = []
    for el in saved:
        if not isinstance(el, Mapping):
            raise TypeError("Each saved constraint must be a JSON object.")
        constraint_type = el.get("type")
        if constraint_type not in TYPE_TO_CLASS:
            raise ValueError(f"Unknown constraint type: {constraint_type!r}.")
        cls = TYPE_TO_CLASS[constraint_type]
        constraints_lst.append(cls.from_dict(skeleton, el))
    return constraints_lst


def save_constraints_lst(path: str | PathLike, constraints_lst):
    from ardy.tools import save_json

    if not constraints_lst:
        print("The constraints lst is empty. Skip saving")
        return

    to_save = []

    def tensor_to_list(obj):
        """Recursively convert tensors to lists for JSON serialization."""
        if isinstance(obj, Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(v) for v in obj]
        else:
            return obj

    for constraint in constraints_lst:
        constraint_info = constraint.get_save_info()
        # Convert all tensors to lists for JSON serialization
        constraint_info = tensor_to_list(constraint_info)
        to_save.append(constraint_info)

    save_json(path, to_save)
    print(f"Saved constraints to {path}")
    return to_save
