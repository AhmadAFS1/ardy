# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch

from ardy.skeleton.kinematics import batch_rigid_transform

SKIN_NAME = "skin_standard.npz"

MATERIAL_BODY = 0
MATERIAL_SKIN = 1
MATERIAL_SCLERA = 2
MATERIAL_IRIS = 3
MATERIAL_PUPIL = 4
MATERIAL_BROW = 5
MATERIAL_MOUTH = 6
MATERIAL_EYE_BACKING = 7

# The released Core skin is a motion-reference mesh, but its nearly closed
# eyelid geometry is easily interpreted as a squint by downstream video
# transfer models.  Core has no facial rig, so correct the bind pose once,
# before linear-blend skinning. Symmetric, smoothly localized displacement
# enlarges the lid silhouette without changing topology, joints, weights, or
# any authored motion. The opening remains slightly emphasized so downstream
# video models read open eyes without making the guidance face look cartoonish.
_EYE_CENTER_X = 0.043
_EYE_CENTER_Y = 1.775
_EYE_HORIZONTAL_SIGMA = 0.019
_EYE_VERTICAL_SIGMA = 0.014
_EYE_VERTICAL_OPENING_METERS = 0.004
_EYE_HORIZONTAL_OPENING_METERS = 0.001

_MOUTH_CORNER_X = 0.036
_MOUTH_CENTER_Y = 1.705
_MOUTH_HORIZONTAL_SIGMA = 0.014
_MOUTH_VERTICAL_SIGMA = 0.018
_MOUTH_CORNER_LIFT_METERS = 0.003


def neutralize_core_eyelids(bind_vertices: torch.Tensor) -> torch.Tensor:
    """Return Core bind vertices with a relaxed, larger open-eye shape."""

    vertices = bind_vertices.clone()
    x, y, z = vertices.unbind(dim=-1)
    eye_side = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
    horizontal_offset = x - eye_side * _EYE_CENTER_X
    vertical_offset = y - _EYE_CENTER_Y
    eye_field = torch.exp(
        -0.5
        * (
            (horizontal_offset / _EYE_HORIZONTAL_SIGMA) ** 2
            + (vertical_offset / _EYE_VERTICAL_SIGMA) ** 2
        )
    )
    # Limit the edit to the forward facial surface rather than the back of the
    # head. The smooth masks avoid seams or shading discontinuities.
    front_face_mask = torch.sigmoid((z - 0.055) * 100.0)
    lid_direction = torch.tanh(vertical_offset / 0.0025)
    corner_direction = torch.tanh(horizontal_offset / 0.0025)
    vertices[:, 0] = (
        x + _EYE_HORIZONTAL_OPENING_METERS * corner_direction * eye_field * front_face_mask
    )
    vertices[:, 1] = (
        y + _EYE_VERTICAL_OPENING_METERS * lid_direction * eye_field * front_face_mask
    )
    return vertices


def neutralize_core_mouth(bind_vertices: torch.Tensor) -> torch.Tensor:
    """Return Core bind vertices with relaxed, subtly lifted mouth corners."""

    vertices = bind_vertices.clone()
    x, y, z = vertices.unbind(dim=-1)
    distance_to_corner = torch.minimum(torch.abs(x - _MOUTH_CORNER_X), torch.abs(x + _MOUTH_CORNER_X))
    vertical_offset = y - _MOUTH_CENTER_Y
    corner_field = torch.exp(
        -0.5
        * (
            (distance_to_corner / _MOUTH_HORIZONTAL_SIGMA) ** 2
            + (vertical_offset / _MOUTH_VERTICAL_SIGMA) ** 2
        )
    )
    front_face_mask = torch.sigmoid((z - 0.068) * 120.0)
    vertices[:, 1] = y + _MOUTH_CORNER_LIFT_METERS * corner_field * front_face_mask
    return vertices


def _ellipsoid_mesh(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    latitude_segments: int = 12,
    longitude_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a compact UV ellipsoid without adding a trimesh dependency."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    cx, cy, cz = center
    rx, ry, rz = radii

    vertices.append((cx, cy + ry, cz))
    for latitude in range(1, latitude_segments):
        phi = np.pi / 2.0 - np.pi * latitude / latitude_segments
        cos_phi = float(np.cos(phi))
        sin_phi = float(np.sin(phi))
        for longitude in range(longitude_segments):
            theta = 2.0 * np.pi * longitude / longitude_segments
            vertices.append(
                (
                    cx + rx * cos_phi * float(np.cos(theta)),
                    cy + ry * sin_phi,
                    cz + rz * cos_phi * float(np.sin(theta)),
                )
            )
    bottom_index = len(vertices)
    vertices.append((cx, cy - ry, cz))

    first_ring = 1
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        faces.append((0, first_ring + longitude, first_ring + next_longitude))

    for latitude in range(latitude_segments - 2):
        ring = 1 + latitude * longitude_segments
        next_ring = ring + longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            faces.append((ring + longitude, next_ring + longitude, next_ring + next_longitude))
            faces.append((ring + longitude, next_ring + next_longitude, ring + next_longitude))

    last_ring = 1 + (latitude_segments - 2) * longitude_segments
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        faces.append((last_ring + longitude, bottom_index, last_ring + next_longitude))

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


class CoreSkin:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        skin_data_path = Path(skeleton.folder) / SKIN_NAME

        assert skeleton.neutral_joints is not None, "CoreSkeleton27 must have neutral joints instantiated"

        device = skeleton.neutral_joints.device

        # bind_rig_transform: [R, 4, 4]
        # bind_vertices: [V, 3]
        # faces: [F, 3]
        # lbs indices, lbs weights: [V, W] (W = max (num joints vertice is related to), in our case W=5)
        skin_data = np.load(skin_data_path)
        bind_rig_np = np.array(skin_data["bind_rig_transform"], dtype=np.float32)
        self.bind_rig_transform = torch.from_numpy(bind_rig_np).to(device=device, dtype=torch.float)
        # Precompute the inverse in numpy to avoid torch lazy evaluation issues
        bind_rig_inv_np = np.linalg.inv(bind_rig_np)
        self.bind_rig_transform_inv = torch.from_numpy(bind_rig_inv_np).to(device=device, dtype=torch.float)
        raw_bind_vertices = torch.tensor(skin_data["bind_vertices"], device=device, dtype=torch.float)
        open_eye_vertices = neutralize_core_eyelids(raw_bind_vertices)
        self.bind_vertices = neutralize_core_mouth(open_eye_vertices)
        self.faces = torch.tensor(skin_data["faces"], device=device, dtype=torch.long)
        self.lbs_indices = torch.tensor(skin_data["lbs_indices"], device=device, dtype=torch.long)
        self.lbs_weights = torch.tensor(skin_data["lbs_weights"], device=device, dtype=torch.float)
        self.vertex_material_ids = torch.full(
            (self.bind_vertices.shape[0],),
            MATERIAL_BODY,
            device=device,
            dtype=torch.long,
        )
        head_surface = (self.bind_vertices[:, 1] > 1.60) & (torch.abs(self.bind_vertices[:, 0]) < 0.18)
        self.vertex_material_ids[head_surface] = MATERIAL_SKIN

        # double check the rig matches expected skeleton
        rig_joint_names = list(skin_data["rig_joint_names"])  # list(str) : [R]
        for sname, rname in zip(self.skeleton.bone_order_names, rig_joint_names):
            if sname != rname:
                raise ValueError(f"MISMATCH in skinnging rig: expected='{sname}' vs rig='{rname}'")

        head_joint_index = self.skeleton.bone_order_names.index("Head")
        self._add_guidance_face(head_joint_index)
        self._configure_blink_deformation()

    def _append_head_ellipsoid(
        self,
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        material_id: int,
        head_joint_index: int,
    ) -> None:
        vertices_np, faces_np = _ellipsoid_mesh(center, radii)
        device = self.bind_vertices.device
        vertex_offset = self.bind_vertices.shape[0]
        vertices = torch.tensor(vertices_np, device=device, dtype=self.bind_vertices.dtype)
        faces = torch.tensor(faces_np + vertex_offset, device=device, dtype=self.faces.dtype)

        influence_count = self.lbs_indices.shape[1]
        indices = torch.full(
            (vertices.shape[0], influence_count),
            head_joint_index,
            device=device,
            dtype=self.lbs_indices.dtype,
        )
        weights = torch.zeros(
            (vertices.shape[0], influence_count),
            device=device,
            dtype=self.lbs_weights.dtype,
        )
        weights[:, 0] = 1.0
        material_ids = torch.full(
            (vertices.shape[0],),
            material_id,
            device=device,
            dtype=self.vertex_material_ids.dtype,
        )

        self.bind_vertices = torch.cat((self.bind_vertices, vertices), dim=0)
        self.faces = torch.cat((self.faces, faces), dim=0)
        self.lbs_indices = torch.cat((self.lbs_indices, indices), dim=0)
        self.lbs_weights = torch.cat((self.lbs_weights, weights), dim=0)
        self.vertex_material_ids = torch.cat((self.vertex_material_ids, material_ids), dim=0)

    def _add_guidance_face(self, head_joint_index: int) -> None:
        """Add high-contrast neutral facial landmarks rigidly bound to Head."""

        for eye_x in (-_EYE_CENTER_X, _EYE_CENTER_X):
            self._append_head_ellipsoid(
                (eye_x, _EYE_CENTER_Y, 0.103),
                (0.021, 0.011, 0.002),
                MATERIAL_EYE_BACKING,
                head_joint_index,
            )
            self._append_head_ellipsoid(
                (eye_x, _EYE_CENTER_Y, 0.106),
                (0.017, 0.007, 0.002),
                MATERIAL_SCLERA,
                head_joint_index,
            )
            self._append_head_ellipsoid(
                (eye_x, _EYE_CENTER_Y, 0.109),
                (0.0053, 0.0053, 0.002),
                MATERIAL_IRIS,
                head_joint_index,
            )
            self._append_head_ellipsoid(
                (eye_x, _EYE_CENTER_Y, 0.112),
                (0.0023, 0.0023, 0.002),
                MATERIAL_PUPIL,
                head_joint_index,
            )
            self._append_head_ellipsoid(
                (eye_x, 1.800, 0.119),
                (0.026, 0.0035, 0.002),
                MATERIAL_BROW,
                head_joint_index,
            )

        self._append_head_ellipsoid(
            (0.0, _MOUTH_CENTER_Y, 0.123),
            (0.038, 0.003, 0.002),
            MATERIAL_MOUTH,
            head_joint_index,
        )

    def _configure_blink_deformation(self) -> None:
        """Precompute local vertical offsets and weights for procedural blinks."""

        x, y, z = self.bind_vertices.unbind(dim=-1)
        eye_side = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
        horizontal_offset = x - eye_side * _EYE_CENTER_X
        vertical_offset = y - _EYE_CENTER_Y
        eyelid_field = torch.exp(
            -0.5
            * (
                (horizontal_offset / 0.018) ** 2
                + (vertical_offset / 0.010) ** 2
            )
        )
        front_face_mask = torch.sigmoid((z - 0.055) * 110.0)
        base_surface = (self.vertex_material_ids == MATERIAL_BODY) | (
            self.vertex_material_ids == MATERIAL_SKIN
        )
        weights = eyelid_field * front_face_mask * base_surface.to(eyelid_field.dtype)

        guidance_eye = (
            (self.vertex_material_ids == MATERIAL_SCLERA)
            | (self.vertex_material_ids == MATERIAL_IRIS)
            | (self.vertex_material_ids == MATERIAL_PUPIL)
        )
        weights = torch.where(guidance_eye, torch.ones_like(weights), weights)
        self.blink_vertical_offsets = vertical_offset
        self.blink_weights = weights


    def lbs(self, posed_transform):
        bind_rig_transform_inv = self.bind_rig_transform_inv
        bind_vertices = self.bind_vertices
        lbs_weights = self.lbs_weights
        # posed_transform: [B, F, J, 4, 4] or [B, J, 4, 4] or [J, 4, 4]
        # unsqueeze to match posed_transform dim
        for _ in range(posed_transform.dim() - 3):
            bind_rig_transform_inv = bind_rig_transform_inv.unsqueeze(0)
            bind_vertices = bind_vertices.unsqueeze(0)
            lbs_weights = lbs_weights.unsqueeze(0)
            # bind_rig_transform_inv: [..., R, 4, 4]
            # bind_vertices: [..., V, 3]
            # lbs_weights: [..., V, W]

        affine_mat = (posed_transform @ bind_rig_transform_inv)[..., :3, :]  # [..., J, 3, 4]
        vs = (
            affine_mat[..., self.lbs_indices, :, :]
            @ torch.concat([bind_vertices, torch.ones_like(bind_vertices[..., 0:1])], dim=-1)[..., None, :, None]
        )  # [..., V, W, 3, 1]
        ws = lbs_weights[..., None, None]
        resv = (vs * ws).sum(dim=-3).squeeze(-1)  # [..., V, 3]
        return resv

    def skin(self, joint_rotmat, joint_pos, rot_is_global=False):
        """
        joint_rotmat: [T, J, 3, 3] local or global joint rotation matrices
        joint_pos: [T, J, 3] global joint positions
        rot_is_global: bool, if True, joint_rotmat is global rotation matrices, otherwise it is local rotation matrices and FK is performed internally
        """
        nF, nJ = joint_pos.shape[:2]
        device = joint_rotmat.device

        # prepare full transformation matrices
        fk_transform = torch.eye(4, device=device)[None, None].repeat(nF, nJ, 1, 1)
        fk_transform[..., :3, 3] = joint_pos
        if rot_is_global:
            fk_transform[..., :3, :3] = joint_rotmat
        else:
            neutral_joints_seq = self.skeleton.neutral_joints[None].repeat((nF, 1, 1)).to(device)
            # FK to get the global rotations
            _, global_joint_rotmat = batch_rigid_transform(
                joint_rotmat,
                neutral_joints_seq,
                self.skeleton.joint_parents.to(device),
                self.skeleton.root_idx,
            )
            fk_transform[..., :3, :3] = global_joint_rotmat

        vertices = self.lbs(fk_transform)
        return vertices
