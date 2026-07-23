"""Focused regression tests for Pose Studio artifact and provenance safety."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from ardy.pose_video.composer import compose_behavior
from ardy.pose_video.provenance import (
    require_motion_provenance,
    require_render_provenance,
    validate_motion_provenance,
)
from ardy.pose_video.spec import BehaviorSpec, CameraSpec, behavior_spec_fingerprint


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HELPERS_PATH = (
    _REPOSITORY_ROOT
    / "scripts"
    / "interactive_demo"
    / "gui"
    / "pose_studio_helpers.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("pose_studio_safety_helpers", _HELPERS_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_HELPERS)
validate_artifact_paths = _HELPERS.validate_artifact_paths


def _spec(**updates: object) -> BehaviorSpec:
    payload: dict[str, object] = {
        "behavior_id": "audit_motion",
        "behavior_type": "one_shot",
        "fps": 10,
        "duration_seconds": 2.0,
        "boundary_seconds": 0.2,
    }
    payload.update(updates)
    return BehaviorSpec.model_validate(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PoseStudioArtifactSafetyTests(unittest.TestCase):
    def test_artifacts_must_be_distinct_and_cannot_replace_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            spec_path = directory / "behavior.json"
            motion_path = directory / "motion.npz"
            video_path = directory / "motion.mp4"
            valid = validate_artifact_paths(
                {
                    "npz": motion_path,
                    "mp4": video_path,
                    "manifest": directory / "render.json",
                    "validation": directory / "validation.json",
                },
                protected_paths=(spec_path,),
            )
            self.assertEqual(valid["npz"], motion_path.resolve())

            with self.assertRaisesRegex(ValueError, "must be distinct"):
                validate_artifact_paths(
                    {"manifest": spec_path, "validation": spec_path},
                    protected_paths=(directory / "other.json",),
                )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                validate_artifact_paths(
                    {"manifest": spec_path}, protected_paths=(spec_path,)
                )

    def test_camera_fov_matches_renderer_limit(self) -> None:
        self.assertEqual(CameraSpec(vertical_fov_degrees=120.0).vertical_fov_degrees, 120.0)
        with self.assertRaises(ValidationError):
            CameraSpec(vertical_fov_degrees=120.01)


class PoseStudioProvenanceTests(unittest.TestCase):
    def test_canonical_fingerprint_is_stable_and_sensitive(self) -> None:
        first = _spec()
        equivalent = BehaviorSpec.model_validate(first.model_dump(mode="json"))
        changed = first.model_copy(update={"description": "changed"})
        self.assertEqual(behavior_spec_fingerprint(first), behavior_spec_fingerprint(equivalent))
        self.assertNotEqual(behavior_spec_fingerprint(first), behavior_spec_fingerprint(changed))

    def test_composed_motion_refuses_a_stale_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "motion.npz"
            spec = _spec()
            compose_behavior(spec).save_npz(path)
            report = validate_motion_provenance(path, spec)
            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))

            stale = spec.model_copy(update={"description": "new edit"})
            with self.assertRaisesRegex(ValueError, "compose it again"):
                require_motion_provenance(path, stale)

    def test_render_manifest_binds_video_to_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            motion = directory / "motion.npz"
            video = directory / "motion.mp4"
            manifest = directory / "render.json"
            motion.write_bytes(b"motion-v1")
            video.write_bytes(b"video-v1")
            manifest.write_text(
                json.dumps(
                    {
                        "input_path": str(motion.resolve()),
                        "output_path": str(video.resolve()),
                        "input_sha256": _sha256(motion),
                        "mp4_sha256": _sha256(video),
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(require_render_provenance(motion, video, manifest)["passed"])
            video.write_bytes(b"video-v2")
            with self.assertRaisesRegex(ValueError, "render it again"):
                require_render_provenance(motion, video, manifest)


if __name__ == "__main__":
    unittest.main()
