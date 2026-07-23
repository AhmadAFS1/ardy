import importlib.util
import tempfile
import unittest
from pathlib import Path

# Load the intentionally dependency-free helper directly. Importing the parent
# GUI package would also import viser, which is an optional ``demo`` extra and
# should not be required by these pure editing tests.
_HELPERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "interactive_demo"
    / "gui"
    / "pose_studio_helpers.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("pose_studio_helpers", _HELPERS_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_HELPERS)

NEW_KEYFRAME_OPTION = _HELPERS.NEW_KEYFRAME_OPTION
delete_keyframe = _HELPERS.delete_keyframe
ensure_workspace_path = _HELPERS.ensure_workspace_path
keyframe_options = _HELPERS.keyframe_options
option_keyframe_index = _HELPERS.option_keyframe_index
remove_joint_target = _HELPERS.remove_joint_target
set_joint_lock = _HELPERS.set_joint_lock
suggested_output_paths = _HELPERS.suggested_output_paths
upsert_joint_keyframe = _HELPERS.upsert_joint_keyframe


def _spec_payload() -> dict:
    return {
        "behavior_id": "test_behavior",
        "duration_seconds": 10.0,
        "boundary_seconds": 2.0,
        "locks": ["Hips"],
        "keyframes": [],
    }


class PoseStudioHelpersTests(unittest.TestCase):
    def test_keyframe_options_round_trip_indices(self) -> None:
        payload = _spec_payload()
        payload["keyframes"] = [
            {
                "time_seconds": 3.25,
                "label": "nod peak",
                "joints": {"Head": {"rotation_degrees": [-12.0, 0.0, 0.0]}},
            }
        ]
        options = keyframe_options(payload)
        self.assertEqual(options[0], NEW_KEYFRAME_OPTION)
        self.assertIn("3.250s", options[1])
        self.assertIsNone(option_keyframe_index(options[0]))
        self.assertEqual(option_keyframe_index(options[1]), 0)

    def test_upsert_merges_joints_at_same_time_and_sorts(self) -> None:
        payload, selected = upsert_joint_keyframe(
            _spec_payload(),
            selected_index=None,
            time_seconds=5.0,
            label="head follow",
            joint_name="Head",
            rotation_degrees=(0.0, 10.0, 0.0),
        )
        self.assertEqual(selected, 0)
        payload, selected = upsert_joint_keyframe(
            payload,
            selected_index=None,
            time_seconds=3.0,
            label="torso lead",
            joint_name="Spine3",
            rotation_degrees=(0.0, 2.0, 0.0),
        )
        self.assertEqual(selected, 0)
        payload, selected = upsert_joint_keyframe(
            payload,
            selected_index=None,
            time_seconds=5.0,
            label="ignored when merging",
            joint_name="Neck",
            rotation_degrees=(0.0, 4.0, 0.0),
        )
        self.assertEqual(selected, 1)
        self.assertEqual([item["time_seconds"] for item in payload["keyframes"]], [3.0, 5.0])
        self.assertEqual(set(payload["keyframes"][1]["joints"]), {"Head", "Neck"})

    def test_upsert_rejects_targets_inside_boundary_blocks(self) -> None:
        for time_seconds in (1.999, 2.0, 8.0, 8.001):
            with self.subTest(time_seconds=time_seconds), self.assertRaisesRegex(ValueError, "editable interior"):
                upsert_joint_keyframe(
                    _spec_payload(),
                    selected_index=None,
                    time_seconds=time_seconds,
                    label="bad",
                    joint_name="Head",
                    rotation_degrees=(0.0, 0.0, 0.0),
                )

    def test_joint_locks_cannot_contradict_keyframes(self) -> None:
        with self.assertRaisesRegex(ValueError, "locked"):
            upsert_joint_keyframe(
                _spec_payload(),
                selected_index=None,
                time_seconds=4.0,
                label="bad",
                joint_name="Hips",
                rotation_degrees=(0.0, 1.0, 0.0),
            )

        payload, _ = upsert_joint_keyframe(
            _spec_payload(),
            selected_index=None,
            time_seconds=4.0,
            label="look",
            joint_name="Head",
            rotation_degrees=(0.0, 8.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "authored targets"):
            set_joint_lock(payload, "Head", True)

        payload = set_joint_lock(payload, "Neck", True)
        self.assertIn("Neck", payload["locks"])
        payload = set_joint_lock(payload, "Neck", False)
        self.assertNotIn("Neck", payload["locks"])

    def test_remove_joint_then_delete_keyframe(self) -> None:
        payload, selected = upsert_joint_keyframe(
            _spec_payload(),
            selected_index=None,
            time_seconds=4.0,
            label="look",
            joint_name="Head",
            rotation_degrees=(0.0, 8.0, 0.0),
        )
        payload, selected = upsert_joint_keyframe(
            payload,
            selected_index=selected,
            time_seconds=4.0,
            label="look",
            joint_name="Neck",
            rotation_degrees=(0.0, 3.0, 0.0),
        )
        payload, selected = remove_joint_target(payload, selected, "Neck")
        self.assertEqual(selected, 0)
        self.assertEqual(set(payload["keyframes"][0]["joints"]), {"Head"})
        self.assertEqual(delete_keyframe(payload, selected)["keyframes"], [])

    def test_workspace_paths_are_scoped_and_outputs_are_predictable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "ardy"
            root.mkdir()
            inside = ensure_workspace_path("pose_specs/test.json", root, suffix=".json")
            self.assertEqual(inside, root / "pose_specs" / "test.json")
            with self.assertRaisesRegex(ValueError, "inside"):
                ensure_workspace_path(root.parent / "escape.json", root, suffix=".json")
            with self.assertRaisesRegex(ValueError, "end in"):
                ensure_workspace_path("pose_specs/test.txt", root, suffix=".json")

            paths = suggested_output_paths(root, "nod_agree")
            self.assertEqual(
                paths["npz"], root / ".cache" / "pose_video" / "nod_agree" / "nod_agree.npz"
            )
            self.assertEqual(paths["mp4"].suffix, ".mp4")


if __name__ == "__main__":
    unittest.main()
