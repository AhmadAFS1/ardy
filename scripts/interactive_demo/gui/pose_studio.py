# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser-based precision authoring for deterministic pose-video assets."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import viser

from ..common import REPO_ROOT
from .pose_studio_helpers import (
    CORE_JOINT_NAMES,
    NEW_KEYFRAME_OPTION,
    delete_keyframe,
    discover_pose_specs,
    display_workspace_path,
    ensure_workspace_path,
    keyframe_options,
    locks_markdown,
    option_keyframe_index,
    remove_joint_target,
    set_joint_lock,
    suggested_output_paths,
    upsert_joint_keyframe,
    validate_artifact_paths,
)


_SPEC_IO_LOCK = threading.Lock()
# Headless skinning/rendering can consume most of a workstation GPU. Serialize
# build jobs across browser clients so two independent tabs cannot corrupt a
# shared output or accidentally exhaust the renderer at the same time.
_BUILD_IO_LOCK = threading.Lock()


@dataclass
class PoseStudioState:
    """All mutable Pose Studio state owned by one browser client."""

    client_id: int
    spec: Any
    spec_path: Path
    ui: SimpleNamespace = field(default_factory=SimpleNamespace)
    selected_keyframe_index: int | None = None
    syncing: bool = False
    dirty: bool = False
    operation_lock: threading.Lock = field(default_factory=threading.Lock)
    preview_lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    preview_motion: Any | None = None
    preview_character: Any | None = None
    preview_playing: bool = False
    preview_thread: threading.Thread | None = None
    worker_thread: threading.Thread | None = None
    keyboard_handler: Callable[[Any], bool] | None = None
    job_controls: tuple[Any, ...] = ()


@dataclass(frozen=True)
class PoseStudioBuildSnapshot:
    """Immutable inputs for one single-asset compose/render transaction."""

    spec: Any
    resolved_spec: Any
    spec_path: Path
    npz_path: Path
    mp4_path: Path
    manifest_path: Path
    validation_path: Path
    draft_resolution: bool
    overwrite: bool

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "npz": self.npz_path,
            "mp4": self.mp4_path,
            "manifest": self.manifest_path,
            "validation": self.validation_path,
        }


@dataclass(frozen=True)
class PoseStudioLibrarySnapshot:
    """Immutable inputs for a cross-asset certified library build."""

    spec_paths: tuple[Path, ...]
    output_dir: Path
    delivery_output_dir: Path | None
    overwrite: bool
    create_switch_test: bool


class _JobCancelled(RuntimeError):
    """Internal signal for a disconnected job that has not started writing."""


def _load_spec(path: Path) -> Any:
    # Import the architecture lazily so opening the ordinary demo never needs
    # the optional offscreen-rendering dependencies.
    from ardy.pose_video.spec import BehaviorSpec

    with path.open("r", encoding="utf-8") as handle:
        return BehaviorSpec.model_validate(json.load(handle))


def _fallback_spec() -> Any:
    from ardy.pose_video.spec import BehaviorSpec

    return BehaviorSpec.model_validate(
        {
            "schema_version": 1,
            "behavior_id": "new_behavior",
            "description": "A precisely authored video-call behavior.",
            "behavior_type": "one_shot",
            "speaking_mode": "either",
            "fps": 30,
            "duration_seconds": 10.0,
            "boundary_seconds": 2.0,
        }
    )


def _write_spec_atomic(spec: Any, destination: Path) -> None:
    """Serialize through the architecture's writer, then atomically replace."""

    from ardy.pose_video.spec import save_behavior_spec

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".json", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_behavior_spec(spec, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _notify(client: Any | None, title: str, body: str, color: str = "blue") -> None:
    if client is not None:
        client.add_notification(
            title=title,
            body=body,
            color=color,
            auto_close_seconds=4.0,
        )


def _error_text(error: BaseException) -> str:
    message = str(error).strip() or type(error).__name__
    return message if len(message) <= 300 else message[:297] + "..."


def _status(state: PoseStudioState, title: str, detail: str, *, icon: str = "ℹ️") -> None:
    state.ui.status.content = f"{icon} **{title}**  \n{detail}"


def _collect_form(state: PoseStudioState) -> Any:
    """Validate all visible controls as one BehaviorSpec transaction."""

    from ardy.pose_video.spec import BehaviorSpec

    ui = state.ui
    payload = state.spec.model_dump(mode="json")
    payload.update(
        {
            "behavior_id": ui.behavior_id.value.strip(),
            "description": ui.description.value.strip(),
            "behavior_type": ui.behavior_type.value,
            "speaking_mode": ui.speaking_mode.value,
            "fps": int(ui.fps.value),
            "duration_seconds": float(ui.duration.value),
            "boundary_seconds": float(ui.boundary.value),
            "source_reference": ui.source_reference.value.strip() or None,
        }
    )
    payload["base_pose"] = {
        "mode": ui.base_mode.value,
        "motion_path": ui.base_motion_path.value.strip() or None,
        "frame": int(ui.base_frame.value),
    }
    payload["ambient_motion"] = {
        "enabled": bool(ui.ambient_enabled.value),
        "breathing_cycles": float(ui.breathing_cycles.value),
        "breathing_degrees": float(ui.breathing_degrees.value),
        "sway_cycles": float(ui.sway_cycles.value),
        "sway_degrees": float(ui.sway_degrees.value),
        "head_sway_degrees": float(ui.head_sway_degrees.value),
    }
    payload["face_intent"] = {
        "gaze": ui.face_gaze.value.strip(),
        "expression": ui.face_expression.value.strip(),
        "blink": ui.face_blink.value.strip(),
        "intensity": float(ui.face_intensity.value),
    }
    payload["camera"] = {
        "resolution": (int(ui.output_width.value), int(ui.output_height.value)),
        "render_resolution": (int(ui.render_width.value), int(ui.render_height.value)),
        "position": tuple(float(value) for value in ui.camera_position.value),
        "look_at": tuple(float(value) for value in ui.camera_look_at.value),
        "up": tuple(float(value) for value in ui.camera_up.value),
        "vertical_fov_degrees": float(ui.camera_fov.value),
        "background_rgb": tuple(int(round(value)) for value in ui.background_rgb.value),
        "body_rgb": tuple(int(round(value)) for value in ui.body_rgb.value),
    }
    state.spec = BehaviorSpec.model_validate(payload)
    state.dirty = True
    return state.spec


def _set_suggested_paths(state: PoseStudioState) -> None:
    behavior_id = state.ui.behavior_id.value.strip() or state.spec.behavior_id
    paths = suggested_output_paths(REPO_ROOT, behavior_id)
    state.ui.npz_path.value = display_workspace_path(paths["npz"], REPO_ROOT)
    state.ui.mp4_path.value = display_workspace_path(paths["mp4"], REPO_ROOT)
    state.ui.manifest_path.value = display_workspace_path(paths["manifest"], REPO_ROOT)
    state.ui.validation_path.value = display_workspace_path(paths["validation"], REPO_ROOT)


def _sync_keyframe_editor(state: PoseStudioState, *, choose_existing_joint: bool = True) -> None:
    ui = state.ui
    payload = state.spec.model_dump(mode="json")
    options = keyframe_options(payload)
    ui.keyframe.options = options
    selected = state.selected_keyframe_index
    if selected is None or selected >= len(payload["keyframes"]):
        state.selected_keyframe_index = None
        ui.keyframe.value = NEW_KEYFRAME_OPTION
        ui.keyframe_time.value = min(
            state.spec.duration_seconds - state.spec.boundary_seconds,
            state.spec.boundary_seconds + 1.0,
        )
        ui.keyframe_label.value = ""
        ui.rotation_x.value = 0.0
        ui.rotation_y.value = 0.0
        ui.rotation_z.value = 0.0
    else:
        keyframe = payload["keyframes"][selected]
        ui.keyframe.value = options[selected + 1]
        ui.keyframe_time.value = float(keyframe["time_seconds"])
        ui.keyframe_label.value = str(keyframe.get("label", ""))
        joints = keyframe["joints"]
        if ui.joint.value not in joints and choose_existing_joint:
            ui.joint.value = next(iter(joints))
        target = joints.get(ui.joint.value)
        degrees = target["rotation_degrees"] if target is not None else (0.0, 0.0, 0.0)
        ui.rotation_x.value = float(degrees[0])
        ui.rotation_y.value = float(degrees[1])
        ui.rotation_z.value = float(degrees[2])
    locked = ui.joint.value in state.spec.locks
    ui.joint_locked.value = locked
    ui.upsert_keyframe.disabled = locked
    ui.locks_summary.content = locks_markdown(payload)
    ui.keyframe_summary.content = (
        f"**Timeline:** {len(payload['keyframes'])} authored keyframes · "
        f"editable interior `{state.spec.boundary_seconds:.3f}s`–"
        f"`{state.spec.duration_seconds - state.spec.boundary_seconds:.3f}s`"
    )


def _sync_all(state: PoseStudioState, *, reset_output_paths: bool) -> None:
    """Push a newly loaded spec into every control without firing edits."""

    ui = state.ui
    spec = state.spec
    state.syncing = True
    try:
        ui.behavior_id.value = spec.behavior_id
        ui.description.value = spec.description
        ui.behavior_type.value = spec.behavior_type
        ui.speaking_mode.value = spec.speaking_mode
        ui.fps.value = spec.fps
        ui.duration.value = spec.duration_seconds
        ui.boundary.value = spec.boundary_seconds
        ui.source_reference.value = spec.source_reference or ""
        ui.base_mode.value = spec.base_pose.mode
        ui.base_motion_path.value = spec.base_pose.motion_path or ""
        ui.base_frame.value = spec.base_pose.frame

        ambient = spec.ambient_motion
        ui.ambient_enabled.value = ambient.enabled
        ui.breathing_cycles.value = ambient.breathing_cycles
        ui.breathing_degrees.value = ambient.breathing_degrees
        ui.sway_cycles.value = ambient.sway_cycles
        ui.sway_degrees.value = ambient.sway_degrees
        ui.head_sway_degrees.value = ambient.head_sway_degrees

        face = spec.face_intent
        ui.face_gaze.value = face.gaze
        ui.face_expression.value = face.expression
        ui.face_blink.value = face.blink
        ui.face_intensity.value = face.intensity

        camera = spec.camera
        ui.output_width.value, ui.output_height.value = camera.resolution
        ui.render_width.value, ui.render_height.value = camera.render_resolution
        ui.camera_position.value = camera.position
        ui.camera_look_at.value = camera.look_at
        ui.camera_up.value = camera.up
        ui.camera_fov.value = camera.vertical_fov_degrees
        ui.background_rgb.value = camera.background_rgb
        ui.body_rgb.value = camera.body_rgb
        ui.preview_frame.max = max(0, spec.num_frames - 1)
        _sync_keyframe_editor(state)
        if reset_output_paths:
            _set_suggested_paths(state)
    finally:
        state.syncing = False


def _resolve_build_paths(
    state: PoseStudioState,
    *,
    resolved_spec: Any,
    spec_path: Path,
) -> dict[str, Path]:
    paths = {
        "npz": ensure_workspace_path(state.ui.npz_path.value, REPO_ROOT, suffix=".npz"),
        "mp4": ensure_workspace_path(state.ui.mp4_path.value, REPO_ROOT, suffix=".mp4"),
        "manifest": ensure_workspace_path(state.ui.manifest_path.value, REPO_ROOT, suffix=".json"),
        "validation": ensure_workspace_path(state.ui.validation_path.value, REPO_ROOT, suffix=".json"),
    }
    protected: list[Path] = [
        spec_path.resolve(),
        ensure_workspace_path(state.ui.spec_path.value, REPO_ROOT, suffix=".json"),
    ]
    for value in (
        resolved_spec.base_pose.motion_path,
        resolved_spec.source_reference,
    ):
        if value and "://" not in value:
            protected.append(Path(value).resolve())
    return validate_artifact_paths(paths, protected_paths=protected)


def _invalidate_preview(state: PoseStudioState, reason: str) -> None:
    """Remove a preview that no longer represents the current editor state."""

    with state.preview_lock:
        state.preview_playing = False
        state.preview_motion = None
        if state.preview_character is not None:
            try:
                state.preview_character.clear()
            except Exception:
                pass
            state.preview_character = None
    ui = state.ui
    if hasattr(ui, "preview_play"):
        ui.preview_play.label = "Play preview"
    if hasattr(ui, "preview_frame"):
        ui.preview_frame.value = 0
        ui.preview_frame.max = max(0, state.spec.num_frames - 1)
    if hasattr(ui, "preview_note"):
        ui.preview_note.content = f"⚠️ **Preview stale:** {reason} Compose again to refresh it."


def _set_preview_pose(state: PoseStudioState, frame: int) -> None:
    with state.preview_lock:
        motion = state.preview_motion
        character = state.preview_character
        if motion is None or character is None:
            return
        frame = min(max(int(frame), 0), motion.num_frames - 1)
        character.set_pose(
            torch.from_numpy(motion.posed_joints[frame]),
            torch.from_numpy(motion.global_rot_mats[frame]),
        )


def _ensure_preview(state: PoseStudioState, client: Any) -> None:
    with state.preview_lock:
        if state.preview_motion is None:
            return
        if state.preview_character is None:
            from ardy.skeleton.registry import build_skeleton
            from ardy.viz.viser_utils import Character

            state.preview_character = Character(
                f"pose_studio_preview_{state.client_id}",
                client,
                build_skeleton(27),
                create_skeleton_mesh=True,
                create_skinned_mesh=True,
                visible_skeleton=True,
                visible_skinned_mesh=bool(state.ui.show_preview.value),
                skinned_mesh_opacity=1.0,
                show_foot_contacts=False,
                mesh_mode="core_skin",
            )
        state.preview_character.set_skeleton_visibility(bool(state.ui.show_preview.value))
        state.preview_character.set_skinned_mesh_visibility(bool(state.ui.show_preview.value))
        state.ui.preview_frame.max = max(0, state.preview_motion.num_frames - 1)
        _set_preview_pose(state, int(state.ui.preview_frame.value))
        if hasattr(state.ui, "preview_note"):
            state.ui.preview_note.content = "✅ Preview matches the last composed specification."


def _start_preview_thread(owner: Any, state: PoseStudioState) -> None:
    if state.preview_thread is not None and state.preview_thread.is_alive():
        return

    def loop() -> None:
        while not state.stop_event.is_set():
            if not owner.client_active(state.client_id):
                break
            started = time.monotonic()
            with state.preview_lock:
                motion = state.preview_motion
                if not state.preview_playing or motion is None:
                    wait_seconds = 0.05
                else:
                    next_frame = (int(state.ui.preview_frame.value) + 1) % motion.num_frames
                    state.ui.preview_frame.value = next_frame
                    _set_preview_pose(state, next_frame)
                    wait_seconds = 1.0 / max(float(motion.fps), 1.0) - (
                        time.monotonic() - started
                    )
            remaining = wait_seconds
            if remaining > 0.0:
                state.stop_event.wait(remaining)

    state.preview_thread = threading.Thread(
        target=loop,
        name=f"pose-studio-preview-{state.client_id}",
        daemon=True,
    )
    state.preview_thread.start()


def _start_job(
    owner: Any,
    state: PoseStudioState,
    name: str,
    job: Callable[[], str],
    *,
    client: Any | None,
) -> None:
    """Run one expensive per-client operation without blocking viser."""

    if not state.operation_lock.acquire(blocking=False):
        _notify(client, "Pose Studio is busy", "Wait for the current build operation to finish.", "orange")
        return
    if state.stop_event.is_set():
        state.operation_lock.release()
        return
    controls = state.job_controls or (
        state.ui.compose,
        state.ui.render,
        state.ui.validate,
        state.ui.build_all,
    )
    disabled_states = tuple((control, bool(control.disabled)) for control in controls)
    for control, _ in disabled_states:
        control.disabled = True
    _status(state, name, "Starting…", icon="⏳")

    def run() -> None:
        try:
            if _BUILD_IO_LOCK.locked() and owner.client_active(state.client_id) and not state.stop_event.is_set():
                _status(state, name, "Waiting for the shared render/build slot…", icon="⏳")
            while not _BUILD_IO_LOCK.acquire(timeout=0.2):
                if state.stop_event.is_set():
                    raise _JobCancelled("browser client disconnected before the build slot was available")
            try:
                if state.stop_event.is_set():
                    raise _JobCancelled("browser client disconnected before the job started")
                detail = job()
            finally:
                _BUILD_IO_LOCK.release()
            if owner.client_active(state.client_id) and not state.stop_event.is_set():
                _status(state, f"{name} complete", detail, icon="✅")
                _notify(client, f"{name} complete", detail, "green")
        except _JobCancelled:
            # Disconnect cleanup already removed the UI. No traceback or stale
            # notification is useful, and no new artifact write was started.
            pass
        except Exception as error:
            traceback.print_exc()
            if owner.client_active(state.client_id) and not state.stop_event.is_set():
                detail = _error_text(error)
                _status(state, f"{name} failed", detail, icon="❌")
                _notify(client, f"{name} failed", detail, "red")
        finally:
            if owner.client_active(state.client_id) and not state.stop_event.is_set():
                for control, was_disabled in disabled_states:
                    control.disabled = was_disabled
            state.worker_thread = None
            state.operation_lock.release()

    state.worker_thread = threading.Thread(
        target=run,
        name=f"pose-studio-{name.lower().replace(' ', '-')}-{state.client_id}",
        daemon=True,
    )
    state.worker_thread.start()


def cleanup_pose_studio(owner: Any, client_id: int) -> None:
    """Stop preview work and release per-client scene/state on disconnect."""

    states = getattr(owner, "_pose_studio_states", None)
    if not states:
        return
    state = states.pop(client_id, None)
    if state is None:
        return
    state.preview_playing = False
    state.stop_event.set()
    state.keyboard_handler = None
    if state.preview_thread is not None and state.preview_thread.is_alive():
        state.preview_thread.join(timeout=0.5)
    with state.preview_lock:
        if state.preview_character is not None:
            try:
                state.preview_character.clear()
            except Exception:
                pass
            state.preview_character = None


def build_pose_studio_tab(owner: Any, client: Any, client_id: int, tab_group: Any) -> None:
    """Add a complete, client-isolated Pose Studio tab to the demo."""

    choices = discover_pose_specs(REPO_ROOT)
    initial_path = next(
        (path for path in choices if path.stem == "neutral_resting"),
        choices[0] if choices else Path(REPO_ROOT) / "pose_specs" / "new_behavior.json",
    )
    try:
        initial_spec = _load_spec(initial_path) if initial_path.is_file() else _fallback_spec()
        initial_error = None
    except Exception as error:
        initial_spec = _fallback_spec()
        initial_error = _error_text(error)

    state = PoseStudioState(client_id=client_id, spec=initial_spec, spec_path=initial_path)
    states = getattr(owner, "_pose_studio_states", None)
    if states is None:
        states = owner._pose_studio_states = {}
    states[client_id] = state
    ui = state.ui

    spec_options = [display_workspace_path(path, REPO_ROOT) for path in choices]
    if not spec_options:
        spec_options = [display_workspace_path(initial_path, REPO_ROOT)]

    with tab_group.add_tab("Pose Studio", viser.Icon.WALK):
        client.gui.add_markdown(
            "### Precision motion authoring\n"
            "Author exact local-joint rotations, preserve canonical transition boundaries, "
            "then compose, preview, render, and verify the asset without recording by hand."
        )

        with client.gui.add_folder("Pose specification", expand_by_default=True):
            ui.spec_choice = client.gui.add_dropdown(
                "Available spec",
                options=spec_options,
                initial_value=display_workspace_path(initial_path, REPO_ROOT),
            )
            ui.spec_path = client.gui.add_text(
                "Spec path",
                initial_value=display_workspace_path(initial_path, REPO_ROOT),
                hint="Workspace-relative JSON path. Writes outside this ARDY checkout are rejected.",
            )
            ui.refresh_specs = client.gui.add_button("Refresh list")
            ui.load_spec = client.gui.add_button("Load selected spec", color="blue")
            ui.save_spec = client.gui.add_button("Save spec", color="green")

        with client.gui.add_folder("Behavior", expand_by_default=True):
            ui.behavior_id = client.gui.add_text("Behavior ID", initial_value=initial_spec.behavior_id)
            ui.description = client.gui.add_text(
                "Description", initial_value=initial_spec.description, multiline=True
            )
            ui.behavior_type = client.gui.add_dropdown(
                "Behavior type", options=["loop", "one_shot"], initial_value=initial_spec.behavior_type
            )
            ui.speaking_mode = client.gui.add_dropdown(
                "Speaking compatibility",
                options=["either", "speaking", "listening"],
                initial_value=initial_spec.speaking_mode,
            )
            ui.fps = client.gui.add_number("FPS", initial_value=initial_spec.fps, min=1, max=120, step=1)
            ui.duration = client.gui.add_number(
                "Duration (seconds)", initial_value=initial_spec.duration_seconds, min=0.1, max=120.0, step=0.01
            )
            ui.boundary = client.gui.add_number(
                "Canonical boundary (seconds)",
                initial_value=initial_spec.boundary_seconds,
                min=0.01,
                max=60.0,
                step=0.01,
                hint="This exact shared motion block is copied to both ends of every clip.",
            )
            ui.source_reference = client.gui.add_text(
                "Source reference", initial_value=initial_spec.source_reference or ""
            )

        with client.gui.add_folder("Base, ambient, and face intent", expand_by_default=False):
            ui.base_mode = client.gui.add_dropdown(
                "Base pose", options=["authored_neutral", "motion_frame"], initial_value=initial_spec.base_pose.mode
            )
            ui.base_motion_path = client.gui.add_text(
                "Base motion NPZ", initial_value=initial_spec.base_pose.motion_path or ""
            )
            ui.base_frame = client.gui.add_number(
                "Base frame", initial_value=initial_spec.base_pose.frame, min=0, step=1
            )
            ui.ambient_enabled = client.gui.add_checkbox(
                "Deterministic ambient motion", initial_value=initial_spec.ambient_motion.enabled
            )
            ui.breathing_cycles = client.gui.add_number(
                "Breathing cycles", initial_value=initial_spec.ambient_motion.breathing_cycles, min=0.01, step=0.05
            )
            ui.breathing_degrees = client.gui.add_number(
                "Breathing amplitude (degrees)",
                initial_value=initial_spec.ambient_motion.breathing_degrees,
                min=0.0,
                max=5.0,
                step=0.05,
            )
            ui.sway_cycles = client.gui.add_number(
                "Sway cycles", initial_value=initial_spec.ambient_motion.sway_cycles, min=0.01, step=0.05
            )
            ui.sway_degrees = client.gui.add_number(
                "Sway amplitude (degrees)",
                initial_value=initial_spec.ambient_motion.sway_degrees,
                min=0.0,
                max=5.0,
                step=0.05,
            )
            ui.head_sway_degrees = client.gui.add_number(
                "Head sway (degrees)",
                initial_value=initial_spec.ambient_motion.head_sway_degrees,
                min=0.0,
                max=3.0,
                step=0.05,
            )
            ui.face_gaze = client.gui.add_text("Gaze intent", initial_value=initial_spec.face_intent.gaze)
            ui.face_expression = client.gui.add_text(
                "Expression intent", initial_value=initial_spec.face_intent.expression
            )
            ui.face_blink = client.gui.add_text("Blink intent", initial_value=initial_spec.face_intent.blink)
            ui.face_intensity = client.gui.add_number(
                "Face intensity", initial_value=initial_spec.face_intent.intensity, min=0.0, max=1.0, step=0.01
            )

        with client.gui.add_folder("Keyframe authoring", expand_by_default=True):
            ui.keyframe_summary = client.gui.add_markdown("")
            ui.keyframe = client.gui.add_dropdown(
                "Keyframe", options=[NEW_KEYFRAME_OPTION], initial_value=NEW_KEYFRAME_OPTION
            )
            ui.keyframe_time = client.gui.add_number(
                "Time (seconds)", initial_value=3.0, min=0.0, max=120.0, step=0.001
            )
            ui.keyframe_label = client.gui.add_text("Label", initial_value="")
            ui.joint = client.gui.add_dropdown(
                "Core joint", options=list(CORE_JOINT_NAMES), initial_value="Head"
            )
            ui.rotation_x = client.gui.add_number(
                "Local X (degrees)", initial_value=0.0, min=-180.0, max=180.0, step=0.05
            )
            ui.rotation_y = client.gui.add_number(
                "Local Y (degrees)", initial_value=0.0, min=-180.0, max=180.0, step=0.05
            )
            ui.rotation_z = client.gui.add_number(
                "Local Z (degrees)", initial_value=0.0, min=-180.0, max=180.0, step=0.05
            )
            ui.upsert_keyframe = client.gui.add_button("Add / update joint target", color="green")
            ui.remove_joint = client.gui.add_button("Remove this joint target", color="orange")
            ui.delete_keyframe = client.gui.add_button("Delete complete keyframe", color="red")

        with client.gui.add_folder("Joint locks", expand_by_default=True):
            ui.locks_summary = client.gui.add_markdown("")
            ui.joint_locked = client.gui.add_checkbox(
                "Selected joint is locked",
                initial_value=False,
                hint="Locked joints cannot receive authored keyframe targets.",
            )
            ui.apply_lock = client.gui.add_button("Apply lock state")

        with client.gui.add_folder("Fixed render camera", expand_by_default=False):
            ui.output_width = client.gui.add_number(
                "Master width", initial_value=initial_spec.camera.resolution[0], min=16, max=7680, step=2
            )
            ui.output_height = client.gui.add_number(
                "Master height", initial_value=initial_spec.camera.resolution[1], min=16, max=7680, step=2
            )
            ui.render_width = client.gui.add_number(
                "Draft width", initial_value=initial_spec.camera.render_resolution[0], min=16, max=7680, step=2
            )
            ui.render_height = client.gui.add_number(
                "Draft height", initial_value=initial_spec.camera.render_resolution[1], min=16, max=7680, step=2
            )
            ui.camera_position = client.gui.add_vector3(
                "Position", initial_value=initial_spec.camera.position, step=0.01
            )
            ui.camera_look_at = client.gui.add_vector3(
                "Look at", initial_value=initial_spec.camera.look_at, step=0.01
            )
            ui.camera_up = client.gui.add_vector3("Up", initial_value=initial_spec.camera.up, step=0.01)
            ui.camera_fov = client.gui.add_number(
                "Vertical FOV (degrees)",
                initial_value=initial_spec.camera.vertical_fov_degrees,
                min=1.01,
                max=120.0,
                step=0.1,
            )
            ui.background_rgb = client.gui.add_vector3(
                "Background RGB",
                initial_value=initial_spec.camera.background_rgb,
                min=(0, 0, 0),
                max=(255, 255, 255),
                step=1,
            )
            ui.body_rgb = client.gui.add_vector3(
                "Body RGB", initial_value=initial_spec.camera.body_rgb, min=(0, 0, 0), max=(255, 255, 255), step=1
            )

        with client.gui.add_folder("Compose, render, and verify", expand_by_default=True):
            ui.npz_path = client.gui.add_text("Motion NPZ", initial_value="")
            ui.mp4_path = client.gui.add_text("Rendered MP4", initial_value="")
            ui.manifest_path = client.gui.add_text("Render manifest", initial_value="")
            ui.validation_path = client.gui.add_text("Validation report", initial_value="")
            ui.draft_resolution = client.gui.add_checkbox(
                "Draft resolution",
                initial_value=False,
                hint="Use camera.render_resolution for quick iteration; turn off for the master MP4.",
            )
            ui.overwrite = client.gui.add_checkbox("Overwrite existing outputs", initial_value=True)
            ui.compose = client.gui.add_button("1 · Compose deterministic motion", color="blue")
            ui.render = client.gui.add_button("2 · Render MP4", color="blue")
            ui.validate = client.gui.add_button("3 · Validate this asset only", color="blue")
            ui.build_all = client.gui.add_button("Build this asset + validate", color="green")
            ui.status = client.gui.add_markdown("ℹ️ **Ready**  \nLoad a spec or start authoring.")

        with client.gui.add_folder("Cross-asset library certification", expand_by_default=True):
            client.gui.add_markdown(
                "This is the master workflow for invisible switching. It renders every active "
                "spec with one shared canonical RGB boundary, validates the clips together, "
                "and optionally creates a switch stress test. Single-asset validation above "
                "does **not** prove cross-asset equality."
            )
            ui.library_spec_source = client.gui.add_text(
                "Active spec source",
                initial_value="pose_specs",
                hint="Workspace-local JSON file or directory. The default uses the three active top-level specs.",
            )
            ui.library_output_dir = client.gui.add_text(
                "Certified library output",
                initial_value=".cache/pose_video/certified_library",
            )
            ui.library_delivery_proxies = client.gui.add_checkbox(
                "Create upload/browser proxies",
                initial_value=True,
                hint=(
                    "After master certification, derive H.264 High@4.0 CQP12 all-IDR copies "
                    "and certify their shared decoded boundaries."
                ),
            )
            ui.library_delivery_output_dir = client.gui.add_text(
                "Delivery proxy output",
                initial_value=".cache/pose_video/certified_delivery",
            )
            ui.library_switch_test = client.gui.add_checkbox(
                "Create switch stress test", initial_value=True
            )
            ui.build_library = client.gui.add_button(
                "Build + certify active library", color="green"
            )

        with client.gui.add_folder("3D motion preview", expand_by_default=True):
            client.gui.add_markdown(
                "Compose first, then scrub or play the deterministic motion directly in the ARDY viewport. "
                "The preview is isolated from generated demo motion."
            )
            ui.show_preview = client.gui.add_checkbox("Show preview avatar", initial_value=True)
            ui.preview_frame = client.gui.add_number(
                "Preview frame", initial_value=0, min=0, max=max(0, initial_spec.num_frames - 1), step=1
            )
            ui.preview_play = client.gui.add_button("Play preview")
            ui.preview_note = client.gui.add_markdown(
                "ℹ️ Compose the current specification to create its preview."
            )

        client.gui.add_markdown(
            "#### Pose Studio shortcuts\n"
            "| Shortcut | Action |\n"
            "|:--|:--|\n"
            "| `Alt+A` | Add/update the selected joint target |\n"
            "| `Alt+[` / `Alt+]` | Previous/next authored keyframe |\n"
            "| `Alt+Delete` | Delete the selected keyframe |\n"
            "| `Ctrl/Cmd+S` | Save the pose specification |\n"
            "| `Ctrl/Cmd+Enter` | Compose deterministic motion |\n"
            "| `Ctrl/Cmd+Shift+Enter` | Build, render, and validate |\n\n"
            "Shortcuts work when focus is outside text and number fields."
        )

    def refresh_specs(event_client: Any | None = None) -> None:
        paths = discover_pose_specs(REPO_ROOT)
        options = [display_workspace_path(path, REPO_ROOT) for path in paths]
        if not options:
            options = [state.ui.spec_path.value]
        current = state.ui.spec_choice.value
        state.ui.spec_choice.options = options
        state.ui.spec_choice.value = current if current in options else options[0]
        _notify(event_client, "Pose specs refreshed", f"Found {len(paths)} JSON specifications.")

    def editing_allowed(event_client: Any | None = None) -> bool:
        if not state.operation_lock.locked():
            return True
        _notify(
            event_client,
            "Pose Studio is busy",
            "Wait for the captured build operation to finish before editing or loading another spec.",
            "orange",
        )
        return False

    def load_action(event_client: Any | None = None) -> None:
        if not editing_allowed(event_client):
            return
        try:
            path = ensure_workspace_path(state.ui.spec_path.value, REPO_ROOT, suffix=".json")
            if not path.is_file():
                raise FileNotFoundError(f"pose specification does not exist: {path}")
            state.spec = _load_spec(path)
            state.spec_path = path
            state.selected_keyframe_index = None
            state.dirty = False
            _invalidate_preview(state, "A different specification was loaded.")
            _sync_all(state, reset_output_paths=True)
            _status(state, "Spec loaded", display_workspace_path(path, REPO_ROOT), icon="✅")
            _notify(event_client, "Pose spec loaded", path.name, "green")
        except Exception as error:
            _status(state, "Load failed", _error_text(error), icon="❌")
            _notify(event_client, "Could not load pose spec", _error_text(error), "red")

    def save_action(event_client: Any | None = None) -> bool:
        if not editing_allowed(event_client):
            return False
        try:
            previous_payload = state.spec.model_dump(mode="json")
            spec = _collect_form(state)
            if spec.model_dump(mode="json") != previous_payload:
                _invalidate_preview(state, "The specification changed.")
            destination = ensure_workspace_path(state.ui.spec_path.value, REPO_ROOT, suffix=".json")
            with _SPEC_IO_LOCK:
                _write_spec_atomic(spec, destination)
            state.spec_path = destination
            state.dirty = False
            refresh_specs(None)
            _status(state, "Spec saved", display_workspace_path(destination, REPO_ROOT), icon="✅")
            _notify(event_client, "Pose spec saved", destination.name, "green")
            return True
        except Exception as error:
            _status(state, "Save failed", _error_text(error), icon="❌")
            _notify(event_client, "Could not save pose spec", _error_text(error), "red")
            return False

    def commit_keyframe(event_client: Any | None = None) -> None:
        if not editing_allowed(event_client):
            return
        try:
            spec = _collect_form(state)
            payload, selected = upsert_joint_keyframe(
                spec.model_dump(mode="json"),
                selected_index=state.selected_keyframe_index,
                time_seconds=float(ui.keyframe_time.value),
                label=ui.keyframe_label.value,
                joint_name=ui.joint.value,
                rotation_degrees=(ui.rotation_x.value, ui.rotation_y.value, ui.rotation_z.value),
            )
            from ardy.pose_video.spec import BehaviorSpec

            state.spec = BehaviorSpec.model_validate(payload)
            state.selected_keyframe_index = selected
            state.dirty = True
            _invalidate_preview(state, "A joint target changed.")
            state.syncing = True
            try:
                _sync_keyframe_editor(state)
            finally:
                state.syncing = False
            _status(
                state,
                "Target authored",
                f"{ui.joint.value} at {ui.keyframe_time.value:.3f}s · save or compose when ready.",
                icon="✅",
            )
            _notify(
                event_client,
                "Joint target authored",
                f"{ui.joint.value} at {ui.keyframe_time.value:.3f}s",
                "green",
            )
        except Exception as error:
            _status(state, "Keyframe rejected", _error_text(error), icon="❌")
            _notify(event_client, "Keyframe rejected", _error_text(error), "red")

    def delete_action(event_client: Any | None = None) -> None:
        if not editing_allowed(event_client):
            return
        try:
            from ardy.pose_video.spec import BehaviorSpec

            payload = delete_keyframe(state.spec.model_dump(mode="json"), state.selected_keyframe_index)
            state.spec = BehaviorSpec.model_validate(payload)
            state.selected_keyframe_index = None
            state.dirty = True
            _invalidate_preview(state, "A keyframe was deleted.")
            state.syncing = True
            try:
                _sync_keyframe_editor(state)
            finally:
                state.syncing = False
            _status(state, "Keyframe deleted", "The complete sparse keyframe was removed.", icon="✅")
            _notify(event_client, "Keyframe deleted", "The selected keyframe was removed.", "green")
        except Exception as error:
            _notify(event_client, "Could not delete keyframe", _error_text(error), "red")

    def remove_joint_action(event_client: Any | None = None) -> None:
        if not editing_allowed(event_client):
            return
        try:
            from ardy.pose_video.spec import BehaviorSpec

            payload, selected = remove_joint_target(
                state.spec.model_dump(mode="json"), state.selected_keyframe_index, ui.joint.value
            )
            state.spec = BehaviorSpec.model_validate(payload)
            state.selected_keyframe_index = selected
            state.dirty = True
            _invalidate_preview(state, "A joint target was removed.")
            state.syncing = True
            try:
                _sync_keyframe_editor(state)
            finally:
                state.syncing = False
            _status(state, "Joint target removed", ui.joint.value, icon="✅")
        except Exception as error:
            _notify(event_client, "Could not remove joint target", _error_text(error), "red")

    def lock_action(event_client: Any | None = None) -> None:
        if not editing_allowed(event_client):
            return
        try:
            from ardy.pose_video.spec import BehaviorSpec

            payload = set_joint_lock(
                state.spec.model_dump(mode="json"), ui.joint.value, bool(ui.joint_locked.value)
            )
            state.spec = BehaviorSpec.model_validate(payload)
            state.dirty = True
            _invalidate_preview(state, "Joint locks changed.")
            state.syncing = True
            try:
                _sync_keyframe_editor(state)
            finally:
                state.syncing = False
            mode = "locked" if ui.joint_locked.value else "unlocked"
            _status(state, f"Joint {mode}", ui.joint.value, icon="✅")
        except Exception as error:
            state.syncing = True
            ui.joint_locked.value = ui.joint.value in state.spec.locks
            state.syncing = False
            _notify(event_client, "Lock change rejected", _error_text(error), "red")

    def capture_build(event_client: Any | None = None) -> PoseStudioBuildSnapshot | None:
        if not editing_allowed(event_client):
            return None
        try:
            spec = _collect_form(state).model_copy(deep=True)
            spec_path = state.spec_path.resolve()
            resolved_spec = spec.resolve_relative_paths(spec_path)
            paths = _resolve_build_paths(
                state,
                resolved_spec=resolved_spec,
                spec_path=spec_path,
            )
            return PoseStudioBuildSnapshot(
                spec=spec,
                resolved_spec=resolved_spec,
                spec_path=spec_path,
                npz_path=paths["npz"],
                mp4_path=paths["mp4"],
                manifest_path=paths["manifest"],
                validation_path=paths["validation"],
                draft_resolution=bool(ui.draft_resolution.value),
                overwrite=bool(ui.overwrite.value),
            )
        except Exception as error:
            _status(state, "Invalid build settings", _error_text(error), icon="❌")
            _notify(event_client, "Invalid build settings", _error_text(error), "red")
            return None

    def compose_job(snapshot: PoseStudioBuildSnapshot) -> str:
        from ardy.pose_video.composer import compose_behavior

        if state.stop_event.is_set():
            raise _JobCancelled("client disconnected before composition")
        if snapshot.npz_path.exists() and not snapshot.overwrite:
            raise FileExistsError(
                f"motion output already exists: {snapshot.npz_path} "
                "(enable 'Overwrite existing outputs' to replace it)"
            )
        motion = compose_behavior(snapshot.resolved_spec)
        if state.stop_event.is_set():
            raise _JobCancelled("client disconnected before the composed motion was written")
        motion.save_npz(snapshot.npz_path)
        with state.preview_lock:
            state.preview_motion = motion
        if owner.client_active(client_id) and not state.stop_event.is_set():
            _ensure_preview(state, client)
        return (
            f"{motion.num_frames} frames written to "
            f"`{display_workspace_path(snapshot.npz_path, REPO_ROOT)}`"
        )

    def render_job(snapshot: PoseStudioBuildSnapshot) -> str:
        if not snapshot.npz_path.is_file():
            raise FileNotFoundError("compose the motion NPZ before rendering")
        from ardy.pose_video.provenance import require_motion_provenance

        require_motion_provenance(snapshot.npz_path, snapshot.resolved_spec)
        if state.stop_event.is_set():
            raise _JobCancelled("client disconnected before rendering started")
        if not snapshot.overwrite:
            existing = [
                path
                for path in (snapshot.mp4_path, snapshot.manifest_path)
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "render artifacts already exist: " + ", ".join(str(path) for path in existing)
                )
        from ardy.pose_video import (
            CameraConfig,
            RenderStyle,
            VideoExportSpec,
            render_motion_npz,
        )

        spec = snapshot.resolved_spec
        resolution = spec.camera.render_resolution if snapshot.draft_resolution else spec.camera.resolution
        camera = CameraConfig(
            width=int(resolution[0]),
            height=int(resolution[1]),
            eye=spec.camera.position,
            target=spec.camera.look_at,
            up=spec.camera.up,
            vertical_fov_degrees=spec.camera.vertical_fov_degrees,
        )
        background = tuple(channel / 255.0 for channel in spec.camera.background_rgb) + (1.0,)
        body = tuple(channel / 255.0 for channel in spec.camera.body_rgb) + (1.0,)
        style = RenderStyle(background_rgba=background, body_rgba=body)
        last_update = [0.0]

        def progress(frame: int, total: int) -> None:
            now = time.monotonic()
            if (
                owner.client_active(client_id)
                and not state.stop_event.is_set()
                and (now - last_update[0] >= 0.5 or frame == total)
            ):
                last_update[0] = now
                _status(state, "Rendering MP4", f"Frame {frame}/{total}", icon="⏳")

        export_spec = VideoExportSpec(
            target_fps=float(spec.fps),
            camera=camera,
            style=style,
            verify_boundary_frames=spec.boundary_frames,
        )
        result = render_motion_npz(
            snapshot.npz_path,
            snapshot.mp4_path,
            spec=export_spec,
            overwrite=snapshot.overwrite,
            manifest_path=snapshot.manifest_path,
            progress=progress,
        )
        return (
            f"{result.output_frames} frames at {result.output_fps:g} FPS written to "
            f"`{display_workspace_path(snapshot.mp4_path, REPO_ROOT)}`"
        )

    def validate_job(snapshot: PoseStudioBuildSnapshot) -> str:
        from ardy.pose_video.provenance import (
            require_motion_provenance,
            require_render_provenance,
        )
        from ardy.pose_video.validation import (
            save_validation_report,
            validate_motion_npz,
            validate_video,
        )

        if not snapshot.npz_path.is_file():
            raise FileNotFoundError("motion NPZ is missing; compose it first")
        provenance_report = require_motion_provenance(
            snapshot.npz_path, snapshot.resolved_spec
        )
        if state.stop_event.is_set():
            raise _JobCancelled("client disconnected before validation started")
        motion_report = validate_motion_npz(snapshot.npz_path)
        video_report = None
        render_provenance = None
        if snapshot.mp4_path.is_file():
            render_provenance = require_render_provenance(
                snapshot.npz_path,
                snapshot.mp4_path,
                snapshot.manifest_path,
            )
            video_report = validate_video(
                snapshot.mp4_path, snapshot.resolved_spec.boundary_frames
            )
        report = {
            "schema_version": 1,
            "validation_scope": "single_asset_only",
            "behavior_id": snapshot.resolved_spec.behavior_id,
            "passed": bool(motion_report["passed"] and (video_report is None or video_report["passed"])),
            "spec_provenance": provenance_report,
            "render_provenance": render_provenance,
            "motion": motion_report,
            "video": video_report,
        }
        if state.stop_event.is_set():
            raise _JobCancelled("client disconnected before the validation report was written")
        if snapshot.validation_path.exists() and not snapshot.overwrite:
            raise FileExistsError(
                f"validation report already exists: {snapshot.validation_path}"
            )
        save_validation_report(report, snapshot.validation_path)
        if not report["passed"]:
            failed = [name for name, passed in motion_report["checks"].items() if not passed]
            if video_report is not None:
                failed.extend(name for name, passed in video_report["checks"].items() if not passed)
            raise RuntimeError("seam validation failed: " + ", ".join(failed))
        scope = "motion + decoded video" if video_report is not None else "motion"
        return (
            f"Single-asset {scope} checks passed (cross-asset certification still required) · "
            f"`{display_workspace_path(snapshot.validation_path, REPO_ROOT)}`"
        )

    def capture_library(event_client: Any | None = None) -> PoseStudioLibrarySnapshot | None:
        if not editing_allowed(event_client):
            return None
        try:
            from ardy.pose_video.library import discover_behavior_specs

            source = ensure_workspace_path(ui.library_spec_source.value, REPO_ROOT)
            spec_paths = tuple(discover_behavior_specs(source))
            for spec_path in spec_paths:
                ensure_workspace_path(spec_path, REPO_ROOT, suffix=".json")
            output_dir = ensure_workspace_path(ui.library_output_dir.value, REPO_ROOT)
            delivery_output_dir = (
                ensure_workspace_path(ui.library_delivery_output_dir.value, REPO_ROOT)
                if bool(ui.library_delivery_proxies.value)
                else None
            )
            root = Path(REPO_ROOT).resolve()
            if output_dir == root:
                raise ValueError("certified library output cannot be the repository root")
            if delivery_output_dir == root:
                raise ValueError("delivery proxy output cannot be the repository root")
            if delivery_output_dir == output_dir:
                raise ValueError("master and delivery proxy outputs must be different directories")
            for spec_path in spec_paths:
                try:
                    spec_path.relative_to(output_dir)
                except ValueError:
                    continue
                raise ValueError(
                    "certified library output must not contain its source specifications"
                )
            if delivery_output_dir is not None:
                for spec_path in spec_paths:
                    try:
                        spec_path.relative_to(delivery_output_dir)
                    except ValueError:
                        continue
                    raise ValueError(
                        "delivery proxy output must not contain its source specifications"
                    )
            return PoseStudioLibrarySnapshot(
                spec_paths=spec_paths,
                output_dir=output_dir,
                delivery_output_dir=delivery_output_dir,
                overwrite=bool(ui.overwrite.value),
                create_switch_test=bool(ui.library_switch_test.value),
            )
        except Exception as error:
            _status(state, "Invalid library settings", _error_text(error), icon="❌")
            _notify(event_client, "Invalid library settings", _error_text(error), "red")
            return None

    def library_job(snapshot: PoseStudioLibrarySnapshot) -> str:
        from ardy.pose_video.library import build_pose_library

        if state.stop_event.is_set():
            raise _JobCancelled("client disconnected before library build started")

        def progress(message: str) -> None:
            if owner.client_active(client_id) and not state.stop_event.is_set():
                _status(state, "Certifying active library", message, icon="⏳")

        result = build_pose_library(
            snapshot.spec_paths,
            snapshot.output_dir,
            render=True,
            overwrite=snapshot.overwrite,
            create_switch_test=snapshot.create_switch_test,
            progress=progress,
        )
        if not result.passed:
            raise RuntimeError("cross-asset library certification failed")
        detail = (
            f"Master certification passed for {len(result.behaviors)} behaviors · "
            f"`{display_workspace_path(result.manifest_path, REPO_ROOT)}`"
        )
        if snapshot.delivery_output_dir is not None:
            from ardy.pose_video.delivery import build_delivery_proxy_library

            video_paths = tuple(
                behavior.video_path
                for behavior in result.behaviors
                if behavior.video_path is not None
            )
            if len(video_paths) != len(result.behaviors):
                raise RuntimeError("master library did not emit every video required for proxies")
            if state.stop_event.is_set():
                raise _JobCancelled("client disconnected before proxy creation")
            if owner.client_active(client_id):
                _status(
                    state,
                    "Certifying upload/browser proxies",
                    f"Encoding {len(video_paths)} High Profile copies…",
                    icon="⏳",
                )
            delivery = build_delivery_proxy_library(
                video_paths,
                snapshot.delivery_output_dir,
                overwrite=snapshot.overwrite,
                source_manifest_path=result.manifest_path,
            )
            if not delivery.passed:
                raise RuntimeError("delivery proxy library certification failed")
            detail += (
                "  \nDelivery proxy certification passed · "
                f"`{display_workspace_path(delivery.manifest_path, REPO_ROOT)}`"
            )
        return detail

    def compose_action(event_client: Any | None = None) -> None:
        captured = capture_build(event_client)
        if captured is None:
            return
        _start_job(owner, state, "Compose", lambda: compose_job(captured), client=event_client)

    def render_action(event_client: Any | None = None) -> None:
        captured = capture_build(event_client)
        if captured is None:
            return
        _start_job(owner, state, "Render", lambda: render_job(captured), client=event_client)

    def validate_action(event_client: Any | None = None) -> None:
        captured = capture_build(event_client)
        if captured is None:
            return
        _start_job(owner, state, "Validate", lambda: validate_job(captured), client=event_client)

    def build_action(event_client: Any | None = None) -> None:
        # A full build is reproducible from the spec on disk, so save first.
        if not save_action(event_client):
            return
        captured = capture_build(event_client)
        if captured is None:
            return

        def job() -> str:
            compose_job(captured)
            if state.stop_event.is_set():
                raise _JobCancelled("client disconnected before rendering")
            render_job(captured)
            if state.stop_event.is_set():
                raise _JobCancelled("client disconnected before validation")
            return validate_job(captured)

        _start_job(owner, state, "Full build", job, client=event_client)

    def build_library_action(event_client: Any | None = None) -> None:
        if state.dirty and not save_action(event_client):
            return
        captured = capture_library(event_client)
        if captured is None:
            return
        _start_job(
            owner,
            state,
            "Library build",
            lambda: library_job(captured),
            client=event_client,
        )

    def cycle_keyframe(delta: int) -> None:
        if state.operation_lock.locked():
            return
        count = len(state.spec.keyframes)
        if count == 0:
            return
        current = state.selected_keyframe_index
        state.selected_keyframe_index = (0 if current is None else current + delta) % count
        state.syncing = True
        try:
            _sync_keyframe_editor(state)
        finally:
            state.syncing = False

    spec_change_controls = (
        ui.behavior_id,
        ui.description,
        ui.behavior_type,
        ui.speaking_mode,
        ui.fps,
        ui.duration,
        ui.boundary,
        ui.source_reference,
        ui.base_mode,
        ui.base_motion_path,
        ui.base_frame,
        ui.ambient_enabled,
        ui.breathing_cycles,
        ui.breathing_degrees,
        ui.sway_cycles,
        ui.sway_degrees,
        ui.head_sway_degrees,
        ui.face_gaze,
        ui.face_expression,
        ui.face_blink,
        ui.face_intensity,
        ui.keyframe_time,
        ui.keyframe_label,
        ui.rotation_x,
        ui.rotation_y,
        ui.rotation_z,
        ui.joint_locked,
        ui.output_width,
        ui.output_height,
        ui.render_width,
        ui.render_height,
        ui.camera_position,
        ui.camera_look_at,
        ui.camera_up,
        ui.camera_fov,
        ui.background_rgb,
        ui.body_rgb,
    )

    def mark_spec_editor_changed(_: Any) -> None:
        if state.syncing or state.operation_lock.locked():
            return
        state.dirty = True
        _invalidate_preview(state, "The editor has uncomposed changes.")

    for control in spec_change_controls:
        control.on_update(mark_spec_editor_changed)

    # Freeze the full authoring/build surface while a captured background job
    # is active. Preview visibility and scrubbing remain usable.
    state.job_controls = (
        ui.spec_choice,
        ui.spec_path,
        ui.refresh_specs,
        ui.load_spec,
        ui.save_spec,
        *spec_change_controls,
        ui.keyframe,
        ui.joint,
        ui.upsert_keyframe,
        ui.remove_joint,
        ui.delete_keyframe,
        ui.apply_lock,
        ui.npz_path,
        ui.mp4_path,
        ui.manifest_path,
        ui.validation_path,
        ui.draft_resolution,
        ui.overwrite,
        ui.compose,
        ui.render,
        ui.validate,
        ui.build_all,
        ui.library_spec_source,
        ui.library_output_dir,
        ui.library_delivery_proxies,
        ui.library_delivery_output_dir,
        ui.library_switch_test,
        ui.build_library,
    )

    @ui.spec_choice.on_update
    def _(_) -> None:
        if not state.syncing:
            ui.spec_path.value = ui.spec_choice.value

    @ui.refresh_specs.on_click
    def _(event: viser.GuiEvent) -> None:
        refresh_specs(event.client)

    @ui.load_spec.on_click
    def _(event: viser.GuiEvent) -> None:
        load_action(event.client)

    @ui.save_spec.on_click
    def _(event: viser.GuiEvent) -> None:
        save_action(event.client)

    @ui.behavior_id.on_update
    def _(_) -> None:
        if not state.syncing:
            state.dirty = True
            _set_suggested_paths(state)

    @ui.keyframe.on_update
    def _(_) -> None:
        if state.syncing:
            return
        try:
            state.selected_keyframe_index = option_keyframe_index(ui.keyframe.value)
            state.syncing = True
            _sync_keyframe_editor(state)
        finally:
            state.syncing = False

    @ui.joint.on_update
    def _(_) -> None:
        if state.syncing:
            return
        state.syncing = True
        try:
            # A joint absent from the selected sparse keyframe is a valid new
            # target: keep the user's choice and present a zero-degree starting
            # point instead of snapping back to the first existing joint.
            _sync_keyframe_editor(state, choose_existing_joint=False)
        finally:
            state.syncing = False

    @ui.upsert_keyframe.on_click
    def _(event: viser.GuiEvent) -> None:
        commit_keyframe(event.client)

    @ui.remove_joint.on_click
    def _(event: viser.GuiEvent) -> None:
        remove_joint_action(event.client)

    @ui.delete_keyframe.on_click
    def _(event: viser.GuiEvent) -> None:
        delete_action(event.client)

    @ui.apply_lock.on_click
    def _(event: viser.GuiEvent) -> None:
        lock_action(event.client)

    @ui.compose.on_click
    def _(event: viser.GuiEvent) -> None:
        compose_action(event.client)

    @ui.render.on_click
    def _(event: viser.GuiEvent) -> None:
        render_action(event.client)

    @ui.validate.on_click
    def _(event: viser.GuiEvent) -> None:
        validate_action(event.client)

    @ui.build_all.on_click
    def _(event: viser.GuiEvent) -> None:
        build_action(event.client)

    @ui.build_library.on_click
    def _(event: viser.GuiEvent) -> None:
        build_library_action(event.client)

    @ui.preview_frame.on_update
    def _(_) -> None:
        _set_preview_pose(state, int(ui.preview_frame.value))

    @ui.show_preview.on_update
    def _(_) -> None:
        if state.preview_character is not None:
            visible = bool(ui.show_preview.value)
            state.preview_character.set_skeleton_visibility(visible)
            state.preview_character.set_skinned_mesh_visibility(visible)

    @ui.preview_play.on_click
    def _(event: viser.GuiEvent) -> None:
        if state.preview_motion is None:
            _notify(event.client, "Nothing to preview", "Compose the deterministic motion first.", "orange")
            return
        state.preview_playing = not state.preview_playing
        ui.preview_play.label = "Pause preview" if state.preview_playing else "Play preview"
        if state.preview_playing:
            _start_preview_thread(owner, state)

    def handle_pose_studio_keyboard(event: viser.KeyboardEvent) -> bool:
        """Handle Pose Studio chords through the demo's single key listener."""

        if event.event_type != "keydown":
            return False
        command = event.ctrl_key or event.meta_key
        key = event.key.lower()
        if command and key == "s":
            save_action(event.client)
        elif command and event.shift_key and event.key == "Enter":
            build_action(event.client)
        elif command and event.key == "Enter":
            compose_action(event.client)
        elif event.alt_key and key == "a":
            commit_keyframe(event.client)
        elif event.alt_key and event.key in ("Delete", "Backspace"):
            delete_action(event.client)
        elif event.alt_key and event.key == "[":
            cycle_keyframe(-1)
        elif event.alt_key and event.key == "]":
            cycle_keyframe(1)
        else:
            return False
        return True

    # The viser fork supports one keyboard callback per client. The IO tab owns
    # that listener for playback/navigation and dispatches modifier chords here
    # so Pose Studio cannot replace the existing controls.
    state.keyboard_handler = handle_pose_studio_keyboard

    _sync_all(state, reset_output_paths=True)
    if initial_error:
        _status(state, "Default spec could not be loaded", initial_error, icon="⚠️")
    else:
        _status(state, "Spec loaded", display_workspace_path(initial_path, REPO_ROOT), icon="✅")
