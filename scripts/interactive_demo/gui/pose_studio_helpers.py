# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the browser Pose Studio.

This module deliberately has no viser, torch, or ARDY imports.  Keeping the
small editing operations independent makes them inexpensive to test and keeps
GUI callbacks focused on presentation and background-job orchestration.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CORE_JOINT_NAMES: tuple[str, ...] = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
)

NEW_KEYFRAME_OPTION = "+ New keyframe"
_OPTION_INDEX = re.compile(r"^\[(\d+)]")


def discover_pose_specs(repo_root: str | Path) -> list[Path]:
    """Return deterministic, workspace-local pose specification choices."""

    root = Path(repo_root).resolve()
    return sorted(
        (root / "pose_specs").rglob("*.json"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def ensure_workspace_path(
    path: str | Path,
    repo_root: str | Path,
    *,
    suffix: str | None = None,
) -> Path:
    """Resolve a UI-supplied path and reject writes outside the repository."""

    root = Path(repo_root).resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay inside {root}") from exc
    if suffix is not None and candidate.suffix.lower() != suffix.lower():
        raise ValueError(f"path must end in {suffix}")
    return candidate


def validate_artifact_paths(
    artifacts: Mapping[str, str | Path],
    *,
    protected_paths: Iterable[str | Path] = (),
) -> dict[str, Path]:
    """Require distinct artifacts that cannot overwrite any source input."""

    resolved = {name: Path(path).resolve() for name, path in artifacts.items()}
    by_path: dict[Path, list[str]] = {}
    for name, path in resolved.items():
        by_path.setdefault(path, []).append(name)
    collisions = {path: names for path, names in by_path.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(
            f"{', '.join(names)} -> {path}" for path, names in sorted(collisions.items())
        )
        raise ValueError(f"artifact paths must be distinct: {detail}")

    protected = {Path(path).resolve() for path in protected_paths}
    overlaps = [(name, path) for name, path in resolved.items() if path in protected]
    if overlaps:
        detail = ", ".join(f"{name} -> {path}" for name, path in overlaps)
        raise ValueError(f"artifacts must not overwrite pose specification or source inputs: {detail}")
    return resolved


def display_workspace_path(path: str | Path, repo_root: str | Path) -> str:
    """Prefer a portable repo-relative path for GUI text controls."""

    candidate = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return str(candidate)


def keyframe_options(spec_payload: Mapping[str, Any]) -> list[str]:
    """Create stable dropdown labels without hiding a keyframe's identity."""

    options = [NEW_KEYFRAME_OPTION]
    for index, keyframe in enumerate(spec_payload.get("keyframes", [])):
        time_seconds = float(keyframe["time_seconds"])
        label = str(keyframe.get("label", "")).strip() or "untitled"
        joint_count = len(keyframe.get("joints", {}))
        suffix = "joint" if joint_count == 1 else "joints"
        options.append(f"[{index}] {time_seconds:.3f}s · {label} · {joint_count} {suffix}")
    return options


def option_keyframe_index(option: str) -> int | None:
    """Decode an option created by :func:`keyframe_options`."""

    if option == NEW_KEYFRAME_OPTION:
        return None
    match = _OPTION_INDEX.match(option)
    if match is None:
        raise ValueError(f"unrecognized keyframe option: {option!r}")
    return int(match.group(1))


def _finite_rotation(rotation_degrees: Sequence[float]) -> list[float]:
    values = [float(component) for component in rotation_degrees]
    if len(values) != 3 or not all(math.isfinite(component) for component in values):
        raise ValueError("rotation must contain three finite XYZ degree values")
    if any(abs(component) > 180.0 for component in values):
        raise ValueError("local XYZ rotation values must stay within [-180, 180] degrees")
    return values


def upsert_joint_keyframe(
    spec_payload: Mapping[str, Any],
    *,
    selected_index: int | None,
    time_seconds: float,
    label: str,
    joint_name: str,
    rotation_degrees: Sequence[float],
) -> tuple[dict[str, Any], int]:
    """Add or update one joint target and return its sorted keyframe index.

    When authoring a new target at an existing timestamp we merge it into that
    sparse keyframe.  This is both friendlier in the UI and preserves the spec's
    strict one-keyframe-per-time invariant.
    """

    payload = deepcopy(dict(spec_payload))
    keyframes = list(payload.get("keyframes", []))
    time_seconds = float(time_seconds)
    if not math.isfinite(time_seconds):
        raise ValueError("keyframe time must be finite")
    duration = float(payload["duration_seconds"])
    boundary = float(payload["boundary_seconds"])
    if time_seconds <= boundary or time_seconds >= duration - boundary:
        raise ValueError(
            f"keyframe time must stay in the editable interior "
            f"({boundary:.3f}, {duration - boundary:.3f}) seconds"
        )
    if joint_name not in CORE_JOINT_NAMES:
        raise ValueError(f"unknown Core joint: {joint_name}")
    if joint_name in payload.get("locks", []):
        raise ValueError(f"{joint_name} is locked; unlock it before authoring a target")

    target: dict[str, Any]
    if selected_index is None:
        matching = next(
            (
                index
                for index, keyframe in enumerate(keyframes)
                if math.isclose(float(keyframe["time_seconds"]), time_seconds, abs_tol=1e-9)
            ),
            None,
        )
        if matching is None:
            target = {"time_seconds": time_seconds, "label": label.strip(), "joints": {}}
            keyframes.append(target)
        else:
            target = keyframes[matching]
    else:
        if selected_index < 0 or selected_index >= len(keyframes):
            raise IndexError("selected keyframe no longer exists")
        target = keyframes[selected_index]
        for index, keyframe in enumerate(keyframes):
            if index != selected_index and math.isclose(
                float(keyframe["time_seconds"]), time_seconds, abs_tol=1e-9
            ):
                raise ValueError("another keyframe already uses that timestamp")
        target["time_seconds"] = time_seconds
        target["label"] = label.strip()

    target.setdefault("joints", {})[joint_name] = {
        "rotation_degrees": _finite_rotation(rotation_degrees)
    }
    keyframes.sort(key=lambda keyframe: float(keyframe["time_seconds"]))
    payload["keyframes"] = keyframes
    new_index = next(index for index, keyframe in enumerate(keyframes) if keyframe is target)
    return payload, new_index


def delete_keyframe(spec_payload: Mapping[str, Any], selected_index: int | None) -> dict[str, Any]:
    """Delete one complete sparse keyframe."""

    if selected_index is None:
        raise ValueError("select an existing keyframe to delete")
    payload = deepcopy(dict(spec_payload))
    keyframes = list(payload.get("keyframes", []))
    if selected_index < 0 or selected_index >= len(keyframes):
        raise IndexError("selected keyframe no longer exists")
    del keyframes[selected_index]
    payload["keyframes"] = keyframes
    return payload


def remove_joint_target(
    spec_payload: Mapping[str, Any], selected_index: int | None, joint_name: str
) -> tuple[dict[str, Any], int | None]:
    """Remove one joint target, deleting its now-empty keyframe if necessary."""

    if selected_index is None:
        raise ValueError("select an existing keyframe first")
    payload = deepcopy(dict(spec_payload))
    keyframes = list(payload.get("keyframes", []))
    if selected_index < 0 or selected_index >= len(keyframes):
        raise IndexError("selected keyframe no longer exists")
    joints = keyframes[selected_index].setdefault("joints", {})
    if joint_name not in joints:
        raise ValueError(f"{joint_name} has no target in the selected keyframe")
    del joints[joint_name]
    if joints:
        return payload, selected_index
    del keyframes[selected_index]
    payload["keyframes"] = keyframes
    return payload, None


def set_joint_lock(spec_payload: Mapping[str, Any], joint_name: str, locked: bool) -> dict[str, Any]:
    """Apply an explicit lock while preventing contradictory authored targets."""

    if joint_name not in CORE_JOINT_NAMES:
        raise ValueError(f"unknown Core joint: {joint_name}")
    payload = deepcopy(dict(spec_payload))
    locks = list(dict.fromkeys(payload.get("locks", [])))
    if locked:
        targeted_at = [
            float(keyframe["time_seconds"])
            for keyframe in payload.get("keyframes", [])
            if joint_name in keyframe.get("joints", {})
        ]
        if targeted_at:
            times = ", ".join(f"{value:.3f}s" for value in targeted_at)
            raise ValueError(f"{joint_name} has authored targets at {times}; remove them before locking")
        if joint_name not in locks:
            locks.append(joint_name)
    else:
        locks = [name for name in locks if name != joint_name]
    payload["locks"] = [name for name in CORE_JOINT_NAMES if name in locks]
    return payload


def suggested_output_paths(repo_root: str | Path, behavior_id: str) -> dict[str, Path]:
    """Return the canonical build locations for a behavior."""

    directory = Path(repo_root).resolve() / ".cache" / "pose_video" / behavior_id
    return {
        "npz": directory / f"{behavior_id}.npz",
        "mp4": directory / f"{behavior_id}.mp4",
        "manifest": directory / f"{behavior_id}.render.json",
        "validation": directory / f"{behavior_id}.validation.json",
    }


def locks_markdown(spec_payload: Mapping[str, Any]) -> str:
    locks = list(spec_payload.get("locks", []))
    if not locks:
        return "**Locked joints (0):** none"
    return f"**Locked joints ({len(locks)}):** " + ", ".join(f"`{name}`" for name in locks)
