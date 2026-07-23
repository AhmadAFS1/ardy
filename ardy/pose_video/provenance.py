"""Bind composed motion archives to their exact behavior specifications."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from .spec import BehaviorSpec, behavior_spec_fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_motion_provenance(path: str | Path, spec: BehaviorSpec) -> dict:
    """Return exact spec/scalar provenance checks for a composed motion NPZ."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as motion:
        if "metadata_json" not in motion:
            raise ValueError(f"{source} has no spec provenance; compose it again")
        metadata_value = np.asarray(motion["metadata_json"])
        if metadata_value.size != 1:
            raise ValueError(f"{source} metadata_json must be a scalar")
        try:
            metadata = json.loads(str(metadata_value.reshape(())))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{source} has invalid spec provenance metadata") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{source} spec provenance metadata must be an object")

        local = np.asarray(motion["local_rot_mats"]) if "local_rot_mats" in motion else None
        fps_value = np.asarray(motion["fps"]) if "fps" in motion else None
        boundary_value = (
            np.asarray(motion["boundary_frames"]) if "boundary_frames" in motion else None
        )
        behavior_value = np.asarray(motion["behavior_id"]) if "behavior_id" in motion else None

    expected_fingerprint = behavior_spec_fingerprint(spec)
    actual_fingerprint = metadata.get("spec_fingerprint")
    checks = {
        "matching_spec_fingerprint": actual_fingerprint == expected_fingerprint,
        "matching_behavior_id": (
            behavior_value is not None
            and behavior_value.size == 1
            and str(behavior_value.reshape(())) == spec.behavior_id
        ),
        "matching_fps": (
            fps_value is not None
            and fps_value.size == 1
            and float(fps_value.reshape(())) == float(spec.fps)
        ),
        "matching_boundary_frames": (
            boundary_value is not None
            and boundary_value.size == 1
            and float(boundary_value.reshape(())) == float(spec.boundary_frames)
        ),
        "matching_frame_count": local is not None and len(local) == spec.num_frames,
    }
    return {
        "schema_version": 1,
        "asset": str(source),
        "behavior_id": spec.behavior_id,
        "passed": all(checks.values()),
        "checks": checks,
        "expected_spec_fingerprint": expected_fingerprint,
        "actual_spec_fingerprint": actual_fingerprint,
    }


def require_motion_provenance(path: str | Path, spec: BehaviorSpec) -> dict:
    """Raise an actionable error when a motion was composed from another spec."""

    report = validate_motion_provenance(path, spec)
    if not report["passed"]:
        failed = ", ".join(name for name, passed in report["checks"].items() if not passed)
        raise ValueError(
            f"motion NPZ does not match the active pose specification ({failed}); "
            "compose it again before rendering or validating"
        )
    return report


def require_render_provenance(
    motion_path: str | Path,
    video_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    """Require a render manifest that binds the current MP4 to the current NPZ."""

    motion = Path(motion_path).resolve()
    video = Path(video_path).resolve()
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise ValueError(
            f"render manifest is missing for {video}; render this motion again before validating"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"render manifest is unreadable: {manifest}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"render manifest must be a JSON object: {manifest}")
    checks = {
        "matching_motion_sha256": payload.get("input_sha256") == _sha256_file(motion),
        "matching_video_sha256": payload.get("mp4_sha256") == _sha256_file(video),
        "matching_motion_path": Path(str(payload.get("input_path", ""))).resolve() == motion,
        "matching_video_path": Path(str(payload.get("output_path", ""))).resolve() == video,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(
            f"rendered MP4 does not match the active motion archive ({failed}); render it again"
        )
    return {"schema_version": 1, "passed": True, "checks": checks}
