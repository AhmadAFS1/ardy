# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision authoring, analysis, rendering, and QA for avatar pose videos.

The public API stays independent of the interactive viser demo. Importing this
package does not initialize OpenGL; the EGL backend is loaded only when a
``CoreMeshRenderer`` is opened.
"""

from .composer import ComposedMotion, compose_behavior
from .config import CameraConfig, RenderStyle
from .delivery import (
    DeliveryProxyAsset,
    DeliveryProxyBuildResult,
    DeliveryProxySpec,
    build_delivery_proxy_library,
    transcode_delivery_proxy,
)
from .library import (
    BuiltBehavior,
    LibraryBuildResult,
    build_pose_library,
    create_switch_stress_test,
    discover_behavior_specs,
    renderer_camera_and_style,
    validate_shared_contract,
)
from .motion import CoreMotion, load_core_motion, resample_core_motion
from .pipeline import (
    RenderResult,
    VideoExportSpec,
    render_canonical_boundary_frames,
    render_motion_npz,
    render_motion_to_mp4,
)
from .reference_analysis import (
    ReferenceAnalysisConfig,
    analyze_frames,
    analyze_reference_set,
    analyze_video,
    infer_behavior,
)
from .renderer import CoreMeshRenderer, RenderingBackendError
from .spec import (
    BehaviorSpec,
    CameraSpec,
    JointTransform,
    PoseKeyframe,
    load_behavior_spec,
    save_behavior_spec,
)
from .validation import (
    VideoValidationExpectations,
    validate_library,
    validate_motion_npz,
    validate_video,
)

__all__ = [
    "BehaviorSpec",
    "BuiltBehavior",
    "CameraConfig",
    "CameraSpec",
    "ComposedMotion",
    "CoreMeshRenderer",
    "CoreMotion",
    "DeliveryProxyAsset",
    "DeliveryProxyBuildResult",
    "DeliveryProxySpec",
    "JointTransform",
    "LibraryBuildResult",
    "PoseKeyframe",
    "ReferenceAnalysisConfig",
    "RenderResult",
    "RenderStyle",
    "RenderingBackendError",
    "VideoExportSpec",
    "VideoValidationExpectations",
    "analyze_frames",
    "analyze_reference_set",
    "analyze_video",
    "build_pose_library",
    "build_delivery_proxy_library",
    "compose_behavior",
    "create_switch_stress_test",
    "discover_behavior_specs",
    "infer_behavior",
    "load_behavior_spec",
    "load_core_motion",
    "render_motion_to_mp4",
    "render_motion_npz",
    "render_canonical_boundary_frames",
    "renderer_camera_and_style",
    "resample_core_motion",
    "save_behavior_spec",
    "transcode_delivery_proxy",
    "validate_library",
    "validate_motion_npz",
    "validate_shared_contract",
    "validate_video",
]
