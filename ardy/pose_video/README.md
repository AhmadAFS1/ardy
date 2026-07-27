# Precision pose-video tools

This package turns versioned local-joint keyframes into deterministic ARDY Core
motion, renders fixed-camera motion-reference MP4s, and certifies exact decoded
transition blocks across a complete behavior library. Pose durations may differ
as long as FPS, camera, encoder, base pose, and canonical boundary timing match.
It also extracts coarse timing from rough reference recordings without claiming
to recover anatomical joint angles.

Build the active library:

```bash
python -m ardy.pose_video build pose_specs \
  --output-dir outputs/pose_library
```

Other subcommands are `analyze`, `compose`, `render`, `proxy`, and `validate`; run any
one with `--help` for its exact inputs. The final library builder uses one
shared canonical RGB block, lossless all-intra H.264, fixed 30 FPS portrait
framing, and both motion- and pixel-level validation.

After approving the lossless master batch, create H.264 High@4.0 upload/browser
copies without weakening the shared decoded boundary:

```bash
python -m ardy.pose_video proxy \
  outputs/pose_library/neutral_resting/neutral_resting.mp4 \
  outputs/pose_library/nod_agree/nod_agree.mp4 \
  outputs/pose_library/look_away_reset/look_away_reset.mp4 \
  --source-manifest outputs/pose_library/library.manifest.json \
  --output-dir outputs/pose_delivery
```

For a variable-duration batch, the proxy command reads each asset's frame count
and duration from the certified source manifest while retaining one strict
shared media and boundary contract.

Analyze one reference:

```bash
python -m ardy.pose_video analyze reference_nod.mp4 \
  --behavior nod \
  --output outputs/reference_nod.analysis.json
```

Analyze and compare canonical blocks across a named reference set:

```bash
python -m ardy.pose_video analyze \
  default_idle_master.mp4 \
  nod_with_idle_boundaries.mp4 \
  look_away_with_idle_boundaries.mp4 \
  --output outputs/reference_set.analysis.json
```

Automatic behavior selection only inspects filenames. The pixel analyzer does
not classify actions, recover anatomy, or estimate yaw/pitch/roll. The generated
keyframes are timing seeds; exact joint rotations must be calibrated in the pose
editor. Fixed ROIs assume a centered video-call portrait and are configurable
through `ReferenceAnalysisConfig` for other compositions.

Analyzer boundary hashes cover reduced-resolution decoded luma frames. They are
fast authoring checks, not production certification. Always use `validate` on
the full-resolution files after Kling, MuseTalk, encoding, cropping, or
resizing. See `docs/pose_video_workflow.md` for the complete workflow and
WebRTC switching protocol.
