# Deterministic motion-reference video renderer

`ardy.pose_video` turns a generated **CoreSkeleton27** motion archive into a
clean, fixed-camera MP4 without opening a browser or recording the viser UI.
It skins ARDY's 9,084-vertex Core humanoid, renders through an EGL offscreen
context, resamples rotations with quaternion SLERP, and pipes raw RGB frames
directly to ffmpeg.

Install the optional renderer dependencies and ensure `ffmpeg` is available:

```bash
python -m pip install -e '.[video]'
ffmpeg -version
```

Render a video-call-framed motion at 30 fps:

```bash
python scripts/render_pose_video.py outputs/motion.npz \
  --output pose_videos/motion.mp4 \
  --fps 30 \
  --manifest pose_videos/motion.render.json \
  --save-camera-json pose_videos/video_call_camera.json
```

Reuse that exact camera contract for every behavior asset:

```bash
python scripts/render_pose_video.py outputs/nod_agree.npz \
  --output pose_videos/nod_agree.mp4 \
  --camera-json pose_videos/video_call_camera.json \
  --manifest pose_videos/nod_agree.render.json
```

The master output is lossless all-intra H.264, yuv420p, silent, BT.709-tagged,
and contains no grid, timeline, gizmos, UI, or camera motion. Existing files
are never replaced unless `--force` is passed. Encoding is atomic, so an
interrupted render does not leave a partial file at the requested destination.

The manifest records source/output hashes, exact frame counts, frame rate,
camera, lighting/material parameters, and the hash of the uncompressed RGB
stream. GPU rasterization can vary a few edge pixels even on one driver; the
pipeline therefore renders one canonical RGB block and reuses it at both ends.
A complete library build reuses that same block across every asset. Interior
pixels need not be byte-reproducible across GPU drivers, and the raw-frame hash
makes such a difference explicit.

The built-in `video-call` camera is waist-up portrait framing. Use
`--camera full-body` while diagnosing whole-body motion. For seamless behavior
assets, camera equality alone is not enough: the source motions must still use
identical canonical boundary frames and matching boundary velocity. Prefer
`python -m ardy.pose_video build ...` for final multi-asset certification.

Lossless x264 can signal these reference masters as High 4:4:4 Intra even with
yuv420p pixels. That profile is intentional for exact archival seams, but it is
not the interchange tier. After the master build passes, create standards-
friendly High@4.0 copies with:

```bash
python -m ardy.pose_video proxy \
  pose_videos/neutral_resting.mp4 \
  pose_videos/nod_agree.mp4 \
  pose_videos/look_away_reset.mp4 \
  --output-dir pose_videos/delivery
```

The proxy builder uses constant QP 12 and all-IDR encoding, then repeats exact
cross-library decoded-pixel certification. It never mutates the source masters.
