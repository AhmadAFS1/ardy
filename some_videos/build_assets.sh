#!/bin/zsh
set -euo pipefail

ICLOUD="/Users/ahmadsmacair/Library/Mobile Documents/com~apple~CloudDocs"
OUT="/Users/ahmadsmacair/Documents/Codex/2026-07-18/okay/outputs/pose_assets_jul20_retry"

# Visual inspection established these roles:
# 190159: neutral/idle
# 190458: nod
# 190609: look away
IDLE="$ICLOUD/Take_20260720_190159_5CF21A/video.mov"
NOD="$ICLOUD/Take_20260720_190458_CB9307/video.mov"
LOOK_AWAY="$ICLOUD/Take_20260720_190609_31A9DF/video.mov"

mkdir -p "$OUT"

ENCODE=(
  -c:v libx264
  -preset slow
  -qp 16
  -pix_fmt yuv420p
  -r 30
  -g 30
  -keyint_min 1
  -sc_threshold 0
  -force_key_frames "expr:eq(n,0)+eq(n,240)+eq(n,299)"
  -movflags +faststart
  -an
)

# Canonical idle: five seconds forward followed by the same path in reverse.
# The overlap windows contain the same source frames on both sides, so the
# fades do not alter the motion. Running the idle through the same two-stage
# fade pipeline as the pose assets also makes their decoded boundary frames
# identical after H.264 encoding.
ffmpeg -hide_banner -loglevel warning -y \
  -i "$IDLE" \
  -filter_complex \
  "[0:v]trim=start=0.5:end=2.833333,setpts=PTS-STARTPTS,fps=30,format=yuv420p[prefix];
   [0:v]trim=start=2.5:end=5.5,setpts=PTS-STARTPTS,fps=30,format=yuv420p,split=2[core_forward][core_reverse_input];
   [core_reverse_input]reverse,setpts=PTS-STARTPTS[core_reverse];
   [core_forward][core_reverse]concat=n=2:v=1:a=0,fps=30,settb=expr=1/30[core];
   [0:v]trim=start=0.5:end=2.833333,setpts=PTS-STARTPTS,fps=30,format=yuv420p,reverse,setpts=PTS-STARTPTS[suffix];
   [prefix][core]xfade=transition=fadeslow:duration=0.333333:offset=2.00[first_join];
   [first_join][suffix]xfade=transition=fadeslow:duration=0.333333:offset=7.666667,format=yuv420p[outv]" \
  -map "[outv]" \
  "${ENCODE[@]}" \
  -metadata title="Canonical Idle — Forward + Reverse" \
  "$OUT/default_idle_master.mp4"

# Nod asset:
# - exact first two seconds of the canonical idle
# - the clean neutral -> nod -> neutral portion of the nod take
# - exact final two seconds of the canonical idle
# The short 1/3-second slow fades occur while both sources are neutral. This
# avoids the long double-image/ghosting that a fade during the nod would cause.
ffmpeg -hide_banner -loglevel warning -y \
  -i "$IDLE" \
  -i "$NOD" \
  -filter_complex \
  "[0:v]trim=start=0.5:end=2.833333,setpts=PTS-STARTPTS,fps=30,format=yuv420p[prefix];
   [1:v]trim=start=0.5:end=6.5,setpts=PTS-STARTPTS,fps=30,format=yuv420p[gesture];
   [0:v]trim=start=0.5:end=2.833333,setpts=PTS-STARTPTS,fps=30,format=yuv420p,reverse,setpts=PTS-STARTPTS[suffix];
   [prefix][gesture]xfade=transition=fadeslow:duration=0.333333:offset=2.00[first_join];
   [first_join][suffix]xfade=transition=fadeslow:duration=0.333333:offset=7.666667,format=yuv420p[outv]" \
  -map "[outv]" \
  "${ENCODE[@]}" \
  -metadata title="Pose Asset — Nod" \
  "$OUT/nod_with_idle_boundaries.mp4"

# Look-away asset. This uses the visually verified 190609 recording; it is the
# take that actually contains the head turn. The newly copied 165259 folder is
# differently framed and does not contain the requested look-away motion.
ffmpeg -hide_banner -loglevel warning -y \
  -i "$IDLE" \
  -i "$LOOK_AWAY" \
  -filter_complex \
  "[0:v]trim=start=0.5:end=2.833333,setpts=PTS-STARTPTS,fps=30,format=yuv420p[prefix];
   [1:v]trim=start=1.8:end=7.8,setpts=PTS-STARTPTS,fps=30,format=yuv420p[gesture];
   [0:v]trim=start=0.5:end=2.833333,setpts=PTS-STARTPTS,fps=30,format=yuv420p,reverse,setpts=PTS-STARTPTS[suffix];
   [prefix][gesture]xfade=transition=fadeslow:duration=0.333333:offset=2.00[first_join];
   [first_join][suffix]xfade=transition=fadeslow:duration=0.333333:offset=7.666667,format=yuv420p[outv]" \
  -map "[outv]" \
  "${ENCODE[@]}" \
  -metadata title="Pose Asset — Look Away" \
  "$OUT/look_away_with_idle_boundaries.mp4"
