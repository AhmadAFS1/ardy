# Precision pose-video workflow

This workflow turns exact ARDY Core body-motion specifications into fixed-camera
MP4 motion references, verifies their transition boundaries, and prepares them
for a downstream identity pass such as Kling Motion Control. The MP4s produced
by ARDY are **motion references**, not finished photoreal avatar calls.

## The shared library contract

Every compatible asset in this repository uses the same contract:

| Property | Required value |
|---|---:|
| Frame rate | 30 FPS |
| Duration | 10.0 seconds |
| Total frames | 300 |
| Opening canonical block | frames 0–59 |
| Authored interior | frames 60–239 |
| Closing canonical block | frames 240–299 |
| Canonical block duration | 2.0 seconds / 60 frames |
| Master resolution | 1080 × 1920 |
| Draft resolution | 540 × 960 |
| Camera position | `[0.0, 0.56, 2.3]` |
| Camera look-at | `[0.0, 0.54, 0.0]` |
| Camera up | `[0.0, 1.0, 0.0]` |
| Vertical field of view | 25 degrees |
| Reference-master encode | Lossless all-intra H.264, yuv420p, BT.709, silent |
| Upload/browser proxy | H.264 High@4.0, constant QP 12, all-IDR, yuv420p, silent |

The precision composer copies the same canonical motion into both ends of every
clip. The two 60-frame blocks are equal frame-for-frame, their endpoint pose is
equal, and their boundary velocity approaches zero. Authored motion and ambient
motion are tapered back to neutral before the closing block.

Do not change the frame rate, duration, boundary duration, camera, resolution,
background, or body color on only one behavior. A library build rejects these
contract mismatches before rendering.

### Neutral silhouette contract

The shared authored neutral is framed to match a seated video-call portrait:
the clavicles remain broad, both upper-arm sockets stay outside the torso, and
the arms hang almost vertically with a small forward offset. At the canonical
pose, both upper arms and forearms remain visible and separated from the torso;
wrists and hands are intentionally below the portrait crop, as in the supplied
reference. The arms must not collapse behind the chest, merge into its
silhouette, or drift toward a T-pose. Every behavior inherits this base pose,
so changing it requires a complete library rebuild and fresh transition
certification.

## Behavior catalog

The top-level `pose_specs/` directory is the active three-behavior set:

| Behavior | Role |
|---|---|
| `neutral_resting` | Quiet idle hub with breathing and minimal sway |
| `nod_agree` | One restrained agreement nod |
| `look_away_reset` | Brief head-led look away and return |

Seven additional files live in `pose_specs/templates/`:

| Template | Intended motion |
|---|---|
| `speaking_neutral` | Stable body loop reserved for downstream lip sync |
| `active_listening` | Slight forward attention and one micro-nod |
| `thinking_glance` | Eyes-down/side intent with small head follow |
| `light_smile` | Facial smile intent with a tiny chin-lift proxy |
| `empathetic_tilt` | Small head tilt and softened attention |
| `curious_brow` | Facial brow intent with a tiny head tilt |
| `amused_reaction` | Restrained facial chuckle intent with a small body reaction |

These seven files are **calibration starters, not certified production assets**.
They pass the schema and deterministic motion composer, but each still needs
visual review against the actual avatar and downstream Kling result. Keep them
in `pose_specs/templates/` until their timing and amplitudes are approved.

## Important facial limitation

ARDY Core has a 27-joint body skeleton. It has no facial rig, eye controls,
eyelids, brows, cheeks, jaw, or lip blendshapes. Therefore:

- `face_intent` is handoff metadata. ARDY does not render it.
- ARDY can provide head, neck, and torso follow for a glance, smile, brow raise,
  or chuckle, but it cannot create the facial event itself.
- Blinks, eye-leading gaze, smiles, brow motion, and mouth motion must be added
  by Kling, a facial-animation model, MuseTalk, or a live facial layer.
- `speaking_neutral` intentionally keeps the body stable so MuseTalk or another
  lip-sync layer can own the mouth without fighting body motion.

Never approve a face-oriented template by looking only at the blue Core mesh.
The mesh can certify body timing and seams, not facial performance.

## 1. Set up the authoring environment

From the repository root, activate the project virtual environment first. All
commands below assume this environment remains active:

```bash
source .venv/bin/activate
python -m pip install -e '.[demo,video]'
ffmpeg -version
```

The video extra installs the headless EGL renderer. The `ffmpeg` executable is
also required. Start the browser authoring server:

```bash
python scripts/run_demo.py
```

Open `http://localhost:2333`. On a rented GPU host, expose or proxy TCP port
2333 through that provider instead of opening `localhost` on your own computer.

## 2. Measure an existing rough reference, if available

Reference analysis estimates visual activity timing; it does not recover a
pose or infer joint angles. Use it to find likely action onset, peak, hold, and
return times:

```bash
python -m ardy.pose_video analyze \
  some_videos/nod_with_idle_boundaries.mp4 \
  --behavior nod \
  --output outputs/reference_analysis/nod.json
```

Analyze the original three as a set to compare their coarse boundary behavior:

```bash
python -m ardy.pose_video analyze \
  some_videos/default_idle_master.mp4 \
  some_videos/nod_with_idle_boundaries.mp4 \
  some_videos/look_away_with_idle_boundaries.mp4 \
  --output outputs/reference_analysis/original_set.json
```

Treat detected timings as suggestions. Use Pose Studio to calibrate the actual
joint angles.

## 3. Author exact keyframes in Pose Studio

Open the **Pose Studio** tab in the browser.

1. Choose a top-level spec from **Available spec**, then select **Load selected
   spec**. To open a nested starter, enter a path such as
   `pose_specs/templates/empathetic_tilt.json` in **Spec path** and load it.
2. Confirm 30 FPS, 10.0-second duration, and 2.0-second canonical boundary.
3. Select an existing keyframe or **+ New keyframe**.
4. Enter its time, label, Core joint, and local XYZ rotation delta in degrees.
5. Select **Add / update joint target**. Adding another joint at the same time
   merges it into that sparse keyframe.
6. Repeat for the fewest joints needed to communicate the motion.
7. Save the spec, compose it, and scrub or play the 3D preview.
8. Iterate in small increments. For restrained video-call acting, a 0.25–1.0
   degree change is often meaningful; avoid large corrective edits.

The rotation values are local joint-space XYZ deltas from the authored neutral
pose. On the supplied Core rig, use X for pitch-like nod/lean changes, Y for
yaw-like turns, and Z for roll-like tilts, then verify the sign visually. Parent
and child rotations accumulate: a little `Spine3`, `Neck`, and `Head` rotation
usually looks more natural than putting the entire angle on `Head`.

Keyframe times must be strictly inside the authored interval:

```text
2.0 seconds < keyframe time < 8.0 seconds
```

Do not place a keyframe at exactly 2.000 or 8.000 seconds. The composer inserts
neutral targets at the interior edges, interpolates authored deltas with eased
spherical rotation interpolation, tapers ambient motion, and finally overwrites
both boundary blocks with the shared canonical block.

Useful shortcuts:

| Shortcut | Action |
|---|---|
| `Alt+A` | Add or update the selected joint target |
| `Alt+[` / `Alt+]` | Select previous or next authored keyframe |
| `Alt+Delete` | Delete the selected complete keyframe |
| `Ctrl/Cmd+S` | Save the spec |
| `Ctrl/Cmd+Enter` | Compose deterministic motion |
| `Ctrl/Cmd+Shift+Enter` | Compose, render, and validate |

These shortcuts work when browser focus is outside text and number fields.

### Lock discipline

Locks prevent contradictory authoring; they are not a downstream physics
simulation. Use them deliberately:

- Keep `Hips` and all lower-body joints locked for a seated, fixed-camera
  video-call library unless lower-body motion is explicitly needed.
- Lock both shoulders for head/neck reactions so an accidental target cannot
  cause arm drift.
- Unlock only the joint being intentionally authored.
- Pose Studio rejects a target on a locked joint.
- Pose Studio also rejects locking a joint that already has authored targets;
  remove those targets first.

Review the full clip after every lock change. A lock does not undo motion
already encoded in a custom base-pose NPZ.

### Draft versus master rendering

Use **Draft resolution** for fast visual iteration only. Turn it off before the
final build. A 540 × 960 draft must never be mixed with the 1080 × 1920 master
library or sent to the master Kling batch.

The browser buttons run the same deterministic pipeline as the CLI:

1. **Compose deterministic motion** writes the Core motion NPZ.
2. **Render MP4** creates the fixed-camera H.264 motion reference.
3. **Validate seams** checks motion arrays and decoded video pixels.
4. **Build + render + validate** runs all three for one working asset. It is
   deliberately labeled single-asset validation.
5. **Build + certify active library** renders every selected top-level spec
   with one shared canonical RGB cache, performs exact cross-asset validation,
   creates the switch stress test, and—when enabled—creates a second certified
   H.264 High-profile proxy library.

The lossless CRF0 masters may be reported by decoders as **High 4:4:4 Intra**
even though their stored pixels are yuv420p. Keep those files as the archival,
reproducible source. Use the separately certified High@4.0 proxy files for
upload/browser interchange. Never replace the master with a proxy or mix the
two tiers inside one validation batch.

## 4. Reproducible command-line builds

Use a new output directory for each review candidate. This retains old assets
and avoids accidental replacement.

Compose and validate one behavior without rendering:

```bash
python -m ardy.pose_video compose \
  pose_specs/nod_agree.json \
  --output-dir outputs/pose_video/nod_candidate_01
```

Render that exact motion with its behavior camera contract:

```bash
python -m ardy.pose_video render \
  outputs/pose_video/nod_candidate_01/nod_agree.motion.npz \
  --output outputs/pose_video/nod_candidate_01/nod_agree.mp4 \
  --behavior-spec pose_specs/nod_agree.json \
  --manifest outputs/pose_video/nod_candidate_01/nod_agree.render.json
```

Validate the motion and decoded MP4 explicitly:

```bash
python -m ardy.pose_video validate motion \
  outputs/pose_video/nod_candidate_01/nod_agree.motion.npz \
  --output outputs/pose_video/nod_candidate_01/nod_agree.motion.validation.json

python -m ardy.pose_video validate video \
  outputs/pose_video/nod_candidate_01/nod_agree.mp4 \
  --boundary-frames 60 \
  --output outputs/pose_video/nod_candidate_01/nod_agree.video.validation.json
```

Build, render, and certify every active top-level behavior together:

```bash
python -m ardy.pose_video build \
  pose_specs \
  --output-dir outputs/pose_video/master_candidate_01
```

The library build creates per-behavior motion/video validation reports,
`library.validation.json`, `library.manifest.json`, and a
`switch_stress_test.mp4`. Review the stress test at normal speed and frame by
frame. A green JSON report proves the measured seam contract; it does not prove
that the acting choice looks good.

Derive upload/browser copies only after the master library passes. Supplying
the source manifest binds the proxies to exactly that approved batch:

```bash
python -m ardy.pose_video proxy \
  outputs/pose_video/master_candidate_01/neutral_resting/neutral_resting.mp4 \
  outputs/pose_video/master_candidate_01/nod_agree/nod_agree.mp4 \
  outputs/pose_video/master_candidate_01/look_away_reset/look_away_reset.mp4 \
  --source-manifest outputs/pose_video/master_candidate_01/library.manifest.json \
  --output-dir outputs/pose_video/master_candidate_01_delivery
```

This command stages the complete batch, encodes at constant QP 12 rather than
CRF/ABR, verifies H.264 High@4.0 media facts and exact decoded 60-frame blocks,
then promotes the outputs. A failed encode or validation does not publish a
partial proxy library.

Exercise all seven calibration starters separately:

```bash
python -m ardy.pose_video build \
  pose_specs/templates \
  --output-dir outputs/pose_video/template_calibration_01
```

Passing this build means the starter files are structurally compatible. It does
not promote them to certified artistic assets. After review, save approved
calibrations as top-level specs and rebuild the complete active library in one
fresh output directory.

Existing output files are not overwritten by default. Use `--force` only when
you intentionally want to replace the exact destination.

## 5. Approval gate before Kling

Approve a motion reference only when all of the following are true:

- The motion NPZ validation report has `"passed": true`.
- The MP4 validation report has `"passed": true`.
- The complete library report has `"passed": true`.
- The delivery proxy library report has `"passed": true` when proxies will be
  uploaded.
- First and last 60-frame decoded blocks match within each clip and across the
  library.
- Frame rate, frame count, dimensions, camera, and render style match.
- The action reads clearly at normal speed but still looks restrained.
- No shoulder, hand, hip, or camera drift appears.
- The final return is calm before the closing canonical block begins.
- The switch stress test has no visible pop at any cut.

Archive the approved spec JSON, motion NPZ, MP4, render manifest, and validation
reports together. They form the reproducible source package for Kling.

## 6. Kling 2.6 Motion Control handoff

Kling controls and model behavior can change over time, so map these invariants
to the current Motion Control UI or API rather than relying on a particular
button name:

1. Use the approved ARDY delivery-proxy MP4 as the **motion reference only**;
   retain its lossless master and both manifests alongside it.
2. Reuse one approved identity image or identity source for every behavior.
3. Keep model version, identity settings, aspect ratio, crop, resolution,
   duration, seed strategy, guidance values, prompt structure, and negative
   prompt structure identical across the batch.
4. Require a static portrait camera: no zoom, pan, dolly, reframing, camera
   shake, stabilization, scene cut, or background animation.
5. Tell the downstream pass to preserve the full action timing and the neutral
   first/last two seconds. Do not ask it to improvise extra gestures.
6. Translate `face_intent` into the downstream facial prompt or facial-reference
   setup. ARDY itself did not render that expression.
7. Generate multiple candidates per behavior. Reject identity drift, framing
   drift, hand/shoulder artifacts, unrequested mouth motion, and any motion in
   the canonical blocks that is not shared by the whole library.
8. Export exactly 300 frames at 30 FPS and 1080 × 1920 when the service permits.
   Do not trim, retime, interpolate, crop, stabilize, or individually color
   grade one result after generation.

Record the Kling model version, request parameters, prompt, seed when exposed,
input hashes, output hash, and generation date for every candidate. A visually
good result that cannot be reproduced is not yet a master.

### Lip-sync layering

Prefer a continuous live MuseTalk/facial layer over the selected body-video
stream. This lets the body state switch independently while the mouth follows
live audio. If lip sync is baked into individual clips, the facial pixels in
every canonical block must also match, which is difficult when the audio differs.
After any MuseTalk or facial pass, repeat video and library validation on those
actual outputs.

## 7. Revalidate the real post-Kling files

The pre-Kling ARDY report cannot certify a stochastic downstream render. Validate
the exact files that will be shipped:

```bash
python -m ardy.pose_video validate video \
  outputs/kling_master/neutral_resting.mp4 \
  outputs/kling_master/nod_agree.mp4 \
  outputs/kling_master/look_away_reset.mp4 \
  --boundary-frames 60 \
  --output outputs/kling_master/library.validation.json
```

When all ten behaviors are ready, include all ten MP4 paths in the same command.
The multi-file report requires:

- each file's first and last decoded frame to match;
- each file's first and last decoded 60-frame block to match;
- all files to share the same opening and ending block hashes; and
- all files to share dimensions, FPS, and frame count.

Then repeat the validation after **every** operation that changes pixels:
Kling generation, MuseTalk, compositing, denoising, color conversion, scaling,
cropping, frame interpolation, and final encoding.

If a post-Kling report fails, do not hide the result with a crossfade. First
identify whether the failure is frame count/format, within-clip boundary drift,
or cross-library identity/pixel drift. Prefer regenerating with locked shared
inputs and settings. Replacing boundary frames after generation is only safe if
the replacement matches the adjacent identity, lighting, pose, and velocity;
otherwise it merely moves the visible pop to the interior edge. Revalidate and
visually inspect any repaired result.

## 8. WebRTC transition scheduling

Pixel-identical assets still require a frame-accurate scheduler. Use one
continuous 30-FPS output clock and one persistent WebRTC video track. Predecode
or keep all-intra assets warm, and swap the frame source inside the compositor;
do not replace the WebRTC track, reset RTP timestamps, or depend on two HTML
video elements starting at the same instant.

The simplest zero-risk policy is:

1. Play the current asset through frame 299.
2. Queue behavior requests rather than cutting immediately.
3. On the next output tick, emit frame 0 of the selected asset.
4. Keep RTP timestamps monotonic and advance exactly one frame per tick.
5. Let one-shot clips complete and return to `neutral_resting` by the same rule.
6. Loop `neutral_resting` from frame 299 to frame 0.

For lower transition latency, phase-align the shared blocks. Once the outgoing
clip is in closing frame `240 + k`, its pixels should equal opening frame `k` in
every approved incoming clip. On the next tick, the compositor can emit incoming
opening frame `k + 1` (or wrap from closing phase 59 to incoming frame 0). Never
phase-align unvalidated post-Kling files.

A minimal scheduler state is:

```text
active behavior + active frame index
pending behavior queue
30-FPS monotonic output timestamp
library version/hash
safe boundary length = 60 frames
```

Do not hard-seek from an arbitrary interior pose to frame 0, and do not crossfade
two faces; both reveal that clips are being switched. If a request arrives in
the authored interior, queue it until the closing canonical block.

With 10-second masters, waiting for a safe block can still introduce noticeable
response latency. The boundary architecture guarantees clean cuts, not instant
cuts. If the product needs faster reactions, create shorter compatible clips or
add more authored neutral-return windows and certify each declared switch point.
Do not trade away seam correctness by cutting the current 10-second asset in the
middle.

## Final production checklist

- Specs are reviewed, versioned, and immutable for this release.
- All active behaviors were built together under one shared contract.
- Draft assets are excluded from the production directory.
- ARDY motion, ARDY MP4, post-Kling MP4, post-face/lip-sync MP4, and final encode
  each have their own validation report.
- Facial behaviors were judged on the downstream avatar, not the Core mesh.
- A human reviewed the final switching sequence frame by frame and at speed.
- The WebRTC scheduler switches only at certified boundary phases.
- Runtime telemetry records behavior, frame index, late frames, dropped frames,
  and unexpected decoder resets so field regressions are detectable.
