# MuseTalk Pose Project — Complete Context and Mission

Last updated: July 20, 2026

## Executive summary

This project is building a library of reusable human-pose videos for an AI video-call application. The backend will combine these videos with MuseTalk lip-sync generation, potentially through Segmind, to produce a hyperrealistic avatar that can change behavior during a call.

The essential requirement is that switching between pose videos must look continuous. A viewer should not be able to tell that separate clips were appended or exchanged. The background is already stable; the remaining visible discontinuities come primarily from changes in the avatar's head position, shoulder position, scale, posture, and facial expression.

The final system must therefore do more than ask the person to approximately return to the same pose. Every finished asset must use exact, shared boundary footage derived from one canonical default video.

## Core mission

Create a recording and asset-building workflow that produces:

- A canonical default pose video whose first and last frames match exactly.
- Gesture videos such as nod, smile, and active listening.
- Identical opening and ending boundary blocks across every finished asset.
- Smooth motion away from the default pose and back to it.
- Reliable validation while recording so poor takes are rejected immediately.
- Finished videos that can be processed through MuseTalk/Segmind and switched during a live video call without noticeable cuts.

Success means that any finished clip can transition to any other finished clip without a visible jump in the avatar or background.

## Original editing specification

The original reference diagram described a reusable master-pose workflow.

### Canonical default video

The default asset should be created from one neutral-pose recording:

1. Use approximately five seconds of the original neutral recording.
2. Append the same footage in reverse.
3. Produce a roughly ten-second default video.
4. Because the second half retraces the first half, the final frame returns to the same visual state as the opening frame.

This corrected an earlier misunderstanding: the default should not simply be a normal ten-second recording with independently captured opening and ending poses.

### Non-default gesture video

Each gesture asset should follow this structure:

1. **A — Canonical opening block:** Copy the first portion directly from the finished default master.
2. **B — Forward gesture:** Perform the nod, smile, glance, listening motion, or other behavior.
3. **C — Reverse return:** Cut the gesture at a clean peak or midpoint and reverse the appropriate segment to create a deterministic return path.
4. **D — Canonical ending block:** Copy the last portion directly from the finished default master.

Conceptually:

```text
A: exact default opening
        ->
B: gesture moving outward
        ->
C: reversed gesture returning to neutral
        ->
D: exact default ending
```

The copied blocks must be exact frame copies, not newly recorded approximations of the default pose.

## Project history

### 1. DaVinci Resolve approach

The work initially centered on a DaVinci Resolve project called **MuseTalk Poses**, which contained the source pose recordings. The first request was to confirm that the project could be modified.

The intended implementation was then described using the reusable master-pose diagram. The goal was to create a default asset and build the other gestures from portions of it plus forward/reverse gesture editing.

Working directly through DaVinci Resolve became slow and required macOS screen-recording or UI-control permissions. Editing project/XML files was briefly considered as a faster alternative.

The DaVinci workflow was then intentionally stopped. The new direction was to use a programmable editing environment and suitable video-processing tools instead of controlling Resolve.

### 2. Correction to the default-video design

An early output did not guarantee that the first and last frames of the default and nod clips were the same. The corrected requirement became:

- Build the default pose from five seconds forward plus the same five seconds reversed.
- Treat that new default video as the canonical source for every other video.
- Construct all gesture assets using exact pieces copied from the canonical default.

This is still the governing editing requirement.

### 3. Lingua and Segmind direction

The generated pose assets were intended to be added to the **Lingua** project. The next intended testing stage was to run at least the default, nod, and smile videos through the Segmind API and inspect the resulting MuseTalk/lip-sync output.

The current recordings should be treated as recording-system tests, not final Segmind production assets. They should not yet be considered the authoritative default, nod, or smile library.

### 4. Existing generated-video diagnosis

An example file named `webrtc_fully_prepared_pose_queue_test_jul_19.mp4` was supplied to evaluate switching quality.

The important clarification was that the background was completely still. The noticeable chops came from the avatar itself, especially changes in head position. Therefore, future analysis and validation should prioritize:

- Head-center translation.
- Head angle and tilt.
- Head/face scale.
- Shoulder position and angle.
- Torso position.
- Facial-expression state.
- The exact boundary frames used during clip changes.

## iOS recording application

A local iOS camera application named **PoseAnchorRecorder** was created in Xcode. It is designed for recording on a personal iPhone and guiding the subject back toward a reference pose.

Relevant project files include:

- `PoseAnchorRecorder/PoseAnchorRecorder/ContentView.swift`
- `PoseAnchorRecorder/PoseAnchorRecorder/Camera/PoseCaptureController.swift`
- `PoseAnchorRecorder/PoseAnchorRecorder/Tracking/PoseAnalyzer.swift`
- `PoseAnchorRecorder/PoseAnchorRecorder/Tracking/PoseComparisonEngine.swift`
- `PoseAnchorRecorder/PoseAnchorRecorder/Models/PoseModels.swift`
- `PoseAnchorRecorder/PoseAnchorRecorder/Services/CaptureExporter.swift`
- `PoseAnchorRecorder/README.md`
- `PoseAnchorRecorder/Documentation/CODE_GUIDE.md`
- `PoseAnchorRecorder/Documentation/DEVICE_SETUP.md`

### Intended recording experience

The app should:

- Capture an image or first-frame reference pose.
- Display a faded reference overlay during recording.
- Detect face/body alignment and motion.
- Show clear red/green guidance.
- Confirm a sufficiently long neutral opening hold.
- Detect a deliberate departure for the gesture.
- Confirm a sufficiently long neutral ending hold.
- Automatically finish or clearly approve a valid take.
- Allow the user to delete and replace the reference when starting over.

### Current important limitation

The green alignment indicator currently means that the detected body geometry is close to the reference. It does **not** necessarily mean that the app has successfully accumulated the required two seconds of stillness.

Natural landmark jitter can keep the stillness measurement above its threshold even when the person believes they are holding still. This explains why stopping immediately after seeing green produced errors such as:

- “The opening anchor hold was shorter than two seconds.”
- “The ending anchor hold was shorter than two seconds.”
- “The gesture never clearly departed from the neutral anchor.”

The recording state and the UI need to distinguish clearly between:

1. Aligned with the reference.
2. Currently holding still.
3. Opening hold successfully completed.
4. Gesture clearly detected.
5. Returned to neutral.
6. Ending hold successfully completed.

## iCloud recording analysis

Five folders matching `Take_202...` were found in iCloud Drive:

1. `Take_20260720_165259_2F72E3`
2. `Take_20260720_190159_5CF21A`
3. `Take_20260720_190458_CB9307`
4. `Take_20260720_190550_128D7F`
5. `Take_20260720_190609_31A9DF`

Each folder contained:

- `video.mov`
- `raw.mov`
- `reference.jpg`
- `session.json`

All five videos were portrait-oriented 1080p H.264 recordings at approximately 30 fps.

### Validation results

All five sessions reported `passed: false`.

All five reported:

- Opening hold duration: `0` seconds.
- Ending hold duration: `0` seconds.
- The opening hold was shorter than two seconds.
- The ending hold was shorter than two seconds.
- The gesture did not clearly depart from the neutral anchor.

Four recordings never progressed beyond the `findingStart` phase. One recording entered `holdingStart` for only a single analysis sample.

The configured stable-motion threshold was `0.35`, while measured motion around apparently still endpoints ranged from approximately `0.595` to over `5.0`. The hold detector was therefore much more sensitive than the real-world tracking signal allowed.

### Reference-image consistency

The four recordings beginning at `190159`, `190458`, `190550`, and `190609` used the exact same reference image.

The earlier `165259` take used a different reference image and visibly different framing. It must not be mixed into the later four-take set.

### First-to-last similarity within each recording

The first and last frames of each individual recording were compared using face-region similarity plus head and shoulder landmark movement.

| Take | Face-region similarity | Interpretation |
|---|---:|---|
| `165259` | 0.977 | Fairly close internally, but belongs to a different reference setup |
| `190159` | 0.975 | Close, but not identical |
| `190458` | 0.991 | Best internal endpoint match |
| `190550` | 0.978 | Close, but not identical |
| `190609` | 0.976 | Close, but not identical |

A score of `1.0` would represent an exact match under this comparison. None of these recordings has a pixel-identical first and last frame.

The relatively high internal scores are encouraging: the recording overlay helps the subject return near the starting pose. However, “near” is not strong enough to guarantee invisible switching.

### Cross-video transition results

Every directed end-to-start transition between the five videos was evaluated.

Best transitions:

- `190550 END -> 190609 START`: similarity approximately `0.976`; likely visually smooth, though not exact.
- `190609 END -> 190550 START`: similarity approximately `0.970`; reasonably good, with some head-position difference.

Moderate transitions:

- `190159 END -> 190458 START`: approximately `0.859`; visible change.
- `190458 END -> 190159 START`: approximately `0.853`; visible change.

Choppy transitions:

- Most other combinations among `190159`, `190458`, `190550`, and `190609` scored roughly `0.70–0.79` and contained noticeable head, shoulder, scale, or expression changes.
- Transitions involving `165259` scored roughly `0.59–0.69` and had the largest head/shoulder shifts. These are clearly unsuitable.

### Current verdict

The recordings are **still choppy as a general interchangeable library**.

Only the `190550` and `190609` pair approaches the desired smoothness. Even that pair does not provide the exact shared-frame guarantee required for arbitrary runtime switching.

These recordings demonstrate that reference-guided capture helps, but capture alone cannot guarantee seamless assets. Exact common boundary footage must be added during asset construction.

## Correct end-to-end workflow

The recommended production workflow is:

### Phase 1 — Fix and validate the recorder

1. Replace or supplement instantaneous frame-to-frame motion checks with a short rolling-window stability measurement.
2. Tolerate natural Vision landmark jitter while remaining strict about real body/head movement.
3. Separate geometric alignment from completed-hold status in both state logic and UI.
4. Require a confirmed opening hold before enabling gesture recording.
5. Require clear gesture departure.
6. Require a confirmed return and ending hold.
7. Auto-stop only after every required phase succeeds.
8. Preserve the ability to delete and recapture the reference.

### Phase 2 — Re-record source footage

1. Use one camera, orientation, lens, distance, lighting setup, chair position, and reference image.
2. Record a clean neutral/default source.
3. Record nod, smile, active listening, and other small gestures.
4. Let the app confirm both anchor holds; do not stop merely because the alignment indicator turns green.
5. Reject recordings with failed session validation.

### Phase 3 — Build the canonical default

1. Select a clean neutral segment, approximately five seconds long.
2. Append an exact reversed copy.
3. Verify that the resulting opening and ending frames are identical.
4. Treat this output as the only canonical boundary source.

### Phase 4 — Build gesture assets

For every gesture:

1. Copy the canonical default opening block exactly.
2. Insert the clean outward gesture.
3. Reverse the gesture from a suitable peak to return to neutral.
4. Copy the canonical default ending block exactly.
5. Avoid dissolves or generated interpolation at the external boundaries because they would destroy exact frame identity.

### Phase 5 — Automated quality assurance

For every finished asset:

- Confirm matching resolution, frame rate, orientation, codec expectations, and color properties.
- Confirm exact hashes for the shared boundary frames or blocks.
- Measure head, shoulder, and expression discontinuity at internal edit points.
- Generate every relevant end-to-start transition pair for visual inspection.
- Reject any asset that does not share the canonical boundaries.

### Phase 6 — MuseTalk/Segmind testing

1. Add the approved default, nod, and smile assets to Lingua.
2. Process each through the Segmind/MuseTalk pipeline with comparable audio.
3. Verify whether lip-sync processing changes cropping, stabilization, timing, or boundary frames.
4. Run a realistic WebRTC pose-switching sequence.
5. Re-check head/body continuity after generation, not only before it.

## What remains unfinished

- The app’s hold/stillness state machine still needs refinement.
- The app currently guides recording but does not yet implement the complete canonical-boundary asset builder from the original diagram.
- A validated canonical default asset has not yet been established.
- The nod, smile, and other gestures have not yet been rebuilt using exact copied default boundary blocks.
- The current five iCloud recordings should not be promoted as final production assets.
- Final Segmind/MuseTalk testing should happen after the recording and asset-building stages pass validation.

## Immediate next action

The highest-value next action is to update **PoseAnchorRecorder** so a user holding still can reliably complete the opening and ending anchors despite landmark noise. The app should expose unambiguous progress and should not accept a take until every phase has actually succeeded.

After that fix, record a new consistent batch, discard the differently framed `165259` setup, build the canonical forward-plus-reverse default, and construct each gesture using exact default boundary footage.

## Definition of done

The project is complete only when:

- A single canonical default master exists.
- Its first and last frames are exactly identical.
- Every gesture asset contains the exact same canonical opening and ending boundary blocks.
- Every recording passes opening-hold, gesture-departure, return, and ending-hold validation.
- Pairwise transition tests show no noticeable avatar jump.
- MuseTalk/Segmind processing preserves the seamless boundaries.
- A live Lingua/WebRTC test can switch among default, nod, smile, and other poses without revealing cuts.

