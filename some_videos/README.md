# Segmind Pose Test Assets — July 20 Retry

## Final assets

| Asset | File | Duration | Frames | Size |
|---|---|---:|---:|---:|
| Canonical idle | `default_idle_master.mp4` | 10.0 s | 300 | 32.0 MB |
| Nod | `nod_with_idle_boundaries.mp4` | 10.0 s | 300 | 31.6 MB |
| Look away | `look_away_with_idle_boundaries.mp4` | 10.0 s | 300 | 31.5 MB |

All three are H.264, 1080 × 1920 portrait, 30 fps, YUV 4:2:0, with no audio.

## Source assignment

Visual inspection of the recordings established the following roles:

- Idle: `Take_20260720_190159_5CF21A/video.mov`
- Nod: `Take_20260720_190458_CB9307/video.mov`
- Look away: `Take_20260720_190609_31A9DF/video.mov`

iCloud added three folders ending in ` 2`, but those files are byte-for-byte duplicates of previously analyzed recordings. The copied `165259` take was not used: it has different framing/reference geometry and does not contain the requested look-away movement. The visually correct look-away is `190609`.

## Construction

The canonical idle follows the required five-seconds-forward plus five-seconds-reversed path.

Every asset has this timeline:

```text
0.000–2.000 s   exact shared canonical opening block
2.000–2.333 s   short slow fade into the selected motion
2.333–7.667 s   pose-specific motion
7.667–8.000 s   short slow fade back to the canonical idle
8.000–10.000 s  exact shared canonical ending block
```

The fades are placed while both sources are approximately neutral. This minimizes the double-image effect that occurs when two differently positioned heads are dissolved during active motion.

## Boundary validation

Validation was performed against the decoded H.264 frames, not only the pre-encode filter graph.

- All three first frames are byte-identical after decoding.
- All three last frames are byte-identical after decoding.
- Within every video, its decoded first and last frames are byte-identical.
- The first 60 decoded frames—exactly two seconds—are identical across all three videos.
- The last 60 decoded frames—exactly two seconds—are identical across all three videos.

Decoded first/last frame MD5:

```text
7ea3a82f9f58468e44a7753e30a34357
```

Ordered decoded-frame checksum for the common first 60-frame block:

```text
c8b4620ab5e4b930c9c7e911a28624483fab60e0839d6935aa9cc25794074e83
```

Ordered decoded-frame checksum for the common last 60-frame block:

```text
dc4b708e390309b56c9fbaed19aa9b6d0ea27b7ae2f514d722ba986c298ae509
```

This means a direct cut from the end of any asset to the beginning of any asset lands on the exact same decoded image and follows the same canonical boundary motion.

## Rebuilding

Run:

```bash
./outputs/pose_assets_jul20_retry/build_assets.sh
```

The script reads the selected source recordings from iCloud Drive and recreates all three outputs.

## Segmind caution

These files are the correct inputs for the next Segmind/MuseTalk test. Segmind may decode, crop, resize, stabilize, or re-encode each asset independently. After generation, repeat the same decoded-frame and real switching tests on the Segmind outputs before treating them as production-ready.

