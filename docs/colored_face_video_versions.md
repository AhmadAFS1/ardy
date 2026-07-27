# Colored-face pose video versions

Both video generations below are repository assets. They intentionally live in
separate paths so the approved two-blink baseline is not overwritten by the
three-blink candidate.

## Two-blink baseline

Blink peaks occur twice in the action section. The first and final 60 frames
remain fully open.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `look_away_reset` | `ee2b19b8a3bd47a9c596f24e6c05ee0343a01c7c7fa1fe2d5eabd1f8fe1d3a45` |
| Master | `neutral_resting` | `4c1de01009b5eab00f0111526a998c3a2c1e402e0d6e8bde1f10d0141f808525` |
| Master | `nod_agree` | `fad67d8a70674b6c4f668f7e73fa2b104dede93e7cba1d583311ec81f0c9f30f` |
| Delivery | `look_away_reset` | `5bd4935b48d89877370190995980100c7135c651d595e605338110bbc9004adf` |
| Delivery | `neutral_resting` | `f65f6b37b0ddcfe96c86a4eec30751d6d15666dddce5e8033d0d46708cc27c07` |
| Delivery | `nod_agree` | `b9bb7a1d1eea745583af8f645941663b98c9822b5a94649a01b8c9fe906ed179` |

Masters are in `outputs/pose_library/`; delivery copies are in
`outputs/pose_delivery/`.

## Three-blink candidate

Blink peaks occur at frames 79, 143, and 206 (approximately 2.63, 4.77, and
6.87 seconds at 30 fps). The middle blink reopens one frame more slowly to
avoid a mechanical repeated rhythm. The first and final 60 frames remain fully
open.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `look_away_reset` | `7400a6f4a2e50f5136d07b70a719ae9f2a9d7c55c812789ba64e4001b68a2160` |
| Master | `neutral_resting` | `37ac5d129e29569b57cada2fbf6cc150aaebf8117d1103201bd3c8d0f8779177` |
| Master | `nod_agree` | `a92ed4df82906c375172bb95b763fdc2eb84760684939dbb8829d5b1bdae23e7` |
| Delivery | `look_away_reset` | `6fe519f80859ced76b4d35b83e6735340f1c1a1001c95374ac4407c99159f192` |
| Delivery | `neutral_resting` | `7c13dc3125beec0e87ce2a2ddd61cdcc66e1e41b95151fd46408946dfd63ff95` |
| Delivery | `nod_agree` | `08ac4ba111af02a36a75bd8bb5df52778d479703d331e6594d4e7d7a326f506a` |

Masters are in `outputs/pose_library_blink3/`; delivery copies are in
`outputs/pose_delivery_blink3/`.

## Three-blink light neutral-resting candidate

This candidate keeps the three-blink timing at frames 79, 143, and 206, but
retains 30% of the open-eye height at each peak. Only `neutral_resting` uses
this softer partial-blink profile; the earlier two- and three-blink versions
remain unchanged in their original paths.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `64c2a403ef28ed407a51514572aa961ed20e6329652d231b7d8e8c7b934a9a49` |
| Delivery | `neutral_resting` | `e3271a12ee8e90c4f2644fee7ff74a97d3f74b539de22dab988314cdf75461b2` |

The master is in `outputs/pose_library_blink3_light/`; the delivery copy is in
`outputs/pose_delivery_blink3_light/`.

## Light-blink neutral with realistic eye proportions

This candidate retains the three soft partial blinks and reduces each apparent
eye from roughly 44 mm to 34 mm wide. Eye height, iris, pupil, and the
underlying eyelid deformation are scaled down together to match realistic
facial proportions.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `0c1414f6a2a629364120371c90beb9ba7af822efd966fbd983c9a25b860ff008` |
| Delivery | `neutral_resting` | `1f958f9cf975ada8c1f285d7ffdaea8187d876696ee9c6a737e5047c85c6b6a6` |

The master is in `outputs/pose_library_blink3_light_realistic_eyes/`; the
delivery copy is in `outputs/pose_delivery_blink3_light_realistic_eyes/`.

## Clean-eye light-blink neutral

This candidate covers the molded Core eyes with a head-colored socket layer
and places the realistic procedural eyes at the natural eye-line. This removes
the second, lower pair of eyes from every rendered frame while preserving the
three soft partial blinks.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `28bfd54c55e9ed4314dcfd4dc5a5bea1999fb33d9eaf90b0b2dfe535fec775f0` |
| Delivery | `neutral_resting` | `a9a7cc3921aa23ef429df701d7e3464907370fda679b35e3bc9672fd880cfe7c` |

The master is in
`outputs/pose_library_blink3_light_realistic_eyes_clean/`; the delivery copy is
in `outputs/pose_delivery_blink3_light_realistic_eyes_clean/`.

## Clean-eye neutral with 50%-slower blinks

This candidate preserves the clean eye sockets, realistic eye proportions,
three blink peaks, and 30% minimum eye height from the clean-eye version. Each
closing and reopening phase lasts 1.5 times as long, while the blink peaks
remain at frames 79, 143, and 206.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `e33b5fd0f4d12cbcff41e1852189027ef0b9fdd0d28b15f5b35b6743f0047cb5` |
| Delivery | `neutral_resting` | `38494a9dedd388f712656c06df5f986e7d0e45ee66abfb484ece1f9c0bd9eb7f` |

The master is in
`outputs/pose_library_blink3_light_realistic_eyes_clean_slow_blink/`; the
delivery copy is in
`outputs/pose_delivery_blink3_light_realistic_eyes_clean_slow_blink/`.

## Slow half-open blink neutral

This candidate keeps the clean eyes, three 50%-slower blinks, and established
blink peak frames. The eyes retain 50% of their normal open height at each
peak, making the motion a gentle partial blink instead of a near closure.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `0389768afdbf1d3ba7d1248f37b7248b8cb27444f13919ae62b10aea5dd847e1` |
| Delivery | `neutral_resting` | `33e521024f80fc62fce95fe636d035fe5dfefd7bfad29349e0c3afd01b01d903` |

The master is in
`outputs/pose_library_blink3_light_realistic_eyes_clean_slow_blink_half_open/`;
the delivery copy is in
`outputs/pose_delivery_blink3_light_realistic_eyes_clean_slow_blink_half_open/`.

## Extra-slow half-open blink neutral

**Status: most successful baseline as of July 25, 2026.** This delivery was
used to produce the successful downstream `neutral_resting_JUL25` video, where
the squinting issue was confirmed absent. Preserve this version as the
preferred fallback while later blink-depth experiments are evaluated.

This candidate slows the preceding half-open blink by another 50%. Each
closing and reopening phase is 1.5 times as long as the preceding version and
2.25 times the original duration. The eyes still retain 50% of their normal
height at the three established blink peaks.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `aca98385be83dd50407116acf9b82b56b28d738afb48ec3313781034fb134020` |
| Delivery | `neutral_resting` | `6c7a11221c1b7016171c19f2fcf814a26f88f5c7bc8e46168560717db1366cce` |

The master is in
`outputs/pose_library_blink3_light_realistic_eyes_clean_extra_slow_half_open/`;
the delivery copy is in
`outputs/pose_delivery_blink3_light_realistic_eyes_clean_extra_slow_half_open/`.

## Experimental extra-slow 30%-open blink

This comparison keeps the geometry and extra-slow timing from the successful
50%-open baseline, but returns the blink peak to 30% of the normal eye height.
It is an experiment only; the 50%-open `neutral_resting_JUL25` source remains
the preferred successful baseline above.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `4dea4f25a9522d388acc64e96b3fef3edc621752691dae4f8f0903675a6eb618` |
| Delivery | `neutral_resting` | `bc8a3dc8a388b22e73b1bccd9e23b1ea54ae8d32c1736c7eda29401b2dce4664` |

The master is in
`outputs/pose_library_blink3_light_realistic_eyes_clean_extra_slow_30pct_open/`;
the delivery copy is in
`outputs/pose_delivery_blink3_light_realistic_eyes_clean_extra_slow_30pct_open/`.

## Extra-slow 40%-open blink candidate

This candidate follows the downstream `CANDIDATE JUL630pm` evaluation. It
keeps the same clean eyes, three extra-slow blink curves, and established peak
frames, while retaining 40% of the normal eye height at each blink peak. It
sits midway between the 30%-open experiment and the successful 50%-open
baseline.

| Tier | Behavior | SHA-256 |
| --- | --- | --- |
| Master | `neutral_resting` | `7fcd28cb7f721cbdfb47e539b4faefee6fe19b46565b2fdf5f84e3863e0e5a99` |
| Delivery | `neutral_resting` | `182d5aca55d289a2359b0393274ded03bf61f19b5a2d071006c7b5fb250ab819` |

The master is in
`outputs/pose_library_blink3_light_realistic_eyes_clean_extra_slow_40pct_open/`;
the delivery copy is in
`outputs/pose_delivery_blink3_light_realistic_eyes_clean_extra_slow_40pct_open/`.

## Ten-pose clean-eye 40%-open library

**Status: complete production candidate as of July 26, 2026.** The established
clean-eye geometry and extra-slow 40%-open blink profile are now applied to a
variable-duration set of ten behaviors. Every asset has the same decoded
15-frame opening and closing block, so all ordered video-to-video transitions
are certified compatible.

See `docs/pose_library_cleaneyes_extra_slow_blink40_v1.md` for durations,
runtime switching instructions, output paths, certification results, and
release hashes.
