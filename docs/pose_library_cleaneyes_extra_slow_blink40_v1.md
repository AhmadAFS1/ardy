# Ten-pose clean-eye library v1

Built July 26, 2026.

This is the first complete variable-duration ARDY pose library using the
clean-eye face overlay, realistic eye proportions, extra-slow natural blink
timing, and a 40%-open blink floor.

## Shared switching contract

- 30 FPS, 1080 × 1920, H.264, `yuv420p`, no audio
- Lossless all-intra masters; H.264 High@4.0 constant-QP delivery copies
- Exactly 15 canonical frames (0.5 seconds) at both ends of every video
- Identical opening and closing decoded boundary blocks across all ten assets
- Master decoded boundary SHA-256:
  `dc64197171239fa84ae4f2a0cb4b23f8f3af5bef8f2c483c091f0ffe051600ea`
- Delivery decoded boundary SHA-256:
  `70233fb87d3d7aa7f19f9e0c7342d8d0434d5c2ee0d3c6da6894dfedfaca352f`
- All 100 ordered source-to-destination switches are certified compatible
- Blink floor: eyes retain 40% of their normal open height
- Blink duration scale: 2.25× the original blink curve
- Duration-aware blinking: three blinks in 10-second clips, two in the
  8-second clip, one in clips of at least 5.5 seconds, and none in shorter
  reaction clips

For a seamless runtime transition, play the outgoing asset through its final
15-frame canonical block, then begin the incoming asset at frame 0. Do not
seek into or skip either boundary block.

## Assets

| Behavior | Duration | Frames | Motion |
| --- | ---: | ---: | --- |
| `neutral_resting` | 10.0 s | 300 | Canonical calm idle |
| `nod_agree` | 2.8 s | 84 | Small acknowledgment nod |
| `look_away_reset` | 4.8 s | 144 | Light side glance and return |
| `active_listening` | 8.0 s | 240 | Attentive listening motion |
| `speaking_direct` | 10.0 s | 300 | Stable front-facing talk support |
| `thinking_glance` | 4.2 s | 126 | Brief down-and-side processing glance |
| `light_smile` | 4.0 s | 120 | Friendly restrained smile |
| `empathetic_head_tilt` | 4.8 s | 144 | Gentle supportive tilt |
| `curious_eyebrow_or_nod` | 3.6 s | 108 | Asymmetric brow lift and tiny curious nod |
| `amused_laugh` | 4.2 s | 126 | Restrained amused reaction |

## Output roots

- Lossless masters and animation data:
  `outputs/pose_library_cleaneyes_extra_slow_blink40_v1/`
- Browser/upload delivery videos:
  `outputs/pose_delivery_cleaneyes_extra_slow_blink40_v1/`
- Master certification:
  `outputs/pose_library_cleaneyes_extra_slow_blink40_v1/library.validation.json`
- Delivery certification:
  `outputs/pose_delivery_cleaneyes_extra_slow_blink40_v1/delivery.library.validation.json`

Each master behavior directory contains its lossless MP4, `.motion.npz`
animation asset, motion validation, video validation, and render metadata.

## Release hashes

| Behavior | Motion NPZ SHA-256 | Master MP4 SHA-256 | Delivery MP4 SHA-256 |
| --- | --- | --- | --- |
| `active_listening` | `ff1b67a0275f9faca03c68f00e8fd3ca64cabc7ab317f398923c00871ac49ebe` | `1a2b5a8b33724606d6a78dbeaa252d48d00a3594cdfd6b0c239e8e82ae989b37` | `cdc01a5c486e7b0b42d8ab6371688315d29a1f61d975f0850e2a2da50dd0ab48` |
| `amused_laugh` | `4ac4d7eebdc1aa09c7223e89e3c18e9a639732af515520306aa1c08c85f23e87` | `91ba1a057100e14471e40ac7383731ee69f343356b0fd849613f8cefbaa3eb4e` | `d4cba5738ef6462260dc9ef38e0bdb94f17452b4f1eaf44c22265acc93046231` |
| `curious_eyebrow_or_nod` | `17fbc2f6687909e19fcb00bc02e3e1d9156a5a03440269e15035ccc5ff9c7550` | `c3056046ef6a33a980c5fcf2af146a08b32c6104afa64d970fac2f6954c70580` | `68fa06a3a83f162b53cc5e5183bb69baa599323c37511e6d081f839716894081` |
| `empathetic_head_tilt` | `8e5951db76e3ed99b079fc11f56642faaca43071f0aaab73160a925701d67ff9` | `438e02208ebcd3e17f5e2bb3ccbda8dc0606d623af78f2857e1dab252431427b` | `a12f2abef843a4cac9375f4542400f39189133c29a01332a2fc4df3f56e09d49` |
| `light_smile` | `d834829a8785bfa809093410e67ab3988275a3f38673169b42faeb5e6ffd2871` | `7cf2b0b5db968fa915d158769fefb39f8f6c46bbe198de96abdf36401ed32657` | `de6e804d5dab577d035c596a4101d15f2ee27bee1356bf8ce6b9a0e0ca4e215b` |
| `look_away_reset` | `a05b9e8085ff26dc079f4a62ebb497bafbbabf8371c776f06007fe31578eedd9` | `346cb096c751d90bffd9a63f4cd05fe01c540fd9109873285c8fda8f40c169b3` | `f32fd540f0fb2e5bfaa07e1da072227704105af2b7ea8bbd3b4554e76d07e414` |
| `neutral_resting` | `5601ea4f9d2bb2e0833a306a293d4101a1b74fa1507a77850e56abce715fc1c4` | `530bcd27d0bdee1a4976816b6c1075500afc9feb00c3c2b76702714363dab48f` | `d92a1dd6da86aeac9a14eb5d84bad39860aee6c1804775c674c865ec4ebcdeb8` |
| `nod_agree` | `4880825a739b338c7f71c67480e000a298d702f4a0c5dec45a5cb88d847c30ca` | `ed694ee05f08113eaaaa924091aae083888bc0621610d8b6b3e9fb1111688534` | `8091082c46c000e7396aa2c3654b08c09b0a6fbf1d2e50aba6d6b1320d41cbe5` |
| `speaking_direct` | `1629ab5ae8b702bf4fe3c71ab1d8c3cd5f3b2352ffc817daff763284f0a68a36` | `7c1718ce4aa8d518f5c25211d3ef6c4180c3d3e8aeb4d2c26962024106afe9a1` | `0ae213d5e9f592be6587bdd8c2d509d6ba534faba0294c83b1afb247e4c7cebb` |
| `thinking_glance` | `9ab044ea9c08a5032e87525bf3b432bcded40388ea4c30f3199b3ac792656bad` | `16cb84f034cb752220e4a5dcf298a45f5cea6c284f46ebabd9fcdb0b00c33cfa` | `35f41bbb95d3174d583a9460faa8fa80699a38ecf2de5c9ca901aaaab3231345` |

The earlier 50%-open `neutral_resting_JUL25` baseline remains preserved in
`outputs/pose_delivery_blink3_light_realistic_eyes_clean_extra_slow_half_open/`.
