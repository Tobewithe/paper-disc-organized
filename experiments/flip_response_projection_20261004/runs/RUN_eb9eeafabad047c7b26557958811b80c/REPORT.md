# Whole-canvas reflection and original-basis projection

Frozen original YOLO26m-seg, no training. All effects are percentage points, normal full-mask decoding.
256 planned dev images, 253 effective images, 1816 candidates.

MIRROR maps anchor locations, not guaranteed instance identity. MATCH uses same level/predicted class and inverse-box overlap. MIX is ordinary equal logit averaging. PROJ solves only a coefficient increment in original full P.

| Group | Pair | Image macro IoU Δ [95% CI] | Candidate IoU Δ | Repair / damage | Coverage Δ | AUC Δ | FPR Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| all | MIX_minus_A | +0.0852 [-0.0621, +0.2356] | +0.0593 | 29 / 20 | -0.1319 | +0.1032 | -0.3198 |
| all | PROJ_minus_A | +0.0346 [-0.1193, +0.1835] | -0.0086 | 27 / 20 | -0.2063 | +0.0999 | -0.3049 |
| all | PROJ_minus_MIX | -0.0506 [-0.0866, -0.0135] | -0.0679 | 6 / 8 | -0.0744 | -0.0033 | +0.0149 |
| all | MATCH_minus_A | -2.7680 [-3.7565, -1.9124] | -2.7754 | 35 / 110 | -3.3496 | -1.0075 | -0.5875 |
| all | MATCH_minus_ORIG_MATCH | -2.7680 [-3.7565, -1.9124] | -2.7754 | 35 / 110 | -3.3496 | -1.0075 | -0.5875 |
| all | MATCH_minus_MIRROR | -2.2760 [-3.3146, -1.3239] | -2.3179 | 15 / 80 | -2.8298 | -0.7049 | -0.7212 |
| all | MIRROR_minus_A | -0.4920 [-1.0089, -0.1425] | -0.4574 | 35 / 45 | -0.5197 | -0.3026 | +0.1337 |
| all | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| all | MIX_minus_MATCH | +2.8532 [+2.0171, +3.8594] | +2.8347 | 100 / 16 | +3.2177 | +1.1107 | +0.2677 |
| box_good_mask_bad | MIX_minus_A | +0.5483 [+0.0998, +1.0515] | +0.3564 | 23 / 0 | +0.0866 | +0.2638 | -0.5330 |
| box_good_mask_bad | PROJ_minus_A | +0.4636 [+0.0547, +0.9457] | +0.2474 | 21 / 0 | -0.0463 | +0.2689 | -0.5244 |
| box_good_mask_bad | PROJ_minus_MIX | -0.0847 [-0.2132, +0.0432] | -0.1090 | 2 / 4 | -0.1329 | +0.0051 | +0.0086 |
| box_good_mask_bad | MATCH_minus_A | -0.6418 [-1.7339, +0.4819] | -1.1970 | 28 / 0 | -1.4712 | -0.8434 | -0.6105 |
| box_good_mask_bad | MATCH_minus_ORIG_MATCH | -0.6418 [-1.7339, +0.4819] | -1.1970 | 28 / 0 | -1.4712 | -0.8434 | -0.6105 |
| box_good_mask_bad | MATCH_minus_MIRROR | -0.7985 [-1.7279, +0.0522] | -0.9668 | 4 / 5 | -1.3313 | -0.2862 | -0.6508 |
| box_good_mask_bad | MIRROR_minus_A | +0.1567 [-0.4803, +0.7473] | -0.2303 | 29 / 0 | -0.1398 | -0.5572 | +0.0403 |
| box_good_mask_bad | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| box_good_mask_bad | MIX_minus_MATCH | +1.1901 [+0.3544, +2.0176] | +1.5534 | 6 / 11 | +1.5578 | +1.1072 | +0.0775 |
| original_success | MIX_minus_A | -0.0648 [-0.2242, +0.0558] | -0.0608 | 0 / 20 | -0.2195 | +0.0252 | -0.2627 |
| original_success | PROJ_minus_A | -0.1231 [-0.2861, -0.0107] | -0.1282 | 0 / 20 | -0.2858 | +0.0015 | -0.1989 |
| original_success | PROJ_minus_MIX | -0.0583 [-0.0965, -0.0197] | -0.0674 | 3 / 3 | -0.0663 | -0.0237 | +0.0638 |
| original_success | MATCH_minus_A | -3.4217 [-4.5163, -2.4632] | -3.9030 | 0 / 110 | -3.8431 | -0.9945 | -0.4179 |
| original_success | MATCH_minus_ORIG_MATCH | -3.4217 [-4.5163, -2.4632] | -3.9030 | 0 / 110 | -3.8431 | -0.9945 | -0.4179 |
| original_success | MATCH_minus_MIRROR | -2.6478 [-3.8287, -1.5431] | -3.2919 | 10 / 75 | -3.1664 | -0.7785 | -0.6960 |
| original_success | MIRROR_minus_A | -0.7738 [-1.3443, -0.3761] | -0.6112 | 0 / 45 | -0.6768 | -0.2160 | +0.2781 |
| original_success | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| original_success | MIX_minus_MATCH | +3.3569 [+2.4173, +4.3880] | +3.8423 | 93 / 3 | +3.6236 | +1.0197 | +0.1552 |
| original_failure | MIX_minus_A | +0.3562 [+0.0029, +0.7788] | +0.2610 | 29 / 0 | +0.1111 | +0.2827 | -0.3340 |
| original_failure | PROJ_minus_A | +0.3158 [-0.0345, +0.7133] | +0.1922 | 27 / 0 | -0.0237 | +0.2884 | -0.4412 |
| original_failure | PROJ_minus_MIX | -0.0405 [-0.1205, +0.0519] | -0.0688 | 3 / 5 | -0.1349 | +0.0057 | -0.1072 |
| original_failure | MATCH_minus_A | -0.5757 [-1.4896, +0.2932] | -0.8826 | 35 / 0 | -1.1893 | -0.8868 | -0.4696 |
| original_failure | MATCH_minus_ORIG_MATCH | -0.5757 [-1.4896, +0.2932] | -0.8826 | 35 / 0 | -1.1893 | -0.8868 | -0.4696 |
| original_failure | MATCH_minus_MIRROR | -0.6839 [-1.4982, -0.0274] | -0.6832 | 5 / 5 | -1.2550 | -0.4065 | -0.6942 |
| original_failure | MIRROR_minus_A | +0.1082 [-0.4081, +0.5853] | -0.1993 | 35 / 0 | +0.0657 | -0.4803 | +0.2246 |
| original_failure | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| original_failure | MIX_minus_MATCH | +0.9319 [+0.3277, +1.6028] | +1.1435 | 7 / 13 | +1.3004 | +1.1695 | +0.1356 |
| P3 | MIX_minus_A | +0.1762 [-0.0940, +0.5080] | +0.0122 | 13 / 9 | +0.5156 | +0.5006 | +0.2304 |
| P3 | PROJ_minus_A | +0.1326 [-0.1291, +0.4159] | -0.0773 | 10 / 11 | +0.3572 | +0.4787 | +0.1790 |
| P3 | PROJ_minus_MIX | -0.0436 [-0.1550, +0.0609] | -0.0894 | 1 / 6 | -0.1584 | -0.0219 | -0.0515 |
| P3 | MATCH_minus_A | -0.3589 [-0.9972, +0.3073] | -0.9431 | 21 / 22 | -0.0559 | -0.7148 | +0.1691 |
| P3 | MATCH_minus_ORIG_MATCH | -0.3589 [-0.9972, +0.3073] | -0.9431 | 21 / 22 | -0.0559 | -0.7148 | +0.1691 |
| P3 | MATCH_minus_MIRROR | -0.2897 [-0.7521, +0.0969] | -0.4203 | 7 / 6 | -0.5911 | -0.2354 | -0.5670 |
| P3 | MIRROR_minus_A | -0.0692 [-0.4874, +0.4358] | -0.5228 | 17 / 19 | +0.5352 | -0.4794 | +0.7361 |
| P3 | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P3 | MIX_minus_MATCH | +0.5351 [+0.1220, +0.9675] | +0.9552 | 15 / 10 | +0.5715 | +1.2154 | +0.0614 |
| P4 | MIX_minus_A | -0.0233 [-0.4386, +0.2661] | +0.1121 | 13 / 6 | -0.1172 | -0.0392 | -0.2597 |
| P4 | PROJ_minus_A | -0.0845 [-0.4948, +0.1894] | +0.0428 | 14 / 6 | -0.1977 | -0.0365 | -0.2284 |
| P4 | PROJ_minus_MIX | -0.0611 [-0.1169, -0.0010] | -0.0693 | 2 / 1 | -0.0804 | +0.0027 | +0.0313 |
| P4 | MATCH_minus_A | -4.2215 [-6.1883, -2.5952] | -4.2599 | 12 / 57 | -4.5524 | -2.2234 | -0.3196 |
| P4 | MATCH_minus_ORIG_MATCH | -4.2215 [-6.1883, -2.5952] | -4.2599 | 12 / 57 | -4.5524 | -2.2234 | -0.3196 |
| P4 | MATCH_minus_MIRROR | -3.9160 [-5.9038, -2.3086] | -3.9233 | 2 / 48 | -4.4221 | -1.8090 | -0.7749 |
| P4 | MIRROR_minus_A | -0.3055 [-0.6362, +0.0043] | -0.3366 | 16 / 15 | -0.1303 | -0.4145 | +0.4553 |
| P4 | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P4 | MIX_minus_MATCH | +4.1982 [+2.6810, +6.1265] | +4.3720 | 56 / 4 | +4.4351 | +2.1843 | +0.0599 |
| P5 | MIX_minus_A | +0.0920 [-0.0469, +0.2641] | +0.0438 | 3 / 5 | -0.3334 | +0.0485 | -0.6504 |
| P5 | PROJ_minus_A | +0.0357 [-0.1096, +0.2038] | -0.0000 | 3 / 3 | -0.4243 | +0.0235 | -0.6447 |
| P5 | PROJ_minus_MIX | -0.0563 [-0.1039, -0.0102] | -0.0438 | 3 / 1 | -0.0909 | -0.0249 | +0.0057 |
| P5 | MATCH_minus_A | -2.7748 [-4.0525, -1.6512] | -2.8626 | 2 / 31 | -3.6932 | -0.4326 | -1.1909 |
| P5 | MATCH_minus_ORIG_MATCH | -2.7748 [-4.0525, -1.6512] | -2.8626 | 2 / 31 | -3.6932 | -0.4326 | -1.1909 |
| P5 | MATCH_minus_MIRROR | -2.2256 [-3.5935, -1.0253] | -2.3249 | 6 / 26 | -2.9000 | -0.3176 | -0.8462 |
| P5 | MIRROR_minus_A | -0.5492 [-1.1152, -0.1191] | -0.5377 | 2 / 11 | -0.7931 | -0.1150 | -0.3447 |
| P5 | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P5 | MIX_minus_MATCH | +2.8668 [+1.7603, +4.1053] | +2.9063 | 29 / 2 | +3.3597 | +0.4811 | +0.5405 |
| small | MIX_minus_A | +0.0929 [-0.1870, +0.3630] | +0.0383 | 19 / 11 | +0.3274 | +0.2803 | +0.4135 |
| small | PROJ_minus_A | +0.0894 [-0.1754, +0.3470] | -0.0313 | 17 / 12 | +0.2502 | +0.3009 | +0.3446 |
| small | PROJ_minus_MIX | -0.0035 [-0.0998, +0.0786] | -0.0696 | 3 / 6 | -0.0772 | +0.0206 | -0.0689 |
| small | MATCH_minus_A | -1.0216 [-1.7210, -0.3595] | -1.2996 | 24 / 29 | -0.7372 | -1.1568 | +1.0181 |
| small | MATCH_minus_ORIG_MATCH | -1.0216 [-1.7210, -0.3595] | -1.2996 | 24 / 29 | -0.7372 | -1.1568 | +1.0181 |
| small | MATCH_minus_MIRROR | -0.4954 [-0.9835, -0.0357] | -0.6698 | 8 / 9 | -0.7546 | -0.2041 | -0.2624 |
| small | MIRROR_minus_A | -0.5262 [-1.1913, +0.0184] | -0.6298 | 22 / 26 | +0.0174 | -0.9527 | +1.2805 |
| small | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| small | MIX_minus_MATCH | +1.1145 [+0.5908, +1.7182] | +1.3380 | 22 / 9 | +1.0646 | +1.4372 | -0.6046 |
| medium | MIX_minus_A | -0.0204 [-0.4801, +0.2578] | +0.0757 | 8 / 6 | -0.2575 | +0.0513 | -0.4913 |
| medium | PROJ_minus_A | -0.1122 [-0.5509, +0.1590] | +0.0011 | 9 / 5 | -0.3832 | +0.0129 | -0.4811 |
| medium | PROJ_minus_MIX | -0.0918 [-0.1522, -0.0321] | -0.0746 | 3 / 1 | -0.1257 | -0.0384 | +0.0102 |
| medium | MATCH_minus_A | -4.3564 [-6.2775, -2.8186] | -4.3238 | 10 / 54 | -5.0084 | -1.7347 | -0.9857 |
| medium | MATCH_minus_ORIG_MATCH | -4.3564 [-6.2775, -2.8186] | -4.3238 | 10 / 54 | -5.0084 | -1.7347 | -0.9857 |
| medium | MATCH_minus_MIRROR | -4.2783 [-6.1799, -2.6887] | -4.1700 | 2 / 46 | -4.8812 | -1.6819 | -0.9408 |
| medium | MIRROR_minus_A | -0.0781 [-0.3293, +0.1598] | -0.1538 | 12 / 12 | -0.1272 | -0.0528 | -0.0449 |
| medium | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| medium | MIX_minus_MATCH | +4.3360 [+2.8473, +6.1629] | +4.3995 | 52 / 6 | +4.7508 | +1.7860 | +0.4944 |
| large | MIX_minus_A | +0.0974 [-0.0738, +0.2768] | +0.0722 | 2 / 3 | -0.3070 | +0.0180 | -0.6387 |
| large | PROJ_minus_A | +0.0472 [-0.1226, +0.2165] | +0.0163 | 1 / 3 | -0.3640 | +0.0009 | -0.6223 |
| large | PROJ_minus_MIX | -0.0502 [-0.0920, -0.0130] | -0.0560 | 0 / 1 | -0.0570 | -0.0171 | +0.0164 |
| large | MATCH_minus_A | -2.8579 [-4.3580, -1.5941] | -3.1354 | 1 / 27 | -3.7178 | -0.5032 | -1.2012 |
| large | MATCH_minus_ORIG_MATCH | -2.8579 [-4.3580, -1.5941] | -3.1354 | 1 / 27 | -3.7178 | -0.5032 | -1.2012 |
| large | MATCH_minus_MIRROR | -2.2155 [-3.7883, -0.8310] | -2.5528 | 5 / 25 | -2.8761 | -0.3408 | -0.8457 |
| large | MIRROR_minus_A | -0.6423 [-1.3349, -0.1403] | -0.5826 | 1 / 7 | -0.8418 | -0.1623 | -0.3556 |
| large | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| large | MIX_minus_MATCH | +2.9552 [+1.7497, +4.4340] | +3.2076 | 26 / 1 | +3.4108 | +0.5212 | +0.5625 |

The paired intervals are exploratory across multiple arms/metrics. Original failure-group damage is structurally zero versus A; it does not imply no continuous degradation.
No candidate is dropped after observing a mask. Undefined AUC/FPR remains in IoU. The projection target, image, class and box contain no GT; evaluation membership itself is GT-conditioned.
Only reliable useful effects warrant a separately registered confirmation. A projection advantage requires PROJ versus MIX, not merely PROJ versus A.
Use AUDIT.json and SOURCE_ROWS.jsonl to assess replay, source correspondence, normal equations and projection error.
