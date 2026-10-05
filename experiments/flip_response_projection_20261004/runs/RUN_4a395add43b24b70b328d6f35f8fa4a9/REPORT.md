# Whole-canvas reflection and original-basis projection

Frozen original YOLO26m-seg, no training. All effects are percentage points, normal full-mask decoding.
2 planned dev images, 2 effective images, 7 candidates.

MIRROR maps anchor locations, not guaranteed instance identity. MATCH uses same level/predicted class and inverse-box overlap. MIX is ordinary equal logit averaging. PROJ solves only a coefficient increment in original full P.

| Group | Pair | Image macro IoU Δ [95% CI] | Candidate IoU Δ | Repair / damage | Coverage Δ | AUC Δ | FPR Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| all | MIX_minus_A | +0.8278 [+0.0876, +1.5680] | +0.5106 | 0 / 0 | +0.6945 | +0.1378 | -0.3755 |
| all | PROJ_minus_A | +0.7139 [+0.1028, +1.3250] | +0.4520 | 0 / 0 | +0.4246 | +0.0507 | -0.5841 |
| all | PROJ_minus_MIX | -0.1139 [-0.2430, +0.0152] | -0.0586 | 0 / 0 | -0.2699 | -0.0871 | -0.2086 |
| all | MATCH_minus_A | +0.7279 [-0.0792, +1.5350] | +0.3820 | 0 / 0 | +0.6524 | -0.0426 | -0.3279 |
| all | MATCH_minus_ORIG_MATCH | +0.7279 [-0.0792, +1.5350] | +0.3820 | 0 / 0 | +0.6524 | -0.0426 | -0.3279 |
| all | MATCH_minus_MIRROR | +0.4479 [+0.0000, +0.8958] | +0.6398 | 0 / 0 | -0.1475 | +0.0344 | -0.6810 |
| all | MIRROR_minus_A | +0.2800 [-0.9749, +1.5350] | -0.2578 | 0 / 0 | +0.8000 | -0.0770 | +0.3531 |
| all | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| all | MIX_minus_MATCH | +0.0999 [+0.0330, +0.1668] | +0.1286 | 0 / 0 | +0.0420 | +0.1804 | -0.0476 |
| box_good_mask_bad | MIX_minus_A | +1.2869 [-0.0494, +2.6232] | +1.2869 | 0 / 0 | +1.3024 | +0.1060 | -0.5906 |
| box_good_mask_bad | PROJ_minus_A | +0.5620 [-0.4552, +1.5793] | +0.5620 | 0 / 0 | +0.7052 | +0.0338 | -0.2531 |
| box_good_mask_bad | PROJ_minus_MIX | -0.7248 [-1.0439, -0.4058] | -0.7248 | 0 / 0 | -0.5972 | -0.0722 | +0.3375 |
| box_good_mask_bad | MATCH_minus_A | +0.9212 [-0.9894, +2.8319] | +0.9212 | 0 / 0 | +1.6288 | -0.3552 | -0.0879 |
| box_good_mask_bad | MATCH_minus_ORIG_MATCH | +0.9212 [-0.9894, +2.8319] | +0.9212 | 0 / 0 | +1.6288 | -0.3552 | -0.0879 |
| box_good_mask_bad | MATCH_minus_MIRROR | +1.9008 [+0.0000, +3.8016] | +1.9008 | 0 / 0 | -0.6242 | +0.1893 | -2.5616 |
| box_good_mask_bad | MIRROR_minus_A | -0.9796 [-4.7910, +2.8319] | -0.9796 | 0 / 0 | +2.2530 | -0.5445 | +2.4738 |
| box_good_mask_bad | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| box_good_mask_bad | MIX_minus_MATCH | +0.3657 [-0.2087, +0.9400] | +0.3657 | 0 / 0 | -0.3264 | +0.4612 | -0.5027 |
| original_success | MIX_minus_A | +0.1219 [+0.1219, +0.1219] | +0.1219 | 0 / 0 | -0.1154 | +0.0162 | -0.4558 |
| original_success | PROJ_minus_A | +0.2422 [+0.2422, +0.2422] | +0.2422 | 0 / 0 | -0.1020 | -0.0025 | -0.6332 |
| original_success | PROJ_minus_MIX | +0.1204 [+0.1204, +0.1204] | +0.1204 | 0 / 0 | +0.0135 | -0.0186 | -0.1774 |
| original_success | MATCH_minus_A | +0.1484 [+0.1484, +0.1484] | +0.1484 | 0 / 0 | -0.2863 | -0.0461 | -0.8152 |
| original_success | MATCH_minus_ORIG_MATCH | +0.1484 [+0.1484, +0.1484] | +0.1484 | 0 / 0 | -0.2863 | -0.0461 | -0.8152 |
| original_success | MATCH_minus_MIRROR | +0.1693 [+0.1693, +0.1693] | +0.1693 | 0 / 0 | -0.0567 | -0.0086 | -0.4216 |
| original_success | MIRROR_minus_A | -0.0209 [-0.0209, -0.0209] | -0.0209 | 0 / 0 | -0.2296 | -0.0375 | -0.3935 |
| original_success | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| original_success | MIX_minus_MATCH | -0.0265 [-0.0265, -0.0265] | -0.0265 | 0 / 0 | +0.1709 | +0.0623 | +0.3593 |
| original_failure | MIX_minus_A | +0.7593 [-0.0494, +1.5680] | +1.0289 | 0 / 0 | +1.0034 | -0.0298 | -0.0026 |
| original_failure | PROJ_minus_A | +0.4349 [-0.4552, +1.3250] | +0.7316 | 0 / 0 | +0.6659 | -0.0388 | -0.0177 |
| original_failure | PROJ_minus_MIX | -0.3244 [-0.4058, -0.2430] | -0.2972 | 0 / 0 | -0.3375 | -0.0089 | -0.0150 |
| original_failure | MATCH_minus_A | +0.2728 [-0.9894, +1.5350] | +0.6935 | 0 / 0 | +1.3947 | -0.4900 | +0.8119 |
| original_failure | MATCH_minus_ORIG_MATCH | +0.2728 [-0.9894, +1.5350] | +0.6935 | 0 / 0 | +1.3947 | -0.4900 | +0.8119 |
| original_failure | MATCH_minus_MIRROR | +1.9008 [+0.0000, +3.8016] | +1.2672 | 0 / 0 | -0.6242 | +0.1893 | -2.5616 |
| original_failure | MIRROR_minus_A | -1.6280 [-4.7910, +1.5350] | -0.5737 | 0 / 0 | +2.0189 | -0.6793 | +3.3735 |
| original_failure | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| original_failure | MIX_minus_MATCH | +0.4865 [+0.0330, +0.9400] | +0.3353 | 0 / 0 | -0.3913 | +0.4601 | -0.8145 |
| P3 | MIX_minus_A | +0.5128 [+0.5128, +0.5128] | +0.5128 | 0 / 0 | +0.7519 | +0.0715 | +0.6944 |
| P3 | PROJ_minus_A | +1.0708 [+1.0708, +1.0708] | +1.0708 | 0 / 0 | +0.7519 | +0.0034 | -0.3472 |
| P3 | PROJ_minus_MIX | +0.5580 [+0.5580, +0.5580] | +0.5580 | 0 / 0 | +0.0000 | -0.0681 | -1.0417 |
| P3 | MATCH_minus_A | +0.2381 [+0.2381, +0.2381] | +0.2381 | 0 / 0 | +0.7519 | -0.0851 | +1.3889 |
| P3 | MATCH_minus_ORIG_MATCH | +0.2381 [+0.2381, +0.2381] | +0.2381 | 0 / 0 | +0.7519 | -0.0851 | +1.3889 |
| P3 | MATCH_minus_MIRROR | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P3 | MIRROR_minus_A | +0.2381 [+0.2381, +0.2381] | +0.2381 | 0 / 0 | +0.7519 | -0.0851 | +1.3889 |
| P3 | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P3 | MIX_minus_MATCH | +0.2747 [+0.2747, +0.2747] | +0.2747 | 0 / 0 | +0.0000 | +0.1566 | -0.6944 |
| P4 | MIX_minus_A | +1.5032 [+0.3833, +2.6232] | +1.1299 | 0 / 0 | +0.8629 | +0.3035 | -1.3973 |
| P4 | PROJ_minus_A | +1.0348 [+0.4903, +1.5793] | +0.8533 | 0 / 0 | +0.3065 | +0.1391 | -1.3805 |
| P4 | PROJ_minus_MIX | -0.4684 [-1.0439, +0.1070] | -0.2766 | 0 / 0 | -0.5565 | -0.1644 | +0.0168 |
| P4 | MATCH_minus_A | +1.7093 [+0.5867, +2.8319] | +1.3351 | 0 / 0 | +0.5648 | +0.1903 | -2.1658 |
| P4 | MATCH_minus_ORIG_MATCH | +1.7093 [+0.5867, +2.8319] | +1.3351 | 0 / 0 | +0.5648 | +0.1903 | -2.1658 |
| P4 | MATCH_minus_MIRROR | +0.1587 [+0.0000, +0.3174] | +0.2116 | 0 / 0 | -0.0504 | -0.0061 | -0.3825 |
| P4 | MIRROR_minus_A | +1.5506 [+0.2693, +2.8319] | +1.1235 | 0 / 0 | +0.6153 | +0.1964 | -1.7833 |
| P4 | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P4 | MIX_minus_MATCH | -0.2061 [-0.2087, -0.2034] | -0.2052 | 0 / 0 | +0.2981 | +0.1132 | +0.7685 |
| P5 | MIX_minus_A | -0.1095 [-0.1095, -0.1095] | -0.1095 | 0 / 0 | +0.2132 | -0.1074 | +0.3090 |
| P5 | PROJ_minus_A | -0.1556 [-0.1556, -0.1556] | -0.1556 | 0 / 0 | +0.2286 | -0.0683 | +0.3979 |
| P5 | PROJ_minus_MIX | -0.0461 [-0.0461, -0.0461] | -0.0461 | 0 / 0 | +0.0155 | +0.0391 | +0.0889 |
| P5 | MATCH_minus_A | -0.5231 [-0.5231, -0.5231] | -0.5231 | 0 / 0 | +0.5137 | -0.4006 | +1.0057 |
| P5 | MATCH_minus_ORIG_MATCH | -0.5231 [-0.5231, -0.5231] | -0.5231 | 0 / 0 | +0.5137 | -0.4006 | +1.0057 |
| P5 | MATCH_minus_MIRROR | +1.2813 [+1.2813, +1.2813] | +1.2813 | 0 / 0 | -0.4245 | +0.1228 | -1.7598 |
| P5 | MIRROR_minus_A | -1.8044 [-1.8044, -1.8044] | -1.8044 | 0 / 0 | +0.9382 | -0.5234 | +2.7655 |
| P5 | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P5 | MIX_minus_MATCH | +0.4136 [+0.4136, +0.4136] | +0.4136 | 0 / 0 | -0.3006 | +0.2933 | -0.6966 |
| small | MIX_minus_A | +1.5680 [+1.5680, +1.5680] | +1.5680 | 0 / 0 | +1.3500 | +0.3433 | -0.4815 |
| small | PROJ_minus_A | +1.3250 [+1.3250, +1.3250] | +1.3250 | 0 / 0 | +0.8305 | +0.1485 | -0.8182 |
| small | PROJ_minus_MIX | -0.2430 [-0.2430, -0.2430] | -0.2430 | 0 / 0 | -0.5195 | -0.1947 | -0.3367 |
| small | MATCH_minus_A | +1.5350 [+1.5350, +1.5350] | +1.5350 | 0 / 0 | +1.2201 | +0.1845 | -0.4105 |
| small | MATCH_minus_ORIG_MATCH | +1.5350 [+1.5350, +1.5350] | +1.5350 | 0 / 0 | +1.2201 | +0.1845 | -0.4105 |
| small | MATCH_minus_MIRROR | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| small | MIRROR_minus_A | +1.5350 [+1.5350, +1.5350] | +1.5350 | 0 / 0 | +1.2201 | +0.1845 | -0.4105 |
| small | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| small | MIX_minus_MATCH | +0.0330 [+0.0330, +0.0330] | +0.0330 | 0 / 0 | +0.1299 | +0.1587 | -0.0710 |
| medium | MIX_minus_A | +0.3833 [+0.3833, +0.3833] | +0.3833 | 0 / 0 | -0.2222 | -0.0081 | -1.1371 |
| medium | PROJ_minus_A | +0.4903 [+0.4903, +0.4903] | +0.4903 | 0 / 0 | -0.2962 | -0.0155 | -1.4719 |
| medium | PROJ_minus_MIX | +0.1070 [+0.1070, +0.1070] | +0.1070 | 0 / 0 | -0.0740 | -0.0074 | -0.3348 |
| medium | MATCH_minus_A | +0.5867 [+0.5867, +0.5867] | +0.5867 | 0 / 0 | -0.5586 | -0.0736 | -2.1217 |
| medium | MATCH_minus_ORIG_MATCH | +0.5867 [+0.5867, +0.5867] | +0.5867 | 0 / 0 | -0.5586 | -0.0736 | -2.1217 |
| medium | MATCH_minus_MIRROR | +0.3174 [+0.3174, +0.3174] | +0.3174 | 0 / 0 | -0.1009 | -0.0122 | -0.7651 |
| medium | MIRROR_minus_A | +0.2693 [+0.2693, +0.2693] | +0.2693 | 0 / 0 | -0.4577 | -0.0614 | -1.3566 |
| medium | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| medium | MIX_minus_MATCH | -0.2034 [-0.2034, -0.2034] | -0.2034 | 0 / 0 | +0.3364 | +0.0655 | +0.9846 |
| large | MIX_minus_A | -0.1095 [-0.1095, -0.1095] | -0.1095 | 0 / 0 | +0.2132 | -0.1074 | +0.3090 |
| large | PROJ_minus_A | -0.1556 [-0.1556, -0.1556] | -0.1556 | 0 / 0 | +0.2286 | -0.0683 | +0.3979 |
| large | PROJ_minus_MIX | -0.0461 [-0.0461, -0.0461] | -0.0461 | 0 / 0 | +0.0155 | +0.0391 | +0.0889 |
| large | MATCH_minus_A | -0.5231 [-0.5231, -0.5231] | -0.5231 | 0 / 0 | +0.5137 | -0.4006 | +1.0057 |
| large | MATCH_minus_ORIG_MATCH | -0.5231 [-0.5231, -0.5231] | -0.5231 | 0 / 0 | +0.5137 | -0.4006 | +1.0057 |
| large | MATCH_minus_MIRROR | +1.2813 [+1.2813, +1.2813] | +1.2813 | 0 / 0 | -0.4245 | +0.1228 | -1.7598 |
| large | MIRROR_minus_A | -1.8044 [-1.8044, -1.8044] | -1.8044 | 0 / 0 | +0.9382 | -0.5234 | +2.7655 |
| large | ORIG_MATCH_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| large | MIX_minus_MATCH | +0.4136 [+0.4136, +0.4136] | +0.4136 | 0 / 0 | -0.3006 | +0.2933 | -0.6966 |

The paired intervals are exploratory across multiple arms/metrics. Original failure-group damage is structurally zero versus A; it does not imply no continuous degradation.
No candidate is dropped after observing a mask. Undefined AUC/FPR remains in IoU. The projection target, image, class and box contain no GT; evaluation membership itself is GT-conditioned.
Only reliable useful effects warrant a separately registered confirmation. A projection advantage requires PROJ versus MIX, not merely PROJ versus A.
Use AUDIT.json and SOURCE_ROWS.jsonl to assess replay, source correspondence, normal equations and projection error.
