# Native semantic response and original-basis projection

Frozen original YOLO26m-seg. No training or parameter selection; all reported mask effects are percentage points.
256 planned dev images, 253 effective images, 1816 fixed official candidates.

A is the original mask. SEM directly decodes the predicted-class semantic logits; MIX averages original and semantic logits equally. PROJ applies the fixed full-support ridge solver to half their logit difference. ROT controls rotate only the semantic response inside the prediction box. AGN_PROJ substitutes the maximum class logit.

| Group | Pair | Image macro IoU delta [95% CI] | Candidate IoU delta | Repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---|---:|---:|---:|---:|---:|---:|
| all | PROJ_minus_A | -29.2864 [-32.0156, -26.7988] | -32.9919 | 12 / 620 | -27.9329 | -1.3603 | +5.3856 |
| all | PROJ_minus_MIX | +0.2454 [+0.1215, +0.3785] | +0.5415 | 30 / 10 | +0.4434 | +0.1037 | +0.5305 |
| all | PROJ_minus_ROT_PROJ | +3.6364 [+3.0152, +4.2794] | +2.6897 | 125 / 33 | +2.3860 | +1.3751 | -3.9387 |
| all | MIX_minus_A | -29.5318 [-32.2135, -27.0183] | -33.5334 | 12 / 640 | -28.3763 | -1.4641 | +4.8550 |
| all | MIX_minus_ROT_MIX | +5.1070 [+4.3768, +5.8630] | +3.6501 | 181 / 26 | +3.3546 | +2.7588 | -5.5078 |
| all | PROJ_minus_AGN_PROJ | -10.0231 [-11.7991, -8.2270] | -14.7562 | 42 / 145 | -15.1420 | +3.7003 | -10.4474 |
| all | SEM_minus_A | -37.3294 [-39.8542, -34.8163] | -39.9525 | 3 / 818 | -30.6911 | -22.8994 | +21.7997 |
| box_good_mask_bad | PROJ_minus_A | -29.3146 [-33.5784, -25.0740] | -29.3180 | 10 / 0 | -35.0353 | -2.8832 | -5.9839 |
| box_good_mask_bad | PROJ_minus_MIX | +0.2056 [-0.1395, +0.5576] | +0.3435 | 1 / 2 | +0.3068 | -0.4959 | +0.1062 |
| box_good_mask_bad | PROJ_minus_ROT_PROJ | +3.9719 [+2.4819, +5.6591] | +3.1870 | 4 / 4 | +4.8867 | +2.4958 | +0.0218 |
| box_good_mask_bad | MIX_minus_A | -29.5202 [-33.8026, -25.2573] | -29.6615 | 11 / 0 | -35.3422 | -2.3873 | -6.0901 |
| box_good_mask_bad | MIX_minus_ROT_MIX | +4.2665 [+2.9666, +5.7498] | +3.3575 | 7 / 2 | +4.7966 | +3.6534 | -1.4955 |
| box_good_mask_bad | PROJ_minus_AGN_PROJ | -12.9812 [-16.3203, -9.2022] | -15.0640 | 1 / 18 | -22.1168 | +1.2409 | -15.2248 |
| box_good_mask_bad | SEM_minus_A | -34.8474 [-38.7188, -31.0370] | -35.2201 | 3 / 0 | -38.8378 | -26.7724 | +1.0495 |
| original_success | PROJ_minus_A | -32.0529 [-35.5602, -28.9267] | -35.4539 | 0 / 620 | -28.8055 | -0.9372 | +8.1521 |
| original_success | PROJ_minus_MIX | +0.3149 [+0.1761, +0.4694] | +0.7127 | 28 / 8 | +0.4967 | +0.2634 | +0.5626 |
| original_success | PROJ_minus_ROT_PROJ | +3.8187 [+3.1163, +4.7202] | +3.1907 | 120 / 27 | +2.1464 | +1.3607 | -4.6502 |
| original_success | MIX_minus_A | -32.3679 [-35.8849, -29.2675] | -36.1666 | 0 / 640 | -29.3022 | -1.2007 | +7.5895 |
| original_success | MIX_minus_ROT_MIX | +5.6629 [+4.8227, +6.6454] | +4.6434 | 174 / 23 | +3.4237 | +2.9002 | -6.3955 |
| original_success | PROJ_minus_AGN_PROJ | -10.9656 [-13.1622, -8.6543] | -14.1836 | 41 / 123 | -14.5663 | +4.6867 | -8.4972 |
| original_success | SEM_minus_A | -40.5355 [-43.6222, -37.5823] | -43.5073 | 0 / 818 | -31.4941 | -23.2924 | +25.4664 |
| original_failure | PROJ_minus_A | -27.4133 [-30.9634, -23.9900] | -28.8596 | 12 / 0 | -34.4498 | -2.8612 | -7.6567 |
| original_failure | PROJ_minus_MIX | +0.1614 [-0.0627, +0.4173] | +0.2541 | 2 / 2 | +0.2972 | -0.4903 | +0.2757 |
| original_failure | PROJ_minus_ROT_PROJ | +3.0243 [+1.7622, +4.3746] | +1.8487 | 5 / 6 | +3.3976 | +1.9104 | -0.5135 |
| original_failure | MIX_minus_A | -27.5747 [-31.1383, -24.1365] | -29.1137 | 12 / 0 | -34.7469 | -2.3710 | -7.9324 |
| original_failure | MIX_minus_ROT_MIX | +3.2089 [+2.1504, +4.4582] | +1.9828 | 7 / 3 | +3.3300 | +2.7846 | -1.6746 |
| original_failure | PROJ_minus_AGN_PROJ | -12.6685 [-15.5578, -9.9452] | -15.7172 | 1 / 22 | -22.0044 | +1.2496 | -16.7065 |
| original_failure | SEM_minus_A | -32.5599 [-35.8252, -29.2965] | -33.9857 | 3 / 0 | -38.1552 | -25.7031 | -0.8855 |
| predclass_correct | PROJ_minus_A | -28.5347 [-31.2158, -26.0432] | -32.6169 | 12 / 593 | -26.9273 | -1.3083 | +5.8813 |
| predclass_correct | PROJ_minus_MIX | +0.2630 [+0.1265, +0.4013] | +0.5444 | 29 / 9 | +0.4594 | +0.1184 | +0.5197 |
| predclass_correct | PROJ_minus_ROT_PROJ | +3.7185 [+3.0803, +4.3754] | +2.7846 | 121 / 31 | +2.4365 | +1.3864 | -4.0400 |
| predclass_correct | MIX_minus_A | -28.7976 [-31.5112, -26.2788] | -33.1613 | 12 / 613 | -27.3866 | -1.4267 | +5.3616 |
| predclass_correct | MIX_minus_ROT_MIX | +5.2372 [+4.4858, +6.0042] | +3.7861 | 178 / 24 | +3.4366 | +2.8041 | -5.6760 |
| predclass_correct | PROJ_minus_AGN_PROJ | -9.3635 [-11.1439, -7.5320] | -14.2659 | 42 / 132 | -14.2607 | +3.7834 | -10.1187 |
| predclass_correct | SEM_minus_A | -36.6442 [-39.2226, -34.1470] | -39.6294 | 3 / 786 | -29.6962 | -22.8061 | +22.4366 |
| predclass_incorrect | PROJ_minus_A | -43.7735 [-52.3270, -34.9863] | -40.5355 | 0 / 27 | -56.0744 | -2.4165 | -20.6861 |
| predclass_incorrect | PROJ_minus_MIX | +0.6090 [+0.0305, +1.2739] | +0.4827 | 1 / 1 | +0.6598 | -0.0461 | +0.2272 |
| predclass_incorrect | PROJ_minus_ROT_PROJ | +1.4143 [-0.5074, +3.5053] | +0.7793 | 4 / 2 | +1.1815 | +2.3782 | -0.5478 |
| predclass_incorrect | MIX_minus_A | -44.3824 [-52.9042, -35.6535] | -41.0182 | 0 / 27 | -56.7342 | -2.3704 | -20.9133 |
| predclass_incorrect | MIX_minus_ROT_MIX | +1.5904 [-0.6078, +3.9314] | +0.9135 | 3 / 2 | +1.4604 | +2.5960 | -0.2927 |
| predclass_incorrect | PROJ_minus_AGN_PROJ | -26.3268 [-34.2564, -18.5807] | -24.6190 | 0 / 13 | -39.4537 | +0.0326 | -24.2621 |
| predclass_incorrect | SEM_minus_A | -49.8422 [-57.1780, -42.0671] | -46.4518 | 0 / 32 | -61.9107 | -23.3887 | -17.5395 |
| sameclass_neighbor | PROJ_minus_A | -36.3419 [-41.6221, -31.1787] | -38.1060 | 5 / 314 | -36.1392 | -2.2787 | +4.0750 |
| sameclass_neighbor | PROJ_minus_MIX | +0.3847 [+0.1934, +0.6144] | +0.4438 | 10 / 3 | +0.3994 | -0.2547 | +0.0930 |
| sameclass_neighbor | PROJ_minus_ROT_PROJ | +2.0708 [+1.3150, +2.9848] | +1.3255 | 39 / 12 | +1.8006 | +0.8326 | -1.7288 |
| sameclass_neighbor | MIX_minus_A | -36.7266 [-41.9507, -31.7300] | -38.5498 | 5 / 321 | -36.5386 | -2.0240 | +3.9821 |
| sameclass_neighbor | MIX_minus_ROT_MIX | +3.0298 [+2.1440, +4.0625] | +1.9129 | 52 / 9 | +2.5328 | +1.7690 | -2.6830 |
| sameclass_neighbor | PROJ_minus_AGN_PROJ | -11.4358 [-14.8462, -8.0907] | -14.1981 | 17 / 47 | -15.9915 | +2.5411 | -7.9541 |
| sameclass_neighbor | SEM_minus_A | -43.2083 [-47.6111, -38.8603] | -44.4082 | 1 / 393 | -39.2219 | -31.5771 | +17.6192 |
| no_sameclass_neighbor | PROJ_minus_A | -28.1903 [-31.1439, -25.4702] | -29.1931 | 7 / 306 | -28.8672 | -1.2642 | +1.9542 |
| no_sameclass_neighbor | PROJ_minus_MIX | +0.3377 [+0.1760, +0.5144] | +0.6140 | 20 / 7 | +0.5297 | +0.2309 | +0.6125 |
| no_sameclass_neighbor | PROJ_minus_ROT_PROJ | +4.2370 [+3.4200, +5.0730] | +3.7029 | 86 / 21 | +2.8928 | +1.6839 | -3.9846 |
| no_sameclass_neighbor | MIX_minus_A | -28.5280 [-31.4464, -25.8754] | -29.8072 | 7 / 319 | -29.3969 | -1.4951 | +1.3416 |
| no_sameclass_neighbor | MIX_minus_ROT_MIX | +5.9958 [+5.0469, +6.9040] | +4.9404 | 129 / 17 | +4.0224 | +3.2598 | -6.0255 |
| no_sameclass_neighbor | PROJ_minus_AGN_PROJ | -11.5072 [-13.6068, -9.3400] | -15.1707 | 25 / 98 | -17.6249 | +4.0396 | -12.6314 |
| no_sameclass_neighbor | SEM_minus_A | -36.0101 [-38.7104, -33.2872] | -36.6427 | 2 / 425 | -32.3990 | -20.1904 | +14.9737 |
| P3 | PROJ_minus_A | -31.0777 [-35.7627, -26.8798] | -29.6752 | 3 / 128 | -37.3908 | -3.3893 | -6.9734 |
| P3 | PROJ_minus_MIX | +0.1918 [-0.0364, +0.4533] | +0.3523 | 8 / 4 | +0.4825 | -0.0010 | +0.6431 |
| P3 | PROJ_minus_ROT_PROJ | -0.4222 [-1.1072, +0.1583] | -0.3747 | 14 / 16 | -0.3601 | -0.5469 | +0.9550 |
| P3 | MIX_minus_A | -31.2695 [-35.8677, -27.0621] | -30.0274 | 2 / 131 | -37.8733 | -3.3883 | -7.6164 |
| P3 | MIX_minus_ROT_MIX | -0.2527 [-0.9587, +0.3675] | -0.3836 | 16 / 15 | -0.0633 | -0.5798 | +0.9933 |
| P3 | PROJ_minus_AGN_PROJ | -15.6650 [-19.6379, -11.7812] | -15.6769 | 8 / 32 | -23.1523 | +0.5812 | -15.4469 |
| P3 | SEM_minus_A | -37.8453 [-42.4077, -33.9813] | -37.7486 | 0 / 177 | -43.7154 | -31.3310 | -1.5684 |
| P4 | PROJ_minus_A | -32.1669 [-36.2931, -28.0641] | -35.0964 | 6 / 271 | -31.3589 | -1.3946 | +5.5963 |
| P4 | PROJ_minus_MIX | +0.6432 [+0.3578, +0.9342] | +0.8297 | 16 / 4 | +0.8444 | +0.4172 | +0.6227 |
| P4 | PROJ_minus_ROT_PROJ | +3.4107 [+2.4417, +4.5743] | +3.0140 | 53 / 11 | +2.4093 | +2.1734 | -3.3060 |
| P4 | MIX_minus_A | -32.8102 [-36.8958, -28.8465] | -35.9261 | 6 / 283 | -32.2033 | -1.8118 | +4.9737 |
| P4 | MIX_minus_ROT_MIX | +4.1063 [+3.1219, +5.2014] | +3.6963 | 69 / 7 | +2.9886 | +3.2389 | -4.1242 |
| P4 | PROJ_minus_AGN_PROJ | -14.2976 [-17.9987, -10.7600] | -18.0030 | 22 / 71 | -20.4118 | +1.2069 | -12.3141 |
| P4 | SEM_minus_A | -39.9750 [-43.8677, -36.0746] | -41.5973 | 2 / 359 | -35.0281 | -24.4662 | +18.5184 |
| P5 | PROJ_minus_A | -31.3392 [-35.4994, -27.5231] | -33.8660 | 3 / 221 | -29.4839 | -0.6487 | +4.6025 |
| P5 | PROJ_minus_MIX | +0.2990 [+0.1368, +0.4743] | +0.3848 | 6 / 2 | +0.3222 | -0.0483 | +0.2060 |
| P5 | PROJ_minus_ROT_PROJ | +5.1252 [+4.2325, +6.0662] | +5.4880 | 58 / 6 | +3.5992 | +1.9478 | -5.1482 |
| P5 | MIX_minus_A | -31.6382 [-35.7098, -27.8301] | -34.2508 | 4 / 226 | -29.8061 | -0.6004 | +4.3965 |
| P5 | MIX_minus_ROT_MIX | +7.9299 [+6.8784, +9.1289] | +7.8014 | 96 / 4 | +5.5706 | +4.3278 | -8.3093 |
| P5 | PROJ_minus_AGN_PROJ | -8.1279 [-10.2994, -5.9594] | -9.8057 | 12 / 42 | -12.1290 | +6.7994 | -8.2763 |
| P5 | SEM_minus_A | -38.5568 [-42.4426, -34.7816] | -40.2304 | 1 / 282 | -31.4543 | -22.4538 | +19.6745 |
| small | PROJ_minus_A | -31.3041 [-35.3692, -27.1773] | -31.0022 | 4 / 176 | -36.3424 | -2.8471 | -4.1933 |
| small | PROJ_minus_MIX | +0.2389 [+0.0597, +0.4105] | +0.4000 | 12 / 3 | +0.5238 | +0.1436 | +0.6553 |
| small | PROJ_minus_ROT_PROJ | +0.6167 [+0.1209, +1.1741] | +0.2207 | 22 / 19 | +0.5979 | +0.3000 | -0.1245 |
| small | MIX_minus_A | -31.5430 [-35.5583, -27.5093] | -31.4022 | 3 / 184 | -36.8662 | -2.9907 | -4.8486 |
| small | MIX_minus_ROT_MIX | +0.7947 [+0.2228, +1.4587] | +0.2174 | 21 / 17 | +0.8177 | +0.4298 | -0.2066 |
| small | PROJ_minus_AGN_PROJ | -14.8142 [-18.1306, -11.5163] | -16.5572 | 9 / 42 | -22.5881 | +0.9897 | -14.7521 |
| small | SEM_minus_A | -37.9240 [-41.7958, -34.1456] | -38.8273 | 0 / 244 | -41.6174 | -30.2604 | +3.0712 |
| medium | PROJ_minus_A | -30.5196 [-35.1939, -26.2401] | -35.0143 | 6 / 264 | -28.3908 | -1.2902 | +6.7336 |
| medium | PROJ_minus_MIX | +0.7914 [+0.4501, +1.1463] | +0.9056 | 16 / 5 | +0.8852 | +0.3464 | +0.5264 |
| medium | PROJ_minus_ROT_PROJ | +4.4627 [+3.1829, +5.9207] | +3.8560 | 64 / 9 | +3.4022 | +2.1797 | -3.4434 |
| medium | MIX_minus_A | -31.3110 [-35.9624, -27.1422] | -35.9200 | 6 / 275 | -29.2759 | -1.6365 | +6.2071 |
| medium | MIX_minus_ROT_MIX | +5.6819 [+4.3822, +7.1023] | +4.9974 | 83 / 6 | +4.4964 | +3.7045 | -4.6752 |
| medium | PROJ_minus_AGN_PROJ | -13.2486 [-16.7301, -9.8798] | -16.4645 | 24 / 63 | -18.7594 | +1.3002 | -10.4733 |
| medium | SEM_minus_A | -39.2828 [-43.3595, -35.4349] | -41.6844 | 2 / 347 | -32.1421 | -23.7457 | +20.7230 |
| large | PROJ_minus_A | -32.6205 [-37.0240, -28.1572] | -33.5663 | 2 / 180 | -30.9059 | -0.4780 | +4.2641 |
| large | PROJ_minus_MIX | +0.2219 [+0.0664, +0.3786] | +0.2821 | 2 / 2 | +0.2141 | -0.0498 | +0.1013 |
| large | PROJ_minus_ROT_PROJ | +5.0241 [+3.9618, +6.2531] | +5.2323 | 39 / 5 | +3.4441 | +1.7690 | -5.2101 |
| large | MIX_minus_A | -32.8423 [-37.2563, -28.3777] | -33.8485 | 3 / 181 | -31.1199 | -0.4282 | +4.1628 |
| large | MIX_minus_ROT_MIX | +7.7607 [+6.5609, +9.1112] | +7.5586 | 77 / 3 | +4.9575 | +4.0132 | -8.5686 |
| large | PROJ_minus_AGN_PROJ | -8.1085 [-10.5068, -5.8421] | -9.4152 | 9 / 40 | -11.7902 | +7.8403 | -8.8326 |
| large | SEM_minus_A | -39.3000 [-43.6255, -34.9675] | -39.4759 | 1 / 227 | -32.5610 | -21.4988 | +18.8619 |

Interpretation boundaries:
- The three primary comparisons are PROJ-A, PROJ-MIX and PROJ-ROT_PROJ. Multiple intervals are exploratory, without a confirmatory familywise-error claim.
- PROJ exceeding MIX only supports this fixed ridge/alpha projection against this direct mixture. PROJ exceeding ROT_PROJ supports this within-box spatial arrangement; it is not proof of general instance ownership.
- Rotation preserves the exact support-value histogram but may change little for nearly symmetric/constant maps. SOURCE_ROWS records RMS, mean absolute and nonzero spatial changes; an inactive contrast cannot reject spatial information.
- AGN takes a maximum across classes, which changes logit scale/distribution. PROJ versus AGN_PROJ alone cannot identify a pure class-specific mechanism.
- Same-class-neighbor strata use other noncrowd GT boxes with positive-area overlap with the recipient prediction box. They describe proximity, not actual mask overlap or occlusion. Predicted-class correctness and neighbor labels enter evaluation only.
- No candidate is removed for an empty mask or undefined AUC. Failure-group damage is structurally zero versus A and does not exclude continuous degradation.
- These reused dev images are not a new blind test. Fixed TAL candidates are not complete deployment outputs or COCO AP. No training or automatic follow-up is authorized by a positive submetric.

Prespecified decisions: {"MIX": {"prespecified_overall_signal": false, "prespecified_target_signal": false, "other_metrics_require_effect_tradeoff_review": true}, "PROJ": {"prespecified_overall_signal": false, "prespecified_target_signal": false, "other_metrics_require_effect_tradeoff_review": true}}
