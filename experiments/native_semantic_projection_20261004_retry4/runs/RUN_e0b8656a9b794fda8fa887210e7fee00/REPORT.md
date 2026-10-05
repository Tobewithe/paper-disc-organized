# Native semantic response and original-basis projection

Frozen original YOLO26m-seg. No training or parameter selection; all reported mask effects are percentage points.
2 planned dev images, 2 effective images, 7 fixed official candidates.

A is the original mask. SEM directly decodes the predicted-class semantic logits; MIX averages original and semantic logits equally. PROJ applies the fixed full-support ridge solver to half their logit difference. ROT controls rotate only the semantic response inside the prediction box. AGN_PROJ substitutes the maximum class logit.

| Group | Pair | Image macro IoU delta [95% CI] | Candidate IoU delta | Repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---|---:|---:|---:|---:|---:|---:|
| all | PROJ_minus_A | -19.6585 [-39.2527, -0.0642] | -11.2609 | 0 / 0 | -23.5979 | -0.4305 | -0.8870 |
| all | PROJ_minus_MIX | +1.2941 [+0.0798, +2.5084] | +1.8145 | 0 / 0 | +0.4434 | +0.1126 | -0.9258 |
| all | PROJ_minus_ROT_PROJ | +4.5613 [+0.3929, +8.7296] | +6.3477 | 0 / 0 | +5.5042 | +1.2300 | -0.5569 |
| all | MIX_minus_A | -20.9526 [-39.3325, -2.5726] | -13.0755 | 0 / 0 | -24.0414 | -0.5431 | +0.0388 |
| all | MIX_minus_ROT_MIX | +4.0663 [+0.5417, +7.5910] | +5.5769 | 0 / 0 | +4.2090 | +2.0053 | -2.7668 |
| all | PROJ_minus_AGN_PROJ | -14.7263 [-34.7567, +5.3041] | -6.1419 | 0 / 0 | -24.8022 | +0.0580 | -12.0227 |
| all | SEM_minus_A | -28.3355 [-47.2007, -9.4704] | -20.2505 | 0 / 0 | -26.6243 | -20.1065 | +19.2163 |
| box_good_mask_bad | PROJ_minus_A | -30.6674 [-66.1877, +4.8530] | -30.6674 | 0 / 0 | -49.6856 | -0.4043 | -17.9138 |
| box_good_mask_bad | PROJ_minus_MIX | +5.7548 [+0.0000, +11.5096] | +5.7548 | 0 / 0 | +1.6670 | +0.6204 | -4.0820 |
| box_good_mask_bad | PROJ_minus_ROT_PROJ | +14.9485 [+0.0000, +29.8970] | +14.9485 | 0 / 0 | +19.2420 | +4.7984 | +0.4898 |
| box_good_mask_bad | MIX_minus_A | -36.4222 [-66.1877, -6.6566] | -36.4222 | 0 / 0 | -51.3526 | -1.0246 | -13.8318 |
| box_good_mask_bad | MIX_minus_ROT_MIX | +10.4263 [+0.0000, +20.8526] | +10.4263 | 0 / 0 | +12.7820 | +6.3921 | -3.0967 |
| box_good_mask_bad | PROJ_minus_AGN_PROJ | -21.7191 [-69.5134, +26.0752] | -21.7191 | 0 / 0 | -52.9799 | +1.3209 | -35.4407 |
| box_good_mask_bad | SEM_minus_A | -39.6854 [-66.1877, -13.1831] | -39.6854 | 0 / 0 | -56.3135 | -22.7753 | -13.9677 |
| original_success | PROJ_minus_A | -1.2935 [-1.2935, -1.2935] | -1.2935 | 0 / 0 | -0.9693 | -0.0026 | +2.8840 |
| original_success | PROJ_minus_MIX | +0.2581 [+0.2581, +0.2581] | +0.2581 | 0 / 0 | +0.2751 | +0.1186 | -0.0565 |
| original_success | PROJ_minus_ROT_PROJ | +3.4378 [+3.4378, +3.4378] | +3.4378 | 0 / 0 | +4.1394 | +0.1619 | -0.5520 |
| original_success | MIX_minus_A | -1.5517 [-1.5517, -1.5517] | -1.5517 | 0 / 0 | -1.2444 | -0.1213 | +2.9405 |
| original_success | MIX_minus_ROT_MIX | +4.2756 [+4.2756, +4.2756] | +4.2756 | 0 / 0 | +4.1314 | +1.6820 | -3.8496 |
| original_success | PROJ_minus_AGN_PROJ | +0.1113 [+0.1113, +0.1113] | +0.1113 | 0 / 0 | -1.4247 | +0.0475 | -3.9456 |
| original_success | SEM_minus_A | -8.5422 [-8.5422, -8.5422] | -8.5422 | 0 / 0 | -5.2213 | -3.5856 | +16.0129 |
| original_failure | PROJ_minus_A | -17.1999 [-39.2527, +4.8530] | -24.5508 | 0 / 0 | -27.0626 | -0.1297 | -6.1322 |
| original_failure | PROJ_minus_MIX | +5.7947 [+0.0798, +11.5096] | +3.8897 | 0 / 0 | +1.6670 | +1.2632 | -4.1688 |
| original_failure | PROJ_minus_ROT_PROJ | +15.1449 [+0.3929, +29.8970] | +10.2276 | 0 / 0 | +19.2420 | +5.0209 | +0.0557 |
| original_failure | MIX_minus_A | -22.9946 [-39.3325, -6.6566] | -28.4405 | 0 / 0 | -28.7296 | -1.3929 | -1.9634 |
| original_failure | MIX_minus_ROT_MIX | +10.6971 [+0.5417, +20.8526] | +7.3120 | 0 / 0 | +12.7820 | +6.2429 | -3.7043 |
| original_failure | PROJ_minus_AGN_PROJ | -4.3407 [-34.7567, +26.0752] | -14.4794 | 0 / 0 | -30.2526 | +1.6963 | -29.8467 |
| original_failure | SEM_minus_A | -30.1919 [-47.2007, -13.1831] | -35.8615 | 0 / 0 | -33.6905 | -20.7949 | +11.8763 |
| predclass_correct | PROJ_minus_A | -19.5209 [-39.2527, +0.2110] | -12.9436 | 0 / 0 | -23.9388 | -0.4066 | -1.1386 |
| predclass_correct | PROJ_minus_MIX | +1.6329 [+0.0798, +3.1860] | +2.1506 | 0 / 0 | +0.5701 | +0.1910 | -1.1562 |
| predclass_correct | PROJ_minus_ROT_PROJ | +5.8529 [+0.3929, +11.3129] | +7.6729 | 0 / 0 | +6.8739 | +1.4842 | -0.9805 |
| predclass_correct | MIX_minus_A | -21.1538 [-39.3325, -2.9750] | -15.0942 | 0 / 0 | -24.5088 | -0.5976 | +0.0176 |
| predclass_correct | MIX_minus_ROT_MIX | +5.1200 [+0.5417, +9.6983] | +6.6461 | 0 / 0 | +5.1320 | +2.4356 | -3.7357 |
| predclass_correct | PROJ_minus_AGN_PROJ | -14.1860 [-34.7567, +6.3848] | -7.3291 | 0 / 0 | -25.3052 | +0.1628 | -13.3611 |
| predclass_correct | SEM_minus_A | -28.3270 [-47.2007, -9.4532] | -22.0357 | 0 / 0 | -27.1385 | -19.7856 | +19.3892 |
| predclass_incorrect | PROJ_minus_A | -1.1651 [-1.1651, -1.1651] | -1.1651 | 0 / 0 | +0.0252 | -0.0429 | +2.2746 |
| predclass_incorrect | PROJ_minus_MIX | -0.2018 [-0.2018, -0.2018] | -0.2018 | 0 / 0 | -0.1261 | +0.0673 | +0.1654 |
| predclass_incorrect | PROJ_minus_ROT_PROJ | -1.6034 [-1.6034, -1.6034] | -1.6034 | 0 / 0 | +0.0504 | +0.0236 | +3.1431 |
| predclass_incorrect | MIX_minus_A | -0.9633 [-0.9633, -0.9633] | -0.9633 | 0 / 0 | +0.1513 | -0.1102 | +2.1092 |
| predclass_incorrect | MIX_minus_ROT_MIX | -0.8382 [-0.8382, -0.8382] | -0.8382 | 0 / 0 | +1.0340 | +0.3583 | +3.4326 |
| predclass_incorrect | PROJ_minus_AGN_PROJ | +0.9813 [+0.9813, +0.9813] | +0.9813 | 0 / 0 | -0.1261 | +0.0279 | -2.1505 |
| predclass_incorrect | SEM_minus_A | -9.5390 [-9.5390, -9.5390] | -9.5390 | 0 / 0 | -4.6406 | -6.4967 | +10.9595 |
| sameclass_neighbor | PROJ_minus_A | +1.8439 [+1.8439, +1.8439] | +1.8439 | 0 / 0 | -4.8029 | +0.3532 | -3.9772 |
| sameclass_neighbor | PROJ_minus_MIX | +5.6539 [+5.6539, +5.6539] | +5.6539 | 0 / 0 | +1.6039 | +1.5312 | -3.9993 |
| sameclass_neighbor | PROJ_minus_ROT_PROJ | +14.1468 [+14.1468, +14.1468] | +14.1468 | 0 / 0 | +19.2672 | +4.8314 | +2.0613 |
| sameclass_neighbor | MIX_minus_A | -3.8100 [-3.8100, -3.8100] | -3.8100 | 0 / 0 | -6.4068 | -1.1780 | +0.0220 |
| sameclass_neighbor | MIX_minus_ROT_MIX | +10.0072 [+10.0072, +10.0072] | +10.0072 | 0 / 0 | +13.2990 | +6.3171 | -1.3804 |
| sameclass_neighbor | PROJ_minus_AGN_PROJ | +13.5283 [+13.5283, +13.5283] | +13.5283 | 0 / 0 | -7.5884 | +2.0856 | -25.3281 |
| sameclass_neighbor | SEM_minus_A | -11.3611 [-11.3611, -11.3611] | -11.3611 | 0 / 0 | -13.7637 | -5.9017 | +4.3112 |
| no_sameclass_neighbor | PROJ_minus_A | -20.2945 [-39.2527, -1.3363] | -16.5029 | 0 / 0 | -22.8975 | -0.4990 | +0.5259 |
| no_sameclass_neighbor | PROJ_minus_MIX | +0.2456 [+0.0798, +0.4114] | +0.2788 | 0 / 0 | +0.2044 | -0.1665 | -0.1521 |
| no_sameclass_neighbor | PROJ_minus_ROT_PROJ | +2.7555 [+0.3929, +5.1182] | +3.2280 | 0 / 0 | +2.7512 | +0.3053 | -1.3259 |
| no_sameclass_neighbor | MIX_minus_A | -20.5401 [-39.3325, -1.7478] | -16.7817 | 0 / 0 | -23.1019 | -0.3325 | +0.6779 |
| no_sameclass_neighbor | MIX_minus_ROT_MIX | +3.2609 [+0.5417, +5.9802] | +3.8048 | 0 / 0 | +2.5819 | +1.1665 | -3.7461 |
| no_sameclass_neighbor | PROJ_minus_AGN_PROJ | -17.4677 [-34.7567, -0.1787] | -14.0099 | 0 / 0 | -23.6560 | -0.3483 | -7.8659 |
| no_sameclass_neighbor | SEM_minus_A | -27.7053 [-47.2007, -8.2099] | -23.8062 | 0 / 0 | -24.9546 | -19.4492 | +21.8935 |
| P3 | PROJ_minus_A | -12.3177 [-12.3177, -12.3177] | -12.3177 | 0 / 0 | +0.7519 | -0.4596 | +21.5278 |
| P3 | PROJ_minus_MIX | +0.1596 [+0.1596, +0.1596] | +0.1596 | 0 / 0 | +0.0000 | +0.8170 | -0.3472 |
| P3 | PROJ_minus_ROT_PROJ | +0.7857 [+0.7857, +0.7857] | +0.7857 | 0 / 0 | +0.0000 | +0.8476 | -1.7361 |
| P3 | MIX_minus_A | -12.4773 [-12.4773, -12.4773] | -12.4773 | 0 / 0 | +0.7519 | -1.2766 | +21.8750 |
| P3 | MIX_minus_ROT_MIX | +1.0833 [+1.0833, +1.0833] | +1.0833 | 0 / 0 | +0.0000 | -0.0885 | -2.4306 |
| P3 | PROJ_minus_AGN_PROJ | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 / 0 | +0.0000 | +0.0000 | +0.0000 |
| P3 | SEM_minus_A | -28.2137 [-28.2137, -28.2137] | -28.2137 | 0 / 0 | +0.7519 | -32.3223 | +77.7778 |
| P4 | PROJ_minus_A | -33.6145 [-66.1877, -1.0412] | -22.7567 | 0 / 0 | -44.8025 | -0.8052 | -11.6568 |
| P4 | PROJ_minus_MIX | +0.0176 [+0.0000, +0.0353] | +0.0235 | 0 / 0 | +0.0237 | -0.8285 | +0.0089 |
| P4 | PROJ_minus_ROT_PROJ | -0.6132 [-1.2265, +0.0000] | -0.8176 | 0 / 0 | +0.1721 | +0.0128 | +1.5112 |
| P4 | MIX_minus_A | -33.6321 [-66.1877, -1.0764] | -22.7802 | 0 / 0 | -44.8262 | +0.0233 | -11.6656 |
| P4 | MIX_minus_ROT_MIX | +0.1854 [+0.0000, +0.3707] | +0.2472 | 0 / 0 | +0.8843 | +0.5083 | +1.2588 |
| P4 | PROJ_minus_AGN_PROJ | -34.3574 [-69.5134, +0.7986] | -22.6387 | 0 / 0 | -45.5658 | -0.7321 | -12.1910 |
| P4 | SEM_minus_A | -38.0917 [-66.1877, -9.9956] | -28.7263 | 0 / 0 | -47.2266 | -23.3310 | -6.7896 |
| P5 | PROJ_minus_A | +0.5871 [+0.5871, +0.5871] | +0.5871 | 0 / 0 | -4.5930 | +0.2813 | -1.0877 |
| P5 | PROJ_minus_MIX | +4.1572 [+4.1572, +4.1572] | +4.1572 | 0 / 0 | +1.4465 | +1.0916 | -2.8085 |
| P5 | PROJ_minus_ROT_PROJ | +15.3670 [+15.3670, +15.3670] | +15.3670 | 0 / 0 | +18.1177 | +3.3837 | -2.4244 |
| P5 | MIX_minus_A | -3.5701 [-3.5701, -3.5701] | -3.5701 | 0 / 0 | -6.0395 | -0.8103 | +1.7209 |
| P5 | MIX_minus_ROT_MIX | +12.4045 [+12.4045, +12.4045] | +12.4045 | 0 / 0 | +12.8509 | +5.9957 | -8.8756 |
| P5 | PROJ_minus_AGN_PROJ | +8.3077 [+8.3077, +8.3077] | +8.3077 | 0 / 0 | -6.7681 | +1.4197 | -20.0918 |
| P5 | SEM_minus_A | -9.1202 [-9.1202, -9.1202] | -9.1202 | 0 / 0 | -11.4488 | -2.2710 | +12.5587 |
| small | PROJ_minus_A | -39.2527 [-39.2527, -39.2527] | -39.2527 | 0 / 0 | -44.4942 | -1.0088 | -2.0354 |
| small | PROJ_minus_MIX | +0.0798 [+0.0798, +0.0798] | +0.0798 | 0 / 0 | +0.0000 | -0.4687 | -0.1736 |
| small | PROJ_minus_ROT_PROJ | +0.3929 [+0.3929, +0.3929] | +0.3929 | 0 / 0 | +0.0000 | +0.4026 | -0.8681 |
| small | MIX_minus_A | -39.3325 [-39.3325, -39.3325] | -39.3325 | 0 / 0 | -44.4942 | -0.5401 | -1.8618 |
| small | MIX_minus_ROT_MIX | +0.5417 [+0.5417, +0.5417] | +0.5417 | 0 / 0 | +0.0000 | +0.2098 | -1.2153 |
| small | PROJ_minus_AGN_PROJ | -34.7567 [-34.7567, -34.7567] | -34.7567 | 0 / 0 | -45.4545 | -0.7507 | -11.1878 |
| small | SEM_minus_A | -47.2007 [-47.2007, -47.2007] | -47.2007 | 0 / 0 | -44.4942 | -36.2831 | +26.0896 |
| medium | PROJ_minus_A | -1.0412 [-1.0412, -1.0412] | -1.0412 | 0 / 0 | +0.1353 | -0.0525 | +2.2850 |
| medium | PROJ_minus_MIX | +0.0353 [+0.0353, +0.0353] | +0.0353 | 0 / 0 | +0.0474 | +0.0974 | +0.0177 |
| medium | PROJ_minus_ROT_PROJ | -1.2265 [-1.2265, -1.2265] | -1.2265 | 0 / 0 | +0.3442 | +0.0679 | +3.0224 |
| medium | MIX_minus_A | -1.0764 [-1.0764, -1.0764] | -1.0764 | 0 / 0 | +0.0879 | -0.1498 | +2.2672 |
| medium | MIX_minus_ROT_MIX | +0.3707 [+0.3707, +0.3707] | +0.3707 | 0 / 0 | +1.7686 | +0.5084 | +2.5175 |
| medium | PROJ_minus_AGN_PROJ | +0.7986 [+0.7986, +0.7986] | +0.7986 | 0 / 0 | -0.2226 | +0.0371 | -2.0064 |
| medium | SEM_minus_A | -9.9956 [-9.9956, -9.9956] | -9.9956 | 0 / 0 | -4.7129 | -6.4180 | +12.0194 |
| large | PROJ_minus_A | +0.5871 [+0.5871, +0.5871] | +0.5871 | 0 / 0 | -4.5930 | +0.2813 | -1.0877 |
| large | PROJ_minus_MIX | +4.1572 [+4.1572, +4.1572] | +4.1572 | 0 / 0 | +1.4465 | +1.0916 | -2.8085 |
| large | PROJ_minus_ROT_PROJ | +15.3670 [+15.3670, +15.3670] | +15.3670 | 0 / 0 | +18.1177 | +3.3837 | -2.4244 |
| large | MIX_minus_A | -3.5701 [-3.5701, -3.5701] | -3.5701 | 0 / 0 | -6.0395 | -0.8103 | +1.7209 |
| large | MIX_minus_ROT_MIX | +12.4045 [+12.4045, +12.4045] | +12.4045 | 0 / 0 | +12.8509 | +5.9957 | -8.8756 |
| large | PROJ_minus_AGN_PROJ | +8.3077 [+8.3077, +8.3077] | +8.3077 | 0 / 0 | -6.7681 | +1.4197 | -20.0918 |
| large | SEM_minus_A | -9.1202 [-9.1202, -9.1202] | -9.1202 | 0 / 0 | -11.4488 | -2.2710 | +12.5587 |

Interpretation boundaries:
- The three primary comparisons are PROJ-A, PROJ-MIX and PROJ-ROT_PROJ. Multiple intervals are exploratory, without a confirmatory familywise-error claim.
- PROJ exceeding MIX only supports this fixed ridge/alpha projection against this direct mixture. PROJ exceeding ROT_PROJ supports this within-box spatial arrangement; it is not proof of general instance ownership.
- Rotation preserves the exact support-value histogram but may change little for nearly symmetric/constant maps. SOURCE_ROWS records RMS, mean absolute and nonzero spatial changes; an inactive contrast cannot reject spatial information.
- AGN takes a maximum across classes, which changes logit scale/distribution. PROJ versus AGN_PROJ alone cannot identify a pure class-specific mechanism.
- Same-class-neighbor strata use other noncrowd GT boxes with positive-area overlap with the recipient prediction box. They describe proximity, not actual mask overlap or occlusion. Predicted-class correctness and neighbor labels enter evaluation only.
- No candidate is removed for an empty mask or undefined AUC. Failure-group damage is structurally zero versus A and does not exclude continuous degradation.
- These reused dev images are not a new blind test. Fixed TAL candidates are not complete deployment outputs or COCO AP. No training or automatic follow-up is authorized by a positive submetric.

Prespecified decisions: {"MIX": {"prespecified_overall_signal": false, "prespecified_target_signal": false, "other_metrics_require_effect_tradeoff_review": true}, "PROJ": {"prespecified_overall_signal": false, "prespecified_target_signal": false, "other_metrics_require_effect_tradeoff_review": true}}
