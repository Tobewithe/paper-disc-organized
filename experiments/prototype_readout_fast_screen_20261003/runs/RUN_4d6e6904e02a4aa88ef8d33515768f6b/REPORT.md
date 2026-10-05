# Prototype-to-coefficient rapid screen

Fixed epoch 3 on 1,024 planned fit images; evaluation is the fixed 256-image dev slice. This is a short-budget developmental screen, not independent confirmation or COCO AP.

Evaluated 1816 fixed official candidates in 253 effective dev images. Missing or unfinished arms: none. An unfinished arm is not a negative result.

All effect sizes below are percentage points. Each CI resamples whole images 1,000 times; the table's point estimate and CI both use image macro.

| Group | Comparison | Macro IoU delta [95% CI] | Candidate IoU delta | Net Mask75 | Repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | N_minus_A | -0.0315 [-0.1657, +0.1319] | -0.1382 | -2 | 16 / 18 | -0.2471 | -0.0531 | -0.2814 |
| all | U_minus_A | -0.0247 [-0.1599, +0.1407] | -0.1285 | 0 | 17 / 17 | -0.2741 | -0.0524 | -0.3559 |
| all | U_minus_N | +0.0069 [+0.0014, +0.0120] | +0.0098 | 2 | 3 / 1 | -0.0270 | +0.0007 | -0.0745 |
| all | Q_minus_A | -0.0269 [-0.1642, +0.1401] | -0.1304 | -3 | 16 / 19 | -0.2952 | -0.0526 | -0.3725 |
| all | Q_minus_N | +0.0046 [-0.0028, +0.0108] | +0.0078 | -1 | 1 / 2 | -0.0481 | +0.0005 | -0.0910 |
| all | P_minus_A | -0.0273 [-0.1620, +0.1378] | -0.1311 | -1 | 17 / 18 | -0.2843 | -0.0525 | -0.3671 |
| all | P_minus_N | +0.0043 [-0.0029, +0.0103] | +0.0072 | 1 | 3 / 2 | -0.0372 | +0.0006 | -0.0857 |
| all | L_minus_A | -0.0246 [-0.1590, +0.1402] | -0.1292 | -1 | 16 / 17 | -0.2737 | -0.0520 | -0.3561 |
| all | L_minus_N | +0.0070 [+0.0016, +0.0120] | +0.0090 | 1 | 2 / 1 | -0.0266 | +0.0011 | -0.0747 |
| all | B_minus_A | -0.0271 [-0.1618, +0.1376] | -0.1309 | 0 | 17 / 17 | -0.2812 | -0.0528 | -0.3623 |
| all | B_minus_N | +0.0044 [-0.0020, +0.0101] | +0.0073 | 2 | 3 / 1 | -0.0341 | +0.0003 | -0.0809 |
| all | P_minus_Q | -0.0003 [-0.0040, +0.0038] | -0.0007 | 2 | 2 / 0 | +0.0109 | +0.0001 | +0.0054 |
| all | P_minus_U | -0.0026 [-0.0055, -0.0001] | -0.0026 | -1 | 0 / 1 | -0.0101 | -0.0001 | -0.0112 |
| all | L_minus_Q | +0.0024 [-0.0021, +0.0076] | +0.0012 | 2 | 2 / 0 | +0.0215 | +0.0006 | +0.0163 |
| all | B_minus_Q | -0.0002 [-0.0043, +0.0044] | -0.0005 | 3 | 3 / 0 | +0.0140 | -0.0002 | +0.0101 |
| box_good_mask_bad | N_minus_A | +0.0707 [-0.5831, +0.7687] | -0.1024 | 11 | 11 / 0 | -0.7356 | -0.2649 | -0.9227 |
| box_good_mask_bad | U_minus_A | +0.0695 [-0.5885, +0.7681] | -0.0965 | 13 | 13 / 0 | -0.7969 | -0.2660 | -1.0353 |
| box_good_mask_bad | U_minus_N | -0.0011 [-0.0171, +0.0141] | +0.0059 | 2 | 2 / 0 | -0.0613 | -0.0011 | -0.1126 |
| box_good_mask_bad | Q_minus_A | +0.0625 [-0.6070, +0.7657] | -0.1106 | 12 | 12 / 0 | -0.8399 | -0.2657 | -1.0571 |
| box_good_mask_bad | Q_minus_N | -0.0082 [-0.0320, +0.0141] | -0.0082 | 1 | 1 / 0 | -0.1042 | -0.0008 | -0.1344 |
| box_good_mask_bad | P_minus_A | +0.0606 [-0.5994, +0.7616] | -0.1035 | 13 | 13 / 0 | -0.8198 | -0.2650 | -1.0509 |
| box_good_mask_bad | P_minus_N | -0.0100 [-0.0317, +0.0090] | -0.0011 | 2 | 2 / 0 | -0.0841 | -0.0000 | -0.1282 |
| box_good_mask_bad | L_minus_A | +0.0656 [-0.5916, +0.7685] | -0.1038 | 12 | 12 / 0 | -0.7944 | -0.2644 | -1.0194 |
| box_good_mask_bad | L_minus_N | -0.0050 [-0.0208, +0.0103] | -0.0014 | 1 | 1 / 0 | -0.0588 | +0.0006 | -0.0967 |
| box_good_mask_bad | B_minus_A | +0.0601 [-0.6021, +0.7616] | -0.1048 | 13 | 13 / 0 | -0.8175 | -0.2671 | -1.0448 |
| box_good_mask_bad | B_minus_N | -0.0105 [-0.0312, +0.0084] | -0.0024 | 2 | 2 / 0 | -0.0818 | -0.0022 | -0.1221 |
| box_good_mask_bad | P_minus_Q | -0.0018 [-0.0129, +0.0094] | +0.0071 | 1 | 1 / 0 | +0.0201 | +0.0008 | +0.0062 |
| box_good_mask_bad | P_minus_U | -0.0089 [-0.0184, -0.0005] | -0.0070 | 0 | 0 / 0 | -0.0228 | +0.0010 | -0.0157 |
| box_good_mask_bad | L_minus_Q | +0.0032 [-0.0124, +0.0191] | +0.0067 | 0 | 0 / 0 | +0.0455 | +0.0014 | +0.0377 |
| box_good_mask_bad | B_minus_Q | -0.0023 [-0.0139, +0.0092] | +0.0058 | 1 | 1 / 0 | +0.0224 | -0.0014 | +0.0124 |
| original_success | N_minus_A | -0.1367 [-0.2196, -0.0622] | -0.2049 | -18 | 0 / 18 | -0.2111 | -0.0184 | -0.1693 |
| original_success | U_minus_A | -0.1295 [-0.2154, -0.0555] | -0.1956 | -17 | 0 / 17 | -0.2291 | -0.0179 | -0.2210 |
| original_success | U_minus_N | +0.0072 [+0.0025, +0.0118] | +0.0092 | 1 | 1 / 0 | -0.0180 | +0.0005 | -0.0517 |
| original_success | Q_minus_A | -0.1272 [-0.2125, -0.0533] | -0.1929 | -19 | 0 / 19 | -0.2432 | -0.0179 | -0.2516 |
| original_success | Q_minus_N | +0.0095 [+0.0037, +0.0153] | +0.0120 | -1 | 0 / 1 | -0.0320 | +0.0005 | -0.0823 |
| original_success | P_minus_A | -0.1291 [-0.2160, -0.0549] | -0.1969 | -18 | 0 / 18 | -0.2342 | -0.0183 | -0.2310 |
| original_success | P_minus_N | +0.0076 [+0.0015, +0.0136] | +0.0080 | 0 | 1 / 1 | -0.0231 | +0.0001 | -0.0617 |
| original_success | L_minus_A | -0.1280 [-0.2135, -0.0531] | -0.1955 | -17 | 0 / 17 | -0.2289 | -0.0180 | -0.2238 |
| original_success | L_minus_N | +0.0087 [+0.0035, +0.0146] | +0.0093 | 1 | 1 / 0 | -0.0178 | +0.0004 | -0.0545 |
| original_success | B_minus_A | -0.1302 [-0.2158, -0.0559] | -0.1971 | -17 | 0 / 17 | -0.2336 | -0.0184 | -0.2280 |
| original_success | B_minus_N | +0.0065 [+0.0012, +0.0118] | +0.0078 | 1 | 1 / 0 | -0.0224 | +0.0000 | -0.0586 |
| original_success | P_minus_Q | -0.0019 [-0.0047, +0.0009] | -0.0040 | 1 | 1 / 0 | +0.0089 | -0.0003 | +0.0205 |
| original_success | P_minus_U | +0.0004 [-0.0026, +0.0042] | -0.0012 | -1 | 0 / 1 | -0.0052 | -0.0004 | -0.0100 |
| original_success | L_minus_Q | -0.0009 [-0.0043, +0.0027] | -0.0027 | 2 | 2 / 0 | +0.0143 | -0.0001 | +0.0278 |
| original_success | B_minus_Q | -0.0030 [-0.0052, -0.0010] | -0.0042 | 2 | 2 / 0 | +0.0096 | -0.0004 | +0.0236 |
| original_failure | N_minus_A | +0.1461 [-0.3551, +0.7170] | -0.0264 | 16 | 16 / 0 | -0.5015 | -0.2384 | -0.6994 |
| original_failure | U_minus_A | +0.1514 [-0.3522, +0.7215] | -0.0157 | 17 | 17 / 0 | -0.5539 | -0.2379 | -0.8282 |
| original_failure | U_minus_N | +0.0053 [-0.0085, +0.0195] | +0.0106 | 1 | 2 / 1 | -0.0524 | +0.0005 | -0.1289 |
| original_failure | Q_minus_A | +0.1457 [-0.3641, +0.7189] | -0.0255 | 16 | 16 / 0 | -0.5912 | -0.2381 | -0.8310 |
| original_failure | Q_minus_N | -0.0004 [-0.0207, +0.0188] | +0.0009 | 0 | 1 / 1 | -0.0897 | +0.0003 | -0.1317 |
| original_failure | P_minus_A | +0.1449 [-0.3630, +0.7190] | -0.0206 | 17 | 17 / 0 | -0.5772 | -0.2381 | -0.8508 |
| original_failure | P_minus_N | -0.0012 [-0.0196, +0.0167] | +0.0057 | 1 | 2 / 1 | -0.0757 | +0.0003 | -0.1514 |
| original_failure | L_minus_A | +0.1508 [-0.3535, +0.7183] | -0.0179 | 16 | 16 / 0 | -0.5523 | -0.2373 | -0.8266 |
| original_failure | L_minus_N | +0.0047 [-0.0096, +0.0193] | +0.0085 | 0 | 1 / 1 | -0.0507 | +0.0011 | -0.1272 |
| original_failure | B_minus_A | +0.1471 [-0.3620, +0.7207] | -0.0199 | 17 | 17 / 0 | -0.5693 | -0.2381 | -0.8415 |
| original_failure | B_minus_N | +0.0010 [-0.0171, +0.0183] | +0.0065 | 1 | 2 / 1 | -0.0677 | +0.0003 | -0.1421 |
| original_failure | P_minus_Q | -0.0008 [-0.0107, +0.0105] | +0.0049 | 1 | 1 / 0 | +0.0140 | +0.0000 | -0.0198 |
| original_failure | P_minus_U | -0.0065 [-0.0152, +0.0011] | -0.0049 | 0 | 0 / 0 | -0.0233 | -0.0002 | -0.0226 |
| original_failure | L_minus_Q | +0.0051 [-0.0085, +0.0203] | +0.0076 | 0 | 0 / 0 | +0.0389 | +0.0008 | +0.0045 |
| original_failure | B_minus_Q | +0.0014 [-0.0088, +0.0134] | +0.0056 | 1 | 1 / 0 | +0.0219 | +0.0000 | -0.0104 |

GT only defines fixed official supervision and evaluation groups; the selectors use original frozen P, c0 and predicted boxes. All arms use the same normal full-prototype decoder and original COCO instance masks. AUC/FPR use continuous input-space logits on the original predicted-box support; undefined AUC cases remain in IoU evaluation.

N is the same-screen native coefficient-branch fine-tuning control. No historical method metrics are joined. U/Q/P/L/B have matched trainable architecture; their frozen point choices differ. Positive differences against another weak arm do not by themselves establish improvement over original A.

The automated screen status concerns the prespecified IoU threshold only. It does not veto a meaningful Mask75, coverage, AUC or FPR improvement, and it does not require all metrics to improve together. Each improvement needs its own magnitude, uncertainty and trade-off assessment.

The screen status applies only to this seed, fit slice and three-epoch budget. No mechanism or final negative conclusion is inferred, and no next experiment starts automatically.

- U: IoU_screen_threshold_not_met_review_other_metrics
- Q: IoU_screen_threshold_not_met_review_other_metrics
- P: IoU_screen_threshold_not_met_review_other_metrics
- L: IoU_screen_threshold_not_met_review_other_metrics
- B: IoU_screen_threshold_not_met_review_other_metrics
