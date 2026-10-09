# QCR DEV fixed-candidate metrics

Protocol status: **incomplete**. Stage II permitted: **False**.

102 candidate rows; 10 / 10 planned images contain rows. 0 planned images contain no rows. Those images receive no artificial zero IoU.

All candidate means weight candidates equally. Image macro first averages eligible candidates within each image, then weights eligible images equally. Subgroup-empty images and images without a defined pixel metric are excluded from that estimand. Confidence intervals resample entire eligible images with replacement, preserving the pairing between arms. Deltas and CI in tables are percentage points (pp).

TAL MaskFail = class_correct and BoxIoU ≥ .75 and original MaskIoU < .75. Strict MaskFail additionally requires an audited raw_arg_maskfail=true: the full class-argmax raw set has a Box75 candidate and no Mask75 candidate for the target GT. Unknown raw-arg labels are excluded from strict metrics and block release; TAL failure never substitutes for strict failure. Baseline success uses the same class/box criteria and original MaskIoU ≥ .75. Size uses original COCO annotation area: small <32², medium [32²,96²), large ≥96².

Coverage uses original-resolution binary-mask intersection / original COCO annToMask area. Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background. All pixels outside the target GT (including other instances) are negative. AUC is exact Mann–Whitney with averaged ties. Undefined metrics remain NA. This is a fixed official one-to-one TAL matched-candidate evaluation and makes no COCO AP claim.

## Protocol criteria

| Criterion | Estimate | Required | Pass |
|---|---:|---:|---|
| heldout_pairwise_accuracy | 52.8197 % | >= 65.0000 % | False |
| step1_true_iou_improvement_fraction | 18.6275 % | > 55.0000 % | False |
| strict_maskfail_D_minus_B | -0.9633 pp | >= 0.2000 pp | False |
| strict_maskfail_D_minus_B_CI_low | -4.7193 pp | > 0.0000 pp | False |
| strict_maskfail_D_minus_A | -2.7511 pp | >= 0.5000 pp | False |
| all_D_minus_A_CI_low | -7.4734 pp | >= -0.1000 pp | False |

Ineligibility/incompleteness: partial_evaluation; scientific_scope_drift.

Scientific drift: Actual training seed 20261005 differs from recorded seed 0.; Quality ranking uses soft IoU rather than normal binary original-mask IoU.; Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.; Failure random-control radius is global rho, not each oracle displacement norm.; Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.; Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.; DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.; Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test..

## Quality diagnostics

Held-out pairwise accuracy: instance equal 52.8197%; pairs weighted 52.8980%; image macro 55.2146%; 2588 non-tie hard-IoU pairs, 100 eligible instances. True hard-IoU ties are excluded; predicted Q ties get 0.5 credit. Spearman uses averaged tie ranks; constant-state instances are undefined.

Spearman: instance equal 0.0556; image macro 0.1019; 100 defined instances.

Step-one true IoU increase: 18.6275% over 102 candidates (19 increase, 77 decrease, 6 ties; ties stay in the denominator).

Nonzero Q/IoU delta sign consistency: 58.3333% over 96 instances. Positive Q but decreased true IoU: 32 / 49 positive-Q instances (65.3061%).

Sign table (Q(c1)−Q(c0) rows / true IoU(D1)−IoU(A) columns; exact zeros kept separately):

| Q delta | IoU decrease | IoU tie | IoU increase |
|---|---:|---:|---:|
| negative | 45 | 0 | 8 |
| zero | 0 | 0 | 0 |
| positive | 32 | 6 | 11 |

Exclusions from nonzero consistency: {"missing_q_or_iou": 0, "both_zero": 0, "q_zero_iou_nonzero": 0, "q_nonzero_iou_zero": 6}.

Q-Rank at these unchanged fixed candidates keeps c0 and therefore the same MaskIoU as A. No ranking/AP effect is estimated here.

## all

102 candidates; 10 eligible images; 0 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 71.8427 / 77.2416 | 46.0784 / 59.0652 | 91.1545 / 91.1298 | 93.3273 / 94.7588 | 36.5627 / 22.5800 |
| B | 69.8012 / 76.1826 | 41.1765 / 60.1995 | 92.2971 / 91.8356 | 93.1618 / 94.5722 | 43.0108 / 26.0617 |
| D1 | 64.2374 / 68.6368 | 32.3529 / 40.4148 | 93.6442 / 94.7485 | 91.2024 / 92.8165 | 58.5544 / 47.0331 |
| D | 68.8688 / 72.9830 | 39.2157 / 52.6318 | 91.2861 / 90.6535 | 92.3117 / 93.7592 | 41.9491 / 30.2029 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.9325 | -3.1997 | [-6.5149, -0.5318] | 102 / 10 |
| D_minus_B | mask75 | -1.9608 | -7.5677 | [-20.0023, 2.3421] | 102 / 10 |
| D_minus_B | coverage | -1.0110 | -1.1821 | [-3.4638, 0.3987] | 102 / 10 |
| D_minus_B | auc | -0.8501 | -0.8130 | [-1.5606, -0.0478] | 102 / 10 |
| D_minus_B | fpr | -1.0617 | 4.1412 | [-2.4161, 11.9915] | 102 / 10 |
| D_minus_A | iou | -2.9739 | -4.2587 | [-7.4734, -1.5012] | 102 / 10 |
| D_minus_A | mask75 | -6.8627 | -6.4334 | [-17.4216, 0.7765] | 102 / 10 |
| D_minus_A | coverage | 0.1316 | -0.4763 | [-2.5810, 0.9718] | 102 / 10 |
| D_minus_A | auc | -1.0156 | -0.9995 | [-1.9450, 0.0154] | 102 / 10 |
| D_minus_A | fpr | 5.3864 | 7.6229 | [1.6205, 14.8277] | 102 / 10 |
| B_minus_A | iou | -2.0414 | -1.0590 | [-1.9530, -0.3014] | 102 / 10 |
| B_minus_A | mask75 | -4.9020 | 1.1343 | [-4.0432, 8.6118] | 102 / 10 |
| B_minus_A | coverage | 1.1426 | 0.7058 | [0.2461, 1.2083] | 102 / 10 |
| B_minus_A | auc | -0.1655 | -0.1865 | [-0.5213, 0.1269] | 102 / 10 |
| B_minus_A | fpr | 6.4481 | 3.4817 | [1.5699, 5.5691] | 102 / 10 |
| D1_minus_A | iou | -7.6053 | -8.6048 | [-12.2475, -4.6089] | 102 / 10 |
| D1_minus_A | mask75 | -13.7255 | -18.6504 | [-40.9882, -1.5619] | 102 / 10 |
| D1_minus_A | coverage | 2.4897 | 3.6187 | [0.9863, 6.9318] | 102 / 10 |
| D1_minus_A | auc | -2.1249 | -1.9423 | [-3.4214, -0.3675] | 102 / 10 |
| D1_minus_A | fpr | 21.9917 | 24.4531 | [18.0212, 30.9210] | 102 / 10 |
| D_minus_D1 | iou | 4.6314 | 4.3461 | [1.5463, 7.1987] | 102 / 10 |
| D_minus_D1 | mask75 | 6.8627 | 12.2170 | [-0.2830, 26.1765] | 102 / 10 |
| D_minus_D1 | coverage | -2.3581 | -4.0951 | [-6.9399, -1.5786] | 102 / 10 |
| D_minus_D1 | auc | 1.1093 | 0.9428 | [0.2325, 1.6563] | 102 / 10 |
| D_minus_D1 | fpr | -16.6053 | -16.8302 | [-21.3807, -11.8395] | 102 / 10 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 1 / 55 | 6 / 47 | 1.8182 / 4.1667 | 12.7660 / 7.4153 | -5 | 1.1343 [-4.0432, 8.6118] |
| D1 | 2 / 55 | 16 / 47 | 3.6364 / 18.7500 | 34.0426 / 34.3870 | -14 | -18.6504 [-40.9882, -1.5619] |
| D | 2 / 55 | 9 / 47 | 3.6364 / 4.1667 | 19.1489 / 12.7094 | -7 | -6.4334 [-17.4216, 0.7765] |

## tal_maskfail

28 candidates; 8 eligible images; 2 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 64.5545 / 61.5007 | 0.0000 / 0.0000 | 86.6355 / 80.9616 | 92.0366 / 88.9194 | 33.4204 / 28.4938 |
| B | 61.7669 / 60.3369 | 3.5714 / 6.2500 | 88.7607 / 82.6988 | 91.6779 / 88.6571 | 39.9751 / 32.6897 |
| D1 | 56.9841 / 61.3739 | 7.1429 / 18.7500 | 93.7883 / 93.1412 | 90.0931 / 89.4709 | 59.8749 / 51.6075 |
| D | 60.9216 / 58.5692 | 3.5714 / 3.1250 | 87.1488 / 78.5114 | 91.2753 / 88.8089 | 40.5456 / 32.8563 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.8453 | -1.7677 | [-4.5869, 0.8713] | 28 / 8 |
| D_minus_B | mask75 | 0.0000 | -3.1250 | [-18.7500, 9.3750] | 28 / 8 |
| D_minus_B | coverage | -1.6120 | -4.1874 | [-13.5631, 1.7861] | 28 / 8 |
| D_minus_B | auc | -0.4026 | 0.1518 | [-2.1544, 3.0829] | 28 / 8 |
| D_minus_B | fpr | 0.5706 | 0.1665 | [-5.4398, 5.1058] | 28 / 8 |
| D_minus_A | iou | -3.6328 | -2.9315 | [-6.1519, 0.3037] | 28 / 8 |
| D_minus_A | mask75 | 3.5714 | 3.1250 | [0.0000, 9.3750] | 28 / 8 |
| D_minus_A | coverage | 0.5132 | -2.4502 | [-11.5495, 3.4108] | 28 / 8 |
| D_minus_A | auc | -0.7613 | -0.1105 | [-3.2423, 3.4353] | 28 / 8 |
| D_minus_A | fpr | 7.1252 | 4.3625 | [-1.6398, 9.5485] | 28 / 8 |
| B_minus_A | iou | -2.7876 | -1.1638 | [-2.7343, 0.2137] | 28 / 8 |
| B_minus_A | mask75 | 3.5714 | 6.2500 | [0.0000, 18.7500] | 28 / 8 |
| B_minus_A | coverage | 2.1252 | 1.7372 | [0.8447, 2.6416] | 28 / 8 |
| B_minus_A | auc | -0.3588 | -0.2623 | [-1.4499, 0.6690] | 28 / 8 |
| B_minus_A | fpr | 6.5547 | 4.1960 | [2.0467, 6.4219] | 28 / 8 |
| D1_minus_A | iou | -7.5704 | -0.1268 | [-11.4215, 15.3161] | 28 / 8 |
| D1_minus_A | mask75 | 7.1429 | 18.7500 | [0.0000, 43.7500] | 28 / 8 |
| D1_minus_A | coverage | 7.1528 | 12.1796 | [2.5271, 24.8245] | 28 / 8 |
| D1_minus_A | auc | -1.9435 | 0.5515 | [-4.3531, 6.8177] | 28 / 8 |
| D1_minus_A | fpr | 26.4545 | 23.1137 | [11.7308, 33.4705] | 28 / 8 |
| D_minus_D1 | iou | 3.9376 | -2.8047 | [-16.9307, 7.2812] | 28 / 8 |
| D_minus_D1 | mask75 | -3.5714 | -15.6250 | [-43.7500, 6.2500] | 28 / 8 |
| D_minus_D1 | coverage | -6.6396 | -14.6298 | [-27.4739, -3.8788] | 28 / 8 |
| D_minus_D1 | auc | 1.1822 | -0.6619 | [-3.7739, 1.8639] | 28 / 8 |
| D_minus_D1 | fpr | -19.3292 | -18.7512 | [-24.3620, -12.5678] | 28 / 8 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 1 / 28 | 0 / 0 | 3.5714 / 6.2500 | NA / NA | 1 | 6.2500 [0.0000, 18.7500] |
| D1 | 2 / 28 | 0 / 0 | 7.1429 / 18.7500 | NA / NA | 2 | 18.7500 [0.0000, 43.7500] |
| D | 1 / 28 | 0 / 0 | 3.5714 / 3.1250 | NA / NA | 1 | 3.1250 [0.0000, 9.3750] |

## strict_maskfail

17 candidates; 6 eligible images; 4 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 60.9034 / 58.3182 | 0.0000 / 0.0000 | 85.0558 / 78.5469 | 89.3606 / 87.0086 | 36.5560 / 30.0085 |
| B | 58.2887 / 56.5304 | 0.0000 / 0.0000 | 86.8961 / 80.2753 | 89.0464 / 87.2573 | 43.2318 / 34.8348 |
| D1 | 56.3784 / 61.6076 | 11.7647 / 25.0000 | 94.5066 / 94.1049 | 87.6491 / 89.0949 | 63.5581 / 53.3123 |
| D | 58.3834 / 55.5671 | 5.8824 / 5.5556 | 84.9979 / 75.2897 | 88.8187 / 88.0255 | 42.3288 / 33.3268 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 0.0947 | -0.9633 | [-4.7193, 2.4993] | 17 / 6 |
| D_minus_B | mask75 | 5.8824 | 5.5556 | [0.0000, 16.6667] | 17 / 6 |
| D_minus_B | coverage | -1.8982 | -4.9856 | [-17.6507, 2.7839] | 17 / 6 |
| D_minus_B | auc | -0.2277 | 0.7682 | [-2.0737, 4.2928] | 17 / 6 |
| D_minus_B | fpr | -0.9030 | -1.5080 | [-8.7236, 5.2510] | 17 / 6 |
| D_minus_A | iou | -2.5200 | -2.7511 | [-7.2767, 1.6853] | 17 / 6 |
| D_minus_A | mask75 | 5.8824 | 5.5556 | [0.0000, 16.6667] | 17 / 6 |
| D_minus_A | coverage | -0.0579 | -3.2572 | [-15.6503, 4.8209] | 17 / 6 |
| D_minus_A | auc | -0.5419 | 1.0169 | [-2.2504, 5.0125] | 17 / 6 |
| D_minus_A | fpr | 5.7728 | 3.3182 | [-4.6057, 11.6175] | 17 / 6 |
| B_minus_A | iou | -2.6147 | -1.7878 | [-3.7993, -0.1228] | 17 / 6 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 17 / 6 |
| B_minus_A | coverage | 1.8403 | 1.7284 | [0.4217, 3.1919] | 17 / 6 |
| B_minus_A | auc | -0.3142 | 0.2487 | [-0.5626, 0.9844] | 17 / 6 |
| B_minus_A | fpr | 6.6758 | 4.8262 | [1.8508, 7.7480] | 17 / 6 |
| D1_minus_A | iou | -4.5250 | 3.2895 | [-12.0639, 22.2614] | 17 / 6 |
| D1_minus_A | mask75 | 11.7647 | 25.0000 | [0.0000, 58.3333] | 17 / 6 |
| D1_minus_A | coverage | 9.4509 | 15.5580 | [2.6159, 31.2317] | 17 / 6 |
| D1_minus_A | auc | -1.7115 | 2.0863 | [-4.0722, 9.6986] | 17 / 6 |
| D1_minus_A | fpr | 27.0021 | 23.3038 | [8.1931, 39.3688] | 17 / 6 |
| D_minus_D1 | iou | 2.0050 | -6.0405 | [-23.5841, 7.5301] | 17 / 6 |
| D_minus_D1 | mask75 | -5.8824 | -19.4444 | [-58.3333, 11.1111] | 17 / 6 |
| D_minus_D1 | coverage | -9.5087 | -18.8152 | [-33.7947, -5.7378] | 17 / 6 |
| D_minus_D1 | auc | 1.1697 | -1.0694 | [-5.1123, 2.8343] | 17 / 6 |
| D_minus_D1 | fpr | -21.2293 | -19.9855 | [-28.8183, -11.1800] | 17 / 6 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 17 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 17 | 0 / 0 | 11.7647 / 25.0000 | NA / NA | 2 | 25.0000 [0.0000, 58.3333] |
| D | 1 / 17 | 0 / 0 | 5.8824 / 5.5556 | NA / NA | 1 | 5.5556 [0.0000, 16.6667] |

## baseline_success

41 candidates; 9 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 87.4724 / 88.7737 | 100.0000 / 100.0000 | 96.7805 / 96.6865 | 98.1774 / 98.6273 | 22.6275 / 15.5934 |
| B | 86.3188 / 87.5657 | 92.6829 / 95.3175 | 97.2898 / 96.9235 | 98.2291 / 98.5766 | 26.3285 / 18.5174 |
| D1 | 77.2498 / 77.2631 | 70.7317 / 66.2169 | 97.6769 / 97.4868 | 96.6225 / 96.6949 | 46.4362 / 41.8610 |
| D | 83.8124 / 83.3264 | 82.9268 / 81.8254 | 96.7223 / 96.7214 | 97.6572 / 97.7136 | 27.4918 / 25.4555 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.5064 | -4.2393 | [-8.2830, -0.6285] | 41 / 9 |
| D_minus_B | mask75 | -9.7561 | -13.4921 | [-35.7143, 0.0000] | 41 / 9 |
| D_minus_B | coverage | -0.5676 | -0.2021 | [-0.9333, 0.4770] | 41 / 9 |
| D_minus_B | auc | -0.5719 | -0.8630 | [-1.5954, -0.2626] | 41 / 9 |
| D_minus_B | fpr | 1.1633 | 6.9381 | [-1.5214, 16.1984] | 41 / 9 |
| D_minus_A | iou | -3.6600 | -5.4473 | [-9.9038, -1.6118] | 41 / 9 |
| D_minus_A | mask75 | -17.0732 | -18.1746 | [-41.5079, -2.7778] | 41 / 9 |
| D_minus_A | coverage | -0.0583 | 0.0349 | [-0.6249, 0.6895] | 41 / 9 |
| D_minus_A | auc | -0.5202 | -0.9137 | [-1.7104, -0.2757] | 41 / 9 |
| D_minus_A | fpr | 4.8643 | 9.8621 | [0.8646, 19.9896] | 41 / 9 |
| B_minus_A | iou | -1.1536 | -1.2080 | [-2.6161, -0.2155] | 41 / 9 |
| B_minus_A | mask75 | -7.3171 | -4.6825 | [-10.5556, -0.7937] | 41 / 9 |
| B_minus_A | coverage | 0.5093 | 0.2370 | [-0.0219, 0.5293] | 41 / 9 |
| B_minus_A | auc | 0.0517 | -0.0506 | [-0.1933, 0.0656] | 41 / 9 |
| B_minus_A | fpr | 3.7011 | 2.9241 | [0.6548, 5.7943] | 41 / 9 |
| D1_minus_A | iou | -10.2225 | -11.5106 | [-17.0932, -6.4021] | 41 / 9 |
| D1_minus_A | mask75 | -29.2683 | -33.7831 | [-60.3704, -11.2679] | 41 / 9 |
| D1_minus_A | coverage | 0.8963 | 0.8003 | [0.1749, 1.4448] | 41 / 9 |
| D1_minus_A | auc | -1.5549 | -1.9324 | [-2.8962, -1.0450] | 41 / 9 |
| D1_minus_A | fpr | 23.8088 | 26.2677 | [15.2118, 37.6698] | 41 / 9 |
| D_minus_D1 | iou | 6.5626 | 6.0633 | [3.2150, 9.2714] | 41 / 9 |
| D_minus_D1 | mask75 | 12.1951 | 15.6085 | [0.0000, 37.8307] | 41 / 9 |
| D_minus_D1 | coverage | -0.9546 | -0.7654 | [-0.9678, -0.5383] | 41 / 9 |
| D_minus_D1 | auc | 1.0347 | 1.0187 | [0.6188, 1.5931] | 41 / 9 |
| D_minus_D1 | fpr | -18.9444 | -16.4055 | [-23.2244, -9.5944] | 41 / 9 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 3 / 41 | NA / NA | 7.3171 / 4.6825 | -3 | -4.6825 [-10.5556, -0.7937] |
| D1 | 0 / 0 | 12 / 41 | NA / NA | 29.2683 / 33.7831 | -12 | -33.7831 [-60.3704, -11.2679] |
| D | 0 / 0 | 7 / 41 | NA / NA | 17.0732 / 18.1746 | -7 | -18.1746 [-41.5079, -2.7778] |

## all/size/small

63 candidates; 4 eligible images; 6 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 65.4911 / 66.7377 | 28.5714 / 24.3636 | 91.2858 / 91.6939 | 91.7555 / 92.3026 | 50.5384 / 46.1190 |
| B | 62.5200 / 64.3819 | 20.6349 / 18.0909 | 92.4466 / 93.0227 | 91.5279 / 92.3443 | 59.5978 / 53.8336 |
| D1 | 60.1412 / 62.6407 | 22.2222 / 19.0909 | 91.9951 / 92.1955 | 89.1014 / 90.6444 | 66.9071 / 58.8460 |
| D | 63.1656 / 63.8779 | 25.3968 / 23.6364 | 90.6362 / 87.1325 | 90.1900 / 90.9963 | 53.9993 / 45.5017 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 0.6456 | -0.5040 | [-3.2074, 1.7590] | 63 / 4 |
| D_minus_B | mask75 | 4.7619 | 5.5455 | [0.0000, 13.6364] | 63 / 4 |
| D_minus_B | coverage | -1.8104 | -5.8902 | [-15.3238, -0.7199] | 63 / 4 |
| D_minus_B | auc | -1.3379 | -1.3480 | [-1.7859, -0.9100] | 63 / 4 |
| D_minus_B | fpr | -5.5985 | -8.3319 | [-13.3163, -1.6927] | 63 / 4 |
| D_minus_A | iou | -2.3255 | -2.8599 | [-5.3536, -0.3661] | 63 / 4 |
| D_minus_A | mask75 | -3.1746 | -0.7273 | [-6.0000, 5.8182] | 63 / 4 |
| D_minus_A | coverage | -0.6495 | -4.5615 | [-13.5113, 0.6111] | 63 / 4 |
| D_minus_A | auc | -1.5655 | -1.3063 | [-2.3359, -0.6558] | 63 / 4 |
| D_minus_A | fpr | 3.4609 | -0.6173 | [-7.2789, 9.0905] | 63 / 4 |
| B_minus_A | iou | -2.9711 | -2.3559 | [-3.8901, -1.0525] | 63 / 4 |
| B_minus_A | mask75 | -7.9365 | -6.2727 | [-8.8182, -2.0000] | 63 / 4 |
| B_minus_A | coverage | 1.1608 | 1.3288 | [0.5909, 2.0666] | 63 / 4 |
| B_minus_A | auc | -0.2276 | 0.0417 | [-0.5386, 0.4914] | 63 / 4 |
| B_minus_A | fpr | 9.0594 | 7.7146 | [4.5892, 10.7984] | 63 / 4 |
| D1_minus_A | iou | -5.3499 | -4.0971 | [-7.6678, -0.5264] | 63 / 4 |
| D1_minus_A | mask75 | -6.3492 | -5.2727 | [-8.5455, -2.0000] | 63 / 4 |
| D1_minus_A | coverage | 0.7093 | 0.5016 | [-0.0592, 1.0625] | 63 / 4 |
| D1_minus_A | auc | -2.6541 | -1.6582 | [-3.7628, 0.4104] | 63 / 4 |
| D1_minus_A | fpr | 16.3687 | 12.7270 | [2.5052, 22.9489] | 63 / 4 |
| D_minus_D1 | iou | 3.0244 | 1.2372 | [-3.6788, 4.4637] | 63 / 4 |
| D_minus_D1 | mask75 | 3.1746 | 4.5455 | [0.0000, 13.6364] | 63 / 4 |
| D_minus_D1 | coverage | -1.3589 | -5.0631 | [-13.5182, -0.3966] | 63 / 4 |
| D_minus_D1 | auc | 1.0886 | 0.3519 | [-1.3375, 1.4320] | 63 / 4 |
| D_minus_D1 | fpr | -12.9078 | -13.3443 | [-21.5847, -8.0659] | 63 / 4 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 45 | 5 / 18 | 0.0000 / 0.0000 | 27.7778 / 28.3333 | -5 | -6.2727 [-8.8182, -2.0000] |
| D1 | 0 / 45 | 4 / 18 | 0.0000 / 0.0000 | 22.2222 / 21.6667 | -4 | -5.2727 [-8.5455, -2.0000] |
| D | 2 / 45 | 4 / 18 | 4.4444 / 8.3333 | 22.2222 / 21.6667 | -2 | -0.7273 [-6.0000, 5.8182] |

## all/size/medium

20 candidates; 5 eligible images; 5 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 82.2806 / 83.4687 | 75.0000 / 74.5455 | 92.5244 / 94.1921 | 97.1886 / 98.0811 | 16.0730 / 16.0036 |
| B | 81.3551 / 82.9151 | 75.0000 / 82.7273 | 94.3046 / 95.2259 | 97.1764 / 97.9425 | 19.3461 / 18.1496 |
| D1 | 67.1140 / 73.9321 | 45.0000 / 63.6364 | 96.0884 / 95.2830 | 95.2932 / 96.1844 | 51.4018 / 38.0915 |
| D | 77.7983 / 81.0757 | 55.0000 / 67.2727 | 94.1847 / 93.7957 | 97.0040 / 97.5050 | 24.7955 / 19.3538 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.5568 | -1.8394 | [-3.9738, -0.0945] | 20 / 5 |
| D_minus_B | mask75 | -20.0000 | -15.4545 | [-35.4545, 0.0000] | 20 / 5 |
| D_minus_B | coverage | -0.1200 | -1.4302 | [-4.9349, 0.7603] | 20 / 5 |
| D_minus_B | auc | -0.1724 | -0.4375 | [-1.2445, 0.0612] | 20 / 5 |
| D_minus_B | fpr | 5.4494 | 1.2042 | [-3.2088, 6.0733] | 20 / 5 |
| D_minus_A | iou | -4.4823 | -2.3930 | [-4.9811, -0.2619] | 20 / 5 |
| D_minus_A | mask75 | -20.0000 | -7.2727 | [-21.8182, 0.0000] | 20 / 5 |
| D_minus_A | coverage | 1.6603 | -0.3964 | [-3.6027, 2.4222] | 20 / 5 |
| D_minus_A | auc | -0.1846 | -0.5762 | [-1.7546, 0.1119] | 20 / 5 |
| D_minus_A | fpr | 8.7225 | 3.3502 | [-0.9251, 9.1359] | 20 / 5 |
| B_minus_A | iou | -0.9255 | -0.5536 | [-0.9901, -0.1675] | 20 / 5 |
| B_minus_A | mask75 | 0.0000 | 8.1818 | [-5.4545, 30.0000] | 20 / 5 |
| B_minus_A | coverage | 1.7803 | 1.0338 | [0.0162, 2.1349] | 20 / 5 |
| B_minus_A | auc | -0.0122 | -0.1387 | [-0.5016, 0.0755] | 20 / 5 |
| B_minus_A | fpr | 3.2731 | 2.1460 | [0.5883, 3.7037] | 20 / 5 |
| D1_minus_A | iou | -15.1666 | -9.5366 | [-16.7602, -2.9750] | 20 / 5 |
| D1_minus_A | mask75 | -30.0000 | -10.9091 | [-32.7273, 0.0000] | 20 / 5 |
| D1_minus_A | coverage | 3.5640 | 1.0909 | [-0.9101, 3.8556] | 20 / 5 |
| D1_minus_A | auc | -1.8955 | -1.8967 | [-4.2855, -0.2914] | 20 / 5 |
| D1_minus_A | fpr | 35.3288 | 22.0880 | [4.8166, 37.7168] | 20 / 5 |
| D_minus_D1 | iou | 10.6843 | 7.1436 | [2.6857, 11.8937] | 20 / 5 |
| D_minus_D1 | mask75 | 10.0000 | 3.6364 | [0.0000, 10.9091] | 20 / 5 |
| D_minus_D1 | coverage | -1.9037 | -1.4874 | [-3.3092, 0.1606] | 20 / 5 |
| D_minus_D1 | auc | 1.7108 | 1.3205 | [0.2055, 2.7016] | 20 / 5 |
| D_minus_D1 | fpr | -26.6063 | -18.7377 | [-28.0488, -5.2629] | 20 / 5 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 1 / 5 | 1 / 15 | 20.0000 / 25.0000 | 6.6667 / 3.1250 | 0 | 8.1818 [-5.4545, 30.0000] |
| D1 | 0 / 5 | 6 / 15 | 0.0000 / 0.0000 | 40.0000 / 18.7500 | -6 | -10.9091 [-32.7273, 0.0000] |
| D | 0 / 5 | 4 / 15 | 0.0000 / 0.0000 | 26.6667 / 12.5000 | -4 | -7.2727 [-21.8182, 0.0000] |

## all/size/large

19 candidates; 8 eligible images; 2 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 81.9158 / 82.7379 | 73.6842 / 75.0000 | 89.2770 / 90.0170 | 94.4745 / 95.9242 | 11.7903 / 10.1663 |
| B | 81.7823 / 82.5945 | 73.6842 / 75.0000 | 89.6880 / 90.5316 | 94.3536 / 95.7808 | 12.9219 / 11.3250 |
| D1 | 74.7913 / 75.0412 | 52.6316 / 56.2500 | 96.5394 / 96.0005 | 93.8627 / 94.7635 | 38.3876 / 33.8355 |
| D | 78.3797 / 78.6792 | 68.4211 / 68.7500 | 90.3897 / 91.2047 | 94.4075 / 95.4503 | 20.0493 / 19.5077 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.4027 | -3.9153 | [-7.9102, -0.4184] | 19 / 8 |
| D_minus_B | mask75 | -5.2632 | -6.2500 | [-18.7500, 0.0000] | 19 / 8 |
| D_minus_B | coverage | 0.7016 | 0.6732 | [-2.3313, 3.9586] | 19 / 8 |
| D_minus_B | auc | 0.0539 | -0.3305 | [-1.3345, 0.5895] | 19 / 8 |
| D_minus_B | fpr | 7.1274 | 8.1826 | [1.9996, 15.9806] | 19 / 8 |
| D_minus_A | iou | -3.5362 | -4.0587 | [-8.2339, -0.1993] | 19 / 8 |
| D_minus_A | mask75 | -5.2632 | -6.2500 | [-18.7500, 0.0000] | 19 / 8 |
| D_minus_A | coverage | 1.1127 | 1.1878 | [-1.5286, 4.5830] | 19 / 8 |
| D_minus_A | auc | -0.0669 | -0.4739 | [-1.6902, 0.7275] | 19 / 8 |
| D_minus_A | fpr | 8.2590 | 9.3414 | [2.7765, 16.9975] | 19 / 8 |
| B_minus_A | iou | -0.1335 | -0.1434 | [-1.1589, 0.5695] | 19 / 8 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 19 / 8 |
| B_minus_A | coverage | 0.4111 | 0.5146 | [0.0405, 1.0610] | 19 / 8 |
| B_minus_A | auc | -0.1208 | -0.1434 | [-0.5543, 0.2418] | 19 / 8 |
| B_minus_A | fpr | 1.1316 | 1.1587 | [0.1230, 2.7906] | 19 / 8 |
| D1_minus_A | iou | -7.1245 | -7.6966 | [-12.1903, -2.7857] | 19 / 8 |
| D1_minus_A | mask75 | -21.0526 | -18.7500 | [-46.8750, 3.1250] | 19 / 8 |
| D1_minus_A | coverage | 7.2624 | 5.9836 | [2.1354, 10.8969] | 19 / 8 |
| D1_minus_A | auc | -0.6117 | -1.1607 | [-2.8762, 0.5923] | 19 / 8 |
| D1_minus_A | fpr | 26.5973 | 23.6692 | [13.9738, 32.5053] | 19 / 8 |
| D_minus_D1 | iou | 3.5884 | 3.6380 | [-0.2253, 7.9026] | 19 / 8 |
| D_minus_D1 | mask75 | 15.7895 | 12.5000 | [-3.1250, 31.2500] | 19 / 8 |
| D_minus_D1 | coverage | -6.1498 | -4.7958 | [-8.4057, -1.7546] | 19 / 8 |
| D_minus_D1 | auc | 0.5448 | 0.6868 | [0.1036, 1.3437] | 19 / 8 |
| D_minus_D1 | fpr | -18.3383 | -14.3278 | [-21.0483, -7.4525] | 19 / 8 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 5 | 0 / 14 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 5 | 6 / 14 | 40.0000 / 37.5000 | 42.8571 / 33.3333 | -4 | -18.7500 [-46.8750, 3.1250] |
| D | 0 / 5 | 1 / 14 | 0.0000 / 0.0000 | 7.1429 / 7.1429 | -1 | -6.2500 [-18.7500, 0.0000] |

## all/pyramid/P3

48 candidates; 3 eligible images; 7 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 65.2384 / 66.9975 | 29.1667 / 32.7778 | 90.9645 / 91.7713 | 91.1603 / 91.7097 | 59.2295 / 60.4738 |
| B | 62.8772 / 64.6072 | 20.8333 / 24.2593 | 91.7917 / 92.5783 | 91.0268 / 91.6442 | 68.7528 / 70.1167 |
| D1 | 61.0332 / 62.4406 | 22.9167 / 25.9259 | 91.3676 / 92.2667 | 88.7482 / 89.3558 | 76.2613 / 78.3695 |
| D | 63.6690 / 65.3541 | 27.0833 / 32.5926 | 90.9287 / 91.7720 | 89.7487 / 90.3738 | 63.1480 / 63.2071 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 0.7918 | 0.7468 | [-0.8003, 2.2003] | 48 / 3 |
| D_minus_B | mask75 | 6.2500 | 8.3333 | [0.0000, 20.0000] | 48 / 3 |
| D_minus_B | coverage | -0.8629 | -0.8063 | [-1.5884, -0.1686] | 48 / 3 |
| D_minus_B | auc | -1.2781 | -1.2704 | [-1.9986, -0.7246] | 48 / 3 |
| D_minus_B | fpr | -5.6048 | -6.9096 | [-15.1311, 2.8814] | 48 / 3 |
| D_minus_A | iou | -1.5694 | -1.6434 | [-3.9654, 0.5693] | 48 / 3 |
| D_minus_A | mask75 | -2.0833 | -0.1852 | [-5.5556, 10.0000] | 48 / 3 |
| D_minus_A | coverage | -0.0358 | 0.0007 | [-1.0618, 1.1098] | 48 / 3 |
| D_minus_A | auc | -1.4116 | -1.3359 | [-3.0447, -0.3322] | 48 / 3 |
| D_minus_A | fpr | 3.9185 | 2.7333 | [-5.5880, 15.8953] | 48 / 3 |
| B_minus_A | iou | -2.3612 | -2.3902 | [-3.1651, -1.6310] | 48 / 3 |
| B_minus_A | mask75 | -8.3333 | -8.5185 | [-10.0000, -5.5556] | 48 / 3 |
| B_minus_A | coverage | 0.8271 | 0.8070 | [0.5266, 1.2784] | 48 / 3 |
| B_minus_A | auc | -0.1335 | -0.0655 | [-1.0461, 0.4571] | 48 / 3 |
| B_minus_A | fpr | 9.5233 | 9.6429 | [6.3718, 13.0139] | 48 / 3 |
| D1_minus_A | iou | -4.2053 | -4.5569 | [-6.0887, -1.6436] | 48 / 3 |
| D1_minus_A | mask75 | -6.2500 | -6.8519 | [-10.0000, -5.0000] | 48 / 3 |
| D1_minus_A | coverage | 0.4031 | 0.4954 | [0.0829, 1.0211] | 48 / 3 |
| D1_minus_A | auc | -2.4121 | -2.3539 | [-4.2143, -1.1450] | 48 / 3 |
| D1_minus_A | fpr | 17.0318 | 17.8957 | [7.1678, 26.1221] | 48 / 3 |
| D_minus_D1 | iou | 2.6358 | 2.9135 | [2.1233, 4.4043] | 48 / 3 |
| D_minus_D1 | mask75 | 4.1667 | 6.6667 | [0.0000, 20.0000] | 48 / 3 |
| D_minus_D1 | coverage | -0.4389 | -0.4947 | [-1.4441, 1.0269] | 48 / 3 |
| D_minus_D1 | auc | 1.0005 | 1.0180 | [0.8128, 1.1696] | 48 / 3 |
| D_minus_D1 | fpr | -13.1133 | -15.1624 | [-25.9852, -9.2751] | 48 / 3 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 34 | 4 / 14 | 0.0000 / 0.0000 | 28.5714 / 34.4444 | -4 | -8.5185 [-10.0000, -5.5556] |
| D1 | 0 / 34 | 3 / 14 | 0.0000 / 0.0000 | 21.4286 / 23.3333 | -3 | -6.8519 [-10.0000, -5.0000] |
| D | 2 / 34 | 3 / 14 | 5.8824 / 13.3333 | 21.4286 / 23.3333 | -1 | -0.1852 [-5.5556, 10.0000] |

## all/pyramid/P4

31 candidates; 6 eligible images; 4 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 76.0264 / 78.7618 | 54.8387 / 53.5417 | 92.8211 / 92.1564 | 95.3732 / 96.4778 | 20.0187 / 18.1717 |
| B | 73.2986 / 77.6763 | 51.6129 / 68.1250 | 94.5991 / 93.6923 | 95.1500 / 96.1416 | 25.4234 / 21.9060 |
| D1 | 64.0389 / 70.9123 | 35.4839 / 47.2917 | 95.0845 / 92.3987 | 92.9568 / 93.9424 | 46.6850 / 34.8753 |
| D | 71.0583 / 75.5124 | 38.7097 / 48.3333 | 92.3184 / 88.9758 | 94.4092 / 95.3713 | 26.4023 / 19.4332 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.2404 | -2.1640 | [-5.8084, 0.9055] | 31 / 6 |
| D_minus_B | mask75 | -12.9032 | -19.7917 | [-53.1250, 0.0000] | 31 / 6 |
| D_minus_B | coverage | -2.2808 | -4.7165 | [-9.4527, -0.6596] | 31 / 6 |
| D_minus_B | auc | -0.7408 | -0.7703 | [-1.2418, -0.3129] | 31 / 6 |
| D_minus_B | fpr | 0.9789 | -2.4728 | [-5.9428, 1.4024] | 31 / 6 |
| D_minus_A | iou | -4.9682 | -3.2494 | [-6.3108, -0.2709] | 31 / 6 |
| D_minus_A | mask75 | -16.1290 | -5.2083 | [-15.6250, 0.0000] | 31 / 6 |
| D_minus_A | coverage | -0.5027 | -3.1806 | [-6.9261, 0.3248] | 31 / 6 |
| D_minus_A | auc | -0.9640 | -1.1065 | [-1.9760, -0.2730] | 31 / 6 |
| D_minus_A | fpr | 6.3836 | 1.2615 | [-2.2202, 6.3873] | 31 / 6 |
| B_minus_A | iou | -2.7278 | -1.0855 | [-2.5908, 0.4312] | 31 / 6 |
| B_minus_A | mask75 | -3.2258 | 14.5833 | [-6.2500, 50.0000] | 31 / 6 |
| B_minus_A | coverage | 1.7781 | 1.5359 | [0.3646, 2.9534] | 31 / 6 |
| B_minus_A | auc | -0.2232 | -0.3362 | [-0.8318, 0.1058] | 31 / 6 |
| B_minus_A | fpr | 5.4048 | 3.7343 | [1.8635, 5.5962] | 31 / 6 |
| D1_minus_A | iou | -11.9875 | -7.8495 | [-13.8009, -2.4551] | 31 / 6 |
| D1_minus_A | mask75 | -19.3548 | -6.2500 | [-18.7500, 0.0000] | 31 / 6 |
| D1_minus_A | coverage | 2.2635 | 0.2423 | [-1.0581, 2.0974] | 31 / 6 |
| D1_minus_A | auc | -2.4165 | -2.5354 | [-5.2120, -0.5342] | 31 / 6 |
| D1_minus_A | fpr | 26.6663 | 16.7036 | [3.8965, 29.0314] | 31 / 6 |
| D_minus_D1 | iou | 7.0193 | 4.6001 | [1.2664, 7.9339] | 31 / 6 |
| D_minus_D1 | mask75 | 3.2258 | 1.0417 | [0.0000, 3.1250] | 31 / 6 |
| D_minus_D1 | coverage | -2.7662 | -3.4229 | [-6.5995, -0.4991] | 31 / 6 |
| D_minus_D1 | auc | 1.4525 | 1.4289 | [-0.0384, 3.3090] | 31 / 6 |
| D_minus_D1 | fpr | -20.2827 | -15.4421 | [-23.9092, -5.2409] | 31 / 6 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 1 / 14 | 2 / 17 | 7.1429 / 20.0000 | 11.7647 / 4.4444 | -1 | 14.5833 [-6.2500, 50.0000] |
| D1 | 0 / 14 | 6 / 17 | 0.0000 / 0.0000 | 35.2941 / 13.3333 | -6 | -6.2500 [-18.7500, 0.0000] |
| D | 0 / 14 | 5 / 17 | 0.0000 / 0.0000 | 29.4118 / 11.1111 | -5 | -5.2083 [-15.6250, 0.0000] |

## all/pyramid/P5

23 candidates; 8 eligible images; 2 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 79.9864 / 80.2658 | 69.5652 / 70.8333 | 89.3045 / 90.4409 | 95.0920 / 95.9160 | 11.5565 / 11.3478 |
| B | 79.5375 / 79.5957 | 69.5652 / 70.8333 | 90.2491 / 91.2913 | 94.9375 / 95.6960 | 12.9928 / 12.9905 |
| D1 | 71.1919 / 70.2227 | 47.8261 / 47.9167 | 96.4540 / 96.4389 | 93.9595 / 94.1306 | 37.5986 / 37.4466 |
| D | 76.7694 / 76.0877 | 65.2174 / 64.5833 | 90.6405 / 91.7703 | 94.8333 / 95.1573 | 18.6620 / 20.6482 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.7681 | -3.5080 | [-7.5398, 0.0472] | 23 / 8 |
| D_minus_B | mask75 | -4.3478 | -6.2500 | [-18.7500, 0.0000] | 23 / 8 |
| D_minus_B | coverage | 0.3914 | 0.4790 | [-0.8883, 1.7909] | 23 / 8 |
| D_minus_B | auc | -0.1043 | -0.5387 | [-1.5327, 0.4371] | 23 / 8 |
| D_minus_B | fpr | 5.6691 | 7.6578 | [1.5530, 15.7056] | 23 / 8 |
| D_minus_A | iou | -3.2171 | -4.1781 | [-8.2823, -0.4325] | 23 / 8 |
| D_minus_A | mask75 | -4.3478 | -6.2500 | [-18.7500, 0.0000] | 23 / 8 |
| D_minus_A | coverage | 1.3360 | 1.3295 | [0.3088, 2.4473] | 23 / 8 |
| D_minus_A | auc | -0.2588 | -0.7587 | [-1.9726, 0.5328] | 23 / 8 |
| D_minus_A | fpr | 7.1055 | 9.3004 | [3.0497, 16.9893] | 23 / 8 |
| B_minus_A | iou | -0.4489 | -0.6702 | [-1.5873, 0.0527] | 23 / 8 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 23 / 8 |
| B_minus_A | coverage | 0.9446 | 0.8504 | [0.0361, 2.1855] | 23 / 8 |
| B_minus_A | auc | -0.1545 | -0.2200 | [-0.6094, 0.1615] | 23 / 8 |
| B_minus_A | fpr | 1.4364 | 1.6427 | [0.3589, 3.3177] | 23 / 8 |
| D1_minus_A | iou | -8.7945 | -10.0432 | [-14.6056, -4.5425] | 23 / 8 |
| D1_minus_A | mask75 | -21.7391 | -22.9167 | [-50.0000, 0.0000] | 23 / 8 |
| D1_minus_A | coverage | 7.1495 | 5.9980 | [2.0874, 10.2392] | 23 / 8 |
| D1_minus_A | auc | -1.1326 | -1.7854 | [-3.5201, 0.1652] | 23 / 8 |
| D1_minus_A | fpr | 26.0421 | 26.0988 | [18.0258, 32.8302] | 23 / 8 |
| D_minus_D1 | iou | 5.5775 | 5.8650 | [1.6385, 10.2539] | 23 / 8 |
| D_minus_D1 | mask75 | 17.3913 | 16.6667 | [-0.0260, 33.3333] | 23 / 8 |
| D_minus_D1 | coverage | -5.8135 | -4.6686 | [-8.3122, -1.4385] | 23 / 8 |
| D_minus_D1 | auc | 0.8738 | 1.0267 | [0.2341, 1.8641] | 23 / 8 |
| D_minus_D1 | fpr | -18.9366 | -16.7983 | [-22.8665, -9.9623] | 23 / 8 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 7 | 0 / 16 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 7 | 7 / 16 | 28.5714 / 30.0000 | 43.7500 / 40.4762 | -5 | -22.9167 [-50.0000, 0.0000] |
| D | 0 / 7 | 1 / 16 | 0.0000 / 0.0000 | 6.2500 / 7.1429 | -1 | -6.2500 [-18.7500, 0.0000] |

## tal_maskfail/size/small

20 candidates; 4 eligible images; 6 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 65.4852 / 64.7878 | 0.0000 / 0.0000 | 92.7916 / 93.8829 | 93.9913 / 93.6825 | 40.0751 / 38.4241 |
| B | 61.5298 / 61.7117 | 0.0000 / 0.0000 | 94.5803 / 95.7869 | 93.5491 / 93.6155 | 48.2412 / 45.4937 |
| D1 | 56.1782 / 57.9435 | 0.0000 / 0.0000 | 94.6690 / 95.5739 | 91.2961 / 92.0751 | 63.2845 / 56.8635 |
| D | 61.1754 / 59.5540 | 5.0000 / 6.2500 | 91.5143 / 86.0315 | 92.6198 / 92.1290 | 46.4939 / 40.7147 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.3544 | -2.1576 | [-6.8716, 1.2584] | 20 / 4 |
| D_minus_B | mask75 | 5.0000 | 6.2500 | [0.0000, 18.7500] | 20 / 4 |
| D_minus_B | coverage | -3.0660 | -9.7554 | [-26.5570, -0.1590] | 20 / 4 |
| D_minus_B | auc | -0.9294 | -1.4865 | [-3.2943, -0.1464] | 20 / 4 |
| D_minus_B | fpr | -1.7473 | -4.7790 | [-11.8931, 1.6276] | 20 / 4 |
| D_minus_A | iou | -4.3098 | -5.2338 | [-9.3950, -1.0726] | 20 / 4 |
| D_minus_A | mask75 | 5.0000 | 6.2500 | [0.0000, 18.7500] | 20 / 4 |
| D_minus_A | coverage | -1.2773 | -7.8514 | [-24.1492, 2.1980] | 20 / 4 |
| D_minus_A | auc | -1.3715 | -1.5535 | [-2.7528, -0.1905] | 20 / 4 |
| D_minus_A | fpr | 6.4188 | 2.2906 | [-7.3974, 10.7837] | 20 / 4 |
| B_minus_A | iou | -3.9554 | -3.0762 | [-5.5235, -1.5311] | 20 / 4 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 20 / 4 |
| B_minus_A | coverage | 1.7887 | 1.9040 | [0.8901, 2.7917] | 20 / 4 |
| B_minus_A | auc | -0.4422 | -0.0670 | [-0.8491, 0.6447] | 20 / 4 |
| B_minus_A | fpr | 8.1661 | 7.0696 | [4.4957, 9.1561] | 20 / 4 |
| D1_minus_A | iou | -9.3070 | -6.8443 | [-13.2509, -0.4377] | 20 / 4 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 20 / 4 |
| D1_minus_A | coverage | 1.8774 | 1.6910 | [-0.1835, 3.5492] | 20 / 4 |
| D1_minus_A | auc | -2.6952 | -1.6075 | [-3.4483, 0.2334] | 20 / 4 |
| D1_minus_A | fpr | 23.2093 | 18.4394 | [3.7612, 33.1177] | 20 / 4 |
| D_minus_D1 | iou | 4.9972 | 1.6105 | [-7.0356, 8.0316] | 20 / 4 |
| D_minus_D1 | mask75 | 5.0000 | 6.2500 | [0.0000, 18.7500] | 20 / 4 |
| D_minus_D1 | coverage | -3.1547 | -9.5423 | [-25.9154, -0.8840] | 20 / 4 |
| D_minus_D1 | auc | 1.3236 | 0.0539 | [-3.0047, 1.9781] | 20 / 4 |
| D_minus_D1 | fpr | -16.7906 | -16.1488 | [-22.3339, -9.9637] | 20 / 4 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 20 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 20 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 1 / 20 | 0 / 0 | 5.0000 / 6.2500 | NA / NA | 1 | 6.2500 [0.0000, 18.7500] |

## tal_maskfail/size/medium

3 candidates; 2 eligible images; 8 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 69.3838 / 70.6435 | 0.0000 / 0.0000 | 76.7574 / 76.5333 | 95.2467 / 95.6420 | 7.8894 / 6.3814 |
| B | 68.8003 / 70.6431 | 33.3333 / 50.0000 | 83.1228 / 82.4756 | 94.7771 / 95.0325 | 12.3504 / 10.6390 |
| D1 | 47.0460 / 50.1499 | 0.0000 / 0.0000 | 89.1757 / 85.5132 | 88.1276 / 88.1604 | 50.6446 / 44.0709 |
| D | 61.4620 / 62.6033 | 0.0000 / 0.0000 | 81.0586 / 77.4226 | 93.8487 / 93.9051 | 18.7990 / 14.2766 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -7.3383 | -8.0398 | [-10.1443, -5.9354] | 3 / 2 |
| D_minus_B | mask75 | -33.3333 | -50.0000 | [-100.0000, 0.0000] | 3 / 2 |
| D_minus_B | coverage | -2.0642 | -5.0531 | [-14.0197, 3.9135] | 3 / 2 |
| D_minus_B | auc | -0.9284 | -1.1274 | [-1.7244, -0.5304] | 3 / 2 |
| D_minus_B | fpr | 6.4487 | 3.6376 | [-4.7957, 12.0708] | 3 / 2 |
| D_minus_A | iou | -7.9218 | -8.0402 | [-8.3954, -7.6850] | 3 / 2 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 2 |
| D_minus_A | coverage | 4.3012 | 0.8893 | [-9.3465, 11.1250] | 3 / 2 |
| D_minus_A | auc | -1.3979 | -1.7369 | [-2.7537, -0.7200] | 3 / 2 |
| D_minus_A | fpr | 10.9096 | 7.8952 | [-1.1483, 16.9386] | 3 / 2 |
| B_minus_A | iou | -0.5835 | -0.0004 | [-1.7496, 1.7489] | 3 / 2 |
| B_minus_A | mask75 | 33.3333 | 50.0000 | [0.0000, 100.0000] | 3 / 2 |
| B_minus_A | coverage | 6.3654 | 5.9424 | [4.6732, 7.2115] | 3 / 2 |
| B_minus_A | auc | -0.4695 | -0.6095 | [-1.0293, -0.1896] | 3 / 2 |
| B_minus_A | fpr | 4.4610 | 4.2576 | [3.6474, 4.8677] | 3 / 2 |
| D1_minus_A | iou | -22.3378 | -20.4935 | [-26.0262, -14.9609] | 3 / 2 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 2 |
| D1_minus_A | coverage | 12.4183 | 8.9799 | [-1.3352, 19.2951] | 3 / 2 |
| D1_minus_A | auc | -7.1191 | -7.4816 | [-8.5691, -6.3940] | 3 / 2 |
| D1_minus_A | fpr | 42.7552 | 37.6895 | [22.4924, 52.8866] | 3 / 2 |
| D_minus_D1 | iou | 14.4159 | 12.4533 | [6.5655, 18.3412] | 3 / 2 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 2 |
| D_minus_D1 | coverage | -8.1171 | -8.0907 | [-8.1701, -8.0112] | 3 / 2 |
| D_minus_D1 | auc | 5.7212 | 5.7447 | [5.6740, 5.8154] | 3 / 2 |
| D_minus_D1 | fpr | -31.8456 | -29.7944 | [-35.9481, -23.6407] | 3 / 2 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 1 / 3 | 0 / 0 | 33.3333 / 50.0000 | NA / NA | 1 | 50.0000 [0.0000, 100.0000] |
| D1 | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## tal_maskfail/size/large

5 candidates; 4 eligible images; 6 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 57.9342 / 57.5709 | 0.0000 / 0.0000 | 67.9381 / 68.7840 | 82.2919 / 84.3988 | 22.1202 / 20.1689 |
| B | 58.4955 / 57.9859 | 0.0000 / 0.0000 | 68.8652 / 69.7274 | 82.3332 / 84.1074 | 23.4853 / 21.3029 |
| D1 | 66.1705 / 65.2216 | 40.0000 / 37.5000 | 93.0331 / 92.5997 | 86.4602 / 88.2347 | 51.7747 / 47.6133 |
| D | 59.5824 / 58.3941 | 0.0000 / 0.0000 | 73.3407 / 73.8715 | 84.3534 / 86.0598 | 29.8007 / 27.3395 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 1.0869 | 0.4082 | [-3.1741, 3.1226] | 5 / 4 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 4 |
| D_minus_B | coverage | 4.4756 | 4.1441 | [0.5081, 7.7800] | 5 / 4 |
| D_minus_B | auc | 2.0202 | 1.9524 | [-1.8929, 6.7367] | 5 / 4 |
| D_minus_B | fpr | 6.3154 | 6.0366 | [1.8149, 9.5628] | 5 / 4 |
| D_minus_A | iou | 1.6482 | 0.8232 | [-3.1160, 4.2232] | 5 / 4 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 4 |
| D_minus_A | coverage | 5.4026 | 5.0875 | [1.2949, 8.9534] | 5 / 4 |
| D_minus_A | auc | 2.0615 | 1.6611 | [-4.3919, 7.5025] | 5 / 4 |
| D_minus_A | fpr | 7.6805 | 7.1706 | [1.8602, 11.6957] | 5 / 4 |
| B_minus_A | iou | 0.5613 | 0.4150 | [-0.2706, 1.1006] | 5 / 4 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 4 |
| B_minus_A | coverage | 0.9270 | 0.9434 | [0.0574, 1.6729] | 5 / 4 |
| B_minus_A | auc | 0.0413 | -0.2914 | [-2.6156, 1.1389] | 5 / 4 |
| B_minus_A | fpr | 1.3651 | 1.1340 | [-0.2726, 2.5406] | 5 / 4 |
| D1_minus_A | iou | 8.2362 | 7.6507 | [-12.8607, 31.7027] | 5 / 4 |
| D1_minus_A | mask75 | 40.0000 | 37.5000 | [0.0000, 75.0000] | 5 / 4 |
| D1_minus_A | coverage | 25.0950 | 23.8156 | [8.4491, 39.5742] | 5 / 4 |
| D1_minus_A | auc | 4.1683 | 3.8359 | [-5.2479, 14.3591] | 5 / 4 |
| D1_minus_A | fpr | 29.6545 | 27.4444 | [12.7367, 39.4462] | 5 / 4 |
| D_minus_D1 | iou | -6.5881 | -6.8275 | [-31.6903, 11.9494] | 5 / 4 |
| D_minus_D1 | mask75 | -40.0000 | -37.5000 | [-75.0000, 0.0000] | 5 / 4 |
| D_minus_D1 | coverage | -19.6924 | -18.7282 | [-36.7500, -1.6392] | 5 / 4 |
| D_minus_D1 | auc | -2.1068 | -2.1749 | [-6.8566, 1.2696] | 5 / 4 |
| D_minus_D1 | fpr | -21.9740 | -20.2738 | [-27.8208, -10.3996] | 5 / 4 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 5 | 0 / 0 | 40.0000 / 37.5000 | NA / NA | 2 | 37.5000 [0.0000, 75.0000] |
| D | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## tal_maskfail/pyramid/P3

12 candidates; 3 eligible images; 7 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 67.4502 / 68.3614 | 0.0000 / 0.0000 | 93.8384 / 94.2103 | 93.7865 / 93.8665 | 49.3795 / 51.0593 |
| B | 65.3615 / 66.2976 | 0.0000 / 0.0000 | 94.4794 / 94.9785 | 93.7418 / 93.8807 | 56.0352 / 58.0623 |
| D1 | 60.8445 / 61.0476 | 0.0000 / 0.0000 | 94.2432 / 94.9669 | 92.8625 / 93.0801 | 70.9749 / 74.9944 |
| D | 65.5223 / 66.1845 | 8.3333 / 11.1111 | 93.2521 / 94.0890 | 93.5909 / 93.8004 | 55.7978 / 58.4042 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 0.1608 | -0.1131 | [-1.4672, 1.8197] | 12 / 3 |
| D_minus_B | mask75 | 8.3333 | 11.1111 | [0.0000, 33.3333] | 12 / 3 |
| D_minus_B | coverage | -1.2273 | -0.8896 | [-3.4141, 0.6386] | 12 / 3 |
| D_minus_B | auc | -0.1509 | -0.0803 | [-0.5103, 0.3371] | 12 / 3 |
| D_minus_B | fpr | -0.2374 | 0.3418 | [-3.8948, 3.0565] | 12 / 3 |
| D_minus_A | iou | -1.9279 | -2.1769 | [-4.0151, 0.2363] | 12 / 3 |
| D_minus_A | mask75 | 8.3333 | 11.1111 | [0.0000, 33.3333] | 12 / 3 |
| D_minus_A | coverage | -0.5863 | -0.1213 | [-3.0251, 2.5543] | 12 / 3 |
| D_minus_A | auc | -0.1956 | -0.0661 | [-0.8820, 0.6715] | 12 / 3 |
| D_minus_A | fpr | 6.4183 | 7.3449 | [0.3364, 11.4548] | 12 / 3 |
| B_minus_A | iou | -2.0887 | -2.0638 | [-3.3232, -1.2847] | 12 / 3 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 3 |
| B_minus_A | coverage | 0.6410 | 0.7682 | [0.0000, 1.9157] | 12 / 3 |
| B_minus_A | auc | -0.0447 | 0.0142 | [-0.3717, 0.3344] | 12 / 3 |
| B_minus_A | fpr | 6.6558 | 7.0030 | [4.2312, 8.3984] | 12 / 3 |
| D1_minus_A | iou | -6.6058 | -7.3138 | [-11.0045, -1.2204] | 12 / 3 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 3 |
| D1_minus_A | coverage | 0.4048 | 0.7566 | [-1.0298, 3.1928] | 12 / 3 |
| D1_minus_A | auc | -0.9240 | -0.7864 | [-1.9763, -0.0576] | 12 / 3 |
| D1_minus_A | fpr | 21.5954 | 23.9351 | [5.7317, 33.8071] | 12 / 3 |
| D_minus_D1 | iou | 4.6779 | 5.1368 | [1.4567, 6.9894] | 12 / 3 |
| D_minus_D1 | mask75 | 8.3333 | 11.1111 | [0.0000, 33.3333] | 12 / 3 |
| D_minus_D1 | coverage | -0.9910 | -0.8780 | [-1.9953, 0.0000] | 12 / 3 |
| D_minus_D1 | auc | 0.7284 | 0.7202 | [0.0697, 1.0943] | 12 / 3 |
| D_minus_D1 | fpr | -15.1771 | -16.5902 | [-22.3523, -5.3953] | 12 / 3 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 12 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 12 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 1 / 12 | 0 / 0 | 8.3333 / 11.1111 | NA / NA | 1 | 11.1111 [0.0000, 33.3333] |

## tal_maskfail/pyramid/P4

10 candidates; 5 eligible images; 5 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 64.0783 / 65.7656 | 0.0000 / 0.0000 | 88.9525 / 89.6826 | 94.2987 / 94.2939 | 23.1504 / 23.7797 |
| B | 58.8151 / 62.5556 | 10.0000 / 20.0000 | 92.4890 / 92.4489 | 93.3998 / 93.5732 | 32.2149 / 31.4834 |
| D1 | 50.6980 / 55.6599 | 0.0000 / 0.0000 | 93.6615 / 91.2959 | 88.8721 / 89.7605 | 50.2891 / 43.3903 |
| D | 56.3548 / 59.5598 | 0.0000 / 0.0000 | 87.3748 / 81.7708 | 91.5550 / 91.5448 | 30.6411 / 24.3456 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.4603 | -2.9958 | [-8.3633, 2.3717] | 10 / 5 |
| D_minus_B | mask75 | -10.0000 | -20.0000 | [-60.0000, 0.0000] | 10 / 5 |
| D_minus_B | coverage | -5.1142 | -10.6782 | [-23.7194, -0.9594] | 10 / 5 |
| D_minus_B | auc | -1.8448 | -2.0285 | [-3.1384, -1.0447] | 10 / 5 |
| D_minus_B | fpr | -1.5738 | -7.1378 | [-13.6655, 0.8775] | 10 / 5 |
| D_minus_A | iou | -7.7235 | -6.2057 | [-10.1981, -2.0300] | 10 / 5 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 10 / 5 |
| D_minus_A | coverage | -1.5777 | -7.9118 | [-20.4872, 1.9529] | 10 / 5 |
| D_minus_A | auc | -2.7437 | -2.7491 | [-4.6360, -1.1147] | 10 / 5 |
| D_minus_A | fpr | 7.4907 | 0.5659 | [-7.8595, 10.3286] | 10 / 5 |
| B_minus_A | iou | -5.2632 | -3.2100 | [-5.9485, -0.0622] | 10 / 5 |
| B_minus_A | mask75 | 10.0000 | 20.0000 | [0.0000, 60.0000] | 10 / 5 |
| B_minus_A | coverage | 3.5365 | 2.7664 | [1.0134, 4.5193] | 10 / 5 |
| B_minus_A | auc | -0.8989 | -0.7207 | [-2.1907, 0.4477] | 10 / 5 |
| B_minus_A | fpr | 9.0645 | 7.7037 | [4.4759, 11.1256] | 10 / 5 |
| D1_minus_A | iou | -13.3803 | -10.1056 | [-16.2062, -4.0037] | 10 / 5 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 10 / 5 |
| D1_minus_A | coverage | 4.7091 | 1.6133 | [-1.1139, 5.9585] | 10 / 5 |
| D1_minus_A | auc | -5.4265 | -4.5334 | [-7.3840, -1.4084] | 10 / 5 |
| D1_minus_A | fpr | 27.1388 | 19.6106 | [8.3802, 31.1218] | 10 / 5 |
| D_minus_D1 | iou | 5.6568 | 3.8999 | [-4.2509, 9.5481] | 10 / 5 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 10 / 5 |
| D_minus_D1 | coverage | -6.2867 | -9.5251 | [-22.0233, -1.5876] | 10 / 5 |
| D_minus_D1 | auc | 2.6829 | 1.7842 | [-1.3667, 4.5483] | 10 / 5 |
| D_minus_D1 | fpr | -19.6481 | -19.0447 | [-23.0622, -14.7987] | 10 / 5 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 1 / 10 | 0 / 0 | 10.0000 / 20.0000 | NA / NA | 1 | 20.0000 [0.0000, 60.0000] |
| D1 | 0 / 10 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 10 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## tal_maskfail/pyramid/P5

6 candidates; 5 eligible images; 5 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 59.5567 / 59.5905 | 0.0000 / 0.0000 | 68.3682 / 69.1310 | 84.7668 / 86.9473 | 18.6190 / 16.3578 |
| B | 59.4974 / 59.2901 | 0.0000 / 0.0000 | 71.1096 / 72.2483 | 84.6801 / 86.5689 | 20.7884 / 18.5031 |
| D1 | 59.7399 / 57.6947 | 33.3333 / 30.0000 | 93.0900 / 92.7546 | 86.5891 / 88.0345 | 53.6510 / 50.6972 |
| D | 59.3316 / 58.3308 | 0.0000 / 0.0000 | 74.5653 / 75.2348 | 86.1779 / 87.9080 | 26.5490 / 23.9297 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.1658 | -0.9593 | [-4.3832, 2.5849] | 6 / 5 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 5 |
| D_minus_B | coverage | 3.4557 | 2.9865 | [-0.3382, 6.6867] | 6 / 5 |
| D_minus_B | auc | 1.4978 | 1.3391 | [-1.8135, 5.6777] | 6 / 5 |
| D_minus_B | fpr | 5.7606 | 5.4266 | [2.3786, 8.5270] | 6 / 5 |
| D_minus_A | iou | -0.2250 | -1.2597 | [-5.9313, 3.4120] | 6 / 5 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 5 |
| D_minus_A | coverage | 6.1971 | 6.1039 | [2.3686, 9.8392] | 6 / 5 |
| D_minus_A | auc | 1.4111 | 0.9607 | [-3.8817, 6.3501] | 6 / 5 |
| D_minus_A | fpr | 7.9300 | 7.5719 | [3.2150, 11.2445] | 6 / 5 |
| B_minus_A | iou | -0.0592 | -0.3004 | [-1.7395, 0.8275] | 6 / 5 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 5 |
| B_minus_A | coverage | 2.7414 | 3.1173 | [0.3392, 7.5278] | 6 / 5 |
| B_minus_A | auc | -0.0867 | -0.3785 | [-2.1446, 0.9803] | 6 / 5 |
| B_minus_A | fpr | 2.1694 | 2.1453 | [0.1681, 4.2039] | 6 / 5 |
| D1_minus_A | iou | 0.1833 | -1.8957 | [-24.2359, 26.5041] | 6 / 5 |
| D1_minus_A | mask75 | 33.3333 | 30.0000 | [0.0000, 70.0000] | 6 / 5 |
| D1_minus_A | coverage | 24.7218 | 23.6236 | [11.3304, 37.7019] | 6 / 5 |
| D1_minus_A | auc | 1.8223 | 1.0871 | [-7.4204, 11.0508] | 6 / 5 |
| D1_minus_A | fpr | 35.0320 | 34.3394 | [15.9167, 50.1977] | 6 / 5 |
| D_minus_D1 | iou | -0.4083 | 0.6361 | [-25.6883, 19.7608] | 6 / 5 |
| D_minus_D1 | mask75 | -33.3333 | -30.0000 | [-70.0000, 0.0000] | 6 / 5 |
| D_minus_D1 | coverage | -18.5247 | -17.5198 | [-34.1098, -3.8486] | 6 / 5 |
| D_minus_D1 | auc | -0.4111 | -0.1265 | [-5.6589, 4.5729] | 6 / 5 |
| D_minus_D1 | fpr | -27.1020 | -26.7675 | [-41.1274, -13.6930] | 6 / 5 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 6 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 6 | 0 / 0 | 33.3333 / 30.0000 | NA / NA | 2 | 30.0000 [0.0000, 70.0000] |
| D | 0 / 6 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail/size/small

12 candidates; 4 eligible images; 6 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 62.0644 / 61.9832 | 0.0000 / 0.0000 | 93.6155 / 93.2045 | 92.2396 / 93.5288 | 44.4001 / 38.8232 |
| B | 58.4830 / 57.9491 | 0.0000 / 0.0000 | 94.9080 / 95.0785 | 91.5219 / 93.2515 | 52.7231 / 47.1367 |
| D1 | 54.4165 / 53.3682 | 0.0000 / 0.0000 | 94.7404 / 94.9856 | 88.3493 / 89.9798 | 67.0769 / 63.7693 |
| D | 58.9365 / 55.7342 | 8.3333 / 8.3333 | 90.5633 / 85.5906 | 90.1501 / 91.3848 | 48.1983 / 42.4873 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 0.4536 | -2.2150 | [-6.7256, 1.9569] | 12 / 4 |
| D_minus_B | mask75 | 8.3333 | 8.3333 | [0.0000, 25.0000] | 12 / 4 |
| D_minus_B | coverage | -4.3447 | -9.4879 | [-26.5095, 0.1203] | 12 / 4 |
| D_minus_B | auc | -1.3719 | -1.8667 | [-3.4094, -0.5137] | 12 / 4 |
| D_minus_B | fpr | -4.5248 | -4.6493 | [-12.6069, 3.9273] | 12 / 4 |
| D_minus_A | iou | -3.1278 | -6.2490 | [-12.4265, -0.0715] | 12 / 4 |
| D_minus_A | mask75 | 8.3333 | 8.3333 | [0.0000, 25.0000] | 12 / 4 |
| D_minus_A | coverage | -3.0522 | -7.6139 | [-24.0118, 2.4794] | 12 / 4 |
| D_minus_A | auc | -2.0895 | -2.1441 | [-3.0617, -0.5654] | 12 / 4 |
| D_minus_A | fpr | 3.7983 | 3.6642 | [-8.7929, 18.3267] | 12 / 4 |
| B_minus_A | iou | -3.5814 | -4.0340 | [-8.4618, -1.3436] | 12 / 4 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 4 |
| B_minus_A | coverage | 1.2924 | 1.8740 | [0.7652, 2.5099] | 12 / 4 |
| B_minus_A | auc | -0.7177 | -0.2774 | [-1.1281, 0.5734] | 12 / 4 |
| B_minus_A | fpr | 8.3231 | 8.3135 | [4.2972, 13.9163] | 12 / 4 |
| D1_minus_A | iou | -7.6479 | -8.6150 | [-16.9160, -0.3954] | 12 / 4 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 4 |
| D1_minus_A | coverage | 1.1248 | 1.7811 | [0.2778, 3.2064] | 12 / 4 |
| D1_minus_A | auc | -3.8903 | -3.5490 | [-7.9502, 0.0266] | 12 / 4 |
| D1_minus_A | fpr | 22.6768 | 24.9462 | [4.4569, 46.0112] | 12 / 4 |
| D_minus_D1 | iou | 4.5201 | 2.3660 | [-7.1492, 9.1827] | 12 / 4 |
| D_minus_D1 | mask75 | 8.3333 | 8.3333 | [0.0000, 25.0000] | 12 / 4 |
| D_minus_D1 | coverage | -4.1771 | -9.3949 | [-25.7302, -0.7271] | 12 / 4 |
| D_minus_D1 | auc | 1.8008 | 1.4050 | [-2.9028, 5.7091] | 12 / 4 |
| D_minus_D1 | fpr | -18.8785 | -21.2820 | [-31.9221, -10.6419] | 12 / 4 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 12 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 12 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 1 / 12 | 0 / 0 | 8.3333 / 8.3333 | NA / NA | 1 | 8.3333 [0.0000, 25.0000] |

## strict_maskfail/size/medium

2 candidates; 1 eligible images; 9 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 66.8643 / 66.8643 | 0.0000 / 0.0000 | 77.2057 / 77.2057 | 94.4561 / 94.4561 | 10.9054 / 10.9054 |
| B | 65.1147 / 65.1147 | 0.0000 / 0.0000 | 84.4172 / 84.4172 | 94.2664 / 94.2664 | 15.7731 / 15.7731 |
| D1 | 40.8382 / 40.8382 | 0.0000 / 0.0000 | 96.5008 / 96.5008 | 88.0620 / 88.0620 | 63.7920 / 63.7920 |
| D | 59.1793 / 59.1793 | 0.0000 / 0.0000 | 88.3307 / 88.3307 | 93.7360 / 93.7360 | 27.8439 / 27.8439 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -5.9354 | -5.9354 | [-5.9354, -5.9354] | 2 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_B | coverage | 3.9135 | 3.9135 | [3.9135, 3.9135] | 2 / 1 |
| D_minus_B | auc | -0.5304 | -0.5304 | [-0.5304, -0.5304] | 2 / 1 |
| D_minus_B | fpr | 12.0708 | 12.0708 | [12.0708, 12.0708] | 2 / 1 |
| D_minus_A | iou | -7.6850 | -7.6850 | [-7.6850, -7.6850] | 2 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_A | coverage | 11.1250 | 11.1250 | [11.1250, 11.1250] | 2 / 1 |
| D_minus_A | auc | -0.7200 | -0.7200 | [-0.7200, -0.7200] | 2 / 1 |
| D_minus_A | fpr | 16.9386 | 16.9386 | [16.9386, 16.9386] | 2 / 1 |
| B_minus_A | iou | -1.7496 | -1.7496 | [-1.7496, -1.7496] | 2 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| B_minus_A | coverage | 7.2115 | 7.2115 | [7.2115, 7.2115] | 2 / 1 |
| B_minus_A | auc | -0.1896 | -0.1896 | [-0.1896, -0.1896] | 2 / 1 |
| B_minus_A | fpr | 4.8677 | 4.8677 | [4.8677, 4.8677] | 2 / 1 |
| D1_minus_A | iou | -26.0262 | -26.0262 | [-26.0262, -26.0262] | 2 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D1_minus_A | coverage | 19.2951 | 19.2951 | [19.2951, 19.2951] | 2 / 1 |
| D1_minus_A | auc | -6.3940 | -6.3940 | [-6.3940, -6.3940] | 2 / 1 |
| D1_minus_A | fpr | 52.8866 | 52.8866 | [52.8866, 52.8866] | 2 / 1 |
| D_minus_D1 | iou | 18.3412 | 18.3412 | [18.3412, 18.3412] | 2 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_D1 | coverage | -8.1701 | -8.1701 | [-8.1701, -8.1701] | 2 / 1 |
| D_minus_D1 | auc | 5.6740 | 5.6740 | [5.6740, 5.6740] | 2 / 1 |
| D_minus_D1 | fpr | -35.9481 | -35.9481 | [-35.9481, -35.9481] | 2 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail/size/large

3 candidates; 2 eligible images; 8 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 52.2857 / 48.7347 | 0.0000 / 0.0000 | 56.0500 / 51.7977 | 74.4475 / 74.7392 | 22.2802 / 18.4578 |
| B | 52.9609 / 49.1743 | 0.0000 / 0.0000 | 56.5011 / 52.0437 | 75.6642 / 75.8781 | 23.5725 / 19.2513 |
| D1 | 74.5865 / 76.8967 | 66.6667 / 75.0000 | 92.2422 / 90.9799 | 84.5729 / 87.1783 | 49.3271 / 39.7806 |
| D | 55.6402 / 51.2924 | 0.0000 / 0.0000 | 60.5144 / 55.1627 | 80.2151 / 81.5589 | 28.5072 / 22.9381 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 2.6793 | 2.1181 | [0.4344, 3.8017] | 3 / 2 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 2 |
| D_minus_B | coverage | 4.0132 | 3.1190 | [0.4364, 5.8017] | 3 / 2 |
| D_minus_B | auc | 4.5509 | 5.6808 | [2.2912, 9.0704] | 3 / 2 |
| D_minus_B | fpr | 4.9347 | 3.6868 | [-0.0570, 7.4305] | 3 / 2 |
| D_minus_A | iou | 3.3545 | 2.5577 | [0.1673, 4.9481] | 3 / 2 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 2 |
| D_minus_A | coverage | 4.4644 | 3.3650 | [0.0667, 6.6633] | 3 / 2 |
| D_minus_A | auc | 5.7676 | 6.8197 | [3.6633, 9.9761] | 3 / 2 |
| D_minus_A | fpr | 6.2270 | 4.4803 | [-0.7598, 9.7204] | 3 / 2 |
| B_minus_A | iou | 0.6752 | 0.4396 | [-0.2671, 1.1464] | 3 / 2 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 2 |
| B_minus_A | coverage | 0.4512 | 0.2460 | [-0.3696, 0.8616] | 3 / 2 |
| B_minus_A | auc | 1.2167 | 1.1389 | [0.9057, 1.3722] | 3 / 2 |
| B_minus_A | fpr | 1.2923 | 0.7935 | [-0.7029, 2.2898] | 3 / 2 |
| D1_minus_A | iou | 22.3008 | 28.1621 | [10.5784, 45.7457] | 3 / 2 |
| D1_minus_A | mask75 | 66.6667 | 75.0000 | [50.0000, 100.0000] | 3 / 2 |
| D1_minus_A | coverage | 36.1923 | 39.1821 | [30.2125, 48.1518] | 3 / 2 |
| D1_minus_A | auc | 10.1253 | 12.4391 | [5.4978, 19.3803] | 3 / 2 |
| D1_minus_A | fpr | 27.0469 | 21.3228 | [4.1506, 38.4950] | 3 / 2 |
| D_minus_D1 | iou | -18.9464 | -25.6044 | [-45.5785, -5.6303] | 3 / 2 |
| D_minus_D1 | mask75 | -66.6667 | -75.0000 | [-100.0000, -50.0000] | 3 / 2 |
| D_minus_D1 | coverage | -31.7278 | -35.8171 | [-48.0851, -23.5492] | 3 / 2 |
| D_minus_D1 | auc | -4.3577 | -5.6194 | [-9.4042, -1.8345] | 3 / 2 |
| D_minus_D1 | fpr | -20.8199 | -16.8426 | [-28.7747, -4.9105] | 3 / 2 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 3 | 0 / 0 | 66.6667 / 75.0000 | NA / NA | 2 | 75.0000 [50.0000, 100.0000] |
| D | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail/pyramid/P3

7 candidates; 3 eligible images; 7 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 64.2755 / 67.0159 | 0.0000 / 0.0000 | 93.8294 / 93.6030 | 92.0635 / 93.7906 | 52.3789 / 54.5844 |
| B | 62.5588 / 64.8919 | 0.0000 / 0.0000 | 94.7746 / 94.6333 | 91.8461 / 93.6505 | 58.8124 / 63.2481 |
| D1 | 59.0630 / 59.5581 | 0.0000 / 0.0000 | 94.9230 / 95.0392 | 90.3197 / 92.4893 | 73.1046 / 83.0429 |
| D | 64.0184 / 65.3276 | 14.2857 / 16.6667 | 93.5313 / 94.0677 | 91.4780 / 93.5523 | 56.8141 / 62.9520 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 1.4597 | 0.4356 | [-1.3067, 3.4059] | 7 / 3 |
| D_minus_B | mask75 | 14.2857 | 16.6667 | [0.0000, 50.0000] | 7 / 3 |
| D_minus_B | coverage | -1.2433 | -0.5656 | [-2.6547, 0.9579] | 7 / 3 |
| D_minus_B | auc | -0.3680 | -0.0982 | [-0.7980, 0.3908] | 7 / 3 |
| D_minus_B | fpr | -1.9984 | -0.2960 | [-4.5062, 3.2000] | 7 / 3 |
| D_minus_A | iou | -0.2570 | -1.6884 | [-5.2894, 1.5209] | 7 / 3 |
| D_minus_A | mask75 | 14.2857 | 16.6667 | [0.0000, 50.0000] | 7 / 3 |
| D_minus_A | coverage | -0.2980 | 0.4647 | [-2.4373, 3.8314] | 7 / 3 |
| D_minus_A | auc | -0.5855 | -0.2383 | [-1.2015, 0.2654] | 7 / 3 |
| D_minus_A | fpr | 4.4352 | 8.3676 | [0.4205, 20.0000] | 7 / 3 |
| B_minus_A | iou | -1.7167 | -2.1240 | [-4.4971, 0.0101] | 7 / 3 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 7 / 3 |
| B_minus_A | coverage | 0.9452 | 1.0303 | [0.0000, 2.8736] | 7 / 3 |
| B_minus_A | auc | -0.2175 | -0.1401 | [-0.4035, 0.1086] | 7 / 3 |
| B_minus_A | fpr | 6.4336 | 8.6636 | [4.2642, 16.8000] | 7 / 3 |
| D1_minus_A | iou | -5.2125 | -7.4578 | [-11.7432, -0.7902] | 7 / 3 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 7 / 3 |
| D1_minus_A | coverage | 1.0936 | 1.4362 | [-0.4808, 4.7893] | 7 / 3 |
| D1_minus_A | auc | -1.7439 | -1.3013 | [-2.3692, -0.3392] | 7 / 3 |
| D1_minus_A | fpr | 20.7257 | 28.4584 | [7.1646, 40.0000] | 7 / 3 |
| D_minus_D1 | iou | 4.9555 | 5.7694 | [2.3111, 10.4466] | 7 / 3 |
| D_minus_D1 | mask75 | 14.2857 | 16.6667 | [0.0000, 50.0000] | 7 / 3 |
| D_minus_D1 | coverage | -1.3917 | -0.9715 | [-1.9565, 0.0000] | 7 / 3 |
| D_minus_D1 | auc | 1.1584 | 1.0630 | [0.6046, 1.4165] | 7 / 3 |
| D_minus_D1 | fpr | -16.2905 | -20.0908 | [-33.5284, -6.7441] | 7 / 3 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 7 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 7 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 1 / 7 | 0 / 0 | 14.2857 / 16.6667 | NA / NA | 1 | 16.6667 [0.0000, 50.0000] |

## strict_maskfail/pyramid/P4

6 candidates; 4 eligible images; 6 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 60.1507 / 61.6765 | 0.0000 / 0.0000 | 91.7456 / 92.4060 | 92.3669 / 93.1488 | 31.1410 / 29.4555 |
| B | 54.9345 / 57.0714 | 0.0000 / 0.0000 | 93.6627 / 94.2989 | 91.2431 / 92.4761 | 40.8721 / 38.2438 |
| D1 | 48.9410 / 51.9383 | 0.0000 / 0.0000 | 95.3418 / 95.1305 | 86.1407 / 88.3409 | 59.6236 / 53.4500 |
| D | 53.2318 / 55.1827 | 0.0000 / 0.0000 | 88.0023 / 85.3714 | 88.9377 / 90.1969 | 37.6798 / 32.6145 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.7027 | -1.8887 | [-7.5491, 3.7716] | 6 / 4 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 4 |
| D_minus_B | coverage | -5.6605 | -8.9275 | [-27.0250, 2.9612] | 6 / 4 |
| D_minus_B | auc | -2.3055 | -2.2792 | [-3.5929, -0.8869] | 6 / 4 |
| D_minus_B | fpr | -3.1923 | -5.6293 | [-14.5566, 9.1893] | 6 / 4 |
| D_minus_A | iou | -6.9189 | -6.4938 | [-12.5489, -0.4387] | 6 / 4 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 4 |
| D_minus_A | coverage | -3.7433 | -7.0345 | [-24.8283, 5.8198] | 6 / 4 |
| D_minus_A | auc | -3.4292 | -2.9519 | [-5.2114, -0.7050] | 6 / 4 |
| D_minus_A | fpr | 6.5388 | 3.1589 | [-9.4877, 19.6913] | 6 / 4 |
| B_minus_A | iou | -5.2162 | -4.6050 | [-7.5843, -2.0205] | 6 / 4 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 4 |
| B_minus_A | coverage | 1.9171 | 1.8929 | [0.6235, 3.1624] | 6 / 4 |
| B_minus_A | auc | -1.1237 | -0.6727 | [-2.4811, 0.6806] | 6 / 4 |
| B_minus_A | fpr | 9.7311 | 8.7883 | [4.8756, 12.1085] | 6 / 4 |
| D1_minus_A | iou | -11.2097 | -9.7382 | [-18.4948, -1.8824] | 6 / 4 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 4 |
| D1_minus_A | coverage | 3.5962 | 2.7245 | [-0.9159, 8.3914] | 6 / 4 |
| D1_minus_A | auc | -6.2262 | -4.8079 | [-9.1075, -0.5529] | 6 / 4 |
| D1_minus_A | fpr | 28.4826 | 23.9945 | [4.8521, 48.7438] | 6 / 4 |
| D_minus_D1 | iou | 4.2908 | 3.2444 | [-6.9441, 10.2938] | 6 / 4 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 6 / 4 |
| D_minus_D1 | coverage | -7.3395 | -9.7590 | [-26.1159, -0.6606] | 6 / 4 |
| D_minus_D1 | auc | 2.7970 | 1.8560 | [-2.7168, 6.6432] | 6 / 4 |
| D_minus_D1 | fpr | -21.9438 | -20.8355 | [-28.7426, -13.3926] | 6 / 4 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 6 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 6 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 6 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail/pyramid/P5

4 candidates; 3 eligible images; 7 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 56.1315 / 55.0461 | 0.0000 / 0.0000 | 59.6672 / 58.0381 | 80.1210 / 82.2066 | 16.9885 / 12.6763 |
| B | 55.8474 / 54.2852 | 0.0000 / 0.0000 | 62.9588 / 62.1397 | 80.8518 / 82.7237 | 19.5053 / 15.2688 |
| D1 | 62.8367 / 60.4602 | 50.0000 / 50.0000 | 92.5253 / 91.7781 | 85.2380 / 87.1967 | 52.7536 / 47.5314 |
| D | 56.2495 / 53.5541 | 0.0000 / 0.0000 | 65.5578 / 63.6712 | 83.9865 / 86.1395 | 23.9530 / 18.7222 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 0.4021 | -0.7311 | [-6.4294, 3.8017] | 4 / 3 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 4 / 3 |
| D_minus_B | coverage | 2.5990 | 1.5315 | [-1.6436, 5.8017] | 4 / 3 |
| D_minus_B | auc | 3.1347 | 3.4158 | [-1.1141, 9.0704] | 4 / 3 |
| D_minus_B | fpr | 4.4477 | 3.4534 | [-0.0570, 7.4305] | 4 / 3 |
| D_minus_A | iou | 0.1181 | -1.4919 | [-9.5912, 4.9481] | 4 / 3 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 4 / 3 |
| D_minus_A | coverage | 5.8907 | 5.6332 | [0.0667, 10.1695] | 4 / 3 |
| D_minus_A | auc | 3.8655 | 3.9329 | [-1.8409, 9.9761] | 4 / 3 |
| D_minus_A | fpr | 6.9645 | 6.0459 | [-0.7598, 9.7204] | 4 / 3 |
| B_minus_A | iou | -0.2840 | -0.7608 | [-3.1618, 1.1464] | 4 / 3 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 4 / 3 |
| B_minus_A | coverage | 3.2916 | 4.1017 | [-0.3696, 11.8130] | 4 / 3 |
| B_minus_A | auc | 0.7308 | 0.5170 | [-0.7268, 1.3722] | 4 / 3 |
| B_minus_A | fpr | 2.5168 | 2.5925 | [-0.7029, 6.1906] | 4 / 3 |
| D1_minus_A | iou | 6.7052 | 5.4142 | [-40.0815, 45.7457] | 4 / 3 |
| D1_minus_A | mask75 | 50.0000 | 50.0000 | [0.0000, 100.0000] | 4 / 3 |
| D1_minus_A | coverage | 32.8581 | 33.7400 | [22.8557, 48.1518] | 4 / 3 |
| D1_minus_A | auc | 5.1170 | 4.9900 | [-9.9081, 19.3803] | 4 / 3 |
| D1_minus_A | fpr | 35.7651 | 34.8551 | [4.1506, 61.9196] | 4 / 3 |
| D_minus_D1 | iou | -6.5872 | -6.9061 | [-45.5785, 30.4904] | 4 / 3 |
| D_minus_D1 | mask75 | -50.0000 | -50.0000 | [-100.0000, 0.0000] | 4 / 3 |
| D_minus_D1 | coverage | -26.9674 | -28.1068 | [-48.0851, -12.6862] | 4 / 3 |
| D_minus_D1 | auc | -1.2515 | -1.0572 | [-9.4042, 8.0672] | 4 / 3 |
| D_minus_D1 | fpr | -28.8005 | -28.8092 | [-52.7423, -4.9105] | 4 / 3 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 4 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 2 / 4 | 0 / 0 | 50.0000 / 50.0000 | NA / NA | 2 | 50.0000 [0.0000, 100.0000] |
| D | 0 / 4 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## baseline_success/size/small

15 candidates; 3 eligible images; 7 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 83.3801 / 82.8480 | 100.0000 / 100.0000 | 97.0678 / 96.8453 | 96.9259 / 96.9184 | 42.5610 / 41.1212 |
| B | 81.4506 / 80.9043 | 86.6667 / 85.0000 | 97.2955 / 97.1015 | 97.0360 / 97.0302 | 49.0372 / 47.2969 |
| D1 | 80.6631 / 80.0233 | 86.6667 / 85.0000 | 96.7872 / 96.5371 | 95.4073 / 95.3788 | 51.4349 / 49.5152 |
| D | 82.8686 / 82.4941 | 86.6667 / 85.0000 | 95.7884 / 95.5004 | 96.0616 / 96.0655 | 38.8476 / 36.7757 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 1.4180 | 1.5898 | [0.6411, 3.2194] | 15 / 3 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 15 / 3 |
| D_minus_B | coverage | -1.5070 | -1.6012 | [-2.2544, -0.8425] | 15 / 3 |
| D_minus_B | auc | -0.9745 | -0.9647 | [-2.6022, -0.0723] | 15 / 3 |
| D_minus_B | fpr | -10.1897 | -10.5212 | [-23.0484, -1.7713] | 15 / 3 |
| D_minus_A | iou | -0.5116 | -0.3539 | [-2.2457, 1.7745] | 15 / 3 |
| D_minus_A | mask75 | -13.3333 | -15.0000 | [-25.0000, 0.0000] | 15 / 3 |
| D_minus_A | coverage | -1.2794 | -1.3449 | [-1.6961, -0.7142] | 15 / 3 |
| D_minus_A | auc | -0.8643 | -0.8528 | [-2.1106, -0.1378] | 15 / 3 |
| D_minus_A | fpr | -3.7134 | -4.3455 | [-13.7632, 5.1039] | 15 / 3 |
| B_minus_A | iou | -1.9295 | -1.9437 | [-3.1548, -1.2315] | 15 / 3 |
| B_minus_A | mask75 | -13.3333 | -15.0000 | [-25.0000, 0.0000] | 15 / 3 |
| B_minus_A | coverage | 0.2276 | 0.2563 | [0.0823, 0.5582] | 15 / 3 |
| B_minus_A | auc | 0.1101 | 0.1118 | [-0.0906, 0.4916] | 15 / 3 |
| B_minus_A | fpr | 6.4762 | 6.1757 | [2.3666, 9.2852] | 15 / 3 |
| D1_minus_A | iou | -2.7171 | -2.8247 | [-4.8450, -1.0075] | 15 / 3 |
| D1_minus_A | mask75 | -13.3333 | -15.0000 | [-25.0000, 0.0000] | 15 / 3 |
| D1_minus_A | coverage | -0.2806 | -0.3082 | [-0.7321, 0.1264] | 15 / 3 |
| D1_minus_A | auc | -1.5186 | -1.5395 | [-3.5555, -0.3742] | 15 / 3 |
| D1_minus_A | fpr | 8.8739 | 8.3940 | [2.3076, 13.3673] | 15 / 3 |
| D_minus_D1 | iou | 2.2055 | 2.4708 | [0.4170, 4.3961] | 15 / 3 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 15 / 3 |
| D_minus_D1 | coverage | -0.9988 | -1.0367 | [-1.7507, -0.3954] | 15 / 3 |
| D_minus_D1 | auc | 0.6542 | 0.6867 | [0.0641, 1.4449] | 15 / 3 |
| D_minus_D1 | fpr | -12.5873 | -12.7394 | [-27.1305, -4.4031] | 15 / 3 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 2 / 15 | NA / NA | 13.3333 / 15.0000 | -2 | -15.0000 [-25.0000, 0.0000] |
| D1 | 0 / 0 | 2 / 15 | NA / NA | 13.3333 / 15.0000 | -2 | -15.0000 [-25.0000, 0.0000] |
| D | 0 / 0 | 2 / 15 | NA / NA | 13.3333 / 15.0000 | -2 | -15.0000 [-25.0000, 0.0000] |

## baseline_success/size/medium

14 candidates; 4 eligible images; 6 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 88.4472 / 90.0188 | 100.0000 / 100.0000 | 96.6560 / 97.3177 | 98.6907 / 98.9563 | 14.7374 / 15.4879 |
| B | 87.4791 / 89.4491 | 92.8571 / 96.4286 | 97.6717 / 97.8312 | 98.7781 / 99.0048 | 17.6213 / 17.2251 |
| D1 | 73.1099 / 80.8773 | 64.2857 / 82.1429 | 98.5938 / 98.2246 | 97.5872 / 98.2677 | 50.5905 / 38.4696 |
| D | 83.7047 / 87.5251 | 71.4286 / 85.7143 | 98.0519 / 98.0053 | 98.7501 / 98.9504 | 24.1293 / 20.6141 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7744 | -1.9240 | [-5.3364, 0.2333] | 14 / 4 |
| D_minus_B | mask75 | -21.4286 | -10.7143 | [-32.1429, 0.0000] | 14 / 4 |
| D_minus_B | coverage | 0.3802 | 0.1741 | [-0.4579, 0.8185] | 14 / 4 |
| D_minus_B | auc | -0.0279 | -0.0543 | [-0.1722, 0.0635] | 14 / 4 |
| D_minus_B | fpr | 6.5080 | 3.3889 | [-0.6311, 9.3268] | 14 / 4 |
| D_minus_A | iou | -4.7425 | -2.4937 | [-6.5972, 0.0599] | 14 / 4 |
| D_minus_A | mask75 | -28.5714 | -14.2857 | [-42.8571, 0.0000] | 14 / 4 |
| D_minus_A | coverage | 1.3959 | 0.6876 | [-0.3342, 2.2493] | 14 / 4 |
| D_minus_A | auc | 0.0595 | -0.0059 | [-0.1593, 0.1600] | 14 / 4 |
| D_minus_A | fpr | 9.3919 | 5.1262 | [-0.5317, 13.1934] | 14 / 4 |
| B_minus_A | iou | -0.9681 | -0.5697 | [-1.2608, -0.0995] | 14 / 4 |
| B_minus_A | mask75 | -7.1429 | -3.5714 | [-10.7143, 0.0000] | 14 / 4 |
| B_minus_A | coverage | 1.0158 | 0.5135 | [-0.0506, 1.4806] | 14 / 4 |
| B_minus_A | auc | 0.0874 | 0.0484 | [-0.0089, 0.1280] | 14 / 4 |
| B_minus_A | fpr | 2.8838 | 1.7373 | [0.2509, 3.5293] | 14 / 4 |
| D1_minus_A | iou | -15.3373 | -9.1415 | [-20.4191, -0.4836] | 14 / 4 |
| D1_minus_A | mask75 | -35.7143 | -17.8571 | [-53.5714, 0.0000] | 14 / 4 |
| D1_minus_A | coverage | 1.9378 | 0.9068 | [-0.7246, 2.7687] | 14 / 4 |
| D1_minus_A | auc | -1.1034 | -0.6886 | [-1.4523, -0.0761] | 14 / 4 |
| D1_minus_A | fpr | 35.8531 | 22.9817 | [0.3108, 44.7965] | 14 / 4 |
| D_minus_D1 | iou | 10.5948 | 6.6478 | [0.5435, 13.8219] | 14 / 4 |
| D_minus_D1 | mask75 | 7.1429 | 3.5714 | [0.0000, 10.7143] | 14 / 4 |
| D_minus_D1 | coverage | -0.5419 | -0.2193 | [-0.7782, 0.7645] | 14 / 4 |
| D_minus_D1 | auc | 1.1629 | 0.6827 | [0.0708, 1.5992] | 14 / 4 |
| D_minus_D1 | fpr | -26.4612 | -17.8555 | [-31.5243, -0.9233] | 14 / 4 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 1 / 14 | NA / NA | 7.1429 / 3.5714 | -1 | -3.5714 [-10.7143, 0.0000] |
| D1 | 0 / 0 | 5 / 14 | NA / NA | 35.7143 / 17.8571 | -5 | -17.8571 [-53.5714, 0.0000] |
| D | 0 / 0 | 4 / 14 | NA / NA | 28.5714 / 14.2857 | -4 | -14.2857 [-42.8571, 0.0000] |

## baseline_success/size/large

12 candidates; 7 eligible images; 3 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 91.4503 / 90.6412 | 100.0000 / 100.0000 | 96.5667 / 96.3433 | 99.1429 / 98.9422 | 6.9156 / 7.7081 |
| B | 91.0502 / 89.8745 | 100.0000 / 100.0000 | 96.8372 / 96.6739 | 99.0799 / 98.8250 | 8.1012 / 9.6657 |
| D1 | 77.8132 / 79.2933 | 58.3333 / 66.6667 | 97.7191 / 97.5060 | 97.0159 / 96.9957 | 35.3412 / 32.4098 |
| D | 85.1178 / 83.6541 | 91.6667 / 85.7143 | 96.3383 / 95.7930 | 98.3765 / 98.0024 | 17.2200 / 19.8915 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -5.9325 | -6.2204 | [-10.2589, -2.4293] | 12 / 7 |
| D_minus_B | mask75 | -8.3333 | -14.2857 | [-42.8571, 0.0000] | 12 / 7 |
| D_minus_B | coverage | -0.4989 | -0.8808 | [-3.2803, 0.6916] | 12 / 7 |
| D_minus_B | auc | -0.7034 | -0.8225 | [-1.7683, -0.0572] | 12 / 7 |
| D_minus_B | fpr | 9.1189 | 10.2258 | [1.7610, 20.1658] | 12 / 7 |
| D_minus_A | iou | -6.3326 | -6.9871 | [-11.7481, -2.5644] | 12 / 7 |
| D_minus_A | mask75 | -8.3333 | -14.2857 | [-42.8571, 0.0000] | 12 / 7 |
| D_minus_A | coverage | -0.2284 | -0.5503 | [-2.3986, 0.7395] | 12 / 7 |
| D_minus_A | auc | -0.7664 | -0.9398 | [-1.9444, -0.0841] | 12 / 7 |
| D_minus_A | fpr | 10.3044 | 12.1834 | [2.1232, 23.8433] | 12 / 7 |
| B_minus_A | iou | -0.4001 | -0.7667 | [-2.6653, 0.3614] | 12 / 7 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 7 |
| B_minus_A | coverage | 0.2704 | 0.3306 | [-0.0870, 0.9196] | 12 / 7 |
| B_minus_A | auc | -0.0630 | -0.1172 | [-0.2655, 0.0010] | 12 / 7 |
| B_minus_A | fpr | 1.1856 | 1.9577 | [-0.0235, 5.5836] | 12 / 7 |
| D1_minus_A | iou | -13.6371 | -11.3479 | [-18.8737, -4.6195] | 12 / 7 |
| D1_minus_A | mask75 | -41.6667 | -33.3333 | [-66.6667, -4.7619] | 12 / 7 |
| D1_minus_A | coverage | 1.1524 | 1.1627 | [0.5156, 1.8619] | 12 / 7 |
| D1_minus_A | auc | -2.1270 | -1.9465 | [-3.2389, -0.7724] | 12 / 7 |
| D1_minus_A | fpr | 28.4256 | 24.7017 | [10.2575, 39.9248] | 12 / 7 |
| D_minus_D1 | iou | 7.3045 | 4.3608 | [-0.0438, 9.4662] | 12 / 7 |
| D_minus_D1 | mask75 | 33.3333 | 19.0476 | [0.0000, 47.6190] | 12 / 7 |
| D_minus_D1 | coverage | -1.3809 | -1.7130 | [-3.8853, -0.4527] | 12 / 7 |
| D_minus_D1 | auc | 1.3606 | 1.0067 | [0.4533, 1.7836] | 12 / 7 |
| D_minus_D1 | fpr | -18.1211 | -12.5183 | [-22.4393, -4.2429] | 12 / 7 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 12 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 0 | 5 / 12 | NA / NA | 41.6667 / 33.3333 | -5 | -33.3333 [-66.6667, -4.7619] |
| D | 0 / 0 | 1 / 12 | NA / NA | 8.3333 / 14.2857 | -1 | -14.2857 [-42.8571, 0.0000] |

## baseline_success/pyramid/P3

12 candidates; 3 eligible images; 7 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 83.6606 / 82.4285 | 100.0000 / 100.0000 | 97.6829 / 97.6763 | 96.3872 / 96.6514 | 51.8928 / 46.3726 |
| B | 81.3673 / 80.1313 | 83.3333 / 76.6667 | 97.8489 / 97.8091 | 96.5414 / 96.7486 | 60.3727 / 53.7748 |
| D1 | 80.1544 / 78.9824 | 83.3333 / 76.6667 | 97.9834 / 97.8373 | 94.6902 / 95.0921 | 67.3375 / 59.3397 |
| D | 82.8104 / 82.2645 | 83.3333 / 76.6667 | 97.3166 / 97.3833 | 95.4589 / 95.8344 | 50.5596 / 44.2934 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 1.4431 | 2.1332 | [0.5969, 4.8936] | 12 / 3 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 3 |
| D_minus_B | coverage | -0.5323 | -0.4259 | [-1.7066, 0.4291] | 12 / 3 |
| D_minus_B | auc | -1.0825 | -0.9141 | [-2.6022, 0.1003] | 12 / 3 |
| D_minus_B | fpr | -9.8131 | -9.4815 | [-23.0484, 2.7587] | 12 / 3 |
| D_minus_A | iou | -0.8502 | -0.1640 | [-2.2457, 2.5809] | 12 / 3 |
| D_minus_A | mask75 | -16.6667 | -23.3333 | [-50.0000, 0.0000] | 12 / 3 |
| D_minus_A | coverage | -0.3663 | -0.2930 | [-1.6243, 0.7453] | 12 / 3 |
| D_minus_A | auc | -0.9283 | -0.8170 | [-2.1106, 0.0314] | 12 / 3 |
| D_minus_A | fpr | -1.3332 | -2.0792 | [-13.7632, 12.5888] | 12 / 3 |
| B_minus_A | iou | -2.2933 | -2.2972 | [-3.1548, -1.4242] | 12 / 3 |
| B_minus_A | mask75 | -16.6667 | -23.3333 | [-50.0000, 0.0000] | 12 / 3 |
| B_minus_A | coverage | 0.1661 | 0.1328 | [0.0000, 0.3162] | 12 / 3 |
| B_minus_A | auc | 0.1542 | 0.0971 | [-0.1313, 0.4916] | 12 / 3 |
| B_minus_A | fpr | 8.4799 | 7.4022 | [3.0913, 9.8301] | 12 / 3 |
| D1_minus_A | iou | -3.5062 | -3.4461 | [-4.8450, -2.2876] | 12 / 3 |
| D1_minus_A | mask75 | -16.6667 | -23.3333 | [-50.0000, 0.0000] | 12 / 3 |
| D1_minus_A | coverage | 0.3005 | 0.1610 | [-0.3968, 0.7535] | 12 / 3 |
| D1_minus_A | auc | -1.6970 | -1.5594 | [-3.5555, -0.1138] | 12 / 3 |
| D1_minus_A | fpr | 15.4447 | 12.9671 | [3.0564, 22.4775] | 12 / 3 |
| D_minus_D1 | iou | 2.6560 | 3.2821 | [1.4603, 5.7868] | 12 / 3 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 12 / 3 |
| D_minus_D1 | coverage | -0.6668 | -0.4540 | [-1.7507, 0.3968] | 12 / 3 |
| D_minus_D1 | auc | 0.7687 | 0.7423 | [0.1452, 1.4449] | 12 / 3 |
| D_minus_D1 | fpr | -16.7780 | -15.0463 | [-27.1305, -8.1198] | 12 / 3 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 2 / 12 | NA / NA | 16.6667 / 23.3333 | -2 | -23.3333 [-50.0000, 0.0000] |
| D1 | 0 / 0 | 2 / 12 | NA / NA | 16.6667 / 23.3333 | -2 | -23.3333 [-50.0000, 0.0000] |
| D | 0 / 0 | 2 / 12 | NA / NA | 16.6667 / 23.3333 | -2 | -23.3333 [-50.0000, 0.0000] |

## baseline_success/pyramid/P4

15 candidates; 5 eligible images; 5 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 88.0644 / 88.5452 | 100.0000 / 100.0000 | 96.5617 / 96.6692 | 98.7216 / 98.7871 | 13.9882 / 14.8989 |
| B | 87.1544 / 87.9673 | 93.3333 / 97.1429 | 97.4589 / 97.2227 | 98.7813 / 98.8147 | 16.2843 / 16.4865 |
| D1 | 74.6315 / 80.8406 | 73.3333 / 88.5714 | 97.5560 / 96.8704 | 97.5353 / 98.0785 | 43.4291 / 33.4206 |
| D | 83.6898 / 86.6108 | 73.3333 / 88.5714 | 96.7455 / 96.2045 | 98.6450 / 98.7261 | 20.1055 / 17.5038 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.4645 | -1.3565 | [-4.3319, 0.7926] | 15 / 5 |
| D_minus_B | mask75 | -20.0000 | -8.5714 | [-25.7143, 0.0000] | 15 / 5 |
| D_minus_B | coverage | -0.7134 | -1.0182 | [-2.6872, 0.1633] | 15 / 5 |
| D_minus_B | auc | -0.1364 | -0.0886 | [-0.1975, 0.0203] | 15 / 5 |
| D_minus_B | fpr | 3.8212 | 1.0173 | [-2.6283, 5.2007] | 15 / 5 |
| D_minus_A | iou | -4.3746 | -1.9344 | [-5.4477, 0.4205] | 15 / 5 |
| D_minus_A | mask75 | -26.6667 | -11.4286 | [-34.2857, 0.0000] | 15 / 5 |
| D_minus_A | coverage | 0.1838 | -0.4646 | [-1.9763, 0.7974] | 15 / 5 |
| D_minus_A | auc | -0.0766 | -0.0610 | [-0.1729, 0.0435] | 15 / 5 |
| D_minus_A | fpr | 6.1173 | 2.6048 | [-1.5681, 7.5226] | 15 / 5 |
| B_minus_A | iou | -0.9101 | -0.5779 | [-1.0239, -0.1950] | 15 / 5 |
| B_minus_A | mask75 | -6.6667 | -2.8571 | [-8.5714, 0.0000] | 15 / 5 |
| B_minus_A | coverage | 0.8972 | 0.5536 | [0.0182, 1.1599] | 15 / 5 |
| B_minus_A | auc | 0.0597 | 0.0276 | [-0.0122, 0.0773] | 15 / 5 |
| B_minus_A | fpr | 2.2961 | 1.5875 | [0.5290, 2.6460] | 15 / 5 |
| D1_minus_A | iou | -13.4329 | -7.7046 | [-15.2252, -1.3585] | 15 / 5 |
| D1_minus_A | mask75 | -26.6667 | -11.4286 | [-34.2857, 0.0000] | 15 / 5 |
| D1_minus_A | coverage | 0.9943 | 0.2013 | [-0.9618, 1.3644] | 15 / 5 |
| D1_minus_A | auc | -1.1863 | -0.7086 | [-1.3689, -0.2078] | 15 / 5 |
| D1_minus_A | fpr | 29.4409 | 18.5217 | [2.0422, 35.0011] | 15 / 5 |
| D_minus_D1 | iou | 9.0583 | 5.7702 | [1.8635, 10.0615] | 15 / 5 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 15 / 5 |
| D_minus_D1 | coverage | -0.8105 | -0.6659 | [-1.6650, 0.4524] | 15 / 5 |
| D_minus_D1 | auc | 1.1097 | 0.6476 | [0.1644, 1.3320] | 15 / 5 |
| D_minus_D1 | fpr | -23.3236 | -15.9169 | [-28.2234, -3.5225] | 15 / 5 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 1 / 15 | NA / NA | 6.6667 / 2.8571 | -1 | -2.8571 [-8.5714, 0.0000] |
| D1 | 0 / 0 | 4 / 15 | NA / NA | 26.6667 / 11.4286 | -4 | -11.4286 [-34.2857, 0.0000] |
| D | 0 / 0 | 4 / 15 | NA / NA | 26.6667 / 11.4286 | -4 | -11.4286 [-34.2857, 0.0000] |

## baseline_success/pyramid/P5

14 candidates; 7 eligible images; 3 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 90.1052 / 89.4853 | 100.0000 / 100.0000 | 96.2416 / 96.1724 | 99.1287 / 98.9993 | 6.7992 / 7.9384 |
| B | 89.6677 / 88.5651 | 100.0000 / 100.0000 | 96.6294 / 96.5172 | 99.0839 / 98.9000 | 7.9095 / 9.9244 |
| D1 | 77.5655 / 77.3924 | 57.1429 / 59.5238 | 97.5437 / 97.4474 | 97.3007 / 97.0685 | 31.7428 / 32.9531 |
| D | 84.8025 / 82.9854 | 92.8571 / 85.7143 | 96.1880 / 96.1508 | 98.4830 / 98.0536 | 15.6334 / 20.1273 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -4.8651 | -5.5797 | [-9.8763, -1.7501] | 14 / 7 |
| D_minus_B | mask75 | -7.1429 | -14.2857 | [-42.8571, 0.0000] | 14 / 7 |
| D_minus_B | coverage | -0.4415 | -0.3664 | [-1.7031, 0.6805] | 14 / 7 |
| D_minus_B | auc | -0.6009 | -0.8465 | [-1.7462, -0.0999] | 14 / 7 |
| D_minus_B | fpr | 7.7238 | 10.2030 | [1.6658, 19.9786] | 14 / 7 |
| D_minus_A | iou | -5.3027 | -6.5000 | [-11.3285, -2.1632] | 14 / 7 |
| D_minus_A | mask75 | -7.1429 | -14.2857 | [-42.8571, 0.0000] | 14 / 7 |
| D_minus_A | coverage | -0.0536 | -0.0216 | [-0.8491, 0.7619] | 14 / 7 |
| D_minus_A | auc | -0.6457 | -0.9457 | [-1.9384, -0.0895] | 14 / 7 |
| D_minus_A | fpr | 8.8341 | 12.1889 | [2.0468, 23.9555] | 14 / 7 |
| B_minus_A | iou | -0.4375 | -0.9202 | [-2.7344, 0.0660] | 14 / 7 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 14 / 7 |
| B_minus_A | coverage | 0.3878 | 0.3448 | [-0.0851, 0.9618] | 14 / 7 |
| B_minus_A | auc | -0.0448 | -0.0993 | [-0.2509, 0.0139] | 14 / 7 |
| B_minus_A | fpr | 1.1103 | 1.9859 | [-0.0285, 5.6249] | 14 / 7 |
| D1_minus_A | iou | -12.5397 | -12.0929 | [-19.1765, -5.4849] | 14 / 7 |
| D1_minus_A | mask75 | -42.8571 | -40.4762 | [-71.4286, -11.9048] | 14 / 7 |
| D1_minus_A | coverage | 1.3021 | 1.2749 | [0.5639, 2.0315] | 14 / 7 |
| D1_minus_A | auc | -1.8280 | -1.9308 | [-3.2349, -0.7049] | 14 / 7 |
| D1_minus_A | fpr | 24.9435 | 25.0147 | [10.4476, 39.9523] | 14 / 7 |
| D_minus_D1 | iou | 7.2370 | 5.5930 | [1.9098, 10.0556] | 14 / 7 |
| D_minus_D1 | mask75 | 35.7143 | 26.1905 | [4.7619, 54.7619] | 14 / 7 |
| D_minus_D1 | coverage | -1.3557 | -1.2966 | [-2.5476, -0.5166] | 14 / 7 |
| D_minus_D1 | auc | 1.1823 | 0.9850 | [0.4195, 1.7716] | 14 / 7 |
| D_minus_D1 | fpr | -16.1094 | -12.8258 | [-22.2859, -4.8591] | 14 / 7 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 14 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 0 | 6 / 14 | NA / NA | 42.8571 / 40.4762 | -6 | -40.4762 [-71.4286, -11.9048] |
| D | 0 / 0 | 1 / 14 | NA / NA | 7.1429 / 14.2857 | -1 | -14.2857 [-42.8571, 0.0000] |

## Timing and limitations

{
  "supplied_timing_scope": "not specified; supplied row observations only",
  "direct_ms": {
    "candidate_mean": 1.1504420681911356,
    "image_macro": 1.0075111124066365,
    "valid_candidates": 102,
    "valid_images": 10,
    "median_ms": 0.7661920972168446,
    "p95_ms": 0.8443882223218678
  },
  "quality_ms": {
    "candidate_mean": 9.37560570481069,
    "image_macro": 9.386899971159863,
    "valid_candidates": 102,
    "valid_images": 10,
    "median_ms": 9.256669785827398,
    "p95_ms": 9.746666718274355
  }
}

Timing is summarized in the supplied unit; no image-total or dataset-total latency is inferred from repeated row values. The unpopulated planned-image list cannot by itself distinguish zero matched candidates from an interrupted evaluation; completion must be established by the run and metadata.

Metadata: `{"split": "dev", "planned_images": 2000, "evaluated_images_requested": 10, "image_ids": [85114, 485462, 571503, 39509, 155707, 448368, 320434, 174137, 224342, 83663], "split_sha256": "a1a3ee0a0e39582dfc13a403faa91ddd63ed5ca5ab16017a0ff84af028765291", "direct_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_direct_seed0_retry5/final.pt", "direct_checkpoint_sha256": "25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8", "quality_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_quality_seed0/final.pt", "quality_checkpoint_sha256": "07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065", "epoch": 3, "rho": 12.0853271484375, "steps": 2, "eta": [6.04266357421875, 3.021331787109375], "lambda": 0.003, "scientific_drift": ["Actual training seed 20261005 differs from recorded seed 0.", "Quality ranking uses soft IoU rather than normal binary original-mask IoU.", "Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.", "Failure random-control radius is global rho, not each oracle displacement norm.", "Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.", "Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.", "DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.", "Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test."], "parameter_updates": false, "oracle_used": false, "streaming": true, "dtype": "float32", "tf32": false, "heldout_states_definition": "Fixed success-state probes: c0 and ±rho/8*u1, ±rho/4*u1, ±rho/4*u2, +rho/8*u2; GT-free and identity-seeded. Added diagnostic, not a preregistered oracle-direction test.", "decoder_definition": "Official process_mask full P@c -> bilinear640 -> predicted box crop -> >0; inverse letterbox binary mask -> >.5. Original COCO GT.", "pixel_metric_version": "qcr-original-pixels-v2", "pixel_metric_definition": "Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background.", "strict_maskfail_definition": "Entire argmax-correct R_arg has a Box75 witness and no Mask75; measured here on its intersection with fixed TAL class-correct Box75 baseline-fail candidates. Unknown on historical DEV cache.", "elapsed_s": 10.926148829981685, "prototype_storage_images": {"torch.float32": 10}, "scope": "Official one2one TAL matched candidates; no AP/postprocessing quality reranking", "integrity_verified": true, "image_membership_verified": true, "native_decode_verified": true, "partial": true, "no_positive_images": 0}`.
