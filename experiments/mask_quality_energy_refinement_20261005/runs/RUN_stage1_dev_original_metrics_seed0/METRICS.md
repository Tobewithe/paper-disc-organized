# QCR DEV fixed-candidate metrics

Protocol status: **incomplete**. Stage II permitted: **False**.

13690 candidate rows; 2000 / 2000 planned images contain rows. 0 planned images contain no rows. Those images receive no artificial zero IoU.

All candidate means weight candidates equally. Image macro first averages eligible candidates within each image, then weights eligible images equally. Subgroup-empty images and images without a defined pixel metric are excluded from that estimand. Confidence intervals resample entire eligible images with replacement, preserving the pairing between arms. Deltas and CI in tables are percentage points (pp).

TAL MaskFail = class_correct and BoxIoU ≥ .75 and original MaskIoU < .75. Strict MaskFail additionally requires an audited raw_arg_maskfail=true: the full class-argmax raw set has a Box75 candidate and no Mask75 candidate for the target GT. Unknown raw-arg labels are excluded from strict metrics and block release; TAL failure never substitutes for strict failure. Baseline success uses the same class/box criteria and original MaskIoU ≥ .75. Size uses original COCO annotation area: small <32², medium [32²,96²), large ≥96².

Coverage uses original-resolution binary-mask intersection / original COCO annToMask area. Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background. All pixels outside the target GT (including other instances) are negative. AUC is exact Mann–Whitney with averaged ties. Undefined metrics remain NA. This is a fixed official one-to-one TAL matched-candidate evaluation and makes no COCO AP claim.

## Protocol criteria

| Criterion | Estimate | Required | Pass |
|---|---:|---:|---|
| heldout_pairwise_accuracy | 53.1602 % | >= 65.0000 % | False |
| step1_true_iou_improvement_fraction | 16.9028 % | > 55.0000 % | False |
| strict_maskfail_D_minus_B | NA pp | >= 0.2000 pp | None |
| strict_maskfail_D_minus_B_CI_low | NA pp | > 0.0000 pp | None |
| strict_maskfail_D_minus_A | NA pp | >= 0.5000 pp | None |
| all_D_minus_A_CI_low | -2.9074 pp | >= -0.1000 pp | False |

Ineligibility/incompleteness: raw_arg_maskfail_unknown; pyramid_level_unknown; scientific_scope_drift.

Scientific drift: Actual training seed 20261005 differs from recorded seed 0.; Quality ranking uses soft IoU rather than normal binary original-mask IoU.; Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.; Failure random-control radius is global rho, not each oracle displacement norm.; Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.; Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.; DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.; Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test..

## Quality diagnostics

Held-out pairwise accuracy: instance equal 53.1602%; pairs weighted 53.2729%; image macro 53.4104%; 372880 non-tie hard-IoU pairs, 13607 eligible instances. True hard-IoU ties are excluded; predicted Q ties get 0.5 credit. Spearman uses averaged tie ranks; constant-state instances are undefined.

Spearman: instance equal 0.0820; image macro 0.0870; 13607 defined instances.

Step-one true IoU increase: 16.9028% over 13690 candidates (2314 increase, 11048 decrease, 328 ties; ties stay in the denominator).

Nonzero Q/IoU delta sign consistency: 64.7433% over 13362 instances. Positive Q but decreased true IoU: 3817 / 5539 positive-Q instances (68.9114%).

Sign table (Q(c1)−Q(c0) rows / true IoU(D1)−IoU(A) columns; exact zeros kept separately):

| Q delta | IoU decrease | IoU tie | IoU increase |
|---|---:|---:|---:|
| negative | 7231 | 26 | 894 |
| zero | 0 | 0 | 0 |
| positive | 3817 | 302 | 1420 |

Exclusions from nonzero consistency: {"missing_q_or_iou": 0, "both_zero": 0, "q_zero_iou_nonzero": 0, "q_nonzero_iou_zero": 328}.

Q-Rank at these unchanged fixed candidates keeps c0 and therefore the same MaskIoU as A. No ranking/AP effect is estimated here.

## all

13690 candidates; 2000 eligible images; 0 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 75.5680 / 80.8471 | 62.7831 / 74.8028 | 90.7355 / 92.3963 | 93.4755 / 95.5142 | 28.1342 / 20.9969 |
| B | 74.5452 / 80.2905 | 59.9123 / 73.4416 | 92.8672 / 93.7744 | 93.4653 / 95.5258 | 34.5030 / 24.7590 |
| D1 | 66.9736 / 72.8029 | 40.3068 / 54.7391 | 92.2365 / 93.4715 | 91.3533 / 93.9302 | 53.4531 / 43.7808 |
| D | 72.0651 / 78.1244 | 54.4558 / 68.7023 | 90.6289 / 92.4304 | 92.6851 / 94.9895 | 36.6692 / 28.0274 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.4802 | -2.1661 | [-2.3445, -1.9861] | 13690 / 2000 |
| D_minus_B | mask75 | -5.4565 | -4.7392 | [-5.3931, -4.1394] | 13690 / 2000 |
| D_minus_B | coverage | -2.2383 | -1.3441 | [-1.5263, -1.1687] | 13690 / 2000 |
| D_minus_B | auc | -0.7802 | -0.5362 | [-0.6018, -0.4723] | 13676 / 1999 |
| D_minus_B | fpr | 2.1662 | 3.2684 | [2.8227, 3.7125] | 13678 / 1999 |
| D_minus_A | iou | -3.5030 | -2.7227 | [-2.9074, -2.5430] | 13690 / 2000 |
| D_minus_A | mask75 | -8.3272 | -6.1004 | [-6.7563, -5.4506] | 13690 / 2000 |
| D_minus_A | coverage | -0.1066 | 0.0340 | [-0.1442, 0.2054] | 13690 / 2000 |
| D_minus_A | auc | -0.7903 | -0.5247 | [-0.5940, -0.4551] | 13676 / 1999 |
| D_minus_A | fpr | 8.5350 | 7.0305 | [6.5767, 7.4895] | 13678 / 1999 |
| B_minus_A | iou | -1.0228 | -0.5566 | [-0.6412, -0.4717] | 13690 / 2000 |
| B_minus_A | mask75 | -2.8707 | -1.3612 | [-1.7110, -0.9912] | 13690 / 2000 |
| B_minus_A | coverage | 2.1316 | 1.3781 | [1.2721, 1.4871] | 13690 / 2000 |
| B_minus_A | auc | -0.0102 | 0.0115 | [-0.0210, 0.0450] | 13676 / 1999 |
| B_minus_A | fpr | 6.3688 | 3.7621 | [3.5606, 3.9713] | 13678 / 1999 |
| D1_minus_A | iou | -8.5945 | -8.0442 | [-8.3965, -7.7138] | 13690 / 2000 |
| D1_minus_A | mask75 | -22.4763 | -20.0637 | [-21.2470, -18.9156] | 13690 / 2000 |
| D1_minus_A | coverage | 1.5010 | 1.0752 | [0.7859, 1.3427] | 13690 / 2000 |
| D1_minus_A | auc | -2.1222 | -1.5840 | [-1.7102, -1.4656] | 13676 / 1999 |
| D1_minus_A | fpr | 25.3189 | 22.7839 | [21.9889, 23.6008] | 13678 / 1999 |
| D_minus_D1 | iou | 5.0915 | 5.3215 | [5.0564, 5.6052] | 13690 / 2000 |
| D_minus_D1 | mask75 | 14.1490 | 13.9633 | [12.9317, 15.0167] | 13690 / 2000 |
| D_minus_D1 | coverage | -1.6076 | -1.0412 | [-1.2695, -0.7907] | 13690 / 2000 |
| D_minus_D1 | auc | 1.3319 | 1.0594 | [0.9765, 1.1494] | 13676 / 1999 |
| D_minus_D1 | fpr | -16.7839 | -15.7534 | [-16.3301, -15.1989] | 13678 / 1999 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 164 / 5095 | 557 / 8595 | 3.2188 / 4.4235 | 6.4805 / 4.0740 | -393 | -1.3612 [-1.7110, -0.9912] |
| D1 | 201 / 5095 | 3278 / 8595 | 3.9450 / 6.2780 | 38.1385 / 30.0261 | -3077 | -20.0637 [-21.2470, -18.9156] |
| D | 267 / 5095 | 1407 / 8595 | 5.2404 / 6.5793 | 16.3700 / 11.5398 | -1140 | -6.1004 [-6.7563, -5.4506] |

## tal_maskfail

2550 candidates; 981 eligible images; 1019 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.7192 / 63.6636 | 0.0000 / 0.0000 | 86.4799 / 85.3681 | 90.5813 / 90.1532 | 35.2505 / 32.9777 |
| B | 62.4548 / 62.9257 | 5.1765 / 5.9439 | 90.3883 / 89.1167 | 90.5208 / 90.1309 | 43.9722 / 40.6654 |
| D1 | 57.5165 / 58.6633 | 6.4314 / 8.0693 | 89.8134 / 88.8291 | 88.0766 / 88.1233 | 56.8935 / 52.8521 |
| D | 60.7243 / 61.2492 | 8.3529 / 9.2573 | 86.8995 / 85.7231 | 89.5906 / 89.3382 | 42.9161 / 39.6379 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.7305 | -1.6764 | [-2.1751, -1.1828] | 2550 / 981 |
| D_minus_B | mask75 | 3.1765 | 3.3134 | [1.7568, 4.8984] | 2550 / 981 |
| D_minus_B | coverage | -3.4889 | -3.3936 | [-4.0445, -2.7661] | 2550 / 981 |
| D_minus_B | auc | -0.9303 | -0.7927 | [-1.0025, -0.5808] | 2550 / 981 |
| D_minus_B | fpr | -1.0560 | -1.0275 | [-1.8258, -0.1961] | 2550 / 981 |
| D_minus_A | iou | -2.9950 | -2.4144 | [-2.9254, -1.9023] | 2550 / 981 |
| D_minus_A | mask75 | 8.3529 | 9.2573 | [7.8215, 10.7621] | 2550 / 981 |
| D_minus_A | coverage | 0.4196 | 0.3550 | [-0.2681, 0.9886] | 2550 / 981 |
| D_minus_A | auc | -0.9908 | -0.8150 | [-1.0513, -0.5824] | 2550 / 981 |
| D_minus_A | fpr | 7.6656 | 6.6602 | [5.7963, 7.5522] | 2550 / 981 |
| B_minus_A | iou | -1.2645 | -0.7379 | [-1.0532, -0.3949] | 2550 / 981 |
| B_minus_A | mask75 | 5.1765 | 5.9439 | [4.7825, 7.1802] | 2550 / 981 |
| B_minus_A | coverage | 3.9084 | 3.7486 | [3.3928, 4.1323] | 2550 / 981 |
| B_minus_A | auc | -0.0605 | -0.0222 | [-0.1370, 0.0950] | 2550 / 981 |
| B_minus_A | fpr | 8.7217 | 7.6877 | [7.1742, 8.2197] | 2550 / 981 |
| D1_minus_A | iou | -6.2028 | -5.0003 | [-5.6690, -4.3042] | 2550 / 981 |
| D1_minus_A | mask75 | 6.4314 | 8.0693 | [6.6612, 9.5533] | 2550 / 981 |
| D1_minus_A | coverage | 3.3335 | 3.4611 | [2.7000, 4.2488] | 2550 / 981 |
| D1_minus_A | auc | -2.5048 | -2.0299 | [-2.3374, -1.7145] | 2550 / 981 |
| D1_minus_A | fpr | 21.6430 | 19.8745 | [18.5720, 21.1869] | 2550 / 981 |
| D_minus_D1 | iou | 3.2078 | 2.5859 | [1.9399, 3.2131] | 2550 / 981 |
| D_minus_D1 | mask75 | 1.9216 | 1.1880 | [-0.5432, 2.8021] | 2550 / 981 |
| D_minus_D1 | coverage | -2.9140 | -3.1060 | [-3.8513, -2.3781] | 2550 / 981 |
| D_minus_D1 | auc | 1.5140 | 1.2149 | [1.0025, 1.4293] | 2550 / 981 |
| D_minus_D1 | fpr | -13.9774 | -13.2142 | [-14.1882, -12.2422] | 2550 / 981 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 132 / 2550 | 0 / 0 | 5.1765 / 5.9439 | NA / NA | 132 | 5.9439 [4.7825, 7.1802] |
| D1 | 164 / 2550 | 0 / 0 | 6.4314 / 8.0693 | NA / NA | 164 | 8.0693 [6.6612, 9.5533] |
| D | 213 / 2550 | 0 / 0 | 8.3529 / 9.2573 | NA / NA | 213 | 9.2573 [7.8215, 10.7621] |

## strict_maskfail

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## baseline_success

8021 candidates; 1962 eligible images; 38 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 86.8540 / 88.0511 | 100.0000 / 100.0000 | 94.9896 / 95.2697 | 97.7000 / 98.0313 | 17.6936 / 15.2518 |
| B | 86.0673 / 87.5495 | 94.2526 / 96.3254 | 95.8307 / 95.8343 | 97.7680 / 98.0892 | 21.2415 / 17.5418 |
| D1 | 76.2358 / 78.7038 | 62.5732 / 70.1974 | 95.7407 / 95.9088 | 95.9398 / 96.6439 | 45.9574 / 39.9855 |
| D | 82.9997 / 84.9940 | 84.3785 / 88.7297 | 94.9495 / 95.2449 | 97.2115 / 97.6537 | 26.9805 / 22.7132 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.0677 | -2.5555 | [-2.7671, -2.3536] | 8021 / 1962 |
| D_minus_B | mask75 | -9.8741 | -7.5957 | [-8.3897, -6.8037] | 8021 / 1962 |
| D_minus_B | coverage | -0.8812 | -0.5894 | [-0.7760, -0.4276] | 8021 / 1962 |
| D_minus_B | auc | -0.5565 | -0.4355 | [-0.4953, -0.3802] | 8019 / 1962 |
| D_minus_B | fpr | 5.7390 | 5.1714 | [4.6996, 5.6413] | 8019 / 1962 |
| D_minus_A | iou | -3.8544 | -3.0571 | [-3.2815, -2.8421] | 8021 / 1962 |
| D_minus_A | mask75 | -15.6215 | -11.2703 | [-12.1517, -10.3929] | 8021 / 1962 |
| D_minus_A | coverage | -0.0401 | -0.0248 | [-0.2129, 0.1370] | 8021 / 1962 |
| D_minus_A | auc | -0.4885 | -0.3776 | [-0.4406, -0.3207] | 8019 / 1962 |
| D_minus_A | fpr | 9.2869 | 7.4614 | [6.9695, 7.9646] | 8019 / 1962 |
| B_minus_A | iou | -0.7867 | -0.5016 | [-0.5630, -0.4465] | 8021 / 1962 |
| B_minus_A | mask75 | -5.7474 | -3.6746 | [-4.1339, -3.2564] | 8021 / 1962 |
| B_minus_A | coverage | 0.8411 | 0.5646 | [0.5242, 0.6060] | 8021 / 1962 |
| B_minus_A | auc | 0.0681 | 0.0579 | [0.0453, 0.0713] | 8019 / 1962 |
| B_minus_A | fpr | 3.5479 | 2.2900 | [2.1331, 2.4505] | 8019 / 1962 |
| D1_minus_A | iou | -10.6182 | -9.3473 | [-9.7496, -8.9649] | 8021 / 1962 |
| D1_minus_A | mask75 | -37.4268 | -29.8026 | [-31.2228, -28.3799] | 8021 / 1962 |
| D1_minus_A | coverage | 0.7511 | 0.6391 | [0.3525, 0.9028] | 8021 / 1962 |
| D1_minus_A | auc | -1.7602 | -1.3875 | [-1.5085, -1.2741] | 8019 / 1962 |
| D1_minus_A | fpr | 28.2638 | 24.7337 | [23.8399, 25.6334] | 8019 / 1962 |
| D_minus_D1 | iou | 6.7638 | 6.2902 | [5.9955, 6.6018] | 8021 / 1962 |
| D_minus_D1 | mask75 | 21.8053 | 18.5323 | [17.3252, 19.7380] | 8021 / 1962 |
| D_minus_D1 | coverage | -0.7912 | -0.6639 | [-0.8953, -0.4209] | 8021 / 1962 |
| D_minus_D1 | auc | 1.2717 | 1.0099 | [0.9292, 1.0970] | 8019 / 1962 |
| D_minus_D1 | fpr | -18.9769 | -17.2723 | [-17.9177, -16.6602] | 8019 / 1962 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 461 / 8021 | NA / NA | 5.7474 / 3.6746 | -461 | -3.6746 [-4.1339, -3.2564] |
| D1 | 0 / 0 | 3002 / 8021 | NA / NA | 37.4268 / 29.8026 | -3002 | -29.8026 [-31.2228, -28.3799] |
| D | 0 / 0 | 1253 / 8021 | NA / NA | 15.6215 / 11.2703 | -1253 | -11.2703 [-12.1517, -10.3929] |

## all/size/small

5736 candidates; 988 eligible images; 1012 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 65.5007 / 67.6379 | 37.2908 / 41.4637 | 89.6383 / 90.5798 | 89.4452 / 91.1263 | 45.8618 / 40.5366 |
| B | 63.4228 / 65.6507 | 32.0258 / 36.9307 | 92.1509 / 92.8416 | 89.3473 / 90.9762 | 56.1358 / 49.5470 |
| D1 | 59.5966 / 61.8779 | 23.0126 / 28.8290 | 90.2805 / 90.1193 | 87.3498 / 89.4662 | 65.3453 / 57.0185 |
| D | 62.3055 / 64.6417 | 30.4219 / 35.6765 | 88.6966 / 89.5910 | 88.5026 / 90.5141 | 52.7265 / 46.5318 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.1173 | -1.0090 | [-1.3018, -0.7377] | 5736 / 988 |
| D_minus_B | mask75 | -1.6039 | -1.2542 | [-2.3402, -0.2390] | 5736 / 988 |
| D_minus_B | coverage | -3.4543 | -3.2507 | [-3.7973, -2.7591] | 5736 / 988 |
| D_minus_B | auc | -0.8447 | -0.4621 | [-0.6900, -0.2064] | 5725 / 988 |
| D_minus_B | fpr | -3.4094 | -3.0151 | [-3.8000, -2.2630] | 5727 / 988 |
| D_minus_A | iou | -3.1952 | -2.9962 | [-3.3250, -2.6879] | 5736 / 988 |
| D_minus_A | mask75 | -6.8689 | -5.7872 | [-6.9869, -4.6556] | 5736 / 988 |
| D_minus_A | coverage | -0.9417 | -0.9888 | [-1.4841, -0.5357] | 5736 / 988 |
| D_minus_A | auc | -0.9426 | -0.6122 | [-0.8249, -0.3779] | 5725 / 988 |
| D_minus_A | fpr | 6.8647 | 5.9952 | [5.2386, 6.7333] | 5727 / 988 |
| B_minus_A | iou | -2.0779 | -1.9871 | [-2.1475, -1.8360] | 5736 / 988 |
| B_minus_A | mask75 | -5.2650 | -4.5329 | [-5.4434, -3.7114] | 5736 / 988 |
| B_minus_A | coverage | 2.5126 | 2.2619 | [2.0479, 2.4866] | 5736 / 988 |
| B_minus_A | auc | -0.0979 | -0.1501 | [-0.2966, -0.0249] | 5725 / 988 |
| B_minus_A | fpr | 10.2740 | 9.0104 | [8.5350, 9.5080] | 5727 / 988 |
| D1_minus_A | iou | -5.9041 | -5.7600 | [-6.2325, -5.3239] | 5736 / 988 |
| D1_minus_A | mask75 | -14.2782 | -12.6347 | [-14.1660, -11.1946] | 5736 / 988 |
| D1_minus_A | coverage | 0.6423 | -0.4604 | [-1.0789, 0.0893] | 5736 / 988 |
| D1_minus_A | auc | -2.0954 | -1.6600 | [-1.9086, -1.4140] | 5725 / 988 |
| D1_minus_A | fpr | 19.4835 | 16.4820 | [15.4099, 17.5566] | 5727 / 988 |
| D_minus_D1 | iou | 2.7089 | 2.7638 | [2.3646, 3.1643] | 5736 / 988 |
| D_minus_D1 | mask75 | 7.4093 | 6.8475 | [5.5211, 8.1859] | 5736 / 988 |
| D_minus_D1 | coverage | -1.5839 | -0.5284 | [-1.0631, 0.0226] | 5736 / 988 |
| D_minus_D1 | auc | 1.1528 | 1.0479 | [0.8853, 1.2179] | 5725 / 988 |
| D_minus_D1 | fpr | -12.6188 | -10.4867 | [-11.3564, -9.6236] | 5727 / 988 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 45 / 3597 | 347 / 2139 | 1.2510 / 1.3368 | 16.2225 / 15.3641 | -302 | -4.5329 [-5.4434, -3.7114] |
| D1 | 68 / 3597 | 887 / 2139 | 1.8905 / 2.5935 | 41.4680 / 37.4391 | -819 | -12.6347 [-14.1660, -11.1946] |
| D | 124 / 3597 | 518 / 2139 | 3.4473 / 3.6639 | 24.2169 / 22.5341 | -394 | -5.7872 [-6.9869, -4.6556] |

## all/size/medium

4630 candidates; 1390 eligible images; 610 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 80.1663 / 80.9868 | 75.6371 / 77.2433 | 90.9262 / 91.7172 | 95.9649 / 96.3610 | 18.2206 / 18.1809 |
| B | 79.6061 / 80.5411 | 73.5205 / 75.7471 | 93.3400 / 93.5521 | 96.0219 / 96.4203 | 23.0574 / 21.9168 |
| D1 | 70.0158 / 73.2191 | 46.9546 / 55.9845 | 92.7527 / 92.2851 | 93.5488 / 94.5731 | 45.5903 / 37.9594 |
| D | 76.1490 / 78.2913 | 64.5356 / 70.0919 | 91.0798 / 91.5134 | 95.1774 / 95.7989 | 27.4943 / 24.1674 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.4571 | -2.2498 | [-2.5618, -1.9595] | 4630 / 1390 |
| D_minus_B | mask75 | -8.9849 | -5.6552 | [-6.7890, -4.5384] | 4630 / 1390 |
| D_minus_B | coverage | -2.2602 | -2.0387 | [-2.3715, -1.7332] | 4630 / 1390 |
| D_minus_B | auc | -0.8445 | -0.6214 | [-0.7511, -0.5005] | 4629 / 1390 |
| D_minus_B | fpr | 4.4370 | 2.2506 | [1.6448, 2.8483] | 4629 / 1390 |
| D_minus_A | iou | -4.0173 | -2.6954 | [-3.0138, -2.3911] | 4630 / 1390 |
| D_minus_A | mask75 | -11.1015 | -7.1514 | [-8.3285, -5.9648] | 4630 / 1390 |
| D_minus_A | coverage | 0.1536 | -0.2038 | [-0.5472, 0.1189] | 4630 / 1390 |
| D_minus_A | auc | -0.7875 | -0.5621 | [-0.7047, -0.4305] | 4629 / 1390 |
| D_minus_A | fpr | 9.2737 | 5.9864 | [5.3480, 6.6304] | 4629 / 1390 |
| B_minus_A | iou | -0.5602 | -0.4456 | [-0.5939, -0.3049] | 4630 / 1390 |
| B_minus_A | mask75 | -2.1166 | -1.4963 | [-2.2670, -0.7304] | 4630 / 1390 |
| B_minus_A | coverage | 2.4138 | 1.8349 | [1.6630, 2.0257] | 4630 / 1390 |
| B_minus_A | auc | 0.0570 | 0.0593 | [0.0128, 0.1042] | 4629 / 1390 |
| B_minus_A | fpr | 4.8368 | 3.7358 | [3.4549, 4.0368] | 4629 / 1390 |
| D1_minus_A | iou | -10.1505 | -7.7676 | [-8.2644, -7.2780] | 4630 / 1390 |
| D1_minus_A | mask75 | -28.6825 | -21.2589 | [-22.9695, -19.5345] | 4630 / 1390 |
| D1_minus_A | coverage | 1.8264 | 0.5679 | [0.1229, 1.0139] | 4630 / 1390 |
| D1_minus_A | auc | -2.4161 | -1.7879 | [-2.0085, -1.5815] | 4629 / 1390 |
| D1_minus_A | fpr | 27.3697 | 19.7785 | [18.6816, 20.8989] | 4629 / 1390 |
| D_minus_D1 | iou | 6.1332 | 5.0722 | [4.6684, 5.4823] | 4630 / 1390 |
| D_minus_D1 | mask75 | 17.5810 | 14.1074 | [12.6250, 15.5392] | 4630 / 1390 |
| D_minus_D1 | coverage | -1.6729 | -0.7717 | [-1.1730, -0.3496] | 4630 / 1390 |
| D_minus_D1 | auc | 1.6286 | 1.2257 | [1.0993, 1.3621] | 4629 / 1390 |
| D_minus_D1 | fpr | -18.0960 | -13.7921 | [-14.5864, -13.0248] | 4629 / 1390 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 79 / 1128 | 177 / 3502 | 7.0035 / 7.2014 | 5.0543 / 4.3898 | -98 | -1.4963 [-2.2670, -0.7304] |
| D1 | 65 / 1128 | 1393 / 3502 | 5.7624 / 6.3592 | 39.7773 / 31.2139 | -1328 | -21.2589 [-22.9695, -19.5345] |
| D | 87 / 1128 | 601 / 3502 | 7.7128 / 8.3159 | 17.1616 / 12.7252 | -514 | -7.1514 [-8.3285, -5.9648] |

## all/size/large

3324 candidates; 1639 eligible images; 361 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 86.5356 / 87.5635 | 88.8688 / 91.2418 | 92.3634 / 93.2468 | 96.9522 / 97.4494 | 11.3863 / 11.0974 |
| B | 86.6893 / 87.7037 | 89.0794 / 91.5204 | 93.4446 / 94.0029 | 96.9996 / 97.4997 | 13.1576 / 12.3145 |
| D1 | 75.4659 / 77.9311 | 60.8905 / 67.8238 | 94.8930 / 95.4927 | 95.1933 / 96.0631 | 43.9076 / 39.5813 |
| D | 83.2180 / 84.9716 | 81.8893 / 86.1729 | 93.3355 / 94.0931 | 96.4203 / 97.0589 | 21.7716 / 19.5435 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.4713 | -2.7322 | [-2.9788, -2.4893] | 3324 / 1639 |
| D_minus_B | mask75 | -7.1901 | -5.3475 | [-6.2753, -4.4697] | 3324 / 1639 |
| D_minus_B | coverage | -0.1091 | 0.0902 | [-0.1330, 0.3028] | 3324 / 1639 |
| D_minus_B | auc | -0.5793 | -0.4408 | [-0.5238, -0.3610] | 3322 / 1638 |
| D_minus_B | fpr | 8.6140 | 7.2291 | [6.6668, 7.8093] | 3322 / 1638 |
| D_minus_A | iou | -3.3176 | -2.5920 | [-2.8587, -2.3335] | 3324 / 1639 |
| D_minus_A | mask75 | -6.9795 | -5.0689 | [-6.0473, -4.1428] | 3324 / 1639 |
| D_minus_A | coverage | 0.9720 | 0.8463 | [0.6096, 1.0775] | 3324 / 1639 |
| D_minus_A | auc | -0.5319 | -0.3905 | [-0.4801, -0.3008] | 3322 / 1638 |
| D_minus_A | fpr | 10.3853 | 8.4461 | [7.8341, 9.0679] | 3322 / 1638 |
| B_minus_A | iou | 0.1536 | 0.1402 | [0.0410, 0.2474] | 3324 / 1639 |
| B_minus_A | mask75 | 0.2106 | 0.2786 | [-0.1993, 0.7688] | 3324 / 1639 |
| B_minus_A | coverage | 1.0812 | 0.7561 | [0.6409, 0.8823] | 3324 / 1639 |
| B_minus_A | auc | 0.0474 | 0.0503 | [0.0146, 0.0858] | 3322 / 1638 |
| B_minus_A | fpr | 1.7713 | 1.2171 | [1.0528, 1.3921] | 3322 / 1638 |
| D1_minus_A | iou | -11.0698 | -9.6324 | [-10.1598, -9.1116] | 3324 / 1639 |
| D1_minus_A | mask75 | -27.9783 | -23.4180 | [-25.1652, -21.6666] | 3324 / 1639 |
| D1_minus_A | coverage | 2.5296 | 2.2459 | [1.8073, 2.6689] | 3324 / 1639 |
| D1_minus_A | auc | -1.7590 | -1.3864 | [-1.5470, -1.2401] | 3322 / 1638 |
| D1_minus_A | fpr | 32.5213 | 28.4840 | [27.4024, 29.5685] | 3322 / 1638 |
| D_minus_D1 | iou | 7.7521 | 7.0405 | [6.6276, 7.4484] | 3324 / 1639 |
| D_minus_D1 | mask75 | 20.9988 | 18.3491 | [16.7553, 20.0044] | 3324 / 1639 |
| D_minus_D1 | coverage | -1.5576 | -1.3996 | [-1.7400, -1.0315] | 3324 / 1639 |
| D_minus_D1 | auc | 1.2270 | 0.9958 | [0.8949, 1.1026] | 3322 / 1638 |
| D_minus_D1 | fpr | -22.1360 | -20.0378 | [-20.8087, -19.2715] | 3322 / 1638 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 40 / 370 | 33 / 2954 | 10.8108 / 11.1661 | 1.1171 / 0.9223 | 7 | 0.2786 [-0.1993, 0.7688] |
| D1 | 68 / 370 | 998 / 2954 | 18.3784 / 18.7569 | 33.7847 / 28.4292 | -930 | -23.4180 [-25.1652, -21.6666] |
| D | 56 / 370 | 288 / 2954 | 15.1351 / 15.1815 | 9.7495 / 7.6236 | -232 | -5.0689 [-6.0473, -4.1428] |

## all/pyramid/P3

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## all/pyramid/P4

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## all/pyramid/P5

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## tal_maskfail/size/small

1551 candidates; 624 eligible images; 1376 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.8957 / 63.9373 | 0.0000 / 0.0000 | 90.9932 / 91.0335 | 90.9461 / 91.1378 | 44.3427 / 43.3385 |
| B | 61.3719 / 61.5045 | 1.9987 / 1.9467 | 93.7394 / 93.6218 | 90.8307 / 91.0353 | 54.1350 / 52.5280 |
| D1 | 57.5294 / 57.9140 | 3.3527 / 4.0425 | 91.3341 / 90.7190 | 88.6605 / 89.0745 | 62.1441 / 59.4005 |
| D | 60.6161 / 60.7610 | 6.1251 / 6.6536 | 90.2378 / 90.0906 | 89.9925 / 90.2763 | 50.7579 / 49.0983 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.7557 | -0.7435 | [-1.2526, -0.2641] | 1551 / 624 |
| D_minus_B | mask75 | 4.1264 | 4.7070 | [3.1758, 6.2559] | 1551 / 624 |
| D_minus_B | coverage | -3.5017 | -3.5312 | [-4.3302, -2.8187] | 1551 / 624 |
| D_minus_B | auc | -0.8382 | -0.7590 | [-0.9862, -0.5392] | 1551 / 624 |
| D_minus_B | fpr | -3.3771 | -3.4297 | [-4.4775, -2.4070] | 1551 / 624 |
| D_minus_A | iou | -3.2796 | -3.1763 | [-3.7154, -2.6782] | 1551 / 624 |
| D_minus_A | mask75 | 6.1251 | 6.6536 | [5.1777, 8.2204] | 1551 / 624 |
| D_minus_A | coverage | -0.7554 | -0.9429 | [-1.6842, -0.2749] | 1551 / 624 |
| D_minus_A | auc | -0.9536 | -0.8615 | [-1.1139, -0.6188] | 1551 / 624 |
| D_minus_A | fpr | 6.4152 | 5.7598 | [4.6921, 6.8279] | 1551 / 624 |
| B_minus_A | iou | -2.5239 | -2.4328 | [-2.7157, -2.1583] | 1551 / 624 |
| B_minus_A | mask75 | 1.9987 | 1.9467 | [1.1752, 2.8072] | 1551 / 624 |
| B_minus_A | coverage | 2.7463 | 2.5883 | [2.3035, 2.9086] | 1551 / 624 |
| B_minus_A | auc | -0.1154 | -0.1025 | [-0.2471, 0.0273] | 1551 / 624 |
| B_minus_A | fpr | 9.7924 | 9.1895 | [8.5638, 9.8238] | 1551 / 624 |
| D1_minus_A | iou | -6.3663 | -6.0234 | [-6.6703, -5.4194] | 1551 / 624 |
| D1_minus_A | mask75 | 3.3527 | 4.0425 | [2.7635, 5.4253] | 1551 / 624 |
| D1_minus_A | coverage | 0.3409 | -0.3145 | [-1.0995, 0.4050] | 1551 / 624 |
| D1_minus_A | auc | -2.2856 | -2.0633 | [-2.4096, -1.7345] | 1551 / 624 |
| D1_minus_A | fpr | 17.8015 | 16.0620 | [14.6834, 17.5066] | 1551 / 624 |
| D_minus_D1 | iou | 3.0867 | 2.8471 | [2.2184, 3.4794] | 1551 / 624 |
| D_minus_D1 | mask75 | 2.7724 | 2.6111 | [1.0261, 4.2394] | 1551 / 624 |
| D_minus_D1 | coverage | -1.0964 | -0.6284 | [-1.5558, 0.2794] | 1551 / 624 |
| D_minus_D1 | auc | 1.3320 | 1.2018 | [0.9784, 1.4290] | 1551 / 624 |
| D_minus_D1 | fpr | -11.3863 | -10.3022 | [-11.5771, -9.0509] | 1551 / 624 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 31 / 1551 | 0 / 0 | 1.9987 / 1.9467 | NA / NA | 31 | 1.9467 [1.1752, 2.8072] |
| D1 | 52 / 1551 | 0 / 0 | 3.3527 / 4.0425 | NA / NA | 52 | 4.0425 [2.7635, 5.4253] |
| D | 95 / 1551 | 0 / 0 | 6.1251 / 6.6536 | NA / NA | 95 | 6.6536 [5.1777, 8.2204] |

## tal_maskfail/size/medium

727 candidates; 489 eligible images; 1511 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 64.6660 / 65.4406 | 0.0000 / 0.0000 | 82.2869 / 83.0817 | 91.9215 / 92.1311 | 21.4276 / 21.5323 |
| B | 64.6851 / 65.4376 | 9.2160 / 8.9673 | 87.9692 / 87.9456 | 91.9321 / 92.2551 | 28.4980 / 27.6280 |
| D1 | 56.5550 / 58.1688 | 7.2902 / 7.3415 | 87.7848 / 87.3884 | 88.8677 / 89.6525 | 46.8646 / 43.7781 |
| D | 61.2653 / 62.6210 | 9.9037 / 10.4158 | 83.6433 / 83.8633 | 90.8808 / 91.4287 | 29.9899 / 28.3610 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.4199 | -2.8167 | [-3.6158, -2.0120] | 727 / 489 |
| D_minus_B | mask75 | 0.6878 | 1.4485 | [-1.2304, 4.0833] | 727 / 489 |
| D_minus_B | coverage | -4.3259 | -4.0823 | [-5.2490, -2.9430] | 727 / 489 |
| D_minus_B | auc | -1.0513 | -0.8264 | [-1.1061, -0.5598] | 727 / 489 |
| D_minus_B | fpr | 1.4919 | 0.7330 | [-0.5806, 2.0165] | 727 / 489 |
| D_minus_A | iou | -3.4007 | -2.8197 | [-3.7040, -1.9286] | 727 / 489 |
| D_minus_A | mask75 | 9.9037 | 10.4158 | [7.9721, 12.9042] | 727 / 489 |
| D_minus_A | coverage | 1.3564 | 0.7816 | [-0.3529, 1.9475] | 727 / 489 |
| D_minus_A | auc | -1.0407 | -0.7024 | [-1.0212, -0.3882] | 727 / 489 |
| D_minus_A | fpr | 8.5623 | 6.8288 | [5.4814, 8.1983] | 727 / 489 |
| B_minus_A | iou | 0.0192 | -0.0030 | [-0.5200, 0.5324] | 727 / 489 |
| B_minus_A | mask75 | 9.2160 | 8.9673 | [6.7791, 11.2307] | 727 / 489 |
| B_minus_A | coverage | 5.6822 | 4.8639 | [4.2976, 5.4696] | 727 / 489 |
| B_minus_A | auc | 0.0106 | 0.1240 | [-0.0133, 0.2622] | 727 / 489 |
| B_minus_A | fpr | 7.0704 | 6.0957 | [5.4403, 6.7984] | 727 / 489 |
| D1_minus_A | iou | -8.1110 | -7.2718 | [-8.3856, -6.1397] | 727 / 489 |
| D1_minus_A | mask75 | 7.2902 | 7.3415 | [5.3408, 9.5331] | 727 / 489 |
| D1_minus_A | coverage | 5.4979 | 4.3067 | [3.0710, 5.5613] | 727 / 489 |
| D1_minus_A | auc | -3.0538 | -2.4786 | [-2.9571, -2.0085] | 727 / 489 |
| D1_minus_A | fpr | 25.4371 | 22.2459 | [20.1294, 24.4253] | 727 / 489 |
| D_minus_D1 | iou | 4.7102 | 4.4521 | [3.4585, 5.4207] | 727 / 489 |
| D_minus_D1 | mask75 | 2.6135 | 3.0743 | [0.7736, 5.3478] | 727 / 489 |
| D_minus_D1 | coverage | -4.1415 | -3.5251 | [-4.7700, -2.2881] | 727 / 489 |
| D_minus_D1 | auc | 2.0131 | 1.7762 | [1.4547, 2.0998] | 727 / 489 |
| D_minus_D1 | fpr | -16.8747 | -15.4171 | [-17.0564, -13.7837] | 727 / 489 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 67 / 727 | 0 / 0 | 9.2160 / 8.9673 | NA / NA | 67 | 8.9673 [6.7791, 11.2307] |
| D1 | 53 / 727 | 0 / 0 | 7.2902 / 7.3415 | NA / NA | 53 | 7.3415 [5.3408, 9.5331] |
| D | 72 / 727 | 0 / 0 | 9.9037 / 10.4158 | NA / NA | 72 | 10.4158 [7.9721, 12.9042] |

## tal_maskfail/size/large

272 candidates; 243 eligible images; 1757 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 60.1824 / 60.0842 | 0.0000 / 0.0000 | 71.9514 / 72.0965 | 84.9194 / 84.5058 | 20.3509 / 20.8038 |
| B | 62.6684 / 62.4194 | 12.5000 / 12.1399 | 77.7455 / 77.7158 | 84.9823 / 84.4419 | 27.3808 / 28.0312 |
| D1 | 60.0122 / 60.0508 | 21.6912 / 20.7819 | 86.5643 / 86.4136 | 82.6325 / 82.3093 | 53.7584 / 53.6532 |
| D | 59.8949 / 59.7496 | 16.9118 / 16.2551 | 76.5670 / 76.3014 | 83.8505 / 83.3788 | 32.7502 / 32.7261 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.7736 | -2.6698 | [-4.2210, -1.1003] | 272 / 243 |
| D_minus_B | mask75 | 4.4118 | 4.1152 | [-0.4115, 8.6420] | 272 / 243 |
| D_minus_B | coverage | -1.1786 | -1.4143 | [-3.3485, 0.4830] | 272 / 243 |
| D_minus_B | auc | -1.1317 | -1.0631 | [-1.8465, -0.2901] | 272 / 243 |
| D_minus_B | fpr | 5.3693 | 4.6948 | [2.4941, 7.0181] | 272 / 243 |
| D_minus_A | iou | -0.2875 | -0.3346 | [-1.9727, 1.3822] | 272 / 243 |
| D_minus_A | mask75 | 16.9118 | 16.2551 | [11.9342, 20.7819] | 272 / 243 |
| D_minus_A | coverage | 4.6155 | 4.2050 | [2.2451, 6.2106] | 272 / 243 |
| D_minus_A | auc | -1.0689 | -1.1270 | [-2.0071, -0.2876] | 272 / 243 |
| D_minus_A | fpr | 12.3992 | 11.9222 | [9.5112, 14.4525] | 272 / 243 |
| B_minus_A | iou | 2.4860 | 2.3352 | [1.2726, 3.4617] | 272 / 243 |
| B_minus_A | mask75 | 12.5000 | 12.1399 | [8.2305, 16.2551] | 272 / 243 |
| B_minus_A | coverage | 5.7941 | 5.6193 | [4.3269, 7.0098] | 272 / 243 |
| B_minus_A | auc | 0.0629 | -0.0639 | [-0.4708, 0.3252] | 272 / 243 |
| B_minus_A | fpr | 7.0299 | 7.2274 | [5.7359, 8.8713] | 272 / 243 |
| D1_minus_A | iou | -0.1702 | -0.0334 | [-2.2505, 2.2190] | 272 / 243 |
| D1_minus_A | mask75 | 21.6912 | 20.7819 | [16.0494, 25.7202] | 272 / 243 |
| D1_minus_A | coverage | 14.6129 | 14.3171 | [12.1400, 16.5710] | 272 / 243 |
| D1_minus_A | auc | -2.2869 | -2.1964 | [-3.1949, -1.2057] | 272 / 243 |
| D1_minus_A | fpr | 33.4075 | 32.8493 | [29.7294, 36.1889] | 272 / 243 |
| D_minus_D1 | iou | -0.1174 | -0.3011 | [-2.2897, 1.6895] | 272 / 243 |
| D_minus_D1 | mask75 | -4.7794 | -4.5267 | [-10.0823, 0.8230] | 272 / 243 |
| D_minus_D1 | coverage | -9.9973 | -10.1121 | [-12.1230, -8.2258] | 272 / 243 |
| D_minus_D1 | auc | 1.2180 | 1.0694 | [0.4391, 1.7242] | 272 / 243 |
| D_minus_D1 | fpr | -21.0082 | -20.9271 | [-23.1840, -18.7562] | 272 / 243 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 34 / 272 | 0 / 0 | 12.5000 / 12.1399 | NA / NA | 34 | 12.1399 [8.2305, 16.2551] |
| D1 | 59 / 272 | 0 / 0 | 21.6912 / 20.7819 | NA / NA | 59 | 20.7819 [16.0494, 25.7202] |
| D | 46 / 272 | 0 / 0 | 16.9118 / 16.2551 | NA / NA | 46 | 16.2551 [11.9342, 20.7819] |

## tal_maskfail/pyramid/P3

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## tal_maskfail/pyramid/P4

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## tal_maskfail/pyramid/P5

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## strict_maskfail/size/small

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## strict_maskfail/size/medium

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## strict_maskfail/size/large

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## strict_maskfail/pyramid/P3

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## strict_maskfail/pyramid/P4

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## strict_maskfail/pyramid/P5

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## baseline_success/size/small

1900 candidates; 708 eligible images; 1292 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 82.6018 / 82.6115 | 100.0000 / 100.0000 | 95.2622 / 95.3002 | 96.2686 / 96.3798 | 31.8278 / 30.7884 |
| B | 80.8006 / 80.8916 | 84.8947 / 85.7889 | 96.3656 / 96.3114 | 96.3714 / 96.4901 | 39.4786 / 37.8652 |
| D1 | 75.0620 / 75.6693 | 58.9474 / 62.5500 | 95.7714 / 95.6145 | 94.9362 / 95.2980 | 55.8506 / 52.1560 |
| D | 78.9492 / 79.3090 | 76.3158 / 77.9484 | 94.7373 / 94.9036 | 95.9068 / 96.0999 | 40.6752 / 38.5475 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.8514 | -1.5826 | [-1.9217, -1.2514] | 1900 / 708 |
| D_minus_B | mask75 | -8.5789 | -7.8405 | [-9.9371, -5.7023] | 1900 / 708 |
| D_minus_B | coverage | -1.6283 | -1.4079 | [-1.7309, -1.1094] | 1900 / 708 |
| D_minus_B | auc | -0.4646 | -0.3903 | [-0.4992, -0.2878] | 1898 / 708 |
| D_minus_B | fpr | 1.1966 | 0.6823 | [-0.2009, 1.5628] | 1898 / 708 |
| D_minus_A | iou | -3.6527 | -3.3025 | [-3.6938, -2.9201] | 1900 / 708 |
| D_minus_A | mask75 | -23.6842 | -22.0516 | [-24.3984, -19.6818] | 1900 / 708 |
| D_minus_A | coverage | -0.5250 | -0.3967 | [-0.7023, -0.1053] | 1900 / 708 |
| D_minus_A | auc | -0.3618 | -0.2799 | [-0.3968, -0.1671] | 1898 / 708 |
| D_minus_A | fpr | 8.8474 | 7.7592 | [6.8009, 8.7725] | 1898 / 708 |
| B_minus_A | iou | -1.8013 | -1.7199 | [-1.8943, -1.5535] | 1900 / 708 |
| B_minus_A | mask75 | -15.1053 | -14.2111 | [-16.2058, -12.2739] | 1900 / 708 |
| B_minus_A | coverage | 1.1033 | 1.0112 | [0.9258, 1.0987] | 1900 / 708 |
| B_minus_A | auc | 0.1028 | 0.1103 | [0.0717, 0.1518] | 1898 / 708 |
| B_minus_A | fpr | 7.6508 | 7.0768 | [6.5600, 7.6046] | 1898 / 708 |
| D1_minus_A | iou | -7.5399 | -6.9422 | [-7.5452, -6.3579] | 1900 / 708 |
| D1_minus_A | mask75 | -41.0526 | -37.4500 | [-40.3935, -34.5526] | 1900 / 708 |
| D1_minus_A | coverage | 0.5091 | 0.3143 | [-0.1040, 0.6731] | 1900 / 708 |
| D1_minus_A | auc | -1.3324 | -1.0818 | [-1.2698, -0.9023] | 1898 / 708 |
| D1_minus_A | fpr | 24.0228 | 21.3676 | [19.8455, 22.9228] | 1898 / 708 |
| D_minus_D1 | iou | 3.8872 | 3.6398 | [3.1824, 4.1135] | 1900 / 708 |
| D_minus_D1 | mask75 | 17.3684 | 15.3984 | [12.8374, 17.9289] | 1900 / 708 |
| D_minus_D1 | coverage | -1.0341 | -0.7110 | [-1.0637, -0.3268] | 1900 / 708 |
| D_minus_D1 | auc | 0.9706 | 0.8019 | [0.6770, 0.9340] | 1898 / 708 |
| D_minus_D1 | fpr | -15.1754 | -13.6085 | [-14.7287, -12.4497] | 1898 / 708 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 287 / 1900 | NA / NA | 15.1053 / 14.2111 | -287 | -14.2111 [-16.2058, -12.2739] |
| D1 | 0 / 0 | 780 / 1900 | NA / NA | 41.0526 / 37.4500 | -780 | -37.4500 [-40.3935, -34.5526] |
| D | 0 / 0 | 450 / 1900 | NA / NA | 23.6842 / 22.0516 | -450 | -22.0516 [-24.3984, -19.6818] |

## baseline_success/size/medium

3270 candidates; 1228 eligible images; 772 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 86.4738 / 86.5902 | 100.0000 / 100.0000 | 94.7870 / 94.9155 | 97.8963 / 97.9343 | 16.1851 / 16.2414 |
| B | 85.6831 / 85.9420 | 95.5657 / 96.0772 | 95.8069 / 95.7860 | 97.9502 / 97.9933 | 19.5907 / 19.1014 |
| D1 | 75.5719 / 77.8632 | 60.9174 / 68.9194 | 95.3659 / 95.0425 | 95.8011 / 96.3303 | 43.5028 / 37.6378 |
| D | 82.3244 / 83.4624 | 83.6697 / 87.5473 | 94.5848 / 94.5047 | 97.2908 / 97.4871 | 25.2520 / 22.6939 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.3586 | -2.4796 | [-2.8327, -2.1644] | 3270 / 1228 |
| D_minus_B | mask75 | -11.8960 | -8.5299 | [-9.9759, -7.1566] | 3270 / 1228 |
| D_minus_B | coverage | -1.2221 | -1.2813 | [-1.5891, -0.9977] | 3270 / 1228 |
| D_minus_B | auc | -0.6595 | -0.5062 | [-0.6312, -0.4042] | 3270 / 1228 |
| D_minus_B | fpr | 5.6613 | 3.5925 | [2.9567, 4.2292] | 3270 / 1228 |
| D_minus_A | iou | -4.1493 | -3.1278 | [-3.5066, -2.7792] | 3270 / 1228 |
| D_minus_A | mask75 | -16.3303 | -12.4527 | [-13.9627, -11.0339] | 3270 / 1228 |
| D_minus_A | coverage | -0.2022 | -0.4108 | [-0.7219, -0.1332] | 3270 / 1228 |
| D_minus_A | auc | -0.6055 | -0.4472 | [-0.5789, -0.3399] | 3270 / 1228 |
| D_minus_A | fpr | 9.0668 | 6.4525 | [5.7534, 7.1655] | 3270 / 1228 |
| B_minus_A | iou | -0.7907 | -0.6482 | [-0.7461, -0.5543] | 3270 / 1228 |
| B_minus_A | mask75 | -4.4343 | -3.9228 | [-4.7756, -3.1201] | 3270 / 1228 |
| B_minus_A | coverage | 1.0199 | 0.8705 | [0.8035, 0.9390] | 3270 / 1228 |
| B_minus_A | auc | 0.0540 | 0.0590 | [0.0380, 0.0823] | 3270 / 1228 |
| B_minus_A | fpr | 3.4056 | 2.8600 | [2.6314, 3.0997] | 3270 / 1228 |
| D1_minus_A | iou | -10.9019 | -8.7271 | [-9.3050, -8.2070] | 3270 / 1228 |
| D1_minus_A | mask75 | -39.0826 | -31.0806 | [-33.2450, -29.0228] | 3270 / 1228 |
| D1_minus_A | coverage | 0.5789 | 0.1270 | [-0.3045, 0.5111] | 3270 / 1228 |
| D1_minus_A | auc | -2.0952 | -1.6040 | [-1.8336, -1.4052] | 3270 / 1228 |
| D1_minus_A | fpr | 27.3177 | 21.3964 | [20.1488, 22.6601] | 3270 / 1228 |
| D_minus_D1 | iou | 6.7526 | 5.5992 | [5.1833, 6.0393] | 3270 / 1228 |
| D_minus_D1 | mask75 | 22.7523 | 18.6279 | [16.8677, 20.4158] | 3270 / 1228 |
| D_minus_D1 | coverage | -0.7811 | -0.5378 | [-0.8873, -0.1503] | 3270 / 1228 |
| D_minus_D1 | auc | 1.4897 | 1.1568 | [1.0328, 1.3014] | 3270 / 1228 |
| D_minus_D1 | fpr | -18.2509 | -14.9439 | [-15.8230, -14.1015] | 3270 / 1228 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 145 / 3270 | NA / NA | 4.4343 / 3.9228 | -145 | -3.9228 [-4.7756, -3.1201] |
| D1 | 0 / 0 | 1278 / 3270 | NA / NA | 39.0826 / 31.0806 | -1278 | -31.0806 [-33.2450, -29.0228] |
| D | 0 / 0 | 534 / 3270 | NA / NA | 16.3303 / 12.4527 | -534 | -12.4527 [-13.9627, -11.0339] |

## baseline_success/size/large

2851 candidates; 1576 eligible images; 424 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 90.1239 / 90.3363 | 100.0000 / 100.0000 | 95.0402 / 95.2644 | 98.4277 / 98.5034 | 10.0142 / 9.9196 |
| B | 90.0181 / 90.2466 | 98.9828 / 99.1529 | 95.5015 / 95.5874 | 98.4888 / 98.5538 | 10.9938 / 10.6391 |
| D1 | 77.7797 / 79.4559 | 66.8888 / 71.7769 | 96.1500 / 96.4651 | 96.7669 / 97.1529 | 42.1865 / 38.8522 |
| D | 86.4735 / 87.2983 | 90.5647 / 92.5072 | 95.5091 / 95.7009 | 97.9891 / 98.1469 | 19.8461 / 18.2810 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.5446 | -2.9484 | [-3.2158, -2.7011] | 2851 / 1576 |
| D_minus_B | mask75 | -8.4181 | -6.6457 | [-7.6890, -5.7016] | 2851 / 1576 |
| D_minus_B | coverage | 0.0076 | 0.1135 | [-0.1108, 0.3104] | 2851 / 1576 |
| D_minus_B | auc | -0.4996 | -0.4068 | [-0.4696, -0.3508] | 2851 / 1576 |
| D_minus_B | fpr | 8.8523 | 7.6419 | [7.0261, 8.2439] | 2851 / 1576 |
| D_minus_A | iou | -3.6504 | -3.0380 | [-3.3185, -2.7803] | 2851 / 1576 |
| D_minus_A | mask75 | -9.4353 | -7.4928 | [-8.5680, -6.5044] | 2851 / 1576 |
| D_minus_A | coverage | 0.4689 | 0.4365 | [0.2126, 0.6398] | 2851 / 1576 |
| D_minus_A | auc | -0.4386 | -0.3565 | [-0.4242, -0.2934] | 2851 / 1576 |
| D_minus_A | fpr | 9.8319 | 8.3614 | [7.7138, 9.0050] | 2851 / 1576 |
| B_minus_A | iou | -0.1058 | -0.0897 | [-0.1657, -0.0191] | 2851 / 1576 |
| B_minus_A | mask75 | -1.0172 | -0.8471 | [-1.2278, -0.5097] | 2851 / 1576 |
| B_minus_A | coverage | 0.4613 | 0.3230 | [0.2716, 0.3770] | 2851 / 1576 |
| B_minus_A | auc | 0.0611 | 0.0504 | [0.0314, 0.0716] | 2851 / 1576 |
| B_minus_A | fpr | 0.9797 | 0.7196 | [0.5960, 0.8536] | 2851 / 1576 |
| D1_minus_A | iou | -12.3443 | -10.8803 | [-11.4298, -10.3384] | 2851 / 1576 |
| D1_minus_A | mask75 | -33.1112 | -28.2231 | [-30.1394, -26.3182] | 2851 / 1576 |
| D1_minus_A | coverage | 1.1098 | 1.2006 | [0.7943, 1.5857] | 2851 / 1576 |
| D1_minus_A | auc | -1.6608 | -1.3505 | [-1.4905, -1.2203] | 2851 / 1576 |
| D1_minus_A | fpr | 32.1723 | 28.9327 | [27.7708, 30.1153] | 2851 / 1576 |
| D_minus_D1 | iou | 8.6938 | 7.8423 | [7.4091, 8.2872] | 2851 / 1576 |
| D_minus_D1 | mask75 | 23.6759 | 20.7303 | [18.9705, 22.4722] | 2851 / 1576 |
| D_minus_D1 | coverage | -0.6409 | -0.7642 | [-1.1026, -0.4066] | 2851 / 1576 |
| D_minus_D1 | auc | 1.2222 | 0.9940 | [0.8988, 1.0959] | 2851 / 1576 |
| D_minus_D1 | fpr | -22.3404 | -20.5712 | [-21.4289, -19.7607] | 2851 / 1576 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 29 / 2851 | NA / NA | 1.0172 / 0.8471 | -29 | -0.8471 [-1.2278, -0.5097] |
| D1 | 0 / 0 | 944 / 2851 | NA / NA | 33.1112 / 28.2231 | -944 | -28.2231 [-30.1394, -26.3182] |
| D | 0 / 0 | 269 / 2851 | NA / NA | 9.4353 / 7.4928 | -269 | -7.4928 [-8.5680, -6.5044] |

## baseline_success/pyramid/P3

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## baseline_success/pyramid/P4

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## baseline_success/pyramid/P5

0 candidates; 0 eligible images; 2000 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| B | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D1 | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |
| D | NA / NA | NA / NA | NA / NA | NA / NA | NA / NA |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_B | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| B_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | iou | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | auc | NA | NA | [NA, NA] | 0 / 0 |
| D1_minus_A | fpr | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | iou | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | mask75 | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | coverage | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | auc | NA | NA | [NA, NA] | 0 / 0 |
| D_minus_D1 | fpr | NA | NA | [NA, NA] | 0 / 0 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D1 | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |
| D | 0 / 0 | 0 / 0 | NA / NA | NA / NA | 0 | NA [NA, NA] |

## Timing and limitations

{
  "supplied_timing_scope": "not specified; supplied row observations only",
  "direct_ms": {
    "candidate_mean": 0.5625091182915637,
    "image_macro": 0.5719952902709721,
    "valid_candidates": 13690,
    "valid_images": 2000,
    "median_ms": 0.5410541780292988,
    "p95_ms": 0.6285374518483877
  },
  "quality_ms": {
    "candidate_mean": 7.067692653365145,
    "image_macro": 7.1317222791628465,
    "valid_candidates": 13690,
    "valid_images": 2000,
    "median_ms": 7.149812765419483,
    "p95_ms": 7.645017979666591
  }
}

Timing is summarized in the supplied unit; no image-total or dataset-total latency is inferred from repeated row values. The unpopulated planned-image list cannot by itself distinguish zero matched candidates from an interrupted evaluation; completion must be established by the run and metadata.

Metadata: `{"split": "dev", "planned_images": 2000, "evaluated_images_requested": 2000, "image_ids": [85114, 485462, 571503, 39509, 155707, 448368, 320434, 174137, 224342, 83663, 122164, 135038, 543618, 375709, 370043, 322149, 157181, 156341, 288481, 144878, 31590, 241102, 255489, 18647, 448269, 357738, 166386, 267932, 224000, 295197, 388498, 339491, 333114, 100012, 526071, 42169, 247818, 179374, 368474, 121632, 29709, 470417, 155269, 542077, 35514, 262991, 100865, 132170, 359687, 473075, 270172, 511572, 193333, 237587, 469389, 317898, 396625, 17198, 388893, 576941, 411027, 364656, 186254, 358085, 464598, 206369, 3690, 496283, 354235, 335865, 559780, 411445, 165203, 39053, 71330, 459151, 102030, 534161, 66129, 73808, 511689, 388299, 547979, 455225, 198604, 306249, 486686, 472707, 340698, 358206, 535933, 15472, 294329, 431764, 206620, 381832, 208132, 443868, 314065, 203458, 514417, 42977, 306724, 12726, 378548, 136699, 248193, 110500, 303546, 335749, 87633, 169826, 516750, 129784, 345997, 516214, 273045, 3272, 450528, 147753, 213539, 295199, 170666, 542234, 87156, 344729, 333485, 539567, 267690, 312536, 104814, 376233, 451504, 162757, 489745, 340215, 187483, 579277, 240301, 529389, 400285, 457805, 191360, 212545, 94000, 29472, 344073, 573762, 298112, 94087, 500162, 252137, 426427, 519565, 367058, 65566, 487825, 498152, 214391, 48160, 228155, 557725, 265160, 55544, 493740, 401287, 454693, 188747, 260190, 422586, 545155, 260363, 401892, 26988, 14044, 547089, 200138, 565595, 29358, 545676, 351914, 5967, 462736, 474847, 248141, 429290, 142051, 360778, 578174, 535602, 393720, 136793, 565331, 529238, 487534, 178421, 29600, 295336, 179408, 130295, 580629, 11222, 578147, 17896, 126198, 154068, 485267, 387000, 492995, 108500, 293974, 549780, 267762, 276749, 205762, 471995, 468471, 263351, 576749, 485945, 81593, 158293, 154257, 493503, 378868, 407097, 253251, 367582, 413868, 370805, 299573, 105015, 361819, 416266, 53359, 59207, 52440, 485123, 417588, 448013, 303986, 550129, 134081, 137564, 312961, 311771, 42267, 287585, 390685, 53842, 47596, 212089, 322827, 96027, 519874, 499903, 206489, 70474, 424307, 282048, 483070, 327794, 306140, 552575, 442417, 51525, 387933, 4794, 535369, 310131, 72382, 530500, 549232, 542304, 156768, 275130, 522221, 524059, 428055, 18485, 434078, 395192, 419764, 574722, 523753, 297976, 14821, 266600, 144832, 532747, 205676, 58364, 383322, 204755, 261271, 567173, 55290, 99856, 225238, 270384, 4069, 277125, 275292, 54755, 139604, 152458, 573456, 549224, 166036, 442016, 42968, 238498, 542150, 499656, 517144, 459919, 379159, 555273, 442225, 544568, 189767, 413312, 318314, 581900, 266922, 529433, 504499, 169907, 295455, 225791, 21097, 484425, 523322, 36559, 252801, 281713, 401201, 536045, 265236, 545237, 142497, 145155, 168480, 114204, 332122, 411238, 40886, 164617, 389948, 276128, 498263, 490084, 31053, 554664, 51555, 44606, 411821, 223112, 196852, 108429, 71621, 205782, 467537, 233697, 412034, 42871, 81099, 497544, 447109, 286171, 542472, 272439, 158494, 95213, 70444, 224396, 95988, 327392, 465229, 60689, 447787, 17468, 414463, 521338, 459716, 83625, 130517, 316648, 64899, 318825, 378092, 105279, 121875, 571780, 128245, 409700, 450634, 148297, 554031, 308372, 539386, 153072, 446106, 126808, 411987, 294528, 401288, 322174, 157188, 439678, 60849, 330173, 75902, 275869, 38721, 213677, 84391, 556591, 252909, 570263, 100977, 102149, 253557, 254834, 88168, 444390, 504664, 306394, 402562, 467604, 514464, 63617, 531703, 499418, 106387, 89181, 357013, 246869, 118970, 487741, 319611, 442249, 361399, 456737, 89696, 86036, 275034, 439143, 532181, 532381, 411400, 410182, 458069, 272518, 31345, 2529, 176596, 320314, 424247, 50627, 538411, 537954, 395828, 282841, 80834, 126135, 511781, 469743, 47713, 310085, 234533, 433647, 24601, 494217, 66406, 256769, 452070, 407585, 431874, 79441, 565938, 538913, 186009, 169023, 141855, 427965, 214454, 168546, 215708, 552354, 568883, 276894, 67805, 28675, 158786, 332499, 151338, 410947, 14591, 512920, 37675, 532009, 240323, 548141, 70294, 20583, 297360, 122583, 550925, 494884, 433366, 365592, 125979, 107959, 549972, 86725, 454053, 34377, 513980, 303368, 16737, 45663, 477392, 262463, 547136, 451364, 410675, 456886, 388453, 431972, 479528, 206257, 234550, 281599, 157693, 72701, 471096, 71589, 395405, 358801, 526695, 304079, 503212, 77806, 212359, 541391, 577953, 403425, 49369, 565650, 378604, 457087, 464888, 135806, 15050, 517031, 57868, 139637, 234889, 270898, 471816, 184386, 466671, 71038, 538792, 575633, 394322, 56187, 510314, 100564, 23779, 231945, 168287, 293704, 93156, 432529, 233354, 166853, 213117, 438024, 489313, 395019, 300415, 456732, 517610, 181860, 64313, 54586, 198277, 63081, 164168, 276739, 516189, 492562, 564655, 186550, 496548, 564934, 478406, 535679, 530038, 168020, 540186, 349912, 434689, 107212, 313002, 60532, 342765, 572108, 88697, 74309, 188945, 343198, 554161, 391397, 145854, 178567, 414592, 428148, 366517, 297023, 385734, 423425, 412302, 481281, 294029, 370793, 266491, 145956, 14781, 266032, 460757, 552018, 571029, 428503, 308764, 410373, 399885, 455169, 95397, 287372, 448689, 344928, 292931, 70660, 358543, 481772, 550394, 532137, 523332, 178876, 139270, 286692, 547419, 58223, 449657, 467187, 244014, 503959, 572724, 449841, 366641, 21285, 67569, 374946, 283118, 12268, 278082, 346823, 431190, 47087, 409064, 359574, 13235, 371705, 264944, 570822, 480322, 45976, 293800, 418418, 357184, 560388, 195204, 534384, 346841, 194421, 336569, 416911, 45941, 97747, 342800, 270215, 2608, 443299, 128140, 148034, 357572, 35814, 215910, 537343, 13482, 64071, 427802, 458205, 52037, 246001, 440763, 55747, 343255, 465047, 191972, 120790, 25003, 401602, 306415, 344862, 522020, 356086, 490318, 113481, 267875, 54655, 126324, 453801, 23511, 32116, 37165, 115887, 261887, 331457, 277341, 137682, 542475, 315229, 342017, 142418, 214008, 19032, 253521, 418282, 18267, 124311, 339312, 341100, 450551, 423327, 80159, 288872, 39016, 534412, 390935, 334185, 107922, 304319, 300159, 322125, 292009, 444831, 268838, 30409, 64797, 15438, 302051, 451391, 318469, 328812, 53774, 191173, 391259, 164006, 204324, 474049, 227069, 80462, 528788, 261272, 512258, 202888, 82935, 116717, 74059, 356767, 102532, 269827, 560272, 521867, 355161, 25239, 225463, 481665, 72096, 460732, 501967, 561128, 60177, 187791, 484304, 150421, 254491, 25722, 480056, 354608, 503021, 495641, 576902, 378458, 353489, 144429, 32523, 289782, 391978, 199575, 270025, 314649, 467112, 378312, 60596, 446827, 124262, 443492, 560349, 316194, 182406, 461561, 242784, 206770, 521048, 21292, 324409, 295362, 14180, 67191, 79453, 531171, 209185, 432363, 50099, 482225, 530059, 317019, 146801, 404349, 202389, 575274, 577744, 175469, 291501, 51663, 539317, 549759, 558864, 257498, 138553, 151808, 202131, 20421, 559975, 477911, 572884, 501212, 491942, 272253, 261315, 506955, 122480, 484551, 68121, 502089, 237277, 38083, 123907, 336477, 376531, 338121, 542573, 426664, 374111, 378393, 340539, 227117, 553935, 343954, 68237, 215353, 221038, 79380, 507520, 162892, 459721, 416281, 308758, 362140, 49123, 35760, 519993, 524044, 501700, 462105, 514934, 370689, 293242, 406646, 158841, 346741, 563592, 3935, 505033, 459502, 199992, 65860, 131579, 481773, 23219, 491981, 195367, 4229, 190805, 479553, 305762, 435076, 466211, 531266, 385464, 556248, 535308, 219752, 89044, 521921, 109708, 218397, 45830, 152255, 138022, 564646, 215564, 474543, 389682, 120783, 260370, 523097, 520860, 267343, 492444, 322008, 274376, 110348, 260634, 51046, 251084, 86008, 237077, 80742, 174313, 225480, 172010, 577526, 550722, 522137, 376397, 549462, 303597, 193074, 42429, 221985, 35768, 158233, 485350, 343860, 65630, 472598, 129067, 70985, 503557, 339674, 338230, 560978, 376972, 132781, 535318, 574992, 342277, 377394, 34343, 203882, 539198, 428986, 24020, 495993, 209178, 197237, 461408, 243627, 531541, 557965, 352143, 8333, 15930, 172377, 356107, 448492, 309406, 262221, 393867, 338104, 465476, 299123, 130181, 46660, 195265, 193373, 555120, 315037, 419271, 533276, 92768, 166794, 322707, 221829, 121884, 357470, 87761, 470472, 532779, 560513, 511406, 284338, 52691, 111032, 496578, 510246, 492545, 403520, 304361, 163589, 336469, 103970, 394139, 425881, 47008, 545675, 219817, 162557, 550844, 273641, 22113, 515054, 497246, 46508, 16344, 86864, 437609, 74268, 557708, 250043, 15600, 440779, 558633, 271780, 23311, 12509, 557804, 197254, 473673, 3008, 362711, 567488, 325727, 517253, 969, 37582, 128320, 551518, 385181, 148965, 9679, 316826, 254183, 491921, 491611, 356765, 295556, 211850, 145549, 461521, 173474, 310958, 386752, 300137, 178192, 522194, 143665, 27844, 157822, 503148, 271917, 201887, 221487, 566088, 274815, 552929, 297182, 89223, 100483, 345787, 106517, 190495, 21644, 51642, 16669, 461464, 183756, 499366, 39335, 535259, 123694, 497807, 444869, 148437, 27065, 1108, 519168, 451392, 320796, 118739, 183394, 326359, 250588, 83587, 47954, 56313, 274184, 314876, 124230, 522947, 143125, 39659, 517619, 123013, 304217, 8583, 245425, 55984, 354322, 54540, 554661, 323479, 458323, 367146, 481462, 100647, 93279, 127603, 561635, 569795, 220819, 381377, 451744, 558350, 537772, 236866, 449387, 350518, 205902, 540783, 220041, 90592, 394113, 486123, 4551, 68442, 93444, 282515, 293899, 79893, 334362, 12551, 347453, 520752, 135576, 412691, 121056, 541456, 300399, 581422, 347139, 84667, 556613, 41404, 314837, 528067, 474461, 87234, 63121, 55580, 268940, 14966, 346954, 99112, 385042, 12620, 400919, 82990, 538190, 330419, 75591, 146979, 194262, 371814, 491505, 167337, 125168, 200181, 434587, 9843, 78060, 103581, 307190, 1098, 287828, 537669, 457086, 81620, 178505, 283977, 364796, 312682, 327130, 559062, 95770, 528931, 482667, 29802, 166977, 337575, 475438, 541374, 572399, 55166, 309940, 386429, 535569, 37102, 427041, 226677, 103890, 141017, 25412, 289425, 275242, 542682, 335503, 390895, 29466, 213863, 227689, 266058, 27104, 129379, 442587, 89225, 162237, 422522, 386693, 83873, 126906, 339100, 142804, 337547, 183642, 272889, 162164, 537291, 70236, 534456, 466095, 144320, 90067, 446726, 212877, 125314, 354527, 69283, 519351, 241945, 138124, 204021, 38026, 435811, 240731, 502959, 171805, 283589, 18792, 26943, 314741, 141955, 296901, 510899, 184578, 4704, 412749, 324971, 57340, 300323, 78361, 489520, 314953, 27428, 414744, 146849, 222917, 514939, 150704, 524877, 226983, 235597, 332836, 418440, 191304, 419384, 315654, 359029, 5113, 539563, 154193, 279940, 304483, 366714, 387833, 496313, 344702, 425359, 537939, 214641, 6964, 148568, 424289, 222751, 152733, 30983, 572826, 558132, 315200, 302511, 112590, 216827, 100958, 114686, 508207, 369285, 66987, 288975, 250924, 321964, 73413, 308034, 214204, 360610, 104625, 468789, 480022, 531201, 512275, 117988, 259146, 404257, 519376, 536175, 220214, 374977, 371603, 147736, 11244, 299148, 547227, 380834, 137094, 243336, 202805, 134752, 4283, 532773, 124367, 560948, 288519, 288770, 316155, 364659, 398616, 291677, 41616, 337042, 200717, 429980, 47151, 356237, 388298, 282009, 219443, 377226, 37771, 330208, 104564, 323379, 76492, 205700, 424408, 463620, 110811, 280084, 154928, 175734, 256190, 391810, 554098, 300578, 430652, 55668, 57945, 529139, 470467, 530600, 373375, 298160, 131918, 429305, 427842, 508477, 561042, 69955, 146432, 259349, 463114, 236963, 444769, 118581, 250556, 23400, 93732, 246014, 414045, 67004, 245965, 425576, 331832, 88425, 515809, 213393, 151396, 286760, 217197, 274599, 382411, 36500, 334700, 338319, 470907, 328381, 261587, 250941, 501013, 141038, 442223, 220106, 235536, 310042, 130271, 416386, 5785, 61137, 156955, 98773, 263274, 200287, 338488, 530278, 428304, 505645, 39514, 477643, 212122, 404504, 468129, 239930, 521351, 29045, 225330, 71224, 491728, 264032, 577464, 404621, 422918, 162252, 483080, 261431, 374734, 240434, 120044, 287304, 34437, 115159, 134078, 153671, 425341, 407963, 352508, 179699, 435328, 124347, 491128, 122147, 263231, 221880, 54277, 292597, 472925, 233430, 5198, 310021, 469603, 96193, 174239, 202321, 513115, 353942, 56359, 13146, 460143, 109281, 480391, 560495, 223832, 496980, 167854, 551657, 115033, 37325, 71409, 84403, 385538, 399837, 344885, 528544, 76654, 12966, 531431, 466402, 236674, 9919, 534735, 223004, 263759, 485139, 107862, 241851, 554892, 153570, 250440, 143528, 358211, 180495, 495872, 530726, 431426, 219025, 551609, 462289, 428991, 576286, 41945, 417384, 353595, 79808, 559902, 91784, 541832, 577967, 36278, 228226, 445211, 558808, 203119, 409468, 363812, 157577, 44065, 364720, 14562, 538281, 301908, 98137, 455165, 51795, 336492, 384930, 397388, 235799, 376608, 169395, 515043, 226928, 446436, 108894, 362114, 268363, 111556, 478174, 455369, 72118, 304065, 220666, 328462, 184806, 352011, 467500, 515354, 230213, 448359, 43921, 127619, 170788, 115599, 71855, 416918, 401783, 75760, 42079, 116048, 255135, 166463, 357241, 371245, 299869, 28288, 308878, 327761, 564, 387105, 471405, 574052, 494706, 533125, 203756, 147926, 41288, 174457, 227663, 482157, 259242, 360157, 145290, 130745, 466926, 125503, 570107, 332851, 104365, 294689, 58926, 178840, 578209, 221172, 485185, 10701, 424489, 207215, 251508, 219313, 505627, 40446, 477867, 394691, 53542, 47614, 453711, 61, 50912, 98639, 272095, 144723, 581038, 449839, 311206, 559720, 575096, 70365, 499837, 571563, 90646, 415539, 189388, 545390, 526033, 221874, 474974, 410272, 131315, 226708, 33555, 170107, 536321, 134858, 330098, 141207, 433938, 338600, 449976, 239461, 395592, 358239, 192496, 61315, 437302, 338802, 491215, 580248, 477468, 524661, 12138, 201928, 112860, 216096, 474155, 70310, 79106, 52523, 70735, 323760, 332853, 4093, 249621, 180970, 443296, 283921, 357852, 384146, 207797, 397903, 219174, 340535, 388398, 520767, 20111, 234577, 47221, 438943, 371794, 432330, 197880, 409316, 180935, 29577, 274931, 533684, 46898, 77681, 288039, 133731, 430872, 307242, 125193, 173464, 325145, 398076, 187349, 380821, 573973, 25525, 374049, 566427, 197609, 364016, 259715, 438095, 503099, 146398, 106154, 274304, 438253, 372913, 327919, 262777, 454203, 453697, 357750, 213039, 5757, 37015, 579325, 519483, 523272, 149284, 468993, 406895, 141121, 386850, 505479, 449870, 508354, 361672, 18743, 268781, 499249, 334631, 141566, 544728, 480961, 317254, 329034, 98872, 141053, 239001, 392177, 9288, 85019, 10123, 348098, 188053, 139953, 379966, 578221, 100611, 67748, 331824, 188511, 63525, 166124, 226211, 211243, 242940, 446045, 381546, 49175, 179578, 434981, 380088, 39022, 353358, 80043, 185681, 119868, 479467, 362192, 581415, 486972, 317622, 171936, 478055, 213455, 157700, 337666, 422729, 127942, 386035, 119561, 347483, 229552, 380300, 284901, 137118, 176478, 41146, 386694, 266151, 506256, 390482, 370170, 521669, 437391, 232769, 565641, 148196, 263957, 325294, 494622, 573568, 489624, 265454, 483587, 434016, 28826, 533250, 375324, 495687, 481099, 401123, 59036, 505750, 203349, 399097, 369208, 367953, 104737, 175427, 413126, 475978, 16297, 431847, 416612, 220736, 202865, 282719, 57940, 77667, 221893, 97744, 380097, 196776, 308606, 214280, 414410, 444703, 474230, 520866, 554587, 420593, 207003, 346109, 237333, 255239, 443259, 221199, 97895, 182349, 241554, 185993, 54007, 330478, 155461, 262718, 517313, 381576, 198112, 300753, 448696, 176985, 79080, 30065, 48867, 70441, 105482, 124202, 551063, 139858, 475667, 486845, 440546, 458649, 376114, 568076, 390213, 574277, 523252, 480599, 110142, 286981, 441539, 196885, 432924, 75213, 168122, 533688, 513600, 270351, 62477, 555800, 524366, 369712, 310618, 87282, 444983, 278219, 550845, 310867, 250720, 228029, 130394, 280266, 191203, 446202, 194247, 392174, 335323, 565805, 325873, 347688, 407221, 351034, 762, 342010, 391474, 570409, 519578, 470882, 529477, 16408, 566746], "split_sha256": "a1a3ee0a0e39582dfc13a403faa91ddd63ed5ca5ab16017a0ff84af028765291", "direct_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_direct_seed0_retry5/final.pt", "direct_checkpoint_sha256": "25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8", "quality_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_quality_seed0/final.pt", "quality_checkpoint_sha256": "07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065", "epoch": 3, "rho": 12.0853271484375, "steps": 2, "eta": [6.04266357421875, 3.021331787109375], "lambda": 0.003, "scientific_drift": ["Actual training seed 20261005 differs from recorded seed 0.", "Quality ranking uses soft IoU rather than normal binary original-mask IoU.", "Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.", "Failure random-control radius is global rho, not each oracle displacement norm.", "Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.", "Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.", "DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.", "Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test."], "parameter_updates": false, "oracle_used": false, "streaming": false, "dtype": "float32", "tf32": false, "heldout_states_definition": "Fixed success-state probes: c0 and ±rho/8*u1, ±rho/4*u1, ±rho/4*u2, +rho/8*u2; GT-free and identity-seeded. Added diagnostic, not a preregistered oracle-direction test.", "decoder_definition": "Official process_mask full P@c -> bilinear640 -> predicted box crop -> >0; inverse letterbox binary mask -> >.5. Original COCO GT.", "pixel_metric_version": "qcr-original-pixels-v2", "pixel_metric_definition": "Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background.", "strict_maskfail_definition": "Entire argmax-correct R_arg has a Box75 witness and no Mask75; measured here on its intersection with fixed TAL class-correct Box75 baseline-fail candidates. Unknown on historical DEV cache.", "elapsed_s": 562.4004586832598, "prototype_storage_images": {"torch.float32": 2000}, "scope": "Official one2one TAL matched candidates; no AP/postprocessing quality reranking", "integrity_verified": true, "image_membership_verified": true, "native_decode_verified": true, "partial": false, "no_positive_images": 0}`.
