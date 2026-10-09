# QCR FINAL fixed-candidate metrics

Protocol status: **incomplete**. Stage II permitted: **False**.

36213 candidate rows; 4952 / 5000 planned images contain rows. 48 planned images contain no rows. Those images receive no artificial zero IoU.

All candidate means weight candidates equally. Image macro first averages eligible candidates within each image, then weights eligible images equally. Subgroup-empty images and images without a defined pixel metric are excluded from that estimand. Confidence intervals resample entire eligible images with replacement, preserving the pairing between arms. Deltas and CI in tables are percentage points (pp).

TAL MaskFail = class_correct and BoxIoU ≥ .75 and original MaskIoU < .75. Strict MaskFail additionally requires an audited raw_arg_maskfail=true: the full class-argmax raw set has a Box75 candidate and no Mask75 candidate for the target GT. Unknown raw-arg labels are excluded from strict metrics and block release; TAL failure never substitutes for strict failure. Baseline success uses the same class/box criteria and original MaskIoU ≥ .75. Size uses original COCO annotation area: small <32², medium [32²,96²), large ≥96².

Coverage uses original-resolution binary-mask intersection / original COCO annToMask area. Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background. All pixels outside the target GT (including other instances) are negative. AUC is exact Mann–Whitney with averaged ties. Undefined metrics remain NA. This is a fixed official one-to-one TAL matched-candidate evaluation and makes no COCO AP claim.

## Protocol criteria

| Criterion | Estimate | Required | Pass |
|---|---:|---:|---|
| heldout_pairwise_accuracy | 53.0261 % | >= 65.0000 % | False |
| step1_true_iou_improvement_fraction | 17.7367 % | > 55.0000 % | False |
| strict_maskfail_D_minus_B | -1.5904 pp | >= 0.2000 pp | False |
| strict_maskfail_D_minus_B_CI_low | -1.9249 pp | > 0.0000 pp | False |
| strict_maskfail_D_minus_A | -2.2400 pp | >= 0.5000 pp | False |
| all_D_minus_A_CI_low | -3.0461 pp | >= -0.1000 pp | False |

Ineligibility/incompleteness: scientific_scope_drift.

Scientific drift: Actual training seed 20261005 differs from recorded seed 0.; Quality ranking uses soft IoU rather than normal binary original-mask IoU.; Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.; Failure random-control radius is global rho, not each oracle displacement norm.; Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.; Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.; DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.; Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test..

## Quality diagnostics

Held-out pairwise accuracy: instance equal 53.0261%; pairs weighted 53.1630%; image macro 53.2171%; 985135 non-tie hard-IoU pairs, 35920 eligible instances. True hard-IoU ties are excluded; predicted Q ties get 0.5 credit. Spearman uses averaged tie ranks; constant-state instances are undefined.

Spearman: instance equal 0.0791; image macro 0.0823; 35920 defined instances.

Step-one true IoU increase: 17.7367% over 36213 candidates (6423 increase, 28947 decrease, 843 ties; ties stay in the denominator).

Nonzero Q/IoU delta sign consistency: 64.2211% over 35370 instances. Positive Q but decreased true IoU: 10093 / 14727 positive-Q instances (68.5340%).

Sign table (Q(c1)−Q(c0) rows / true IoU(D1)−IoU(A) columns; exact zeros kept separately):

| Q delta | IoU decrease | IoU tie | IoU increase |
|---|---:|---:|---:|
| negative | 18854 | 70 | 2562 |
| zero | 0 | 0 | 0 |
| positive | 10093 | 773 | 3861 |

Exclusions from nonzero consistency: {"missing_q_or_iou": 0, "both_zero": 0, "q_zero_iou_nonzero": 0, "q_nonzero_iou_zero": 843}.

Q-Rank at these unchanged fixed candidates keeps c0 and therefore the same MaskIoU as A. No ranking/AP effect is estimated here.

## all

36213 candidates; 4952 eligible images; 48 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 74.6649 / 79.5417 | 60.9836 / 71.9923 | 90.1461 / 91.6882 | 93.1578 / 95.0747 | 28.6483 / 22.0712 |
| B | 73.7044 / 78.9946 | 58.4293 / 70.4853 | 92.5439 / 93.3069 | 93.1600 / 95.0997 | 35.3836 / 26.3045 |
| D1 | 66.0322 / 71.4846 | 38.6629 / 52.4099 | 91.7940 / 92.8336 | 90.9716 / 93.4233 | 54.5054 / 45.0962 |
| D | 70.9784 / 76.6308 | 52.2713 / 65.8009 | 90.0640 / 91.7043 | 92.3233 / 94.4910 | 37.5662 / 29.4515 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.7260 | -2.3638 | [-2.4892, -2.2401] | 36213 / 4952 |
| D_minus_B | mask75 | -6.1580 | -4.6844 | [-5.0874, -4.2904] | 36213 / 4952 |
| D_minus_B | coverage | -2.4799 | -1.6026 | [-1.7449, -1.4696] | 36213 / 4952 |
| D_minus_B | auc | -0.8368 | -0.6088 | [-0.6597, -0.5590] | 36173 / 4952 |
| D_minus_B | fpr | 2.1826 | 3.1470 | [2.8684, 3.4231] | 36179 / 4952 |
| D_minus_A | iou | -3.6865 | -2.9109 | [-3.0461, -2.7771] | 36213 / 4952 |
| D_minus_A | mask75 | -8.7123 | -6.1914 | [-6.6354, -5.7463] | 36213 / 4952 |
| D_minus_A | coverage | -0.0821 | 0.0161 | [-0.1222, 0.1474] | 36213 / 4952 |
| D_minus_A | auc | -0.8345 | -0.5838 | [-0.6394, -0.5304] | 36173 / 4952 |
| D_minus_A | fpr | 8.9180 | 7.3803 | [7.0917, 7.6844] | 36179 / 4952 |
| B_minus_A | iou | -0.9605 | -0.5471 | [-0.6032, -0.4930] | 36213 / 4952 |
| B_minus_A | mask75 | -2.5543 | -1.5071 | [-1.7576, -1.2559] | 36213 / 4952 |
| B_minus_A | coverage | 2.3978 | 1.6187 | [1.5465, 1.6956] | 36213 / 4952 |
| B_minus_A | auc | 0.0023 | 0.0250 | [0.0036, 0.0464] | 36173 / 4952 |
| B_minus_A | fpr | 6.7353 | 4.2333 | [4.0963, 4.3707] | 36179 / 4952 |
| D1_minus_A | iou | -8.6327 | -8.0572 | [-8.2980, -7.8269] | 36213 / 4952 |
| D1_minus_A | mask75 | -22.3207 | -19.5825 | [-20.2980, -18.8618] | 36213 / 4952 |
| D1_minus_A | coverage | 1.6479 | 1.1455 | [0.9325, 1.3447] | 36213 / 4952 |
| D1_minus_A | auc | -2.1862 | -1.6514 | [-1.7449, -1.5651] | 36173 / 4952 |
| D1_minus_A | fpr | 25.8571 | 23.0249 | [22.5274, 23.5302] | 36179 / 4952 |
| D_minus_D1 | iou | 4.9462 | 5.1462 | [4.9570, 5.3405] | 36213 / 4952 |
| D_minus_D1 | mask75 | 13.6084 | 13.3910 | [12.7770, 14.0041] | 36213 / 4952 |
| D_minus_D1 | coverage | -1.7300 | -1.1293 | [-1.3026, -0.9461] | 36213 / 4952 |
| D_minus_D1 | auc | 1.3517 | 1.0677 | [1.0115, 1.1285] | 36173 / 4952 |
| D_minus_D1 | fpr | -16.9392 | -15.6447 | [-15.9976, -15.2873] | 36179 / 4952 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 510 / 14129 | 1435 / 22084 | 3.6096 / 4.3550 | 6.4979 / 4.5342 | -925 | -1.5071 [-1.7576, -1.2559] |
| D1 | 628 / 14129 | 8711 / 22084 | 4.4448 / 6.6865 | 39.4448 / 31.1004 | -8083 | -19.5825 [-20.2980, -18.8618] |
| D | 740 / 14129 | 3895 / 22084 | 5.2375 / 6.5689 | 17.6372 / 12.2160 | -3155 | -6.1914 [-6.6354, -5.7463] |

## tal_maskfail

6783 candidates; 2577 eligible images; 2423 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 62.8919 / 62.8232 | 0.0000 / 0.0000 | 85.6175 / 84.6335 | 90.2434 / 90.1283 | 35.1392 / 33.0887 |
| B | 61.9769 / 62.2360 | 5.7939 / 6.3087 | 90.3988 / 89.2068 | 90.3114 / 90.2050 | 44.7886 / 41.8090 |
| D1 | 57.0622 / 57.9718 | 7.2534 / 9.2794 | 90.0684 / 89.1151 | 87.8808 / 87.9884 | 58.1645 / 54.6441 |
| D | 60.1806 / 60.5643 | 8.6835 / 9.5092 | 86.8079 / 85.8147 | 89.3267 / 89.2554 | 43.6783 / 41.0721 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.7963 | -1.6718 | [-1.9832, -1.3687] | 6783 / 2577 |
| D_minus_B | mask75 | 2.8896 | 3.2005 | [2.3329, 4.0800] | 6783 / 2577 |
| D_minus_B | coverage | -3.5909 | -3.3921 | [-3.7985, -2.9938] | 6783 / 2577 |
| D_minus_B | auc | -0.9847 | -0.9497 | [-1.1011, -0.8004] | 6780 / 2574 |
| D_minus_B | fpr | -1.1102 | -0.7368 | [-1.2899, -0.1845] | 6780 / 2574 |
| D_minus_A | iou | -2.7113 | -2.2589 | [-2.5979, -1.9237] | 6783 / 2577 |
| D_minus_A | mask75 | 8.6835 | 9.5092 | [8.6122, 10.4101] | 6783 / 2577 |
| D_minus_A | coverage | 1.1904 | 1.1812 | [0.7776, 1.5922] | 6783 / 2577 |
| D_minus_A | auc | -0.9167 | -0.8730 | [-1.0407, -0.7079] | 6780 / 2574 |
| D_minus_A | fpr | 8.5391 | 7.9835 | [7.4026, 8.5715] | 6780 / 2574 |
| B_minus_A | iou | -0.9150 | -0.5872 | [-0.8240, -0.3532] | 6783 / 2577 |
| B_minus_A | mask75 | 5.7939 | 6.3087 | [5.5422, 7.0836] | 6783 / 2577 |
| B_minus_A | coverage | 4.7813 | 4.5733 | [4.2925, 4.8566] | 6783 / 2577 |
| B_minus_A | auc | 0.0680 | 0.0767 | [-0.0009, 0.1532] | 6780 / 2574 |
| B_minus_A | fpr | 9.6494 | 8.7203 | [8.3861, 9.0670] | 6780 / 2574 |
| D1_minus_A | iou | -5.8297 | -4.8514 | [-5.2824, -4.4227] | 6783 / 2577 |
| D1_minus_A | mask75 | 7.2534 | 9.2794 | [8.3535, 10.2157] | 6783 / 2577 |
| D1_minus_A | coverage | 4.4509 | 4.4815 | [3.9762, 4.9917] | 6783 / 2577 |
| D1_minus_A | auc | -2.3626 | -2.1399 | [-2.3554, -1.9316] | 6780 / 2574 |
| D1_minus_A | fpr | 23.0253 | 21.5554 | [20.7385, 22.3896] | 6780 / 2574 |
| D_minus_D1 | iou | 3.1184 | 2.5925 | [2.2308, 2.9710] | 6783 / 2577 |
| D_minus_D1 | mask75 | 1.4300 | 0.2298 | [-0.7129, 1.1836] | 6783 / 2577 |
| D_minus_D1 | coverage | -3.2605 | -3.3003 | [-3.7712, -2.8214] | 6783 / 2577 |
| D_minus_D1 | auc | 1.4459 | 1.2669 | [1.1401, 1.3961] | 6780 / 2574 |
| D_minus_D1 | fpr | -14.4862 | -13.5720 | [-14.1719, -12.9516] | 6780 / 2574 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 393 / 6783 | 0 / 0 | 5.7939 / 6.3087 | NA / NA | 393 | 6.3087 [5.5422, 7.0836] |
| D1 | 492 / 6783 | 0 / 0 | 7.2534 / 9.2794 | NA / NA | 492 | 9.2794 [8.3535, 10.2157] |
| D | 589 / 6783 | 0 / 0 | 8.6835 / 9.5092 | NA / NA | 589 | 9.5092 [8.6122, 10.4101] |

## strict_maskfail

5304 candidates; 2228 eligible images; 2772 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 61.1795 / 60.9552 | 0.0000 / 0.0000 | 84.4017 / 83.0055 | 89.2474 / 88.9929 | 35.7373 / 33.4904 |
| B | 60.1276 / 60.3055 | 4.2421 / 4.5267 | 89.3831 / 87.9043 | 89.2636 / 89.0122 | 45.9009 / 42.8883 |
| D1 | 55.5766 / 56.3405 | 5.7881 / 7.0639 | 89.1521 / 88.0921 | 86.7820 / 86.6639 | 58.7682 / 55.6701 |
| D | 58.4564 / 58.7151 | 6.0143 / 6.6706 | 85.5954 / 84.3389 | 88.2550 / 88.0090 | 44.3099 / 41.7638 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.6712 | -1.5904 | [-1.9249, -1.2650] | 5304 / 2228 |
| D_minus_B | mask75 | 1.7722 | 2.1439 | [1.2797, 3.0187] | 5304 / 2228 |
| D_minus_B | coverage | -3.7877 | -3.5654 | [-4.0373, -3.0944] | 5304 / 2228 |
| D_minus_B | auc | -1.0086 | -1.0033 | [-1.1752, -0.8321] | 5302 / 2226 |
| D_minus_B | fpr | -1.5909 | -1.1245 | [-1.7622, -0.4517] | 5302 / 2226 |
| D_minus_A | iou | -2.7231 | -2.2400 | [-2.6159, -1.8712] | 5304 / 2228 |
| D_minus_A | mask75 | 6.0143 | 6.6706 | [5.8328, 7.5466] | 5304 / 2228 |
| D_minus_A | coverage | 1.1936 | 1.3334 | [0.8684, 1.8239] | 5304 / 2228 |
| D_minus_A | auc | -0.9924 | -0.9839 | [-1.1824, -0.7875] | 5302 / 2226 |
| D_minus_A | fpr | 8.5726 | 8.2734 | [7.5965, 8.9894] | 5302 / 2226 |
| B_minus_A | iou | -1.0519 | -0.6496 | [-0.8965, -0.3943] | 5304 / 2228 |
| B_minus_A | mask75 | 4.2421 | 4.5267 | [3.8115, 5.2242] | 5304 / 2228 |
| B_minus_A | coverage | 4.9813 | 4.8988 | [4.5864, 5.2235] | 5304 / 2228 |
| B_minus_A | auc | 0.0162 | 0.0193 | [-0.0671, 0.1064] | 5302 / 2226 |
| B_minus_A | fpr | 10.1636 | 9.3979 | [9.0129, 9.7975] | 5302 / 2226 |
| D1_minus_A | iou | -5.6029 | -4.6147 | [-5.0931, -4.1584] | 5304 / 2228 |
| D1_minus_A | mask75 | 5.7881 | 7.0639 | [6.1934, 7.9812] | 5304 / 2228 |
| D1_minus_A | coverage | 4.7504 | 5.0865 | [4.4841, 5.6808] | 5304 / 2228 |
| D1_minus_A | auc | -2.4654 | -2.3289 | [-2.5840, -2.0836] | 5302 / 2226 |
| D1_minus_A | fpr | 23.0309 | 22.1797 | [21.2522, 23.1263] | 5302 / 2226 |
| D_minus_D1 | iou | 2.8798 | 2.3747 | [1.9583, 2.7812] | 5304 / 2228 |
| D_minus_D1 | mask75 | 0.2262 | -0.3933 | [-1.3150, 0.5425] | 5304 / 2228 |
| D_minus_D1 | coverage | -3.5568 | -3.7531 | [-4.3004, -3.2007] | 5304 / 2228 |
| D_minus_D1 | auc | 1.4730 | 1.3450 | [1.1968, 1.4995] | 5302 / 2226 |
| D_minus_D1 | fpr | -14.4583 | -13.9063 | [-14.6334, -13.1966] | 5302 / 2226 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 225 / 5304 | 0 / 0 | 4.2421 / 4.5267 | NA / NA | 225 | 4.5267 [3.8115, 5.2242] |
| D1 | 307 / 5304 | 0 / 0 | 5.7881 / 7.0639 | NA / NA | 307 | 7.0639 [6.1934, 7.9812] |
| D | 319 / 5304 | 0 / 0 | 6.0143 / 6.6706 | NA / NA | 319 | 6.6706 [5.8328, 7.5466] |

## baseline_success

20207 candidates; 4791 eligible images; 209 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 86.6977 / 87.7515 | 100.0000 / 100.0000 | 94.9199 / 95.2034 | 97.6616 / 97.9989 | 17.9539 / 15.7238 |
| B | 85.9172 / 87.2361 | 94.2050 / 95.9291 | 95.8145 / 95.8075 | 97.7220 / 98.0454 | 21.6111 / 18.1785 |
| D1 | 75.8615 / 78.3225 | 61.3005 / 69.6398 | 95.6744 / 95.7689 | 95.7930 / 96.5649 | 46.9772 / 40.5306 |
| D | 82.5587 / 84.6005 | 83.1494 / 88.4502 | 94.8795 / 95.2112 | 97.1211 / 97.5923 | 27.8856 / 23.4916 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.3586 | -2.6355 | [-2.7674, -2.5069] | 20207 / 4791 |
| D_minus_B | mask75 | -11.0556 | -7.4789 | [-7.9862, -6.9679] | 20207 / 4791 |
| D_minus_B | coverage | -0.9350 | -0.5963 | [-0.7050, -0.4985] | 20207 / 4791 |
| D_minus_B | auc | -0.6008 | -0.4532 | [-0.4971, -0.4132] | 20200 / 4791 |
| D_minus_B | fpr | 6.2745 | 5.3131 | [5.0143, 5.6138] | 20200 / 4791 |
| D_minus_A | iou | -4.1390 | -3.1510 | [-3.2934, -3.0184] | 20207 / 4791 |
| D_minus_A | mask75 | -16.8506 | -11.5498 | [-12.1301, -10.9942] | 20207 / 4791 |
| D_minus_A | coverage | -0.0405 | 0.0078 | [-0.1017, 0.1066] | 20207 / 4791 |
| D_minus_A | auc | -0.5404 | -0.4067 | [-0.4509, -0.3664] | 20200 / 4791 |
| D_minus_A | fpr | 9.9317 | 7.7678 | [7.4444, 8.0859] | 20200 / 4791 |
| B_minus_A | iou | -0.7805 | -0.5154 | [-0.5552, -0.4776] | 20207 / 4791 |
| B_minus_A | mask75 | -5.7950 | -4.0709 | [-4.3963, -3.7621] | 20207 / 4791 |
| B_minus_A | coverage | 0.8945 | 0.6041 | [0.5758, 0.6323] | 20207 / 4791 |
| B_minus_A | auc | 0.0604 | 0.0465 | [0.0307, 0.0586] | 20200 / 4791 |
| B_minus_A | fpr | 3.6572 | 2.4547 | [2.3516, 2.5632] | 20200 / 4791 |
| D1_minus_A | iou | -10.8362 | -9.4289 | [-9.6889, -9.1743] | 20207 / 4791 |
| D1_minus_A | mask75 | -38.6995 | -30.3602 | [-31.2620, -29.4372] | 20207 / 4791 |
| D1_minus_A | coverage | 0.7544 | 0.5655 | [0.3724, 0.7460] | 20207 / 4791 |
| D1_minus_A | auc | -1.8686 | -1.4340 | [-1.5193, -1.3560] | 20200 / 4791 |
| D1_minus_A | fpr | 29.0233 | 24.8068 | [24.2238, 25.3688] | 20200 / 4791 |
| D_minus_D1 | iou | 6.6971 | 6.2780 | [6.0742, 6.4870] | 20207 / 4791 |
| D_minus_D1 | mask75 | 21.8489 | 18.8104 | [18.0224, 19.5870] | 20207 / 4791 |
| D_minus_D1 | coverage | -0.7949 | -0.5577 | [-0.7242, -0.3872] | 20207 / 4791 |
| D_minus_D1 | auc | 1.3282 | 1.0273 | [0.9728, 1.0846] | 20200 / 4791 |
| D_minus_D1 | fpr | -19.0916 | -17.0390 | [-17.4179, -16.6432] | 20200 / 4791 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 1171 / 20207 | NA / NA | 5.7950 / 4.0709 | -1171 | -4.0709 [-4.3963, -3.7621] |
| D1 | 0 / 0 | 7820 / 20207 | NA / NA | 38.6995 / 30.3602 | -7820 | -30.3602 [-31.2620, -29.4372] |
| D | 0 / 0 | 3405 / 20207 | NA / NA | 16.8506 / 11.5498 | -3405 | -11.5498 [-12.1301, -10.9942] |

## all/size/small

15150 candidates; 2614 eligible images; 2386 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 64.8219 / 67.2998 | 36.3762 / 41.6218 | 89.0494 / 89.9276 | 89.1491 / 91.0002 | 46.2931 / 41.0355 |
| B | 62.7864 / 65.4217 | 31.6106 / 37.0809 | 91.7733 / 92.4045 | 89.0535 / 90.9524 | 56.8924 / 50.2176 |
| D1 | 58.7038 / 61.7009 | 22.0792 / 28.5405 | 89.4669 / 89.6634 | 86.9904 / 89.2113 | 65.5774 / 57.3767 |
| D | 61.6032 / 64.4502 | 29.3795 / 35.4798 | 88.0385 / 89.0156 | 88.1117 / 90.1391 | 52.8679 / 46.7369 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.1832 | -0.9716 | [-1.1588, -0.7880] | 15150 / 2614 |
| D_minus_B | mask75 | -2.2310 | -1.6011 | [-2.3148, -0.8846] | 15150 / 2614 |
| D_minus_B | coverage | -3.7349 | -3.3888 | [-3.6687, -3.1169] | 15150 / 2614 |
| D_minus_B | auc | -0.9418 | -0.8134 | [-0.9305, -0.6964] | 15118 / 2614 |
| D_minus_B | fpr | -4.0244 | -3.4806 | [-3.9163, -3.0565] | 15124 / 2614 |
| D_minus_A | iou | -3.2186 | -2.8496 | [-3.0475, -2.6488] | 15150 / 2614 |
| D_minus_A | mask75 | -6.9967 | -6.1420 | [-6.9450, -5.3741] | 15150 / 2614 |
| D_minus_A | coverage | -1.0109 | -0.9120 | [-1.1673, -0.6560] | 15150 / 2614 |
| D_minus_A | auc | -1.0374 | -0.8612 | [-0.9972, -0.7224] | 15118 / 2614 |
| D_minus_A | fpr | 6.5748 | 5.7014 | [5.2297, 6.1831] | 15124 / 2614 |
| B_minus_A | iou | -2.0354 | -1.8780 | [-1.9825, -1.7760] | 15150 / 2614 |
| B_minus_A | mask75 | -4.7657 | -4.5410 | [-5.1549, -3.9510] | 15150 / 2614 |
| B_minus_A | coverage | 2.7239 | 2.4768 | [2.3340, 2.6370] | 15150 / 2614 |
| B_minus_A | auc | -0.0956 | -0.0478 | [-0.1139, 0.0181] | 15118 / 2614 |
| B_minus_A | fpr | 10.5993 | 9.1821 | [8.9120, 9.4648] | 15124 / 2614 |
| D1_minus_A | iou | -6.1181 | -5.5989 | [-5.8553, -5.3440] | 15150 / 2614 |
| D1_minus_A | mask75 | -14.2970 | -13.0814 | [-14.1032, -12.0559] | 15150 / 2614 |
| D1_minus_A | coverage | 0.4175 | -0.2642 | [-0.5842, 0.0479] | 15150 / 2614 |
| D1_minus_A | auc | -2.1588 | -1.7890 | [-1.9721, -1.6110] | 15118 / 2614 |
| D1_minus_A | fpr | 19.2843 | 16.3412 | [15.6911, 17.0287] | 15124 / 2614 |
| D_minus_D1 | iou | 2.8995 | 2.7493 | [2.5147, 2.9798] | 15150 / 2614 |
| D_minus_D1 | mask75 | 7.3003 | 6.9393 | [6.0161, 7.8541] | 15150 / 2614 |
| D_minus_D1 | coverage | -1.4285 | -0.6478 | [-0.9953, -0.3018] | 15150 / 2614 |
| D_minus_D1 | auc | 1.1213 | 0.9278 | [0.8197, 1.0443] | 15118 / 2614 |
| D_minus_D1 | fpr | -12.7094 | -10.6398 | [-11.1842, -10.0908] | 15124 / 2614 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 143 / 9639 | 865 / 5511 | 1.4836 / 1.3993 | 15.6959 / 14.4105 | -722 | -4.5410 [-5.1549, -3.9510] |
| D1 | 207 / 9639 | 2373 / 5511 | 2.1475 / 2.6688 | 43.0593 / 38.2995 | -2166 | -13.0814 [-14.1032, -12.0559] |
| D | 297 / 9639 | 1357 / 5511 | 3.0812 / 3.4816 | 24.6235 / 21.9878 | -1060 | -6.1420 [-6.9450, -5.3741] |

## all/size/medium

12388 candidates; 3470 eligible images; 1530 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 79.4793 / 80.3432 | 74.3300 / 76.0092 | 90.5638 / 91.5111 | 95.7686 / 96.2498 | 18.6425 / 18.3380 |
| B | 78.9656 / 79.8583 | 72.4411 / 74.5377 | 93.1450 / 93.4546 | 95.8488 / 96.3257 | 23.7339 / 22.3254 |
| D1 | 69.2900 / 72.5754 | 45.4876 / 54.5887 | 92.7949 / 92.4331 | 93.3496 / 94.5284 | 47.4395 / 39.0396 |
| D | 75.0815 / 77.2803 | 62.3830 / 68.5934 | 90.7529 / 91.2140 | 95.0221 / 95.7327 | 28.8450 / 24.8388 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.8841 | -2.5780 | [-2.7652, -2.3983] | 12388 / 3470 |
| D_minus_B | mask75 | -10.0581 | -5.9443 | [-6.6382, -5.2697] | 12388 / 3470 |
| D_minus_B | coverage | -2.3921 | -2.2406 | [-2.4391, -2.0378] | 12388 / 3470 |
| D_minus_B | auc | -0.8266 | -0.5930 | [-0.6637, -0.5269] | 12384 / 3470 |
| D_minus_B | fpr | 5.1111 | 2.5133 | [2.1443, 2.8914] | 12384 / 3470 |
| D_minus_A | iou | -4.3978 | -3.0630 | [-3.2611, -2.8702] | 12388 / 3470 |
| D_minus_A | mask75 | -11.9470 | -7.4158 | [-8.1518, -6.6766] | 12388 / 3470 |
| D_minus_A | coverage | 0.1892 | -0.2970 | [-0.5031, -0.0957] | 12388 / 3470 |
| D_minus_A | auc | -0.7464 | -0.5171 | [-0.5978, -0.4373] | 12384 / 3470 |
| D_minus_A | fpr | 10.2026 | 6.5008 | [6.0952, 6.9134] | 12384 / 3470 |
| B_minus_A | iou | -0.5137 | -0.4850 | [-0.5704, -0.3999] | 12388 / 3470 |
| B_minus_A | mask75 | -1.8889 | -1.4715 | [-1.9175, -1.0215] | 12388 / 3470 |
| B_minus_A | coverage | 2.5813 | 1.9436 | [1.8312, 2.0585] | 12388 / 3470 |
| B_minus_A | auc | 0.0802 | 0.0759 | [0.0402, 0.1165] | 12384 / 3470 |
| B_minus_A | fpr | 5.0915 | 3.9874 | [3.8112, 4.1633] | 12384 / 3470 |
| D1_minus_A | iou | -10.1893 | -7.7679 | [-8.0835, -7.4623] | 12388 / 3470 |
| D1_minus_A | mask75 | -28.8424 | -21.4205 | [-22.4938, -20.3352] | 12388 / 3470 |
| D1_minus_A | coverage | 2.2311 | 0.9220 | [0.6409, 1.1969] | 12388 / 3470 |
| D1_minus_A | auc | -2.4190 | -1.7214 | [-1.8500, -1.5974] | 12384 / 3470 |
| D1_minus_A | fpr | 28.7970 | 20.7016 | [20.0159, 21.3994] | 12384 / 3470 |
| D_minus_D1 | iou | 5.7914 | 4.7049 | [4.4495, 4.9560] | 12388 / 3470 |
| D_minus_D1 | mask75 | 16.8954 | 14.0047 | [13.0251, 14.9550] | 12388 / 3470 |
| D_minus_D1 | coverage | -2.0420 | -1.2190 | [-1.4805, -0.9544] | 12388 / 3470 |
| D_minus_D1 | auc | 1.6726 | 1.2043 | [1.1228, 1.2915] | 12384 / 3470 |
| D_minus_D1 | fpr | -18.5945 | -14.2008 | [-14.6770, -13.7373] | 12384 / 3470 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 222 / 3180 | 456 / 9208 | 6.9811 / 7.0356 | 4.9522 / 4.3510 | -234 | -1.4715 [-1.9175, -1.0215] |
| D1 | 199 / 3180 | 3772 / 9208 | 6.2579 / 7.4975 | 40.9644 / 32.1224 | -3573 | -21.4205 [-22.4938, -20.3352] |
| D | 271 / 3180 | 1751 / 9208 | 8.5220 / 9.0821 | 19.0161 / 13.3485 | -1480 | -7.4158 [-8.1518, -6.6766] |

## all/size/large

8675 candidates; 4062 eligible images; 938 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 84.9799 / 86.2069 | 84.8991 / 87.7713 | 91.4648 / 92.3788 | 96.4180 / 96.8536 | 12.1625 / 11.7706 |
| B | 85.2586 / 86.4508 | 85.2565 / 88.0348 | 93.0312 / 93.5021 | 96.4798 / 96.9261 | 14.5062 / 13.4327 |
| D1 | 74.1784 / 76.8971 | 57.8790 / 65.5126 | 94.4288 / 94.9943 | 94.5167 / 95.4579 | 45.2852 / 40.4529 |
| D | 81.4921 / 83.6297 | 77.8098 / 82.7670 | 92.6176 / 93.4343 | 95.8116 / 96.4654 | 23.3327 / 20.6996 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7665 | -2.8210 | [-3.0001, -2.6508] | 8675 / 4062 |
| D_minus_B | mask75 | -7.4467 | -5.2679 | [-5.8630, -4.6660] | 8675 / 4062 |
| D_minus_B | coverage | -0.4136 | -0.0678 | [-0.2546, 0.1028] | 8675 / 4062 |
| D_minus_B | auc | -0.6682 | -0.4607 | [-0.5235, -0.3983] | 8671 / 4062 |
| D_minus_B | fpr | 8.8265 | 7.2668 | [6.9028, 7.6202] | 8671 / 4062 |
| D_minus_A | iou | -3.4877 | -2.5772 | [-2.7684, -2.3937] | 8675 / 4062 |
| D_minus_A | mask75 | -7.0893 | -5.0043 | [-5.6347, -4.3680] | 8675 / 4062 |
| D_minus_A | coverage | 1.1528 | 1.0556 | [0.8755, 1.2333] | 8675 / 4062 |
| D_minus_A | auc | -0.6064 | -0.3882 | [-0.4581, -0.3175] | 8671 / 4062 |
| D_minus_A | fpr | 11.1701 | 8.9289 | [8.5297, 9.3115] | 8671 / 4062 |
| B_minus_A | iou | 0.2787 | 0.2438 | [0.1591, 0.3328] | 8675 / 4062 |
| B_minus_A | mask75 | 0.3573 | 0.2636 | [-0.0842, 0.6142] | 8675 / 4062 |
| B_minus_A | coverage | 1.5664 | 1.1233 | [1.0124, 1.2406] | 8675 / 4062 |
| B_minus_A | auc | 0.0618 | 0.0725 | [0.0437, 0.1018] | 8671 / 4062 |
| B_minus_A | fpr | 2.3437 | 1.6621 | [1.5210, 1.8122] | 8671 / 4062 |
| D1_minus_A | iou | -10.8015 | -9.3099 | [-9.6611, -8.9676] | 8675 / 4062 |
| D1_minus_A | mask75 | -27.0202 | -22.2587 | [-23.3866, -21.1476] | 8675 / 4062 |
| D1_minus_A | coverage | 2.9640 | 2.6155 | [2.3133, 2.9120] | 8675 / 4062 |
| D1_minus_A | auc | -1.9013 | -1.3957 | [-1.5117, -1.2848] | 8671 / 4062 |
| D1_minus_A | fpr | 33.1227 | 28.6823 | [27.9888, 29.3729] | 8671 / 4062 |
| D_minus_D1 | iou | 7.3137 | 6.7327 | [6.4485, 7.0245] | 8675 / 4062 |
| D_minus_D1 | mask75 | 19.9308 | 17.2544 | [16.2529, 18.2832] | 8675 / 4062 |
| D_minus_D1 | coverage | -1.8112 | -1.5600 | [-1.8099, -1.3005] | 8675 / 4062 |
| D_minus_D1 | auc | 1.2949 | 1.0075 | [0.9396, 1.0815] | 8671 / 4062 |
| D_minus_D1 | fpr | -21.9526 | -19.7533 | [-20.2534, -19.2601] | 8671 / 4062 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 145 / 1310 | 114 / 7365 | 11.0687 / 10.9207 | 1.5479 / 1.2766 | 31 | 0.2636 [-0.0842, 0.6142] |
| D1 | 222 / 1310 | 2566 / 7365 | 16.9466 / 18.3774 | 34.8405 / 29.2315 | -2344 | -22.2587 [-23.3866, -21.1476] |
| D | 172 / 1310 | 787 / 7365 | 13.1298 / 13.6216 | 10.6857 / 8.2308 | -615 | -5.0043 [-5.6347, -4.3680] |

## all/pyramid/P3

11330 candidates; 2245 eligible images; 2755 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.5763 / 66.1370 | 33.2480 / 39.3209 | 88.7704 / 89.3993 | 87.8920 / 89.5254 | 52.7351 / 48.1376 |
| B | 61.4679 / 64.0940 | 28.5702 / 34.3462 | 91.3833 / 91.9176 | 87.7703 / 89.4484 | 64.6894 / 59.5048 |
| D1 | 57.8940 / 60.3919 | 19.9294 / 25.2354 | 89.3648 / 89.5582 | 85.6083 / 87.4388 | 73.0323 / 67.6540 |
| D | 60.3510 / 62.9554 | 26.5137 / 32.3890 | 87.7193 / 88.5085 | 86.7410 / 88.4723 | 60.0818 / 55.6559 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.1169 | -1.1386 | [-1.3453, -0.9341] | 11330 / 2245 |
| D_minus_B | mask75 | -2.0565 | -1.9572 | [-2.7398, -1.1762] | 11330 / 2245 |
| D_minus_B | coverage | -3.6640 | -3.4091 | [-3.7563, -3.0781] | 11330 / 2245 |
| D_minus_B | auc | -1.0293 | -0.9762 | [-1.1314, -0.8268] | 11302 / 2245 |
| D_minus_B | fpr | -4.6076 | -3.8489 | [-4.4176, -3.2887] | 11305 / 2245 |
| D_minus_A | iou | -3.2253 | -3.1816 | [-3.4112, -2.9589] | 11330 / 2245 |
| D_minus_A | mask75 | -6.7343 | -6.9319 | [-7.8170, -6.0582] | 11330 / 2245 |
| D_minus_A | coverage | -1.0511 | -0.8908 | [-1.2158, -0.5687] | 11330 / 2245 |
| D_minus_A | auc | -1.1509 | -1.0532 | [-1.2218, -0.8881] | 11302 / 2245 |
| D_minus_A | fpr | 7.3467 | 7.5182 | [6.9198, 8.1584] | 11305 / 2245 |
| B_minus_A | iou | -2.1084 | -2.0430 | [-2.1580, -1.9294] | 11330 / 2245 |
| B_minus_A | mask75 | -4.6778 | -4.9747 | [-5.6946, -4.3142] | 11330 / 2245 |
| B_minus_A | coverage | 2.6129 | 2.5183 | [2.3517, 2.6840] | 11330 / 2245 |
| B_minus_A | auc | -0.1217 | -0.0770 | [-0.1571, 0.0052] | 11302 / 2245 |
| B_minus_A | fpr | 11.9544 | 11.3671 | [11.0401, 11.7126] | 11305 / 2245 |
| D1_minus_A | iou | -5.6823 | -5.7451 | [-6.0294, -5.4800] | 11330 / 2245 |
| D1_minus_A | mask75 | -13.3186 | -14.0855 | [-15.2059, -13.0310] | 11330 / 2245 |
| D1_minus_A | coverage | 0.5944 | 0.1589 | [-0.1974, 0.5005] | 11330 / 2245 |
| D1_minus_A | auc | -2.2836 | -2.0866 | [-2.3083, -1.8742] | 11302 / 2245 |
| D1_minus_A | fpr | 20.2972 | 19.5164 | [18.7456, 20.2622] | 11305 / 2245 |
| D_minus_D1 | iou | 2.4570 | 2.5634 | [2.3032, 2.8224] | 11330 / 2245 |
| D_minus_D1 | mask75 | 6.5843 | 7.1536 | [6.1952, 8.0896] | 11330 / 2245 |
| D_minus_D1 | coverage | -1.6455 | -1.0497 | [-1.4360, -0.6549] | 11330 / 2245 |
| D_minus_D1 | auc | 1.1327 | 1.0334 | [0.9036, 1.1668] | 11302 / 2245 |
| D_minus_D1 | fpr | -12.9505 | -11.9981 | [-12.6424, -11.3653] | 11305 / 2245 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 101 / 7563 | 631 / 3767 | 1.3354 / 1.5507 | 16.7507 / 16.4831 | -530 | -4.9747 [-5.6946, -4.3142] |
| D1 | 123 / 7563 | 1632 / 3767 | 1.6263 / 1.9899 | 43.3236 / 41.0710 | -1509 | -14.0855 [-15.2059, -13.0310] |
| D | 196 / 7563 | 959 / 3767 | 2.5916 / 3.0758 | 25.4579 / 24.5221 | -763 | -6.9319 [-7.8170, -6.0582] |

## all/pyramid/P4

14099 candidates; 3432 eligible images; 1568 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 77.2689 / 77.9699 | 68.1467 / 69.3786 | 90.7113 / 91.3902 | 94.9960 / 95.2892 | 21.8821 / 22.0714 |
| B | 76.4410 / 77.2786 | 65.5224 / 67.2973 | 93.3211 / 93.5596 | 95.0574 / 95.3578 | 27.5787 / 26.9300 |
| D1 | 68.0471 / 70.9404 | 42.3363 / 50.4704 | 92.3084 / 92.1514 | 92.7064 / 93.5686 | 48.6892 / 42.3704 |
| D | 73.1762 / 74.9350 | 57.1814 / 62.2923 | 90.6244 / 91.0084 | 94.2279 / 94.7105 | 31.2921 / 28.7270 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.2647 | -2.3436 | [-2.5436, -2.1484] | 14099 / 3432 |
| D_minus_B | mask75 | -8.3410 | -5.0050 | [-5.7143, -4.2988] | 14099 / 3432 |
| D_minus_B | coverage | -2.6966 | -2.5512 | [-2.7745, -2.3366] | 14099 / 3432 |
| D_minus_B | auc | -0.8295 | -0.6473 | [-0.7266, -0.5688] | 14091 / 3432 |
| D_minus_B | fpr | 3.7134 | 1.7970 | [1.3965, 2.2017] | 14094 / 3432 |
| D_minus_A | iou | -4.0926 | -3.0349 | [-3.2371, -2.8352] | 14099 / 3432 |
| D_minus_A | mask75 | -10.9653 | -7.0863 | [-7.8061, -6.3780] | 14099 / 3432 |
| D_minus_A | coverage | -0.0869 | -0.3818 | [-0.5975, -0.1703] | 14099 / 3432 |
| D_minus_A | auc | -0.7681 | -0.5786 | [-0.6669, -0.4915] | 14091 / 3432 |
| D_minus_A | fpr | 9.4101 | 6.6556 | [6.2278, 7.0777] | 14094 / 3432 |
| B_minus_A | iou | -0.8279 | -0.6913 | [-0.7862, -0.5955] | 14099 / 3432 |
| B_minus_A | mask75 | -2.6243 | -2.0813 | [-2.5472, -1.6325] | 14099 / 3432 |
| B_minus_A | coverage | 2.6098 | 2.1694 | [2.0462, 2.3005] | 14099 / 3432 |
| B_minus_A | auc | 0.0614 | 0.0686 | [0.0334, 0.1051] | 14091 / 3432 |
| B_minus_A | fpr | 5.6967 | 4.8586 | [4.6716, 5.0605] | 14094 / 3432 |
| D1_minus_A | iou | -9.2218 | -7.0295 | [-7.3284, -6.7416] | 14099 / 3432 |
| D1_minus_A | mask75 | -25.8103 | -18.9082 | [-19.9450, -17.8951] | 14099 / 3432 |
| D1_minus_A | coverage | 1.5971 | 0.7612 | [0.4856, 1.0276] | 14099 / 3432 |
| D1_minus_A | auc | -2.2897 | -1.7206 | [-1.8615, -1.5880] | 14091 / 3432 |
| D1_minus_A | fpr | 26.8071 | 20.2991 | [19.6000, 21.0254] | 14094 / 3432 |
| D_minus_D1 | iou | 5.1292 | 3.9946 | [3.7529, 4.2363] | 14099 / 3432 |
| D_minus_D1 | mask75 | 14.8450 | 11.8219 | [10.9235, 12.7270] | 14099 / 3432 |
| D_minus_D1 | coverage | -1.6839 | -1.1430 | [-1.3931, -0.8865] | 14099 / 3432 |
| D_minus_D1 | auc | 1.5215 | 1.1420 | [1.0493, 1.2416] | 14091 / 3432 |
| D_minus_D1 | fpr | -17.3970 | -13.6435 | [-14.1330, -13.1412] | 14094 / 3432 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 235 / 4491 | 605 / 9608 | 5.2327 / 4.9878 | 6.2968 / 5.6400 | -370 | -2.0813 [-2.5472, -1.6325] |
| D1 | 253 / 4491 | 3892 / 9608 | 5.6335 / 6.5215 | 40.5079 / 32.1070 | -3639 | -18.9082 [-19.9450, -17.8951] |
| D | 321 / 4491 | 1867 / 9608 | 7.1476 / 7.3957 | 19.4317 / 14.4655 | -1546 | -7.0863 [-7.8061, -6.3780] |

## all/pyramid/P5

10784 candidates; 4413 eligible images; 587 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 82.9106 / 84.4041 | 80.7585 / 83.9711 | 90.8525 / 91.8618 | 96.2756 / 96.7043 | 12.2347 / 11.9594 |
| B | 82.9827 / 84.5158 | 80.5267 / 83.9429 | 92.7472 / 93.2368 | 96.3306 / 96.7736 | 14.8549 / 13.8460 |
| D1 | 71.9482 / 75.1527 | 53.5423 / 62.0487 | 93.6738 / 94.1859 | 94.3270 / 95.2678 | 42.6805 / 38.2493 |
| D | 79.2705 / 81.7487 | 72.9136 / 78.7201 | 91.7947 / 92.6565 | 95.6862 / 96.3134 | 22.1570 / 19.8606 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7121 | -2.7671 | [-2.9409, -2.5962] | 10784 / 4413 |
| D_minus_B | mask75 | -7.6131 | -5.2229 | [-5.8176, -4.6607] | 10784 / 4413 |
| D_minus_B | coverage | -0.9524 | -0.5803 | [-0.7562, -0.4069] | 10784 / 4413 |
| D_minus_B | auc | -0.6444 | -0.4602 | [-0.5196, -0.3988] | 10780 / 4413 |
| D_minus_B | fpr | 7.3022 | 6.0147 | [5.6920, 6.3451] | 10780 / 4413 |
| D_minus_A | iou | -3.6400 | -2.6554 | [-2.8465, -2.4739] | 10784 / 4413 |
| D_minus_A | mask75 | -7.8450 | -5.2511 | [-5.8891, -4.6440] | 10784 / 4413 |
| D_minus_A | coverage | 0.9423 | 0.7947 | [0.6124, 0.9767] | 10784 / 4413 |
| D_minus_A | auc | -0.5894 | -0.3909 | [-0.4592, -0.3216] | 10780 / 4413 |
| D_minus_A | fpr | 9.9223 | 7.9012 | [7.5496, 8.2764] | 10780 / 4413 |
| B_minus_A | iou | 0.0721 | 0.1117 | [0.0311, 0.1949] | 10784 / 4413 |
| B_minus_A | mask75 | -0.2318 | -0.0282 | [-0.3637, 0.3057] | 10784 / 4413 |
| B_minus_A | coverage | 1.8947 | 1.3751 | [1.2606, 1.4951] | 10784 / 4413 |
| B_minus_A | auc | 0.0550 | 0.0693 | [0.0359, 0.1034] | 10780 / 4413 |
| B_minus_A | fpr | 2.6202 | 1.8865 | [1.7475, 2.0310] | 10780 / 4413 |
| D1_minus_A | iou | -10.9623 | -9.2515 | [-9.5788, -8.9276] | 10784 / 4413 |
| D1_minus_A | mask75 | -27.2162 | -21.9225 | [-22.9197, -20.9000] | 10784 / 4413 |
| D1_minus_A | coverage | 2.8213 | 2.3241 | [2.0302, 2.6034] | 10784 / 4413 |
| D1_minus_A | auc | -1.9486 | -1.4365 | [-1.5478, -1.3306] | 10780 / 4413 |
| D1_minus_A | fpr | 30.4458 | 26.2898 | [25.6761, 26.9272] | 10780 / 4413 |
| D_minus_D1 | iou | 7.3223 | 6.5960 | [6.3383, 6.8696] | 10784 / 4413 |
| D_minus_D1 | mask75 | 19.3713 | 16.6714 | [15.7863, 17.5609] | 10784 / 4413 |
| D_minus_D1 | coverage | -1.8791 | -1.5294 | [-1.7637, -1.2713] | 10784 / 4413 |
| D_minus_D1 | auc | 1.3592 | 1.0456 | [0.9788, 1.1196] | 10780 / 4413 |
| D_minus_D1 | fpr | -20.5235 | -18.3886 | [-18.8538, -17.9435] | 10780 / 4413 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 174 / 2075 | 199 / 8709 | 8.3855 / 8.3717 | 2.2850 / 1.7271 | -25 | -0.0282 [-0.3637, 0.3057] |
| D1 | 252 / 2075 | 3187 / 8709 | 12.1446 / 13.4776 | 36.5943 / 30.1103 | -2935 | -21.9225 [-22.9197, -20.9000] |
| D | 223 / 2075 | 1069 / 8709 | 10.7470 / 11.6690 | 12.2747 / 9.0112 | -846 | -5.2511 [-5.8891, -4.6440] |

## tal_maskfail/size/small

4033 candidates; 1637 eligible images; 3363 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.4081 / 63.6052 | 0.0000 / 0.0000 | 90.4004 / 90.5074 | 90.5344 / 91.0786 | 44.2386 / 42.3075 |
| B | 60.9789 / 61.2011 | 2.3804 / 2.2827 | 93.7896 / 93.7147 | 90.5548 / 91.0791 | 55.0264 / 52.3636 |
| D1 | 57.0712 / 57.4361 | 3.5457 / 4.0822 | 91.4424 / 90.8776 | 88.5835 / 89.1569 | 62.7808 / 59.1760 |
| D | 60.4041 / 60.7078 | 5.2814 / 5.2546 | 90.2945 / 90.2288 | 89.7835 / 90.3146 | 51.0837 / 48.4312 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.5748 | -0.4933 | [-0.7879, -0.2137] | 4033 / 1637 |
| D_minus_B | mask75 | 2.9011 | 2.9719 | [2.0760, 3.8897] | 4033 / 1637 |
| D_minus_B | coverage | -3.4950 | -3.4859 | [-3.9322, -3.0703] | 4033 / 1637 |
| D_minus_B | auc | -0.7713 | -0.7646 | [-0.9170, -0.6223] | 4033 / 1637 |
| D_minus_B | fpr | -3.9426 | -3.9324 | [-4.5706, -3.2928] | 4033 / 1637 |
| D_minus_A | iou | -3.0040 | -2.8974 | [-3.2134, -2.5905] | 4033 / 1637 |
| D_minus_A | mask75 | 5.2814 | 5.2546 | [4.4235, 6.1766] | 4033 / 1637 |
| D_minus_A | coverage | -0.1059 | -0.2786 | [-0.6994, 0.1165] | 4033 / 1637 |
| D_minus_A | auc | -0.7509 | -0.7640 | [-0.9426, -0.5963] | 4033 / 1637 |
| D_minus_A | fpr | 6.8452 | 6.1237 | [5.4379, 6.8269] | 4033 / 1637 |
| B_minus_A | iou | -2.4292 | -2.4041 | [-2.5944, -2.2066] | 4033 / 1637 |
| B_minus_A | mask75 | 2.3804 | 2.2827 | [1.7500, 2.8595] | 4033 / 1637 |
| B_minus_A | coverage | 3.3891 | 3.2073 | [2.9691, 3.4536] | 4033 / 1637 |
| B_minus_A | auc | 0.0204 | 0.0005 | [-0.0931, 0.0900] | 4033 / 1637 |
| B_minus_A | fpr | 10.7878 | 10.0560 | [9.6497, 10.4721] | 4033 / 1637 |
| D1_minus_A | iou | -6.3369 | -6.1690 | [-6.5735, -5.7752] | 4033 / 1637 |
| D1_minus_A | mask75 | 3.5457 | 4.0822 | [3.3108, 4.9112] | 4033 / 1637 |
| D1_minus_A | coverage | 1.0420 | 0.3701 | [-0.1414, 0.8753] | 4033 / 1637 |
| D1_minus_A | auc | -1.9510 | -1.9217 | [-2.1631, -1.6931] | 4033 / 1637 |
| D1_minus_A | fpr | 18.5422 | 16.8685 | [15.9491, 17.8263] | 4033 / 1637 |
| D_minus_D1 | iou | 3.3329 | 3.2717 | [2.8858, 3.6550] | 4033 / 1637 |
| D_minus_D1 | mask75 | 1.7357 | 1.1724 | [0.2471, 2.1025] | 4033 / 1637 |
| D_minus_D1 | coverage | -1.1478 | -0.6487 | [-1.2271, -0.0536] | 4033 / 1637 |
| D_minus_D1 | auc | 1.2000 | 1.1577 | [1.0147, 1.3115] | 4033 / 1637 |
| D_minus_D1 | fpr | -11.6971 | -10.7448 | [-11.5650, -9.9390] | 4033 / 1637 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 96 / 4033 | 0 / 0 | 2.3804 / 2.2827 | NA / NA | 96 | 2.2827 [1.7500, 2.8595] |
| D1 | 143 / 4033 | 0 / 0 | 3.5457 / 4.0822 | NA / NA | 143 | 4.0822 [3.3108, 4.9112] |
| D | 213 / 4033 | 0 / 0 | 5.2814 / 5.2546 | NA / NA | 213 | 5.2546 [4.4235, 6.1766] |

## tal_maskfail/size/medium

1862 candidates; 1192 eligible images; 3808 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.6905 / 64.0912 | 0.0000 / 0.0000 | 81.7597 / 82.0710 | 91.7577 / 91.9085 | 21.9986 / 21.7813 |
| B | 64.0873 / 64.5004 | 9.2911 / 9.5799 | 88.3253 / 88.0832 | 91.9463 / 92.1177 | 29.9955 / 29.1227 |
| D1 | 55.8832 / 57.3178 | 8.1096 / 9.2514 | 88.2397 / 87.6186 | 88.5368 / 89.1700 | 48.8329 / 45.2653 |
| D | 60.2511 / 61.2209 | 11.9764 / 12.9813 | 83.4046 / 83.3066 | 90.6908 / 91.0625 | 31.2894 / 29.5985 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.8362 | -3.2795 | [-3.8928, -2.6891] | 1862 / 1192 |
| D_minus_B | mask75 | 2.6853 | 3.4014 | [1.7305, 5.0388] | 1862 / 1192 |
| D_minus_B | coverage | -4.9207 | -4.7767 | [-5.6052, -4.0049] | 1862 / 1192 |
| D_minus_B | auc | -1.2555 | -1.0553 | [-1.3253, -0.8076] | 1861 / 1191 |
| D_minus_B | fpr | 1.2939 | 0.4757 | [-0.4421, 1.3647] | 1861 / 1191 |
| D_minus_A | iou | -3.4394 | -2.8703 | [-3.5220, -2.2025] | 1862 / 1192 |
| D_minus_A | mask75 | 11.9764 | 12.9813 | [11.2424, 14.6481] | 1862 / 1192 |
| D_minus_A | coverage | 1.6449 | 1.2356 | [0.4195, 2.0279] | 1862 / 1192 |
| D_minus_A | auc | -1.0669 | -0.8461 | [-1.1343, -0.5850] | 1861 / 1191 |
| D_minus_A | fpr | 9.2908 | 7.8171 | [6.9090, 8.7619] | 1861 / 1191 |
| B_minus_A | iou | 0.3968 | 0.4092 | [-0.0230, 0.8643] | 1862 / 1192 |
| B_minus_A | mask75 | 9.2911 | 9.5799 | [8.1260, 11.0936] | 1862 / 1192 |
| B_minus_A | coverage | 6.5656 | 6.0122 | [5.5074, 6.5501] | 1862 / 1192 |
| B_minus_A | auc | 0.1886 | 0.2092 | [0.0882, 0.3329] | 1861 / 1191 |
| B_minus_A | fpr | 7.9968 | 7.3414 | [6.8591, 7.8584] | 1861 / 1191 |
| D1_minus_A | iou | -7.8073 | -6.7734 | [-7.5473, -6.0006] | 1862 / 1192 |
| D1_minus_A | mask75 | 8.1096 | 9.2514 | [7.7279, 10.8350] | 1862 / 1192 |
| D1_minus_A | coverage | 6.4801 | 5.5476 | [4.6639, 6.4103] | 1862 / 1192 |
| D1_minus_A | auc | -3.2209 | -2.7385 | [-3.0854, -2.4020] | 1861 / 1191 |
| D1_minus_A | fpr | 26.8343 | 23.4840 | [22.0399, 24.8631] | 1861 / 1191 |
| D_minus_D1 | iou | 4.3679 | 3.9031 | [3.1975, 4.5927] | 1862 / 1192 |
| D_minus_D1 | mask75 | 3.8668 | 3.7300 | [1.8857, 5.5314] | 1862 / 1192 |
| D_minus_D1 | coverage | -4.8352 | -4.3120 | [-5.1579, -3.4563] | 1862 / 1192 |
| D_minus_D1 | auc | 2.1540 | 1.8924 | [1.6572, 2.1238] | 1861 / 1191 |
| D_minus_D1 | fpr | -17.5435 | -15.6668 | [-16.7111, -14.6190] | 1861 / 1191 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 173 / 1862 | 0 / 0 | 9.2911 / 9.5799 | NA / NA | 173 | 9.5799 [8.1260, 11.0936] |
| D1 | 151 / 1862 | 0 / 0 | 8.1096 / 9.2514 | NA / NA | 151 | 9.2514 [7.7279, 10.8350] |
| D | 223 / 1862 | 0 / 0 | 11.9764 / 12.9813 | NA / NA | 223 | 12.9813 [11.2424, 14.6481] |

## tal_maskfail/size/large

888 candidates; 728 eligible images; 4272 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 58.8731 / 58.9010 | 0.0000 / 0.0000 | 71.9841 / 72.0059 | 85.7381 / 85.4582 | 21.3206 / 21.8223 |
| B | 62.0842 / 61.9657 | 13.9640 / 13.3585 | 79.3469 / 78.9149 | 85.7697 / 85.5082 | 29.2590 / 29.4925 |
| D1 | 59.4936 / 60.0598 | 22.2973 / 23.2944 | 87.6627 / 87.3681 | 83.3044 / 83.2703 | 56.7520 / 56.2301 |
| D | 59.0177 / 59.3358 | 17.2297 / 17.6969 | 78.1088 / 77.7669 | 84.3824 / 84.2514 | 35.9918 / 35.9447 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.0665 | -2.6299 | [-3.6033, -1.6761] | 888 / 728 |
| D_minus_B | mask75 | 3.2658 | 4.3384 | [1.6595, 6.9940] | 888 / 728 |
| D_minus_B | coverage | -1.2380 | -1.1480 | [-2.3117, -0.0039] | 888 / 728 |
| D_minus_B | auc | -1.3872 | -1.2568 | [-1.6812, -0.8578] | 886 / 726 |
| D_minus_B | fpr | 6.7328 | 6.4522 | [5.0461, 7.9312] | 886 / 726 |
| D_minus_A | iou | 0.1446 | 0.4349 | [-0.5460, 1.4706] | 888 / 728 |
| D_minus_A | mask75 | 17.2297 | 17.6969 | [15.0870, 20.4441] | 888 / 728 |
| D_minus_A | coverage | 6.1248 | 5.7610 | [4.6053, 6.8939] | 888 / 728 |
| D_minus_A | auc | -1.3557 | -1.2068 | [-1.6526, -0.7582] | 886 / 726 |
| D_minus_A | fpr | 14.6712 | 14.1224 | [12.5124, 15.7762] | 886 / 726 |
| B_minus_A | iou | 3.2111 | 3.0647 | [2.3655, 3.8372] | 888 / 728 |
| B_minus_A | mask75 | 13.9640 | 13.3585 | [11.1493, 15.6825] | 888 / 728 |
| B_minus_A | coverage | 7.3628 | 6.9090 | [6.0725, 7.7751] | 888 / 728 |
| B_minus_A | auc | 0.0315 | 0.0500 | [-0.1559, 0.2627] | 886 / 726 |
| B_minus_A | fpr | 7.9384 | 7.6702 | [6.8462, 8.4909] | 886 / 726 |
| D1_minus_A | iou | 0.6205 | 1.1588 | [-0.1240, 2.4633] | 888 / 728 |
| D1_minus_A | mask75 | 22.2973 | 23.2944 | [20.4670, 26.2477] | 888 / 728 |
| D1_minus_A | coverage | 15.6786 | 15.3622 | [14.1275, 16.6414] | 888 / 728 |
| D1_minus_A | auc | -2.4338 | -2.1879 | [-2.7371, -1.6173] | 886 / 726 |
| D1_minus_A | fpr | 35.4314 | 34.4079 | [32.4464, 36.3425] | 886 / 726 |
| D_minus_D1 | iou | -0.4758 | -0.7240 | [-1.8001, 0.3508] | 888 / 728 |
| D_minus_D1 | mask75 | -5.0676 | -5.5975 | [-8.4821, -2.6557] | 888 / 728 |
| D_minus_D1 | coverage | -9.5538 | -9.6012 | [-10.6836, -8.4930] | 888 / 728 |
| D_minus_D1 | auc | 1.0781 | 0.9811 | [0.6503, 1.3041] | 886 / 726 |
| D_minus_D1 | fpr | -20.7602 | -20.2855 | [-21.6543, -18.8847] | 886 / 726 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 124 / 888 | 0 / 0 | 13.9640 / 13.3585 | NA / NA | 124 | 13.3585 [11.1493, 15.6825] |
| D1 | 198 / 888 | 0 / 0 | 22.2973 / 23.2944 | NA / NA | 198 | 23.2944 [20.4670, 26.2477] |
| D | 153 / 888 | 0 / 0 | 17.2297 / 17.6969 | NA / NA | 153 | 17.6969 [15.0870, 20.4441] |

## tal_maskfail/pyramid/P3

2845 candidates; 1272 eligible images; 3728 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.7436 / 64.2382 | 0.0000 / 0.0000 | 91.2406 / 91.2532 | 89.7447 / 90.1352 | 51.8887 / 50.4616 |
| B | 61.1524 / 61.5655 | 2.0738 / 2.2802 | 94.2843 / 94.2208 | 89.7575 / 90.1429 | 64.0547 / 62.5496 |
| D1 | 57.7290 / 58.0391 | 2.6011 / 3.1476 | 92.6614 / 92.3085 | 87.7485 / 88.1431 | 71.6232 / 69.9603 |
| D | 60.7458 / 61.0936 | 4.6046 / 4.8163 | 91.2400 / 91.1489 | 88.9595 / 89.3767 | 59.5241 / 58.1658 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.4066 | -0.4719 | [-0.8147, -0.1537] | 2845 / 1272 |
| D_minus_B | mask75 | 2.5308 | 2.5361 | [1.5144, 3.5259] | 2845 / 1272 |
| D_minus_B | coverage | -3.0443 | -3.0719 | [-3.6089, -2.5975] | 2845 / 1272 |
| D_minus_B | auc | -0.7980 | -0.7662 | [-0.9544, -0.5849] | 2845 / 1272 |
| D_minus_B | fpr | -4.5305 | -4.3838 | [-5.2158, -3.5535] | 2845 / 1272 |
| D_minus_A | iou | -2.9978 | -3.1446 | [-3.5129, -2.7908] | 2845 / 1272 |
| D_minus_A | mask75 | 4.6046 | 4.8163 | [3.8477, 5.8080] | 2845 / 1272 |
| D_minus_A | coverage | -0.0005 | -0.1043 | [-0.6087, 0.3465] | 2845 / 1272 |
| D_minus_A | auc | -0.7852 | -0.7585 | [-0.9780, -0.5521] | 2845 / 1272 |
| D_minus_A | fpr | 7.6355 | 7.7042 | [6.8250, 8.6024] | 2845 / 1272 |
| B_minus_A | iou | -2.5912 | -2.6727 | [-2.8934, -2.4666] | 2845 / 1272 |
| B_minus_A | mask75 | 2.0738 | 2.2802 | [1.6240, 3.0051] | 2845 / 1272 |
| B_minus_A | coverage | 3.0437 | 2.9676 | [2.7027, 3.2568] | 2845 / 1272 |
| B_minus_A | auc | 0.0128 | 0.0077 | [-0.1121, 0.1206] | 2845 / 1272 |
| B_minus_A | fpr | 12.1660 | 12.0880 | [11.5844, 12.6348] | 2845 / 1272 |
| D1_minus_A | iou | -6.0145 | -6.1991 | [-6.6320, -5.7708] | 2845 / 1272 |
| D1_minus_A | mask75 | 2.6011 | 3.1476 | [2.3326, 4.0222] | 2845 / 1272 |
| D1_minus_A | coverage | 1.4209 | 1.0553 | [0.4518, 1.6017] | 2845 / 1272 |
| D1_minus_A | auc | -1.9962 | -1.9921 | [-2.2894, -1.7071] | 2845 / 1272 |
| D1_minus_A | fpr | 19.7346 | 19.4987 | [18.3931, 20.6061] | 2845 / 1272 |
| D_minus_D1 | iou | 3.0168 | 3.0545 | [2.5946, 3.5026] | 2845 / 1272 |
| D_minus_D1 | mask75 | 2.0035 | 1.6688 | [0.5768, 2.7421] | 2845 / 1272 |
| D_minus_D1 | coverage | -1.4214 | -1.1596 | [-1.8157, -0.4850] | 2845 / 1272 |
| D_minus_D1 | auc | 1.2110 | 1.2336 | [1.0515, 1.4204] | 2845 / 1272 |
| D_minus_D1 | fpr | -12.0991 | -11.7945 | [-12.7338, -10.8340] | 2845 / 1272 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 59 / 2845 | 0 / 0 | 2.0738 / 2.2802 | NA / NA | 59 | 2.2802 [1.6240, 3.0051] |
| D1 | 74 / 2845 | 0 / 0 | 2.6011 / 3.1476 | NA / NA | 74 | 3.1476 [2.3326, 4.0222] |
| D | 131 / 2845 | 0 / 0 | 4.6046 / 4.8163 | NA / NA | 131 | 4.8163 [3.8477, 5.8080] |

## tal_maskfail/pyramid/P4

2521 candidates; 1465 eligible images; 3535 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.6626 / 63.7022 | 0.0000 / 0.0000 | 84.9435 / 85.2019 | 91.7040 / 91.6921 | 24.9733 / 25.4212 |
| B | 63.0938 / 63.1961 | 7.1797 / 7.0174 | 90.5587 / 90.4838 | 91.8416 / 91.8356 | 33.2640 / 33.2357 |
| D1 | 56.5483 / 57.3188 | 7.6557 / 8.4746 | 88.5121 / 88.0429 | 89.1822 / 89.4871 | 47.1448 / 44.8580 |
| D | 60.5153 / 60.9943 | 10.0754 / 10.3076 | 85.7108 / 85.7515 | 90.7961 / 90.9541 | 32.6130 / 31.8558 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.5785 | -2.2018 | [-2.6685, -1.7616] | 2521 / 1465 |
| D_minus_B | mask75 | 2.8957 | 3.2902 | [1.9670, 4.6438] | 2521 / 1465 |
| D_minus_B | coverage | -4.8479 | -4.7323 | [-5.3751, -4.1117] | 2521 / 1465 |
| D_minus_B | auc | -1.0455 | -0.8814 | [-1.0958, -0.6831] | 2520 / 1464 |
| D_minus_B | fpr | -0.6510 | -1.3799 | [-2.1533, -0.6377] | 2520 / 1464 |
| D_minus_A | iou | -3.1473 | -2.7079 | [-3.2205, -2.1957] | 2521 / 1465 |
| D_minus_A | mask75 | 10.0754 | 10.3076 | [8.9234, 11.6700] | 2521 / 1465 |
| D_minus_A | coverage | 0.7673 | 0.5496 | [-0.0729, 1.1878] | 2521 / 1465 |
| D_minus_A | auc | -0.9078 | -0.7380 | [-0.9826, -0.5022] | 2520 / 1464 |
| D_minus_A | fpr | 7.6397 | 6.4345 | [5.6670, 7.2345] | 2520 / 1464 |
| B_minus_A | iou | -0.5688 | -0.5061 | [-0.8535, -0.1286] | 2521 / 1465 |
| B_minus_A | mask75 | 7.1797 | 7.0174 | [5.8951, 8.1798] | 2521 / 1465 |
| B_minus_A | coverage | 5.6152 | 5.2819 | [4.8937, 5.7182] | 2521 / 1465 |
| B_minus_A | auc | 0.1377 | 0.1434 | [0.0189, 0.2890] | 2520 / 1464 |
| B_minus_A | fpr | 8.2907 | 7.8144 | [7.4077, 8.2315] | 2520 / 1464 |
| D1_minus_A | iou | -7.1143 | -6.3833 | [-7.0057, -5.7608] | 2521 / 1465 |
| D1_minus_A | mask75 | 7.6557 | 8.4746 | [7.1918, 9.7387] | 2521 / 1465 |
| D1_minus_A | coverage | 3.5685 | 2.8410 | [2.1184, 3.5578] | 2521 / 1465 |
| D1_minus_A | auc | -2.5218 | -2.2051 | [-2.5031, -1.9122] | 2520 / 1464 |
| D1_minus_A | fpr | 22.1714 | 19.4368 | [18.2443, 20.6388] | 2520 / 1464 |
| D_minus_D1 | iou | 3.9670 | 3.6754 | [3.1277, 4.2499] | 2521 / 1465 |
| D_minus_D1 | mask75 | 2.4197 | 1.8330 | [0.4061, 3.2657] | 2521 / 1465 |
| D_minus_D1 | coverage | -2.8013 | -2.2914 | [-3.0240, -1.5249] | 2521 / 1465 |
| D_minus_D1 | auc | 1.6139 | 1.4671 | [1.2826, 1.6611] | 2520 / 1464 |
| D_minus_D1 | fpr | -14.5317 | -13.0023 | [-13.9479, -12.0203] | 2520 / 1464 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 181 / 2521 | 0 / 0 | 7.1797 / 7.0174 | NA / NA | 181 | 7.0174 [5.8951, 8.1798] |
| D1 | 193 / 2521 | 0 / 0 | 7.6557 / 8.4746 | NA / NA | 193 | 8.4746 [7.1918, 9.7387] |
| D | 254 / 2521 | 0 / 0 | 10.0754 / 10.3076 | NA / NA | 254 | 10.3076 [8.9234, 11.6700] |

## tal_maskfail/pyramid/P5

1417 candidates; 1089 eligible images; 3911 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 59.8108 / 60.1483 | 0.0000 / 0.0000 | 75.5266 / 75.8161 | 88.6451 / 88.5923 | 19.5673 / 19.7816 |
| B | 61.6452 / 61.9117 | 10.7975 / 10.5387 | 82.3132 / 82.0510 | 88.7001 / 88.6600 | 26.5764 / 26.3617 |
| D1 | 56.6378 / 57.8133 | 15.8786 / 16.9238 | 87.6311 / 87.2888 | 85.8291 / 86.1500 | 50.7296 / 48.9860 |
| D | 58.4503 / 59.2926 | 14.3966 / 15.2908 | 79.8610 / 79.7572 | 87.4482 / 87.5711 | 31.5252 / 30.7893 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.1949 | -2.6191 | [-3.3138, -1.9407] | 1417 / 1089 |
| D_minus_B | mask75 | 3.5992 | 4.7521 | [2.8926, 6.6606] | 1417 / 1089 |
| D_minus_B | coverage | -2.4522 | -2.2938 | [-3.1468, -1.4600] | 1417 / 1089 |
| D_minus_B | auc | -1.2519 | -1.0889 | [-1.3652, -0.7992] | 1415 / 1087 |
| D_minus_B | fpr | 4.9488 | 4.4275 | [3.4139, 5.4145] | 1415 / 1087 |
| D_minus_A | iou | -1.3604 | -0.8557 | [-1.6045, -0.0941] | 1417 / 1089 |
| D_minus_A | mask75 | 14.3966 | 15.2908 | [13.2935, 17.2578] | 1417 / 1089 |
| D_minus_A | coverage | 4.3344 | 3.9412 | [3.0830, 4.7864] | 1417 / 1089 |
| D_minus_A | auc | -1.1968 | -1.0211 | [-1.3160, -0.7042] | 1415 / 1087 |
| D_minus_A | fpr | 11.9579 | 11.0077 | [9.8761, 12.1621] | 1415 / 1087 |
| B_minus_A | iou | 1.8345 | 1.7634 | [1.2324, 2.3011] | 1417 / 1089 |
| B_minus_A | mask75 | 10.7975 | 10.5387 | [8.8795, 12.3035] | 1417 / 1089 |
| B_minus_A | coverage | 6.7865 | 6.2350 | [5.6380, 6.8663] | 1417 / 1089 |
| B_minus_A | auc | 0.0550 | 0.0677 | [-0.0859, 0.2184] | 1415 / 1087 |
| B_minus_A | fpr | 7.0092 | 6.5802 | [5.9733, 7.1938] | 1415 / 1087 |
| D1_minus_A | iou | -3.1730 | -2.3350 | [-3.3377, -1.3246] | 1417 / 1089 |
| D1_minus_A | mask75 | 15.8786 | 16.9238 | [14.8178, 19.1247] | 1417 / 1089 |
| D1_minus_A | coverage | 12.1045 | 11.4727 | [10.4320, 12.5139] | 1417 / 1089 |
| D1_minus_A | auc | -2.8160 | -2.4423 | [-2.8472, -2.0285] | 1415 / 1087 |
| D1_minus_A | fpr | 31.1623 | 29.2044 | [27.6892, 30.7692] | 1415 / 1087 |
| D_minus_D1 | iou | 1.8126 | 1.4792 | [0.6657, 2.2715] | 1417 / 1089 |
| D_minus_D1 | mask75 | -1.4820 | -1.6330 | [-3.8568, 0.5832] | 1417 / 1089 |
| D_minus_D1 | coverage | -7.7701 | -7.5315 | [-8.3872, -6.6697] | 1417 / 1089 |
| D_minus_D1 | auc | 1.6192 | 1.4211 | [1.1750, 1.6765] | 1415 / 1087 |
| D_minus_D1 | fpr | -19.2044 | -18.1968 | [-19.2789, -17.1317] | 1415 / 1087 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 153 / 1417 | 0 / 0 | 10.7975 / 10.5387 | NA / NA | 153 | 10.5387 [8.8795, 12.3035] |
| D1 | 225 / 1417 | 0 / 0 | 15.8786 / 16.9238 | NA / NA | 225 | 16.9238 [14.8178, 19.1247] |
| D | 204 / 1417 | 0 / 0 | 14.3966 / 15.2908 | NA / NA | 204 | 15.2908 [13.2935, 17.2578] |

## strict_maskfail/size/small

3303 candidates; 1447 eligible images; 3553 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 61.7867 / 61.7295 | 0.0000 / 0.0000 | 89.7858 / 89.7026 | 89.7952 / 90.2208 | 45.0796 / 43.4278 |
| B | 59.2979 / 59.2471 | 1.7257 / 1.5801 | 93.4658 / 93.2797 | 89.7834 / 90.1871 | 56.3515 / 54.1784 |
| D1 | 55.4711 / 55.3916 | 2.1798 / 2.0353 | 90.8667 / 90.0895 | 87.7151 / 88.0362 | 63.5670 / 60.6885 |
| D | 58.6732 / 58.7076 | 3.2092 / 3.0294 | 89.6449 / 89.4254 | 88.9411 / 89.3218 | 52.0978 / 49.8786 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.6248 | -0.5395 | [-0.8817, -0.2175] | 3303 / 1447 |
| D_minus_B | mask75 | 1.4835 | 1.4494 | [0.6696, 2.2155] | 3303 / 1447 |
| D_minus_B | coverage | -3.8210 | -3.8543 | [-4.3637, -3.3929] | 3303 / 1447 |
| D_minus_B | auc | -0.8423 | -0.8653 | [-1.0515, -0.6823] | 3303 / 1447 |
| D_minus_B | fpr | -4.2537 | -4.2998 | [-5.0551, -3.5539] | 3303 / 1447 |
| D_minus_A | iou | -3.1136 | -3.0218 | [-3.3788, -2.6800] | 3303 / 1447 |
| D_minus_A | mask75 | 3.2092 | 3.0294 | [2.3497, 3.7533] | 3303 / 1447 |
| D_minus_A | coverage | -0.1409 | -0.2773 | [-0.7639, 0.1885] | 3303 / 1447 |
| D_minus_A | auc | -0.8541 | -0.8990 | [-1.1176, -0.6840] | 3303 / 1447 |
| D_minus_A | fpr | 7.0182 | 6.4508 | [5.6705, 7.2372] | 3303 / 1447 |
| B_minus_A | iou | -2.4888 | -2.4824 | [-2.7067, -2.2742] | 3303 / 1447 |
| B_minus_A | mask75 | 1.7257 | 1.5801 | [1.0986, 2.1065] | 3303 / 1447 |
| B_minus_A | coverage | 3.6800 | 3.5771 | [3.3014, 3.8680] | 3303 / 1447 |
| B_minus_A | auc | -0.0118 | -0.0337 | [-0.1449, 0.0721] | 3303 / 1447 |
| B_minus_A | fpr | 11.2719 | 10.7506 | [10.3138, 11.2210] | 3303 / 1447 |
| D1_minus_A | iou | -6.3156 | -6.3379 | [-6.7833, -5.9115] | 3303 / 1447 |
| D1_minus_A | mask75 | 2.1798 | 2.0353 | [1.5083, 2.6093] | 3303 / 1447 |
| D1_minus_A | coverage | 1.0809 | 0.3869 | [-0.2258, 0.9831] | 3303 / 1447 |
| D1_minus_A | auc | -2.0801 | -2.1846 | [-2.4809, -1.8926] | 3303 / 1447 |
| D1_minus_A | fpr | 18.4874 | 17.2607 | [16.2318, 18.2772] | 3303 / 1447 |
| D_minus_D1 | iou | 3.2021 | 3.3161 | [2.8837, 3.7585] | 3303 / 1447 |
| D_minus_D1 | mask75 | 1.0294 | 0.9941 | [0.2492, 1.7605] | 3303 / 1447 |
| D_minus_D1 | coverage | -1.2218 | -0.6641 | [-1.3341, 0.0051] | 3303 / 1447 |
| D_minus_D1 | auc | 1.2260 | 1.2856 | [1.1032, 1.4764] | 3303 / 1447 |
| D_minus_D1 | fpr | -11.4692 | -10.8099 | [-11.6965, -9.9404] | 3303 / 1447 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 57 / 3303 | 0 / 0 | 1.7257 / 1.5801 | NA / NA | 57 | 1.5801 [1.0986, 2.1065] |
| D1 | 72 / 3303 | 0 / 0 | 2.1798 / 2.0353 | NA / NA | 72 | 2.0353 [1.5083, 2.6093] |
| D | 106 / 3303 | 0 / 0 | 3.2092 / 3.0294 | NA / NA | 106 | 3.0294 [2.3497, 3.7533] |

## strict_maskfail/size/medium

1349 candidates; 916 eligible images; 4084 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 61.8270 / 62.1797 | 0.0000 / 0.0000 | 79.2296 / 79.5487 | 90.8724 / 91.0347 | 20.2971 / 20.0862 |
| B | 62.1715 / 62.5188 | 7.2646 / 7.1488 | 86.2829 / 86.0737 | 90.9657 / 91.1436 | 28.5914 / 27.7862 |
| D1 | 53.9630 / 55.1720 | 6.2268 / 6.9141 | 86.5727 / 85.9736 | 87.3337 / 87.9510 | 47.8276 / 44.6636 |
| D | 58.1961 / 59.1219 | 8.3766 / 9.0312 | 80.9199 / 80.9264 | 89.7227 / 90.1590 | 29.7248 / 28.2304 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.9754 | -3.3969 | [-4.0191, -2.7703] | 1349 / 916 |
| D_minus_B | mask75 | 1.1119 | 1.8824 | [0.2058, 3.4973] | 1349 / 916 |
| D_minus_B | coverage | -5.3631 | -5.1474 | [-6.1116, -4.2011] | 1349 / 916 |
| D_minus_B | auc | -1.2431 | -0.9846 | [-1.2394, -0.7377] | 1349 / 916 |
| D_minus_B | fpr | 1.1334 | 0.4442 | [-0.6148, 1.5645] | 1349 / 916 |
| D_minus_A | iou | -3.6308 | -3.0578 | [-3.7616, -2.3448] | 1349 / 916 |
| D_minus_A | mask75 | 8.3766 | 9.0312 | [7.3455, 10.7624] | 1349 / 916 |
| D_minus_A | coverage | 1.6902 | 1.3776 | [0.4158, 2.3230] | 1349 / 916 |
| D_minus_A | auc | -1.1498 | -0.8758 | [-1.1780, -0.5908] | 1349 / 916 |
| D_minus_A | fpr | 9.4277 | 8.1442 | [7.0699, 9.2875] | 1349 / 916 |
| B_minus_A | iou | 0.3446 | 0.3391 | [-0.0988, 0.7985] | 1349 / 916 |
| B_minus_A | mask75 | 7.2646 | 7.1488 | [5.7386, 8.7318] | 1349 / 916 |
| B_minus_A | coverage | 7.0533 | 6.5250 | [5.9545, 7.1110] | 1349 / 916 |
| B_minus_A | auc | 0.0933 | 0.1089 | [-0.0285, 0.2400] | 1349 / 916 |
| B_minus_A | fpr | 8.2943 | 7.7000 | [7.0898, 8.3398] | 1349 / 916 |
| D1_minus_A | iou | -7.8640 | -7.0077 | [-7.8259, -6.1867] | 1349 / 916 |
| D1_minus_A | mask75 | 6.2268 | 6.9141 | [5.4310, 8.4975] | 1349 / 916 |
| D1_minus_A | coverage | 7.3431 | 6.4248 | [5.4594, 7.3940] | 1349 / 916 |
| D1_minus_A | auc | -3.5387 | -3.0837 | [-3.4981, -2.6772] | 1349 / 916 |
| D1_minus_A | fpr | 27.5305 | 24.5774 | [22.9087, 26.2294] | 1349 / 916 |
| D_minus_D1 | iou | 4.2331 | 3.9499 | [3.1614, 4.7368] | 1349 / 916 |
| D_minus_D1 | mask75 | 2.1497 | 2.1171 | [0.3078, 3.8795] | 1349 / 916 |
| D_minus_D1 | coverage | -5.6529 | -5.0472 | [-6.1043, -4.0447] | 1349 / 916 |
| D_minus_D1 | auc | 2.3889 | 2.2079 | [1.9328, 2.4871] | 1349 / 916 |
| D_minus_D1 | fpr | -18.1028 | -16.4332 | [-17.7249, -15.1690] | 1349 / 916 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 98 / 1349 | 0 / 0 | 7.2646 / 7.1488 | NA / NA | 98 | 7.1488 [5.7386, 8.7318] |
| D1 | 84 / 1349 | 0 / 0 | 6.2268 / 6.9141 | NA / NA | 84 | 6.9141 [5.4310, 8.4975] |
| D | 113 / 1349 | 0 / 0 | 8.3766 / 9.0312 | NA / NA | 113 | 9.0312 [7.3455, 10.7624] |

## strict_maskfail/size/large

652 candidates; 565 eligible images; 4435 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 56.7639 / 56.7567 | 0.0000 / 0.0000 | 67.8273 / 67.8113 | 83.0910 / 82.8037 | 20.3084 / 20.8338 |
| B | 60.1018 / 59.9941 | 10.7362 / 10.2360 | 75.1141 / 74.8238 | 83.0892 / 82.8209 | 28.7194 / 29.1750 |
| D1 | 59.4497 / 59.7808 | 23.1595 / 23.3628 | 85.8030 / 85.4715 | 80.8954 / 80.7449 | 57.0890 / 56.9052 |
| D | 57.8970 / 58.1207 | 15.3374 / 15.4867 | 74.7544 / 74.5524 | 81.7224 / 81.5352 | 35.0055 / 35.3265 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.2047 | -1.8734 | [-2.9046, -0.8289] | 652 / 565 |
| D_minus_B | mask75 | 4.6012 | 5.2507 | [2.4189, 8.2006] | 652 / 565 |
| D_minus_B | coverage | -0.3597 | -0.2714 | [-1.5969, 1.0666] | 652 / 565 |
| D_minus_B | auc | -1.3669 | -1.2857 | [-1.8005, -0.7560] | 650 / 563 |
| D_minus_B | fpr | 6.2861 | 6.1515 | [4.5097, 7.8583] | 650 / 563 |
| D_minus_A | iou | 1.1331 | 1.3639 | [0.2863, 2.5025] | 652 / 565 |
| D_minus_A | mask75 | 15.3374 | 15.4867 | [12.6549, 18.4963] | 652 / 565 |
| D_minus_A | coverage | 6.9271 | 6.7410 | [5.4031, 8.1332] | 652 / 565 |
| D_minus_A | auc | -1.3687 | -1.2686 | [-1.8370, -0.6868] | 650 / 563 |
| D_minus_A | fpr | 14.6971 | 14.4928 | [12.5952, 16.4775] | 650 / 563 |
| B_minus_A | iou | 3.3378 | 3.2374 | [2.5112, 4.0028] | 652 / 565 |
| B_minus_A | mask75 | 10.7362 | 10.2360 | [7.9351, 12.7139] | 652 / 565 |
| B_minus_A | coverage | 7.2868 | 7.0124 | [6.1233, 8.0146] | 652 / 565 |
| B_minus_A | auc | -0.0018 | 0.0172 | [-0.2415, 0.2722] | 650 / 563 |
| B_minus_A | fpr | 8.4110 | 8.3412 | [7.3573, 9.3670] | 650 / 563 |
| D1_minus_A | iou | 2.6857 | 3.0240 | [1.5894, 4.4802] | 652 / 565 |
| D1_minus_A | mask75 | 23.1595 | 23.3628 | [20.0878, 26.7847] | 652 / 565 |
| D1_minus_A | coverage | 17.9757 | 17.6602 | [16.2415, 19.1468] | 652 / 565 |
| D1_minus_A | auc | -2.1956 | -2.0588 | [-2.7702, -1.3468] | 650 / 563 |
| D1_minus_A | fpr | 36.7806 | 36.0714 | [33.8505, 38.3181] | 650 / 563 |
| D_minus_D1 | iou | -1.5526 | -1.6601 | [-2.8035, -0.5306] | 652 / 565 |
| D_minus_D1 | mask75 | -7.8221 | -7.8761 | [-11.2094, -4.7198] | 652 / 565 |
| D_minus_D1 | coverage | -11.0486 | -10.9192 | [-12.0702, -9.7532] | 652 / 565 |
| D_minus_D1 | auc | 0.8269 | 0.7903 | [0.3807, 1.1872] | 650 / 563 |
| D_minus_D1 | fpr | -22.0835 | -21.5787 | [-23.2208, -19.9534] | 650 / 563 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 70 / 652 | 0 / 0 | 10.7362 / 10.2360 | NA / NA | 70 | 10.2360 [7.9351, 12.7139] |
| D1 | 151 / 652 | 0 / 0 | 23.1595 / 23.3628 | NA / NA | 151 | 23.3628 [20.0878, 26.7847] |
| D | 100 / 652 | 0 / 0 | 15.3374 / 15.4867 | NA / NA | 100 | 15.4867 [12.6549, 18.4963] |

## strict_maskfail/pyramid/P3

2387 candidates; 1138 eligible images; 3862 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 62.3859 / 62.7485 | 0.0000 / 0.0000 | 90.8745 / 90.8017 | 89.1435 / 89.5478 | 52.3321 / 50.9879 |
| B | 59.7046 / 59.9352 | 1.3825 / 1.3632 | 94.1660 / 94.0289 | 89.1251 / 89.5165 | 64.9644 / 63.6752 |
| D1 | 56.2963 / 56.3220 | 1.8433 / 1.9927 | 92.3793 / 91.8802 | 86.9993 / 87.2740 | 72.2103 / 70.9951 |
| D | 59.2653 / 59.3950 | 2.9744 / 3.0458 | 90.8438 / 90.6159 | 88.2341 / 88.5962 | 60.1734 / 58.9485 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.4393 | -0.5402 | [-0.9196, -0.1770] | 2387 / 1138 |
| D_minus_B | mask75 | 1.5920 | 1.6826 | [0.7892, 2.6424] | 2387 / 1138 |
| D_minus_B | coverage | -3.3223 | -3.4131 | [-4.0076, -2.8474] | 2387 / 1138 |
| D_minus_B | auc | -0.8910 | -0.9202 | [-1.1358, -0.7018] | 2387 / 1138 |
| D_minus_B | fpr | -4.7910 | -4.7267 | [-5.7124, -3.7774] | 2387 / 1138 |
| D_minus_A | iou | -3.1206 | -3.3535 | [-3.7555, -2.9453] | 2387 / 1138 |
| D_minus_A | mask75 | 2.9744 | 3.0458 | [2.2383, 3.9385] | 2387 / 1138 |
| D_minus_A | coverage | -0.0308 | -0.1858 | [-0.7436, 0.3536] | 2387 / 1138 |
| D_minus_A | auc | -0.9094 | -0.9516 | [-1.2078, -0.7022] | 2387 / 1138 |
| D_minus_A | fpr | 7.8412 | 7.9607 | [6.9361, 8.9319] | 2387 / 1138 |
| B_minus_A | iou | -2.6813 | -2.8134 | [-3.0535, -2.5814] | 2387 / 1138 |
| B_minus_A | mask75 | 1.3825 | 1.3632 | [0.8574, 1.9576] | 2387 / 1138 |
| B_minus_A | coverage | 3.2915 | 3.2272 | [2.9274, 3.5614] | 2387 / 1138 |
| B_minus_A | auc | -0.0184 | -0.0314 | [-0.1637, 0.0974] | 2387 / 1138 |
| B_minus_A | fpr | 12.6322 | 12.6874 | [12.1104, 13.2888] | 2387 / 1138 |
| D1_minus_A | iou | -6.0895 | -6.4266 | [-6.8993, -5.9469] | 2387 / 1138 |
| D1_minus_A | mask75 | 1.8433 | 1.9927 | [1.3279, 2.6950] | 2387 / 1138 |
| D1_minus_A | coverage | 1.5047 | 1.0785 | [0.4303, 1.7190] | 2387 / 1138 |
| D1_minus_A | auc | -2.1442 | -2.2739 | [-2.6233, -1.9189] | 2387 / 1138 |
| D1_minus_A | fpr | 19.8781 | 20.0073 | [18.8325, 21.2264] | 2387 / 1138 |
| D_minus_D1 | iou | 2.9690 | 3.0731 | [2.5442, 3.5835] | 2387 / 1138 |
| D_minus_D1 | mask75 | 1.1311 | 1.0531 | [0.1154, 2.0396] | 2387 / 1138 |
| D_minus_D1 | coverage | -1.5355 | -1.2643 | [-2.0753, -0.4922] | 2387 / 1138 |
| D_minus_D1 | auc | 1.2348 | 1.3222 | [1.1057, 1.5410] | 2387 / 1138 |
| D_minus_D1 | fpr | -12.0369 | -12.0466 | [-13.1474, -10.9747] | 2387 / 1138 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 33 / 2387 | 0 / 0 | 1.3825 / 1.3632 | NA / NA | 33 | 1.3632 [0.8574, 1.9576] |
| D1 | 44 / 2387 | 0 / 0 | 1.8433 / 1.9927 | NA / NA | 44 | 1.9927 [1.3279, 2.6950] |
| D | 71 / 2387 | 0 / 0 | 2.9744 / 3.0458 | NA / NA | 71 | 3.0458 [2.2383, 3.9385] |

## strict_maskfail/pyramid/P4

1847 candidates; 1164 eligible images; 3836 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 61.4714 / 61.3054 | 0.0000 / 0.0000 | 83.0222 / 83.1256 | 90.7112 / 90.5272 | 24.2477 / 24.9575 |
| B | 60.8092 / 60.6768 | 5.3059 / 4.9870 | 89.1008 / 88.9676 | 90.7706 / 90.5557 | 32.9292 / 33.3978 |
| D1 | 54.4637 / 54.9278 | 5.1435 / 5.1540 | 86.8485 / 86.3114 | 88.0192 / 88.0720 | 46.0190 / 44.4455 |
| D | 58.1572 / 58.4130 | 6.4970 / 6.5003 | 83.7410 / 83.6674 | 89.7421 / 89.7103 | 31.8183 / 31.4797 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.6520 | -2.2638 | [-2.7694, -1.7655] | 1847 / 1164 |
| D_minus_B | mask75 | 1.1911 | 1.5133 | [0.2461, 2.7719] | 1847 / 1164 |
| D_minus_B | coverage | -5.3599 | -5.3002 | [-6.0751, -4.5377] | 1847 / 1164 |
| D_minus_B | auc | -1.0285 | -0.8453 | [-1.0721, -0.6213] | 1847 / 1164 |
| D_minus_B | fpr | -1.1108 | -1.9182 | [-2.8198, -1.0174] | 1847 / 1164 |
| D_minus_A | iou | -3.3141 | -2.8924 | [-3.4366, -2.3594] | 1847 / 1164 |
| D_minus_A | mask75 | 6.4970 | 6.5003 | [5.2802, 7.7879] | 1847 / 1164 |
| D_minus_A | coverage | 0.7188 | 0.5418 | [-0.1978, 1.2964] | 1847 / 1164 |
| D_minus_A | auc | -0.9691 | -0.8169 | [-1.0914, -0.5505] | 1847 / 1164 |
| D_minus_A | fpr | 7.5707 | 6.5222 | [5.6362, 7.4512] | 1847 / 1164 |
| B_minus_A | iou | -0.6622 | -0.6286 | [-0.9755, -0.2662] | 1847 / 1164 |
| B_minus_A | mask75 | 5.3059 | 4.9870 | [3.9517, 6.0749] | 1847 / 1164 |
| B_minus_A | coverage | 6.0787 | 5.8420 | [5.4111, 6.2940] | 1847 / 1164 |
| B_minus_A | auc | 0.0594 | 0.0285 | [-0.0974, 0.1483] | 1847 / 1164 |
| B_minus_A | fpr | 8.6815 | 8.4403 | [7.9483, 8.9657] | 1847 / 1164 |
| D1_minus_A | iou | -7.0077 | -6.3777 | [-7.0277, -5.7535] | 1847 / 1164 |
| D1_minus_A | mask75 | 5.1435 | 5.1540 | [4.0693, 6.3087] | 1847 / 1164 |
| D1_minus_A | coverage | 3.8263 | 3.1858 | [2.3540, 4.0197] | 1847 / 1164 |
| D1_minus_A | auc | -2.6920 | -2.4552 | [-2.8171, -2.1192] | 1847 / 1164 |
| D1_minus_A | fpr | 21.7713 | 19.4880 | [18.1422, 20.9201] | 1847 / 1164 |
| D_minus_D1 | iou | 3.6936 | 3.4852 | [2.8818, 4.0919] | 1847 / 1164 |
| D_minus_D1 | mask75 | 1.3535 | 1.3463 | [0.0333, 2.6854] | 1847 / 1164 |
| D_minus_D1 | coverage | -3.1075 | -2.6440 | [-3.5244, -1.7384] | 1847 / 1164 |
| D_minus_D1 | auc | 1.7228 | 1.6383 | [1.4226, 1.8584] | 1847 / 1164 |
| D_minus_D1 | fpr | -14.2006 | -12.9659 | [-14.1258, -11.8669] | 1847 / 1164 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 98 / 1847 | 0 / 0 | 5.3059 / 4.9870 | NA / NA | 98 | 4.9870 [3.9517, 6.0749] |
| D1 | 95 / 1847 | 0 / 0 | 5.1435 / 5.1540 | NA / NA | 95 | 5.1540 [4.0693, 6.3087] |
| D | 120 / 1847 | 0 / 0 | 6.4970 / 6.5003 | NA / NA | 120 | 6.5003 [5.2802, 7.7879] |

## strict_maskfail/pyramid/P5

1070 candidates; 867 eligible images; 4133 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 57.9845 / 58.3470 | 0.0000 / 0.0000 | 72.3432 / 72.5175 | 86.9479 / 86.8816 | 18.5177 / 18.6977 |
| B | 59.8946 / 60.2218 | 8.7850 / 8.3218 | 79.2002 / 78.9288 | 86.9667 / 86.9100 | 25.7269 / 25.6842 |
| D1 | 55.8920 / 56.9668 | 15.7009 / 16.3322 | 85.9294 / 85.5847 | 84.1566 / 84.3468 | 50.7735 / 49.6664 |
| D | 57.1684 / 58.0159 | 11.9626 / 12.7855 | 77.0881 / 77.1088 | 85.7299 / 85.7629 | 30.4578 / 30.2219 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.7262 | -2.2059 | [-2.9718, -1.4258] | 1070 / 867 |
| D_minus_B | mask75 | 3.1776 | 4.4637 | [2.4702, 6.5688] | 1070 / 867 |
| D_minus_B | coverage | -2.1121 | -1.8200 | [-2.8765, -0.7797] | 1070 / 867 |
| D_minus_B | auc | -1.2368 | -1.1471 | [-1.5254, -0.7935] | 1068 / 865 |
| D_minus_B | fpr | 4.7310 | 4.5377 | [3.2912, 5.8195] | 1068 / 865 |
| D_minus_A | iou | -0.8161 | -0.3310 | [-1.1799, 0.5323] | 1070 / 867 |
| D_minus_A | mask75 | 11.9626 | 12.7855 | [10.6805, 14.9771] | 1070 / 867 |
| D_minus_A | coverage | 4.7448 | 4.5913 | [3.5358, 5.6691] | 1070 / 867 |
| D_minus_A | auc | -1.2181 | -1.1187 | [-1.5235, -0.7256] | 1068 / 865 |
| D_minus_A | fpr | 11.9402 | 11.5242 | [10.1831, 12.9258] | 1068 / 865 |
| B_minus_A | iou | 1.9101 | 1.8748 | [1.3676, 2.4264] | 1070 / 867 |
| B_minus_A | mask75 | 8.7850 | 8.3218 | [6.6417, 10.1019] | 1070 / 867 |
| B_minus_A | coverage | 6.8570 | 6.4113 | [5.7263, 7.1510] | 1070 / 867 |
| B_minus_A | auc | 0.0187 | 0.0284 | [-0.1431, 0.1989] | 1068 / 865 |
| B_minus_A | fpr | 7.2092 | 6.9865 | [6.2674, 7.7331] | 1068 / 865 |
| D1_minus_A | iou | -2.0925 | -1.3801 | [-2.5191, -0.2648] | 1070 / 867 |
| D1_minus_A | mask75 | 15.7009 | 16.3322 | [13.9791, 18.7431] | 1070 / 867 |
| D1_minus_A | coverage | 13.5862 | 13.0672 | [11.8561, 14.2779] | 1070 / 867 |
| D1_minus_A | auc | -2.7913 | -2.5348 | [-3.0498, -2.0245] | 1068 / 865 |
| D1_minus_A | fpr | 32.2559 | 30.9687 | [29.0812, 32.7935] | 1068 / 865 |
| D_minus_D1 | iou | 1.2764 | 1.0491 | [0.1880, 1.9169] | 1070 / 867 |
| D_minus_D1 | mask75 | -3.7383 | -3.5467 | [-5.9213, -1.1534] | 1070 / 867 |
| D_minus_D1 | coverage | -8.8413 | -8.4759 | [-9.4330, -7.5189] | 1070 / 867 |
| D_minus_D1 | auc | 1.5732 | 1.4161 | [1.0963, 1.7259] | 1068 / 865 |
| D_minus_D1 | fpr | -20.3157 | -19.4445 | [-20.6903, -18.1717] | 1068 / 865 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 94 / 1070 | 0 / 0 | 8.7850 / 8.3218 | NA / NA | 94 | 8.3218 [6.6417, 10.1019] |
| D1 | 168 / 1070 | 0 / 0 | 15.7009 / 16.3322 | NA / NA | 168 | 16.3322 [13.9791, 18.7431] |
| D | 128 / 1070 | 0 / 0 | 11.9626 / 12.7855 | NA / NA | 128 | 12.7855 [10.6805, 14.9771] |

## baseline_success/size/small

4753 candidates; 1831 eligible images; 3169 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 82.4782 / 82.5056 | 100.0000 / 100.0000 | 95.0314 / 95.0543 | 96.1376 / 96.2009 | 32.5853 / 31.8365 |
| B | 80.7157 / 80.8875 | 85.3145 / 86.6467 | 96.1857 / 96.1730 | 96.2469 / 96.3102 | 40.1903 / 38.8099 |
| D1 | 74.7035 / 75.7675 | 57.3953 / 62.7175 | 95.4532 / 95.3358 | 94.7481 / 95.0922 | 56.7002 / 52.3371 |
| D | 78.8329 / 79.3447 | 75.8258 / 78.6991 | 94.6490 / 94.6000 | 95.7049 / 95.8354 | 41.4870 / 39.1709 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.8828 | -1.5428 | [-1.7837, -1.3031] | 4753 / 1831 |
| D_minus_B | mask75 | -9.4887 | -7.9476 | [-9.2904, -6.6023] | 4753 / 1831 |
| D_minus_B | coverage | -1.5367 | -1.5730 | [-1.8231, -1.3296] | 4753 / 1831 |
| D_minus_B | auc | -0.5419 | -0.4748 | [-0.5950, -0.3713] | 4748 / 1831 |
| D_minus_B | fpr | 1.2967 | 0.3610 | [-0.1761, 0.9150] | 4748 / 1831 |
| D_minus_A | iou | -3.6453 | -3.1609 | [-3.4316, -2.8955] | 4753 / 1831 |
| D_minus_A | mask75 | -24.1742 | -21.3009 | [-22.7604, -19.8048] | 4753 / 1831 |
| D_minus_A | coverage | -0.3823 | -0.4543 | [-0.7067, -0.2124] | 4753 / 1831 |
| D_minus_A | auc | -0.4327 | -0.3655 | [-0.4897, -0.2548] | 4748 / 1831 |
| D_minus_A | fpr | 8.9018 | 7.3344 | [6.7035, 7.9737] | 4748 / 1831 |
| B_minus_A | iou | -1.7625 | -1.6181 | [-1.7259, -1.5126] | 4753 / 1831 |
| B_minus_A | mask75 | -14.6855 | -13.3533 | [-14.5636, -12.1591] | 4753 / 1831 |
| B_minus_A | coverage | 1.1543 | 1.1188 | [1.0586, 1.1807] | 4753 / 1831 |
| B_minus_A | auc | 0.1092 | 0.1094 | [0.0794, 0.1411] | 4748 / 1831 |
| B_minus_A | fpr | 7.6050 | 6.9734 | [6.6685, 7.2662] | 4748 / 1831 |
| D1_minus_A | iou | -7.7747 | -6.7381 | [-7.1185, -6.3756] | 4753 / 1831 |
| D1_minus_A | mask75 | -42.6047 | -37.2825 | [-39.0508, -35.4649] | 4753 / 1831 |
| D1_minus_A | coverage | 0.4218 | 0.2816 | [-0.0019, 0.5525] | 4753 / 1831 |
| D1_minus_A | auc | -1.3896 | -1.1087 | [-1.2704, -0.9588] | 4748 / 1831 |
| D1_minus_A | fpr | 24.1149 | 20.5006 | [19.5456, 21.4612] | 4748 / 1831 |
| D_minus_D1 | iou | 4.1294 | 3.5773 | [3.2685, 3.8936] | 4753 / 1831 |
| D_minus_D1 | mask75 | 18.4305 | 15.9816 | [14.3830, 17.6001] | 4753 / 1831 |
| D_minus_D1 | coverage | -0.8041 | -0.7358 | [-1.0172, -0.4364] | 4753 / 1831 |
| D_minus_D1 | auc | 0.9569 | 0.7432 | [0.6544, 0.8290] | 4748 / 1831 |
| D_minus_D1 | fpr | -15.2131 | -13.1662 | [-13.8612, -12.4784] | 4748 / 1831 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 698 / 4753 | NA / NA | 14.6855 / 13.3533 | -698 | -13.3533 [-14.5636, -12.1591] |
| D1 | 0 / 0 | 2025 / 4753 | NA / NA | 42.6047 / 37.2825 | -2025 | -37.2825 [-39.0508, -35.4649] |
| D | 0 / 0 | 1149 / 4753 | NA / NA | 24.1742 / 21.3009 | -1149 | -21.3009 [-22.7604, -19.8048] |

## baseline_success/size/medium

8482 candidates; 3032 eligible images; 1968 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 86.2917 / 86.3784 | 100.0000 / 100.0000 | 94.7874 / 94.9813 | 97.8693 / 97.9418 | 16.2553 / 16.3106 |
| B | 85.5084 / 85.7230 | 95.5199 / 95.9787 | 95.8479 / 95.8575 | 97.9155 / 97.9935 | 19.7324 / 19.2423 |
| D1 | 75.1125 / 77.7642 | 59.6675 / 68.5636 | 95.6400 / 95.3355 | 95.6880 / 96.3453 | 45.2362 / 38.1597 |
| D | 81.6742 / 83.1768 | 81.6081 / 87.1868 | 94.6017 / 94.6358 | 97.2506 / 97.4925 | 26.4378 / 23.0607 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.8342 | -2.5462 | [-2.7386, -2.3619] | 8482 / 3032 |
| D_minus_B | mask75 | -13.9118 | -8.7919 | [-9.6247, -7.9852] | 8482 / 3032 |
| D_minus_B | coverage | -1.2462 | -1.2216 | [-1.3756, -1.0757] | 8482 / 3032 |
| D_minus_B | auc | -0.6648 | -0.5010 | [-0.5619, -0.4470] | 8480 / 3032 |
| D_minus_B | fpr | 6.7054 | 3.8184 | [3.4139, 4.2457] | 8480 / 3032 |
| D_minus_A | iou | -4.6175 | -3.2016 | [-3.4027, -3.0008] | 8482 / 3032 |
| D_minus_A | mask75 | -18.3919 | -12.8132 | [-13.7211, -11.9191] | 8482 / 3032 |
| D_minus_A | coverage | -0.1857 | -0.3455 | [-0.4969, -0.2005] | 8482 / 3032 |
| D_minus_A | auc | -0.6187 | -0.4493 | [-0.5092, -0.3938] | 8480 / 3032 |
| D_minus_A | fpr | 10.1825 | 6.7501 | [6.3092, 7.2190] | 8480 / 3032 |
| B_minus_A | iou | -0.7833 | -0.6554 | [-0.7229, -0.5905] | 8482 / 3032 |
| B_minus_A | mask75 | -4.4801 | -4.0213 | [-4.5575, -3.5287] | 8482 / 3032 |
| B_minus_A | coverage | 1.0606 | 0.8762 | [0.8362, 0.9203] | 8482 / 3032 |
| B_minus_A | auc | 0.0462 | 0.0517 | [0.0353, 0.0684] | 8480 / 3032 |
| B_minus_A | fpr | 3.4771 | 2.9317 | [2.7741, 3.0948] | 8480 / 3032 |
| D1_minus_A | iou | -11.1792 | -8.6142 | [-8.9686, -8.2666] | 8482 / 3032 |
| D1_minus_A | mask75 | -40.3325 | -31.4364 | [-32.8131, -30.1156] | 8482 / 3032 |
| D1_minus_A | coverage | 0.8526 | 0.3542 | [0.0976, 0.5966] | 8482 / 3032 |
| D1_minus_A | auc | -2.1813 | -1.5965 | [-1.7259, -1.4786] | 8480 / 3032 |
| D1_minus_A | fpr | 28.9808 | 21.8491 | [21.0586, 22.6593] | 8480 / 3032 |
| D_minus_D1 | iou | 6.5616 | 5.4126 | [5.1356, 5.6885] | 8482 / 3032 |
| D_minus_D1 | mask75 | 21.9406 | 18.6232 | [17.4343, 19.8225] | 8482 / 3032 |
| D_minus_D1 | coverage | -1.0383 | -0.6997 | [-0.9249, -0.4679] | 8482 / 3032 |
| D_minus_D1 | auc | 1.5627 | 1.1472 | [1.0586, 1.2434] | 8480 / 3032 |
| D_minus_D1 | fpr | -18.7984 | -15.0990 | [-15.6508, -14.5591] | 8480 / 3032 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 380 / 8482 | NA / NA | 4.4801 / 4.0213 | -380 | -4.0213 [-4.5575, -3.5287] |
| D1 | 0 / 0 | 3421 / 8482 | NA / NA | 40.3325 / 31.4364 | -3421 | -31.4364 [-32.8131, -30.1156] |
| D | 0 / 0 | 1560 / 8482 | NA / NA | 18.3919 / 12.8132 | -1560 | -12.8132 [-13.7211, -11.9191] |

## baseline_success/size/large

6972 candidates; 3811 eligible images; 1189 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 90.0681 / 90.2268 | 100.0000 / 100.0000 | 95.0053 / 95.1152 | 98.4467 / 98.4835 | 10.0558 / 9.9333 |
| B | 89.9607 / 90.1690 | 98.6661 / 98.8939 | 95.5207 / 95.4766 | 98.4912 / 98.5282 | 11.2436 / 10.7799 |
| D1 | 77.5622 / 79.3663 | 65.9495 / 71.4816 | 95.8670 / 96.2154 | 96.6323 / 97.0578 | 42.4735 / 38.9362 |
| D | 86.1746 / 87.2027 | 90.0172 / 92.3917 | 95.3745 / 95.5976 | 97.9281 / 98.1013 | 20.3840 / 18.5567 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7860 | -2.9663 | [-3.1395, -2.7992] | 6972 / 3811 |
| D_minus_B | mask75 | -8.6489 | -6.5023 | [-7.1422, -5.8582] | 6972 / 3811 |
| D_minus_B | coverage | -0.1462 | 0.1210 | [-0.0129, 0.2486] | 6972 / 3811 |
| D_minus_B | auc | -0.5631 | -0.4269 | [-0.4795, -0.3757] | 6972 / 3811 |
| D_minus_B | fpr | 9.1404 | 7.7768 | [7.3962, 8.1663] | 6972 / 3811 |
| D_minus_A | iou | -3.8935 | -3.0242 | [-3.2078, -2.8427] | 6972 / 3811 |
| D_minus_A | mask75 | -9.9828 | -7.6083 | [-8.2937, -6.9303] | 6972 / 3811 |
| D_minus_A | coverage | 0.3692 | 0.4824 | [0.3467, 0.6147] | 6972 / 3811 |
| D_minus_A | auc | -0.5186 | -0.3821 | [-0.4397, -0.3272] | 6972 / 3811 |
| D_minus_A | fpr | 10.3282 | 8.6234 | [8.2147, 9.0326] | 6972 / 3811 |
| B_minus_A | iou | -0.1075 | -0.0579 | [-0.1018, -0.0149] | 6972 / 3811 |
| B_minus_A | mask75 | -1.3339 | -1.1061 | [-1.3693, -0.8615] | 6972 / 3811 |
| B_minus_A | coverage | 0.5154 | 0.3614 | [0.3245, 0.3987] | 6972 / 3811 |
| B_minus_A | auc | 0.0444 | 0.0447 | [0.0228, 0.0642] | 6972 / 3811 |
| B_minus_A | fpr | 1.1878 | 0.8466 | [0.7499, 0.9486] | 6972 / 3811 |
| D1_minus_A | iou | -12.5059 | -10.8606 | [-11.2169, -10.5128] | 6972 / 3811 |
| D1_minus_A | mask75 | -34.0505 | -28.5184 | [-29.7387, -27.3290] | 6972 / 3811 |
| D1_minus_A | coverage | 0.8617 | 1.1002 | [0.8227, 1.3592] | 6972 / 3811 |
| D1_minus_A | auc | -1.8144 | -1.4257 | [-1.5297, -1.3265] | 6972 / 3811 |
| D1_minus_A | fpr | 32.4176 | 29.0029 | [28.2564, 29.7497] | 6972 / 3811 |
| D_minus_D1 | iou | 8.6125 | 7.8364 | [7.5484, 8.1411] | 6972 / 3811 |
| D_minus_D1 | mask75 | 24.0677 | 20.9101 | [19.8105, 22.0750] | 6972 / 3811 |
| D_minus_D1 | coverage | -0.4925 | -0.6178 | [-0.8496, -0.3665] | 6972 / 3811 |
| D_minus_D1 | auc | 1.2958 | 1.0436 | [0.9799, 1.1118] | 6972 / 3811 |
| D_minus_D1 | fpr | -22.0895 | -20.3795 | [-20.9297, -19.8447] | 6972 / 3811 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 93 / 6972 | NA / NA | 1.3339 / 1.1061 | -93 | -1.1061 [-1.3693, -0.8615] |
| D1 | 0 / 0 | 2374 / 6972 | NA / NA | 34.0505 / 28.5184 | -2374 | -28.5184 [-29.7387, -27.3290] |
| D | 0 / 0 | 696 / 6972 | NA / NA | 9.9828 / 7.6083 | -696 | -7.6083 [-8.2937, -6.9303] |

## baseline_success/pyramid/P3

3120 candidates; 1454 eligible images; 3546 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 82.3239 / 82.3237 | 100.0000 / 100.0000 | 94.9471 / 94.9504 | 95.5295 / 95.6094 | 38.2763 / 37.3963 |
| B | 80.4034 / 80.4557 | 84.2949 / 84.7427 | 96.0723 / 96.0725 | 95.6513 / 95.7166 | 47.5700 / 46.2999 |
| D1 | 74.8673 / 75.2915 | 57.0513 / 59.9389 | 95.4798 / 95.2639 | 94.1029 / 94.3301 | 65.0070 / 61.6319 |
| D | 78.6166 / 78.8568 | 75.0321 / 76.3991 | 94.6334 / 94.5900 | 95.0621 / 95.1656 | 48.9186 / 46.9423 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.7868 | -1.5988 | [-1.8820, -1.3293] | 3120 / 1454 |
| D_minus_B | mask75 | -9.2628 | -8.3436 | [-9.8762, -6.7737] | 3120 / 1454 |
| D_minus_B | coverage | -1.4389 | -1.4825 | [-1.7905, -1.2048] | 3120 / 1454 |
| D_minus_B | auc | -0.5891 | -0.5510 | [-0.6948, -0.4197] | 3115 / 1454 |
| D_minus_B | fpr | 1.3486 | 0.6424 | [-0.0572, 1.3645] | 3115 / 1454 |
| D_minus_A | iou | -3.7073 | -3.4669 | [-3.7924, -3.1492] | 3120 / 1454 |
| D_minus_A | mask75 | -24.9679 | -23.6009 | [-25.4379, -21.8243] | 3120 / 1454 |
| D_minus_A | coverage | -0.3136 | -0.3605 | [-0.6609, -0.0857] | 3120 / 1454 |
| D_minus_A | auc | -0.4673 | -0.4437 | [-0.5996, -0.3029] | 3115 / 1454 |
| D_minus_A | fpr | 10.6423 | 9.5461 | [8.7492, 10.3841] | 3115 / 1454 |
| B_minus_A | iou | -1.9205 | -1.8680 | [-1.9998, -1.7362] | 3120 / 1454 |
| B_minus_A | mask75 | -15.7051 | -15.2573 | [-16.8047, -13.7829] | 3120 / 1454 |
| B_minus_A | coverage | 1.1252 | 1.1221 | [1.0513, 1.1943] | 3120 / 1454 |
| B_minus_A | auc | 0.1218 | 0.1072 | [0.0681, 0.1473] | 3115 / 1454 |
| B_minus_A | fpr | 9.2937 | 8.9036 | [8.5042, 9.3054] | 3115 / 1454 |
| D1_minus_A | iou | -7.4566 | -7.0322 | [-7.4618, -6.6260] | 3120 / 1454 |
| D1_minus_A | mask75 | -42.9487 | -40.0611 | [-42.2722, -37.9403] | 3120 / 1454 |
| D1_minus_A | coverage | 0.5327 | 0.3135 | [-0.0608, 0.6655] | 3120 / 1454 |
| D1_minus_A | auc | -1.4266 | -1.2793 | [-1.4866, -1.0892] | 3115 / 1454 |
| D1_minus_A | fpr | 26.7307 | 24.2356 | [23.0830, 25.3715] | 3115 / 1454 |
| D_minus_D1 | iou | 3.7493 | 3.5654 | [3.1819, 3.9546] | 3120 / 1454 |
| D_minus_D1 | mask75 | 17.9808 | 16.4602 | [14.5518, 18.3862] | 3120 / 1454 |
| D_minus_D1 | coverage | -0.8464 | -0.6740 | [-1.0887, -0.2495] | 3120 / 1454 |
| D_minus_D1 | auc | 0.9593 | 0.8355 | [0.7304, 0.9464] | 3115 / 1454 |
| D_minus_D1 | fpr | -16.0884 | -14.6895 | [-15.5261, -13.8670] | 3115 / 1454 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 490 / 3120 | NA / NA | 15.7051 / 15.2573 | -490 | -15.2573 [-16.8047, -13.7829] |
| D1 | 0 / 0 | 1340 / 3120 | NA / NA | 42.9487 / 40.0611 | -1340 | -40.0611 [-42.2722, -37.9403] |
| D | 0 / 0 | 779 / 3120 | NA / NA | 24.9679 / 23.6009 | -779 | -23.6009 [-25.4379, -21.8243] |

## baseline_success/pyramid/P4

8836 candidates; 2907 eligible images; 2093 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 85.8282 / 85.9798 | 100.0000 / 100.0000 | 94.9612 / 95.1781 | 97.6852 / 97.7495 | 18.3590 / 18.6386 |
| B | 84.9278 / 85.2119 | 94.1829 / 94.9178 | 96.0547 / 96.1143 | 97.7411 / 97.8118 | 22.2458 / 22.0276 |
| D1 | 75.4059 / 77.8872 | 60.0837 / 68.8656 | 95.8382 / 95.7020 | 95.5528 / 96.1323 | 47.2637 / 41.2659 |
| D | 81.3952 / 82.7404 | 81.0774 / 86.2886 | 94.8263 / 94.9051 | 97.0617 / 97.2805 | 28.7057 / 26.0999 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.5325 | -2.4716 | [-2.6744, -2.2717] | 8836 / 2907 |
| D_minus_B | mask75 | -13.1055 | -8.6291 | [-9.4981, -7.7674] | 8836 / 2907 |
| D_minus_B | coverage | -1.2283 | -1.2092 | [-1.3933, -1.0384] | 8836 / 2907 |
| D_minus_B | auc | -0.6793 | -0.5312 | [-0.6054, -0.4646] | 8834 / 2907 |
| D_minus_B | fpr | 6.4599 | 4.0723 | [3.6195, 4.5215] | 8834 / 2907 |
| D_minus_A | iou | -4.4329 | -3.2395 | [-3.4571, -3.0303] | 8836 / 2907 |
| D_minus_A | mask75 | -18.9226 | -13.7114 | [-14.6257, -12.8099] | 8836 / 2907 |
| D_minus_A | coverage | -0.1349 | -0.2730 | [-0.4572, -0.0962] | 8836 / 2907 |
| D_minus_A | auc | -0.6235 | -0.4689 | [-0.5427, -0.4016] | 8834 / 2907 |
| D_minus_A | fpr | 10.3467 | 7.4613 | [6.9770, 7.9449] | 8834 / 2907 |
| B_minus_A | iou | -0.9004 | -0.7679 | [-0.8336, -0.7026] | 8836 / 2907 |
| B_minus_A | mask75 | -5.8171 | -5.0822 | [-5.6817, -4.5319] | 8836 / 2907 |
| B_minus_A | coverage | 1.0935 | 0.9362 | [0.8938, 0.9811] | 8836 / 2907 |
| B_minus_A | auc | 0.0558 | 0.0623 | [0.0476, 0.0780] | 8834 / 2907 |
| B_minus_A | fpr | 3.8867 | 3.3890 | [3.2288, 3.5566] | 8834 / 2907 |
| D1_minus_A | iou | -10.4223 | -8.0926 | [-8.4356, -7.7694] | 8836 / 2907 |
| D1_minus_A | mask75 | -39.9163 | -31.1344 | [-32.4555, -29.8745] | 8836 / 2907 |
| D1_minus_A | coverage | 0.8769 | 0.5239 | [0.2754, 0.7443] | 8836 / 2907 |
| D1_minus_A | auc | -2.1325 | -1.6171 | [-1.7654, -1.4832] | 8834 / 2907 |
| D1_minus_A | fpr | 28.9046 | 22.6274 | [21.8193, 23.4444] | 8834 / 2907 |
| D_minus_D1 | iou | 5.9893 | 4.8532 | [4.6016, 5.1103] | 8836 / 2907 |
| D_minus_D1 | mask75 | 20.9937 | 17.4230 | [16.2703, 18.5862] | 8836 / 2907 |
| D_minus_D1 | coverage | -1.0118 | -0.7969 | [-0.9973, -0.5819] | 8836 / 2907 |
| D_minus_D1 | auc | 1.5090 | 1.1482 | [1.0506, 1.2505] | 8834 / 2907 |
| D_minus_D1 | fpr | -18.5580 | -15.1660 | [-15.7042, -14.6302] | 8834 / 2907 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 514 / 8836 | NA / NA | 5.8171 / 5.0822 | -514 | -5.0822 [-5.6817, -4.5319] |
| D1 | 0 / 0 | 3527 / 8836 | NA / NA | 39.9163 / 31.1344 | -3527 | -31.1344 [-32.4555, -29.8745] |
| D | 0 / 0 | 1672 / 8836 | NA / NA | 18.9226 / 13.7114 | -1672 | -13.7114 [-14.6257, -12.8099] |

## baseline_success/pyramid/P5

8251 candidates; 4120 eligible images; 880 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 89.2828 / 89.5939 | 100.0000 / 100.0000 | 94.8655 / 95.0704 | 98.4411 / 98.4878 | 9.8479 / 9.8769 |
| B | 89.0618 / 89.4552 | 97.9760 / 98.4599 | 95.4598 / 95.4877 | 98.4833 / 98.5322 | 11.1314 / 10.7955 |
| D1 | 76.7254 / 78.8076 | 64.2104 / 70.4884 | 95.5725 / 95.9406 | 96.6882 / 97.1005 | 39.8638 / 36.7559 |
| D | 85.2952 / 86.5034 | 88.4378 / 91.4717 | 95.0294 / 95.3401 | 97.9621 / 98.1228 | 19.0671 / 17.6334 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7666 | -2.9518 | [-3.1227, -2.7923] | 8251 / 4120 |
| D_minus_B | mask75 | -9.5382 | -6.9883 | [-7.6108, -6.3967] | 8251 / 4120 |
| D_minus_B | coverage | -0.4304 | -0.1476 | [-0.2846, -0.0182] | 8251 / 4120 |
| D_minus_B | auc | -0.5212 | -0.4094 | [-0.4574, -0.3641] | 8251 / 4120 |
| D_minus_B | fpr | 7.9357 | 6.8380 | [6.4981, 7.1950] | 8251 / 4120 |
| D_minus_A | iou | -3.9876 | -3.0905 | [-3.2700, -2.9209] | 8251 / 4120 |
| D_minus_A | mask75 | -11.5622 | -8.5283 | [-9.1959, -7.8887] | 8251 / 4120 |
| D_minus_A | coverage | 0.1639 | 0.2697 | [0.1323, 0.4000] | 8251 / 4120 |
| D_minus_A | auc | -0.4790 | -0.3650 | [-0.4161, -0.3175] | 8251 / 4120 |
| D_minus_A | fpr | 9.2192 | 7.7565 | [7.3949, 8.1470] | 8251 / 4120 |
| B_minus_A | iou | -0.2209 | -0.1387 | [-0.1827, -0.0979] | 8251 / 4120 |
| B_minus_A | mask75 | -2.0240 | -1.5401 | [-1.8207, -1.2760] | 8251 / 4120 |
| B_minus_A | coverage | 0.5943 | 0.4173 | [0.3847, 0.4492] | 8251 / 4120 |
| B_minus_A | auc | 0.0421 | 0.0443 | [0.0263, 0.0594] | 8251 / 4120 |
| B_minus_A | fpr | 1.2836 | 0.9185 | [0.8260, 1.0126] | 8251 / 4120 |
| D1_minus_A | iou | -12.5573 | -10.7863 | [-11.1312, -10.4528] | 8251 / 4120 |
| D1_minus_A | mask75 | -35.7896 | -29.5116 | [-30.6770, -28.3780] | 8251 / 4120 |
| D1_minus_A | coverage | 0.7071 | 0.8702 | [0.6204, 1.1047] | 8251 / 4120 |
| D1_minus_A | auc | -1.7529 | -1.3873 | [-1.4831, -1.2984] | 8251 / 4120 |
| D1_minus_A | fpr | 30.0159 | 26.8789 | [26.1973, 27.5856] | 8251 / 4120 |
| D_minus_D1 | iou | 8.5698 | 7.6958 | [7.4201, 7.9786] | 8251 / 4120 |
| D_minus_D1 | mask75 | 24.2274 | 20.9832 | [19.9312, 22.0208] | 8251 / 4120 |
| D_minus_D1 | coverage | -0.5432 | -0.6005 | [-0.8176, -0.3734] | 8251 / 4120 |
| D_minus_D1 | auc | 1.2739 | 1.0223 | [0.9605, 1.0890] | 8251 / 4120 |
| D_minus_D1 | fpr | -20.7967 | -19.1224 | [-19.6331, -18.6300] | 8251 / 4120 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 167 / 8251 | NA / NA | 2.0240 / 1.5401 | -167 | -1.5401 [-1.8207, -1.2760] |
| D1 | 0 / 0 | 2953 / 8251 | NA / NA | 35.7896 / 29.5116 | -2953 | -29.5116 [-30.6770, -28.3780] |
| D | 0 / 0 | 954 / 8251 | NA / NA | 11.5622 / 8.5283 | -954 | -8.5283 [-9.1959, -7.8887] |

## Timing and limitations

{
  "supplied_timing_scope": "not specified; supplied row observations only",
  "direct_ms": {
    "candidate_mean": 0.7761803917312959,
    "image_macro": 0.7830361821898053,
    "valid_candidates": 36213,
    "valid_images": 4952,
    "median_ms": 0.7528793066740036,
    "p95_ms": 0.9128522127866745
  },
  "quality_ms": {
    "candidate_mean": 9.148856597597046,
    "image_macro": 9.181491645052523,
    "valid_candidates": 36213,
    "valid_images": 4952,
    "median_ms": 9.129862301051617,
    "p95_ms": 11.08179185539484
  }
}

Timing is summarized in the supplied unit; no image-total or dataset-total latency is inferred from repeated row values. The unpopulated planned-image list cannot by itself distinguish zero matched candidates from an interrupted evaluation; completion must be established by the run and metadata.

Metadata: `{"split": "final", "planned_images": 5000, "evaluated_images_requested": 5000, "image_ids": [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818, 1993, 2006, 2149, 2153, 2157, 2261, 2299, 2431, 2473, 2532, 2587, 2592, 2685, 2923, 3156, 3255, 3501, 3553, 3661, 3845, 3934, 4134, 4395, 4495, 4765, 4795, 5001, 5037, 5060, 5193, 5477, 5503, 5529, 5586, 5600, 5992, 6012, 6040, 6213, 6460, 6471, 6614, 6723, 6763, 6771, 6818, 6894, 6954, 7088, 7108, 7278, 7281, 7386, 7511, 7574, 7784, 7795, 7816, 7818, 7888, 7977, 7991, 8021, 8211, 8277, 8532, 8629, 8690, 8762, 8844, 8899, 9378, 9400, 9448, 9483, 9590, 9769, 9772, 9891, 9914, 10092, 10363, 10583, 10707, 10764, 10977, 10995, 11051, 11122, 11149, 11197, 11511, 11615, 11699, 11760, 11813, 12062, 12120, 12280, 12576, 12639, 12667, 12670, 12748, 13004, 13177, 13201, 13291, 13348, 13546, 13597, 13659, 13729, 13774, 13923, 14007, 14038, 14226, 14380, 14439, 14473, 14831, 14888, 15079, 15254, 15272, 15278, 15335, 15338, 15440, 15497, 15517, 15597, 15660, 15746, 15751, 15956, 16010, 16228, 16249, 16439, 16451, 16502, 16598, 16958, 17029, 17031, 17115, 17178, 17182, 17207, 17379, 17436, 17627, 17714, 17899, 17905, 17959, 18150, 18193, 18380, 18491, 18519, 18575, 18737, 18770, 18833, 18837, 19042, 19109, 19221, 19402, 19432, 19742, 19786, 19924, 20059, 20107, 20247, 20333, 20553, 20571, 20992, 21167, 21465, 21503, 21604, 21839, 21879, 21903, 22192, 22371, 22396, 22479, 22589, 22623, 22705, 22755, 22892, 22935, 22969, 23023, 23034, 23126, 23230, 23272, 23359, 23666, 23751, 23781, 23899, 23937, 24021, 24027, 24144, 24243, 24567, 24610, 24919, 25057, 25096, 25139, 25181, 25228, 25386, 25393, 25394, 25424, 25560, 25593, 25603, 25986, 26204, 26465, 26564, 26690, 26926, 26941, 27186, 27620, 27696, 27768, 27932, 27972, 27982, 28285, 28449, 28452, 28809, 28993, 29187, 29393, 29397, 29596, 29640, 29675, 29984, 30213, 30494, 30504, 30675, 30785, 30828, 31050, 31093, 31118, 31217, 31248, 31269, 31296, 31322, 31620, 31735, 31749, 31817, 32038, 32081, 32285, 32334, 32570, 32610, 32735, 32811, 32817, 32861, 32887, 32901, 32941, 33005, 33104, 33109, 33114, 33221, 33368, 33638, 33707, 33759, 33854, 34071, 34139, 34205, 34257, 34417, 34452, 34760, 34873, 35062, 35197, 35279, 35326, 35682, 35770, 35963, 36494, 36539, 36660, 36678, 36844, 36861, 36936, 37670, 37689, 37740, 37751, 37777, 37988, 38048, 38070, 38118, 38210, 38576, 38678, 38825, 38829, 39405, 39477, 39480, 39484, 39551, 39670, 39769, 39785, 39914, 39951, 39956, 40036, 40083, 40471, 40757, 41488, 41633, 41635, 41872, 41888, 41990, 42070, 42102, 42178, 42276, 42296, 42528, 42563, 42628, 42888, 42889, 43314, 43435, 43581, 43737, 43816, 44068, 44195, 44260, 44279, 44590, 44652, 44699, 44877, 45070, 45090, 45229, 45472, 45550, 45596, 45728, 46031, 46048, 46252, 46378, 46463, 46497, 46804, 46872, 47010, 47112, 47121, 47571, 47585, 47740, 47769, 47801, 47819, 47828, 48153, 48396, 48504, 48555, 48564, 48924, 49060, 49091, 49259, 49269, 49759, 49761, 49810, 50006, 50145, 50149, 50165, 50326, 50331, 50380, 50638, 50679, 50811, 50828, 50844, 50896, 50943, 51008, 51309, 51314, 51326, 51598, 51610, 51712, 51738, 51938, 51961, 51976, 52007, 52017, 52412, 52413, 52462, 52507, 52565, 52591, 52891, 52996, 53505, 53529, 53624, 53626, 53909, 53994, 54123, 54164, 54592, 54593, 54605, 54628, 54654, 54931, 54967, 55002, 55022, 55072, 55150, 55167, 55299, 55528, 55950, 56127, 56288, 56344, 56350, 56545, 57027, 57149, 57150, 57232, 57238, 57244, 57597, 57672, 57725, 57760, 58029, 58111, 58350, 58384, 58393, 58539, 58636, 58655, 58705, 59044, 59386, 59598, 59635, 59920, 60052, 60090, 60102, 60347, 60363, 60449, 60507, 60770, 60823, 60835, 60855, 60886, 60899, 60932, 61108, 61171, 61268, 61333, 61418, 61471, 61584, 61658, 61747, 61960, 62025, 62353, 62355, 62554, 62692, 62808, 63047, 63154, 63552, 63602, 63740, 63965, 64084, 64359, 64462, 64495, 64499, 64523, 64574, 64718, 64868, 64898, 65074, 65288, 65350, 65455, 65485, 65736, 65798, 66038, 66135, 66231, 66523, 66561, 66635, 66706, 66771, 66817, 66841, 66886, 66926, 67180, 67213, 67310, 67315, 67406, 67534, 67616, 67896, 68078, 68093, 68286, 68387, 68409, 68628, 68765, 68833, 68933, 69106, 69138, 69213, 69224, 69356, 69795, 70048, 70158, 70229, 70254, 70739, 70774, 71226, 71451, 71711, 71756, 71877, 71938, 72281, 72795, 72813, 72852, 73118, 73153, 73326, 73533, 73702, 73946, 74058, 74092, 74200, 74209, 74256, 74457, 74646, 74733, 74860, 75393, 75456, 75612, 76211, 76261, 76416, 76417, 76468, 76547, 76625, 76731, 77396, 77460, 77595, 78032, 78170, 78266, 78404, 78420, 78426, 78565, 78748, 78823, 78843, 78915, 78959, 79014, 79031, 79034, 79144, 79188, 79229, 79408, 79565, 79588, 79651, 79837, 79969, 80022, 80057, 80153, 80273, 80274, 80340, 80413, 80659, 80666, 80671, 80932, 80949, 81061, 81394, 81594, 81738, 81766, 81988, 82085, 82180, 82688, 82696, 82715, 82765, 82807, 82812, 82821, 82846, 82986, 83113, 83172, 83531, 83540, 84031, 84170, 84241, 84270, 84362, 84431, 84477, 84492, 84650, 84664, 84674, 84752, 85089, 85157, 85195, 85329, 85376, 85478, 85576, 85665, 85682, 85772, 85823, 85911, 86220, 86483, 86582, 86755, 86956, 87038, 87144, 87244, 87470, 87476, 87742, 87875, 88040, 88218, 88250, 88265, 88269, 88345, 88432, 88462, 88485, 88848, 88951, 88970, 89045, 89078, 89271, 89296, 89556, 89648, 89670, 89697, 89761, 89880, 90003, 90062, 90108, 90155, 90208, 90284, 90631, 90891, 90956, 91406, 91495, 91500, 91615, 91619, 91654, 91779, 91921, 92053, 92091, 92124, 92177, 92416, 92660, 92839, 92939, 93154, 93261, 93353, 93437, 93717, 93965, 94157, 94185, 94326, 94336, 94614, 94751, 94852, 94871, 94944, 95069, 95155, 95707, 95786, 95843, 95862, 95899, 96001, 96427, 96493, 96549, 96825, 96960, 97022, 97230, 97278, 97337, 97585, 97679, 97924, 97988, 97994, 98018, 98261, 98287, 98392, 98497, 98520, 98633, 98716, 98839, 98853, 99024, 99039, 99053, 99054, 99114, 99182, 99242, 99428, 99810, 100238, 100274, 100283, 100428, 100489, 100510, 100582, 100624, 100723, 101022, 101068, 101420, 101762, 101780, 101787, 101884, 102331, 102356, 102411, 102644, 102707, 102805, 102820, 103548, 103585, 103723, 104119, 104198, 104424, 104455, 104572, 104603, 104612, 104619, 104666, 104669, 104782, 104803, 105014, 105249, 105264, 105335, 105455, 105912, 105923, 106048, 106235, 106266, 106281, 106330, 106389, 106563, 106757, 106881, 106912, 107087, 107094, 107226, 107339, 107554, 107851, 108026, 108244, 108253, 108440, 108495, 108503, 108864, 109055, 109118, 109313, 109441, 109798, 109827, 109900, 109916, 109976, 109992, 110042, 110211, 110282, 110359, 110449, 110638, 110721, 110784, 110884, 110972, 110999, 111036, 111086, 111179, 111207, 111609, 111951, 112110, 112298, 112378, 112626, 112634, 112798, 112997, 113051, 113235, 113354, 113403, 113589, 113720, 113867, 114049, 114770, 114871, 114884, 114907, 115118, 115245, 115870, 115885, 115898, 115946, 116068, 116206, 116208, 116362, 116439, 116479, 116589, 116825, 117197, 117374, 117425, 117492, 117525, 117645, 117719, 117744, 117908, 117914, 118209, 118367, 118405, 118515, 118594, 118921, 119038, 119088, 119233, 119365, 119445, 119452, 119516, 119641, 119677, 119828, 119911, 119995, 120420, 120572, 120584, 120777, 120853, 121031, 121153, 121242, 121417, 121497, 121506, 121586, 121591, 121673, 121744, 122046, 122166, 122217, 122606, 122672, 122745, 122927, 122962, 122969, 123131, 123213, 123321, 123480, 123585, 123633, 124277, 124442, 124636, 124659, 124798, 124975, 125062, 125072, 125129, 125211, 125245, 125257, 125405, 125472, 125572, 125778, 125806, 125850, 125936, 125952, 126107, 126110, 126137, 126216, 126226, 126592, 127092, 127135, 127182, 127263, 127270, 127394, 127476, 127494, 127517, 127530, 127624, 127660, 127955, 127987, 128051, 128112, 128148, 128372, 128476, 128598, 128654, 128658, 128675, 128699, 128748, 129054, 129062, 129113, 129135, 129322, 129416, 129492, 129756, 129812, 129945, 130386, 130465, 130566, 130579, 130586, 130599, 130613, 130699, 130826, 131131, 131138, 131273, 131379, 131386, 131431, 131444, 131556, 131938, 132116, 132329, 132375, 132408, 132544, 132587, 132622, 132703, 132796, 132931, 133000, 133087, 133233, 133244, 133343, 133418, 133567, 133631, 133645, 133778, 133819, 133969, 134034, 134096, 134112, 134322, 134689, 134722, 134856, 134882, 134886, 135410, 135561, 135604, 135670, 135673, 135872, 135890, 135902, 136033, 136334, 136355, 136466, 136600, 136633, 136715, 136772, 136915, 137106, 137246, 137294, 137576, 137727, 137950, 138115, 138241, 138492, 138550, 138639, 138819, 138856, 138954, 138979, 139077, 139099, 139260, 139684, 139871, 139872, 139883, 140076, 140203, 140270, 140286, 140420, 140439, 140556, 140583, 140640, 140658, 140840, 140929, 140987, 141328, 141597, 141671, 141821, 142092, 142238, 142324, 142472, 142585, 142620, 142790, 142971, 143068, 143556, 143572, 143931, 143961, 143998, 144003, 144114, 144300, 144333, 144706, 144784, 144798, 144932, 144984, 145020, 145591, 145597, 145620, 145665, 145781, 146155, 146358, 146363, 146457, 146489, 146498, 146667, 146825, 146831, 147205, 147223, 147338, 147415, 147498, 147518, 147725, 147729, 147740, 147745, 148508, 148620, 148662, 148707, 148719, 148730, 148739, 148783, 148957, 148999, 149222, 149375, 149406, 149568, 149622, 149770, 150224, 150265, 150417, 150638, 150649, 150726, 150930, 151000, 151051, 151480, 151516, 151629, 151657, 151662, 151820, 151857, 151938, 151962, 152120, 152214, 152465, 152686, 152740, 152771, 152870, 153011, 153217, 153229, 153299, 153343, 153510, 153527, 153529, 153568, 153632, 153669, 153782, 153797, 154000, 154004, 154087, 154213, 154339, 154358, 154425, 154431, 154644, 154705, 154718, 154947, 155051, 155145, 155154, 155179, 155291, 155341, 155443, 155451, 155571, 156071, 156076, 156278, 156292, 156372, 156643, 156924, 157046, 157098, 157124, 157138, 157213, 157365, 157390, 157418, 157601, 157756, 157767, 157807, 157847, 157928, 158227, 158548, 158660, 158744, 158945, 158956, 159112, 159282, 159311, 159399, 159458, 159684, 159791, 159977, 160012, 160556, 160666, 160728, 160772, 160864, 161008, 161032, 161044, 161128, 161397, 161609, 161642, 161781, 161799, 161820, 161861, 161875, 161879, 161925, 161978, 162035, 162092, 162130, 162366, 162415, 162543, 162581, 162732, 162858, 163057, 163117, 163118, 163155, 163257, 163258, 163290, 163314, 163562, 163611, 163640, 163682, 163746, 163951, 164115, 164363, 164602, 164637, 164883, 164885, 164969, 165039, 165257, 165336, 165351, 165500, 165518, 165681, 165713, 165831, 166165, 166166, 166259, 166277, 166287, 166391, 166426, 166478, 166509, 166521, 166563, 166642, 166664, 166747, 166768, 166918, 167067, 167122, 167128, 167159, 167240, 167353, 167486, 167540, 167572, 167898, 167902, 168330, 168337, 168458, 168593, 168619, 168883, 168974, 169076, 169169, 169356, 169996, 170099, 170116, 170191, 170278, 170474, 170545, 170595, 170613, 170670, 170739, 170893, 170955, 171050, 171190, 171298, 171382, 171611, 171740, 171757, 171788, 172083, 172330, 172396, 172547, 172571, 172595, 172617, 172648, 172649, 172856, 172877, 172935, 172946, 172977, 173004, 173008, 173033, 173044, 173057, 173091, 173183, 173302, 173371, 173383, 173799, 173830, 174004, 174018, 174123, 174231, 174371, 174482, 175251, 175364, 175387, 175438, 175443, 175535, 176037, 176232, 176446, 176606, 176634, 176701, 176778, 176799, 176847, 176857, 176901, 177015, 177065, 177213, 177357, 177383, 177489, 177539, 177714, 177861, 177893, 177934, 177935, 178028, 178469, 178618, 178744, 178982, 179112, 179141, 179174, 179214, 179265, 179285, 179392, 179487, 179642, 179653, 179765, 179898, 180011, 180101, 180135, 180188, 180296, 180383, 180487, 180560, 180751, 180792, 180798, 180878, 181303, 181421, 181499, 181542, 181666, 181753, 181796, 181816, 181859, 181969, 182021, 182155, 182162, 182202, 182417, 182441, 182611, 182805, 182923, 183049, 183104, 183127, 183246, 183391, 183437, 183500, 183648, 183675, 183709, 183716, 183965, 184321, 184324, 184338, 184384, 184400, 184611, 184762, 184791, 184978, 185157, 185250, 185292, 185409, 185472, 185473, 185599, 185802, 185890, 185950, 186042, 186282, 186296, 186345, 186422, 186449, 186624, 186632, 186637, 186873, 186929, 186938, 186980, 187055, 187144, 187236, 187243, 187249, 187271, 187362, 187513, 187585, 187734, 187745, 187990, 188296, 188439, 188465, 188592, 188689, 188906, 189078, 189213, 189226, 189310, 189436, 189451, 189475, 189698, 189752, 189775, 189806, 189820, 189828, 190007, 190140, 190236, 190307, 190637, 190648, 190676, 190753, 190756, 190841, 190853, 190923, 191013, 191288, 191471, 191580, 191614, 191672, 191761, 191845, 192047, 192191, 192607, 192670, 192699, 192716, 192871, 192904, 192964, 193162, 193181, 193245, 193348, 193429, 193494, 193674, 193717, 193743, 193884, 193926, 194216, 194471, 194506, 194716, 194724, 194746, 194832, 194875, 194940, 195045, 195165, 195754, 195842, 195918, 196009, 196141, 196185, 196442, 196754, 196759, 196843, 197004, 197022, 197388, 197528, 197658, 197796, 197870, 198489, 198510, 198641, 198805, 198915, 198928, 198960, 199055, 199236, 199310, 199395, 199442, 199551, 199681, 199771, 199977, 200152, 200162, 200252, 200421, 200667, 200839, 200961, 201025, 201072, 201148, 201418, 201426, 201646, 201676, 201775, 201934, 202001, 202228, 202339, 202445, 203095, 203294, 203317, 203389, 203488, 203546, 203580, 203629, 203639, 203864, 203931, 204186, 204329, 204871, 205105, 205282, 205289, 205324, 205333, 205401, 205514, 205542, 205647, 205776, 205834, 206025, 206027, 206135, 206218, 206271, 206411, 206487, 206579, 206831, 206838, 206994, 207306, 207538, 207585, 207728, 207844, 208208, 208363, 208423, 208901, 209142, 209222, 209530, 209613, 209747, 209753, 209757, 209829, 209972, 210030, 210032, 210099, 210230, 210273, 210299, 210388, 210394, 210502, 210520, 210708, 210789, 210855, 210915, 211042, 211069, 211120, 211674, 211825, 212072, 212166, 212226, 212453, 212559, 212573, 212800, 212895, 213033, 213035, 213086, 213171, 213224, 213255, 213422, 213445, 213547, 213593, 213605, 213816, 213830, 213935, 214192, 214200, 214205, 214224, 214539, 214703, 214720, 214753, 214869, 215072, 215114, 215245, 215259, 215644, 215723, 215778, 216277, 216296, 216419, 216497, 216516, 216636, 216739, 217060, 217219, 217285, 217400, 217425, 217614, 217753, 217872, 217948, 217957, 218091, 218249, 218362, 218424, 218439, 218997, 219271, 219283, 219440, 219485, 219578, 220310, 220584, 220732, 220764, 220858, 221017, 221155, 221213, 221281, 221291, 221502, 221693, 221708, 221754, 221872, 222094, 222118, 222235, 222299, 222317, 222455, 222458, 222559, 222735, 222825, 222863, 222991, 223090, 223130, 223182, 223188, 223738, 223747, 223789, 223955, 223959, 224051, 224093, 224119, 224200, 224222, 224337, 224664, 224675, 224724, 224807, 225184, 225405, 225532, 225670, 225757, 225946, 226058, 226111, 226130, 226147, 226154, 226171, 226408, 226417, 226592, 226662, 226802, 226883, 226903, 226984, 227044, 227187, 227399, 227478, 227482, 227491, 227511, 227686, 227765, 227898, 227985, 228144, 228214, 228436, 228771, 228942, 228981, 229111, 229216, 229221, 229311, 229358, 229553, 229601, 229659, 229747, 229753, 229849, 229858, 229948, 229997, 230008, 230166, 230362, 230450, 230819, 230983, 230993, 231088, 231097, 231125, 231169, 231237, 231339, 231508, 231527, 231549, 231580, 231747, 231822, 231831, 231879, 232088, 232244, 232348, 232489, 232538, 232563, 232646, 232649, 232684, 232692, 233033, 233139, 233238, 233370, 233567, 233727, 233771, 233825, 234366, 234413, 234526, 234607, 234660, 234757, 234779, 234807, 235057, 235064, 235241, 235252, 235399, 235778, 235784, 235836, 235857, 236166, 236308, 236412, 236426, 236592, 236599, 236690, 236721, 236730, 236784, 236845, 236914, 237071, 237118, 237316, 237517, 237864, 237928, 237984, 238013, 238039, 238410, 238866, 239041, 239274, 239318, 239347, 239537, 239627, 239717, 239773, 239843, 239857, 240023, 240049, 240250, 240754, 240767, 240940, 241297, 241319, 241326, 241602, 241668, 241677, 242060, 242287, 242411, 242678, 242724, 242934, 242946, 243034, 243075, 243148, 243199, 243204, 243344, 243495, 243626, 243867, 243989, 244019, 244099, 244181, 244379, 244411, 244496, 244592, 244750, 244833, 245026, 245102, 245173, 245311, 245320, 245448, 245513, 245576, 245651, 245764, 245915, 246308, 246436, 246454, 246522, 246883, 246963, 246968, 247806, 247838, 247917, 248111, 248112, 248284, 248314, 248334, 248400, 248616, 248631, 248752, 248810, 248980, 249025, 249129, 249180, 249219, 249550, 249643, 249786, 250127, 250137, 250205, 250282, 250619, 250758, 250766, 250901, 251065, 251119, 251140, 251537, 251572, 251824, 252216, 252219, 252294, 252332, 252507, 252559, 252701, 252716, 252776, 253002, 253386, 253433, 253452, 253695, 253742, 253819, 253835, 254016, 254368, 254516, 254814, 255165, 255401, 255483, 255536, 255664, 255718, 255747, 255749, 255824, 255912, 255917, 255965, 256192, 256195, 256407, 256518, 256775, 256868, 256916, 256941, 257084, 257169, 257370, 257478, 257566, 257624, 257865, 257896, 258388, 258541, 258793, 258883, 258911, 259097, 259382, 259571, 259597, 259625, 259640, 259690, 259830, 259854, 260105, 260106, 260261, 260266, 260470, 260657, 260925, 261036, 261061, 261097, 261116, 261161, 261318, 261535, 261706, 261712, 261732, 261796, 261888, 261982, 262048, 262227, 262440, 262487, 262587, 262631, 262682, 262895, 262938, 263068, 263299, 263403, 263425, 263463, 263474, 263594, 263644, 263679, 263796, 263860, 263966, 263969, 264335, 264441, 264535, 264968, 265108, 265518, 265777, 265816, 266082, 266206, 266400, 266409, 266768, 266892, 266981, 267169, 267191, 267300, 267351, 267434, 267537, 267670, 267903, 267933, 267940, 267946, 268000, 268375, 268378, 268729, 268831, 268996, 269113, 269121, 269196, 269314, 269316, 269632, 269682, 269866, 269932, 269942, 270066, 270122, 270244, 270297, 270386, 270402, 270474, 270677, 270705, 270883, 270908, 271116, 271402, 271457, 271471, 271728, 271997, 272049, 272136, 272148, 272212, 272364, 272416, 272566, 273132, 273198, 273232, 273420, 273493, 273551, 273617, 273642, 273711, 273712, 273715, 273760, 274066, 274219, 274272, 274411, 274460, 274687, 274708, 275058, 275198, 275392, 275727, 275749, 275791, 276018, 276024, 276055, 276284, 276285, 276434, 276707, 276720, 276804, 276921, 277005, 277020, 277051, 277197, 277584, 277689, 278006, 278353, 278463, 278705, 278749, 278848, 278973, 279145, 279278, 279541, 279714, 279730, 279769, 279774, 279887, 279927, 280325, 280710, 280779, 280891, 280918, 280930, 281032, 281179, 281409, 281414, 281447, 281687, 281693, 281754, 281759, 281929, 282037, 282046, 282296, 282298, 282912, 283037, 283038, 283070, 283113, 283268, 283318, 283412, 283520, 283717, 283785, 284106, 284279, 284282, 284296, 284445, 284623, 284698, 284725, 284743, 284762, 284764, 284991, 285047, 285349, 285788, 285894, 286182, 286422, 286458, 286503, 286507, 286523, 286553, 286660, 286708, 286849, 286907, 286908, 286994, 287291, 287347, 287527, 287545, 287649, 287667, 287714, 287874, 287959, 288042, 288062, 288391, 288430, 288584, 288685, 288762, 288862, 288882, 289059, 289222, 289229, 289343, 289393, 289415, 289417, 289516, 289586, 289594, 289659, 289702, 289741, 289938, 289960, 289992, 290081, 290163, 290179, 290248, 290293, 290592, 290619, 290768, 290771, 290833, 290843, 291490, 291551, 291619, 291634, 291664, 291791, 291861, 292005, 292024, 292060, 292082, 292155, 292225, 292236, 292330, 292415, 292446, 292456, 292488, 292908, 292997, 293044, 293071, 293200, 293245, 293300, 293324, 293390, 293474, 293625, 293794, 293804, 293858, 294162, 294163, 294350, 294695, 294783, 294831, 294855, 295138, 295231, 295316, 295420, 295478, 295713, 295797, 295809, 296222, 296224, 296231, 296284, 296317, 296634, 296649, 296657, 296969, 297022, 297084, 297085, 297147, 297343, 297353, 297396, 297427, 297562, 297578, 297595, 297681, 297698, 297830, 298251, 298396, 298697, 298738, 298904, 298994, 299355, 299553, 299609, 299720, 299887, 300039, 300155, 300233, 300276, 300341, 300659, 300842, 300913, 301061, 301135, 301376, 301421, 301563, 301718, 301867, 301981, 302030, 302107, 302165, 302452, 302536, 302760, 302882, 302990, 303305, 303499, 303566, 303653, 303713, 303818, 303863, 303893, 303908, 304180, 304291, 304365, 304396, 304404, 304545, 304560, 304812, 304817, 304984, 305309, 305317, 305343, 305609, 305695, 306136, 306139, 306437, 306582, 306700, 306733, 306893, 307074, 307145, 307172, 307598, 307658, 308165, 308193, 308328, 308391, 308394, 308430, 308466, 308476, 308531, 308545, 308587, 308631, 308753, 308793, 308799, 309173, 309391, 309452, 309467, 309484, 309495, 309655, 309678, 309938, 309964, 310072, 310200, 310622, 310862, 310980, 311002, 311081, 311180, 311190, 311295, 311303, 311392, 311394, 311518, 311789, 311883, 311909, 311928, 311950, 312192, 312213, 312237, 312263, 312278, 312340, 312406, 312421, 312489, 312549, 312552, 312586, 312720, 313034, 313130, 313182, 313454, 313562, 313588, 313783, 314034, 314177, 314182, 314251, 314264, 314294, 314541, 314709, 314914, 315001, 315187, 315219, 315257, 315450, 315492, 316015, 316054, 316404, 316666, 317024, 317433, 317999, 318080, 318114, 318138, 318238, 318455, 318908, 319100, 319184, 319369, 319534, 319607, 319617, 319696, 319721, 319935, 320232, 320425, 320490, 320554, 320632, 320642, 320664, 320696, 320706, 320743, 321118, 321214, 321333, 321557, 321790, 321887, 322163, 322211, 322352, 322429, 322574, 322610, 322724, 322829, 322844, 322864, 322895, 322944, 322959, 322968, 323151, 323202, 323263, 323355, 323496, 323571, 323709, 323751, 323799, 323828, 323895, 324158, 324258, 324614, 324715, 324818, 324927, 325031, 325114, 325306, 325347, 325483, 325527, 325838, 325991, 326082, 326128, 326174, 326248, 326462, 326541, 326542, 326627, 326970, 327306, 327592, 327601, 327605, 327617, 327701, 327769, 327780, 327890, 328030, 328117, 328238, 328286, 328337, 328430, 328601, 328683, 328959, 329041, 329080, 329219, 329319, 329323, 329447, 329455, 329456, 329542, 329614, 329827, 330369, 330396, 330554, 330790, 330818, 331075, 331280, 331317, 331352, 331569, 331604, 331799, 331817, 332318, 332351, 332455, 332570, 332845, 332901, 333069, 333237, 333402, 333697, 333745, 333772, 333956, 334006, 334309, 334371, 334399, 334417, 334483, 334521, 334530, 334555, 334719, 334767, 334977, 335081, 335177, 335328, 335427, 335450, 335529, 335658, 335800, 335954, 336053, 336209, 336232, 336265, 336309, 336356, 336587, 336628, 336658, 337055, 337498, 337987, 338191, 338219, 338304, 338325, 338428, 338532, 338560, 338624, 338625, 338718, 338901, 338905, 338986, 339442, 339823, 339870, 340015, 340175, 340272, 340451, 340697, 340894, 340930, 341058, 341094, 341196, 341469, 341681, 341719, 341828, 341921, 341973, 342006, 342128, 342186, 342295, 342367, 342397, 342971, 343076, 343149, 343218, 343315, 343453, 343466, 343496, 343524, 343561, 343706, 343803, 343934, 343937, 343976, 344029, 344059, 344100, 344268, 344611, 344614, 344621, 344795, 344816, 344888, 344909, 345027, 345252, 345261, 345356, 345361, 345385, 345397, 345466, 345469, 345941, 346232, 346638, 346703, 346707, 346905, 346968, 347163, 347174, 347254, 347265, 347335, 347370, 347456, 347544, 347664, 347693, 347930, 348012, 348045, 348216, 348243, 348481, 348488, 348708, 348881, 349152, 349184, 349302, 349480, 349594, 349678, 349837, 349860, 350002, 350003, 350019, 350023, 350054, 350122, 350148, 350388, 350405, 350488, 350607, 350679, 350833, 351096, 351331, 351362, 351530, 351559, 351589, 351609, 351810, 351823, 352491, 352582, 352584, 352618, 352684, 352760, 352900, 353027, 353051, 353096, 353180, 353518, 353970, 354072, 354307, 354547, 354753, 354829, 355169, 355240, 355257, 355325, 355610, 355677, 355817, 355905, 356094, 356125, 356169, 356248, 356261, 356347, 356387, 356424, 356427, 356428, 356432, 356498, 356505, 356531, 356612, 356968, 357060, 357081, 357238, 357430, 357459, 357501, 357567, 357737, 357742, 357748, 357816, 357888, 357903, 357941, 357978, 358195, 358427, 358525, 358923, 359135, 359219, 359540, 359677, 359781, 359833, 359855, 359937, 360097, 360137, 360325, 360393, 360564, 360661, 360943, 360951, 360960, 361103, 361142, 361147, 361180, 361238, 361268, 361506, 361551, 361571, 361586, 361621, 361730, 361919, 362434, 362520, 362682, 362716, 363072, 363188, 363207, 363461, 363666, 363784, 363840, 363875, 364102, 364126, 364166, 364297, 364322, 364557, 364587, 364636, 364884, 365095, 365098, 365207, 365208, 365385, 365387, 365521, 365642, 365655, 365745, 365766, 365886, 366141, 366178, 366199, 366225, 366611, 366711, 366884, 367082, 367095, 367195, 367228, 367386, 367569, 367680, 367818, 368038, 368212, 368294, 368335, 368456, 368684, 368752, 368900, 368940, 368961, 368982, 369037, 369081, 369310, 369323, 369370, 369442, 369503, 369541, 369675, 369751, 369757, 369771, 369812, 370042, 370208, 370270, 370375, 370478, 370486, 370677, 370711, 370813, 370818, 370900, 370999, 371042, 371472, 371529, 371552, 371677, 371699, 371749, 372203, 372260, 372307, 372317, 372349, 372466, 372577, 372718, 372819, 373315, 373353, 373382, 373705, 374052, 374083, 374369, 374545, 374551, 374727, 374982, 375015, 375078, 375278, 375430, 375469, 375493, 375763, 376093, 376112, 376206, 376264, 376278, 376284, 376307, 376310, 376322, 376365, 376442, 376478, 376625, 376856, 376900, 377000, 377113, 377239, 377368, 377393, 377486, 377497, 377575, 377588, 377635, 377670, 377723, 377814, 377882, 377946, 378099, 378116, 378139, 378244, 378284, 378453, 378454, 378515, 378605, 378673, 378873, 379332, 379441, 379453, 379476, 379533, 379800, 379842, 380203, 380706, 380711, 380913, 381360, 381587, 381639, 381971, 382009, 382030, 382088, 382111, 382122, 382125, 382696, 382734, 382743, 383289, 383337, 383339, 383384, 383386, 383443, 383606, 383621, 383676, 383838, 383842, 383921, 384136, 384350, 384468, 384513, 384527, 384616, 384651, 384661, 384666, 384670, 384808, 384850, 384949, 385029, 385190, 385205, 385719, 385997, 386134, 386210, 386277, 386352, 386457, 386879, 386912, 387098, 387148, 387383, 387387, 387916, 388056, 388215, 388258, 388846, 388903, 388927, 389109, 389197, 389315, 389316, 389381, 389451, 389532, 389566, 389684, 389804, 389812, 389933, 390246, 390301, 390555, 390826, 390902, 391140, 391144, 391290, 391375, 391648, 391722, 392228, 392481, 392722, 392818, 392933, 393014, 393056, 393093, 393115, 393226, 393282, 393469, 393569, 393838, 394199, 394206, 394275, 394328, 394510, 394559, 394611, 394677, 394940, 395180, 395343, 395388, 395575, 395633, 395701, 395801, 395903, 396200, 396205, 396274, 396338, 396518, 396526, 396568, 396580, 396729, 396863, 396903, 397133, 397279, 397303, 397327, 397351, 397354, 397639, 397681, 398028, 398203, 398237, 398377, 398438, 398652, 398742, 398810, 398905, 399205, 399296, 399462, 399560, 399655, 399764, 400044, 400082, 400161, 400367, 400573, 400794, 400803, 400815, 400922, 401244, 401250, 401446, 401862, 401991, 402096, 402118, 402334, 402346, 402433, 402473, 402519, 402615, 402720, 402765, 402774, 402783, 402992, 403122, 403353, 403385, 403565, 403584, 403817, 404128, 404191, 404249, 404479, 404484, 404534, 404568, 404601, 404678, 404805, 404839, 404922, 404923, 405195, 405205, 405249, 405279, 405306, 405432, 405691, 405970, 405972, 406129, 406417, 406570, 406611, 406997, 407002, 407083, 407298, 407403, 407518, 407524, 407574, 407614, 407646, 407650, 407825, 407868, 407943, 407960, 408112, 408120, 408696, 408774, 408830, 409198, 409211, 409268, 409358, 409424, 409475, 409542, 409630, 409867, 410221, 410428, 410456, 410487, 410496, 410510, 410612, 410650, 410712, 410735, 410878, 410880, 410934, 411530, 411665, 411754, 411774, 411817, 411938, 411953, 412240, 412286, 412362, 412531, 412887, 412894, 413247, 413395, 413404, 413552, 413689, 414034, 414133, 414170, 414261, 414340, 414385, 414510, 414638, 414673, 414676, 414795, 415194, 415238, 415536, 415716, 415727, 415741, 415748, 415882, 415990, 416104, 416170, 416256, 416269, 416330, 416343, 416451, 416534, 416745, 416758, 416837, 416885, 416991, 417043, 417085, 417249, 417285, 417465, 417608, 417632, 417779, 417876, 417911, 418062, 418281, 418696, 418959, 418961, 419096, 419098, 419201, 419312, 419379, 419408, 419601, 419653, 419882, 419974, 420069, 420230, 420281, 420472, 420840, 420916, 421060, 421455, 421757, 421834, 421923, 422670, 422706, 422836, 422886, 422998, 423104, 423123, 423229, 423506, 423519, 423617, 423798, 423944, 423971, 424135, 424162, 424349, 424521, 424545, 424551, 424642, 424721, 424776, 424975, 425221, 425226, 425227, 425361, 425390, 425702, 425906, 425925, 426166, 426203, 426241, 426253, 426268, 426297, 426329, 426372, 426376, 426795, 426836, 427034, 427055, 427077, 427160, 427256, 427338, 427500, 427649, 427655, 427997, 428111, 428218, 428280, 428454, 428562, 428867, 429011, 429109, 429281, 429530, 429598, 429623, 429690, 429718, 429761, 430048, 430056, 430073, 430286, 430377, 430871, 430875, 430961, 430973, 431140, 431545, 431568, 431693, 431727, 431848, 431876, 431896, 432085, 432468, 432553, 432898, 433103, 433134, 433192, 433204, 433243, 433374, 433515, 433774, 433915, 433980, 434204, 434230, 434247, 434297, 434459, 434479, 434548, 434996, 435003, 435081, 435205, 435206, 435208, 435299, 435880, 436315, 436551, 436617, 436738, 436883, 437110, 437205, 437239, 437331, 437351, 437392, 437514, 437898, 438017, 438226, 438269, 438304, 438774, 438862, 438876, 438907, 438955, 439180, 439290, 439426, 439522, 439525, 439593, 439623, 439715, 439773, 439854, 439994, 440171, 440184, 440336, 440475, 440507, 440508, 440617, 441247, 441286, 441442, 441468, 441491, 441543, 441553, 441586, 442009, 442161, 442306, 442323, 442456, 442463, 442480, 442661, 442746, 442822, 442836, 442993, 443303, 443426, 443498, 443844, 443969, 444142, 444275, 444879, 445248, 445365, 445439, 445602, 445658, 445675, 445722, 445792, 445834, 445846, 445999, 446005, 446117, 446206, 446207, 446522, 446574, 446651, 446703, 447088, 447169, 447187, 447200, 447313, 447314, 447342, 447465, 447522, 447611, 447789, 447917, 448076, 448256, 448263, 448365, 448410, 448448, 448810, 449190, 449198, 449312, 449406, 449432, 449579, 449603, 449661, 449909, 449996, 450075, 450100, 450202, 450303, 450399, 450439, 450488, 450559, 450686, 450758, 451043, 451084, 451090, 451144, 451150, 451155, 451308, 451435, 451571, 451693, 451714, 451879, 452084, 452122, 452321, 452515, 452784, 452793, 452891, 453001, 453040, 453166, 453302, 453341, 453584, 453634, 453708, 453722, 453841, 453860, 453981, 454067, 454404, 454661, 454750, 454798, 454978, 455085, 455157, 455219, 455267, 455301, 455352, 455448, 455555, 455597, 455624, 455716, 455872, 455937, 455981, 456015, 456143, 456292, 456303, 456394, 456496, 456559, 456662, 456865, 457078, 457262, 457559, 457848, 457884, 458045, 458054, 458109, 458223, 458255, 458325, 458410, 458663, 458702, 458755, 458768, 458790, 458992, 459153, 459195, 459272, 459396, 459437, 459467, 459500, 459634, 459662, 459757, 459809, 459887, 459954, 460147, 460160, 460229, 460333, 460347, 460379, 460494, 460682, 460683, 460841, 460927, 460929, 460967, 461009, 461036, 461275, 461405, 461573, 461751, 462031, 462371, 462576, 462614, 462629, 462643, 462728, 462756, 462904, 463037, 463174, 463199, 463283, 463522, 463527, 463542, 463618, 463647, 463690, 463730, 463802, 463842, 463849, 463918, 464089, 464144, 464251, 464358, 464476, 464522, 464689, 464786, 464824, 464872, 465129, 465179, 465180, 465430, 465549, 465585, 465675, 465718, 465806, 465822, 465836, 466085, 466125, 466156, 466256, 466339, 466416, 466567, 466602, 466835, 466986, 467176, 467315, 467511, 467776, 467848, 468124, 468233, 468245, 468332, 468501, 468505, 468577, 468632, 468925, 468954, 468965, 469067, 469174, 469192, 469246, 469652, 469828, 470121, 470173, 470773, 470779, 470924, 470952, 471023, 471087, 471450, 471567, 471756, 471789, 471869, 471893, 471991, 472030, 472046, 472298, 472375, 472623, 472678, 473015, 473118, 473121, 473219, 473237, 473406, 473821, 473869, 473974, 474021, 474028, 474039, 474078, 474095, 474164, 474167, 474170, 474293, 474344, 474452, 474786, 474854, 474881, 475064, 475150, 475191, 475223, 475365, 475387, 475484, 475572, 475678, 475732, 475779, 475904, 476119, 476215, 476258, 476415, 476491, 476514, 476704, 476770, 476787, 476810, 477118, 477227, 477288, 477441, 477623, 477689, 477805, 477955, 478136, 478286, 478393, 478420, 478474, 478721, 478862, 479030, 479099, 479126, 479155, 479248, 479448, 479596, 479732, 479912, 479953, 480021, 480122, 480212, 480275, 480842, 480936, 480944, 480985, 481159, 481386, 481390, 481404, 481413, 481480, 481567, 481573, 481582, 482100, 482275, 482319, 482436, 482477, 482487, 482585, 482719, 482735, 482800, 482917, 482970, 482978, 483050, 483531, 483667, 483999, 484029, 484296, 484351, 484404, 484415, 484760, 484893, 484978, 485027, 485071, 485130, 485237, 485424, 485480, 485802, 485844, 485895, 485972, 486040, 486046, 486104, 486112, 486438, 486479, 486573, 487583, 488075, 488166, 488251, 488270, 488385, 488592, 488664, 488673, 488710, 488736, 489014, 489046, 489091, 489305, 489339, 489611, 489764, 489842, 489924, 490125, 490171, 490413, 490470, 490515, 490936, 491008, 491071, 491090, 491130, 491213, 491216, 491366, 491464, 491470, 491497, 491613, 491683, 491725, 491757, 491867, 492077, 492110, 492282, 492284, 492362, 492758, 492878, 492905, 492937, 492968, 492992, 493019, 493284, 493286, 493334, 493442, 493566, 493613, 493772, 493799, 493864, 493905, 494188, 494427, 494634, 494759, 494863, 494869, 494913, 495054, 495146, 495448, 495732, 496409, 496571, 496597, 496722, 496854, 496954, 497344, 497568, 497599, 497628, 497867, 498032, 498286, 498463, 498709, 498747, 498807, 498857, 498919, 499031, 499109, 499181, 499266, 499313, 499622, 499768, 499775, 500049, 500211, 500257, 500270, 500423, 500464, 500477, 500478, 500565, 500613, 500663, 500716, 500826, 501005, 501023, 501243, 501368, 501523, 502136, 502168, 502229, 502336, 502347, 502599, 502732, 502737, 502910, 503755, 503823, 503841, 503855, 504000, 504074, 504389, 504415, 504439, 504580, 504589, 504635, 504711, 505169, 505451, 505565, 505573, 505638, 505789, 505942, 506004, 506178, 506279, 506310, 506454, 506656, 506707, 506933, 507015, 507037, 507042, 507081, 507223, 507235, 507473, 507575, 507667, 507797, 507893, 507975, 508101, 508312, 508370, 508482, 508586, 508602, 508639, 508730, 508917, 509008, 509014, 509131, 509258, 509260, 509403, 509451, 509656, 509699, 509719, 509735, 509824, 510095, 510329, 511076, 511321, 511384, 511398, 511453, 511599, 511647, 511760, 511999, 512194, 512248, 512330, 512403, 512476, 512564, 512648, 512657, 512776, 512836, 512929, 512985, 513041, 513181, 513283, 513484, 513524, 513567, 513580, 513688, 514376, 514508, 514540, 514586, 514797, 514914, 514979, 515025, 515077, 515266, 515350, 515445, 515577, 515579, 515828, 515982, 516038, 516143, 516173, 516316, 516318, 516601, 516677, 516708, 516804, 516871, 516916, 517056, 517069, 517523, 517687, 517832, 518213, 518326, 518770, 519039, 519208, 519338, 519491, 519522, 519569, 519611, 519688, 519764, 520009, 520077, 520264, 520301, 520324, 520531, 520659, 520707, 520832, 520871, 520910, 521052, 521141, 521231, 521259, 521282, 521405, 521509, 521540, 521601, 521717, 521719, 521819, 521956, 522007, 522156, 522393, 522638, 522713, 522751, 522889, 522940, 523033, 523100, 523175, 523194, 523229, 523241, 523782, 523807, 523811, 523957, 524108, 524280, 524456, 524742, 524850, 525083, 525155, 525247, 525286, 525322, 525600, 526103, 526197, 526256, 526392, 526706, 526728, 526751, 527029, 527215, 527220, 527427, 527528, 527616, 527695, 527750, 527784, 527960, 528314, 528399, 528524, 528578, 528705, 528862, 528977, 528980, 529105, 529122, 529148, 529528, 529568, 529762, 529939, 529966, 530052, 530061, 530099, 530146, 530162, 530457, 530466, 530470, 530624, 530820, 530836, 530854, 530975, 531036, 531134, 531135, 531495, 531707, 531771, 532058, 532071, 532129, 532481, 532493, 532530, 532575, 532690, 532761, 532855, 532901, 533145, 533206, 533493, 533536, 533816, 533855, 533958, 534041, 534270, 534394, 534601, 534605, 534639, 534664, 534673, 534827, 535094, 535156, 535253, 535306, 535523, 535578, 535608, 535858, 536038, 536073, 536343, 536947, 537053, 537153, 537241, 537270, 537355, 537506, 537672, 537802, 537812, 537827, 537964, 537991, 538067, 538236, 538364, 538458, 539143, 539445, 539883, 539962, 540280, 540414, 540466, 540502, 540928, 540932, 540962, 541055, 541123, 541291, 541634, 541664, 541773, 541952, 542073, 542089, 542127, 542423, 542625, 542776, 542856, 543043, 543047, 543300, 543528, 543581, 544052, 544306, 544444, 544519, 544565, 544605, 544811, 545007, 545100, 545129, 545219, 545407, 545594, 545730, 545826, 545958, 546011, 546219, 546325, 546475, 546556, 546626, 546659, 546717, 546823, 546826, 546829, 546964, 546976, 547144, 547336, 547383, 547502, 547519, 547816, 547854, 547886, 548246, 548267, 548339, 548506, 548524, 548555, 548780, 549055, 549136, 549167, 549220, 549390, 549674, 549738, 549930, 550084, 550322, 550349, 550426, 550471, 550691, 550714, 550797, 550939, 551215, 551304, 551350, 551439, 551660, 551780, 551794, 551804, 551815, 551820, 551822, 552371, 552612, 552775, 552842, 552883, 552902, 553094, 553221, 553339, 553511, 553664, 553669, 553731, 553776, 553788, 553990, 554002, 554156, 554266, 554291, 554328, 554579, 554595, 554735, 554838, 555005, 555009, 555012, 555050, 555412, 555597, 555705, 555972, 556000, 556158, 556193, 556498, 556765, 556873, 557172, 557258, 557501, 557672, 557884, 557916, 558073, 558114, 558213, 558421, 558558, 558854, 559099, 559160, 559348, 559513, 559543, 559547, 559707, 559842, 559956, 560011, 560178, 560256, 560266, 560279, 560312, 560371, 560474, 560880, 560911, 561009, 561223, 561256, 561335, 561366, 561465, 561679, 561889, 561958, 562059, 562121, 562197, 562207, 562229, 562243, 562443, 562448, 562561, 562581, 562818, 562843, 563267, 563281, 563349, 563470, 563603, 563604, 563648, 563653, 563702, 563758, 563882, 564023, 564091, 564127, 564133, 564280, 564336, 565012, 565045, 565153, 565391, 565469, 565563, 565597, 565607, 565624, 565776, 565778, 565853, 565877, 565962, 565989, 566042, 566282, 566436, 566524, 566758, 566923, 567011, 567197, 567432, 567640, 567740, 567825, 567886, 567898, 568147, 568195, 568213, 568290, 568439, 568584, 568690, 568710, 568814, 568981, 569030, 569059, 569273, 569565, 569700, 569825, 569917, 569972, 569976, 570169, 570448, 570456, 570471, 570539, 570664, 570688, 570736, 570756, 570782, 570834, 571008, 571264, 571313, 571598, 571718, 571804, 571857, 571893, 571943, 572303, 572388, 572408, 572462, 572517, 572555, 572620, 572678, 572900, 572956, 573008, 573094, 573258, 573391, 573626, 573943, 574297, 574315, 574425, 574520, 574702, 574810, 574823, 575081, 575187, 575205, 575243, 575357, 575372, 575500, 575815, 575970, 576031, 576052, 576566, 576654, 576955, 577149, 577182, 577539, 577584, 577735, 577862, 577864, 577932, 577959, 577976, 578093, 578236, 578489, 578500, 578545, 578792, 578871, 578922, 578967, 579070, 579091, 579158, 579307, 579321, 579635, 579655, 579818, 579893, 579900, 579902, 579970, 580197, 580294, 580410, 580418, 580757, 581062, 581100, 581206, 581317, 581357, 581482, 581615, 581781], "split_sha256": "00f8420d2aa16bf8dad421bca425c7d19c3e508f3ebe4836cf73d4380468eff8", "direct_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_direct_seed0_retry5/final.pt", "direct_checkpoint_sha256": "25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8", "quality_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_quality_seed0/final.pt", "quality_checkpoint_sha256": "07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065", "epoch": 3, "rho": 12.0853271484375, "steps": 2, "eta": [6.04266357421875, 3.021331787109375], "lambda": 0.003, "scientific_drift": ["Actual training seed 20261005 differs from recorded seed 0.", "Quality ranking uses soft IoU rather than normal binary original-mask IoU.", "Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.", "Failure random-control radius is global rho, not each oracle displacement norm.", "Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.", "Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.", "DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.", "Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test."], "parameter_updates": false, "oracle_used": false, "streaming": true, "dtype": "float32", "tf32": false, "heldout_states_definition": "Fixed success-state probes: c0 and ±rho/8*u1, ±rho/4*u1, ±rho/4*u2, +rho/8*u2; GT-free and identity-seeded. Added diagnostic, not a preregistered oracle-direction test.", "decoder_definition": "Official process_mask full P@c -> bilinear640 -> predicted box crop -> >0; inverse letterbox binary mask -> >.5. Original COCO GT.", "pixel_metric_version": "qcr-original-pixels-v2", "pixel_metric_definition": "Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background.", "strict_maskfail_definition": "Entire argmax-correct R_arg has a Box75 witness and no Mask75; measured here on its intersection with fixed TAL class-correct Box75 baseline-fail candidates. Unknown on historical DEV cache.", "elapsed_s": 1637.370605725795, "prototype_storage_images": {"torch.float32": 5000}, "scope": "Official one2one TAL matched candidates; no AP/postprocessing quality reranking", "integrity_verified": true, "image_membership_verified": true, "native_decode_verified": true, "partial": false, "no_positive_images": 48, "statistical_repair": "Map unchanged native pyramid_level 0/1/2 to P3/P4/P5. No model execution or raw-row change.", "source_evaluation_summary": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_final_complete_seed0_retry1/SUMMARY.json"}`.
