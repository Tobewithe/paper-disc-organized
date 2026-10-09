# QCR FINAL fixed-candidate metrics

Protocol status: **incomplete**. Stage II permitted: **False**.

21 candidate rows; 2 / 2 planned images contain rows. 0 planned images contain no rows. Those images receive no artificial zero IoU.

All candidate means weight candidates equally. Image macro first averages eligible candidates within each image, then weights eligible images equally. Subgroup-empty images and images without a defined pixel metric are excluded from that estimand. Confidence intervals resample entire eligible images with replacement, preserving the pairing between arms. Deltas and CI in tables are percentage points (pp).

TAL MaskFail = class_correct and BoxIoU ≥ .75 and original MaskIoU < .75. Strict MaskFail additionally requires an audited raw_arg_maskfail=true: the full class-argmax raw set has a Box75 candidate and no Mask75 candidate for the target GT. Unknown raw-arg labels are excluded from strict metrics and block release; TAL failure never substitutes for strict failure. Baseline success uses the same class/box criteria and original MaskIoU ≥ .75. Size uses original COCO annotation area: small <32², medium [32²,96²), large ≥96².

Coverage uses original-resolution binary-mask intersection / original COCO annToMask area. Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background. All pixels outside the target GT (including other instances) are negative. AUC is exact Mann–Whitney with averaged ties. Undefined metrics remain NA. This is a fixed official one-to-one TAL matched-candidate evaluation and makes no COCO AP claim.

## Protocol criteria

| Criterion | Estimate | Required | Pass |
|---|---:|---:|---|
| heldout_pairwise_accuracy | 50.3968 % | >= 65.0000 % | False |
| step1_true_iou_improvement_fraction | 14.2857 % | > 55.0000 % | False |
| strict_maskfail_D_minus_B | -1.0900 pp | >= 0.2000 pp | False |
| strict_maskfail_D_minus_B_CI_low | -1.0900 pp | > 0.0000 pp | False |
| strict_maskfail_D_minus_A | -2.6544 pp | >= 0.5000 pp | False |
| all_D_minus_A_CI_low | -8.5685 pp | >= -0.1000 pp | False |

Ineligibility/incompleteness: partial_evaluation; pyramid_level_unknown; scientific_scope_drift.

Scientific drift: Actual training seed 20261005 differs from recorded seed 0.; Quality ranking uses soft IoU rather than normal binary original-mask IoU.; Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.; Failure random-control radius is global rho, not each oracle displacement norm.; Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.; Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.; DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.; Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test..

## Quality diagnostics

Held-out pairwise accuracy: instance equal 50.3968%; pairs weighted 50.7799%; image macro 56.9940%; 577 non-tie hard-IoU pairs, 21 eligible instances. True hard-IoU ties are excluded; predicted Q ties get 0.5 credit. Spearman uses averaged tie ranks; constant-state instances are undefined.

Spearman: instance equal 0.0163; image macro 0.1329; 21 defined instances.

Step-one true IoU increase: 14.2857% over 21 candidates (3 increase, 17 decrease, 1 ties; ties stay in the denominator).

Nonzero Q/IoU delta sign consistency: 65.0000% over 20 instances. Positive Q but decreased true IoU: 6 / 9 positive-Q instances (66.6667%).

Sign table (Q(c1)−Q(c0) rows / true IoU(D1)−IoU(A) columns; exact zeros kept separately):

| Q delta | IoU decrease | IoU tie | IoU increase |
|---|---:|---:|---:|
| negative | 11 | 0 | 1 |
| zero | 0 | 0 | 0 |
| positive | 6 | 1 | 2 |

Exclusions from nonzero consistency: {"missing_q_or_iou": 0, "both_zero": 0, "q_zero_iou_nonzero": 0, "q_nonzero_iou_zero": 1}.

Q-Rank at these unchanged fixed candidates keeps c0 and therefore the same MaskIoU as A. No ranking/AP effect is estimated here.

## all

21 candidates; 2 eligible images; 0 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 72.6242 / 82.9019 | 52.3810 / 75.0000 | 91.0631 / 94.2764 | 90.6978 / 93.6487 | 44.1583 / 31.9155 |
| B | 71.8814 / 82.4610 | 47.6190 / 72.5000 | 92.9550 / 95.2140 | 91.3561 / 93.9663 | 52.3407 / 36.1970 |
| D1 | 62.2338 / 34.3194 | 33.3333 / 17.5000 | 91.0497 / 49.4478 | 87.2810 / 90.0750 | 65.6210 / 34.4510 |
| D | 71.3013 / 78.1374 | 42.8571 / 70.0000 | 93.1416 / 90.4192 | 90.8503 / 93.7620 | 47.6245 / 29.6276 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -0.5801 | -4.3237 | [-8.4612, -0.1861] | 21 / 2 |
| D_minus_B | mask75 | -4.7619 | -2.5000 | [-5.0000, 0.0000] | 21 / 2 |
| D_minus_B | coverage | 0.1866 | -4.7948 | [-10.3007, 0.7110] | 21 / 2 |
| D_minus_B | auc | -0.5059 | -0.2043 | [-0.5376, 0.1290] | 21 / 2 |
| D_minus_B | fpr | -4.7162 | -6.5694 | [-8.6177, -4.5211] | 21 / 2 |
| D_minus_A | iou | -1.3229 | -4.7646 | [-8.5685, -0.9606] | 21 / 2 |
| D_minus_A | mask75 | -9.5238 | -5.0000 | [-10.0000, 0.0000] | 21 / 2 |
| D_minus_A | coverage | 2.0785 | -3.8572 | [-10.4178, 2.7033] | 21 / 2 |
| D_minus_A | auc | 0.1525 | 0.1133 | [0.0700, 0.1566] | 21 / 2 |
| D_minus_A | fpr | 3.4662 | -2.2879 | [-8.6476, 4.0719] | 21 / 2 |
| B_minus_A | iou | -0.7428 | -0.4409 | [-0.7746, -0.1072] | 21 / 2 |
| B_minus_A | mask75 | -4.7619 | -2.5000 | [-5.0000, 0.0000] | 21 / 2 |
| B_minus_A | coverage | 1.8919 | 0.9376 | [-0.1171, 1.9923] | 21 / 2 |
| B_minus_A | auc | 0.6583 | 0.3176 | [-0.0590, 0.6942] | 21 / 2 |
| B_minus_A | fpr | 8.1824 | 4.2815 | [-0.0299, 8.5930] | 21 / 2 |
| D1_minus_A | iou | -10.3904 | -48.5825 | [-90.7949, -6.3702] | 21 / 2 |
| D1_minus_A | mask75 | -19.0476 | -57.5000 | [-100.0000, -15.0000] | 21 / 2 |
| D1_minus_A | coverage | -0.0134 | -44.8286 | [-94.3613, 4.7040] | 21 / 2 |
| D1_minus_A | auc | -3.4168 | -3.5737 | [-3.7470, -3.4003] | 21 / 2 |
| D1_minus_A | fpr | 21.4627 | 2.5355 | [-18.3839, 23.4550] | 21 / 2 |
| D_minus_D1 | iou | 9.0675 | 43.8180 | [5.4095, 82.2264] | 21 / 2 |
| D_minus_D1 | mask75 | 9.5238 | 52.5000 | [5.0000, 100.0000] | 21 / 2 |
| D_minus_D1 | coverage | 2.0918 | 40.9714 | [-2.0008, 83.9436] | 21 / 2 |
| D_minus_D1 | auc | 3.5693 | 3.6870 | [3.5569, 3.8171] | 21 / 2 |
| D_minus_D1 | fpr | -17.9964 | -4.8234 | [-19.3831, 9.7363] | 21 / 2 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 10 | 1 / 11 | 0.0000 / 0.0000 | 9.0909 / 5.0000 | -1 | -2.5000 [-5.0000, 0.0000] |
| D1 | 1 / 10 | 5 / 11 | 10.0000 / 10.0000 | 45.4545 / 70.0000 | -4 | -57.5000 [-100.0000, -15.0000] |
| D | 0 / 10 | 2 / 11 | 0.0000 / 0.0000 | 18.1818 / 10.0000 | -2 | -5.0000 [-10.0000, 0.0000] |

## tal_maskfail

5 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.1501 / 63.1501 | 0.0000 / 0.0000 | 85.3614 / 85.3614 | 92.3959 / 92.3959 | 41.9445 / 41.9445 |
| B | 61.5857 / 61.5857 | 0.0000 / 0.0000 | 86.9761 / 86.9761 | 92.4118 / 92.4118 | 47.3543 / 47.3543 |
| D1 | 52.2707 / 52.2707 | 0.0000 / 0.0000 | 92.7562 / 92.7562 | 87.5560 / 87.5560 | 64.8253 / 64.8253 |
| D | 60.4956 / 60.4956 | 0.0000 / 0.0000 | 88.2751 / 88.2751 | 91.6392 / 91.6392 | 46.4880 / 46.4880 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.0900 | -1.0900 | [-1.0900, -1.0900] | 5 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D_minus_B | coverage | 1.2990 | 1.2990 | [1.2990, 1.2990] | 5 / 1 |
| D_minus_B | auc | -0.7726 | -0.7726 | [-0.7726, -0.7726] | 5 / 1 |
| D_minus_B | fpr | -0.8663 | -0.8663 | [-0.8663, -0.8663] | 5 / 1 |
| D_minus_A | iou | -2.6544 | -2.6544 | [-2.6544, -2.6544] | 5 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D_minus_A | coverage | 2.9138 | 2.9138 | [2.9138, 2.9138] | 5 / 1 |
| D_minus_A | auc | -0.7567 | -0.7567 | [-0.7567, -0.7567] | 5 / 1 |
| D_minus_A | fpr | 4.5436 | 4.5436 | [4.5436, 4.5436] | 5 / 1 |
| B_minus_A | iou | -1.5644 | -1.5644 | [-1.5644, -1.5644] | 5 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| B_minus_A | coverage | 1.6148 | 1.6148 | [1.6148, 1.6148] | 5 / 1 |
| B_minus_A | auc | 0.0159 | 0.0159 | [0.0159, 0.0159] | 5 / 1 |
| B_minus_A | fpr | 5.4099 | 5.4099 | [5.4099, 5.4099] | 5 / 1 |
| D1_minus_A | iou | -10.8793 | -10.8793 | [-10.8793, -10.8793] | 5 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D1_minus_A | coverage | 7.3948 | 7.3948 | [7.3948, 7.3948] | 5 / 1 |
| D1_minus_A | auc | -4.8399 | -4.8399 | [-4.8399, -4.8399] | 5 / 1 |
| D1_minus_A | fpr | 22.8809 | 22.8809 | [22.8809, 22.8809] | 5 / 1 |
| D_minus_D1 | iou | 8.2249 | 8.2249 | [8.2249, 8.2249] | 5 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D_minus_D1 | coverage | -4.4810 | -4.4810 | [-4.4810, -4.4810] | 5 / 1 |
| D_minus_D1 | auc | 4.0832 | 4.0832 | [4.0832, 4.0832] | 5 / 1 |
| D_minus_D1 | fpr | -18.3373 | -18.3373 | [-18.3373, -18.3373] | 5 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail

5 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 63.1501 / 63.1501 | 0.0000 / 0.0000 | 85.3614 / 85.3614 | 92.3959 / 92.3959 | 41.9445 / 41.9445 |
| B | 61.5857 / 61.5857 | 0.0000 / 0.0000 | 86.9761 / 86.9761 | 92.4118 / 92.4118 | 47.3543 / 47.3543 |
| D1 | 52.2707 / 52.2707 | 0.0000 / 0.0000 | 92.7562 / 92.7562 | 87.5560 / 87.5560 | 64.8253 / 64.8253 |
| D | 60.4956 / 60.4956 | 0.0000 / 0.0000 | 88.2751 / 88.2751 | 91.6392 / 91.6392 | 46.4880 / 46.4880 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -1.0900 | -1.0900 | [-1.0900, -1.0900] | 5 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D_minus_B | coverage | 1.2990 | 1.2990 | [1.2990, 1.2990] | 5 / 1 |
| D_minus_B | auc | -0.7726 | -0.7726 | [-0.7726, -0.7726] | 5 / 1 |
| D_minus_B | fpr | -0.8663 | -0.8663 | [-0.8663, -0.8663] | 5 / 1 |
| D_minus_A | iou | -2.6544 | -2.6544 | [-2.6544, -2.6544] | 5 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D_minus_A | coverage | 2.9138 | 2.9138 | [2.9138, 2.9138] | 5 / 1 |
| D_minus_A | auc | -0.7567 | -0.7567 | [-0.7567, -0.7567] | 5 / 1 |
| D_minus_A | fpr | 4.5436 | 4.5436 | [4.5436, 4.5436] | 5 / 1 |
| B_minus_A | iou | -1.5644 | -1.5644 | [-1.5644, -1.5644] | 5 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| B_minus_A | coverage | 1.6148 | 1.6148 | [1.6148, 1.6148] | 5 / 1 |
| B_minus_A | auc | 0.0159 | 0.0159 | [0.0159, 0.0159] | 5 / 1 |
| B_minus_A | fpr | 5.4099 | 5.4099 | [5.4099, 5.4099] | 5 / 1 |
| D1_minus_A | iou | -10.8793 | -10.8793 | [-10.8793, -10.8793] | 5 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D1_minus_A | coverage | 7.3948 | 7.3948 | [7.3948, 7.3948] | 5 / 1 |
| D1_minus_A | auc | -4.8399 | -4.8399 | [-4.8399, -4.8399] | 5 / 1 |
| D1_minus_A | fpr | 22.8809 | 22.8809 | [22.8809, 22.8809] | 5 / 1 |
| D_minus_D1 | iou | 8.2249 | 8.2249 | [8.2249, 8.2249] | 5 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| D_minus_D1 | coverage | -4.4810 | -4.4810 | [-4.4810, -4.4810] | 5 / 1 |
| D_minus_D1 | auc | 4.0832 | 4.0832 | [4.0832, 4.0832] | 5 / 1 |
| D_minus_D1 | fpr | -18.3373 | -18.3373 | [-18.3373, -18.3373] | 5 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 5 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## baseline_success

10 candidates; 2 eligible images; 0 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 86.8536 / 90.1460 | 100.0000 / 100.0000 | 96.0508 / 96.8407 | 95.2594 / 95.9931 | 35.8676 / 28.0971 |
| B | 85.8584 / 89.5455 | 90.0000 / 94.4444 | 96.6859 / 97.1414 | 95.6109 / 96.1622 | 45.1579 / 33.2451 |
| D1 | 70.9854 / 40.9770 | 60.0000 / 33.3333 | 88.7335 / 50.8371 | 93.8323 / 93.5349 | 59.5116 / 33.0620 |
| D | 83.8444 / 84.6661 | 80.0000 / 88.8889 | 96.2701 / 92.3324 | 94.9564 / 95.8559 | 39.7162 / 26.3918 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -2.0140 | -4.8794 | [-8.4612, -1.2977] | 10 / 2 |
| D_minus_B | mask75 | -10.0000 | -5.5556 | [-11.1111, 0.0000] | 10 / 2 |
| D_minus_B | coverage | -0.4158 | -4.8091 | [-10.3007, 0.6825] | 10 / 2 |
| D_minus_B | auc | -0.6545 | -0.3063 | [-0.7415, 0.1290] | 10 / 2 |
| D_minus_B | fpr | -5.4417 | -6.8532 | [-8.6177, -5.0888] | 10 / 2 |
| D_minus_A | iou | -3.0092 | -5.4800 | [-8.5685, -2.3915] | 10 / 2 |
| D_minus_A | mask75 | -20.0000 | -11.1111 | [-22.2222, 0.0000] | 10 / 2 |
| D_minus_A | coverage | 0.2193 | -4.5083 | [-10.4178, 1.4012] | 10 / 2 |
| D_minus_A | auc | -0.3030 | -0.1372 | [-0.3444, 0.0700] | 10 / 2 |
| D_minus_A | fpr | 3.8487 | -1.7052 | [-8.6476, 5.2371] | 10 / 2 |
| B_minus_A | iou | -0.9952 | -0.6005 | [-1.0938, -0.1072] | 10 / 2 |
| B_minus_A | mask75 | -10.0000 | -5.5556 | [-11.1111, 0.0000] | 10 / 2 |
| B_minus_A | coverage | 0.6351 | 0.3008 | [-0.1171, 0.7186] | 10 / 2 |
| B_minus_A | auc | 0.3515 | 0.1691 | [-0.0590, 0.3971] | 10 / 2 |
| B_minus_A | fpr | 9.2903 | 5.1480 | [-0.0299, 10.3259] | 10 / 2 |
| D1_minus_A | iou | -15.8683 | -49.1690 | [-90.7949, -7.5431] | 10 / 2 |
| D1_minus_A | mask75 | -40.0000 | -66.6667 | [-100.0000, -33.3333] | 10 / 2 |
| D1_minus_A | coverage | -7.3173 | -46.0035 | [-94.3613, 2.3543] | 10 / 2 |
| D1_minus_A | auc | -1.4272 | -2.4582 | [-3.7470, -1.1694] | 10 / 2 |
| D1_minus_A | fpr | 23.6440 | 4.9649 | [-18.3839, 28.3138] | 10 / 2 |
| D_minus_D1 | iou | 12.8591 | 43.6890 | [5.1516, 82.2264] | 10 / 2 |
| D_minus_D1 | mask75 | 20.0000 | 55.5556 | [11.1111, 100.0000] | 10 / 2 |
| D_minus_D1 | coverage | 7.5365 | 41.4952 | [-0.9531, 83.9436] | 10 / 2 |
| D_minus_D1 | auc | 1.1242 | 2.3210 | [0.8250, 3.8171] | 10 / 2 |
| D_minus_D1 | fpr | -19.7953 | -6.6702 | [-23.0766, 9.7363] | 10 / 2 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 1 / 10 | NA / NA | 10.0000 / 5.5556 | -1 | -5.5556 [-11.1111, 0.0000] |
| D1 | 0 / 0 | 4 / 10 | NA / NA | 40.0000 / 66.6667 | -4 | -66.6667 [-100.0000, -33.3333] |
| D | 0 / 0 | 2 / 10 | NA / NA | 20.0000 / 11.1111 | -2 | -11.1111 [-22.2222, 0.0000] |

## all/size/small

11 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 66.4701 / 66.4701 | 36.3636 / 36.3636 | 91.6526 / 91.6526 | 86.8294 / 86.8294 | 60.3573 / 60.3573 |
| B | 65.4340 / 65.4340 | 27.2727 / 27.2727 | 93.9530 / 93.9530 | 88.0922 / 88.0922 | 73.4460 / 73.4460 |
| D1 | 64.4884 / 64.4884 | 27.2727 / 27.2727 | 94.9179 / 94.9179 | 83.6380 / 83.6380 | 78.9170 / 78.9170 |
| D | 67.8845 / 67.8845 | 27.2727 / 27.2727 | 94.0568 / 94.0568 | 88.1518 / 88.1518 | 59.9454 / 59.9454 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 2.4505 | 2.4505 | [2.4505, 2.4505] | 11 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 11 / 1 |
| D_minus_B | coverage | 0.1038 | 0.1038 | [0.1038, 0.1038] | 11 / 1 |
| D_minus_B | auc | 0.0597 | 0.0597 | [0.0597, 0.0597] | 11 / 1 |
| D_minus_B | fpr | -13.5006 | -13.5006 | [-13.5006, -13.5006] | 11 / 1 |
| D_minus_A | iou | 1.4144 | 1.4144 | [1.4144, 1.4144] | 11 / 1 |
| D_minus_A | mask75 | -9.0909 | -9.0909 | [-9.0909, -9.0909] | 11 / 1 |
| D_minus_A | coverage | 2.4042 | 2.4042 | [2.4042, 2.4042] | 11 / 1 |
| D_minus_A | auc | 1.3225 | 1.3225 | [1.3225, 1.3225] | 11 / 1 |
| D_minus_A | fpr | -0.4119 | -0.4119 | [-0.4119, -0.4119] | 11 / 1 |
| B_minus_A | iou | -1.0361 | -1.0361 | [-1.0361, -1.0361] | 11 / 1 |
| B_minus_A | mask75 | -9.0909 | -9.0909 | [-9.0909, -9.0909] | 11 / 1 |
| B_minus_A | coverage | 2.3004 | 2.3004 | [2.3004, 2.3004] | 11 / 1 |
| B_minus_A | auc | 1.2628 | 1.2628 | [1.2628, 1.2628] | 11 / 1 |
| B_minus_A | fpr | 13.0887 | 13.0887 | [13.0887, 13.0887] | 11 / 1 |
| D1_minus_A | iou | -1.9817 | -1.9817 | [-1.9817, -1.9817] | 11 / 1 |
| D1_minus_A | mask75 | -9.0909 | -9.0909 | [-9.0909, -9.0909] | 11 / 1 |
| D1_minus_A | coverage | 3.2653 | 3.2653 | [3.2653, 3.2653] | 11 / 1 |
| D1_minus_A | auc | -3.1914 | -3.1914 | [-3.1914, -3.1914] | 11 / 1 |
| D1_minus_A | fpr | 18.5597 | 18.5597 | [18.5597, 18.5597] | 11 / 1 |
| D_minus_D1 | iou | 3.3961 | 3.3961 | [3.3961, 3.3961] | 11 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 11 / 1 |
| D_minus_D1 | coverage | -0.8611 | -0.8611 | [-0.8611, -0.8611] | 11 / 1 |
| D_minus_D1 | auc | 4.5138 | 4.5138 | [4.5138, 4.5138] | 11 / 1 |
| D_minus_D1 | fpr | -18.9717 | -18.9717 | [-18.9717, -18.9717] | 11 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 7 | 1 / 4 | 0.0000 / 0.0000 | 25.0000 / 25.0000 | -1 | -9.0909 [-9.0909, -9.0909] |
| D1 | 1 / 7 | 2 / 4 | 14.2857 / 14.2857 | 50.0000 / 50.0000 | -1 | -9.0909 [-9.0909, -9.0909] |
| D | 0 / 7 | 1 / 4 | 0.0000 / 0.0000 | 25.0000 / 25.0000 | -1 | -9.0909 [-9.0909, -9.0909] |

## all/size/medium

8 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 75.3282 / 75.3282 | 62.5000 / 62.5000 | 88.4059 / 88.4059 | 94.1975 / 94.1975 | 27.6341 / 27.6341 |
| B | 74.8016 / 74.8016 | 62.5000 / 62.5000 | 90.2425 / 90.2425 | 94.1860 / 94.1860 | 31.5208 / 31.5208 |
| D1 | 62.4183 / 62.4183 | 37.5000 / 37.5000 | 95.5753 / 95.5753 | 90.0940 / 90.0940 | 55.9275 / 55.9275 |
| D | 71.0025 / 71.0025 | 50.0000 / 50.0000 | 91.8027 / 91.8027 | 92.7589 / 92.7589 | 37.4696 / 37.4696 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7992 | -3.7992 | [-3.7992, -3.7992] | 8 / 1 |
| D_minus_B | mask75 | -12.5000 | -12.5000 | [-12.5000, -12.5000] | 8 / 1 |
| D_minus_B | coverage | 1.5602 | 1.5602 | [1.5602, 1.5602] | 8 / 1 |
| D_minus_B | auc | -1.4272 | -1.4272 | [-1.4272, -1.4272] | 8 / 1 |
| D_minus_B | fpr | 5.9488 | 5.9488 | [5.9488, 5.9488] | 8 / 1 |
| D_minus_A | iou | -4.3257 | -4.3257 | [-4.3257, -4.3257] | 8 / 1 |
| D_minus_A | mask75 | -12.5000 | -12.5000 | [-12.5000, -12.5000] | 8 / 1 |
| D_minus_A | coverage | 3.3968 | 3.3968 | [3.3968, 3.3968] | 8 / 1 |
| D_minus_A | auc | -1.4386 | -1.4386 | [-1.4386, -1.4386] | 8 / 1 |
| D_minus_A | fpr | 9.8355 | 9.8355 | [9.8355, 9.8355] | 8 / 1 |
| B_minus_A | iou | -0.5265 | -0.5265 | [-0.5265, -0.5265] | 8 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 8 / 1 |
| B_minus_A | coverage | 1.8366 | 1.8366 | [1.8366, 1.8366] | 8 / 1 |
| B_minus_A | auc | -0.0114 | -0.0114 | [-0.0114, -0.0114] | 8 / 1 |
| B_minus_A | fpr | 3.8867 | 3.8867 | [3.8867, 3.8867] | 8 / 1 |
| D1_minus_A | iou | -12.9099 | -12.9099 | [-12.9099, -12.9099] | 8 / 1 |
| D1_minus_A | mask75 | -25.0000 | -25.0000 | [-25.0000, -25.0000] | 8 / 1 |
| D1_minus_A | coverage | 7.1694 | 7.1694 | [7.1694, 7.1694] | 8 / 1 |
| D1_minus_A | auc | -4.1034 | -4.1034 | [-4.1034, -4.1034] | 8 / 1 |
| D1_minus_A | fpr | 28.2934 | 28.2934 | [28.2934, 28.2934] | 8 / 1 |
| D_minus_D1 | iou | 8.5842 | 8.5842 | [8.5842, 8.5842] | 8 / 1 |
| D_minus_D1 | mask75 | 12.5000 | 12.5000 | [12.5000, 12.5000] | 8 / 1 |
| D_minus_D1 | coverage | -3.7726 | -3.7726 | [-3.7726, -3.7726] | 8 / 1 |
| D_minus_D1 | auc | 2.6648 | 2.6648 | [2.6648, 2.6648] | 8 / 1 |
| D_minus_D1 | fpr | -18.4580 | -18.4580 | [-18.4580, -18.4580] | 8 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 3 | 0 / 5 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 3 | 2 / 5 | 0.0000 / 0.0000 | 40.0000 / 40.0000 | -2 | -25.0000 [-25.0000, -25.0000] |
| D | 0 / 3 | 1 / 5 | 0.0000 / 0.0000 | 20.0000 / 20.0000 | -1 | -12.5000 [-12.5000, -12.5000] |

## all/size/large

2 candidates; 2 eligible images; 0 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 95.6555 / 95.6555 | 100.0000 / 100.0000 | 98.4497 / 98.4497 | 97.9756 / 97.9756 | 21.1607 / 21.1607 |
| B | 95.6609 / 95.6609 | 100.0000 / 100.0000 | 98.3157 / 98.3157 | 97.9886 / 97.9886 | 19.5413 / 19.5413 |
| D1 | 49.0952 / 49.0952 | 50.0000 / 50.0000 | 51.6729 / 51.6729 | 96.0648 / 96.0648 | 31.2663 / 31.2663 |
| D | 91.2885 / 91.2885 | 100.0000 / 100.0000 | 93.4635 / 93.4635 | 98.0573 / 98.0573 | 20.4796 / 20.4796 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -4.3723 | -4.3723 | [-8.4612, -0.2835] | 2 / 2 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 2 |
| D_minus_B | coverage | -4.8521 | -4.8521 | [-10.3007, 0.5964] | 2 / 2 |
| D_minus_B | auc | 0.0687 | 0.0687 | [0.0084, 0.1290] | 2 / 2 |
| D_minus_B | fpr | 0.9383 | 0.9383 | [-8.6177, 10.4944] | 2 / 2 |
| D_minus_A | iou | -4.3670 | -4.3670 | [-8.5685, -0.1655] | 2 / 2 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 2 |
| D_minus_A | coverage | -4.9862 | -4.9862 | [-10.4178, 0.4454] | 2 / 2 |
| D_minus_A | auc | 0.0817 | 0.0817 | [0.0700, 0.0934] | 2 / 2 |
| D_minus_A | fpr | -0.6811 | -0.6811 | [-8.6476, 7.2853] | 2 / 2 |
| B_minus_A | iou | 0.0054 | 0.0054 | [-0.1072, 0.1180] | 2 / 2 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 2 |
| B_minus_A | coverage | -0.1340 | -0.1340 | [-0.1510, -0.1171] | 2 / 2 |
| B_minus_A | auc | 0.0130 | 0.0130 | [-0.0590, 0.0850] | 2 / 2 |
| B_minus_A | fpr | -1.6195 | -1.6195 | [-3.2090, -0.0299] | 2 / 2 |
| D1_minus_A | iou | -46.5604 | -46.5604 | [-90.7949, -2.3258] | 2 / 2 |
| D1_minus_A | mask75 | -50.0000 | -50.0000 | [-100.0000, 0.0000] | 2 / 2 |
| D1_minus_A | coverage | -46.7768 | -46.7768 | [-94.3613, 0.8077] | 2 / 2 |
| D1_minus_A | auc | -1.9108 | -1.9108 | [-3.7470, -0.0745] | 2 / 2 |
| D1_minus_A | fpr | 10.1055 | 10.1055 | [-18.3839, 38.5950] | 2 / 2 |
| D_minus_D1 | iou | 42.1934 | 42.1934 | [2.1603, 82.2264] | 2 / 2 |
| D_minus_D1 | mask75 | 50.0000 | 50.0000 | [0.0000, 100.0000] | 2 / 2 |
| D_minus_D1 | coverage | 41.7906 | 41.7906 | [-0.3623, 83.9436] | 2 / 2 |
| D_minus_D1 | auc | 1.9925 | 1.9925 | [0.1679, 3.8171] | 2 / 2 |
| D_minus_D1 | fpr | -10.7867 | -10.7867 | [-31.3096, 9.7363] | 2 / 2 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 2 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 0 | 1 / 2 | NA / NA | 50.0000 / 50.0000 | -1 | -50.0000 [-100.0000, 0.0000] |
| D | 0 / 0 | 0 / 2 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |

## all/pyramid/P3

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

2 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 68.5328 / 68.5328 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 88.7680 / 88.7680 | 82.6484 / 82.6484 |
| B | 66.3243 / 66.3243 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 89.1135 / 89.1135 | 90.4110 / 90.4110 |
| D1 | 66.5690 / 66.5690 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 85.4524 / 85.4524 | 89.4977 / 89.4977 |
| D | 69.1702 / 69.1702 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 89.5350 / 89.5350 | 80.5936 / 80.5936 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 2.8458 | 2.8458 | [2.8458, 2.8458] | 2 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_B | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_B | auc | 0.4216 | 0.4216 | [0.4216, 0.4216] | 2 / 1 |
| D_minus_B | fpr | -9.8174 | -9.8174 | [-9.8174, -9.8174] | 2 / 1 |
| D_minus_A | iou | 0.6374 | 0.6374 | [0.6374, 0.6374] | 2 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_A | auc | 0.7670 | 0.7670 | [0.7670, 0.7670] | 2 / 1 |
| D_minus_A | fpr | -2.0548 | -2.0548 | [-2.0548, -2.0548] | 2 / 1 |
| B_minus_A | iou | -2.2085 | -2.2085 | [-2.2085, -2.2085] | 2 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| B_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| B_minus_A | auc | 0.3455 | 0.3455 | [0.3455, 0.3455] | 2 / 1 |
| B_minus_A | fpr | 7.7626 | 7.7626 | [7.7626, 7.7626] | 2 / 1 |
| D1_minus_A | iou | -1.9638 | -1.9638 | [-1.9638, -1.9638] | 2 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D1_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D1_minus_A | auc | -3.3156 | -3.3156 | [-3.3156, -3.3156] | 2 / 1 |
| D1_minus_A | fpr | 6.8493 | 6.8493 | [6.8493, 6.8493] | 2 / 1 |
| D_minus_D1 | iou | 2.6012 | 2.6012 | [2.6012, 2.6012] | 2 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_D1 | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_D1 | auc | 4.0826 | 4.0826 | [4.0826, 4.0826] | 2 / 1 |
| D_minus_D1 | fpr | -8.9041 | -8.9041 | [-8.9041, -8.9041] | 2 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## tal_maskfail/size/medium

3 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 59.5616 / 59.5616 | 0.0000 / 0.0000 | 77.8976 / 77.8976 | 94.8145 / 94.8145 | 14.8085 / 14.8085 |
| B | 58.4266 / 58.4266 | 0.0000 / 0.0000 | 80.5888 / 80.5888 | 94.6107 / 94.6107 | 18.6499 / 18.6499 |
| D1 | 42.7386 / 42.7386 | 0.0000 / 0.0000 | 90.2223 / 90.2223 | 88.9583 / 88.9583 | 48.3771 / 48.3771 |
| D | 54.7126 / 54.7126 | 0.0000 / 0.0000 | 82.7538 / 82.7538 | 93.0419 / 93.0419 | 23.7510 / 23.7510 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7140 | -3.7140 | [-3.7140, -3.7140] | 3 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_B | coverage | 2.1650 | 2.1650 | [2.1650, 2.1650] | 3 / 1 |
| D_minus_B | auc | -1.5688 | -1.5688 | [-1.5688, -1.5688] | 3 / 1 |
| D_minus_B | fpr | 5.1011 | 5.1011 | [5.1011, 5.1011] | 3 / 1 |
| D_minus_A | iou | -4.8490 | -4.8490 | [-4.8490, -4.8490] | 3 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_A | coverage | 4.8563 | 4.8563 | [4.8563, 4.8563] | 3 / 1 |
| D_minus_A | auc | -1.7726 | -1.7726 | [-1.7726, -1.7726] | 3 / 1 |
| D_minus_A | fpr | 8.9425 | 8.9425 | [8.9425, 8.9425] | 3 / 1 |
| B_minus_A | iou | -1.1350 | -1.1350 | [-1.1350, -1.1350] | 3 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| B_minus_A | coverage | 2.6913 | 2.6913 | [2.6913, 2.6913] | 3 / 1 |
| B_minus_A | auc | -0.2038 | -0.2038 | [-0.2038, -0.2038] | 3 / 1 |
| B_minus_A | fpr | 3.8414 | 3.8414 | [3.8414, 3.8414] | 3 / 1 |
| D1_minus_A | iou | -16.8231 | -16.8231 | [-16.8231, -16.8231] | 3 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D1_minus_A | coverage | 12.3247 | 12.3247 | [12.3247, 12.3247] | 3 / 1 |
| D1_minus_A | auc | -5.8562 | -5.8562 | [-5.8562, -5.8562] | 3 / 1 |
| D1_minus_A | fpr | 33.5686 | 33.5686 | [33.5686, 33.5686] | 3 / 1 |
| D_minus_D1 | iou | 11.9741 | 11.9741 | [11.9741, 11.9741] | 3 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_D1 | coverage | -7.4684 | -7.4684 | [-7.4684, -7.4684] | 3 / 1 |
| D_minus_D1 | auc | 4.0836 | 4.0836 | [4.0836, 4.0836] | 3 / 1 |
| D_minus_D1 | fpr | -24.6261 | -24.6261 | [-24.6261, -24.6261] | 3 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## tal_maskfail/size/large

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

## tal_maskfail/pyramid/P3

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

2 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 68.5328 / 68.5328 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 88.7680 / 88.7680 | 82.6484 / 82.6484 |
| B | 66.3243 / 66.3243 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 89.1135 / 89.1135 | 90.4110 / 90.4110 |
| D1 | 66.5690 / 66.5690 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 85.4524 / 85.4524 | 89.4977 / 89.4977 |
| D | 69.1702 / 69.1702 | 0.0000 / 0.0000 | 96.5570 / 96.5570 | 89.5350 / 89.5350 | 80.5936 / 80.5936 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 2.8458 | 2.8458 | [2.8458, 2.8458] | 2 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_B | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_B | auc | 0.4216 | 0.4216 | [0.4216, 0.4216] | 2 / 1 |
| D_minus_B | fpr | -9.8174 | -9.8174 | [-9.8174, -9.8174] | 2 / 1 |
| D_minus_A | iou | 0.6374 | 0.6374 | [0.6374, 0.6374] | 2 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_A | auc | 0.7670 | 0.7670 | [0.7670, 0.7670] | 2 / 1 |
| D_minus_A | fpr | -2.0548 | -2.0548 | [-2.0548, -2.0548] | 2 / 1 |
| B_minus_A | iou | -2.2085 | -2.2085 | [-2.2085, -2.2085] | 2 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| B_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| B_minus_A | auc | 0.3455 | 0.3455 | [0.3455, 0.3455] | 2 / 1 |
| B_minus_A | fpr | 7.7626 | 7.7626 | [7.7626, 7.7626] | 2 / 1 |
| D1_minus_A | iou | -1.9638 | -1.9638 | [-1.9638, -1.9638] | 2 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D1_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D1_minus_A | auc | -3.3156 | -3.3156 | [-3.3156, -3.3156] | 2 / 1 |
| D1_minus_A | fpr | 6.8493 | 6.8493 | [6.8493, 6.8493] | 2 / 1 |
| D_minus_D1 | iou | 2.6012 | 2.6012 | [2.6012, 2.6012] | 2 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_D1 | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 1 |
| D_minus_D1 | auc | 4.0826 | 4.0826 | [4.0826, 4.0826] | 2 / 1 |
| D_minus_D1 | fpr | -8.9041 | -8.9041 | [-8.9041, -8.9041] | 2 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 2 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail/size/medium

3 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 59.5616 / 59.5616 | 0.0000 / 0.0000 | 77.8976 / 77.8976 | 94.8145 / 94.8145 | 14.8085 / 14.8085 |
| B | 58.4266 / 58.4266 | 0.0000 / 0.0000 | 80.5888 / 80.5888 | 94.6107 / 94.6107 | 18.6499 / 18.6499 |
| D1 | 42.7386 / 42.7386 | 0.0000 / 0.0000 | 90.2223 / 90.2223 | 88.9583 / 88.9583 | 48.3771 / 48.3771 |
| D | 54.7126 / 54.7126 | 0.0000 / 0.0000 | 82.7538 / 82.7538 | 93.0419 / 93.0419 | 23.7510 / 23.7510 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.7140 | -3.7140 | [-3.7140, -3.7140] | 3 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_B | coverage | 2.1650 | 2.1650 | [2.1650, 2.1650] | 3 / 1 |
| D_minus_B | auc | -1.5688 | -1.5688 | [-1.5688, -1.5688] | 3 / 1 |
| D_minus_B | fpr | 5.1011 | 5.1011 | [5.1011, 5.1011] | 3 / 1 |
| D_minus_A | iou | -4.8490 | -4.8490 | [-4.8490, -4.8490] | 3 / 1 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_A | coverage | 4.8563 | 4.8563 | [4.8563, 4.8563] | 3 / 1 |
| D_minus_A | auc | -1.7726 | -1.7726 | [-1.7726, -1.7726] | 3 / 1 |
| D_minus_A | fpr | 8.9425 | 8.9425 | [8.9425, 8.9425] | 3 / 1 |
| B_minus_A | iou | -1.1350 | -1.1350 | [-1.1350, -1.1350] | 3 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| B_minus_A | coverage | 2.6913 | 2.6913 | [2.6913, 2.6913] | 3 / 1 |
| B_minus_A | auc | -0.2038 | -0.2038 | [-0.2038, -0.2038] | 3 / 1 |
| B_minus_A | fpr | 3.8414 | 3.8414 | [3.8414, 3.8414] | 3 / 1 |
| D1_minus_A | iou | -16.8231 | -16.8231 | [-16.8231, -16.8231] | 3 / 1 |
| D1_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D1_minus_A | coverage | 12.3247 | 12.3247 | [12.3247, 12.3247] | 3 / 1 |
| D1_minus_A | auc | -5.8562 | -5.8562 | [-5.8562, -5.8562] | 3 / 1 |
| D1_minus_A | fpr | 33.5686 | 33.5686 | [33.5686, 33.5686] | 3 / 1 |
| D_minus_D1 | iou | 11.9741 | 11.9741 | [11.9741, 11.9741] | 3 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_D1 | coverage | -7.4684 | -7.4684 | [-7.4684, -7.4684] | 3 / 1 |
| D_minus_D1 | auc | 4.0836 | 4.0836 | [4.0836, 4.0836] | 3 / 1 |
| D_minus_D1 | fpr | -24.6261 | -24.6261 | [-24.6261, -24.6261] | 3 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |
| D | 0 / 3 | 0 / 0 | 0.0000 / 0.0000 | NA / NA | 0 | 0.0000 [0.0000, 0.0000] |

## strict_maskfail/size/large

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

3 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 84.4281 / 84.4281 | 100.0000 / 100.0000 | 96.6847 / 96.6847 | 95.8356 / 95.8356 | 46.5690 / 46.5690 |
| B | 81.3764 / 81.3764 | 66.6667 / 66.6667 | 96.6847 / 96.6847 | 96.8254 / 96.8254 | 72.0933 / 72.0933 |
| D1 | 80.1775 / 80.1775 | 66.6667 / 66.6667 | 96.6847 / 96.6847 | 97.4386 / 97.4386 | 76.7648 / 76.7648 |
| D | 83.9951 / 83.9951 | 66.6667 / 66.6667 | 96.5378 / 96.5378 | 96.8349 / 96.8349 | 42.5666 / 42.5666 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | 2.6187 | 2.6187 | [2.6187, 2.6187] | 3 / 1 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_B | coverage | -0.1468 | -0.1468 | [-0.1468, -0.1468] | 3 / 1 |
| D_minus_B | auc | 0.0096 | 0.0096 | [0.0096, 0.0096] | 3 / 1 |
| D_minus_B | fpr | -29.5267 | -29.5267 | [-29.5267, -29.5267] | 3 / 1 |
| D_minus_A | iou | -0.4330 | -0.4330 | [-0.4330, -0.4330] | 3 / 1 |
| D_minus_A | mask75 | -33.3333 | -33.3333 | [-33.3333, -33.3333] | 3 / 1 |
| D_minus_A | coverage | -0.1468 | -0.1468 | [-0.1468, -0.1468] | 3 / 1 |
| D_minus_A | auc | 0.9993 | 0.9993 | [0.9993, 0.9993] | 3 / 1 |
| D_minus_A | fpr | -4.0025 | -4.0025 | [-4.0025, -4.0025] | 3 / 1 |
| B_minus_A | iou | -3.0518 | -3.0518 | [-3.0518, -3.0518] | 3 / 1 |
| B_minus_A | mask75 | -33.3333 | -33.3333 | [-33.3333, -33.3333] | 3 / 1 |
| B_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| B_minus_A | auc | 0.9897 | 0.9897 | [0.9897, 0.9897] | 3 / 1 |
| B_minus_A | fpr | 25.5243 | 25.5243 | [25.5243, 25.5243] | 3 / 1 |
| D1_minus_A | iou | -4.2506 | -4.2506 | [-4.2506, -4.2506] | 3 / 1 |
| D1_minus_A | mask75 | -33.3333 | -33.3333 | [-33.3333, -33.3333] | 3 / 1 |
| D1_minus_A | coverage | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D1_minus_A | auc | 1.6029 | 1.6029 | [1.6029, 1.6029] | 3 / 1 |
| D1_minus_A | fpr | 30.1958 | 30.1958 | [30.1958, 30.1958] | 3 / 1 |
| D_minus_D1 | iou | 3.8176 | 3.8176 | [3.8176, 3.8176] | 3 / 1 |
| D_minus_D1 | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 3 / 1 |
| D_minus_D1 | coverage | -0.1468 | -0.1468 | [-0.1468, -0.1468] | 3 / 1 |
| D_minus_D1 | auc | -0.6036 | -0.6036 | [-0.6036, -0.6036] | 3 / 1 |
| D_minus_D1 | fpr | -34.1982 | -34.1982 | [-34.1982, -34.1982] | 3 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 1 / 3 | NA / NA | 33.3333 / 33.3333 | -1 | -33.3333 [-33.3333, -33.3333] |
| D1 | 0 / 0 | 1 / 3 | NA / NA | 33.3333 / 33.3333 | -1 | -33.3333 [-33.3333, -33.3333] |
| D | 0 / 0 | 1 / 3 | NA / NA | 33.3333 / 33.3333 | -1 | -33.3333 [-33.3333, -33.3333] |

## baseline_success/size/medium

5 candidates; 1 eligible images; 1 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 84.7881 / 84.7881 | 100.0000 / 100.0000 | 94.7109 / 94.7109 | 93.8272 / 93.8272 | 35.3294 / 35.3294 |
| B | 84.6267 / 84.6267 | 100.0000 / 100.0000 | 96.0346 / 96.0346 | 93.9312 / 93.9312 | 39.2433 / 39.2433 |
| D1 | 74.2261 / 74.2261 | 60.0000 / 60.0000 | 98.7871 / 98.7871 | 90.7755 / 90.7755 | 60.4578 / 60.4578 |
| D | 80.7763 / 80.7763 | 80.0000 / 80.0000 | 97.2320 / 97.2320 | 92.5890 / 92.5890 | 45.7007 / 45.7007 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -3.8503 | -3.8503 | [-3.8503, -3.8503] | 5 / 1 |
| D_minus_B | mask75 | -20.0000 | -20.0000 | [-20.0000, -20.0000] | 5 / 1 |
| D_minus_B | coverage | 1.1974 | 1.1974 | [1.1974, 1.1974] | 5 / 1 |
| D_minus_B | auc | -1.3422 | -1.3422 | [-1.3422, -1.3422] | 5 / 1 |
| D_minus_B | fpr | 6.4574 | 6.4574 | [6.4574, 6.4574] | 5 / 1 |
| D_minus_A | iou | -4.0118 | -4.0118 | [-4.0118, -4.0118] | 5 / 1 |
| D_minus_A | mask75 | -20.0000 | -20.0000 | [-20.0000, -20.0000] | 5 / 1 |
| D_minus_A | coverage | 2.5211 | 2.5211 | [2.5211, 2.5211] | 5 / 1 |
| D_minus_A | auc | -1.2382 | -1.2382 | [-1.2382, -1.2382] | 5 / 1 |
| D_minus_A | fpr | 10.3713 | 10.3713 | [10.3713, 10.3713] | 5 / 1 |
| B_minus_A | iou | -0.1615 | -0.1615 | [-0.1615, -0.1615] | 5 / 1 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 5 / 1 |
| B_minus_A | coverage | 1.3237 | 1.3237 | [1.3237, 1.3237] | 5 / 1 |
| B_minus_A | auc | 0.1040 | 0.1040 | [0.1040, 0.1040] | 5 / 1 |
| B_minus_A | fpr | 3.9139 | 3.9139 | [3.9139, 3.9139] | 5 / 1 |
| D1_minus_A | iou | -10.5620 | -10.5620 | [-10.5620, -10.5620] | 5 / 1 |
| D1_minus_A | mask75 | -40.0000 | -40.0000 | [-40.0000, -40.0000] | 5 / 1 |
| D1_minus_A | coverage | 4.0762 | 4.0762 | [4.0762, 4.0762] | 5 / 1 |
| D1_minus_A | auc | -3.0518 | -3.0518 | [-3.0518, -3.0518] | 5 / 1 |
| D1_minus_A | fpr | 25.1284 | 25.1284 | [25.1284, 25.1284] | 5 / 1 |
| D_minus_D1 | iou | 6.5502 | 6.5502 | [6.5502, 6.5502] | 5 / 1 |
| D_minus_D1 | mask75 | 20.0000 | 20.0000 | [20.0000, 20.0000] | 5 / 1 |
| D_minus_D1 | coverage | -1.5551 | -1.5551 | [-1.5551, -1.5551] | 5 / 1 |
| D_minus_D1 | auc | 1.8136 | 1.8136 | [1.8136, 1.8136] | 5 / 1 |
| D_minus_D1 | fpr | -14.7571 | -14.7571 | [-14.7571, -14.7571] | 5 / 1 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 5 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 0 | 2 / 5 | NA / NA | 40.0000 / 40.0000 | -2 | -40.0000 [-40.0000, -40.0000] |
| D | 0 / 0 | 1 / 5 | NA / NA | 20.0000 / 20.0000 | -1 | -20.0000 [-20.0000, -20.0000] |

## baseline_success/size/large

2 candidates; 2 eligible images; 0 planned images have no subgroup member.

| Arm | IoU candidate / image % | Mask75 candidate / image % | Coverage candidate / image % | AUC candidate / image % | FPR candidate / image % |
|---|---:|---:|---:|---:|---:|
| A | 95.6555 / 95.6555 | 100.0000 / 100.0000 | 98.4497 / 98.4497 | 97.9756 / 97.9756 | 21.1607 / 21.1607 |
| B | 95.6609 / 95.6609 | 100.0000 / 100.0000 | 98.3157 / 98.3157 | 97.9886 / 97.9886 | 19.5413 / 19.5413 |
| D1 | 49.0952 / 49.0952 | 50.0000 / 50.0000 | 51.6729 / 51.6729 | 96.0648 / 96.0648 | 31.2663 / 31.2663 |
| D | 91.2885 / 91.2885 | 100.0000 / 100.0000 | 93.4635 / 93.4635 | 98.0573 / 98.0573 | 20.4796 / 20.4796 |

| Paired comparison | Metric | Candidate Δ pp | Image Δ pp | Image bootstrap 95% CI pp | Valid candidates / images |
|---|---|---:|---:|---|---:|
| D_minus_B | iou | -4.3723 | -4.3723 | [-8.4612, -0.2835] | 2 / 2 |
| D_minus_B | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 2 |
| D_minus_B | coverage | -4.8521 | -4.8521 | [-10.3007, 0.5964] | 2 / 2 |
| D_minus_B | auc | 0.0687 | 0.0687 | [0.0084, 0.1290] | 2 / 2 |
| D_minus_B | fpr | 0.9383 | 0.9383 | [-8.6177, 10.4944] | 2 / 2 |
| D_minus_A | iou | -4.3670 | -4.3670 | [-8.5685, -0.1655] | 2 / 2 |
| D_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 2 |
| D_minus_A | coverage | -4.9862 | -4.9862 | [-10.4178, 0.4454] | 2 / 2 |
| D_minus_A | auc | 0.0817 | 0.0817 | [0.0700, 0.0934] | 2 / 2 |
| D_minus_A | fpr | -0.6811 | -0.6811 | [-8.6476, 7.2853] | 2 / 2 |
| B_minus_A | iou | 0.0054 | 0.0054 | [-0.1072, 0.1180] | 2 / 2 |
| B_minus_A | mask75 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 2 / 2 |
| B_minus_A | coverage | -0.1340 | -0.1340 | [-0.1510, -0.1171] | 2 / 2 |
| B_minus_A | auc | 0.0130 | 0.0130 | [-0.0590, 0.0850] | 2 / 2 |
| B_minus_A | fpr | -1.6195 | -1.6195 | [-3.2090, -0.0299] | 2 / 2 |
| D1_minus_A | iou | -46.5604 | -46.5604 | [-90.7949, -2.3258] | 2 / 2 |
| D1_minus_A | mask75 | -50.0000 | -50.0000 | [-100.0000, 0.0000] | 2 / 2 |
| D1_minus_A | coverage | -46.7768 | -46.7768 | [-94.3613, 0.8077] | 2 / 2 |
| D1_minus_A | auc | -1.9108 | -1.9108 | [-3.7470, -0.0745] | 2 / 2 |
| D1_minus_A | fpr | 10.1055 | 10.1055 | [-18.3839, 38.5950] | 2 / 2 |
| D_minus_D1 | iou | 42.1934 | 42.1934 | [2.1603, 82.2264] | 2 / 2 |
| D_minus_D1 | mask75 | 50.0000 | 50.0000 | [0.0000, 100.0000] | 2 / 2 |
| D_minus_D1 | coverage | 41.7906 | 41.7906 | [-0.3623, 83.9436] | 2 / 2 |
| D_minus_D1 | auc | 1.9925 | 1.9925 | [0.1679, 3.8171] | 2 / 2 |
| D_minus_D1 | fpr | -10.7867 | -10.7867 | [-31.3096, 9.7363] | 2 / 2 |

| Arm vs A | Repair / baseline failures | Damage / baseline successes | Repair rate candidate / image % | Damage rate candidate / image % | Net count | Net image pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| B | 0 / 0 | 0 / 2 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |
| D1 | 0 / 0 | 1 / 2 | NA / NA | 50.0000 / 50.0000 | -1 | -50.0000 [-100.0000, 0.0000] |
| D | 0 / 0 | 0 / 2 | NA / NA | 0.0000 / 0.0000 | 0 | 0.0000 [0.0000, 0.0000] |

## baseline_success/pyramid/P3

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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

0 candidates; 0 eligible images; 2 planned images have no subgroup member.

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
    "candidate_mean": 2.519113233401662,
    "image_macro": 1.7051146365702152,
    "valid_candidates": 21,
    "valid_images": 2,
    "median_ms": 0.7657483220100403,
    "p95_ms": 0.8460693061351776
  },
  "quality_ms": {
    "candidate_mean": 10.87104311833779,
    "image_macro": 11.097247176803648,
    "valid_candidates": 21,
    "valid_images": 2,
    "median_ms": 11.012816801667213,
    "p95_ms": 12.172517366707325
  }
}

Timing is summarized in the supplied unit; no image-total or dataset-total latency is inferred from repeated row values. The unpopulated planned-image list cannot by itself distinguish zero matched candidates from an interrupted evaluation; completion must be established by the run and metadata.

Metadata: `{"split": "final", "planned_images": 5000, "evaluated_images_requested": 2, "image_ids": [139, 285], "split_sha256": "00f8420d2aa16bf8dad421bca425c7d19c3e508f3ebe4836cf73d4380468eff8", "direct_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_direct_seed0_retry5/final.pt", "direct_checkpoint_sha256": "25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8", "quality_checkpoint": "/root/autodl-tmp/qcr_run/runs/RUN_stage1_quality_seed0/final.pt", "quality_checkpoint_sha256": "07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065", "epoch": 3, "rho": 12.0853271484375, "steps": 2, "eta": [6.04266357421875, 3.021331787109375], "lambda": 0.003, "scientific_drift": ["Actual training seed 20261005 differs from recorded seed 0.", "Quality ranking uses soft IoU rather than normal binary original-mask IoU.", "Ranking uses relu(0.05 - true_soft_difference * predicted_difference), abs(true_difference)>0.01; locked rule specifies sign ordering margin 0.02.", "Failure random-control radius is global rho, not each oracle displacement norm.", "Training averages candidates within image then weights images equally; equal-instance weighting was not implemented.", "Quality input has no explicit class/score, scale embedding, or coefficient displacement descriptor.", "DEV uses historical compact cache; its prototype storage precision must be reported separately from the FP32 streaming FINAL.", "Held-out random-state diagnostic was added after training and is not a frozen oracle-direction state test."], "parameter_updates": false, "oracle_used": false, "streaming": true, "dtype": "float32", "tf32": false, "heldout_states_definition": "Fixed success-state probes: c0 and ±rho/8*u1, ±rho/4*u1, ±rho/4*u2, +rho/8*u2; GT-free and identity-seeded. Added diagnostic, not a preregistered oracle-direction test.", "decoder_definition": "Official process_mask full P@c -> bilinear640 -> predicted box crop -> >0; inverse letterbox binary mask -> >.5. Original COCO GT.", "pixel_metric_version": "qcr-original-pixels-v2", "pixel_metric_definition": "Primary AUC: continuous cropped logits bilinearly inverse-letterboxed, exact average-tie pixel ranking against original COCO GT inside inverse-letterboxed predicted-box support (>0.5). Primary FPR: actual normal binary decoded mask positives among original non-GT pixels in that same support. Letterbox640 AUC/logit>0 FPR also retained as auxiliary fields. FPR negatives include other instances, not just background.", "strict_maskfail_definition": "Entire argmax-correct R_arg has a Box75 witness and no Mask75; measured here on its intersection with fixed TAL class-correct Box75 baseline-fail candidates. Unknown on historical DEV cache.", "elapsed_s": 2.4550982378423214, "prototype_storage_images": {"torch.float32": 2}, "scope": "Official one2one TAL matched candidates; no AP/postprocessing quality reranking", "integrity_verified": true, "image_membership_verified": true, "native_decode_verified": true, "partial": true, "no_positive_images": 0}`.
