# Frozen U evidence-path usage audit

FULL_minus_A +0.186694 [+0.159258, +0.214129] pp; FULL_minus_NATIVE -0.069581 [-0.192523, +0.053360] pp; FULL_minus_MEAN -0.000380 [-0.000760, +0.000000] pp; FULL_minus_ROLL +0.011215 [-0.009543, +0.031974] pp.

One existing epoch-3 U checkpoint, no updates. NATIVE is the native cv4 inside that very checkpoint, not the independently trained N arm.

A uses original c0; FULL uses the unchanged U forward; NATIVE removes its extra correction; MEAN replaces the 64 evidence values by their within-candidate mean; ROLL moves them by four rows and four columns on the verified 8x8 U grid. P, K, prediction boxes, scores and decoding stay fixed.

## Absolute values

IoU/Mask75/coverage/AUC/FPR are percentages; BCE is in its original loss units. Each cell is image macro / candidate mean.

| Population | Arm | IoU | Mask75 | Coverage | AUC | FPR | BCE |
|---|---|---:|---:|---:|---:|---:|---:|
| fit:all | A | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:all | FULL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:all | NATIVE | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:all | MEAN | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:all | ROLL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:failure | A | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:failure | FULL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:failure | NATIVE | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:failure | MEAN | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:failure | ROLL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:success | A | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:success | FULL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:success | NATIVE | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:success | MEAN | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:success | ROLL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:box_good_mask_bad | A | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:box_good_mask_bad | FULL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:box_good_mask_bad | NATIVE | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:box_good_mask_bad | MEAN | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| fit:box_good_mask_bad | ROLL | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined | undefined / undefined |
| dev:all | A | +73.548857 / +79.458138 | +40.000000 / +57.142857 | +90.402635 / +93.160390 | +97.671353 / +97.941933 | +17.577386 / +14.863249 | +3.895803 / +4.588828 |
| dev:all | FULL | +73.735551 / +79.656589 | +40.000000 / +57.142857 | +90.185243 / +92.849830 | +97.697651 / +97.971197 | +17.157283 / +14.361759 | +3.735645 / +4.359891 |
| dev:all | NATIVE | +73.805132 / +79.673481 | +40.000000 / +57.142857 | +90.388928 / +92.979692 | +97.698858 / +97.972081 | +17.353610 / +14.528358 | +3.742641 / +4.368715 |
| dev:all | MEAN | +73.735931 / +79.657132 | +40.000000 / +57.142857 | +90.186164 / +92.851145 | +97.697713 / +97.971314 | +17.161634 / +14.367974 | +3.732419 / +4.356122 |
| dev:all | ROLL | +73.724336 / +79.654271 | +40.000000 / +57.142857 | +90.190715 / +92.857646 | +97.700623 / +97.974634 | +17.183207 / +14.379062 | +3.729291 / +4.352783 |
| dev:failure | A | +59.464150 / +59.562945 | +0.000000 / +0.000000 | +87.073178 / +86.038077 | +95.247740 / +95.845161 | +22.033858 / +22.659363 | +3.031679 / +2.780701 |
| dev:failure | FULL | +59.927808 / +59.925137 | +0.000000 / +0.000000 | +85.995268 / +85.319470 | +95.320970 / +95.900439 | +20.851501 / +21.794391 | +2.862331 / +2.667911 |
| dev:failure | NATIVE | +59.995208 / +60.034245 | +0.000000 / +0.000000 | +86.253427 / +85.616889 | +95.323064 / +95.902490 | +21.056654 / +22.019724 | +2.869987 / +2.673926 |
| dev:failure | MEAN | +59.932584 / +59.928321 | +0.000000 / +0.000000 | +85.997774 / +85.321141 | +95.321379 / +95.900690 | +20.848609 / +21.792463 | +2.861262 / +2.666545 |
| dev:failure | ROLL | +59.931618 / +59.917019 | +0.000000 / +0.000000 | +86.015322 / +85.332839 | +95.334315 / +95.909965 | +20.868737 / +21.821228 | +2.857929 / +2.663445 |
| dev:success | A | +94.379532 / +94.379532 | +100.000000 / +100.000000 | +98.502125 / +98.502125 | +99.514511 / +99.514511 | +9.016163 / +9.016163 | +5.944922 / +5.944922 |
| dev:success | FULL | +94.455179 / +94.455179 | +100.000000 / +100.000000 | +98.497600 / +98.497600 | +99.524265 / +99.524265 | +8.787285 / +8.787285 | +5.628877 / +5.628877 |
| dev:success | NATIVE | +94.402909 / +94.402909 | +100.000000 / +100.000000 | +98.501793 / +98.501793 | +99.524274 / +99.524274 | +8.909834 / +8.909834 | +5.639806 / +5.639806 |
| dev:success | MEAN | +94.453740 / +94.453740 | +100.000000 / +100.000000 | +98.498649 / +98.498649 | +99.524281 / +99.524281 | +8.799608 / +8.799608 | +5.623304 / +5.623304 |
| dev:success | ROLL | +94.457209 / +94.457209 | +100.000000 / +100.000000 | +98.501252 / +98.501252 | +99.523135 / +99.523135 | +8.797438 / +8.797438 | +5.619787 / +5.619787 |
| dev:box_good_mask_bad | A | +62.677751 / +62.677751 | +0.000000 / +0.000000 | +89.959371 / +89.959371 | +95.304710 / +95.304710 | +22.877934 / +22.877934 | +3.041071 / +3.041071 |
| dev:box_good_mask_bad | FULL | +63.221038 / +63.221038 | +0.000000 / +0.000000 | +88.881460 / +88.881460 | +95.358692 / +95.358692 | +21.580476 / +21.580476 | +2.865424 / +2.865424 |
| dev:box_good_mask_bad | NATIVE | +63.128290 / +63.128290 | +0.000000 / +0.000000 | +88.951650 / +88.951650 | +95.360066 / +95.360066 | +21.744864 / +21.744864 | +2.874171 / +2.874171 |
| dev:box_good_mask_bad | MEAN | +63.225815 / +63.225815 | +0.000000 / +0.000000 | +88.883967 / +88.883967 | +95.359069 / +95.359069 | +21.577583 / +21.577583 | +2.865045 / +2.865045 |
| dev:box_good_mask_bad | ROLL | +63.208862 / +63.208862 | +0.000000 / +0.000000 | +88.901515 / +88.901515 | +95.372981 / +95.372981 | +21.620732 / +21.620732 | +2.861925 / +2.861925 |

## Paired comparisons

Each interval resamples whole images, including candidate-mean intervals. BCE differences retain original loss units; other differences are percentage points.

| Population | Comparison | Metric | Image macro delta [95% CI] | Candidate delta [95% CI] | Repair / damage / net Mask75 |
|---|---|---|---:|---:|---:|
| fit:all | FULL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | NATIVE_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | NATIVE_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | NATIVE_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | NATIVE_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | NATIVE_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | NATIVE_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | MEAN_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | MEAN_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | MEAN_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | MEAN_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | MEAN_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | MEAN_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | ROLL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | ROLL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | ROLL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | ROLL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | ROLL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | ROLL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_NATIVE | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_NATIVE | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_NATIVE | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_NATIVE | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_NATIVE | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_NATIVE | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_MEAN | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_MEAN | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_MEAN | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_MEAN | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_MEAN | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_MEAN | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_ROLL | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_ROLL | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_ROLL | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_ROLL | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_ROLL | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:all | FULL_minus_ROLL | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | NATIVE_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | NATIVE_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | NATIVE_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | NATIVE_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | NATIVE_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | NATIVE_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | MEAN_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | MEAN_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | MEAN_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | MEAN_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | MEAN_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | MEAN_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | ROLL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | ROLL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | ROLL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | ROLL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | ROLL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | ROLL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_NATIVE | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_NATIVE | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_NATIVE | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_NATIVE | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_NATIVE | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_NATIVE | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_MEAN | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_MEAN | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_MEAN | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_MEAN | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_MEAN | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_MEAN | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_ROLL | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_ROLL | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_ROLL | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_ROLL | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_ROLL | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:failure | FULL_minus_ROLL | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | NATIVE_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | NATIVE_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | NATIVE_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | NATIVE_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | NATIVE_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | NATIVE_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | MEAN_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | MEAN_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | MEAN_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | MEAN_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | MEAN_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | MEAN_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | ROLL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | ROLL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | ROLL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | ROLL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | ROLL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | ROLL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_NATIVE | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_NATIVE | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_NATIVE | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_NATIVE | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_NATIVE | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_NATIVE | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_MEAN | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_MEAN | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_MEAN | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_MEAN | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_MEAN | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_MEAN | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_ROLL | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_ROLL | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_ROLL | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_ROLL | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_ROLL | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:success | FULL_minus_ROLL | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | NATIVE_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | NATIVE_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | NATIVE_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | NATIVE_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | NATIVE_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | NATIVE_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | MEAN_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | MEAN_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | MEAN_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | MEAN_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | MEAN_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | MEAN_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | ROLL_minus_A | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | ROLL_minus_A | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | ROLL_minus_A | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | ROLL_minus_A | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | ROLL_minus_A | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | ROLL_minus_A | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_NATIVE | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_NATIVE | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_NATIVE | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_NATIVE | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_NATIVE | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_NATIVE | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_MEAN | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_MEAN | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_MEAN | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_MEAN | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_MEAN | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_MEAN | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_ROLL | iou | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_ROLL | mask75 | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_ROLL | coverage | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_ROLL | auc | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_ROLL | fpr | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| fit:box_good_mask_bad | FULL_minus_ROLL | bce | undefined [undefined, undefined] | undefined [undefined, undefined] | 0 / 0 / 0 |
| dev:all | FULL_minus_A | iou | +0.186694 [+0.159258, +0.214129] | +0.198451 [+0.159258, +0.214129] | 0 / 0 / 0 |
| dev:all | FULL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_A | coverage | -0.217392 [-0.434784, +0.000000] | -0.310560 [-0.434784, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_A | auc | +0.026298 [+0.019376, +0.033219] | +0.029264 [+0.019376, +0.033219] | 0 / 0 / 0 |
| dev:all | FULL_minus_A | fpr | -0.420103 [-0.610004, -0.230203] | -0.501489 [-0.610004, -0.230203] | 0 / 0 / 0 |
| dev:all | FULL_minus_A | bce | -0.160158 [-0.320641, +0.000324] | -0.228936 [-0.320641, +0.000324] | 0 / 0 / 0 |
| dev:all | NATIVE_minus_A | iou | +0.256275 [+0.160768, +0.351782] | +0.215344 [+0.160768, +0.351782] | 0 / 0 / 0 |
| dev:all | NATIVE_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | NATIVE_minus_A | coverage | -0.013707 [-0.403354, +0.375940] | -0.180699 [-0.403354, +0.375940] | 0 / 0 / 0 |
| dev:all | NATIVE_minus_A | auc | +0.027505 [+0.021338, +0.033672] | +0.030148 [+0.021338, +0.033672] | 0 / 0 / 0 |
| dev:all | NATIVE_minus_A | fpr | -0.223776 [-0.483042, +0.035490] | -0.334890 [-0.483042, +0.035490] | 0 / 0 / 0 |
| dev:all | NATIVE_minus_A | bce | -0.153162 [-0.309381, +0.003057] | -0.220113 [-0.309381, +0.003057] | 0 / 0 / 0 |
| dev:all | MEAN_minus_A | iou | +0.187073 [+0.159258, +0.214889] | +0.198994 [+0.159258, +0.214889] | 0 / 0 / 0 |
| dev:all | MEAN_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | MEAN_minus_A | coverage | -0.216471 [-0.432943, +0.000000] | -0.309245 [-0.432943, +0.000000] | 0 / 0 / 0 |
| dev:all | MEAN_minus_A | auc | +0.026360 [+0.019311, +0.033409] | +0.029381 [+0.019311, +0.033409] | 0 / 0 / 0 |
| dev:all | MEAN_minus_A | fpr | -0.415753 [-0.601303, -0.230203] | -0.495274 [-0.601303, -0.230203] | 0 / 0 / 0 |
| dev:all | MEAN_minus_A | bce | -0.163385 [-0.325135, -0.001635] | -0.232706 [-0.325135, -0.001635] | 0 / 0 / 0 |
| dev:all | ROLL_minus_A | iou | +0.175478 [+0.127284, +0.223672] | +0.196133 [+0.127284, +0.223672] | 0 / 0 / 0 |
| dev:all | ROLL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | ROLL_minus_A | coverage | -0.211921 [-0.423841, +0.000000] | -0.302744 [-0.423841, +0.000000] | 0 / 0 / 0 |
| dev:all | ROLL_minus_A | auc | +0.029269 [+0.021262, +0.037277] | +0.032701 [+0.021262, +0.037277] | 0 / 0 / 0 |
| dev:all | ROLL_minus_A | fpr | -0.394179 [-0.604196, -0.184162] | -0.484186 [-0.604196, -0.184162] | 0 / 0 / 0 |
| dev:all | ROLL_minus_A | bce | -0.166512 [-0.328754, -0.004271] | -0.236044 [-0.328754, -0.004271] | 0 / 0 / 0 |
| dev:all | FULL_minus_NATIVE | iou | -0.069581 [-0.192523, +0.053360] | -0.016892 [-0.192523, +0.053360] | 0 / 0 / 0 |
| dev:all | FULL_minus_NATIVE | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_NATIVE | coverage | -0.203685 [-0.375940, -0.031430] | -0.129861 [-0.375940, -0.031430] | 0 / 0 / 0 |
| dev:all | FULL_minus_NATIVE | auc | -0.001207 [-0.001962, -0.000452] | -0.000884 [-0.001962, -0.000452] | 0 / 0 / 0 |
| dev:all | FULL_minus_NATIVE | fpr | -0.196327 [-0.265692, -0.126962] | -0.166599 [-0.265692, -0.126962] | 0 / 0 / 0 |
| dev:all | FULL_minus_NATIVE | bce | -0.006996 [-0.011260, -0.002733] | -0.008824 [-0.011260, -0.002733] | 0 / 0 / 0 |
| dev:all | FULL_minus_MEAN | iou | -0.000380 [-0.000760, +0.000000] | -0.000543 [-0.000760, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_MEAN | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_MEAN | coverage | -0.000921 [-0.001841, +0.000000] | -0.001315 [-0.001841, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_MEAN | auc | -0.000062 [-0.000190, +0.000065] | -0.000117 [-0.000190, +0.000065] | 0 / 0 / 0 |
| dev:all | FULL_minus_MEAN | fpr | -0.004351 [-0.008701, +0.000000] | -0.006215 [-0.008701, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_MEAN | bce | +0.003226 [+0.001959, +0.004494] | +0.003770 [+0.001959, +0.004494] | 0 / 0 / 0 |
| dev:all | FULL_minus_ROLL | iou | +0.011215 [-0.009543, +0.031974] | +0.002319 [-0.009543, +0.031974] | 0 / 0 / 0 |
| dev:all | FULL_minus_ROLL | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_ROLL | coverage | -0.005471 [-0.010943, +0.000000] | -0.007816 [-0.010943, +0.000000] | 0 / 0 / 0 |
| dev:all | FULL_minus_ROLL | auc | -0.002971 [-0.004057, -0.001886] | -0.003437 [-0.004057, -0.001886] | 0 / 0 / 0 |
| dev:all | FULL_minus_ROLL | fpr | -0.025924 [-0.046041, -0.005808] | -0.017303 [-0.046041, -0.005808] | 0 / 0 / 0 |
| dev:all | FULL_minus_ROLL | bce | +0.006354 [+0.004595, +0.008113] | +0.007108 [+0.004595, +0.008113] | 0 / 0 / 0 |
| dev:failure | FULL_minus_A | iou | +0.463658 [+0.159258, +0.768058] | +0.362191 [+0.159258, +0.768058] | 0 / 0 / 0 |
| dev:failure | FULL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_A | coverage | -1.077910 [-2.155821, +0.000000] | -0.718607 [-2.155821, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_A | auc | +0.073229 [+0.019376, +0.127082] | +0.055278 [+0.019376, +0.127082] | 0 / 0 / 0 |
| dev:failure | FULL_minus_A | fpr | -1.182357 [-2.134511, -0.230203] | -0.864972 [-2.134511, -0.230203] | 0 / 0 / 0 |
| dev:failure | FULL_minus_A | bce | -0.169348 [-0.339019, +0.000324] | -0.112791 [-0.339019, +0.000324] | 0 / 0 / 0 |
| dev:failure | NATIVE_minus_A | iou | +0.531059 [+0.351782, +0.710335] | +0.471300 [+0.351782, +0.710335] | 0 / 0 / 0 |
| dev:failure | NATIVE_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | NATIVE_minus_A | coverage | -0.819751 [-2.015442, +0.375940] | -0.421187 [-2.015442, +0.375940] | 0 / 0 / 0 |
| dev:failure | NATIVE_minus_A | auc | +0.075324 [+0.021338, +0.129310] | +0.057329 [+0.021338, +0.129310] | 0 / 0 / 0 |
| dev:failure | NATIVE_minus_A | fpr | -0.977203 [-1.989896, +0.035490] | -0.639639 [-1.989896, +0.035490] | 0 / 0 / 0 |
| dev:failure | NATIVE_minus_A | bce | -0.161692 [-0.326440, +0.003057] | -0.106775 [-0.326440, +0.003057] | 0 / 0 / 0 |
| dev:failure | MEAN_minus_A | iou | +0.468434 [+0.159258, +0.777611] | +0.365376 [+0.159258, +0.777611] | 0 / 0 / 0 |
| dev:failure | MEAN_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | MEAN_minus_A | coverage | -1.075404 [-2.150807, +0.000000] | -0.716936 [-2.150807, +0.000000] | 0 / 0 / 0 |
| dev:failure | MEAN_minus_A | auc | +0.073638 [+0.019311, +0.127966] | +0.055529 [+0.019311, +0.127966] | 0 / 0 / 0 |
| dev:failure | MEAN_minus_A | fpr | -1.185249 [-2.140295, -0.230203] | -0.866900 [-2.140295, -0.230203] | 0 / 0 / 0 |
| dev:failure | MEAN_minus_A | bce | -0.170417 [-0.339199, -0.001635] | -0.114156 [-0.339199, -0.001635] | 0 / 0 / 0 |
| dev:failure | ROLL_minus_A | iou | +0.467468 [+0.127284, +0.807653] | +0.354074 [+0.127284, +0.807653] | 0 / 0 / 0 |
| dev:failure | ROLL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | ROLL_minus_A | coverage | -1.057856 [-2.115712, +0.000000] | -0.705237 [-2.115712, +0.000000] | 0 / 0 / 0 |
| dev:failure | ROLL_minus_A | auc | +0.086575 [+0.021262, +0.151888] | +0.064804 [+0.021262, +0.151888] | 0 / 0 / 0 |
| dev:failure | ROLL_minus_A | fpr | -1.165121 [-2.146080, -0.184162] | -0.838135 [-2.146080, -0.184162] | 0 / 0 / 0 |
| dev:failure | ROLL_minus_A | bce | -0.173749 [-0.343227, -0.004271] | -0.117257 [-0.343227, -0.004271] | 0 / 0 / 0 |
| dev:failure | FULL_minus_NATIVE | iou | -0.067401 [-0.192523, +0.057722] | -0.109108 [-0.192523, +0.057722] | 0 / 0 / 0 |
| dev:failure | FULL_minus_NATIVE | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_NATIVE | coverage | -0.258159 [-0.375940, -0.140379] | -0.297420 [-0.375940, -0.140379] | 0 / 0 / 0 |
| dev:failure | FULL_minus_NATIVE | auc | -0.002095 [-0.002227, -0.001962] | -0.002050 [-0.002227, -0.001962] | 0 / 0 / 0 |
| dev:failure | FULL_minus_NATIVE | fpr | -0.205153 [-0.265692, -0.144615] | -0.225333 [-0.265692, -0.144615] | 0 / 0 / 0 |
| dev:failure | FULL_minus_NATIVE | bce | -0.007656 [-0.012579, -0.002733] | -0.006015 [-0.012579, -0.002733] | 0 / 0 / 0 |
| dev:failure | FULL_minus_MEAN | iou | -0.004777 [-0.009553, +0.000000] | -0.003184 [-0.009553, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_MEAN | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_MEAN | coverage | -0.002507 [-0.005014, +0.000000] | -0.001671 [-0.005014, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_MEAN | auc | -0.000409 [-0.000883, +0.000065] | -0.000251 [-0.000883, +0.000065] | 0 / 0 / 0 |
| dev:failure | FULL_minus_MEAN | fpr | +0.002892 [+0.000000, +0.005785] | +0.001928 [+0.000000, +0.005785] | 0 / 0 / 0 |
| dev:failure | FULL_minus_MEAN | bce | +0.001069 [+0.000179, +0.001959] | +0.001366 [+0.000179, +0.001959] | 0 / 0 / 0 |
| dev:failure | FULL_minus_ROLL | iou | -0.003810 [-0.039595, +0.031974] | +0.008118 [-0.039595, +0.031974] | 0 / 0 / 0 |
| dev:failure | FULL_minus_ROLL | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_ROLL | coverage | -0.020054 [-0.040108, +0.000000] | -0.013369 [-0.040108, +0.000000] | 0 / 0 / 0 |
| dev:failure | FULL_minus_ROLL | auc | -0.013346 [-0.024805, -0.001886] | -0.009526 [-0.024805, -0.001886] | 0 / 0 / 0 |
| dev:failure | FULL_minus_ROLL | fpr | -0.017236 [-0.046041, +0.011569] | -0.026837 [-0.046041, +0.011569] | 0 / 0 / 0 |
| dev:failure | FULL_minus_ROLL | bce | +0.004402 [+0.004208, +0.004595] | +0.004466 [+0.004208, +0.004595] | 0 / 0 / 0 |
| dev:success | FULL_minus_A | iou | +0.075647 [+0.075647, +0.075647] | +0.075647 [+0.075647, +0.075647] | 0 / 0 / 0 |
| dev:success | FULL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | FULL_minus_A | coverage | -0.004525 [-0.004525, -0.004525] | -0.004525 [-0.004525, -0.004525] | 0 / 0 / 0 |
| dev:success | FULL_minus_A | auc | +0.009754 [+0.009754, +0.009754] | +0.009754 [+0.009754, +0.009754] | 0 / 0 / 0 |
| dev:success | FULL_minus_A | fpr | -0.228878 [-0.228878, -0.228878] | -0.228878 [-0.228878, -0.228878] | 0 / 0 / 0 |
| dev:success | FULL_minus_A | bce | -0.316046 [-0.316046, -0.316046] | -0.316046 [-0.316046, -0.316046] | 0 / 0 / 0 |
| dev:success | NATIVE_minus_A | iou | +0.023377 [+0.023377, +0.023377] | +0.023377 [+0.023377, +0.023377] | 0 / 0 / 0 |
| dev:success | NATIVE_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | NATIVE_minus_A | coverage | -0.000332 [-0.000332, -0.000332] | -0.000332 [-0.000332, -0.000332] | 0 / 0 / 0 |
| dev:success | NATIVE_minus_A | auc | +0.009762 [+0.009762, +0.009762] | +0.009762 [+0.009762, +0.009762] | 0 / 0 / 0 |
| dev:success | NATIVE_minus_A | fpr | -0.106329 [-0.106329, -0.106329] | -0.106329 [-0.106329, -0.106329] | 0 / 0 / 0 |
| dev:success | NATIVE_minus_A | bce | -0.305116 [-0.305116, -0.305116] | -0.305116 [-0.305116, -0.305116] | 0 / 0 / 0 |
| dev:success | MEAN_minus_A | iou | +0.074208 [+0.074208, +0.074208] | +0.074208 [+0.074208, +0.074208] | 0 / 0 / 0 |
| dev:success | MEAN_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | MEAN_minus_A | coverage | -0.003477 [-0.003477, -0.003477] | -0.003477 [-0.003477, -0.003477] | 0 / 0 / 0 |
| dev:success | MEAN_minus_A | auc | +0.009770 [+0.009770, +0.009770] | +0.009770 [+0.009770, +0.009770] | 0 / 0 / 0 |
| dev:success | MEAN_minus_A | fpr | -0.216555 [-0.216555, -0.216555] | -0.216555 [-0.216555, -0.216555] | 0 / 0 / 0 |
| dev:success | MEAN_minus_A | bce | -0.321619 [-0.321619, -0.321619] | -0.321619 [-0.321619, -0.321619] | 0 / 0 / 0 |
| dev:success | ROLL_minus_A | iou | +0.077677 [+0.077677, +0.077677] | +0.077677 [+0.077677, +0.077677] | 0 / 0 / 0 |
| dev:success | ROLL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | ROLL_minus_A | coverage | -0.000874 [-0.000874, -0.000874] | -0.000874 [-0.000874, -0.000874] | 0 / 0 / 0 |
| dev:success | ROLL_minus_A | auc | +0.008624 [+0.008624, +0.008624] | +0.008624 [+0.008624, +0.008624] | 0 / 0 / 0 |
| dev:success | ROLL_minus_A | fpr | -0.218725 [-0.218725, -0.218725] | -0.218725 [-0.218725, -0.218725] | 0 / 0 / 0 |
| dev:success | ROLL_minus_A | bce | -0.325135 [-0.325135, -0.325135] | -0.325135 [-0.325135, -0.325135] | 0 / 0 / 0 |
| dev:success | FULL_minus_NATIVE | iou | +0.052270 [+0.052270, +0.052270] | +0.052270 [+0.052270, +0.052270] | 0 / 0 / 0 |
| dev:success | FULL_minus_NATIVE | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | FULL_minus_NATIVE | coverage | -0.004193 [-0.004193, -0.004193] | -0.004193 [-0.004193, -0.004193] | 0 / 0 / 0 |
| dev:success | FULL_minus_NATIVE | auc | -0.000009 [-0.000009, -0.000009] | -0.000009 [-0.000009, -0.000009] | 0 / 0 / 0 |
| dev:success | FULL_minus_NATIVE | fpr | -0.122549 [-0.122549, -0.122549] | -0.122549 [-0.122549, -0.122549] | 0 / 0 / 0 |
| dev:success | FULL_minus_NATIVE | bce | -0.010930 [-0.010930, -0.010930] | -0.010930 [-0.010930, -0.010930] | 0 / 0 / 0 |
| dev:success | FULL_minus_MEAN | iou | +0.001438 [+0.001438, +0.001438] | +0.001438 [+0.001438, +0.001438] | 0 / 0 / 0 |
| dev:success | FULL_minus_MEAN | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | FULL_minus_MEAN | coverage | -0.001048 [-0.001048, -0.001048] | -0.001048 [-0.001048, -0.001048] | 0 / 0 / 0 |
| dev:success | FULL_minus_MEAN | auc | -0.000016 [-0.000016, -0.000016] | -0.000016 [-0.000016, -0.000016] | 0 / 0 / 0 |
| dev:success | FULL_minus_MEAN | fpr | -0.012323 [-0.012323, -0.012323] | -0.012323 [-0.012323, -0.012323] | 0 / 0 / 0 |
| dev:success | FULL_minus_MEAN | bce | +0.005573 [+0.005573, +0.005573] | +0.005573 [+0.005573, +0.005573] | 0 / 0 / 0 |
| dev:success | FULL_minus_ROLL | iou | -0.002031 [-0.002031, -0.002031] | -0.002031 [-0.002031, -0.002031] | 0 / 0 / 0 |
| dev:success | FULL_minus_ROLL | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:success | FULL_minus_ROLL | coverage | -0.003651 [-0.003651, -0.003651] | -0.003651 [-0.003651, -0.003651] | 0 / 0 / 0 |
| dev:success | FULL_minus_ROLL | auc | +0.001130 [+0.001130, +0.001130] | +0.001130 [+0.001130, +0.001130] | 0 / 0 / 0 |
| dev:success | FULL_minus_ROLL | fpr | -0.010152 [-0.010152, -0.010152] | -0.010152 [-0.010152, -0.010152] | 0 / 0 / 0 |
| dev:success | FULL_minus_ROLL | bce | +0.009089 [+0.009089, +0.009089] | +0.009089 [+0.009089, +0.009089] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_A | iou | +0.543287 [+0.318517, +0.768058] | +0.543287 [+0.318517, +0.768058] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_A | coverage | -1.077910 [-2.155821, +0.000000] | -1.077910 [-2.155821, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_A | auc | +0.053982 [-0.019119, +0.127082] | +0.053982 [-0.019119, +0.127082] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_A | fpr | -1.297458 [-2.134511, -0.460405] | -1.297458 [-2.134511, -0.460405] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_A | bce | -0.175647 [-0.339019, -0.012275] | -0.175647 [-0.339019, -0.012275] | 0 / 0 / 0 |
| dev:box_good_mask_bad | NATIVE_minus_A | iou | +0.450539 [+0.190743, +0.710335] | +0.450539 [+0.190743, +0.710335] | 0 / 0 / 0 |
| dev:box_good_mask_bad | NATIVE_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | NATIVE_minus_A | coverage | -1.007721 [-2.015442, +0.000000] | -1.007721 [-2.015442, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | NATIVE_minus_A | auc | +0.055356 [-0.018598, +0.129310] | +0.055356 [-0.018598, +0.129310] | 0 / 0 / 0 |
| dev:box_good_mask_bad | NATIVE_minus_A | fpr | -1.133070 [-1.989896, -0.276243] | -1.133070 [-1.989896, -0.276243] | 0 / 0 / 0 |
| dev:box_good_mask_bad | NATIVE_minus_A | bce | -0.166900 [-0.326440, -0.007359] | -0.166900 [-0.326440, -0.007359] | 0 / 0 / 0 |
| dev:box_good_mask_bad | MEAN_minus_A | iou | +0.548064 [+0.318517, +0.777611] | +0.548064 [+0.318517, +0.777611] | 0 / 0 / 0 |
| dev:box_good_mask_bad | MEAN_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | MEAN_minus_A | coverage | -1.075404 [-2.150807, +0.000000] | -1.075404 [-2.150807, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | MEAN_minus_A | auc | +0.054359 [-0.019249, +0.127966] | +0.054359 [-0.019249, +0.127966] | 0 / 0 / 0 |
| dev:box_good_mask_bad | MEAN_minus_A | fpr | -1.300350 [-2.140295, -0.460405] | -1.300350 [-2.140295, -0.460405] | 0 / 0 / 0 |
| dev:box_good_mask_bad | MEAN_minus_A | bce | -0.176026 [-0.339199, -0.012852] | -0.176026 [-0.339199, -0.012852] | 0 / 0 / 0 |
| dev:box_good_mask_bad | ROLL_minus_A | iou | +0.531110 [+0.254568, +0.807653] | +0.531110 [+0.254568, +0.807653] | 0 / 0 / 0 |
| dev:box_good_mask_bad | ROLL_minus_A | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | ROLL_minus_A | coverage | -1.057856 [-2.115712, +0.000000] | -1.057856 [-2.115712, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | ROLL_minus_A | auc | +0.068271 [-0.015347, +0.151888] | +0.068271 [-0.015347, +0.151888] | 0 / 0 / 0 |
| dev:box_good_mask_bad | ROLL_minus_A | fpr | -1.257202 [-2.146080, -0.368324] | -1.257202 [-2.146080, -0.368324] | 0 / 0 / 0 |
| dev:box_good_mask_bad | ROLL_minus_A | bce | -0.179146 [-0.343227, -0.015064] | -0.179146 [-0.343227, -0.015064] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_NATIVE | iou | +0.092748 [+0.057722, +0.127774] | +0.092748 [+0.057722, +0.127774] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_NATIVE | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_NATIVE | coverage | -0.070190 [-0.140379, +0.000000] | -0.070190 [-0.140379, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_NATIVE | auc | -0.001374 [-0.002227, -0.000520] | -0.001374 [-0.002227, -0.000520] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_NATIVE | fpr | -0.164388 [-0.184162, -0.144615] | -0.164388 [-0.184162, -0.144615] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_NATIVE | bce | -0.008747 [-0.012579, -0.004915] | -0.008747 [-0.012579, -0.004915] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_MEAN | iou | -0.004777 [-0.009553, +0.000000] | -0.004777 [-0.009553, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_MEAN | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_MEAN | coverage | -0.002507 [-0.005014, +0.000000] | -0.002507 [-0.005014, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_MEAN | auc | -0.000377 [-0.000883, +0.000130] | -0.000377 [-0.000883, +0.000130] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_MEAN | fpr | +0.002892 [+0.000000, +0.005785] | +0.002892 [+0.000000, +0.005785] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_MEAN | bce | +0.000379 [+0.000179, +0.000578] | +0.000379 [+0.000179, +0.000578] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_ROLL | iou | +0.012177 [-0.039595, +0.063948] | +0.012177 [-0.039595, +0.063948] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_ROLL | mask75 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_ROLL | coverage | -0.020054 [-0.040108, +0.000000] | -0.020054 [-0.040108, +0.000000] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_ROLL | auc | -0.014289 [-0.024805, -0.003772] | -0.014289 [-0.024805, -0.003772] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_ROLL | fpr | -0.040256 [-0.092081, +0.011569] | -0.040256 [-0.092081, +0.011569] | 0 / 0 / 0 |
| dev:box_good_mask_bad | FULL_minus_ROLL | bce | +0.003499 [+0.002789, +0.004208] | +0.003499 [+0.002789, +0.004208] | 0 / 0 / 0 |

## Interpretation boundaries

- MEAN is constant evidence before multiplication by K, not a constant mask-logit bias. Its projected correction may remain spatially nonuniform.
- ROLL retains every candidate's evidence multiset and mean, but breaks its spatial correspondence with fixed K. A loss under ROLL demonstrates sensitivity to this intervention, not by itself useful segmentation or an identified failure cause.
- The native branch and extra head were jointly trained. FULL versus NATIVE is a same-checkpoint path ablation, not an estimate of independently training a native-only model.
- Fit is the first 128 planned fit images; dev is the existing 256 planned images (smoke uses only its first two effective images). These are reused research data, not an independent test.
- All groups are defined from original A only. Invalid operators, empty corrected masks and undefined AUC cases stay in the candidate population; undefined metrics have separate counts.
- Per-candidate mean/std/RMS, centered-evidence energy share, coefficient increments and A_full-projected 16x16 logit RMS are saved without using GT to construct the interventions.
- The 16x16 operator diagnostic is not the normal-image mask metric. No threshold, sample, checkpoint or parameter is selected from these outcomes.
- No new training, model design or automatic follow-up is authorized by this audit.
