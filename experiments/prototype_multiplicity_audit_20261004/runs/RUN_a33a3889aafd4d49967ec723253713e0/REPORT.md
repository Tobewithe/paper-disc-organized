# Prototype multiplicity: local sensitivity audit

This is a same-image GT-assisted diagnostic. Each image starts from the same original parameters; three equal-norm temporary negative-gradient displacements are evaluated and then restored. No model is trained or selected.

Planned images: 128; effective one-to-one images: 127; evaluated one-to-one candidates: 756.

| Group | Comparison | Macro IoU delta [95% CI], pp | Repair/damage | Coverage delta, pp | AUC delta, pp | FPR delta, pp |
|---|---|---:|---:|---:|---:|---:|
| all | M_minus_A | +0.2983 [+0.1871, +0.3947] | 1/0 | +0.0658 | +0.1411 | -0.2993 |
| all | E_minus_A | +0.2984 [+0.1863, +0.3943] | 1/0 | +0.0695 | +0.1402 | -0.2994 |
| all | O_minus_A | +0.3423 [+0.2293, +0.4468] | 1/0 | +0.1259 | +0.1666 | -0.2572 |
| all | E_minus_M | +0.0001 [-0.0092, +0.0098] | 0/0 | +0.0037 | -0.0009 | -0.0001 |
| all | O_minus_M | +0.0440 [+0.0181, +0.0713] | 0/0 | +0.0601 | +0.0255 | +0.0421 |
| box_good_mask_bad | M_minus_A | +0.2725 [-0.3404, +0.6662] | 1/0 | +0.1142 | +0.3150 | -0.1356 |
| box_good_mask_bad | E_minus_A | +0.2856 [-0.3092, +0.6744] | 1/0 | +0.1169 | +0.3347 | -0.1544 |
| box_good_mask_bad | O_minus_A | +0.4462 [-0.1332, +0.8604] | 1/0 | +0.3010 | +0.4028 | -0.1147 |
| box_good_mask_bad | E_minus_M | +0.0131 [-0.0272, +0.0541] | 0/0 | +0.0028 | +0.0197 | -0.0189 |
| box_good_mask_bad | O_minus_M | +0.1737 [+0.0175, +0.4101] | 0/0 | +0.1868 | +0.0877 | +0.0208 |
| original_success | M_minus_A | +0.2502 [+0.1936, +0.3086] | 0/0 | +0.0799 | +0.0723 | -0.2799 |
| original_success | E_minus_A | +0.2413 [+0.1841, +0.2982] | 0/0 | +0.0756 | +0.0699 | -0.2803 |
| original_success | O_minus_A | +0.2442 [+0.1857, +0.3035] | 0/0 | +0.0918 | +0.0739 | -0.2344 |
| original_success | E_minus_M | -0.0089 [-0.0171, -0.0016] | 0/0 | -0.0043 | -0.0024 | -0.0005 |
| original_success | O_minus_M | -0.0059 [-0.0186, +0.0064] | 0/0 | +0.0119 | +0.0016 | +0.0455 |
| original_failure | M_minus_A | +0.3740 [-0.1195, +0.7226] | 1/0 | +0.0114 | +0.3164 | -0.1215 |
| original_failure | E_minus_A | +0.3882 [-0.1237, +0.7336] | 1/0 | +0.0300 | +0.3175 | -0.1254 |
| original_failure | O_minus_A | +0.5429 [+0.0052, +0.9069] | 1/0 | +0.2148 | +0.4074 | -0.0392 |
| original_failure | E_minus_M | +0.0142 [-0.0155, +0.0466] | 0/0 | +0.0186 | +0.0011 | -0.0038 |
| original_failure | O_minus_M | +0.1690 [+0.0727, +0.2779] | 0/0 | +0.2034 | +0.0911 | +0.0824 |
| P3 | M_minus_A | +0.3666 [+0.2377, +0.5246] | 1/0 | +0.3227 | +0.3232 | +0.0473 |
| P3 | E_minus_A | +0.3933 [+0.2617, +0.5560] | 1/0 | +0.3553 | +0.3196 | +0.0510 |
| P3 | O_minus_A | +0.4858 [+0.3157, +0.6873] | 1/0 | +0.5335 | +0.4280 | +0.1651 |
| P3 | E_minus_M | +0.0267 [-0.0130, +0.0746] | 0/0 | +0.0325 | -0.0035 | +0.0037 |
| P3 | O_minus_M | +0.1192 [+0.0324, +0.2196] | 0/0 | +0.2108 | +0.1048 | +0.1178 |
| P4 | M_minus_A | +0.3729 [+0.2841, +0.4621] | 0/0 | +0.1544 | +0.1459 | -0.2004 |
| P4 | E_minus_A | +0.3712 [+0.2795, +0.4647] | 0/0 | +0.1549 | +0.1407 | -0.2141 |
| P4 | O_minus_A | +0.4706 [+0.3551, +0.5902] | 0/0 | +0.3083 | +0.1633 | -0.0661 |
| P4 | E_minus_M | -0.0017 [-0.0318, +0.0348] | 0/0 | +0.0005 | -0.0052 | -0.0136 |
| P4 | O_minus_M | +0.0977 [+0.0352, +0.1670] | 0/0 | +0.1539 | +0.0174 | +0.1344 |
| P5 | M_minus_A | +0.2322 [+0.0868, +0.3534] | 0/0 | -0.0170 | +0.0786 | -0.3587 |
| P5 | E_minus_A | +0.2251 [+0.0824, +0.3450] | 0/0 | -0.0192 | +0.0762 | -0.3577 |
| P5 | O_minus_A | +0.2302 [+0.0767, +0.3573] | 0/0 | -0.0289 | +0.0790 | -0.3750 |
| P5 | E_minus_M | -0.0072 [-0.0128, -0.0025] | 0/0 | -0.0021 | -0.0024 | +0.0010 |
| P5 | O_minus_M | -0.0021 [-0.0186, +0.0150] | 0/0 | -0.0118 | +0.0004 | -0.0163 |

M: official one-to-many positive-candidate mean. E: the identical coefficients, owners and pixel losses, averaged within GT and then equally across represented GTs. O: frozen one-to-one coefficient loss artificially differentiated through P as a diagnostic direction (the original forward detaches P).

Parameters, source features, coefficients, boxes, labels and assignments are restored/frozen. BN affine parameters and running buffers are both frozen; unused semseg is frozen. M is only the official many mask term, not the full original prototype-training gradient including semantic supervision and upstream paths. Normal-image evaluation always uses original one-to-one coefficients and boxes with the temporarily changed complete prototype. The variant helper's A metrics are discarded.

Parameter- and prototype-output-space gradient cosines are reported separately. Neither negative cosine nor local improvement identifies the history of pretraining, a universal root cause, or generalization. M/E use all their own official many positives; different represented GTs from O are disclosed.

Fixed diagnostic decision: {"direction_signal": false, "local_task_signal": false, "local_all_guard": true, "finite_difference_interpretable": false, "finite_difference_joint_faithful_images": 101, "finite_difference_joint_faithful_fraction": 0.7952755905511811, "finite_difference_joint_minimum_fraction": 0.9, "mechanism_activated": true, "multiplicity_varied_images": 40, "multiplicity_varied_fraction": 0.31496062992125984, "diagnostic_signal": false, "outcome": "inconclusive_local_linearization", "automatic_training": false, "automatic_followup": false, "interpretation": "Only same-image GT-assisted local sensitivity; no learned or held-out method efficacy"}
