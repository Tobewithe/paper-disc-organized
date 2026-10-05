# Prototype multiplicity: local sensitivity audit

This is a same-image GT-assisted diagnostic. Each image starts from the same original parameters; three equal-norm temporary negative-gradient displacements are evaluated and then restored. No model is trained or selected.

Planned images: 2; effective one-to-one images: 2; evaluated one-to-one candidates: 6.

| Group | Comparison | Macro IoU delta [95% CI], pp | Repair/damage | Coverage delta, pp | AUC delta, pp | FPR delta, pp |
|---|---|---:|---:|---:|---:|---:|
| all | M_minus_A | +0.4977 [+0.2901, +0.7052] | 0/0 | +0.3394 | +0.1369 | -0.1908 |
| all | E_minus_A | +0.4331 [+0.2901, +0.5761] | 0/0 | +0.2807 | +0.1572 | -0.2196 |
| all | O_minus_A | +0.4509 [+0.3283, +0.5735] | 0/0 | +0.2681 | +0.1647 | -0.2562 |
| all | E_minus_M | -0.0645 [-0.1291, +0.0000] | 0/0 | -0.0587 | +0.0202 | -0.0288 |
| all | O_minus_M | -0.0468 [-0.1318, +0.0382] | 0/0 | -0.0713 | +0.0278 | -0.0653 |
| box_good_mask_bad | M_minus_A | +0.0724 [+0.0724, +0.0724] | 0/0 | +0.1075 | +0.0026 | +0.0000 |
| box_good_mask_bad | E_minus_A | -0.1856 [-0.1856, -0.1856] | 0/0 | +0.0000 | -0.0013 | +0.0457 |
| box_good_mask_bad | O_minus_A | -0.1856 [-0.1856, -0.1856] | 0/0 | +0.0000 | -0.0012 | +0.0457 |
| box_good_mask_bad | E_minus_M | -0.2580 [-0.2580, -0.2580] | 0/0 | -0.1075 | -0.0039 | +0.0457 |
| box_good_mask_bad | O_minus_M | -0.2580 [-0.2580, -0.2580] | 0/0 | -0.1075 | -0.0038 | +0.0457 |
| original_success | M_minus_A | +0.6559 [+0.2901, +1.0216] | 0/0 | +0.5356 | +0.1945 | -0.1523 |
| original_success | E_minus_A | +0.6236 [+0.2901, +0.9570] | 0/0 | +0.4745 | +0.2258 | -0.2069 |
| original_success | O_minus_A | +0.6407 [+0.3283, +0.9530] | 0/0 | +0.4607 | +0.2379 | -0.2445 |
| original_success | E_minus_M | -0.0323 [-0.0646, +0.0000] | 0/0 | -0.0611 | +0.0313 | -0.0546 |
| original_success | O_minus_M | -0.0152 [-0.0686, +0.0382] | 0/0 | -0.0750 | +0.0435 | -0.0922 |
| original_failure | M_minus_A | +0.0724 [+0.0724, +0.0724] | 0/0 | +0.1075 | +0.0026 | +0.0000 |
| original_failure | E_minus_A | -0.1856 [-0.1856, -0.1856] | 0/0 | +0.0000 | -0.0013 | +0.0457 |
| original_failure | O_minus_A | -0.1856 [-0.1856, -0.1856] | 0/0 | +0.0000 | -0.0012 | +0.0457 |
| original_failure | E_minus_M | -0.2580 [-0.2580, -0.2580] | 0/0 | -0.1075 | -0.0039 | +0.0457 |
| original_failure | O_minus_M | -0.2580 [-0.2580, -0.2580] | 0/0 | -0.1075 | -0.0038 | +0.0457 |
| P3 | M_minus_A | +1.6878 [+1.6878, +1.6878] | 0/0 | +1.9704 | +0.6915 | +0.0000 |
| P3 | E_minus_A | +1.6878 [+1.6878, +1.6878] | 0/0 | +1.9704 | +0.8201 | +0.0000 |
| P3 | O_minus_A | +1.6878 [+1.6878, +1.6878] | 0/0 | +1.9704 | +0.8764 | +0.0000 |
| P3 | E_minus_M | +0.0000 [+0.0000, +0.0000] | 0/0 | +0.0000 | +0.1286 | +0.0000 |
| P3 | O_minus_M | +0.0000 [+0.0000, +0.0000] | 0/0 | +0.0000 | +0.1849 | +0.0000 |
| P4 | M_minus_A | +0.4526 [+0.4526, +0.4526] | 0/0 | -0.1100 | +0.0146 | -0.6539 |
| P4 | E_minus_A | +0.4526 [+0.4526, +0.4526] | 0/0 | -0.1100 | +0.0146 | -0.6539 |
| P4 | O_minus_A | +0.5232 [+0.5232, +0.5232] | 0/0 | -0.1572 | +0.0116 | -0.7876 |
| P4 | E_minus_M | +0.0000 [+0.0000, +0.0000] | 0/0 | +0.0000 | +0.0000 | +0.0000 |
| P4 | O_minus_M | +0.0706 [+0.0706, +0.0706] | 0/0 | -0.0472 | -0.0030 | -0.1337 |
| P5 | M_minus_A | +0.2114 [+0.2089, +0.2139] | 0/0 | +0.0439 | +0.0289 | -0.1228 |
| P5 | E_minus_A | +0.1146 [+0.0203, +0.2089] | 0/0 | -0.0441 | +0.0271 | -0.1659 |
| P5 | O_minus_A | +0.1236 [+0.0163, +0.2308] | 0/0 | -0.0512 | +0.0251 | -0.1873 |
| P5 | E_minus_M | -0.0968 [-0.1936, +0.0000] | 0/0 | -0.0880 | -0.0018 | -0.0432 |
| P5 | O_minus_M | -0.0878 [-0.1976, +0.0219] | 0/0 | -0.0951 | -0.0038 | -0.0646 |

M: official one-to-many positive-candidate mean. E: the identical coefficients, owners and pixel losses, averaged within GT and then equally across represented GTs. O: frozen one-to-one coefficient loss artificially differentiated through P as a diagnostic direction (the original forward detaches P).

Parameters, source features, coefficients, boxes, labels and assignments are restored/frozen. BN affine parameters and running buffers are both frozen; unused semseg is frozen. M is only the official many mask term, not the full original prototype-training gradient including semantic supervision and upstream paths. Normal-image evaluation always uses original one-to-one coefficients and boxes with the temporarily changed complete prototype. The variant helper's A metrics are discarded.

Parameter- and prototype-output-space gradient cosines are reported separately. Neither negative cosine nor local improvement identifies the history of pretraining, a universal root cause, or generalization. M/E use all their own official many positives; different represented GTs from O are disclosed.

Fixed diagnostic decision: {"direction_signal": false, "local_task_signal": false, "local_all_guard": false, "finite_difference_interpretable": true, "finite_difference_joint_faithful_images": 2, "finite_difference_joint_faithful_fraction": 1.0, "finite_difference_joint_minimum_fraction": 0.9, "mechanism_activated": true, "multiplicity_varied_images": 1, "multiplicity_varied_fraction": 0.5, "diagnostic_signal": false, "outcome": "stop_current_multiplicity_hypothesis", "automatic_training": false, "automatic_followup": false, "interpretation": "Only same-image GT-assisted local sensitivity; no learned or held-out method efficacy"}
