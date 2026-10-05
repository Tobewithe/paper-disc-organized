# Frozen A/MIX confirmation

2 planned images; 2 effective images; 7 official TAL candidates.

Same frozen original model, full-canvas horizontal flip, source matcher and equal continuous-logit fusion. No training or projection.

| Group | Images / candidates | Macro IoU change pp [95% CI] | Candidate IoU change pp | Repair / damage | Coverage pp | AUC pp | FPR pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 2 / 7 | +0.8278 [+0.0876, +1.5680] | +0.5106 | 0 / 0 | +0.6945 | +0.1378 | -0.3755 |
| box_good_mask_bad | 2 / 2 | +1.2869 [-0.0494, +2.6232] | +1.2869 | 0 / 0 | +1.3024 | +0.1060 | -0.5906 |
| original_success | 1 / 4 | +0.1219 [+0.1219, +0.1219] | +0.1219 | 0 / 0 | -0.1154 | +0.0162 | -0.4558 |
| original_failure | 2 / 3 | +0.7593 [-0.0494, +1.5680] | +1.0289 | 0 / 0 | +1.0034 | -0.0298 | -0.0026 |
| retained_anyclass | 2 / 7 | +0.8278 [+0.0876, +1.5680] | +0.5106 | 0 / 0 | +0.6945 | +0.1378 | -0.3755 |
| retained_exactclass | 2 / 7 | +0.8278 [+0.0876, +1.5680] | +0.5106 | 0 / 0 | +0.6945 | +0.1378 | -0.3755 |
| retained_exactclass_correct | 2 / 6 | +0.7596 [-0.0487, +1.5680] | +0.4902 | 0 / 0 | +0.7120 | +0.1330 | -0.2385 |
| P3 | 1 / 1 | +0.5128 [+0.5128, +0.5128] | +0.5128 | 0 / 0 | +0.7519 | +0.0715 | +0.6944 |
| P4 | 2 / 3 | +1.5032 [+0.3833, +2.6232] | +1.1299 | 0 / 0 | +0.8629 | +0.3035 | -1.3973 |
| P5 | 1 / 3 | -0.1095 [-0.1095, -0.1095] | -0.1095 | 0 / 0 | +0.2132 | -0.1074 | +0.3090 |
| small | 1 / 2 | +1.5680 [+1.5680, +1.5680] | +1.5680 | 0 / 0 | +1.3500 | +0.3433 | -0.4815 |
| medium | 1 / 2 | +0.3833 [+0.3833, +0.3833] | +0.3833 | 0 / 0 | -0.2222 | -0.0081 | -1.1371 |
| large | 1 / 3 | -0.1095 [-0.1095, -0.1095] | -0.1095 | 0 / 0 | +0.2132 | -0.1074 | +0.3090 |

All candidates remain in the denominator after correction, including empty or degraded masks. Undefined AUC/FPR does not remove IoU observations.
The original-failure group has structurally zero Mask75 damage versus A; this does not establish absence of continuous degradation.
Retained subsets are secondary same-candidate diagnostics, not all output predictions or COCO AP. Source matching uses predicted classes only; GT class agreement is evaluation metadata.
Predeclared confirmation decisions and historical image exclusion provenance are specified by this Study's frozen protocol. No automatic method changes follow.
