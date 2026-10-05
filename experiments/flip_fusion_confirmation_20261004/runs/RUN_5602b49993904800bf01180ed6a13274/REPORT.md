# Frozen A/MIX confirmation

1536 planned images; 1525 effective images; 11034 official TAL candidates.

Same frozen original model, full-canvas horizontal flip, source matcher and equal continuous-logit fusion. No training or projection.

| Group | Images / candidates | Macro IoU change pp [95% CI] | Candidate IoU change pp | Repair / damage | Coverage pp | AUC pp | FPR pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 1525 / 11034 | +0.0077 [-0.0541, +0.0664] | +0.0162 | 146 / 133 | -0.1636 | +0.0726 | -0.1674 |
| box_good_mask_bad | 817 / 2158 | +0.3330 [-0.0225, +0.7117] | +0.2707 | 124 / 0 | +0.1639 | +0.2871 | -0.0119 |
| original_success | 1497 / 6831 | -0.0314 [-0.0761, +0.0094] | -0.0434 | 0 / 133 | -0.1966 | +0.0230 | -0.2201 |
| original_failure | 947 / 4203 | +0.1779 [-0.1185, +0.4918] | +0.1131 | 146 / 0 | +0.0278 | +0.2727 | +0.0407 |
| retained_anyclass | 1525 / 10618 | +0.0036 [-0.0539, +0.0562] | -0.0031 | 136 / 129 | -0.1633 | +0.0800 | -0.1628 |
| retained_exactclass | 1525 / 10618 | +0.0036 [-0.0539, +0.0562] | -0.0031 | 136 / 129 | -0.1633 | +0.0800 | -0.1628 |
| retained_exactclass_correct | 1522 / 10224 | +0.0164 [-0.0329, +0.0625] | +0.0145 | 132 / 119 | -0.1451 | +0.0900 | -0.1552 |
| P3 | 679 / 3425 | -0.0621 [-0.1842, +0.0538] | -0.0916 | 57 / 54 | +0.0921 | +0.3111 | +0.6133 |
| P4 | 1084 / 4287 | +0.0829 [-0.0014, +0.1677] | +0.0991 | 68 / 57 | -0.1306 | +0.0440 | -0.2823 |
| P5 | 1340 / 3322 | +0.0010 [-0.0849, +0.0828] | +0.0203 | 21 / 22 | -0.3023 | +0.0207 | -0.4812 |
| small | 803 / 4623 | +0.0049 [-0.1144, +0.1161] | -0.0279 | 73 / 71 | +0.0360 | +0.2540 | +0.4197 |
| medium | 1086 / 3737 | +0.0323 [-0.0577, +0.1162] | +0.0782 | 55 / 48 | -0.1555 | +0.0497 | -0.2801 |
| large | 1236 / 2674 | -0.0131 [-0.1060, +0.0744] | +0.0058 | 18 / 14 | -0.3213 | -0.0008 | -0.5337 |

All candidates remain in the denominator after correction, including empty or degraded masks. Undefined AUC/FPR does not remove IoU observations.
The original-failure group has structurally zero Mask75 damage versus A; this does not establish absence of continuous degradation.
Retained subsets are secondary same-candidate diagnostics, not all output predictions or COCO AP. Source matching uses predicted classes only; GT class agreement is evaluation metadata.
Predeclared confirmation decisions and historical image exclusion provenance are specified by this Study's frozen protocol. No automatic method changes follow.
