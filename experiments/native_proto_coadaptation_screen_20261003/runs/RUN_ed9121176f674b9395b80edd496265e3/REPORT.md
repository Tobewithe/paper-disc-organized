# Native prototype/coefficient coadaptation screen

P and PC are evaluated with their own trained full prototypes. A uses original P and c0; N reuses the same-candidate epoch-3 native-coefficient control from the prior screen.

256 planned / 253 effective dev images; 1816 fixed official candidates. These development images were previously viewed; no blind confirmation or COCO AP claim.

All differences are percentage points. The 1000 paired-image bootstrap intervals are descriptive; candidate-mean intervals also resample whole images.

| Group | Comparison | Macro IoU Δ [95% CI] | Candidate IoU Δ | Net Mask75 | Repair/damage | Coverage Δ | AUC Δ | FPR Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | P_minus_A | -0.1679 [-0.3198, +0.0116] | -0.3122 | -9 | 25/34 | -0.5403 | -0.1990 | -0.3794 |
| all | PC_minus_A | -0.2537 [-0.4393, -0.0454] | -0.4232 | -10 | 23/33 | -0.6922 | -0.1900 | -0.5053 |
| all | P_minus_N | -0.1364 [-0.2826, +0.0032] | -0.1739 | -7 | 26/33 | -0.2932 | -0.1459 | -0.0980 |
| all | PC_minus_N | -0.2222 [-0.3505, -0.1099] | -0.2850 | -8 | 20/28 | -0.4451 | -0.1369 | -0.2239 |
| all | PC_minus_P | -0.0858 [-0.2156, +0.0241] | -0.1111 | -1 | 14/15 | -0.1519 | +0.0090 | -0.1259 |
| box_good_mask_bad | P_minus_A | +0.0131 [-0.6068, +0.7462] | -0.1485 | 20 | 20/0 | -1.1402 | -0.6255 | -1.4377 |
| box_good_mask_bad | PC_minus_A | -0.1960 [-0.9248, +0.6072] | -0.2949 | 18 | 18/0 | -1.2503 | -0.7505 | -1.2933 |
| box_good_mask_bad | P_minus_N | -0.0575 [-0.4955, +0.3995] | -0.0461 | 9 | 15/6 | -0.4045 | -0.3606 | -0.5150 |
| box_good_mask_bad | PC_minus_N | -0.2667 [-0.6746, +0.0935] | -0.1925 | 7 | 10/3 | -0.5146 | -0.4856 | -0.3706 |
| box_good_mask_bad | PC_minus_P | -0.2092 [-0.6349, +0.1443] | -0.1464 | -2 | 5/7 | -0.1101 | -0.1250 | +0.1444 |
| original_success | P_minus_A | -0.2619 [-0.3746, -0.1580] | -0.3871 | -34 | 0/34 | -0.4177 | -0.0672 | -0.1260 |
| original_success | PC_minus_A | -0.3550 [-0.5792, -0.1881] | -0.5378 | -33 | 0/33 | -0.5891 | -0.0563 | -0.2751 |
| original_success | P_minus_N | -0.1252 [-0.2387, -0.0256] | -0.1823 | -16 | 9/25 | -0.2066 | -0.0488 | +0.0433 |
| original_success | PC_minus_N | -0.2183 [-0.3788, -0.0892] | -0.3329 | -15 | 8/23 | -0.3780 | -0.0379 | -0.1058 |
| original_success | PC_minus_P | -0.0931 [-0.2441, +0.0117] | -0.1507 | 1 | 8/7 | -0.1714 | +0.0109 | -0.1492 |
| original_failure | P_minus_A | -0.0320 [-0.5234, +0.5644] | -0.1863 | 25 | 25/0 | -0.9931 | -0.6909 | -1.0937 |
| original_failure | PC_minus_A | -0.2188 [-0.7862, +0.3794] | -0.2310 | 23 | 23/0 | -1.2238 | -0.7861 | -1.1102 |
| original_failure | P_minus_N | -0.1781 [-0.5811, +0.1895] | -0.1600 | 9 | 17/8 | -0.4916 | -0.4525 | -0.3944 |
| original_failure | PC_minus_N | -0.3649 [-0.6802, -0.0849] | -0.2046 | 7 | 12/5 | -0.7223 | -0.5477 | -0.4108 |
| original_failure | PC_minus_P | -0.1869 [-0.5190, +0.1108] | -0.0446 | -2 | 6/8 | -0.2307 | -0.0952 | -0.0165 |

Evaluation scope:

- A is always the unchanged original model. Helper evaluations with variant P also compute an internal reference; all those reference metrics and classifications are discarded.
- P trains the native prototype path with original c0; PC jointly trains the native prototype and coefficient paths. The inference prefix, original boxes, assignment and candidate identities stay fixed.
- N is joined from the historical screen by all seven identity fields; it is not rerun or retrained. Same checkpoint budget is epoch 3, with no dev checkpoint selection.
- Normal masks use process_mask(upsample=True), original predicted boxes, original zero threshold and true letterbox inverse. IoU/coverage use original COCO masks. AUC/FPR use continuous input-grid logits on fixed predicted-box support.
- Original success/failure and box-good/mask-bad groups are defined exclusively by original A. No candidate is removed because it worsens or produces an empty mask; undefined AUC does not remove IoU.
- Prototypes change in this study, so these results do not establish fixed-P coefficient reachability. Three-epoch negative outcomes do not disprove the mechanism. Consider useful individual metric improvements together with their costs; not all metrics must improve.
- No threshold search, gate, new GT-based selection, full AP evaluation or automatic next experiment.
