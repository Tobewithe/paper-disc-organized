# Frozen local-view coefficient replay

Zero training. Frozen local-view and original-view reselection sources supply responses to direct rendering and the unchanged OGPS solver. Original P, target c0, predicted boxes and evaluation candidate identities remain fixed.

256 planned dev images, 253 effective images, 1816 candidates. These images were previously viewed; this is exploratory reuse, not a new blind test or COCO AP.

All deltas below are percentage points. Macro estimates and confidence intervals use the same image-equal statistic; candidate intervals in SUMMARY also resample whole images. The 1000-bootstrap intervals are descriptive across these multiple exploratory comparisons.

| Group | Comparison | Macro IoU Δ [95% CI] | Candidate IoU Δ | Net Mask75 | Repair/damage | Coverage Δ | AUC Δ | FPR Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | VIEW_SOLVE_minus_A | -3.6204 [-4.3176, -2.9848] | -4.1259 | -127 | 42/169 | -3.8185 | -2.5631 | +1.8892 |
| all | VIEW_SOLVE_minus_BASE_SOLVE | -3.2919 [-3.9756, -2.6675] | -3.8463 | -117 | 41/158 | -3.2788 | -2.4053 | +1.2653 |
| all | VIEW_SOLVE_minus_VIEW_FULL | -0.9153 [-1.1952, -0.6512] | -1.3372 | -60 | 34/94 | +0.8897 | -0.1686 | +4.7124 |
| all | VIEW_SOLVE_minus_VIEW_GRID | +3.3802 [+2.7795, +3.9883] | +1.8675 | 42 | 121/79 | +3.7820 | +1.1111 | +0.0839 |
| all | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | -2.2147 [-2.8839, -1.5801] | -2.2502 | -34 | 81/115 | -3.9612 | -1.9481 | -2.4598 |
| all | VIEW_FULL_minus_ORIG_RESELECT | -2.6212 [-3.3224, -1.9688] | -2.6533 | -67 | 82/149 | -4.5805 | -2.3251 | -2.7240 |
| all | ORIG_RESELECT_minus_A | -0.0840 [-0.2415, +0.0244] | -0.1354 | 0 | 1/1 | -0.1277 | -0.0695 | -0.0992 |
| all | ORIG_RESELECT_SOLVE_minus_A | -1.4057 [-1.6134, -1.2067] | -1.8757 | -93 | 16/109 | +0.1428 | -0.6150 | +4.3490 |
| all | BASE_SOLVE_minus_A | -0.3285 [-0.4207, -0.2376] | -0.2796 | -10 | 10/20 | -0.5397 | -0.1579 | +0.6239 |
| all | VIEW_FULL_minus_A | -2.7052 [-3.4047, -2.0579] | -2.7887 | -67 | 81/148 | -4.7082 | -2.3946 | -2.8232 |
| all | VIEW_GRID_minus_A | -7.0006 [-7.9143, -6.1595] | -5.9934 | -169 | 65/234 | -7.6005 | -3.6742 | +1.8053 |
| box_good_mask_bad | VIEW_SOLVE_minus_A | -5.0138 [-6.4503, -3.5834] | -5.5436 | 33 | 33/0 | -5.0292 | -6.5633 | +4.1211 |
| box_good_mask_bad | VIEW_SOLVE_minus_BASE_SOLVE | -4.7877 [-6.2134, -3.3125] | -5.2970 | 24 | 26/2 | -4.3361 | -6.3445 | +4.2678 |
| box_good_mask_bad | VIEW_SOLVE_minus_VIEW_FULL | -1.8566 [-2.6361, -1.1685] | -2.5410 | -26 | 6/32 | +0.7557 | -0.4665 | +5.6620 |
| box_good_mask_bad | VIEW_SOLVE_minus_VIEW_GRID | +1.6470 [+0.4220, +3.0993] | +0.5458 | -18 | 11/29 | +4.5390 | +0.8598 | +4.0114 |
| box_good_mask_bad | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | -3.1675 [-4.6569, -1.6959] | -3.1981 | 22 | 28/6 | -5.6812 | -5.3927 | -1.9971 |
| box_good_mask_bad | VIEW_FULL_minus_ORIG_RESELECT | -3.0788 [-4.6959, -1.3593] | -2.8031 | 58 | 59/1 | -5.6476 | -5.9848 | -1.4540 |
| box_good_mask_bad | ORIG_RESELECT_minus_A | -0.0784 [-0.2758, +0.0939] | -0.1994 | 1 | 1/0 | -0.1373 | -0.1120 | -0.0869 |
| box_good_mask_bad | ORIG_RESELECT_SOLVE_minus_A | -1.8463 [-2.3954, -1.3361] | -2.3455 | 11 | 11/0 | +0.6521 | -1.1705 | +6.1182 |
| box_good_mask_bad | BASE_SOLVE_minus_A | -0.2261 [-0.3897, -0.0689] | -0.2465 | 9 | 9/0 | -0.6930 | -0.2188 | -0.1467 |
| box_good_mask_bad | VIEW_FULL_minus_A | -3.1572 [-4.7552, -1.4213] | -3.0025 | 59 | 59/0 | -5.7849 | -6.0968 | -1.5408 |
| box_good_mask_bad | VIEW_GRID_minus_A | -6.6608 [-8.5344, -4.5774] | -6.0893 | 51 | 51/0 | -9.5681 | -7.4231 | +0.1098 |
| original_success | VIEW_SOLVE_minus_A | -3.5401 [-4.2843, -2.8240] | -3.9162 | -169 | 0/169 | -3.5618 | -1.3900 | +1.7307 |
| original_success | VIEW_SOLVE_minus_BASE_SOLVE | -3.2163 [-3.9547, -2.5014] | -3.5872 | -149 | 7/156 | -3.1228 | -1.2915 | +0.9280 |
| original_success | VIEW_SOLVE_minus_VIEW_FULL | -0.8738 [-1.1701, -0.5828] | -0.9753 | -21 | 23/44 | +0.7813 | -0.1466 | +4.5976 |
| original_success | VIEW_SOLVE_minus_VIEW_GRID | +3.6990 [+3.1475, +4.2454] | +2.9297 | 65 | 103/38 | +3.4133 | +1.2721 | -1.0730 |
| original_success | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | -2.1923 [-2.9013, -1.5005] | -2.1917 | -60 | 46/106 | -3.7570 | -1.0341 | -2.5702 |
| original_success | VIEW_FULL_minus_ORIG_RESELECT | -2.6595 [-3.3720, -1.9632] | -2.9216 | -147 | 1/148 | -4.3353 | -1.2407 | -2.8698 |
| original_success | ORIG_RESELECT_minus_A | -0.0068 [-0.0202, +0.0000] | -0.0193 | -1 | 0/1 | -0.0078 | -0.0027 | +0.0028 |
| original_success | ORIG_RESELECT_SOLVE_minus_A | -1.3478 [-1.5811, -1.1342] | -1.7244 | -109 | 0/109 | +0.1952 | -0.3559 | +4.3009 |
| original_success | BASE_SOLVE_minus_A | -0.3239 [-0.4138, -0.2249] | -0.3289 | -20 | 0/20 | -0.4390 | -0.0985 | +0.8026 |
| original_success | VIEW_FULL_minus_A | -2.6663 [-3.3850, -1.9640] | -2.9409 | -148 | 0/148 | -4.3432 | -1.2434 | -2.8670 |
| original_success | VIEW_GRID_minus_A | -7.2391 [-8.1340, -6.3777] | -6.8459 | -234 | 0/234 | -6.9751 | -2.6621 | +2.8037 |
| original_failure | VIEW_SOLVE_minus_A | -3.9953 [-5.0854, -2.9394] | -4.4779 | 42 | 42/0 | -4.5843 | -5.3149 | +2.6700 |
| original_failure | VIEW_SOLVE_minus_BASE_SOLVE | -3.7882 [-4.8954, -2.7085] | -4.2812 | 32 | 34/2 | -3.9900 | -4.9670 | +2.6923 |
| original_failure | VIEW_SOLVE_minus_VIEW_FULL | -1.3627 [-1.9759, -0.8236] | -1.9446 | -39 | 11/50 | +1.4129 | -0.4077 | +6.4055 |
| original_failure | VIEW_SOLVE_minus_VIEW_GRID | +1.4216 [+0.4249, +2.4682] | +0.0847 | -23 | 18/41 | +4.4432 | +0.3600 | +4.9897 |
| original_failure | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | -2.2847 [-3.4553, -1.1208] | -2.3484 | 26 | 35/9 | -5.0660 | -3.8799 | -3.4260 |
| original_failure | VIEW_FULL_minus_ORIG_RESELECT | -2.4446 [-3.5992, -1.2456] | -2.2031 | 80 | 81/1 | -5.6572 | -4.7316 | -3.3571 |
| original_failure | ORIG_RESELECT_minus_A | -0.1881 [-0.6805, +0.1700] | -0.3302 | 1 | 1/0 | -0.3400 | -0.1756 | -0.3784 |
| original_failure | ORIG_RESELECT_SOLVE_minus_A | -1.7107 [-2.3124, -1.1885] | -2.1295 | 16 | 16/0 | +0.4817 | -1.4350 | +6.0960 |
| original_failure | BASE_SOLVE_minus_A | -0.2071 [-0.3298, -0.0829] | -0.1967 | 10 | 10/0 | -0.5943 | -0.3479 | -0.0223 |
| original_failure | VIEW_FULL_minus_A | -2.6327 [-3.8164, -1.4148] | -2.5333 | 81 | 81/0 | -5.9972 | -4.9072 | -3.7355 |
| original_failure | VIEW_GRID_minus_A | -5.4170 [-6.7520, -4.0502] | -4.5626 | 65 | 65/0 | -9.0275 | -5.6749 | -2.3197 |

Arm definitions:

- A: original full prototype and original coefficient, with the original normal-image decoder.
- BASE_SOLVE: sigmoid(A8 c0) passed through the formal probability clipping and fixed ridge solver. Clipping can change c0, so this is not an identity arm.
- VIEW_FULL: local-view continuous responses reprojected to the original 640 input, with source-defined baseline fill outside observed support; fixed original box crop and identical original-image inverse.
- VIEW_GRID: pool sigmoid(VIEW_FULL) using the legacy 8×8 floor/ceil crop, form t8=logit(clamp(q8,.01,.99)), resize these logits into the same integer box, then crop/threshold and restore the original image.
- VIEW_SOLVE: exactly the same q8/t8 passed through the unchanged FP64 ridge (lambda=.003) into the original full prototype. No local-view coefficients or prototypes replace original P/c0.

- ORIG_RESELECT: reselect from the frozen original-image official top300 by predicted-box IoU, then source score and raw ID; decode that source coefficient with original full P, while retaining the target candidate's box and identity.
- ORIG_RESELECT_SOLVE: the same reselected source's full 640 input logits are transformed by sigmoid, the same 8x8 probability pooling, clipping and solver as VIEW_SOLVE; the regularization center stays the original target c0.

VIEW_FULL versus VIEW_GRID changes pooling, clipping and rendering jointly; it is not a pure resolution experiment. VIEW_SOLVE versus VIEW_GRID compares prototype projection with direct rendering of the same target. VIEW_SOLVE versus BASE_SOLVE checks added local-view information against self-projection/calibration effects.

VIEW_SOLVE versus ORIG_RESELECT_SOLVE and VIEW_FULL versus ORIG_RESELECT control original-image source reselection. A local-view gain over A alone cannot distinguish image-view effects from changing the source raw position. Source selection is GT-free in both views; this comparison still includes differences in source predictions induced by the view.

Raw class/score or GT never selects an evaluation subset after replay. Every fixed candidate remains, including empty masks and worsening predictions. GT is supplied only to metrics; the source receives an explicit GT-free field allowlist. Original successes/failures and the box-good/mask-bad stratum are defined by A.

AUC uses continuous input-space logits with original COCO labels on the fixed original predicted-box support. Undefined AUC entries are counted, not removed from IoU. Model execution completed is not a scientific mechanism claim; no further training is launched.
