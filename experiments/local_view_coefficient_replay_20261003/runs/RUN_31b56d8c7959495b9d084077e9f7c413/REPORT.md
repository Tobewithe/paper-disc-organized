# Frozen local-view coefficient replay

Zero training. Frozen local-view and original-view reselection sources supply responses to direct rendering and the unchanged OGPS solver. Original P, target c0, predicted boxes and evaluation candidate identities remain fixed.

SMOKE ONLY: 2 planned dev images, 2 effective images, 7 candidates. These images were previously viewed; this is exploratory reuse, not a new blind test or COCO AP.

All deltas below are percentage points. Macro estimates and confidence intervals use the same image-equal statistic; candidate intervals in SUMMARY also resample whole images. The 1000-bootstrap intervals are descriptive across these multiple exploratory comparisons.

| Group | Comparison | Macro IoU Δ [95% CI] | Candidate IoU Δ | Net Mask75 | Repair/damage | Coverage Δ | AUC Δ | FPR Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | VIEW_SOLVE_minus_A | -6.7958 [-11.5176, -2.0740] | -4.7722 | 0 | 0/0 | +0.2093 | -5.4630 | +16.7059 |
| all | VIEW_SOLVE_minus_BASE_SOLVE | -7.0780 [-11.9734, -2.1826] | -4.9800 | 0 | 0/0 | +0.5957 | -5.1674 | +17.5368 |
| all | VIEW_SOLVE_minus_VIEW_FULL | -2.3462 [-3.4120, -1.2803] | -1.8894 | 0 | 0/0 | +1.0783 | +3.0421 | +6.9206 |
| all | VIEW_SOLVE_minus_VIEW_GRID | +0.3473 [-0.6822, +1.3768] | +0.7885 | 0 | 0/0 | +2.6173 | +5.0984 | +2.8643 |
| all | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | -3.6153 [-6.1934, -1.0373] | -2.5105 | 0 | 0/0 | +0.2620 | -4.4027 | +11.7977 |
| all | VIEW_FULL_minus_ORIG_RESELECT | -4.4496 [-8.1055, -0.7937] | -2.8828 | 0 | 0/0 | -0.8690 | -8.5052 | +9.7853 |
| all | ORIG_RESELECT_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 | 0/0 | +0.0000 | +0.0000 | +0.0000 |
| all | ORIG_RESELECT_SOLVE_minus_A | -3.1805 [-5.3242, -1.0367] | -2.2617 | 0 | 0/0 | -0.0528 | -1.0604 | +4.9081 |
| all | BASE_SOLVE_minus_A | +0.2822 [+0.1085, +0.4559] | +0.2078 | 0 | 0/0 | -0.3864 | -0.2957 | -0.8310 |
| all | VIEW_FULL_minus_A | -4.4496 [-8.1055, -0.7937] | -2.8828 | 0 | 0/0 | -0.8690 | -8.5052 | +9.7853 |
| all | VIEW_GRID_minus_A | -7.1431 [-10.8354, -3.4508] | -5.5607 | 0 | 0/0 | -2.4081 | -10.5614 | +13.8415 |
| box_good_mask_bad | VIEW_SOLVE_minus_A | -2.8634 [-8.3204, +2.5937] | -2.8634 | 0 | 0/0 | -0.1034 | -0.6405 | +2.7626 |
| box_good_mask_bad | VIEW_SOLVE_minus_BASE_SOLVE | -2.2123 [-7.1775, +2.7528] | -2.2123 | 0 | 0/0 | +1.2501 | -0.2351 | +2.9933 |
| box_good_mask_bad | VIEW_SOLVE_minus_VIEW_FULL | -3.0469 [-4.4426, -1.6512] | -3.0469 | 0 | 0/0 | +1.9998 | -0.7677 | +5.7505 |
| box_good_mask_bad | VIEW_SOLVE_minus_VIEW_GRID | -0.0118 [-0.6998, +0.6762] | -0.0118 | 0 | 0/0 | +5.6641 | +2.9784 | +5.1591 |
| box_good_mask_bad | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | +0.2303 [-8.2521, +8.7126] | +0.2303 | 0 | 0/0 | -0.6406 | -0.5423 | -2.8500 |
| box_good_mask_bad | VIEW_FULL_minus_ORIG_RESELECT | +0.1835 [-6.6692, +7.0363] | +0.1835 | 0 | 0/0 | -2.1031 | +0.1272 | -2.9879 |
| box_good_mask_bad | ORIG_RESELECT_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 | 0/0 | +0.0000 | +0.0000 | +0.0000 |
| box_good_mask_bad | ORIG_RESELECT_SOLVE_minus_A | -3.0936 [-6.1189, -0.0684] | -3.0936 | 0 | 0/0 | +0.5373 | -0.0983 | +5.6126 |
| box_good_mask_bad | BASE_SOLVE_minus_A | -0.6510 [-1.1430, -0.1590] | -0.6510 | 0 | 0/0 | -1.3534 | -0.4054 | -0.2307 |
| box_good_mask_bad | VIEW_FULL_minus_A | +0.1835 [-6.6692, +7.0363] | +0.1835 | 0 | 0/0 | -2.1031 | +0.1272 | -2.9879 |
| box_good_mask_bad | VIEW_GRID_minus_A | -2.8516 [-7.6207, +1.9175] | -2.8516 | 0 | 0/0 | -5.7675 | -3.6190 | -2.3965 |
| original_success | VIEW_SOLVE_minus_A | -0.5124 [-0.5124, -0.5124] | -0.5124 | 0 | 0/0 | +0.4945 | -0.2259 | +1.9968 |
| original_success | VIEW_SOLVE_minus_BASE_SOLVE | -0.9338 [-0.9338, -0.9338] | -0.9338 | 0 | 0/0 | +1.2051 | -0.1258 | +3.9934 |
| original_success | VIEW_SOLVE_minus_VIEW_FULL | -1.1876 [-1.1876, -1.1876] | -1.1876 | 0 | 0/0 | +1.0627 | -0.1939 | +5.0497 |
| original_success | VIEW_SOLVE_minus_VIEW_GRID | +1.8959 [+1.8959, +1.8959] | +1.8959 | 0 | 0/0 | +1.9580 | +0.7641 | -2.7186 |
| original_success | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | +0.7664 [+0.7664, +0.7664] | +0.7664 | 0 | 0/0 | +0.2962 | -0.1757 | -0.7193 |
| original_success | VIEW_FULL_minus_ORIG_RESELECT | +0.6752 [+0.6752, +0.6752] | +0.6752 | 0 | 0/0 | -0.5683 | -0.0321 | -3.0529 |
| original_success | ORIG_RESELECT_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 | 0/0 | +0.0000 | +0.0000 | +0.0000 |
| original_success | ORIG_RESELECT_SOLVE_minus_A | -1.2788 [-1.2788, -1.2788] | -1.2788 | 0 | 0/0 | +0.1983 | -0.0502 | +2.7162 |
| original_success | BASE_SOLVE_minus_A | +0.4214 [+0.4214, +0.4214] | +0.4214 | 0 | 0/0 | -0.7106 | -0.1001 | -1.9966 |
| original_success | VIEW_FULL_minus_A | +0.6752 [+0.6752, +0.6752] | +0.6752 | 0 | 0/0 | -0.5683 | -0.0321 | -3.0529 |
| original_success | VIEW_GRID_minus_A | -2.4083 [-2.4083, -2.4083] | -2.4083 | 0 | 0/0 | -1.4635 | -0.9900 | +4.7154 |
| original_failure | VIEW_SOLVE_minus_A | -9.9190 [-11.5176, -8.3204] | -10.4519 | 0 | 0/0 | +0.3444 | -5.8475 | +19.9957 |
| original_failure | VIEW_SOLVE_minus_BASE_SOLVE | -9.5755 [-11.9734, -7.1775] | -10.3748 | 0 | 0/0 | +1.4773 | -5.3971 | +20.2493 |
| original_failure | VIEW_SOLVE_minus_VIEW_FULL | -2.5316 [-3.4120, -1.6512] | -2.8251 | 0 | 0/0 | +1.5777 | +2.8183 | +6.4440 |
| original_failure | VIEW_SOLVE_minus_VIEW_GRID | -0.6910 [-0.6998, -0.6822] | -0.6880 | 0 | 0/0 | +4.4953 | +6.3388 | +6.3111 |
| original_failure | VIEW_SOLVE_minus_ORIG_RESELECT_SOLVE | -7.2227 [-8.2521, -6.1934] | -6.8796 | 0 | 0/0 | +0.4103 | -4.7814 | +16.1038 |
| original_failure | VIEW_FULL_minus_ORIG_RESELECT | -7.3874 [-8.1055, -6.6692] | -7.6268 | 0 | 0/0 | -1.2333 | -8.6658 | +13.5517 |
| original_failure | ORIG_RESELECT_minus_A | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0 | 0/0 | +0.0000 | +0.0000 | +0.0000 |
| original_failure | ORIG_RESELECT_SOLVE_minus_A | -2.6963 [-5.3242, -0.0684] | -3.5723 | 0 | 0/0 | -0.0659 | -1.0661 | +3.8918 |
| original_failure | BASE_SOLVE_minus_A | -0.3435 [-1.1430, +0.4559] | -0.0771 | 0 | 0/0 | -1.1330 | -0.4503 | -0.2537 |
| original_failure | VIEW_FULL_minus_A | -7.3874 [-8.1055, -6.6692] | -7.6268 | 0 | 0/0 | -1.2333 | -8.6658 | +13.5517 |
| original_failure | VIEW_GRID_minus_A | -9.2280 [-10.8354, -7.6207] | -9.7638 | 0 | 0/0 | -4.1510 | -12.1862 | +13.6846 |

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
