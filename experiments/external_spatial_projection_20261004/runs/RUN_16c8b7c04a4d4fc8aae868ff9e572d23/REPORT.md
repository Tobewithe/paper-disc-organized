# External frozen spatial evidence projected into original YOLO prototypes

STOP after this fixed replay; no automatic training, prompts, threshold search or solver changes.

SMOKE ONLY. 7 fixed candidates in 2 effective images. External SAM ViT-B is an additional pretrained model and compute source, not a lightweight YOLO-only improvement.

| Group | Arm | Macro IoU | Candidate IoU | Macro Mask75 | Coverage | AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| all | A | +73.54886 | +79.45814 | +40.00000 | +90.40264 | +97.67135 | +17.57739 |
| all | BASE_SOLVE | +70.36839 | +77.19642 | +40.00000 | +90.34984 | +96.61098 | +22.48549 |
| all | SAM_FULL | +70.02651 | +76.39867 | +65.00000 | +77.69277 | +94.37993 | +8.64764 |
| all | SAM_GRID | +65.80002 | +71.97946 | +40.00000 | +75.14801 | +92.49052 | +12.87362 |
| all | SAM_SOLVE | +67.78651 | +74.86605 | +40.00000 | +78.74505 | +93.16821 | +13.82413 |
| box_good_mask_bad | A | +62.67775 | +62.67775 | +0.00000 | +89.95937 | +95.30471 | +22.87793 |
| box_good_mask_bad | BASE_SOLVE | +59.58413 | +59.58413 | +0.00000 | +90.49664 | +95.20645 | +28.49053 |
| box_good_mask_bad | SAM_FULL | +62.57948 | +62.57948 | +50.00000 | +69.50964 | +94.71353 | +6.41620 |
| box_good_mask_bad | SAM_GRID | +51.55099 | +51.55099 | +0.00000 | +59.86711 | +91.50187 | +8.44102 |
| box_good_mask_bad | SAM_SOLVE | +58.34534 | +58.34534 | +0.00000 | +67.55711 | +92.77421 | +9.18178 |
| original_success | A | +94.37953 | +94.37953 | +100.00000 | +98.50213 | +99.51451 | +9.01616 |
| original_success | BASE_SOLVE | +93.10071 | +93.10071 | +100.00000 | +98.70044 | +99.46426 | +11.73231 |
| original_success | SAM_FULL | +94.11778 | +94.11778 | +100.00000 | +95.84672 | +98.35808 | +3.64352 |
| original_success | SAM_GRID | +92.15748 | +92.15748 | +100.00000 | +96.46347 | +98.58795 | +11.89373 |
| original_success | SAM_SOLVE | +94.21961 | +94.21961 | +100.00000 | +97.62270 | +99.27920 | +6.95014 |
| original_failure | A | +59.46415 | +59.56295 | +0.00000 | +87.07318 | +95.24774 | +22.03386 |
| original_failure | BASE_SOLVE | +56.76787 | +55.99069 | +0.00000 | +87.00723 | +94.18166 | +25.92569 |
| original_failure | SAM_FULL | +51.58072 | +52.77319 | +25.00000 | +59.16958 | +91.29935 | +7.68231 |
| original_failure | SAM_GRID | +41.92250 | +45.07544 | +0.00000 | +49.96280 | +87.70842 | +8.60743 |
| original_failure | SAM_SOLVE | +47.95819 | +49.06132 | +0.00000 | +58.00140 | +88.66699 | +11.42817 |

All changes below are percentage points. Image macro point estimates and intervals use the same statistic; candidate-mean intervals also resample whole images.

| Group | Comparison | Macro IoU delta [95% CI] | Candidate delta | Repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---|---:|---:|---:|---:|---:|---:|
| all | BASE_SOLVE_minus_A | -3.18047 [-5.32420, -1.03673] | -2.26172 | 0 / 0 | -0.05279 | -1.06037 | +4.90810 |
| all | SAM_FULL_minus_A | -3.52235 [-4.60240, -2.44230] | -3.05947 | 1 / 0 | -12.70987 | -3.29143 | -8.92974 |
| all | SAM_GRID_minus_A | -7.74884 [-8.37921, -7.11846] | -7.47867 | 0 / 0 | -15.25463 | -5.18083 | -4.70376 |
| all | SAM_SOLVE_minus_A | -5.76235 [-8.49296, -3.03173] | -4.59208 | 0 / 0 | -11.65758 | -4.50314 | -3.75326 |
| all | SAM_SOLVE_minus_BASE_SOLVE | -2.58188 [-3.16876, -1.99500] | -2.33036 | 0 / 0 | -11.60479 | -3.44277 | -8.66136 |
| all | SAM_SOLVE_minus_SAM_FULL | -2.24000 [-3.89056, -0.58944] | -1.53261 | 0 / 1 | +1.05229 | -1.21171 | +5.17649 |
| all | SAM_SOLVE_minus_SAM_GRID | +1.98649 [-0.11375, +4.08673] | +2.88659 | 0 / 0 | +3.59704 | +0.67769 | +0.95051 |
| all | SAM_GRID_minus_SAM_FULL | -4.22649 [-4.67616, -3.77681] | -4.41920 | 0 / 1 | -2.54476 | -1.88941 | +4.22598 |
| box_good_mask_bad | BASE_SOLVE_minus_A | -3.09362 [-6.11889, -0.06836] | -3.09362 | 0 / 0 | +0.53727 | -0.09826 | +5.61259 |
| box_good_mask_bad | SAM_FULL_minus_A | -0.09827 [-11.16446, +10.96792] | -0.09827 | 1 / 0 | -20.44973 | -0.59118 | -16.46173 |
| box_good_mask_bad | SAM_GRID_minus_A | -11.12676 [-26.70408, +4.45056] | -11.12676 | 0 / 0 | -30.09226 | -3.80284 | -14.43692 |
| box_good_mask_bad | SAM_SOLVE_minus_A | -4.33241 [-14.51897, +5.85415] | -4.33241 | 0 / 0 | -22.40227 | -2.53050 | -13.69616 |
| box_good_mask_bad | SAM_SOLVE_minus_BASE_SOLVE | -1.23879 [-14.45061, +11.97304] | -1.23879 | 0 / 0 | -22.93953 | -2.43224 | -19.30875 |
| box_good_mask_bad | SAM_SOLVE_minus_SAM_FULL | -4.23414 [-5.11377, -3.35451] | -4.23414 | 0 / 1 | -1.95253 | -1.93932 | +2.76558 |
| box_good_mask_bad | SAM_SOLVE_minus_SAM_GRID | +6.79435 [+1.40359, +12.18511] | +6.79435 | 0 / 0 | +7.69000 | +1.27234 | +0.74076 |
| box_good_mask_bad | SAM_GRID_minus_SAM_FULL | -11.02849 [-15.53962, -6.51736] | -11.02849 | 0 / 1 | -9.64253 | -3.21166 | +2.02482 |
| original_success | BASE_SOLVE_minus_A | -1.27883 [-1.27883, -1.27883] | -1.27883 | 0 / 0 | +0.19832 | -0.05025 | +2.71615 |
| original_success | SAM_FULL_minus_A | -0.26176 [-0.26176, -0.26176] | -0.26176 | 0 / 0 | -2.65541 | -1.15643 | -5.37265 |
| original_success | SAM_GRID_minus_A | -2.22205 [-2.22205, -2.22205] | -2.22205 | 0 / 0 | -2.03866 | -0.92656 | +2.87757 |
| original_success | SAM_SOLVE_minus_A | -0.15993 [-0.15993, -0.15993] | -0.15993 | 0 / 0 | -0.87943 | -0.23531 | -2.06603 |
| original_success | SAM_SOLVE_minus_BASE_SOLVE | +1.11890 [+1.11890, +1.11890] | +1.11890 | 0 / 0 | -1.07774 | -0.18507 | -4.78218 |
| original_success | SAM_SOLVE_minus_SAM_FULL | +0.10183 [+0.10183, +0.10183] | +0.10183 | 0 / 0 | +1.77598 | +0.92112 | +3.30662 |
| original_success | SAM_SOLVE_minus_SAM_GRID | +2.06213 [+2.06213, +2.06213] | +2.06213 | 0 / 0 | +1.15923 | +0.69125 | -4.94359 |
| original_success | SAM_GRID_minus_SAM_FULL | -1.96030 [-1.96030, -1.96030] | -1.96030 | 0 / 0 | +0.61675 | +0.22987 | +8.25021 |
| original_failure | BASE_SOLVE_minus_A | -2.69628 [-5.32420, -0.06836] | -3.57225 | 0 / 0 | -0.06594 | -1.06608 | +3.89183 |
| original_failure | SAM_FULL_minus_A | -7.88343 [-11.16446, -4.60240] | -6.78975 | 1 / 0 | -27.90360 | -3.94840 | -14.35154 |
| original_failure | SAM_GRID_minus_A | -17.54165 [-26.70408, -8.37921] | -14.48750 | 0 / 0 | -37.11038 | -7.53933 | -13.42642 |
| original_failure | SAM_SOLVE_minus_A | -11.50596 [-14.51897, -8.49296] | -10.50163 | 0 / 0 | -29.07178 | -6.58075 | -10.60569 |
| original_failure | SAM_SOLVE_minus_BASE_SOLVE | -8.80968 [-14.45061, -3.16876] | -6.92938 | 0 / 0 | -29.00584 | -5.51467 | -14.49752 |
| original_failure | SAM_SOLVE_minus_SAM_FULL | -3.62253 [-3.89056, -3.35451] | -3.71188 | 0 / 1 | -1.16818 | -2.63235 | +3.74586 |
| original_failure | SAM_SOLVE_minus_SAM_GRID | +6.03568 [-0.11375, +12.18511] | +3.98587 | 0 / 0 | +8.03860 | +0.95858 | +2.82074 |
| original_failure | SAM_GRID_minus_SAM_FULL | -9.65822 [-15.53962, -3.77681] | -7.69775 | 0 / 1 | -9.20678 | -3.59093 | +0.92512 |

Arm definitions and limits:

- A is original c0 with complete original P and normal frozen-box decoding.
- BASE_SOLVE uses original full640 logits -> sigmoid once -> identical floor/ceil predicted-box pooling -> 8x8 -> probability clip [.01,.99] -> logit -> existing FP64 ridge (.003). It is not the previous sigmoid(A8 c0) baseline.
- SAM_FULL uses postprocessed continuous SAM logits at 640, the original prediction-box support, and the same original-image inverse. The input is exact cached RGB uint8; SAM internally performs its official 1024 preprocessing.
- SAM_GRID and SAM_SOLVE share precisely the same q8 and clipped target logits. GRID bilinearly renders those logits into the floor/ceil box; SOLVE fits original-P coefficients centered on c0 and uses normal original-P decoding.
- FULL -> GRID changes pooling, clipping and rendering together, not only resolution. P pooling operates on the original160 grid while probability pooling uses640; legacy integer-quantization differences remain and a negative result is not a prototype-capacity proof.
- Quality predictions are recorded but never select masks, candidates, prompts or subsets. No point prompts, GT boxes/classes, multimask oracle selection or iterative mask inputs are used.
- Finite degenerate/nonintersecting prediction boxes remain: SAM_FULL falls back to original logits; the same baseline q8 is used by GRID/SOLVE, explicitly marked unavailable. Nonfinite assets fail instead of being silently repaired.
- Every candidate, including empty outputs and undefined AUC, remains. Original groups use A only. This is reused developer data on fixed official supervision candidates, not full inference AP or a new blind test.
- External pretrained knowledge and encoder/prompt latency are extra costs; a positive replay does not establish distillability, YOLO-internal information sufficiency, or a deployable lightweight contribution.
- Replay q8/coefficient tensors remain on the server under REPLAY_TENSORS for independent reconstruction. No historical OGPS effect size or oracle-recovery ratio is imported.
