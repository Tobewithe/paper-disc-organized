# External frozen spatial evidence projected into original YOLO prototypes

STOP after this fixed replay; no automatic training, prompts, threshold search or solver changes.

1816 fixed candidates in 253 effective images. External SAM ViT-B is an additional pretrained model and compute source, not a lightweight YOLO-only improvement.

| Group | Arm | Macro IoU | Candidate IoU | Macro Mask75 | Coverage | AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| all | A | +80.25922 | +75.93827 | +72.88784 | +92.18980 | +95.66236 | +21.00947 |
| all | BASE_SOLVE | +78.94393 | +74.19897 | +69.49107 | +92.47222 | +95.08949 | +25.47481 |
| all | SAM_FULL | +76.03362 | +73.50306 | +66.35415 | +83.55142 | +92.79949 | +12.73601 |
| all | SAM_GRID | +72.06062 | +70.63133 | +57.36433 | +80.79002 | +92.34988 | +17.36313 |
| all | SAM_SOLVE | +75.26891 | +72.01113 | +65.36736 | +84.92180 | +93.28595 | +18.45698 |
| box_good_mask_bad | A | +62.27843 | +63.07913 | +0.00000 | +84.50430 | +89.93721 | +33.20876 |
| box_good_mask_bad | BASE_SOLVE | +60.52194 | +60.88963 | +3.14421 | +85.27888 | +88.87416 | +39.37296 |
| box_good_mask_bad | SAM_FULL | +61.25474 | +63.47066 | +23.88815 | +75.95073 | +85.80130 | +22.37780 |
| box_good_mask_bad | SAM_GRID | +57.43579 | +60.29137 | +14.50346 | +72.52458 | +84.80158 | +24.48878 |
| box_good_mask_bad | SAM_SOLVE | +58.12486 | +59.76636 | +13.17118 | +76.76153 | +85.31204 | +31.05752 |
| original_success | A | +87.71125 | +86.60107 | +100.00000 | +95.23482 | +97.97144 | +15.15043 |
| original_success | BASE_SOLVE | +86.36869 | +84.89154 | +92.51378 | +95.43687 | +97.61836 | +19.45101 |
| original_success | SAM_FULL | +82.36317 | +82.40042 | +84.57613 | +86.31435 | +95.26638 | +8.00234 |
| original_success | SAM_GRID | +78.33179 | +79.09356 | +75.99011 | +83.85295 | +94.91304 | +13.34873 |
| original_success | SAM_SOLVE | +82.13872 | +81.84411 | +84.68820 | +87.81780 | +96.17314 | +12.84312 |
| original_failure | A | +58.97834 | +58.04113 | +0.00000 | +83.03133 | +88.38383 | +37.76913 |
| original_failure | BASE_SOLVE | +57.49985 | +56.25185 | +3.31257 | +83.88267 | +87.02675 | +44.22943 |
| original_failure | SAM_FULL | +57.97808 | +58.56913 | +17.02287 | +74.70614 | +84.71295 | +25.92359 |
| original_failure | SAM_GRID | +54.95560 | +56.42777 | +10.44377 | +71.96084 | +84.12729 | +27.87665 |
| original_failure | SAM_SOLVE | +55.60631 | +55.50680 | +11.33414 | +75.97093 | +84.23617 | +35.08575 |

All changes below are percentage points. Image macro point estimates and intervals use the same statistic; candidate-mean intervals also resample whole images.

| Group | Comparison | Macro IoU delta [95% CI] | Candidate delta | Repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---|---:|---:|---:|---:|---:|---:|
| all | BASE_SOLVE_minus_A | -1.31528 [-1.50128, -1.14053] | -1.73931 | 15 / 108 | +0.28242 | -0.57288 | +4.46534 |
| all | SAM_FULL_minus_A | -4.22559 [-5.52289, -3.02249] | -2.43522 | 126 / 174 | -8.63838 | -2.86287 | -8.27346 |
| all | SAM_GRID_minus_A | -8.19860 [-9.62350, -6.83484] | -5.30694 | 91 / 251 | -11.39978 | -3.31248 | -3.64633 |
| all | SAM_SOLVE_minus_A | -4.99030 [-6.21177, -3.86251] | -3.92714 | 64 / 182 | -7.26800 | -2.37641 | -2.55249 |
| all | SAM_SOLVE_minus_BASE_SOLVE | -3.67502 [-4.85721, -2.52426] | -2.18784 | 117 / 142 | -7.55042 | -1.80354 | -7.01783 |
| all | SAM_SOLVE_minus_SAM_FULL | -0.76471 [-1.12827, -0.42655] | -1.49192 | 56 / 126 | +1.37039 | +0.48646 | +5.72097 |
| all | SAM_SOLVE_minus_SAM_GRID | +3.20830 [+2.60170, +3.79870] | +1.37980 | 133 / 91 | +4.13178 | +0.93607 | +1.09385 |
| all | SAM_GRID_minus_SAM_FULL | -3.97301 [-4.48507, -3.45220] | -2.87172 | 18 / 130 | -2.76139 | -0.44961 | +4.62713 |
| box_good_mask_bad | BASE_SOLVE_minus_A | -1.75650 [-2.23014, -1.28519] | -2.18951 | 10 / 0 | +0.77457 | -1.06305 | +6.16420 |
| box_good_mask_bad | SAM_FULL_minus_A | -1.02369 [-3.15657, +1.15938] | +0.39153 | 100 / 0 | -8.55357 | -4.13591 | -10.83096 |
| box_good_mask_bad | SAM_GRID_minus_A | -4.84265 [-7.28385, -2.54100] | -2.78776 | 70 / 0 | -11.97972 | -5.13563 | -8.71997 |
| box_good_mask_bad | SAM_SOLVE_minus_A | -4.15358 [-6.21501, -2.10823] | -3.31278 | 44 / 0 | -7.74278 | -4.62517 | -2.15124 |
| box_good_mask_bad | SAM_SOLVE_minus_BASE_SOLVE | -2.39708 [-4.60072, -0.22105] | -1.12327 | 41 / 7 | -8.51735 | -3.56212 | -8.31544 |
| box_good_mask_bad | SAM_SOLVE_minus_SAM_FULL | -3.12988 [-4.04871, -2.32563] | -3.70430 | 8 / 64 | +0.81079 | -0.48926 | +8.67972 |
| box_good_mask_bad | SAM_SOLVE_minus_SAM_GRID | +0.68907 [-0.49753, +1.82424] | -0.52501 | 16 / 42 | +4.23695 | +0.51046 | +6.56873 |
| box_good_mask_bad | SAM_GRID_minus_SAM_FULL | -3.81895 [-4.96959, -2.81182] | -3.17929 | 1 / 31 | -3.42615 | -0.99972 | +2.11098 |
| original_success | BASE_SOLVE_minus_A | -1.34256 [-1.56916, -1.11830] | -1.70954 | 0 / 108 | +0.20205 | -0.35308 | +4.30058 |
| original_success | SAM_FULL_minus_A | -5.34808 [-6.65648, -4.12261] | -4.20065 | 0 / 174 | -8.92047 | -2.70507 | -7.14809 |
| original_success | SAM_GRID_minus_A | -9.37946 [-10.91552, -7.96642] | -7.50751 | 0 / 251 | -11.38187 | -3.05840 | -1.80170 |
| original_success | SAM_SOLVE_minus_A | -5.57253 [-6.79953, -4.39213] | -4.75696 | 0 / 182 | -7.41702 | -1.79831 | -2.30731 |
| original_success | SAM_SOLVE_minus_BASE_SOLVE | -4.22996 [-5.48223, -3.05081] | -3.04742 | 59 / 133 | -7.61907 | -1.44523 | -6.60789 |
| original_success | SAM_SOLVE_minus_SAM_FULL | -0.22444 [-0.57168, +0.12238] | -0.55631 | 39 / 47 | +1.50345 | +0.90676 | +4.84078 |
| original_success | SAM_SOLVE_minus_SAM_GRID | +3.80693 [+3.25585, +4.41790] | +2.75055 | 108 / 39 | +3.96485 | +1.26010 | -0.50561 |
| original_success | SAM_GRID_minus_SAM_FULL | -4.03138 [-4.54365, -3.55858] | -3.30686 | 10 / 87 | -2.46140 | -0.35334 | +5.34639 |
| original_failure | BASE_SOLVE_minus_A | -1.47849 [-1.84691, -1.10881] | -1.78927 | 15 / 0 | +0.85134 | -1.35708 | +6.46031 |
| original_failure | SAM_FULL_minus_A | -1.00026 [-2.98983, +0.72996] | +0.52801 | 126 / 0 | -8.32519 | -3.67087 | -11.84554 |
| original_failure | SAM_GRID_minus_A | -4.02274 [-5.99510, -2.17709] | -1.61336 | 91 / 0 | -11.07048 | -4.25654 | -9.89248 |
| original_failure | SAM_SOLVE_minus_A | -3.37203 [-5.13001, -1.69789] | -2.53432 | 64 / 0 | -7.06040 | -4.14766 | -2.68338 |
| original_failure | SAM_SOLVE_minus_BASE_SOLVE | -1.89354 [-3.81448, -0.15036] | -0.74505 | 58 / 9 | -7.91175 | -2.79058 | -9.14368 |
| original_failure | SAM_SOLVE_minus_SAM_FULL | -2.37177 [-3.01630, -1.72026] | -3.06233 | 17 / 79 | +1.26479 | -0.47679 | +9.16217 |
| original_failure | SAM_SOLVE_minus_SAM_GRID | +0.65071 [-0.31058, +1.66215] | -0.92096 | 25 / 52 | +4.01008 | +0.10888 | +7.20910 |
| original_failure | SAM_GRID_minus_SAM_FULL | -3.02248 [-3.91986, -2.20005] | -2.14137 | 8 / 43 | -2.74530 | -0.58566 | +1.95306 |

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
