# Mask supervision support redistribution: frozen epoch-3 evaluation

This is a reused-development diagnostic, not an independent confirmation or COCO AP result. Only original sparse official one-to-one candidate identities are evaluated; no assignment, threshold, gate or checkpoint is selected here.

A is original YOLO; N is the completed ordinary native-branch control; D changes the fixed training supervision support while preserving the native architecture and training budget. The support change is not a claim of pure sample-count causality.

The main metric is dev image-macro original-image Mask IoU. Candidate means also resample whole images. All deltas below are percentage points; CIs are descriptive 1,000-image-bootstrap intervals.

| Split / group | Images / candidates | Comparison | Macro IoU delta [95% CI] | Candidate IoU delta | Mask75 repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fit / all | 127 / 756 | D_minus_A | -0.0063 [-0.2622, +0.2230] | +0.1973 | 3 / 4 | -0.0732 | +0.0065 | -0.0380 |
| fit / all | 127 / 756 | D_minus_N | -0.3116 [-0.7032, -0.0304] | -0.2011 | 2 / 8 | +0.0826 | -0.1280 | +0.4348 |
| fit / all | 127 / 756 | N_minus_A | +0.3053 [-0.0557, +0.7451] | +0.3984 | 7 / 2 | -0.1558 | +0.1345 | -0.4728 |
| fit / box_good_mask_bad | 64 / 136 | D_minus_A | +0.0990 [-1.5679, +1.7863] | +0.4055 | 3 / 0 | +0.2274 | +0.3149 | +0.4868 |
| fit / box_good_mask_bad | 64 / 136 | D_minus_N | -1.2389 [-2.7188, -0.2553] | -0.9890 | 1 / 4 | +0.1185 | -0.3158 | +1.2996 |
| fit / box_good_mask_bad | 64 / 136 | N_minus_A | +1.3379 [-0.6302, +3.3584] | +1.3944 | 6 / 0 | +0.1089 | +0.6307 | -0.8128 |
| fit / original_success | 125 / 500 | D_minus_A | +0.0374 [-0.0340, +0.1060] | +0.0604 | 0 / 4 | -0.0653 | +0.0091 | -0.1900 |
| fit / original_success | 125 / 500 | D_minus_N | +0.0462 [-0.0666, +0.2063] | +0.1482 | 1 / 3 | +0.0223 | +0.0673 | -0.0161 |
| fit / original_success | 125 / 500 | N_minus_A | -0.0088 [-0.1635, +0.0942] | -0.0878 | 0 / 2 | -0.0876 | -0.0582 | -0.1739 |
| fit / original_failure | 77 / 256 | D_minus_A | -0.1555 [-1.4289, +0.8478] | +0.4648 | 3 / 0 | +0.0870 | -0.0108 | +0.9438 |
| fit / original_failure | 77 / 256 | D_minus_N | -1.1695 [-2.2209, -0.3555] | -0.8833 | 1 / 5 | +0.3694 | -0.7498 | +1.7398 |
| fit / original_failure | 77 / 256 | N_minus_A | +1.0139 [-0.5035, +2.5298] | +1.3481 | 7 / 0 | -0.2825 | +0.7389 | -0.7960 |
| fit / pyramid_L0 | 56 / 196 | D_minus_A | -0.1048 [-0.3039, +0.1062] | -0.0625 | 1 / 3 | +0.4172 | -0.2925 | +1.3541 |
| fit / pyramid_L0 | 56 / 196 | D_minus_N | -0.3332 [-0.5334, -0.1405] | -0.3304 | 1 / 2 | +0.0186 | -0.7396 | +0.9743 |
| fit / pyramid_L0 | 56 / 196 | N_minus_A | +0.2284 [+0.0433, +0.4190] | +0.2679 | 0 / 1 | +0.3986 | +0.4471 | +0.3798 |
| fit / pyramid_L1 | 77 / 300 | D_minus_A | -0.0037 [-0.4668, +0.3953] | +0.5150 | 2 / 0 | +0.3948 | +0.1165 | +0.4773 |
| fit / pyramid_L1 | 77 / 300 | D_minus_N | -0.3014 [-0.8683, +0.2470] | +0.0041 | 1 / 1 | +0.1657 | +0.0536 | +0.5454 |
| fit / pyramid_L1 | 77 / 300 | N_minus_A | +0.2978 [-0.1918, +0.7033] | +0.5109 | 3 / 1 | +0.2291 | +0.0629 | -0.0681 |
| fit / pyramid_L2 | 115 / 260 | D_minus_A | +0.0168 [-0.2949, +0.2772] | +0.0266 | 0 / 1 | -0.3222 | +0.0070 | -0.6635 |
| fit / pyramid_L2 | 115 / 260 | D_minus_N | -0.2946 [-0.7305, -0.0427] | -0.3404 | 0 / 5 | +0.0309 | -0.0784 | +0.1579 |
| fit / pyramid_L2 | 115 / 260 | N_minus_A | +0.3114 [-0.1468, +0.8120] | +0.3670 | 4 / 0 | -0.3531 | +0.0854 | -0.8213 |
| dev / all | 253 / 1816 | D_minus_A | -0.0553 [-0.1945, +0.0859] | -0.0961 | 15 / 19 | -0.1208 | -0.0870 | -0.0288 |
| dev / all | 253 / 1816 | D_minus_N | -0.0238 [-0.1479, +0.0867] | +0.0421 | 16 / 18 | +0.1263 | -0.0339 | +0.2526 |
| dev / all | 253 / 1816 | N_minus_A | -0.0315 [-0.1831, +0.1381] | -0.1382 | 16 / 18 | -0.2471 | -0.0531 | -0.2814 |
| dev / box_good_mask_bad | 141 / 353 | D_minus_A | +0.2474 [-0.3353, +0.8382] | -0.0242 | 13 / 0 | -0.1768 | -0.2614 | -0.5752 |
| dev / box_good_mask_bad | 141 / 353 | D_minus_N | +0.1768 [-0.2944, +0.7055] | +0.0782 | 8 / 6 | +0.5589 | +0.0035 | +0.3475 |
| dev / box_good_mask_bad | 141 / 353 | N_minus_A | +0.0707 [-0.5769, +0.7477] | -0.1024 | 11 / 0 | -0.7356 | -0.2649 | -0.9227 |
| dev / original_success | 250 / 1138 | D_minus_A | -0.1005 [-0.1967, -0.0163] | -0.1082 | 0 / 19 | -0.1594 | +0.0082 | -0.0433 |
| dev / original_success | 250 / 1138 | D_minus_N | +0.0362 [-0.0675, +0.1644] | +0.0967 | 8 / 9 | +0.0517 | +0.0266 | +0.1260 |
| dev / original_success | 250 / 1138 | N_minus_A | -0.1367 [-0.2282, -0.0656] | -0.2049 | 0 / 18 | -0.2111 | -0.0184 | -0.1693 |
| dev / original_failure | 164 / 678 | D_minus_A | +0.0904 [-0.3469, +0.5620] | -0.0759 | 15 / 0 | -0.0491 | -0.3838 | -0.1162 |
| dev / original_failure | 164 / 678 | D_minus_N | -0.0557 [-0.3402, +0.2479] | -0.0496 | 8 / 9 | +0.4524 | -0.1455 | +0.5832 |
| dev / original_failure | 164 / 678 | N_minus_A | +0.1461 [-0.3964, +0.7045] | -0.0264 | 16 / 0 | -0.5015 | -0.2384 | -0.6994 |
| dev / pyramid_L0 | 115 / 579 | D_minus_A | -0.2083 [-0.6796, +0.1185] | -0.2068 | 5 / 8 | +0.0736 | -0.1657 | +0.8715 |
| dev / pyramid_L0 | 115 / 579 | D_minus_N | -0.1818 [-0.5495, +0.1065] | -0.1525 | 6 / 4 | +0.1563 | -0.1201 | +0.6894 |
| dev / pyramid_L0 | 115 / 579 | N_minus_A | -0.0266 [-0.1642, +0.1064] | -0.0543 | 3 / 8 | -0.0826 | -0.0456 | +0.1821 |
| dev / pyramid_L1 | 179 / 682 | D_minus_A | -0.1403 [-0.4850, +0.1146] | +0.0482 | 8 / 3 | -0.0852 | -0.1362 | +0.1609 |
| dev / pyramid_L1 | 179 / 682 | D_minus_N | +0.1522 [-0.0274, +0.3739] | +0.2089 | 8 / 5 | +0.4169 | -0.0245 | +0.2968 |
| dev / pyramid_L1 | 179 / 682 | N_minus_A | -0.2925 [-0.7486, -0.0192] | -0.1607 | 9 / 7 | -0.5021 | -0.1117 | -0.1359 |
| dev / pyramid_L2 | 226 / 555 | D_minus_A | -0.1619 [-0.3793, +0.0257] | -0.1579 | 2 / 8 | -0.3459 | -0.0983 | -0.4205 |
| dev / pyramid_L2 | 226 / 555 | D_minus_N | -0.0335 [-0.2827, +0.2548] | +0.0402 | 2 / 9 | +0.0176 | +0.0013 | +0.0852 |
| dev / pyramid_L2 | 226 / 555 | N_minus_A | -0.1285 [-0.4054, +0.1138] | -0.1981 | 4 / 3 | -0.3635 | -0.0996 | -0.5056 |

## Original sparse diagnostic loss

BCE uses the original sparse official GT-mask support, original gain and normalization for all arms, even though D trained with a different support set. It is not the dense training trajectory loss.

| Split | Candidate BCE A / N / D | Image-macro BCE A / N / D |
|---|---|---|
| fit | 2.39868563 / 2.24942133 / 2.32026114 | 2.04503480 / 1.93515045 / 1.98917059 |
| dev | 2.56644822 / 2.57340981 / 2.60502785 | 2.13413551 / 2.13833294 / 2.15909221 |

## Frozen-state and interpretation checks

- A and N replay the historical dev metrics at the exact permanent candidate identities; maximum allowable absolute error is 1e-12. Original coefficient replay uses atol=rtol=3e-5.
- Full P and the frozen predicted box feed the existing process_mask(upsample=True) decoder and true letterbox inverse. Original COCO instance masks evaluate IoU/coverage; continuous input-grid logits and the fixed prediction-box support evaluate AUC/FPR.
- AUC/FPR undefined cases remain in IoU and Mask75 evaluation. Original success/failure and the box-good-mask-bad subgroup are defined by A. A pyramid level is not an object-size class.
- The fit evaluation is the first 128 planned fit images, not the full 1,024-image training set. Smoke uses two effective images per split and makes no scientific decision.
- Prespecified screen status: **prespecified_IoU_screen_not_passed**. Other meaningful metrics require explicit magnitude, uncertainty and trade-off review. No automatic new run, expanded data, or extra training follows.
