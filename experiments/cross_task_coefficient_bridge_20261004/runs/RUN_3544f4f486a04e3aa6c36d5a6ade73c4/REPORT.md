# Cross-task coefficient bridge: fixed epoch-three screen

A is original YOLO; N is historical ordinary native fine-tuning. T reads frozen classification hidden features; R reads frozen regression hidden features; M trains with the same architecture as T but a fixed wrong-image source. TW reuses frozen T parameters with M's wrong source, without training or changing the candidate.

Primary evidence is normal original-image mask quality. This is reused-development evaluation of GT-conditioned fixed official candidates, not a new blind test or COCO AP result. T/R have different input dimensions and are not a matched-capacity mechanism contrast.

Deltas below are percentage points. Each confidence interval resamples whole images 1,000 times. Fit is the first 128 planned images; dev uses all 256 planned images. No-positive images remain in the manifest.

| Split / group | Images / candidates | Pair | Macro IoU delta [95% CI] | Candidate IoU delta | Repair / damage | Coverage delta | AUC delta | FPR delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fit / all | 127 / 756 | T_minus_A | +0.3741 [-0.0086, +0.8448] | +0.4019 | 7 / 7 | -0.0107 | +0.1779 | -0.4196 |
| fit / all | 127 / 756 | T_minus_N | +0.0688 [-0.0143, +0.1550] | +0.0035 | 0 / 5 | +0.1451 | +0.0434 | +0.0532 |
| fit / all | 127 / 756 | T_minus_M | -0.0512 [-0.1280, +0.0286] | -0.0547 | 1 / 5 | +0.0562 | +0.0042 | +0.0497 |
| fit / all | 127 / 756 | R_minus_N | +0.0571 [+0.0012, +0.1145] | +0.0430 | 0 / 0 | +0.0983 | +0.0210 | +0.0004 |
| fit / all | 127 / 756 | T_minus_TW | +0.0700 [+0.0035, +0.1443] | +0.0397 | 2 / 6 | +0.1792 | +0.0309 | +0.1556 |
| fit / box_good_mask_bad | 64 / 136 | T_minus_A | +1.6931 [-0.4920, +3.9172] | +1.7000 | 6 / 0 | +0.6967 | +0.7569 | -0.6237 |
| fit / box_good_mask_bad | 64 / 136 | T_minus_N | +0.3552 [+0.0593, +0.6673] | +0.3056 | 0 / 0 | +0.5879 | +0.1263 | +0.1891 |
| fit / box_good_mask_bad | 64 / 136 | T_minus_M | -0.2169 [-0.4183, -0.0431] | -0.1372 | 0 / 2 | +0.0351 | -0.0914 | +0.2100 |
| fit / box_good_mask_bad | 64 / 136 | R_minus_N | +0.1710 [+0.0221, +0.3521] | +0.1328 | 0 / 0 | +0.1830 | +0.0431 | +0.0348 |
| fit / box_good_mask_bad | 64 / 136 | T_minus_TW | +0.2200 [-0.0689, +0.4840] | +0.2709 | 0 / 2 | +0.6609 | +0.0096 | +1.0034 |
| fit / original_success | 125 / 500 | T_minus_A | -0.0490 [-0.2196, +0.0800] | -0.2507 | 0 / 7 | -0.1248 | -0.0619 | -0.1382 |
| fit / original_success | 125 / 500 | T_minus_N | -0.0402 [-0.1134, +0.0172] | -0.1628 | 0 / 5 | -0.0372 | -0.0037 | +0.0356 |
| fit / original_success | 125 / 500 | T_minus_M | -0.0267 [-0.0901, +0.0293] | -0.0482 | 1 / 2 | -0.0023 | -0.0062 | +0.0148 |
| fit / original_success | 125 / 500 | R_minus_N | -0.0122 [-0.0357, +0.0090] | -0.0047 | 0 / 0 | +0.0082 | -0.0079 | +0.0270 |
| fit / original_success | 125 / 500 | T_minus_TW | -0.0203 [-0.0764, +0.0311] | -0.1008 | 2 / 4 | +0.0162 | +0.0034 | +0.0380 |
| fit / original_failure | 77 / 256 | T_minus_A | +1.4268 [-0.1905, +3.0329] | +1.6765 | 7 / 0 | +0.3516 | +0.9002 | -0.6422 |
| fit / original_failure | 77 / 256 | T_minus_N | +0.4129 [+0.1793, +0.6622] | +0.3285 | 0 / 0 | +0.6341 | +0.1613 | +0.1538 |
| fit / original_failure | 77 / 256 | T_minus_M | -0.0846 [-0.3424, +0.1851] | -0.0674 | 0 / 3 | +0.1847 | -0.0085 | +0.1012 |
| fit / original_failure | 77 / 256 | R_minus_N | +0.2410 [+0.0953, +0.4164] | +0.1361 | 0 / 0 | +0.2920 | +0.0919 | -0.0462 |
| fit / original_failure | 77 / 256 | T_minus_TW | +0.2488 [-0.0060, +0.5052] | +0.3142 | 0 / 2 | +0.6469 | +0.0423 | +0.7646 |
| fit / same_predicted_class_wrong_source | 127 / 755 | T_minus_TW | +0.0719 [+0.0048, +0.1453] | +0.0404 | 2 / 6 | +0.1787 | +0.0319 | +0.1532 |
| fit / same_predicted_class_wrong_source_target | 64 / 135 | T_minus_TW | +0.2177 [-0.0736, +0.4808] | +0.2767 | 0 / 2 | +0.6609 | +0.0173 | +1.0201 |
| fit / pyramid_L0 | 56 / 196 | T_minus_A | +0.2935 [+0.0763, +0.5098] | +0.3174 | 0 / 2 | +0.4843 | +0.4518 | +0.5379 |
| fit / pyramid_L0 | 56 / 196 | T_minus_N | +0.0652 [-0.0364, +0.1690] | +0.0494 | 0 / 1 | +0.0857 | +0.0047 | +0.1581 |
| fit / pyramid_L0 | 56 / 196 | T_minus_M | +0.0212 [-0.0637, +0.1032] | +0.0533 | 0 / 1 | -0.0642 | -0.0830 | -0.2543 |
| fit / pyramid_L0 | 56 / 196 | R_minus_N | -0.0089 [-0.0759, +0.0463] | -0.0034 | 0 / 0 | -0.0740 | +0.0012 | -0.1485 |
| fit / pyramid_L0 | 56 / 196 | T_minus_TW | +0.0267 [-0.0903, +0.1495] | +0.0226 | 0 / 2 | -0.0088 | -0.0069 | -0.1302 |
| fit / pyramid_L1 | 77 / 300 | T_minus_A | +0.3081 [-0.2155, +0.7292] | +0.3739 | 3 / 4 | +0.2893 | +0.0941 | +0.0580 |
| fit / pyramid_L1 | 77 / 300 | T_minus_N | +0.0103 [-0.1473, +0.1509] | -0.1369 | 0 / 3 | +0.0602 | +0.0312 | +0.1261 |
| fit / pyramid_L1 | 77 / 300 | T_minus_M | -0.0986 [-0.2517, +0.0321] | -0.1392 | 1 / 3 | -0.1045 | -0.0078 | +0.1730 |
| fit / pyramid_L1 | 77 / 300 | R_minus_N | +0.1243 [-0.0021, +0.3044] | +0.0769 | 0 / 0 | +0.2653 | +0.0185 | +0.1221 |
| fit / pyramid_L1 | 77 / 300 | T_minus_TW | +0.1805 [-0.0480, +0.4901] | -0.0036 | 1 / 3 | +0.4285 | +0.0371 | +0.5193 |
| fit / pyramid_L2 | 115 / 260 | T_minus_A | +0.4322 [-0.0876, +0.9959] | +0.4980 | 4 / 1 | -0.1614 | +0.1310 | -0.8342 |
| fit / pyramid_L2 | 115 / 260 | T_minus_N | +0.1208 [+0.0257, +0.2370] | +0.1310 | 0 / 1 | +0.1917 | +0.0455 | -0.0128 |
| fit / pyramid_L2 | 115 / 260 | T_minus_M | -0.0566 [-0.2010, +0.0815] | -0.0387 | 0 / 1 | +0.0914 | +0.0148 | +0.0617 |
| fit / pyramid_L2 | 115 / 260 | R_minus_N | +0.0357 [-0.0190, +0.0966] | +0.0389 | 0 / 0 | +0.0784 | +0.0246 | +0.0333 |
| fit / pyramid_L2 | 115 / 260 | T_minus_TW | +0.1008 [-0.0200, +0.2332] | +0.1026 | 1 / 1 | +0.1620 | +0.0365 | +0.0871 |
| dev / all | 253 / 1816 | T_minus_A | -0.1037 [-0.2962, +0.1055] | -0.2001 | 19 / 18 | -0.3269 | -0.0553 | -0.3017 |
| dev / all | 253 / 1816 | T_minus_N | -0.0722 [-0.1742, +0.0203] | -0.0619 | 9 / 6 | -0.0798 | -0.0022 | -0.0203 |
| dev / all | 253 / 1816 | T_minus_M | +0.0282 [-0.0707, +0.1262] | -0.0199 | 14 / 5 | +0.0317 | +0.0117 | -0.0382 |
| dev / all | 253 / 1816 | R_minus_N | -0.0426 [-0.1065, +0.0080] | -0.0432 | 2 / 6 | -0.0287 | -0.0036 | +0.0284 |
| dev / all | 253 / 1816 | T_minus_TW | -0.0277 [-0.1156, +0.0377] | -0.0256 | 12 / 5 | +0.0010 | -0.0002 | -0.0020 |
| dev / box_good_mask_bad | 141 / 353 | T_minus_A | -0.0496 [-0.7407, +0.7520] | -0.1885 | 14 / 0 | -0.7340 | -0.2591 | -0.7684 |
| dev / box_good_mask_bad | 141 / 353 | T_minus_N | -0.1202 [-0.3764, +0.1321] | -0.0861 | 5 / 2 | +0.0017 | +0.0058 | +0.1543 |
| dev / box_good_mask_bad | 141 / 353 | T_minus_M | +0.1388 [-0.1624, +0.4566] | +0.1751 | 6 / 1 | +0.1564 | +0.0382 | -0.0397 |
| dev / box_good_mask_bad | 141 / 353 | R_minus_N | -0.1127 [-0.3437, +0.0702] | -0.0927 | 1 / 2 | -0.1175 | -0.0392 | +0.0230 |
| dev / box_good_mask_bad | 141 / 353 | T_minus_TW | +0.0686 [-0.0941, +0.2351] | +0.1467 | 3 / 1 | +0.0919 | +0.0573 | -0.0264 |
| dev / original_success | 250 / 1138 | T_minus_A | -0.2062 [-0.4196, -0.0691] | -0.2637 | 0 / 18 | -0.3097 | -0.0178 | -0.2263 |
| dev / original_success | 250 / 1138 | T_minus_N | -0.0695 [-0.2222, +0.0196] | -0.0588 | 4 / 4 | -0.0986 | +0.0006 | -0.0570 |
| dev / original_success | 250 / 1138 | T_minus_M | -0.0303 [-0.1530, +0.0489] | -0.0644 | 4 / 3 | +0.0014 | +0.0082 | -0.0148 |
| dev / original_success | 250 / 1138 | R_minus_N | -0.0169 [-0.0407, +0.0060] | -0.0299 | 1 / 3 | -0.0232 | -0.0012 | -0.0075 |
| dev / original_success | 250 / 1138 | T_minus_TW | -0.0586 [-0.2271, +0.0265] | -0.0842 | 6 / 4 | -0.0595 | -0.0020 | -0.0498 |
| dev / original_failure | 164 / 678 | T_minus_A | +0.0716 [-0.5084, +0.6865] | -0.0934 | 19 / 0 | -0.5497 | -0.2320 | -0.6972 |
| dev / original_failure | 164 / 678 | T_minus_N | -0.0745 [-0.2840, +0.0949] | -0.0670 | 5 / 2 | -0.0482 | +0.0064 | +0.0021 |
| dev / original_failure | 164 / 678 | T_minus_M | +0.1055 [-0.1145, +0.3699] | +0.0548 | 10 / 2 | +0.0699 | +0.0301 | -0.0717 |
| dev / original_failure | 164 / 678 | R_minus_N | -0.1049 [-0.2932, +0.0588] | -0.0656 | 1 / 3 | -0.0628 | -0.0162 | +0.0793 |
| dev / original_failure | 164 / 678 | T_minus_TW | +0.0277 [-0.0986, +0.1615] | +0.0726 | 6 / 1 | +0.0586 | -0.0014 | -0.0577 |
| dev / same_predicted_class_wrong_source | 246 / 1732 | T_minus_TW | -0.0268 [-0.1152, +0.0458] | -0.0312 | 11 / 5 | -0.0000 | +0.0012 | +0.0040 |
| dev / same_predicted_class_wrong_source_target | 138 / 339 | T_minus_TW | +0.0547 [-0.0942, +0.2231] | +0.1246 | 3 / 1 | +0.0951 | +0.0559 | -0.0087 |
| dev / pyramid_L0 | 115 / 579 | T_minus_A | -0.2000 [-0.5111, +0.0396] | -0.1217 | 4 / 7 | -0.3116 | -0.0929 | +0.1499 |
| dev / pyramid_L0 | 115 / 579 | T_minus_N | -0.1734 [-0.4446, -0.0043] | -0.0674 | 3 / 1 | -0.2290 | -0.0473 | -0.0322 |
| dev / pyramid_L0 | 115 / 579 | T_minus_M | -0.0358 [-0.2121, +0.1056] | -0.0073 | 3 / 2 | -0.1576 | -0.0225 | -0.2741 |
| dev / pyramid_L0 | 115 / 579 | R_minus_N | -0.1309 [-0.3730, +0.0100] | -0.0539 | 0 / 2 | -0.1983 | -0.0090 | -0.0678 |
| dev / pyramid_L0 | 115 / 579 | T_minus_TW | -0.0063 [-0.1432, +0.1278] | -0.0042 | 3 / 1 | -0.1240 | -0.0515 | -0.1713 |
| dev / pyramid_L1 | 179 / 682 | T_minus_A | -0.3708 [-0.7776, -0.0803] | -0.2846 | 10 / 8 | -0.4495 | -0.0732 | -0.1043 |
| dev / pyramid_L1 | 179 / 682 | T_minus_N | -0.0784 [-0.2720, +0.1104] | -0.1239 | 4 / 4 | +0.0527 | +0.0385 | +0.0316 |
| dev / pyramid_L1 | 179 / 682 | T_minus_M | +0.0552 [-0.1416, +0.2946] | -0.0254 | 6 / 2 | +0.1097 | +0.0387 | -0.0511 |
| dev / pyramid_L1 | 179 / 682 | R_minus_N | -0.0480 [-0.1151, +0.0148] | -0.0409 | 2 / 2 | +0.0344 | +0.0123 | +0.0956 |
| dev / pyramid_L1 | 179 / 682 | T_minus_TW | +0.0095 [-0.1792, +0.1913] | -0.0044 | 6 / 3 | +0.1405 | +0.0290 | +0.0128 |
| dev / pyramid_L2 | 226 / 555 | T_minus_A | -0.0881 [-0.2932, +0.1310] | -0.1781 | 5 / 3 | -0.4121 | -0.1000 | -0.5794 |
| dev / pyramid_L2 | 226 / 555 | T_minus_N | +0.0403 [-0.0724, +0.1865] | +0.0200 | 2 / 1 | -0.0487 | -0.0004 | -0.0738 |
| dev / pyramid_L2 | 226 / 555 | T_minus_M | -0.0160 [-0.1690, +0.1277] | -0.0262 | 5 / 1 | +0.0235 | +0.0155 | -0.0102 |
| dev / pyramid_L2 | 226 / 555 | R_minus_N | -0.0139 [-0.0526, +0.0284] | -0.0350 | 0 / 2 | +0.0030 | -0.0120 | +0.0185 |
| dev / pyramid_L2 | 226 / 555 | T_minus_TW | -0.0537 [-0.1598, +0.0259] | -0.0741 | 3 / 1 | -0.0299 | +0.0137 | +0.0260 |

## Official diagnostic loss

The same complete sparse official mask BCE, gain and normalization are used for every arm. No teacher or output regularizer is added.

| Split | Candidate BCE (A / N / T / R / M / TW) | Image-macro BCE (same order) |
|---|---|---|
| fit | 2.39868563 / 2.24942133 / 2.20909027 / 2.24092109 / 2.20818754 / 2.24689685 | 2.04503480 / 1.93515045 / 1.90734261 / 1.92945360 / 1.90144059 / 1.93078986 |
| dev | 2.56644822 / 2.57340981 / 2.58734462 / 2.57771157 / 2.59365368 / 2.59208814 | 2.13413551 / 2.13833294 / 2.14684351 / 2.14211997 / 2.15888420 / 2.15709415 |

## Decision and validity

Prespecified screen status: **prespecified_screen_not_passed**. Other metric improvements require an explicit magnitude and damage review; this script does not authorize a new experiment.

T-M compares equally trained identical architectures. T-TW is a frozen-parameter input intervention; it can expose reliance on source correspondence but alone cannot establish superiority over the baseline. Wrong-source fallback cases are excluded from the separately reported same-predicted-class subset; matching is based on original predicted class, never GT class.

A/N dev replay is checked at every permanent identity and all five historical mask metrics within 1e-12; original coefficient replay uses atol=rtol=3e-5. Full prototypes, original predicted boxes, process_mask(upsample=True), and the original letterbox inverse are unchanged. AUC uses continuous input-grid logits and the fixed prediction-box support. Undefined AUC/FPR retain their IoU/Mask75 evaluation.

Full input hashes, checkpoint/initialization/order checks, buffer checks, donor counts and numeric replay errors are in AUDIT.json; all pair/group metrics are in SUMMARY.json. Layer indices L0/L1/L2 denote P3/P4/P5, not object-size classes.
