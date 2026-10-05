# YOLO26 mechanism-gate uncertainty audit

Read-only bootstrap audit of the existing fixed cache; no model forward, training, or GT edits.
Bootstrap unit is image (all GT rows from a sampled image stay together). Seed: 20260905.

## GT-level rates (95% image-cluster bootstrap CI)

| Scope | N GT | Total failure | Relation failure | Miss |
|---|---:|---:|---:|---:|
| ALL | 7374 | 19.84% [18.50%, 21.14%] | 11.05% [10.19%, 11.85%] | 3.63% [3.19%, 4.11%] |
| PigLife_public_test | 4474 | 8.81% [7.86%, 9.75%] | 7.53% [6.70%, 8.37%] | 0.29% [0.14%, 0.45%] |
| FaroPigSeg_test | 1752 | 45.49% [43.17%, 47.95%] | 20.21% [18.29%, 22.21%] | 9.70% [8.26%, 11.23%] |
| BamaPig2D_eval | 1148 | 23.69% [20.93%, 26.35%] | 10.80% [8.94%, 12.82%] | 7.40% [5.94%, 8.89%] |

## Scene-stratified total failure

| Dataset | Scene | N GT | Rate (95% CI) |
|---|---|---:|---:|
| PigLife_public_test | Isolated | 1202 | 4.41% [3.37%, 5.54%] |
| PigLife_public_test | Near/Dense | 907 | 7.72% [5.99%, 9.39%] |
| PigLife_public_test | Touching | 2365 | 11.46% [10.03%, 12.84%] |
| FaroPigSeg_test | Isolated | 140 | 41.43% [32.87%, 50.00%] |
| FaroPigSeg_test | Near/Dense | 452 | 42.92% [38.67%, 47.07%] |
| FaroPigSeg_test | Touching | 1160 | 46.98% [44.15%, 49.72%] |
| BamaPig2D_eval | Isolated | 283 | 19.79% [15.33%, 24.55%] |
| BamaPig2D_eval | Near/Dense | 33 | 12.12% [3.03%, 22.86%] |
| BamaPig2D_eval | Touching | 832 | 25.48% [22.28%, 28.47%] |

## Existing NMS threshold sensitivity

The following values are copied from the cache-only comparison; they are not re-estimated here.

| Method | Threshold | Total failure | Relation failure | O -> C | C -> failure |
|---|---:|---:|---:|---:|---:|
| BASELINE | - | 19.84% | 11.05% | 0 | 0 |
| MASK_NMS | 0.7 | 13.59% | 3.72% | 460 | 0 |
| MASK_NMS | 0.75 | 13.74% | 4.08% | 449 | 0 |
| MASK_NMS | 0.8 | 14.04% | 4.53% | 427 | 0 |
| MASK_NMS | 0.85 | 14.25% | 4.96% | 411 | 0 |
| MASK_NMS | 0.9 | 14.63% | 5.48% | 384 | 0 |
| MASK_NMS | 0.95 | 15.31% | 6.33% | 334 | 0 |
| SOURCE_AWARE | 0.7 | 14.05% | 4.33% | 426 | 0 |
| SOURCE_AWARE | 0.75 | 14.13% | 4.57% | 420 | 0 |
| SOURCE_AWARE | 0.8 | 14.24% | 4.77% | 412 | 0 |
| SOURCE_AWARE | 0.85 | 14.43% | 5.14% | 398 | 0 |
| SOURCE_AWARE | 0.9 | 14.73% | 5.57% | 377 | 0 |
| SOURCE_AWARE | 0.95 | 15.36% | 6.39% | 330 | 0 |

## Interpretation boundary

The intervals quantify uncertainty for this fixed checkpoint/cache and do not establish causality or a deployable postprocessor. The threshold scan remains a control: it shows duplicate suppression changes the error mix, but does not identify the mechanism responsible for residual low-overlap failures.
