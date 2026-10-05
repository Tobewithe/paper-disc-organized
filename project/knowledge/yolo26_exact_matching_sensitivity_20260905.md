# YOLO26 exact-runtime matching sensitivity

## Scope

- Input: `experiments/yolo26_formal_1280_rect_replay_20260905/full_exact_8427_20260905/yolo26_gt_candidate_trace.csv`.
- Runtime and geometry: Ultralytics `8.4.27`, `imgsz=1280`, `rect=True`, 72 reconstructed-sample images.
- This is a read-only reclassification of retained candidate metrics. No inference, checkpoint, GT, threshold used by the original replay, or training was changed.

## Rule

A candidate is associated at threshold `t` when `box IoU >= t OR coverage >= t OR (purity >= t AND coverage >= 0.10)`. For each failed GT, the table counts whether any associated candidate exists, then whether any such candidate passes the original `conf=0.05` and belongs to Top-300.

| t | State | Failed GT | Any associated | Any conf-qualified | Any Top-300 | Median associated candidates |
|---:|---|---:|---:|---:|---:|---:|
| 0.30 | O | 24 | 24 | 24 | 24 | 109 |
| 0.30 | M | 8 | 8 | 8 | 8 | 216 |
| 0.30 | X | 21 | 21 | 21 | 21 | 221 |
| 0.50 | O | 24 | 24 | 24 | 24 | 96 |
| 0.50 | M | 8 | 8 | 8 | 8 | 211 |
| 0.50 | X | 21 | 21 | 21 | 21 | 218 |
| 0.70 | O | 24 | 24 | 24 | 24 | 92 |
| 0.70 | M | 8 | 8 | 8 | 8 | 188 |
| 0.70 | X | 21 | 21 | 20 | 20 | 209 |

## Boundary

For this reconstructed PigLife/Touching trace, threshold variation does not make candidate absence or Top-K loss a prevalent explanation for O/M failures. This does not establish that mask construction or candidate crowding is causal: association is a screening rule, the sample is reconstructed, and `SPLIT_TYPE` labels for the archived formal sample were not recovered.
