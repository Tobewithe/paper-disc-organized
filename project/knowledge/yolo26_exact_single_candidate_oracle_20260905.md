# YOLO26 exact-runtime single-candidate oracle

## Scope

- Input: `experiments/yolo26_formal_1280_rect_replay_20260905/full_exact_8427_20260905/yolo26_gt_candidate_trace.csv`.
- The analysis uses retained per-candidate `mask_iou` only; it does not rerun inference and does not construct unions of multiple masks.
- The 53 failed GT rows come from the reconstructed O/M/X sample and remain audit-only.

## Result

Counts are failed GT rows with at least one candidate whose mask IoU against that GT reaches the threshold.

| State | Failed GT | Raw >=0.50 | Conf-qualified >=0.50 | Top-300 >=0.50 | Raw >=0.75 | Top-300 >=0.75 |
|---|---:|---:|---:|---:|---:|---:|
| O | 24 | 24 | 24 | 24 | 23 | 23 |
| M | 8 | 8 | 8 | 8 | 5 | 5 |
| X | 21 | 21 | 19 | 19 | 18 | 18 |

## Interpretation boundary

For O and M in this reconstructed trace, a moderately good single mask candidate is already present and survives Top-300, yet the final relation component still fails. This points toward mask/relationship assignment or multi-instance consistency as the next counterfactual target, not a broad candidate-absence explanation. X contains a small selection-sensitive slice (2/21 lose the `>=0.50` candidate between raw and conf), but this is not the dominant explanation. The oracle is single-candidate only and cannot establish causality or evaluate complementary mask unions.
