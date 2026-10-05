# YOLO26 1280/rect reconstructed-sample replay

## Scope

- Run: `experiments/yolo26_formal_1280_rect_replay_20260905/full_reconstructed`
- Protocol geometry: `imgsz=1280`, `rect=True`, input `736x1280`, `19,320` raw positions/image, `Top-300`, `conf=0.05`, formal relation score `0.50`.
- Runtime: Ultralytics `8.4.100`; archived baseline reference is `8.4.27`. This is geometry-aligned, not exact-runtime replay.
- Checkpoint and manifest are the archived PigLife YOLO26-seg inputs; no training or checkpoint edits.

## Sample provenance

The original 118-row sample CSV was absent. A reconstructed CSV was generated from the retained `gt_analysis.csv`, selecting 25 O, 9 M, and 25 X PigLife/Touching rows, then one nearest C row per failure. It contains 118 rows, 59 failure rows, and 59 control rows, but 49 unique controls and 72 unique images. It must not be treated as the archived 54-image/30-control sample.

## Replay result

| Item | Result |
|---|---:|
| Images processed | 72 |
| Raw positions / image | 19,320 (all 72) |
| Top-300 parity | 72/72 |
| Traced failure GT rows | 53 |
| Traced control GT rows | 49 |
| Relation components O/M/X | 24/4/13 |
| Onset: LOW_SCORE | 1 X component |
| Onset: MASK_RELATION_FAILURE | 40 components |
| Onset: TOPK_RANKING_LOSS | 0 |
| Onset: NO_SEPARATE_GOOD_CANDIDATES | 0 |
| Onset: DUPLICATE_TOPK_SURVIVAL | 0 |

## Interpretation boundary

The replay does not reproduce the archived `25 O duplicate-survival, 2 LOW_SCORE, 16 MASK_RELATION_FAILURE` distribution. The discrepancy is confounded by reconstructed sampling, retained prediction/cache provenance, and the 8.4.100 versus 8.4.27 runtime difference. Therefore this result is an audit signal and a reproducibility artifact, not evidence that duplicate survival is absent or that mask construction is causal.

## Gate status

Mechanism Gate remains open. The exact-runtime replay is now available, but the original sample list remains missing. Do not merge this output with the 1024-square exploratory cache or the archived 1280/rect report.

## Exact-runtime replay update (2026-09-05)

- The isolated Ultralytics `8.4.27` wheel was imported through `PYTHONPATH`; the Conda `pytorch` environment was not modified.
- A 4-image case-only smoke passed, then the full reconstructed sample completed on 72 images.
- Exact-runtime aggregate results are unchanged from the 8.4.100 geometry-aligned replay: 19,320 raw positions/image, Top-300 parity 72/72, 53 traced failure GT rows, 49 controls, O/M/X components 24/4/13, 40 `MASK_RELATION_FAILURE`, 1 `LOW_SCORE`, and zero `DUPLICATE_TOPK_SURVIVAL`.
- Per-row CSV hashes are not identical across runtimes, so only these aggregate invariants are treated as reproduced. The original sample list is still missing; the Mechanism Gate remains open.

## Candidate-stage descriptive signal

- In the exact-runtime reconstructed trace, all 53 failure GT rows reached Top-300; no failure was attributed to Top-K ranking loss.
- The median number of relation-window candidates was 156 for failures versus 88 for the 49 reconstructed C controls. This is a candidate-crowding association under the fixed trace, not proof that crowding causes the mask/relationship failure.
- The failure-vs-control medians (all failures) were best box IoU `0.932`, best mask IoU `0.857`, and best score `0.926`; these summaries are descriptive and remain sample/provenance scoped.

## Files

- `experiments/yolo26_formal_1280_rect_replay_20260905/yolo_dense_failure_samples_reconstructed_20260905.csv`
- `experiments/yolo26_formal_1280_rect_replay_20260905/full_reconstructed/inference_config.json`
- `experiments/yolo26_formal_1280_rect_replay_20260905/full_reconstructed/pipeline_inspection.json`
- `experiments/yolo26_formal_1280_rect_replay_20260905/full_reconstructed/yolo26_failure_onset_distribution.csv`
- `experiments/yolo26_formal_1280_rect_replay_20260905/full_reconstructed/yolo26_gt_candidate_trace.csv`
- `experiments/yolo26_formal_1280_rect_replay_20260905/full_exact_8427_20260905/`

## Exact-runtime multi-candidate union oracle (2026-09-05)

- Reused retained exact-runtime boxes, mask coefficients, prototypes, and Top-300 flags; no model forward or training was run.
- On the reconstructed 59 failure rows, Top-300 best-single mask IoU >= 0.50 occurred for O/M/X = 25/25, 9/9, 23/25.
- The best two-candidate union reached IoU >= 0.50 for O/M/X = 5/25, 8/9, 19/25; all associated Top-300 union reached 25/25, 9/9, 23/25.
- This is an oracle upper bound, not a deployable postprocessor. It provides evidence for complementary candidate structure in many M/X rows, while O is usually already explained by one candidate or remains unexplained by a pair. Because the sample is reconstructed, this does not close the Mechanism Gate or establish causality.
- Artifacts: `experiments/yolo26_formal_1280_rect_replay_20260905/union_oracle_exact_20260905/union_oracle_gt.csv`, `union_oracle_summary.json`, and `union_oracle_report.md`.
