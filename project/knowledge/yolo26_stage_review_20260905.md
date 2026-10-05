# YOLO26 Stage Availability Review (2026-09-05)

## Review Status

- Scope: read-only review of `tools/analyze_yolo26_stage_availability.py`, its unit test, `experiments/yolo26_stage_availability_full_20260905_v1`, and `experiments/yolo26_trace_replay_full_20260905_v1`.
- Review class: same-family, provisional. This is a code-and-artifact integrity review, not an independent replication or a method evaluation.
- Verdict: **WARN.** The stage counts are supported as frozen-trace, per-failed-GT availability measurements. They do not approve the candidate-scorer/ranking Gate or a method route.
- No GPU inference was run. `C:\Dpan\envsfiles\CondaData\envs\pytorch\python.exe -m unittest tools.test_yolo26_stage_availability -v` passed (2 tests).

## What The Evidence Supports

The replay audit completed without a violation for both input traces: 426 PigLife images plus 160 FaroPigSeg images, 8,327 final masks total. The audit recorded exact replay equality for retained raw arrays, Top-K IDs, final predictions, and final-source mappings; old raw `prototype`, coefficient, and box decoded with `process_mask_native` in batches of two gave final-mask XOR total and maximum of zero. This supports that the stage audit reads a stable frozen output set under the stated local runtime.

For the failed-GT denominator only, the reported availability counts are:

| Dataset | Failed GT | Raw | Top-K | Confidence | Final |
|---|---:|---:|---:|---:|---:|
| PigLife | 394 | 390 | 390 | 373 | 373 |
| FaroPigSeg | 797 | 746 | 698 | 553 | 553 |

Each count means that at least one candidate independently reaches mask IoU >= 0.50 for that GT at the named stage. It does not allocate candidates across GT, require a one-to-one feasible assignment, establish a corrected taxonomy, or estimate deployable recall/AP.

## Raw-Count Reuse And Screen Proof

The R006 source calculates `raw_good_count` for each GT before its separate maximum-cardinality assignment: `tools/run_yolo26_candidate_perturbation.py:250-296`, followed by assignment only at `:297-314`. Thus the stage script's reuse of `raw_good_count` is sufficient for the stated independent per-GT raw-availability numerator; it is not a reuse of the assignment result.

The screen is lossless for the declared IoU threshold. `process_mask_native` crops every reconstructed binary mask to the candidate box (`ultralytics/utils/ops.py:477-497` and `:531+`). Therefore, for a mask M and GT G,

`IoU(M,G) <= coverage(M,G) <= coverage(box envelope,G)`.

R006 decodes every candidate whose integer-enlarged box envelope has GT coverage >= 0.50 (`tools/run_yolo26_candidate_perturbation.py:251-256`). The floor/ceil envelope is a superset of the crop. No box-purity filter or candidate quota is applied to this inclusion path. This proves that it cannot omit a raw candidate satisfying the declared mask-IoU >= 0.50 condition. It does not make the result a simultaneous matching result or a deployment simulation.

The stage script separately reconstructs the relevant Top-K masks, checks the strict `score > 0.05` source set equals the framework mask-input source set, and checks final-RLE eligibility against the reconstructed source mask (`tools/analyze_yolo26_stage_availability.py:57-94`). This supports the observed Top-K, confidence, and final columns for the same per-GT definition.

## Bootstrap Denominator

`bootstrap_rate` groups only the rows supplied to it by image, resamples those images, and divides sampled available-GT counts by sampled failed-GT counts (`tools/bootstrap_yolo26_gate.py:25-36`). For `ALL_FAILED`, the point denominators are 394 PigLife failed GT across 203 images and 797 FaroPigSeg failed GT across 152 images; the summary records both values. The intervals are therefore image-cluster bootstrap intervals conditional on images with at least one baseline failed GT and on the fixed failure taxonomy. They are not benchmark-wide GT uncertainty intervals, nor do they include uncertainty from the original failure classifier, GT annotations, threshold choice, or the GT-guided raw screen.

## Prototype And Provenance Boundary

The replay audit adds the missing new-forward check: the actual prototype passed to `process_mask_native` exactly matches that new forward's raw `one2one.proto`; it also proves old raw tensor replay and old final-mask reconstruction parity. The legacy trace NPZ does not contain its historical `framework_prototype` argument. Consequently, the audit cannot prove the historical call's prototype identity directly, even though old raw prototype/coefficient/box reconstruct to the old final RLE with XOR zero. Any claim must retain this distinction.

`experiments/yolo26_stage_availability_full_20260905_v1/provenance.json` hashes the stage script, native mask implementation, legacy taxonomy input, R006 table inputs, and replay summary. This is adequate for a trace-bound audit, but the minimal provenance improvement is to also carry the replay `run_metadata.json` hash and its recorded `diagnostic_inferencer.py` hash into the stage provenance. That makes the new-forward prototype assertion part of the explicit stage evidence chain instead of only an indirect dependency through `summary.json`.

## Claim Language Review

`research-wiki/yolo26_stage_availability_20260905.md` already correctly says “per-GT availability,” rejects causal ranking interpretation, and identifies R006 as a GT-guided output-set upper bound. Keep “upper bound” attached to the R006 GT-informed output-set counterfactual, not to the stage availability percentages themselves. The latter are thresholded existence rates within frozen candidates.

Two phrases need tightening before use in a proposal or paper:

1. `RESEARCH_BRIEF.md:56` says FaroPigSeg has a “raw formation gap.” Replace it with “a 51/797 per-failed-GT raw-mask availability deficit at IoU >= 0.50.” The current evidence does not isolate mask formation from box localization, prototype representation, or GT semantics. Likewise, PigLife's 4/394 raw-unavailable count only says this frozen candidate set rarely lacks an individually qualifying mask among its baseline failed GT.
2. `RESEARCH_BRIEF.md:58` still lists batch-2 parity, greedy one-to-one matching, and I/L/S box-purity screening as R006 blockers. The completed R006 code uses batch-2 final-source parity and a maximum-cardinality assignment, while its raw inclusion screen has no purity filter. Replace this stale blocker list with the remaining claim boundary: GT-guided output selection and per-GT availability do not establish deployable ranking benefit, causal mechanism, AP improvement, or cross-architecture generalization.

The current wiki phrase “FaroPigSeg has a material raw-mask gap” should also be replaced by the observed count and scope. “Material” is an unsupported magnitude judgment; “raw-mask formation” is a mechanism label. The precise result is the 51/797 raw-unavailable slice, plus 48 first lost at Top-K and 145 first lost at confidence, all conditional on baseline failed GT.

## Gate Decision

Do not approve the candidate-scorer/ranking Gate. The stage audit narrows where individual mask-IoU availability disappears in two frozen traces, but it has no deployable selector, no non-GT candidate selection rule, no AP evaluation, no independent causal control, and no cross-architecture confirmation. The minimal next evidence is an explicitly pre-registered deployable intervention with overall AP, failure taxonomy, and adjacent-instance regressions reported separately; it must not use the stage table's GT-conditioned choices at inference time.
