# YOLO26 stage availability audit (2026-09-05)

This is a frozen-trace, per-GT availability audit. It is not a causal ranking experiment and does not use `private_test_v1`, locked-test, guard-band, or target data.

## Evidence

- Full same-forward traces: PigLife 426 images / 4,843 final masks; FaroPigSeg 160 images / 3,484 final masks.
- Replay audit: 586 images, 8,327 final masks; raw tensors, Top-K IDs, final predictions, final-source mappings and batch-2 final-mask reconstruction all matched exactly; XOR total and maximum were zero.
- R006 full oracle baseline classifier matched the saved GT taxonomy for all 6,226 GT. Its recovery is the achieved outcome of a specified GT-guided output-set replacement, not a proven optimum over all replacements.
- Stage audit uses all R006 raw-good rows. The `.50` box-envelope screen is a lossless inclusion screen for mask IoU `.50`; no box-purity or candidate quota is used. Top-K and confidence source sets are checked against the framework mask-input source set.

## Failed-GT availability

| Dataset | Failed GT | Raw | Top-K | Confidence | Final |
|---|---:|---:|---:|---:|---:|
| PigLife | 394 | 390 (98.98%) | 390 (98.98%) | 373 (94.67%) | 373 (94.67%) |
| FaroPigSeg | 797 | 746 (93.60%) | 698 (87.58%) | 553 (69.39%) | 553 (69.39%) |

Image-cluster bootstrap 95% intervals are in `experiments/yolo26_stage_availability_full_20260905_v1/stage_summary.csv`.

The first unavailable stage counts were PigLife: raw 4, confidence 17, final-available 373; FaroPigSeg: raw 51, Top-K 48, confidence 145, final-available 553. “Final-available” means at least one final mask reaches mask IoU `.50` for that GT; it does not mean the fixed taxonomy becomes correct.

## Interpretation boundary

At mask IoU >= 0.50, 4/394 PigLife and 51/797 FaroPigSeg baseline failed GT have no individually qualifying reconstructed raw mask. This does not isolate box localization, prototype representation, mask quality at stricter thresholds, or annotation semantics. Faro also has 48 GT first losing availability at Top-K and 145 at confidence; PigLife has 17 first losing it at confidence. These are candidate-retention observations, not proof of a universal ranking cause or of an expected scorer gain.

The old traces did not capture the actual framework prototype argument. The replay proves exact new-forward prototype identity, exact old/new raw tensors and zero-XOR old-mask reconstruction; it cannot directly establish the missing historical call argument. Bootstrap intervals are conditional on images containing baseline failed GT, not uncertainty intervals for the whole benchmark.

R006 `ADDITION_REMOVAL` recovered 799 failed GT to class C, while `ADDITION` and `REMOVAL` recovered 65 and 74 respectively, with one C-to-failure regression under addition. These are GT-guided output-set counterfactuals; they are not deployable scorer evidence and do not pass the ranking/competition Gate.

The v1 R006 assignment was greedy. The completed v2 run uses maximum-cardinality assignment and a final-source XOR guard. All 1,136 GT with raw-good candidates received distinct source assignments within each image. All four conditions' 6,226 GT labels and all selected candidate IDs are identical to v1; this dataset had no observed greedy assignment loss. The per-GT raw counts above were computed before assignment. Independent stage review: `research-wiki/yolo26_stage_review_20260905.md` (same-family, provisional).

R006 v2 joint replacement recovered 322/394 PigLife failed GT (81.73%, 95% image-cluster interval 76.84%-86.46%) and 477/797 Faro failed GT (59.85%, 56.34%-63.21%). Correct-GT regressions were 0/4,080 and 1/955. Statistics resample all manifest images, whereas the stage intervals condition on images with eligible failed GT. These are different, explicitly recorded bootstrap populations.

The R006 script received a post-launch change limited to parity source sorting, bounds checks and metadata. The launch version was recovered to the SHA256 observed before launch, `fdc393636537cf38a358ab56e221f86e63ff1e2a63f8c1f0c524bdca7a7cade7`; see `experiments/yolo26_runtime_source_snapshot_20260905_v1/v2_run_sources.json`. A separate completed sorted-source parity check also compared all 8,327 final masks with zero XOR. No historical R006 outputs were rewritten.

## Artifacts

- `experiments/yolo26_trace_replay_full_20260905_v1/summary.json`
- `experiments/yolo26_stage_availability_full_20260905_v1/stage_gt.csv`
- `experiments/yolo26_stage_availability_full_20260905_v1/stage_summary.csv`
- `experiments/yolo26_stage_availability_full_20260905_v1/provenance.json`
- `experiments/r006_candidate_perturbation_fulltrace_20260905_v1/report.md`
- `experiments/r006_candidate_perturbation_fulltrace_20260905_v2/run_summary.json`
- `experiments/r006_fulltrace_statistics_20260905_v2/report.md`
- `experiments/r006_fulltrace_statistics_20260905_v2/v1_v2_comparison.json`
- `experiments/r006_fulltrace_statistics_20260905_v2/sorted_final_source_parity.json`
- `experiments/yolo26_runtime_source_snapshot_20260905_v1/provenance.json`
