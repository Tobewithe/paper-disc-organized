# Full Local Comparison Integrity Audit

**Date:** 2026-09-11  
**Scope:** `experiments/coco_clean_20260911/diagnostics/full_local_comparison_20260911` and the bounded implementation/provenance files named in the review request  
**Review independence:** Fresh same-family reviewer; provisional  
**Verdict:** **WARN — no fatal implementation or data-integrity bug found**

The completed full-val table is valid within its stated frozen-candidate, head-only diagnostic scope. The audit independently confirmed the 5,000-image COCO val2017 population, official COCOeval use, exact checkpoint identities, and exact reconstruction of the reported AP and custom R75 aggregates from the saved evaluator artifacts. The audit found no blocker that invalidates the main table. The warnings below constrain the claims and identify provenance/reporting fixes needed before treating this as a durable final experiment record.

## Integrity checks

| Check | Status | Evidence |
|---|---|---|
| Original GT provenance | PASS | Evaluation loads `data/annotations/instances_val2017.json`; independently hashed as `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. It contains 5,000 images, 36,781 annotations (36,335 non-crowd and 446 crowd), and 80 categories. The evaluated image IDs equal the complete, unique GT image-ID set. GT is loaded only after all arm masks are decoded for an image and is used for evaluation/spatial diagnostics, not inference. |
| Model and cache provenance | PASS with recording gaps | All 15 recorded head checkpoint hashes match the files used by the run. The BCE training-protocol hash and cache-protocol hash match. The cache protocol records the official base-weight hash `16b636f...`, zero train images, 5,000 val images, and official `pycocotools` mask rasterization. The current Dice protocol, normalizers, imported helper scripts, GT JSON, and census manifest were independently checked, but their hashes are not all recorded in the evaluation protocol. |
| Official COCO evaluation | PASS | `eval_full_local_comparison.py` constructs `COCOeval(gt, dt, "segm")`, restricts `imgIds` to the complete set, and calls `evaluate`, `accumulate`, and `summarize` with default COCO IoU thresholds, area ranges, and max detections. Independent recomputation from every saved `cocoeval_*.npz` gives `ap_max_absdiff = 0.0` against `summary.csv`. The common cached box AP is `0.5255434778769922`. |
| Full-val scope | PASS with scope limitation | `COMPLETE.json` reports 5,000 images and 16 arms; the protocol image list is unique and exactly equals all COCO val2017 image IDs. This is a full-val diagnostic, but the same validation split was used during method development. It is exploratory evidence, not a pristine held-out final test. |
| Decoder and zero parity | WARN | The initial mask is constructed with the frozen prototype/coefficient decoder and compared per image with `ops.process_mask` using `assert torch.equal(initial, stock)`. Thus a normal Python run would abort on any mismatch. However, `decode()` returns literal `0`, so `initial_replay_pixel_xor: 0` is not an accumulated measured XOR count. Assertions can also be disabled under optimized Python. This is a reporting/provenance flaw, not evidence that the completed normal run mismatched. |
| Aggregate counts | PASS | `gt_recovery.csv` has 581,360 rows, exactly `16 × 36,335` non-crowd GT instances. Independent aggregation exactly reproduces all reported R75 all/high/low values (`r75_max_absdiff = 0.0`). |
| Seed aggregation | WARN | `summary.csv` contains the individual seed arms; the evaluator itself does not create aggregate rows. Claims must use paired three-seed means for seeds 0/1/2 and must not select individual seeds. `PAIRED_ANALYSIS.json` uses fixed three-seed means and 2,000 paired image-cluster bootstrap draws for the custom ratios. Its confidence intervals are exploratory percentile intervals without multiplicity adjustment, and it does not provide AP confidence intervals. |
| Oracle scope | PASS | No evaluation GT is supplied to any arm's inference path. GT is used in training the small residual heads and for post-decoding evaluation/diagnostics. Detection boxes, classes, scores, prototypes, and candidate coefficient vectors remain frozen. Arms can remove masks after thresholding, so the number of non-empty predictions can differ; this behavior is part of the evaluated method and should remain disclosed. |

## Implementation assessment

`local_coefficient_head.py` implements the stated shared 73-to-128 MLP and either averages four residual coefficient vectors (global arm) or bilinearly interpolates them within the predicted box (local arm). The final layer is zero-initialized. `train_local_coefficients.py` and `train_local_dice.py` use fixed seeds 0/1/2, paired global/local configurations, a fixed epoch-15 endpoint, and frozen cached features/candidates. There is no full-val checkpoint selection in the reviewed scripts. The evaluation code normalizes the frozen features with the matching family normalizer, applies the saved residual head, reconstructs masks, encodes them for COCO, and leaves scores/classes/boxes fixed.

The documented training/evaluation coordinate-sampling mismatch remains a bounded implementation limitation: the training loss samples original-image GT pixels through cached coordinates, while evaluation reconstructs and resizes full masks. It does not invalidate the run, but the result supports this exact implementation rather than an abstract locality mechanism.

## Claim corrections

1. **Do not claim broad local-head superiority.** For BCE, local minus global has a small positive high-ICI R75 effect of `+0.330` percentage points (paired image interval `[+0.123, +0.547]`) and pair-recovery effect of `+0.286` points (`[+0.090, +0.483]`). Overall and low-ICI R75 intervals cross zero, AP is essentially tied, and mean AP75 is slightly lower. For Dice, local adds no supported high-ICI R75 or pair-recovery benefit over Dice global.
2. **Do not describe the Dice gain as pure instance separation.** Dice global versus initial improves high-ICI R75 by `+1.837` points (`[+1.263, +2.449]`) and spatial IoU by `+0.330` points, while high-ICI coverage falls `-2.395` points and same-neighbor/GT leakage falls `-0.962` points. Relative to equal BCE global, Dice raises coverage by `+1.849` points, same-neighbor/GT leakage by `+1.997` points, and IoU by `+0.333` points. The gain mixes coverage, leakage, and shape effects.
3. **Name R75 precisely.** It is a custom micro-recall over non-ignored GT instances using COCO's IoU-0.75 matching and maxDet=100, and pair recovery is a custom same-category/bbox-overlap diagnostic. Neither is official COCO recall nor precision-matched. AP/AP50/AP75 are official COCOeval metrics.
4. **Keep the experiment scope explicit.** This is a frozen-candidate, coefficient-residual, head-only comparison on reused COCO val2017. It supports diagnostic mechanism claims for these cached predictions; it does not establish end-to-end detector improvement or held-out generalization.

## Required record fixes

- Replace the hard-coded parity return with a measured per-image XOR/mismatch count, accumulate it, retain an explicit failure gate, and record whether Python assertions were enabled.
- Extend `protocol.json` with direct hashes for the GT JSON, census manifest, Dice training protocol, both normalizers, `spatial_coefficient_head.py`, and `frozen_mechanism_probe.py`; distinguish BCE and Dice protocol fields.
- Record hashes for `summary.csv`, `gt_recovery.csv`, `spatial.csv`, and each COCOeval artifact in the completion record.
- Report paired three-seed aggregates and the paired-analysis limitations alongside any local-versus-global or Dice-versus-BCE claims.

## Final determination

No reviewed defect makes the main full-val comparison table invalid. Acceptance is **provisional/WARN** because the parity counter is not actually measured, the evaluation protocol omits several direct provenance hashes, and the strongest mechanism interpretations require the narrower wording above. This audit does not accept any broader experiment or paper claim outside the bounded full-local-comparison output.
