# Threshold Coverage Control Experiment Audit

**Date:** 2026-09-11  
**Auditor:** fresh Codex same-family reviewer (read-only semantic audit)  
**Review status:** provisional; same-family, not cross-family accepted  
**Bounded scope:** `threshold_coverage_control.py`, `frozen_mechanism_probe.py`, `eval_full_local_comparison.py`, `run_threshold_coverage.sh`, completed `threshold_coverage_control_20260911` outputs, and its frozen `full_local_comparison_20260911` input  
**Overall verdict:** **WARN**  
**Integrity status:** **WARN — no fatal implementation, provenance, or numeric-reconstruction defect; the development coverage match did not transfer to evaluation and the split is reused COCO val2017.**

## Executive finding

The completed experiment is a real-GT, frozen-candidate, fixed-bbox-ownership diagnostic. The threshold parameters were selected from 300 development images and locked before the 600-image evaluation pass. I reconstructed the split, both selected thresholds, development confirmations, all saved means, and all 2,000 image-cluster bootstrap contrasts from the raw remote CSV files. The largest reconstruction difference was `4.44e-16`; all recorded input/output hashes matched; zero-threshold decoding and prior spatial metrics replayed exactly.

The primary claim limitation is substantive: development high-ICI coverage was matched essentially exactly, but evaluation high-ICI coverage was not. The learned BCE arm was `1.290 pp` below its threshold control (95% interval `[-1.851, -0.772]`), and learned Dice was `1.102 pp` below (`[-1.567, -0.648]`). Consequently, this output supports a **development-calibrated threshold transfer test whose coverage transfer failed**. It does not support an evaluation-set “matched-coverage,” “equal-coverage,” “lossless separation,” or shrinkage-controlled causal claim.

No fatal bug was found, and the saved tables are not invalidated when described within that scope.

## A. Ground-truth provenance and parameter isolation: PASS

- COCO instance GT is loaded from `data/annotations/instances_val2017.json`; the exact GT and ICI-manifest hashes are frozen in `diagnostics/threshold_coverage_control_20260911/protocol.json:940`. `threshold_coverage_control.py:44` first consults GT only after the cached logits, boxes, coefficients, and fixed ownership mapping have been loaded and the zero decoder has been constructed (`threshold_coverage_control.py:29`, `threshold_coverage_control.py:35`, `threshold_coverage_control.py:39`).
- Fixed ownership is annotation-to-original-prediction matching, not an oracle mask assignment. The supplied upstream helper constructs one-to-one official COCO bbox matches at IoU `0.5` (`frozen_mechanism_probe.py:93`, `frozen_mechanism_probe.py:102`, `frozen_mechanism_probe.py:106`). The full comparison decodes every arm before reading GT ownership (`eval_full_local_comparison.py:23`, `eval_full_local_comparison.py:99`, `eval_full_local_comparison.py:100`).
- The development IDs come exactly from the frozen 300-image `local_coeff_eval_20260911/protocol.json`. Evaluation IDs equal the first 600 SHA-256-ranked IDs from the prior 5,000-image protocol after excluding all 300 development IDs (`threshold_coverage_control.py:93`, `threshold_coverage_control.py:99`). Reconstruction found 300 unique development IDs, 600 unique evaluation IDs, zero overlap, and exact agreement with that hash rule.
- Threshold selection uses only development high-ICI rows (`threshold_coverage_control.py:127`, `threshold_coverage_control.py:130`). It targets the mean of the three learned seeds, requires a monotone bracket, and interpolates once (`threshold_coverage_control.py:132`, `threshold_coverage_control.py:134`, `threshold_coverage_control.py:136`). `SELECTED_THRESHOLDS.json` is written before development confirmation and before `run(evaluate, ...)` (`threshold_coverage_control.py:138`, `threshold_coverage_control.py:140`, `threshold_coverage_control.py:146`). File timestamps agree with this sequence: thresholds `18:56:59`, confirmation `18:57:10`, evaluation CSV `18:57:29` Asia/Shanghai.
- The full 5,000-image comparison, including these evaluation IDs, existed before this control was designed. Thus the code-level parameter isolation passes, but this is not a pristine or previously unseen test split; that limitation is assessed under E.

## B. Score normalization: PASS

No metric is divided by the model's own maximum, minimum, mean, or another prediction-derived scale. `coverage`, `same_neighbor`, `neighbor`, and `background` are pixel counts divided by the matched target's valid GT area, while `mask_iou` uses the GT/prediction union denominator (`threshold_coverage_control.py:53`, `threshold_coverage_control.py:63`, `threshold_coverage_control.py:65`; equivalent upstream definition at `frozen_mechanism_probe.py:147`). Independent checking recovered the identity `mask_iou = coverage / (1 + neighbor + background)` to `2.22e-16`, and every value was finite. These are GT-area-normalized diagnostic ratios; they should not be described as probabilities, but they are not self-normalized scores.

The learned arm's mean development coverage is deliberately used as the scalar threshold target (`threshold_coverage_control.py:132`). That is an explicitly labeled comparator-calibration rule, not ground truth and not score inflation.

## C. Result existence and numeric reconstruction: PASS

`COMPLETE.json:2` records `COMPLETE`, 300 development images, 600 evaluation images, measured decoder XOR `0`, and spatial replay maximum error `0.0` (`COMPLETE.json:4`, `COMPLETE.json:6`). Every hash in `COMPLETE.json:9` matched its remote output, and every input hash in the protocol matched the referenced artifact. Local and remote source hashes also matched the protocol: threshold script `916390ca...`, helper `e1cd6a48...`, main comparison script `9bfaea1d...`, prior spatial input `3578a56e...`, and threshold evaluation rows `279fff5f...`.

Independent reconstruction from the raw remote files produced:

| Check | Reconstructed result |
|---|---:|
| Development grid | 77,496 rows; 24 thresholds; 3,229 unique mapped targets per threshold |
| Evaluation grid | 10,785 rows; 3 fixed thresholds; 3,595 unique mapped targets per threshold |
| Development/evaluation curve maximum error | `0.0` / `0.0` |
| Selected threshold error | `0.0` for BCE and Dice |
| Development-confirmation error | `0.0` for BCE and Dice |
| `summary.csv` maximum error | `0.0` |
| 60 paired means/interval endpoints maximum error | `4.44e-16` |
| Zero-threshold spatial replay maximum error | `0.0` over 6,824 development/evaluation rows |
| Metric identity maximum error | `2.22e-16` |

The selected BCE threshold is `0.4008067121400105` and Dice threshold is `0.21247638627885146` (`SELECTED_THRESHOLDS.json:5`, `SELECTED_THRESHOLDS.json:9`). Development high-ICI confirmation gaps were only `-0.000982 pp` and `+0.000487 pp` (`DEVELOPMENT_CONFIRMATION.json:6`, `DEVELOPMENT_CONFIRMATION.json:12`), within the predeclared `0.25 pp` development tolerance.

## D. Called functions and decoder checks: PASS

All metric-producing functions in `threshold_coverage_control.py` are live: `measure()` is called inside `run()` (`threshold_coverage_control.py:115`, `threshold_coverage_control.py:120`), `cohort_summary()` produces both curves and confirmation summaries (`threshold_coverage_control.py:127`, `threshold_coverage_control.py:140`, `threshold_coverage_control.py:146`), and all five `FIELDS` feed saved summaries and paired contrasts (`threshold_coverage_control.py:150`, `threshold_coverage_control.py:165`, `threshold_coverage_control.py:169`). Imported hashing and JSON/CSV writers are also exercised. The runner invokes this exact script with the `pytorch` Conda interpreter and `--evaluation-images 600` (`run_threshold_coverage.sh:3`, `run_threshold_coverage.sh:5`).

Exact threshold decoding is: coefficient-prototype logits, bilinear resize to model-input resolution with `align_corners=False`, strict `z > tau`, crop by the unchanged predicted bbox, stock `scale_masks`, and final `> 0.5` at original resolution (`threshold_coverage_control.py:39`, `threshold_coverage_control.py:59`, `threshold_coverage_control.py:61`, `threshold_coverage_control.py:62`). At `tau=0`, every image is compared pixel-for-pixel with `ops.process_mask`; any XOR aborts (`threshold_coverage_control.py:40`, `threshold_coverage_control.py:43`). Each fixed-ownership spatial row is additionally compared against the prior run at tolerance `1e-10` (`threshold_coverage_control.py:68`, `threshold_coverage_control.py:72`). The completed measurements were XOR `0` and replay error `0.0`.

One future-run guard should be added: evaluation seed cohorts are explicitly required to equal the initial annotation queue (`threshold_coverage_control.py:156`, `threshold_coverage_control.py:158`), while development selection only requires the pooled family rows to be nonempty (`threshold_coverage_control.py:130`, `threshold_coverage_control.py:131`). A missing development seed could silently change the purported “three-seed mean.” This run is unaffected: independently checked BCE and Dice seeds 0/1/2 each contained exactly the same 3,229 development targets, including 705 high-ICI targets, with no duplicates.

## E. Scope assessment: WARN

- The primary analysis is conditional on original bbox ownership and excludes unmatched GT, as declared in `protocol.json:939`. Of 600 selected evaluation images, 588 contributed at least one mapped target: 3,595 targets total, with 475 high-ICI targets across 168 images and 3,120 low-ICI targets across 583 images. This supports a matched-detection mask-attribution diagnostic, not population-level instance recall.
- The learned result is the pointwise average of three observed seeds (`threshold_coverage_control.py:155`, `threshold_coverage_control.py:160`). The 2,000 paired bootstrap draws resample evaluation images and correctly preserve within-image pairing (`threshold_coverage_control.py:161`, `threshold_coverage_control.py:170`). All current cohorts were nonempty; minimum resampled target denominators were 3,139 (all), 326 (high), and 2,684 (low). Intervals remain exploratory, pointwise, conditional on the selected thresholds and observed three-seed average; they do not propagate development-selection uncertainty, seed variance, or multiplicity (`PAIRED_ANALYSIS.json:2`).
- Coverage calibration transferred poorly. On high ICI, learned BCE versus its threshold control was `-1.290 pp` coverage, 95% interval `[-1.851, -0.772]` (`PAIRED_ANALYSIS.json:260`); learned Dice was `-1.102 pp`, `[-1.567, -0.648]` (`PAIRED_ANALYSIS.json:360`). All-target gaps were `-1.777 pp` and `-1.689 pp` (`PAIRED_ANALYSIS.json:60`, `PAIRED_ANALYSIS.json:160`). These intervals exclude zero.
- With the coverage mismatch, high-ICI same-neighbor differences do not isolate separation from mask shrinkage: BCE `-0.356 pp`, interval `[-0.886, 0.197]`, and Dice `-0.089 pp`, `[-0.682, 0.552]` (`PAIRED_ANALYSIS.json:270`, `PAIRED_ANALYSIS.json:370`). Background ratios are lower, but target coverage is also lower. Neither result is evidence of lossless separation.
- The code intentionally computes no task AP, precision-matched recall, or detector recovery metric; the protocol explicitly excludes those claims (`protocol.json:938`). The 600-image evaluation is drawn from the already explored COCO val2017 full comparison, so it cannot establish untouched-test or deployment generalization.

## F. Evaluation classification: real_gt

Classification is **`real_gt`**: regions and denominators come from original COCO `annToMask` annotations, crowd pixels are excluded, and the ICI cohort label comes from the frozen census manifest (`threshold_coverage_control.py:45`, `threshold_coverage_control.py:47`, `threshold_coverage_control.py:50`, `threshold_coverage_control.py:75`). The learned arm supplies a development coverage target, but that target is not represented as GT. The more specific subtype is **real-GT, frozen-candidate, fixed-bbox-ownership mask diagnostic**.

## Claim impact

| Claim | Audit assessment |
|---|---|
| Parameters were fixed on disjoint development IDs before evaluation | Supported |
| Scalar thresholds matched learned high-ICI mean coverage on development | Supported |
| The evaluation comparison is matched/equal coverage | Unsupported; transfer failed |
| Learned residuals improve separation beyond uniform shrinkage at equal coverage | Unsupported by experiment A |
| Saved values describe fixed-ownership pixel behavior on the evaluated mapped cohort | Supported |
| Results establish AP, recall, unmatched-instance recovery, or deployment behavior | Unsupported; not evaluated |

## Required reporting constraints

1. Call this a development-calibrated threshold transfer control and report that coverage matching failed on evaluation.
2. Do not use “matched coverage,” “equal coverage,” “lossless,” or “pure separation” for experiment A's evaluation contrasts.
3. Keep the fixed-bbox, matched-target, reused-val2017, three-observed-seed, pointwise-bootstrap, and no-AP limitations adjacent to comparative claims.
4. For future reruns, assert the complete per-seed development annotation queues before averaging, and fail explicitly if any requested cohort is absent.

This same-family review is provisional. It verifies the supplied code and completed artifacts against the experiment-integrity checklist; it is not a cross-family acceptance decision.
