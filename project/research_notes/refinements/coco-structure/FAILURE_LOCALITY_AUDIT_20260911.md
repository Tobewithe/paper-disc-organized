# Failure-locality experiment integrity audit

**Date:** 2026-09-11  
**Audit route:** fresh Codex subagent, read-only semantic review  
**Assurance:** `same-family` / `provisional`  
**Overall verdict:** **WARN**  
**Fatal findings:** **none**

The run is internally coherent and the reported S012 result is usable as a conditional, GT-informed representation diagnostic. It is not a deployable-method result, COCO AP result, independent-image readout result, or causal proof about the coefficient head. Two accounting/selection issues and the resolution dependence of normalized distance bins require visible qualification.

## Material findings

### W1. Held-out GT does not tune the readout, but it does select the reported cohort

**Status:** WARN

For each direction of the spatial cross-fit, ridge weights use only `ytr`, threshold selection uses only training-fold scores and `ytr` (or shuffled `ytr`), and `yte` first enters the held-out metric. This part is clean (`failure_locality_probe.py:126-143`).

Before fitting, however, the code inspects the sampled labels in both folds and drops the entire seed when either fold has fewer than four positives or four negatives (`failure_locality_probe.py:110-113`). The summary then retains only targets with all six arms, all three sample seeds, and both directions (`summarize_failure_locality.py:38-47`). Thus held-out GT affects cohort inclusion, even though it does not affect fitted weights or thresholds.

Full-run attrition is substantial and is disclosed in the result report:

| Split | Bbox-matched targets | Complete readout targets | Too-small support | Class-insufficient targets |
|---|---:|---:|---:|---:|
| train | 2,060 | 1,579 | 431 | 50 |
| val | 2,629 | 1,955 | 624 | 50 |

Evidence: `cohorts.csv:5-10,14-19`; `readout_statuses.csv`; `FAILURE_LOCALITY_RESULTS_20260911.md:13-17`.

**Impact:** all readout claims must say "GT-conditioned, all-seed eligible targets." They cannot be generalized to all matched GT, all dense scenes, or deployable inference.

### W2. `partial_targets` undercounts excluded targets

**Status:** WARN

`bytarget` is built only from rows that exist in `readouts.csv`; `partial_targets=len(bytarget)-len(complete)` therefore counts only targets with some, but not all, readout rows (`summarize_failure_locality.py:38-41,112-113`). It omits targets that generated zero rows.

The full output reports `complete_targets=3534` and `partial_targets=1` (`READOUT_ANALYSIS.json:11181-11182`), but the actual accounting is:

- 4,689 bbox-matched targets;
- 3,534 complete targets;
- 1 target with partial nonzero rows;
- 1,154 bbox-matched targets with no readout rows.

The cohort table and status counters preserve the true exclusions, and the narrative report uses those correct counts. The defect is confined to the meaning of the JSON field `partial_targets`; it does not change any contrast because contrasts use the explicit complete-target intersection.

**Impact:** do not cite `partial_targets=1` as the total excluded or incomplete count. Rename it conceptually to `partial_nonzero_targets` and derive total exclusion from matched minus complete.

### W3. Normalized distance bins have a discrete-pixel floor; neighbor-core sensitivity is not fully resolved by target-size stratification

**Status:** WARN

Distances are integer-pixel Euclidean distances divided by `sqrt(valid target area)` and binned at 0.02 and 0.05 (`failure_locality_probe.py:19-23,76-85`). Consequently:

- if valid area is below 2,500 pixels, the smallest nonzero distance (1 px) cannot enter the `near` bin;
- if valid area is below 400 pixels, every nonzero distance enters `far` directly;
- for `same_neighbor_core`, a neighbor with valid area below 2,500 has a threshold below 1 px, so every rasterized neighbor pixel is classified as core (`failure_locality_probe.py:62-64`).

In the val high-ICI failure group, 118/166 targets have valid area below 2,500 and 54/166 are below 400. The report appropriately adds a standard COCO target-size sensitivity: among 73 medium/large failed targets, 66 have same-neighbor errors and the instance-macro far/core fractions remain 72.17%/71.87% (`LOCALITY_ANALYSIS.json:3062-3225`; `FAILURE_LOCALITY_RESULTS_20260911.md:32-40`). This removes the most severe current-target quantization (`valid_area<400`) but does not fully remove the near-bin floor: 25/73 medium/large targets still have fewer than 2,500 valid pixels after crowd exclusion. More importantly, target-size stratification does not stratify the neighboring component whose area defines the core threshold.

The top-10 concentration statistic is correctly computed and reported: 10 targets contribute 72.39% of pooled same-neighbor error pixels in the val high-failure group. The instance-macro far fraction (81.04%) is therefore the safer primary description than the pooled far fraction (90.75%).

**Impact:** the evidence supports "same-neighbor errors are often not confined to a one-pixel target boundary" under the stated scale. It does not validate 0.02/0.05 as semantic identity boundaries, and the neighbor-core percentage needs a fixed-pixel or neighbor-area sensitivity before being treated as scale-robust.

### W4. The split holds out interpolation cells, not images or receptive fields

**Status:** WARN, correctly disclosed

The checkerboard construction keeps a pixel only when all four bilinear stencil cells belong to one fold, and the sampled prototype-cell sets are asserted disjoint (`failure_locality_probe.py:94-103,115-116`). All 10,604 successful target-seed status rows in the full run record `no_shared_proto_cells=True`; no contrary row was found.

Both folds still use the same image, target, frozen prototype tensor, and overlapping network receptive fields (`failure_locality_probe.py:109,117-119`; `failure_locality_protocol.json:10`). The result report states this at `FAILURE_LOCALITY_RESULTS_20260911.md:44-46`.

**Impact:** describe this as same-image spatial cross-fitting or interpolation-cell holdout. It is not cross-image generalization.

### W5. Statistical and experimental scope remains exploratory

**Status:** WARN, correctly disclosed

Contrasts are paired because every arm shares the same target/fold/seed positions; folds and three sample seeds are averaged per target before treatment-minus-control differences (`failure_locality_probe.py:110-115,134-143`; `summarize_failure_locality.py:43-47,86-96`). The bootstrap resamples the preselected images as clusters and recomputes the target-weighted mean (`summarize_failure_locality.py:26-30,50-51`). Metric denominators are correct.

The 95% intervals are pointwise percentile intervals over 2,000 cluster draws. There is no permutation test and no multiplicity correction across the 924 stored contrasts. The protocol and output call them pointwise and treat category/size results as descriptive (`failure_locality_protocol.json:27`; `READOUT_ANALYSIS.json:2`). The frozen checkpoint is single; seeds 0/1/2 are pixel-sampling seeds, not independent model trainings (`failure_locality_protocol.json:11-15,24-29`). The 400 val images were reused after project exploration, so they are not pristine confirmation data.

## Checklist A-F

### A. Ground-truth provenance: PASS

- The selected train and val annotation files independently hash to `610fce...01d` and `e8c7f7...0b6f`, exactly matching the run protocol. They are the local COCO 2017 instance JSON files under `datasets/coco/annotations/`.
- Direct parsing of those JSONs found 2,281 non-crowd GT plus 41 crowd annotations on the selected train images and 2,913 non-crowd GT plus 30 crowd annotations on val. The 5,194 output `(image_id, annotation_id)` pairs exactly equal all selected non-crowd annotations.
- `COCO.annToMask` calls `annToRLE`; multipart polygons are converted and merged before decoding (`pycocotools/coco.py:417-445`). The audited subsets contain 222 multipart train instances and 300 multipart val instances, so this path is materially exercised.
- Original-resolution masks use `annToMask`; ordinary GT is unioned, crowd is accumulated separately, and all evaluated regions are intersected with `valid=~crowd` (`failure_locality_probe.py:56-75`). Own GT has priority, same-category neighbor is disjoint from own, and other-category pixels exclude both own and same-category union (`failure_locality_probe.py:71-85`).
- Input-grid GT uses the cached original/input shapes with nearest-exact resize and the same rounded letterbox offsets as the decoder path; padding and crowd are invalid (`crossimage_response_experiment.py:40-51`). Original-resolution and input-grid diagnostics remain separately labeled.
- Cached GT-to-prediction ownership was produced by category-aware official COCO bbox evaluation at IoU 0.5 with one-to-one prediction mapping (`frozen_mechanism_probe.py:93-109`). This is a conditional bbox-matched diagnostic, not official mask AP.

### B. Metrics, denominators, and score normalization: PASS

- Held-out sampled IoU is `TP/(GT-positive+FP)`; coverage, same-neighbor FP, background FP, and all FP are divided by the held-out own-GT-positive count (`failure_locality_probe.py:50-53`). Exact original-resolution IoU is equivalently `TP/(valid own area+FP)` (`failure_locality_probe.py:67-89`).
- AUC is a Mann-Whitney statistic with half credit for ties and is evaluated only after both classes pass the minimum count (`failure_locality_probe.py:26-29,110-113`).
- Threshold candidates and strict `score>tau` accounting are consistent; the maximized training IoU denominator is `training positives+training FP` (`failure_locality_probe.py:32-39`).
- No performance metric is divided by a prediction maximum, minimum, or mean. Train-score standard deviations are used only for the explicitly labeled signed-margin direction descriptor (`failure_locality_probe.py:144-153`), not to inflate task scores.
- The independent local verifier checked count partitions, integer reconstruction, threshold/AUC/distance helpers, target means, and all contrast point estimates with maximum identity error `1.42e-13` (`verify_failure_locality.py:19-90`; `LOCAL_VERIFICATION.json:2-10`).

### C. Files and numeric evidence: WARN

- `COMPLETE.json` records `COMPLETE`, `network_training=false`, `GT_oracle=true`, 160+400 images, 5,194 exact rows, 127,248 readout rows, and zero prior-spatial replay error (`COMPLETE.json:2-13`).
- All 12 artifact hashes in `COMPLETE.json` and `SUMMARY_COMPLETE.json` match current bytes. The full protocol hashes the exact main script, fixed protocol, S011 selection, both cache manifests, and both annotation JSONs; all match. Its 560 cache digests equal the source S011 manifests. The caches themselves were not downloaded into this audit workspace, so their byte hashes were enforced by the runtime assertion rather than independently reopened by the auditor (`failure_locality_probe.py:163-180`; `LOCAL_VERIFICATION.json:10`).
- The summarizer SHA in `SUMMARY_COMPLETE.json:3` matches the final source. The earlier smoke summary was regenerated after the size-group update; its final verifier is PASS. Smoke remains execution coverage only, not inferential evidence.
- Main narrative values in `FAILURE_LOCALITY_RESULTS_20260911.md:13-87` match the final JSON/CSV artifacts, including cohort counts, exact error rates, macro/pooled distance summaries, high-group readout table, paired intervals, train/held-out gaps, and fold stability.
- The tracker is `DONE / PARTIAL SUPPORT`, consistent with existing receipts (`EXPERIMENT_TRACKER.md:22`).
- WARN is retained because the machine field `partial_targets` omits zero-row exclusions and because imported runtime dependencies are not all directly bound in the S012 receipt. The source S011 protocol separately records the weight, decoder, experiment, and Ultralytics ops hashes (`diagnostics/crossimage_response_20260911/protocol.json:29-34`).

### D. Actual calls, dead code, and leakage: WARN

- `exact_regions` and `probe_target` are called for every selected image and every bbox-matched target; their outputs populate the four primary CSVs (`failure_locality_probe.py:172-193`). Output cardinalities and receipts demonstrate execution.
- `prepare` from `crossimage_response_decoder.py` is called and asserts parity with official `process_mask` for the original crop (`crossimage_response_decoder.py:7-30`). Its prediction-only `features`, `score`, and `decode` functions are part of S011, not S012; they are actually called by `crossimage_response_experiment.py:54-94,97-167,197-227` and are not phantom S012 metrics.
- No model output is used as GT. The only data-leakage concern is W1: held-out labels affect eligibility, not fit or threshold.

### E. Scope, selection, seeds, and oracle boundary: WARN

- Images are fixed S011 calibration160/evaluation400 lists; S011 selected them by exclusions plus a deterministic hash ordering without reading S012 outcomes (`crossimage_response_experiment.py:21-33`; `failure_locality_probe.py:163-170`).
- Results condition on original category-aware bbox50 matching, valid crop support, both sampled classes, all folds, and all sample seeds. Unmatched and skipped targets are retained in status/cohort artifacts.
- The protocol, receipts, report, and tracker consistently distinguish GT oracle, no network training, reused val, one checkpoint, and non-deployable diagnostic from AP or a learned method (`failure_locality_protocol.json:4-29`; `FAILURE_LOCALITY_RESULTS_20260911.md:71-87`; `EXPERIMENT_TRACKER.md:22`).
- The previous learned local BCE+Dice control is not erased: the report correctly notes that local did not improve over the strong global arm, consistent with `FULL_COCO_PILOT_COMPARISON_20260911.md:31-49` and `FULL_LOCAL_COMPARISON_AUDIT_20260911.md:25-41`.

### F. Evaluation type: PASS

**Classification:** `real_gt`, subtype `GT-informed within-image spatial-oracle diagnostic`.

The reference masks and labels are dataset-provided COCO GT. They are neither model-generated nor synthetic proxy GT. Because each oracle is fitted per target from one spatial GT fold, its result has a stricter claim ceiling than ordinary real-GT task evaluation: representation opportunity on the eligible cohort only.

## Result-to-claim impact

| Scoped statement in the supplied S012 report | Audit disposition |
|---|---|
| Original matched masks contain same-category-neighbor pixels beyond a thin target boundary | **Supported with qualification.** Use instance-macro and medium/large sensitivity; retain pixel-floor and neighbor-core caveats. |
| A per-target prototype oracle outperforms a per-target threshold on held-out interpolation cells | **Supported on the GT-conditioned complete cohort.** Val high group: +2.965 pp sampled IoU, pointwise image-cluster CI [1.685, 4.279]. |
| A higher-capacity local oracle adds +0.932 pp over the global prototype oracle in val high targets | **Supported as an unequal-capacity opportunity probe.** It is not evidence that the existing local head learns the gain. |
| The oracle establishes a coefficient-similarity cause, a head bottleneck, cross-image generalization, AP gain, or a deployable method | **Unsupported and not claimed by the supplied report.** |
| Prior local BCE+Dice learned control remains negative relative to its strong global control | **Consistent with the cited prior audited artifacts.** S012 does not override it. |

## Required reporting constraints

1. Use matched and complete cohort counts, not `partial_targets`, for exclusions.
2. Call the readout cohort `GT-conditioned all-seed eligible` and the split `same-image spatial cross-fit`.
3. Keep all oracle values out of official task/AP tables and out of deployable-method comparisons.
4. Describe 95% intervals as pointwise exploratory image-cluster bootstrap intervals; do not imply multiplicity-adjusted significance or an interaction test.
5. Keep the normalized-distance thresholds descriptive. For stronger neighbor-interior language, add sensitivity by valid target area, valid neighbor area, and fixed raw-pixel bands.
6. A subsequent method claim requires prediction-time features only, cross-image fitting/calibration, untouched-image evaluation, independent training seeds, and comparison with the existing BCE+Dice global/local controls.

## Audit limits

This audit read source, final outputs, smoke outputs, raw COCO annotation JSONs, result narrative, tracker, and prior local-control reports. It did not rerun GPU inference, independently reconstruct prototype features from the 560 remote cache files, or review an unprovided paper manuscript/new claim. The local verifier explicitly records the missing independent re-inference witness. Per the user's write restriction, only this Markdown report and its JSON companion were created; the skill's normal `.aris/traces/` side artifact was not written.
