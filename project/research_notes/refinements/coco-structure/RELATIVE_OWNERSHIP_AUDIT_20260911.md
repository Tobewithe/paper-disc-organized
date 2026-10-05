# Relative Ownership Experiment Audit

**Date:** 2026-09-11  
**Auditor:** fresh same-family Codex reviewer; read-only source review  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Overall verdict:** **WARN (no fatal integrity defect)**  
**Evaluation type:** **real_gt**

## Bottom-line judgment

The implementation and the reported negative decision are internally consistent. COCO annotations provide the measurement ground truth, GT does not enter the decoder, metrics are not self-normalized, the development and evaluation phases are separated by a hash-bound settings lock, and completed receipts/results exist. The predeclared development rule selected `alpha=0` for relative, geometry, and threshold arms. Consequently, the 600-image evaluation is an exact zero-intervention consistency check; it is not an independent test of nonzero decoder efficacy and gives no effective nonzero random-shuffle comparison.

No fatal A-D integrity failure was found. The WARN verdict concerns evidence scope and causal wording: the only nonzero effects are development-set, fixed-bbox50, partially eligible diagnostics; exact cardinality holds on the 640-input grid, not necessarily after image-space scaling; and the pointwise bootstrap intervals are descriptive after selection.

## A. Ground Truth Provenance — PASS

- The decoder contract accepts only `proto`, coefficients, predicted boxes, detections, and input shape (`relative_ownership_decoder.py:25-36`). The caller explicitly constructs this filtered input and calls `decode` before passing masks to the GT measurement function (`relative_ownership_experiment.py:207-211`). Neither annotation IDs nor GT ownership mappings can affect neighbor selection or pixel reranking.
- COCO annotations are loaded from `instances_train2017.json` or `instances_val2017.json` (`relative_ownership_experiment.py:182-185`). Dataset masks are rasterized only inside `spatial_and_predictions` to measure own/neighbor/background regions (`relative_ownership_experiment.py:65-89`).
- Fixed attribution is genuine bbox IoU=0.50 COCO matching: `ownership` invokes `COCOeval(..., 'bbox')`, fixes `iouThrs=[.5]`, and records `gtMatches` (`frozen_mechanism_probe.py:93-109`). The fixed mapping is cached, not recomputed from modified masks.
- Official task results use `COCOeval(..., 'segm')` over all selected image IDs (`summarize_relative_ownership.py:17-31`). This is dataset GT, not a model-derived proxy.
- The v1.1 development sampling amendment is documented as occurring after an unattainable quota failed and before decoder output (`relative_ownership_protocol.json:10`). It changes sampling scope, not GT provenance.

## B. Score Normalization — PASS

- The per-prediction logit RMS is an intervention scale only (`relative_ownership_decoder.py:53-54`, `relative_ownership_decoder.py:78-82`). It changes the ranking penalty; it is never used as an evaluation denominator or presented as an accuracy score.
- Spatial measurements divide pixel counts by dataset GT area, or compute IoU from TP/neighbor/background counts (`relative_ownership_experiment.py:107-117`). Task AP/R75 comes from COCOeval (`summarize_relative_ownership.py:17-31`). No metric is divided by a maximum, mean, or RMS of the model's own output.
- `correction_rms` is recorded as intervention amplitude (`relative_ownership_decoder.py:114-118`), not as performance. No fabricated near-one score was found.

## C. Result File Existence and Provenance — PASS

- Development and evaluation completion receipts report `COMPLETE` for 160 and 600 images (`DEVELOPMENT_COMPLETE.json:2-4`, `EVALUATION_COMPLETE.json:2-4`); the final task receipt reports `COMPLETE` for 600 images and binds the summarizer source hash (`TASK_COMPLETE.json:2-15`).
- Before summarization, active code requires the evaluation receipt and verifies every receipt hash (`summarize_relative_ownership.py:49-53`). The final task receipt hashes the generated summaries, recovery tables, prediction hashes, and paired analysis (`summarize_relative_ownership.py:136-137`). The separate local deterministic verification is `PASS`: it checked 28 receipt entries, 12,246 area records, seven identical locked task rows, and the transition identity (maximum numerical error `3.0965e-16`; `experiments/coco_clean_20260911/diagnostics/relative_ownership_20260911/LOCAL_VERIFICATION.json:2-8`).
- The locked task table contains one unique result row across all seven arm labels because every locked value is zero: Mask AP `0.4562982143`, AP75 `0.4863927036`, 49,073 predictions, high-ICI R75 `0.5076045627`, and high-pair R75 `0.4069981584` (`task_summary.csv:2-8`). Semantic prediction identity is explicitly stored for every non-initial arm (`PAIRED_ANALYSIS.json:5222`).
- The remote full artifact directory additionally contained 600 per-image prediction files for each of the seven arms at audit time. The retained 6-development/8-evaluation smoke run is pipeline evidence only, not main-result evidence.

## D. Active Code / Dead Code — PASS WITH LIMITATION

- `decode` is imported and called in the active experiment pipeline (`relative_ownership_experiment.py:21-22`, `relative_ownership_experiment.py:207-210`). Development constructs and executes every nonzero relative and geometry grid arm (`relative_ownership_experiment.py:159-164`); correction records show 2,041 eligible records per nonzero arm and zero input-grid area error.
- Development selection actively applies the coverage guard, ranks high-ICI mask IoU, writes `LOCKED_SETTINGS.json`, and hashes the selected data and sources (`relative_ownership_experiment.py:234-250`). Evaluation reloads the lock and asserts script, decoder, and protocol hashes before using its arms (`relative_ownership_experiment.py:169-180`).
- Official task evaluation and paired analyses are active (`summarize_relative_ownership.py:58-77`, `summarize_relative_ownership.py:79-137`). The post-hoc transfer script only replays the existing development arms and labels its output as descriptive with no validation modification (`diagnose_relative_transfers.py:1-3`, `diagnose_relative_transfers.py:18-28`, `diagnose_relative_transfers.py:58-66`).
- Limitation: the main evaluation never exercises a nonzero random-shuffle intervention, because the selected relative alpha is zero. Thus the three random labels are not three training seeds or effective nonzero controls. This is correctly disclosed in the protocol (`relative_ownership_protocol.json:17`, `relative_ownership_protocol.json:23`) and results narrative.

## E. Scope and Statistical Validity — WARN

- Scope is one frozen pretrained model, 160 hash-selected development images from a heavily high-ICI-enriched mounted train pool, and 600 previously explored val images (`relative_ownership_protocol.json:6-11`). There is no training and no independent model-seed replication. This supports the predeclared stop decision for this concrete decoder, not broad claims about all relational decoding, neighbor information, or CCL.
- Main task denominators are 4,042 non-crowd GT instances and 771 same-class bbox-adjacent pairs. Fixed-attribution spatial denominators are 3,618 matched/valid targets, leaving 424 task GT unmatched or without valid pixels (`PAIRED_ANALYSIS.json:5218-5221`). The task endpoints retain all GT; only pixel attribution is conditional on original bbox50 matching.
- Intervention eligibility is sparse relative to all detections: development has 24,370 detections and 2,041 eligible correction records per nonzero arm (8.38%). Within fixed-attribution targets, relative arms are eligible for 999/1,855 overall and 331/426 high-ICI targets; they actually change 1,627 predictions at alpha 0.25. All-target means therefore estimate the predeclared policy including unchanged ineligible targets, not an effect conditional only on eligible predictions.
- Candidate identity, class, and score are asserted unchanged for exact-area arms (`summarize_relative_ownership.py:63-65`), and decoder code never mutates boxes or scores. Exact area is checked per modified prediction on the input grid (`relative_ownership_decoder.py:112-122`). Image-space masks are subsequently geometrically scaled and thresholded (`relative_ownership_experiment.py:95-105`), so exact original-image area or actual own-GT coverage is not guaranteed.
- Geometry uses the same input-grid pixel coordinates and predicted input-grid boxes for normalized center-distance penalties (`relative_ownership_decoder.py:83-89`); no GT geometry enters it. This is a valid control for this implementation, not proof that all geometry controls are exhausted.
- The 2,000 draws resample images as clusters (`summarize_relative_ownership.py:82`, `summarize_relative_ownership.py:100-115`). This respects within-image dependence, but intervals are pointwise with no multiplicity correction, development intervals reuse the hyperparameter-selection data, and alpha-zero evaluation intervals are tautologically `[0,0]`. Category-by-area standardization is a point estimate only (`summarize_relative_ownership.py:116-125`).
- Development alpha 0.25 provides a real nonzero tradeoff: among 426 high-ICI fixed-attribution targets, same-neighbor error changes by `-0.1340` pp (pointwise CI `[-0.2342,-0.0478]`), background error by `+0.1384` pp (`[+0.0717,+0.2092]`), coverage by `-0.0428` pp, and mask IoU by `-0.0290` pp (`[-0.1052,+0.0402]`). These are descriptive development results and cannot be called held-out efficacy.

## F. Evaluation Type — real_gt

- **Task AP/R75 and pair recovery:** `real_gt`, measured against COCO dataset annotations with official COCO segmentation evaluation.
- **Fixed-attribution pixel metrics:** `real_gt`, conditional on an original-prediction bbox50 match; GT is used only after decoding to define own, same-class-neighbor, other-instance, overlap, crowd-excluded, and background regions.
- **Random control:** prediction-derived spatial-shuffle control by design, but its outcome is still measured against real GT. In the locked main evaluation it is inactive because alpha=0.
- **Development hyperparameter selection:** supervised calibration on train GT, properly separated from the locked val run; it is not training of model parameters.

## Claim Impact

- **Supported:** The predeclared development gate selected zero, so this bounded relative-response, predicted-neighbor, exact-input-area reranker did not earn promotion to a nonzero held-out test or formal training.
- **Supported with qualifier:** On development data, nonzero relative reranking reduces fixed-attribution same-class-neighbor occupancy while shifting pixels toward background and failing to improve mean mask IoU.
- **Supported:** The 600-image locked run reproduces the baseline exactly for all zero-valued arms and verifies score/candidate consistency and the no-retuning execution path.
- **Unsupported:** Nonzero relative reranking independently fails, ties, or beats baseline/geometry/random on the 600-image evaluation; the selected alpha is zero.
- **Unsupported:** Three-seed robustness, because seeds 0/1/2 are spatial permutations and are inactive in the main run.
- **Unsupported:** Exact original-image area/coverage preservation, general ineffectiveness of neighbor information, rejection of CCL, or equivalence to the earlier full relative-response AUC probe.

## Action Items

- Keep the results report's present zero-selection wording and the stated stop rule. Do not turn the all-zero val table or `[0,0]` contrasts into evidence about nonzero efficacy.
- State both spatial and task denominators whenever quoting pixel metrics: 3,618 matched/valid of 4,042 total GT, with 424 absent from the spatial diagnostic; give 771 as the pair denominator.
- When discussing area preservation, always say “exact input-grid foreground cardinality”; retain the observed image-space area/coverage differences.
- Treat development bootstrap intervals and post-hoc pixel transfers as descriptive, and retain the no-multiplicity and no-independent-seed limitations.
- Treat the narrative's 41.4/165.3-second runtimes as approximate; completion receipts record about 41.6 and 165.6 seconds. This is editorial and does not affect results.

## Unperformed Checks / Remaining Limits

- No independent cross-family reviewer was used; this verdict remains provisional.
- No nonzero arm was evaluated on the 600-image locked val sample, and no effective nonzero shuffle control or model-seed replication exists.
- No pristine test set or second dataset/model was evaluated.
- This bounded audit did not independently reproduce GPU inference or rerun COCOeval; it verified active code, receipts, hashes, tables, and deterministic local checks.
