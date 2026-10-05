# Experiment Plan

## 2026-09-07 User-Authorized All-Split External Inference

The user confirmed that the active checkpoint was fine-tuned only on PigLife and authorized all FaroPigSeg train/val/test and BamaPig2D train/eval images for fixed-model inference. Archived training provenance agrees. R012 expands external evaluation to 4,858 images / 27,745 GT, retaining split-specific and pooled results under the frozen .05/Top300/1024 protocol. See `experiments/yolo26_external_full_20260907_protocol.md`. Original target-dataset split names do not imply checkpoint training exposure; prior exploratory test/eval inspection remains disclosed. This execution does not select a new method or pass its Gate.

R012 is completed and deterministically verified: Faro all-split AP .398061 / failure rate 46.83%; Bama .641492 / 23.34%. All 492 historical image prediction lists are unchanged. Bama's 77 byte-identical pairs include annotation variation and 11 cross-split pairs; keep group/label policy explicit in future validation. Full results: `research-wiki/yolo26_external_full_20260907.md`. The existing raw-trace scope remains 586 PigLife/Faro images.

**Problem**: Explain and improve high YOLO26-seg failure rates in dense, touching pig scenes.
**Method Thesis**: Freeze YOLO26, localize the dominant failure stage with provenance-complete candidate traces, then test one stage-matched intervention.
**Date**: 2026-09-05

## Claim Map

Execution clarification (2026-09-05): R006 GT-guided addition/removal is a specified output-set oracle experiment, not a causal decomposition or a proven optimal recovery bound. R005 per-GT IoU >= .50 availability is distinct from simultaneous matching and class-C recovery. The numeric 40%/30% route Gate cannot be satisfied by relabeling these measurements as ranking/competition. Full trace replay and R005b strict-quality/all-GT matching passed their integrity checks; method selection remains gated pending stage-specific intervention evidence.

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Failures are stage-heterogeneous and reproducible | Establishes the real bottleneck | New manifest, taxonomy, bootstrap intervals, replication on two datasets | B1, B2 |
| C2: The intervention improves the diagnosed stage without hiding regressions | Supports the paper contribution | Baseline/control/oracle/intervention plus deletion and relation metrics | B3, B4 |

## Paper Storyline

- Main paper must prove: a reproducible YOLO26 failure mechanism and a minimal aligned improvement.
- Appendix can support: threshold sensitivity, qualitative cases, latency details, and secondary dataset slices.
- Experiments intentionally cut: cross-architecture training, LLM/VLM modules, broad backbone sweeps, and repeated cache-only analyses.

## Experiment Blocks

### Block 1: Provenance and protocol sanity
- Claim tested: evaluation and candidate tracing are reproducible.
- Dataset / split: all eligible PigLife test images; FaroPigSeg and BamaPig2D where labels are compatible.
- Compared systems: exact YOLO26 checkpoint/runtime; no intervention.
- Metrics: image/GT counts, cache integrity, candidate count, Top-K parity, mask AP sanity.
- Setup: freeze resize, `imgsz`, `rect`, confidence, Top-K, matching thresholds, and a new sample manifest.
- Success criterion: manifest and raw traces are reproducible; target at least 100 dense/touching images per dataset where available.
- Failure interpretation: stop and repair provenance or metric code.
- Priority: MUST-RUN

### Block 2: Stage diagnosis and counterfactuals
- Claim tested: dense failures are attributable to formation, ranking/competition, or mask quality.
- Dataset / split: manifest from B1.
- Compared systems: baseline, Mask-NMS threshold sweep, candidate oracle, candidate removal/addition perturbations.
- Metrics: O/M/X/MISS, candidate recall, oracle recovery, ranking-inversion rate, complete/local coexistence, union upper bound, bootstrap intervals.
- Setup: pre-register complete (`IoU>=0.50`, coverage `>=0.70`), local (`0.10<=IoU<0.50`, coverage `>=0.10`, inside-GT `>=0.60`), duplicate (`IoU>=0.80`).
- Success criterion: route gate is met only if oracle recovery is `>=40%`, ranking/competition accounts for `>=30%` of recovered cases, and direction replicates on two datasets.
- Failure interpretation: do not train a scorer; report the dominant alternative stage and design only its matched test.
- Priority: MUST-RUN

### Block 3: Conditional minimum intervention
- Claim tested: one stage-matched intervention reduces the diagnosed failure.
- Dataset / split: primary dataset plus one replication dataset; three seeds if training is required.
- Compared systems: baseline, Mask-NMS control, selected intervention, intervention-deletion control.
- Metrics: mask AP/AP50/AP75, dense failure, O/M/X/SPLIT_TYPE correction/regression, candidate recall, latency, ranking-inversion frequency.
- Setup: if Route A passes, two-layer MLP candidate scorer with pairwise margin ranking and calibration; YOLO26 backbone/neck/mask prototypes frozen. Otherwise use the single formation- or mask-side branch indicated by B2.
- Success criterion: improvement in the diagnosed slice, non-inferior overall AP, and no unexplained increase in unrelated failure classes.
- Failure interpretation: mechanism is not actionable under this intervention; retain negative result and stop expansion.
- Priority: MUST-RUN only after B2 gate

### Block 4: Robustness and presentation
- Claim tested: result is not a threshold or sample artifact.
- Dataset / split: held-out or second dataset; fixed protocol.
- Compared systems: final method and strongest simple control.
- Metrics: threshold sensitivity, image-cluster bootstrap, latency/memory, qualitative failure transitions.
- Priority: NICE-TO-HAVE after B3

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Freeze manifest and evaluator | R001 | reproducible counts and hashes | 2-4 h | label/protocol mismatch |
| M1 | Reproduce YOLO26 baseline | R002-R004 | baseline valid on each dataset | 0.5-1 GPU-day | runtime drift |
| M2 | Identify mechanism | R005-R007 | 40%/30% gate and two-dataset direction | 4-8 h cache analysis | oracle overinterpretation |
| M3 | Test minimum intervention | R008-R010 | aligned gain or stop | 1-3 GPU-days | scorer does not transfer |
| M4 | Robustness/figures | R011-R013 | stable reportable result | 4-8 h | secondary slice noise |

## Compute and Data Budget

### Immediate Diagnostic Continuation (2026-09-05)

The new replay and per-GT availability evidence are recorded in `research-wiki/yolo26_stage_availability_20260905.md`. Continue the current idea-discovery run; these diagnostic artifacts do not close method acceptance.

| Order | Question | Minimum Experiment | Required Check | Decision |
|---|---|---|---|---|
| 1 | Does the matching repair change R006 recovery? | New v2 output directory, final-source XOR guard, maximum-cardinality assignment, all-GT reclassification, dataset-wise recovery/regression intervals | Every 6,226 GT key and transition conserved; compare v1/v2 at per-GT and selected-source levels | Preserve both versions and explain any change; never label achieved recovery a proven optimum |
| 2 | Does a candidate meet the fixed single-mask quality criterion? | Add the fixed taxonomy's coverage >= .75 AND purity >= .75 criterion alongside IoU >= .50/.75, at raw/Top-K/conf/final | Lossless coverage-only inclusion screen; native full-resolution decoding; verify final-RLE decisions | Distinguish loose overlap availability from mask quality; do not equate absent candidates with a uniquely identified formation cause |
| 3 | Can neighboring instances be represented simultaneously? | Exact matching on each GT component, including originally C neighbors, using explicitly mask-defined admissible edges | State whether an edge is only pairwise admissibility or also excludes other-GT contamination; report matching conflicts separately | Separate per-GT existence, simultaneous allocation, and final output-set failure |
| 4 | Which single retention change has an observable effect? | Freeze the lost-at-Top-K/lost-at-confidence/final-available strata; separate one-gate retention, deletion control, and joint GT-informed replacement | Full GT regressions and COCO AP; no threshold optimization on this diagnostic public-test set | Only a reproducible stage-specific effect can justify comparing a deployable intervention with nearest literature |

Primary coordinator owns protocol, evidence review and method Gate. Terra workers own bounded implementation, metric audits and literature verification. Reuse current code and the existing Conda `pytorch` runtime; assign separate output directories. Model, scorer and loss design remains pending the evidence and the user's major-direction decision.

Execution update: R006 v2 and its integrity/statistics audit are complete. R005b strict-quality extraction and all-GT matching are complete in `experiments/yolo26_strict_quality_fulltrace_20260905_v1` and `experiments/yolo26_strict_matching_fulltrace_20260905_v1`; independent CPU integrity audit is PASS in `research-wiki/yolo26_strict_full_integrity_20260905.md`. Strict coverage/purity raw availability is PigLife 360/394 and Faro 490/797; final availability/matched counts are 312/310 and 224/224. The 20 assignment deficits are summed across quality/stage graphs (8+8 at IoU .50 and 2+2 at coverage/purity .75, confidence/final), not 20 distinct affected GT. The CPU audit `experiments/yolo26_gate_binding_20260905_v1` proves that every raw score > .05 source is already in Top-300 across all 586 images. Therefore skip a redundant Top-K-only mask replay at fixed conf=.05: its post-confidence source set is identical. First-loss-at-Top-K remains a sequential availability label, not the isolated effect of changing Top-K. The next retention diagnostic must isolate confidence while holding Top-K and scores fixed, with no public-test threshold optimization. See `research-wiki/yolo26_strict_quality_20260905.md` for the result table and scope limits.

Literature anchors: Mask Scoring R-CNN (1903.00241, Section 3.2) is established quality-calibration prior art; its described inference recalibrates retained masks. TIDE (2008.08115, Sections 2.2-2.3) motivates separate baseline-relative interventions and explicit AP denominators. Use their verified Wiki pages when interpreting the next results.

### R005c: Fixed Confidence Retention and Matched GT-Guided Controls

Protocol frozen before the new output run. This continues B2 diagnosis; no trained method is selected.

- Use the same 426 PigLife / 160 Faro traces, cached Top-300 order, original scores, batch-2 native decoder, and fixed classifier. Baseline is score > .05; the only confidence comparison is score > .01, reused from the existing R007 experiment. No threshold or Top-K sweep.
- Freeze the target pool from R005b: baseline non-C GT, strict coverage/purity .75 candidate available at Top-K but absent at confidence .05. This pool is PigLife 32 and Faro 146 GT. Within this pool, admissible added candidates must be in cached Top-300, have .01 < score <= .05, meet coverage >= .75 AND purity >= .75, and be absent from final sources. Pre-run table accounting gives 7 edges / 5 GT for PigLife and 45 edges / 38 GT for Faro; maximum-cardinality source-unique assignment uses the existing deterministic R006 order.
- Compare BASELINE, CONFIDENCE_001, GT_ADDITION, GT_REMOVAL, and GT_ADDITION_REMOVAL, each constructed independently from BASELINE. The three GT-guided controls share exactly one selected-source map and target set. Removal uses only the selected target's baseline core predictions, as defined by the fixed classifier. Record mapping and target-key hashes for all three controls.
- Reclassify all 6,226 GT for every condition. Report per-dataset 8x8 transitions, failed-to-C, C-to-failure and C-to-O/M/X. Export real condition predictions as COCO segmentation RLEs with unchanged scores; evaluate AP/AP50/AP75 per dataset using standard maxDets=100, distinct from the model's Top-300 gate.
- Stop on changed provenance hashes, manifest/source ID mismatch, baseline prediction or source-mapping mismatch, nonzero final-source XOR, candidate-quality drift, reused sources, changed target/mapping hashes, or incomplete GT accounting. Keep new output directories empty before launch; preserve historical scripts and results.
- Existing R007 reports PigLife 2 recoveries / 136 correct-GT regressions and Faro 10 / 133 at .01. These historical counts need verification against the current full traces before being joined with newly exported AP. GT-guided results remain achieved intervention outcomes, not a deployable selector, optimal bound, or attribution of independent internal causes.

PigLife public-test and Faro test have been used for exploratory diagnosis. Any later tuned method must disclose that use and choose thresholds or learn parameters on permitted train/validation data. The forbidden private/locked/guard/target data remain outside this plan.

- Total estimated GPU-hours: 24-72 for the first intervention cycle; diagnostics reuse existing caches where valid.
- Data preparation: new manifest, annotation-semantics audit, and raw-candidate retention.
- Human evaluation: targeted audit of ambiguous complete/local cases only; no broad relabeling before B2.
- Biggest bottleneck: provenance-complete candidate traces under one fixed YOLO26 runtime.

## Risks and Mitigations

- Missing archived manifest: generate a new manifest and keep reconstructed replay audit-only.
- Candidate oracle uses GT: report achieved recovery under its specified policy; do not claim an optimal bound or deployable performance.
- Dataset semantics differ: record modal/visible/amodal status before pooling results.
- Training gain is just more capacity: freeze backbone and include scorer-deletion and parameter-matched controls.

## Final Checklist

### 2026-09-07 continuation

The user selected Faro single-mask quality as the next diagnostic priority: box localization, mask expression and annotation semantics. R005c v2 computations are complete and statistics are in `experiments/yolo26_confidence_statistics_20260907_v1`; fresh integrity review is being completed. Method selection remains gated.

R005d implements the first bounded test under that priority. Its preregistered protocol is `experiments/faro_mask_quality_20260907_protocol.md`: all 160 images and 1,752 GT, exact box support at native pixel coordinates, original YOLO-to-COCO raster parity, and a same-stage join with frozen strict mask availability. Results are in `experiments/faro_box_support_20260907_v1`. The next scientific distinction is box support insufficiency versus support-feasible mask-quality absence, with official annotation policy checked separately. Support feasibility is not proof of mask-head capacity or a selected intervention.

- [ ] Main claims covered by B1-B3
- [ ] Novelty isolated by stage oracle and deletion control
- [ ] Simplicity defended by one-component budget
- [x] Frontier contribution explicitly not claimed
- [x] Nice-to-have runs separated from must-run
