# Paper Plan

**Working title**: Mechanism-Driven Diagnosis and Improvement of Instance Segmentation Failures in Dense Pig Scenes  
**中文题目**：密集猪群实例分割失败机制分析与证据驱动改进  
**Paper type**: Empirical diagnostic study with a mechanism-dependent intervention  
**Target**: SCI Q2 computer-vision / agricultural-vision venue (to be selected after evidence and venue-fit review)  
**Status**: Research plan; no method or main contribution is locked  
**Date**: 2026-09-05

## 1. Story and Research Questions

The paper first determines why instance segmentation fails in dense, touching pig scenes, then tests the smallest intervention supported by the diagnosis. The central distinction is between ordinary high-overlap duplicates and low-overlap, complementary or competing instance hypotheses.

**RQ1**: Which failure mechanisms dominate across density, contact, and occlusion conditions?  
**RQ2**: Are residual errors caused by mask construction, candidate formation, candidate scoring/competition, or a mixture?  
**RQ3**: Which mechanism family gives a measurable and reproducible improvement without increasing unrelated errors?  
**RQ4**: Does the finding transfer across PigLife, FaroPigSeg, and BamaPig2D under the same YOLO26-seg protocol? Cross-architecture replication is deferred unless a second runnable baseline becomes available without delaying the YOLO study.

## 2. Claims-Evidence Matrix

| Claim | Existing evidence | Status | Required confirmation | Paper location |
|---|---|---|---|---|
| Dense scenes contain multiple distinct failure modes, not one generic error | 7,374 GT; failure taxonomy; scene-conditioned analysis | Partial | Re-run taxonomy on all target datasets with fixed protocol | §3, §5.1 |
| High-overlap duplicate predictions are largely removable by Mask-NMS | 427/675 O cases changed to C at Mask-NMS@0.80 | Supported for current cache | Report threshold sweep and held-out validation | §3.4, §5.2 |
| Residual O errors include low-overlap complementary fragments | 196 residual O; SPLIT_TYPE=50; union IoU/coverage gains | Partial, descriptive | Independent annotation/audit and cross-dataset replication | §3.5-3.6, §5.3 |
| Fragmentation involves candidate formation and competition, not only mask pixels | 36/50 box-level or box-mask mixed; complete/local candidate analysis | Partial, no causal proof | Counterfactual candidate oracle and source ablations | §3.8-3.11, §5.4 |
| A mechanism-targeted intervention improves dense-scene segmentation and relation metrics | No intervention results yet | Open | Method selection after RQ1-RQ3; main and ablation experiments | §4, §5.5-5.8 |
| The mechanism is not a YOLO-only artifact | No second architecture result; MaskDINO loading mismatch remains | Deferred / scoped limitation | Only add a second architecture if runtime parity is available; otherwise state YOLO26 scope explicitly | §5.7, §6 |

## 3. Paper Structure

### §1 Introduction

- Motivate dense pig instance segmentation for monitoring and behavior analysis.
- State why touching, similar appearance, partial visibility, and candidate ambiguity make failure analysis necessary.
- Contrast high-overlap duplicates with low-overlap complementary fragments.
- Present the diagnosis-to-improvement workflow, not a preselected module.
- Contributions must remain conditional until §5 confirms them:
  1. A reproducible failure taxonomy and candidate-level tracing protocol.
  2. Evidence separating ordinary duplicate suppression from residual fragmentation/competition.
  3. A mechanism-targeted intervention selected by the diagnosis and tested with relation-aware metrics.
  4. Cross-dataset evidence under a fixed YOLO26-seg protocol; cross-architecture generalization is explicitly scoped as future work unless parity is available.

**Hero figure**: a four-panel visual showing (a) dense pig scene, (b) high-overlap duplicate corrected by Mask-NMS, (c) one GT explained by complementary local candidates, and (d) the diagnostic-to-intervention pipeline. Caption must state that panels (b)-(c) are different error mechanisms.

### §2 Related Work

Organize by mechanisms, not by paper chronology:

1. General instance segmentation and query/candidate prediction: Mask R-CNN, YOLO-seg, Mask2Former/MaskDINO (baseline context; verify exact editions before citation).
2. Dense and occluded instance segmentation: BCNet, ORM, OCFusion, Relook, Layering, OVIS.
3. Candidate duplication, assignment, and global consistency: SCE duplicate removal, relation/distillation methods, recurrent instance segmentation.
4. Pig and animal instance segmentation: Panoptic Instance Segmentation on Pigs, CClusnet-Inseg, livestock monitoring, cattle/biological-image segmentation.
5. Positioning: existing work addresses occlusion, layering, clustering, or duplicate removal, but does not directly establish the project's complete/local low-IoU candidate competition mechanism.

### §3 Failure Analysis and Problem Formulation

#### §3.1 Data, annotation semantics, and baseline

- Datasets: PigLife, FaroPigSeg, BamaPig2D; record split, image counts, and modal/visible/amodal semantics.
- Baseline: YOLO26-seg fixed prediction and raw-candidate caches; document checkpoint and inference versions.
- Define GT-level labels C/I/L/S/O/M/X/MISS and scene strata Isolated/Near-Dense/Touching.

#### §3.2 Global failure taxonomy

- Report 7,374 GT and baseline failure rate 19.84% as current-cache evidence.
- Add confidence intervals or bootstrap intervals when the protocol is rerun.

#### §3.3 Scene-conditioned failures

- Compare density, contact count, nearest-neighbor distance, area, and failure type.
- Treat associations as descriptive, not causal.

#### §3.4 Duplicate suppression control

- Mask-NMS threshold sweep, with @0.80 as the existing reference.
- Compare ordinary Mask-NMS with source-aware suppression; retain the negative result if reproduced.

#### §3.5 Residual error mechanisms

- Analyze residual O subclasses: LOCAL_REPEAT, SPLIT_TYPE, MULTI_PRED_COMPLEX, OTHER_UNRESOLVED.
- Report prediction-pair IoU, candidate count, and union-vs-best-single coverage.

#### §3.6 Fragmentation and candidate tracing

- For SPLIT_TYPE, report union IoU gain (+0.084 median), coverage gain (+0.143 median), and complementary-region statistics.
- Trace box-mask split source, grid-box correspondence, raw candidate existence, rank, and score.
- Separate: no complete candidate, complete candidate dropped, and complete/local candidates jointly surviving.

#### §3.7 Mechanism hypotheses and falsification tests

Define competing explanations:

- H1: ordinary duplicate suppression is sufficient.
- H2: mask construction is the primary bottleneck.
- H3: candidate formation is insufficient.
- H4: candidate scoring/competition is insufficient.
- H5: multiple mechanisms coexist.

Each hypothesis must map to an oracle, perturbation, or source ablation before an intervention is selected.

### §4 Mechanism-Targeted Intervention (to be filled after §3 Gate)

This section is intentionally parameterized until the diagnosis Gate is passed.

Required subsections:

- §4.1 Design requirements derived from the confirmed mechanism.
- §4.2 Candidate/instance representation and supervision.
- §4.3 Relation, consistency, or formation module (only if supported by §3).
- §4.4 Training objective and inference path.
- §4.5 Complexity and compatibility with the baseline.
- §4.6 Failure cases where the intervention is not expected to help.

Allowed candidate families for comparison, not commitments: candidate-set relation scoring, global candidate selection, explained-region memory, center/offset clustering, visible/full-region modeling, and failure-slice augmentation.

### §5 Experiments

#### §5.1 Protocol and datasets

- Freeze train/validation/test splits, random seeds, image resizing, confidence thresholds, and matching rules.
- Validate annotation semantics before interpreting outside-GT pixels.

#### §5.2 Baseline and post-processing controls

- Baseline; Box-NMS/Mask-NMS threshold sweep; source-aware suppression.
- Report box AP, mask AP, AP50, AP75, latency, and relation metrics.

#### §5.3 Mechanism replication

- Reproduce taxonomy, residual O analysis, SPLIT_TYPE union analysis, and complete/local candidate statistics on each dataset.
- Report bootstrap uncertainty and per-scene stratification.

#### §5.4 Oracle and counterfactual tests

- Candidate oracle upper bound: best valid candidate/subset under GT matching.
- Candidate removal/addition perturbations to isolate formation versus scoring.
- Score calibration and rank perturbation tests.
- Keep all oracle results labeled as diagnostic upper bounds, never as deployable performance.

#### §5.5 Main intervention comparison

| Method | Box AP | Mask AP | AP75 | Dense failure | O/M/X | SPLIT_TYPE | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Best control | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Proposed, diagnosis-selected | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

#### §5.6 Failure-specific evaluation

- Measure correction and regression for O, M, X, MISS, LOCAL_REPEAT, and SPLIT_TYPE.
- Require overall AP improvement or a clearly stated tradeoff; a slice-only gain is not sufficient for the main claim.

#### §5.7 Ablation and mechanism alignment

- Remove one relation/consistency/formation component at a time.
- Include score-only, NMS-only, and oracle-inspired controls.
- Check whether the component changes the diagnosed mechanism rather than only changing confidence thresholds.

#### §5.8 Generalization and efficiency

- Cross-dataset: PigLife, FaroPigSeg, BamaPig2D with the same YOLO26-seg protocol.
- Cross-architecture: deferred; report architecture scope explicitly unless a second runnable baseline is already available with parity.
- Report parameters, FLOPs, FPS, memory, and training cost.

#### §5.9 Qualitative evidence

- Show duplicate, fragmentation, candidate-missing, candidate-dropped, and jointly-surviving cases.
- Include negative examples where adjacent pigs remain separated.

### §6 Discussion and Limitations

- Explain why Mask-NMS addresses high-overlap duplicates but not necessarily low-IoU complementary hypotheses.
- Distinguish candidate formation, ranking/competition, and mask construction.
- Discuss annotation semantics, limited SPLIT_TYPE sample size, domain specificity, and architecture dependence.
- State which conclusions are descriptive, causal, or experimentally supported.

### §7 Conclusion

- Restate only claims that pass the Claims-Evidence Gate.
- Summarize the confirmed failure mechanism(s), intervention effect, and remaining limitations.

## 4. Figure and Table Plan

| ID | Type | Purpose | Evidence source | Status |
|---|---|---|---|---|
| Fig. 1 | Pipeline/qualitative | Diagnosis-to-improvement story | manual + final examples | planned |
| Fig. 2 | Distribution plots | Failure taxonomy by scene condition | baseline cache | existing evidence, regenerate |
| Fig. 3 | Threshold curves | Mask-NMS control and residual O | cache analysis | planned |
| Fig. 4 | Candidate diagrams | Complete/local competition and fragmentation | raw candidate cache | planned |
| Fig. 5 | Main results | Overall and dense-scene comparison | experiment results | pending |
| Fig. 6 | Error transitions | Before/after mechanism-specific corrections | experiment results | pending |
| Table 1 | Related work | Mechanism, location, dataset, limitation | research-wiki papers | planned |
| Table 2 | Dataset/protocol | Splits, annotation semantics, density statistics | dataset audit | pending |
| Table 3 | Main results | AP and relation metrics | experiment results | pending |
| Table 4 | Ablation | Mechanism component contribution | experiment results | pending |
| Table 5 | Efficiency | Params/FLOPs/FPS/memory | experiment results | pending |

## 5. Citation Plan

- Introduction: Panoptic Instance Segmentation on Pigs; CClusnet-Inseg; OVIS; Layering.
- Dense/occluded related work: BCNet, ORM, OCFusion, Relook, Layering, OVIS.
- Candidate and consistency mechanisms: Recurrent Instance Segmentation; SCE duplicate removal; relation/distillation and query-selection papers, each marked `[VERIFY]` until official metadata is checked.
- Domain context: Livestock Monitoring with Transformer; Beef Cattle Instance Segmentation; Object-Guided Instance Segmentation for Biological Images; agricultural dense-scene review.
- Citation rule: every paper used in the manuscript must have a verified author list, year, venue, and DOI/arXiv/official URL in `research-wiki/literature_registry.jsonl`; never create BibTeX from memory.

## 6. Experiment-Fill Order

1. Verify annotation semantics and freeze the evaluation protocol.
2. Reproduce existing baseline diagnostics on all available target datasets.
3. Complete the missing-source literature pass, including the Gemini CLI fallback; this pass is now recorded in `research-wiki/raw-search/` and the registry, with zero verifiable Gemini hits.
4. Run oracle/counterfactual tests to decide which mechanism(s) are causal candidates.
5. Select the minimum intervention that directly targets the confirmed mechanism.
6. Run main, failure-specific, ablation, cross-dataset, and efficiency experiments.
7. Update the Claims-Evidence Matrix; remove any claim whose evidence remains partial and keep cross-architecture generalization explicitly out of scope if it is not run.

## 7. Current Gate

The paper outline is ready, but the method-selection Gate is not passed. No network, loss, or main contribution is fixed until the mechanism replication and counterfactual experiments establish which failure mechanism is dominant and actionable.
