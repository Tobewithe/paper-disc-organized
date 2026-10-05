# Research Proposal: Diagnosis-Driven Improvement of Dense-Pig YOLO26 Segmentation

## Problem Anchor
- Bottom-line problem: explain why YOLO26-seg has a high failure rate in dense, touching pig scenes, then improve the confirmed failure mechanism.
- Must-solve bottleneck: distinguish candidate formation, candidate competition/selection, mask construction, and ordinary duplicate suppression instead of treating all residual errors as one class.
- Non-goals: no cross-architecture study in this phase; no preselected loss, module, or paper contribution; no causal claim from the reconstructed 1280/rect replay.
- Constraints: YOLO26-first; use PigLife, FaroPigSeg, and BamaPig2D when protocol parity is available; preserve all provenance and failure labels; use existing checkpoints and caches before new training.
- Success condition: a provenance-complete YOLO26 protocol identifies a reproducible dominant mechanism and a minimal intervention improves overall mask AP plus the corresponding dense-scene failure slice without unacceptable regressions.

## Technical Gap

The recovered 1024-square cache covers 918 images and 7,374 GT instances with a 19.84% total failure rate. Mask-NMS@0.80 repairs 427/675 O cases, so ordinary high-overlap duplication is not sufficient as the main explanation. Residual `SPLIT_TYPE` evidence contains complete/local candidate coexistence and positive union-oracle gains, but the formal sample manifest is missing and the 1280/rect replay is audit-only. Existing occlusion, layering, clustering, and duplicate-removal papers do not directly test this low-IoU candidate-set distinction in YOLO26 or pig scenes.

The operational gap is therefore a mechanism test: under a fixed YOLO26 protocol, which failure stage removes or misranks the evidence needed to explain one GT instance? A larger backbone or generic NMS cannot answer this.

## Method Thesis

- Provisional thesis: retain the YOLO26 generator, trace a provenance-complete candidate set per GT, and add only the smallest candidate-set decision mechanism supported by the replicated failure stage.
- Route A (minimal, primary candidate): a lightweight candidate-set scorer/re-ranker using overlap, complementary coverage, containment, box-mask consistency, score calibration, and local density; it acts after candidate generation and before final selection.
- Route B (diagnosis-dependent fallback): if replicated failures are dominated by missing complete candidates rather than ranking, use failure-slice data construction or a small formation-side head; if mask construction dominates, test a mask-side intervention instead.
- No route is locked until the mechanism gate is passed on a newly generated manifest.

## Contribution Focus

- Dominant contribution: a reproducible diagnosis-to-intervention protocol that separates dense-scene failure mechanisms and tests the minimum mechanism-targeted YOLO26 improvement.
- Optional supporting contribution: a candidate-level metric suite for complete/local coexistence, candidate recall, union upper bound, and relation-specific correction/regression.
- Explicit non-contributions: cross-architecture generalization, a new universal segmentation architecture, or a generic NMS heuristic.

## Proposed Method

### Complexity Budget
- Frozen/reused: YOLO26-seg backbone, neck, mask prototype path, train/test split, and standard predictor.
- New trainable component: at most one small candidate-set scorer; training is deferred until its offline signal is confirmed.
- Excluded: graph stacks, VLM/LLM inference, multi-stage mask generators, and simultaneous changes to data, mask head, and post-processing.

### System Overview

`YOLO26 raw candidates -> candidate trace + GT-independent features -> stage-specific diagnostic oracle -> (if justified) lightweight set scorer -> final selection -> standard mask metrics + failure metrics`

### Core Mechanism

For each image, preserve raw candidates before Top-K/confidence/final-set filters. Build pair/set features from geometry, mask overlap, complementary coverage, containment, score gap, and box-mask agreement. First use deterministic oracle and perturbation tests to identify whether the failure is formation, ranking, or mask quality. Only if ranking/competition is reproducibly dominant, train a scorer with positives defined by valid one-to-one GT matching and hard negatives from surviving local/duplicate candidates. Use pairwise/listwise ranking plus a calibration term; keep the scorer frozen at inference except for the final candidate selection stage.

### Training and Inference Boundary

Training, loss, and candidate labels are not fixed by this proposal. They are selected only after the stage-specific oracle establishes a stable signal across datasets. Inference must expose candidate recall and candidate-set decisions so a gain cannot be attributed to a hidden threshold change.

### Failure Modes and Diagnostics
- No complete candidate: scorer cannot recover it; report formation failure and test data/formation branch only if dominant.
- Complete candidate exists but is dropped: ranking branch is testable with counterfactual re-ranking.
- Complete and local candidates coexist: evaluate relation-aware selection and union upper bound.
- Mask quality is intrinsically poor: report as mask-construction failure and do not claim candidate selection solved it.
- Protocol drift: regenerate and retain a sample manifest; never merge reconstructed 1280/rect rows with the 1024 exploratory cache.

## Claim-Driven Validation Sketch

### Claim 1: Dense YOLO26 failures are stage-heterogeneous and reproducible
- Minimal experiment: rerun the taxonomy and candidate trace on a new provenance-complete sample for PigLife, then replicate on FaroPigSeg and BamaPig2D where labels permit.
- Metrics: total failure, O/M/X/MISS, candidate recall, stage transition counts, bootstrap intervals.
- Expected evidence: stable stage-specific patterns, not a single aggregate error rate.

### Claim 2: The selected intervention targets the confirmed stage
- Minimal experiment: compare baseline, Mask-NMS control, stage-matched oracle, and one intervention; include deletion of the new component.
- Metrics: mask AP/AP50/AP75, dense-slice failure, relation-specific correction/regression, latency.
- Expected evidence: improvement in the diagnosed slice and no unexplained overall regression.

## Experiment Handoff Inputs
- Must-prove: mechanism replication, candidate-stage counterfactual, intervention alignment.
- Must-run controls: baseline, Mask-NMS threshold sweep, oracle upper bound, scorer deletion/control.
- Critical data: PigLife first; FaroPigSeg/BamaPig2D replication; newly retained sample manifest.
- Highest-risk assumptions: low-IoU candidate competition is not an artifact of the missing manifest or matching rule; ranking signal transfers across datasets.

## Compute & Timeline Estimate
- Diagnostic replay: existing GPU/cache plus one protocol-complete YOLO26 inference per dataset.
- Intervention prototype: one small training job only after the gate; estimate 1-3 GPU-days total for the first decision cycle.
- Timeline: protocol/replication first, then mechanism gate, then one intervention branch; no parallel architecture expansion.
