# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: explain why YOLO26-seg has a high failure rate in dense, touching pig scenes, then improve the confirmed failure mechanism.
- Must-solve bottleneck: distinguish candidate formation, candidate competition/selection, mask construction, and ordinary duplicate suppression instead of treating all residual errors as one class.
- Non-goals: no cross-architecture study in this phase; no preselected loss, module, or paper contribution; no causal claim from the reconstructed 1280/rect replay.
- Constraints: YOLO26-first; use PigLife, FaroPigSeg, and BamaPig2D when protocol parity is available; preserve all provenance and failure labels; use existing checkpoints and caches before new training.
- Success condition: a provenance-complete YOLO26 protocol identifies a reproducible dominant mechanism and a minimal intervention improves overall mask AP plus the corresponding dense-scene failure slice without unacceptable regressions.

## Anchor Check

- Original bottleneck: dense-scene YOLO26 failures have heterogeneous, incompletely localized mechanisms.
- The revised method still addresses it by making stage attribution a prerequisite for intervention.
- Rejected as drift: adding an LLM/VLM or expanding to another architecture merely to improve frontier language.

## Simplicity Check

- Dominant contribution: a stage-localized diagnosis protocol plus one minimum intervention.
- Components removed: graph stacks, multi-stage mask generators, and simultaneous data/mask/post-processing changes.
- The remaining mechanism is the smallest adequate route because the frozen generator exposes the candidates that already contain the diagnostic evidence.

## Changes Made

### 1. Hard mechanism gate

- Reviewer said: resolve the branching problem with a quantitative trigger.
- Action: after a new manifest, choose Route A only if (i) the candidate oracle recovers at least 40% of residual O/M/X failures, (ii) at least 30% of those recoveries have a complete candidate already present but incorrectly ranked or jointly selected, and (iii) the effect is directionally consistent on two datasets. Otherwise retain the dominant stage as a diagnostic result and test only its matched branch.
- Impact: the scorer is no longer an unconstrained default.

### 2. Concrete Route A scorer

- Reviewer said: define architecture and loss.
- Action: use a two-layer MLP over candidate and pair features, followed by differentiable listwise scoring over each image's candidate set. Train with pairwise margin ranking between valid complete candidates and hard local/duplicate negatives, plus a small score-calibration term; no backbone or mask-head updates.
- Impact: an engineer can implement the route without adding a graph architecture.

### 3. Formal definitions and sample target

- Reviewer said: formalize complete/local and state scale.
- Action: pre-register complete candidates as mask IoU >= 0.50 and GT coverage >= 0.70; local candidates as mask IoU in [0.10, 0.50) with GT coverage >= 0.10 and candidate-inside-GT fraction >= 0.60. Keep duplicate labels separate when candidate IoU >= 0.80. Generate a new manifest covering all eligible test images, with a minimum target of 100 dense/touching images per dataset where available; report actual counts and bootstrap intervals.
- Impact: the mechanism slice becomes auditable and threshold-sensitive analysis is planned.

## Revised Proposal

### Method thesis

Freeze YOLO26-seg and expose its pre-filter candidate set. Use a provenance-complete, stage-specific oracle to decide whether dense failures are primarily formation, ranking/competition, or mask construction. Only when the pre-registered candidate-recovery gate is met, train one lightweight candidate-set scorer; otherwise test the matched stage intervention and preserve the negative result.

### Pipeline

`raw YOLO26 candidates -> provenance-complete trace -> stage oracle/perturbation -> gate -> one stage-matched intervention -> standard AP + dense/relation metrics`

### Candidate representation

For each candidate, retain confidence, box geometry, mask area, mask-to-box ratio, GT-independent pair overlap, containment, complementary coverage proxy, local density, and score rank. Candidate labels are created only for evaluation/training splits with fixed matching rules; test GT is never used at inference.

### Route A implementation if gate passes

Use a two-layer MLP (hidden width fixed before training) on candidate and pair features. Produce a scalar utility per candidate and apply listwise selection under the existing final-set budget. Optimize pairwise margin ranking for valid complete versus hard local/duplicate candidates plus a calibration term against held-out candidate quality. Compare against score-only and Mask-NMS controls.

### Route selection and falsification

The gate is falsified if the candidate oracle recovers <40% of residual failures, if fewer than 30% are ranking/competition cases, or if the signal fails to replicate on two datasets. Missing-candidate dominance sends the study to formation-side data/representation tests; intrinsically poor masks send it to mask-side tests. No branch is allowed to claim a gain outside its diagnosed stage.

### Minimal validation

1. Re-run taxonomy and candidate trace on a new manifest across PigLife, FaroPigSeg, and BamaPig2D, targeting at least 100 dense/touching images per dataset where possible.
2. Run deterministic candidate oracle, removal/addition perturbation, and matching-threshold sensitivity before any training.
3. If Route A passes, compare baseline, Mask-NMS, scorer, and scorer-deletion control on mask AP/AP50/AP75, dense failure, O/M/X/SPLIT_TYPE correction/regression, latency, and bootstrap intervals.

The missing archived manifest remains an explicit limitation; reconstructed 1280/rect rows remain audit-only and are never pooled with the 1024 cache.
