---
type: experiment
node_id: exp:yolo26-candidate-gate-20260905
slug: yolo26-candidate-gate-20260905
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: medium
metrics: "1,463 failed GT; Top-K mask-IoU>=0.50 oracle 665/1,463=45.45%; retained scalar table has 0 measurable top-k selection losses"
reasoning: "Full retained candidate table indicates a potentially large candidate upper bound, but its schema does not preserve enough pre-final candidates to identify ranking loss. The two-dataset replication gate is therefore not met."
provenance: "experiments/yolo26_candidate_gate_20260905/summary.json; experiments/yolo26_candidate_gate_20260905/candidate_gate_by_dataset_class.csv; C:\\Dpan\\codexproject\\pigcv_research\\data\\analysis\\pigcv_analysis.db"
added: 2026-09-05T19:45:00+08:00
---

# YOLO26 candidate gate audit

## Result

Across the retained scalar database, 665 of 1,463 failed GT rows (45.45%) have at least one candidate with mask IoU >= 0.50 in the recorded Top-K/final candidate table. However, the retained candidate rows with a good mask are exclusively primary class O; this is not a cross-class oracle. The table records zero Top-K-good candidates later dropped, so it cannot support a ranking-loss estimate.

The R006 coverage audit confirms the limitation: candidate rows exist for every failed O row (PigLife 289/289, FaroPigSeg 291/291, BamaPig2D 95/95), while every failed I/L/M/X/MISS/S row has zero retained candidate rows. This is a schema/retention coverage gap, not evidence that the runtime generated no candidates for those classes.

## Gate interpretation

The 40% oracle threshold is exceeded only for the O-covered slice of one dataset (PigLife); the required two-dataset direction and cross-class coverage are absent. R005 therefore remains partial-audit and does not authorize training a candidate scorer. A provenance-complete candidate perturbation trace retaining pre-final candidates across O/M/X and MISS is still required.

## Artifacts

- `experiments/yolo26_candidate_gate_20260905/summary.json`
- `experiments/yolo26_candidate_gate_20260905/candidate_gate_by_dataset.csv`
- `experiments/yolo26_candidate_gate_20260905/candidate_gate_by_dataset_class.csv`
- `experiments/yolo26_candidate_gate_20260905/candidate_coverage_by_class.csv`
- `experiments/yolo26_candidate_gate_20260905/candidate_coverage_by_class.json`
