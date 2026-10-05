---
type: experiment
node_id: exp:yolo26-unified-diagnostic-20260905
slug: yolo26-unified-diagnostic-20260905
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: high
metrics: "918 images; 7,374 GT; PigLife COCO alignment 426/4,474; total failure PigLife 8.81%, Faro 45.49%, Bama 23.69%"
reasoning: "Unified the legacy GT-aligned YOLO26 baseline and failure-stage tables into machine-readable ARIS artifacts and verified the recovered PigLife COCO view against legacy counts."
provenance: "experiments/yolo26_unified_diagnostic_20260905/summary.json; experiments/yolo26_unified_diagnostic_20260905/provenance_audit.json; legacy source C:\\Dpan\\codexproject\\pigcv_research\\artifacts\\analysis\\yolo26seg_diagnostic_full_single_forward_20260901_final"
added: 2026-09-05T19:40:00+08:00
---

# YOLO26 unified diagnostic

## Outcome

Partial. The baseline replay tables are now available as JSON/CSV and the recovered PigLife COCO test view matches the legacy GT/image counts exactly. This closes the data-alignment blocker but does not establish a causal mechanism or intervention gain.

## Key results

| Dataset | Images | GT | Total failure | Relation failure | Miss |
|---|---:|---:|---:|---:|---:|
| PigLife | 426 | 4,474 | 8.81% | 7.53% | 0.29% |
| FaroPigSeg | 160 | 1,752 | 45.49% | 20.21% | 9.70% |
| BamaPig2D | 332 | 1,148 | 23.69% | 10.80% | 7.40% |

## Interpretation boundary

These are fixed-cache replay and inherited evaluator outputs. They are suitable for protocol and descriptive baseline reporting. Candidate-stage causal claims remain gated on provenance-complete oracle and perturbation evidence.

## Artifacts

- `experiments/yolo26_unified_diagnostic_20260905/summary.json`
- `experiments/yolo26_unified_diagnostic_20260905/baseline_summary.csv`
- `experiments/yolo26_unified_diagnostic_20260905/scene_failure_taxonomy.csv`
- `experiments/yolo26_unified_diagnostic_20260905/failure_onset_taxonomy.csv`
- `experiments/yolo26_unified_diagnostic_20260905/provenance_audit.json`
