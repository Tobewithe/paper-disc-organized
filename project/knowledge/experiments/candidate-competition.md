---
type: experiment
node_id: exp:candidate-competition
slug: candidate-competition
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: medium
metrics: "SPLIT_TYPE=50; complete raw candidate absent 11, dropped 8, survived with local co-survival 31; local-complete score gap median .185"
reasoning: "Candidate formation/quality, score or rank loss, and final local-complete co-survival all occur; no single competition mechanism is sufficient."
provenance: "C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-02-YOLO26-SPLIT_TYPE完整候选与局部候选竞争分析.md (SHA256 63BAFBF0EF16B66C61B5BDA8A92BC71B527231FB3189D6585F40F369DF5775B7); C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-02-YOLO26-SPLIT_TYPE原始候选多峰与正确候选存在性分析.md (SHA256 8B731A77EA1597B74478344CAC3A4A570729F58F1C2AC68FC7C590DBEF55F2C4)"
added: 2026-09-05T01:00:00+08:00
---

# Complete/local candidate competition

## Outcome

Partial. In 50 SPLIT_TYPE GT, 11 had no complete raw candidate, 8 had one that was dropped, and 31 retained complete and local candidates together. The local-minus-complete score gap median was .185 versus .013 in matched C controls.

## Interpretation boundary

The complete candidate is a fixed box-quality proxy, not the model's true proposal. Multimodality was similarly common in SPLIT_TYPE and controls (96% vs 94%), so multiple spatial peaks alone are not discriminative.

## Reusable evidence

Fills candidate-formation and selection analysis in Section 3 and motivates oracle/counterfactual tests without locking a method.

## Follow-up

Reproduce with exact runtime, raw mask reconstruction, and matched controls across datasets.

## Connections

Auto-generated from graph/edges.jsonl; no edges yet.
