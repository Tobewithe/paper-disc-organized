---
type: experiment
node_id: exp:residual-o-mechanisms
slug: residual-o-mechanisms
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: medium
metrics: "Residual O=196; SPLIT_TYPE=50; union coverage gain median .143; box/mask split A/B/C/D=6/17/19/8"
reasoning: "Residual O is heterogeneous; SPLIT_TYPE is the clearest complementary fragment pattern, while other groups are redundant or contaminated."
provenance: "C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01-YOLO26-残留O多预测联合恢复分析.md (SHA256 8AD3BA4FF8DC7AF7F51C53253BFB43A7D6774E6083A553D8BF6055815C6EAE91); C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01-YOLO26-残留O框掩码分裂来源定位.md (SHA256 8630049698365B388811649A1BA09589DE6C1F005BD3BB23664464CA4540ABF3); C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01-YOLO26-残留O新增区域贡献分析.md (SHA256 91FF71E078AA238AD329BC8D38BF7C40AEC779F12F8E1C53BACC4FA817413D1)"
added: 2026-09-05T01:00:00+08:00
---

# Residual O mechanisms

## Outcome

Partial. Across residual O, union IoU gain was not uniformly positive. For SPLIT_TYPE (50 GT), union coverage gain median was .143 and union IoU gain .084; only 6/50 were box-complete/mask-split, 17/50 box-split, 19/50 mixed, and 8/50 unresolved.

## Interpretation boundary

SPLIT_TYPE is a fixed descriptive label, not a universal failure class. Complementary additions and neighbor/background contamination coexist; no internal causal stage is established.

## Reusable evidence

Supports a mechanism taxonomy and targeted metrics for fragmentation versus contamination.

## Follow-up

Validate label stability across datasets and architectures, then test oracle/counterfactual interventions.

## Connections

Auto-generated from graph/edges.jsonl; no edges yet.
