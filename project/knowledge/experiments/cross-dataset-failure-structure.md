---
type: experiment
node_id: exp:cross-dataset-failure-structure
slug: cross-dataset-failure-structure
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: medium
metrics: "PigLife/FaroPigSeg/BamaPig2D; 3 models; GT failure families and joint hard/disagreement profiles"
reasoning: "Failure structure changes across datasets and architectures; this is descriptive evidence for a diagnosis-first study, not a causal architecture comparison."
provenance: "C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-08-27-PigLife-FaroPigSeg-BamaPig2D跨数据集失败结构对比.md; SHA256 01EF23FDE26FEB77396D1D992F12A965FCD22F9B4BE337069D37EAA5321B6AC0; C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-08-27-FaroPigSeg与BamaPig2D三模型外部失败分析.md; SHA256 55C0CD26FC73B4BC4359440F549EEF9A190BFE37693E6CABF1B61A919EFB47AC"
added: 2026-09-05T01:00:00+08:00
---

# Cross-dataset failure structure

## Outcome

Partial. On fixed PigLife-trained checkpoints, PigLife hard failures were 2.26%, FaroPigSeg 17.12%, and BamaPig2D 13.50%. The dominant failure family differed by dataset/model.

## Interpretation boundary

Dataset, viewpoint, scale, density, labels, and architecture change together. These results support cross-dataset slicing and do not identify a causal module or rank architectures intrinsically.

## Reusable evidence

Use for paper Sections 3 (failure analysis), 5 (generalization), and the protocol requiring overall AP plus dense/relationship slices.

## Follow-up

Align annotation semantics and run the same taxonomy across architectures before selecting an intervention.

## Connections

Auto-generated from graph/edges.jsonl; no edges yet.
