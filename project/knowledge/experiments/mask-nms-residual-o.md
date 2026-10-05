---
type: experiment
node_id: exp:mask-nms-residual-o
slug: mask-nms-residual-o
idea: idea:diagnosis-to-improvement
verdict: no
confidence: high
metrics: "Mask-NMS@0.80: 427/675 O→C; 196 residual O; source-aware suppression had no extra mAP or DIFFERENT_GT advantage"
reasoning: "High-overlap duplicate suppression explains a large but bounded portion of O failures; residual errors are lower-overlap and heterogeneous."
provenance: "C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01-YOLO26-重复预测抑制干预实验.md; SHA256 7F6A7753764439B7DF32126E9FC161DCE6E0854CC44CF9561D43402AAF9A5717; C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01-YOLO26-NMS与来源感知抑制对比实验.md; SHA256 825F247CF2CE3F13410135D61130996CCF59AAE662A5F2DB0D14A24685E6BDD0; C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01-YOLO26-MaskNMS可修复与残留O对照分析.md; SHA256 E5FCC57916B7FD9AB65DC4A485FC72F76D5DEFB11BC923D2040D42259A80BAC4"
added: 2026-09-05T01:00:00+08:00
---

# Mask-NMS and residual O

## Outcome

Negative for the proposed source-aware post-processing route as a primary contribution. Ordinary Mask-NMS@0.80 converted 427/675 O cases to C, but left 196 O; source-aware suppression showed no additional advantage in the scanned fixed-cache runs.

## Interpretation boundary

This is an offline prediction-set intervention, not a production post-processing causal study. O→C does not imply every residual O is a fragmentation case.

## Reusable evidence

Provides a strong baseline and a principled reason not to make ordinary duplicate suppression the paper contribution.

## Follow-up

Analyze residual morphology and candidate formation before designing any intervention.

## Connections

Auto-generated from graph/edges.jsonl; no edges yet.
