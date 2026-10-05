---
type: experiment
node_id: exp:yolo26-failure-taxonomy
slug: yolo26-failure-taxonomy
idea: idea:diagnosis-to-improvement
verdict: partial
confidence: medium
metrics: "918 images; 7,374 GT; 9,628 predictions; baseline failure 19.84%; C 80.16%; I/L/S 5.15%; O/M/X 11.05%; MISS 3.63%"
reasoning: "A reproducible taxonomy and onset trace separate duplicate survival from mask/relationship failure and score loss in a fixed YOLO26 pipeline."
provenance: "C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-08-27-PigLife-public_test-三模型失败分析重建归档.md; SHA256 DE505257F51C9B45296F5646862FB6823A26E9D6278424F1392FA10F5D59BD30; C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-08-31-YOLO26-Dense-Touching-关系失败内部诊断归档.md; SHA256 C364650588F45B5E8C22813DCC1BDFBFD434DA316A70A4519F7623328932BED1; C:\\Dpan\\document\\pigcv_reserch\\03-成果与资产\\2026-09-01 目前所有实验的链路存档.md"
added: 2026-09-05T01:00:00+08:00
---

# YOLO26 failure taxonomy

## Outcome

Partial. Full fixed-cache diagnosis gives 19.84% GT failure. In the Dense/Touching relation sample, 25/43 components were duplicate survival, 16/43 mask/relationship failure, and 2/43 score-threshold loss; no Top-300 ranking-loss case was observed in that sample.

## Interpretation boundary

The onset labels depend on fixed association rules, candidate screening, and resolution. They localize an evidence entry point, not an exhaustive proof over every raw candidate or a causal attribution to a network stage.

## Reusable evidence

Directly fills failure taxonomy, diagnostic protocol, and claims-evidence matrix in the paper outline.

## Follow-up

Repeat with exact runtime parity and pre-register relation-component metrics.

## Connections

Auto-generated from graph/edges.jsonl; no edges yet.
