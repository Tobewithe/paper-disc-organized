---
type: experiment
node_id: exp:yolo26-confidence-20260907
title: "YOLO26 fixed confidence retention and matched GT controls"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: medium
date: "2026-09-07"
hardware: ""
duration: ""
provenance: "experiments/yolo26_confidence_diagnostic_fulltrace_20260905_v2; experiments/yolo26_confidence_statistics_20260907_v1; research-wiki/yolo26_confidence_diagnostic_20260907.md"
added: 2026-09-06T17:29:08Z
tags: ["yolo26", "diagnosis", "confidence", "oracle"]
---

# YOLO26 fixed confidence retention and matched GT controls

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
586 images, 6226 GT, five conditions. Confidence .01 recovers 12 and regresses 269. GT joint recovers 43 selected targets plus 2 neighbors. Paired image bootstrap complete.

## Reasoning
Frozen-threshold relaxation worsens relation failures; oracle selection is not deployable. Independent integrity review pending; method Gate remains blocked.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

