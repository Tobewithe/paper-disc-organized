---
type: experiment
node_id: exp:yolo26-gate-binding-20260905
title: "Isolated Top-K relaxation at fixed confidence is inactive on the frozen traces"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: high
date: "2026-09-05"
hardware: ""
duration: ""
provenance: "experiments/yolo26_gate_binding_20260905_v1/validation.json; experiments/yolo26_trace_replay_full_20260905_v1/summary.json"
added: 2026-09-05T12:37:05Z
tags: ["yolo26", "diagnosis", "retention"]
---

# Isolated Top-K relaxation at fixed confidence is inactive on the frozen traces

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
PigLife: 426 images, 4843 raw sources score > .05, zero outside Top-300. Faro: 160 images, 3515 raw sources score > .05, zero outside Top-300.

## Reasoning
Expanding only Top-K leaves the strict-confidence source set unchanged on these fixed traces. This establishes the effect of this one intervention, not score calibration, formation cause or general ranking behavior.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

