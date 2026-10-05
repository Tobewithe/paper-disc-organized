---
type: experiment
node_id: exp:yolo26-r006-fixed-fulltrace-20260905
title: "R006 fixed full-trace GT-guided output-set recovery"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: medium
date: "2026-09-05"
hardware: ""
duration: ""
provenance: "experiments/r006_fulltrace_statistics_20260905_v2; research-wiki/yolo26_final_integrity_20260905.md; experiments/yolo26_runtime_source_snapshot_20260905_v1/v2_run_sources.json"
added: 2026-09-05T12:09:25Z
tags: ["yolo26", "diagnosis", "oracle", "integrity"]
---

# R006 fixed full-trace GT-guided output-set recovery

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
PigLife: 322/394 recovered, 0/4080 regressions. Faro: 477/797 recovered, 1/955 regressions. Full coverage: 6226 GT. Unique assigned sources: 1136. Final-source masks: 8327, XOR=0. Joint predictions: 8327-1566+1035=7796. V1/V2 classes and selected sources identical.

## Reasoning
Terra integrity PASS, scientific interpretation WARN, same-family provisional. Achieved recovery under a specified GT-guided matching policy; no optimality, causal ranking attribution, deployed AP gain, or method acceptance.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

