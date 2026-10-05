---
type: experiment
node_id: exp:yolo26-strict-quality-20260905
title: "YOLO26 strict mask quality and all-GT simultaneous matching"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: high
date: "2026-09-05"
hardware: ""
duration: ""
provenance: "experiments/yolo26_strict_quality_fulltrace_20260905_v1; experiments/yolo26_strict_matching_fulltrace_20260905_v1; research-wiki/yolo26_strict_full_integrity_20260905.md"
added: 2026-09-05T13:43:04Z
tags: ["yolo26", "diagnosis", "mask-quality", "matching", "integrity"]
---

# YOLO26 strict mask quality and all-GT simultaneous matching

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
586 images; 6226 GT; 193413 edges; quality availability and exact maximum-cardinality matching for IoU50, coverage75_purity75 and IoU75 across raw/Top-K/conf/final; final-source XOR=0; 20 condition-stage assignment-deficit records; strict coverage/purity failed-GT final available/matched PigLife 312/310 of 394, Faro 224/224 of 797.

## Reasoning
Integrity and independent CPU audit pass. Strict quality reveals a larger Faro deficit and separates pairwise availability from simultaneous allocation, but edges do not exclude cross-GT contamination and matching is not output-set recovery. No internal causal attribution, AP gain, deployable intervention or Route-A acceptance is established.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

