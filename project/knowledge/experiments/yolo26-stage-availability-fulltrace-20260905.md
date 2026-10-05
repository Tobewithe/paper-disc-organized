---
type: experiment
node_id: exp:yolo26-stage-availability-fulltrace-20260905
title: "YOLO26 full-trace per-GT stage availability"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: medium
date: "2026-09-05"
hardware: "RTX 5060 Ti 16 GB; Conda pytorch"
duration: ""
provenance: "research-wiki/yolo26_stage_availability_20260905.md; research-wiki/yolo26_stage_review_20260905.md; experiments/yolo26_stage_availability_full_20260905_v1/provenance.json"
added: 2026-09-05T11:45:44Z
tags: ["yolo26", "diagnosis", "public", "availability"]
---

# YOLO26 full-trace per-GT stage availability

**verdict:** `partial`  ·  **confidence:** `medium`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
PigLife failed GT: raw/top-k/conf/final=390/390/373/373 of 394. FaroPigSeg:746/698/553/553 of 797. IoU>=.50 existence per GT. Replay:586 images,8327 final masks,zero XOR.

## Reasoning
Supports a dataset-dependent description of where individual candidate availability is lost. Does not establish simultaneous instance matching, causal ranking failure, optimal oracle recovery, or deployable scorer improvement. Same-family stage review is provisional.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

