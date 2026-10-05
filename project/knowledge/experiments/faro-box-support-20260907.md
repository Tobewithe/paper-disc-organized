---
type: experiment
node_id: exp:faro-box-support-20260907
title: "Faro box support and source polygon consistency"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: high
date: "2026-09-07"
hardware: ""
duration: ""
provenance: "experiments/faro_box_support_20260907_v1/deterministic_verification.json; research-wiki/faro_mask_quality_20260907.md; research-wiki/papers/compte2025_housed_pig_identification.md"
added: 2026-09-06T17:35:47Z
tags: ["faropigseg", "diagnosis", "box-support", "annotation"]
---

# Faro box support and source polygon consistency

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
160 images, 1752 GT, 7008 stage records. 307 raw strict-mask-absent failures split into 37 support-infeasible and 270 support-feasible. Source raster parity passed; 3484 final masks inside support. Independent OpenCV/COCO implementation reproduced 7008 rows and verified 489 hashes.

## Reasoning
Deterministic support-count verification PASS only. Support feasibility is necessary and does not identify prototype, coefficient, mask-head or annotation-policy cause. Official annotation semantics remain unverified; method Gate remains blocked.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

