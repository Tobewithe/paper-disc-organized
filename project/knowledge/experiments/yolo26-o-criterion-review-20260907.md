---
type: experiment
node_id: exp:yolo26-o-criterion-review-20260907
title: "Review of O relation criterion and fragmentation blind spots"
idea_id: "idea:diagnosis-to-improvement"
verdict: partial
confidence: high
date: "2026-09-07"
hardware: "CPU; fixed Faro/Bama segmentation exports"
duration: ""
provenance: "experiments/o_criterion_review_20260907_v1/summary.json; rule_summary.csv; transitions.csv; synthetic_cases.json; research-wiki/o_criterion_review_20260907.md. Coordinator deterministic review, not independent acceptance."
added: 2026-09-06T18:35:31Z
tags: ["yolo26", "failure-taxonomy", "criterion-review", "fragmentation", "diagnostic"]
---

# Review of O relation criterion and fragmentation blind spots

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:diagnosis-to-improvement`

## Metrics
27745 baseline labels reproduced; 12 source hashes unchanged; seven actual-classifier counterexamples. Current MISS with high-quality unions of high-purity sub-half predictions: Faro 245, Bama 29. Current O with pair IoU >= .80: Faro 1498/2801; Bama 731/841.

## Reasoning
Current O is coherent as a strong one-to-many component, but cannot enumerate all fragmentation or distinguish duplication. Blanket purity guards destroy merge edges. Screen positives are not validated new labels. Alternative rules were audited, not adopted; baseline and method gates unchanged.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

