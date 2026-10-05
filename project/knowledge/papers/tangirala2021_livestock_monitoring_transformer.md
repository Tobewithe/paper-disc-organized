---
type: paper
node_id: paper:tangirala2021_livestock_monitoring_transformer
title: "Livestock Monitoring with Transformer"
authors: ["Bhavesh Tangirala", "Ishan Bhandari", "Daniel Laszlo", "Deepak K. Gupta", "Rajat M. Thomas", "Devanshu Arya"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2111.00801"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Livestock Monitoring with Transformer

## One-line thesis
Starformer jointly performs pig instance segmentation, tracking, action recognition, and re-identification with transformer embeddings.

## Problem / Gap
Group-housed pigs look similar and experience long-duration occlusion, viewpoint, illumination, and scale changes, making isolated vision tasks hard to generalize.

## Method
The end-to-end transformer learns instance-level embeddings while optimizing the STAR tasks jointly. The authors introduce PigTrace with boxes, masks, tracks, and activity labels from real indoor farms.

## Key Results
The abstract reports that Starformer outperforms popular single-task baselines; exact metrics are not stated in the ingested source.

## Assumptions
- Assumes synchronized video and multi-task annotations.
- Uses temporal context and identity embeddings beyond single-image YOLO inference.

## Limitations / Failure Modes
- Multi-task tracking evidence cannot directly identify single-frame mask candidate failures.
- Dataset and model details require full-text verification before quantitative citation.

## Reusable Ingredients
- PigTrace annotation schema and real-farm difficulty factors.
- Instance embeddings as a domain-specific relation signal.

## Open Questions
- Whether temporal identity cues can help explain or repair single-frame `SPLIT_TYPE` errors is open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is a pig-domain context and annotation precedent. It supports reporting realistic occlusion and identity conditions, but the current study intentionally focuses first on YOLO26 single-image instance-mask failure mechanisms.

## Abstract (original)

> Tracking the behaviour of livestock enables early detection and thus prevention of contagious diseases in modern animal farms. Apart from economic gains, this would reduce the amount of antibiotics used in livestock farming which otherwise enters the human diet exasperating the epidemic of antibiotic resistance - a leading cause of death. We could use standard video cameras, available in most modern farms, to monitor livestock. However, most computer vision algorithms perform poorly on this task, primarily because, (i) animals bred in farms look identical, lacking any obvious spatial signature, (ii) none of the existing trackers are robust for long duration, and (iii) real-world conditions such as changing illumination, frequent occlusion, varying camera angles, and sizes of the animals make it hard for models to generalize. Given these challenges, we develop an end-to-end behaviour monitoring system for group-housed pigs to perform simultaneous instance level segmentation, tracking, action recognition and re-identification (STAR) tasks. We present starformer, the first end-to-end multiple-object livestock monitoring framework that learns instance-level embeddings for grouped pigs through the use of transformer architecture. For benchmarking, we present Pigtrace, a carefully curated dataset comprising video sequences with instance level bounding box, segmentation, tracking and activity classification of pigs in real indoor farming environment. Using simultaneous optimization on STAR tasks we show that starformer outperforms popular baseline models trained for individual tasks.

