---
type: paper
node_id: paper:baselizadeh2025_occlusionordered_semantic_instance
title: "Occlusion-Ordered Semantic Instance Segmentation"
authors: ["Soroosh Baselizadeh", "Cheuk-To Yu", "Olga Veksler", "Yuri Boykov"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2504.14054"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Occlusion-Ordered Semantic Instance Segmentation

## One-line thesis
OOSIS jointly predicts instance masks and relative occlusion order from oriented occlusion boundaries and semantic segmentation.

## Problem / Gap
Absolute monocular depth is difficult, while ordinary instance segmentation does not provide relative 3D ordering needed for occlusion analysis.

## Method
The method extracts instances and their ordering simultaneously from oriented occlusion boundaries and semantic labels, formulating the task as a labeling problem rather than a standard detect-and-segment pipeline. It also introduces an oriented-boundary predictor and a joint mask/order metric.

## Key Results
The abstract reports better performance than strong baselines on KINS and COCOA; exact numbers are not stated in the ingested source.

## Assumptions
- Assumes occlusion boundaries provide reliable relative-depth cues.
- Evaluates datasets with occlusion-order annotations or compatible supervision.

## Limitations / Failure Modes
- The method depends on visible occlusion boundaries and does not address low-overlap complementary candidates directly.
- Peer-reviewed status and detailed failure cases require source verification.

## Reusable Ingredients
- Relative occlusion-order labels and joint mask/order evaluation.
- Boundary-aware analysis for separating occluder and occludee errors.

## Open Questions
- Whether relative order can be inferred robustly for touching pigs without explicit boundary labels remains open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This provides a relation-aware evaluation and mechanism analogue for dense pig masks, but the project's `SPLIT_TYPE` cases may lack order annotations and may not contain visible occlusion boundaries.

## Abstract (original)

> Standard semantic instance segmentation provides useful, but inherently 2D information from a single image. To enable 3D analysis, one usually integrates absolute monocular depth estimation with instance segmentation. However, monocular depth is a difficult task. Instead, we leverage a simpler single-image task, occlusion-based relative depth ordering, providing coarser but useful 3D information. We show that relative depth ordering works more reliably from occlusions than from absolute depth. We propose to solve the joint task of relative depth ordering and segmentation of instances based on occlusions. We call this task Occlusion-Ordered Semantic Instance Segmentation (OOSIS). We develop an approach to OOSIS that extracts instances and their occlusion order simultaneously from oriented occlusion boundaries and semantic segmentation. Unlike popular detect-and-segment framework for instance segmentation, combining occlusion ordering with instance segmentation allows a simple and clean formulation of OOSIS as a labeling problem. As a part of our solution for OOSIS, we develop a novel oriented occlusion boundaries approach that significantly outperforms prior work. We also develop a new joint OOSIS metric based both on instance mask accuracy and correctness of their occlusion order. We achieve better performance than strong baselines on KINS and COCOA datasets.

