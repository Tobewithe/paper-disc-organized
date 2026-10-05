---
type: paper
node_id: paper:ke2021_deep_occlusionaware_instance
title: "Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers"
authors: ["Lei Ke", "Yu-Wing Tai", "Chi-Keung Tang"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2103.12340"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers

## One-line thesis
BCNet models occluder and occludee masks as two interacting layers to separate true contours from occlusion boundaries.

## Problem / Gap
Highly overlapping objects create ambiguity between object contours and boundaries caused by occlusion, which standard instance segmentation does not explicitly model.

## Method
A top GCN layer detects occluding objects and a bottom GCN layer infers partially occluded objects. The bilayer representation models their interaction during mask regression and can be inserted into one-stage or two-stage detectors.

## Key Results
Experiments on COCO and KINS report consistent gains, especially under heavy occlusion; exact values are not stated in the abstract.

## Assumptions
- Assumes meaningful occluder/occludee relationships can be inferred from image evidence.
- Relies on detector features and mask regression compatible with the bilayer module.

## Limitations / Failure Modes
- Evidence centers on high-overlap occlusion, not low-IoU complementary fragments.
- It does not establish a causal explanation for YOLO26 score/rank failures.

## Reusable Ingredients
- Explicit bilayer relation modeling as a controlled mechanism family.
- Heavy-occlusion evaluation slices for stress testing.

## Open Questions
- Whether bilayer supervision or inferred relations are available in the project's pig annotations remains open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
BCNet is a strong relation-modeling precedent for the diagnostic stage, but it should remain a comparison mechanism until the YOLO26 failure evidence shows that occlusion relations, rather than candidate formation or ranking, dominate.

## Abstract (original)

> Segmenting highly-overlapping objects is challenging, because typically no distinction is made between real object contours and occlusion boundaries. Unlike previous two-stage instance segmentation methods, we model image formation as composition of two overlapping layers, and propose Bilayer Convolutional Network (BCNet), where the top GCN layer detects the occluding objects (occluder) and the bottom GCN layer infers partially occluded instance (occludee). The explicit modeling of occlusion relationship with bilayer structure naturally decouples the boundaries of both the occluding and occluded instances, and considers the interaction between them during mask regression. We validate the efficacy of bilayer decoupling on both one-stage and two-stage object detectors with different backbones and network layer choices. Despite its simplicity, extensive experiments on COCO and KINS show that our occlusion-aware BCNet achieves large and consistent performance gain especially for heavy occlusion cases. Code is available at https://github.com/lkeab/BCNet.

