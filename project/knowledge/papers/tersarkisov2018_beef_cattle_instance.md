---
type: paper
node_id: paper:tersarkisov2018_beef_cattle_instance
title: "Beef Cattle Instance Segmentation Using Fully Convolutional Neural Network"
authors: ["Aram Ter-Sarkisov", "Robert Ross", "John Kelleher", "Bernadette Earley", "Michael Keane"]
year: 2018
venue: "arXiv"
external_ids:
  arxiv: "1807.01972"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Beef Cattle Instance Segmentation Using Fully Convolutional Neural Network

## One-line thesis
The paper converts a fully convolutional network into an instance segmenter that predicts one mask per beef cattle animal in CCTV footage.

## Problem / Gap
Livestock monitoring needs individual animal masks for behavior and welfare analysis, but farm imagery contains repeated, similar animals.

## Method
The method trains a fully convolutional network to output a separate prediction for every animal instance in the scene.

## Key Results
The abstract establishes a proof-of-concept on winter finishing cattle CCTV; no quantitative metrics are stated in the ingested source.

## Assumptions
- Assumes a fixed CCTV domain and animal appearance distribution.
- Targets cattle rather than pigs and does not describe heavy occlusion controls.

## Limitations / Failure Modes
- Early domain-specific work with limited evidence for modern dense-scene failure analysis.
- No candidate-level or relation-level diagnostics are reported.

## Reusable Ingredients
- Livestock-specific instance-mask framing and welfare-monitoring use case.

## Open Questions
- How the approach behaves under modern crowded pig benchmarks remains open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This supports the applied livestock motivation but offers little direct evidence for the YOLO26 dense-pig failure mechanism; it remains contextual rather than a method precedent.

## Abstract (original)

> We present an instance segmentation algorithm trained and applied to a CCTV recording of beef cattle during a winter finishing period. A fully convolutional network was transformed into an instance segmentation network that learns to label each instance of an animal separately. We introduce a conceptually simple framework that the network uses to output a single prediction for every animal. These results are a contribution towards behaviour analysis in winter finishing beef cattle for early detection of animal welfare-related problems.

