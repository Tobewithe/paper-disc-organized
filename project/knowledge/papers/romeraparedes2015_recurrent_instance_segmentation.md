---
type: paper
node_id: paper:romeraparedes2015_recurrent_instance_segmentation
title: "Recurrent Instance Segmentation"
authors: ["Bernardino Romera-Paredes", "Philip H. S. Torr"]
year: 2015
venue: "ECCV 2016. 14th European Conference on Computer Vision"
external_ids:
  arxiv: "1511.08250"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Recurrent Instance Segmentation

## One-line thesis
Recurrent Instance Segmentation predicts objects sequentially while maintaining spatial memory of already explained pixels.

## Problem / Gap
Independently trained detection and segmentation modules miss joint reasoning opportunities, especially when objects overlap or must be explained as a set.

## Method
An RNN emits one instance at a time and updates spatial memory after each prediction. The memory supports occlusion handling, and a task-specific loss represents sequential instance segmentation.

## Key Results
The authors report gains on multiple-person segmentation and state-of-the-art results on a plant phenotyping leaf-counting dataset; exact metrics are not stated here.

## Assumptions
- Assumes instances can be ordered and decoded sequentially.
- Requires a memory state that records explained pixels.

## Limitations / Failure Modes
- Sequential decoding differs substantially from YOLO26's fixed candidate set.
- The historical experiments do not test dense pig masks or modern real-time constraints.

## Reusable Ingredients
- Spatial explained-region memory as a conceptual candidate-competition baseline.
- Sequential versus parallel decoding as a diagnostic contrast.

## Open Questions
- Whether an explained-region state can be approximated without abandoning YOLO26 latency is open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This paper motivates testing whether global explained-region context, rather than ordinary NMS, distinguishes complete/local candidate competition; it is a mechanism precedent, not a selected implementation.

## Abstract (original)

> Instance segmentation is the problem of detecting and delineating each distinct object of interest appearing in an image. Current instance segmentation approaches consist of ensembles of modules that are trained independently of each other, thus missing opportunities for joint learning. Here we propose a new instance segmentation paradigm consisting in an end-to-end method that learns how to segment instances sequentially. The model is based on a recurrent neural network that sequentially finds objects and their segmentations one at a time. This net is provided with a spatial memory that keeps track of what pixels have been explained and allows occlusion handling. In order to train the model we designed a principled loss function that accurately represents the properties of the instance segmentation problem. In the experiments carried out, we found that our method outperforms recent approaches on multiple person segmentation, and all state of the art approaches on the Plant Phenotyping dataset for leaf counting.

