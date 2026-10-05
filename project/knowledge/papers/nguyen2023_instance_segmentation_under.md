---
type: paper
node_id: paper:nguyen2023_instance_segmentation_under
title: "Instance Segmentation under Occlusions via Location-aware Copy-Paste Data Augmentation"
authors: ["Son Nguyen", "Mikel Lainsa", "Hung Dao", "Daeyoung Kim", "Giang Nguyen"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2310.17949"
  doi: null
  s2: null
tags: ["occlusion", "data augmentation"]
added: 2026-09-04T17:13:59Z
---

# Instance Segmentation under Occlusions via Location-aware Copy-Paste Data Augmentation

## One-line thesis
Location-aware Copy-Paste augments occlusion layouts and, with HTC-based inference, improves instance segmentation in the DeepSportRadar setting.

## Problem / Gap
Occluded and deformable objects create limited training diversity and ambiguous visible regions, making dense instance segmentation difficult.

## Method
The method generates training samples with location-aware copy-paste augmentation. The reported winning pipeline combines HTC, CBNetV2, a MaskIoU head, and stochastic weight averaging; the augmentation changes training data rather than post-hoc candidate selection.

## Key Results
The paper reports an occlusion score of 0.533 and first place in the MMSports 2023 challenge; these results are for DeepSportRadar and should not be transferred to pig data.

## Assumptions
- Assumes a challenge distribution where occlusion layout and object deformation can be simulated by copy-paste.
- Evaluates a two-stage HTC-family pipeline rather than YOLO26-seg.

## Limitations / Failure Modes
- Domain is basketball human segmentation, so animal-domain transfer is unverified.
- The source does not test low-IoU complementary candidate co-survival or YOLO score calibration.

## Reusable Ingredients
- Location-aware copy-paste as a controlled occlusion-distribution intervention.
- MaskIoU and SWA are reproducibility baselines for an occlusion benchmark.

## Open Questions
- Whether location-aware augmentation changes the project's observed failure taxonomy without changing inference selection remains open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This paper supplies a training-side comparison route for dense-scene failures. It is useful for separating data-distribution effects from YOLO26 candidate-selection effects, but it does not explain the project's low-overlap complete/local competition.

## Abstract (original)

> Occlusion is a long-standing problem in computer vision, particularly in instance segmentation. ACM MMSports 2023 DeepSportRadar has introduced a dataset that focuses on segmenting human subjects within a basketball context and a specialized evaluation metric for occlusion scenarios. Given the modest size of the dataset and the highly deformable nature of the objects to be segmented, this challenge demands the application of robust data augmentation techniques and wisely-chosen deep learning architectures. Our work (ranked 1st in the competition) first proposes a novel data augmentation technique, capable of generating more training samples with wider distribution. Then, we adopt a new architecture - Hybrid Task Cascade (HTC) framework with CBNetV2 as backbone and MaskIoU head to improve segmentation performance. Furthermore, we employ a Stochastic Weight Averaging (SWA) training strategy to improve the model's generalization. As a result, we achieve a remarkable occlusion score (OM) of 0.533 on the challenge dataset, securing the top-1 position on the leaderboard. Source code is available at this https://github.com/nguyendinhson-kaist/MMSports23-Seg-AutoID.

