---
type: paper
node_id: paper:yi2019_multiscale_cell_instance
title: "Multi-scale Cell Instance Segmentation with Keypoint Graph based Bounding Boxes"
authors: ["Jingru Yi", "Pengxiang Wu", "Qiaoying Huang", "Hui Qu", "Bo Liu", "Daniel J. Hoeppner", "Dimitris N. Metaxas"]
year: 2019
venue: "arXiv"
external_ids:
  arxiv: "1907.09140"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Multi-scale Cell Instance Segmentation with Keypoint Graph based Bounding Boxes

## One-line thesis
The method detects five cell keypoints, groups them with a keypoint graph, derives boxes, and segments each cell within its box.

## Problem / Gap
Touching cells are hard to separate because box-free methods rely on local pixels and anchor detectors suffer class imbalance.

## Method
Keypoint detection finds five predefined points per cell; graph grouping forms a cell-level box, after which box-local features drive segmentation.

## Key Results
Experiments on two cell datasets reportedly outperform other instance-segmentation methods; exact metrics are not stated in the abstract.

## Assumptions
- Assumes each cell has a stable set of five detectable keypoints.
- Uses object grouping before mask prediction.

## Limitations / Failure Modes
- Cell morphology and keypoint annotations do not transfer directly to pigs.
- The method does not address YOLO26 raw candidate ranking or score calibration.

## Reusable Ingredients
- Keypoint-graph grouping as a touching-object separation baseline.
- Box-local segmentation to inject object-level context.

## Open Questions
- Whether a pig-specific center/keypoint representation could reduce candidate fragmentation is untested.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is an adjacent object-grouping mechanism for dense touching instances. It supports comparing center/keypoint grouping against YOLO26 candidate behavior without implying that the cell solution is directly portable.

## Abstract (original)

> Most existing methods handle cell instance segmentation problems directly without relying on additional detection boxes. These methods generally fails to separate touching cells due to the lack of global understanding of the objects. In contrast, box-based instance segmentation solves this problem by combining object detection with segmentation. However, existing methods typically utilize anchor box-based detectors, which would lead to inferior instance segmentation performance due to the class imbalance issue. In this paper, we propose a new box-based cell instance segmentation method. In particular, we first detect the five pre-defined points of a cell via keypoints detection. Then we group these points according to a keypoint graph and subsequently extract the bounding box for each cell. Finally, cell segmentation is performed on feature maps within the bounding boxes. We validate our method on two cell datasets with distinct object shapes, and empirically demonstrate the superiority of our method compared to other instance segmentation techniques. Code is available at: https://github.com/yijingru/KG_Instance_Segmentation.

