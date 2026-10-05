---
type: paper
node_id: paper:haidar2024_maskuno_switchsplit_block
title: "MaskUno: Switch-Split Block For Enhancing Instance Segmentation"
authors: ["Jawad Haidar", "Marc Mouawad", "Imad Elhajj", "Daniel Asmar"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2407.21498"
  doi: null
  s2: null
tags: ["instance segmentation", "mask separation"]
added: 2026-09-04T17:14:15Z
---

# MaskUno: Switch-Split Block For Enhancing Instance Segmentation

## One-line thesis
MaskUno routes refined ROIs through a Switch-Split block and specialized mask predictors to reduce competing mask kernels.

## Problem / Gap
The paper identifies competition among mask kernels when models learn multiple object classes synchronously and seeks to improve instance-mask specialization.

## Method
After ROI refinement and classification, the Switch-Split block assigns regions to specialized mask predictors instead of sending all masks through a shared predictor. The intervention is inside the mask head and is evaluated as a plug-in to existing instance-segmentation models.

## Key Results
The abstract reports a 2.03% mAP increase for DetectoRS on COCO; broader exact results are not included in the ingested source.

## Assumptions
- Assumes ROI-based instance-segmentation architectures with a mask head.
- Frames competition primarily across classes and predictor kernels.

## Limitations / Failure Modes
- The reported setting is COCO and ROI-based models, not YOLO26-seg.
- It does not establish handling of same-class complete/local low-overlap candidates.

## Reusable Ingredients
- Specialized mask predictors as a possible internal competition-control baseline.
- A clear distinction between mask-head competition and post-processing duplicate suppression.

## Open Questions
- Whether specialized predictors help when competition is among same-class pig instances remains untested.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
MaskUno is a relevant architectural neighbor for the project's candidate-competition hypothesis, but its evidence concerns shared mask-kernel competition in ROI models and cannot be claimed as a YOLO26 solution or novelty gap closure.

## Abstract (original)

> Instance segmentation is an advanced form of image segmentation which, beyond traditional segmentation, requires identifying individual instances of repeating objects in a scene. Mask R-CNN is the most common architecture for instance segmentation, and improvements to this architecture include steps such as benefiting from bounding box refinements, adding semantics, or backbone enhancements. In all the proposed variations to date, the problem of competing kernels (each class aims to maximize its own accuracy) persists when models try to synchronously learn numerous classes. In this paper, we propose mitigating this problem by replacing mask prediction with a Switch-Split block that processes refined ROIs, classifies them, and assigns them to specialized mask predictors. We name the method MaskUno and test it on various models from the literature, which are then trained on multiple classes using the benchmark COCO dataset. An increase in the mean Average Precision (mAP) of 2.03% was observed for the high-performing DetectoRS when trained on 80 classes. MaskUno proved to enhance the mAP of instance segmentation models regardless of the number and typ

