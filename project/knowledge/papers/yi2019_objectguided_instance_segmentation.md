---
type: paper
node_id: paper:yi2019_objectguided_instance_segmentation
title: "Object-Guided Instance Segmentation for Biological Images"
authors: ["Jingru Yi", "Hui Tang", "Pengxiang Wu", "Bo Liu", "Daniel J. Hoeppner", "Dimitris N. Metaxas", "Lianyi Han", "Wei Fan"]
year: 2019
venue: "arXiv"
external_ids:
  arxiv: "1911.09199"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Object-Guided Instance Segmentation for Biological Images

## One-line thesis
Object-Guided Instance Segmentation uses center-derived boxes and object features to separate clustered biological instances inside an RoI.

## Problem / Gap
Clustered, occluded, and adhered biological objects cause box-free methods to over- or under-segment because they lack a global object view.

## Method
Centers locate object boxes; object features are reused to guide the segmentation branch. Instance normalization models the target distribution and suppresses attached neighbors within the RoI.

## Key Results
The paper reports state-of-the-art performance on three biological datasets; exact values are not stated in the ingested source.

## Assumptions
- Assumes object centers can be detected reliably.
- Uses RoI-local features and instance normalization for clustered objects.

## Limitations / Failure Modes
- Biological datasets and box-based architecture differ from YOLO26 pig segmentation.
- It does not isolate whether errors arise during candidate formation, ranking, or mask reconstruction.

## Reusable Ingredients
- Center-guided object context and neighbor-distribution suppression.
- A useful comparator for domain-specific grouping mechanisms.

## Open Questions
- Can center-guided features improve complete/local candidate separability in pigs without a new detector?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This provides a biologically adjacent precedent for object-level context in clustered instances. It is relevant to the mechanism taxonomy, but current evidence does not justify selecting center guidance over relation, training, or ranking interventions.

## Abstract (original)

> Instance segmentation of biological images is essential for studying object behaviors and properties. The challenges, such as clustering, occlusion, and adhesion problems of the objects, make instance segmentation a non-trivial task. Current box-free instance segmentation methods typically rely on local pixel-level information. Due to a lack of global object view, these methods are prone to over- or under-segmentation. On the contrary, the box-based instance segmentation methods incorporate object detection into the segmentation, performing better in identifying the individual instances. In this paper, we propose a new box-based instance segmentation method. Mainly, we locate the object bounding boxes from their center points. The object features are subsequently reused in the segmentation branch as a guide to separate the clustered instances within an RoI patch. Along with the instance normalization, the model is able to recover the target object distribution and suppress the distribution of neighboring attached objects. Consequently, the proposed model performs excellently in segmenting the clustered objects while retaining the target object details. The proposed method achieves state-of-the-art performances on three biological datasets: cell nuclei, plant phenotyping dataset, and neural cells.

