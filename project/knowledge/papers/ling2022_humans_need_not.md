---
type: paper
node_id: paper:ling2022_humans_need_not
title: "Humans need not label more humans: Occlusion Copy & Paste for Occluded Human Instance Segmentation"
authors: ["Evan Ling", "Dezhao Huang", "Minhoe Hur"]
year: 2022
venue: "arXiv"
external_ids:
  arxiv: "2210.03686"
  doi: null
  s2: null
tags: []
added: 2026-09-04T15:20:57Z
---

# Humans need not label more humans: Occlusion Copy & Paste for Occluded Human Instance Segmentation

## One-line thesis
Occlusion Copy & Paste creates same-class occlusion examples from existing data without extra manual labels.

## Problem / Gap
Crowded human scenes expose weaknesses that model-centric changes cannot fully exploit without enough occluded training examples.

## Method
The augmentation adapts copy-paste to same-class occlusion and systematically studies add-ons. It is model-agnostic and changes the training distribution rather than the inference selector.

## Key Results
The paper reports state-of-the-art performance on OCHuman; exact metrics are not stated in the ingested abstract.

## Assumptions
- Assumes existing instance masks can be recomposed into realistic occlusions.
- Targets same-class occlusion and benefits from a large source dataset.

## Limitations / Failure Modes
- Human/OCHuman results do not establish transfer to pigs.
- It does not diagnose or correct low-IoU candidate co-survival at inference.

## Reusable Ingredients
- A data-centric occlusion baseline that is compatible with many model families.
- Add-on ablations for separating augmentation effects.

## Open Questions
- Whether copy-paste reduces `SPLIT_TYPE` without changing candidate ranking is untested.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is a useful training-side control for the diagnosis-to-improvement study, allowing data-distribution effects to be separated from YOLO26 candidate-selection effects.

## Abstract (original)

> Modern object detection and instance segmentation networks stumble when picking out humans in crowded or highly occluded scenes. Yet, these are often scenarios where we require our detectors to work well. Many works have approached this problem with model-centric improvements. While they have been shown to work to some extent, these supervised methods still need sufficient relevant examples (i.e. occluded humans) during training for the improvements to be maximised. In our work, we propose a simple yet effective data-centric approach, Occlusion Copy & Paste, to introduce occluded examples to models during training - we tailor the general copy & paste augmentation approach to tackle the difficult problem of same-class occlusion. It improves instance segmentation performance on occluded scenarios for "free" just by leveraging on existing large-scale datasets, without additional data or manual labelling needed. In a principled study, we show whether various proposed add-ons to the copy & paste augmentation indeed contribute to better performance. Our Occlusion Copy & Paste augmentation is easily interoperable with any models: by simply applying it to a recent generic instance segmentation model without explicit model architectural design to tackle occlusion, we achieve state-of-the-art instance segmentation performance on the very challenging OCHuman dataset. Source code is available at https://github.com/levan92/occlusion-copy-paste.

