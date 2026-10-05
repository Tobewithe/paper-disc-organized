---
type: paper
node_id: paper:he2025_segment_concealed_objects
title: "Segment Concealed Objects with Incomplete Supervision"
authors: ["Chunming He", "Kai Li", "Yachao Zhang", "Ziyun Yang", "Youwei Pang", "Longxiang Tang", "Chengyu Fang", "Yulun Zhang", "Linghe Kong", "Xiu Li", "Sina Farsiu"]
year: 2025
venue: "10.1109/TPAMI.2025.3576209"
external_ids:
  arxiv: "2506.08955"
  doi: null
  s2: null
tags: ["incomplete supervision", "concealment", "partial masks"]
added: 2026-09-04T17:13:57Z
---

# Segment Concealed Objects with Incomplete Supervision

## One-line thesis
SEE combines mean-teacher pseudo-labeling with hybrid-granularity feature grouping for concealed-object segmentation under incomplete supervision.

## Problem / Gap
Concealed-object segmentation is difficult because training annotations may be weak or incomplete and object appearance can resemble the background.

## Method
SEE uses a unified mean-teacher framework and prompts SAM with coarse teacher masks to generate pseudo-labels. It stores high-quality pseudo-labels and selects reliable components for student supervision, while a hybrid-granularity feature grouping module clusters features at multiple scales to improve mask coherence.

## Key Results
The abstract reports state-of-the-art performance across multiple ISCOS tasks and describes SEE as a plug-and-play enhancement for existing models; exact metrics are not stated in the ingested source.

## Assumptions
- Assumes weak or semi-annotated training data and access to SAM for pseudo-label generation.
- Targets concealed objects whose appearance may be intrinsically similar to background.

## Limitations / Failure Modes
- The source addresses concealed-object and incomplete-supervision settings, not YOLO26 candidate competition directly.
- No evidence is provided here for dense pig instance segmentation or low-IoU complete/local candidate co-survival.

## Reusable Ingredients
- Mean-teacher pseudo-label storage and reliability selection.
- Multi-granularity feature grouping as a coherence-oriented mechanism.

## Open Questions
- Whether the pseudo-label and grouping design transfers to fully supervised dense animal instance segmentation remains open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is an adjacent completeness/coherence mechanism for the diagnosis-to-improvement study; it motivates testing whether incomplete local evidence, rather than only duplicate overlap, limits mask completeness, but it does not establish the project's `SPLIT_TYPE` mechanism.

## Abstract (original)

> Incompletely-Supervised Concealed Object Segmentation (ISCOS) involves segmenting objects that seamlessly blend into their surrounding environments, utilizing incompletely annotated data, such as weak and semi-annotations, for model training. This task remains highly challenging due to (1) the limited supervision provided by the incompletely annotated training data, and (2) the difficulty of distinguishing concealed objects from the background, which arises from the intrinsic similarities in concealed scenarios. In this paper, we introduce the first unified method for ISCOS to address these challenges. To tackle the issue of incomplete supervision, we propose a unified mean-teacher framework, SEE, that leverages the vision foundation model, ``\emph{Segment Anything Model (SAM)}'', to generate pseudo-labels using coarse masks produced by the teacher model as prompts. To mitigate the effect of low-quality segmentation masks, we introduce a series of strategies for pseudo-label generation, storage, and supervision. These strategies aim to produce informative pseudo-labels, store the best pseudo-labels generated, and select the most reliable components to guide the student model, thereby ensuring robust network training. Additionally, to tackle the issue of intrinsic similarity, we design a hybrid-granularity feature grouping module that groups features at different granularities and aggregates these results. By clustering similar features, this module promotes segmentation coherence, facilitating more complete segmentation for both single-object and multiple-object images. We validate the effectiveness of our approach across multiple ISCOS tasks, and experimental results demonstrate that our method achieves state-of-the-art performance. Furthermore, SEE can serve as a plug-and-play solution, enhancing the performance of existing models.

