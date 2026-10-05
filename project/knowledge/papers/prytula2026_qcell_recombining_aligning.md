---
type: paper
node_id: paper:prytula2026_qcell_recombining_aligning
title: "QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation"
authors: ["Yaroslav Prytula", "Anton Popov", "Dmytro Fishman"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2608.29253"
  doi: null
  s2: null
tags: ["instance-segmentation", "query-recombination", "contrastive-learning", "overlap", "cell-microscopy"]
added: 2026-09-05T13:38:21Z
---

# QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation

## One-line thesis
Latent decomposition and recombination of amodal, visible, and invisible cell queries, combined with contrastive alignment, targets complete instance reasoning under microscopy overlap.

## Problem / Gap
Semi-transparent overlapping cells mix visual evidence and weaken boundaries, making complete object structure and same-class instance separation difficult. The paper targets overlap-aware object reasoning inside a query-based segmentation model.

## Method
QCell extends MaskDINO. Each decoder query is decomposed into amodal, visible, and invisible sub-representations, which generate sub-masks and are recombined into a refined full-instance query with consistency regularization. Matched main queries and denoising queries provide positive/negative pairs for instance-discriminative contrastive learning and cosine query alignment. This is an in-network query representation method, not post-hoc frozen-candidate subset selection.

## Key Results
The paper evaluates ISBI2014, Revvity-25, and a new Organoids benchmark with three-seed test averages. On ISBI2014, MaskDINO reports 63.7 AP and 75.9 AJI while QCell reports 65.9 AP and 78.6 AJI; the abstract summarizes these as +2.2 AP and +2.7 AJI. The ISBI2014 ablation reports complementary gains from recombination and contrastive alignment, including 13.68 versus 11.67 AP on a highly overlapping subset.

## Assumptions
Training requires compatible amodal, visible, and invisible component masks and MaskDINO-style query/DN supervision. Experiments use amodal mask contracts, ResNet-50-FPN, 100 queries on ISBI2014/Revvity-25 and 300 on Organoids, with all models trained rather than frozen candidate traces.

## Limitations / Failure Modes
The method depends on microscopy overlap and component-level supervision that may not exist for visible/modal pig masks. It changes query representations and mask prediction during training, so it does not isolate candidate retention, ranking, Top-K, NMS, or output-set selection. Results are author-reported and do not establish transfer to YOLO26 or pig datasets.

## Reusable Ingredients
A query-level decomposition/recombination objective and explicit same-class query separation are candidate mechanism families for overlap, but only after annotation semantics and training/evaluation splits support them. Heavy-overlap slices and object-level false-negative measures can complement AP when instance identity matters.

## Open Questions
Can pig annotations provide trustworthy visible/occluded or amodal components? If not, can the representation idea be tested without inventing hidden-region labels? Would query separation alter the project's complete/local failure taxonomy or merely change mask quality?

## Claims
Literature claim only: query recombination and contrastive query alignment are established near-neighbor mechanisms for overlapping cell segmentation. No project claim or novelty clearance is inferred.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This paper creates high novelty risk for generic query-recombination/alignment wording, while its amodal microscopy supervision and trained MaskDINO architecture leave the frozen YOLO26 pig-candidate setting unresolved. It is an adjacent mechanism comparator, not project evidence.

## Abstract (original)

> Instance segmentation of overlapping cells in microscopy remains challenging due to semi-transparent structures that produce weak boundaries and mixed visual evidence in overlap regions. Existing methods address this through local regions of interest or shape priors but lack global reasoning across overlapping objects. We present QCell, a novel query-based model that de-overlaps cell instances in microscopy scenes. Our approach combines (i) an instance recombination module that decomposes and recombines query representations in latent space, enabling the model to reason about complete object structure under overlap, and (ii) a contrastive query alignment objective that combines distinctive instance feature learning and separation of overlapping cell queries. We additionally introduce a new Organoid dataset benchmark for overlapping cell segmentation. We show that QCell outperforms state-of-the-art methods across multiple benchmarks, achieving +2.2 AP and +2.7 AJI on ISBI2014. Code is available at https://github.com/SlavkoPrytula/QCell

