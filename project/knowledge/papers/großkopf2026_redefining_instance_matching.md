---
type: paper
node_id: paper:großkopf2026_redefining_instance_matching
title: "Redefining Instance Matching: A Unified Framework for Part-Aware Matching in Panoptic Segmentation Evaluation"
authors: ["Erik Großkopf", "Soumya Snigdha Kundu", "Hendrik Möller", "Nicolas Münster", "Mehdi Astaraki", "Paula Tamara Buzduga", "Kerstin Ritter", "Benedikt Wiestler", "Jan Kirschke", "Jonathan Shapey", "Tom Vercauteren", "Florian Kofler"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.31094"
  doi: null
  s2: null
tags: ["instance matching", "evaluation", "partial regions"]
added: 2026-09-04T17:14:02Z
---

# Redefining Instance Matching: A Unified Framework for Part-Aware Matching in Panoptic Segmentation Evaluation

## One-line thesis
The paper reformulates panoptic instance matching as constrained bipartite assignment and compares one-to-one, many-to-one, one-to-many, and many-to-many strategies.

## Problem / Gap
Standard PQ matching is straightforward above IoU 0.5 but becomes underspecified when predictions fragment, adjacent objects are hard to delineate, or annotations are noisy.

## Method
The framework independently bounds prediction-side and ground-truth-side matching degrees, yielding four matching families. It uses vertex-based TP/FN/FP accounting and adds region-wise and part-aware analysis through the Panoptica package.

## Key Results
The abstract states that one-to-one, many-to-one, and one-to-many strategies are well-defined within PQ, while many-to-many falls outside it; configurable case studies demonstrate threshold- and strategy-dependent behavior.

## Assumptions
- Assumes panoptic-style segment matching and explicit IoU thresholds.
- Treats evaluation matching as separable from the model's prediction process.

## Limitations / Failure Modes
- It is an evaluation framework, not a YOLO inference intervention.
- The abstract does not establish which matching strategy best diagnoses dense pig failures.

## Reusable Ingredients
- Degree-constrained matching vocabulary for fragmented predictions.
- Region-wise analysis and area-under-threshold curves for robustness reporting.

## Open Questions
- Whether part-aware or many-to-one diagnostics correlate with YOLO26 raw candidate co-survival on pig masks is open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This paper directly informs the evaluation side of the project: it provides principled language for fragmented/adjacent predictions and can help audit whether `SPLIT_TYPE` is an error label or an artifact of one-to-one matching. It does not prove a causal model mechanism.

## Abstract (original)

> The Panoptic Quality (PQ) metric is the standard for jointly evaluating instance and semantic segmentation. However, its original definition relies on a One-to-One matching between predicted and ground truth segments, which is only straightforward when the IoU threshold exceeds 0.5. Below 0.5, multiple matching strategies emerge in a poorly explored problem space. We systematically elucidate this space by recasting segment matching as a constrained bipartite assignment problem. Independently bounding the prediction- and ground-truth-side degrees yields four matching strategies: One-to-One, Many-to-One, One-to-Many, and Many-to-Many. We show that the first three are well-defined within the PQ framework, while Many-to-Many falls outside it. These strategies become relevant when instances are fragmented, adjacent objects are difficult to delineate, or annotations are noisy. Central to our framework is a vertex-based accounting of TP, FN, and FP, anchored to ground truth and predicted segments rather than to matching edges. We further show that the framework extends naturally to part-aware panoptic segmentation, and we explore part-aware evaluation on biomedical data. Across configurable case studies we report how different combinations of thresholds and matching strategies behave in practice. We release a unified open-source package built on Panoptica. It exposes Voronoi-based region-wise analysis, part-aware evaluation, and Area Under Threshold Curve computations as configurable options.

