---
type: paper
node_id: paper:bolya2020_tide_general_toolbox
title: "TIDE: A General Toolbox for Identifying Object Detection Errors"
authors: ["Daniel Bolya", "Sean Foley", "James Hays", "Judy Hoffman"]
year: 2020
venue: "arXiv"
external_ids:
  arxiv: "2008.08115"
  doi: null
  s2: null
tags: ["instance-segmentation", "error-diagnosis", "oracle"]
added: 2026-09-05T12:29:28Z
---

# TIDE: A General Toolbox for Identifying Object Detection Errors

## One-line thesis
TIDE measures output-error contributions using separate GT-informed AP oracles relative to the same baseline, avoiding order-dependent progressive error correction.

## Problem / Gap
AP mixes confidence, localization, duplicates, background and missing-instance errors. Counting error cases alone does not quantify their AP effect. Verified from official arXiv v2 PDF, Sections 1-2.

## Method
Define classification, localization, combined classification/localization, duplicate, background and missed-GT errors; measure each correction oracle's AP change from the same unmodified baseline. Foreground/background IoU thresholds default to .50/.10. The missed-GT oracle changes the recall denominator, rather than assigning an arbitrary score to a newly invented detection. Section 2.3 explains why progressive correction changes apparent error contribution with correction order.

## Key Results
Cross-model and cross-dataset analysis on COCO, Pascal VOC, Cityscapes and LVIS, including instance segmentation and Mask Scoring R-CNN. Figure 2 holds error quantities fixed and swaps correction order to expose the change in apparent contribution. No numeric AP gain is borrowed for this project.

## Assumptions
Predictions, scores, category labels and evaluation GT are available. The evaluation and correction oracle are explicitly defined. The framework operates on model outputs and does not require internal model access.

## Limitations / Failure Modes
An output-error oracle's AP delta is conditional on that oracle and metric. It is not a controlled measurement of an internal formation/ranking mechanism, nor a deployable intervention. The project's C/I/L/S/O/M/X/MISS relation taxonomy differs from TIDE's error types; the two must not be relabeled as equivalent.

## Reusable Ingredients
Start each retention/deletion intervention from the same baseline; state prediction and GT denominator treatment; preserve scores; compute AP separately from relation recovery. If full TIDE analysis is needed, use https://github.com/dbolya/tide rather than recreating its evaluator.

## Open Questions
Does a stage-specific intervention improve AP and relation failures together? Does a score-only correction show a different pattern from deletion or retention? These need project experiments.

## Claims
Literature support for evaluation design only; no project method acceptance or internal causal attribution.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Supports reporting R006 as a specified GT-informed experiment and keeping joint versus isolated corrections separate. Scholar Feed returns ECCV venue metadata; title/authors and Sections 1-3 verified against official arXiv v2, pages 1-9. Venue status has not been independently checked at the publisher in this pass. PDF: `papers/2008.08115.pdf`, SHA256 `32c48d5e34973caf6d496ffa0a1e8f70a2303c8c831a9c53d98ef47521d63d8d`. Official project: https://dbolya.github.io/tide/.

## Abstract (original)

> We introduce TIDE, a framework and associated toolbox for analyzing the sources of error in object detection and instance segmentation algorithms. Importantly, our framework is applicable across datasets and can be applied directly to output prediction files without required knowledge of the underlying prediction system. Thus, our framework can be used as a drop-in replacement for the standard mAP computation while providing a comprehensive analysis of each model's strengths and weaknesses. We segment errors into six types and, crucially, are the first to introduce a technique for measuring the contribution of each error in a way that isolates its effect on overall performance. We show that such a representation is critical for drawing accurate, comprehensive conclusions through in-depth analysis across 4 datasets and 7 recognition models. Available at https://dbolya.github.io/tide/

