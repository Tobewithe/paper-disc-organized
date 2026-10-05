---
type: paper
node_id: paper:henrich2025_benchmarking_pig_detection
title: "Benchmarking pig detection and tracking under diverse and challenging conditions"
authors: ["Jonathan Henrich", "Christian Post", "Maximilian Zilke", "Parth Shiroya", "Emma Chanut", "Amir Mollazadeh Yamchi", "Ramin Yahyapour", "Thomas Kneib", "Imke Traulsen"]
year: 2025
venue: "Computers and Electronics in Agriculture, Volume 241, 2026, 111264"
external_ids:
  arxiv: "2507.16639"
  doi: null
  s2: null
tags: ["pig", "benchmark", "dense scenes"]
added: 2026-09-04T17:14:21Z
---

# Benchmarking pig detection and tracking under diverse and challenging conditions

## One-line thesis
PigDetect and PigTrack benchmark individual pig localization and tracking under realistic barn conditions, including occlusion and poor visibility.

## Problem / Gap
Pig monitoring lacks a systematic benchmark spanning diverse real-barn conditions and characteristic failure cases, limiting reproducible comparison and generalization analysis.

## Method
The authors curate PigDetect for detection and PigTrack for multi-object tracking, compare state-of-the-art and real-time detectors, and analyze end-to-end tracking failures. The study also tests training on challenging images and unseen pens.

## Key Results
Challenging training images improve detection beyond random sampling; state-of-the-art models outperform real-time alternatives for detection, while SORT-based methods have stronger detection and end-to-end models stronger association. Models generalize to unseen pens; exact metrics are not stated in the abstract.

## Assumptions
- Assumes realistic barn imagery and detection/tracking labels rather than dense instance masks.
- Separates localization quality from temporal association quality.

## Limitations / Failure Modes
- It is primarily detection and tracking, so it does not identify mask-construction or mask-candidate causes.
- Transfer to YOLO26-seg instance masks and `SPLIT_TYPE` requires direct evaluation.

## Reusable Ingredients
- PigDetect/PigTrack dataset design and unseen-pen generalization protocol.
- Failure-case reporting that distinguishes model family and data difficulty.

## Open Questions
- Whether the challenging-image subsets predict dense mask failure strata is open.

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is a domain and evaluation anchor for the YOLO-first study. It supports stratifying failures by realistic barn difficulty and testing generalization, but it is not direct evidence for the project's instance-mask candidate competition mechanism.

## Abstract (original)

> To ensure animal welfare and effective management in pig farming, monitoring individual behavior is a crucial prerequisite. While monitoring tasks have traditionally been carried out manually, advances in machine learning have made it possible to collect individualized information in an increasingly automated way. Central to these methods is the localization of animals across space (object detection) and time (multi-object tracking). Despite extensive research of these two tasks in pig farming, a systematic benchmarking study has not yet been conducted. In this work, we address this gap by curating two datasets: PigDetect for object detection and PigTrack for multi-object tracking. The datasets are based on diverse image and video material from realistic barn conditions, and include challenging scenarios such as occlusions or bad visibility. For object detection, we show that challenging training images improve detection performance beyond what is achievable with randomly sampled images alone. Comparing different approaches, we found that state-of-the-art models offer substantial improvements in detection quality over real-time alternatives. For multi-object tracking, we observed that SORT-based methods achieve superior detection performance compared to end-to-end trainable models. However, end-to-end models show better association performance, suggesting they could become strong alternatives in the future. We also investigate characteristic failure cases of end-to-end models, providing guidance for future improvements. The detection and tracking models trained on our datasets perform well in unseen pens, suggesting good generalization capabilities. This highlights the importance of high-quality training data. The datasets and research code are made publicly available to facilitate reproducibility, re-use and further development.

