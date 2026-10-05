# Research Proposal: Diagnosis-Driven Improvement of Dense-Pig YOLO26 Segmentation

> 2026-09-07 状态更正：下文是历史条件性方案，不是当前方法批准。评分器和两层 MLP 不具优先执行权；40%/30% 不构成因果识别。当前按 `research-wiki/mechanism_evidence_ledger_20260907.md` 恢复证据、比较竞争解释，Faro 框/mask/标注诊断优先。跨架构历史证据需要恢复和口径对齐，不能沿用下文的范围限制排除它。原文保留用于追溯。

## Problem Anchor

Explain why YOLO26-seg has a high failure rate in dense, touching pig scenes, then improve the confirmed failure mechanism. The bottleneck is to separate candidate formation, candidate competition/selection, mask construction, and ordinary duplicate suppression. Cross-architecture claims, a preselected loss/module, and causal claims from reconstructed 1280/rect data are out of scope.

## Technical Gap

The recovered 1024-square cache covers 918 images and 7,374 GT instances with 19.84% total failure. Mask-NMS@0.80 repairs 427/675 O cases, while residual `SPLIT_TYPE` cases show complete/local candidate coexistence and positive union-oracle gains. The formal sample manifest is missing, so the 1280/rect replay is audit-only. Existing occlusion, layering, clustering, and duplicate-removal work does not directly test this low-IoU candidate-set distinction under YOLO26 in pig scenes.

## Method Thesis

Freeze the YOLO26 generator and expose a provenance-complete pre-filter candidate set. Use stage-specific deterministic oracles and perturbations to identify formation, ranking/competition, or mask-quality dominance. Only if the preregistered gate is met, train one lightweight candidate-set scorer; otherwise test the stage-matched branch and preserve the negative result.

## Decision Gate

Route A candidate-set scorer is enabled only when a new manifest shows: (i) candidate oracle recovery >=40% of residual O/M/X failures; (ii) >=30% of recovered cases have a complete candidate present but incorrectly ranked or jointly selected; and (iii) the direction replicates on two datasets. Otherwise the dominant stage determines the single matched intervention.

## Minimal Route A

For each candidate retain confidence, box geometry, mask area, mask-to-box ratio, candidate overlap, containment, complementary-coverage proxy, local density, and rank. Pre-register complete as mask IoU >=0.50 and GT coverage >=0.70; local as IoU in [0.10,0.50), GT coverage >=0.10, and candidate-inside-GT fraction >=0.60; keep IoU >=0.80 duplicates separate. Use a fixed two-layer MLP to score candidates and listwise select under the existing final-set budget. Train with pairwise margin ranking between valid complete candidates and hard local/duplicate negatives plus a small calibration term. Log ranking-inversion frequency per epoch. Freeze the YOLO26 backbone, neck, and mask prototype path.

## Validation

1. Generate a new provenance-complete manifest covering all eligible test images, targeting at least 100 dense/touching images per dataset where available, and replicate the taxonomy on PigLife, FaroPigSeg, and BamaPig2D.
2. Run candidate oracle, removal/addition perturbations, and matching-threshold sensitivity before training.
3. If the gate passes, compare baseline, Mask-NMS, scorer, and scorer-deletion controls using mask AP/AP50/AP75, dense failure, O/M/X/SPLIT_TYPE correction/regression, candidate recall, latency, and bootstrap intervals.

All reconstructed 1280/rect rows remain audit-only and are never pooled with the 1024 cache. No LLM/VLM component is added because it does not answer the candidate-stage causal question.
