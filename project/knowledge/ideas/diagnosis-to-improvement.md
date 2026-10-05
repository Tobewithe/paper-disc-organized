---
type: idea
node_id: idea:diagnosis-to-improvement
slug: diagnosis-to-improvement
title: "密集猪群实例分割的诊断到改进"
stage: proposed
outcome: pending
thesis: "先定位密集场景实例分割失败的可重复机制，再针对被证实的主导机制设计并验证最小干预。"
risks: "现有诊断主要来自固定 checkpoint/cache；失败标签可能依赖阈值、数据集与架构，不能提前选择方法路线。"
target_gaps: "G1-low-iou-fragment,G2-candidate-selection,G3-pig-domain-transfer,G4-evaluation"
added: 2026-09-05T01:05:00+08:00
---

# 密集猪群实例分割的诊断到改进

## Current status

Exploration only. The imported experiments establish evidence and exclusions, not a selected model, loss, or paper contribution.

## Evidence basis

See `exp:yolo26-failure-taxonomy`, `exp:cross-dataset-failure-structure`, `exp:mask-nms-residual-o`, `exp:residual-o-mechanisms`, and `exp:candidate-competition`.

## Decision gate

Run cross-dataset/cross-architecture mechanism validation and oracle/counterfactual tests before any method route is chosen.

## Connections

Auto-generated from graph/edges.jsonl; no edges yet.
