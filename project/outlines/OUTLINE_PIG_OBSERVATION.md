---
schema_version: 1
id: OUTLINE_PIG_OBSERVATION
kind: outline
title: 判断一头猪究竟需要哪些观察？
parent_id: OUTLINE_PIG_INSTANCE_SEGMENTATION
outline_level: 2
summary: 区分当前帧已有信息与历史、视角、分辨率提供的新证据，理解何时应获取更多观察。
author: AI依据用户三层结构要求及既有研究整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 猪视频系统、STARFORMER与多尺度表示的输入权限和限制
    relation: references
---

## 关键问题

错误可能来自没有用好本帧，也可能因为本帧缺少足够证据。历史帧、其他视角和原图细节改变的信息不同，应分别理解它们什么时候有价值、什么时候不可信，以及成本是否值得。

## 已有认识与排除范围

猪视频文献显示额外观察与强模型可以组成实用系统，同时常含已知猪数、参考帧、未来信息和质量控制。旧视频或预训练线的局部阴性不等于本项目猪实例任务已经检验这些问题。当前原视频可得性和原图额外细节尚未确认。

## 当前探索反馈

[局部重看V1](../../experiments/pig_local_relook_20261103/REPORT_V1.md)的128TRAIN仅15个合格失败，未达32，当前病例入口已结束、方法未测；不能把它外推成无需更多观察。随后[夜间四臂训练](../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)已完整结束：同帧局部细节训练未过正常效用线，保持项相对有效但仍有净退化和超限损伤，当前配置关闭。它研究同帧信息的利用，不等于检验了新的原图、视角或历史帧；广义观察问题继续未知。

## 2026-11-04综合反馈

[具体夜间训练Idea](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)和[广义细节Idea](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)已补清：P高于同条件普通R，却仍低于多阶段保留N；同帧训练的相对收益和更换已有模型的价值分别判断。它没有新增像素或历史信息，[综合复盘](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)继续保留新观察价值未知，不把病例不足、裁剪训练阴性合并成“更多观察无用”；具体边界见[细节缺口](OUTLINE_PIG_DETAIL_EVIDENCE.md)。

## 具体未解缺口

- [少量看清过的历史观察能否解除当前归属歧义？](OUTLINE_PIG_HISTORY_EVIDENCE.md)
- [原图细节是否提供当前全图路径没有利用的信息？](OUTLINE_PIG_DETAIL_EVIDENCE.md)

## 从这里形成的Idea

[历史可见证据](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)和[按需获取细节](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)可以与[可信归属](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)连接。观察不足与表示不足应共同比较，不能从一个方法失败就自动得出必须视频或必须更高分辨率。

[视频对应与身份知识](OUTLINE_VIDEO_OCCLUSION.md)提供跨线参考，原线的输入、预算和实验结论各自保留。观察信息的判断继续回到[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。
