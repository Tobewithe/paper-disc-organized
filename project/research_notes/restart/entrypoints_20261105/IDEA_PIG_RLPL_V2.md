# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_RLPL_V2.md](../../../ideas/IDEA_PIG_RLPL_V2.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_RLPL_V2
kind: idea
title: 合成共视原型对应的原生Mask适配（RLPL V2，已停止）
summary: D2虽优于连续对应C2，仍低于共同合成曝光的无辅助B2及项目保留N且损伤超限；关闭原三配置，保留对应信息、迁移与适配的开放问题。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_DATA, OUTLINE_PIG_TRAINING, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
author: AI于2026-11-03据原完整结案报告回填具体Idea身份；非新增实验或事前注册
curation_status: source_based_draft
scope:
  claim: 原V2合成共视离散原型对应适配能相对匹配对照和原生得到满足保护的净分割收益。
  boundary: 原数据、B2／C2／D2接口、共同曝光及各2000更新的冻结配置。
  retrospective_record: true
evidence:
  - path: experiments/spm_rlpl_pipeline_20261101/FINAL_REPORT.md
    revision: f7f8f9b737a3ebea593443fb67682142572db800c68973753a4c49f46f85accd
    locator: 三臂正式训练、完整评价、净效用与损伤及停止决定；包含Run和交付来源
    relation: references
---

## 当时想验证什么

利用合成同实例、同表面且双方可见的局部对应，在训练期约束离散原型分配，部署回到原生Mask模型，能否获得独立且可采用的真实分割收益？比较共同曝光无辅助B2、连续对应C2和原型对应D2，三臂各2000更新。

该条是对已执行想法的事后索引，原协议和Run是事实来源；不将稳定ID回填解释为当时已预注册。

## 实际结果与停止理由

[完整结案报告](../../../../experiments/spm_rlpl_pipeline_20261101/FINAL_REPORT.md)给出正常Mask AP：N57.343500、B2 56.222099、C2 52.357215、D2 54.481384。D2较C2高2.124169点，却较B2低1.740715点、较N低2.862116点；相对N的原成功损伤113/2126＝5.3151%，超过原1%限。

训练、完整评价和实物交付完成；停止B2／C2／D2这三配置的继续投入，保留N。相对较差控制变好不足以支持“该配置有可用净收益”。因此进度为已关闭，这一限定范围的联合主张记为被结果反驳；不抹去D2相对C2的局部事实。

## 排除范围与仍值得理解的问题

本结果不否定所有原型或对应学习。原报告未区分BN、权重变化等竞争解释，不能把停止原因补写为已识别的单一机制；原报告提出的历史下一步也不自动成为当前动作。

[真实历史观察](../../../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)可以提出不同于合成共视的证据来源，[场景迁移与保持](../../../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)可以研究适配风险。这些方向仍为候选，不把旧失败当作其阳性依据。若未来回到原型对应，应指出新的可用信息或机制证据为何足以改变旧判断，而非仅增加训练或更改名称。

## 2026-11-04综合反馈：共同曝光控制与退化解释

[综合复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#5-rlpl原型优于连续对应但低于共同曝光无辅助对照)明确：B2共同接触合成RGB及Mask路径BN，不能当作纯真实续训；N则是[多阶段项目保留权重](../../PIG_YOLO_BASELINE_LINEAGE_20261104.md)，不能叫未经目标训练的官方模型。D2−B2为−1.740715 AP，仍是关闭本配置的直接依据，D2−C2的局部优势保留。

后来BN完全固定的概念／夜间续训也有退化，因此BN不足以统一解释这些失败；它在RLPL中的作用仍未被独立干预区分。继续`closed/contradicted`，不因来源澄清重开三配置，也不把同一批结果重复算成新证据；本认识反馈到[训练](../../../outlines/OUTLINE_PIG_TRAINING.md)、[适配损伤](../../../outlines/OUTLINE_PIG_TRANSFER_GAP.md)与[猪体知识](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)。

## 回到纲要

停止范围与保留问题已写入[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，开放联系见[探索图](../../../outlines/OUTLINE_PIG_HYPOTHESES.md)。本记录不降低原Goal，也不恢复原配置。
