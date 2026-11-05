---
schema_version: 1
id: IDEA_PIG_RLPL_V2
kind: idea
title: 合成共视原型对应的原生Mask适配（RLPL V2，已停止）
summary: 原型对应优于连续对应，却低于共同曝光无辅助控制及保留N，损伤超限；原配置关闭，迁移与适配机制仍待区分。
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

## 本版主张与比较

用合成同实例、同表面且双方可见的局部对应约束训练期原型分配，部署回到原生Mask模型，检验能否取得满足保护的真实收益。共同曝光无辅助B2、连续对应C2、离散原型对应D2作匹配比较；本条为已执行实验的事后Idea索引。

## 当前判断与理由

三配置结束并停止投入，保留N。D2优于C2，但低于B2及N且损伤超限，未满足本版净效用主张。局部相对优势与关闭理由均由[完整报告](../../experiments/spm_rlpl_pipeline_20261101/FINAL_REPORT.md)支撑。

[后续复盘](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#5-rlpl原型优于连续对应但低于共同曝光无辅助对照)明确B2也接触合成RGB及Mask路径BN，N则为多阶段项目保留权重。D2相对B2的比较仍是方法判断的重要依据。BN在本版中的作用未被独立干预区分，后续固定BN仍退化的事实也限制了统一归因。

## 保留问题

哪些对应信息能够迁移并被实例任务使用、普通适配怎样造成损伤，仍待区分。新的[历史观察](IDEA_PIG_OBSERVATION_INFORMATION.md)或[迁移保持](IDEA_PIG_COVERAGE_TRANSFER.md)可以提供不同的信息来源与解释；回到原型对应时，应有足以改变旧判断的新前提或机制证据。

当前行动见[当前安排](../ACTIVE_PLAN.md)，广义判断见[猪体知识Idea](IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_RLPL_V2.md)保留原数值与复盘演变。
