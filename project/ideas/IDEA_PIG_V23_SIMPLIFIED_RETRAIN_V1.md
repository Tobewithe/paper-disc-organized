---
schema_version: 1
id: IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1
kind: idea
title: 对V23低依赖组件施加约束后重新学习候选与选择
summary: 功能约束重训支持优先保留共享保留图、形变置零作为次选；实际结构压缩与计算收益待测。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_MODEL, OUTLINE_PIG_TRAINING, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: supported
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: B组9配置、简化配对差、护栏、空Mask与实际耗时
    relation: references
---

## 本版主张与依据

[固定权重消融](IDEA_PIG_C21_V23_ABLATION_V1.md)发现形变和逐部位保留的影响较小。本版进一步检验：从共同V23起点施加功能约束并重新学习候选和选择，能否保持正常性能？比较完整重训、形变置零和九部位保留平均共享，主要条件见[协议](../../experiments/pig_multi_idea_night_20261105/PROTOCOL_V1.md)。

## 当前判断与理由

功能简化得到支持，本版验证结束。两种简化均通过筛选与护栏，共享版本三次配对方向一致，优先保留；形变置零作为次选，原V23继续作成熟参照。作用量及来源波动适合支持精简候选选择，完整结构继续重训以明显涨点的优先级降低。

当前均值／置零实现仍执行原分支，尚未兑现计算节省。完整控制的空Mask和损伤、各臂配对差与耗时见[报告B组](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#5-b组功能精简有依据共享保留图优先)。

## 后继判断依据

先辨明目标分支的实际成本占比，再决定是否删除计算并核对输出等价性或近似误差。压缩后的精度、空Mask、原成功损伤和正常预热时延共同决定采用；收益不足或关键护栏失败时停止该压缩对象，保留原V23。预算、目标与退出条件集中在[报告后继设计](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)。

当前行动见[当前安排](../ACTIVE_PLAN.md)；[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md)保留数值与判断演变。
