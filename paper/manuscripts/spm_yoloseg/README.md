# SPM-YOLO：猪体知识与实例分割

本线探索怎样获取并利用对象、部位和空间内容，改善接触与遮挡中的实例归属。它是[真实猪场实例分割中心命题](../../../project/outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)的一条探索方向，核心主张由[猪体知识Idea](../../../project/ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)维护；原型记忆、概念监督和读出修正是已探索的实现途径。

## 当前研究判断

保留原V23作为成熟参照，原相对N至少提高5 AP的目标尚未达到。已有功能消融支持内容重建、细尺度、外观匹配与选择保护的作用；正确语义相对非语义控制的独立增量仍不稳定，内容总量与正确对应的分工也未解决。广义猪体知识判断继续未知。

最近夜间实验使三个版本分别收束：同COCO起点语义联合训练和C21局部保持结束；V23共享保留图获得功能简化支持，作为实际结构压缩的优先候选。实际计算收益待测，继续投入取决于成本与精度证据。各版本的理由和复判条件在对应Idea维护，完整数值与收益损伤见[夜间报告](../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)。

## Idea与证据入口

| 研究内容 | 主张与当前取舍 | 实验依据 |
|---|---|---|
| 组件的实际功能 | [C21／V23消融](../../../project/ideas/IDEA_PIG_C21_V23_ABLATION_V1.md) | [消融报告](../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md) |
| 从共同起点获得语义增量 | [COCO联合学习](../../../project/ideas/IDEA_PIG_COCO_JOINT_V1.md) | 夜间报告A组 |
| 保持性能并简化读出 | [V23功能简化](../../../project/ideas/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md) | 夜间报告B组 |
| 局部可塑性与输出保持 | [C21细尺度保持](../../../project/ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md) | 夜间报告C组 |
| 实例接口与训练分配 | [分配—部位—Mask](../../../project/ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md) | [所属Study](../../../experiments/spm_concept_assignment_bridge_20261102/study.json)，评价暂停 |

旧[固定池质量头](../../../project/ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)、[RLPL](../../../project/ideas/IDEA_PIG_RLPL_V2.md)和[原生部位V1](../../../project/ideas/IDEA_PIG_NATIVE_PART_V1.md)保留各自关闭范围；后继解释应继承这些证据约束。

## 接续与追溯

当前行动及暂停事项统一见[当前安排](../../../project/ACTIVE_PLAN.md)，预算与停止条件见[报告中的后继设计](../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)，实际执行状态见[夜间Study](../../../experiments/pig_multi_idea_night_20261105/study.json)。test426继续封存，旧分配矩阵及LCF的暂停保持。

方法和经验从[知识入口](../../../project/research_notes/README.md)进入Skill。V1至V32的设计、执行与写作材料可从[整理前完整入口](../../../project/research_notes/restart/entrypoints_20261105/LINE_spm_yoloseg.md)追溯；其中旧“当前／下一步”按原时点解释。稳定身份为`LINE_SPM_YOLOSEG_20261021`，目录key沿用`spm_yoloseg`。
