# 密集视频个体身份保持与遮挡恢复

本线研究密集、相似目标的跨帧身份与遮挡后恢复，猪和马等动物作为应用对象。核心问题由[时序证据Idea](../../../project/ideas/IDEA_TEMPORAL_IDENTITY_EVIDENCE.md)连接；已完成系统、实验与阴性结果继续作为前期证据和下游应用来源。

## 当前研究判断

既有确认、回退与收益学习在部分视频有局部作用，但当前配置未建立相对成熟跟踪器的稳定优势。可见性模型在合成数据中学会了状态预测，真实身份恢复增量有限，打乱权重对照未支持原机制；该配置已停止扩训。具体比较、事件分母与限制由[可见性迁移报告](../../../experiments/dense_video_occlusion_transfer_20261019/REPORT.md)承载。

后续核心表征问题已由独立的[视频遮挡表征预训练主线](../occlusion_representation_pretraining/README.md)组织。此关联保留旧实验身份及结论，不把旧运行重新记作新线结果。

## 关键实验与交付

| 内容 | 原始入口 |
|---|---|
| 检测分割跟踪原型 | [第一轮报告](../../../experiments/dense_video_identity_20261018/REPORT.md)与[交付说明](../../../experiments/dense_video_identity_20261018/DELIVERY.md) |
| 确认与回退的作用 | [第三轮报告](../../../experiments/dense_video_reversible_identity_20261019/REPORT.md) |
| 竞争证据与动作收益 | [第四轮报告](../../../experiments/dense_video_evidence_decision_20261019/REPORT.md) |
| 可控视频与几何标签 | [合成报告](../../../experiments/dense_video_synthetic_occlusion_20261019/REPORT.md) |
| 可见性学习与真实迁移 | 上述迁移报告及[Study](../../../experiments/dense_video_occlusion_transfer_20261019/study.json) |

当前行动见[当前安排](../../../project/ACTIVE_PLAN.md)，知识见[遮挡与视频身份Skill](../../../project/skills/occlusion-video-identity/SKILL.md)。全部历史轮次、协议与展示入口保留在[整理前全文](../../../project/research_notes/restart/entrypoints_20261105/LINE_dense_video_identity.md)。稳定身份为`LINE_6dd1154d9754466da9c6441369c41fb9`。
