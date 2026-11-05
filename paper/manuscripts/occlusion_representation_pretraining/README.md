# 视频遮挡表征预训练

本线研究物理可控视频监督能否让表征保持目标内容、响应可见证据变化，并改善遮挡后的对应与实际实例关联。主张见[可迁移遮挡表征Idea](../../../project/ideas/IDEA_TRANSFERABLE_OCCLUSION_REPRESENTATION.md)，方法背景见[遮挡与视频身份Skill](../../../project/skills/occlusion-video-identity/SKILL.md)。

## 当前判断

已有四臂v0.2试验完成训练与评价，[2026-10-22验收报告](../../../experiments/occlusion_representation_pretraining_20261019/ACCEPTANCE_REVIEW_VOR_20261022.md)确认当前物理配对相对同预算普通对应训练未取得预定实用增量，按原出口结案。原入口停留在E0与提交阶段的状态已由该报告及[Study](../../../experiments/occlusion_representation_pretraining_20261019/study.json)取代。

当前帧对照的重现恢复较高，同时错误接受更多。历史信息的作用仍需结合覆盖和接受策略理解，原报告提出的等覆盖分析保留为未执行建议。已有结果限定单seed、合成留出资产和末点；真实视频迁移及成熟跟踪器比较仍未测。

## 设计与证据入口

- [视频研究方案](../../../experiments/occlusion_representation_pretraining_20261019/PROTOCOL_VIDEO_V02_DRAFT.md)：问题、竞争解释与贡献边界。
- [数据合同](../../../experiments/occlusion_representation_pretraining_20261019/DATA_CONTRACT_VIDEO_V02_DRAFT.md)、[模型与损失](../../../experiments/occlusion_representation_pretraining_20261019/MODEL_AND_LOSS_VIDEO_V02_DRAFT.md)：监督、输入及候选机制。
- [四臂执行协议](../../../experiments/occlusion_representation_pretraining_20261019/EXECUTION_E1_V02.md)与上述验收报告：实际比较、预算、结果与局限。

本线承接[密集视频身份研究](../dense_video_identity/README.md)的部分问题，保留独立Study和原始证据。当前行动统一见[当前安排](../../../project/ACTIVE_PLAN.md)，执行信息沿Study核对。

[整理前完整入口](../../../project/research_notes/restart/entrypoints_20261105/LINE_occlusion_representation_pretraining.md)保留历史设计和当时状态。稳定身份为`LINE_01701394050a46a3835bc63fb0d95f86`。
