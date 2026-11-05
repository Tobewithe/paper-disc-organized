---
schema_version: 1
id: REVIEW_PIG_LOCAL_RELOOK_SUPPORT_20261103
kind: review
title: 局部重看最小验证：原128TRAIN的合格失败仅15例，当前入口支持不足
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_DETAIL_EVIDENCE, OUTLINE_PIG_TASK, OUTLINE_PIG_EVALUATION]
idea_ids: [IDEA_PIG_LOCAL_RELOOK_V1, IDEA_PIG_ADAPTIVE_DETAIL]
verdict: insufficient_current_pool_method_not_tested
coverage: declared
author: AI依据真实服务器Slurm1160及独立复核1161
---

在原128 TRAIN／1460实例及原正常输出上，按冻结规则得到1456个同条框IoU≥0.75关联，其中只有15个MaskIoU处于0.50–0.75的失败，另外1441个MaskIoU≥0.75。失败不足预定32，因此停止这个样本池入口，**没有进行局部重看模型前向，不判定方法无效**。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| 问题 | 已见TRAIN的IoU50计数触顶，推荐检验同条框较准、Mask未达0.75的实例能否由局部重新编码修复；必须先确认固定池支持32失败与32成功。 |
| 机制假说 | 局部视觉证据可能有增量；也可能缺少真实细节或裁剪不适配。此次只检验病例资格，尚未检验这些机制。 |
| 针对性方法 | 依[原协议](PROTOCOL_V1.md)核全部128图、原160独立Mask及完整正常预测，按分数顺序固定同条检测框关联，计算该检测自身MaskIoU；不拼接另一检测的成功，不修改GT。 |
| 整体结果 | 1460原GT、2261正常输出、1456个框75关联，合格失败15、成功1441；低于32失败门槛，未建立缩小版正式样本。独立实现重新做全部关联及位运算MaskIoU，1456行一致，最大误差0。无方法相对原生／已有方法收益数据，因方法未测。 |
| 研究决策 | 具体入口已结束／支持不足；广义按需细节未知。没有降低阈值、扩到其他图或用15替32，0新模型前向／0优化。用户随后新增夜间训练授权，另立[四臂训练Study](../pig_detail_training_night_20261103/PROTOCOL_V1.md)，不改本Study的停止条件或事实。 |

Slurm1160与1161均COMPLETED／0:0，分配2秒与1秒、各2CPU，共6CPU-core秒、0GPU。首次提交因不存在cpu分区被拒，未创建Run；改为现有gpu分区申请纯CPU后成功。传输准备的引号错误未开始科学运行，也未覆盖数据。

来源、精简结果与SHA见[RESULT_SUMMARY.json](RESULT_SUMMARY.json)，服务器位置见[REMOTE_STORAGE.json](REMOTE_STORAGE.json)。完整代码、原始关联、逐例表、冻结源和Run仅保存在服务器，本地没有下载数据。相应Run为`RUN_PIG_LOCAL_RELOOK_SELECTION_20261103_a44d3fd00405`及`RUN_PIG_LOCAL_RELOOK_VERIFY_SELECTION_20261103_875ba7406284`。
