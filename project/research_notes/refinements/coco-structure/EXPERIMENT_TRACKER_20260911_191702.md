# COCO结构诊断执行表

日期：2026-09-11。没有重启正式CCL训练；本表的TODO不是已开始运行。

| ID | 实验 | 数据 | 状态 | 输出/门槛 |
|---|---|---|---|---|
| S000 | 运行时预测路径见证 | COCO val图1000 | DONE | one-to-many＋NMS，FP32重放coeff/box/proto均0误差 |
| S001 | 全候选索引追踪冒烟 | 预定32图 | TODO | raw→过滤→NMS→max_det→空mask，每阶段source index与原输出一致 |
| S002 | 全候选失败分解 | 300图，所有GT | TODO | 区分候选、类别/分数、抑制、框与掩码；不丢弃漏检 |
| S003 | 精简逐层空间可读出性 | 与S002相同图；探针设置用train开发 | TODO | 同类/异类/距离/尺度控制；own-vs-neighbor与own-vs-background分开 |
| S004 | 系数选择与原型可表达性 | S002固定实例和对照 | CONDITIONAL | GT oracle、随机方向、实际系数、同RMS方向；覆盖/背景共同报告 |
| S005 | 共同支持与外扩环归属 | S002固定图；COCO CCL权重未就绪 | CONDITIONAL | 不同模型不直接交换P与c；无干净CCL权重则只做官方模型诊断 |
| S006 | 一个对应机制的方法验证 | 预先冻结协议＋三种子 | NOT STARTED | 机制定位后另定方法与正式训练预算 |

最新计划见同目录EXPERIMENT_PLAN.md。旧全量AP、阈值和CCL结果仍在各自目录，不覆写。
