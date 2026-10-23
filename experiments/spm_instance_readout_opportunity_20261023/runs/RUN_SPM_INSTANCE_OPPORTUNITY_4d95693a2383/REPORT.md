# 实例均值原型稠密读出机会：完整实验

本实验冻结已有权重，在432张真实开发图上完成无GT读出和GT辅助机会核验，训练0步。原生Mask AP为57.3153，训练原型的无GT读出为25.2765；本实验不将GT辅助结果计作部署收益。

| 维度 | 本次实验的整体研究链路 |
|---|---|
| **问题** | V4.1全量训练的原型对同预算B仅+0.0417pp，关闭原型仅损失0.0565pp，小目标低于B。要区分实例信息没有被全局字典正确读出，还是现有投影在真实图上没有学出可用均值原型。 |
| **机制假说** | H_readout预测现有T嵌入经每实例GT支持能改善固定好框差Mask；H_target预测T在真实稠密图上仍无有效机会，随机/原始表征及实际采样覆盖可定位训练与压缩缺口。 |
| **针对性方法** | 同B官方Box75最终检测匹配与同一Mask，T训练128D/U同结构随机128D/R原始512D均采用单位均值、支持点leave-one-out与其他实例/bg竞争。无GT分支仅使用RGB和原生预测，先锁定全量输出，再读GT；同T循环错配实例、删除整个读出回到B。R维数不同有混杂，GT上界不作无GT AP。 |
| **整体结果** | 固定好框差Mask 305 条。T/U/R失败组均值IoU差分别为-0.0952/-0.2117/-0.1402；恢复Mask75比例0.66%/0.00%/0.00%。无GT T/U/R AP为25.2765/4.5954/13.4874，T对原生-32.0387pp，错配T为0.1323。真实train 8001/14642实例进入固定原型episode，详见分层覆盖。 |
| **研究决策** | 机器预冻结裁决：`single_mean_readout_no_practical_opportunity_in_frozen_coordinates`。我选定的正向重点为`direct_mask_supervised_query_and_basis_learning_with_matched_input_and_capacity`。本次结果决定后继原型训练方向，正式新训练尚未启动，必须另冻匹配容量/信息/步数及定向干预；+5目标仍未完成。 |

同一432图/2863可见GT、COCO百分制mAP50–95；差值单位为绝对pp。参照已有G/R正常AP57.2036/57.2395，新读出T分别差-31.9271/-31.9630pp，因后处理与信息形式不完全匹配，不作独立机制因果归因。GT-only上界不是最优oracle，不排除其他读出形式；负结果只关闭本冻结坐标/单均值/固定margin配置。

证据：[协议](../../PROTOCOL.md)、[运行](run.json)、[重放](REPLAY.json)、[指标](METRICS.json)、[机会分层](OPPORTUNITY_SUMMARY.json)、[逐对象](OBJECT_LEDGER.json)、[训练覆盖](TRAINING_INSTANCE_COVERAGE.json)、[裁决](DECISION.json)、[源码/输入冻结](SOURCE/FREEZE.json)。图集在figures/，实际检测身份在OUTPUT_IDENTITIES.json；原test未使用，无阈值或混合幅度搜索。
