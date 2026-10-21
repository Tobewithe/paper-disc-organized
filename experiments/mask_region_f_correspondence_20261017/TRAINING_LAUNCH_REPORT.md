# 区域 F 空间对应匹配对照：真实核验与训练启动

2026-10-17 12:51:25（UTC+8）状态快照。原生集成、真实smoke及原2000图输入导出均已通过，A臂正式训练已实际完成75/1500步、看过300图；Q臂排在A之后，尚未开始。清理六个旧传输包释放26.12GiB，原Run、缓存和checkpoint全部保留。

| 维度 | 本次完整实验的研究链路与当前状态 |
|---|---|
| **问题** | 固定P已有可修系数机会，现有G只兑现一部分。静态代码审查确认旧修正器的F来自检测raw位置的单点，区域P/z0有8×8布局；需要检验加入与它对应的区域F能否提升修正方向，而不是继续在旧输入上换loss。 |
| **机制假说** | 有序区域F对P/z0提供有效的空间条件，因而保留布局应胜过同特征数值的打乱组。竞争解释是额外区域统计或修正幅度足以解释变化，布局本身没有增量；原监督到最终掩码效用的偏差也可能令两组都失败。原P/z0布局对照的实际增量支持检验此方向，不能据此提前认定区域F有效。 |
| **针对性方法** | A保留区域F布局，Q仅置换位置；保留原单点F、P/c0与全部检测身份。共享原投影和卷积，均167872参数/17张量，沿用原初始化、2000图、soft-IoU与保护项，各3轮/1500步；固定last。后续按原协议比较N、已有G、A/Q及同checkpoint置换/等功能幅度控制。 |
| **整体结果** | 当前是工程核验通过、正式训练进行中。真实smoke的256张上下文/29101条输出和完整2000图/218209条训练输出均精确重放；两臂各2步smoke的17张量梯度通过。A已训练75步，Q待执行。**相对原生N和已有G的效用、损伤与机制增量均尚未评价，不能报告新收益或宣布科研门槛通过。** |
| **研究决策** | 优先完成已经有来源依据、能区分布局与额外特征解释的这一次匹配对照。唯一下一步是让当前有限序列完成A→Q各1500步、固定最终checkpoint并做训练结束配置审计。任一工程检查失败即保留错误并停止序列；不改科学门槛或追加预算。两臂训练已授权并启动执行序列；S2科学评价、S3/S4均未启动。 |

| 执行环节 | Run与状态 |
|---|---|
| CPU组件与集成测试 | `RUN_RFC_ENGINEERING_5df98dcee62b4cea8ad51c701abbba7e`，29项通过，旧工程Run保留 |
| 笔记本冻结环境/来源预检 | `RUN_RFC_PREFLIGHT_7aaf3a57b5b4448e955bc7de51a0473f`，完成 |
| 原批次真实输入smoke | `RUN_RFC_INPUT_SMOKE_192922660ab94ad1a8e342aa68f71572`，完成 |
| A / Q梯度smoke | `RUN_RFC_SMOKE_A_133d4a85e54a4edb9fb332561b48a421` / `RUN_RFC_SMOKE_Q_bd05243a99704dedbf5b4bf63bfc674d`，各2步完成 |
| 完整训练区域F | `RUN_RFC_INPUT_FULL_5959628271bc4cfeb792e9b5905d81ee`，2000图/218209行完成 |
| A正式训练 | `RUN_RFC_TRAIN_A_59b094a35f9b4fc3a45f7a86862ec677`，运行中，快照75/1500步，PID22976 |
| Q正式训练 | 单次序列中的下一臂，须A完成及其审计通过；尚无Run |

序列PID18168，笔记本工作根为 `D:/coco_wire/baseline_native_20261013`。本次只启动隐藏的有限进程序列，没有创建定时任务或goal。完成两臂后程序写 `TRAINING_PAIR_FREEZE.json` 并退出；不会自动开始科学评价、换checkpoint、追加训练或提交推送。报告数字为上述时刻的快照，后续进度以笔记本真实进程、`EXECUTION_STATUS.json`及Run为准。

[执行协议](EXECUTION_PROTOCOL.md) · [配置复核](EXECUTION_CONFIGURATION_REVIEW.md) · [正式训练启动核验](preflight/FORMAL_TRAINING_START_VERIFICATION.json) · [smoke独立对账](preflight/INDEPENDENT_SMOKE_RECEIPT_CHECK.json) · [完整输入独立对账](preflight/INDEPENDENT_FULL_INPUT_RECEIPT_CHECK.json) · [296项回传核验与运行中快照](preflight/TRANSFER_VERIFICATION_04.json) · [清理记录](STORAGE_CLEANUP.md)。区域F大文件留在笔记本，以Run清单/SHA关联；运行中的A仅回传阶段快照，不能冒充训练结案。
