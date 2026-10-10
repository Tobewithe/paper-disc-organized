# 官方 TAL one-to-one 原生共享系数头冲突审计结果

停止“冲突感知实例条件系数路由”方法立项。`val_small` 的可恢复实例没有比不可恢复实例呈现更高的共享末层梯度冲突，且冲突不能稳定预测 oracle gap；预注册阳性门槛未达到，因此不启动 teacher、basis、residual、solver 或冲突损失训练。最终文件见 [正式 Run](runs/RUN_CONFLICT_AUDIT_S4_60) 和 [独立验证](runs/RUN_CONFLICT_VERIFY_S4)。

40 图面板先按协议执行，但 `val_small` 只有 49 个 recoverable 候选、18 张 recoverable 图，低于预注册的 50 候选/20 图有效性门槛。按协议从同一确定性清单扩到 60 图，没有新增数据或训练。最终 Run 使用 fit/dev/val 各 60 图、1,290 个候选和 4,120 个 pair（3,528 个同图同尺度 pair，592 个跨图状态匹配 pair）；val 有 68 recoverable/89 unrecoverable 候选，覆盖 30/26 张图。

最终 `val` 的 recoverable−unrecoverable 负余弦比例差为：候选口径同图 −0.0552，95% bootstrap CI [−0.1313, 0.0210]；image-macro 同图 −0.1717，CI [−0.3246, −0.0327]；候选口径跨图 −0.0867，CI [−0.2507, 0.0838]；image-macro 跨图 −0.1292，CI [−0.3496, 0.1102]。跨图冲突与 oracle gap 的候选级 Spearman 为 0.1171，95% CI [−0.0461, 0.2723]；image-macro 为 −0.0634，CI [−0.4074, 0.2990]。`dev` 候选级点估计为 0.0617，未形成稳定正向证据。

实现复核通过：笔记本使用 vendor Ultralytics 8.4.100、官方权重 SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，原生 one-to-one cv4 末层三组形状均为 `32×64×1×1`。4 图 smoke 覆盖每个 split 的 4 张图和三个 pyramid level；共享 W/b autograd、方向有限差分和原生头重构的最大误差为 `1.34e−6 < 3e−5`，固定 BCE 与既有 `objective_A` 最大差为 `3.17e−6`。独立验证脚本重算 67,251 项 identity、pair、摘要、bootstrap、Spearman 和判定检查，失败数为 0。

这项结论只针对冻结官方 cache 中 native one-to-one cv4 最后一层的局部 mask-BCE 实例梯度统计；它不宣称整个 cv4 分支或所有全局参数不存在冲突，也不把约一半负余弦解释为因果证明。若以后按同一冻结协议独立重跑的 val panel 同时使四个主要 CI 下界都大于 0、val 冲突−oracle-gap CI 下界大于 0 且 dev 点估计保持正号，推翻本结论并重新立项；当前后备动作是整理“Recoverable but Unrealized”机制论文。
