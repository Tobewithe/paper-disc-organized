# 实际训练 TAL 稀疏事件记录：工程原型（2026-10-12）

状态：工程草稿，尚未真实训练，未完成 hook parity smoke，不批准大型训练或从头 COCO 预训练。

主科研决定：[算力受限研究路线](../../project/audits/candidate_eval_20261012/COMPUTE_CONSTRAINED_TAL_DECISION.md)。
详细数据合同：[动态监督图记录协议](../../project/audits/candidate_eval_20261012/TAL_TRAINING_TRACE_PROTOCOL.md)。

## 为什么暂停

固定官方 COCO 预训练权重可研究当前 TAL 与 Mask 质量的失配，不需要重建最初完整训练过程。

在 Ultralytics 8.4.100 中，TAL 依赖分类预测、框、anchor 与增强后 GT，不直接使用 Mask 系数或 Prototype。若仅微调 Mask coefficient 分支并固定分类/框/上游特征及 BN，那么同一输入的 TAL 正样本不随这项微调改变；没有必要每个 step 重复存储这些关系。

## 已提供的代码及其限制

scripts/tal_trace.py 为真实 criterion 读取正样本事件的钩子草稿。尚未连接实际 SegmentationTrainer 的 lazy criterion 生命周期；尚未通过 AMP、DDP、增强 GT lineage、loss/梯度/optimizer-step 等价及运行开销验证。**不得声称已经捕获训练轨迹。**

## 将来启用前的验收

1. 在真实训练 criterion 的 one-to-one 与 one-to-many assigner 上挂只读 hook，不能重新实例化 criterion 来模拟。
2. 同样输入、同样随机态，无/有 hook 的 TAL 正样本、Loss、梯度和优化器更新数值等价；若失败，停止。
3. 增强后 GT 的来源需贯穿 Mosaic/MixUp/裁切到 batch；无法可靠追溯时只记录增强后本地身份，跨 step 原 GT 轨迹标记 unknown。
4. 每次 microbatch 只记录最终稀疏正样本边、GT 总数、各尺度 raw 数和来源状态，不为每个负样本编造 GT owner。
5. 必须先明确一个需要真实训练动态而无法用现成静态缓存回答的科学命题，以及固定预算和停止条件。

**当前唯一动作**：复用官方固定权重进行静态 GT–TAL × 正常输出关系审计；不得因为现有代码存在而启动训练。
