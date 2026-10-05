# 选择性归属门控 pilot（S060）

本轮尝试了一个受守恒约束的候选间归属修正：对同类候选共同支持区域使用反对称 logits 修正，并在推理阶段只允许正响应且 margin 较小的像素被修改。训练在 1,200 张 COCO train2017 冻结输出上进行，3 个随机种子，GT 仅用于训练标签；transfer 评估使用预先固定的 300 张图。

训练 holdout 的 pair BCE 从 0.5976 降至约 0.5764，但 20 张 transfer 冒烟图中三个种子均为 **0 个改变像素**，Mask AP、AP50、AP75 均与原模型完全一致。由此可见，当前 pair 训练目标没有形成可部署的二值归属修正；300 图版本因逐候选支持构造过慢而停止，未产生可报告的完整 COCO 结果。

这不是方法正结果。主要实现教训是：训练使用 GT 独占区域的有向样本，而推理使用候选共同支持区域，存在训练/推理分布错位；同时连续 logits 损失下降不足以保证阈值后的掩码发生改变。该具体配方停止，不继续扩展数据或调阈值。

产物：`experiments/coco_clean_20260911/antisymmetric_ownership.py`、`eval_antisymmetric_ownership.py`、`diagnostics/antisymmetric_ownership_train_20260913_v3`、`diagnostics/antisymmetric_ownership_eval_20_20260913_v3`。
