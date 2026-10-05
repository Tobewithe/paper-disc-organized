# Experiment 7V：冻结修正器的独立确认与标量阈值对照

## Goal

在不训练新网络、不改 loss、不新增 gate 和不扩大 prototype 输入的条件下，回答两个预注册问题：

1. 已冻结的 `big-global` 修正器相对官方原输出的 native mask IoU 小幅收益，能否在此前没有参与该修正器选择的独立 COCO val2017 图像上复现？
2. 该收益是否超过在同一固定支持区上只调整一个全局 mask-logit 阈值的简单校准基线？

这是固定候选、固定框和固定类别分数的 GT-positive 诊断，不是完整 COCOeval AP。

## Frozen methods

- **A official**：原始 one-to-one 候选的官方 mask logits，固定框/候选/分数，native letterbox 解码。
- **B scalar threshold**：A 的连续 logits 减去一个全数据集共用的标量 `b` 后再二值化；`b=0` 必须逐像素复现 A。`b` 只在既有 100 图 dev 集上按图片级 macro IoU 选择一次，然后冻结。
- **C h-only**：`7T_rep3/h.pt` 冻结预测器，仅用于次要对照。
- **D big-global**：`7U_bigglobal/big_global.pt` 冻结预测器，使用其已冻结 h 与 4×4 pooled prototype 输入；不在本轮改结构或重训。

所有方法复用同一个官方 one-to-one 正样本映射；不按 IoU 另选候选，不用 GT 选择预测输出。GT 只用于官方匹配映射与评价。

## Image set and independence

- 评价清单固定在 `independent_val_ids.json`，共 512 张 val2017 图像。
- 清单在读取本轮预测前生成并冻结；选择规则为 val2017 图像 ID 排序后，排除系数预测主线中已经登记的图像级列表与 reviewed val 清单，再取前 512 张。
- 排除来源和数量见 `EXCLUSION_SOURCES.json`、`EXCLUDED_VAL_IMAGE_IDS.json`。其他不属于本系数修正主线的历史研究未被当作修正器开发数据；若论文要求更严格的全项目隔离，应改用 train holdout 并在报告中标注其对 COCO 预训练权重的限制。

## Sample-size plan

7U 审计的图片级 macro IoU 区间为 [0.00043, 0.00530]（100 图）。由此估计配对差标准差约 0.01245；以双侧 α=0.05、功效 0.90、实用效应 0.002 IoU（0.2 pp）计算，正态近似需要约 408 张图。本轮预先冻结 512 张，超过该规划量。计算记录在 `SAMPLE_SIZE_PLAN.json`。

## Evaluation

主指标：图片级 macro native Mask IoU；以图片为重采样单位。主要比较为 D−A 与 D−B，双侧 α=0.05，并对两个主要比较做 Holm 校正。候选级平均为次指标，区间同样按图片重采样。

同时记录：Mask75 rate、repair/damage（相对 A 的实例阈值跨越）、GT coverage、pixel ROC AUC、background FPR。原失败/成功与目标大小只作预先声明的次要分层，不以单一分层替代总体主指标。方向 cosine不作为成功判据。

B 的候选阈值搜索网格在脚本中固定，包含 `b=0`；只使用既有 100 图 dev 的原图差值选择，独立评价图像不参与选择或追加。

## Decision rules

- D−A 与 D−B 均稳定为正，且达到预注册 0.002 IoU 实用量级：保留“冻结修正器在独立图像上有超出标量阈值的独立收益”的表述；再规划跨 seed/完整推理验证。
- D−A 为正但 D−B 未建立优势：只保留小幅收益，降低复杂修正器优先级，不宣称 prototype 条件独立贡献。
- D−A 不能复现且区间排除 0.002：结束本路线的确认，不沿同一结构继续加参数。
- 区间较宽：记为不确定，不追加临时模型搜索或图像。

## Reproducibility

远端运行只保留冻结模型、原始 forward 和评价产物；运行目录保留命令、来源和结果。不得将本轮固定候选结果改写成 COCO AP 或全候选部署收益。
