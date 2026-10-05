# S075：预测交集支持的候选归属门控复验（2026-09-13）

## 问题与目的

S054 的 GT 候选归属诊断表明，错误像素往往落在同类邻居候选的响应区域内；S060 则使用了训练期 GT 独占区域、推理期候选支持区域，存在明显的支持分布错位。S075 复验一个更严格的配方：训练和推理都使用同一预测支持域，即同类候选预测框的交集；GT 只在训练时提供像素归属标签，推理不访问 GT。

目标是判断“候选间归属读出”能否在不改变主干和原型的情况下产生可部署的掩码修正。每个候选对采用平衡的 00/01/10/11 归属状态，门控输出为有界的反对称像素修正，随后使用官方掩码裁切和二值化流程评估。

## 协议

- 数据：COCO val 的冻结预测缓存；120 张拟合图、20 张独立 transfer 图。
- 候选：112 个同类候选对，28,672 个支持域像素；22,937 个拟合像素、5,735 个留出像素。
- 训练：3 个随机种子（0/1/2），15 个 epoch，每轮保存 checkpoint。
- 训练支持域：同类候选预测框交集；标签由 COCO GT 仅用于拟合阶段生成。
- 推理支持域：同类候选预测框交集；`GT_used_in_inference=false`。
- 评价：官方 COCO segm evaluator，保留原预测框、类别和分数，采用与基线一致的官方 `crop_mask` 解码。

## 结果

| 方案 | Mask AP | AP50 | AP75 | 预测数 | 改变二值像素数 |
|---|---:|---:|---:|---:|---:|
| Initial | 0.583003 | 0.792886 | 0.620112 | 1,804 | 0 |
| Gate seed 0 | 0.583003 | 0.792886 | 0.620112 | 1,804 | 0 |
| Gate seed 1 | 0.583003 | 0.792886 | 0.620112 | 1,804 | 0 |
| Gate seed 2 | 0.583003 | 0.792886 | 0.620112 | 1,804 | 0 |

三种子均执行 7,413,270 次像素更新。留出 BCE 在 15 个 epoch 后仅下降约 `1.06e-5`、`1.18e-5`、`1.06e-5`；活跃留出像素为 366、338、343。软修正没有使任何 transfer 预测跨过二值掩码阈值，因此 AP、AP50、AP75 均保持不变。

## 结论与边界

1. 在同一预测支持域内，候选归属门控仍不能转化为可见的实例分割改进；S060 的失败不只是训练/推理支持域不一致。
2. 该配方的训练信号极弱，且修正幅度不足以改变官方解码后的二值掩码。不能把 BCE 下降或软更新次数写成方法收益。
3. S054 的 GT 归属竞争增益仍是诊断上限，不能据此声称一个简单的候选门控即可自动恢复掩码。停止该具体公式，不再围绕它扩展超参数。
4. 下一步若继续方法探索，应转向 S074 暴露的“固定原型上可留出泛化的响应读出”问题：设计共享、可训练的系数残差预测器，并在独立 COCO 图像上直接测 AP；不得再次使用 GT 求解系数作为方法结果。

## 产物

- 实现：[joint_support_ownership_pilot.py](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/joint_support_ownership_pilot.py)
- 训练记录：[TRAINING.json](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260913_joint_support_ownership_120_20/training/TRAINING.json)
- 评价汇总：[SUMMARY.json](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260913_joint_support_ownership_120_20/evaluation/SUMMARY.json)
- checkpoint：`experiments/coco_clean_20260913_joint_support_ownership_120_20/training/checkpoints/`
