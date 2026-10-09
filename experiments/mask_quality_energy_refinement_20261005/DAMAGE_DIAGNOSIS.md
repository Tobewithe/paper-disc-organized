# QCR 损伤原因诊断（2026-10-06）

**当前证据支持：固定大步把已有好 mask 推离局部可用区域，同时质量头监督的 soft 目标与正常原图 hard 评价不一致。** 第一项已由同方向、不同位移的固定诊断验证；第二项在很小位移也出现。它们可以重叠，不能按比例分摊全部损伤。排序损失另有明确的尺度冲突，但其独立致损贡献尚未隔离。

## 范围与复现

租用服务器已关闭；本次在用户授权的本地 RTX 5060 Ti 上执行，Python 3.10.19、Torch 2.9.1+cu128、Ultralytics 8.4.100、FP32、TF32 关闭。历史服务器为 Torch 2.8.0+cu128，不能宣称跨环境完整数值等价。

主 Run 固定取既有 DEV 清单前 64 图，包含全部 537 个官方 TAL 候选，没有按损伤结果选样。Quality epoch-3 checkpoint、原 YOLO、rho=12.085327、lambda=.003、原两步长度 6.042664/3.021332 均冻结；e 固定于 c0。模型参数及原模型 buffers 未改变。GT 只进入事后标签与梯度诊断，未输入质量头，未用于选步长或模型。

与历史 DEV 的 537 个候选身份完全配对；A/D1/D 的 Mask75 判定均零差异，IoU 最大绝对误差分别为 1.66e−5/2.01e−4/9.24e−5。独立前 2 图 smoke 的 19 个候选身份及 A/D1/D IoU 完全一致，只能认证这些样本。

## 主要证据

全量统计来自既有 FINAL 36,213 行和 DEV 13,690 行；以下损伤计数按候选，历史主 IoU 按图片 macro。局部曲线按 64 图的候选均值，两种口径不可混用。

| 证据范围 | 观察 | 说明 |
|---|---|---|
| FINAL 原始成功 22,084 个 | 第一步损伤 8,711；第二步恢复 5,312，又新增 496；最终损伤 3,895（17.637%） | 主要损伤从第一步发生，第二步回撤缓解但未消除 |
| 本地 537 个 | 同方向位移 .094417 时 Q 全部上升；实际 6.042664 时 318 个 Q 反而下降 | 固定大步跨过质量头自身的局部上升区间 |
| 本地 323 个原始成功 | 同方向 hard-IoU 平均增量：.094417 为 −0.0624 pp，1.510666 为 −2.0592 pp，3.021332 为 −5.6415 pp，6.042664 为 −11.0865 pp | 较大位移显著加重损伤；小步也没有建立平均收益 |
| 本地小步 .094417 | soft 上升 472 个，其中 201 个 hard 下降；成功组 soft 上升 288 个，其中 131 个 hard 下降 | surrogate 改善可以与正常解码退化同时发生 |
| FINAL 固定 held-out 扰动 | pairwise 53.026%、Spearman .0791；第一步 Q 上升的 14,727 个中，10,093 个真实 IoU 下降（68.534%） | Q 不能作为可靠的真实质量改善保证 |

下图对同一批 323 个成功候选展示固定方向的位移曲线。横轴为 symlog，连线仅连接已测点，不表示中间状态已测；所有点均为候选均值，没有据此选择步长。[可导出 PDF](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/runs/RUN_damage_curve_20261006_retry1/QCR_SUCCESS_DISPLACEMENT.pdf)。

![成功候选固定方向诊断曲线](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/runs/RUN_damage_curve_20261006_retry1/QCR_SUCCESS_DISPLACEMENT.png)

### 1. 更新过大，而且没有保持 c0 的选择

实际 Success 训练状态最远只到 rho/4=3.021332；推理第一步归一化梯度后几乎固定走两倍距离 6.042664。在 c0，距离惩罚梯度为零，因此第一步不因该惩罚缩短。[状态构造源码](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/scripts/qcr_train_fixed.py:89)与[更新源码](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/scripts/qcr_train_fixed.py:177)直接确认这一结构。

没有 Q/E 改善检查、接受或回退，也没有“保持原系数”的输出选择；第二步固定长度仅为第一步的一半，不能完整撤销第一步。全量 FINAL 的 D1 和 D 位移均超过 Success 训练最大半径。这里的错位特指成功分支，Failure 状态包含更大半径，不能说整个模型从未见过 6.04。

跨峰现象并不独属于最终损伤组：局部最终损伤组 42/57 与保持成功组 196/266 的第一步 Q 下降率都约 73.7%。它验证优化更新的问题，不能单独解释哪些实例会跌破 Mask75。

### 2. 训练目标与正常解码质量不一致

实际 Huber 和 rank 都使用 640 尺度、预测框内裁剪的官方 overlap GT soft-IoU；评价使用原图独立 COCO mask、正常二值解码。框外 GT、重叠处理、连续概率及二值阈值都会改变目标，[实际标签路径](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/scripts/qcr_train_fixed.py:139)没有使用正常原图 hard 排序标签。

局部 gQ 与该训练 soft 目标的梯度夹角为负者只有 58/537；小步 soft 上升者达 472/537。因此不能把损伤简单说成梯度普遍朝错方向：相当多更新沿 soft 改善方向，hard 却下降。全量最终损伤组固定预测框内 FPR 图片 macro 增加 25.005 pp，coverage 下降 3.094 pp，显示大量背景阳性伴随损伤；这是按结局分组的关联，不能据此量化 FPR 的独立因果效应，也不代表整图 FP 面积。

### 3. 排序约束存在尺度冲突，因果尚待隔离

[实际 rank](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/scripts/qcr_train_fixed.py:147)为 `relu(.05 − Δsoft·ΔQ)`，只保留 |Δsoft|>.01 的对。把状态对按真实质量从高到低定向后，零损失要求 ΔQ≥.05/Δsoft：真实差 .02 时要求预测差至少 2.5，真实差 .10 时要求至少 .5，与 Huber 回归 soft-IoU 的校准尺度冲突。即使 Q 完美等于 soft，|Δsoft|<sqrt(.05) 的有效对仍有正 rank 损失。这是已确认实现问题，也偏离原 sign ranking 规则；尚未做隔离训练，不能声称它造成了某个比例的损伤。

## 第二步为何恢复部分结果

局部 323 个成功实例先损伤 136 个，第二步恢复 81 个，最终损伤 57 个（55 个残留、2 个新增）。固定相同第二步长度的机制干预中，Q-only 恢复 77 个、penalty-only 恢复 51 个；这是诊断对照，不是新方法性能确认。

537 个候选的 gQ1/正则梯度范数比中位数为 3.681，只有 1 个正则范数更大；Q 梯度自身在 423 个候选上朝 c0，加入正则后的能量梯度在 517 个上朝 c0。恢复是质量地形回向与惩罚共同作用，不能称第二步全靠正则。

第二步也会跨峰。候选 image 448269 / annotation 1594519 / raw 6892 的正常 IoU 为 A .77746 → D1 .78012 → D .03595，Q 为 .72701 → .87322 → .52046，soft 为 .71835 → .75960 → .14192；同样长度的 penalty-only 第二步 IoU 为 .79880，Q-only 为 .08469。另一个候选 image 35514 / annotation 1054032 / raw 8287 从 A .94797、D1 .95149 跌到 D .54155，Q2 也下降。这是两个事后反例，说明固定第二步仍可能严重破坏，不能用于选择新方法或步长。

## 可复查记录与限制

已完成 [全量纯统计 Run](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/runs/RUN_damage_existing_rows_20261006_retry1/SUMMARY.json)、[smoke retry1](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/runs/RUN_damage_local_smoke_20261006_retry1/run.json)、[64 图主 Run](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/runs/RUN_damage_gradient_local_20261006/run.json)及[纯统计绘图 Run](C:/Dpan/codexproject/paper-disc-organized/experiments/mask_quality_energy_refinement_20261005/runs/RUN_damage_curve_20261006_retry1/run.json)。首个纯统计及绘图 Run 均退出码 1，精确原因未知；首个 smoke 子模型完成，但 launcher 后处理缺少 `metrics` 字段导致失败，修复后另建 retry1。失败及部分产物均保留。

独立复核已从逐候选行重算全部分组计数、均值与分位数，并核对主 Run 的输入、产物及 13 个快照 SHA，未发现错误。537 个 tiny Q 增量最小为 2.97e−5，小步增量最小为 9.33e−4，均明显高于 FP32 舍入幅度。

关键输入 SHA256 如下；JPEG 清单、标注哈希、代码快照与逐候选输出均在主 Run 内保存。

| 输入 | SHA256 |
|---|---|
| Quality checkpoint | `07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065` |
| 官方 YOLO 权重 | `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5` |
| 图片划分 | `a1a3ee0a0e39582dfc13a403faa91ddd63ed5ca5ab16017a0ff84af028765291` |
| DEV 历史逐候选行 | `a16932c640820156add318839593754b9e3043e90b9952da094bae88cf54279c` |
| FINAL 历史逐候选行 | `457bb2a5062d9a8919fd1106af4d76cb0c092b78c6ce9f50ced5a5a1d6ef4192` |

后续若开展新研究版本，应先恢复正常 hard 排序监督并核对 rank/calibration，再分别检验更新接受机制与成功实例的保持选择，避免同时修改后无法归因。本次没有启动训练、调参或 Stage II；已有 STOP 保持。64 图是固定小样本机制诊断，不替代全量终点，不证明某个修复能泛化，也没有隔离 FIT 量化、采样或 seed 偏差的贡献。
