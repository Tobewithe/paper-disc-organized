# P3 辅助分类损失消融

状态：三组训练、六次独立评估和配对汇总均已完成（2026-09-15 13:06:48 +08:00）；结果与日志已回传，本轮完成科学评价。

## 完成后的判断

<!-- research-note: {"id":"FIND_p3_aux_class_ablation_seed0_20260915","kind":"finding","title":"去掉额外分类 BCE 恢复晚期局部性能，但部署增益仍有限","scope":{"dataset":"COCO train2017 fixed random 20000 / val2017 all 5000","seed":0,"epochs":3,"primary_checkpoint":"last","target_panel":"239 failures + 199 controls","version":"Ultralytics 8.4.100"},"coverage":"declared","limitations":["single seed","best checkpoint is supplementary","diagnostic target matching is not unique COCO recall","descriptive intervals without multiplicity correction"],"evidence":[{"path":"experiments/p3_aux_class_ablation_20260915/runs/RUN_3cc1a111bf2c4802bb86e93f1f2f445d/official_contrasts.csv","revision":"4528892fbb5c6cba99a5a301ae771df1f8c17e5e1008bcebaa6b431eef32946b","relation":"derived_from"},{"path":"experiments/p3_aux_class_ablation_20260915/runs/RUN_3cc1a111bf2c4802bb86e93f1f2f445d/targeted_contrasts.csv","revision":"faf518dc121cd0bb15d389254bf2c860533990fcbea3a9e188286a587e07c3a2","relation":"derived_from"},{"path":"experiments/p3_aux_class_ablation_20260915/runs/RUN_4848d7f735d74261877b04f41cfda828/failure_control_interactions.csv","revision":"002e92c5a63faddb00eff83553c082634f4fc5fb80fcd92118e0540dc3c8a5df","relation":"derived_from"}]} -->
这轮有用：它把“额外分类 BCE 可能伤害方法”推进到了实际训练对照。去掉该项后，第 3 轮受辅助监督的 one-to-many 分支保有对失败对象的 raw P3 几何改善，同时恢复原方法损失的小目标 AP；部署 one-to-one 的失败面板也优于原方法。但不能把这个结论扩大成“reg_only 已稳定超过官方基线”：总体 AP 近似持平，部署失败面板对官方基线的多数改善区间仍跨零，且第 1 轮 best 上不重复同样的下游收益。

这不是要求所有指标上涨。raw P3 的可重复改善值得保留为机制证据；但论文若承诺正常部署的失败修复，还要解决训练阶段和分支之间的收益传递。

### 预先规定的第 3 轮主结果

以下 AP/AR 按 0–100 显示；均为原始 COCO 标注的官方 COCOeval，不混入训练时转换标签的原生验证数值。

| 分支 | 指标 | 官方基线 | 原方法 | 去额外 BCE | 去 BCE−原方法 | 去 BCE−基线 |
|---|---|---:|---:|---:|---:|---:|
| one2many | Box AP | 50.379 | 49.668 | 50.181 | +0.514 | -0.198 |
| one2many | Mask AP | 42.214 | 41.641 | 42.037 | +0.396 | -0.177 |
| one2many | Mask AP50 | 65.155 | 64.262 | 64.783 | +0.521 | -0.372 |
| one2many | Mask AP75 | 45.748 | 45.031 | 45.564 | +0.533 | -0.184 |
| one2many | Mask AP small | 22.162 | 21.029 | 22.085 | +1.056 | -0.077 |
| one2many | Mask AR small | 39.544 | 38.362 | 39.154 | +0.792 | -0.390 |
| one2many | Mask AR75 | 62.469 | 62.024 | 62.256 | +0.232 | -0.213 |
| one2one | Box AP | 48.770 | 48.912 | 48.746 | -0.166 | -0.024 |
| one2one | Mask AP | 41.141 | 41.147 | 41.070 | -0.077 | -0.070 |
| one2one | Mask AP50 | 63.613 | 63.629 | 63.417 | -0.212 | -0.196 |
| one2one | Mask AP75 | 44.666 | 44.476 | 44.483 | +0.008 | -0.182 |
| one2one | Mask AP small | 21.084 | 21.075 | 21.168 | +0.093 | +0.085 |
| one2one | Mask AR small | 39.346 | 39.775 | 39.884 | +0.109 | +0.539 |
| one2one | Mask AR75 | 63.364 | 63.481 | 63.319 | -0.162 | -0.045 |

第 3 轮 one-to-many 导出预测数：基线 596,652，原方法 823,073，去 BCE 601,799（相同 conf=0.001）。这与 BCE 抬高额外位置置信度、影响候选排序的假设一致，但预测数并非假阳性数，不能据此直接宣称误报增加多少。总 AP 是全部 COCO 类别的汇总；高低分变化和局部几何变化需要分别解释。

### 239 个失败目标：第 3 轮

变化乘 100；CI 为 4,000 次按图片配对 bootstrap 的描述性 95% 区间。中心误差来自全部尺度最佳 raw 框，越低越好。最终掩码与框由 GT 关联同类最佳框，不能称为正式 COCO 唯一匹配召回。

| 分支 | 比较 | 指标 | 变化 | 95% CI |
|---|---|---|---:|---|
| one2many | reg_only minus baseline | raw_p3_best_box_iou | +2.015 | [+0.710, +3.375] |
| one2many | original minus baseline | raw_p3_best_box_iou | +1.309 | [-0.135, +2.781] |
| one2one | reg_only minus original | raw_center_error_norm | -1.935 | [-3.904, -0.155] |
| one2one | reg_only minus original | final_box_iou | +1.828 | [+0.187, +3.528] |
| one2one | reg_only minus original | mask_iou | +1.651 | [+0.267, +3.156] |
| one2one | reg_only minus original | prediction_purity | +1.951 | [+0.312, +3.748] |
| one2one | reg_only minus original | final_mask50 | +2.929 | [+0.418, +5.858] |
| one2one | reg_only minus baseline | mask_iou | +1.142 | [-0.188, +2.512] |
| one2one | reg_only minus baseline | target_coverage | +3.168 | [-0.012, +6.249] |
| one2one | reg_only minus baseline | same_neighbor_leak_pred | -2.034 | [-4.343, +0.104] |

reg_only 对基线的 one-to-many raw P3 Box IoU：失败组 +2.015 点，控制组 −0.521 点；失败减控制的净变化 **+2.536，CI [+0.809, +4.246]**。这是目标相关几何改善的支持证据。相应 one-to-one 的净变化 −0.407 点，CI 跨零，说明该几何优势尚未直接传到部署 raw 分支。

one-to-one 的面板 Mask50 成功数从基线/原方法的 18/239 变为 reg_only 的 25/239；Mask75 从 1/239 变为 4/239。新增 7 个和 3 个仅属于这个已固定的失败面板，不能换算为全 COCO mAP 的百分点，也不能因小样本倍数大而夸大。

### 第 1 轮 best 的补充证据

三组原生 Box/Mask AP 都在第 1 轮达到最高，因此本次 best 对比也发生在相同轮次。主结果仍固定为第 3 轮，不因这里更好看而改协议。

| 分支 | 指标 | 官方基线 | 原方法 | 去额外 BCE |
|---|---|---:|---:|---:|
| one2many | Mask AP | 42.976 | 43.023 | 42.417 |
| one2many | Mask AP small | 22.411 | 22.598 | 22.454 |
| one2many | Mask AR small | 38.782 | 39.820 | 39.825 |
| one2one | Mask AP | 42.379 | 42.435 | 42.372 |
| one2one | Mask AP small | 21.805 | 21.868 | 22.063 |
| one2one | Mask AR small | 40.082 | 40.028 | 40.072 |

第 1 轮原方法的 one-to-many 失败目标仍有明确收益：相对基线，Mask IoU +1.807 点，CI [+0.636, +3.008]；边界 F1 +4.027，CI [+2.093, +6.210]。这保留了原方法早期局部修复的正面证据。去 BCE 在该时点的 one-to-many Mask AP 比原方法低 0.606 点，所以不能声称额外 BCE 始终有害。

第 1 轮部署 one-to-one：reg_only 小目标 Mask AP 对基线 +0.258 点，总体 Mask AP −0.007 点；但失败面板 Mask IoU 对基线 −0.011 点，CI 跨零。第 3 轮看到的失败掩码改善没有在这个时点重复。

### 对问题定位的意义与下一步

1. **保留几何信号。** reg_only 的 one-to-many raw P3 改善在两个 checkpoint 都存在：第 1 轮 +2.219 点、第 3 轮 +2.015 点，各自面板区间排零。两轮来自同一次训练，不是独立复现。
2. **额外 BCE 存在阶段相关代价。** 删除 BCE 后，第 3 轮原方法的 one-to-many 小目标 AP 回升 1.056 点，部署失败掩码/纯度也改善；但早期原方法的局部下游收益更好。因此 BCE 是影响结果的一个因素，不能列为唯一根因。此前梯度方向冲突本身也不等于因果伤害。
3. **应优先分离阶段权重与归属问题。** 在已安装 8.4.100 中，官方 one-to-many 权重按三轮从 0.8→0.45→0.1 衰减，one-to-one 相应增强；本方法额外辅助权重固定 0.5，写在官方双分支加权总损失之外。于是辅助项相对于原生 one-to-many 项的标量系数从 0.625→1.111→5.0。这是代码层面的目标比例变化，不是实测梯度范数，也尚不能证明它导致后期退化。三组基线自身也随训练退化，不能全归责于辅助项。
4. **下一项建议单变量消融。** 在 reg_only 上只令辅助系数随官方 one-to-many 权重同步衰减，保持首轮有效权重不变（lambda_aux=0.5×w_o2m/0.8），对比当前固定 0.5；其余 gate、教师、数据和监督均固定。这样直接检验“后期相对辅助权重变大”的假设。之后再用归属对齐与等数量随机筛选对照分离位置冲突因素。不要同时改 gate、分支和权重而失去归因。这是下一步建议，本轮没有启动新训练。

当前可用于论文的主张：**输入干预引出的训练信号能够改善特定失败对象的原始定位；辅助分类约束和分支训练动态会影响这种收益能否保留到最终分割。** 当前不能承诺全 COCO AP 显著上涨，也不能将去 BCE 本身包装为已完成的强创新。
<!-- /research-note -->

### 记录与适用范围

- 原始汇总：[RUN_3cc1a111bf2c4802bb86e93f1f2f445d](runs/RUN_3cc1a111bf2c4802bb86e93f1f2f445d/COMPLETE.json)，全量官方指标、分类别表及失败面板差值均已回传。
- 本次复核：[RUN_4848d7f735d74261877b04f41cfda828](runs/RUN_4848d7f735d74261877b04f41cfda828/verification.json)，核对三组实际配置仅输出名不同、两种方法执行源码相同、24 行官方结果及 5,256 行面板记录齐全，并补充失败减控制的交互区间。
- [失败减控制统计](runs/RUN_4848d7f735d74261877b04f41cfda828/failure_control_interactions.csv)；[成功目标数量](runs/RUN_4848d7f735d74261877b04f41cfda828/target_success_counts.csv)；[当前版本损失调度源码](runs/RUN_4848d7f735d74261877b04f41cfda828/loss_schedule_source.txt)。
- 本地仅做结果分析，未重新预测或训练。所有训练/推理均在原远端同一环境执行；本地 NumPy/Torch 环境差异没有用于重新生成预测。单 seed、多个描述性指标区间未经多重比较校正，须在后续独立训练或新验证面板复核。
- 每轮 checkpoint 和无损压缩预测导出保留远端原 Run 目录；本地已回传日志、配置、源码快照、元数据及结果表。当前 GPU 空闲，没有追加队列。

## 问题与实验

已有诊断发现，原方法的额外 hard-positive BCE 在 89 个入选位置中，有 64 个与官方分类输出梯度方向相反；其中包括 44 个官方背景位置。这是值得检验的冲突信号，尚不能证明其造成 AP 损失。原始证据见 [诊断报告](../p3_branch_gate_diagnostic_20260915/REPORT.md)。

三组均从同一 COCO 预训练 YOLO26m-seg 权重开始，Ultralytics 8.4.100，沿用权重携带的训练配置。固定随机抽取 20,000 张 train2017 图片，训练 3 epoch，seed=0，batch=2，workers=8。

| 组别 | 官方损失 | 额外 P3 回归 | 额外正类 BCE |
|---|---|---|---|
| baseline | 保留 | 无 | 无 |
| original | 保留 | 原方法，权重 0.5 | 回归项内部权重 0.05 |
| reg_only | 保留 | 与 original 相同 | 0 |

主要对比是 reg_only−original；两者对 baseline 的差异用于判断收益与代价。目标选择、编辑视图、冻结教师、gate、回归形式及权重不变。教师在每次全新训练初始化时复制同一原始学生，不接续旧微调权重。当前启动器不提供 resume，避免无意更换教师。后续若恢复中断需另行正确恢复原教师。

随机子集只由 train2017 图片 ID 和固定随机种子 20260915 决定，不用验证集表现筛选，也不重新转换任何多边形标签。沿用已有官方格式数据及多多边形合并结果。subset 内执行原增强，因此 Mosaic/MixUp 的取图范围也是该固定子集；三组一致，这不等于全量训练。

## 评价与判断

- 主结果固定用第 3 轮 last.pt；best.pt 为补充，按各组相同的原生验证 fitness 选择，不择优混用。
- 所有组在完整 5,000 张 COCO val2017 上重新预测，分别显式执行 one-to-many 和 one-to-one。最终 COCOeval 使用原始标注、正确 COCO category ID、包含零预测图片，分割面积从 RLE 计算。
- 输出 Box/Mask AP、AP50、AP75、APS/M/L、AR、AR50/75 及分类别结果。主要关注部署 one-to-one 的小目标 Mask AP/AR；one-to-many 用于观察受训练分支与部署分支间的差异。
- 对冻结的 438 个目标（239 失败、199 控制）比较 raw P3 Box IoU、中心误差、最终框/掩码 IoU、目标覆盖、纯度、边界 F1、邻居/背景泄漏和 Mask75。该面板采用 GT 关联的同类最佳框，只是诊断，不等同于 COCO 唯一匹配召回。中心误差来自全部尺度最佳 raw 框，不误称 P3 中心误差。
- 面板差值使用按图片配对 bootstrap，4,000 次；单 seed 结果用于路线筛选，不声称跨训练种子的稳定性。AP 多个指标的探索性涨跌不等于显著性证据；不要求所有指标都提高。
- 若 reg_only 保住定位收益且在分类/召回或小目标分割上优于 original，再决定进入归属对齐或 one-to-one 直接监督。当前队列只执行这三组，不自动增加多 seed 或后续方法。

## 执行与产物

数据准备、一次短程 reg_only 冒烟、三次训练、六次独立 checkpoint 评估各自保留 Run ID。每轮 checkpoint 全部保存；队列失败即停止并保留日志。正式训练后自动完成两分支全量评估和配对汇总。执行端为 `/root/autodl-tmp/p3_aux_class_ablation_20260915`；本地 `runs/<Run ID>/` 回传同一执行记录，不创建重复 Run。

协议与固定 ID：`protocol.json`；执行状态：`queue.json`；每次运行：`runs/<Run ID>/run.json`。所有计时使用实际执行端时间。训练与评估的完成状态不代表假设得到支持。

## 启动记录

- Study：`STUDY_4e1c74a75b3b4988b7181511c45799da`。
- 数据准备完成：`RUN_a5374d46b43246029e867aae474b3a43`，固定 20,000 张 train2017、完整 5,000 张 val2017，未重做标签转换。
- 冒烟完成：`RUN_7857532ba1aa40a6952009222ba3be7e`，32 张训练图、16 张验证图、1 epoch，数值有限；30 个入选目标中 8 个触发辅助回归。该小样本只验证执行，不作为方法性能证据。
- 首次基线 `RUN_36f5517015704aa78ea93048648ca13e` 在训练数据迭代前因官方 AMP 检查模型的 GitHub 下载超时停滞，已终止该启动并保留失败记录。没有已训练 epoch 或 checkpoint 被丢弃。后续启动复用冒烟时下载的官方 `yolo26n.pt` 缓存，仍实际执行 AMP 数值检查，不改变训练模型 YOLO26m-seg 或 AMP 配置。
- 正式基线重新启动为 `RUN_8c0d01745a4843ada1dac9640d45d5cc`；原方法为 `RUN_0c943bc6c5bb433193ed0f37bbcd6f3c`，reg_only 为 `RUN_b80a0c0d93db465789d4d7ff5ba9c45e`。
- `deployment.zip` 是首次部署历史包。现行代码看 `scripts/`，实际各 Run 的执行代码看其 `snapshots/`；已执行快照不回写。

远端只读监控命令（Ctrl+C 仅关闭监控）：

```bash
/root/miniconda3/bin/python /root/autodl-tmp/p3_aux_class_ablation_20260915/scripts/monitor.py --study /root/autodl-tmp/p3_aux_class_ablation_20260915 --watch 15
```
