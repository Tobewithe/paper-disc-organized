# S057：同权重原生one-to-one推理路径对照

2026-09-13，本地RTX5060Ti，完整COCO val2017的5,000图、36,335普通GT。运行330.515秒，不训练、不调整分数、不搜索超参数。

## 1. 对照结果

| 指标（0–100） | 原one-to-many+NMS | 原生one-to-one | 变化（点） |
|---|---:|---:|---:|
| Mask AP | 43.698 | 43.686 | -0.012 |
| Mask AP50 | 66.669 | 66.557 | -0.112 |
| Mask AP75 | 47.082 | 47.216 | +0.134 |
| Box AP | 52.554 | 52.295 | -0.260 |
| 全部GT Mask R75 | 59.254 | 62.692 | +3.437 |
| 低E(4) Mask R75 | 60.740 | 64.382 | +3.642 |
| 中E(4) Mask R75 | 67.054 | 70.273 | +3.220 |
| 高E(4) Mask R75 | 52.132 | 55.231 | +3.100 |
| 低−高R75差距 | 8.609 | 9.151 | +0.542 |
| 全部GT R75@全局P>=90% | 24.582 | 23.358 | -1.225 |
| 高E(4) R75@全局P>=90% | 13.096 | 12.684 | -0.413 |
| 最终预测数 | 446,097 | 555,714 | +109,617 |

高组R75变化的图片簇bootstrap95%CI为[2.543,3.670]点，差距变化CI为[-0.152,1.160]点。后者没有确认差距缩小或扩大。AP未计算区间；点估计接近不等于统计等价。P90为每臂在同一个全局分数阈值下取得的描述性工作点，阈值不同，不是各密度组分别挑阈值。

**切换原生one-to-one路径没有消除高拥挤缺口。** 它提高全分数候选池中的R75，但整体AP几乎不变，P90下的召回点估计下降。因此不能仅凭R75宣布切换路径已经解决了任务问题。

## 2. 如何保证比较对象明确

两臂权重SHA256均为`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。使用相同本地torch2.9.1+cu128、隔离Ultralytics8.4.143源码、640输入、rect=False、conf=.001、max_det=300、FP32、禁TF32、retina_masks=False和原框正常裁切。每幅JPEG与S036回执哈希相同。

本地该资产加载后的`head.end2end`实际为False；本轮显式设置True，逐图验证backend/head确实启用one-to-one。该事实不能解释为用户配置错误，也不能未经核实声称是所有官方资产/版本的默认行为。

本轮采用官方SegmentationPredictor，其输出二值mask通过与S036相同的scale_masks及>.5恢复到原图，再由pycocotools官方评价。全部GT保留，空图不删除，E(4)/ICI沿用原GT-only字段。

两条路径更换了预测分支及候选保留方式，候选、框、分数和系数都会变化；这不是固定候选下的单独NMS开关因果实验。新结果不能与S036的冻结one-to-many系数头直接组合成同口径方法提升。

## 3. 对主线的影响

1. 将原生one-to-one列为后续方法必须面对的基线；原one-to-many机制诊断仍保留，但论文明确分支口径。
2. 保留S056揭示的“同类错误修复对高低差距更有针对性”作为候选研究对象。S056的GT修复数量和AP收益只对其原路径成立，不能直接搬到本轮路径。
3. 方法门槛仍是正常Mask AP、全局同Precision下的高拥挤Recall及高低差距共同改善。继续增加低分候选或强制像素互斥都不能替代这一门槛。

目前没有完成新方法训练或验证创新性；本轮确认研究对象不因切换原生路径而自动消失。

## 4. 产物

- [任务对照表](../../experiments/coco_clean_20260911/diagnostics/native_one2one_fullval_20260913/comparison.csv)
- [配对R75统计](../../experiments/coco_clean_20260911/diagnostics/native_one2one_fullval_20260913/paired_stats.json)
- [Box AP](../../experiments/coco_clean_20260911/diagnostics/native_one2one_fullval_20260913/box_ap.json)
- [逐图预测及协议目录](../../experiments/coco_clean_20260911/diagnostics/native_one2one_fullval_20260913)
- [执行脚本](../../experiments/coco_clean_20260911/eval_native_one2one.py)

全部推理与评价进程已完成；没有修改训练队列或定时任务。
