# 官方 TAL 同域：原生末层共享可达性与掩码效用审计

本协议执行用户 2026-09-30 给定的审计指示。只检查固定官方 one-to-one TAL 对象，原几何阶段保留为历史证据，不参与本轮绑定、求解或分母。

## 目标与范围

固定候选身份、h、完整 prototype、预测框、类别分数、GT 标签处理，只求解三尺度原生系数末层。禁止添加新头、MLP、gate、阈值校准、边界监督、输入特征，禁止解冻上游、重新 assignment、按结果删候选。完成本报告即停止；不开展完整 COCO AP。

官方权重 YOLO26m-seg；Ultralytics 8.4.100；由权重 train_args 读取 mask_ratio=1、overlap_mask=True、分割增益 box=9.83241。远端 GPU 56018 执行；本地只编辑、回传和记录。

## 固定数据身份

| 分片 | 本轮冻结清单图片 | 官方候选 |
|---|---:|---:|
| fit | 796 | 6058 |
| dev | 197 | 1402 |
| val | 196 | 1346 |

清单来自已有官方有效图片列表，不补样凑数。更早原始计划中未进入这份列表的图片不反推、不补造。本轮列表内部无正样本图片仍须保留并单独列出。

YOLODataset 使用官方转换、排重、overlap-mask 排序；one-to-one TAL 由原冻结权重产生，保存 split/image_id/annotation_id/branch/raw_id/pyramid_level/target_gt_idx。target_boxes、owners 是正样本紧凑顺序，不能用 full-image gt_index 索引该子集。原系数、h、预测框都用同一 raw_id。任何输出 top-k/置信度统计不用于入组。

## 参考损失与 Gate A

最终权威参考为实际导入的官方 v8SegmentationLoss.single_mask_loss / calculate_segmentation_loss。GT 来自官方 640×640 overlap raster；完整 32×160×160 prototype 以 bilinear、align_corners=False 插值到 640×640。官方 GT 框支持为坐标 >= 下界、< 上界；crop_mask 在 8.4.100 原地修改输入，所以每实例必须新建支持画布。

官方 pixel BCE 在完整 640×640 网格上 mean，再除以归一化 GT 框面积，乘分割增益 9.83241。移除零权重框外像素后，等价为框内 BCE sum/GT 框像素面积，再乘该增益。本轮最终汇总为等候选均值，不按类别分数、图片或尺度改求解权重。

原官方缓存已有直接 criterion.loss 与 ROI 的扰动审计。本轮额外在各分片 6 张固定图、原系数与固定随机扰动处检查官方值、ROI 值和系数梯度，以及解析双精度梯度。atol=rtol=3e-5；不因失败放宽。Gate A 不通过则不正式求解。

旧几何阶段 BCE 与本轮不混称：旧回放 diag_bce 使用 160 网格、原始 COCO GT 最近邻降采样和预测框支持，且无官方分割增益；它与阶段 A 缓存目标不是同一实现。7D 旧数值不用于本轮比例。

## 相同诊断目标

\[
F_i(\delta)=L_i^{ref}(c_{0,i}+\delta)+\frac{0.003}{2}\|\delta\|^2,
\qquad J=N^{-1}\sum_i F_i.
\]

A: delta=0；B: delta=[标准化 h;1]A_l，仅 fit 求解；C: 每候选独立有限 GT oracle。正则只罚系数输出增量，不是 W,b 的 weight decay。特征均值/std 只来自 fit。理论最优有 J_ind* ≤ J_shared* ≤ J0；实际只能报告当前数值解及剩余差距，不把剩余比例说成不可达上限。

## 求解记录

确定性双精度 L-BFGS，零初始化与固定随机初始化比较。精确解析梯度与官方梯度先验证；固定 prototype 像素保存在 GPU、逐候选累积，不抽样、不量化。记录目标/BCE/正则、候选尺度权重、退出原因、梯度定义及初始化差异。迭代上限不是最优证书。

已经构造的 B 参数在正常掩码评价前冻结；不根据 dev/val 调 B、不调 lambda、不关闭尺度。独立 C 在三个分片生成，只作 GT 辅助诊断；逐实例状态全部保留。

## Gate B 与正常原图评价

U=A_l[:-1], v=A_l[-1]；delta_W=(U/std).T；delta_b=v-(mean/std)@U。根据 checkpoint 实际 conv 形状合并到 one2one_cv4 的原生 1×1 卷积，原 checkpoint 不覆盖。保存单独完整合并副本及 SHA256。

直接 c0+delta 与合并卷积输出逐候选核对（atol=1e-4、rtol=1e-5），原系数重建 atol=rtol=1e-5。完整 prototype 解码 process_mask(upsample=True)：组合 logits→双线性上采样到640→预测框裁剪→0 logit 二值化；scale_masks 根据缓存真实 ratio_pad 去 letterbox、默认双线性还原到原图→0.5 二值化。记录两条 B 路径阈值附近像素差异。缓存既有 val baseline IoU 逐候选复现 atol=1e-6。

IoU/coverage 使用原始 COCO annToMask；AUC/FPR 使用完整连续 logit 上采样到640，在固定预测框支持、最近邻放缩后的原始 COCO GT上计算；AUC ties 取平均秩。FPR 定义为该支持内 GT-negative 像素中 logit>0 的比例。标签/支持在 A/B/C 间固定；AUC 单类支持未定义单独计数，不能删除 IoU 候选。

参考损失表使用官方 overlap 标签、GT 框支持和官方增益，不拿上述原始 COCO 支持指标替代官方目标。

## 汇总与结束

主指标 val 图片 macro 原图 Mask IoU B−A；候选平均为次指标。原成功按 A 的 IoU≥0.75 定义。P3/P4/P5 是候选层级，不等同 COCO 实际大小。固定5000次整图配对bootstrap，seed=20260930；候选均值也重采样整图。报告 fit/dev/val，全 val 各层及原成功/失败分层；逐候选、逐图片结果保存。

这批图已被前序研究查看，只称官方同域定位性评价；不能称新盲测或 COCO AP。若数值解损失降低但 dev/val 掩码下降，结论为当前拟合收益未稳定迁移，不能归咎原型、证明全 YOLO 无信息或宣称官方优化器失误。

完成报告后停止。
