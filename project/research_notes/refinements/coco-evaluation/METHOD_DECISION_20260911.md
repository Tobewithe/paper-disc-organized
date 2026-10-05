# 从 COCO 固定解码器结果决定下一步方法

**后续取证更新（2026-09-11）：COCO训练标签存在多polygon实例拆分，详见TRAINING_LABEL_AUDIT_20260911.md。本文件的局部观察只适用于现有旧权重；先修复数据并重训原CCL/基线，再决定是否需要改方法，不把当前结果解释为干净监督下的根因。**

状态：2026-09-11。诊断完成，新的训练方法尚未验证。

## 1. 可以保留的研究问题

相邻实例共享 prototype 解码，系数方向是可干预的表示因素。但“让两输出不同”没有直接规定“每个输出对应哪个 GT 区域”。当前 COCO 结果把这两件事区分开：系数推开降低预测间原生重叠，GT邻居错误未明确下降，覆盖和背景激活同时增加，严格实例指标未获明确收益。

这个问题适合作为方法修订的动机，不能写成已经证明所有密集失败都源自系数相似，也不能写成联合 prototype 适应的必要性定理。200图、可匹配对象、固定 decoder 的局部结果不能替代训练后的全量任务评估。

## 2. 先核对原方法，再改方法

实际旧实现见 `gemini/remote_runs/export_light/scripts/train_coco_ccl.py`：对每个GT的分配候选取平均系数；两个GT框IoU>0.05时，惩罚 ReLU(cos−margin)，没有类别条件，也没有在额外正则中区分自身、邻居独占区域和背景。

因此“只在相邻GT上施加CCL”已经存在，不能再当作新改进。更具体的内部消融是同类别邻接门控；更有区别的替代方向是用训练GT约束输出的空间归属。两者不是同一个改动，应分别归因。

## 3. 最小可检验方法与对照

主对照保留三组：同初始化的原版Baseline、原CCL、空间归属正则。最终若保留两种约束组合，则另加组合消融，不跳过单独空间项。原有分割、检测、分类监督保持一致。

在训练图像中，对GT邻接对 i,j 定义有效像素域V（排除crowd和无效padding）；令 U 为全部非crowd GT并集。空间域S_i可先固定为GT框及20%外扩的并集支持，20%只是开发配置，不代表已验证最佳。必须在训练/开发划分决定外扩和权重，不能根据现有val AP反复调参。

- 自身 T_i = G_i ∩ V。
- 邻居 N_i = (同类邻接GT并集) ∩ S_i ∩ V ∖ G_i。
- 背景 B_i = S_i ∩ V ∖ U。
- 不同类别邻居不混入背景；由原有分割监督处理，或在独立预定消融中使用所有类别邻居。

对匹配候选的原生logit z_i=c_i^T P，开发候选额外项为区域均值 softplus(z_i) 在 N_i/B_i 上的加权和，配合原有GT分割损失保持 T_i 覆盖。若另加 softplus(−z_i) 自身项，必须单列权重并说明它改变了自身监督强度。空区域跳过该分量，按有效目标数归一化，记录各区域支持量与梯度范数。训练的GT掩码坐标、letterbox和低分辨率栅格必须一致；保留各实例二值掩码和crowd信息，不能用错误的多边形拆分或不可逆重叠ID合并替代。

下一步先做两种固定张量方向探针：邻居抑制方向，以及加入自身覆盖/背景保护的方向，用与原push相同的系数范数和扰动预算比较。该探针用GT构造方向，必须标为oracle诊断，不能当作可部署模型或AP结果。梯度零/区域空/投影不可行的情况保留，不剔除失败样本。

若GT知情的方向在冻结P下仍无法改善空间误差，先检查当前prototype表示能力、栅格分辨率和匹配；不能直接开展大范围lambda搜索。若能够改善，再把相同目标带入训练，检查能否从输入学习这种变化。

## 4. 评估与推进条件

训练/开发先用COCO train2017中的固定互斥划分；200图机制集及现有Dense1576结果已被查看，后续不称为独立确认测试。若全val5000中其余图像未用于训练选择，固定方案后可作为额外留出确认并明确选择记录；同时完整报告标准val5000性能。

训练时三组共享预训练权重哈希、样本顺序、增广、优化器、完整学习率日程、实际epoch数、seed与checkpoint选择方式。先seed0开发，最终配置再以0/1/2复现。图片bootstrap与训练seed方差分开。

首要任务指标沿用高ICI实例 R75、高低拥挤增益差D、同Precision Recall和同类PairRecall，保留官方Mask/Box AP50:95与AP50。GT诊断必须同时显示邻居错误、背景错误、覆盖和IoU；不能以缩小掩码换取低泄漏。门槛：开发集中GT邻居错误下降且覆盖/背景没有抵消，随后留出集中严格实例指标正向、多seed可重复。没有达到门槛就报告失败并调整假设，不更换主指标找正值。

## 5. 可用于修稿的英文段落

> To distinguish representation separation from spatial correctness, we performed norm-preserving bidirectional coefficient rotations on frozen COCO decoders. A GT-defined sample of 200 images yielded 170 eligible pairs for the baseline and 166 for CCL. With prototypes, boxes, scores, and assignments fixed, a 5-degree outward rotation reduced raw pairwise mask IoU by 4.54 and 3.56 percentage points, respectively. However, the corresponding changes in same-class neighbor error did not show a clear reduction. Among targets with ICI above 0.5, target coverage increased together with background error, while mean cropped mask IoU remained nearly unchanged. These controlled responses support a local effect of coefficient direction on raw overlap, but do not establish that coefficient decorrelation improves GT separation or mediates a training-time AP gain. They motivate spatially targeted regularization that distinguishes the intended instance, neighboring instances, and background.

段落证据来自 `gemini/results/coco_frozen_decoder/20260911_070048/INTERVENTION_REPORT.md`、`validation.json`与全部CSV。将来正文若加入新训练结果，应替换“motivate”后的方法状态，不能在当前版本提前写成提升已经实现。
