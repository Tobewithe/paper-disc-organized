# OGPS Gate 0：复用7O预测的零训练重放

2026-10-02，执行前锁定。仅在笔记本 `ssh 28358lan` 运算；不训练网络，不改变候选、不调阈值、不给下一阶段自动放行。

## 对象与资产

复用7O修复raw索引后有效val：196张图片、1,346个官方one-to-one TAL候选。7O有效构建RUN_07ed2b4d206841f2b0a7ee5cfe6ff2c5、预测RUN_fcbacf1a76ab4dcc91d451a92264f085；有限oracle使用direction_predictability RUN_783b955a16ac42369921f39803a26eee。按(image_id,annotation_id,raw_id)严格join，不取有利交集。完整原型与640标签从笔记本prepared缓存复用，并逐图记录hash；h、c0、pooled原logit、GT8标签及原图baseline须复现。

已有有限oracle为λ=.003、未乘segmentation gain的BCE位移正则目标；只是同身份恢复参照，不是本轮Ridge同目标最优解。评价图已参与历史研究，属于定位性复用，不是新盲测或COCO AP。

## 固定求解器

对预测框在原生160网格的32通道prototype裁剪（左上floor、右下ceil，截边且至少1格），adaptive average pooling到8×8，得A∈R^(64×32)。不标准化通道。

所有正式输入统一为概率q，t=logit(clamp(q,.01,.99))。7O保存的是logit，先sigmoid一次；GT soft occupancy不sigmoid。固定：

`min_delta ||A(c0+delta)-t||²/(2*64) + .003*||delta||²/2`

`delta = (A.T A/64 + .003 I)^(-1) A.T (t-A c0)/64`。

FP64 Cholesky，无优化更新、无sweep、无jitter；正常解码显式转回FP32。lambda与oracle相同不意味着目标相同。GT8输入不是正常IoU的严格上界；阴性只决定这个固定实现是否继续。

正式臂：A原模型；O有限oracle；G_GTsolver GT8；T_TRUE旧7O真实ROI；N_BASEH旧空ROI；W_INST、W_IMAGE旧错配ROI；S_BASE以sigmoid(Ac0)自投影。后六者全部使用同一求解器。概率截断可能使自投影不恒等，S_BASE正是测量这一校准效应。另用不截断的t=Ac0作纯数学恒等检查，不作为新方法臂。

保留D_TRUE/D_BASEH/D_INST/D_IMAGE及D_GT直接图作为对照。已有wrong臂是分别训练的同构头；同图wrong有207/1346回退错图，不能解释成同权重下的严格输入干预。

## 解码与统计

系数臂用完整原型→官方8.4.100 process_mask(upsample=True)→输入二值mask→真实ratio_pad逆变换→>.5。预测框固定、零logit阈值固定。直接8×8图没有历史原图评价，故本轮预先固定：将logit双线性align_corners=False缩放到640输入中floor/ceil预测框矩形，放回画布，按同一精确预测框裁剪、零阈值，然后同一逆变换。GT直接图用同一概率截断转logit。原7O的8×8 IoU与此处正常原图指标不可混比。

IoU、coverage使用原始COCO实例mask。AUC/FPR用640连续logit、nearest映射原图GT及固定预测框支持；AUC无定义保留候选并计数。主表全1,346候选；预定分组：原成功/失败、框IoU≥.75且原maskIoU<.75、P3/4/5。层级不等同大小。

5,000次按图片配对bootstrap，seed=20261002；报告图片macro和候选均值，两种区间均整图抽样。不因空mask或下降删实例。

Gate 0A使用框好掩码失败组上**macro增益之比** `(G_GTsolver-A)/(O-A)`，不是逐候选比值平均。缺oracle或非正分母记不确定。R<.5停止；.5≤R<.7灰区不训练；R≥.7仅必要条件，继续查看0B。

Gate 0B在全体候选同时要求：T_TRUE−A macro IoU点估计>0；T_TRUE−D_TRUE≥.002且95% CI下界>0；T_TRUE对W_INST、W_IMAGE、S_BASE的macro IoU差各自95% CI下界>0；对A的Mask75净修复>0。区间为定位性逐项区间，不宣称多重校正的确认性结论。若不通过不训练同类ownership predictor。即使全通过，本轮也只报告，正式训练需后续任务。

## 完成与审计

资产一致容差：h/c0 1e-5，pooled base 2e-5，GT8 1e-6，原图baselineIoU 1e-6。恒等增量≤1e-10；求解驻点和条件数完整保存。超界停止定位，不按结果放宽阈值。

产物：输入hash、固定代码快照、数学自检、身份/解码/求解审计、全部逐候选/逐图指标、系数副本、汇总及两Gate决策。失败Run保留，重试新Run。完成后停止，不自动新增头、损失、输入或训练。
