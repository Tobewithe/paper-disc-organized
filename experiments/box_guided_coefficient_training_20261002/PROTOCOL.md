# 预测框引导原生系数分支：大样本配对训练

研究动机来自[上一轮复核](../box_guided_residual_readout_20261002/TRAINING_AUDIT_REPORT.md)：冻结旧H、均值差分及单个线性层没有可靠收益，但这不充分检验原生系数分支能否学习使用框内空间信息。用户明确授权真正的框引导训练、扩大数据；本轮只执行这一套固定方案，不自动扩展更多结构。

## 方法与可训练范围

\[
H_l=\phi_{\psi,l}(F_l),\quad h_i=H_l[p_i],\quad
T_i=\operatorname{ROIAlign}_{7\times7}(H_l,\operatorname{detach}(B_i^{pred})),
\]
\[
r_i=\operatorname{CrossAttention}_{\eta}(h_i,T_i+E_{xy}+E_l),\qquad
c_i=W_l(h_i+\operatorname{Proj}_{\eta}(r_i))+b_l.
\]

49个ROI位置始终保留；二维相对位置和尺度嵌入参与读取。候选点表示作为query，token作为key/value。两层64维、4头cross-attention，每层FFN宽128，dropout为0。新增融合共71,872个参数，三尺度共享融合，但原生cv4的卷积和末层仍各尺度独立。最后残差projection零初始化，原生末层不清零；初始模型复现原系数，projection移动后上游attention得到梯度。

训练原生one-to-one系数分支的前两层卷积、BN仿射、末层及新增融合。BN运行统计固定；backbone、neck、prototype、box、class保持原权重。缓存完整FP32的neck特征F，每次训练重新运行cv4产生新的H，绝不把旧64维候选h当作可重学习表示。ROI使用输入640坐标预测框，aligned=True、sampling_ratio=2，只对采样副本clamp；最终解码仍用原预测框。

GT用于官方监督及评价，不输入框读出或注意力。框好掩码差是预先固定的主要评价分层；训练仍保留全部官方正样本，避免按测试失败身份选样。没有加入新的加权loss或教师项。是否需要针对性重加权不是本轮问题。

## 对照

| 组 | 内容 | 用途 |
|---|---|---|
| A | 官方原权重输出 | 静态原模型参照 |
| N | 只微调完整原生系数分支 | 分离追加训练的作用 |
| P | 原生分支加同样融合，但49个内容token均为候选点h | 与R严格同参数量，分离容量与区域内容 |
| R | 原生分支加预测框7×7真实空间token | 主要方法 |
| RW | 冻结最终R，仅将读ROI框换为同图其他候选的预测框 | 区域对应关系的次要诊断；不重新训练 |

N/P/R从相同官方cv4初始化；P/R融合初始化随机种子一致。RW按预测类别、框面积/宽高比选择供体，不用GT决定供体；供体池本身来自GT条件官方候选，因此仅为受控诊断。保存供体annotation关系，区分同GT和不同GT，不把这个控制称为完整部署评价。

## 数据与固定身份

复用7L历史清单的10,000张train2017作为fit；开发集为旧200张dev加从剩余train2017按固定SHA256排序取800张；val复用其历史2,000张val2017清单。SPLIT.json在读新模型效果之前冻结，不按正样本数补图。无官方正样本图片保留记录，不计算不存在的正样本loss，也不换图补足。

这些图像没有用于本轮另一分片的参数拟合，但已被历史研究使用，不称为新盲测。模型仍为Ultralytics8.4.100官方COCO预训练YOLO26m-seg。标签沿用其mask_ratio=1、overlap_mask=True，以官方convert_coco与YOLODataset转换；annotation ID跟随去重/排序，再由原始冻结模型的one-to-one TAL绑定raw位置。每个候选的框、特征、系数和标签必须来自同一raw_id。

资产核对发现笔记本仅2,000张train原图，不能把旧10k的候选h/FP16原型缓存冒充完整F。缺失原图从官方COCO公共图像源补齐；主机仅下载和中转文件，模型计算全部在笔记本。旧模型缓存和数据标注不覆盖。

## 训练目标与预算

使用已核对的官方**实例mask BCE分量**：完整原型双线性到640、系数组合、GT overlap实例标签、GT框裁剪、归一面积、checkpoint分割增益。一个有效batch内按候选等权平均。像素分块只安排内存，不抽样/改变像素权重；分块取得dL/dc后按链式法则反传系数分支，并在smoke中与普通autograd核对。

这是固定上游的分支微调，不宣称复制完整官方历史训练（没有上游更新、增强、one-to-many训练等）。N/P/R使用完全相同的数据顺序和loss，使主要比较只改变读取机制/容量。

执行值锁定于RUN_CONFIG.json：seed0，15轮，microbatch2图、累积至16图；AdamW，原生参数lr1e−4、融合lr5e−4，二维及以上权重weight_decay1e−4，bias/归一参数不decay；一轮线性warmup，之后cosine降到初始lr的10%；全局梯度范数上限10。FP32，不用AMP/TF32。固定末轮checkpoint，不依据dev/val选epoch。3个训练组共用18小时训练预算，预算不足保留已完成epoch的checkpoint并报告未完成，不拿部分训练当正式结论。缓存上限250 GB，不删除旧资产。

## 必要检查与评价

运行前只检查新增路径：初始系数复现、官方候选/监督重放、原生前级卷积实际更新、两步后attention梯度非零、BN运行统计不变、P/R参数一致。小样本只用于代码检查，不充当正式方法结果，正式训练完全重置初始化。

主比较：**val框好掩码差组的图片macro原图Mask IoU，R−N**。该组由A的BoxIoU≥0.75、原图MaskIoU<0.75定义。预设+1个IoU百分点为值得重视的实用量级；报告95%按图片配对bootstrap区间，不在结果出来后换主指标。

同时报告R−P和R−A、全体val的修复/损伤、净Mask75、coverage、连续logit AUC、FPR及fit/dev结果。全体、原成功/失败、P3/P4/P5和COCO大小分层保留；层级不等同大小。次指标为描述性95%区间，不做事后子组择优。一个有实际意义的指标改善可以保留，但不能拿它冒充整体收益。

所有模型使用同一完整原型及正常process_mask、既定letterbox逆变换，IoU对原始COCO GT评价。保留空掩码、负改善和全部冻结候选；5,000次整图配对bootstrap，候选均值也整图重采样。本轮是固定官方候选的分支训练评价，不宣称COCO AP提升。

R胜过N但未胜过P，只支持当前新增结构训练的效果，不能独立归于框区域信息；R胜过两者且真实ROI优于错ROI，更支持区域读取有用。阴性结果限定本架构、数据和预算，不自动增加模块、种子、轮数或新loss。完成本轮报告后停止。

## 设计参考与主张限制

ROIAlign用于避免区域特征的粗坐标量化，参考[Mask R-CNN, ICCV2017](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)。区域/实例信息与细粒度特征结合有既有研究，参见[BlendMask, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_BlendMask_Top-Down_Meets_Bottom-Up_for_Instance_Segmentation_CVPR_2020_paper.html)。这些工作提供设计依据，不证明本实现新颖或有效；本轮贡献判断先取决于配对结果和机制控制。
