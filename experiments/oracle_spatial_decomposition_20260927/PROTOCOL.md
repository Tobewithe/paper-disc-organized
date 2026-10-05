# 7R：有限 oracle 修正的空间分解

2026-09-27，查看7R结果前登记。来源为7Q固定模型、有限oracle（lambda=0.003）和预测缓存；不训练、不选择新的checkpoint。远端当前拒绝连接，本地同版本8.4.100、相同官方权重重建原型，逐图核对已存储的系数、框、ROI原型。沿用已分析过的7Q测试集，是探索性后续诊断，不声称新的独立确认实验。

## 问题与数学对象
在输入640坐标下，令S为固定预测框与有效图像区域的交集。oracle修正为 d*=Upsample(P delta_c*)，预测修正为 d_hat；区域指示函数相加为1：
d*|S = sum_r 1[R_r] d*，r属于boundary, interior, neighbor, background。
这是函数的区域分解；各区域单独截取的修正通常不再属于prototype列空间，不是四个可独立实现的系数方向。

原始COCO annToMask是区域真值；不使用训练的overlap owner mask替代原始GT。目标掩码G按官方实际缩放尺寸最近邻缩放并补零到640。邻居O为该图全部其他标注前景（包括crowd）的并集，扣除目标G。crowd单独保存为子区域，不能把crowd前景当背景。

## 互斥区域
主半径r=3个输入像素，r=1、5为边界宽度敏感性；方形(2r+1)形态学核。
- boundary = (dilate(G)-erode(G)) intersect S（内外侧均包括）
- interior = erode(G) intersect S
- neighbor = O minus dilate(G)，再intersect S
- background = S minus上述三区域
四者互斥、穷尽S。边界优先，因此另外报告inner boundary、outer boundary on neighbor、outer boundary on background、crowd区域，防止类别优先顺序掩盖邻居错误。此处background仅指未标注为COCO实例的区域，不保证物理上不存在其他物体。

## 样本与对照
沿用7Q的TEST.pt全部14,911个固定one2one正样本候选，来自2,000张预选图中的1,982张有效图。不重新运行GT分配，不改变候选框/系数身份。
预设分组：全部；原Mask75失败；失败且BoxIoU>=0.75；原Mask75成功。
所有分组以7Q原图COCO结果定义，不以本轮640结果重分组。
主预测器p_coefficient、p_spatial；参照h_coefficient、wrong_instance_spatial、wrong_image_spatial；各2个已有种子。原始与oracle为端点。两个种子的指标先分别算再平均，不平均预测来假装一个模型。

## 测量
1. 区域面积、oracle修正平方能量与能量密度；区域cosine、相对误差、幅度投影系数(dot(d_hat,d*)/||d*||²)、能量加权反向比例。
2. 分原始FN/FP/TP/TN统计：oracle/预测的修正符号、修复像素、误伤像素。避免把所有背景下降都当作有效修复。
3. GT辅助区域替换：仅将区域r的预测修正换为oracle，其余区域保留预测。比较与完整预测的IoU差；另作“仅该区域oracle”及“oracle去掉该区域”控制。IoU效应不具可加性，不能直接加和当贡献分摊。
4. 四区域结果使用640输入栅格的原始GT映射，便于严格区域替换和像素记账；同时复核原始图分辨率解码的baseline/oracle/主预测IoU与7Q缓存一致性。640统计不冒称原图COCO AP。
5. 2,000次配对逐图cluster bootstrap，区域空/零oracle能量保持undefined并报告有效样本数。预先指定主分析good-box failures下p_spatial的四区域hybrid gain，Holm校正4项；其他分组/宽度/预测器对照均描述性。报告两个种子单独结果。

## 如何解释
区域cosine低并不自动意味着该区域最值得修；同时看oracle实际修复量与区域替换能带回多少IoU。内部已学会但邻居未学会、边界缺失、普遍幅度收缩等结果均可成立；不得预设答案。边界缺失不等于证明高频特征不足，邻居缺失不等于证明相邻实例导致。下一步只根据稳定且有实际收益的区域差异设计干预。

本轮不增加网络、不扫loss、不以测试GT指导可部署门控；区域替换明确是GT辅助诊断上界。

补充（完整运行完成/读取前）：主半径下同时报告区域替换使IoU增加至少0.01、降低至少0.01的实例比例，检验平均收益是否只由少数异常值驱动。

## 输出网格对照（完整结果读取前补充）
沿用7Q的oracle_grid：把P delta_c*先通过同一16x16 ROI算子，再用相同双线性插值回填预测框。比较各区域的oracle-grid与原oracle方向、区域仅修复收益和grid预测补回该区域oracle的增益。该对照无需训练，使用GT，属于表示上界。若边界oracle-grid保留大量收益而预测器未保留，不能把边界短板简单归咎于16x16输出分辨率；仍不证明哪层输入信息缺失。
