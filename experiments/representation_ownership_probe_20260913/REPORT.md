# S081：COCO 图片级迁移的目标/邻居表示判别 probe

状态：COMPLETE。复用冻结 readout cache，无新增前向、训练或 AP。fit 图片拟合线性判别器，transfer 图片只用于评估；COCO GT 只定义像素归属标签。

图片 1500，匹配目标 9626，可用目标/任务记录 12439。每个目标最多平衡采样自身与负类各 32 个像素。

特征含义：coord=位置；head_input=候选头输入 h（同一候选内为空间常量）；prototype=原型向量；coefficient=实例系数；logit=原型与系数的标量乘积；proto_coeff=逐通道贡献；all=位置+h+逐通道贡献。

| task | feature | transfer micro AUC | transfer macro target AUC (95% image bootstrap) |
|---|---|---:|---:|
| own_vs_same | coord | 0.5084 | 0.4876 [0.4648, 0.5105] |
| own_vs_same | head_input | 0.5031 | 0.5000 [0.5000, 0.5000] |
| own_vs_same | prototype | 0.6838 | 0.6980 [0.6821, 0.7150] |
| own_vs_same | coefficient | 0.5039 | 0.5000 [0.5000, 0.5000] |
| own_vs_same | logit | 0.9449 | 0.9371 [0.9268, 0.9470] |
| own_vs_same | proto_coeff | 0.9551 | 0.9457 [0.9356, 0.9552] |
| own_vs_same | head_plus_proto_coeff | 0.9550 | 0.9457 [0.9357, 0.9552] |
| own_vs_same | all | 0.9548 | 0.9456 [0.9356, 0.9552] |
| own_vs_background | coord | 0.4986 | 0.4993 [0.4895, 0.5090] |
| own_vs_background | head_input | 0.5008 | 0.5000 [0.5000, 0.5000] |
| own_vs_background | prototype | 0.9097 | 0.9069 [0.9005, 0.9135] |
| own_vs_background | coefficient | 0.5013 | 0.5000 [0.5000, 0.5000] |
| own_vs_background | logit | 0.9468 | 0.9505 [0.9467, 0.9544] |
| own_vs_background | proto_coeff | 0.9508 | 0.9510 [0.9471, 0.9548] |
| own_vs_background | head_plus_proto_coeff | 0.9524 | 0.9508 [0.9469, 0.9547] |
| own_vs_background | all | 0.9523 | 0.9508 [0.9469, 0.9546] |

解释边界：高 AUC 说明冻结表示中存在可利用的像素区分信息，不证明某个可部署损失或模块已经有效；head_input 单独应接近随机，因为它对同一候选的所有像素相同。若 proto_coeff/head_plus_proto_coeff 显著优于 prototype，说明实例系数提供了必要的条件化；若加入 h 无增益，则瓶颈更可能在系数映射而非候选头缺少空间输入。
