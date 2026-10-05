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

## 分层结果

AUC 按 transfer 目标分层；S078 的 ICI/失败状态仅用于事后分组。区间是目标级 bootstrap，不是图像级不确定性。

| task | feature | stratum | n | mean AUC | 95% target bootstrap |
|---|---|---|---:|---:|---:|
| own_vs_same | coord | all | 688 | 0.4876 | [0.4648, 0.5105] |
| own_vs_same | coord | high_ici | 163 | 0.4133 | [0.3727, 0.4538] |
| own_vs_same | coord | low_ici | 525 | 0.5106 | [0.4837, 0.5385] |
| own_vs_same | coord | support_low_mask_bad | 76 | 0.5103 | [0.4509, 0.5719] |
| own_vs_same | coord | support_sufficient_mask_bad | 182 | 0.4444 | [0.4019, 0.4867] |
| own_vs_same | coord | mask_good | 430 | 0.5018 | [0.4735, 0.5322] |
| own_vs_same | coord | area_small_<1024 | 236 | 0.4353 | [0.3976, 0.4730] |
| own_vs_same | coord | area_medium_1024_9216 | 272 | 0.4918 | [0.4542, 0.5299] |
| own_vs_same | coord | area_large_>=9216 | 180 | 0.5497 | [0.5071, 0.5933] |
| own_vs_background | coord | all | 1697 | 0.4993 | [0.4900, 0.5085] |
| own_vs_background | coord | high_ici | 175 | 0.5384 | [0.5034, 0.5754] |
| own_vs_background | coord | low_ici | 1522 | 0.4948 | [0.4851, 0.5049] |
| own_vs_background | coord | support_low_mask_bad | 166 | 0.4993 | [0.4659, 0.5331] |
| own_vs_background | coord | support_sufficient_mask_bad | 424 | 0.4959 | [0.4759, 0.5159] |
| own_vs_background | coord | mask_good | 1107 | 0.5006 | [0.4890, 0.5124] |
| own_vs_background | coord | area_small_<1024 | 611 | 0.5067 | [0.4899, 0.5235] |
| own_vs_background | coord | area_medium_1024_9216 | 596 | 0.4901 | [0.4744, 0.5063] |
| own_vs_background | coord | area_large_>=9216 | 490 | 0.5013 | [0.4853, 0.5182] |
| own_vs_same | head_input | all | 688 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | high_ici | 163 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | low_ici | 525 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | support_low_mask_bad | 76 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | support_sufficient_mask_bad | 182 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | mask_good | 430 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | area_small_<1024 | 236 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | area_medium_1024_9216 | 272 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | area_large_>=9216 | 180 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | all | 1697 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | high_ici | 175 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | low_ici | 1522 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | support_low_mask_bad | 166 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | support_sufficient_mask_bad | 424 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | mask_good | 1107 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | area_small_<1024 | 611 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | area_medium_1024_9216 | 596 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | area_large_>=9216 | 490 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | prototype | all | 688 | 0.6980 | [0.6812, 0.7147] |
| own_vs_same | prototype | high_ici | 163 | 0.7034 | [0.6704, 0.7354] |
| own_vs_same | prototype | low_ici | 525 | 0.6963 | [0.6763, 0.7143] |
| own_vs_same | prototype | support_low_mask_bad | 76 | 0.5739 | [0.5195, 0.6289] |
| own_vs_same | prototype | support_sufficient_mask_bad | 182 | 0.6673 | [0.6332, 0.7008] |
| own_vs_same | prototype | mask_good | 430 | 0.7329 | [0.7146, 0.7528] |
| own_vs_same | prototype | area_small_<1024 | 236 | 0.7181 | [0.6866, 0.7491] |
| own_vs_same | prototype | area_medium_1024_9216 | 272 | 0.7050 | [0.6791, 0.7298] |
| own_vs_same | prototype | area_large_>=9216 | 180 | 0.6610 | [0.6291, 0.6907] |
| own_vs_background | prototype | all | 1697 | 0.9069 | [0.9003, 0.9134] |
| own_vs_background | prototype | high_ici | 175 | 0.8615 | [0.8356, 0.8858] |
| own_vs_background | prototype | low_ici | 1522 | 0.9122 | [0.9052, 0.9188] |
| own_vs_background | prototype | support_low_mask_bad | 166 | 0.8129 | [0.7822, 0.8397] |
| own_vs_background | prototype | support_sufficient_mask_bad | 424 | 0.8546 | [0.8390, 0.8698] |
| own_vs_background | prototype | mask_good | 1107 | 0.9411 | [0.9352, 0.9467] |
| own_vs_background | prototype | area_small_<1024 | 611 | 0.8781 | [0.8652, 0.8894] |
| own_vs_background | prototype | area_medium_1024_9216 | 596 | 0.9186 | [0.9086, 0.9285] |
| own_vs_background | prototype | area_large_>=9216 | 490 | 0.9287 | [0.9176, 0.9392] |
| own_vs_same | coefficient | all | 688 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | high_ici | 163 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | low_ici | 525 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | support_low_mask_bad | 76 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | support_sufficient_mask_bad | 182 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | mask_good | 430 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | area_small_<1024 | 236 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | area_medium_1024_9216 | 272 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | area_large_>=9216 | 180 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | all | 1697 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | high_ici | 175 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | low_ici | 1522 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | support_low_mask_bad | 166 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | support_sufficient_mask_bad | 424 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | mask_good | 1107 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | area_small_<1024 | 611 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | area_medium_1024_9216 | 596 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | area_large_>=9216 | 490 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | logit | all | 688 | 0.9371 | [0.9261, 0.9470] |
| own_vs_same | logit | high_ici | 163 | 0.9409 | [0.9212, 0.9578] |
| own_vs_same | logit | low_ici | 525 | 0.9359 | [0.9228, 0.9468] |
| own_vs_same | logit | support_low_mask_bad | 76 | 0.8110 | [0.7584, 0.8597] |
| own_vs_same | logit | support_sufficient_mask_bad | 182 | 0.8872 | [0.8628, 0.9098] |
| own_vs_same | logit | mask_good | 430 | 0.9804 | [0.9739, 0.9854] |
| own_vs_same | logit | area_small_<1024 | 236 | 0.9070 | [0.8849, 0.9271] |
| own_vs_same | logit | area_medium_1024_9216 | 272 | 0.9391 | [0.9217, 0.9545] |
| own_vs_same | logit | area_large_>=9216 | 180 | 0.9733 | [0.9605, 0.9836] |
| own_vs_background | logit | all | 1697 | 0.9505 | [0.9465, 0.9544] |
| own_vs_background | logit | high_ici | 175 | 0.9303 | [0.9175, 0.9422] |
| own_vs_background | logit | low_ici | 1522 | 0.9529 | [0.9490, 0.9568] |
| own_vs_background | logit | support_low_mask_bad | 166 | 0.8729 | [0.8520, 0.8914] |
| own_vs_background | logit | support_sufficient_mask_bad | 424 | 0.9106 | [0.8999, 0.9204] |
| own_vs_background | logit | mask_good | 1107 | 0.9775 | [0.9752, 0.9796] |
| own_vs_background | logit | area_small_<1024 | 611 | 0.9223 | [0.9133, 0.9302] |
| own_vs_background | logit | area_medium_1024_9216 | 596 | 0.9614 | [0.9557, 0.9661] |
| own_vs_background | logit | area_large_>=9216 | 490 | 0.9726 | [0.9673, 0.9776] |
| own_vs_same | proto_coeff | all | 688 | 0.9457 | [0.9352, 0.9557] |
| own_vs_same | proto_coeff | high_ici | 163 | 0.9491 | [0.9308, 0.9658] |
| own_vs_same | proto_coeff | low_ici | 525 | 0.9446 | [0.9317, 0.9556] |
| own_vs_same | proto_coeff | support_low_mask_bad | 76 | 0.8243 | [0.7710, 0.8735] |
| own_vs_same | proto_coeff | support_sufficient_mask_bad | 182 | 0.9052 | [0.8823, 0.9280] |
| own_vs_same | proto_coeff | mask_good | 430 | 0.9843 | [0.9783, 0.9893] |
| own_vs_same | proto_coeff | area_small_<1024 | 236 | 0.9237 | [0.9040, 0.9419] |
| own_vs_same | proto_coeff | area_medium_1024_9216 | 272 | 0.9434 | [0.9270, 0.9588] |
| own_vs_same | proto_coeff | area_large_>=9216 | 180 | 0.9779 | [0.9664, 0.9875] |
| own_vs_background | proto_coeff | all | 1697 | 0.9510 | [0.9469, 0.9551] |
| own_vs_background | proto_coeff | high_ici | 175 | 0.9332 | [0.9202, 0.9448] |
| own_vs_background | proto_coeff | low_ici | 1522 | 0.9530 | [0.9488, 0.9569] |
| own_vs_background | proto_coeff | support_low_mask_bad | 166 | 0.8776 | [0.8577, 0.8965] |
| own_vs_background | proto_coeff | support_sufficient_mask_bad | 424 | 0.9097 | [0.8993, 0.9197] |
| own_vs_background | proto_coeff | mask_good | 1107 | 0.9778 | [0.9755, 0.9800] |
| own_vs_background | proto_coeff | area_small_<1024 | 611 | 0.9226 | [0.9144, 0.9306] |
| own_vs_background | proto_coeff | area_medium_1024_9216 | 596 | 0.9619 | [0.9559, 0.9672] |
| own_vs_background | proto_coeff | area_large_>=9216 | 490 | 0.9730 | [0.9675, 0.9777] |
| own_vs_same | head_plus_proto_coeff | all | 688 | 0.9457 | [0.9353, 0.9553] |
| own_vs_same | head_plus_proto_coeff | high_ici | 163 | 0.9489 | [0.9291, 0.9647] |
| own_vs_same | head_plus_proto_coeff | low_ici | 525 | 0.9447 | [0.9331, 0.9558] |
| own_vs_same | head_plus_proto_coeff | support_low_mask_bad | 76 | 0.8260 | [0.7729, 0.8733] |
| own_vs_same | head_plus_proto_coeff | support_sufficient_mask_bad | 182 | 0.9040 | [0.8790, 0.9265] |
| own_vs_same | head_plus_proto_coeff | mask_good | 430 | 0.9845 | [0.9784, 0.9892] |
| own_vs_same | head_plus_proto_coeff | area_small_<1024 | 236 | 0.9239 | [0.9035, 0.9432] |
| own_vs_same | head_plus_proto_coeff | area_medium_1024_9216 | 272 | 0.9433 | [0.9256, 0.9591] |
| own_vs_same | head_plus_proto_coeff | area_large_>=9216 | 180 | 0.9780 | [0.9668, 0.9875] |
| own_vs_background | head_plus_proto_coeff | all | 1697 | 0.9508 | [0.9468, 0.9546] |
| own_vs_background | head_plus_proto_coeff | high_ici | 175 | 0.9334 | [0.9211, 0.9448] |
| own_vs_background | head_plus_proto_coeff | low_ici | 1522 | 0.9529 | [0.9486, 0.9571] |
| own_vs_background | head_plus_proto_coeff | support_low_mask_bad | 166 | 0.8765 | [0.8565, 0.8958] |
| own_vs_background | head_plus_proto_coeff | support_sufficient_mask_bad | 424 | 0.9093 | [0.8987, 0.9191] |
| own_vs_background | head_plus_proto_coeff | mask_good | 1107 | 0.9779 | [0.9756, 0.9800] |
| own_vs_background | head_plus_proto_coeff | area_small_<1024 | 611 | 0.9217 | [0.9131, 0.9297] |
| own_vs_background | head_plus_proto_coeff | area_medium_1024_9216 | 596 | 0.9624 | [0.9570, 0.9673] |
| own_vs_background | head_plus_proto_coeff | area_large_>=9216 | 490 | 0.9731 | [0.9676, 0.9782] |
| own_vs_same | all | all | 688 | 0.9456 | [0.9353, 0.9557] |
| own_vs_same | all | high_ici | 163 | 0.9486 | [0.9302, 0.9650] |
| own_vs_same | all | low_ici | 525 | 0.9447 | [0.9321, 0.9558] |
| own_vs_same | all | support_low_mask_bad | 76 | 0.8258 | [0.7735, 0.8738] |
| own_vs_same | all | support_sufficient_mask_bad | 182 | 0.9039 | [0.8797, 0.9273] |
| own_vs_same | all | mask_good | 430 | 0.9844 | [0.9785, 0.9890] |
| own_vs_same | all | area_small_<1024 | 236 | 0.9238 | [0.9026, 0.9423] |
| own_vs_same | all | area_medium_1024_9216 | 272 | 0.9434 | [0.9260, 0.9591] |
| own_vs_same | all | area_large_>=9216 | 180 | 0.9776 | [0.9658, 0.9875] |
| own_vs_background | all | all | 1697 | 0.9508 | [0.9468, 0.9546] |
| own_vs_background | all | high_ici | 175 | 0.9331 | [0.9210, 0.9448] |
| own_vs_background | all | low_ici | 1522 | 0.9529 | [0.9488, 0.9571] |
| own_vs_background | all | support_low_mask_bad | 166 | 0.8765 | [0.8572, 0.8961] |
| own_vs_background | all | support_sufficient_mask_bad | 424 | 0.9093 | [0.8978, 0.9191] |
| own_vs_background | all | mask_good | 1107 | 0.9779 | [0.9756, 0.9800] |
| own_vs_background | all | area_small_<1024 | 611 | 0.9217 | [0.9132, 0.9300] |
| own_vs_background | all | area_medium_1024_9216 | 596 | 0.9624 | [0.9568, 0.9674] |
| own_vs_background | all | area_large_>=9216 | 490 | 0.9731 | [0.9674, 0.9780] |

解释边界：高 AUC 说明冻结表示中存在可利用的像素区分信息，不证明某个可部署损失或模块已经有效；head_input 单独应接近随机，因为它对同一候选的所有像素相同。若 proto_coeff/head_plus_proto_coeff 显著优于 prototype，说明实例系数提供了必要的条件化；若加入 h 无增益，则瓶颈更可能在系数映射而非候选头缺少空间输入。
