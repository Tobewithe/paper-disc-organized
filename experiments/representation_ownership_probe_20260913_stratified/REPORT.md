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
| own_vs_background | coord | all | 1697 | 0.4993 | [0.4905, 0.5088] |
| own_vs_background | coord | high_ici | 175 | 0.5384 | [0.5019, 0.5737] |
| own_vs_background | coord | low_ici | 1522 | 0.4948 | [0.4854, 0.5046] |
| own_vs_background | coord | support_low_mask_bad | 166 | 0.4993 | [0.4691, 0.5304] |
| own_vs_background | coord | support_sufficient_mask_bad | 424 | 0.4959 | [0.4755, 0.5156] |
| own_vs_background | coord | mask_good | 1107 | 0.5006 | [0.4888, 0.5118] |
| own_vs_same | head_input | all | 688 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | high_ici | 163 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | low_ici | 525 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | support_low_mask_bad | 76 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | support_sufficient_mask_bad | 182 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | head_input | mask_good | 430 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | all | 1697 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | high_ici | 175 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | low_ici | 1522 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | support_low_mask_bad | 166 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | support_sufficient_mask_bad | 424 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | head_input | mask_good | 1107 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | prototype | all | 688 | 0.6980 | [0.6812, 0.7141] |
| own_vs_same | prototype | high_ici | 163 | 0.7034 | [0.6688, 0.7364] |
| own_vs_same | prototype | low_ici | 525 | 0.6963 | [0.6767, 0.7159] |
| own_vs_same | prototype | support_low_mask_bad | 76 | 0.5739 | [0.5188, 0.6293] |
| own_vs_same | prototype | support_sufficient_mask_bad | 182 | 0.6673 | [0.6328, 0.6992] |
| own_vs_same | prototype | mask_good | 430 | 0.7329 | [0.7128, 0.7519] |
| own_vs_background | prototype | all | 1697 | 0.9069 | [0.8999, 0.9132] |
| own_vs_background | prototype | high_ici | 175 | 0.8615 | [0.8365, 0.8850] |
| own_vs_background | prototype | low_ici | 1522 | 0.9122 | [0.9055, 0.9187] |
| own_vs_background | prototype | support_low_mask_bad | 166 | 0.8129 | [0.7825, 0.8398] |
| own_vs_background | prototype | support_sufficient_mask_bad | 424 | 0.8546 | [0.8384, 0.8697] |
| own_vs_background | prototype | mask_good | 1107 | 0.9411 | [0.9352, 0.9467] |
| own_vs_same | coefficient | all | 688 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | high_ici | 163 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | low_ici | 525 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | support_low_mask_bad | 76 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | support_sufficient_mask_bad | 182 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | coefficient | mask_good | 430 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | all | 1697 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | high_ici | 175 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | low_ici | 1522 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | support_low_mask_bad | 166 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | support_sufficient_mask_bad | 424 | 0.5000 | [0.5000, 0.5000] |
| own_vs_background | coefficient | mask_good | 1107 | 0.5000 | [0.5000, 0.5000] |
| own_vs_same | logit | all | 688 | 0.9371 | [0.9267, 0.9471] |
| own_vs_same | logit | high_ici | 163 | 0.9409 | [0.9216, 0.9572] |
| own_vs_same | logit | low_ici | 525 | 0.9359 | [0.9238, 0.9480] |
| own_vs_same | logit | support_low_mask_bad | 76 | 0.8110 | [0.7574, 0.8598] |
| own_vs_same | logit | support_sufficient_mask_bad | 182 | 0.8872 | [0.8623, 0.9099] |
| own_vs_same | logit | mask_good | 430 | 0.9804 | [0.9740, 0.9855] |
| own_vs_background | logit | all | 1697 | 0.9505 | [0.9467, 0.9543] |
| own_vs_background | logit | high_ici | 175 | 0.9303 | [0.9167, 0.9425] |
| own_vs_background | logit | low_ici | 1522 | 0.9529 | [0.9486, 0.9571] |
| own_vs_background | logit | support_low_mask_bad | 166 | 0.8729 | [0.8535, 0.8917] |
| own_vs_background | logit | support_sufficient_mask_bad | 424 | 0.9106 | [0.9003, 0.9206] |
| own_vs_background | logit | mask_good | 1107 | 0.9775 | [0.9752, 0.9796] |
| own_vs_same | proto_coeff | all | 688 | 0.9457 | [0.9354, 0.9552] |
| own_vs_same | proto_coeff | high_ici | 163 | 0.9491 | [0.9308, 0.9651] |
| own_vs_same | proto_coeff | low_ici | 525 | 0.9446 | [0.9323, 0.9558] |
| own_vs_same | proto_coeff | support_low_mask_bad | 76 | 0.8243 | [0.7718, 0.8709] |
| own_vs_same | proto_coeff | support_sufficient_mask_bad | 182 | 0.9052 | [0.8817, 0.9269] |
| own_vs_same | proto_coeff | mask_good | 430 | 0.9843 | [0.9780, 0.9890] |
| own_vs_background | proto_coeff | all | 1697 | 0.9510 | [0.9471, 0.9547] |
| own_vs_background | proto_coeff | high_ici | 175 | 0.9332 | [0.9210, 0.9452] |
| own_vs_background | proto_coeff | low_ici | 1522 | 0.9530 | [0.9485, 0.9572] |
| own_vs_background | proto_coeff | support_low_mask_bad | 166 | 0.8776 | [0.8579, 0.8970] |
| own_vs_background | proto_coeff | support_sufficient_mask_bad | 424 | 0.9097 | [0.8989, 0.9204] |
| own_vs_background | proto_coeff | mask_good | 1107 | 0.9778 | [0.9754, 0.9799] |
| own_vs_same | head_plus_proto_coeff | all | 688 | 0.9457 | [0.9351, 0.9554] |
| own_vs_same | head_plus_proto_coeff | high_ici | 163 | 0.9489 | [0.9293, 0.9653] |
| own_vs_same | head_plus_proto_coeff | low_ici | 525 | 0.9447 | [0.9312, 0.9552] |
| own_vs_same | head_plus_proto_coeff | support_low_mask_bad | 76 | 0.8260 | [0.7738, 0.8726] |
| own_vs_same | head_plus_proto_coeff | support_sufficient_mask_bad | 182 | 0.9040 | [0.8797, 0.9264] |
| own_vs_same | head_plus_proto_coeff | mask_good | 430 | 0.9845 | [0.9783, 0.9890] |
| own_vs_background | head_plus_proto_coeff | all | 1697 | 0.9508 | [0.9466, 0.9548] |
| own_vs_background | head_plus_proto_coeff | high_ici | 175 | 0.9334 | [0.9206, 0.9448] |
| own_vs_background | head_plus_proto_coeff | low_ici | 1522 | 0.9529 | [0.9486, 0.9570] |
| own_vs_background | head_plus_proto_coeff | support_low_mask_bad | 166 | 0.8765 | [0.8566, 0.8952] |
| own_vs_background | head_plus_proto_coeff | support_sufficient_mask_bad | 424 | 0.9093 | [0.8983, 0.9191] |
| own_vs_background | head_plus_proto_coeff | mask_good | 1107 | 0.9779 | [0.9756, 0.9800] |
| own_vs_same | all | all | 688 | 0.9456 | [0.9354, 0.9553] |
| own_vs_same | all | high_ici | 163 | 0.9486 | [0.9292, 0.9654] |
| own_vs_same | all | low_ici | 525 | 0.9447 | [0.9330, 0.9563] |
| own_vs_same | all | support_low_mask_bad | 76 | 0.8258 | [0.7745, 0.8757] |
| own_vs_same | all | support_sufficient_mask_bad | 182 | 0.9039 | [0.8780, 0.9265] |
| own_vs_same | all | mask_good | 430 | 0.9844 | [0.9781, 0.9891] |
| own_vs_background | all | all | 1697 | 0.9508 | [0.9467, 0.9549] |
| own_vs_background | all | high_ici | 175 | 0.9331 | [0.9197, 0.9449] |
| own_vs_background | all | low_ici | 1522 | 0.9529 | [0.9485, 0.9570] |
| own_vs_background | all | support_low_mask_bad | 166 | 0.8765 | [0.8560, 0.8951] |
| own_vs_background | all | support_sufficient_mask_bad | 424 | 0.9093 | [0.8987, 0.9195] |
| own_vs_background | all | mask_good | 1107 | 0.9779 | [0.9756, 0.9800] |

解释边界：高 AUC 说明冻结表示中存在可利用的像素区分信息，不证明某个可部署损失或模块已经有效；head_input 单独应接近随机，因为它对同一候选的所有像素相同。若 proto_coeff/head_plus_proto_coeff 显著优于 prototype，说明实例系数提供了必要的条件化；若加入 h 无增益，则瓶颈更可能在系数映射而非候选头缺少空间输入。
