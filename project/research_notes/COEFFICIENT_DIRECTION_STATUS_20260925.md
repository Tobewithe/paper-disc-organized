# 系数修正方向研究现状（始于 2026-09-25，更新至 2026-09-26）

当前对象是官方 COCO 预训练 YOLO26m-seg 固定 one-to-one 正样本中的**系数修正方向可预测性**，不是完整 COCOeval AP 方法结果。7D 给出 \(\lambda=.003\) 的有限、稳定 GT 条件修正目标；GT 仅构造监督和评价，不进入推理。

| 问题 | 已有事实 | 证据 |
| --- | --- | --- |
| 方向需要准到什么程度才有实际收益？ | 在合成、GT 辅助、固定 oracle 半径的 7E.1 曲线上，平均 IoU 净益首次稳定出现在作用余弦约 .5，Mask75 净益约 .65；这不是部署阈值。 | [7E.1](../../experiments/coefficient_direction_utility_20260925/RESULTS.md) |
| 更多相同输入数据能否解决？ | `h`-only 在 800→10,000 训练图像中 val 作用余弦 .353→.362，配对区间跨零；原失败组约 .16–.19，原图 IoU 仍负。 | [7F-A](../../experiments/coefficient_direction_scaling_20260925/RESULTS.md) |
| 更细的局部原型能否解决？ | 真实 8×8 未超过同容量 4×4 或 `h`-only，亦未可靠超过错配原型。 | [7F-B](../../experiments/coefficient_spatial_descriptor_20260925/RESULTS.md) |
| 相似 `h` 是否承载同一修正方向？ | 同类同尺度近邻在全体验证实例好于随机；原失败，特别是框好掩码差组的近邻优势很弱且区间可跨零。不能据此宣称信息论歧义。 | [7G](../../experiments/coefficient_feature_ambiguity_20260925/RESULTS.md) |
| 直接输入当前 \(Pc_0\) 能否补足？ | 真实 8×8 原生 logit 不优于错配 logit，原图 IoU 相对 `h`-only 进一步下降。 | [7H](../../experiments/coefficient_raw_mask_condition_20260925/RESULTS.md) |
| 不看 GT 能否识别谁失败／谁有高 oracle-gap？ | 2026-09-26 的 7I 在 GT 固定正样本上，检测输出识别 Mask75 失败 AUROC .863；框好子组 .828。高 G 仅约 .74–.75，空间图不优于检测输出。 | [7I](../../experiments/coefficient_failure_identifiability_20260926/RESULTS.md) |
| 投影前 64 通道局部空间图能否给方向？ | 7J 真实 ROI 未超过 `h_only` 或错配 ROI 的配对区间；原失败组作用余弦 .177，原图 IoU 无稳定增益。 | [7J](../../experiments/coefficient_spatial_head_information_20260926/RESULTS.md) |
| 再往上游的 256/512 通道 neck ROI 能否给方向？ | 7J-N 中真实 ROI 在全体作用余弦 .329，低于同容量 `h_only` 的 .354；原 Mask75 失败组 .172 对 .191，配对区间未支持改善。全体原图 IoU 相对原模型仅 +.0003，Mask75 救回 24 个与 `h_only` 相同。 | [7J-N](../../experiments/coefficient_neck_roi_20260926/RESULTS.md) |
| 用失败评分只修坏实例能否避免损伤？ | 7K 主 gate 在 dev 5% 目标成功误选率下，val 抓到 48.2% 原失败，但 `h_only` 仅抓到 3/24 个实际可救回者，3 修复/4 损伤；真实 neck ROI 3/3。原失败组内，失败评分对“当前修正器可救回”的 AUROC 仅 .261，95% 图像聚类 CI [.186,.351]。失败识别不能充当收益预测。 | [7K](../../experiments/coefficient_failure_gated_correction_20260927/RESULTS.md) |

**当前判断：** “附近有好系数”成立；“当前共享头只需更多同类数据、更多未加权原型细节或原掩码几何就能预测好方向”没有得到支持。真正需要修复的实例比原成功实例更难预测且所需位移更大。不能把这归因成官方 loss、原型容量或类别混淆的确定机制，也还没有可报告的 COCO AP 方法收益。

**当前分叉：** 7I/7J/7J-N/7K 分开检验了失败识别、修正方向、局部输入与应用范围。检测输出能识别部分失败，但当前失败分数偏向更严重、较难由冻结系数头救回的候选。不能把“谁失败”当作“修谁有益”，也不能靠对所有实例修正或只对高失败分数实例修正得到稳定收益。按当前证据，停止沿**冻结表示＋同类共享方向头**反复换输入和简单 failure gate；若继续 whether/where/how far，应先在无训练泄漏的条件下定义并检验可预测的 correction-benefit 目标，再验证所有推理候选和完整 COCOeval。不能把这些单种子、小训练集、已审阅 val 的诊断写成对上游信息或新方法的最终否定。
