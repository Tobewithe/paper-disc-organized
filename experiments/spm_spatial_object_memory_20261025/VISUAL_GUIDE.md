# V21真实权重与结果图册

共46张科学/技术/效果图，均来自实际固定末模型/输出，已逐张打开检查。图像生成不作为效用证据；整体成绩仍是432真实dev Mask AP+3.0730、框不变、损伤0.8467%、+5未达。实际看图的SHA和说明见[记录](FINAL_VISUAL_QA_PROGRESS.json)。

## 先看完整验收与训练

- [完整真实成绩](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_VISUALS_V1_ff4a948ae5a2/real_full_acceptance.png)：完整AP、尺度和护栏，不以局部好例代替全量。
- [实际训练曲线](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_VISUALS_V1_ff4a948ae5a2/actual_training_curves.png)：8000步对象/候选及8epoch策略；loss或训练成本不是AP。

## 原型内容、检索与真实调用

- [64槽整体与九部位字段](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_VISUALS_V1_ff4a948ae5a2/trained_memory_64.png)：形状来自单个几何身体，颜色是既定九部位；强锚使binary没有变化，不是新学形状的证据。
- [实际键及正常匹配](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_VISUALS_V1_ff4a948ae5a2/trained_addresses_and_normal_matching.png)：地址相似度/激活可见集中，不能把64容量写成64个不同语义对象。
- [检索与动作](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_VISUALS_V1_ff4a948ae5a2/normal_retrieval_and_actions.png)及[三层调用统计](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_ACTUAL_MEMORY_USAGE_V1_d9c0b6bce1a9/actual_memory_usage_three_levels.png)：最高检索、最终选用、真正改像素分别统计。5/40/40槽并不是同一种激活；真实选择91.50%不同于top1。

## 八个实际合成查询的部位组成与遮挡

`mechanism_0.png`至`mechanism_7.png`位于[主图目录](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_VISUALS_V1_ff4a948ae5a2)；对应`composition_0.png`至`composition_7.png`位于[新增细节目录](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FINAL_BOUNDARY_AND_COMPOSITION_VISUALS_42b7448b85e1)。每对说明同一正常查询的检索、整体、部位、物理rho、相对深度、加删修正及实际决策。

组成图显示实际最高检索槽的`G L_j`映射、给定附件/亲缘关系和接触约定；若最终选用不同槽，标题明确标注，不把两者混用。这些框是2D规范部位变换，不是估计出的3D骨架。部位柱形以解码的自身可见对象总质量为分母，比较rho前后保留量，不是逐部位真实可见率准确度。热图颜色用于单图定位，没有共同绝对色标，跨图幅度以原NPZ数值为准。

八图均为训练见过的合成图。可以看到身体/耳/腿等既定解码结构；不能据此宣称真实部位识别或新体形泛化。

## 十二个真实固定样例与边界增删

图像ID为94、320、565、586、798、955、1030、1203、1269、1324、1410、1565，按冻结SHA规则固定。每张`real_ID.png`是RGB/原生/完整方法，展示阈值.25，**官方AP仍用.001**。对应新增`boundary_ID.png`固定同一检测行，青色为原生、紫色为方法、黄色为说明用GT，绿色增加/红色删除。

边界说明的GT是同类非crowd中最大BoxIoU≥.5的参考；标题的同预测MaskIoU不是COCOeval独立segm匹配，不把它们拼成官方成功。ROI按最高分非KEEP选取，不按GT改善挑样例。

| 样例 | 该固定行MaskIoU（近似三位） | 实际看图结论 |
|---|---|---|
| 94 | .926→.944 | 局部边缘改善 |
| 320 | .842→.908 | 身体轮廓改善 |
| 565 | .933→.920 | 轻微退化，保留 |
| 586 | .783→.766 | 轻微退化，且BoxIoU约.740，不标作Box75成功 |
| 798 | .863→.874 | 夜间样例改善，原检测漏项没有被新增 |
| 955 | .924→.921 | 轻微退化，密集小猪框仍受原生限制 |
| 1030 | .913→.927 | 局部边界改善 |
| 1203 | .887→.902 | 局部边界改善 |
| 1269 | .904→.913 | 笼栏下边缘增删 |
| 1324 | .906→.913 | 身体边缘与腿部局部增删 |
| 1410 | .700→.814 | 耳/身体连接及过伸区域改善 |
| 1565 | .931→.949 | 身体外围小幅修正 |

这些图说明实际模型如何改变输出，也同时保留收益和损伤。它们不证明改动来自正确原型内容；该点以[完整字段依赖核验](final_delivery/db838c03da4e/runs/RUN_SPM_SPATIAL_FIELD_DEPENDENCE_EVALUATE_V1_d2339e19e261/SUMMARY.json)的阴性实用幅度判断为准。
