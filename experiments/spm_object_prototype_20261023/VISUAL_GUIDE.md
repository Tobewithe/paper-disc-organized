# 对象原型v0技术图集

18张图均已查看和SHA核验。完整实验、门槛失败及真实域边界见[报告](REPORT.md)。所有对象结果来自固定末步模型；给定网格和部位先验与学习后的结果分别标注。

| 图 | 内容 | 阅读边界 |
|---|---|---|
| [训练曲线](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/01_training_curves.png) | 01 | 总Loss、顶点和跨视角点损失，训练拟合不等于真实AP收益。 |
| [完整对象能力](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/02_held_out_object_ability.png) | 02 | 288遮挡输入的身体误差、完整投影IoU和体形码漂移；同时展示收益与稳定性缺口。 |
| [8族配对结果](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/03_paired_family_results.png) | 03 | 预先留出的8个形变族，均有几何误差增量；不是8个独立资产。 |
| [遮挡观察到其他视角](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/05_occlusion_to_different_view.png) | 05 | 固定3个源样本；输入只有源RGB与相机，白色目标是完整身体投影，不是部署可见Mask。 |
| [学习到的身体特征场](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/06_learned_surface_field.png) | 06 | 左为给定粗部位颜色；右为64维学习特征的训练域PCA颜色，不代表独立实例ID或无监督语义发现。 |
| [同一体形的姿态控制](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/07_controlled_pose_edit.png) | 07 | 输入推断体形后请求六种姿态；姿态控制值来自给定资产，不冒充姿态估计精度。 |

12个固定对象重建沿用评价源码预先指定的family24/31、姿态0/3/4、view1、外观0和清洁/遮挡两状态；每一行目标/U/S使用相同坐标和尺度，不按结果挑案例。

| 固定样本 | 对象重建图 |
|---|---|
| f024_p0_v1_a0_o0 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_00.png) |
| f024_p0_v1_a0_o1 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_01.png) |
| f024_p3_v1_a0_o0 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_02.png) |
| f024_p3_v1_a0_o1 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_03.png) |
| f024_p4_v1_a0_o0 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_04.png) |
| f024_p4_v1_a0_o1 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_05.png) |
| f031_p0_v1_a0_o0 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_06.png) |
| f031_p0_v1_a0_o1 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_07.png) |
| f031_p3_v1_a0_o0 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_08.png) |
| f031_p3_v1_a0_o1 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_09.png) |
| f031_p4_v1_a0_o0 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_10.png) |
| f031_p4_v1_a0_o1 | [目标、普通、结构三者对照](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/04_fixed_object_11.png) |

[学习后对象OBJ](artifacts/demo_inference_232/object_prediction.obj)由单次独立推断生成；[原始资产六姿态QA](presentation/01_given_asset_pose_QA.png)是给定先验，不能当作训练结果。

来源：[展示Run](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/run.json)、[固定样本清单](runs/RUN_SPM_OBJECT_PRESENTATION_0760243e2f46/SUMMARY.json)、[绘图源码](presentation_tools/publish_results.py)。可视化不修改原始评价或科学裁决。
