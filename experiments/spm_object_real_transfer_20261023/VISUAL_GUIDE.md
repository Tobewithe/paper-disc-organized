# 对象原型真实迁移 v1 技术图

全部对应正式Run235的固定末步与完整432dev结果，当前效用阴性。橙色为可见Mask预测，绿色为真实可见轮廓；候选身体图是内部完整投影先验，不能当成可见GT。6个真实案例按开发帧均匀位置、首条合格预测固定选择，没有挑修复成功例。

| 图 | 内容 |
|---|---|
| [01 训练曲线](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/01_training.png) | S/Q各4000实际步，两曲线接近；loss下降不作AP证据 |
| [02 全量真实成绩](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/02_real_segmentation.png) | 原生、Q、S及同S关/乱对象输入的AP/AP75/小目标AP |
| [03 修复与损伤](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/03_repair_damage.png) | 官方同GT任务匹配；S修复23、损伤106 |
| [04 已训练身体候选](presentation_revision/04_learned_body_candidates_r1.png) | 12个固定均匀候选，颜色为已训练特征的固定投影，不是语义ID；R1仅修标题排版 |
| [05 真实案例0](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/05_fixed_real_00.png) | 开发序号0，原生/完整候选/S/Q |
| [06 真实案例1](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/05_fixed_real_01.png) | 开发序号86 |
| [07 真实案例2](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/05_fixed_real_02.png) | 开发序号172 |
| [08 真实案例3](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/05_fixed_real_03.png) | 开发序号258 |
| [09 真实案例4](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/05_fixed_real_04.png) | 开发序号344 |
| [10 真实案例5](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/05_fixed_real_05.png) | 开发序号431 |

固定案例多为高置信度输出，肉眼变化较小；完整12906条输出AP与106条成功损伤是整体结论，不用这些局部图弱化负收益。[案例身份](runs/RUN_SPM_REAL_OBJECT_PRESENTATION_9d90d1af0976/SUMMARY.json)、[排版修订](presentation_revision/FIGURE_LAYOUT_R1.json)、[完整报告](REPORT.md)。
