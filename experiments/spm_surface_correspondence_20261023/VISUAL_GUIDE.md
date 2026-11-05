# V14结果与身体对应技术图

全部来自本项真实固定模型前向或独立CPU复算；8组PNG/SVG均已检查。没有新训练或新的AP成绩。

## 全量成绩

- [能力、姿态、双猪干扰总图](runs/RUN_SPM_V14_ACCEPTANCE_7c2921be1d26/figures/00_capability.png)：主阈值70%、各姿态50%均在图中预先标明。双猪下降8.75pp虽通过损伤护栏，绝对定位仍低。
- [身体表面误差累积分布](runs/RUN_SPM_V14_ACCEPTANCE_7c2921be1d26/figures/01_pck_curve.png)：学习度量在紧阈值有精确点优势，同时保留明显的大错配尾部。真值部位参照有额外GT权限，不能当部署AP比较。

## 身体身份与误差

每图从左到右：RGB；规范身体GT颜色；实际学习度量检索出的身体顶点颜色；归一化图测地误差。颜色固定由规范xyz决定，跨姿态相同颜色表示相同身体位置。GT身体区域只限定展示/评分范围，未输入网络；图中显示的是主体条件身体检索，不是前景分割结果。

- [站立固定样例](runs/RUN_SPM_V14_EVALUATE_1e7bfe487d58/figures/single_f024_p0_v0_a0_o0.png)
- [俯卧固定样例](runs/RUN_SPM_V14_EVALUATE_1e7bfe487d58/figures/single_f024_p3_v1_a0_o0.png)
- [侧卧固定样例](runs/RUN_SPM_V14_EVALUATE_1e7bfe487d58/figures/single_f024_p4_v2_a0_o0.png)

## 邻猪干扰

固定索引0/24/48，均在运行前指定。两猪是已有dev单体的2D平移层叠，不能称物理3D重渲染或真实猪群；各实例ID/身体点身份用于后验评分。背景/目标实例归属接口并未实现。

- [双猪000](runs/RUN_SPM_V14_EVALUATE_1e7bfe487d58/figures/crowd_000.png)
- [双猪024](runs/RUN_SPM_V14_EVALUATE_1e7bfe487d58/figures/crowd_024.png)
- [双猪048](runs/RUN_SPM_V14_EVALUATE_1e7bfe487d58/figures/crowd_048.png)

同名SVG与PNG并列保存，可用于导出。[完整研究链路](REPORT.md)、[下一训练设计](NEXT_TRAINING_PLAN.md)。
