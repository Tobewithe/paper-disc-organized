# SPM-YOLO V4重点技术与最终结果可视化

**实际训练已完成，单seed、54张real-dev图。** F Mask AP=50.9267，比同预算continued +1.3045pp，比G +1.3725pp，比B* +0.2306pp；AP75相对B*下降1.1407pp。

推理关闭检索几乎不改变结果；优先使用已导出的原生head权重。图中的gate/相似度/残差全部来自真实checkpoint。开发选例不是独立test或统计显著性证据。

接触放大图经复核：不存在C token的样本改取猪-背景边界，并在图下注明；旧版crop保留在原Run，不再作为正式展示。

[完整报告](REPORT.md) · [权重与推理](USAGE.md) · [独立验收](ACCEPTANCE_RECEIPT.json)

## 实现结构与冻结/训练边界

![实现结构与冻结/训练边界](runs/RUN_V4_METHOD_DIAGRAM_1b13f671f212/figures/00_method_overview.png)

## 训练曲线与四臂真实开发成绩

![训练曲线与四臂真实开发成绩](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/01_training_and_results.png)

## 独立原型训练损失和梯度

![独立原型训练损失和梯度](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/02_prototype_training.png)

## 数据来源、猪体尺度和姿态分布

![数据来源、猪体尺度和姿态分布](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/03_data_distribution.png)

## 16个前景原型的真实近邻与来源

![16个前景原型的真实近邻与来源](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/04_prototype_atlas_fg.png)

## 16个背景原型的真实近邻与来源

![16个背景原型的真实近邻与来源](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/04_prototype_atlas_bg.png)

## 32槽原型利用率与来源占比

![32槽原型利用率与来源占比](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/05_memory_support.png)

## 真实/合成及C/B/I/BG嵌入分布

![真实/合成及C/B/I/BG嵌入分布](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/06_embedding_PCA.png)

## 固定开发样本的改善与退化全图

![固定开发样本的改善与退化全图](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/07_improvement_and_damage.png)

## 接触与猪-背景边界放大；无接触时明确标注回退

![接触与猪-背景边界放大；无接触时明确标注回退](runs/RUN_V4_VISUAL_REVIEW_564b07a78306/figures/08_contact_and_boundary_zoom.png)

## 实际检索相似度、分配、gate和有效残差

![实际检索相似度、分配、gate和有效残差](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/09_retrieval_gate_residual.png)

## 推理开销与同checkpoint必要干预

![推理开销与同checkpoint必要干预](runs/RUN_V4_VISUALIZE_b8797cb13710/figures/10_cost_and_interventions.png)

## 4个推断family的开发收益与损失

![4个推断family的开发收益与损失](runs/RUN_V4_VISUAL_REVIEW_564b07a78306/figures/11_grouped_development_effect.png)
