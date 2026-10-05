# Experiment 7K：Failure-Gated Correction

## 问题与对象

本实验只问：在已冻结的系数修正器不变时，使用 7I 的 GT-free 失败评分选择少量候选，能否保留修复而减少原成功实例的损伤？对象仍为 7E/7I 的官方 one-to-one、**GT 条件入组**候选，不代表任意推理候选的完整 COCO AP。训练 GT 只用于既有 7I gate 与 7J-N 修正器训练；本轮选择阈值只用 dev 的原 Mask75 标签，val 标签只用于最后评价。

必须说明：val 图像虽未训练 gate 或修正器，但已用于 7I/7J 的结果审阅，所以本轮是既有验证集上的**探索性组合诊断**，不能称新盲测。若出现可观收益，另用未参与这条路线决策的图像/候选及标准 COCOeval 确认。

## 冻结输入与事前选择

- 主要 gate：7I `h+raw detection` 的原 Mask75 failure Logistic Regression。完全复用 7I 的特征标准化、正则和 fit 训练；重算 val 评分须逐项等于已存 `SCORES.json`。
- 主要 correction：7J-N 的 `h_only` 最佳 dev checkpoint；次要 correction：同次训练的 `true_local`。作为跨实验敏感性对照，另报告 7J 的 `h_only`。不重训这三组。
- gate 对照：7I 的 `box_score` failure 评分，检验检测输出是否已足够；`h+raw detection` high-gap 评分作为探索性的“修正价值”门控，不作为主判断。
- dev 上分别用**原 Mask75 成功实例**评分设定目标误选率 1%、3%、5%、10% 的阈值，采用 `score > threshold`，相同阈值直接用于 val。主操作点为 dev 5%；其他点用于 risk–coverage 曲线。所有 arm 都有 `all-on` 和 `official/no-correction` 对照。

## 评价与判别

逐候选唯一键为 `(image_id, annotation_id)`；检查 7I 评分与 7J/7J-N 原图评价的键集合及原 IoU 完全一致。Gate 未选中时逐值保持官方原输出；选中时使用已有修正器保存的**正常原图、原预测框**掩码结果。记录实际 val 误选成功率、覆盖率、失败捕获率、原失败救回数、原成功损伤数、净 Mask75、全部候选平均 Mask IoU 和框内 ROC AUC；按图像聚类对与官方和 all-on 的配对差给 95% 区间。还需按原成功/失败统计两种 7J-N 修正器的预测 `||Δc||`，检验 true-local 的较少损伤是否只是幅度收缩。

若 gate 只降低损伤、也等比例失去救回，或净收益区间跨零，不称为有效方法。即使形成正的候选级净收益，仍需在未参与决策的新图片上作完整推理和 COCOeval，验证 gate 对所有候选可用、分数/NMS 交互及总体 AP，才能成为论文方法证据。
