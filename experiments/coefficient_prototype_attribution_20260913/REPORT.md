# S080：系数机会与固定原型限制归因

状态：COMPLETE / S074 复算。无新增前向、训练或 COCO AP。

## 判定

对原始 IoU<0.75 的实例，若固定原型下的 GT-assisted all-fit IoU 达到 0.75，记为“系数可修复”；否则记为“固定原型条件下仍受限”。后者不能单独证明原型因果，只表示即使给出全像素最优系数，当前固定 P、框和支持域仍不足以达到阈值。

## 结果

| 密度 | 失败子型 | n | 系数可修复 | 固定原型仍低于 .75 | all-fit 增益(点) | 固定原型余 gap(点) |
|---|---|---:|---:|---:|---:|---:|
| all | requires_target_pixel_recovery | 48 | 8 (16.7%) | 40 (83.3%) | 18.79 | 14.62 |
| all | sufficient_true_pixels_but_residual_false_pixels | 46 | 25 (54.3%) | 21 (45.7%) | 21.91 | 5.50 |
| high | requires_target_pixel_recovery | 24 | 4 (16.7%) | 20 (83.3%) | 17.23 | 16.57 |
| high | sufficient_true_pixels_but_residual_false_pixels | 24 | 13 (54.2%) | 11 (45.8%) | 25.62 | 2.99 |
| low | requires_target_pixel_recovery | 24 | 4 (16.7%) | 20 (83.3%) | 20.34 | 12.68 |
| low | sufficient_true_pixels_but_residual_false_pixels | 22 | 12 (54.5%) | 10 (45.5%) | 17.86 | 8.24 |

## 解释边界

两类失败同时存在。`coefficient_recoverable` 是系数读出方法的条件机会；`fixed_prototype_limited` 说明只修系数不够，但其剩余瓶颈可能来自原型表达、目标支持、低分辨率和模型外的误差，不能直接命名为原型失败。高低密度均出现相同结构，因此 S080 不支持把该归因写成密集场景独有机制。下一步若做方法，应在独立图像上学习共享系数修正，并同时报告固定原型仍受限的残差，避免把 GT oracle 增益写成可部署收益。

详见 `SUMMARY.json`、`SUMMARY.csv`、`instances_attributed.csv`；原始输入为 S074 `instances.csv`。
