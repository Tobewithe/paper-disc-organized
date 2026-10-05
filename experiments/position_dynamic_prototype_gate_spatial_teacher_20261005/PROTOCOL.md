# Protocol: pixel-logit spatial teacher for dynamic prototype readout

## 独立问题
当前 G/P 位置相关动态读出只有最终 mask BCE 间接监督。上一轮 warm-start 已证明动态路径被激活仍无效。本轮只检验：用同一官方候选的有限 oracle 产生可识别的像素 logit 修正目标，是否能让动态分支学习到有用的空间修正。

## 方法变量
保持 G/P 的网络结构、预测框、候选、固定 prototype、输入特征、数据划分、seed、3 epoch、学习率和解码不变。新增变量只有训练期辅助项：

\[
\Delta z_i^*=P_i(c_{\lambda,i}^*-c_{0,i}),\quad \lambda=0.003
\]

在固定 GT 框支持内监督动态残差 `z_dynamic-z_native`：

\[
L=L_{mask}+\beta |U_i|^{-1}\sum_{u\in U_i}
\operatorname{Huber}_\tau((z'_i-z_{native,i})_u-\Delta z^*_{i,u})
\]

固定 `beta=0.1`, `tau=1.0`; teacher 仅用于 fit 训练，评价阶段不使用 GT/oracle。native 分支仍按 warm-start 设置联合训练，但教师项作用在残差，避免把 native 输出差异误当作位置教师作用。

## 对照与数据
- A：原始 native 输出。
- N：已有同预算 native 微调 checkpoint，直接复用。
- G/P：上一轮 warm-start、仅官方 mask BCE 的冻结 checkpoint，作为无教师对照。
- G-T/P-T：本轮同结构、同预算、同初始化，增加上述 logit teacher。
- fit 1024 图片、dev 256 图片、seed 0、3 epochs；dev 不用于选择 beta 或 checkpoint。
- 有限 oracle 只按完整身份 `(image_id, annotation_id, raw_id, pyramid_level, target_gt_idx)` 绑定。若 fit 身份缺失，停止训练，不用部分身份。

## 关键核验
smoke 必须核对：oracle 身份全覆盖、初始输出与 native 一致、teacher target 有限、teacher 梯度非零、prototype/native/box 输入未被修改。评价必须复现 A 的正常原图解码。

## 主指标与继续条件
在 dev 正常原图评价，主比较为目标组（A 中 BoxIoU≥0.75 且 MaskIoU<0.75）的图片 macro IoU：P-T−P 与 G-T−G；同时看 P-T−A、Mask75 修复/损伤、coverage、AUC、FPR，以及全体 macro IoU。只有目标组至少 +0.5 pp 且全体没有明显负迁移（CI 下界不低于 −0.1 pp），并且 P-T 优于 P，才值得进入更大规模确认。否则停止本配置，不扫 beta、温度、学习率、epoch 或另加损失。

## 边界
这不是对 `a_k(u)` 通道权重的唯一监督；监督的是可识别的像素效果。generic coefficient teacher 已有阴性结果，本轮只验证 spatial residual teacher 这一不同目标，不把阳性/阴性外推到所有教师监督。
