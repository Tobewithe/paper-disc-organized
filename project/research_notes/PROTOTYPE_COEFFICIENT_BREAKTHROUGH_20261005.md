# 原型到系数链路：突破当前困境的判断与下一轮入口

## 当前判断

当前困境不是“没有更好的 coefficient”，而是已有 GT oracle 方向在无 GT 前向中无法稳定兑现。最近的证据已经把问题范围收窄：

| 证据 | 支持/排除 |
|---|---|
| 严格 argmax-MaskFail 全量 oracle | 6,459 个严格失败中 4,263 个（66.00%）至少有一个 raw 在固定 prototype 下可由 coefficient 达到 Mask75；固定 P/解码仍受限约 33.58% |
| 同 GT 多 raw 位置 | oracle 最大值与中位数平均只差 1.293 个百分点；只训练位置选择器不能解释主要差距 |
| OGPS/ownership→solver | GT ownership 可兑现，但现有 GT-free ownership 经 solver 仍低于原模型；不能继续重复 ownership 投影 |
| 直接空间/框读出 | true box 有输入依赖，但联合训练和空间 logit teacher 没有产生正常掩码实质收益 |
| 空间 logit teacher | teacher loss 明显下降，P-T 相对 P 只有 +0.013 pp 左右，主要失败组 CI 跨零；“监督不够”不是充分解释 |

因此下一轮不能再是“换一个 teacher、再加一个 gate、再拼 prototype 统计”或延长同一训练。它们都仍然要求网络直接预测高维 `delta-c`/`delta-z`，没有改变 prototype 到 coefficient 的函数约束。

## 建议突破：功能空间信赖域读出

把网络预测目标从“直接给出 32 维系数修正”改为“提出少量像素响应证据，由固定 prototype 几何把证据转换为系数修正”。设原始响应为 (z_0=P^Tc_0)，模型只输出低维的候选 logit 修正 (d_	heta) 以及修复/保护权重；最终修正由确定性算子得到：

\[
\delta c=\arg\min_d
 \|W_E^{1/2}(P^Td-d_\theta)\|_2^2
 \lambda_R\|W_R^{1/2}P^Td\|_2^2
 \lambda_c\|d\|_2^2,
\qquad c'=c_0+d.
\]

其中 (W_E) 只强调当前 mask 的低 margin/不确定位置，(W_R) 保护当前已经可靠的前景和背景；两者第一版由 (z_0)、预测框和固定原型确定，不读取 GT。求解器显式保证修正属于当前 (P) 的可实现空间，并且把“修多少”和“不要伤哪里”分开。

这和已停止方案的区别必须保留：

- OGPS 是 ownership 图直接投影，失败在 ownership 预测本身；
- 空间 teacher 是把 dense Δz 直接回归出来，失败在高维目标和泛化；
- 本方案预测低维证据，**由 (P) 的功能几何完成投影和保护**，网络不直接承担矩阵求逆和全像素一致性。

加权最小二乘、动态核和 masked attention 都有文献先例，因此不能把 solver 或名称本身写成原创。潜在贡献只能来自：严格失败对象上的机制证据、prototype 几何约束如何减少误伤，以及正常原图上的实用收益。

## 先做不训练的判别，不直接烧一轮训练

第一步应使用已有的 oracle target（优先复用 `position_dynamic_prototype_gate_spatial_teacher_20261005/assets/ORACLE_fit.pt` 与已审计的官方候选缓存）做一个**功能投影审计**：

1. 用 (z_0) 的不确定性构造 (W_E)，用高 margin 区域构造 (W_R)；
2. 将已有 (Δc^*) 或 (Δz^*=P^TΔc^*) 投影到上述信赖域/低秩子空间；
3. 与 full oracle、无保护投影、随机同维子空间比较；
4. 记录 oracle IoU/AUC/coverage、保护区 logit 变化和 (Δc) 范数。

第一阶段只问：**在不增加网络信息的情况下，功能空间约束能否保留大部分 oracle 收益，同时减少对原本正确区域的改动？**

## 预先固定的放行条件

- 若功能约束在至少 80% 的 oracle IoU 增益上保持，并且保护区扰动相对无保护投影下降 ≥20%，才进入短预算训练；
- 若与普通投影/直接 delta-c 没有稳定差异，停止，不再训练新头；
- 若只在 GT 选定区域成立，或完整原图解码不成立，记为 oracle 几何现象，不放行方法训练。

## 训练阶段（只有通过判别审计才启动）

保持现有 G/P 的特征来源、候选身份、原型、训练图片和预算不变，比较三组：

1. 原生/普通 residual 对照（复用已有结果）；
2. 直接 spatial-logit teacher 对照（复用本轮阴性）；
3. 信赖域读出：网络只预测低维 (d_	heta,W_E/W_R)，固定求解器输出 (c')。

评价以正常原图为准：目标严格 MaskFail 组图片 macro IoU、全体 macro IoU、Mask75 修复/损伤、coverage、AUC、FPR 和增量时延。放行门槛固定为目标组至少 +0.5 pp 且 bootstrap 下界大于 0；全体不能出现超过 −0.1 pp 的明显负迁移，并必须优于同预算直接 residual 对照。

## 如果仍失败

停止 prototype→coefficient 的可学习纠错方法线，不再补 teacher/gate/solver。转向当前固定 (P)/解码仍受限的 33.58% 对象，研究高分辨率 prototype 或局部细节表示；这才是证据要求改变上游表示，而不是继续调整系数读出。

本记录是研究决策与待登记协议，不代表该方法已经运行或有效。
