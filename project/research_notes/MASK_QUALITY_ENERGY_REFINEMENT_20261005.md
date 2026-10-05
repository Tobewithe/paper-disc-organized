# Mask quality energy → coefficient refinement：训练方案（待执行）

## 研究问题

现有质量分支已经能对固定掩码做质量排序，并在 one-to-one 候选保留阶段带来小幅 Mask AP 提升；但它没有改变固定 raw 的 coefficient。现在检验：质量函数若显式接收候选 coefficient 和 \(Pc\)，能否学习当前 coefficient 附近的质量地形，并用其对 \(c\) 做无 GT 局部优化。

这不是回归 \(c^*\) 或 \(\Delta c^*\)，也不是把质量分数直接当分类分数重排。

## 训练对象与冻结范围

- YOLO26m-seg、Ultralytics 8.4.100、原型 \(P\)、框、类别分数、候选身份和正常解码全部冻结；
- 只训练一个质量能量头 \(Q_\phi\)；首轮不解冻原生系数头；
- 训练图片使用已有官方 fit 清单，独立 dev 清单只作最终评价；
- oracle coefficient 只用于 fit 阶段构造扰动和质量标签，推理阶段不输入 GT、oracle 或 GT 框。

## 质量头输入

对一个候选，保留可微路径：

\[
z(c)=P c.
\]

质量头输入为：

1. 原实例特征/系数分支描述（沿用已有质量分支输入）；
2. 归一化预测框和尺度 embedding；
3. 当前 candidate coefficient \(c\) 与 \(c-c_0\)；
4. \(z(c)\) 在固定 prototype 网格上的 \(8\times8\) 平均响应；
5. 原始类别/目标分数描述。

\(P\)、\(c\) 和 \(z(c)\) 在质量头内部不能 detach，否则无法得到 \(\nabla_c Q_\phi\)。GT 只参与标签生成。

## 训练样本如何生成

每个 fit 候选使用已有有限 oracle 方向 \(d^*=c^*-c_0\)，固定生成以下 candidate 集：

\[
 c_0,\quad
 c_0+0.25d^*,\quad
 c_0+0.5d^*,\quad
 c_0+0.75d^*,\quad
 c_0+d^*,\quad
 c_0+r u_1,\quad
 c_0+r u_2,\quad
 c_0-0.5d^*,
\]

其中 \(r=\lVert d^*\rVert\)，\(u_1,u_2\) 是固定随机单位方向。随机方向控制质量头是否只是学会“修正幅度越大越好”。若 \(r\) 极小，保留 \(c_0\) 与随机控制，不放大该实例。

每个 candidate 用固定预测框、完整 prototype 和原始 GT 计算两个标签：

- 连续 soft-IoU：用于稳定回归；
- 正常二值解码后的 Mask IoU：用于同一实例内排序监督。

## 损失

质量头输出标量 \(\hat q_\phi\)。首轮固定：

\[
L_Q=L_{Huber}(\hat q,q_{soft})+
0.5\,L_{rank}.
\]

对同一实例的 candidate 对 \((a,b)\)，若真实 \(q_a>q_b\)，使用：

\[
L_{rank}=\max(0,\,0.02-(\hat q_a-\hat q_b)).
\]

不加入新的 mask BCE、Dice、负像素权重或 teacher loss，避免再次把监督对象混回原分割目标。训练 batch 内实例权重相等，不能让大框/大像素实例主导质量头。

## 推理时怎样改 coefficient

对每个原始候选从 \(c^{(0)}=c_0\) 开始，计算：

\[
E(c)=Q_\phi(F,B,Pc,c)-\frac{\lambda}{2}\lVert c-c_0\rVert^2.
\]

首轮主结果固定为一步：

\[
g=\nabla_c E(c^{(0)}),\qquad
c^{(1)}=c_0+\eta\frac{g}{\lVert g\rVert+10^{-8}}.
\]

\(\eta\) 在 fit 上预先固定为 oracle 位移范数的中位数的 0.5 倍；不使用 dev 调参。两步版本只作为预注册的次要诊断，第二步步长减半，不扫描步数、步长或 gate。

最终用原始 \(P c^{(1)}\) 走完整正常 mask 解码。质量头不参与类别分数和候选身份改变，先隔离“改 coefficient”的效应。

## 对照与评价

固定三组：

1. A：原始 \(c_0\)；
2. Q-rank：已有质量头只做固定 candidate quality 排序，不更新 \(c\)；
3. Q-grad：本方案对 \(c\) 做一步质量梯度更新。

另保留 GT oracle 作为诊断上界，不能当方法成绩。

主指标按正常原图、图片 macro 配对统计：

- Box75 且原 Mask<.75 的目标组 Mask IoU；
- 全体候选/图片 macro Mask IoU；
- Mask75 修复/损伤；
- coverage、AUC、FPR；
- 单候选额外推理时间。

同时报告质量头本身的 Spearman、pairwise accuracy，以及 \(Q(c^{(1)})-Q(c_0)\) 与真实 IoU 变化的一致率。质量分数预测得准但梯度更新无效，必须单独记为失败。

## 预先停止条件

质量头进入下一阶段的最低条件：held-out candidate pairwise accuracy ≥65%，且质量梯度一步使真实 Mask IoU 提升的比例高于 55%。

方法继续条件：目标组图片 macro IoU 相对 A 至少 +0.5 个百分点且图片 bootstrap 95% CI 下界 >0；全体相对 A 不出现 CI 下界低于 −0.1 个百分点的明显负迁移；Q-grad 必须优于 Q-rank 和已有 direct residual 对照。

若质量预测有效但梯度无效，停止“质量地形→系数更新”而保留质量排序结论；若训练集有效、held-out 无效，判断为质量地形泛化不足；若目标组改善但成功组损伤明显，不自动加 gate，直接停止当前配置并报告适用范围。

本方案尚未执行，当前只作为下一轮独立 Study 的训练协议。

## 冻结边界与阶段安排（最终锁定）

### 阶段 I：机制判别

官方 YOLO26m-seg 完全冻结：backbone、neck、分类头、box 头、prototype 分支、原生 coefficient 分支、BN affine 参数和 running statistics 全部冻结。每个候选固定产生 (P,h,c_0,B,class)。

Direct32 与 Q-Refine 共享同一个新增 evidence encoder、同一输入、同一训练图片、同一候选、同一训练预算和同一 trust radius。两组的唯一变量是：

- Direct32：(e
ightarrowDelta c)；
- Q-Refine：(e
ightarrow Q(F,B,Pc,c))，推理时通过 (
abla_c Q) 更新临时 coefficient。

推理中的 (c_1,c_2) 是实例级临时变量，不会更新 YOLO 参数。阶段 I 主比较为 Original、Direct32、Q-Refine；Q-Rank只作质量排序次要对照。

### 阶段 II：有限表示适配

只有阶段 I 在严格 MaskFail 目标组和正常原图上确认 Q-Refine 相对 Direct32 的稳定增益，才登记阶段 II。阶段 II 只解冻 one-to-one coefficient branch，继续冻结 backbone、neck、prototype、box 和 class 分支。比较 Direct32 与 Q-Refine 的有限解冻版本，判断表示适配与质量地形优化是否互补。

当前不允许全模型微调，也不允许解冻 prototype；否则无法保持本轮因果对照。


## 当前执行状态（2026-10-05）

已将阶段 I 锁定为 20,000 FIT 图片、2,000 DEV 图片、COCO val2017 5,000 FINAL；FIT 预期 144,995 个普通实例，DEV 预期 13,730 个。Failure/Success 按实例近似 1:1，目标至少各 20,000，达到目标后不因结果扩容。AdamW `lr=3e-4`、`weight_decay=1e-4`、无 scheduler、梯度累积 8 图、3 epoch、seed 0，epoch 3 固定为主 checkpoint；Direct32 与 Q-Refine 同数据和预算。

执行包已写入 `experiments/mask_quality_energy_refinement_20261005/`，包括 `prepare_qcr_cache.py`、`select_candidates.py`、`build_oracle_selected.py`、`qcr_train.py`、配置与代码 SHA 清单。训练不在桌面运行。指定 3080Ti 服务器 `connect.bjb2.seetacloud.com:33953` 当前端口拒绝连接，因此尚未创建实际训练 Run；待服务器可达后先做缓存/身份预检和 2 图 smoke，再启动独立 Direct32 与 Q-Refine Run。此状态是执行阻塞，不是方法阴性。
