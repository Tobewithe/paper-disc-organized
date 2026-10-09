# Quality-guided coefficient refinement（QCR）执行协议

## 目标

比较两种在同一冻结 YOLO26m-seg 表示上的学习形式：

\[
\text{Direct32: }e\rightarrow\Delta c
\qquad\text{vs}\qquad
\text{Q-Refine: }e\rightarrow Q(c)\rightarrow\nabla_cQ\rightarrow\Delta c.
\]

目标不是证明质量分数本身可预测，而是检验学习质量地形后，是否能在每个实例自己的 prototype 空间内改进 coefficient。

## 数据

- COCO train2017：20,000 张 fit 图片；
- COCO train2017：2,000 张独立 dev 图片；
- COCO val2017：5,000 张最终评价图片；
- 图片级划分，禁止按 candidate 随机拆分；
- 训练期有限 oracle 只用于 fit 图片构造扰动和质量标签；
- dev/val 的 GT 不进入模型输入、步长或半径选择。

训练开始前保存三份冻结图片清单和 SHA256。

## 阶段 I：完全冻结原模型

冻结：

- backbone、neck；
- 分类头、box 头；
- prototype 分支；
- 原生 one-to-one coefficient 分支；
- BN affine 参数与 running statistics。

原模型固定产生 \(P,h,c_0,B,class\)。只训练新增 evidence encoder 和 Direct32/Q-Refine 分支。

Direct32 与 Q-Refine 必须共享 evidence encoder、图片、candidate、训练预算、参数量级和 trust radius。

### Direct32

\[
\hat c=c_0+\Delta c_\theta(e),\qquad
L_D=1-\operatorname{SoftIoU}(P\hat c,Y)
\frac{0.003}{2}\|\Delta c_\theta\|^2.
\]

不使用 \(c^*\) 作为 coefficient L2 标签；输出限制在固定 \(\rho\) 内。

### Q-Refine

质量函数输入 \(e,F,B,c,Pc\)，其中 \(c\) 到 \(Pc\) 的计算图保持可微：

\[
Q_\phi(F,B,c,Pc)\rightarrow \hat q.
\]

每个 fit candidate 使用有限 oracle \(d^*=c^*-c_0\) 构造：

\[
c_0,\;c_0+0.25d^*,\;c_0+0.5d^*,\;c_0+0.75d^*,\;c^*,
\;c_0-0.5d^*,\;c_0+r u_1,\;c_0+r u_2.
\]

每个状态计算 soft-IoU 和正常二值解码 Mask IoU。损失固定为：

\[
L_Q=L_{\mathrm{Huber}}(\hat q,q_{\mathrm{soft}})
0.5L_{\mathrm{rank}},
\]

其中 \(L_{\mathrm{rank}}\) 只比较同一实例的状态排序。

推理从 \(c^{(0)}=c_0\) 开始，固定两步：

\[
E(c)=Q_\phi(F,B,c,Pc)-\frac{\lambda}{2}\|c-c_0\|^2,
\]
\[
c^{t+1}=c^t+\eta_t
\frac{\nabla_cE(c^t)}{\|\nabla_cE(c^t)\|+\epsilon}.
\]

\(\rho\) 取 fit oracle 位移范数中位数；总位移限制为 \(\|c^t-c_0\|\le\rho\)，第一步 \(\eta_0=\rho/2\)，第二步 \(\eta_1=\rho/4\)。不使用 dev/val 扫描步长、半径或迭代次数。

## 阶段 I 对照

- A：原始 \(c_0\)；
- B：Direct32；
- C：Q-Rank，同一质量函数只排序，不更新 \(c\)；
- D：Q-Refine；
- Oracle：GT 辅助上界，不是部署方法。

主机制比较是 D−B。C 只判断质量排序本身的作用。

## 评价

在正常原图解码下报告：

- 严格 MaskFail 目标组图片 macro Mask IoU；
- 全体图片 macro Mask IoU；
- Mask75 修复/损伤；
- coverage、AUC、FPR；
- small/medium/large 分层；
- 额外推理时间。

同时报告 Q 的 held-out pairwise accuracy、Spearman 以及 \(Q(c^{1})-Q(c^0)\) 与真实 IoU 增量的一致率。

阶段 I 放行条件：

1. Q held-out pairwise accuracy ≥65%；
2. Q 梯度一步使真实 Mask IoU 上升的比例 >55%；
3. D−B 在严格 MaskFail 目标组达到至少 +0.2 pp，图片 bootstrap 95% CI 下界 >0；
4. D 相对 A 的目标组达到至少 +0.5 pp，且全体 CI 下界不低于 −0.1 pp。

任一关键条件失败，停止当前阶段，不扫超参、不延长训练拯救。

## 阶段 II：有限解冻

只有阶段 I 的 D−B 通过后才启动。保持数据、输入、预算和对照不变，只解冻 one-to-one coefficient branch；继续冻结 backbone、neck、prototype、box 和 class 分支。比较：

- frozen Direct32 / Q-Refine；
- coefficient branch 可训 + Direct32；
- coefficient branch 可训 + Q-Refine。

阶段 II 仍不解冻 prototype。

## 阶段 III：上游表示（条件性）

只有阶段 II 显示 Q-Refine 与 coefficient representation 适配互补，才允许逐层解冻更上游系数分支。每次只解冻一个明确层级，并保留原始冻结对照；不直接全模型微调。

若阶段 I/II 均不能建立 D−B 的实质收益，则停止 prototype→coefficient 可学习纠错线，转向固定 \(P\)/解码仍受限的高分辨率 prototype/局部细节问题。

本协议已登记，尚未启动远端训练。

## 训练规模与实例采样（最终锁定）

本轮正式机制判别预算固定为：

- train2017：20,000 张 FIT 图片；
- train2017：2,000 张 DEV 图片；
- val2017：5,000 张 FINAL 图片。

按 image_id 划分，禁止同图 candidate 跨 FIT/DEV。val2017 不参与训练、checkpoint、步长、trust radius 或方案选择。

FIT 中按实例构造：

- Failure：argmax class=GT、BoxIoU≥0.75、MaskIoU<0.75；
- Success：argmax class=GT、BoxIoU≥0.75、MaskIoU≥0.75。

Failure 与 Success 按实例近似 1:1 采样，而不是按图片强制 1:1。目标至少获得 20,000–30,000 个 Failure 及数量相当的 Success；20,000 张图片达到目标后不继续扩大，未达到时只按预先目标增加图片，不根据结果临时扩容。

Failure 使用已有有限 oracle 构造 coefficient states。Success 不强制全部求 oracle，只围绕 (c_0) 生成固定小扰动，监督质量头识别“当前已经足够好，不应乱改”。所有采样清单、Failure/Success 数量和 oracle 覆盖率在训练前冻结。

20,000 张 FIT 是本轮机制判别预算；阴性结果不自动升级到 full train2017。只有阶段 I 明显阳性，才另登记更大规模的正式性能确认。


## 训练运行锁定（启动前冻结）

- **FIT/DEV/FINAL**：20,000 / 2,000 / 5,000 张图片，按 `image_id` 分割；FIT 预期 144,995 个普通 COCO 实例，DEV 预期 13,730 个，实际有效候选和 Failure/Success 数量以缓存与清单为准。训练前保存 `IMAGE_SPLIT.json`、候选清单及 SHA256。
- **Stage I 可训练范围**：原始 YOLO 不加载为可训练对象；其 P、h、c0、预测框、类别输出和 BN 状态作为只读缓存。仅训练新增 EvidenceEncoder 与对应 Direct32 或 QualityHead。这样两臂的变化只来自“直接输出 Δc”与“学习 Q(c) 后按 ∇cQ 更新”，不混入原生分支重训。
- **优化器/预算**：AdamW，学习率 `3e-4`，weight decay `1e-4`，无 scheduler，梯度裁剪 5.0，按 8 张图片的梯度累积更新；每臂 3 个完整 FIT epoch，seed 0。Direct32 与 Q-Refine 使用相同图片顺序、批累积、更新数和半径。只保存 epoch 1–3；Stage I 主评价固定 epoch 3，不以 DEV 结果事后挑 checkpoint。
- **质量标签与 oracle**：Failure 仅使用 FIT 上已经固定身份的有限系数 oracle 构造 8 个状态；Success 使用预先固定的随机小扰动，不求逐实例 oracle。`rho` 是 FIT Failure oracle 位移范数中位数，计算一次后冻结。
- **公平对照**：A 为原始 c0；B 为同一 EvidenceEncoder、同一半径和训练预算的 Direct32；D 为同一 EvidenceEncoder、同一输入和训练预算的 Q-Refine；C 为 D 的质量排序次要诊断。两种新增头的参数量、输入和数据状态在 `MODEL.json` 中记录，不把质量头的标量输出误称为等价参数量。
- **逐级解冻**：Stage I 阴性时不加数据、不扫学习率/半径、不延长 epoch；只有达到协议中的正向证据后才登记 Stage II，且仅解冻原生 one-to-one coefficient branch。Stage II 仍冻结 prototype、backbone、neck、box、class；只有 Stage II 明确显示上游表示限制，才登记单层级上游解冻 Run。
- **checkpoint/seed/失败保留**：每个 seed 与每次重试独立 Run；异常、超时或部分产物保留并写入 `FAILURE.json`，不纳入效用比较。Stage I 首轮 seed 0，只有放行条件满足才增加 seed 1/2。

