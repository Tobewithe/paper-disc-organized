# Derivation Package

## Target

为 YOLO26 prototype–coefficient mask decoder 推导“系数正交与未裁切掩码分离”的关系。先反驳无条件蕴含，再给出成立的条件性重叠上界。本文是独立代数推导，不声称新颖性，不把附加假设视为已有实验事实。

输出包括精确恒等式、显式反例、非负激活分离命题、原始 CCL 可联系到的条件性上界，以及实验核验方案。不会据此直接证明 AP 提升、FPN 因果机制或原始损失已满足假设。

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

原命题“系数正交必然使阈值掩码分离”错误。修正后：系数相似度可在 prototype 几何、符号抵消和正激活裕量受控时约束重叠面积；非负 logit 激活的正交性则与零阈值二值掩码分离精确等价。

## Invariant Object

固定一张图像、两个实例与一个共同像素区域 Ω，最终考察该区域内未裁切二值预测的重叠比例：

\[
o_{ij}=\frac1N\sum_{x\in\Omega}\mathbf1[z_i(x)>0]\mathbf1[z_j(x)>0],\qquad N=|\Omega|.
\]

这是预测间交集占 Ω 的比例，不是 mask IoU，不是相对 GT 的泄漏率，不等于分割准确率。整个推导始终以 o_ij 为目标；系数余弦、logit 内积和正激活乘积是中间对象。

## Assumptions

- 一张图像和同一解码分支内，两个实例共享 P，且 z_i=Pᵀc_i。P 和 c 在单次分析中固定，训练中可以共同变化。
- Ω 事先定义且两个实例相同，可以是全图或 GT 框并集；不同 Ω 的比例不能直接混用。
- 主推导针对 uncropped masks 与概率阈值 0.5，即 z>0；非 0.5 阈值另见边界说明。
- P 表示目标评价网格上的有效 prototype；若实际解码含 logit 的线性插值，须将该操作纳入 P，并正确还原 padding。不得把 sigmoid 后插值与 logit 插值视为相同。
- 条件上界额外要求 G 接近 αI、实际被评估的系数相似度不超过 m、符号抵消与小正 logit 比例可控。这些条件均待实测，原始 CCL 不自动保证它们。
- 不假设 sigmoid 输出非零区域可作为有限支撑集；有限 logits 经 sigmoid 后处处严格为正。

## Notation

- P∈R^{k×N}：prototype 矩阵；p_x∈R^k 为其第 x 列。
- c_i,c_j∈R^k：两个非零系数列向量；z_i=Pᵀc_i∈R^N。
- s_i=σ(z_i)：软掩码；b_i(x)=1[z_i(x)>0]：阈值 0.5 的硬掩码。
- G=PPᵀ/N：未中心化 Gram 矩阵；ρ_c=c_iᵀc_j/(||c_i||||c_j||)。ρ_c 是余弦而非中心化统计相关系数。
- u_i=[z_i]_+，v_i=[−z_i]_+，因此 z_i=u_i−v_i。
- Q_ij=u_iᵀu_j/N：共同正激活能量；D_ij=v_iᵀv_j/N：共同负激活能量。
- R_ij=(u_iᵀv_j+v_iᵀu_j)/N：异号乘积的绝对量。
- a_i(γ)=N^{-1}|{x:0<z_i(x)<γ}|，γ>0：接近零阈值的小正激活像素比例。
- α>0，ε∈[0,1)：Gram 近似各向同性参数；m∈[0,1]：实际系数余弦的上界。

## Derivation Strategy

从 o_ij 出发，使用非负正激活乘积 Q_ij 给重叠建立上界，再展开 Q_ij 与带符号 logit 内积的差别，最后联系系数和 Gram 矩阵。先处理精确关系，再引入可测的附加条件。

## Derivation Map

1. **identity**：z_iᵀz_j/N=c_iᵀGc_j，解释系数与输出几何差异。
2. **counterexample**：G=αI 也不足以从 logit 正交推出支撑分离。
3. **proposition**：Q_ij=0 当且仅当 o_ij=0。
4. **proposition**：o_ij≤Q_ij/γ²+a_i(γ)+a_j(γ)，不需要 Gram 假设。
5. **identity**：Q_ij=c_iᵀGc_j+R_ij−D_ij。
6. **conditional proposition**：若 ||G−αI||op≤αε 且 ρ_c≤m，则得到原始系数余弦与重叠的条件上界。
7. **interpretation**：上述关系提供可证伪的机制假设，不是 CCL 有效性或创新性的证明。

## Main Derivation

### Step 1. 系数欧氏几何与 logit 几何（identity）

\[
\frac{z_i^\top z_j}{N}=\frac{c_i^\top PP^\top c_j}{N}=c_i^\top Gc_j.
\]

对非零 logits，其未中心化余弦为

\[
\rho_z=\frac{c_i^\top Gc_j}{\sqrt{c_i^\top Gc_i}\sqrt{c_j^\top Gc_j}}.
\]

因此 c_iᵀc_j=0 通常不推出 z_iᵀz_j=0。若 G=αI，α>0，则 ρ_z=ρ_c。仅在这个条件下两种正交完全一致。

还存在坐标依赖性：对可逆 A，令 P'=AP、c_i'=A^{-T}c_i，则 P'ᵀc_i'=Pᵀc_i。一般 A 会改变系数欧氏余弦，却不改变 masks。这说明系数余弦不是解码函数本身的不变量；不表示网络在既定架构内一定能实现任意 A。

### Step 2. 即使 Gram 各向同性仍不能推出分离（counterexample）

取 k=2、N=3：

\[
P=\begin{bmatrix}2&1&-2\\1&2&2\end{bmatrix},\quad
c_i=\begin{bmatrix}1\\0\end{bmatrix},\quad
c_j=\begin{bmatrix}0\\1\end{bmatrix}.
\]

此时 PPᵀ=9I，所以 G=3I，系数与 logits 都严格正交：

\[
z_i=(2,1,-2)^\top,\quad z_j=(1,2,2)^\top,\quad z_i^\top z_j=2+2-4=0.
\]

阈值掩码却是 b_i=(1,1,0)、b_j=(1,1,1)，因此 o_ij=2/3，mask IoU 也为 2/3。抵消的 −4 来自一个实例为负、另一个实例为正的像素。

这排除了“仅增加 prototype 正交条件即可挽救原结论”的路线。该反例中的 P 满行秩，问题并非只发生在重复 prototype 或退化矩阵中。

### Step 3. 与支撑分离精确对应的正交对象（proposition）

**命题 1：** 在有限像素集上，Q_ij=0 当且仅当 o_ij=0。

**证明：** 对每个像素，u_i(x)u_j(x)≥0，因此有限和为零当且仅当每个乘积为零。又有 u_i(x)u_j(x)>0 当且仅当 z_i(x)>0 且 z_j(x)>0，恰好对应该像素同时属于两个硬掩码。证毕。

因此成立的精确说法是：**正 logit 激活的内积为零，与阈值 0.5 的掩码支撑不相交等价。** 它不是原始系数欧氏正交，也不是完整 sigmoid 输出正交。

可将 Q 写回系数空间。定义

\[
G^{++}_{ij}=\frac1N\sum_{x\in\Omega}\mathbf1[z_i(x)>0,z_j(x)>0]p_xp_x^\top.
\]

则 Q_ij=c_iᵀG^{++}_{ij}c_j。G^{++}_{ij} 是半正定矩阵，但依赖当前两个实例的激活门控，不能把它当成固定的 prototype Gram。这个式子是精确重写，不是免费获得的独立分离保证。

### Step 4. 从小正激活内积到小重叠（proposition）

只有 Q 小还不够：若两张图在所有像素都是很小的正 logit，Q 可任意小而 o=1。必须控制阈值附近的像素。

**命题 2：** 对任意 γ>0，均有

\[
\boxed{\quad o_{ij}\le\min\left\{1,\frac{Q_{ij}}{\gamma^2}+a_i(\gamma)+a_j(\gamma)\right\}.\quad}
\]

**证明：** 把重叠像素分为强正激活部分 H={x:z_i(x)≥γ,z_j(x)≥γ} 和剩余部分。H 中每个像素贡献至少 γ² 的正乘积，所以 |H|/N≤Q_ij/γ²。重叠像素若不属于 H，则至少一个 logit 位于 (0,γ)，剩余比例由并集上界控制为 a_i+a_j。两项相加，并利用 o≤1 即得。

若所有被激活像素都有正裕量 γ，即 a_i=a_j=0，则 o≤Q/γ²。只要求概率阈值高于 0.5 时也可以直接使用正 logit 阈值作为 γ，但那改变了评价阈值，不能替代默认 0.5 的结论。

### Step 5. 与原始系数 CCL 的联系（identity + conditional proposition）

首先精确展开：

\[
c_i^\top Gc_j=\frac{(u_i-v_i)^\top(u_j-v_j)}N=Q_{ij}+D_{ij}-R_{ij}.
\]

所以

\[
Q_{ij}=c_i^\top Gc_j+R_{ij}-D_{ij}\le c_i^\top Gc_j+R_{ij}.
\]

条件：G=αI+E，||E||op≤αε，且实际评估系数满足 ρ_c≤m。由谱范数定义，

\[
c_i^\top Gc_j
=\alpha c_i^\top c_j+c_i^\top Ec_j
\le\alpha(m+\varepsilon)\|c_i\|\|c_j\|.
\]

代入命题 2，得到：

\[
\boxed{
o_{ij}\le\min\left\{1,
\frac{\alpha(m+\varepsilon)\|c_i\|\|c_j\|+R_{ij}}{\gamma^2}
+a_i(\gamma)+a_j(\gamma)
\right\}.
}
\]

这是原始系数余弦与硬掩码重叠之间的一条成立的条件性数学关系。若不假设各向同性，可先使用更直接的精确上界：

\[
o_{ij}\le\min\{1,[c_i^\top Gc_j+R_{ij}]/\gamma^2+a_i+a_j\}.
\]

c_iᵀGc_j+R_ij=Q_ij+D_ij≥0，所以这里没有负上界问题。

**特例：** 若 m=0、ε=0、R=0、a_i=a_j=0，则 o=0。这个特例的额外条件非常强，不能默认用于实际网络。一般情况下，仅降低 m 可能使该上界的一项变小；P、系数范数、R 和 a 都会随训练变化，因此不能据此断言实际 o 必然下降。即使所有其他量固定，上界下降也不是实际值单调下降的证明。

### Step 6. 原实现的训练/推理对象差异（scope）

tools/train_ccl_true.py 对每个 GT 的多个 foreground anchors 先取系数均值，再施加余弦惩罚。因而惩罚对象是 c̄_g，而可视化和推理使用的是被选中预测的 c_a。

平均系数满足某一余弦条件，不保证每个 anchor 或另一解码分支满足该条件。上述命题成立于实际代入的 c_i,c_j；若要以训练损失解释推理结果，还需实测组内系数离散程度、选中预测的余弦，以及损失覆盖的解码分支。

原始 hinge loss [ρ−m]_+ 只是软惩罚，不保证训练后每个 pair 都有 ρ≤m。不能将设定的 margin 直接当成命题中实测余弦的上界。

## Remarks and Interpretation

- 可用于论文的理论表述：系数去相关是一种依赖 prototype 几何的间接约束；硬掩码重叠受非负激活相互作用和阈值附近质量共同控制。由此建立条件界，并用实验检验各条件是否满足。
- 若提出 prototype-aware cosine，只约束 c_iᵀGc_j 仍会受到符号抵消影响；不能把替换 Gram 当作已解决分离问题。
- 一个更直接的候选正则化是 L_pos=平均 Q_ij，但 Q 可通过缩小 logits 而降低，在阈值 0.5 下硬掩码甚至完全不变。因此必须配合自身 GT 拟合，并检查 a_i(γ)、覆盖率和置信裕量。这只是方法候选，尚无性能保证。
- 预测分离并不等于正确识别：全空、漏掉困难目标或分错位置也可使重叠下降。必须结合 GT fidelity，且先核实数据标注是否允许实例 mask 真实重叠。

## Boundaries and Non-Claims

- 不证明一般的“系数正交⇒分离”，也不证明“分离⇒高 AP”。
- 不证明 CCL 解决 FPN feature assimilation，不证明因果链“感受野重叠⇒系数相似⇒泄漏”。
- 不把数学有效当成 bound 有用：上界常可能超过 1，截断后只剩平凡界。
- 不从现有一次 +0.489 AP points 的复评得出这些额外假设成立。
- 对概率阈值 τ≠0.5，硬掩码由 z>logit(τ) 决定；正激活对象应改为 [z−logit(τ)]_+。原始 G 恒等式连接的是未平移 z，需重新展开阈值平移的常数项，不能照搬最终系数界。
- 对不同实例使用不同 crop 的结果，不应直接套用共享 P 的式子；本文件讨论 uncropped masks。crop 会额外改变支撑。
- 未作相关工作查新，不能声称这些基本恒等式、命题或正则项是首次提出。

## Open Risks

1. 在两个现有 checkpoint 上，G 是否接近各向同性？k=32 时可直接计算谱；若 rank deficient，ε<1 的全空间条件不成立，但可以继续使用无各向同性假设的 Q 或 exact-G 上界。
2. 符号抵消 R 是否很大，使 logit cosine 严重低估共同正激活？要按 ICI 和模型报告分布。
3. 在统一 logit 坐标下扫描 γ，报告 a_i、a_j、上界和实际 o；γ 与 logits 的绝对尺度有关，不应为每个测试 pair 事后选择最有利值。
4. 当比较模型时，P 与 c 的尺度及基底都可能变化，需要报告各项并核对其组合；不可只看单一余弦。
5. 使用固定、一对一 GT 匹配协议，并报告无法匹配的样本，避免仅挑可视化最成功的 pair。
6. 若界在绝大多数 pair 上都为 1，应承认它缺少解释力；以命题 1–2 和实测诊断为主，不能把空泛的条件界作为机制已证实的证据。

## Verification

反例与条件上界的数值检查脚本位于 gemini/scripts/verify_orthogonality_bounds.py。脚本结果仅辅助发现代数错误，不替代上面的证明，也不代表实际模型满足附加条件。
