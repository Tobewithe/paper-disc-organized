# 分割损失：定义、偏好与证据

设有效监督中的像素概率p_i、二值目标y_i。下面公式用于说明学习偏好，不规定所有任务都采用相同归一化或权重。论文引用与阅读范围见[论文卡片](papers.md)。

## BCE、CE与Focal

二值BCE为：

\[
L_{BCE}=-\frac1N\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
\]

它来自Bernoulli似然，直接惩罚局部概率错误。多类CE对应互斥类别的分类分布。Mask R-CNN对选定实例／类别使用二值Mask目标，并不让不同类别Mask在每个像素进行softmax竞争。[MASKRCNN] 这为区分实例Mask与语义图提供明确先例。

Focal定义p_t为正确类别的概率：

\[
L_{FL}=-\alpha_t(1-p_t)^\gamma\log p_t.
\]

γ抑制容易样本的贡献，α控制类别权重；二者作用不同。[FOCAL] 原研究针对密集检测极端正负不均衡。一个正实例裁剪区域的Mask loss未必有同样比例。原文明确指出它与降低异常点影响的鲁棒损失方向不同，因此困难标注噪声可能被强调。

## Dice与IoU：区域比值

一种常见soft Dice为：

\[
D=\frac{2\sum_i p_i y_i+\epsilon}{\sum_i p_i+\sum_i y_i+\epsilon},\quad L_D=1-D.
\]

V-Net原文使用分母Σp_i²+Σy_i²的可微版本。[VNET] 平方／线性分母和平滑项具有不同梯度，不能把它们说成一个完全相同的loss。

一种soft IoU为：

\[
J=\frac{\sum_i p_i y_i+\epsilon}{\sum_i p_i+\sum_i y_i-\sum_i p_i y_i+\epsilon},\quad L_J=1-J.
\]

区域比值减少单纯像素计数的影响，但把像素梯度通过分母耦合。逐实例平均通常更接近每个实例同权；合并全图／全batch时大实例可能贡献更多。空目标时平滑和定义决定惩罚，不能假定任意Dice变体都能有效压制假阳性。

BCE＋Dice之所以有动机，在于局部概率与区域重叠的互补。这里只给出候选理由，不把固定1:1或任意权重说成公认配方。[LOSSODYSSEY]提供分布型、区域型、边界型和组合型目标的分类；本次读取的是2020综述预印本，不借用未核读的后续基准结论。

## Tversky：代价不对称

\[
T=\frac{TP}{TP+\alpha FP+\beta FN},\quad L_T=1-T,
\]

其中TP=Σp_i y_i，FP=Σp_i(1-y_i)，FN=Σ(1-p_i)y_i；平滑可按所用变体加入。按这里约定，增大β相对α更重视漏分。[TVERSKY] α=β=0.5与线性soft Dice相关，但不等同所有平方分母变体。

**适合：** 风险定义确实要求更高召回，如小病灶或必须保留细部。**可能损伤：** 更容易把邻猪／背景纳入；概率校准与AP排序也可能变化。只看FN下降不够。可见标签不应借此变成补全完整身体。

## Lovász：排序型Jaccard代理

Lovász利用Jaccard集合误差的结构，把排序后的误差按集合增量加权，得到可优化代理。[LOVASZ] 二值Lovász-hinge基于logit margin，多类Lovász-Softmax基于类别概率；独立实例的二值掩码通常应先考虑二值问题，不能直接把所有实例当互斥语义类别。

其重要价值是直接考虑重叠度量的非分解结构。原文也区分image／batch与dataset mIoU，并指出随机小批次代理不直接等于全数据IoU。因此它更不构成COCO AP的直接优化证明。与BCE／soft IoU的差别应从误差排序与权重理解，而不是“loss更高级”。

## Boundary loss：距离信息

Kervadec等使用GT边界的有符号距离φ_G，内部为负、外部为正：

\[
L_B=\int_\Omega \phi_G(q)p(q)dq.
\]

它来自轮廓距离的区域表达，可与区域目标组合。[BOUNDARY] 这不是“只在一圈边缘做BCE”，也不是所有边界loss的统一名称。距离单位、裁剪与尺度改变有效权重；GT轮廓误差会成为强监督。其值可以为负，不应据此说训练坏了。

若问题是两实例接触边界，前景／背景边缘与实例间边缘不是同一目标。需要定义本实例与邻实例的对应；整图前景边缘可能完全看不到猪—猪分界。

## clDice与拓扑先验

clDice将预测骨架落入GT的比例与GT骨架落入预测的比例取调和平均，并发展soft-skeleton训练。[CLDICE] 理论结论有特定二值／骨架与拓扑假设；一个soft训练目标下降不保证任意神经网络输出满足拓扑。

原研究聚焦管状／网络结构。遮挡动物的可见Mask可以是多块，整图猪群也不应连成网络。因此不能把clDice或“强制单连通”当YOLOSeg通用改进。只有目标拓扑本身有依据时，才让该先验进入方法选择。

## 为什么IoU好看而AP可能不变

实例AP还依赖对象是否被提出、分类与排序、重复预测、匹配和各IoU阈值上的精度召回。Mask Scoring R-CNN通过预测MaskIoU来校正分类分数，展示形状质量与打分并非同一件事。[MASKSCORING] 它的目标不是一个通用Mask形状loss，也不能替代形状学习。

因此先选择应改善的学习对象，再看该对象是否限制当前任务。把理论上的代理偏好与实际AP收益分别论证。

[BOUNDARY]: https://arxiv.org/abs/1812.07032v4
[CLDICE]: https://arxiv.org/abs/2003.07311v7
[FOCAL]: https://arxiv.org/abs/1708.02002v2
[LOSSODYSSEY]: https://arxiv.org/abs/2005.13449v1
[LOVASZ]: https://arxiv.org/abs/1705.08790v2
[MASKRCNN]: https://arxiv.org/abs/1703.06870v3
[MASKSCORING]: https://arxiv.org/abs/1903.00241v1
[TVERSKY]: https://arxiv.org/abs/1706.05721v1
[VNET]: https://arxiv.org/abs/1606.04797v1
