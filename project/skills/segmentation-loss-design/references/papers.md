# 论文依据与阅读边界

本包核对原始论文并选择性阅读与方法有关的正文／理论段落，未逐篇全文审读、复现实验或开展计量式影响力排名。下面按“论文能支持什么”和“不能由此推出什么”区分来源结论与应用迁移。

方法正文中的YOLOSeg组合、主任务优先变体及部分代数说明是基于文献的综合或推导，文中已作标识；不能把它们作为原论文已测结果。

## BOUNDARY

**Boundary loss for highly unbalanced segmentation**

Kervadec, Hoel, Bouchtiba, Jihene, Desrosiers, Christian, Granger, Eric, Dolz, Jose, Ayed, Ismail Ben。MIDL 2019; read expanded arXiv v4。

- 阅读版本：[1812.07032v4](https://arxiv.org/abs/1812.07032v4)。
- 核读范围：§2 signed-distance reformulation and combined objective。
- 方法依据：将轮廓距离表达为有符号GT距离加权的区域概率积分，与区域loss组合。
- 结论边界：不等于任意边缘BCE；边界噪声、尺度和监督语义影响迁移。
- PDF身份：`9746e5741c759ba10e05d9b18245f4d7f580f58ae8eb4812bfc73992dde1bf40`。

## CLDICE

**clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**

Shit, Suprosanna, Paetzold, Johannes C., Sekuboyina, Anjany, Ezhov, Ivan, Unger, Alexander, Zhylka, Andrey, Pluim, Josien P. W., Bauer, Ulrich, Menze, Bjoern H.。CVPR 2021; read arXiv v7。

- 阅读版本：[2003.07311v7](https://arxiv.org/abs/2003.07311v7)。
- 核读范围：§2–4 definition, topology assumptions and soft-skeleton motivation。
- 方法依据：以预测与真值骨架覆盖定义clDice，并在特定拓扑条件下讨论保证及训练代理。
- 结论边界：主要面向管状／网络结构；不支持对任意遮挡动物可见Mask强制连通。
- PDF身份：`2e952692e01cb534b8851712e27846ab5d0dc06dfd170c9c64daf4e70d7f3170`。

## CONDINST

**Conditional Convolutions for Instance Segmentation**

Tian, Zhi, Shen, Chunhua, Chen, Hao。ECCV 2020; read arXiv v4。

- 阅读版本：[2003.05664v4](https://arxiv.org/abs/2003.05664v4)。
- 核读范围：§2.1 instance-conditioned filters and §2.3 Dice objective。
- 方法依据：动态实例参数与相对位置使共享特征产生对象特定Mask，并采用相应目标。
- 结论边界：支持实例条件的重要性，不证明任何部位／门控组合都有增量。
- PDF身份：`6c3643a3002fc887938c00fbbbf2c89d2fb3f6595deec3b70b1a1f8a59434981`。

## FOCAL

**Focal Loss for Dense Object Detection**

Lin, Tsung-Yi, Goyal, Priya, Girshick, Ross, He, Kaiming, Dollár, Piotr。ICCV 2017。

- 阅读版本：[1708.02002v2](https://arxiv.org/abs/1708.02002v2)。
- 核读范围：§3.1–3.2 weighting, easy examples and focal definition。
- 方法依据：减少大量易分类样本的相对贡献，区分类别权重与难度调制。
- 结论边界：不是普遍抑制异常噪声的鲁棒损失；原重点是密集检测。
- PDF身份：`2e6f45b97007009f32b82a3c19f35638d8da7022d8d4e4416b9078de91339f71`。

## LOSSODYSSEY

**Segmentation Loss Odyssey**

Ma, Jun。arXiv 2020 taxonomy preprint, not the later journal benchmark。

- 阅读版本：[2005.13449v1](https://arxiv.org/abs/2005.13449v1)。
- 核读范围：abstract, §1 taxonomy and §2.1 CE。
- 方法依据：整理分布、区域、边界及组合型分割目标，说明目标之间的分类联系。
- 结论边界：本次未把该预印本当作后续Medical Image Analysis基准的完整阅读或验证。
- PDF身份：`3b45605aa2847f293ee290560dfbf3f5fc8f66286339868c8fbbc66beb29fa05`。

## LOVASZ

**The Lov\'asz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks**

Berman, Maxim, Triki, Amal Rannen, Blaschko, Matthew B.。CVPR 2018。

- 阅读版本：[1705.08790v2](https://arxiv.org/abs/1705.08790v2)。
- 核读范围：§2.1–3.1 binary/multiclass surrogates and image versus dataset IoU。
- 方法依据：由Jaccard集合误差构造排序型代理，区分二值hinge与多类Softmax。
- 结论边界：batch／image代理不等于dataset IoU，更不等同COCO实例AP。
- PDF身份：`82bdd01cacbab500abab94c78d6c27e2c1f2a7fafee16c9b302f8d1225c38b1e`。

## MASKRCNN

**Mask R-CNN**

He, Kaiming, Gkioxari, Georgia, Dollár, Piotr, Girshick, Ross。ICCV 2017。

- 阅读版本：[1703.06870v3](https://arxiv.org/abs/1703.06870v3)。
- 核读范围：§3 mask target, sigmoid/CE distinction and alignment。
- 方法依据：对象条件的二值Mask与语义Softmax目标不同，强调精细空间表示。
- 结论边界：不能把所有实例目标当互斥语义图，或将ROI结果直接当YOLO结论。
- PDF身份：`b595a882b763d55d933543131f32a2c4b6f41d0818d418ef46a2b66f946b67a8`。

## MASKSCORING

**Mask Scoring R-CNN**

Huang, Zhaojin, Huang, Lichao, Gong, Yongchao, Huang, Chang, Wang, Xinggang。CVPR 2019。

- 阅读版本：[1903.00241v1](https://arxiv.org/abs/1903.00241v1)。
- 核读范围：§3.2 mask scoring, MaskIoU target and inference role。
- 方法依据：预测MaskIoU并结合分类分数排序，区分质量估计与掩码形状。
- 结论边界：不是一个通用形状loss，也不允许用测试GT给部署预测排序。
- PDF身份：`87fb301a260af6c76f4821a0298085a122289f4d16644dcba30e9a18cd5cfd97`。

## POINTREND

**PointRend: Image Segmentation as Rendering**

Kirillov, Alexander, Wu, Yuxin, He, Kaiming, Girshick, Ross。CVPR 2020。

- 阅读版本：[1912.08193v2](https://arxiv.org/abs/1912.08193v2)。
- 核读范围：§3.1–3.2 sampling and fine/coarse instance representations。
- 方法依据：组合精细特征与粗对象预测，在不确定位置细读出，讨论区域特定信息。
- 结论边界：不表示原生YOLO每种边界错误都应增加同款点头，也不保证标签粗糙时的收益。
- PDF身份：`dc544aba7f76d1c395154d638c52e6196095eb9468700828b3b35d1fa250244c`。

## TVERSKY

**Tversky loss function for image segmentation using 3D fully convolutional deep networks**

Salehi, Seyed Sadegh Mohseni, Erdogmus, Deniz, Gholipour, Ali。MLMI 2017。

- 阅读版本：[1706.05721v1](https://arxiv.org/abs/1706.05721v1)。
- 核读范围：§2.2 Tversky loss and precision/recall tradeoff。
- 方法依据：用不同FP与FN系数调节区域目标，研究病灶分割召回取舍。
- 结论边界：更大FN代价可能增加FP；不能照搬原数值或声称直接提高实例AP。
- PDF身份：`1ea63897815a4e5c9af06e7d5f73517f167d0f3fb53c74e46b68ca8d935c8673`。

## VNET

**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**

Milletari, Fausto, Navab, Nassir, Ahmadi, Seyed-Ahmad。3DV 2016。

- 阅读版本：[1606.04797v1](https://arxiv.org/abs/1606.04797v1)。
- 核读范围：§3 Dice loss layer。
- 方法依据：为体数据分割使用平方分母的可微Dice目标，处理前景比例问题。
- 结论边界：与常见线性soft Dice变体不同；医学体数据结果不直接外推动物实例AP。
- PDF身份：`f3d7093a00e3a2db384e34c918d3ef86a74f9d3339c558a87cb4e9db443464d3`。

## YOLACT

**YOLACT: Real-time Instance Segmentation**

Bolya, Daniel, Zhou, Chong, Xiao, Fanyi, Lee, Yong Jae。ICCV 2019。

- 阅读版本：[1904.02689v2](https://arxiv.org/abs/1904.02689v2)。
- 核读范围：§3.1–3.3 prototypes, coefficients and mask assembly。
- 方法依据：共享空间原型与实例系数组合Mask，支持分解表示的学习与分析。
- 结论边界：原型不自动等于器官语义；具体YOLO实现可能有差异。
- PDF身份：`223c83bd8f229beeaaf316b4eb4d4a92b1798c4ff61346d4ed142ad81e345d16`。
