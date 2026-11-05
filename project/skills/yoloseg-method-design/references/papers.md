# 论文依据与阅读边界

本包核对原始论文并选择性阅读与方法有关的正文／理论段落，未逐篇全文审读、复现实验或开展计量式影响力排名。下面按“论文能支持什么”和“不能由此推出什么”区分来源结论与应用迁移。

方法正文中的YOLOSeg组合、主任务优先变体及部分代数说明是基于文献的综合或推导，文中已作标识；不能把它们作为原论文已测结果。

## ADAIN

**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**

Huang, Xun, Belongie, Serge。ICCV 2017。

- 阅读版本：[1703.06868v2](https://arxiv.org/abs/1703.06868v2)。
- 核读范围：§5 Adaptive Instance Normalization and §6.1 architecture。
- 方法依据：以风格特征统计调制内容特征的逐通道均值与方差。
- 结论边界：原证据来自风格迁移；用于检测分支统计适应属于方法迁移，不是现成效用结论。
- PDF身份：`31290e56ec08b58f7297ffef87dd0f1a62a88935fb61836b4a31d7eb6b7ae258`。

## AT

**Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer**

Zagoruyko, Sergey, Komodakis, Nikos。ICLR 2017。

- 阅读版本：[1612.03928v3](https://arxiv.org/abs/1612.03928v3)。
- 核读范围：§3 activation-based attention and transfer objective。
- 方法依据：由激活统计构造空间注意图，归一化后迁移教师关注结构，并允许不同通道表示。
- 结论边界：空间激活对应不等于实例身份、因果解释或当前分割收益。
- PDF身份：`e4f13e6dda75d070d4524e2e8fac3074b3b7af3744ed5d66dbbe39170b067be8`。

## BOUNDARY

**Boundary loss for highly unbalanced segmentation**

Kervadec, Hoel, Bouchtiba, Jihene, Desrosiers, Christian, Granger, Eric, Dolz, Jose, Ayed, Ismail Ben。MIDL 2019; read expanded arXiv v4。

- 阅读版本：[1812.07032v4](https://arxiv.org/abs/1812.07032v4)。
- 核读范围：§2 signed-distance reformulation and combined objective。
- 方法依据：将轮廓距离表达为有符号GT距离加权的区域概率积分，与区域loss组合。
- 结论边界：不等于任意边缘BCE；边界噪声、尺度和监督语义影响迁移。
- PDF身份：`9746e5741c759ba10e05d9b18245f4d7f580f58ae8eb4812bfc73992dde1bf40`。

## CONDINST

**Conditional Convolutions for Instance Segmentation**

Tian, Zhi, Shen, Chunhua, Chen, Hao。ECCV 2020; read arXiv v4。

- 阅读版本：[2003.05664v4](https://arxiv.org/abs/2003.05664v4)。
- 核读范围：§2.1 instance-conditioned filters and §2.3 Dice objective。
- 方法依据：动态实例参数与相对位置使共享特征产生对象特定Mask，并采用相应目标。
- 结论边界：支持实例条件的重要性，不证明任何部位／门控组合都有增量。
- PDF身份：`6c3643a3002fc887938c00fbbbf2c89d2fb3f6595deec3b70b1a1f8a59434981`。

## CORAL

**Deep CORAL: Correlation Alignment for Deep Domain Adaptation**

Sun, Baochen, Saenko, Kate。ECCV Workshops 2016。

- 阅读版本：[1607.01719v1](https://arxiv.org/abs/1607.01719v1)。
- 核读范围：§3.1–3.2 covariance objective and joint discriminative learning。
- 方法依据：用协方差距离匹配源与目标域，同时保留判别目标，原文指出只对齐可导致退化。
- 结论边界：二阶统计一致不保证类别条件、实例关系或目标域准确率一致。
- PDF身份：`9691e1d6b20e0d5b75f580076516ff8a2e6a2ea3529eeb3e4ca3dd33b85e1ccc`。

## DANN

**Domain-Adversarial Training of Neural Networks**

Ganin, Yaroslav, Ustinova, Evgeniya, Ajakan, Hana, Germain, Pascal, Larochelle, Hugo, Laviolette, François, Marchand, Mario, Lempitsky, Victor。JMLR 2016。

- 阅读版本：[1505.07818v4](https://arxiv.org/abs/1505.07818v4)。
- 核读范围：§3.3 target-risk bound; §4.2 domain-adversarial objective。
- 方法依据：通过域判别与特征对抗目标学习域不变表示；理论同时含联合可达误差条件。
- 结论边界：不支持把任意两模块或不相容标签强行对齐，也不自动授权任何对抗训练。
- PDF身份：`39349d71ca79d44b9cfb24be39ceb5802d58f4e004287655b7df49337938bb59`。

## FILM

**FiLM: Visual Reasoning with a General Conditioning Layer**

Perez, Ethan, Strub, Florian, de Vries, Harm, Dumoulin, Vincent, Courville, Aaron。AAAI 2018。

- 阅读版本：[1709.07871v2](https://arxiv.org/abs/1709.07871v2)。
- 核读范围：§2.1–2.2 conditioning and visual reasoning model。
- 方法依据：条件输入生成逐通道缩放和平移，对视觉表示进行特征级调制。
- 结论边界：支持条件化形式；未证明任意对象条件都可靠或能处理空间错配。
- PDF身份：`765acea6f2a78b1d76961392099cd0af49cb2e7d6f06d0b3d82f1dbe9cdcd363`。

## FITNETS

**FitNets: Hints for Thin Deep Nets**

Romero, Adriana, Ballas, Nicolas, Kahou, Samira Ebrahimi, Chassang, Antoine, Gatta, Carlo, Bengio, Yoshua。ICLR 2015。

- 阅读版本：[1412.6550v4](https://arxiv.org/abs/1412.6550v4)。
- 核读范围：§2.2 hint-based training。
- 方法依据：用教师中间层提示与学生回归器训练较深较窄网络，讨论提示层选择及过度正则化。
- 结论边界：支持投影后表征提示的原理；不证明任意两模块应逐值匹配，也未测本包的YOLOSeg组合。
- PDF身份：`584ff258c88dd5b992b1df7977d856189840cdf89b0ff93331a69054f5f5f9c7`。

## FPN

**Feature Pyramid Networks for Object Detection**

Lin, Tsung-Yi, Dollár, Piotr, Girshick, Ross, He, Kaiming, Hariharan, Bharath, Belongie, Serge。CVPR 2017。

- 阅读版本：[1612.03144v2](https://arxiv.org/abs/1612.03144v2)。
- 核读范围：§3 top-down and lateral construction。
- 方法依据：以自顶向下路径和侧向投影融合不同层级，给细尺度带来语义信息。
- 结论边界：不证明再加任意尺度路径或额外颈部必然改善当前YOLO实例分割。
- PDF身份：`3a33e7ad38c17037624defde57f380f49dee918c1c14abd5dbcad032bae91428`。

## GRADNORM

**GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks**

Chen, Zhao, Badrinarayanan, Vijay, Lee, Chen-Yu, Rabinovich, Andrew。ICML 2018。

- 阅读版本：[1711.02257v4](https://arxiv.org/abs/1711.02257v4)。
- 核读范围：§3.1–3.2 gradient norm targets and algorithm。
- 方法依据：根据相对学习速度调整共享层上的加权梯度范数，区分大小与速度。
- 结论边界：不直接处理方向冲突，也不决定主任务优先级或保证验证指标。
- PDF身份：`e72930c1f875396f16955748e5af9f2b14c9365c50c1e055cbc36612f326a726`。

## INVARIANT

**On Learning Invariant Representation for Domain Adaptation**

Zhao, Han, Combes, Remi Tachet des, Zhang, Kun, Gordon, Geoffrey J.。ICML 2019。

- 阅读版本：[1901.09453v2](https://arxiv.org/abs/1901.09453v2)。
- 核读范围：§4.1 invariant-representation counterexample。
- 方法依据：构造边缘特征分布完全相同但标签关系不同、联合误差很大的反例。
- 结论边界：说明域不变性非充分条件，不意味着所有域适应无用。
- PDF身份：`9d359c0d73cff25ceb8c8bae52894d7d9d85181de74e98e3343359680490f7cd`。

## L2SP

**Explicit Inductive Bias for Transfer Learning with Convolutional Networks**

Li, Xuhong, Grandvalet, Yves, Davoine, Franck。ICML 2018。

- 阅读版本：[1802.01483v2](https://arxiv.org/abs/1802.01483v2)。
- 核读范围：regularization section: L2, L2-SP and shared/new parameters。
- 方法依据：以预训练起点为已有参数的L2中心，对新增参数另作正则化。
- 结论边界：参数距离不等于功能／表征距离；源目标差异大时先验可能不合适。
- PDF身份：`7761dd752ed486915e1ccbc1582ba1dbe164b5d9ccef40408d191aa0da6676a9`。

## LPFT

**Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution**

Kumar, Ananya, Raghunathan, Aditi, Jones, Robbie, Ma, Tengyu, Liang, Percy。ICLR 2022。

- 阅读版本：[2202.10054v1](https://arxiv.org/abs/2202.10054v1)。
- 核读范围：§3.4 head initialization/LP-FT and §4 opening。
- 方法依据：分析头初始化与预训练特征扭曲，并研究先probe后微调。
- 结论边界：线性理论与分类分布偏移证据不能直接当YOLO多头系统定理。
- PDF身份：`dc52ebad1dba8876ecad06788754a387d4274aa7bcda0d0296a96928aa747cf8`。

## LWF

**Learning without Forgetting**

Li, Zhizhong, Hoiem, Derek。ECCV 2016 / IEEE TPAMI 2018; read arXiv v3。

- 阅读版本：[1606.09282v3](https://arxiv.org/abs/1606.09282v3)。
- 核读范围：§3 objective, procedure and distribution limitation。
- 方法依据：用新任务图像上的旧响应约束共享学习，讨论缺少旧域图像时的局限。
- 结论边界：原分类任务扩展不保证同任务YOLOSeg的任何实例损伤率。
- PDF身份：`f62623b398b2dfa417686e2fe7ec981fcb39ef1ce94837656fc64726c8085a66`。

## MASKSCORING

**Mask Scoring R-CNN**

Huang, Zhaojin, Huang, Lichao, Gong, Yongchao, Huang, Chang, Wang, Xinggang。CVPR 2019。

- 阅读版本：[1903.00241v1](https://arxiv.org/abs/1903.00241v1)。
- 核读范围：§3.2 mask scoring, MaskIoU target and inference role。
- 方法依据：预测MaskIoU并结合分类分数排序，区分质量估计与掩码形状。
- 结论边界：不是一个通用形状loss，也不允许用测试GT给部署预测排序。
- PDF身份：`87fb301a260af6c76f4821a0298085a122289f4d16644dcba30e9a18cd5cfd97`。

## NONLOCAL

**Non-local Neural Networks**

Wang, Xiaolong, Girshick, Ross, Gupta, Abhinav, He, Kaiming。CVPR 2018。

- 阅读版本：[1711.07971v3](https://arxiv.org/abs/1711.07971v3)。
- 核读范围：§3.1–3.3 non-local operator and residual block。
- 方法依据：按内容关系聚合远处位置，以残差形式加入，并讨论零输出投影保留初始行为。
- 结论边界：不保证相关性就是正确实例归属或训练后的不损伤。
- PDF身份：`415319e3506ca0d95851eafbce6ae46df76f86584375e09cbf118001a1968268`。

## PCGRAD

**Gradient Surgery for Multi-Task Learning**

Yu, Tianhe, Kumar, Saurabh, Gupta, Abhishek, Levine, Sergey, Hausman, Karol, Finn, Chelsea。NeurIPS 2020。

- 阅读版本：[2001.06782v4](https://arxiv.org/abs/2001.06782v4)。
- 核读范围：§2.3 algorithm and opening of §2.4 assumptions。
- 方法依据：对负内积任务梯度做随机顺序投影，减少局部相反分量。
- 结论边界：简化理论与经验支持不保证所有深网任务、每步loss或验证AP改善。
- PDF身份：`4b960b81a001f376e8cb551c61c5578129dd99a9d1ff147be5ea56b9e578cbdf`。

## POINTREND

**PointRend: Image Segmentation as Rendering**

Kirillov, Alexander, Wu, Yuxin, He, Kaiming, Girshick, Ross。CVPR 2020。

- 阅读版本：[1912.08193v2](https://arxiv.org/abs/1912.08193v2)。
- 核读范围：§3.1–3.2 sampling and fine/coarse instance representations。
- 方法依据：组合精细特征与粗对象预测，在不确定位置细读出，讨论区域特定信息。
- 结论边界：不表示原生YOLO每种边界错误都应增加同款点头，也不保证标签粗糙时的收益。
- PDF身份：`dc544aba7f76d1c395154d638c52e6196095eb9468700828b3b35d1fa250244c`。

## RKD

**Relational Knowledge Distillation**

Park, Wonpyo, Kim, Dongju, Lu, Yan, Cho, Minsu。CVPR 2019。

- 阅读版本：[1904.05068v2](https://arxiv.org/abs/1904.05068v2)。
- 核读范围：§3.2 distance-wise and angle-wise distillation。
- 方法依据：迁移样本表示间的归一化距离与三元夹角，减少对绝对表示坐标的依赖。
- 结论边界：原实证含度量学习、分类与少样本；转为像素或实例关系需重新定义对应与价值。
- PDF身份：`f95d58df95d542a63557ea25dee6734736c9249b1bb3040124427ec44a0d15a3`。

## TVERSKY

**Tversky loss function for image segmentation using 3D fully convolutional deep networks**

Salehi, Seyed Sadegh Mohseni, Erdogmus, Deniz, Gholipour, Ali。MLMI 2017。

- 阅读版本：[1706.05721v1](https://arxiv.org/abs/1706.05721v1)。
- 核读范围：§2.2 Tversky loss and precision/recall tradeoff。
- 方法依据：用不同FP与FN系数调节区域目标，研究病灶分割召回取舍。
- 结论边界：更大FN代价可能增加FP；不能照搬原数值或声称直接提高实例AP。
- PDF身份：`1ea63897815a4e5c9af06e7d5f73517f167d0f3fb53c74e46b68ca8d935c8673`。

## YOLACT

**YOLACT: Real-time Instance Segmentation**

Bolya, Daniel, Zhou, Chong, Xiao, Fanyi, Lee, Yong Jae。ICCV 2019。

- 阅读版本：[1904.02689v2](https://arxiv.org/abs/1904.02689v2)。
- 核读范围：§3.1–3.3 prototypes, coefficients and mask assembly。
- 方法依据：共享空间原型与实例系数组合Mask，支持分解表示的学习与分析。
- 结论边界：原型不自动等于器官语义；具体YOLO实现可能有差异。
- PDF身份：`223c83bd8f229beeaaf316b4eb4d4a92b1798c4ff61346d4ed142ad81e345d16`。
