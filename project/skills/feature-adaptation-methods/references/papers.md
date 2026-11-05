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

## BN

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**

Ioffe, Sergey, Szegedy, Christian。ICML 2015。

- 阅读版本：[1502.03167v3](https://arxiv.org/abs/1502.03167v3)。
- 核读范围：§3 normalization and population versus minibatch statistics。
- 方法依据：以逐通道统计及可学习缩放平移重参数化特征，区分训练batch与推理统计。
- 结论边界：计算定义有依据；原始内部协变量偏移解释不是已被确认为唯一机制。
- PDF身份：`bdc69e0b568f41d8d3181cc9077a461f58fcdd1c70c9f49efe60a05ee6109655`。

## BNMECHANISM

**How Does Batch Normalization Help Optimization?**

Santurkar, Shibani, Tsipras, Dimitris, Ilyas, Andrew, Madry, Aleksander。NeurIPS 2018。

- 阅读版本：[1805.11604v5](https://arxiv.org/abs/1805.11604v5)。
- 核读范围：§2.1 internal-shift counter-experiment and §3.1 smoothness explanation。
- 方法依据：用受控研究挑战BN收益仅来自内部协变量偏移降低的解释，讨论优化平滑性。
- 结论边界：一种解释与其条件不是任意现代模型唯一机制。
- PDF身份：`41d5195d959a8c37f3f23df34023032b2a0361887495cdb558a2402f60437d1e`。

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

## GN

**Group Normalization**

Wu, Yuxin, He, Kaiming。ECCV 2018。

- 阅读版本：[1803.08494v3](https://arxiv.org/abs/1803.08494v3)。
- 核读范围：§3.1 normalization sets and group formulation。
- 方法依据：按每个样本的通道组和空间位置标准化，统计不依赖其他batch样本。
- 结论边界：支持小batch归一化候选；不保证直接替换既有预训练BN网络后更优。
- PDF身份：`b2863f7fc089095da466c4318a4764ae666ce60b7d264b5de129d1867a4541a3`。

## INVARIANT

**On Learning Invariant Representation for Domain Adaptation**

Zhao, Han, Combes, Remi Tachet des, Zhang, Kun, Gordon, Geoffrey J.。ICML 2019。

- 阅读版本：[1901.09453v2](https://arxiv.org/abs/1901.09453v2)。
- 核读范围：§4.1 invariant-representation counterexample。
- 方法依据：构造边缘特征分布完全相同但标签关系不同、联合误差很大的反例。
- 结论边界：说明域不变性非充分条件，不意味着所有域适应无用。
- PDF身份：`9d359c0d73cff25ceb8c8bae52894d7d9d85181de74e98e3343359680490f7cd`。

## RKD

**Relational Knowledge Distillation**

Park, Wonpyo, Kim, Dongju, Lu, Yan, Cho, Minsu。CVPR 2019。

- 阅读版本：[1904.05068v2](https://arxiv.org/abs/1904.05068v2)。
- 核读范围：§3.2 distance-wise and angle-wise distillation。
- 方法依据：迁移样本表示间的归一化距离与三元夹角，减少对绝对表示坐标的依赖。
- 结论边界：原实证含度量学习、分类与少样本；转为像素或实例关系需重新定义对应与价值。
- PDF身份：`f95d58df95d542a63557ea25dee6734736c9249b1bb3040124427ec44a0d15a3`。
