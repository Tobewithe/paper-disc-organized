# 论文依据与阅读边界

本包核对原始论文并选择性阅读与方法有关的正文／理论段落，未逐篇全文审读、复现实验或开展计量式影响力排名。下面按“论文能支持什么”和“不能由此推出什么”区分来源结论与应用迁移。

方法正文中的YOLOSeg组合、主任务优先变体及部分代数说明是基于文献的综合或推导，文中已作标识；不能把它们作为原论文已测结果。

## ADAPTERS

**Learning multiple visual domains with residual adapters**

Rebuffi, Sylvestre-Alvise, Bilen, Hakan, Vedaldi, Andrea。NeurIPS 2017。

- 阅读版本：[1705.08045v5](https://arxiv.org/abs/1705.08045v5)。
- 核读范围：§3.2–3.3 residual visual adapters and domain-specific learning。
- 方法依据：用域特定小型残差适配器共享视觉主体并限制新增变化。
- 结论边界：保留原路径不意味着适配器启用时预测完全不变。
- PDF身份：`12c5877e98dc11b99ef5cf7be40a1145d5f22e7a790ea785f3f6ae5148137c9d`。

## AT

**Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer**

Zagoruyko, Sergey, Komodakis, Nikos。ICLR 2017。

- 阅读版本：[1612.03928v3](https://arxiv.org/abs/1612.03928v3)。
- 核读范围：§3 activation-based attention and transfer objective。
- 方法依据：由激活统计构造空间注意图，归一化后迁移教师关注结构，并允许不同通道表示。
- 结论边界：空间激活对应不等于实例身份、因果解释或当前分割收益。
- PDF身份：`e4f13e6dda75d070d4524e2e8fac3074b3b7af3744ed5d66dbbe39170b067be8`。

## FITNETS

**FitNets: Hints for Thin Deep Nets**

Romero, Adriana, Ballas, Nicolas, Kahou, Samira Ebrahimi, Chassang, Antoine, Gatta, Carlo, Bengio, Yoshua。ICLR 2015。

- 阅读版本：[1412.6550v4](https://arxiv.org/abs/1412.6550v4)。
- 核读范围：§2.2 hint-based training。
- 方法依据：用教师中间层提示与学生回归器训练较深较窄网络，讨论提示层选择及过度正则化。
- 结论边界：支持投影后表征提示的原理；不证明任意两模块应逐值匹配，也未测本包的YOLOSeg组合。
- PDF身份：`584ff258c88dd5b992b1df7977d856189840cdf89b0ff93331a69054f5f5f9c7`。

## KD

**Distilling the Knowledge in a Neural Network**

Hinton, Geoffrey, Vinyals, Oriol, Dean, Jeff。arXiv 2015 technical report。

- 阅读版本：[1503.02531v1](https://arxiv.org/abs/1503.02531v1)。
- 核读范围：§2–2.1 temperature distillation and logit matching。
- 方法依据：教师软输出提供类别关系，温度及T²因子影响学习信号。
- 结论边界：分类softmax推导不能直接替代独立二值实例Mask的概率定义。
- PDF身份：`74e6689115c539db499f9d6153edecb364059926fd45402454f8210442444c5a`。

## L2SP

**Explicit Inductive Bias for Transfer Learning with Convolutional Networks**

Li, Xuhong, Grandvalet, Yves, Davoine, Franck。ICML 2018。

- 阅读版本：[1802.01483v2](https://arxiv.org/abs/1802.01483v2)。
- 核读范围：regularization section: L2, L2-SP and shared/new parameters。
- 方法依据：以预训练起点为已有参数的L2中心，对新增参数另作正则化。
- 结论边界：参数距离不等于功能／表征距离；源目标差异大时先验可能不合适。
- PDF身份：`7761dd752ed486915e1ccbc1582ba1dbe164b5d9ccef40408d191aa0da6676a9`。

## LORA

**LoRA: Low-Rank Adaptation of Large Language Models**

Hu, Edward J., Shen, Yelong, Wallis, Phillip, Allen-Zhu, Zeyuan, Li, Yuanzhi, Wang, Shean, Wang, Lu, Chen, Weizhu。ICLR 2022。

- 阅读版本：[2106.09685v2](https://arxiv.org/abs/2106.09685v2)。
- 核读范围：§4.1 low-rank update and §4.2 application scope。
- 方法依据：以冻结权重加低秩增量进行参数高效适配，原主要实证是语言模型。
- 结论边界：不声称某种YOLO卷积低秩更新已获原论文验证或天然防遗忘。
- PDF身份：`e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a`。

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

## MASKRCNN

**Mask R-CNN**

He, Kaiming, Gkioxari, Georgia, Dollár, Piotr, Girshick, Ross。ICCV 2017。

- 阅读版本：[1703.06870v3](https://arxiv.org/abs/1703.06870v3)。
- 核读范围：§3 mask target, sigmoid/CE distinction and alignment。
- 方法依据：对象条件的二值Mask与语义Softmax目标不同，强调精细空间表示。
- 结论边界：不能把所有实例目标当互斥语义图，或将ROI结果直接当YOLO结论。
- PDF身份：`b595a882b763d55d933543131f32a2c4b6f41d0818d418ef46a2b66f946b67a8`。

## RKD

**Relational Knowledge Distillation**

Park, Wonpyo, Kim, Dongju, Lu, Yan, Cho, Minsu。CVPR 2019。

- 阅读版本：[1904.05068v2](https://arxiv.org/abs/1904.05068v2)。
- 核读范围：§3.2 distance-wise and angle-wise distillation。
- 方法依据：迁移样本表示间的归一化距离与三元夹角，减少对绝对表示坐标的依赖。
- 结论边界：原实证含度量学习、分类与少样本；转为像素或实例关系需重新定义对应与价值。
- PDF身份：`f95d58df95d542a63557ea25dee6734736c9249b1bb3040124427ec44a0d15a3`。

## YOLACT

**YOLACT: Real-time Instance Segmentation**

Bolya, Daniel, Zhou, Chong, Xiao, Fanyi, Lee, Yong Jae。ICCV 2019。

- 阅读版本：[1904.02689v2](https://arxiv.org/abs/1904.02689v2)。
- 核读范围：§3.1–3.3 prototypes, coefficients and mask assembly。
- 方法依据：共享空间原型与实例系数组合Mask，支持分解表示的学习与分析。
- 结论边界：原型不自动等于器官语义；具体YOLO实现可能有差异。
- PDF身份：`223c83bd8f229beeaaf316b4eb4d4a92b1798c4ff61346d4ed142ad81e345d16`。
