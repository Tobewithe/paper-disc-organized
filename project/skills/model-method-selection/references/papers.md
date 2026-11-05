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

## CONDINST

**Conditional Convolutions for Instance Segmentation**

Tian, Zhi, Shen, Chunhua, Chen, Hao。ECCV 2020; read arXiv v4。

- 阅读版本：[2003.05664v4](https://arxiv.org/abs/2003.05664v4)。
- 核读范围：§2.1 instance-conditioned filters and §2.3 Dice objective。
- 方法依据：动态实例参数与相对位置使共享特征产生对象特定Mask，并采用相应目标。
- 结论边界：支持实例条件的重要性，不证明任何部位／门控组合都有增量。
- PDF身份：`6c3643a3002fc887938c00fbbbf2c89d2fb3f6595deec3b70b1a1f8a59434981`。

## FITNETS

**FitNets: Hints for Thin Deep Nets**

Romero, Adriana, Ballas, Nicolas, Kahou, Samira Ebrahimi, Chassang, Antoine, Gatta, Carlo, Bengio, Yoshua。ICLR 2015。

- 阅读版本：[1412.6550v4](https://arxiv.org/abs/1412.6550v4)。
- 核读范围：§2.2 hint-based training。
- 方法依据：用教师中间层提示与学生回归器训练较深较窄网络，讨论提示层选择及过度正则化。
- 结论边界：支持投影后表征提示的原理；不证明任意两模块应逐值匹配，也未测本包的YOLOSeg组合。
- PDF身份：`584ff258c88dd5b992b1df7977d856189840cdf89b0ff93331a69054f5f5f9c7`。

## GRADNORM

**GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks**

Chen, Zhao, Badrinarayanan, Vijay, Lee, Chen-Yu, Rabinovich, Andrew。ICML 2018。

- 阅读版本：[1711.02257v4](https://arxiv.org/abs/1711.02257v4)。
- 核读范围：§3.1–3.2 gradient norm targets and algorithm。
- 方法依据：根据相对学习速度调整共享层上的加权梯度范数，区分大小与速度。
- 结论边界：不直接处理方向冲突，也不决定主任务优先级或保证验证指标。
- PDF身份：`e72930c1f875396f16955748e5af9f2b14c9365c50c1e055cbc36612f326a726`。

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

## MGDA

**Multi-Task Learning as Multi-Objective Optimization**

Sener, Ozan, Koltun, Vladlen。NeurIPS 2018。

- 阅读版本：[1810.04650v2](https://arxiv.org/abs/1810.04650v2)。
- 核读范围：§3–3.1 Pareto formulation and minimum-norm combination。
- 方法依据：把多任务视为多目标优化，以共享梯度凸组合研究共同下降与Pareto驻点。
- 结论边界：驻点不必是全局最优，也不决定部署所需的主任务折衷。
- PDF身份：`f667a8e903dc07efeadfd0204051ffbeffc93aee10c78f1a7da410e43eb11e1d`。

## MTLSURVEY

**Multi-Task Learning for Dense Prediction Tasks: A Survey**

Vandenhende, Simon, Georgoulis, Stamatios, Van Gansbeke, Wouter, Proesmans, Marc, Dai, Dengxin, Van Gool, Luc。IEEE TPAMI 2022; read arXiv v3。

- 阅读版本：[2004.13379v3](https://arxiv.org/abs/2004.13379v3)。
- 核读范围：abstract/introduction and §3.1 task balancing comparison。
- 方法依据：比较稠密多任务的共享架构、权重与梯度优化，整理不同机制。
- 结论边界：本包只使用所读方法比较，不冒称重现综述中的全部基准结论。
- PDF身份：`f53a99bb8cbbf48dd6e989f8e70bad02f53e7babf9d6c181f53a23557f9e197a`。

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

## YOLACT

**YOLACT: Real-time Instance Segmentation**

Bolya, Daniel, Zhou, Chong, Xiao, Fanyi, Lee, Yong Jae。ICCV 2019。

- 阅读版本：[1904.02689v2](https://arxiv.org/abs/1904.02689v2)。
- 核读范围：§3.1–3.3 prototypes, coefficients and mask assembly。
- 方法依据：共享空间原型与实例系数组合Mask，支持分解表示的学习与分析。
- 结论边界：原型不自动等于器官语义；具体YOLO实现可能有差异。
- PDF身份：`223c83bd8f229beeaaf316b4eb4d4a92b1798c4ff61346d4ed142ad81e345d16`。
