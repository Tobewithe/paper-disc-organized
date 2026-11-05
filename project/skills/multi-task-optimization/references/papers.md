# 论文依据与阅读边界

本包核对原始论文并选择性阅读与方法有关的正文／理论段落，未逐篇全文审读、复现实验或开展计量式影响力排名。下面按“论文能支持什么”和“不能由此推出什么”区分来源结论与应用迁移。

方法正文中的YOLOSeg组合、主任务优先变体及部分代数说明是基于文献的综合或推导，文中已作标识；不能把它们作为原论文已测结果。

## CAGRAD

**Conflict-Averse Gradient Descent for Multi-task Learning**

Liu, Bo, Liu, Xingchao, Jin, Xiaojie, Stone, Peter, Liu, Qiang。NeurIPS 2021。

- 阅读版本：[2110.14048v2](https://arxiv.org/abs/2110.14048v2)。
- 核读范围：§3.1 objective and algorithm; convergence scope in §3.2 opening。
- 方法依据：在平均梯度邻域中改善最不利任务的一阶变化，控制聚合目标与冲突的折衷。
- 结论边界：平均目标不等于指定主任务优先；收敛结论有光滑性、步长和参数条件。
- PDF身份：`47770cef6bda95b28d1cf091c4e6d7d1e903c6565f4b93a4f68193ae2aeb5b80`。

## GRADNORM

**GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks**

Chen, Zhao, Badrinarayanan, Vijay, Lee, Chen-Yu, Rabinovich, Andrew。ICML 2018。

- 阅读版本：[1711.02257v4](https://arxiv.org/abs/1711.02257v4)。
- 核读范围：§3.1–3.2 gradient norm targets and algorithm。
- 方法依据：根据相对学习速度调整共享层上的加权梯度范数，区分大小与速度。
- 结论边界：不直接处理方向冲突，也不决定主任务优先级或保证验证指标。
- PDF身份：`e72930c1f875396f16955748e5af9f2b14c9365c50c1e055cbc36612f326a726`。

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

## UNCERTAINTY

**Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics**

Kendall, Alex, Gal, Yarin, Cipolla, Roberto。CVPR 2018。

- 阅读版本：[1705.07115v3](https://arxiv.org/abs/1705.07115v3)。
- 核读范围：§3.1–3.2 task uncertainty and likelihood derivation。
- 方法依据：从同方差任务噪声和联合似然推导目标加权，分类与回归处理不同。
- 结论边界：不是逐像素可信度；任意Dice／KL套用不自动保留原似然解释。
- PDF身份：`408fd70c33271009143e873e41efe9bd805d7bce6c9f72813ec378de2f16b30c`。
