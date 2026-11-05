# 部位监督到实例分割：近邻方法定向审查

2026-11-02。针对“概念/部位监督与实例任务存在gap，应该采用什么连接形式”的讨论。实际进行了3次arXiv API查询，返回4个不同ID（含1个无关透明物体工作）；下载2篇PDF并阅读下面的正文范围。不是穷尽综述、全文逐页阅读或代码复现。

查询原文和返回摘要见[queries.json](queries.json)，PDF来源、字节数、SHA及临时文本路径见[sources.json](sources.json)。通过本机`pdftotext -enc UTF-8 -layout`提取；未安装新环境或运行模型。

## TAPPS：最直接的任务连接依据

[Task-aligned Part-aware Panoptic Segmentation through Joint Object-Part Representations](https://arxiv.org/abs/2406.10114)，Daan de Geus、Gijs Dubbelman，2024。PDF SHA `131282a6a94a797e2c48fc7f83bc50f6f836a4ed474352d02f3ba34a0faada7c`。实际读取提取文本工具显示的1–669行，覆盖方法、主要实验、消融、实例分离分析及部分附录实现/成本。

- 对照方法分别预测实例和全图部位；同类部位跨实例合成一个语义目标。TAPPS改为每个共享查询同时表示一个对象及其内部部位，学习与任务定义对齐。
- 同一个256维query经原Mask2Former的3层MLP预测对象Mask；另一条适配MLP及每部位固定FC生成该对象的部位queries，与同一高分辨率特征相乘预测部位Mask。训练以对象类/Mask做二分匹配，再监督该对象的已知部位。**对象Mask不是部位Mask的硬并集，也不是先正确分完部位才能得到对象。**
- Pascal-PP、ImageNet初始化：表1/5的thing PQ为62.0→65.6，instance-agnostic thing mIoU为69.8→69.9。作者据此解释主要改善实例分离，而非只改善类别识别。Cityscapes-PP相同设置thing PQ53.8→55.6，mIoU77.6→78.2。
- 这是PPS/PQ结果，不是猪Mask AP；训练有真实部位标注，不是仅合成部位到真实域迁移。Pascal和Cityscapes分别有多类与类别—部位兼容性简化，而本项目仅猪一类，不能转移那部分收益。
- 表10：基线约44M、TAPPS约48M参数；不是严格等容量独立归因。原实验Pascal 60k（ImageNet）或10k（COCO）、Cityscapes90k更新，不拿这些预算直接代替本项目冻结规模。没有把该论文当作本项目已验证的效用或机制。

**可借用的主张：**部位目标应绑定具体正常实例，并作用到承载对象预测的实例表示，而不只把共享图像特征做成可读的部位语义图。**不能冒称新意：**共享实例query预测对象与部位已有直接先例。

## Panoptic-PartFormer：联合推理不等于实例绑定

[Panoptic-PartFormer: Learning a Unified Model for Panoptic Part Segmentation](https://arxiv.org/abs/2204.04655)，2022。PDF SHA `1764b1fcfc6677c50dc922118835e1ebb78a8268e70afa5f10506eecb38c6f09`。实际读取提取文本60–389行，主要为动机与方法；未读完整结果表，不据本轮阅读追加数值效果声称。

方法设thing/stuff/part不同queries，场景和部位有各自特征decoder；通过mask pooling、动态卷积和query self-attention联合迭代。它是“在架构中联合处理”而不是TAPPS的“每个对象共享query产生其所属部位”。二者不能因为都叫联合学习就当作同一连接形式。

## 本项目差异与待验证价值

优先讨论：把合成可见部位的辅助目标绑定到**原生YOLO真实使用的实例读出**，而不是新增一个独立实例网络；例如从实际实例系数派生各部位辅助系数，共用原生空间基，正常输出仍为原`c_i^T P`。推理可以删除辅助头，目标是在原生参数中保留收益。

与TAPPS相比，研究难点是：原生低维读出可表达性、仅合成部位标签向真实任务迁移、物理可见标签与真实Mask标注语义不完全相同，以及正确解剖是否超过同容量非解剖分区。差异有研究价值，但目前没有新Run证明它成立，也不能据此宣称新颖性已充分或已准正式训练。
