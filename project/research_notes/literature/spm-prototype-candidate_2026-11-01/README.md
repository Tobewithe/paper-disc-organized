# SPM原型候选：定向近邻审查（2026-11-01）

对应[RLPL候选](../../../../paper/manuscripts/spm_yoloseg/PAPER_CANDIDATE_RLPL_20261101.md)。本次是围绕明确命题的定向检索/方法正文阅读，**不是穷尽多库系统综述或新颖性保证**。检索元数据、读取位置、内容SHA和范围见[sources.json](sources.json)；全文临时缓存不提交大文件。

## 1. 主判断

**“原型库＋注意力＋YOLO”“局部原型”“原型跨域训练”“训练完删除辅助头”都已有充分近邻，不能单独充当创新。** 我选择继续发展的是：让渲染器核实的同实例局部对应及可见性条件，直接约束原生分割分支的训练，并严格检验它比普通连续对应学习和既有原型学习多解决了什么。

这更适合农业视觉应用型训练方法论文的定位，而非宣称发明全新原型范式。最有意义的新增结果应是：**在同输入、同监督、同训练预算、原生推理结构不变时，正确的可见对应关系与原型分配确实带来独立真实收益。** 当前只有可检验设计，没有该结果；仅更换为猪任务不构成机制差异。

## 2. 最相关工作：先比较机制而不是名称

| 工作与一手来源 | 已核查的方法 | 与本候选的关系/不能据为己有的部分 | 必须验证的真实差异 |
|---|---|---|---|
| [SwAV：Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882) | §3.1、§3.2及附录C.1/C.2；跨增强视图预测共享原型的平衡软分配，Sinkhorn、原型归一化、多crop。 | **本候选分配loss的直接来源**，不是新loss。原型共享、无pairwise大量负例均已存在。 | 对象局部、受控三维成像、双方可见支持及原生分割梯度的联合训练，须超过任务适配后的原型对照E。 |
| [DenseCL：Dense Contrastive Learning for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2011.09157) | §2.1–2.4；dense projection、局部InfoNCE、按backbone余弦最近邻建立对应，训练后去掉投影。 | 局部对应学习、投影只用于训练都已有。不能把DenseCL描述为仅图像级对比。 | 实际登记的同位置/owner可见关系是否比特征近邻更有用；E保留同数据与原型目标，只改变对应来源。 |
| [PixPro：Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043) | §3.1–3.3；同一图像crop的精确坐标关系，像素传播、非负例一致性，且可预训练dense head。 | **精确像素对应、训练dense head也不是本候选首创**。 | 多个独立渲染中的外观/遮挡干预不是简单裁剪；但价值必须由同信息连续学习C和真实任务成绩证明。 |
| [Exploring Cross-Image Pixel Contrast for Semantic Segmentation](https://arxiv.org/abs/2101.11939) | §3.2–3.3；跨图同类像素正对、类别pixel/region memory，训练后移除投影与memory。 | “跨图局部知识”“区域原型”“零新增部署网络”已有直接先例。 | 不是把全体猪前景都拉近，而是匹配同实例同可见局部；不依赖该方法的多语义类负对，但仍须排除普通表征正则。 |
| [Calibration-based Dual Prototypical Contrastive Learning Approach for Domain Generalization Semantic Segmentation](https://arxiv.org/abs/2309.14282) | §3.1–3.3；源域/风格增强域的类别原型，域差异与难原型校准。 | 原型帮助域泛化、跨风格对齐均已有；其文中也指出原型本身可能有域偏差。 | 本候选没有声称原型天然域不变；依靠可核局部关系，而非整个类别中心或校准模块。 |
| [Prototypical Progressive Alignment and Reweighting for Generalizable Semantic Segmentation](https://arxiv.org/abs/2507.11955) | §IV-A–D；CLIP文本/视觉文本原型、分层渐进对齐与像素重加权，仅用于训练。 | 无额外推理开销及原型正则不是空白；本次只按arXiv正文记录，不把未知期刊身份写成录用。 | 本候选不调用CLIP/额外教师，没有多层重加权；若拿其结果作对照须另控制额外预训练信息，不能直接跨表比AP。 |
| [ASGNet：Adaptive Prototype Learning and Allocation for Few-Shot Segmentation](https://arxiv.org/abs/2104.01893) | 前轮已读方法正文；支持图superpixel多原型、逐query像素分配。 | 局部多原型分配已有，不因它是few-shot就忽略机制相似。 | 本候选部署不提供带标注support；这是输入合同差异，不自动等于算法创新。 |
| [UNICL-SAM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.html) | 前轮已额外读取PDF方法；示例不确定性、特征细化与part prototype提示。 | “部位原型提高分割”已有；不能只引用标题而忽略其支持/查询条件。 | 不采用SAM、上下文示例或推理提示；本候选也不预先把无监督槽命名为语义部位。 |
| [ShapeMask：Learning to Segment Novel Objects by Refining Shape Priors](https://arxiv.org/abs/1904.03239) | §3.1–3.4；对32×32 GT形状聚类、框特征加权形状先验、粗/细Mask与实例embedding。 | 形状字典＋实例条件读出是强近邻，不应把该组合当新idea。 | 本候选不做形状模板生成与粗Mask再精修；不恢复已终止的后置掩码方向。 |
| [GRASP：Learnable Shape Prototypes with Occlusion-Geometry-Guided Injection for Amodal Instance Segmentation](https://arxiv.org/abs/2605.24533) | 前轮已读方法；可学习形状记忆、cross-attention、空间门控，输入含可见Mask，目标amodal。 | 可学习形状原型、注意力和遮挡门控均非空白。**按预印本记录，未核实期刊录用**。 | 本候选只做visible任务、无输入Mask、无部署记忆。不能只靠visible/amodal不同宣称新的原型算法。 |

上述正文核查的是指定方法部分，不声称逐页审读全部论文、复现代码或确认全部作者的实验证据。ASGNet/UNICL-SAM/GRASP沿用本对话前轮的实际全文缓存和已读位置；新source清单与旧仅摘要清单分开，不回写历史审阅状态。

## 3. 三个改变设计的发现

### 3.1 训练期原型本身已有强近邻

Cross-Image Pixel Contrast与PPAR已经在训练后移除辅助结构。因此候选必须把“零新增部署分支”定位为实用约束，不能写作首创贡献。DenseCL/PixPro也不支持把“直接训练分割特征而非只训练独立投影”描述为文献空白。

与项目早期[基础实现](../../../../paper/manuscripts/spm_yoloseg/BASE_IMPLEMENTATION_SPEC.md)相比，本候选的明确改变是：不再只冻结YOLO学习投影、构库后部署检索，而是让辅助损失进入原生分割专用参数，部署删除辅助头。早期已经有对应视图/VICReg，不能把这部分重报成新想法。

### 3.2 “学到原型内容”比“分配训练有效”更难成立

SwAV附录C.2报告其设置中学习原型73.9、固定随机原型73.1，并明确随机原型也接近有效。这些是该文ImageNet线性评价，不是本任务AP；但它直接说明：**分配学习收益不必依赖具名语义内容。** 因而候选只先主张有用的训练约束；若要进一步声称猪体知识存储，需要固定随机原型的匹配训练与更多内容证据，不能看可视化后命名。

### 3.3 两个容易制造假机制的控制被排除

- 固定目标的均值原型MSE与对所有目标的等权MSE，对学生给出相同梯度；平均不是新机制。
- 无序原型及分配列的共同置换保持损失；这种“打乱”不是破坏。训练期头在导出时删除也按设计不影响结果，不能沿用旧runtime bank的off归因。

对应解析恒等式及构造反例在[本地核验脚本](../../../../experiments/spm_registered_prototype_learning_20261101/verify_design.py)中保存；纯抽象测试不是科学效用证据。

## 4. 创新性与论文取舍

- **保留并发展：** 可靠实例标签＋条件变化下正确局部对应＋原生分割专用训练，以及同数据/同信息的收益分解。这与当前生成器能力和旧失败直接衔接，较再加推理检索支路有更明确的最小验证路径。
- **不认领：** 原型、聚类、Sinkhorn、跨图/像素一致性、对比学习、无推理增量本身；也不把猪场应用或模块重新命名当机制差异。
- **新颖性风险：** 属于已有表征技术的任务化组合。若不能超过C/E，方法创新主张不成立；若只有数据B有效，写数据/受控合成研究，不继续凑原型模块。
- **论文明确信息边界：** 当前未完成广泛引用追踪、未复现近邻代码、未做RLPL真实训练。正式投稿前应以已得到的明确贡献再补针对性检索，而不是现在声称排除了全部先例。
- **投稿定位：** 农业视觉应用研究。先取得独立真实泛化、完整损伤与成本证据，再定具体期刊；不以+5、1–2AP或模块数量保证发表。
