# 对象原型融合与训练：一手文献核验（2026-10-23）

已核验22篇一手来源：7篇相关方法段，1篇官方代码沿用V6核验，14篇摘要/官方元数据。此处“方法段”不表示逐页通读或复现实验。完整书目信息和阅读层级见 [sources.json](sources.json)。

核心判断、数学推导与候选训练顺序见 [V7研究备忘](../../../../paper/manuscripts/spm_yoloseg/OBJECT_FUSION_THEORY_AND_LITERATURE_V7_20261023.md)。它是研究建议，不是冻结协议或追加训练授权；V6源码与预算不改，按其真实结果验收。

| 来源 | 状态/阅读范围 | 可借鉴的内容 | 对当前任务的边界 |
|---|---|---|---|
| [GRASP2026](https://arxiv.org/abs/2605.24533) | arXiv preprint; submitted to IEEE TCSVT, acceptance not verified；methods_checked | 借鉴按位置控制先验影响；直接相关的2026先例 | 目标是amodal；SDF外部也包含背景，不能直接当遮挡或可见Mask收益 |
| [UNICLSAM2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.html) | CVPR 2025；abstract_checked | 支持局部原型和可靠性，反对只用全局均值 | few-shot/in-context场景；不证明原型对应真实身体部位 |
| [ASGNet2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Adaptive_Prototype_Learning_and_Allocation_for_Few-Shot_Segmentation_CVPR_2021_paper.html) | CVPR 2021；abstract_checked | 原型需要覆盖局部外观变化并分配到具体像素 | 少样本语义分割；不能将其mIoU差值当猪群Mask AP预期 |
| [PANet2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.html) | ICCV 2019；abstract_checked | 训练空间与查询空间要一致 | support/query循环不是带物理身份的表面对应监督 |
| [DensePose2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Guler_DensePose_Dense_Human_CVPR_2018_paper.html) | CVPR 2018；abstract_checked | 把对象设计成可查询的身体表面，而非单个向量 | 人体标签与猪体不同；不能直接迁移其精度 |
| [CSE2020](https://proceedings.neurips.cc/paper/2020/hash/c81e728d9d4c2f636f067f89cc14862c-Abstract.html) | NeurIPS 2020；methods_checked | 本轮首选：用合成表面ID监督像素到身体表面的映射 | 真实图无表面GT；合成到真实的对应可迁移性仍待检验 |
| [NOCS2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.html) | CVPR 2019；abstract_checked | 合成数据导出规范坐标，与实例Mask一起训练 | RGB-D/尺度与刚体设定；不能声称仅RGB获得其6D姿态能力 |
| [AnimalAvatars2024](https://ai.meta.com/research/publications/animal-avatars-reconstructing-animatable-3d-animals-from-casual-videos/) | ECCV 2024；abstract_checked | 几何对象与真实视觉对应一起使用，比单纯形状库更接近对象概念 | 主要为狗的视频重建，需多帧/相机等条件，不是实时猪群实例分割 |
| [CondInst2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1105_ECCV_2020_paper.php) | ECCV 2020；abstract_checked | 融合需保留每个检测实例的查询与局部像素关系 | 动态权重本身不提供跨图像对象知识 |
| [SAM2023](https://arxiv.org/abs/2304.02643) | ICCV 2023；official_code_checked | 像素特征与实例查询共同解码 | 通用prompt分割不是对象原型；注意力权重不自动等于身体对应 |
| [HQSAM2023](https://papers.neurips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html) | NeurIPS 2023；methods_checked | 借鉴保留预训练主路、在高分辨率私有特征上做条件修正 | SAM规模/数据/任务不同；冻结权重不保证所有原生输出函数都完全不变 |
| [Mask2Former2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html) | CVPR 2022；abstract_checked | 实例查询必须通过有空间细节的像素解码器形成Mask | 硬预测Mask可能屏蔽漏分区域，不建议直接照搬硬限制 |
| [YOLACT2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html) | ICCV 2019；abstract_checked | 明确原生P是每图空间基，跨图像对象库是另一个对象 | 原生P不能因名称prototype就被解释为持久身体对象 |
| [C2FSeg2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Coarse-to-Fine_Amodal_Segmentation_with_Shape_Prior_ICCV_2023_paper.html) | ICCV 2023；methods_checked | 粗对象形状需要当前图像高频细节修正 | amodal目标不同；VQ prior不是精确可见轮廓，也不是身体点语义 |
| [PLUG2024](https://arxiv.org/abs/2405.16094) | arXiv preprint; venue not verified；abstract_checked | 可见区域和遮挡补全的目标应分清 | 不把amodal扩张当可见Mask收益；未核实的会议信息不写成已录用 |
| [BLADE2024](https://arxiv.org/abs/2401.01642) | AAAI 2024, 38(4):3846–3854；abstract_checked | 重叠/非重叠位置需要不同处理 | 其扩张目标与当前可见Mask评价不相同；不是AAAI 2025 |
| [SuperGlue2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) | CVPR 2020；methods_checked | 像素可拒绝错误身体对应，不能强迫背景/遮挡匹配 | 稀疏关键点近一对一；不可照搬成像素到表面的平衡一对一约束 |
| [L2SP2018](https://proceedings.mlr.press/v80/li18a.html) | ICML 2018；methods_checked | 若需解冻原生参数，应控制相对原起点的变化 | 分类迁移证据，不证明本轮下降由遗忘导致，也不保证Mask AP |
| [LPFT2022](https://arxiv.org/abs/2202.10054) | ICLR 2022；methods_checked | 先对齐新增接口，再考虑解冻，避免两者同时大幅漂移 | 其全特征FT/线性OOD假设不是当前头部微调任务，不能直接作根因证明 |
| [ControlNet2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | ICCV 2023；abstract_checked | 初始化时精确回到预训练函数，逐步学习条件修正 | 扩散生成领域的工程先例，不能外推分割增益 |
| [Lovasz2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html) | CVPR 2018；abstract_checked | 解释交叉熵下降与IoU/AP上升不是同一命题 | 本轮不据此临时更换loss，论文也不保证COCO AP |
| [LoRA2022](https://arxiv.org/abs/2106.09685) | ICLR 2022；abstract_checked | 把适配维持在较小子空间的参考 | LLM证据不决定视觉最优秩，不作为本轮额外模块建议 |

优先级：CSE/DensePose解决像素—身体对应；HQ-SAM/CondInst支持保留原生能力的实例条件稠密读出；L2-SP/LP-FT支持分阶段适配。GRASP与C2F-Seg帮助检查结构先验和图像细节的关系，但其amodal目标不可混入可见Mask评价。

[Math检查](MATH_CHECKS.json)仅验证本备忘中代数例子和闭式解，与模型实验、对应泛化或AP提升分账。可用 `python literature_search/spm-object-fusion_2026-10-23/verify_math.py` 复算。

