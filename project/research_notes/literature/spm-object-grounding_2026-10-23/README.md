# 从身体几何到可读视觉原型：第二轮文献筛选

2026-10-23。在上一轮[22篇一手来源](../spm-object-fusion_2026-10-23/sources.json)基础上，新核10篇，累计32篇；新增6篇查阅相关方法段，4篇仅摘要/官方元数据。数量不作为路线正确性的证据，也不声称穷尽文献。

当前选择：用同一冻结视觉空间监督身体表面字段，再学习实例条件可见Mask读出；保留原生函数。完整方法和闭式解见[V8决策稿](../../../../paper/manuscripts/spm_yoloseg/GROUNDED_OBJECT_ROUTE_V8_20261023.md)。

| 一手来源 | 阅读范围 | 能改变本项目决策的内容 | 范围限制 |
|---|---|---|---|
| [ShapeMask2019](https://arxiv.org/html/1904.03239) | methods_checked；3.1–3.3; appendix D table 7 | 借鉴形状与当前实例图像的协作；不照抄低分辨率套框接口 | 完整监督表7：35.5→37.2；仅shape36.9，仅embedding36.7。不能引用跨类别大增益作为本项目+5依据 |
| [MagicPony2023](https://arxiv.org/html/2211.12497) | methods_checked；3.4, equation 3; limitations | 最关键正面依据：对象表面字段必须有实际图像特征锚定 | 单对象重建与多视点歧义；左右腿可能无法区分，不是猪群可见Mask AP结果 |
| [Fauna2024](https://arxiv.org/html/2401.02400) | methods_checked；3.1–3.2; feature rendering and staged training | 几何先验与预训练语义应共同形成对象，并采用分阶段学习 | 128物种/78,168图的重建研究；不把其规模、时间或重建精度外推本数据 |
| [Dessie2024](https://arxiv.org/html/2410.03438v1) | methods_checked；3.2–3.3; 4.2–4.3 table 1; 5.1 | 多样化需跨外观/姿态/背景并由真实分布适配，不能只增固定模板重复次数 | 2 A100约2天；单马crop。更多直接shape/pose GT并非每项都更好，不能承诺小模型短训复制 |
| [Farm3D2023](https://arxiv.org/html/2304.10535) | methods_checked；introduction; 3.1–3.3 | 保留对象级建模原则，暂不引入完整扩散生成/3D重建训练链 | 方法偏向无明显遮挡的单体；生成图不提供已核实的逐像素3D标签，不宜直接替换本轮几何标签 |
| [DepthAnythingV22024](https://arxiv.org/html/2406.09414) | methods_checked；3–5.1, synthetic-real gap and semantic feature preservation | 直接合成到真实是必要风险，须利用真实训练分布做接口锚定 | 原文595K合成/62M真实、DINOv2-G；深度任务，不保证小型YOLO对应迁移 |
| [MaskTransfiner2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html) | abstract_checked；official abstract and first-page PDF text | 保留高分辨率图像细节原则，暂不另加quadtree/transformer | 论文不同框架报告的增益不能作为猪群+5预期；尚未核当前错误主要在边界 |
| [PointRend2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) | abstract_checked；official abstract | 空间细节应由实际图像维持，原型不替代最终可见边界 | 只说明像素细节的一种处理，不提供身体知识或同类实例归属 |
| [FeatUp2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html) | abstract_checked；official proceedings abstract | 低分辨率语义与高分辨率细节需要共同保留 | 不是三维身体对应，也不是当前应用直接有效的证据 |
| [TwoFeatures2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8e9bdc23f169a05ea9b72ccef4574551-Abstract-Conference.html) | abstract_checked；official proceedings abstract | 即使引入预训练语义，也需对应误差/实例归属核验，暂不引入额外扩散特征 | DINO匹配可稀疏、扩散匹配可语义错；不把foundation当无误老师 |

结论是本项目对论文及本地实现的推论。MagicPony/3D-Fauna提供了表面特征与真实视觉特征关联的先例；ShapeMask支持形状和实例证据共同读出；Dessie/Depth Anything V2提示强几何标签不能替代域桥接。四者不构成已经在猪群任务实现+5 AP的证据。

