# 来源与阅读范围

2026-11-05从既有本地阅读和实证总结整理。本次没有新外部检索、论文全文审阅、代码复现或模型验证。逐文件路径、SHA与范围见[整理来源](CURATION_SOURCES.json)。

主要来源为`project/research_notes/literature/occlusion-pretraining_2026-10-19/summary.md`及`notes/E_findings.md`；原`evidence_matrix.json`、`SEARCH_STATUS.json`、`pdfs/manifest.json`保存主张位置、获取与失败记录。下载成功不等于逐页精读。项目经验原报告为`experiments/dense_video_occlusion_transfer_20261019/REPORT.md`；本次沿用已有总结中的数字，没有重新计算。

- TCOW：https://tcow.cs.columbia.edu/ ，https://arxiv.org/abs/2305.03052 。遮挡持续存在与目标／遮挡物／容器监督。
- WALT：https://openaccess.thecvf.com/content/CVPR2022/papers/Reddy_WALT_Watch_and_Learn_2D_Amodal_Representation_From_Time-Lapse_Imagery_CVPR_2022_paper.pdf 。分层amodal监督。
- SAMEO：https://arxiv.org/abs/2503.06261 ；SAMBA：https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Towards_Efficient_Foundation_Model_for_Zero-shot_Amodal_Segmentation_CVPR_2025_paper.pdf 。框提示及可见／完整分工。
- Amodal SAM：https://arxiv.org/abs/2604.20748 。预印本；没有核实一篇独立的“AmodalSAM2”。
- SAM2：https://arxiv.org/abs/2408.00714 ，https://github.com/facebookresearch/sam2 ；SAM3／3.1：https://github.com/facebookresearch/sam3 。按原记录的发布日期和接口解释，不当成当前安装验收。
- CoTracker3：https://arxiv.org/abs/2410.11831 。真实伪轨迹与可见性头边界。
- DiffVAS：https://diffusion-vas.github.io/ ；TABE：https://finlay-hudson.github.io/tabe/ 。隐藏内容预测及原记录未核范围保留。
- Occlusion-Enhanced Distillation：https://arxiv.org/abs/2409.01573 。苹果检测中的相关蒸馏先例，不外推为猪／视频／YOLO已验证收益。
- 视频猪模型、STARFORMER与基础模型跟踪的指定段落证据来自`project/research_notes/literature/pig-instance-open-analysis_2026-11-02/EVIDENCE_CARDS.md`和`READING_LOG.json`。作者自报、抽样评价、已知数量、双向传播及人工QC条件均保留。

补入A/B/C/G/S各来源桶中已有的指定方法、元数据和实现边界，并从2026-10-14视频可靠性来源稿提炼COCO/HOTA/IDF1、ByteTrack/OC-SORT、SAM2Long及DeepStream的接口与时延知识。429回退、摘要层来源、未定位代码等仍保持原状态；主张不含历史模型选择和分阶段训练建议。

本Skill没有迁移原报告中的模型优先级、分阶段实验方案、拟议损失或“唯一下一步”。
