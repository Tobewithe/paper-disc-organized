# ARIS 全源文献登记册

检索批次：`2026-09-05T09:30:00+08:00`（增量复检；2026-09-04 批次保留）。研究对象：密集猪群实例分割中的低重叠实例碎片化、候选竞争与全局/关系一致性。

## 登记规则

- canonical key 优先级：DOI -> arXiv ID -> Semantic Scholar/OpenAlex ID -> 规范化标题。
- `core-verified`：有 arXiv 或官方出版页，且已核对作者、年份、venue/标识。
- `related-unread`：命中且主题相关，但本批次未完成原文深读。
- `duplicate`：与另一来源同一 canonical paper，保留来源集合。
- `excluded-noise`：检索命中但与实例分割问题无直接关系，保留 raw hit 以便以后识别。
- `pending-verification`：标识或日期需要原始页面复核，不能作为已发表事实。

## 来源调用审计

| 来源 | 查询/范围 | 结果 | 状态与证据 |
|---|---|---:|---|
| Zotero MCP | 四组主题 | 0 | 未发现 `mcp__zotero__*` 工具，未配置；无静默跳过 |
| Obsidian MCP | vault 主题检索 | 65 条笔记库，命中既有研究资产 | 可用；命中笔记路径见下文 |
| local PDFs | `papers/`, `literature/` | 0 PDF | 明确记录：目录不存在可用 PDF |
| arXiv | 四组查询，每组 max=10 | 40 raw hits | 成功；`raw-search/arxiv_*.txt` |
| OpenAlex | 四组查询，每组 max=10 | 40 个文本结果 | 成功；`raw-search/openalex_*.txt`；部分为低相关噪声 |
| DeepXiv | 四组查询，每组 max=10 | 40 raw hits | health/token 成功；`raw-search/deepxiv_*.txt` |
| Semantic Scholar | 四组查询 | 28 raw hits（10+2+10+6） | 首轮成功，随后 HTTP 429；不得将限流后的空结果解释为无论文；`raw-search/s2_*.txt` |
| Scholar Feed | YOLO 定向四组查询 + 图/全文追踪 | 30 条结构化命中；top-k 排序查询及部分追踪遇瞬时上游错误 | 30 条已追加登记；失败不解释为零覆盖；见 `literature_search_log.md` |
| Exa | 四组查询 | 0 | `EXA_API_KEY` 缺失，四次均失败；`raw-search/exa_*.txt` |
| Gemini MCP/CLI | 四组查询（上一批） | 0 可核验命中 | CLI 可认证但搜索链未产出可核验结果；模型记忆候选未入册 |
| Web/arXiv 官方页 | 核心论文核验 | 12 篇完成 | URL 记录在论文页；arXiv raw 结果保留 |

原始结果不会被覆盖：所有 API/CLI 输出保留在 `research-wiki/raw-search/`，调用摘要在 `raw-search/source_calls.log`。

## Obsidian 与旧项目证据

命中笔记：`03-成果与资产/猪CV难题与文献解决方案汇总_2026-08.md`、`03-成果与资产/2026-08-31-YOLO26-Dense-Touching-关系失败内部诊断归档.md`、`03-成果与资产/2026-09-01-YOLO26-O类最终预测竞争结构分析.md`、`03-成果与资产/2026-09-02-YOLO26-SPLIT_TYPE完整候选与局部候选竞争分析.md`、`03-成果与资产/2026-09-02-YOLO26-SPLIT_TYPE原始候选多峰与正确候选存在性分析.md`。

项目诊断证据：7,374 GT；基线失败率 19.84%；Mask-NMS@0.80 将 427/675 个 O 类 GT 转为 C，仍有 196 个 O 类；其中 `SPLIT_TYPE=50`，联合 IoU 增益中位数 0.084、coverage 增益中位数 0.143；31/50 个样本完整与局部候选共同存活，score gap 中位数 0.185。以上是描述性证据，不是因果证明。

## 核心论文（去重后）

| Key | 论文 | 年份/venue | 主题 | 核验与来源 | 状态 |
|---|---|---|---|---|---|
| arxiv:2208.04438 | Ke, Tai, Tang, *Occlusion-Aware Instance Segmentation via BiLayer Network Architectures* | 2022, arXiv | 双层 occluder/occludee 关系 | [arXiv](https://arxiv.org/abs/2208.04438)；BCNet 官方代码链接在摘要 | core-verified |
| arxiv:2210.03551 | Chen, Wu, Merhof, *Instance Segmentation of Dense and Overlapping Objects via Layering* | 2022, BMVC | object layering | [arXiv](https://arxiv.org/abs/2210.03551)；S2、DeepXiv 重复命中 | core-verified |
| arxiv:2001.07475 | Wada et al., *Instance Segmentation of Visible and Occluded Regions for Finding and Picking Target from a Pile of Objects* | 2018, IROS | relook 与实例间关系 | [arXiv](https://arxiv.org/abs/2001.07475)；DeepXiv | core-verified |
| arxiv:2001.07481 | Wada, Okada, Inaba, *Joint Learning of Instance and Semantic Segmentation for Robotic Pick-and-Place with Heavy Occlusions in Clutter* | 2020, arXiv | 实例/语义联合遮挡分割 | [arXiv](https://arxiv.org/abs/2001.07481)；DeepXiv | core-verified |
| arxiv:2012.02107 | Yuan et al., *Robust Instance Segmentation through Reasoning about Multi-Object Occlusion* | 2021, CVPR | ORM 与遮挡顺序推理 | [arXiv](https://arxiv.org/abs/2012.02107) | core-verified |
| arxiv:1906.05896 | Lazarow et al., *Learning Instance Occlusion for Panoptic Segmentation* | 2020, CVPR | OCFusion 二元遮挡关系 | [arXiv](https://arxiv.org/abs/1906.05896) | core-verified |
| arxiv:2206.01942 | Huang et al., *Occlusion-Resistant Instance Segmentation of Piglets in Farrowing Pens Using Center Clustering Network* | 2022, arXiv | 中心预测、DBSCAN、C2M/RC2M | [arXiv](https://arxiv.org/abs/2206.01942)；DeepXiv | core-verified |
| arxiv:2005.10499 | Brünger et al., *Panoptic Instance Segmentation on Pigs* | 2020, arXiv | 猪群 panoptic 数据与评测 | [arXiv](https://arxiv.org/abs/2005.10499)；DeepXiv | core-verified |
| doi:10.1049/ipr2.12880 | Tu et al., *MaskDis R-CNN: An instance segmentation algorithm with adversarial network for herd pigs* | 2023, IET Image Processing | 猪群领域实例分割 | [DOI](https://doi.org/10.1049/ipr2.12880) | core-verified |
| doi:10.25165/j.ijabe.20211404.6398 | Bello et al., *Enhanced Mask R-CNN for herd segmentation* | 2021, IJABE | 猪群 Mask R-CNN | [DOI](https://doi.org/10.25165/j.ijabe.20211404.6398) | core-verified |
| doi:10.1007/s11263-022-01629-1 | Qi et al., *Occluded Video Instance Segmentation: A Benchmark* | 2022, IJCV | OVIS 遮挡评测基准 | [DOI](https://doi.org/10.1007/s11263-022-01629-1) | core-verified |
| doi:10.3390/s20051520 | Zhang et al., *Applications of Deep Learning for Dense Scenes Analysis in Agriculture: A Review* | 2020, Sensors | 农业密集场景综述 | [DOI](https://doi.org/10.3390/s20051520) | core-verified |

## 相关命中与筛选状态

以下条目均保留在原始 raw 文件中；状态用于后续快速识别，不代表质量排序。

### related-unread / pending-verification

- Ling, Huang, Hur, *Humans need not label more humans: Occlusion Copy & Paste for Occluded Human Instance Segmentation*, BMVC 2022, arXiv:2210.03686，数据增强路线。
- Chen et al., *BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation*, arXiv:2003.14031。
- Zhou et al., *Completely Occluded and Dense Object Instance Segmentation Using Box Prompt-Based Segmentation Foundation Models*, arXiv:2401.08174。
- Baselizadeh et al., *Occlusion-Ordered Semantic Instance Segmentation*, arXiv:2504.14054，较新预印本，需核验同行评审状态。
- Qi et al., *Sequential Context Encoding for Duplicate Removal*, NeurIPS 2018, arXiv:1810.08770。
- Liu et al., *Graph Relation Distillation for Efficient Biomedical Instance Segmentation*, arXiv:2401.06370。
- Xue et al., *Transformer-based Open-world Instance Segmentation with Cross-task Consistency Regularization*, ACM MM 2023, DOI:10.1145/3581783.3612493。
- Zhao et al., *SheepInst*, Animals 2023, DOI:10.3390/ani13081338。
- Huang et al., *Automated recognition of postures and drinking behaviour...*, Scientific Reports 2020, DOI:10.1038/s41598-020-70688-6（猪视觉监测，非核心实例分割）。
- *Towards re-identification for long-term tracking of group housed pigs*, Biosystems Engineering 2022, DOI:10.1016/j.biosystemseng.2022.07.017（跟踪，非核心实例分割）。
- *Automated Segmentation and Tracking of Group Housed Pigs Using Foundation Models*, arXiv:2604.03426；日期显示为 2026-04-03，需核验原始页面后再引用。
- *Topology-Aware Query Selection for Surgical Instrument Instance Segmentation*, arXiv:2608.11607；官方 arXiv 元数据与 v1 PDF 已核验，canonical page `papers/zhang2026_topologyaware_query_selection.md`；机制为 fixed Mask2Former 候选集合选择，非猪域因果证据。
- *QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation*, arXiv:2608.29253；官方 arXiv 元数据与 v1 PDF 已核验，canonical page `papers/prytula2026_qcell_recombining_aligning.md`；机制为 MaskDINO query 重组/对齐，非 frozen YOLO26 候选选择证据。

### excluded-noise（仍已登记）

Semantic Scholar fragmentation 查询命中 2 条非 CV 文献；arXiv fragmentation 命中 `cond-mat/0106135`、`1412.6158`、`1410.0637`、`2409.17988`、`2007.03702`、`1605.02746`、`1909.04695` 等非实例分割论文；relation 查询含认知流、竞赛与医学语义分割噪声；pig_animal 查询含肾脏/脑肿瘤、肖像、建筑、珊瑚等非猪实例分割论文；OpenAlex 结果含 CPN、医学分割、活动识别等低相关条目。其完整原始标识和标题见对应 `raw-search/*.txt`，不得因排除而删除。

### duplicate

`2208.04438`、`2012.02107`、`2210.03551`、`2001.07475`、`2209.13899`、`2110.00470`、`2212.09277` 在多个 arXiv/DeepXiv/S2 查询重复出现；canonical entry 只保留一份，来源集合在 raw 文件与本表可追溯。

## 全量 raw-hit 索引

四组 arXiv、四组 DeepXiv、四组 OpenAlex、四组 Semantic Scholar 的逐条返回结果（共 40+40+40+28 个接口结果，含重复和噪声）保留在：

`research-wiki/raw-search/{arxiv,deepxiv,openalex,s2}_{dense,fragmentation,relation,pig_animal}.txt`。

该索引是全量登记的一部分；后续新增来源必须追加文件并更新本册，不得覆盖旧批次。

## 机制分类比较

| 机制族 | 代表论文 | 作用位置 | 主要目标 | 对当前问题的证据强度 |
|---|---|---|---|---|
| 遮挡关系/顺序 | BCNet; ORM; OCFusion; Relook | 网络内部关系模块或重看阶段 | 分离遮挡者/被遮挡者，修正遮挡边界 | 中：支持关系建模，但不覆盖低 IoU 碎片 |
| 层/集合结构 | Layering | 分层分组与实例推理 | 在拥挤场景组织互相重叠实例 | 中：接近候选集合组织，需验证 query 迁移 |
| 中心聚类 | CClusnet-Inseg | 中心预测 + DBSCAN/C2M | 猪仔遮挡下实例关联与完整检测 | 中-高（猪域），但架构机制不同 |
| 数据增强 | Occlusion Copy & Paste 等 | 训练数据构造 | 增强遮挡分布与泛化 | 低-中：不能直接解决推理时共同存活 |
| 评测/基准 | OVIS; Panoptic Instance Segmentation on Pigs | 数据与指标 | 定义遮挡/猪群任务压力测试 | 中：支撑评测设计，不提供修复机制 |
| 后处理去重 | SCE duplicate removal（待核验）; Mask-NMS（项目已有） | 推理后处理 | 删除重复候选 | 低：低重叠互补候选通常不满足合并条件 |

表中“证据强度”是针对本项目 `SPLIT_TYPE` 假设的映射判断，不是论文质量排名；具体作者结果仍以原始来源为准。
