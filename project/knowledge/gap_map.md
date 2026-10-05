# Gap Map

更新时间：`2026-09-07`。本文件只表达证据支持的缺口与未成立推断，不选择论文主贡献。

最新续接：R005c 五条件计算和统计已完成，统一 .01 保留恢复 12、退化 269；GT 联合恢复 43 个所选目标及 2 个邻居。下一步不再重复同一阈值实验。用户选择优先研究 Faro 单 mask 质量；R005d 将 307 个 raw 严格质量缺失失败分为框支持不足 37、支持可行但 mask 缺失 270。原始标签到 COCO 的栅格转换一致；modal/amodal 政策仍待原文核验。这些结果正在补齐独立审查，不接受 scorer 路线。最新报告为 `yolo26_confidence_diagnostic_20260907.md` 和 `faro_mask_quality_20260907.md`；下表中 2026-09-05 的“下一步”保留为历史决策依据。

| Gap ID | 文献已覆盖机制 | 项目观测 | 尚未成立的推断 | 可审查下一步 |
|---|---|---|---|---|
| G1-low-iou-fragment | Layering、BCNet、ORM、OCFusion 主要处理密集重叠、遮挡边界或遮挡顺序 | `SPLIT_TYPE=50` 中联合 IoU 增益中位数 0.084、coverage 增益 0.143；31/50 完整与局部候选共同存活 | 低 IoU 互补碎片是否构成独立机制类别，文献没有直接证明 | 检索 low-IoU fragmentation、partial mask candidates、instance split error 的原始论文；对候选对做关系标注 |
| G2-candidate-selection | Mask Scoring R-CNN 已提出 mask 质量评分；已核验的 Topology-Aware Query Selection 使用 frozen Mask2Former 候选图、基数预测和精确子集选择 | 严格 coverage/purity .75 raw 可用性为 PigLife 360/394、Faro 490/797；final 可用/匹配为 312/310 和 224/224 | 通用关系选择已有直接近邻；GT 引导恢复不能证明排序因果或部署收益；固定 conf=.05 单独扩大 Top-K 无效 | 从同一基线分别检验置信度保留、删除及联合操作，并保留整体 AP 与全部回退 |
| G3-pig-domain-transfer | CClusnet-Inseg 用中心+DBSCAN；Pig Panoptic 提供猪群任务基线；MaskDis/Enhanced Mask R-CNN 为领域先例 | 旧诊断显示 36/50 `SPLIT_TYPE` 有 box 参与或 box-mask 混合 | 猪域中心聚类或遮挡机制能否迁移到 YOLO26 query 竞争未知 | 核对 PigLife、FaroPigSeg、BamaPig2D 的标注语义与评测，建立跨数据集失败切片 |
| G4-evaluation | TIDE 从同一基线分别修正错误并测 AP；OVIS 提供遮挡评测语境 | 当前已分别报告 COCO AP 和关系失败，但 GT 引导联合增删没有可部署 AP 证据 | 仅 AP 或仅关系恢复均不足以证明机制；连续增删的差额不能直接叫作独立因果贡献 | 同基线单因素干预，保留原分数、完整 GT 和全部退化；并列 AP 与关系指标 |
| G5-source-coverage | arXiv/OpenAlex/DeepXiv 已有结果；Scholar Feed 本轮新增结构化检索；Exa/Gemini/Zotero/WebSearch 仍有缺口 | Exa 缺 key；S2/Scholar Feed 有限流或瞬时错误；Gemini CLI 本轮无可核验命中；Zotero/WebSearch 未暴露 | 空结果不是“无论文”，覆盖率仍不完整 | 复用相同 query 做增量重试；修复 Exa/Gemini/Zotero/WebSearch 后按 canonical key 去重 |
| G6-fragmentation-vocabulary | 已核验 Topology-Aware Query Selection 明确讨论 duplicate/fragmented/merged 输出；QCell 讨论重叠细胞的 query 重组/对齐与 amodal 监督 | 项目内部 `SPLIT_TYPE` 是工作标签，尚无统一文献术语映射 | 文献相邻机制不等于同一标注语义、架构或失败原因；QCell 不以碎片化为明确主问题 | 比较原始失败定义和监督要求；用维度化对照限定创新表述，不凭这两篇宣称全面查新通过 |
| G7-evaluation-matching | Part-aware matching 工作区分多种分配约束；PigDetect/PigTrack 提供猪舍评测语境 | R005b 已包含全部正确邻居：raw/Top-K 无分配缺口，PigLife conf/final 在 IoU .50 各缺 8、严格 coverage/purity 各缺 2，Faro 均为零 | 成对合格与 source 唯一分配不排除其他 GT 污染；跨质量/阶段的 20 条缺口记录不能当作 20 个独立 GT | 对实际输出集合重新分类所有 GT，并列恢复、回退与 AP，不将匹配可行性解释为方法收益 |
| G8-training-vs-selection | Location-aware Copy-Paste 与 SEE/MaskUno 分别代表训练分布、伪标签完整性和 mask-head 竞争干预 | 项目已有 Mask-NMS 后处理证据，但尚未将训练侧与推理侧干预置于同一失败分层 | 未来改进若有效，原因可能是数据覆盖、mask 头竞争或候选选择，不能仅凭整体 AP 归因 | 为候选干预建立最小正交对照：训练分布、mask 生成、候选选择分别单独改变 |
| G9-protocol-parity | 当前诊断采用新 full trace：1024、rect=false、Ultralytics 8.4.100；旧 1280/rect 与 8.4.27 重放保留为历史审计 | 586 图 raw/Top-K/source mapping 和 8,327 个最终 mask 重建一致，XOR=0 | 历史 trace 未记录实际 prototype 实参，不能倒推实参身份；公开 test 已用于探索 | 使用当前全量证据链；未来调参/学习在允许的 train/validation 上完成，并披露 test 探索使用 |

## 当前结论边界

文献支持“显式实例关系、层/集合结构和领域特定聚类”是可复用机制族；不支持直接宣称它们解决低 IoU 完整/局部候选共存。当前最稳妥的研究缺口是：对该候选集合竞争现象进行机制化验证，并检查其跨数据集泛化，而不是预先确定某个网络或损失。

新增论文进一步表明：低阈值匹配可以改变碎片化错误的统计解释（Großkopf et al.），训练侧遮挡增强与伪标签完整性是独立于候选选择的替代解释（Nguyen et al.; He et al.），而 MaskUno 的 mask-kernel 专门化只提供 ROI 架构邻近证据。猪域基准支持按真实猪舍难例报告泛化，但没有直接解释 YOLO26 mask 候选竞争。

## Imported experiment linkage
- G1: `exp:residual-o-mechanisms` and `exp:candidate-competition` provide descriptive fixed-cache evidence.
- G2: `exp:mask-nms-residual-o` establishes duplicate suppression as a baseline, not a selected contribution.
- G3/G4: `exp:cross-dataset-failure-structure` and `exp:yolo26-failure-taxonomy` motivate cross-dataset and cross-architecture validation.

## Replay audit update (2026-09-05)

- A 1280/rect geometry-aligned replay on a reconstructed sample produced 40/41 relation components as mask/relationship failures and one low-score component, with zero duplicate-survival components.
- This does not close G2 or G9: the archived sample list is still unavailable, and the replay cannot be merged with the archived 25-component duplicate-survival result. Exact 8.4.27 aggregate replay is now available, but per-row values are runtime-sensitive.

## Union-oracle update (2026-09-05)

- A retained-output oracle over 59 reconstructed failure rows found Top-300 best-single IoU >= 0.50 in 57/59, best-pair union success in 32/59, and all-associated-candidate union success in 57/59. Best-pair success was O/M/X = 5/25, 8/9, 19/25.
- This narrows G2: complementary candidates are plausible for many reconstructed M/X failures, whereas O is not generally explained by pairwise union. The result still does not distinguish candidate association, mask ownership, or postprocessing and cannot replace the missing original sample list.

## Full PigLife raw-trace update (2026-09-05)

Historical restricted/downsampled search, superseded by the full-resolution audit below.

- A new provenance-complete same-forward trace over all 426 PigLife test images found raw mask-IoU>=0.50 candidates for 388/394 failed GT rows and Top-300 retention for 387/394. I/L/M/X raw coverage was 100%; MISS was 8/13.
- This narrows G2 on PigLife: candidate absence is not the dominant explanation for non-MISS failures, while the remaining Top-300-sensitive slice is too small to justify a scorer without perturbation. Cross-dataset replication and candidate removal/addition remain open.

## FaroPigSeg raw-trace update (2026-09-05)

Historical restricted/downsampled search, superseded by the full-resolution audit below.

- FaroPigSeg replication gives raw candidate oracle coverage 466/797 (58.47%) and Top-300 coverage 388/797 (48.68%); 78 raw-good rows are removed by Top-300.
- Together with PigLife, candidate availability clears the 40% descriptive threshold on two datasets, but the observed selection-loss fraction is below the 30% competition gate. This supports perturbation testing, not a scorer decision.

## Current full-resolution evidence (2026-09-05)

- Failed-GT raw/Top-K/conf/final availability is PigLife 390/390/373/373 of 394 and Faro 746/698/553/553 of 797. The earlier restricted/downsampled counts cannot establish current raw absence or selection fractions.
- R006 v2 achieves 322/394 and 477/797 joint recoveries with 0 and 1 correct-GT regressions. Terra final integrity PASS is same-family provisional; an optimal bound, internal cause or deployable gain is not established.
- Every raw candidate with score > .05 is already in Top-300 on both frozen traces. Only-Top-K expansion is an exact no-op for the confidence-passing source set. Sequential first-loss labels and isolated interventions answer different questions.
- Strict-quality extraction and all-GT matching are complete over 6,226 GT: 193,413 edges, final-source XOR=0, and 20 condition-stage assignment-deficit records. Method selection remains gated; no model, loss or scorer has been selected.
- Official v1 PDFs for Topology-Aware Query Selection (2608.11607) and QCell (2608.29253) have been verified and canonically ingested. They constrain generic selector/recombination novelty claims; their surgical/cell evidence does not establish pig-domain transfer or project failure causality.
