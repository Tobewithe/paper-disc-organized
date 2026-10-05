# YOLO26 全量严格质量与同时匹配诊断

日期：2026-09-05。该报告只描述冻结的 PigLife public test（426 图）与 FaroPigSeg test（160 图）trace，不涉及训练、模型选择或受保护数据。

## 结果

严格质量提取覆盖 586 张图、6,226 个 GT、193,413 条稀疏候选质量边。coverage = 交集面积 / GT 面积，purity = 交集面积 / 预测面积；质量条件为：mask IoU >= 0.50、coverage >= 0.75 且 purity >= 0.75、mask IoU >= 0.75。仅使用 box-envelope GT coverage >= .50 的无损预筛选，无 purity 预筛选、候选配额或降采样；复用全分辨率 batch-2 decoder，最终阶段直接读取完整 final-RLE 矩阵。提取耗时 3,023.281 秒。

trace runtime 为 Ultralytics 8.4.100，输入为 imgsz=1024、rect=false、conf=.05、max_det=300；checkpoint 历史 metadata 为 8.4.27。四阶段 membership 嵌套、manifest/GT 数量、source mapping、final-RLE 解码和哈希均通过；final-source XOR 总数/最大值为 0/0。全部 1,191 个历史 IoU .50 失败 GT 的四阶段候选计数逐项一致。

下表报告失败 GT 的数量。单个数字表示 available = matched；双数表示 available / matched。available 是逐 GT 合格候选存在数，matched 是在包含全部原本正确邻居的 all-GT 图上，约束每个 source 最多分配一次后的最大基数匹配结果。

| 数据集/条件 | raw | Top-K | conf | final |
|---|---:|---:|---:|---:|
| PigLife，IoU >= .50，N=394 | 390 | 390 | 373/365 | 373/365 |
| PigLife，coverage/purity >= .75，N=394 | 360 | 344 | 312/310 | 312/310 |
| PigLife，IoU >= .75，N=394 | 330 | 317 | 296 | 296 |
| Faro，IoU >= .50，N=797 | 746 | 698 | 553 | 553 |
| Faro，coverage/purity >= .75，N=797 | 490 | 370 | 224 | 224 |
| Faro，IoU >= .75，N=797 | 395 | 296 | 196 | 196 |

严格 coverage/purity 的 image-cluster bootstrap 95% CI：PigLife raw 360/394 = 91.37% [88.56%, 93.83%]，final available 312/394 = 79.19% [74.94%, 83.29%]，final matched 310/394 = 78.68% [74.46%, 82.77%]；Faro raw 490/797 = 61.48% [58.11%, 64.59%]，final/matched 224/797 = 28.11% [25.06%, 31.04%]。

bootstrap 使用 3,000 次图像簇重采样，seed=20260905；失败 GT 范围条件化于含此类 GT 的图像，PigLife 为 203 图，Faro 为 152 图。区间反映固定图像集的重采样不确定性，不覆盖训练随机性或不同猪舍的独立性。

独立 CPU 审计用 SciPy 最大二分匹配重算了 586 x 3 x 4 = 7,032 个图，并独立重建每个图的连通分量划分；全部匹配基数、边代数、组件统计和 216 个 bootstrap 区间一致。20 条分配缺口记录由 PigLife IoU .50 的 conf/final 各 8 条，加 coverage/purity .75 的 conf/final 各 2 条组成，跨条件和阶段不能按独立 GT 相加。raw/Top-K 两数据集均无分配缺口；Faro 各条件各阶段均无分配缺口。该匹配下 baseline-C 邻居没有可用但未匹配的实例；IoU .75 本身比 baseline-C 的规则更严格。

## 失败分层

严格 coverage/purity 条件下，首次不可用的互斥计数为 PigLife：raw 34、Top-K 16、confidence 32、最终仍可用 312；Faro：raw 307、Top-K 120、confidence 146、最终仍可用 224。它们是按固定顺序记账的标签，不能解释为分别放宽门槛的独立效果。

raw 严格质量缺失的基线类别：PigLife 为 I=10、MISS=7、O=5、L=5、M=4、X=3；Faro 为 I=111、MISS=109、O=46、X=16、L=14、S=9、M=2。final 严格质量仍可用的失败 GT 为 PigLife O=276、M=11、X=25，Faro O=204、M=1、X=19。Faro 在更严格标准下暴露出较大的单 mask 质量缺口；在当前成对质量边定义下，同时分配冲突较少。两者均未唯一定位内部原因。

## 解释边界

这些边是成对质量合格关系，不排除候选对其他 GT 的污染；最大基数匹配是 source uniqueness 诊断，不能替代最终输出集合恢复。结果未区分 box 定位、prototype 表达、mask head 或标注语义的影响，尚无部署收益、AP 增益或 Route-A Gate 证据。固定 conf=.05 下，PigLife 4,843 和 Faro 3,515 个 score > .05 的 raw source 全部位于 Top-300，因此仅扩大 Top-K 的下游来源集合不变。Faro 最终 3,484 个预测与 3,515 个 confidence 输入的差额来自空 mask 过滤。

验收状态：实验完整性 PASS，方法接受 PENDING；Terra 审查为 same-family / provisional。历史 trace 未保留实际 framework prototype 实参，不能将解码一致性解释为历史实参身份验证。

## 下一步

继续既定冻结 trace 诊断：固定 Top-K 与原始分数，分别从同一基线检验单因素 confidence retention、deletion control 与 joint replacement，报告全部 GT 的 COCO AP、关系失败转移和原本正确 GT 回退。两份 public test 已用于探索；不在它们上调阈值或选方法。若后续证据支持可部署路线，才在允许的 train/validation 上学习或定参，并单独进行方法决策。

## 证据路径

- 严格质量：[summary.json](C:/Dpan/codexproject/paper-disc/experiments/yolo26_strict_quality_fulltrace_20260905_v1/summary.json)
- 同时匹配及区间：[summary.csv](C:/Dpan/codexproject/paper-disc/experiments/yolo26_strict_matching_fulltrace_20260905_v1/summary.csv)
- 独立验收：[yolo26_strict_full_integrity_20260905.md](C:/Dpan/codexproject/paper-disc/research-wiki/yolo26_strict_full_integrity_20260905.md)
- 实验计划：[EXPERIMENT_PLAN.md](C:/Dpan/codexproject/paper-disc/refine-logs/EXPERIMENT_PLAN.md)
