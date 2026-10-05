# 既有实验证据取证与重启建议

日期：2026-09-05  
范围：仅审计既有 YOLO26 固定缓存、原始候选 trace、R006/R007 与其脚本；不选择方法路线。

## 裁决

现有工作已经支持一个较窄的描述性结论：同一冻结 YOLO26 评估器在三个数据集的失败率和失败类别构成不同，且 PigLife 与 FaroPigSeg 的部分失败 GT 存在 `mask IoU >= 0.50` 的原始候选。它**尚不支持**“候选竞争/排序是主导根因”或任何候选 scorer 的训练立项。ARIS 机制 Gate 应保持未通过。

## 证据链与主张裁决

| 主张 | 状态 | 已核实证据 | 裁决边界 |
|---|---|---|---|
| 三个数据集的固定 YOLO26 基线存在明显不同的失败结构 | 部分支持 | 统一表给出 PigLife/Faro/Bama 分别 4,474/1,752/1,148 GT，以及 8.81%/45.49%/23.69% 总失败率；PigLife COCO 标签数与 legacy GT 一致。[`experiments/yolo26_unified_diagnostic_20260905/summary.json`](../experiments/yolo26_unified_diagnostic_20260905/summary.json)；[`provenance_audit.json`](../experiments/yolo26_unified_diagnostic_20260905/provenance_audit.json) | 这是继承缓存和既有 evaluator 的描述统计，不是新推理复现，更不是 AP 或跨架构比较。PigLife 全量 raw trace 用 Ultralytics 8.4.100，原始运行时记录为 8.4.27，故只有 GT 分类/计数可比，未见逐预测字节级 parity。[`inference_config.json`](../experiments/yolo26_raw_trace_piglife_full_20260905_v1/inference_config.json) |
| 数据、split 与实例标注可追溯 | 部分支持 | 清单登记 Faro 160、Bama 332、PigLife public-test 426 图；PigLife 派生 COCO 有 4,474 标注，Bama 采用 COCO polygon 而非 pose 标签。[`research-wiki/formal_manifest_audit_20260905.md`](formal_manifest_audit_20260905.md) | 原 118 行 dense-failure 样本清单、`SPLIT_TYPE` 标签来源和 54 图/30 控制组均未恢复；1280/rect 重放是重建样本，不能并入主结果。[`research-wiki/protocol_audit_20260905.md:41`](protocol_audit_20260905.md:41) |
| PigLife 非 MISS 失败通常不是“完全没有好候选” | 部分支持 | 完整 PigLife trace 覆盖 426 图，394 个 legacy failed GT 中 raw/Top-300 good 为 388/387；I/M/X 均为 raw 100%，O 为 289/289。[`piglife_raw_trace_candidate_coverage.json`](../experiments/yolo26_candidate_gate_20260905/piglife_raw_trace_candidate_coverage.json) | 脚本只解码 `Top-300 + box 覆盖>=0.05 + 每 GT box-IoU 前120` 的 screen，故所谓 raw-good 是受筛选器限制的下界，不是对全部 raw mask 的穷尽搜索。[`tools/analyze_yolo26_raw_trace_coverage.py:85`](../tools/analyze_yolo26_raw_trace_coverage.py:85) |
| FaroPigSeg 中有可测的 Top-300 候选损失 | 部分支持 | 797 failed GT 中 raw-good 466、Top-300-good 388，差 78。[`faropigseg_raw_trace_candidate_coverage.json`](../experiments/yolo26_candidate_gate_20260905/faropigseg_raw_trace_candidate_coverage.json) | 这是 mask-IoU oracle availability，不等价于改变 Top-K 可恢复的部署收益；R007 将 K 从 300 到 1000（conf=0.05）没有 GT 分类变化，说明这些例子不能直接转译为门控收益。[`sweep_summary.csv`](../experiments/r007_score_retention_20260905_v1/sweep_summary.csv) |
| `SPLIT_TYPE=50` 显示完整/局部候选竞争 | 未验证（仅历史探索性） | 简报记有 11 缺失、8 丢失、31 共存及 0.185 中位 score gap。[`RESEARCH_BRIEF.md:35`](../RESEARCH_BRIEF.md:35) | 该 50 例来自独立的 1024-square/21,504-position 缓存；缺原始样本清单与标签，可比性和样本归属都不能审计。项目协议已明确不得将其报作正式跨数据集证据。[`research-wiki/protocol_audit_20260905.md:21`](protocol_audit_20260905.md:21) |
| R006 证明排序/竞争导致失败 | 不成立 | R006 的 baseline 分类与保存的 legacy GT 表进行一致性检查，并对全部 GT 重分类；因此它是自洽的 evaluator-level oracle 实验。[`tools/run_yolo26_candidate_perturbation.py:324`](../tools/run_yolo26_candidate_perturbation.py:324) | 候选由 GT mask IoU 选择（>=0.50），addition 直接加入 oracle 候选，removal 删除相对该 GT 的 core 预测；这同时使用 GT 选择和 GT 定义干预，不能隔离模型的 ranking、NMS、mask 形成或关系匹配因果作用。[`tools/run_yolo26_candidate_perturbation.py:168`](../tools/run_yolo26_candidate_perturbation.py:168)；[`tools/run_yolo26_candidate_perturbation.py:257`](../tools/run_yolo26_candidate_perturbation.py:257) |
| R006 的 799 个 `ADDITION_REMOVAL` 转 C 可作为方法潜力 | 仅 oracle 上界 | 全量汇总确有 799 failed-to-C、1 C-to-failure，说明固定分类器可在 oracle 替换输出时显著改变标签。[`counterfactual_summary.csv`](../experiments/r006_candidate_perturbation_20260905_v1/counterfactual_summary.csv) | 这不是可部署 intervention，也不能作为“竞争占比”或预期 AP 增益。脚本的 source candidate 每图强制一对一分配，且按 GT IoU 而非模型分数选取，进一步改变了自然候选集。[`tools/run_yolo26_candidate_perturbation.py:199`](../tools/run_yolo26_candidate_perturbation.py:199) |
| R006 的后验 competition gate 支持候选路线 | 未验证 | v3 后验表报告关系失败 oracle recovery 281/354（Faro）和 291/337（PigLife），并把 pre-Top300、置信度损失、final 共存做并集。[`competition_gate_summary_v3.csv`](../experiments/r006_candidate_perturbation_20260905_v1/competition_gate_summary_v3.csv) | 指标依赖 R006 oracle-selected candidate，且将 final complete/local 共存纳入 competition；共存本身不能区分错误排序、NMS、匹配规则或 mask 构造。此表是后验描述，不能替代预注册的因果 Gate。[`tools/analyze_r006_competition_gate.py:172`](../tools/analyze_r006_competition_gate.py:172) |
| R007 表明简单放宽 Top-K/置信度不能修复当前失败 | 支持，但仅限冻结 trace | 脚本重建 `Top-K -> score > conf`，并要求 K=300/conf=.05 的最终预测与 GT taxonomy 完全一致才继续。K=1000/.05 与基线相同；降至 .01 使总失败从 19.13% 升至 23.26%，仅 12 failed-to-C、269 C-to-failure。[`tools/run_yolo26_score_retention_sweep.py:169`](../tools/run_yolo26_score_retention_sweep.py:169)；[`sweep_summary.csv`](../experiments/r007_score_retention_20260905_v1/sweep_summary.csv) | 它不含 NMS（脚本明确如此），只覆盖两个阈值与两数据集，且是缓存重放；不能声称全模型、所有 threshold 或跨数据集的门控机制已被排除。[`tools/run_yolo26_score_retention_sweep.py:301`](../tools/run_yolo26_score_retention_sweep.py:301) |
| Mask-NMS 是有价值的主方法 | 不成立 | 固定缓存中 Mask-NMS 改变错误构成；项目已将其定义为 control。[`research-wiki/yolo26_gate_uncertainty_20260905.md`](yolo26_gate_uncertainty_20260905.md) | 无新 forward、无统一正式候选溯源，且它不证明残留低重叠失败的机制；只能保留为基线控制。 |

## 阻断缺陷

1. **协议混用。** 1024/square 固定缓存、1280/rect 重建样本、8.4.27 与 8.4.100 都存在；`SPLIT_TYPE` 历史结果不可与后续正式表合并。
2. **raw 语义尚非统一。** PigLife 新 trace 保存全 raw NPZ，但 coverage 脚本只对 screened candidates 解码；旧 scalar DB 又只保留 O 类失败的候选行。缺行是保留盲区，不能解释成候选不存在。[`candidate_coverage_by_class.csv`](../experiments/yolo26_candidate_gate_20260905/candidate_coverage_by_class.csv)
3. **R006 的名称误导。** 它检验“GT 知道正确候选并可替换/删除输出时的上界”，不是 candidate addition/removal 的可部署因果试验；不得把其 recovery 或 gate share 用作 scorer 路线证据。
4. **基线 parity 粒度不足。** 现有检查保证 fixed classifier 的 GT 标签一致，却没有对新 PigLife 8.4.100 forward 与原 8.4.27 final mask/source mapping 做逐图逐预测对照。
5. **没有正式性能终点。** 本轮核心表均为 failure taxonomy/GT oracle；无同一 frozen manifest 上的 COCO mask AP、置信区间和独立模型干预结果。

## 最小可信重启实验

不训练、不选方法。以一个冻结的 PigLife + FaroPigSeg 样本清单执行一次同版本、同 checkpoint、同预处理的前向，并为每图保留：全 raw score/box/mask coefficient/prototype、Top-K indices、每个最终预测到 raw source 的映射、COCO GT hash 与最终预测 JSON。先验定义所有 IoU/coverage/purity、失败类别、candidate complete/local 和匹配阈值。

验收顺序：

1. 逐图验证 raw 重建的 `K=300, conf=.05` 最终预测与保存最终输出完全一致，并生成 manifest/结果哈希。
2. 在**全部 raw candidates**（非 box-screen）上计算 raw、Top-K、conf、final 四阶段的候选可用性；按数据集、失败类和 image-cluster bootstrap 报告。
3. 仅对“raw-good 且经某个明确门丢失”的预注册子集，运行两个可区分的 oracle：保留正确候选、删除明确竞争候选；分别报告相邻 GT 回归。任何结论仍标注 oracle，只有在两数据集方向一致后才决定是否进入方法设计。

该实验是当前最小能修复 provenance、阶段语义和因果边界的工作；它不会预设 candidate scorer、mask 分支或训练方案。
