# YOLO26 R006 Oracle 代码与协议独立审查

日期：2026-09-05  
范围：只读审查 `tools/run_yolo26_candidate_perturbation.py`、其导入的 `pigcv_research/scripts/analyze_yolo26_diagnostic_cache_full.py`、当前两份 full same-forward trace 的已存取证。未运行 GPU，未读取 `private_test_v1`、locked-test、guard band 或 target 数据，也未触碰运行中的 R006。

审查结论：当前 R006 不能用于通过 candidate-scorer 或 ranking/competition 的 B2 Gate。修复下列三项后，它可作为**GT-guided 输出集合替换上界**保留，用来否定不可能的路线或给出同一冻结 trace 上的可恢复性上限；它仍不构成部署推理、排序因果归因或 scorer 收益证据。

本审查是同家族独立代码审查：`review_independence: same-family`，`acceptance_status: provisional`。没有跨家族验收通过的结论。

## 修复状态（2026-09-05）

代码已实施：R006 改为最大基数一对一 matching、以相同 batch-2 decoder 对全部 final source 做 XOR guard 并把摘要写入新 `run_summary.json`；I/L/S 的前三阶段只以 box coverage 预筛选，标签改为 `RAW_BOX/TOPK_BOX/CONF_BOX`。下文的 blocker 描述原 R006 实现及修复理由。

旧 R006 输出保持历史证据，不因代码修改而升级。只有在新的空输出目录完成完整双数据集 R006 后，新的 assignment、parity summary 和 counterfactual 结果才可被审查；该重跑仍只是 GT-guided output-set upper bound。

## 已通过的边界检查

### Box-mask coverage 不会漏掉 `mask IoU >= 0.50`

`box_mask_coverage` 使用整数矩形 `[floor(x1), ceil(x2)) x [floor(y1), ceil(y2))` 计算 GT 覆盖率。[`run_yolo26_candidate_perturbation.py:98`](../tools/run_yolo26_candidate_perturbation.py:98) 当前 PyTorch 环境的 `process_mask_native` 最终以 `x >= x1, x < x2, y >= y1, y < y2` 裁剪二值 mask。[`ops.py:477`](C:/Dpan/envsfiles/CondaData/envs/pytorch/lib/site-packages/ultralytics/utils/ops.py:477) 因而 native crop 的像素集合包含在 R006 的 floor/ceil 矩形中，而重建 mask 又包含在 native crop 中。对任意 GT `G` 和候选 mask `M`：

```
IoU(M, G) <= |M intersect G| / |G| <= |box_envelope intersect G| / |G|
```

所以任何 `IoU(M,G) >= .50` 的 raw candidate 必定通过 `.50` coverage 预筛选。[`run_yolo26_candidate_perturbation.py:151`](../tools/run_yolo26_candidate_perturbation.py:151) floor/ceil 只会增加待解码候选，不会造成这个阈值的假阴性。`.70` complete 和 `.10` local 是后续描述字段，不改变该 inclusion 结论。[`run_yolo26_candidate_perturbation.py:194`](../tools/run_yolo26_candidate_perturbation.py:194)

这里不需要改动。

### 已捕获 prototype 消除了“缺 prototype”解释

当前 PigLife 和 Faro full trace 的取证分别记录 4,843/4,843 和 3,484/3,484 最终 RLE 与 same-forward `process_mask_native` 输出 `xor_pixels=0`，且框架输入的 coefficient、box 和 source candidate 都解析回记录的 raw candidate。[`yolo26_full_trace_audit_20260905.md`](yolo26_full_trace_audit_20260905.md)；[`yolo26_faro_full_trace_audit_20260905.md`](yolo26_faro_full_trace_audit_20260905.md) 每图 NPZ 也保留 prototype，故这两个 full trace 没有“prototype 未捕获而无法重建”的上限。

这项证据只覆盖 framework 实际送入 native-mask 分支的最终候选。未进入最终集合的 raw candidate 没有 framework 输出 RLE 可比；由已保存 `prototype + coefficient + box` 重建它们是 mask-head 函数的反事实计算，不说明正常 NMS/postprocess 会输出它们。

## Blocker 与最小修复

### B1. R006 自己的 batch-2 重建没有和 captured framework mask 做 parity

R006 在 `decode_masks` 中每次只放两个 candidate 重跑 `ops.process_mask_native`。[`run_yolo26_candidate_perturbation.py:110`](../tools/run_yolo26_candidate_perturbation.py:110) 这一路径既用于 oracle eligibility，也用于写入 added output。[`run_yolo26_candidate_perturbation.py:157`](../tools/run_yolo26_candidate_perturbation.py:157)；[`run_yolo26_candidate_perturbation.py:273`](../tools/run_yolo26_candidate_perturbation.py:273) 已有 trace audit 验证的是 captured same-forward framework batch 与最终 RLE 的一致性，并没有验证**这个脚本的 batch-2 decoder**与该 captured RLE 的一致性。

当前 `process_mask_native` 先做矩阵乘和上采样阈值，再 crop；其具体实现和内部 chunk 策略由运行时的 Ultralytics 提供。[`ops.py:531`](C:/Dpan/envsfiles/CondaData/envs/pytorch/lib/site-packages/ultralytics/utils/ops.py:531) 没有发现数值差异的证据，也没有不存在差异的证据。尤其在 `.50/.70/.75` 边界，不能把两种 batch context 当作已验证等价。

最小修复：R006 开始选择候选前，对每个 `final_pred_to_source_candidate` source 用**同一 `decode_masks` batch size=2 路径**重建，并与记录的 `final_predictions.mask_rle` 做逐像素 XOR；任一非零即终止。保存 final-source 总数、XOR 最大值、Ultralytics 版本、decoder batch size。非最终 raw candidate 仍标为 `reconstructed raw-mask oracle`，不可改写成 framework 最终候选。

### B2. R006 的贪心一对一分配可能把可行 oracle 判为不可行

R006 按候选数排序后给每个 GT 取第一个未使用 raw source。[`run_yolo26_candidate_perturbation.py:198`](../tools/run_yolo26_candidate_perturbation.py:198) 这不是最大基数二分图匹配。

该逻辑有无 GPU 的最小反例：按代码的目标顺序，候选集为 `A={0}`、`B={1,2}`、`C={0,1}`。贪心选择 `A->0, B->1` 后会令 `C` 失败；但完整一对一分配 `A->0, B->2, C->1` 存在。因此 `oracle_assigned`、`candidate_assignment_conflict`、`raw/topk/conf/mask_separable` 及后续 addition/removal 可能被低估或错误分类。

最小修复：在每图使用精确最大基数二分图匹配；仅在存在多个最大匹配时，再以已预先声明的 `(raw_mask_iou, coverage, score, rank, source_id)` 规则打破平局。修复前不以 assignment conflict 或 oracle coverage 触发任何 Gate。

### B3. 四阶段对 I/L/S 把 box purity 误作 raw-mask 正确性的必要条件

四阶段代码对 I/L/S 使用 `box_cov >= .75 and box_pur >= .75` 作为 `box_correct`，并仅从这些候选构造 `relevant` 后再做真实 mask 重建。[`analyze_yolo26_diagnostic_cache_full.py:563`](../../pigcv_research/scripts/analyze_yolo26_diagnostic_cache_full.py:563)；[`analyze_yolo26_diagnostic_cache_full.py:580`](../../pigcv_research/scripts/analyze_yolo26_diagnostic_cache_full.py:580)

对 raw mask `M` 与其 crop box `B`，`M subseteq B` 保证 `coverage(M,G) <= coverage(B,G)`，但不保证 `purity(M,G) <= purity(B,G)`；box 可多含 GT 外像素。因此 `M` 可以同时达到 `.75` coverage 和 `.75` purity，而 `B` 的 purity 低于 `.75`。这类候选不会进入 `relevant`，会被报告为 RAW/TOPK/CONF failure，尽管真正的 raw mask 可达 C。该问题不影响上节仅以 `.50` coverage 筛选 `IoU >= .50` 的 R006 bound。

最小修复：I/L/S 的 box 预筛选只保留 raw-mask 正确性的必要条件 `box_cov >= .75`，然后在 `reconstruct_candidate_edges` 后用实际 mask 的 coverage/purity 判定；不要以 box purity 剔除待解码候选。输出列名也应改为 `RAW_BOX`、`TOPK_BOX`、`CONF_BOX`，避免把 box feasibility 写成 raw mask formation。

## Oracle 操作、分母与可用解释

R006 的四个条件不应合并成“competition recovery”：

| 条件 | 实际操作 | 可回答的问题 | 不能回答的问题 |
| --- | --- | --- | --- |
| `ADDITION` | 将 GT 选中的非最终 raw mask 插入最终输出 | 正确 mask 被允许加入时 taxonomy 的输出集合上限 | 学习的 scorer 会否选到它 |
| `REMOVAL` | 删除相对该 GT 的 final core prediction，且保护被其他 GT 选中的 source | 删除关联输出的集合效应 | 被删除项是否是模型中的唯一竞争者 |
| `ADDITION_REMOVAL` | 同时插入和删除 | 特定 GT-guided 替换集合的上限 | ranking、NMS、mask 形成各自的因果贡献 |

代码明确将 `ADDITION_REMOVAL` 同时做两种改变。[`run_yolo26_candidate_perturbation.py:257`](../tools/run_yolo26_candidate_perturbation.py:257) 但报告把它称为“因果定位 oracle”。[`run_yolo26_candidate_perturbation.py:345`](../tools/run_yolo26_candidate_perturbation.py:345) 这是超出操作本身的表述。删除的 core prediction 还可能对相邻原本 C 的 GT 有用；脚本会记录 `C_to_failure`，却没有把它作为替换上界的必要回归指标或分母。[`run_yolo26_candidate_perturbation.py:300`](../tools/run_yolo26_candidate_perturbation.py:300)

另有三套分母：`n_gt` 是全部 GT，`target_failed_gt` 是全部基线 non-C GT，`oracle_assigned` 是贪心唯一分配后的 eligible GT。[`run_yolo26_candidate_perturbation.py:345`](../tools/run_yolo26_candidate_perturbation.py:345) 目前汇总只报 transition count 与全体 GT 失败率。[`run_yolo26_candidate_perturbation.py:357`](../tools/run_yolo26_candidate_perturbation.py:357) 这不能直接与计划中的 `>=40% oracle recovery`、`>=30% ranking/competition` Gate 对齐；计划还要求 image-cluster bootstrap 和两个数据集方向一致。[`EXPERIMENT_PLAN.md`](../refine-logs/EXPERIMENT_PLAN.md)

最小修复：每 dataset、每 condition 固定并同时报告：

- baseline non-C `N_fail`；
- exact-matching eligible `N_match`；
- `recovered_to_C / N_fail`、`recovered_to_C / N_match`；
- `C_to_failure / baseline_C`；
- image-cluster bootstrap interval。

只将 `ADDITION`、`REMOVAL`、`ADDITION_REMOVAL` 分开作为 output-set upper-bound 条件。`post_ranking_exclusion` 只能写为“raw score rank 高于该 GT core output 但未在 final set 中”，不能归因给 ranking，因为该脚本不分解 Top-K、confidence、NMS 或最终 mask selection。[`run_yolo26_candidate_perturbation.py:211`](../tools/run_yolo26_candidate_perturbation.py:211)

## 可复用的四阶段资产

| 资产 | 可复用范围 | 继续使用前的最小条件 |
| --- | --- | --- |
| `classify_all`、`matrices`、`normalize_gt`、全 GT transition reclassification | 保持 C/I/L/S/O/M/X/MISS 的同一 evaluator，且能暴露相邻回归 | 保留 R006 已有的 baseline class-map equality 检查。[`run_yolo26_candidate_perturbation.py:330`](../tools/run_yolo26_candidate_perturbation.py:330) |
| `rectangle_metrics` | 对 relation 类的 `.50` mask-IoU availability 做无假阴性的 box coverage 预筛选 | 不把它称作 raw-mask metric；I/L/S 依 B3 改成 coverage-only screen。 |
| `reconstruct_candidate_edges` | 在 frozen `prototype/coefficient/box` 上计算 raw/Top-K/conf 后的真实重建 mask 边 | 先完成 B1 batch parity；仅对未入 final 的候选标为 reconstructed counterfactual。[`analyze_yolo26_diagnostic_cache_full.py:501`](../../pigcv_research/scripts/analyze_yolo26_diagnostic_cache_full.py:501) |
| `audit_cache` 的 schema、map、score 和最终 RLE 检查 | 作为 trace admission 的基础 | R006 当前只检查 JSONL image coverage，未调用它。[`run_yolo26_candidate_perturbation.py:67`](../tools/run_yolo26_candidate_perturbation.py:67) 将其抽成 single-trace guard，并要求 completed status、artifact hash、final-source parity。 |
| `covers_targets` 和 O subtype 分析 | 描述 unit-level separability / final-set现象 | `covers_targets` 已要求 candidate 对全图只有一个 target edge，故其贪心不共享 source；O 分支目前直接以 final mask subtype 定 stage，未经过四个 gate，需单列 `O_final_set_subtype`，不能与其他类的 RAW/TOPK/CONF 比例合并。[`analyze_yolo26_diagnostic_cache_full.py:606`](../../pigcv_research/scripts/analyze_yolo26_diagnostic_cache_full.py:606) |

完成 B1-B3 后的最小可信表述是：在两个冻结、same-forward、provenance-complete trace 上，GT-guided 的一对一输出集合替换具有某个带 image-cluster interval 的 taxonomy recovery upper bound，并同时报告所有 GT 回归。它只用于决定某个候选路线是否值得进入一个独立、可部署的干预试验；不选择 scorer，也不替代 Block 3 的 AP、deletion control 和跨数据集验证。
