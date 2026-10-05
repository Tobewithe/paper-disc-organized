# COCO 结构诊断实验完整性审计

日期：2026-09-11  
审计者：任务隔离的 Codex 审计子代理（同模型家族、只读语义审查）  
独立性：`same-family`  
接受状态：`provisional`  
总体结论：**WARN；未发现使主 300 图候选追踪数据作废的 fatal 问题。**

本审计按 `experiment-audit` 检查 A–F，但父任务明确禁止再启动新审查代理。因此，本文件是独立任务上下文中的同家族临时审查，不是技能所要求的跨家族接受结论。审计未修改实验源码或远端输出。

## 已核验范围

- 源码：`structure_candidate_trace.py`、`structure_features.py`、`frozen_mechanism_probe.py`、`inspect_runtime_path.py`、`summarize_structure_trace.py`、`report_structure_diagnostic.py`。
- 计划与叙述：`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`、`STRUCTURE_RESULTS_20260911.md`。
- 主输出：`structure_smoke4_20260911`、失败的 `structure_train_witness32_20260911`、完成的 `structure_train_witness32_v2_20260911`、完成的 `structure_main300_20260911`，以及本地同步的汇总文件。
- 未核验：本轮没有对应新输出的 `frozen_mechanism_probe.py` 干预数值；没有复算每个 raw NPZ 的内容哈希；没有做跨家族外部复审。

## A. GT 来源、索引重放与参数隔离：PASS（带范围限定）

1. GT 来自 COCO 官方 `instances_{split}2017.json`，以 annotation ID 为实例并排除 crowd；类别映射使用 COCO category ID 的排序。候选选择、框/掩码官方匹配和像素标签均未使用模型输出作为 GT（`structure_candidate_trace.py:46-48,119-130`；`structure_features.py:53-68`；`summarize_structure_trace.py:20-42`）。
2. 300 图运行逐图重新调用官方冻结权重，禁用 TF32，且 `head.end2end=False`。NMS 以 `return_idxs=True` 返回 source index；代码逐图断言 NMS 输出等于运行时进入 mask 构造的输出，随后断言 raw coefficient 和缩放框按 source index 精确等于最终捕获值（`structure_candidate_trace.py:72-75,94-105`）。
3. val 运行还逐图把 `coeff/boxes/detections/proto` 与既有 full-val cache 比较，要求形状相同且最大误差严格为 0（`structure_candidate_trace.py:106-112`）。300 个 witness 完成，缓存误差为 0；主回执记录 300 图、3,635 GT、1,174 对、66,960 feature rows。
4. 模型处于 `eval()` 且 `requires_grad_(False)`；表示探针只拟合临时 ridge readout，不更新 YOLO 参数（`structure_candidate_trace.py:72,94`；`structure_features.py:36-48`）。`frozen_mechanism_probe.py` 的 oracle 梯度也只作用于临时 `2x32` coefficient tensor，并有零干预官方解码重放与坐标重参数化不变性断言（`frozen_mechanism_probe.py:159-175,199-215`）。本轮未审到该脚本的新运行输出，因此这里只接受源码隔离设计，不接受其未提供的效果数值。
5. 运行类型为 one-to-many + class-aware NMS；`inspect_runtime_path.py` 保存了实际 head 状态、Ultralytics 源码和单图 FP32 cache replay（`inspect_runtime_path.py:14-43`）。

结论：未发现 fake GT、source-index 近似反查或模型参数被探针暗中更新。val 官方任务指标属于 `real_gt`。GT 拟合的表示读出和空间 oracle 必须另标为 oracle diagnostic，不能当作模型的无 GT 推理性能。

## B. 分母、归一化与代理范围：PASS / WARN

- PASS：阶段率和官方恢复率以所选 300 图的全部 3,635 个非 crowd GT 为微平均分母，保留漏检；高组为 `ICI > 0.5 + 1e-10`。pair 表保留 1,174 个同类相邻 GT 对；`pair_exists` 的逻辑等价于要求两侧集合中存在两个不同 source candidates（`structure_candidate_trace.py:39-40,119-139,148-154`）。
- PASS：未发现用模型自身最大值、均值或最佳输出归一化来制造接近 1 的分数。AUC 是标准正负排序统计；`relative_logit_change` 除以原始 logit RMS 只是有原始量并列的干预幅度诊断，不是主性能分数（`structure_features.py:28-30`；`frozen_mechanism_probe.py:216-231`）。
- WARN：feature probe 每图、每 pair type 只哈希选择一对，而不是使用全部合格对（`structure_features.py:19-26`）。主 proto 数值的 186/176/167 是“每图至多一对且像素合格”的富集队列，不能外推到 877 个全部高相邻对。
- WARN：val 像素标签的一半用于同图 readout 拟合，另一半用于棋盘格测试。这是 GT-informed within-image oracle，不是仅在 train2017 拟合后对 val 的模型泛化测试（`structure_features.py:74-97`）。结果报告已按 oracle 表述，故不构成假 GT，但与确认性 val 评估必须隔离。
- WARN：图像来自此前探索过的 dense-enriched 300 图。bootstrap 只刻画该固定富集样本内、以图像为簇的点态不确定性，不能恢复 COCO 总体代表性；没有多重比较校正（`report_structure_diagnostic.py:33-53,109-110`）。

## C. 输出存在性、计数与聚合：PASS（溯源封存 WARN）

- 主 300 图输出完成且计数精确对齐：`instances.csv` 3,635 数据行，`pairs.csv` 1,174 行，`features.csv` 66,960 行，`feature_skipped.csv` 159 行，`witness.csv` 300 行，`stage_summary.csv` 3 行。`COMPLETE.json` 中相应计数一致。
- train witness 初次运行只留下 `protocol.json`，因所选 JPEG 不在挂载池而失败；它不是完成实验。v2 明确限定 37,855 个实际存在的 train JPEG，排除 1,200 个 adaptation IDs，并完成 32 图、380 GT、106 pairs、6,384 feature rows。结果叙述和 tracker 已如实区分失败目录与 v2。
- 逐 GT 复算确认高组 `bbox75=72.2081%`、`segm75=53.0457%`、`score_available75=79.8223%`、`nms_available75=73.8579%`；新 `segm75` 与历史 `official_mask75_recovered` 零分歧，新 `bbox50` 与 cache mapping 零分歧。报告中的 5.9645 个百分点差值与配对 JSON 一致。
- Proto cohort 复算确认：all eligible 186；A bbox50 matched 176；A、B 均 bbox50 matched 167。176 队中 oracle AUC 0.992852、actual-A AUC 0.934085、配对差 0.058768；167 队中 pair-difference AUC 0.969876。
- WARN：`COMPLETE.json` 只哈希输出目录根部文件，没有给 `raw/*.npz` 建逐文件或聚合 manifest；而 0.75 阶段后处理依赖 raw arrays（`structure_candidate_trace.py:140,154`；`summarize_structure_trace.py:23-31`）。已生成的派生 CSV 有哈希且当前数字内部一致，但从 raw 到派生结果的链路不是完整的防篡改封存。
- WARN：本地 `PAIRED_STRUCTURE_ANALYSIS.json` 是 reporter 的派生输出；reporter 校验主回执列出的根文件及 summary 的三个派生文件，但不校验 raw 目录，也不把自己的输出纳入先验回执（`report_structure_diagnostic.py:28-35,111-117`）。

## D. 实际调用与 phantom/dead code：PASS（轻微清理项）

- `TraceCapture.postprocess/construct_result`、`matrix_iou`、`pair_exists`、`probe_image`、`choose_pairs`、`readout`、`auc`、`cosine` 均处在已执行的 `--features` 主路径；三 seed 分组若不是恰好三行，summarizer 会直接失败（`structure_candidate_trace.py:25-40,83-90,121-147`；`structure_features.py:19-48,90-107`；`summarize_structure_trace.py:54-67`）。
- 未发现报告引用了从未调用的指标函数。轻微死代码包括 candidate trace 的 `anchors`、feature probe 未使用的 `seed_offset`，以及 summarizer 导入但未使用的 `pair_exists`；它们不影响现有数值。
- `frozen_mechanism_probe.py` 的主要函数由其 `main()` 串联，但本轮指定的新远端目录不含该脚本的新结果，因此不可据源码宣称干预已成功或有效。

## E. 表示读出、空间几何与 matched cohort：WARN

1. **Matched cohort 非对称。** `actual_a_auc` 仅在 annotation A 有官方 bbox50 mapping 时生成；B 不必匹配。A 由 COCO annotation 遍历/哈希选中 pair 的原有顺序决定，不是对 A/B 对称平均（`structure_features.py:98-103`）。176 队的 +5.877pp 只能表述为“像素合格且 A bbox50 matched 的目标 A 队列”。报告已在 `STRUCTURE_RESULTS_20260911.md:53` 明示；不能把它改写为两实例均匹配或 pair recovery 改善。
2. **167 队是更严格但仍为 GT 选择。** `actual_pair_difference_auc` 需要 A、B 均 bbox50 matched；`(c_A-c_B)^T p(x)` 的提升是在 GT 选择的 pair-exclusive pixels 上计算，且没有背景/覆盖约束。因此它支持相对响应假说，不是可部署方法收益（`structure_features.py:101-103`；`report_structure_diagnostic.py:88-106`）。
3. **几何对照很强。** all-eligible proto oracle AUC 0.9919，而二维归一化坐标 AUC 0.9613；oracle-coordinate 差只有约 3.06pp。pair 组成尚未按类别、面积、距离或接触形态匹配，不能把高 AUC 解释为纯网络实例身份信息。
4. **空间折叠并不独立。** 4x4 checkerboard 的训练/测试块相邻，深层 receptive fields 跨块；像素也来自同一图、同一实例对（`structure_features.py:69-91`）。区间不能解释为跨图泛化。
5. **seed 含义有限。** Proto 恰为 32 维、coordinate 为 2 维，因此这两项不触发随机投影；其主 AUC 三个 seed 实际相同，seed 只改变 shuffled-label null。通道数大于 32 的层才使用三个随机高斯投影（`structure_features.py:36-48,95-97`）。三行平均不是三次模型训练或三次独立读出复现；当前报告已说明这一点。
6. **跨层比较不完备。** 通道大于 32 的层只看 32D 随机投影，没有原维数 readout 或与坐标等容量的构造，不能据跨层 AUC 排序定位信息在哪一层丢失。
7. **区域并非全局独占。** `ma`/`mb` 只互相排除并排除 crowd，没有排除与第三个 ordinary instance 的重叠；`frozen_mechanism_probe.py` 的 `own/same/union` 指标也没有单列 GT overlap（`structure_features.py:55-68`；`frozen_mechanism_probe.py:184-191,147-157`）。这低于计划中“GT overlap 单列”的要求。
8. `normalized_center_distance` 用 A 的面积尺度归一化，交换 A/B 会改变数值（`structure_features.py:63-65`）。当前只记录而未据它做匹配控制，所以现有主数值不因此错误；以后若用于控制，应改为对称尺度或明确方向。

## F. 评价类型：PASS（分类必须保持）

| 输出 | 类型 | 可支持的上限 |
|---|---|---|
| 候选 availability、官方 bbox/segm R50/R75 | `real_gt`，COCO 数据集 GT | 所选 300 图上的描述性阶段/恢复率 |
| `auc` ridge readout | `real_gt_oracle_probe`，同图 GT 拟合与测试 | 固定线性探针对空间块的可读出性 |
| `actual_a_auc` / pair-difference AUC | `real_gt_diagnostic`，GT 选 pair/像素与 bbox-matched cohort | 原始未 crop logits 的像素排序诊断 |
| `spatial_oracle` coefficient direction | `gt_oracle_diagnostic`，同图 GT 生成方向 | 机会/可表达性诊断；不能称 AP 上界或部署效果 |

没有 synthetic/model-generated GT。Oracle 使用被源码和结果报告明确披露，因此诚信上为 WARN 范围限制，不是 fake-GT FAIL。

## 对结果主张的影响

- **支持（限定样本）**：300 图 source-index 生命周期、缓存严格重放、候选阶段可用率、同阈值官方 Box/Mask R75、逐 GT/逐 pair 数值及图像簇点态区间。
- **支持为描述性现象**：高组 score→NMS 的 bbox75 availability 下降 5.96pp；低组也下降 5.09pp，因此不能称为高拥挤特有机制或因果归因。
- **支持为 oracle/readout 线索**：176 队上 proto oracle 比 actual-A 高 5.877pp；167 队上相对响应比 single-A 高约 3.00pp。两者都不是方法增益、AP 增益或根因证明。
- **仅部分支持 C1**：证据把后续检查缩小到严格定位候选筛选和实例相对响应，但类别/尺度/距离 matched controls、被抑制候选归属、共同支持域、背景与覆盖代价仍缺失，尚未建立“具体阶段导致高拥挤额外失败”的因果链。
- **不支持 C2**：没有新方法、同预算基础对照、三模型种子或正式全 val5k 验证；没有重启训练。

## 必须保留的限定语

1. 不得把 candidate availability 称为 AP、可实现 recall 上界或同一预测的因果归属。
2. 不得把 bbox 与 mask 的独立 COCO matching 差直接归因给同一候选的 coefficient。
3. 不得把 176 队 oracle-minus-actual 写成两实例均匹配、跨图泛化、模型提升或可部署干预。
4. 不得把 3 个 projection seeds 当作 3 个模型 seed；proto/coordinate 主 AUC 本身没有随机投影复现。
5. 不得宣称根因已定位；当前最强结论是“存在值得做有界干预的候选阶段和相对响应线索”。

## 最终判定

**Fatal：无。Overall：WARN。** 主数据的索引、GT、计数和报告数字没有发现已证实的致命错误。警告来自 oracle/非对称 matched cohort、强几何可分性、未匹配组成、空间 RF 泄漏、区域 overlap 未单列、raw 目录未完整封存，以及确认性控制尚未实现。当前 `STRUCTURE_RESULTS_20260911.md` 基本保持了这些边界，可作为第一轮探索性诊断报告；不能升级成机制根因或方法有效性结论。
