# Output Manifest

2026-09-13 S054：候选归属响应诊断完成。[结果报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md)、[协议](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/candidate_ownership_20260913/protocol.json)、[指标](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/candidate_ownership_20260913/metrics.csv)、[回执](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/candidate_ownership_20260913/COMPLETE.json)索引75条区域记录及15份候选logit/掩码产物。没有训练或定时任务修改。

2026-09-12 S053：15目标P3混合/独占格拆分完成，当前事实覆盖S000—S053。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md)、[冻结计划](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CELL_PARTITION_PLAN_20260912.md)、[完整统计](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/feature_cell_partition_20260912/ANALYSIS.json)、[产物回执](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/feature_cell_partition_20260912/ANALYSIS_RECEIPT.json)索引780条指标、15份分区与子集/掩码/P3通道值、脚本及图表。没有训练或定时任务修改。

2026-09-12 S052：完成44目标原型分支局部特征插入/恢复，当前事实覆盖S000—S052。[结果报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md)、[计划](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FEATURE_SWAP_PLAN_20260912.md)、[原始指标](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/proto_feature_swap_20260912_v2/metrics.csv)、[全部分组](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/proto_feature_swap_20260912_v2/ANALYSIS.json)、[后处理回执](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/proto_feature_swap_20260912_v2/ANALYSIS_RECEIPT.json)索引全部脚本/44份掩码与网格/3例特征张量/图表。首次失败单列proto_feature_swap_20260912/FAILED.json；无训练、无自动化修改。

2026-09-12 S047：新增[邻居背景替换结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NEIGHBOR_BACKGROUND_RESULTS_20260912.md)、[原始统计](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/neighbor_background_20260912/ANALYSIS.json)与[案例图](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/neighbor_background_20260912/CASE_PANEL.png)；64图输入干预完成、无训练、未确认稳定正常恢复，事实记录已更新。

事实沉淀入口：[RESEARCH_FACTS.md](C:/Dpan/codexproject/paper-disc/RESEARCH_FACTS.md)，2026-09-12建立，覆盖可核实的历史COCO审计和S000—S046，区分测量、解释、更正及未执行计划。最新补录：[S046稀疏身份提示结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SPARSE_IDENTITY_RESULTS_20260912.md)。

| Timestamp | Skill | File | Stage | Description |
| --- | --- | --- | --- | --- |
| 2026-09-11 03:27:03 | experiment-plan | refine-logs/coco-evaluation/audit_existing_protocol.py | audit | Read-only GT/prediction audit and official COCOeval regression fixture |
| 2026-09-11 03:27:03 | experiment-plan | refine-logs/coco-evaluation/PROTOCOL_AUDIT_20260911_032703.json | audit | Verified inputs, hashes, density composition and reproduced empty-category bug |
| 2026-09-11 03:27:03 | experiment-plan | refine-logs/coco-evaluation/COCO_INSTANCE_MANIFEST_20260911_032703.csv | data | 36,335 valid noncrowd COCO val2017 instances with GT-only density attributes |
| 2026-09-11 03:34:12 | experiment-plan | refine-logs/coco-evaluation/audit_bootstrap_support.py | audit | Replays existing bootstrap draw sequence to assess empty-category branch exposure |
| 2026-09-11 03:34:12 | experiment-plan | refine-logs/coco-evaluation/BOOTSTRAP_SUPPORT_AUDIT_20260911_033412.json | audit | Five model/draw combinations trigger the reproduced bug; AP impact not computed |
| 2026-09-11 03:28:26 | experiment-plan | refine-logs/coco-evaluation/EXPERIMENT_PLAN_20260911_032826.md | plan | Chinese COCO-only metrics and five-block claim-driven experiment plan |
| 2026-09-11 03:28:26 | experiment-plan | refine-logs/coco-evaluation/EXPERIMENT_PLAN.md | latest | Latest scoped COCO experiment plan |
| 2026-09-11 03:28:26 | experiment-plan | refine-logs/coco-evaluation/EXPERIMENT_TRACKER_20260911_032826.md | tracker | Completed input checks and outstanding model evaluation/training steps |
| 2026-09-11 03:28:26 | experiment-plan | refine-logs/coco-evaluation/EXPERIMENT_TRACKER.md | latest | Latest scoped COCO tracker |
| 2026-09-11 03:28:26 | experiment-plan | refine-logs/coco-evaluation/PAPER_REVIEW_20260911_032826.md | review | Constructive review linking current claims to measurable next evidence |
| 2026-09-11 03:28:26 | experiment-plan | refine-logs/coco-evaluation/PAPER_REVIEW.md | latest | Latest scoped paper review |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/scripts/run_fast_bootstrap.py | implementation | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/scripts/evaluate_stratified_benchmarks.py | implementation | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/scripts/coco_instance_metrics.py | implementation | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/scripts/test_coco_instance_metrics.py | implementation | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/scripts/report_coco_instance_metrics.py | implementation | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/scripts/validate_coco_instance_run.py | implementation | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/analysis.log | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/attempt1_run.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/bootstrap_draws.npz | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/bootstrap_metrics.npz | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/execution.log | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/gt_matches_baseline.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/gt_matches_ccl01.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/gt_matches_ccl05.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/gt_pairs.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/instance_crowding_results.pdf | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/instance_crowding_results.png | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/instance_crowding_results.svg | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/INTERPRETATION.md | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/matches_baseline.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/matches_baseline.npz | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/matches_ccl01.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/matches_ccl01.npz | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/matches_ccl05.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/matches_ccl05.npz | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/pair_recall.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/paired_gt_changes.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/paired_gt_changes_summary.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/protocol_tests.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/recall_by_density.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/RESULTS_REPORT.md | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/run.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/standardization_support.csv | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/summary.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | gemini/results/coco_instance_eval/20260911_034617/validation.json | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | refine-logs/coco-evaluation/EXPERIMENT_TRACKER_20260911_035217.md | results | COCO-Dense GT instance evaluation: source, data, verification, or report |
| 2026-09-11 03:57:24 | experiment-plan execution | refine-logs/coco-evaluation/EXPERIMENT_TRACKER.md | latest | Updated after Dense1576 evaluation |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/analysis_bootstrap_draws.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/analysis.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/execution.log | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/gt_selection.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/intervention_direction_contrasts.csv | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/INTERVENTION_REPORT.md | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/intervention_summary.csv | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/interventions.csv | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_102411.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_10707.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_11197.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_114907.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_118594.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_119452.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_124975.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_12670.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_127270.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_129054.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_130386.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_132544.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_133244.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_140203.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_145597.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_156372.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_157767.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_157928.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_163117.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_166166.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_166259.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_167486.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_168330.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_171190.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_17182.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_174018.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_174371.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_181421.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_182155.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_182611.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_183437.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_18491.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_185890.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_19109.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_194940.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_201426.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_205324.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_208363.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_210394.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_211674.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_215644.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_224724.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_226662.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_227399.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_231508.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_231747.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_231879.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_233033.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_233238.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_236426.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_239347.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_239717.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_244592.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_252294.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_257624.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_262227.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_262487.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_266768.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_268375.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_269113.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_272364.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_273711.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_277005.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_279278.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_280891.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_280918.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_284445.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_286182.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_28993.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_290163.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_292997.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_295809.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_296231.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_296969.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_297085.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_297427.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_301376.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_301867.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_303818.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_304404.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_306582.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_30785.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_309391.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_31322.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_320632.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_322429.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_322895.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_324258.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_328117.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_330396.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_333772.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_334555.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_338905.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_34205.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_34417.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_345466.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_351530.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_351810.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_363207.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_365207.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_371699.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_372260.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_374083.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_377946.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_378284.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_380913.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_384808.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_388846.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_389566.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_391144.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_3934.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_394206.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_400367.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_40083.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_410650.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_411817.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_414795.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_415741.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_416758.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_419601.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_419882.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_42178.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_422706.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_426203.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_426329.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_427256.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_431568.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_432553.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_432898.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_433103.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_43314.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_434459.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_438862.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_439715.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_447088.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_447187.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_447313.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_448256.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_450202.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_450303.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_450399.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_455267.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_458755.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_459437.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_461009.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_468632.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_470773.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_471023.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_474293.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_475678.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_481390.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_481582.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_492758.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_493799.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_497599.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_500257.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_502336.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_504635.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_50679.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_508917.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_51309.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_520077.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_520324.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_520707.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_52413.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_525600.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_526392.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_529966.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_537270.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_537991.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_539962.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_542127.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_546219.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_55022.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_551820.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_555972.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_559842.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_561958.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_563470.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_564336.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_567640.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_570539.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_57150.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_571857.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_573094.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_579655.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_59044.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_60932.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_62554.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_63047.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_67213.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_76625.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_776.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_78823.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_81988.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_82688.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_85823.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_8844.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_94852.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_baseline_97337.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_102411.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_10707.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_11197.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_114907.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_118594.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_119452.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_124975.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_12670.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_127270.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_129054.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_130386.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_132544.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_133244.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_140203.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_145597.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_156372.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_157767.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_157928.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_163117.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_166166.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_166259.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_167486.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_168330.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_171190.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_17182.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_174018.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_174371.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_181421.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_182155.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_182611.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_183437.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_18491.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_185890.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_19109.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_194940.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_201426.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_205324.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_208363.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_210394.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_211674.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_215644.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_224724.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_226662.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_227399.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_231508.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_231747.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_231879.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_233033.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_233238.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_236426.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_239347.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_239717.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_244592.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_252294.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_257624.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_262227.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_262487.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_266768.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_268375.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_269113.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_272364.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_273711.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_277005.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_279278.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_280891.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_280918.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_284445.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_286182.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_28993.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_290163.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_292997.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_295809.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_296231.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_296969.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_297085.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_297427.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_301376.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_301867.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_303818.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_304404.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_306582.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_30785.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_309391.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_31322.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_320632.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_322429.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_322895.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_324258.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_328117.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_330396.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_333772.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_334555.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_338905.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_34205.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_34417.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_345466.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_351530.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_351810.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_363207.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_365207.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_371699.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_372260.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_374083.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_377946.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_378284.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_380913.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_384808.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_388846.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_389566.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_391144.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_3934.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_394206.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_400367.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_40083.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_410650.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_411817.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_414795.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_415741.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_416758.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_419601.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_419882.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_42178.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_422706.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_426203.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_426329.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_427256.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_431568.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_432553.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_432898.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_433103.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_43314.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_434459.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_438862.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_439715.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_447088.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_447187.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_447313.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_448256.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_450202.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_450303.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_450399.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_455267.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_458755.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_459437.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_461009.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_468632.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_470773.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_471023.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_474293.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_475678.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_481390.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_481582.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_492758.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_493799.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_497599.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_500257.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_502336.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_504635.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_50679.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_508917.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_51309.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_520077.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_520324.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_520707.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_52413.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_525600.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_526392.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_529966.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_537270.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_537991.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_539962.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_542127.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_546219.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_55022.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_551820.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_555972.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_559842.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_561958.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_563470.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_564336.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_567640.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_570539.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_57150.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_571857.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_573094.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_579655.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_59044.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_60932.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_62554.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_63047.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_67213.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_76625.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_776.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_78823.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_81988.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_82688.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_85823.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_8844.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_94852.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/pair_ccl01_97337.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/run.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/statuses.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/validation.json | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_102411.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_10707.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_11197.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_114907.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_118594.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_119452.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_124975.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_12670.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_127270.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_129054.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_130386.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_132544.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_133244.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_140203.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_156372.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_157767.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_157928.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_163117.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_166166.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_166259.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_167486.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_168330.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_171190.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_17182.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_174018.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_174371.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_181421.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_182155.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_183437.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_18491.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_19109.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_194940.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_201426.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_205324.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_210394.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_211674.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_224724.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_227399.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_231508.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_231747.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_231879.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_233033.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_233238.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_236426.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_239717.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_244592.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_257624.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_262227.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_262487.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_266768.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_272364.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_273711.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_277005.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_280891.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_280918.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_284445.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_286182.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_28993.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_290163.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_292997.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_296231.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_296969.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_297427.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_301376.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_301867.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_303818.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_304404.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_30785.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_309391.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_31322.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_322429.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_322895.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_324258.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_328117.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_330396.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_333772.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_334555.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_338905.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_34205.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_34417.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_345466.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_351810.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_363207.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_365207.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_371699.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_372260.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_374083.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_377946.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_378284.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_380913.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_384808.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_388846.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_389566.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_391144.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_3934.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_394206.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_40083.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_410650.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_411817.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_414795.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_416758.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_419601.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_419882.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_422706.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_426329.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_427256.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_431568.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_434459.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_438862.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_439715.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_447088.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_447187.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_447313.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_448256.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_450202.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_450303.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_450399.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_455267.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_458755.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_459437.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_461009.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_468632.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_470773.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_471023.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_475678.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_481390.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_481582.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_492758.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_493799.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_497599.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_500257.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_502336.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_504635.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_50679.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_508917.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_51309.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_520077.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_520324.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_520707.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_52413.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_525600.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_526392.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_529966.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_537270.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_537991.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_539962.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_55022.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_551820.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_555972.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_559842.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_563470.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_564336.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_567640.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_570539.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_57150.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_571857.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_573094.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_579655.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_59044.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_60932.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_62554.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_67213.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_76625.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_776.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_78823.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_81988.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_82688.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_85823.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_8844.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/baseline_94852.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_102411.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_10707.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_11197.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_114907.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_118594.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_119452.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_124975.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_12670.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_127270.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_129054.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_130386.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_132544.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_133244.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_140203.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_156372.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_157767.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_157928.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_163117.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_166166.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_166259.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_167486.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_168330.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_171190.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_17182.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_174018.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_181421.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_182155.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_183437.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_18491.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_19109.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_194940.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_201426.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_205324.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_210394.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_224724.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_227399.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_231508.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_231747.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_231879.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_233033.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_233238.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_236426.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_239717.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_244592.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_257624.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_262227.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_262487.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_266768.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_272364.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_273711.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_277005.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_280891.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_280918.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_284445.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_286182.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_28993.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_290163.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_292997.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_296231.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_296969.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_297427.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_301376.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_301867.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_303818.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_304404.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_30785.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_309391.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_31322.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_322429.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_322895.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_324258.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_328117.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_330396.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_333772.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_334555.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_338905.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_34205.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_34417.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_345466.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_351810.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_363207.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_365207.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_371699.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_374083.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_377946.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_378284.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_380913.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_384808.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_388846.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_389566.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_391144.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_3934.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_394206.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_400367.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_40083.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_410650.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_411817.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_414795.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_416758.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_419601.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_422706.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_426203.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_426329.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_427256.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_431568.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_434459.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_438862.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_439715.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_447088.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_447187.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_447313.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_448256.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_450202.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_450303.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_450399.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_455267.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_458755.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_459437.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_461009.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_468632.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_470773.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_475678.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_481390.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_481582.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_492758.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_493799.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_497599.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_500257.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_502336.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_504635.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_50679.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_508917.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_51309.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_520077.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_520324.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_520707.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_52413.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_526392.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_529966.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_537270.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_539962.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_542127.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_546219.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_55022.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_551820.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_555972.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_559842.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_563470.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_564336.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_567640.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_570539.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_57150.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_573094.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_579655.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_59044.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_60932.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_62554.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_67213.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_76625.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_776.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_78823.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_81988.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_82688.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_85823.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_8844.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/tensors/ccl01_94852.npz | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/frozen_decoder_tradeoff.png | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/frozen_decoder_tradeoff.pdf | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/frozen_decoder_tradeoff.svg | experiment | Frozen-decoder saved input, tensor, per-pair output, or analysis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/availability_and_sensitivity.json | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/box50_assignment_baseline.npy | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/box50_assignment_ccl01.npy | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/candidate_availability_summary.csv | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/diagnostic_matching_coverage.csv | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/execution.log | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/outcome_category_size.csv | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/per_gt_candidate_availability.csv | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/per_gt_spatial.csv | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/run.json | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_diagnosis_aggregate.pdf | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_diagnosis_aggregate.png | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_diagnosis_aggregate.svg | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_diagnosis_cases.pdf | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_diagnosis_cases.png | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_diagnosis_cases.svg | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/spatial_metric_summary.csv | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/SPATIAL_REPORT.md | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/summary.json | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_spatial_diagnosis/20260911_040704/validation.json | spatial-diagnosis | Completed COCO GT pixel and candidate diagnosis |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/coco_frozen_decoder.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/test_coco_frozen_decoder.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/analyze_frozen_decoder_run.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/test_frozen_decoder_analysis.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/validate_frozen_decoder_run.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/report_frozen_decoder_run.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/plot_frozen_decoder_run.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/coco_spatial_diagnosis.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/test_coco_spatial_diagnosis.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/coco_candidate_availability.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/report_coco_spatial_diagnosis.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/scripts/validate_coco_spatial_run.py | source | Diagnostic implementation, paired analysis, reporting or verification |
| 2026-09-11 07:42:38 | run-experiment | gemini/results/coco_frozen_decoder/20260911_070048/artifact_index.csv | index | 750 artifact hashes; raw inputs preserved |
| 2026-09-11 07:42:38 | run-experiment | refine-logs/coco-evaluation/EXPERIMENT_TRACKER.md | latest | Updated E2/E3 results, failures and next-step status |
| 2026-09-11 07:42:38 | run-experiment | refine-logs/coco-evaluation/METHOD_DECISION_20260911.md | method | Result-grounded spatial-supervision design and manuscript paragraph |
| 2026-09-11 07:53:06 | experiment-plan continuation | refine-logs/coco-evaluation/recheck_instance_protocol.py | audit | GT identity and instance support verification |
| 2026-09-11 07:53:06 | experiment-plan continuation | refine-logs/coco-evaluation/INSTANCE_PROTOCOL_RECHECK_20260911.json | audit | GT and manifest hashes match; 5206 of 5245 high ICI instances covered |
| 2026-09-11 07:53:06 | experiment-plan continuation | refine-logs/coco-evaluation/INSTANCE_EVALUATION_CLARIFICATION_20260911.md | protocol | Instance evaluation, historical polygon error and cache reuse boundaries |
| 2026-09-11 07:53:06 | experiment-plan continuation | refine-logs/coco-evaluation/EXPERIMENT_PLAN.md | latest | Correct execution status and link to instance protocol clarification |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/check_training_label_source.py | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/TRAINING_LABEL_CONNECTION_CHECK_20260911.json | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/audit_training_label_fragmentation.py | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/manual_converter_source_snapshot_20260911.py.txt | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/TRAINING_LABEL_FORENSICS_20260911.json | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/TRAINING_LABEL_FRAGMENTATION_20260911.csv | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/TRAINING_LABEL_AUDIT_20260911.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/EXPERIMENT_TRACKER.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/EXPERIMENT_PLAN.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/METHOD_DECISION_20260911.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/INSTANCE_EVALUATION_CLARIFICATION_20260911.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | refine-logs/coco-evaluation/PAPER_REVIEW.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | gemini/results/coco_instance_eval/20260911_034617/TRAINING_PROVENANCE_UPDATE.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | gemini/results/coco_spatial_diagnosis/20260911_040704/TRAINING_PROVENANCE_UPDATE.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:05:45 | training-label audit | gemini/results/coco_frozen_decoder/20260911_070048/TRAINING_PROVENANCE_UPDATE.md | audit | Archived manual polygon-splitting route, exact log fingerprints, and interpretation/retraining update |
| 2026-09-11 08:13:19 | remote read-only inspection | refine-logs/coco-evaluation/inspect_remote_training_data.py | source | Password-prompt SFTP inspector without stored credentials |
| 2026-09-11 08:13:19 | remote read-only inspection | refine-logs/coco-evaluation/REMOTE_CONNECTION_51352_20260911.json | audit | Authentication succeeded; historical dataset and cache absent from inspected server locations |
| 2026-09-11 | clean COCO setup | experiments/coco_clean_20260911/env-spec.json | environment | Pinned remote pytorch environment, official model witness and source weight receipt |
| 2026-09-11 | clean COCO setup | experiments/coco_clean_20260911/PROTOCOL.md | plan | Standard supervised Baseline/CCL protocol; official conversion and loading, formal training not yet launched |
| 2026-09-11 | clean COCO setup | experiments/coco_clean_20260911/convert_official.py | audit source | Calls official converter unchanged and audits annotation IDs, two-part fixture and exact class+bbox deduplication |
| 2026-09-11 | COCO full census | experiments/coco_clean_20260911/coco_crowding_census.py | source | Full original annotation-ID crowding census with per-instance/per-image outputs and scalar formula checks |
| 2026-09-11 | COCO full census | experiments/coco_clean_20260911/report_crowding_census.py | source | Local aggregate validation, Chinese report and standalone plots |
| 2026-09-11 | COCO full census | experiments/coco_clean_20260911/census/COCO2017_CROWDING_REPORT.md | report | All train/val instance crowding, test image inventory, exact legacy subset identity, count and ratio distributions |
| 2026-09-11 | COCO full census | experiments/coco_clean_20260911/census/COCO2017_CROWDING_CENSUS.json | result | Full source hashes, counts, crowd/invalid annotation accounting, category-aware and all-category statistics |
| 2026-09-11 | COCO full census | experiments/coco_clean_20260911/census/ARTIFACT_SHA256.json | manifest | Hashes for six per-image/instance/category CSVs, report, validation JSON and PNG/PDF figures |
| 2026-09-11 09:34 | clean COCO launch | experiments/coco_clean_20260911/PREFLIGHT_REPORT.md | audit | Official loader identity, corrected ICI boundaries, CCL oracle/gradient checks and paired smoke evidence |
| 2026-09-11 09:34 | clean COCO launch | experiments/coco_clean_20260911/ccl_loss.py | source | Original hinge-cosine CCL with reused assignment and FP32 reductions; lambda-zero exact stock equivalence |
| 2026-09-11 09:34 | clean COCO launch | experiments/coco_clean_20260911/train_config.json | protocol | Six paired runs across seeds 0/1/2, fixed 15 epochs, every-epoch checkpoint |
| 2026-09-11 09:34 | clean COCO launch | experiments/coco_clean_20260911/audits/LAUNCH_GATE.json | integrity | Frozen source/data/protocol hashes and passing prerequisite receipts |
| 2026-09-11 09:34 | clean COCO launch | experiments/coco_clean_20260911/TRAINING_STATUS.md | status | Baseline seed0 verified training on GPU; five later jobs pending, no formal efficacy result yet |
| 2026-09-11 09:42 | remote training monitor | experiments/coco_clean_20260911/monitor_training.py | source | Read-only standard-library terminal dashboard for six jobs, live progress, processes, GPU, CSV metrics and registered checkpoints |
| 2026-09-11 09:42 | remote training monitor | experiments/coco_clean_20260911/monitor.sh | launcher | Remote one-command launcher in pytorch Python, configurable refresh and JSON snapshot |
| 2026-09-11 09:42 | remote training monitor | experiments/coco_clean_20260911/audits/MONITOR_CHECK.json | audit | Actual remote metrics/progress checks, partial writes, missing files, dead process and stale log checks, refresh and safe Ctrl+C |
| 2026-09-11 11:16 | training failure diagnosis | experiments/coco_clean_20260911/diagnostics/nonfinite_20260911/DIAGNOSIS.md | diagnosis | Baseline seed0 nonfinite gradient at epoch9 batch2278; finite recovery state; bounded isolated epoch replay did not reproduce |
| 2026-09-11 12:04 | COCO mechanism localization | experiments/coco_clean_20260911/frozen_mechanism_probe.py | source | Official pretrained fixed-decoder 300-image no-training experiment; norm-preserving push/pull/random and spatial oracle |
| 2026-09-11 12:04 | COCO mechanism localization | experiments/coco_clean_20260911/frozen_capacity_check.py | source | GT-informed closed-form coefficient replacement with fixed prototypes and predicted boxes, no network optimization |
| 2026-09-11 12:04 | COCO mechanism localization | experiments/coco_clean_20260911/audit_frozen_mechanism.py | audit source | Independent ICI, official mask replay, RLE/NumPy counts and bootstrap checks; post-hoc bounded oracle direction |
| 2026-09-11 12:04 | COCO mechanism localization | experiments/coco_clean_20260911/diagnostics/mechanism_official_300_20260911/LOCAL_ARTIFACT_SHA256.json | result manifest | Local copies of full per-target results, paired draws, source hashes and audit receipts; tensors retained remotely |
| 2026-09-11 12:04 | COCO mechanism localization | refine-logs/coco-evaluation/MECHANISM_LOCALIZATION.md | report | Chinese report with all positive and negative controls, scope limits and next mechanism decision |
| 2026-09-11 12:04 | COCO mechanism localization | experiments/coco_clean_20260911/TRAINING_PAUSED_BY_USER.json | state | User requested mechanism first; formal training remains paused and historical queue FAILED preserved |

| 2026-09-11 15:45:14 | COCO mechanism continuation | refine-logs/coco-evaluation/MECHANISM_LEARNING_STATUS_20260911_154514.md | report | Full-val per-instance failure decomposition, all completed 3-seed learned pilots, failed-candidate disposition and current decision |

| 2026-09-11 15:52:57 | COCO mechanism continuation | refine-logs/coco-evaluation/MECHANISM_LEARNING_STATUS_20260911_155257.md | report | Full-val per-instance failure decomposition, all completed 3-seed learned pilots, failed-candidate disposition and current decision |

| 2026-09-11 16:24:32 | COCO mechanism continuation | refine-logs/coco-evaluation/MECHANISM_LEARNING_STATUS_20260911_162432.md | report | Full-val per-instance failure decomposition, all completed 3-seed learned pilots, failed-candidate disposition and current decision |

| 2026-09-11 | Full COCO frozen-head comparison | refine-logs/coco-evaluation/FULL_COCO_PILOT_COMPARISON_20260911.md | report | All 5,000 val images, 16 arms, 3-seed mean/SD, paired image intervals, coverage/leakage tradeoff, limits and next mechanism decision |
| 2026-09-11 | Full COCO paired statistics | experiments/coco_clean_20260911/summarize_full_local_comparison.py | source | Remote original-annotation pair reconstruction, cohort checks, 2,000 paired image bootstrap draws, source hashes; no training or inference |
| 2026-09-11 | Full COCO paired statistics | experiments/coco_clean_20260911/diagnostics/full_local_comparison_20260911/PAIRED_ANALYSIS.json | result | Task means/SD, spatial means, paired interval contrasts; fixed seeds and exploratory validation limits |

| 2026-09-11 | Threshold coverage control | refine-logs/coco-evaluation/THRESHOLD_COVERAGE_CONTROL_20260911.md | report | 300-image development-selected thresholds, 600-image transfer check, separately labeled post-hoc equal-mean-coverage curves and spatial tradeoffs |
| 2026-09-11 | Threshold coverage control | experiments/coco_clean_20260911/threshold_coverage_control.py | source | Frozen official decoder parity measured, fixed bbox attribution, dev-only scalar selection and paired image intervals |
| 2026-09-11 | Threshold coverage curve | experiments/coco_clean_20260911/threshold_coverage_curve.py | source | Explicitly descriptive evaluation-GT operating points, fixed grid, per-draw interpolation and bracket disclosure; no deployable parameter or AP claim |
| 2026-09-11 | Threshold coverage figure | refine-logs/coco-evaluation/figures/THRESHOLD_COVERAGE_20260911.svg | figure | Editable comparison of same-neighbor error, background error and attributed IoU at matched mean coverage |
| 2026-09-11 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_PLAN_20260911_191702.md | implementation | Structure-first COCO mechanism plan; raw candidates, representation readout, fixed-decoder and common-crop interventions |
| 2026-09-11 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_PLAN.md | implementation | Latest scoped structural diagnostic plan; preserves prior project plans |
| 2026-09-11 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_TRACKER_20260911_191702.md | implementation | Runtime witness DONE; full-candidate/feature experiments explicitly TODO or CONDITIONAL |
| 2026-09-11 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_TRACKER.md | implementation | Latest scoped execution tracker |
| 2026-09-11 | runtime path witness | experiments/coco_clean_20260911/audits/RUNTIME_BRANCH_WITNESS_FP32_20260911.json | review | Actual one-to-many and NMS path, six feature shapes, exact single-image FP32 cache replay |
| 2026-09-11 | runtime path witness | experiments/coco_clean_20260911/inspect_runtime_path.py | implementation | Read-only runtime structure and cache witness source |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/structure_candidate_trace.py | source | Exact raw-candidate indices and original-output replay; 300 exploratory val images |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/structure_features.py | source | Within-image GT-informed linear readouts; coordinate and shuffled-label controls |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/summarize_structure_trace.py | source | Official same-IoU box/mask matches and descriptive feature summaries |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/report_structure_diagnostic.py | source | Hash verification and 2000 image-cluster paired intervals |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/plot_structure_diagnostic.py | figure source | Standalone SVG PDF PNG using same-cohort AUC and nonexclusive candidate labels |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/diagnostics/structure_main300_20260911/COMPLETE.json | receipt | 300 images 3635 GT 1174 adjacent pairs; frozen official model |
| 2026-09-11 | COCO structural diagnostic | experiments/coco_clean_20260911/diagnostics/structure_main300_20260911/PAIRED_STRUCTURE_ANALYSIS.json | result | Paired AUC, candidate loss and same-IoU pair recovery; no method AP claim |
| 2026-09-11 | COCO structural diagnostic | refine-logs/coco-structure/STRUCTURE_RESULTS_20260911.md | report | Structural first-pass results and bounded mechanism follow-ups; formal training remains paused |
| 2026-09-11 | Frozen relative ownership | experiments/coco_clean_20260911/relative_ownership_protocol.json | protocol | Pre-evaluation protocol; train pool amendment before inference, lock and stop rule |
| 2026-09-11 | Frozen relative ownership | experiments/coco_clean_20260911/relative_ownership_decoder.py | source | Prediction-only neighbor reranking with exact input-grid foreground cardinality |
| 2026-09-11 | Frozen relative ownership | experiments/coco_clean_20260911/relative_ownership_experiment.py | source | Train calibration, locked val decoding, original GT spatial metrics and preserved scores |
| 2026-09-11 | Frozen relative ownership | experiments/coco_clean_20260911/summarize_relative_ownership.py | source | Official COCO task recovery/AP, prediction identity checks and paired statistics |
| 2026-09-11 | Frozen relative ownership | experiments/coco_clean_20260911/diagnose_relative_transfers.py | source | Post-hoc existing development-arm pixel transitions without retuning |
| 2026-09-11 | Frozen relative ownership | experiments/coco_clean_20260911/diagnostics/relative_ownership_20260911/LOCAL_VERIFICATION.json | verification | 28 hash entries checked; 12246 exact-area records; transition identities agree |
| 2026-09-11 | Frozen relative ownership | refine-logs/coco-structure/RELATIVE_OWNERSHIP_RESULTS_20260911.md | report | NO-GO for concrete intervention: neighbor error decreases, background rises, zero selected; no training |

| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/READOUT_IDENTIFIABILITY_PLAN_20260912_060756.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/READOUT_IDENTIFIABILITY_PLAN.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/EXPERIMENT_PLAN_before_readout_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/EXPERIMENT_PLAN_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_PLAN_before_readout_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_PLAN_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_PLAN.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/EXPERIMENT_TRACKER_before_readout_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_TRACKER_before_readout_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_TRACKER_20260912_061054.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 06:10:54 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_TRACKER.md | implementation | S023-S025 readout/input discrimination plan; history preserved; no experiment launched |
| 2026-09-12 | Local readout pilot S023 | refine-logs/coco-structure/READOUT_INPUT_PILOT_RESULTS_20260912.md | report | 32 fit /64 heldout train2017, three seeds, overfit/transfer contrast; no method claim |
| 2026-09-12 | Local readout scale S024 | refine-logs/coco-structure/READOUT_INPUT_SCALE_RESULTS_20260912.md | report | 1200 fit /300 heldout, three seeds, full ordinaryGT official COCO matching; current spatial layout NO_GO |
| 2026-09-12 | Local readout scale S024 | experiments/coco_clean_20260911/diagnostics/readout_input_scale1200_20260912/COMPLETE.json | receipt | Frozen official model, 225 checkpoints, task and fixed-attribution evaluation complete |
| 2026-09-12 | Error localization | refine-logs/coco-structure/READOUT_FAILURE_ROUTE_20260912.md | report | All-GT failure states, crop support, GT pixel-edit opportunities, exposure versus conditional FPR; next route untrained |
| 2026-09-12 | Error localization | experiments/coco_clean_20260911/diagnostics/readout_input_scale1200_20260912/error_localization_v2/SUMMARY.json | result | Original normal crop, rawCOCO masks, own/neighbor/background errors; fixed vs task matching separated |
| 2026-09-12 | Error exposure | experiments/coco_clean_20260911/diagnostics/readout_input_scale1200_20260912/error_localization_v2/EXPOSURE_ANALYSIS.json | result | Image-cluster exploratory intervals and exact valid-domain IoU identity; unadjusted composition |
| 2026-09-12 | Limited code review | refine-logs/coco-structure/READOUT_INPUT_RAPID_REVIEW_20260912.md | review | Same-family provisional rapid preflight; no fatal observed, limited scope, receipt-validation gap retained |
| 2026-09-12 | Composition-controlled mask errors | refine-logs/coco-structure/READOUT_COMPOSITION_CONTROL_20260912.md | report | Category, size and box-IoU common strata; neighbor burden persists, task failure contrast uncertain; old pixel MLP accounted for |
| 2026-09-12 | Composition-controlled mask errors | experiments/coco_clean_20260911/diagnostics/readout_input_scale1200_20260912/composition_control_v5/SUMMARY.json | result | 14 image-cluster comparisons, common-support sample counts and qualified scope; v1-v4 failures preserved |
| 2026-09-12 | Composition-control implementation | experiments/coco_clean_20260911/readout_composition_control.py | source | Reuses hashed annotation subset and existing predictions; explicit einsum avoids local matrix multiply exit |
| 2026-09-12 | Rich pixel readout S026 | experiments/coco_clean_20260911/run_rich_pixel_readout.py | source | Frozen realCOCO native640 pixel+global BCE+Dice comparison, three seeds, allGT task evaluation |
| 2026-09-12 | Rich pixel readout S026 | experiments/coco_clean_20260911/rich_pixel_readout.py | source | Prediction-only P(x), h and scalar-response input controls; zero initialized residual |
| 2026-09-12 | Rich pixel readout S026 | experiments/coco_clean_20260911/diagnostics/rich_pixel_readout_20260912/protocol.json | protocol | Pre-execution same512pixel/15epoch/3seed controls; frozen original crop, no inference GT, exploratory transfer |
| 2026-09-12 | Rich pixel readout S026 | refine-logs/coco-structure/RICH_PIXEL_READOUT_RESULTS_20260912.md | report | Complete NO_GO; rich P+h+neighbor fails to beat original/scalar dense recovery; global control degradation disclosed |
| 2026-09-12 | Rich pixel readout S026 | experiments/coco_clean_20260911/diagnostics/rich_pixel_readout_20260912/RESEARCH_ANALYSIS.json | result | All15fitted runs and22arms, official allGT task, spatial diagnostics, 2000pairedimage intervals, original parity PASS |
| 2026-09-12 | Rich pixel readout S026 | experiments/coco_clean_20260911/diagnostics/rich_pixel_readout_20260912/COMPLETE.json | receipt | 7811fit targets,300transfer images,2002ordinaryGT,225checkpoints,632.6seconds,original decoder actual XOR0 |
| 2026-09-12 | S027-S029 saved-readout diagnosis | refine-logs/coco-structure/READOUT_FIT_AND_CALIBRATION_RESULTS_20260912.md | report | Fit/unused-pixel behavior, finite unconstrained GT solve, prediction-only affine projection and full-task component ablation; no new shared training |
| 2026-09-12 | S027 fixed-fit diagnosis | experiments/coco_clean_20260911/diagnostics/readout_fit_transfer_20260912/ANALYSIS.json | result | 968targets,160GT-assisted coefficient solves, explicit empty-pixel counts and extreme-oracle limits |
| 2026-09-12 | S028 prediction projection | experiments/coco_clean_20260911/diagnostics/readout_affine_projection_20260912/ANALYSIS.json | result | Fit-only prediction leastsquares, three saved seeds, response-energy approximation not causal share |
| 2026-09-12 | S029 component ablation | experiments/coco_clean_20260911/diagnostics/readout_affine_ablation_20260912/ANALYSIS.json | result | All300images/2002GT, original/full exact prediction replay, no gain from removing affine component |
| 2026-09-12 | S030-S031 label and norm controls | refine-logs/coco-structure/READOUT_REGULARIZATION_AND_LABELS_20260912.md | report | Same160targets constrained oracle, independent overlap labels and rawCOCO controls; no inference method claim |
| 2026-09-12 | S030 regularization | experiments/coco_clean_20260911/diagnostics/readout_regularized_oracle_20260912/ANALYSIS.json | result | Predeclared .01/.1 regularization, norms finite, unused pixels improved, fullmask uncertainty retained |
| 2026-09-12 | S031 label semantics | experiments/coco_clean_20260911/diagnostics/readout_label_semantics_20260912/ANALYSIS.json | result | Label-only controlled fits, nativecache exact replay, four catastrophic cases reversed by independent targets |
| 2026-09-12 | S032 shared label controls | refine-logs/coco-structure/SHARED_LABEL_CONTROL_RESULTS_20260912.md | report | Complete samehead3labels3seeds, rawCOCO improves AP/highR75, gap not confirmed, neighbor errors not reduced |
| 2026-09-12 | S032 shared training | experiments/coco_clean_20260911/train_readout_label_controls.py | source | Frozen original model, same512samples, three target encodings, all135checkpoints, allGT official evaluation |
| 2026-09-12 | S032 raw results | experiments/coco_clean_20260911/diagnostics/shared_label_controls_20260912/ANALYSIS.json | result | AP53.313 vs51.902, highR7554.586 vs51.678, full300train2017 exploratory, noGT inference |
| 2026-09-12 | S033 boundary and error flow | refine-logs/coco-structure/READOUT_BOUNDARY_FLOW_RESULTS_20260912.md | report | 96.946%label disagreement near2px; rawreadout shrinks81.763%targets, own loss beyond boundary; no dense-specific support |
| 2026-09-12 | S033 raw pixel accounting | experiments/coco_clean_20260911/diagnostics/readout_boundary_flow_20260912_v2/ANALYSIS.json | result | All2002GT labels,1815matchedtargets,3savedseeds,2000image-clusterintervals and category-size common strata |
| 2026-09-12 | S033 implementation | experiments/coco_clean_20260911/readout_boundary_flow.py | source | Frozen exact7mode spatial replay, labelpartitions and own/neighbor/background pixel transitions; no training |
| 2026-09-12 | S034 predicted-area control | refine-logs/coco-structure/READOUT_EQUAL_AREA_RESULTS_20260912.md | report | Original ranking with learned K reproduces task gain; no dense sorting superiority, resize-area limits disclosed |
| 2026-09-12 | S034 implementation | experiments/coco_clean_20260911/readout_equal_area_control.py | source | Stable topK without GT, saved three seeds, all300images/2002GT, exactS032prediction replay |
| 2026-09-12 | S034 results | experiments/coco_clean_20260911/diagnostics/readout_equal_area_20260912/ANALYSIS.json | result | AP53.219/highR7555.257 for area-transfer; 73695input-area witnesses, originalresolution drift measured |
| 2026-09-12 | S035 scalar bias controls | refine-logs/coco-structure/READOUT_BIAS_CONTROL_RESULTS_20260912.md | report | Instancebias raises highR75 without reordering pixels; AP below coefficienthead, density gap not confirmed, coverage tradeoff |
| 2026-09-12 | S035 implementation | experiments/coco_clean_20260911/train_readout_bias_controls.py | source | Same rawCOCO512labels, constant+conditional three-seed heads,90checkpoints, train-only stationary scalar reference |
| 2026-09-12 | S035 raw results | experiments/coco_clean_20260911/diagnostics/readout_bias_controls_20260912/ANALYSIS.json | result | 300exploredtrain2017 officialallGT, AP52.689/highR7554.474 conditionalbias, exactoriginal/coefficient replay |
| 2026-09-12 | S036 frozen full-val protocol | refine-logs/coco-structure/FROZEN_FULL_VAL_PROTOCOL_20260912.md | protocol | Full5000val/allGT,11frozenarms, three saved seeds, no training/selection; low-middle-highR75 and pooledP90 operating curve |
| 2026-09-12 | S036 resumable evaluation | experiments/coco_clean_20260911/eval_frozen_readouts_val.py | source | Fresh local pinned forward, sourceindex/decode witnesses, atomic image shards, official COCOeval before density bins |
| 2026-09-12 | S036 paired statistics | experiments/coco_clean_20260911/summarize_frozen_readouts_val.py | source | Seedmeans first,2000pairedimagebootstrap R75/gap; AP/P90 point metrics with explicit limits |
| 2026-09-12 | S036 full-val results | refine-logs/coco-structure/FROZEN_FULL_VAL_RESULTS_20260912.md | report | 5000images/36335GT, coeffAP+1.298/highR75+1.684 but nonhigh-high gap widens1.146; scalar strong control |
| 2026-09-12 | S036 continuation | experiments/coco_clean_20260911/eval_frozen_readouts_val_v2.py | source | Zero-element NMS source index int64 fix only;4809hash-verified parent shards+191new,allGT retained |
| 2026-09-12 | S036 verified results | experiments/coco_clean_20260911/diagnostics/frozen_readouts_fullval_20260912_v2/ANALYSIS.json | result | All11official arms complete; pointwise2000pairedimageCI, no freshblind claim, no dense-specific victory |
| 2026-09-12 | S037 equal coverage | refine-logs/coco-structure/EQUAL_COVERAGE_RESULTS_20260912.md | report | Same90%ownTP sharedheads raise highneighborFP1.870points; threshold-insufficient residuals, exposure vs FPR separated |
| 2026-09-12 | S037 source | experiments/coco_clean_20260911/readout_equal_coverage.py | source | Frozen300trainimages/1815matches, rawCOCO640 GT-assisted whole-tie coverage/threshold sweeps, originalIoU7260exact |
| 2026-09-12 | S037 statistics | experiments/coco_clean_20260911/summarize_equal_coverage.py | source | Three savedseeds,2000imagebootstrap,category-size-box commonstrata, residual and allmatched denominators |
| 2026-09-12 | S037 error budget | experiments/coco_clean_20260911/equal_coverage_error_budget.py | source | ExactIoU=coverage/(1+FP/GT) accounting on diagnosticdomain; not newtheory orAP |
| 2026-09-12 | S037 results | experiments/coco_clean_20260911/diagnostics/readout_equal_coverage_20260912/ANALYSIS.json | result | 82/100matchedresidualhigh targets fail bestscalar inall3heads,80cropfeasible; notsharedtrainingcausality |
| 2026-09-12 | S038 support and gradients | refine-logs/coco-structure/SUPPORT_GRADIENT_RESULTS_20260912.md | report | Actualneighborlabels present, within-target suppression has ownBCE cost; genericpositive-negative opposition notdensecause |
| 2026-09-12 | S038 source | experiments/coco_clean_20260911/readout_support_gradients.py | source | 160fit+300transfer native support/Yreplay,128targets3heads exactregionalVJP,temporarydc localsteps |
| 2026-09-12 | S038 summaries | experiments/coco_clean_20260911/summarize_support_gradients.py | source | Three seeds firstmean,2000imageclusterCI,fitvsanalogoustransfersupport distinction |
| 2026-09-12 | S038 results | experiments/coco_clean_20260911/diagnostics/readout_support_gradients_20260912/ANALYSIS.json | result | 2998supporttargets,384gradientrows,fitneighbor131.6/512,suppressionownBCEcost,notAPorhistoricalAdam |

- S039 selective direction: refine-logs/coco-structure/SELECTIVE_DIRECTION_RESULTS_20260912.md; experiments/coco_clean_20260911/diagnostics/readout_selective_direction_20260912/ANALYSIS.json; source experiments/coco_clean_20260911/readout_selective_direction.py and summarize_selective_direction.py.

- S040 spatial control: refine-logs/coco-structure/SPATIAL_CONTROL_RESULTS_20260912.md; experiments/coco_clean_20260911/diagnostics/readout_spatial_control_20260912/ANALYSIS.json; source experiments/coco_clean_20260911/readout_spatial_control.py and summarize_spatial_control.py.

- S041: refine-logs\coco-structure\SOLVER_RESPONSE_RESULTS_20260912.md; diagnostics/solver_response_20260912/ANALYSIS.json; train_solver_response.py and summarize_solver_response.py.
- S042: refine-logs\coco-structure\TEACHER_FITTING_RESULTS_20260912.md; diagnostics/teacher_fitting_20260912/ANALYSIS.json; diagnose_teacher_fitting.py.

- S043: refine-logs\coco-structure\CONVEX_SHARED_READOUT_RESULTS_20260912.md; associated diagnostics ANALYSIS.json and source archived.

- S044: refine-logs\coco-structure\TEACHER_REPEATABILITY_RESULTS_20260912.md; associated diagnostics ANALYSIS.json and source archived.

- S045: refine-logs/coco-structure/PROTOTYPE_CONDITION_RESULTS_20260912.md; diagnostics/prototype_residual_condition_20260912 protocol, raw crossfit/permutation rows, 24 states, ANALYSIS and VERIFICATION. Completed, current global-prototype summary recipe NO_GO.


S048/S049 completed outputs (per-image raw artifacts and hashes are indexed by each COMPLETE.json):

| Timestamp | Skill | File | Stage | Description |
| --- | --- | --- | --- | --- |
| 2026-09-12 | experiment-plan | RESEARCH_FACTS.md | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_PLAN.md | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_PLAN_20260912_205700.md | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/mask_geometry_failure_census.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/summarize_mask_geometry_census.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/crowded_failure_branch_probe.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/summarize_crowded_branches.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/plot_geometry_branch_results.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/archive_geometry_branch_results.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/ANALYSIS.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/ANALYSIS_COMPLETE.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/COMPLETE.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/CROWDED_SUCCESS_POOL.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/CROWDED_SUPPORTED_MASK_FAILURE_POOL.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/DIFFERENT_CROWDED_MASK_FAILURE_POOL.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/instances.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/instances_classified.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/mask_geometry_failure_census.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/progress.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/protocol.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/summarize_mask_geometry_census.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/WITNESS.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/ANALYSIS.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/ANALYSIS_COMPLETE.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/archive_geometry_branch_results.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/COMPLETE.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/crowded_failure_branch_probe.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/GEOMETRY_BRANCH_OVERVIEW.png | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/GEOMETRY_BRANCH_OVERVIEW.svg | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/image_effects.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/manifest.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/metrics.csv | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/plot_geometry_branch_results.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/progress.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/protocol.json | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/summarize_crowded_branches.py | implementation | S048/S049 completed; scope and limitations in reports |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_failure_branch_20260912/WITNESS.json | implementation | S048/S049 completed; scope and limitations in reports |


S050 saved-output pixel flow (per-image RLE files indexed in COMPLETE.json):

| Timestamp | Skill | File | Stage | Description |
| --- | --- | --- | --- | --- |
| 2026-09-12 | experiment-plan | RESEARCH_FACTS.md | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | refine-logs/coco-structure/EXPERIMENT_TRACKER.md | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/crowded_pixel_flow_probe.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/summarize_crowded_pixel_flow.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/plot_crowded_pixel_flow.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/archive_crowded_pixel_flow.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/ANALYSIS.json | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/ANALYSIS_COMPLETE.json | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/archive_crowded_pixel_flow.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/COMPLETE.json | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/crowded_pixel_flow_probe.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/fill_consistency.csv | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/NEXT_CANDIDATE_INVENTORY.json | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/paired_image_effects.csv | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/PIXEL_FLOW_CASES.png | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/PIXEL_FLOW_CASES.svg | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/pixel_flows.csv | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/plot_crowded_pixel_flow.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/protocol.json | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/summarize_crowded_pixel_flow.py | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/UNUSED_FAILURE_SUBTYPE_CANDIDATES.csv | implementation | S050 same-output spatial analysis; no new training/inference |
| 2026-09-12 | experiment-plan | experiments/coco_clean_20260911/diagnostics/crowded_pixel_flow_20260912/WITNESS.json | implementation | S050 same-output spatial analysis; no new training/inference |


S051 completed local-neighbor subtype experiment (raw per-image outputs indexed in COMPLETE.json):

| Timestamp | Skill | File | Stage | Description |
| --- | --- | --- | --- | --- |
| 2026-09-12 | run-experiment | RESEARCH_FACTS.md | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | refine-logs/coco-structure/SUBTYPE_NEIGHBOR_PLAN.md | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | refine-logs/coco-structure/SUBTYPE_NEIGHBOR_PLAN_20260912_215811.md | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | refine-logs/EXPERIMENT_TRACKER.md | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | refine-logs/coco-structure/EXPERIMENT_TRACKER.md | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/subtype_neighbor_probe.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/summarize_subtype_neighbor.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/subtype_region_followup.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/summarize_subtype_regions.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/report_subtype_neighbor.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/archive_subtype_neighbor.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/ANALYSIS.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/ANALYSIS_COMPLETE.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/archive_subtype_neighbor.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/COMPLETE.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/crowded_failure_branch_probe.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/ILLUSTRATIVE_CASES.csv | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/image_effects.csv | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/manifest.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/metrics.csv | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/neighbor_background_probe.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/progress.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/protocol.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/report_subtype_neighbor.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/SUBTYPE_CASES.png | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/SUBTYPE_CASES.svg | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/subtype_neighbor_probe.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/subtype_region_followup.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/summarize_subtype_neighbor.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/summarize_subtype_regions.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/WITNESS.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/ANALYSIS.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/ANALYSIS_COMPLETE.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/case_responses.csv | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/COMPLETE.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/crowded_pixel_flow_probe.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/protocol.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/regions.csv | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/subtype_region_followup.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/summarize_subtype_regions.py | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
| 2026-09-12 | run-experiment | experiments/coco_clean_20260911/diagnostics/subtype_neighbor_20260912/regional_followup/WITNESS.json | implementation | S051 completed,44stratifiedinputs; original/paired contrast and limits retained |
