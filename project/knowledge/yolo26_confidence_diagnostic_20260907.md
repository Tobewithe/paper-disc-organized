# R005c：置信度保留与固定目标增删诊断

> 当前状态更正：`experiments/yolo26_confidence_audit_20260907_v2/audit_results.json` 已完成且 deterministic_verdict=fail。下列数值保留为原计算结果，不是已通过完整性审查的结论。COCO 一致不抵消 stage/schema、映射和 7 对输出差异；旧“审计正在运行”文字仅描述此前状态。后续落实见 `mechanism_evidence_ledger_20260907.md`。原 FAIL 回执保留。

日期：2026-09-07。承接 `dense-pig-yolo26-20260905` 的 `idea-discovery` 阶段；本轮完成冻结结果的恢复、统计与审计衔接。方法路线尚未选定。

## 结果与范围

统一将置信度从 .05 降至 .01，在两份固定 trace 上合计恢复 12 个失败 GT，却使 269 个原本正确 GT 退化。Faro 的 AP 同时有所提高，说明本次 AP 变化与固定输出集合的关系正确性并不同向。GT 引导联合增删恢复 43 个所选目标及 2 个目标外邻居，但这些操作使用评测 GT，不能作为可部署方法结果。

- 冻结输入：PigLife public-test 426 图、4,474 GT；FaroPigSeg test 160 图、1,752 GT。基线失败分别为 394 与 797。
- 运行条件：既有 YOLO26 trace、Ultralytics 8.4.100、1024 / rect=false、Top-300、原始 scores、batch-2 native mask 解码。基线 score > .05，唯一放宽条件为 score > .01。
- 三个 GT 对照均独立从基线构造，共享同一个目标集合与 source 分配。目标要求基线非 C、严格 coverage/purity .75 候选在 Top-K 可用而在 .05 不可用；新增 source 还须满足 .01 < score <= .05。
- 目标池为 PigLife 32、Faro 146；可分配并选中的目标为 5、38。未按 public test 结果调整阈值、训练模型或改动预测分数。
- v1 没有完成回执；正式计算来源为 `experiments/yolo26_confidence_diagnostic_fulltrace_20260905_v2/`，不能把 v1 目录存在解释为成功运行。

## 原始结果表

AP、AP50、AP75 使用 [COCO 官方评测定义](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)，表中为 0–100 标度；模型 Top-300 与 COCO maxDets=100 是不同限制。失败率来自固定 C/I/L/S/O/M/X/MISS 分类器。恢复是基线非 C → C；退化是基线 C → 非 C。

| 数据集 | 条件 | 预测数 | AP | AP50 | AP75 | 恢复 | 退化 | 失败率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PigLife | 基线 .05 | 4,843 | 83.452206 | 98.744168 | 96.544634 | 0 | 0 | 8.8064% |
| PigLife | 统一 .01 | 5,187 | 83.452206 | 98.744168 | 96.544634 | 2 | 136 | 11.8015% |
| PigLife | GT 增加 | 4,848 | 83.452206 | 98.744168 | 96.544634 | 2 | 0 | 8.7617% |
| PigLife | GT 删除 | 4,839 | 83.450925 | 98.743097 | 96.543977 | 0 | 0 | 8.8064% |
| PigLife | GT 联合增删 | 4,844 | 83.450925 | 98.743097 | 96.543977 | 5 | 0 | 8.6947% |
| Faro | 基线 .05 | 3,484 | 39.878423 | 66.789851 | 40.416895 | 0 | 0 | 45.4909% |
| Faro | 统一 .01 | 4,979 | 40.459277 | 67.524708 | 41.196455 | 10 | 133 | 52.5114% |
| Faro | GT 增加 | 3,522 | 40.334440 | 67.220366 | 41.034231 | 16 | 0 | 44.5776% |
| Faro | GT 删除 | 3,460 | 39.720615 | 66.162155 | 40.477330 | 2 | 0 | 45.3767% |
| Faro | GT 联合增删 | 3,498 | 40.262459 | 67.028280 | 41.098902 | 40 | 0 | 43.2078% |

机器可读统计保留完整浮点数：`experiments/yolo26_confidence_statistics_20260907_v1/condition_statistics.csv`。相对 AP 变化、相对失败数变化、AP 点数差与失败率百分点差均单独命名，不能混用。

## 主要发现

1. **固定 .01 保留产生净退化。** PigLife 净增加 134 个失败，失败率 +2.9951 个百分点；Faro 净增加 123 个失败，失败率 +7.0205 个百分点。相对基线失败数分别增加 34.01%、15.43%。PigLife 的 136 个退化均为 C→O；Faro 为 C→O 130 个、C→X 3 个。该结论针对指定 checkpoint、数据与阈值，不能外推为所有低分候选均无价值。
2. **AP 与关系指标确实分离。** PigLife AP 不变；Faro AP +0.580854 点（相对 +1.4566%），同时关系退化明显。COCO 使用分数排序和插值精度，固定分类器判定实际保留集合的多实例关系。两者问题不同；本轮没有进一步把 AP 分歧归因到某一个插值点或 maxDets 截断。
3. **选择性增删可修复小切片，但没有建立评分器贡献。** GT 增加恢复 18 个 GT，GT 删除恢复 2 个，联合恢复 45 个。43 个所选目标全部恢复，另 2 个恢复发生在目标外：Faro `(image_id=92, annotation_id=758)` 与 `(109,963)`，均为 X→C。它们在删除和联合条件中均恢复，不能算作额外选中的 source。联合恢复占所有基线失败 45/1,191=3.78%；PigLife 5/394=1.27%，Faro 40/797=5.02%。该小范围对照不能用来推断更宽候选池的最优恢复率。
4. **更好的关系恢复不保证更高 AP。** Faro 联合条件比仅增加多恢复 24 个 GT，但 AP 增益由 +0.456017 点变为 +0.384036 点。PigLife 联合 AP 比基线低 0.001281 点。不能声称两个数据集均有 AP 提升，也不能把联合与单独条件的差直接称为内部因果贡献。

## 不确定性

复用项目既有 `ratio_bootstrap`，每个数据集对全部 manifest 图像进行 3,000 次配对图像聚类重采样，seed=20260907；条件之间使用相同抽样，保留分子与分母配对。

| 条件 | 数据集 | 指标 | 点估计 | 95% 区间 |
|---|---|---|---:|---:|
| 统一 .01 | PigLife | 失败率变化，百分点 | +2.995 | [+2.451, +3.554] |
| 统一 .01 | Faro | 失败率变化，百分点 | +7.021 | [+5.777, +8.294] |
| 统一 .01 | PigLife | 原正确 GT 退化率 | 3.333% | [2.733%, 3.943%] |
| 统一 .01 | Faro | 原正确 GT 退化率 | 13.927% | [11.801%, 16.167%] |
| GT 联合增删 | PigLife | 原失败 GT 恢复率 | 1.269% | [0.259%, 2.500%] |
| GT 联合增删 | Faro | 原失败 GT 恢复率 | 5.019% | [3.430%, 6.766%] |

未估计 AP 区间。图像重采样不解决同视频/猪舍图像间相关性。GT 对照未观测到 C→失败，其经验 bootstrap 的零宽区间不证明总体退化风险为零。所有结果来自单个 checkpoint，无训练多 seed 结论。

## 文献与方法 Gate

沿用已核验的 [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241) 和 [TIDE](https://dbolya.com/tide/)：前者已研究 mask 质量评分，后者支持把错误诊断与整体 AP 联系起来。本轮分别在同一基线上做保留、删除及联合对照，并同时报告 AP 与关系分类；没有把项目分类器称为 TIDE。2026-09-07 复查了这两项工作的官方页面与 COCO 源码；机制细节以已有 Wiki 原文核验记录为准。

已有近邻核验 `recent_neighbor_verification_20260905.md` 也限制通用关系选择器和 query 重组的创新表述。本轮未做新的全面查新。

candidate-scorer/ranking Gate 继续未通过：本轮没有证实内部排序因果、可部署选择规则、两数据集一致 AP 收益或既定 40%/30% 路线标准。已批准诊断主线的关键缺口转为：解释严格质量缺失与保留集合关系失败各自的可操作原因，尤其是 Faro 的单 mask 质量缺口；后续仍须比较 box 支持、mask 表达和标注语义，再与用户共同决定方法路线。

## 完整性与可恢复状态

2026-09-07 项目审查更正：下文“审计正在续接”为此前状态。新回执 `experiments/yolo26_confidence_audit_20260907_v2/audit_results.json` 已完成导出但报告 `deterministic_verdict=fail`，同时有矛盾的 accepted 字段。stage flags/source mapping 与 confidence 集合 7 missing/7 unexpected 尚未解释；COCO 重算无差异不代表整体审计通过。本报告数字保留为原计算结果，不能称为已通过独立完整性审查。详见 `project_mechanism_goal_audit_20260907.md`。

独立 `experiment-audit` 正在续接；此段将在审计回执生成后更新。统计脚本自行校验了每条件全部 6,226 GT 与 manifest 的对应、原汇总计数及输入哈希前后稳定；这些机械检查不代替独立预测完整性审查。

- 原始输出：`experiments/yolo26_confidence_diagnostic_fulltrace_20260905_v2/`。
- 新统计：`experiments/yolo26_confidence_statistics_20260907_v1/`，含 50 组区间、目标外恢复清单及输入输出 SHA256。
- 分析脚本：`tools/analyze_yolo26_confidence_results.py`。
- 独立审计预定位置：`experiments/yolo26_confidence_audit_20260905_v2/`。
- 原 run：`.aris/runs/dense-pig-yolo26-20260905.json`；维持 `idea-discovery`，不以诊断完成替代方法验收。

复算命令（输出目录必须为空或不存在）：

```powershell
& C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe tools/analyze_yolo26_confidence_results.py --run-dir experiments/yolo26_confidence_diagnostic_fulltrace_20260905_v2 --output-dir experiments/yolo26_confidence_statistics_20260907_v1
```
