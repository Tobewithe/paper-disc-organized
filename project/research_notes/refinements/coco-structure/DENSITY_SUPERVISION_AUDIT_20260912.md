# S016/S017 实验完整性审计及 S018 补充

日期：2026-09-12。审计者：新上下文 GPT-5.6-Sol ultra，代理 `audit_density_supervision`。`review_independence: same-family`，`acceptance_status: provisional`。以下保存审计返回结论；非作者自行判定为PASS。

**Overall WARN。** 未发现伪GT、自归一化、虚构结果或隐藏分母；警告来自范围和一个不影响执行的未使用包装器。S018为补充代码/结果审查，未由审计者独立重做全部COCO解码。

| 检查 | 结论 | 审计证据 |
|---|---|---|
| A GT来源 | PASS | assignment_supervision_probe.py:38,68；crossimage_response_experiment.py:40；candidate_lineage_probe.py:115。COCO原始JSON、annotation ID、annToMask多片合并，非预测造GT。S018复用S017标签并annToRLE评测。 |
| B归一化 | PASS | no_candidate_readout_probe.py:46,56,165。分母为GT/union/FP；RMS仅参数条件化，不是分数自归一化。阈值与旧系数选择明确GT同图oracle；S018不按IoU选状态。 |
| C文件存在性 | PASS | assignment 304、low463、S018 490回执文件hash通过；3,635/1,021/887目标计数与CSV一致。S016交互、空间12组复算一致。远端loss/tal源码归档hash与执行时protocol一致。 |
| D执行链 | WARN（轻微） | assignment_supervision_probe.py:79实际直接调用官方assigner和calculate_segmentation_loss。:26的TracedLoss.get_assigned_targets_and_loss未被调用，无结果依赖self.assignment，不影响主调用。保留执行时脚本以维持hash。 |
| E范围 | WARN | S016配对只覆盖254+742个fit-eligible，不是全GT或密度因果；S017非历史训练/原预训练标签复现；空间7例zero-valid已逐项披露。S01897/887达到迭代上限，只是achieved solve。 |
| F类型 | PASS | S017为real_gt冻结监督路径诊断；S016/S018为real_gt+same-image GT-assisted oracle。不是独立泛化、正式AP或训练种子成绩。 |

审计初步将本地另一Ultralytics版本的源码hash与远端protocol比较，提出源码不一致警告；取得远端 `official_source/{loss,tal,ops}.py` 后核对完全一致，审计者已撤回该警告。空间表7个排除最初仅体现在n差异，现已补 `skipped_zero_valid_area=7` 和逐项ID，保留旧JSON。

审计允许的结论：在该300图缓存、当前冻结检查点和显式mask recipe下，分配/直接系数梯度及同图oracle的实际达到行为。不能由此声称密度因果、历史监督缺失、预训练标签损坏、表示能力上界或可部署AP改善。

结果报告：DENSITY_SUPERVISION_RESULTS_20260912.md。确定性核验：`experiments/coco_clean_20260911/diagnostics/DENSITY_SUPERVISION_LOCAL_VERIFICATION.json`。
