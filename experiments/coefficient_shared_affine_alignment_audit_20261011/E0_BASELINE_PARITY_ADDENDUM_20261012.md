# E0 基线一致性补充 Gate（2026-10-12）

这是 **结果完整性附加校验**，不改变 2026-10-11 的 B/C 目标、dev 选择、val CI 门槛或求解规则；不追称它在历史运行开始前已经预注册。

旧几何候选与官方 TAL raw 仅有 311/1301 精确相同，旧 Phase-B +5.4838pp 是几何集合结果，换官方候选后为 −0.1821pp（CI 跨零）。没有原始 A 逐候选一致性，不得跨报告比较或关闭方法线。

## 强制校验

1. A native、B old160、C official640 均使用同一官方 TAL 缓存。fit/dev/val 预期为 796/197/196 图、6058/1402/1346 候选。
2. 必须提供之前官方 TAL 同缓存评价的 `PER_CANDIDATE.jsonl`，包括 split/image_id/annotation_id/branch/raw_id/pyramid_level/iou_A。该逐候选文件不在当前 GitHub 镜像内，需从本地或执行机器取得；不能用 SUMMARY 汇总均值代替。
3. 只比较 **E0 A vs 历史官方 A**，不能把旧报告里的 B（另一共享映射）或 C（逐 GT oracle）与 E0 B/C 按字母比较；两次的 B/C 含义不同。
4. 强制要求精确候选 identity 集相等、逐行 A IoU 最大差 ≤1e-6、图像/候选数量一致、B/C 双图梯度 `gradient_inf<1e-6`、image-macro 点估计可从逐候选记录重算。
5. 任何失败、原始 per-candidate 缺失或源 Run 不完整，一律判 `insufficient_inputs`；不要仅凭汇总均值或 CI 进行科学裁决。

## 正式 E0 裁决唯一入口

在仓库根目录先执行：

    python -m unittest discover -s project/audits/candidate_eval_20261012 -p 'test_validate_e0_receipt.py'

再用已存在的评估结果执行：

    python project/audits/candidate_eval_20261012/adjudicate_e0_with_gate.py --eval-dir "<E0_EVAL_RUN>" --official-per-candidate "<OFFICIAL_BASELINE_PER_CANDIDATE.jsonl>"

通过后生成评估 Run 下 `E0_GATE.json` 并依原决策规则生成 `DECISION.json`；未通过只留下阻塞凭据、不启动新实验。此工具不重算 bootstrap CI，也不独立验证物理 checkpoint 一致性；这些仍依赖原协议 Run/哈希记录。

旧 `DECISION_ADDENDUM_20261011.md` 的选择和停止门槛保持不变，历史 Run 不覆盖，单独记录新审计收据。