# 本地 Codex 一次性执行交接：候选/评价审计收口

执行对象：`paper-disc-organized` 本地仓库及与之对应的大缓存/原始预测。**本任务只执行证据校验与必要的评价重放；不得启动新训练、重配超参或拓展研究路线。** 本交接并不表示可以不看原研究线的 RESEARCH_RULES、项目通用规范及每个 Run 的 PROTOCOL。

## 0. 同步与预检

- 本地先运行 `git status --short` 并保护未提交改动，再 `git pull --ff-only`；如果本地有阻塞或冲突，停止并报告，**不要 reset、clean、force pull**。确认新增 `AGENTS.md`、`项目通用规范/候选身份与评估口径.md` 和本目录脚本可用。
- 核对当前 E0 所依赖的官方 TAL 缓存路径、`INDEX.json` SHA、checkpoint SHA、annotations SHA、Ultralytics 实际版本。原 E0 队列可能在笔记本运行，不要重复启动或修改运行中产物。
- 先执行 `python -m unittest discover -s project/audits/candidate_eval_20261012 -p 'test_validate_e0_receipt.py'`；失败则记录测试输出，停止对 E0 的科学裁决。

## 1. 只收集两个真实来源

**官方基线：** 在旧 `coefficient_official_tal_affine_20260930` 的 `RUN_official_eval` 或实际执行机器找到完整 `PER_CANDIDATE.jsonl`，记录其绝对路径与 SHA256。它应是 3 个 split 的官方同位 A 原始 IoU，不能从仅有 `SUMMARY.json` 的 GitHub 目录逆造。若丢失，先检查已有旧评估缓存/Run 输出；只有确有必要才按旧评估代码重放 native 评价，不重新训练。

**E0 当前结果：** 在 `coefficient_shared_affine_alignment_audit_20261011` 本地 `runs/` 中定位真实最新的 `RUN_CROSS_EVAL_*` 及其两份 `SHARED_old160.pt`/`SHARED_official640.pt`，对照队列 `QUEUE_STATUS.json`、`run.json`、`SUMMARY.json`、`PER_CANDIDATE.jsonl`、`manifest.sha256`。若队列未完成，标记等待本地已有进程的产物（不是从本会话声称后台作业），不要重复发起。

## 2. E0 同位严格验收

运行（命令从仓库根）：

    python project/audits/candidate_eval_20261012/adjudicate_e0_with_gate.py --eval-dir "<E0_EVAL_RUN>" --official-per-candidate "<OFFICIAL_TAL_PER_CANDIDATE.jsonl>"

- `E0_GATE.json` 必须 `status=valid_for_E0_adjudication` 且所有 checks pass。
- 全部官方候选精确 key 相等，native A 原图 IoU 与历史 A 的逐行差 ≤1e-6，B/C 拟合梯度无穷范数 <1e-6，fit/dev/val 行与图数一致，image-macro 聚合点数一致。
- 旧官方 PER_CANDIDATE 里的 B/C 是历史共享参数/独立 GT oracle，不是 E0 的 B/C；验证器只核对 A。
- Gate 通过后沿用原有 10/11 决策协议生成 DECISION；不能基于 val 改选 B/C 或调整阈值。补充对基线/模型 hash 的物理输入检查，若缺失标注为未核实而不将其视为自动通过。

阻塞时产出新的审计 Run 说明原因、损失文件路径、需要的唯一最小修复动作；**禁止**写“共享修正不可能”或“模型优化失败”作为最终科研结论。

## 3. 高风险历史证据修订，不重新训练

- `coefficient_oracle_alignment_20261009`、`coefficient_trust_region_audit_20261009`：查候选 manifest/读取脚本，明确 1,375 面板是否来自 `native_bank` 旧几何关联；必要时在其论文引用处加适用域附注，而不是覆盖旧报告。论文需要官方 TAL 可学性证据时先写独立最小重放协议。
- `position_objective_complement_20261004`：复用已完成的 2×2 数值，别重新做同样四格；保留 G-OFF stationarity 不充分的限制。
- `neighbor_sensitive_p3` 论文 `EVIDENCE_MAP.md` 引用的 `counterfactual_p3_fullcoco_20260914/RESULTS_20260915.md` 在当前 GitHub 镜像不存在；从本地找真正的更正重评分 Run、类别 ID 和有效图像清单，核对 SHA、范围与正文声明，不可填空。
- 对历史报告只写额外审计记录/修订证据图；旧 `Run`/`REPORT` 均不覆盖。源材料缺失就留 `unknown`。

## 4. 最终回传

回传轻量文本 JSON/MD：`E0_GATE.json`、`DECISION.json`、具体核对的源路径与 SHA256、`E0_AUDIT_RUN.md`、若修订了论文证据图则提供最小 diff。保留不一致和失败日志，不传 COCO 原图、checkpoint、大 NPZ/PT 或秘钥。

**停止条件**：E0 同位 Gate 达标并作出一次预注册裁决，或明确报告唯一阻塞与需补证数据；同时完成两项高风险历史结论的适用域确认。不得滚动追加独立实验。
