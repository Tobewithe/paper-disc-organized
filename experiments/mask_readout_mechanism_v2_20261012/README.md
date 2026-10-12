# 固定候选掩码读出与空间残差：新研究 Study（草案）

本目录保存 2026-10-12 制定的**论文级机制证据检验计划**。研究线：`neighbor_sensitive_p3`。这是独立的新 Study，**不覆盖**已有 `representation_reachability_20261010` 的冻结协议、Smoke 和失败 Run。

## 从这里开始

**本地 Agent 优先阅读 [LOCAL_AGENT_START_HERE.md](LOCAL_AGENT_START_HERE.md)**，再完整阅读本目录的 [PROTOCOL.md](PROTOCOL.md) 与 [GATES_SPEC.json](GATES_SPEC.json)。开展科研操作前仍须遵守仓库根目录 `AGENTS.md`、`project/科研项目通用规范/` 和研究线 `RESEARCH_RULES.md`。

- `PROTOCOL.md`：完整研究命题、旧证据复用、E0–E5 的实验设计、资源限制、判据与停止条件。
- `GATES_SPEC.json`：随原计划封存的机器可读**规格**，不是已实现的训练器或 Gate 判定器。
- `study.json`：本研究在仓库工作台的登记记录；当前为草案，`training_started=false`。
- `LOCAL_AGENT_START_HERE.md`：提供给本地 Codex 的执行交接指令和阶段顺序。

## 状态和路线

1. **E0 前置任务**：核查已有 `coefficient_shared_affine_alignment_audit_20261011` 的真实远端/本地产物，使用其原有 `DECISION_ADDENDUM_20261011.md` 做判断。不因 GitHub 没有结果就重复启动后台队列。
2. E0 没有有效结果：收集／完成既有任务；**禁止启动新 20k**。
3. E0 有效且阳性：依旧有原共享头路线继续；本新表示 Study 暂停。
4. E0 有效但未过阳性：可按本研究计划启动**真实 E1/G0**，通过后再按冻结方案进入 20k 两 seed／五臂训练。

## 重要语义

本目录的 PROTOCOL/GATES 源文件在进入 GitHub 之前生成，因此其中 `remote_writes_performed=false` 等字段表示**生成时的历史快照**，**不表示现在仓库不存在该文件**，也不证明本地未运行旧实验。实际状态由最新且已核验的 Run/收集记录决定。文件随提交进入 GitHub 并**不等于协议已冻结或实验已授权运行**。

原样保留两个生成文件，避免破坏其中已登记的 `protocol_sha256`。正式实验启动前，本地 Agent 必须确认现有主分支版本、核查执行可行性，按协议把实验版冻结在独立 commit 中，保存 protocol SHA、源码哈希、数据清单与环境锁。冻结以后禁止静默更改指标/对照/门槛；变更另立清楚的版本记录。

## 数据边界

GitHub 仅保存可审计轻量文件（协议、脚本、Run 元信息、汇总与核验）。训练图、标注、权重、checkpoint、大型逐候选缓存保留在本地/计算环境；用哈希、文件大小、原始位置及 `transfer.json` / `manifest.sha256` 追踪。敏感账号和连接凭据不得提交。
