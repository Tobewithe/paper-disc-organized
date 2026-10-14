# paper-disc-organized

这是用户的 `paper-disc` 科研工作台 GitHub 镜像，保存实验脚本、协议、报告、论文草稿和可追溯的轻量元数据；大型权重、COCO 图像、预测张量、原始逐 GT 大表主要留在本地/计算环境，**仓库没有文件不代表机器上没有对应 Run**。

## 唯一的当前执行入口

**[project/ACTIVE_PLAN.md — 当前研究总控、Gate 队列和历史计划去重](project/ACTIVE_PLAN.md)**

当前用户指定的研究线是 **“原型—系数链路”**（历史技术标识与目录名：`neighbor_sensitive_p3`）：**固定训练完成的 YOLO26m-seg，研究 Prototype→Coefficient 链路的后训练实例修复**。已有 TAL/原始 raw/COCO 口径审计用于证据校正，不是后训练主线的前置训练任务。各阶段的实际状态、Gate 裁决与唯一下一步以总控及其引用的真实 Run 为准。

- [S0：官方原生 COCO GT 失败账本协议](experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) — 官方发表 AP 复现与原生 GT 账本有效性分别判定。
- [S1–S4：冻结权重 Prototype→Coefficient 后训练计划](experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md) · [Codex 工作入口](experiments/mask_readout_mechanism_v2_20261012/LOCAL_AGENT_START_HERE.md) — 具体执行顺序与已结案配置以总控为准，协议不自动授权续跑。
- [AGENTS.md：仓库工作规则](AGENTS.md) · [跨项目科研通用规范](project/科研项目通用规范/README.md) · [研究线专属规则](paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md)。

研究线元信息由 [project/research_lines.json](project/research_lines.json) 记录；此处的当前任务**不改变其它论文线**。例如独立的 [mask_boundary_calibration](paper/manuscripts/mask_boundary_calibration/README.md) 仍有自己的原冻结结果、稿件与历史研究计划。

## 目录职责（不让历史笔记争夺执行权）

| 目录 | 权威边界 |
|---|---|
| `project/ACTIVE_PLAN.md` | **仅此处回答现在做什么、哪个 Gate 已获得真正凭据、下一步是什么** |
| `project/科研项目通用规范/` | 工作协议、评价身份、指标选择、训练资源门槛；不决定具体 Study 的实验结果 |
| `paper/manuscripts/<line>/` | 论文线事实、研究问题与历史稿件；`RESEARCH_RULES.md` 只约束所属研究线 |
| `experiments/<study>/PROTOCOL.md` / 当前指向的 protocol | 运行前明确冻结该 Study 的方法、输入、数据划分、评价、门槛、停止与交付；不同 Study 不用名字相似就混在一起 |
| `experiments/<study>/runs/<RUN_ID>/` | **唯一运行事实**：原始命令、环境、原始产物、Run 身份、校验、结果；失败/不完整也保留 |
| `project/research_notes/`、`project/hypotheses/`、`project/audits/` | 历史推导、旧目标与审计来源；出现“当前”“下一步”时先按当前总控校验时间和适用域 |

## 历史计划及基线不重写

- [旧 2026-10-12 五臂机制方案](experiments/mask_readout_mechanism_v2_20261012/PROTOCOL.md) 和 [机器门槛草案](experiments/mask_readout_mechanism_v2_20261012/GATES_SPEC.json) **保留为未冻结历史草案**，不再授权 TAL-E0→20k 五臂训练。
- [2026-10-12 候选/评价历史审计](project/audits/candidate_eval_20261012/REPORT.md) 和 [固定 8.4.100 代码/官方 COCO 基线核验](project/audits/candidate_eval_20261012/O2O_TOPK_OFFICIAL_COCO_BASELINE_20261013.md) 仍是来源事实；旧自定义 Box/Mask AP `51.980/43.518` **不是**官网 `52.5/44.1` 的原生复现。
- 历史 `raw_all` / `raw_argmax_correct` 的 5000 图统计、几何面板及旧 TAL 回放保留原数值与分母；仅当身份、分支、输入、解码全部可比，才做跨 Run 关联；不能从旧 GT 辅助 oracle 结果声称正常推理已救回。

**仓库整理本身只修改索引、当前规范和未执行计划路由，不改历史权重、源码快照、原 Run、工作台数据库或用户本地数据；更新 README 不意味着已经完成训练或评估。**


## 计划查找与本地交接

- [计划文件目录](project/PLAN_CATALOG.json)：项目/论文目录计划文件与当前两份 Study 的角色清单；历史命令不自动生效。
- [后训练执行合同](experiments/mask_readout_mechanism_v2_20261012/EXECUTION_CONTRACT.md)：S0–S4 映射、唯一目标分母、P/c0 重放、统计与正式运行前冻结项。
- [运行登记与本地协同](project/科研项目通用规范/运行登记与本地协同.md)：Run 真状态、轻量回传、文件散列与并发交接。

## 最小自检（不启动实验）

本地拉取后执行 `python project/check_active_plan.py`：只读核对当前入口、两份 Study 登记、历史计划警示与核心文件的相对链接。它**不**检查 GPU/Run 完成、不修改工作台，不把旧文件未同步到 GitHub 视为运行失败。当前科研结论仍以对应 Run 和 [总控](project/ACTIVE_PLAN.md) 为准。
