# 当前科研总控：固定权重 Prototype→Coefficient 后训练救援

**更新：2026-10-13｜状态：仓库规划已统一，真实执行证据仍待本地确认。**

本页是 **`neighbor_sensitive_p3` 这条研究线唯一的「现在做什么」索引**，不是新 Study、科学结果或替代历史实验协议。对其它独立研究线（例如 `mask_boundary_calibration`）只做边界标记，不抢占其原冻结计划。新指令出现时先更新本页的**指向和状态**，不要在旧日报/原计划里继续追加「最新计划」。

## 0. 唯一研究命题与路线裁决

> 固定官方 `yolo26m-seg.pt` 的预训练网络、Box/Class、原生 Prototype P、原始系数 c0 和正常最终检测身份，只对真实推理输出进行**后训练的无 GT 条件系数残差修复**：`c'=c0+Δcθ`。检验哪些原 Mask75 任务失败的 GT 在同一最终 Box75 检测上具备固定 P 的有限可恢复空间，以及新读出器是否能救回且不会损伤原成功实例。

**当前采纳：** `output-first GT 任务账本 → 同最终检测 fixed-P coefficient finite oracle → 通过 Gate 后单个轻量 adapter pilot → 全量同输出 COCO AP/损伤验收 → 方法开发外确认`。

**当前不做：** 重建当年预训练 TAL 动态、重新预训练 YOLO、重扫 30828840 raw、以 GT 选最优 raw 冒充实际推理、为挽救旧阴性结果重开教师/门控/共享仿射配置、未过初筛直接做五臂×20k。历史 TAL 是可选独立审计，不是本后训练主线的依赖。

## 1. 当前工作队列：同一时刻只激活一个 Gate

| 阶段 | 研究对象与唯一动作 | 必需输入/通过判据 | 产物/进度 |
|---|---|---|---|
| **S0 当前激活**：原生 COCO 输出任务账本 | 按固定 `v8.4.100` 读取实际 O2O 输出，确认 `end2end=True`、`agnostic_nms=False`、类别/图片/面积/decoder、官方 bbox 与 segm 分别匹配，同输出 ID 不混用 | 模型 SHA 和标注、5000 图；`native_gt_ledger_validity=passed`。官网公布 AP parity 是**另一个独立 Gate**，未过不能自称刊载基线复现，但只要原生任务账本有效，可继续在明示 `native_config_scoped` 下做科研诊断 | [冻结 GT 基线协议](../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md)。GitHub Study：`planned_not_executed`；缺少本地执行结果，不冒称 Run 完成 |
| **S1 条件下一步**：同最终检测系数可恢复性 | 在**官方 Mask75 失败 GT** 中，以 Box75 正确匹配且自身 MaskIoU<.75 的**具体输出检测 d** 为对象（集合 T）；原 Box、class、score、P、decoder、c0 固定，只对 `Δc` 做有限 GT-assisted oracle。保留输出 ID、Mask 修复及数值收敛证据 | 按[后训练重定向](../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md)的预设机会 Gate：有效 baseline 失败且 Box75 配对的不同 GT ≥200，其中有限 oracle 达 Mask75 ≥20%；仅是**已找到的可恢复性下界**，非 P 完整表达能力证书 | 当前未执行；未过则结束 c-only 输出救援配置，不直接否定所有表示方法 |
| **S2 条件训练**：只新增 delta-c adapter | GT 在 train2017 的**冻结最终输出**经正确类别/BoxIoU确定性关联产生标签；基础权重（包括 cv4/Proto/BN）全部冻结；零残差原样重放；不得在推理阶段用 GT 或将其配对称为 TAL | 单 seed、先 2000 张 fit 的独立短预算，成功保护与统一输出身份；预注册门槛：目标组 image-macro ΔMaskIoU ≥+0.5pp 且 CI 下界 >0，原成功损伤率单侧95%上界 ≤1%，总体不能明显负迁移（详见方法协议） | 未授权启动，先过 S1；不得直接使用旧五臂草案 |
| **S3 条件全量验证**：方法是否有真实效用 | 先用 S2 同一 pilot checkpoint（不扩训练）与原生基线**同一批输出记录**，以全量 COCO Mask AP、AP75/APsmall、Box AP 不变和原成功损伤做配对验证 | 方法协议门槛：Mask AP 相对同协议 baseline ≥+0.3 点并有预登记的配对不确定性/损伤护栏；不足则保留诊断、关闭本配置 | 未启动 |
| **S4 条件确认**：独立/外部可迁移性 | 在与方法选择不重叠的图片、场景或视频组独立重验；扩 20k/multi-seed 只在前置效用 Gate 通过后授权 | 未见数据与泄漏审计必须成立，既有 COCO val2017 研发集不能改称盲测 | 未授权 |

**当前唯一下一项实动作：S0 的本地数据/模型/预测路径预检与最小原生验证。** 任何 Gate 已执行与否必须依据`runs/<RUN_ID>/run.json`、实际产物和本地任务状态，**不能仅凭 GitHub 缺文件判定本机任务没跑**。S0 结束后按真 Run 更新本页当前阶段，避免两个代理同时开启 S1。

## 2. 文件角色和权威来源（找不到时不要自行创造新「总计划」）

| 层级 | 唯一入口 | 职责与限制 |
|---|---|---|
| 仓库执行路由 | [根 AGENTS](../AGENTS.md) | 先读本页，再按任务选择对应研究线规范与冻结协议；不在 AGENTS 堆每日事件 |
| 全项目通用约束 | [规范索引](科研项目通用规范/README.md) | 文件身份、评价语义、最少指标、冻结/停止与算力；跨研究线生效 |
| **当前方法主线** | [后训练主协议](../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md) · [Codex 工作入口](../experiments/mask_readout_mechanism_v2_20261012/LOCAL_AGENT_START_HERE.md) | 定义 S1–S4 ；新方法科研参数/阈值必须在开 Run 前冻结，以该具体协议而非本页摘要为准 |
| **当前依赖实验** | [GT 失败重基线协议](../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) | S0 的 G0–G5，含官方发表 AP parity 与原生输出 GT 账本合法性双状态 |
| 研究线边界 | [neighbor_sensitive_p3 RESEARCH_RULES](../paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md) | 版本/权重/作用对象、已有阴性证据与预算；不能把旧论文任务误激活 |
| 独立旧研究线 | [mask_boundary_calibration README](../paper/manuscripts/mask_boundary_calibration/README.md) | RCMC 及原稿/旧4500图自有协议，**不列入本次方法训练队列** |
| 官方源码/基线事实 | [8.4.100 O2O/COCO 基线核验](audits/candidate_eval_20261012/O2O_TOPK_OFFICIAL_COCO_BASELINE_20261013.md) | 权重 SHA相同、官方 Box/Mask AP 52.5/44.1、本地旧自定义 51.980/43.518、有待原生 parity；这不是研究方法 |

研究线身份元数据仍由 [research_lines.json](research_lines.json) 描述；此页记录**本轮用户明确的当前研究任务**，不强行设置该 JSON 的全项目默认研究线，也不改写其他论文的完成状态。

## 3. 历史计划去重和禁自动续跑清单

| 老入口/证据 | 当前身份 | 不再自动执行什么 |
|---|---|---|
| [2026-10-12 五臂机制草案 PROTOCOL](../experiments/mask_readout_mechanism_v2_20261012/PROTOCOL.md)、[旧 GATES_SPEC](../experiments/mask_readout_mechanism_v2_20261012/GATES_SPEC.json) | **未冻结历史草案**，完整保留来源/hash；当前执行权被 `POSTTRAIN_RESCOPE` 替代 | **E0 TAL 共享仿射审计→五臂×两 seed×20k** 的启动链 |
| [2026-10-11 共享仿射候选审计](../experiments/coefficient_shared_affine_alignment_audit_20261011/REPORT.md) 及 [原审计报告](audits/candidate_eval_20261012/REPORT.md) | **独立历史主张核验**；原 Gate 仅对那一项旧声明有效 | 为了新后训练方法强制先做历史 TAL 交叉审计 |
| [2026-10-03 “当前目标”长日志](research_notes/PROTOTYPE_COEFFICIENT_ACTIVE_GOAL_20261003.md)、[2026-09-30 冻结记录](research_notes/MAINLINE_STATUS_20260930.md) | 历史科研过程/证据索引，不作当前指挥文件 | 依据旧主机、旧 Run、旧停止状态“自动续跑” |
| [2026-09-15 ACTIVE_FAILURE_ROUTE](hypotheses/ACTIVE_FAILURE_ROUTE.md)、[旧 P3 DEVELOPMENT_PLAN](../paper/manuscripts/neighbor_sensitive_p3/DEVELOPMENT_PLAN.md) | 独立历史 P3/one-to-many/干预研究线记录 | 把一对多无槽位 Box50 问题冒充当前 O2O Mask75 修复总体 |
| [原生旧 raw 5000 报告](../experiments/coco_raw_capability_5000_20260916/REPORT.md)、[argmax 条件 raw 五类](../experiments/coco_raw_argmax_conditioned_20261005/REPORT.md) | **保留原指标与范围**，分别是 GT 辅助候选存在性和 argmax 正确条件 | 原始 oracle/argmax 数直接代表当前最终输出 GT 成败 |
| [ACD、短训可学性阴性](../experiments/coefficient_oracle_alignment_20261009/REPORT.md)、[固定 P 表达复导](../experiments/coefficient_recoverability_rederive_20261010/REPORT.md) | 旧面板和预算下有价值的正/负证据 | 重复训练同样的原生 cv4、oracle 蒸馏、信赖域等来追阳性 |

**历史冲突处理规则：** 旧文件中的“当前”“下一步”“必须先 TAL/E0”如与本页不一致，视为**该文件产生时的历史上下文**。这不是删除旧 protocol 事实，也不授权修改任何*已冻结 Run* 的 Gate。要在已冻结实验外开新命题，新建版本并先写差异，不在旧文件追加相互矛盾的「最新决策」。

## 4. 评价/身份的不可变边界

- v8.4.100 原生 O2O 是 `head.end2end=True`；该版本 `nms=False` **不是**验证分支选择器。Head 默认非 agnostic：先 K raw，再 K 个`(raw,class)`，`conf` 对组合行筛；同 raw 可跨类别重复输出、共享一个框和 Mask 系数，最终检测必须有独立 ID。
- COCOeval `bbox`/`segm` 是两次独立 GT 匹配：`bbox75_match_id`、`segm75_match_id` 可不同。主任务失败是 **segm@.75、area=all、useCats=1、maxDets=100** 的普通 GT 未匹配；需要解释掩码时固定同一条 bbox 匹配的输出，再计算其自己的 BoxIoU/MaskIoU。
- 官方公布基线 **52.5/44.1** 对比旧自定义 **51.980/43.518**；同一官方权重 SHA 已一致。发布数字 parity 与**某次原生输出的 GT 账本有效性**分开结案。原始 `predictions.json` 的字段/area policy 不得无记录改写。
- 方法端只允许改变所选检测的 Mask；任何 GT oracle、raw 全局最大、TAL 分配或几何 Hungarian 都不能作为“部署任务已救回”的成绩。正常完整 COCO AP 不能把按图、按候选、按 GT 的平均代替。

## 5. 修改计划与变更审计的最小手续

1. **新决策只改一个当前入口：** 本页用于阶段和路由；具体科学 Gate/阈值由该 Study 的待冻结/冻结 `PROTOCOL` 控制。维护`last_reviewed`与所依据的 Run/commit，严禁无原始凭据写 `completed`。
2. **在运行前冻结**：确定版本、权重/图片/GT、输出分支、数据拆分、候选身份、解码、主指标、损伤护栏、预算、停止条件。工程烟雾测试先于正式运行；失败留下独立 Run，不运行中调参、另起研究命题救场。
3. **完成后只回写事实**：`run.json` + `SUMMARY.json` + `DECISION.json`/`REPORT.md` 和确切的来源、差异、唯一后继；文稿/历史报告保留旧范围。没证据写 `unknown`/`blocked`，不从“已计划”推“已完成”。
4. **同名冲突先查版本和对象**：核实是否同 `checkpoint/image_ids/branch/input_hw/decoder/prediction_identity`；否就保留两份结果，不平均、不拼接。
5. **仅文档整理，不碰大文件和远端任务**：不改 `project/structure.json`、工作台索引数据库、旧 Run、COCO、预训练权重、原先冻结的 `PROTOCOL`；不会因更新本页自动连接机器或触发训练。

**GitHub 可核验的状态：** 当前已有两份规划文档和 Study 登记；尚无新基线 Run/输出 GT 账本/G1 oracle/后训练 checkpoint 在本次仓库规划任务中被执行、核验。实际本机/服务器是否已有新的 Run，需由本地 Codex 读取其真实状态，不能从镜像缺失断定没有。


## 6. 本次治理审计与后续轻量自检

此前一轮（变更前）的仓库镜像树检查记录：已检查根路由、当前两份 Study、三份通用规范与研究线入口：**12 份当前执行文件共 76 个内部文件链接均在 GitHub tree 有落点**。补充核对的五份旧历史文件仍有 7 个目标缺失于 GitHub（例如旧 Run 结果或未同步标准），已保留为**历史来源可用性待本地核验**，不伪造数据、也不把它们作为当前 Gate 阻塞理由。这个链接检查是**文件导航核验**，不是 336 个实验目录/全部历史 Run 的科学复核。

本地维护命令：`python project/check_active_plan.py`（只读、Python 标准库）；检查核心链接、历史导航提示和两个当前 Study 的实际指向。它不是模型评估器，也不能替代手动核实本机正在运行的任务。新增科学结果/数值门槛仍必须通过真实 Run 另行裁决。

## 7. 2026-10-13 执行歧义收敛

- 阶段、目标分母、特征缓存桥接和统计冻结项统一到 [EXECUTION_CONTRACT](../experiments/mask_readout_mechanism_v2_20261012/EXECUTION_CONTRACT.md)，它属于当前未冻结方法协议的组成部分，不是第二份总控。
- 队列固定为 S0 → S1 → S2 的 2k pilot → S3 同 checkpoint 全量 AP → S4 有条件扩量/独立确认。修正原文中“G2 后直接20k”的歧义。
- S1/S2 目标 T 同时要求 baseline 任务失败、Box75 配对、该条输出自身 MaskIoU<.75；原成功损伤按所有基线任务成功 GT 统计，不能缩成局部成功检测子集。
- S0 JSON 到 S1 P/c0 的桥接必须验证；未缓存时只预登记必要图片的特征重放，不能用 COCO dt_id 猜 raw_id。
- [运行登记与本地协同](科研项目通用规范/运行登记与本地协同.md) 固定真 Run、冻结快照、回传清单和并发交接；[计划目录](PLAN_CATALOG.json) 仅管文件角色，不复制实时阶段状态。
- oracle 数值预算、adapter 候选结构/配置和损伤 CI 设计已有明确实现/默认；正式开 Run 前仍需冻结真实输入、模型层/通道、资源和其余重采样参数，并完成原生 smoke。剩余执行冻结项不以“计划写完”冒称可无条件启动训练。

本次覆盖项目/论文目录树的计划索引与当前两份 Study 的协议一致性；336 个实验目录仅枚举，**未逐一复算历史实验或审查其全部子文件**。旧 Run、源码快照、其它论文线及工作台自动索引未修改。

## 8. S0/S1 执行准备进度（2026-10-13）

[工具说明](../shared/evaluation/POSTTRAIN_TOOLS.md) 已提供离线 COCO 账本、目标冻结/机会统计/输出不变量检查、有限系数求解、固定版本 P/c0/F 原生导出和 decoder 重放组件及合成测试。[方法假说与预设配置](../experiments/mask_readout_mechanism_v2_20261012/MECHANISM_AND_DEFAULTS.md) 将主假说收敛到预测框内 P/z0 的显式空间条件读出，对照改为 P/z0 联合空间置换，保留数值、通道与同结构预算；GT/基线保持 loss 分别归一化。[损伤验收](../shared/evaluation/DAMAGE_ACCEPTANCE.md) 固定有限成功 GT 总体、checkpoint 冻结后无放回抽样及精确单侧95%上界，S3 全量报告精确比例。

**当前科研阶段仍为 S0**：本次只执行标准库工程测试，未运行真实 COCO、torch solver、原生模型特征导出或训练。旧 solver 的 cv2 GT 栅格化/框内交并集不用于新 S1；真实 native decoder 桥接是进入 S1 的工程依赖。原生 GT 账本有效性与科学 Gate 不能由测试退出码或文件存在自动赋值。
