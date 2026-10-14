# 当前科研总控：固定权重 Prototype→Coefficient 后训练救援

**更新：2026-10-14 20:44核验｜状态：S1机会Gate通过；旧BCE版与本次soft-IoU版S2配置均关闭。soft-IoU评价 `RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0` 已完成，原S2三项Gate均失败；训练/评价配置、回传SHA与本机独立复算全部通过。笔记本无在运行Python任务和Study锁。唯一下一步为同checkpoint、现有缓存的同ROI→原图配对诊断，尚未启动；不追加训练，不进入S3/S4。S0本地基线有效，官方发表AP复现未通过。**

本页是 **`neighbor_sensitive_p3` 这条研究线唯一的「现在做什么」索引**，不是新 Study、科学结果或替代历史实验协议。对其它独立研究线（例如 `mask_boundary_calibration`）只做边界标记，不抢占其原冻结计划。新指令出现时先更新本页的**指向和状态**，不要在旧日报/原计划里继续追加「最新计划」。

## 0. 唯一研究命题与路线裁决

> 固定官方 `yolo26m-seg.pt` 的预训练网络、Box/Class、原生 Prototype P、原始系数 c0 和正常最终检测身份，只对真实推理输出进行**后训练的无 GT 条件系数残差修复**：`c'=c0+Δcθ`。检验哪些原 Mask75 任务失败的 GT 在同一最终 Box75 检测上具备固定 P 的有限可恢复空间，以及新读出器是否能救回且不会损伤原成功实例。

**当前采纳：** `output-first GT 任务账本 → 同最终检测 fixed-P coefficient finite oracle → 通过 Gate 后单个轻量 adapter pilot → 全量同输出 COCO AP/损伤验收 → 方法开发外确认`。

**当前不做：** 重建当年预训练 TAL 动态、重新预训练 YOLO、重扫 30828840 raw、以 GT 选最优 raw 冒充实际推理、为挽救旧阴性结果重开教师/门控/共享仿射配置、未过初筛直接做五臂×20k。历史 TAL 是可选独立审计，不是本后训练主线的依赖。

## 1. 当前工作队列：同一时刻只激活一个 Gate

| 阶段 | 研究对象与唯一动作 | 必需输入/通过判据 | 产物/进度 |
|---|---|---|---|
| **S0 已结案**：采用本地配置范围的原生基线 | 按固定 `v8.4.100` 读取实际 O2O 输出，确认 `end2end=True`、`agnostic_nms=False`、类别/图片/面积/decoder、官方 bbox 与 segm 分别匹配，同输出 ID 不混用 | 模型 SHA 和标注、5000 图；`native_gt_ledger_validity=passed`。官网公布 AP parity 是**另一个独立 Gate**，未过不能自称刊载基线复现，但只要原生任务账本有效，可继续在明示 `native_config_scoped` 下做科研诊断 | [冻结 GT 基线协议](../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) · [真实执行状态](../experiments/official_gt_failure_rebaseline_20261013/EXECUTION_STATUS.md)。原生账本通过，发表 AP 复现未过；主 Run `RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170`，详见[结果报告](../experiments/official_gt_failure_rebaseline_20261013/REPORT.md) |
| **S1 已通过**：同最终检测系数可恢复性 | 在基线 Mask75 失败且同一最终 Box75 检测自身 MaskIoU<.75 的集合 T，只优化系数；P、框、类别、score、decoder和输出身份冻结 | 预注册 ≥200个GT、找到 Mask75 ≥20%；有限机会下界，不是可部署成绩 | [S1 报告](../experiments/mask_readout_mechanism_v2_20261012/S1_REPORT_20261013.md)：`RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949`，122/256，全部收敛，控制/配置/回传审计通过 |
| **S2 已结案：BCE及soft-IoU当前配置均拒绝** | soft-IoU从原B initial.pt、原2000图/逐轮顺序/3epoch/1500步，只替换GT loss；执行完整结束 | 原门槛：目标ΔIoU≥.005且95%下界>0、总体≥−.001、成功损伤上界≤1% | [soft-IoU结案报告](../experiments/mask_readout_soft_iou_pilot_20261014/REPORT.md)：评价 `RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0`，目标+.006799但CI下界−.001267、总体−.002618、损伤53/1024且上界6.4338%，三项失败；配置/回传/独立复算通过。旧[BCE结案](../experiments/mask_readout_mechanism_v2_20261012/S2_REPORT_20261013.md)保留 |
| **S3 本配置不启动**：全量效用验收 | 原计划是同一pilot checkpoint全5000图COCO AP和损伤评价 | 需先通过S2 | S2失败，未启动，不产生全量方法AP结论 |
| **S4 本配置不启动**：扩量/独立确认 | 原计划是效用Gate通过后的20k/multi-seed及独立数据确认 | 需S2/S3通过并有新授权 | 前置门槛未过，未启动 |

**当前唯一步骤（尚未启动）：同一soft-IoU checkpoint、现有缓存上的同ROI→原图固定检测配对诊断，重点定位原成功实例损伤出现于proto网格、原图解码还是COCO匹配阶段；零训练。** [本次结案报告](../experiments/mask_readout_soft_iou_pilot_20261014/REPORT.md)与[复核收据](../experiments/mask_readout_soft_iou_pilot_20261014/RESULT_VERIFICATION.json)记录三项Gate失败、相对旧BCE的目标效果改善及其开发集范围。按[启动前协议](../experiments/mask_readout_soft_iou_pilot_20261014/PROTOCOL.md)关闭本配置；不因目标均值转正而追加训练或放行S3。后台执行与轻量结果回传已完成，禁止重复启动同一训练/评价。新诊断需先冻结具体配对、指标和判定对象，再按用户安排执行。

2026-10-14 路由复核先依据 [S1 报告](../experiments/mask_readout_mechanism_v2_20261012/S1_REPORT_20261013.md)及 [S2 结案报告](../experiments/mask_readout_mechanism_v2_20261012/S2_REPORT_20261013.md)纠正残留的“S1 正在运行”旧句，随后按用户授权完成上述独立零训练诊断。S0 REPORT/DECISION 中“继续取得官方发布回执”的下一步保留为当时结案记录，当前队列以本页为准。任何 Gate 已执行与否必须依据 `runs/<RUN_ID>/run.json`、实际产物和任务状态，不能仅凭 GitHub 缺文件判定本机没跑。

## 2. 文件角色和权威来源（找不到时不要自行创造新「总计划」）

| 层级 | 唯一入口 | 职责与限制 |
|---|---|---|
| 仓库执行路由 | [根 AGENTS](../AGENTS.md) | 先读本页，再按任务选择对应研究线规范与冻结协议；不在 AGENTS 堆每日事件 |
| 全项目通用约束 | [规范索引](科研项目通用规范/README.md) | 文件身份、评价语义、最少指标、冻结/停止与算力；跨研究线生效 |
| **当前方法主线** | [后训练主协议](../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md) · [Codex 工作入口](../experiments/mask_readout_mechanism_v2_20261012/LOCAL_AGENT_START_HERE.md) | 定义 S1–S4 ；新方法科研参数/阈值必须在开 Run 前冻结，以该具体协议而非本页摘要为准 |
| **当前依赖实验** | [GT 失败重基线协议](../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) | S0 的 G0–G5，含官方发表 AP parity 与原生输出 GT 账本合法性双状态 |
| 研究线边界 | [neighbor_sensitive_p3 RESEARCH_RULES](../paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md) | 版本/权重/作用对象、已有阴性证据与预算；不能把旧论文任务误激活 |
| 独立旧研究线 | [mask_boundary_calibration README](../paper/manuscripts/mask_boundary_calibration/README.md) | RCMC 及原稿/旧4500图自有协议，**不列入本次方法训练队列** |
| 官方源码/基线事实 | [8.4.100 O2O/COCO 基线核验](audits/candidate_eval_20261012/O2O_TOPK_OFFICIAL_COCO_BASELINE_20261013.md) | 权重 SHA相同、官方 Box/Mask AP 52.5/44.1、本地旧自定义 51.980/43.518、本次原生 52.351/44.014 仍未过发表 parity；这不是研究方法 |

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

**交接时的 GitHub 规划状态是历史快照。** 本次已核实笔记本 smoke、工程、恢复、全量基线与历史核对 Run；原生 GT 账本有效，发表 AP 未复现，轻量证据已按原 Run ID 回传，具体见执行状态。GitHub 缺少大文件不代表执行端没有产物。


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

工程证据链：原24项基础工程测试、S1真实smoke/正式Run、S2原生缓存smoke/全2000图、真实adapter smoke、正式B/C训练和1299图配对评价均已完成。S2新增4项训练行为测试与2项评价测试通过；逐Run配置核验、独立checkpoint/日志审计及回传后bootstrap复算通过。此前工程失败Run保留。工程有效与科研失败分别记录，结论见上表和S2报告；不追加epoch或自动扩量挽救本配置。
