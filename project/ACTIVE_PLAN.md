# 当前科研总控：原型—系数链路

**更新：2026-10-17｜CET-v1真实系数误差迁移核验已在笔记本完成并结案。仅拟合2510条教师估计误差库和强度，在581条/208图留出完成33698次等功能幅度压力；真实中心化结构相对随机/错配的额外损失+.001973/+.002291，区间均跨零且均值低于.005，冻结判定INCONCLUSIVE_NOT_READY，不进入原型容错训练。三类扰动均使教师IoU损失约.147–.149的共同现象保留。166项回传、配置和两机独立复核通过；训练/forward/Oracle求解均0，任务已停用。唯一后续建议是先设计端点保真且功能尺度受控的一般扰动稳定性可行性对照，尚未启动。**

本页是 **“原型—系数链路”这条研究线唯一的「现在做什么」索引**，不是新 Study、科学结果或替代历史实验协议。对其它独立研究线（例如 `mask_boundary_calibration`）只做边界标记，不抢占其原冻结计划。新指令出现时先更新本页的**指向和状态**，不要在旧日报/原计划里继续追加「最新计划」。

**研究线命名（2026-10-14，用户指定）：** 展示名称统一为“原型—系数链路”（Prototype→Coefficient）。`neighbor_sensitive_p3` 保留为历史技术标识和目录名，维持既有证据引用；邻域机制、P3 干预等具体方法是否采用，由证据与所属实验协议决定，名称不预设机制结论。本次更名不改变下列阶段、冻结配置或 Gate。

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
| **S2 新固定面积策略：开发数值通过** | 原生原图掩码面积≥9216时才应用现有Δc，完整1314图/185621检测重新评价，新增训练0 | 原T655/dev2925/成功1024及三项数值门槛不变；已看过开发数据，不能作为独立确认 | [面积门控结案](../experiments/mask_readout_large_mask_gate_20261015/REPORT.md)，`RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8`；三项通过，配置/独立复算/回传通过 |
| **S2 用户实验一：旧版本关闭** | 原生c0的固定强度局部Gram秩一排斥，完整1314图/185621输出，新增训练0 | 原T655/dev2925/成功1024及三项门槛不变 | [实验一结案](../experiments/mask_readout_subspace_repulsion_20261015/REPORT.md)，`RUN_REPULSION_FULL_8b2e19a348d1476fbf05b64aabb10c7f`；目标−.086124、总体−.045933、损伤84/1024且上界9.7182%，三项失败；配置/独立复算/回传通过 |
| **S2 实验一修订：已结案** | A前景冲突区域对照；B整框正则局部响应拟合，零训练 | 原T655/dev2925/成功1024，完整1314图；先smoke再正式 | [E1-R2协议](../experiments/mask_readout_repulsion_revision_20261015/PROTOCOL.md)；`RUN_REVISION_FULL_77f26718c91c4f908cb77610a7803fc5` 两臂S2三项均失败；独立全输出/统计及1269图FP64残差核验完成，一条额外系数误差检查例外保留，见[数值复核](../experiments/mask_readout_repulsion_revision_20261015/INDEPENDENT_LAPTOP_FULL.json) |
| **S2 实验二：已结案** | 球面方向与冲突门控adapter；原2000图/3轮及完整评价 | 原S2三项Gate | [E2报告](../experiments/mask_readout_sphere_recovery_20261015/REPORT.md)：目标−.000602、总体−.000509、损伤9/1024上界1.5115%；目标与损伤未通过。工程恢复开销单列，完整回传/独立复核通过 |
| **S1机会扩展·实验三：已结案** | 同256条面板的Q-all/Q-local有限空间oracle；不训练adapter | 找到比例≥20%且相对原global32净增≥13 | [E3报告](../experiments/mask_readout_quadrant_oracle_20261015/REPORT.md)：Q-all132/256，新增28、丢失18、净增10；未过Gate。Q-local仅覆盖104条，启动前可达性检查遗漏已记录，不把全量失败当作局部表达能力否定 |
| **S2 实验四：已结案** | 解码输入尺度GT BCE对照R与R+.1排斥REP，各原2000图/3轮 | 两臂各按原S2三项；排斥增量另需目标配对区间下界>0且REP全部通过 | [E4报告](../experiments/mask_readout_aligned_repulsion_20261015/REPORT.md)：R/REP目标+.009783/+.009617均通过，总体均通过；损伤57/52，上界6.8607%/6.3249%均失败；REP−R区间跨零。配置/回传/两机统计复核完成，训练描述性均值回执差异保留 |
| **S3 已结案：增益幅度未通过** | 原soft-IoU checkpoint、9216面积门控；全部5000图/539128最终输出，新增训练0 | 冻结Mask AP≥+0.3点且配对区间下界>0、全22957成功GT损伤≤1%、Box不变 | [S3结案](../experiments/mask_gate_s3r_20261015/REPORT.md)：`RUN_S3_FULL_670fffd010664a648e096572fc7c476d`，AP 44.014175→44.155748，Δ+0.141573，CI[+0.084732,+0.204922]，损伤90/22957，恢复189，Box不变；仅幅度Gate失败。96文件回传、配置和两机独立复核通过；[旧smoke工程失败与V2恢复](../experiments/mask_gate_s3r_20261015/RECOVERY.md)保留 |
| **S1—S3机会兑现核对：已结案** | 原S1固定256条/243图、同soft-IoU checkpoint及9216门控，零训练 | 冻结L/G/R互斥分解；原生/oracle/旧输出精确重放与独立复算 | [机会核对报告](../experiments/mask_opportunity_audit_20261015/REPORT.md)：原122条oracle阳性中100/10/12；按冻结规则学习侧优先，配置/回传/两机核验通过；不是新的S2/S3 Gate |
| **训练内拟合/开发落差核验：已结案** | 同soft-IoU checkpoint、原fit2000、U/G完整输出；dev512直接复用，零训练 | 训练内修复/保护诊断要求及冻结的组成可比性；不放行S3 | [核验报告](../experiments/mask_soft_fit_audit_20261016/REPORT.md)：`RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770`，G训练三项满足，U训练损伤4.04%仍失败；门控内train/devΔ为+.054824/+.054241。共同支持仅29/82、28图且差距区间跨零；不确立小样本因果。配置、140文件回传和两机独立核验通过 |
| **AMT-v1已结案：增量失败、损伤精度不足** | 原2000图/初始化/架构/3轮1500步，仅替换接受失败实例的可达目标；同last固定9216门控 | 原S2三项加相对原G增量；教师与训练前置均通过 | [AMT结案](../experiments/mask_attainable_teacher_20261016/REPORT.md)，`RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357`；目标/总体通过，损伤7/1024上界1.263%精度不足，较原G目标增量−.004605，CI[−.009708,−.001204]；配置/166项回传/两机复核通过，不晋级S3 |
| **有限修复—保护机制：已结案，数值未决；EK-v1不晋级** | 同原S1的256条/243图完成C/E/EK及等幅对照，768次有限求解；网络训练0 | C256/256收敛，159条实际见证；E/EK有5/18解未过残差标准。EK−C −.002237，保留86.79%；只补未决项的最有利保留上界88.05%仍不达95% | [结案报告](../experiments/mask_repair_preservation_20261016/REPORT.md) · [配置复核](../experiments/mask_repair_preservation_20261016/CONFIGURATION_REVIEW.md) · [投资决定](../experiments/mask_repair_preservation_20261016/INVESTMENT_DECISION.json)；80项原Run回传、532项哈希检查及两机统计复核通过 |
| **方向与幅度：已结案，方向预测优先** | 原256条/243图、原soft-IoU方向及缓存；GT辅助有限调幅，训练0 | q=26.08%（95%上界30.27%），C见证捕获32/159=20.13%（上界26.74%），均<50%；数值有效 | [报告](../experiments/mask_adapter_ray_audit_20261016/REPORT.md) · [配置复核](../experiments/mask_adapter_ray_audit_20261016/CONFIGURATION_REVIEW.md) · [Study](../experiments/mask_adapter_ray_audit_20261016/study.json)；正式 `RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b`，R−N +.034469、R−G +.027240，均为GT辅助IoU，不是新AP；547项哈希、371文件回传及两机复核通过 |
| **CLD-v1已结案：机制/效用两轴阴性** | REAL/PERM各1600拟合图×3轮1200步；固定last完成400图完整评价；停止当前配置 | R−P/R−J单项通过，R−N失败；M−N目标、M−G增量与总体护栏失败，损伤14/1808通过；不晋级S2 | [整体报告](../experiments/mask_direction_learnability_20261016/REPORT.md) · [配置复核](../experiments/mask_direction_learnability_20261016/EVALUATION_CONFIGURATION_REVIEW.md) · [最终裁决](../experiments/mask_direction_learnability_20261016/EVALUATION_DECISION.json)。`RUN_CLD_EVALUATION_3bf05f04bb6e47c9abf9bb46cc4e1f47`；1352项回传和两机核验通过 |
| **CET-v1已结案：结构迁移证据不足，不进入训练** | 2510条/867图拟合；581条/208图留出、33698次固定强度压力；训练0 | S−R额外损失+.001973 CI[−.001149,.005096]；S−M+.002291 CI[−.000305,.005035]。两项均未满足≥.005且下界>0，上界尚未低于.005，不能判实质无效 | [整体报告](../experiments/mask_coefficient_error_transfer_20261017/REPORT.md) · [配置复核](../experiments/mask_coefficient_error_transfer_20261017/CONFIGURATION_REVIEW.md) · [研究决定](../experiments/mask_coefficient_error_transfer_20261017/RESEARCH_DECISION.json)；正式 `RUN_CET_FULL_3d2de9164a2f434eba4ff3a756690c45`；166项回传、两机全量核验通过；无后继运行 |
| **S4 不启动**：扩量/独立确认 | S3通过后才规划扩量或开发外确认 | 需S3通过并有新授权/冻结预算 | 本候选S3未通过，不晋级、不扩20k，不把重复开发COCO当独立确认 |

**已完成的原S1固定256条机会兑现核对保留。** [结案报告](../experiments/mask_opportunity_audit_20261015/REPORT.md)对应正式Run `RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd`：原122条oracle阳性分解为L=100（无门控Δc仍未修好）、G=10（已修好但门控撤回）、R=12（修好且保留）；134条原oracle未找到解完整保留。110条未兑现机会中学习侧占90.91%，按事前协议优先选择可恢复失败的条件修复学习；门控也损失10/22条已实现修复，不能忽略，但放宽门控不能解决另100条。49,976行原生/门控重放与旧输出精确一致、256条oracle重放和独立RLE IoU通过；[配置复核](../experiments/mask_opportunity_audit_20261015/CONFIGURATION_REVIEW.md)、80文件回传与两机独立核验通过。本机首次Python3.10描述性均值末位相等检查失败已保留；原冻结脚本在本机Python3.12复验全通过，未改容差或科学结论。

**此前soft-IoU训练内/开发落差核验已完成，以下作为本次目标对照的来源证据。** [报告](../experiments/mask_soft_fit_audit_20261016/REPORT.md)与[配置复核](../experiments/mask_soft_fit_audit_20261016/CONFIGURATION_REVIEW.md)对应正式Run `RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770`。训练218209输出、T2669、原成功9748；门控内有效监督3453/14268=24.2%，涉及1620张图。G训练目标+.010484、总体+.002194、损伤31/9748，诊断三项满足；开发G目标+.010024、损伤8/2355（本次512图census，不替代旧S2抽样上界）。无门控U训练损伤394/9748，说明保护问题在训练内也存在。分层差距区间跨零且共同支持不足，不认定小样本因果，也不宣称train/dev统计等效。140文件回传、两机独立复核、完整输出身份及12097配对RLE核验均通过，新增训练/基础模型forward/oracle求解均为0。

**证据综合已完成：** [论文证据与主张表](../paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_AND_CLAIMS_20261017.md)区分固定P的GT辅助机会、无GT方法实际兑现、原成功GT损伤及各主张的未证边界，另附完整实验链路和可引用正文草稿。[来源清单](../paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_AND_CLAIMS_20261017.sources.json)保存16个正式Run的字段、原摘要哈希和154项来源文件；本轮只读复核原产物并整理文档，不创建新Study/Run或重算模型评价。原CLD结案中“下一步尚未启动”保留为当时历史状态，当前完成状态以本页为准。

**结果与讨论正文已完成：** [中文正文](../paper/manuscripts/neighbor_sensitive_p3/RESULTS_AND_DISCUSSION_ZH_20261017.md)按“可修机会—实际兑现—损伤代价”组织，含同配置原生参照、全量效用、AMT/CLD双参照、诊断解释及局限；[正文核验收据](../paper/manuscripts/neighbor_sensitive_p3/RESULTS_AND_DISCUSSION_ZH_20261017.audit.json)记录来源、表格与引用核查。旧raw/argmax和P3完整稿与当前O2O正文节选分开保留，没有将历史候选总体移作当前最终输出结论。本次授权写作已完成，证据表中“正文尚未改写”是上一轮历史状态。

**当前唯一后续建议：先冻结“原生输出与可达修复端点保真下的一般功能扰动稳定性”最小可行性对照设计，尚未启动。** [CET整体报告](../experiments/mask_coefficient_error_transfer_20261017/REPORT.md)记录本次完整实验链路与[研究裁决](../experiments/mask_coefficient_error_transfer_20261017/RESEARCH_DECISION.json)。本轮正面证据是三类等幅扰动均显著损伤可达教师，结构专属额外损伤仍未获支持；因此不按真实误差协方差启动原型容错训练。后继可行性设计须保护原生与修复端点、控制功能尺度、排除整体logit放大或缩小扰动的伪收益；未验证这一条件前不进入训练。均值符号非对称只作预登记描述，不能挽救本次Gate，也不自动推出可部署偏置修正。原“统一O2O引言与方法”写作建议暂缓；已有论文正文、原S3及CLD/AMT/EK Gate均保留。当前没有任何科研运行或自动后续触发。

本次取舍依据为累计实测：旧方向有限调幅只兑现32/159见证；新CLD正确对应相对PERM/J分别+.045044/+.021467，但留出/拟合GT辅助方向相对N分别−.101720/−.056781，正常候选相对历史G目标−.009738且区间全负。因此保留“实例对应有相对作用”的有限事实，关闭“该直接方向监督能形成有益修复”的当前配置，不把它概括成输入信息绝对无用或只因样本少。正向重点收束到**已有G的低损伤实际效用与可修机会的兑现边界**；旧G在5000图的+.141573 AP与未达到+.30门槛同时保留。AMT、EK-v1及其它旧阴性约束不改写。

**上一轮修复—保护结案的两层状态：**[原始DECISION](../experiments/mask_repair_preservation_20261016/runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01/DECISION.json)保持`INCONCLUSIVE_NUMERICAL`，不因补充分析回写Gate。EK-v1不继续投入训练的依据是19条C/EK均已收敛的丢失：即使把2条未决丢失全部视为修复、保持已收敛结果不变，最高保留仍只有140/159=88.05%，见[算术边界](../experiments/mask_repair_preservation_20261016/runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01/SUPPLEMENTAL_DECISION_BOUND.json)。C的正向见证完整保留；EK实际输出相对原生+.132921、相对C−.002237同时报告，均不是新AP。停止当前保护目标不证明所有保护机制无效，方向判断也不直接把目标失败推成输入不足；下一项才区分方向与幅度。

**本次路由修正（依据c23d8a2c及项目讨论）：** AMT结案中的181维“可修失败/原成功”可分性建议尚未执行，`superseded_by=experiments/mask_repair_preservation_20261016/PROTOCOL_DRAFT.md`；旧报告仍保留当时建议。类别可分不等于能产生有益Δc，它不能优先替代对修复方向和误伤的直接干预。讨论稿中“先做Oracle—Adapter—Gate核对”的语句也已被真实完成Run超越，不再重跑。此次调整是设计判断，不将讨论升级为已证科学结论；不启动更多loss、门控或网络的并行试错。

**AMT-v1证据已结案，原协议保留。** 用户先授权冻结并训练，随后“开始”授权[冻结S2评价](../experiments/mask_attainable_teacher_20261016/EVALUATION_EXECUTION.md)。训练 `RUN_AMT_TRAIN_e10144e6d3d94803824aef1c636d831f` 于02:30完成1500步；last SHA `b39df4f2720bf60f3ffe64f79ce426d2dc55ee28188c0068e6b1aa3061cf79b7`。教师4521求解全部收敛、3091接受、门控内360/305图及训练配置审计均通过，但不能代表学生有效。S2一次抽签1024成功GT与dev512并集1311图；正式 `RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357` 于03:15完成，独立流水线03:18结束。G目标+.00541994、总体+.00022803；原G目标+.01002449，新减原−.00460455，CI[−.00970826,−.00120439]；损伤7/1024上界1.263231%为精度不足而非已证总体风险超1%。按原协议关闭AMT-v1。见[结案报告](../experiments/mask_attainable_teacher_20261016/REPORT.md)和[配置复核](../experiments/mask_attainable_teacher_20261016/S2_CONFIGURATION_REVIEW.md)。188708行zero-delta精确重放、原G固定端点复现、完整COCO匹配和两机统计均通过；166项最终文件回传SHA一致，无Python任务/锁/后续触发。无新增训练、无S3/S4、无自动提交推送。原[训练结案](../experiments/mask_attainable_teacher_20261016/TRAINING_REPORT.md)和[教师结案](../experiments/mask_attainable_teacher_20261016/TEACHER_REPORT.md)为各自阶段历史记录。

此前直接提出教师方案的未执行建议曾 `superseded_by=experiments/mask_soft_fit_audit_20261016/PROTOCOL.md`，本轮先补齐训练内证据才作上述方向选择。启动前RLE格式工程修订及首版冻结见[记录](../experiments/mask_soft_fit_audit_20261016/PRELAUNCH_ENGINEERING_REVIEW.md)，不改变科学协议。S3 +0.3点幅度Gate失败不变，不晋级S4/20k；E1/E2/E4阴性及E3净增Gate均保留。

**追加用户授权（2026-10-15）：** 用户要求随后自主完成实验2、3、4。E1修订先结案，再依次冻结并完成E2球面门控adapter pilot、E3子区域有限oracle、E4解码对齐监督与排斥loss比较；继续使用已授权笔记本。旧协议中“不得自动启动后续实验”是旧授权边界，本指令明确扩展执行范围；仍不启动S3/20k或扫描参数。每项独立Run、配置审计、结果回传和判定，阴性不自动取消用户明确要求的其它独立实验。

用户指定的四实验队列如下，现已全部完成；旧协议与Run判定保留，不自动重跑：

| 顺序 | 候选 | 当前状态 |
|---|---|---|
| 1 | 局部Gram邻居排斥 | E1-R2两臂及旧E1-R1-v1均已失败，证据保留 |
| 2 | 球面方向更新与冲突门控 | 已结案、目标与损伤Gate失败；[E2协议](../experiments/mask_readout_sphere_recovery_20261015/PROTOCOL.md)恢复1500步和完整评价均完成，独立复核通过；78步+3张反传开销单列 |
| 3 | 2×2子区域系数残差 | 已完成，Q-all净增10未过13门槛，Q-local覆盖限制；[E3协议](../experiments/mask_readout_quadrant_oracle_20261015/PROTOCOL.md)；改变全局32维系数参数化，不能混用S1身份 |
| 4 | 解码对齐BCE与跨实例排斥loss | 已结案；[E4两臂协议](../experiments/mask_readout_aligned_repulsion_20261015/PROTOCOL.md)，R/REP训练及完整评价完成，目标/总体通过、损伤均失败，排斥增量未获支持；321文件回传核验完成，自主队列已退出 |

四实验授权当时仅改变执行优先级：S3待执行路由曾登记 `superseded_by=experiments/mask_readout_subspace_repulsion_20261015/PROTOCOL.md`，没有撤销[面积门控S2开发筛查通过](../experiments/mask_readout_large_mask_gate_20261015/REPORT.md)。四实验结案后按用户“继续”恢复并完成了上方S3，现已关闭该候选的晋级。面积门控S2的目标Δ+0.010024、总体+0.000891、损伤2/1024仍保留；它不替代新的全量AP幅度裁决。附件仅引用旧soft-IoU失败，不能据此抹去已有正结果；“缩放范数就改二值边界”“无邻居就保证GT零损”“邻域必然是主因”等说法不作为已知事实。旧面板已暴露，S2与本次S3均是开发证据，不具新的独立95%覆盖保证。

路由变更理由：旧诊断53例损伤中15/19/19的阶段签名和三个固定方向AUC失败均保留；两个19并列时的事前选择顺序没有证明Lovász-hinge最优，也没有检验完整邻域网络。原报告中的具体换loss建议（未冻结/未运行）`superseded_by=experiments/mask_readout_stratified_audit_20261015/PROTOCOL.md`。本次以新探索协议补齐缺项，获得面积风险关系的正面依据；这不撤销旧简单信号判据的失败。soft-IoU旧配置三项Gate仍失败，[原结案报告](../experiments/mask_readout_soft_iou_pilot_20261014/REPORT.md)和[复核收据](../experiments/mask_readout_soft_iou_pilot_20261014/RESULT_VERIFICATION.json)保留。

本次诊断正式Run为 `RUN_BENEFIT_AUDIT_FULL_44b006cece39486aa502ca64398166d4`，1314图、185621条输出，配置与独立复算通过，新增训练步0。V1中断Run `RUN_BENEFIT_AUDIT_FULL_1ab7051a0f39457090f9ba6fdde8a336` 因显存资源瓶颈保留为工程失败；[V2恢复协议](../experiments/mask_readout_benefit_audit_20261014/PROTOCOL_V2.md)只改变逐图显存释放，全部2995条重叠配对完全一致，科学判据沿用[原诊断协议](../experiments/mask_readout_benefit_audit_20261014/PROTOCOL.md)。禁止重复运行已完成诊断或原训练/评价。

2026-10-14 路由复核先依据 [S1 报告](../experiments/mask_readout_mechanism_v2_20261012/S1_REPORT_20261013.md)及 [S2 结案报告](../experiments/mask_readout_mechanism_v2_20261012/S2_REPORT_20261013.md)纠正残留的“S1 正在运行”旧句，随后按用户授权完成上述独立零训练诊断。S0 REPORT/DECISION 中“继续取得官方发布回执”的下一步保留为当时结案记录，当前队列以本页为准。任何 Gate 已执行与否必须依据 `runs/<RUN_ID>/run.json`、实际产物和任务状态，不能仅凭 GitHub 缺文件判定本机没跑。

## 2. 文件角色和权威来源（找不到时不要自行创造新「总计划」）

| 层级 | 唯一入口 | 职责与限制 |
|---|---|---|
| 仓库执行路由 | [根 AGENTS](../AGENTS.md) | 先读本页，再按任务选择对应研究线规范与冻结协议；不在 AGENTS 堆每日事件 |
| 全项目通用约束 | [规范索引](科研项目通用规范/README.md) | 文件身份、评价语义、最少指标、冻结/停止与算力；跨研究线生效 |
| **当前方法主线** | [后训练主协议](../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md) · [Codex 工作入口](../experiments/mask_readout_mechanism_v2_20261012/LOCAL_AGENT_START_HERE.md) | 定义 S1–S4 ；新方法科研参数/阈值必须在开 Run 前冻结，以该具体协议而非本页摘要为准 |
| **当前依赖实验** | [GT 失败重基线协议](../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) | S0 的 G0–G5，含官方发表 AP parity 与原生输出 GT 账本合法性双状态 |
| 研究线边界 | [原型—系数链路研究规则](../paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md) | 版本/权重/作用对象、已有阴性证据与预算；不能把旧论文任务误激活 |
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
