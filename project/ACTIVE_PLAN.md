# 当前科研总控与研究线路由

按用户当前指定的研究线读取相应入口，各线的实验状态、执行设备和冻结 Gate 独立。研究线身份见 [research_lines.json](research_lines.json)，原型—系数链路的既有总控在下方保留。

**2026-10-22 用户最新安排：暂不执行LCF正式统计结案及其后继实验，保留 `evaluated_pending_statistics` 状态、已有验收结论和未完成项。已有结果已推送GitHub；用户随后明确选择SPM-YOLO，当前任务见下节。** 不改变LCF任何冻结Gate、科学结论或训练预算。

## 当前用户指定主线：SPM-YOLO 猪群原型记忆

**2026-10-23 当前+5点goal与V4.1完整裁决：** 用户明确要求优化原型生成和训练、让原型真正帮助分割，至少提升5个绝对百分点。[规模Study](../experiments/spm_v4_scale_20261023/study.json)已真实完成1272train/432dev、2048几何/4096外观、基线120轮/12720步、原型20000步及B/G/R/F各8000步；Slurm199/204/205/207/208/209均完成并验收，五模型+memory/投影及17组图已本地SHA核验。F/B/G/R Mask AP57.3570/57.3153/57.2036/57.2395，F对B仅+0.0417pp；同F关闭原型仅−0.0565pp、通道错配反而+0.0861pp，小目标对B−0.5772pp。**未达+5且原型特异效用未获支持，关闭本配置全局字典扩训；[完整报告](../experiments/spm_v4_scale_20261023/REPORT.md)、[本地验收](../experiments/spm_v4_scale_20261023/FINAL_LOCAL_ACCEPTANCE.json)保留阴性与对照收益。** 原型训练loss下降不作效用证据，最强原生参照及[goal验收](../experiments/spm_v4_scale_20261023/GOAL_ACCEPTANCE_ADDENDUM.md)不降低。

**2026-10-23 实例读出诊断已完成并验收：** [Study](../experiments/spm_instance_readout_opportunity_20261023/study.json)的Slurm221/222完成432dev零训练读出及独立CPU复算，120项AP统计和2385条对象IoU误差均0，41项本地SHA通过。训练128D均值原型比随机投影有区分优势，但GT支持下305条失败仅修复2条，平均IoU 0.6613→0.5660；无GT AP25.2765，相对B57.3153下降32.0387pp。[完整报告](../experiments/spm_instance_readout_opportunity_20261023/REPORT.md)关闭单均值读出扩训。GT支持是有限读出诊断，不是所有读出器的性能上界；旧机器建议及阴性保留。

**唯一当前动作与V5路线：** 用户明确授权的对象v0已按[冻结合同](../experiments/spm_object_prototype_20261023/PROTOCOL_V0.md)完成Slurm229–232：2304图、两臂各4000步、固定评价和独立CPU验收。[报告](../experiments/spm_object_prototype_20261023/REPORT.md)、[权重与推断](../experiments/spm_object_prototype_20261023/artifacts/README.md)、[18张图](../experiments/spm_object_prototype_20261023/VISUAL_GUIDE.md)已交付，195项回传SHA通过。S完整身体误差0.01913体长，对U按族平均减少46.67%（95%区间41.67%–51.48%），不同视角完整投影IoU0.89934对U0.85639；体形码漂移0.22384>0.10，联合Gate未过，原始DECISION不改。保留几何结构先验与已训练S，关闭当前配置扩训；**唯一后继建议**按[体形不变性问题](../experiments/spm_object_prototype_20261023/NEXT_OBJECT_STATE_QUESTION.md)复用已有模型/全因子样本，零训练区分潜码漂移与实际几何漂移，再定真实域对象对齐。该后继尚未启动，本轮授权实验已完成，无自动新训练或定时任务。实际单卡分配298秒，在授权120分钟内；+5点真实分割目标未完成、最强真实dev原生57.3153保留。来源和范围见[Study](../experiments/spm_object_prototype_20261023/study.json)与[授权](../experiments/spm_object_prototype_20261023/EXECUTION_AUTHORIZATION.json)。

**2026-10-23 V4快速首轮已完成并回传：用户要求按V4快速正收益、技术可视化并显式要求goal。** [报告](../experiments/spm_v4_fastpilot_20261023/REPORT.md)、[Study](../experiments/spm_v4_fastpilot_20261023/study.json)、[图集](../experiments/spm_v4_fastpilot_20261023/VISUAL_GUIDE.md)与[验收](../experiments/spm_v4_fastpilot_20261023/ACCEPTANCE_RECEIPT.json)：Slurm191完成64几何/128外观，192完成B*600步、原型1000步、continued/G/F各400步，193完成免检索导出。固定172train/54dev，F Mask AP50.9267，比continued +1.3045pp、G +1.3725pp、B* +0.2306pp；AP75相对B* −1.1407pp。关检索AP50.9272，未支持部署检索必要性。5模型+32槽memory、13张正式展示图和预测/日志已本地SHA核验。**当前首轮交付完成；唯一后继建议：优先用保留SPM训练权重的原生head，在独立真实test核验正收益，尚未启动，不自动追加训练/多seed。** 单seed开发阳性不冒充独立test或机制成立。本次用户指令取代下段保存暂停及旧probe作为前置的动作顺序，不重判旧Gate，无定时器。

**2026-10-23 当前保存检查点：用户在推进多样化猪场场景后要求“你先保存现有成果”。** [SPM-FARM](../experiments/spm_farm_scenes_20261023/study.json)已完成资产预检和Slurm189工程渲染：`RUN_SPM_FARM_SMOKE_c065813560a6`，8个独立场景×2外观、90头几何实例/180条可见Mask，640×384原生标签；32,768条独立射线ID核验0错，RLE往返与成对标签一致。站/走/进食/腹卧/左右侧卧、4布局/4地面/2相机已生成；3次前置工程失败与R0/R1源码保留。见[保存与接续说明](../experiments/spm_farm_scenes_20261023/CHECKPOINT.md)、[报告](../experiments/spm_farm_scenes_20261023/REPORT.md)。**当前动作仅保存与回传，不再追加作业；正式128场景/256图及新版2000步原型更新均未启动。** 原P1的已训练memory/projector保留；新训练执行器尚未完成，TRAIN_PROTOCOL与输入准备脚手架不能视为已训练。恢复时唯一下一步是按真实猪体尺度完善相机近景/局部群体取景并复核姿态样例，再按原预算执行正式生成。此安排取代下段未启动的“先接入旧库smoke”顺序；不改旧Run/Gate，不把工程样例视为Mask AP收益，无定时任务或自动续训。

**2026-10-22：按用户明确目标，已训练出双源原型并交付。** Slurm184/`RUN_SPM_PROTO_TRAIN_2e46d0049349`正式完成2000步投影训练及构库（exit0，1分55秒），真实226图＋既有Blender Pig 384帧，共610图提取、492图train-only学习；P3编码器固定官方COCO预训练，未先训猪群YOLO B。已交付[原型memory](../experiments/spm_prototype_training_20261022/artifacts/pig_prototype_memory.pt)和[投影权重](../experiments/spm_prototype_training_20261022/artifacts/pig_projector.pt)：fg/bg各128个128维中心，26项回传SHA一致、单位中心及512条dev特征独立检索误差0，编码器不变。两分区均触及容量上限，不能称最优K；训练loss12.3533→.9709不是Mask AP收益。Slurm182工程失败保留，183重试通过，累计32工程步独立于2000正式步。见[报告](../experiments/spm_prototype_training_20261022/REPORT.md)、[协议](../experiments/spm_prototype_training_20261022/PROTOCOL.md)、[Study](../experiments/spm_prototype_training_20261022/study.json)。**唯一下一步：将已验证memory接入同编码器YOLO P3，先做零残差等价与真实检索响应smoke，尚未启动。** 下方SPM-S1失败不重判，其Gate不阻止本轮原型训练；“先接触普查”建议由用户新指令取代。服务器独立根`/srv/gpu-lab/users/a25023104/projects/spm-prototype-training-20261022`，无遗留作业、定时器或目标模式。

**2026-10-22服务器SPM-S1已执行，在P0覆盖Gate停止。** 用户授权完善后直接服务器实验并允许按需多卡；实际按[冻结合同](../experiments/spm_yoloseg_20261021/EXECUTION_PROTOCOL.md)完成226张原train图/38视频的标签核验，Slurm181/`RUN_SPM_P0_a5037a9c76d5`正常结束84.42秒。55图产生212个采样C token，B/I为5424/4813；17个推断family中仅4个达到各类≥16的覆盖要求，低于预冻结8组，故本配置停止，特征前向与所有训练均0，未占GPU。Slurm180输入准备工程失败保留。20项回传清单SHA和账本计数复算通过，226个cache留服务器；无probe AP或方法效用结果。**当前正向重点与唯一下一步建议：对已有原train全量标注做零训练接触来源普查，区分均匀6帧漏采稀有接触与来源本身不足；尚未启动，不降低原Gate或重选原226图。** 文件名family不是已核实物理猪栏，不能声称跨栏泛化。见[报告](../experiments/spm_yoloseg_20261021/REPORT_SPM_S1_20261022.md)、[回执](../experiments/spm_yoloseg_20261021/P0_ACCEPTANCE_RECEIPT_20261022.json)、[Study](../experiments/spm_yoloseg_20261021/study.json)。服务器独立根`/srv/gpu-lab/users/a25023104/projects/spm-yoloseg-20261022`；本轮不训练YOLO/原型、不渲染Blender，无定时器或目标模式，下方准备快照不再代表未执行。

**本次拉取更新（2026-10-22，`a7e47c44`）：** 新增[V3重点难点与科学决策树](../paper/manuscripts/spm_yoloseg/PRIORITIES_RISKS_AND_DECISION_TREE_V3.md)及[接触边界Probe草案](../experiments/spm_yoloseg_20261021/CONTACT_BOUNDARY_PROBE_DRAFT.md)。本地基础接口方案保留；当前P0输入清点同时覆盖真实同类接触C/猪-背景B/内部I账本及P3/P4提取接口，后续科研取舍按V3建议顺序另行冻结。下段V2拉取与基础设计记录保留，不视为绕过V3建议直接构库/训练的授权。

**2026-10-22：已拉取远端5项V2设计更新（HEAD `9b03dfe6`），用户要求“先把最基本的方案做出来，在讨论实验效果，先把必须做的完善了”。** 当前优先完成[基础实现方案](../paper/manuscripts/spm_yoloseg/BASE_IMPLEMENTATION_SPEC.md)与[配置草案](../experiments/spm_yoloseg_20261021/BASE_CONFIG_DRAFT.json)：可见实例RLE与共用加载器、Blender几何标注、单P3投影/动态原型、冻结查询坐标、原生训练梯度、检索融合及模型打包。真实高频迁移可独立关闭，完整效果矩阵留待基础链路接通后讨论。**唯一下一步：清点本地已登记真实RGB/可见Mask与3D资产，补齐路径、来源和哈希。** 仍为设计/`not_started`/`hypothesis_only`，未实现运行代码、未渲染或训练，未选择本线执行设备或冻结预算；不自动启动远程任务。见[主线入口](../paper/manuscripts/spm_yoloseg/README.md)与[Study](../experiments/spm_yoloseg_20261021/study.json)。其它独立研究线的既有结果与状态保持不变。

## 用户指定独立诊断：YOLOseg低中高频分析

**2026-10-20 FREQ-S2服务器尺度条件化复析已结案。** Slurm171/`RUN_FREQ_S2_FULL_4db146ff8acd405094115055754f0bca`完整复析28,377条既有频谱，6.01秒，零训练/新推理。按同类、输入面积细分层后失败减成功低频份额S/M/L为+2.44/+11.95/+33.34点；L区间[32.51,34.20]点，覆盖963/981，Hann方向保持。L失败低频方向差占86.32%，低频余弦0.7189（成功0.9798），支持优先核验大目标粗轮廓/覆盖空间定位；不证明纯相位或具体因果。**唯一下一步建议：冻结同检测、等改动预算的低频空间纠正vs低频仅调幅零训练对照，尚未启动。** 同数据探索性关联，不冒称独立FFT/模型复现或AP增益；原始24项份额差0，28,377身份与168+18项均值独立核验、18项回传及4份修订图SHA通过。见[完整报告](../experiments/yoloseg_frequency_scale_reanalysis_20261020/REPORT.md)、[冻结协议](../experiments/yoloseg_frequency_scale_reanalysis_20261020/PROTOCOL.md)、[Study](../experiments/yoloseg_frequency_scale_reanalysis_20261020/study.json)。下方FREQ-v1原结案建议由本结果细化，LCF/VOR及其它任务不变。

**2026-10-20（last_reviewed）：FREQ-v1已结案并交付可视化。** YANGLEI于17:17完成5000图原型与28,377条固定Box75配对，正式Run `RUN_FREQ_FULL_666fc3b6b2aa427d80220d4a6956a41a`，零训练/前向。P低/中/高能量59.85/27.35/12.80%；自身失败组高频误差56.24%（95%区间55.20–57.21%），但S/M/L分别77.40/47.53/17.52%，高频预测/GT能量比均不低于1，不支持统一高频不足。**本诊断的正向重点：大目标低频轮廓/覆盖空间错配；唯一后继建议是冻结同检测、同改动能量预算的低频空间纠正/高频增幅零训练对照，尚未启动。** 28,377项IoU差0、20,000项图/缓存SHA、71项回传及180项独立均值复算通过，5组PNG/SVG已查看。见[完整报告](../experiments/yoloseg_frequency_audit_20261020/REPORT.md)、[协议](../experiments/yoloseg_frequency_audit_20261020/PROTOCOL.md)、[Study](../experiments/yoloseg_frequency_audit_20261020/study.json)。频谱是描述性证据，不证明机制或方法效用；不改变下方服务器LCF及视频任务的运行、Gate或预算，无定时器/目标模式。

## 独立主线 视频遮挡表征预训练

**2026-10-21/22 VOR E1 四臂已评价并结案（验收通过）。** 本线 key 为 `occlusion_representation_pretraining`，入口为[视频遮挡表征预训练](../paper/manuscripts/occlusion_representation_pretraining/README.md)。Slurm160 生成 40,320 帧，Slurm161 四臂各 2,000 步并完成冻结评价，Run `RUN_VOR_EVALUATE_PILOT_bacf87dfcd02`（exit 0；1,708 项产物 SHA 全对，四臂 `step_2000.pt` 与 `trained_*.json` 逐臂匹配，7 个结果臂预测互不相同）。裁决 `no_practical_pair_increment` 经独立复算逐位一致：**P−B R1 仅 +0.0065pp**（CI[−0.3527,+0.2749]，门槛 ≥3pp 且下界>0）→ 物理配对无实用增量；**P−N +3.3516 与 B−N +3.3451 几乎相等** → 增益来自适配参数而非配对，支持 H_alt；**P−I −3.1578pp**（CI[−5.3727,−1.3953]）→ 历史递归摘要在本数据为负贡献，I 臂 R1 10.19%、条件实例恢复 14.03% 为四臂最高，代价是 W 41.50%（+3.60pp）。护栏 W 通过但仅剩 0.05pp 余量（+0.7909，上界 +0.9495），U 通过。**唯一下一步建议：用已存 P/I 预测做零训练、等覆盖率对齐复算，区分"历史有害"与"I 只是接受更激进"，尚未启动、需授权。** 见[验收报告](../experiments/occlusion_representation_pretraining_20261019/ACCEPTANCE_REVIEW_VOR_20261022.md)、[验收回执](../experiments/occlusion_representation_pretraining_20261019/ACCEPTANCE_RECEIPT_VOR_20261022.json)、[Study](../experiments/occlusion_representation_pretraining_20261019/study.json)。边界：单 seed、末步 checkpoint、合成留出资产、无真实视频与成熟跟踪器对照；条件定位不等于 IDF1。

- **状态：**已结案。Slurm160 单卡生成 40,320 帧并核验缓存；Slurm161 运行 B/P/I/Hpos 各 2,000 步与冻结评价，四臂真实 8,000 步。科学结论 `no_practical_pair_increment`，训练来源与评价各为独立 Run。产物缺陷一处已登记：`VALIDATION_THRESHOLDS.json` 中两个干预臂的 `f1/tp/accepted` 为 P 的整条副本（源码 `evaluate.py:256` 沿用 P 阈值），不影响 R1/W/U。
- **唯一下一步：**等覆盖率对齐的 P/I 零训练复算（见上）；不追加预算、不重选阈值、不重复提交。
- **范围：**实际首版底座为公开DINOv2 ViT-S/14；DINOv3权重访问未就绪，已在执行前登记替换。当前验证同渲染器的留出动物资产（训练 Horse/Pig/Cow/Sheep/Zebra，验证 Llama，测试 Pug/Fox），真实视频/固定YOLO跟踪迁移未测。执行账号 `a25023104@192.168.71.10`，主存目录 `video-occlusion-pretraining-v02`。旧草案、旧Run和小论文Gate保留；无定时器或自动续训。

## 原型—系数链路独立总控

**2026-10-22 LCF-S1 服务器四臂评价完成并通过执行验收，科学判定为联合门槛未通过。** Slurm178/`RUN_LCFS_ALL_EVALUATE_a126a83488fd46728faf91ea579e9371` 正常完成 5,000 图/539,125 输出/313 批，训练来源 Slurm154 四臂各 15,000 步（均 exit 0、无早停）；74 项产物 SHA、四份 last.pt 与 composite/evaluate 摘要指针、8 个科学源文件 SHA、A/C 难度匹配（99.76%/99.78%）及 Box 身份（六集 Box AP 均 52.34795017624022）核验通过，服务器无遗留作业。独立复算（2000 次/seed1729/按图配对）：segm AP 为 N 44.013836、旧G 44.152393、A 44.758755、B 44.755534、C 44.734297、**D 44.667286**；主方法 D 相对 N 的拓扑并集率**恶化 5.43%**（配对差 −0.014286，CI[−0.017951,−0.010731]），跨实例混入显著上升（N−D −0.022043），D 显著劣于 B（−0.012140，CI[−0.014652,−0.009670]），D−C 仅 3.07%（门槛 10%），D 相对 Dproj/等幅控制亦为负；四臂原成功 GT 损伤 **1.768–1.964% 全部超过 1% 门槛**（D 448/22958），旧G 90/22958。四臂 AP 增益 +0.65～+0.74 点均超旧 S3 +0.30 幅度线，但保护护栏失败，不能据此称实用方法成立。**判定 `requirements_not_jointly_met`，属方向性失败而非精度不足。** 缺口：statistics 阶段未执行，`STATISTICS.json`/`DECISION.json` 与 AP 配对 CI 未生成，`run.json` 仍为 `evaluated_pending_statistics`。**唯一下一步建议：冻结 B 检查点做零训练单变量对照，只施加旧G的 9216 原生掩码面积门控，检验损伤能否压回 ≤1% 且保留 ≥+0.5 点增益；尚未启动、需新授权，不训练不扫阈值。** 见[验收报告](../experiments/mask_local_connectivity_server_20261020/ACCEPTANCE_REVIEW_20261022.md)、[验收回执](../experiments/mask_local_connectivity_server_20261020/ACCEPTANCE_RECEIPT_20261022.json)、[Study](../experiments/mask_local_connectivity_server_20261020/study.json)。

**2026-10-20 V2两项独立零训练机制核验已授权并提交，2026-10-22 两项均已结案。** [冻结协议](../experiments/yoloseg_v2_mechanism_audits_20261020/PROTOCOL.md)、[报告](../experiments/yoloseg_v2_mechanism_audits_20261020/REPORT.md)、[Study](../experiments/yoloseg_v2_mechanism_audits_20261020/study.json)：Slurm169 输出链归因、Slurm170 采样位置稳定性，各单卡/4CPU/16GiB，零训练、不扫阈值。**实验一**（576 帧/1,728 可见GT/14,704 条原生输出精确重放）：100 条正确检测失败中 82 条首失于固定 .25 工作点、18 条 raw cow 分数≤.001，无 Top-K 截断首失；P/c 单因子替换准入未通过，不支持直接恢复分散场景 P/c。**实验二**（`RUN_V2_SHIFT_FULL_e9754eecd2314965b3731d28ba1d0c3c`，5,184 张前向，Slurm170 exit 0，4,662 项产物 SHA 全对）：±1 任一状态翻转的观测比例 **2.7922%**，三 seed 3.95/2.00/2.43% → **0/3 达 5%** 门槛；ΔIoU −0.0050；掩码坏一致性仅 0.36%、连续读出 RMS 0.0727 → 未触发阈值/裁剪放大分支。**判定：本场景与幅度下采样位置不是主瓶颈，两项统一的唯一下一步回到 .25 工作点与类别置信度形成/保持。** 主量与 per-seed/per-variant 由原始行独立复算，与 `SUMMARY.json` 逐位相同（差 0.00e+00）。旧 Slurm167/168 工程失败与旧 Run 保留，R1 仅规范 JSON 表示。不训练、不扫阈值、不设定时/目标、不提交推送。

**2026-10-20 Blender V2已结案。** Slurm164正常完成16帧smoke及12段/576帧正式测试，训练0步；正式`RUN_YB2_FULL_9154f356ba5f435f8c73db254817abb3`。三seed分散组适用性、投影及14704条原生重放通过；中心近邻/遮挡失败增量+7.6389/+9.7222个百分点，均未过预设10点门槛。近邻共同132条正确配对实质混入0→55；不据此宣称P容量不足或新方法增益。完整1303项Run清单SHA、本机冻结复算及16280项独立拓扑核算通过。见[报告](../experiments/yoloseg_blender_scale_controlled_20261020/REPORT.md)、[配置复核](../experiments/yoloseg_blender_scale_controlled_20261020/CONFIGURATION_REVIEW.md)、[Study](../experiments/yoloseg_blender_scale_controlled_20261020/study.json)。正向重点为区分实例输出断续与掩码混入的来源；唯一下一步建议冻结候选/最终输出/P/c分段归因协议，尚未启动。两项空闲卡分析建议仅讨论；不追加训练、不改LCF/VOR，不设定时/目标、不提交推送。

**2026-10-20 Blender合成视频首轮已结案。** Slurm162/`RUN_YB_SMOKE_c6f0333ccc994518bdf0790b52265526`正常完成8段/384帧，单卡15分11秒、训练0步；49152射线检查与33967条原生输出zero-delta重放通过。交叉条件正确检测189/192→168/192，共同168条观测混入9→69（补充描述）；三组控制可识别性未过80%，不作原型因果结论。59项回传SHA及账本/配置复算通过；[报告](../experiments/yoloseg_blender_stress_20261020/REPORT.md)、[配置复核](../experiments/yoloseg_blender_stress_20261020/CONFIGURATION_REVIEW.md)、[Study](../experiments/yoloseg_blender_stress_20261020/study.json)。唯一下一步建议冻结以可识别分散场景为参照、目标投影尺度受控的V2场景协议，尚未启动；12类/20736帧大矩阵仍仅设计，不改LCF及其它视频任务，无定时/目标/自动重启。

**2026-10-20 LCF服务器四卡正式训练，Slurm154（LCFS-train-ABCD）——已于 2026-10-22 完成评价，状态见本线上方验收段。** [迁移Study](../experiments/mask_local_connectivity_server_20261020/study.json)、[迁移协议](../experiments/mask_local_connectivity_server_20261020/MIGRATION_PROTOCOL.md)及[初始化工程修订](../experiments/mask_local_connectivity_server_20261020/INITIALIZATION_RECOVERY.md)已冻结；Slurm153四卡smoke通过。四臂实际initial.pt与笔记本零步张量精确一致、新建空优化器，首批日志12/13/15/14步确认。每臂20k×3轮15000步不变，笔记本失败2165步单列，累计正式尝试上限62165；服务器smoke32步另列丢弃。GPU各一张A6000，无DDP/跨卡batch扩大；预计约19小时为smoke外推。见[启动配置复核](../experiments/mask_local_connectivity_server_20261020/STARTUP_CONFIGURATION_REVIEW.md)，157项回传SHA核验通过。训练结果未出，后续按原LCF Gate完整5000图评价；当前Slurm仅提交train，没有自动评价任务。原失败保留，无定时/目标/自动提交推送。

**2026-10-20实机核验更正：下方启动快照已过时。** LCF笔记本Run已于台北时间06:05因`CUDA reserved exceeds 6 GiB`退出，无活跃训练；实际日志A542/B541/C541/D541，共2165步（run.json总计2160为较早快照）。尚无训练后checkpoint，仅四份initial.pt，未评价，不能断点续接或声称科研失败。用户已要求改到算力服务器；只读连接master-01/a25023104成功，本账号Slurm队列为空，尚未提交迁移作业。旧Run、冻结文件与日志保留；重启及预算记账须先在迁移执行补充协议明确。见[实机核验回执](../experiments/mask_local_connectivity_factorial_20261020/LAPTOP_FAILURE_STATUS.json)。

**历史笔记本启动快照（已被工程失败及上方服务器迁移取代）：LCF-v1/V2实例局部解码×连接监督四臂训练。** 用户“试试”承接方案，笔记本独立进程PID24120。正式Run `RUN_LCF_TRAIN_ced92f98a1aa4d8898fe5a24c0b13a39`，回传真日志确认四臂各36/15000步，未完成/未出科研成绩。A/B全局系数约束、C/D局部残差；A/C同类同难度监督控制，B/D连接路径监督，D为唯一主方法。四臂均71361有效参数、同seed0实际初始化张量，原20k图×3轮，总60000步；本Study独立新预算，不冒称旧2k/S3已放行或S4扩量。完整训练标注、25k图片SHA、原历史批次通过；真实smoke32图/每臂4步与额外16图原生/旧G2345条重放及point-F相同，显存峰值约2.40GiB、保守估时79.92小时<96小时。见[启动配置复核](../experiments/mask_local_connectivity_factorial_20261020/STARTUP_CONFIGURATION_REVIEW.md)、[协议V2](../experiments/mask_local_connectivity_factorial_20261020/PROTOCOL.md)、[Study](../experiments/mask_local_connectivity_factorial_20261020/study.json)。完成后同一checkpoint全5000图拓扑/AP/损伤评价与统计，再复核配置；不追加训练/阈值搜索，不恢复SSR/PLF，不设定时任务/目标、不自动提交推送。

**2026-10-20 STA-v1拓扑审计已结案；原设计建议已由上方LCF新授权取代。** [报告](../experiments/mask_split_merge_topology_audit_20261020/REPORT.md)：笔记本CPU完成原256条/243图及6144个已有C压力掩码，训练/推理/系数求解均0。C的97条Mask75失败中47条有实质分裂/混入候选，159条成功中仍有58条；C相对N/旧G既有GT辅助IoU收益+.135158/+.127930，混入量下降、分裂量差区间跨0。408个像素变化≤1%的有效方向中，仅3个实质告警翻转，不能把细碎拓扑变化夸成主要结构瓶颈。20480项独立连通域字段及6144精度标记通过，20项回传SHA与本机6配对均值通过；[配置复核](../experiments/mask_split_merge_topology_audit_20261020/CONFIGURATION_REVIEW.md)。**正向研究重点：局部切断跨实例错误连接，同时保持目标自身正确连接。唯一下一步是设计并冻结全局系数/独立局部编辑的同质量、同幅度、同编辑预算可控性对照，尚未启动。** 现有有限Oracle不是全局最优；全局耦合因果、固定P拓扑不可达及局部新方法效用均未证。SSR保持停止，PLF暂缓，不训练、不设定时任务/目标。

**PLF暂缓：**线性可分性/有理证书程序已准备并在笔记本生成执行前freeze，尚无smoke或正式科学Run。用户新指令优先拓扑审计；保留准备文件与冻结快照，不启动求解，不抹去原计划。

**SSR用户终止：**缺图16344/16344已回传并保留；接续在输入准备标注SHA断言处失败，正式训练0步，已核对本机及笔记本均无SSR活跃Python进程。错误、全部数据与工程证据保留；本次按用户要求停止，不修复或重启。见[终止回执](../experiments/mask_sampling_readout_20261020/USER_STOP_RECEIPT.json)和[原Study](../experiments/mask_sampling_readout_20261020/study.json)。

**历史结案：2026-10-20原256条完整评价已完成（其Run无活跃任务）。** 笔记本RSM-C1完成256条/243图、零训练，C/W同掩码端点精确一致；原10%logit幅度压力下W-C IoU=-0.001422，95%区间[-0.002319,-0.000560]，159见证平均保留95.02096%→94.73270%。218/256中心接受、38回退、1条代理越限，W原幅度12/6144及实际logit差49/6144超差，科学状态仍为INCONCLUSIVE_NUMERICAL；完整读数不冒充原Gate通过。13056次full replay、13312项RLE/IoU、旧113条前缀、96文件回传与两机统计核验通过。见[完整报告](../experiments/mask_repair_margin_completion_20261020/REPORT.md)、[配置复核](../experiments/mask_repair_margin_completion_20261020/CONFIGURATION_REVIEW.md)、[Study](../experiments/mask_repair_margin_completion_20261020/study.json)。**停止该最大最小余量重居中配置的训练及追加修复投入；当时建议恢复G_budget备份作独立确认，尚未执行，现行动作已由上方STA-v1取代（SSR已终止、PLF暂缓）。** 原114工程失败及V3协议不改，不训练、不自动推送、不设置定时或目标。

以下内容是 **“原型—系数链路”这条研究线的「现在做什么」索引**，不是新 Study、科学结果或替代历史实验协议。其设备、预算和 Gate 仅适用于该线，不抢占上方遮挡表征主线或其它独立研究线的计划。新指令出现时更新本页对应研究线的**指向和状态**，不要在旧日报/原计划里继续追加「最新计划」。

**当前执行设备：算力服务器 `a25023104@192.168.71.10`（用户2026-10-20新授权，Slurm）。笔记本 `ssh 28358lan` 保留失败产物，不再训练。** 已核对主机 `YANGLEI`、账号 `yanglei\28358`。2026-10-20当前授权为上方LCF-v1；SSR已按用户要求终止、旧RSM已结案，原生响应漂移草案不执行。服务器登录及既有结果保留，不把服务器AP当作本机同配对重放；已知视频服务保留，不改其它会话任务。新Run是否启动以Study与真实run.json为准。

**研究线命名（2026-10-14，用户指定）：** 展示名称统一为“原型—系数链路”（Prototype→Coefficient）。`neighbor_sensitive_p3` 保留为历史技术标识和目录名，维持既有证据引用；邻域机制、P3 干预等具体方法是否采用，由证据与所属实验协议决定，名称不预设机制结论。本次更名不改变下列阶段、冻结配置或 Gate。

## 0. 唯一研究命题与路线裁决

> 已有系数链路的基准命题：固定官方 `yolo26m-seg.pt` 的预训练网络、Box/Class、原生 Prototype P、原始系数 c0 和正常最终检测身份，只对真实推理输出进行**后训练的无 GT 条件系数残差修复**：`c'=c0+Δcθ`。检验哪些原 Mask75 任务失败的 GT 在同一最终 Box75 检测上具备固定 P 的有限可恢复空间，以及新读出器是否能救回且不会损伤原成功实例。

**主线已执行的评价链：** `output-first GT 任务账本 → 同最终检测 fixed-P coefficient finite oracle → 轻量 adapter pilot → 全量同输出 COCO AP/损伤验收`。局部像素残差、区域F对照及原生掩码分支联合适配均已结案。最后一项按用户授权允许PC的P及原生系数头共同变化、每臂20k×3轮，检测分支/身份/评价配置冻结；不能误写为PC仍固定P。实际PC显著弱于旧G、G_budget和C，关闭本配置继续投入。G_budget的新增收益属于既有适配器在共同新配方下的成果，不能归为PC机制贡献，也不能单独归因为扩数据。旧Gate与冻结阴性保留。

**当前不做：** 重建当年预训练 TAL 动态、重新预训练 YOLO、重扫 30828840 raw、以 GT 选最优 raw 冒充实际推理、为挽救旧阴性结果重开教师/门控/共享仿射配置、未过初筛直接做五臂×20k。历史 TAL 是可选独立审计，不是本后训练主线的依赖。

## 1. 当前工作队列：同一时刻只激活一个 Gate

| 阶段 | 研究对象与唯一动作 | 必需输入/通过判据 | 产物/进度 |
|---|---|---|---|
| **LCF-S1：评价已完成，联合门槛未通过（2026-10-22验收）** | 四臂各15,000步已完成；Slurm178 完整5000图评价，主D局部解码+连接监督 | 拓扑相对N恶化5.43%、混入显著上升、D显著劣于B、D−C仅3.07%；四臂损伤1.768–1.964%超1% | Slurm154训练+Slurm178评价；[验收报告](../experiments/mask_local_connectivity_server_20261020/ACCEPTANCE_REVIEW_20261022.md) · [验收回执](../experiments/mask_local_connectivity_server_20261020/ACCEPTANCE_RECEIPT_20261022.json) · [Study](../experiments/mask_local_connectivity_server_20261020/study.json)；statistics阶段待授权 |
| **STA-v1：已结案，现象支持、因果未证** | 分裂/混入与已有C压力；256条/243图、6144方向；唯一下一步设计局部断桥—连通保持匹配对照，未启动 | 无新增训练/推理；GT多部件与漏覆盖分开；独立连通域/精度/传输核验通过 | [报告](../experiments/mask_split_merge_topology_audit_20261020/REPORT.md) · [配置复核](../experiments/mask_split_merge_topology_audit_20261020/CONFIGURATION_REVIEW.md) · [Study](../experiments/mask_split_merge_topology_audit_20261020/study.json) |
| **PLF-v1：用户改题后暂缓** | 已准备线性可分性程序、freeze；尚无科学Run | 不启动；不宣称精确像素不可分等于Mask75不可达 | [原协议](../experiments/prototype_linear_feasibility_20261020/PROTOCOL.md)及准备文件保留 |
| **SSR-v1：用户终止，训练未启动** | 原六臂预算停止，缺图保留 | 不再接续、修复或启动训练 | [终止回执](../experiments/mask_sampling_readout_20261020/USER_STOP_RECEIPT.json)；旧协议与失败证据保留 |
| **RSM-C1：完整读数已结案，数值未决** | 原256条/243图完成；W-C受压IoU=-0.001422，见证保留率增量-0.28826个百分点 | 原门槛不变；218/256候选、幅度/代理未全有效，不进入稳定性训练 | [报告](../experiments/mask_repair_margin_completion_20261020/REPORT.md) · [配置复核](../experiments/mask_repair_margin_completion_20261020/CONFIGURATION_REVIEW.md) · [Study](../experiments/mask_repair_margin_completion_20261020/study.json)；两机核验/回传完成，无活跃Run |
| **RSM-v1：114数值恢复核验结案；无有效完整修复** | 原16图重新生成精确一致；W原方向22/24、实际双读数16/24，原C24/24 | 原精度/科学协议未改；未启动新smoke/256；下一步仅建议先设计稳定系数表示的目标等价性检查 | [恢复报告](../experiments/mask_repair_margin_20261019/RECOVERY_REPORT_20261020.md) · [配置复核](../experiments/mask_repair_margin_20261019/RECOVERY_CONFIGURATION_REVIEW_20261020.md) · [Study](../experiments/mask_repair_margin_20261019/study.json)；3工程Run、117候选独立核验、157文件回传完成 |
| **联合掩码分支已结案；论文证据整合已完成** | G_budget收益和PC/C对照已并入当前结果、讨论及证据主张表；作为历史证据保留 | 三臂20k×3轮各15000步、5000图、2000配对统计完成；PC四项主增量失败，三护栏通过；最高点估计G不等于事后通过主Gate | [正文](../paper/manuscripts/neighbor_sensitive_p3/RESULTS_AND_DISCUSSION_ZH_20261017.md) · [证据表](../paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_AND_CLAIMS_20261017.md) · [Study](../experiments/mask_branch_joint_replay_20261018/study.json)；G/N/旧G为44.265173/44.013836/44.152393，G损伤81/22958；PC44.017703，损伤198/22958；配置/回传核验沿用结案回执 |
| **RFC-v1 已结案；证据入稿建议暂缓** | 有序区域F的A/Q原预算训练与固定1294图评价完成；关闭本配置新增投入 | A−N修复/总体护栏通过；A−G实用增量未过；损伤上界精度不足；四项机制CI跨0；配置/身份通过 | [报告](../experiments/mask_region_f_correspondence_20261017/REPORT.md) · [配置复核](../experiments/mask_region_f_correspondence_20261017/CONFIGURATION_REVIEW.md) · [Study](../experiments/mask_region_f_correspondence_20261017/study.json)；RUN_RFC_EVAL_RECOVERY_e2b485f03a5f4e96a0f96ab327f4e39f；两机复核及2041项回传通过；原结案后继建议为整合论文证据；当前被上方新路线设计暂缓。不补训/补样/晋升Q或S3 |
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
| **LEA-v1：已结案，当前域阴性** | 原256条/243图，原生低置信可编辑域；GT辅助支持域上界及限制已有G的完整输出；零训练 | 见证保留≥152/159、额外机会与定位价值、局部性≤25%、原成功损伤≤1%且不多于G；先smoke再正式 | [协议](../experiments/mask_local_editability_20261017/PROTOCOL.md) · [Study](../experiments/mask_local_editability_20261017/study.json)；正式RUN_LEA_FULL_b383a8db922c451cbc77f2069d385cd7；C见证保留132/159、局部占比32.99%，两项失败；GT上界204/256，G_B较G退化，242项回传与两机复核通过 |
| **LRP-v1：已结案，实用幅度/历史增量失败** | LOCAL/UNIFORM各1600图×3轮1200步；固定last完整400图42167输出评价 | T490上L−N+.002100、L−U+.001884，两项95%上界<.005；L−G−.007460区间全负；损伤1/1808及总体护栏通过 | [报告](../experiments/mask_local_residual_pilot_20261017/REPORT.md) · [配置复核](../experiments/mask_local_residual_pilot_20261017/CONFIGURATION_REVIEW.md)；RUN_LRP_EVALUATION_f89d8084795244a985b37031a6b395a9，两机核验及2145项回传通过；关闭当前配置 |
| **S4 不启动**：扩量/独立确认 | S3通过后才规划扩量或开发外确认 | 需S3通过并有新授权/冻结预算 | 本候选S3未通过，不晋级、不扩20k，不把重复开发COCO当独立确认 |

**已完成的原S1固定256条机会兑现核对保留。** [结案报告](../experiments/mask_opportunity_audit_20261015/REPORT.md)对应正式Run `RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd`：原122条oracle阳性分解为L=100（无门控Δc仍未修好）、G=10（已修好但门控撤回）、R=12（修好且保留）；134条原oracle未找到解完整保留。110条未兑现机会中学习侧占90.91%，按事前协议优先选择可恢复失败的条件修复学习；门控也损失10/22条已实现修复，不能忽略，但放宽门控不能解决另100条。49,976行原生/门控重放与旧输出精确一致、256条oracle重放和独立RLE IoU通过；[配置复核](../experiments/mask_opportunity_audit_20261015/CONFIGURATION_REVIEW.md)、80文件回传与两机独立核验通过。本机首次Python3.10描述性均值末位相等检查失败已保留；原冻结脚本在本机Python3.12复验全通过，未改容差或科学结论。

**此前soft-IoU训练内/开发落差核验已完成，以下作为本次目标对照的来源证据。** [报告](../experiments/mask_soft_fit_audit_20261016/REPORT.md)与[配置复核](../experiments/mask_soft_fit_audit_20261016/CONFIGURATION_REVIEW.md)对应正式Run `RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770`。训练218209输出、T2669、原成功9748；门控内有效监督3453/14268=24.2%，涉及1620张图。G训练目标+.010484、总体+.002194、损伤31/9748，诊断三项满足；开发G目标+.010024、损伤8/2355（本次512图census，不替代旧S2抽样上界）。无门控U训练损伤394/9748，说明保护问题在训练内也存在。分层差距区间跨零且共同支持不足，不认定小样本因果，也不宣称train/dev统计等效。140文件回传、两机独立复核、完整输出身份及12097配对RLE核验均通过，新增训练/基础模型forward/oracle求解均为0。

**证据综合于2026-10-19更新：** [论文证据与主张表](../paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_AND_CLAIMS_20261017.md)保留E0—E15，并以E16补入完整三臂命题，区分GT辅助机会、无GT实际兑现和原成功GT损伤。[来源清单](../paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_AND_CLAIMS_20261017.sources.json)保留历史154项来源并增加三臂Run、冻结配置及结案核验的哈希和字段；新增10个Run摘要按原POINTERS/产物清单核对。本轮只写作与读取原产物，不创建新Study/Run或重算评价。各旧报告中“下一步尚未启动”保留其历史状态，本页登记论文整合已完成。

**结果与讨论正文已补齐本次授权内容：** [中文正文](../paper/manuscripts/neighbor_sensitive_p3/RESULTS_AND_DISCUSSION_ZH_20261017.md)新增三臂双参照AP/区间/损伤、PC四项Gate与归因边界，明确G_budget收益不归PC，PC不固定P，历史22957与服务器22958成功GT分开；[正文核验收据](../paper/manuscripts/neighbor_sensitive_p3/RESULTS_AND_DISCUSSION_ZH_20261017.audit.json)记录来源、表格与引用核查。旧S3和失败Run原判定保留，旧raw/argmax与P3完整稿未改写；当前是结果讨论节选完成，不等于整篇投稿稿已统一合稿。

**RFC-v1已完成训练与评价并关闭新增投入。** 当前状态只见本页顶部与Study；本段保留设计依据。[执行协议](../experiments/mask_region_f_correspondence_20261017/EXECUTION_PROTOCOL.md)启用原[方案](../experiments/mask_region_f_correspondence_20261017/PROTOCOL_DRAFT.md)：依据[空间信息审查](audits/repair_information_20261017/REPORT.md)及[275项来源收据](audits/repair_information_20261017/EXISTING_ARTIFACT_REVIEW.json)，保留旧单点F，用共享投影把同层8×8区域F接入既有卷积，不加参数；A/Q只差F排列，预登记同checkpoint置换与等功能幅度对照。两臂沿用G的原2000图/1500步，在原512图比较，避开旧G训练过400图的问题。此前[CPU核心工程报告](../experiments/mask_region_f_correspondence_20261017/IMPLEMENTATION_REPORT.md)及design_snapshot保留当时范围；现已补齐原批次重放和训练集成，新增批次选择测试后29项通过，真实运行独立登记。额外效用、原S2护栏和机制分别裁决；若效用未过则停止该配置并整理论文证据，不自动变网格/换loss/加训。不解冻P、不扩量。

**此前空间信息审查暂缓的论文补写建议：** 把LEA/LRP结案证据补入论文结果讨论仍未执行，其既有事实保持。[LRP整体报告](../experiments/mask_local_residual_pilot_20261017/REPORT.md)已按冻结门槛关闭：相对原生和统一偏移控制有小幅正值，但95%上界低于.005、对历史G增量全负；G训练过本400图，差值不能作匹配训练条件的架构因果结论。G全量实证保留，LRP独有恢复1例且独有损伤1例，不直接组合。[LEA结案](../experiments/mask_local_editability_20261017/REPORT.md)覆盖/局部性阴性原样保留。该次审查仅读取原产物；其后的RFC设计/工程进度以上方当前路由为准。

CLD结案时的历史取舍依据为累计实测：旧方向有限调幅只兑现32/159见证；新CLD正确对应相对PERM/J分别+.045044/+.021467，但留出/拟合GT辅助方向相对N分别−.101720/−.056781，正常候选相对历史G目标−.009738且区间全负。因此保留“实例对应有相对作用”的有限事实，关闭“该直接方向监督能形成有益修复”的当前配置，不把它概括成输入信息绝对无用或只因样本少。正向重点收束到**已有G的低损伤实际效用与可修机会的兑现边界**；旧G在5000图的+.141573 AP与未达到+.30门槛同时保留。AMT、EK-v1及其它旧阴性约束不改写。

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


**2026-10-17此前训练授权与参数化范围（已履行）：** 用户要求“不论门槛如何继续至完成一个实际训练”，这是对LEA旧“后继不启动”执行限制的明确扩展，现已履行。原LEA冻结协议/阴性不改。本次LRP用逐像素δz替代全局Δc作为独立探索参数化，P/c0/基础网络/检测身份仍冻结；仅在固定B内最终硬合成。证据依据为B内GT上界204/256及对等面积随机域+.095538 IoU，不把低置信当正确性保证。该科学问题已按协议完成：可测小收益存在，实用幅度与历史方法增量不足；两臂训练和400图评价后已停止，唯一后继见上方结案路由。旧未冻结后继建议均服从此路由。

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
