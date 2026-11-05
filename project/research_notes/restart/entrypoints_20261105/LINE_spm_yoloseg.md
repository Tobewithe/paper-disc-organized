# 入口整理前快照（2026-11-05）

原文件：[paper/manuscripts/spm_yoloseg/README.md](../../../../paper/manuscripts/spm_yoloseg/README.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
# SPM-YOLO：合成—真实原型记忆增强的猪群实例分割

**2026-11-05夜间三组Idea验收完成：** [完整报告](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)覆盖24配置、三配对seed、102192正式更新及完整432dev；原79作业与新增1258／1259验收全部COMPLETED／0:0，独立指标／状态误差0，24最终包精确重放通过。累计25.591389GPUh／102.875556CPU-coreh，该次验收结束时队列为空，见[结案汇总](../../../../experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json)。

同COCO起点正确部位C−Q平均−0.513151且异号，C−R+1.960429低于非语义Q−R+2.473579，A本版停止扩训；C21细尺度保持仅+0.024948、成功状态变化0／−1／+1，C本版同样结束。V23形变置零／共享保留相对完整重训+0.010234／+0.148434，共享三seed正向且简化均过护栏；优先保留共享功能约束用于下一步真实结构压缩，原V23继续作成熟参照。共享平均相对原V23仅+0.063368，原N+5目标尚未达到。

认识已反馈具体Idea、学习／表示／评价及相关上层／中心纲要。[固定反馈流程补充](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)已将本次工程重试、复判条件、全流程复盘、Skill评估和下一步带回原Study。本轮汇总级诊断发现A的9个配对配置从5到20轮Mask AP和输出数全部下降，中尺度损失明显、大尺度持平或提高；下一步先拆解覆盖／排序／匹配Mask质量，再剖析B的实际成本以选择压缩对象。原V23继续作成熟参照，A/C旧版保持关闭；范围、预算和退出条件以报告第8.6节为准。当前远端SSH超时，下一轮逐例和成本作业尚未提交，见[总控](../../../ACTIVE_PLAN.md)。

**启动历史（2026-11-05）：夜间三组Idea正式训练已启动：** 新[Study／协议](../../../../experiments/pig_multi_idea_night_20261105/study.json)比较[COCO起点联合学习](../../../ideas/IDEA_PIG_COCO_JOINT_V1.md)、[V23功能约束后重训](../../../ideas/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md)及[C21细尺度适配与保持](../../../ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md)。工程67次scratch与共同初值／实际梯度／成本风险／重载通过，1189–1196已同时在8张不同GPU上真实更新。24配置、三个配对seed及训练→评价→独立验收共70任务已排队，见[启动交接](../../../../experiments/pig_multi_idea_night_20261105/HANDOFF.md)。用户要求启动交接后暂停会话，远端继续；当前研究判断待完整结果。

本批围绕已核缺口检验学习起点、低依赖组件重训补偿和局部可塑性，既有内容正确对应问题继续开放。依用户“显卡全用”本Study使用8卡例外，GPU总帽64小时、请求上界61.75小时，默认六卡规则保留。旧N+5目标、test426封存与原分配矩阵／LCF暂停保持；实际当前安排见[总控](../../../ACTIVE_PLAN.md)。

**2026-11-02用户澄清：** [中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)讨论更大的真实猪场实例分割命题，本线从对象／猪体知识切入，只是一个探索入口。新的[开放探索图](../../../outlines/OUTLINE_PIG_HYPOTHESES.md)比较表示、观察、标注、合成工具、可靠性与迁移等方向，不要求都服务于概念训练。上一轮[论文叙事草案](../../../../paper/manuscripts/spm_yoloseg/PAPER_NARRATIVE_AND_IDEAS_20261102.md)保留为局部历史材料，不是当前写作任务或中心命题的定义。

**纲要中的取舍与开放问题：** 原质量头V25／V26、RLPL V2和原生部位V1的停止范围已回写[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，并连接各具体Idea与原报告。纲要由这些认识及其他资料形成新Idea，本线组织其中所选想法的探索，实际结果再反馈到纲要；不把研究线登记或本目录变成中心命题的全部范围。

**实际状态仍以[总控](../../../ACTIVE_PLAN.md)为准：** RLPL、原生概念V1及其重放已完成并按原判定关闭；分配矩阵六臂各2000末点元数据已核，CP/QP终态已于2026-11-03对账，评价仍暂停、dev未读。研究假设不恢复训练/评价/生成或远端重试，test426仍封存。以下保留各历史阶段记录，旧“尚未开训”“当前目标”“唯一下一步”均只按其日期和范围解释。

**2026-11-04 C21／V23消融已完成验收与知识反馈：** [具体Idea](../../../ideas/IDEA_PIG_C21_V23_ABLATION_V1.md)及[完整报告](../../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)覆盖固定原模型、七项组件移除、完整432dev。1173–1176全部COMPLETED／0:0，十组各12899条输出；独立指标、GT状态、来源重抽与原始证据SHA复核通过。原[启动交接](../../../../experiments/pig_c21_v23_ablation_20261104/HANDOFF.md)继续保留提交时快照。

**本轮对研究选择的调整：** C21内容输入／细尺度移除损失2.306973／2.565348 AP，V23外观码损失1.315483；强制完整候选使两模型损伤大增并出现1376／1816条空Mask，更支持内容重建与选择保护协同。当前保留原V23，将形变和分部位保留图列为简化候选。下一科学问题优先辨别保持内容总量时正确空间／部位对应与正常选择器响应的作用，相关认识已回到[猪知识Idea](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)、表示／学习／评价缺口及中心纲要。分配矩阵继续暂停，test426封存，该轮结束时队列为空。

**2026-11-04认识回流：** [综合复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)将N的多阶段保留权重身份、C/Q弱且seed异号、D2低于共同曝光无辅助B2等区别写回[部位V1](../../../ideas/IDEA_PIG_NATIVE_PART_V1.md)、[RLPL V2](../../../ideas/IDEA_PIG_RLPL_V2.md)和[广义猪体知识](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)，再反馈[监督问题](../../../outlines/OUTLINE_PIG_SUPERVISION.md)、[学习缺口](../../../outlines/OUTLINE_PIG_TRAINING.md)与中心纲要。已有阴性不因训练整合尚有未知被撤销，同COCO起点直接比较的缺项也不自动授权新训练或限制其他开放Idea。

**2026-11-01正式训练请求已做启动前核验，尚未开训：** [本次报告](../../../../experiments/spm_rlpl_pipeline_20261101/TRAINING_REQUEST_REPORT.md)记录936/940真实监督语义复核：样例两Mask交集占较小者48.9177%，另有设施填入，关闭未执行的“严格可见合成/真实GT直接同口径监督”默认。选定[监督V2](../../../../experiments/spm_rlpl_pipeline_20261101/SUPERVISION_RESCOPE_V2.md)：原真实GT分割监督、合成仅供共视辅助，B2/C2/D2比较口径显式重登记。它未实现/训练，不能冒充原B/C/D已运行。此前[工程报告](../../../../experiments/spm_rlpl_pipeline_20261101/REPORT.md)的28/18/37项检查与12合成样例仍有效；938新生成器入口失败、修复未重试，200组release/正式合同仍缺。优化0步，无新AP/训练权重；原GT不改、不进生成器，test426封存，旧结果/事故保留。实际路由只见[总控](../../../ACTIVE_PLAN.md)。

**2026-11-01论文候选审查已完成：** [RLPL主张与最小对照](../../../../paper/manuscripts/spm_yoloseg/PAPER_CANDIDATE_RLPL_20261101.md)选择训练期可见局部对应的原型分配，部署仍用原生YOLO；[近邻审查](../../literature/spm-prototype-candidate_2026-11-01/README.md)明确SwAV/DenseCL/PixPro等已有做法与本候选需要验证的增量。[新Study](../../../../experiments/spm_registered_prototype_learning_20261101/study.json)仅完成设计/本地抽象检查，无新模型训练或AP；不恢复后置Mask修补，不把旧V23收益记为新数据/新机制成果。旧+5目标与test426封存不变，后续实际动作只读[总控](../../../ACTIVE_PLAN.md)；下文历史“下一步”不自动生效。

**当前完整训练交付目标v4（已重建并active）：** [目标与完成方式](../../../../paper/manuscripts/spm_yoloseg/MODEL_DESIGN_GOAL_20261025.md)、[创建快照](../../../../paper/manuscripts/spm_yoloseg/MODEL_TRAIN_DELIVERY_GOAL_V4_20261025.json)包含实际训练、接入微调、完整真实验收、权重/复现入口和技术/效果图。V21现已完成8000+2568更新及[完整交付](../../../../experiments/spm_spatial_object_memory_20261025/REPORT.md)，Mask AP+3.073026、框不变、护栏通过、46图逐张实际看过，+5未达。V22[固定模型空间破坏](../../../../experiments/spm_spatial_structure_audit_20261026/REPORT.md)完整432正常RGB仅−0.044647点、0优化；它削弱当前有序内容的实用作用，未唯一识别所有收益来源。[V23内容约束组合重建](../../../../paper/manuscripts/spm_yoloseg/CONTENT_CONSTRAINED_ARCHITECTURE_V23_20261026.md)已完成8000+2568实际更新及全部九项验收作业：[完整真实结果](../../../../experiments/spm_content_constrained_memory_20261026/FORMAL_PROGRESS.json)为Mask AP60.753278，较N+3.409778、较C21+0.336752，框及护栏通过，+5未达；96项统计和12RGB重载一致。[934原文件及实物交付](../../../../experiments/spm_content_constrained_memory_20261026/FINAL_ITERATION_DELIVERY_VERIFIED.json)完成，57图已实际逐张看过；+5目标继续，下一步为固定候选的有限机会裁决。全部理论依据只维护[统一理论文件](../../../../paper/manuscripts/spm_yoloseg/SPATIAL_OBJECT_THEORY_SUPPORT_20261025.md)，goal不能在设计/smoke处结案。旧Run/Gate保留，唯一执行路由只见[总控](../../../ACTIVE_PLAN.md)。

**功能适配对象记忆v2.2设计草案：** [17篇文献与五轮数学复审](../../../../paper/manuscripts/spm_yoloseg/FUNCTION_ALIGNED_MEMORY_V2_2_MATH_REVIEW_20261025.md)、[接口合同](../../../../paper/manuscripts/spm_yoloseg/FUNCTION_ALIGNED_MEMORY_V2_2_CONTRACT_20261025.json)及[22项代数/反例核验](../../literature/spm-function-math_2026-10-25/MATH_CHECKS.json)将[v2.1](../../../../paper/manuscripts/spm_yoloseg/FUNCTION_ALIGNED_MEMORY_V2_1_REVIEW_20261025.md)收敛为任务一致检索、可见部位组成及成本敏感联合幅度策略。四类功能表示/原生检测和Mask锚定保留，修订旧alpha gate与C阶段梯度选择。当前为完成文献/数学审查的设计，实际新网络/训练均未开始；唯一执行路由仍见[总控](../../../ACTIVE_PLAN.md)。

**2026-10-25 V20训练与完整验收已完成：** 双A6000实际10000步、160000图曝光，493条支持词表与解码器已导出回传。[报告](../../../../experiments/spm_semantic_dense_task_20261025/REPORT.md)保留真实Box AP66.884503→58.231517、Mask AP57.343500→51.961884；合成Mask AP+15.857点，真实效用及语义检索必要条件未通过。真实整体/部位平均读入仅1.401%/.128%，源prone亦复核为屈腿代理。[模型用法](../../../../experiments/spm_semantic_dense_task_20261025/USAGE.md)、[完整技术图集](../../../../experiments/spm_semantic_dense_task_20261025/VISUAL_GUIDE.md)、[独立及本地校验](../../../../experiments/spm_semantic_dense_task_20261025/LOCAL_RETURN_VERIFIED.json)已交付；旧结果不改。当前动作与后继推荐只看[总控](../../../ACTIVE_PLAN.md)。

**2026-10-24 V17/V18完整结案及Blender原因诊断完成：** [训练报告](../../../../experiments/spm_object_memory_training_20261024/V18_REPORT.md)保留末C30真实Mask AP57.343500→47.697063（−9.646437点）、修复35/损伤405、同框FN+294.29%；[权重](../../../../experiments/spm_object_memory_training_20261024/V18_MODEL_USAGE.md)和[48组技术/结果图](../../../../experiments/spm_object_memory_training_20261024/C_VISUAL_GUIDE.md)均已核验。[Blender报告](../../../../experiments/spm_blender_domain_replay_20261024/REPORT.md)完成8布局/32新图/64RGB零优化对照：简单外观+1.811点、参考材质/照明−5.681点，四种外观同框IoU均下降，栏杆标注语义另影响相对成绩。支持读出过删且外观放大，不证明合成原则上无法拟合真实；[8份场景与18组图](../../../../experiments/spm_blender_domain_replay_20261024/VISUAL_GUIDE.md)已交付。用户要求结案后自动训练循环停止；当前唯一动作及后继推荐只看[总控](../../../ACTIVE_PLAN.md)。

**历史V17设计快照（已实际训练，设计稿不是当前状态）：** 按用户明确的“数据学习对象→原型存储→实例检索→特征与Mask重建”，形成[三阶段方案v1](../../../../experiments/spm_object_memory_training_20261024/TRAINING_DESIGN_V1.md)及替代其网络/优化建议的[v2详细设计](../../../../experiments/spm_object_memory_training_20261024/NETWORK_AND_OPTIMIZATION_V2.md)。冻结YOLO，采用全局候选加局部空间组合、三类记忆与约4.98M新增参数，先记忆重建、再实例匹配、最后真实分割适配；实际执行/阴性和原C依赖替代见上方报告及[Study](../../../../experiments/spm_object_memory_training_20261024/study.json)，旧结果保留。

**2026-10-24 V16真实身体聚合读出训练与验收完成：** [报告](../../../../experiments/spm_instance_body_readout_20261024/REPORT.md)完成1272真实train、10817参数、固定4000步末EMA；432dev/2863GT的Mask AP57.3585→52.6420（−4.7165pp），框及12906条检测身份精确不变，修复32、损伤187/2127（8.7917%）。同一Box75检测的FP−42.66%、FN+97.09%，实测主要损伤为真前景过度删除。[完整权重与用法](../../../../experiments/spm_instance_body_readout_20261024/USAGE.md)、[11组主要图](../../../../experiments/spm_instance_body_readout_20261024/VISUAL_GUIDE.md)、[本地验收](../../../../experiments/spm_instance_body_readout_20261024/LOCAL_ACCEPTANCE.json)完成；48项独立AP差0、6次RGB一致，GPU已释放。关闭本配置扩训，保留V13原生与V15合成身体能力；优先核实真实像素归属信息，具体唯一下一步只看[总控](../../../ACTIVE_PLAN.md)。新核验未启动，原型+5目标未完成，无GitHub推送。

**2026-10-23 V15对象已实际训练并交付：** [报告](../../../../experiments/spm_canonical_object_training_20261023/REPORT.md)完成6000步固定末EMA，框条件身体PCK@0.10为89.8878%、左右侧卧89.6118%/86.3037%，五项几何要求通过；完整可见归属IoU57.2013%未达75%。[完整对象权重/用法](../../../../experiments/spm_canonical_object_training_20261023/USAGE.md)、[11组主要技术图](../../../../experiments/spm_canonical_object_training_20261023/VISUAL_GUIDE.md)、12次RGB重载及2389项独立核验完成。当时选定的[真实可见实例密集残差读出](../../../../experiments/spm_canonical_object_training_20261023/NEXT_MASK_READOUT_PLAN.md)现由上方V16执行完成。合成验收有GT实例框/新增上下文，不作真实AP或独立几何因果结论；原型+5目标未完成，当前动作只看[总控](../../../ACTIVE_PLAN.md)。

**2026-10-23 V14身体对应能力验收完成：** [完整报告](../../../../experiments/spm_surface_correspondence_20261023/REPORT.md)在576合成dev/147456身体点测得当前PCK@0.10为38.7465%（要求70%），侧卧16.5527%/19.9422%；学习度量改善了初始19.1779%，局部精细对应仍有价值，但全身对应不可靠。96张双猪受控拼合同点再降8.7524pp。Slurm265/266完成，优化0；10648项独立复算与[8组图](../../../../experiments/spm_surface_correspondence_20261023/VISUAL_GUIDE.md)验收。已选定[跨姿态规范身体对应训练方向](../../../../experiments/spm_surface_correspondence_20261023/NEXT_TRAINING_PLAN.md)，新神经训练尚未启动；V13独立留出建议顺延，当前动作只看[总控](../../../ACTIVE_PLAN.md)。

**2026-10-23 当前框优先结果已交付：** [V9–V13完整实验链路](../../../../paper/manuscripts/spm_yoloseg/BOX_FIRST_RESULTS_V9_V13_20261023.md)选定[原生纯几何组合](../../../../experiments/spm_geometry_score_preservation_20261023/REPORT.md)：相对原N Box AP+0.3930pp、Mask AP+0.0432pp，Box75成功2385→2388、18修复/15损伤（0.6289%）。[完整权重](../../../../experiments/spm_geometry_score_preservation_20261023/USAGE.md)、[技术/结果图](../../../../experiments/spm_geometry_score_preservation_20261023/VISUAL_GUIDE.md)、72项独立复算与SHA验收完成。当前首选不含新增原型；原型额外联合收益和+5目标仍未完成，静态字段/类前景/共享检测融合当前配置关闭扩训。结案时建议的独立留出仍未启动，其当前顺序由V14后的总控取代。

**历史V6结案与当时V8候选（已由V9–V13执行更新）：** [报告](../../../../experiments/spm_object_fusion_design_20261023/REPORT.md)/[27组图](../../../../experiments/spm_object_fusion_design_20261023/VISUAL_GUIDE.md)：F−B−0.0911pp，修复5、损伤11/2101，+5未达；模型/预测/独立统计和纯RGB重载均验收通过。累计[32篇一手文献](../../literature/spm-object-grounding_2026-10-23/README.md)与源码核验后，选定[V8视觉锚定表面场](../../../../paper/manuscripts/spm_yoloseg/GROUNDED_OBJECT_ROUTE_V8_20261023.md)，保留几何与最强原生，先补对象字段和图像的共同特征监督。[候选参考代码](../../../../experiments/spm_grounded_surface_20261023/study.json)已准备，实际模型预检/字段拟合/新训练尚未执行；当前唯一动作见[ACTIVE_PLAN](../../../ACTIVE_PLAN.md)。

**历史18:11进度（当时验收待补）：** 两组各8000步正常结束，均选2000步；F/B Mask AP56.2637/56.3548，差−0.0911pp，均低于初始57.3153。见[最终进度摘要](../../../../experiments/spm_object_fusion_design_20261023/STATUS_20261023_1811.json)。唯一执行动作仍是既有模型/预测、护栏和图的最终验收，不追加训练。用户新增文献研究已完成：[22篇一手来源](../../literature/spm-object-fusion_2026-10-23/README.md)、[V7数学推导与训练建议](../../../../paper/manuscripts/spm_yoloseg/OBJECT_FUSION_THEORY_AND_LITERATURE_V7_20261023.md)，优先规范表面对应与原生函数保护；属于研究备忘，未实施或开训，不替代V6冻结合同。

**历史17:24 V6进度快照：** 原生`243_1`已完成8000步并释放显卡，融合`243_0`在3350/8000步正常运行；同2000步完整432dev的B/F Mask AP56.3548/56.2637，融合暂低0.0911pp，均低于初始57.3153。原生按冻结规则选2000步，融合尚未跑完，不提前作最终结论。见[状态读数](../../../../experiments/spm_object_fusion_design_20261023/STATUS_20261023_1724.json)、[Study](../../../../experiments/spm_object_fusion_design_20261023/study.json)、[执行合同](../../../../experiments/spm_object_fusion_design_20261023/EXECUTION_PROTOCOL_V1.md)及[启动检查点](../../../../experiments/spm_object_fusion_design_20261023/CHECKPOINT.md)，当前动作以[ACTIVE_PLAN](../../../ACTIVE_PLAN.md)为准。继续现有融合组冻结预算，不追加消融/训练；+5未达成，不推送或创建接续任务。

**2026-10-23 对象真实迁移v1已结案：** 用户“试试”后完成Slurm233–238、真实1272train/432dev及S/Q各4000步。[报告](../../../../experiments/spm_object_real_transfer_20261023/REPORT.md)、[两模型与候选库](../../../../experiments/spm_object_real_transfer_20261023/artifacts/README.md)、[10组技术图](../../../../experiments/spm_object_real_transfer_20261023/VISUAL_GUIDE.md)已交付。S/N/Q Mask AP50.9072/57.3153/50.2174；S对N−6.4080pp，关对象输入反而52.1663，当前接口效用阴性。独立60统计误差0，原生完整parity误差0，test未读；+5目标未完成。保留原生与合成对象生成器，关闭当前候选套框→ROI修正扩训；结案时建议的监督关联与采样核查（76背景占25%）没有执行，当前已由V6融合研究取代其前置优先级，具体路由见[ACTIVE_PLAN](../../../ACTIVE_PLAN.md)。

**2026-10-23 V5对象v0已真实训练并结案：** [报告](../../../../experiments/spm_object_prototype_20261023/REPORT.md)、[模型/推断](../../../../experiments/spm_object_prototype_20261023/artifacts/README.md)和[18张技术图](../../../../experiments/spm_object_prototype_20261023/VISUAL_GUIDE.md)已交付。用户明确授权后Slurm229–232完成2304图、两臂各4000步、固定评价及独立验收；S完整身体误差0.01913体长，对U按族平均减少46.67%，不同视角完整投影IoU0.89934对U0.85639。但体形码漂移0.22384>0.10，联合Gate未过；v0本身不评价真实Mask AP、+5目标未完成。保留几何先验和S权重，关闭该配置扩训；当时后继建议为潜码/几何漂移核验，现行动作由上方真实迁移结案及ACTIVE_PLAN取代，旧Gate与结论不改。

**2026-10-23 V5设计快照（本轮训练前）：** 用户将重点明确为“对象本身如何形成”。[V5 设计与训练](../../../../paper/manuscripts/spm_yoloseg/OBJECT_PROTOTYPE_DESIGN_V5_20261023.md)选择共享可变形身体、标准部位及特征场，分离个体体形、姿态、视角和可见性；通过跨视角预测与遮挡恢复学对象，再适配YOLO。[对象Study](../../../../experiments/spm_object_prototype_20261023/study.json)的实际执行已见上段，具体路由只看[ACTIVE_PLAN](../../../ACTIVE_PLAN.md)。前驱[实例均值读出实验](../../../../experiments/spm_instance_readout_opportunity_20261023/REPORT.md)Slurm221/222已结案：有部分特征区分作用，但GT支持仅修复2/305，无GT AP25.2765；不扩大该读出配置。V5取代未启动的直接Mask监督建议，旧Run与冻结规则保留。

**2026-10-23 全量V4.1完成：** 已完成1272train/432dev、多样场景、120轮基线、20000步实例原型及四臂各8000步。F/B/G/R Mask AP57.3570/57.3153/57.2036/57.2395，F对B仅+0.0417pp；off仅−0.0565pp，原型特异效用未获支持，至少+5点goal未完成。[完整报告](../../../../experiments/spm_v4_scale_20261023/REPORT.md)、[17组图](../../../../experiments/spm_v4_scale_20261023/VISUAL_GUIDE.md)、[本地验收](../../../../experiments/spm_v4_scale_20261023/FINAL_LOCAL_ACCEPTANCE.json)已交付。后继[实例条件稠密读出机会](../../../../experiments/spm_instance_readout_opportunity_20261023/study.json)核验现有嵌入能否形成真实实例Mask；GT上界不算部署收益。唯一动作路由见[当前总控](../../../ACTIVE_PLAN.md)，不继续扩大本配置固定全局字典。

**研究线 Key：** `spm_yoloseg`  

**2026-10-23 V4快速首轮已完成：** [报告](../../../../experiments/spm_v4_fastpilot_20261023/REPORT.md)、[13张技术/结果图](../../../../experiments/spm_v4_fastpilot_20261023/VISUAL_GUIDE.md)和[权重用法](../../../../experiments/spm_v4_fastpilot_20261023/USAGE.md)已交付。Slurm191/192/193完成64几何/128图、B*600步、原型1000步、continued/G/F各400步及免检索导出。F Mask AP50.9267，对同预算continued +1.3045pp、G +1.3725pp、B* +0.2306pp；AP75相对B*下降。关闭检索几乎无变化，优先使用保留SPM训练权重的原生head（AP50.9272）。单seed54图开发结果，不称独立测试或机制证明；当前路由仅看ACTIVE_PLAN。下方旧保存、not_started和Probe顺序保留历史范围。

**2026-10-23保存检查点：** 用户要求先保存。新猪栏生成器已完成Slurm189工程样例（8独立场景、16外观图、90头几何实例），标签与射线核验通过；见[接续说明](../../../../experiments/spm_farm_scenes_20261023/CHECKPOINT.md)与[报告](../../../../experiments/spm_farm_scenes_20261023/REPORT.md)。正式128场景和新版原型更新尚未启动，旧P1原型完整保留。当前仅保存，不追加作业；恢复时先补符合真实猪体尺度的相机取景，当前路由以ACTIVE_PLAN为准。

**历史P1产物（2026-10-22）：** 按用户“训练出原型”的明确要求，Slurm184正式完成2000步投影训练与双源构库。已有[256个128维原型](../../../../experiments/spm_prototype_training_20261022/artifacts/pig_prototype_memory.pt)和[投影权重](../../../../experiments/spm_prototype_training_20261022/artifacts/pig_projector.pt)，重载/来源/检索核验通过；见[训练报告](../../../../experiments/spm_prototype_training_20261022/REPORT.md)与[用法](../../../../experiments/spm_prototype_training_20261022/USAGE.md)。下一步是同编码器P3接入smoke，尚未启动；YOLO AP效用未测。旧S1覆盖结果保留，不再作为本轮构库前置。

**历史SPM-S1状态（2026-10-22）：** 用户授权服务器最小试验；SPM-S1的P0已完成226图标签/覆盖核验，4个推断来源组达到要求、低于预冻结8组，故停在训练前。特征前向、probe、YOLO和原型优化均0；属于数据支持不足，不是方法效用失败。见[完整报告](../../../../experiments/spm_yoloseg_20261021/REPORT_SPM_S1_20261022.md)。以下V1/V2/V3设计与“未运行”文字保留其历史范围；当前状态以本段、Study与ACTIVE_PLAN为准。
**登记日期：** 2026-10-21  
**定位：** 独立应用型 SCI 论文主线；目标是先完成可重复、可投稿的猪场 YOLO26-Seg 改进研究（优先满足毕业需求，期刊分区与录用均不预先保证）。  
**初始设计状态（历史）：** `research_draft / hypothesis_only / not_started`。2026-10-22 已补充 V2 详细设计与 V3 难点排序/冻结特征接触边界 Probe 草案；仍未渲染、训练、产生本线科学结果或冻结正式协议。 本地[基础实现方案](../../../../paper/manuscripts/spm_yoloseg/BASE_IMPLEMENTATION_SPEC.md)和配置草案已保留，包含RLE加载、P3检索、梯度与打包接口；它们不替代下方V3科学决策顺序。

## V1 至 V4 的核心科研问题

面对群养猪密集接触、部分遮挡和高度相似外观，是否能够从 **Blender 合成图像 + 公开/自采真实猪场数据**中形成一组**可冻结的可学习视觉原型**，并在不更换原生 YOLO26-Seg Mask 解码器的前提下，通过**条件检索 + 残差门控**改善实例分割？

核心可检验主张 H：在**同一份真实/合成训练数据、同样训练预算、相近可训练容量**的公平对照下，带来源约束的冻结原型记忆与条件检索，将在预定义的密集/遮挡子集改善可见区域 Mask 质量，同时不显著损伤全量表现和推理效率。

竞争解释：收益可能仅来自增加 Blender 样本、增加参数、普通特征重校准、额外监督或域随机化；需要相同训练信息与机制消融区分。

## V4 数据、监督、训练与评价补充（2026-10-23）

新增 [V4 可执行候选协议](../../../../paper/manuscripts/spm_yoloseg/DATA_SUPERVISION_TRAINING_EVALUATION_PROTOCOL_V4_20261023.md)，把数据准备（真实实例/接触 crop、cutout 合成与 Blender）、C/B/I/D 标签、监督目标、冻结原型与 P3 注入、数据/模型融合训练、公平消融和真实测试评价细化为可审计合同。**V4初版为候选设计；用户随后明确授权按V4快速实施，本轮以快速Study启动前冻结合同和真实Run为准，不重判旧Probe Gate。**固定 K=32 仅是 pilot 对照，动态 K 研究仍保留。

## 历史决策优先级（V3，2026-10-22）

**最高优先级科学问题：** 真正失败的猪-猪接触区域（同类不同实例）在原生 YOLO26-Seg 冻结 P3/P4 中，是否存在跨真实视频/猪栏可稳定读取、超出简单像素边缘的分离信息？仅前景/背景原型可能无法提供实例归属线索。

**研究顺序：** P0 真实数据簇/可见 Mask/原生模型审计及接触错例账本 → 小规模冻结 P3/P4 接触边界 probe（需独立训练授权）+ Blender 64–128 张标签质量 smoke → 仅在有效时进行同几何的原始/普通高频/真实高频比较 → 按证据建立跨域冻结原型 → P3 注入与等容量消融。外观原型、几何原型及二者组合由上述证据决定，**不预设 V3 或 memory 一定有收益**。相关操作合同见上方入口。

## 暂定方法（未获实验验证）

1. **Blender 数据：** 不同姿态、视角、光照、密度与遮挡的 RGB、bbox、实例可见 Mask；另外导出完整（amodal）Mask/深度/姿态作为合成可观测标签，*第一版不改变原生可见 Mask 任务定义*。
2. **联合表征：** 初版只用基线YOLO26-Seg的P3 Neck特征与轻量投影器，结合真实/合成对应训练视图构建128维归一化空间；P4留作扩展。
3. **自适应原型发现：** 从猪体区域与背景区域分别平衡采样特征，采用 DP-means 风格的新建/更新/合并/剔除策略确定有效原型数量 K；K 不代表预设部位数量。
4. **冻结原型库：** 保存向量、投影/归一化定义、来源统计与哈希；确保 YOLO 检索端和构库端特征空间一致。
5. **注入 YOLO26-Seg：** 在 Neck P3（第一版）/P4（第二版）到 Segment26 头的输入处，按局部相似度 Top-k 查原型并做门控残差；P5 与原生 Proto26/掩码系数链路先保持原样。
6. **分阶段微调：** 零残差初始化后只训练融合与原生头；保留原生one2many/one2one训练与O2O推理，冻结构库所用Neck、BN统计及投影器。解冻查询上游需要另立设计。

**严格区分：** 本主线的「知识原型」是跨图像持久记忆向量；原 YOLO-Seg 的 `Proto26` 是每张图独立计算的空间 Mask 基底，两者不替换也不混称。第一版只做实例检测/可见 Mask；跟踪、身份识别、世界模型和完整轮廓推断均不纳入最低可行研究。

## 研究范围与独立性

- 这是一条**独立论文主线**，不是 `neighbor_sensitive_p3` 的续跑，也不是原 Prototype→Coefficient 失败诊断实验的改名；旧 Run、冻结 Gate、超参、结论不得迁移充当本线新结果。
- 可引用旧研究中「密集遮挡导致合并/漏检」等观察作为研究动机，但本线必须有自己的可追溯基线与独立结果，不能继承旧 AP 表格作为新论文结果。
- 本线当前优先动作由[总控](../../../ACTIVE_PLAN.md)登记；旧原型—系数线的协议、Run和科学结论保持独立。

## 文档入口

| 文件 | 内容 |
|---|---|
| [V4 数据—监督—训练—评价协议](../../../../paper/manuscripts/spm_yoloseg/DATA_SUPERVISION_TRAINING_EVALUATION_PROTOCOL_V4_20261023.md) | **新增：** 真实/cutout/Blender 数据与标签、接触边界监督、冻结原型、P3 融合、消融和评价；仅候选、未开训 |
| [V3 重点难点与科学决策树](../../../../paper/manuscripts/spm_yoloseg/PRIORITIES_RISKS_AND_DECISION_TREE_V3.md) | **优先核验 P3/P4 同类接触边界信息**，再决定外观/几何原型；难点排序、证据门槛和停止条件 |
| [接触边界 Probe 草案](../../../../experiments/spm_yoloseg_20261021/CONTACT_BOUNDARY_PROBE_DRAFT.md) | 可见 GT 接触 C/B/I 标签、冻结 P3/P4 线性/非线性 probe、数据对照/统计与三出口（未冻结、未运行） |
| [基础实现方案 V0.1](../../../../paper/manuscripts/spm_yoloseg/BASE_IMPLEMENTATION_SPEC.md) | **基础工程接口**：必须实现的数据、模型、梯度、存取接口及完成条件；先接通链路，再讨论效果 |
| [基础配置草案](../../../../experiments/spm_yoloseg_20261021/BASE_CONFIG_DRAFT.json) | 已选择的设计默认值与待核路径/预算分开；执行开关关闭 |
| [V2 详细实施蓝图](../../../../paper/manuscripts/spm_yoloseg/IMPLEMENTATION_BLUEPRINT_V2.md) | **场景统计→Blender 几何与标注→真实纹理/高频→共同原型空间→YOLO 注入**，每步算法、输入输出与QA |
| [真实频率迁移规范](../../../../paper/manuscripts/spm_yoloseg/REAL_FREQUENCY_TRANSFER_SPEC.md) | 类别条件真实 Pig/背景高频、UV/PBR 纹理、FFT 幅度/相位、相机退化、真实数据防泄漏及 QA |
| [V2 Pilot 与消融设计](../../../../experiments/spm_yoloseg_20261021/PILOT_AND_ABLATION_V2.md) | 同场景同标签 V0/V2/V3 高频对照、真实/合成原型独立贡献、数据/计算公平性与阶段 Gate |
| [研究章程](../../../../paper/manuscripts/spm_yoloseg/RESEARCH_CHARTER.md) | 机制假说、数学接口、Blender→真实原型→YOLO26-Seg 的完整技术路线及失败出口 |
| 本线规则（原路径：`paper/manuscripts/spm_yoloseg/RESEARCH_RULES.md`；[服务器归档回执](../../maintenance/nonrun_file_compliance_20261104T093520Z/ARCHIVE_RECEIPT.json)） | 数据、权重、评价、冻结、训练授权和因果归因规范 |
| [实验设计草案](../../../../experiments/spm_yoloseg_20261021/PROTOCOL_DRAFT.md) | P0–P5 分阶段 Gate、最低实验矩阵与报告要求；**未冻结且未开训** |
| [Study 状态](../../../../experiments/spm_yoloseg_20261021/study.json) | 计划、执行、科学结果和产物状态 |
| [研究线总索引](../../../research_lines.json) | 独立主线正式登记 |

## 历史建议（执行顺序已由V4快速合同取代）

**SPM-S1之后：** 先对已有原train全部标注做零训练接触来源普查，确定能否满足来源覆盖，再决定是否另冻接触覆盖采样合同；不足则优先补独立真实接触来源，暂停纯外观原型大库。本轮固定226图probe配置已停止，不降Gate、不换图续跑。下段P0建议为此前设计记录，不表示重跑已完成阶段。

**原P0建议（历史设计）：** 只读核实数据/权重/Ultralytics 版本与可见掩码评价口径，确认视频/猪栏无泄漏分组及原生 P3/P4 提取接口，形成**猪-猪接触 C / 猪-背景 B / 内部 I** 的真实训练域标签与难例账本。P0 完成后先评估冻结特征接触边界可读性（Probe 草案，未授权开训），再决定是否大规模 Blender 渲染、构建原型与正式训练。 同时清点本地已登记3D资产并补齐[基础配置草案](../../../../experiments/spm_yoloseg_20261021/BASE_CONFIG_DRAFT.json)中的路径、来源与哈希；基础接口设计保留，当前不启动渲染或训练。
