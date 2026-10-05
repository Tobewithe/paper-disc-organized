# Experiment Tracker

2026-09-13 S054 COMPLETE：固定S053的15个同类邻居误报目标完成候选归属响应诊断，75条区域记录，68.859秒；15/15找到原图真实邻居候选。leak像素目标候选响应胜率59.7%；GT选邻居的诊断竞争使Mask IoU +4.755[0.494,8.772]点，12/15提高，邻居误报−15.204[−25.727,−6.637]自身面积点。无输入编辑/训练；这是诊断上限，不是自动方法/AP。主线收窄为可学习的候选间实例归属竞争。报告[CANDIDATE_OWNERSHIP_RESULTS_20260913.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md)。

2026-09-12 S053 COMPLETE：P3混合格/邻居独占格拆分15/15目标、780条记录、32.734秒。原c0/b0固定，每图M/E/B等k=1—4格、三冻结几何子集平均，两填充各seed0。独占格实际插入IoU纹理/颜色+0.474[−0.060,1.120]/+0.715[−0.043,1.611]点，恢复−1.345[−2.368,−0.488]/−1.056[−2.132,−0.163]；相对背景正反向区间排零。混合−独占四比较均跨零，不判区域主导；独占格仍有重叠感受野，等格数不等范数。71原产物哈希/390自替换/120旧掩码重放通过，无排除或训练。停止扩大格/层盲扫，下一候选为原图错误像素的目标/邻居归属响应及普通竞争读出对照，尚未执行。报告[CELL_PARTITION_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md)。事实至S053，全部进程完成，无定时任务修改。

2026-09-12 S052 COMPLETE：S051全部44例完成原型分支5层×3区域局部插入/恢复，5632条记录，194.281秒；固定旧系数/原框、纹理与颜色仅各seed0。同类误报15例P3编辑位置插入IoU +1.925[0.189,3.854]/+2.357[0.556,4.542]点，逆向恢复−2.467[−4.032,−1.027]/−2.581[−4.520,−0.950]，邻居−背景对照区间亦排零。所有15例P3编辑格与目标格直接重叠，平均占编辑格30.445%，只定位可逆传播路径，未定位自然错误起源或新方法。自身遗漏整组融合目标区域尚不一致；保留正反例。v1原加载模块CPU引用失败、无干预结果；v2改用AutoBackend实际GPU副本，99哈希/220自替换/176端点对照通过。下一项仅候选为同15例P3混合格/独占格拆分，尚未启动。报告[FEATURE_SWAP_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md)。事实至S052，全部本轮进程完成；无训练和定时任务修改。

2026-09-12 S051 COMPLETE：错误类型×大小配额的局部邻居干预完成44/44（同类误报15、自身遗漏15、背景14，小中大不替补），107.187秒；原型/系数输出空间复算1584臂、14.594秒。两填充各3seed正常流程稳定Mask75恢复同类3例、自身2例。自身遗漏只换原型相对原IoU纹理+2.541[0.650,4.777]、颜色+1.347[−2.115,4.510]；临邻自身净补回+2.259[0.404,5.372]/+2.783[0.636,6.031]自身面积点，但未编辑邻居错误亦增加。系数未单独解释主要个体恢复，原型主因/容量不足未确认；背景控制损伤、填充敏感性、正反例保留。下一可判别项为原输入下预定层特征注入与逆向恢复，本轮未启动；不重复S051、不训练CCL。报告[SUBTYPE_NEIGHBOR_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md)。事实至S051；全部进程完成，无定时任务修改。

2026-09-12 S050 COMPLETE：复用S049全部63图c/P，25.516秒解码2268输出，无新模型前向/训练。32主失败中29小目标、30背景误报/2自身遗漏、0同类邻居主导，故不代表粘连失败池。单换原型邻居相对原完整IoU纹理/颜色−0.054/−0.132点，背景控制−1.455/−2.147点：此前相对优势主要含对照损伤。仍有局部临邻自身边界净补回+0.569[0.198,1.005]/+0.715[0.306,1.179]自身面积点，但完整IoU未改善。两填充均相对控制>1点原型12例、均<−1点9例，保留正反例。下一项须按同类误报/自身遗漏及大小固定组成，候选清单已存但未启动新输入实验；不再重复同一四格。报告[PIXEL_FLOW_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md)。事实记录至S050，所有本轮进程完成，无自动化修改。

2026-09-12 S048/S049 COMPLETE：全COCO val5000图/36335普通GT完成同一预测框与mask失败清点；box好mask差5575，进一步初筛支持充分且没有其他保留好mask者4260，真实同类边界拥挤低重叠失败池511/397图、Box90为102。严格输入/输出对照64入选63可评，同类失败32例：固定原框，邻居减距离匹配背景控制，只换系数纹理/颜色IoU +0.305/−0.179点，只换原型+1.402/+2.015，各完整IoU区间与两路径差区间均跨零，尚不支持系数或原型为主因。原型路径增加自身覆盖，未稳定减少邻居/背景错误。全部进程完成，未启动训练或更上游模块注入；当前应复用保存c/P及像素区域定位具体亚型，勿重复已完成队列。报告[全量失败清点](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md)、[路径对照](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md)。事实记录已更新至S049。

2026-09-12 S047邻居背景替换64图已完成。固定原框、邻居减等面积背景对照，同类近邻纹理/颜色IoU约+1.14/+1.22点、区间跨零；主要框好mask差只有4个完整对照且填充方向不一致。原图/模型输入自身像素严格未变。发现高ICI与实际边界接近需分开；下一步须改进失败选样与位置对照，尚未扩量/模块替换。报告[NEIGHBOR_BACKGROUND_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NEIGHBOR_BACKGROUND_RESULTS_20260912.md)，已更新事实记录。无训练、无定时任务修改。

事实入口：[RESEARCH_FACTS.md](C:/Dpan/codexproject/paper-disc/RESEARCH_FACTS.md)。优先查阅当前结论、数据使用范围与已纠正前提，勿将较早的下一步计划当成当前待执行任务。

2026-09-12 S046已完成并补齐报告：64fit目标、38个提示共同可行、3保存头状态，684拟合全收敛。高可行28目标，自身＋邻居8点相对均衡提示完整IoU +1.292[-.306,3.224]，未用点改善但完整掩码优势未确认；相对背景提示有改善和背景错误取舍。详见[coco-structure/SPARSE_IDENTITY_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SPARSE_IDENTITY_RESULTS_20260912.md)。停止当前稀疏提示配方。邻居外观换背景尚仅方案；本次仅事实整理，无训练和定时任务修改。

2026-09-12 S045 COMPLETE / GLOBAL PROTOTYPE SUMMARY NO_GO：49.437秒，旧1200fit中1187图7811匹配目标，三种子两图片折。base/真实原型/打乱原型18个PCG均收敛，纯实例增强6个未达1e-7、保留800上限。留折全体MSE base2.553、prototype3.315、shuffle3.320，真实−错配−.00535[-.07879,+.06710]，高采样IoU prototype72.510 vs base74.077，−1.568点区间排零。停止全图均值+二阶矩PCA32双线性交互配方，不上升到所有原型条件无效。7650目标同图留一残差公共修正，高1020目标MSE2.443→3.288，仅14.71%改善；置换对照3.613但都不如不修正，空间支持外推/极值保留。旧上游头已见全部fit，交叉拟合只限新增映射；不是独立新图/AP。报告PROTOTYPE_CONDITION_RESULTS_20260912.md，diagnostics/prototype_residual_condition_20260912/ANALYSIS.json。所有本轮进程完成，CCL/端到端继续暂停。下一窄问题为少量自身/邻居身份提示是否能兑现实例GT读出机会，须随机点/背景/错配提示同预算控制，尚未启动。

2026-09-12 S043/S044 COMPLETE / FIXED LINEAR GAP, REPEATABLE ORACLE COMPONENT：S0437811fit三状态四固定表征29.829秒，12/12凸二次PCG真残差<1e−7、max目标差界1.731e−10。保存头MSE2.278→固定hidden最后层2.098(约−7.94%)→adaptedhidden2.004；KL/IoU口径不同，不能断言全网输入缺失。S044预选64fit目标47图A/B/E坐标互斥、等预算k23–512/66–512，384fit全收敛4.781秒：高teacher修正cos.815、E二值分歧4.36%，完整IoU A/B+9.405/+8.784、均增自身降邻居/背景。采样变化存在但不是已证明的全部瓶颈。报告CONVEX_SHARED_READOUT_RESULTS及TEACHER_REPEATABILITY_RESULTS_20260912.md；所有进程完成，正式CCL暂停。下一有增量候选为图像原型基底条件与全局读出映射适配，必须区别S026P+h像素MLP，先残差/条件诊断，不直接当新方法；尚未启动。

2026-09-12 S041/S042 COMPLETE / SIMPLE DISTILLATION NO_GO, FIT OPTIMIZATION UNRESOLVED：S041三臂三种子135checkpoint，solver高R7555.257vsdirect55.369差−.112[-.678,.518]点，AP+.069无CI，P90高同17.450；覆盖↓邻居↓的收缩取舍。停止混合KL版。S042仅fit7811/3种子48.437秒，混合KL减少约5.24%、pureAdam7.03%、finitefullLBFGS20.24%，采样IoU81.545→82.388；不是完整/AP。H/T起点梯度cos−.0265/.848/.336，无一致抵消，LBFGS全100迭代触预算且非平稳，不得判输入信息缺失/容量上限。报告SOLVER_RESPONSE_RESULTS及TEACHER_FITTING_RESULTS_20260912.md，diagnostics/solver_response_20260912与teacher_fitting_20260912。所有进程完成。下一可判别项为固定中间表征下凸共享最后层teacher拟合，不是调更大头/迭代追分，尚未启动；正式CCL/端到端暂停。

2026-09-12 S041 RUNNING / SOLVER RESPONSE LEARNABILITY：沿用1200fit/300已探索transfer，7811fit目标×3保存S032头GT教师。direct/self_response/solver_response三臂×三种子，各从相同保存权重、Adam状态、样本顺序继续15轮，每轮checkpoint；主干/原型/框/分数冻结。teacher仅fitGT，原512点S040正则凸求解，概率KL权重1/T1，禁止按效果筛教师。训练与正常官方评价在本地运行，diagnostics/solver_response_20260912/progress.json与日志为准，不重复启动。未有任务结论，筛查不当新确认集。

2026-09-12 S040 COMPLETE / REGULARIZED SOLVER OPPORTUNITY, SPATIAL SPECIFICITY UNCONFIRMED：同128目标97图×3保存头×5臂1920凸拟合，34.437秒，无共享训练。transfer高32原IoU72.482，全局32GT解85.589(+13.106[8.852,17.822])，覆盖+3.559/邻居−10.653/背景−13.614点，IoU75由21/20/20→26/26/26；不是任务R75/AP。空间128为87.038，但相对同维全局非线性+.820[-.358,1.966]点，未通过完整控制门槛；非高背景退化亦保留。1920全收敛，global系数范数比中位2.451/最大13.178，非硬有界完美教师。135hash/128坐标、baselineIoU与零decoder重放通过。停止扩空间基或投影。下一项PLANNED为train-only正则教师响应vs同输入同容量同预算继续像素监督的可学习性，三种子、正常原框、不能直接称创新。报告SPATIAL_CONTROL_RESULTS_20260912.md，diagnostics/readout_spatial_control_20260912/ANALYSIS.json。

2026-09-12 S039 COMPLETE / LOCAL SELECTIVITY, NO TASK RECOVERY：128固定目标/97图×3保存头，6方向×2幅度、25.328秒，无训练。transfer高32未用像素自身BCE增量+.001045→+.00007565、邻居−.001774→−.002183，但原框完整IoU仅+.01897[-.02087,.04601]点、未优于常数，全部方向两幅度均0救回/0损伤IoU75；相对普通邻居方向背景错误+.13632点。停止投影版本扩展/步长调分，不外推所有有限优化失败。下一问题为输出空间控制粒度，同P比较全局/局部系数及同维全局非线性、纯空间偏置；尚未运行。报告SELECTIVE_DIRECTION_RESULTS_20260912.md，diagnostics/readout_selective_direction_20260912/ANALYSIS.json。

2026-09-12 S038支持/同目标区域梯度完成，460图/2998目标、64秒。fit高邻居平均132/512点，非普遍漏采；临时邻居抑制降低邻居BCE却使自身BCE升，背景/非高也类似，不可称密集专属训练根因。下一步同状态自身约束方向在排重未用像素验证，尚未启动，无新训练。报告coco-structure/SUPPORT_GRADIENT_RESULTS_20260912.md。

2026-09-12 S037等自身覆盖诊断完成，300探索train/1815匹配、3保存种子44秒，无训练。90%覆盖高237目标，新系数邻居FP/自身面积增加1.870点CI排零；100有框残余高失败中82即使GT穷举阈值三种子均失败（640域），80非裁切支持不足。说明归属排序限制仍在，不等于密集特有训练因果。报告coco-structure/EQUAL_COVERAGE_RESULTS_20260912.md；下一步实际采样/同目标区域梯度诊断，未启动。

2026-09-12 S036完整val5000/36335GT/11输出已完成：原→系数AP43.698→44.996、高R7550.829→52.514，但非高−高差距9.846→10.992显著扩大。实例偏置44.439/52.113也未缩差距；确认普通读出收益，不是密集问题已解决。v1空检测错误修复后完整接续，未丢GT。详见coco-structure/FROZEN_FULL_VAL_RESULTS_20260912.md。全部进程结束，下一步仅train剩余机制诊断，正式CCL不恢复。

2026-09-12 14:04 S036空检测索引dtype错误已定位，保留v1失败源码/数据，v2仅修复零元素index转换，校验并复用4,809图继续191图；输出frozen_readouts_fullval_20260912_v2，未丢弃空图、未训练或调参。

2026-09-12 13:28 S036完整COCOval5000图冻结确认已在本地运行，11输出复用已有三种子，不训练/调参。逐图可接续，主报告coco-structure/FROZEN_FULL_VAL_PROTOCOL_20260912.md，真实状态见experiments/coco_clean_20260911/diagnostics/frozen_readouts_fullval_20260912/progress.json。不得重复启动。

2026-09-12 S035已完成三种子常数/实例偏置学习控制，90checkpoint。条件偏置保持像素排序，300探索图AP52.689、高R7554.474；高组改善但覆盖下降、差距缩小未确认，不是新方法。下一步冻结头进行COCOval确认，当前无后台训练。详见coco-structure/READOUT_BIAS_CONTROL_RESULTS_20260912.md。

2026-09-12 S034等预测面积排序对照完成，无训练。原像素排序借用新头面积AP53.219接近新头53.313、高R7555.257；尚无新增排序/实例归属优势，面积严格相等只在640输入域。下一步标量偏置强对照尚未启动，详见coco-structure/READOUT_EQUAL_AREA_RESULTS_20260912.md。

2026-09-12 S033边界/像素流诊断完成，无新训练。标签差异集中边界，S032读出呈普遍面积收缩并牺牲部分自身内部；尚非密集专属分离。见coco-structure/READOUT_BOUNDARY_FLOW_RESULTS_20260912.md。下一步为等预测面积排序控制，未运行。

2026-09-12 S032已完成：原COCO栅格监督同全局头三种子在300张已探索train2017图AP+1.411、高R75+2.908点；差距缩小尚不显著、邻居错误未减少。属于普通强基线与机制控制，不是新方法。所有后台训练结束；详见coco-structure/SHARED_LABEL_CONTROL_RESULTS_20260912.md。

2026-09-12 S030—S031已完成，定位到有限目标中的overlap监督/原COCO标签冲突；S032同全局头三标签、三种子本地学习对照运行中。不是独立标签开关创新、不是原训练有错。当前报告coco-structure/READOUT_REGULARIZATION_AND_LABELS_20260912.md；具体状态以coco-structure/EXPERIMENT_TRACKER.md顶部为准。

2026-09-12 S026局部原型P(x)+候选h的逐像素读出已完成1200/300图三种子、225checkpoint和官方评价，NO_GO；高R75未超过原始/标量对照、未缩小差距。见coco-structure/RICH_PIXEL_READOUT_RESULTS_20260912.md。原型可拟合机会与密集像素错误仍成立，但新的可部署方法尚未建立；正式CCL仍暂停。

2026-09-12 当前COCO进度见 [coco-structure/EXPERIMENT_TRACKER.md](coco-structure/EXPERIMENT_TRACKER.md)。S023本地32/64图拟合完成；S024快速纯BCE版1200/300图三种子完成，当前空间布局方案未过门槛，未完成原计划BCE+Dice方法强对照。新增全部GT失败分解、裁切支持和GT像素编辑，路线见 [READOUT_FAILURE_ROUTE_20260912.md](coco-structure/READOUT_FAILURE_ROUTE_20260912.md)。S025条件实验及新逐像素方法未启动；正式CCL训练仍暂停，后文为先前项目记录。


2026-09-07 project review status notice: R005c's newer receipt `experiments/yolo26_confidence_audit_20260907_v2/audit_results.json` reports `deterministic_verdict=fail` despite an inconsistent `acceptance_status=accepted`. Stage/source checks and 7 missing/7 unexpected confidence-set records require reconciliation. The historical row below is not a PASS. Full audit: `research-wiki/project_mechanism_goal_audit_20260907.md`. No prediction or original audit receipt was changed.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | create provenance-complete manifest | YOLO26 protocol audit | FaroPigSeg test + BamaPig2D eval; PigLife inventory | counts, paths, hashes, label semantics | MUST | DONE | 160 Faro + 332 Bama COCO polygon images; PigLife recovered at `derived/task05_v1` |
| R002 | M1 | baseline replay | YOLO26-seg | PigLife | AP, failure taxonomy | MUST | DONE-AUDIT | unified cache replay; COCO/image counts now verified 426/4,474; fresh forward remains optional |
| R003 | M1 | external replication | YOLO26-seg | PigLife | AP, failure taxonomy | MUST | DONE | recovered COCO test labels aligned with legacy GT table |
| R004 | M1 | external replication | YOLO26-seg | BamaPig2D eval | AP, failure taxonomy | MUST | DONE | COCO polygons; pose labels excluded; prediction validation valid |
| R005 | M2 | stage oracle | cached candidates + same-forward raw trace | PigLife + FaroPigSeg | candidate recall, stage availability, bootstrap | MUST | DONE-AUDIT | Full same-forward replay passed 586 images/8,327 masks with XOR=0. Failed-GT availability: PigLife raw/top-k/conf/final 390/390/373/373 of 394; Faro 746/698/553/553 of 797. This is availability, not causal ranking evidence. |
| R005b | M2 | strict mask quality and simultaneous allocation | IoU .50/.75; coverage/purity .75; all-GT exact matching | frozen PigLife + Faro traces | quality availability, matching conflicts, final RLE parity | MUST | DONE-AUDIT | Full 586-image/6,226-GT extraction: 193,413 edges, final-source XOR=0; 7,032 exact matching graphs independently reproduced, 20 condition-stage deficit records (not distinct GT). Pairwise admissibility does not ensure clean output-set relations; method acceptance remains pending. |
| R005c | M2 | isolate confidence retention and fixed target controls | K=300, conf .05/.01; strict GT-guided addition/removal/joint | frozen PigLife + Faro traces | per-dataset AP, all-GT recovery/regression, source/target parity | MUST | AUDIT-FAIL-UNRESOLVED | v2 original computation retained: 12 recovered/269 regressed; not integrity-cleared. Audit 20260907_v2 FAIL: stage schema, mapping, 7 confidence pairs unresolved; COCO matches alone do not pass audit. See `research-wiki/mechanism_evidence_ledger_20260907.md`. |
| R006 | M2 | perturbation | candidate removal/addition | PigLife + FaroPigSeg | achieved GT-guided recovery, taxonomy transitions | MUST | DONE-AUDIT | v2 exact matching and XOR guards passed; four-condition GT labels and 1,136 selected sources equal v1. ADDITION/REMOVAL/JOINT recovered 65/74/799; joint correct-GT regressions=1. Dataset recovery intervals audited. This does not meet the causal ranking/scorer Gate. |
| R005d | M2 | localize Faro strict single-mask quality gap | exact box support and source annotation parity | frozen Faro test, all 160 images / 1,752 GT | same-stage support/mask availability, original polygon raster parity | MUST | DONE-AUDIT | User chose this priority 2026-09-07. 307 raw strict-mask-absent failures split into 37 support-infeasible / 270 feasible. Original label raster parity passed; 3,484 final masks inside source support. Independent deterministic implementation reproduced 7,008 support rows and verified 489 hashes. Annotation semantics remain unverified; no model/method acceptance. |
| R007 | M2 | control | Mask-NMS sweep | all valid datasets | AP, residual O | MUST | DONE-AUDIT | existing fixed-cache control and uncertainty audit |
| R012 | M2 | all-split external baseline | frozen PigLife-only YOLO26 checkpoint | Faro train/val/test + Bama train/eval | mask AP/AP50/AP75/AR100, full GT taxonomy, historical parity | USER-AUTHORIZED | DONE-VERIFIED | 4,858 images / 27,745 GT / 45,664 predictions; 6,397 input hashes passed. Historical 492-image predictions and 2,900 GT classes unchanged. All-split mask AP: Faro .398061, Bama .641492. Bama has 77 byte-identical pairs with annotation differences; all records retained. Coordinator deterministic audit, not independent method acceptance. See `research-wiki/yolo26_external_full_20260907.md`. |
| R012b | M2 | duplicate-record sensitivity | frozen Bama predictions, fixed ID policies | all splits, min/max representative and singleton only | pooled mask AP, non-C failure rate | USER-CONTINUATION | DONE | v2 completed; AP .644483/.644492/.647532 versus all-record .641492, changes +.2990/+.3000/+.6040 points. No label choice based on model performance. v1 partial failure retained; original full run unchanged. |
| R012c | M2 | review O construct and threshold sensitivity | fixed masks, current classifier and audit-only rules | all Faro/Bama 27,745 GT | baseline parity, O transitions, sub-half part screen | USER-REVIEW | DONE | Core-only O is a strong one-to-many relation, not exhaustive fragmentation. Screen found 245 Faro / 29 Bama current MISS with high-quality unions of sub-half parts; not manually verified new labels. Seven actual-classifier counterexamples, 12 source hashes and all saved GT labels passed. Rules and Gates unchanged. |
| R008 | M3 | conditional intervention | two-layer MLP scorer or stage-matched branch | primary dataset | AP, dense failure, latency | MUST-GATED | TODO | run only after R005-R006 |
| R009 | M3 | deletion control | intervention removed | primary dataset | same metrics | MUST-GATED | TODO | isolate contribution |
| R010 | M3 | replication | final intervention | second dataset | same metrics | MUST-GATED | TODO | stop if protocol mismatch |
| R011 | M4 | robustness | threshold/bootstrap | held-out | intervals, sensitivity | NICE | TODO | appendix if budget permits |

R005, R005b and R006 now cover the full 426-image PigLife public-test and 160-image Faro test traces. The older R005 388/394 and 466/797 raw-good figures used a restricted, downsampled candidate search and remain historical only. Current exact-threshold availability is 390/394 and 746/797; strict coverage/purity final availability is 312/394 and 224/797. R008-R010 stay gated.
