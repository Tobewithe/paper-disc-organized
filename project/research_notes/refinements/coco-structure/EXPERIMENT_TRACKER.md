# COCO结构诊断执行表

2026-09-13 S077 COMPLETE：冻结COCO train2017 readout缓存，fit/transfer 7811/1815目标，七个裁切尺度0.8—1.4，按官方解码计算逐实例IoU。transfer固定1.2倍裁切整体比1.0下降2.265点，高ICI组下降3.163点；逐实例oracle最佳尺度相对1.0整体提升1.047点，高组1.493点，低组1.022点。fit-only原始边界外响应启发式在transfer整体仅+0.115点，高组−0.078点，低组+0.137点。结论：裁切支持参与失败且高组有略大oracle余量，但固定外扩有害，当前无GT统计量未形成稳定方法；无训练、无完整AP。报告CROP_SUPPORT_PROBE_20260913/REPORT.md，原始结果experiments/coco_clean_20260911/diagnostics/crop_support_probe_20260913。

2026-09-13 S076 COMPLETE：GT教师系数残差可预测性探针完成。137个有效实例、5个图像外推折、4种推理可见特征组合×3个岭强度；教师平均IoU机会+15.508点，但最佳共享预测器（coeff_box，λ=.1）留出平均IoU−9.391点，高/低组−11.358/−7.337，残差余弦≤.039，正向比例≤28.5%。停止当前简单共享残差回归配方；教师上限仍是诊断，不是方法或完整AP。报告TEACHER_DELTA_PREDICTABILITY_20260913.md，原始结果experiments/coco_clean_20260913_teacher_delta_probe。

2026-09-13 S075 COMPLETE：预测交集支持的候选归属门控复验完成。120张拟合图/20张独立transfer图/112对候选/3种子/15 epoch；训练和推理均使用同类预测框交集，GT仅生成训练标签。每种子7,413,270次软更新，留出BCE仅下降约1.06e-5—1.18e-5；1,804个transfer预测三种子均0像素二值改变，Mask AP/AP50/AP75与基线完全一致。停止该具体公式，不能把GT归属诊断上限写成自动方法。报告JOINT_SUPPORT_OWNERSHIP_PILOT_20260913.md，原始结果experiments/coco_clean_20260913_joint_support_ownership_120_20。

2026-09-13 S074 COMPLETE：固定S070张量的原型/预测框/官方解码不变，140个有效实例（高低密度×3先验分层）进行32维系数残差的像素留出检验，4.4秒；GT只用于诊断标签，无网络训练/AP。拟合一半自身/非自身640网格像素后，在另一半留出像素高失败组AUC提高11.97/16.16点（低组14.95/8.61），完整输入网格split IoU提高高16.40/24.28、低19.01/16.89；mask-good控制仅高6.06、低4.70。说明固定原型上存在可迁移的实例响应读出机会，不能称原型无表达；高低均有收益，不能称密集专属。实际队列来自S070冻结条件样本，GT solver不是方法。报告FIXED_PROTOTYPE_SPLIT_RESULTS_20260913.md，原始目录diagnostics/fixed_prototype_split_20260913_v5，保留v1-v4失败目录；下一步是冻结独立训练/测试图检验共享预测器能否学习响应修正，禁止重复GT系数oracle。

2026-09-13 S057 COMPLETE：同权重原生one-to-one全val5000图/36335普通GT，330.515秒，555714预测。MaskAP43.686vs43.698、BoxAP52.295vs52.554；高E(4)R75 55.231vs52.132，+3.100[2.543,3.670]，低−高差距9.151vs8.609，差+0.542[-0.152,1.160]；高P90召回12.684vs13.096。原生路径未消除密集缺口，增加全分数召回未变成AP/P90收益。每图head/backend真one-to-one及JPEG哈希通过；加载该资产初始end2endFalse，不据此归责用户配置。报告NATIVE_ONE2ONE_RESULTS_20260913.md，全部本轮进程完成、无训练/自动化修改。下一方法须区分可信竞争者/像素归属并保护自身，不能原样重做已失败硬竞争/S013排序/旧邻居MLP。

2026-09-13 S057 RUNNING：本地RTX5060Ti原生one-to-one路径，同官方权重/8.4.143隔离运行时/640/rectFalse/conf.001/正常crop，全val5000图。显式end2end=True，每图验证实际head/backend标志，JPEG与S036哈希一致。比较原one-to-many+NMS的完整预测集合，不称只干预NMS。PID33732，diagnostics/native_one2one_fullval_20260913，eval_native_one2one.py；无训练或参数搜索。

2026-09-13 S056 COMPLETE：9臂全val5000图/36335GT/446097槽位，385.641秒。只对固定5575框好mask差GT做同类/背景/异类/遗漏/全FP/完整GT修复；其他候选保持。GT同类删除AP+0.378，高E(4)R75+5.035[4.582,5.488]点，低−高差距8.609→3.583；GT背景删除AP+2.709，但差距+0.261[-0.444,1.042]。自动分数/中心竞争AP−1.047/−5.354，高R75−2.698/−8.421。合成分区/竞争验证、S048像素复算、官方基线AP重放通过。S055错误表面/重复子型撤回，主拓扑保留。报告MASK_ERROR_AP_RESULTS_20260913.md；无训练，进程已完成。

2026-09-13 S056 RUNNING：本地CPU复用5,000图冻结预测，固定S048一对一归属。六种GT错误修复诊断与两种纯预测竞争规则（分数优先/框归一化中心距离）交给官方COCOeval重算。GT修复只改5,575个同槽位框好mask差；自动规则在全部原预测上执行、不访问GT；所有框/分数/类别/槽位保留。100图准备通过，全量PID27888，diagnostics/mask_error_ap_20260913。更正S055：组件预测并集会误计邻居自己的正确输出，旧错误表面与重复子型撤回；高组M/X中546/2073个Mask75已成功。详情见S055报告更正。正式CCL未恢复，下一动作按当前进程完成情况决定。

2026-09-13 S055 COMPLETE：基于冻结的COCO val2017最终预测缓存完成5,000图/36,335普通GT关系图分类，无新增前向或训练。全候选口径GT关系为C9932/I521/L1180/S227/O14383/M94/X7809/MISS2189；score≥0.10口径为C17929/I983/L3498/S371/O3993/M917/X2070/MISS6574。E(4)分组在score≥0.10下高组M+X 21.93%、低组1.64%，高组M/X主要错误表面为同类邻居。分数敏感性显示全候选低分尾部会放大O/X，故主关系必须和候选口径、错误表面、阶段证据分离；MISS不能解释成无raw候选。阶段字段统一标记UNRESOLVED_FINAL_CACHE_ONLY。报告[COCO_RELATION_TAXONOMY_RESULTS_20260913.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/COCO_RELATION_TAXONOMY_RESULTS_20260913.md)，脚本[分类脚本](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/classify_coco_relation_graph.py)。事实记录更新至S055；无训练和定时任务修改。

2026-09-13 S054 COMPLETE：固定S053的15个同类邻居误报目标完成候选归属响应诊断，75条区域记录，68.859秒；15/15找到原图真实邻居候选。leak像素目标候选响应胜率59.7%；GT选邻居的诊断竞争使Mask IoU +4.755[0.494,8.772]点，12/15提高，邻居误报−15.204[−25.727,−6.637]自身面积点。无输入编辑/训练；这是诊断上限，不是自动方法/AP。主线收窄为可学习的候选间实例归属竞争。报告[CANDIDATE_OWNERSHIP_RESULTS_20260913.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md)。

2026-09-12 S053 COMPLETE：P3混合格/邻居独占格拆分15/15目标、780条记录、32.734秒。原c0/b0固定，每图M/E/B等k=1—4格、三冻结几何子集平均，两填充各seed0。独占格实际插入IoU纹理/颜色+0.474[−0.060,1.120]/+0.715[−0.043,1.611]点，恢复−1.345[−2.368,−0.488]/−1.056[−2.132,−0.163]；相对背景正反向区间排零。混合−独占四比较均跨零，不判区域主导；独占格仍有重叠感受野，等格数不等范数。71原产物哈希/390自替换/120旧掩码重放通过，无排除或训练。停止扩大格/层盲扫，下一候选为原图错误像素的目标/邻居归属响应及普通竞争读出对照，尚未执行。报告[CELL_PARTITION_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md)。事实至S053，全部进程完成，无定时任务修改。

2026-09-12 S052 COMPLETE：S051全部44例完成原型分支5层×3区域局部插入/恢复，5632条记录，194.281秒；固定旧系数/原框、纹理与颜色仅各seed0。同类误报15例P3编辑位置插入IoU +1.925[0.189,3.854]/+2.357[0.556,4.542]点，逆向恢复−2.467[−4.032,−1.027]/−2.581[−4.520,−0.950]，邻居−背景对照区间亦排零。所有15例P3编辑格与目标格直接重叠，平均占编辑格30.445%，只定位可逆传播路径，未定位自然错误起源或新方法。自身遗漏整组融合目标区域尚不一致；保留正反例。v1原加载模块CPU引用失败、无干预结果；v2改用AutoBackend实际GPU副本，99哈希/220自替换/176端点对照通过。下一项仅候选为同15例P3混合格/独占格拆分，尚未启动。报告[FEATURE_SWAP_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md)。事实至S052，全部本轮进程完成；无训练和定时任务修改。

2026-09-12 S051 COMPLETE：错误类型×大小配额的局部邻居干预完成44/44（同类误报15、自身遗漏15、背景14，小中大不替补），107.187秒；原型/系数输出空间复算1584臂、14.594秒。两填充各3seed正常流程稳定Mask75恢复同类3例、自身2例。自身遗漏只换原型相对原IoU纹理+2.541[0.650,4.777]、颜色+1.347[−2.115,4.510]；临邻自身净补回+2.259[0.404,5.372]/+2.783[0.636,6.031]自身面积点，但未编辑邻居错误亦增加。系数未单独解释主要个体恢复，原型主因/容量不足未确认；背景控制损伤、填充敏感性、正反例保留。下一可判别项为原输入下预定层特征注入与逆向恢复，本轮未启动；不重复S051、不训练CCL。报告[SUBTYPE_NEIGHBOR_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md)。事实至S051；全部进程完成，无定时任务修改。

2026-09-12 S050 COMPLETE：复用S049全部63图c/P，25.516秒解码2268输出，无新模型前向/训练。32主失败中29小目标、30背景误报/2自身遗漏、0同类邻居主导，故不代表粘连失败池。单换原型邻居相对原完整IoU纹理/颜色−0.054/−0.132点，背景控制−1.455/−2.147点：此前相对优势主要含对照损伤。仍有局部临邻自身边界净补回+0.569[0.198,1.005]/+0.715[0.306,1.179]自身面积点，但完整IoU未改善。两填充均相对控制>1点原型12例、均<−1点9例，保留正反例。下一项须按同类误报/自身遗漏及大小固定组成，候选清单已存但未启动新输入实验；不再重复同一四格。报告[PIXEL_FLOW_RESULTS_20260912.md](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md)。事实记录至S050，所有本轮进程完成，无自动化修改。

2026-09-12 S048/S049 COMPLETE：全COCO val5000图/36335普通GT完成同一预测框与mask失败清点；box好mask差5575，进一步初筛支持充分且没有其他保留好mask者4260，真实同类边界拥挤低重叠失败池511/397图、Box90为102。严格输入/输出对照64入选63可评，同类失败32例：固定原框，邻居减距离匹配背景控制，只换系数纹理/颜色IoU +0.305/−0.179点，只换原型+1.402/+2.015，各完整IoU区间与两路径差区间均跨零，尚不支持系数或原型为主因。原型路径增加自身覆盖，未稳定减少邻居/背景错误。全部进程完成，未启动训练或更上游模块注入；当前应复用保存c/P及像素区域定位具体亚型，勿重复已完成队列。报告[全量失败清点](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md)、[路径对照](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md)。事实记录已更新至S049。

2026-09-12 S047 COMPLETE / INPUT INTERVENTION, NO ROBUST RECOVERY：64张新抽取train图，同类近邻32/异类16/同类远邻16；两填充×3 seed、原source四格、完整正常匹配，78.703秒。自身原图与实际模型输入零变化，64零修改/NMS身份/decoder通过，842hash核验。完整背景对照下同类近邻固定框IoU差纹理+1.136[-.156,2.899]、颜色+1.218[-1.148,3.716]；主要goodBox/badMask仅4个完整对照，+1.673[-1.326,4.672]与-2.311[-8.865,1.774]，无稳定正常检出收益。部分案例改善但填充敏感，背景距离失配明显；高ICI可伴真实mask距离56px。报告NEIGHBOR_BACKGROUND_RESULTS_20260912.md，diagnostics/neighbor_background_20260912/ANALYSIS.json及CASE_PANEL.png。所有进程结束，无训练/自动化修改。更严格实际边界失败队列和模块替换尚未执行；不得重复S047。

事实入口：[RESEARCH_FACTS.md](C:/Dpan/codexproject/paper-disc/RESEARCH_FACTS.md)。该记录汇总当前有效事实、纠正过的前提、停止的具体配方及未执行方案；下方较早的“下一步”属于历史状态，不据此重复启动。

2026-09-12 S046 COMPLETE / SPARSE CUE FULL-MASK ADVANTAGE UNCONFIRMED：补录已完成结果，非新增训练。64个fit目标/47图、3保存S032头状态，38个共同提示可行目标（高28/非高10），684/684凸拟合收敛，9.453秒。高可行组自身＋邻居8点相对正负均衡8点：完整原框IoU +1.292[-.306,3.224]点、未用E点IoU +2.182[.771,3.986]；相对自身＋背景8点完整IoU +3.334[2.046,5.025]但背景错误 +3.387[1.171,5.872]点。未通过完整掩码相对普通提示的主要门槛，停止该单一8点配方；不宣称训练缺GT身份标签。70哈希、64坐标存档、原输出与S044精确重放通过，非独立GPU复现。报告SPARSE_IDENTITY_RESULTS_20260912.md，diagnostics/sparse_identity_probe_20260912/ANALYSIS.json。最新用户提出的邻居背景替换仍仅方案，未生成清单/编辑/推理；本次事实归档未恢复训练、未修改定时任务。

2026-09-12 S045 COMPLETE / GLOBAL PROTOTYPE SUMMARY NO_GO：49.437秒，旧1200fit中1187图7811匹配目标，三种子两图片折。base/真实原型/打乱原型18个PCG均收敛，纯实例增强6个未达1e-7、保留800上限。留折全体MSE base2.553、prototype3.315、shuffle3.320，真实−错配−.00535[-.07879,+.06710]，高采样IoU prototype72.510 vs base74.077，−1.568点区间排零。停止全图均值+二阶矩PCA32双线性交互配方，不上升到所有原型条件无效。7650目标同图留一残差公共修正，高1020目标MSE2.443→3.288，仅14.71%改善；置换对照3.613但都不如不修正，空间支持外推/极值保留。旧上游头已见全部fit，交叉拟合只限新增映射；不是独立新图/AP。报告PROTOTYPE_CONDITION_RESULTS_20260912.md，diagnostics/prototype_residual_condition_20260912/ANALYSIS.json。所有本轮进程完成，CCL/端到端继续暂停。下一窄问题为少量自身/邻居身份提示是否能兑现实例GT读出机会，须随机点/背景/错配提示同预算控制，尚未启动。

2026-09-12 S043/S044 COMPLETE / FIXED LINEAR GAP, REPEATABLE ORACLE COMPONENT：S0437811fit三状态四固定表征29.829秒，12/12凸二次PCG真残差<1e−7、max目标差界1.731e−10。保存头MSE2.278→固定hidden最后层2.098(约−7.94%)→adaptedhidden2.004；KL/IoU口径不同，不能断言全网输入缺失。S044预选64fit目标47图A/B/E坐标互斥、等预算k23–512/66–512，384fit全收敛4.781秒：高teacher修正cos.815、E二值分歧4.36%，完整IoU A/B+9.405/+8.784、均增自身降邻居/背景。采样变化存在但不是已证明的全部瓶颈。报告CONVEX_SHARED_READOUT_RESULTS及TEACHER_REPEATABILITY_RESULTS_20260912.md；所有进程完成，正式CCL暂停。下一有增量候选为图像原型基底条件与全局读出映射适配，必须区别S026P+h像素MLP，先残差/条件诊断，不直接当新方法；尚未启动。

2026-09-12 S041/S042 COMPLETE / SIMPLE DISTILLATION NO_GO, FIT OPTIMIZATION UNRESOLVED：S041三臂三种子135checkpoint，solver高R7555.257vsdirect55.369差−.112[-.678,.518]点，AP+.069无CI，P90高同17.450；覆盖↓邻居↓的收缩取舍。停止混合KL版。S042仅fit7811/3种子48.437秒，混合KL减少约5.24%、pureAdam7.03%、finitefullLBFGS20.24%，采样IoU81.545→82.388；不是完整/AP。H/T起点梯度cos−.0265/.848/.336，无一致抵消，LBFGS全100迭代触预算且非平稳，不得判输入信息缺失/容量上限。报告SOLVER_RESPONSE_RESULTS及TEACHER_FITTING_RESULTS_20260912.md，diagnostics/solver_response_20260912与teacher_fitting_20260912。所有进程完成。下一可判别项为固定中间表征下凸共享最后层teacher拟合，不是调更大头/迭代追分，尚未启动；正式CCL/端到端暂停。

2026-09-12 S041 RUNNING / SOLVER RESPONSE LEARNABILITY：沿用1200fit/300已探索transfer，7811fit目标×3保存S032头GT教师。direct/self_response/solver_response三臂×三种子，各从相同保存权重、Adam状态、样本顺序继续15轮，每轮checkpoint；主干/原型/框/分数冻结。teacher仅fitGT，原512点S040正则凸求解，概率KL权重1/T1，禁止按效果筛教师。训练与正常官方评价在本地运行，diagnostics/solver_response_20260912/progress.json与日志为准，不重复启动。未有任务结论，筛查不当新确认集。

2026-09-12 S040 COMPLETE / REGULARIZED SOLVER OPPORTUNITY, SPATIAL SPECIFICITY UNCONFIRMED：同128目标97图×3保存头×5臂1920凸拟合，34.437秒，无共享训练。transfer高32原IoU72.482，全局32GT解85.589(+13.106[8.852,17.822])，覆盖+3.559/邻居−10.653/背景−13.614点，IoU75由21/20/20→26/26/26；不是任务R75/AP。空间128为87.038，但相对同维全局非线性+.820[-.358,1.966]点，未通过完整控制门槛；非高背景退化亦保留。1920全收敛，global系数范数比中位2.451/最大13.178，非硬有界完美教师。135hash/128坐标、baselineIoU与零decoder重放通过。停止扩空间基或投影。下一项PLANNED为train-only正则教师响应vs同输入同容量同预算继续像素监督的可学习性，三种子、正常原框、不能直接称创新。报告SPATIAL_CONTROL_RESULTS_20260912.md，diagnostics/readout_spatial_control_20260912/ANALYSIS.json。

2026-09-12 S039 COMPLETE / LOCAL SELECTIVITY, NO TASK RECOVERY：128固定目标/97图×3保存头，6方向×2幅度、25.328秒，无训练。transfer高32未用像素自身BCE增量+.001045→+.00007565、邻居−.001774→−.002183，但原框完整IoU仅+.01897[-.02087,.04601]点、未优于常数，全部方向两幅度均0救回/0损伤IoU75；相对普通邻居方向背景错误+.13632点。停止投影版本扩展/步长调分，不外推所有有限优化失败。下一问题为输出空间控制粒度，同P比较全局/局部系数及同维全局非线性、纯空间偏置；尚未运行。报告SELECTIVE_DIRECTION_RESULTS_20260912.md，diagnostics/readout_selective_direction_20260912/ANALYSIS.json。

2026-09-12 15:52 S038 COMPLETE / SUPPORT PRESENT, LOCAL COUPLING：160hashfit+300transfer/2998匹配，128预选目标×3保存头384梯度、64.27秒，无训练。fit高154目标512点中自身245.79/邻居131.60/背景105.51；无支持内有邻居却完全漏采者。fit高136有错误目标支持外错误均值19.018[14.525,23.769]%，多数支持内但实际不同错误采样覆盖17.375%。transfer仅类似几何不是已监督。fit高31有效梯度目标自身/邻居共享cos−.901，背景−.955且非高也负，不能宣称密集特有异常。邻居临时dc在采样logitRMS.01使邻居BCE−.001658、自身+.001213，全31目标3种子自身BCE都升；正像素变化很小、非覆盖/AP。总方向768步完整loss全降。1183fitY/2998支持重放、梯度sum误差<4e−8、8hash通过。报告SUPPORT_GRADIENT_RESULTS_20260912.md，diagnostics/readout_support_gradients_20260912/ANALYSIS.json。下一窄诊断为同128目标自身约束局部方向＋排重未用像素/完整mask，不重训或包装梯度投影创新；未启动。所有进程结束。

2026-09-12 15:01 S037 COMPLETE / EQUAL-COVERAGE NEIGHBOR ERROR：复用300探索train图、1815bbox50固定匹配、3保存S032头，44秒、无训练。原COCO固定IoU7260项精确重放。90%覆盖可达高237，同目标四输出TP数严格相同；新系数邻居FP/自身面积11.705→13.575%，+1.870[.791,3.138]点，背景16.210→15.572、IoU75.065→74.781未改善。邻居FPR225目标+1.499[.327,2.677]；80/95覆盖也增邻居。全GT高298中三头均失败132，32无bbox50、100有；100中82三头穷举阈值均未达640IoU75，其中80原框支持充分。GT删邻居重扫救23、删背景35，可重叠，非AP。类别×大小×框IoU共同高142/非高412仍有邻居面积负担差，条件邻居FPR未更高，不能宣称更难辨别或密集特有因果。当前方向是同覆盖下归属误判/暴露负担，不是已证实训练原因。EQUAL_COVERAGE_RESULTS_20260912.md；diagnostics/readout_equal_coverage_20260912/ANALYSIS.json。下一步仅诊断当前正增益头实际512采样支持及同目标区域梯度，区分于旧S022跨目标冲突，不重复S013排序损失/S026MLP。所有进程结束，正式CCL继续暂停。

2026-09-12 14:13 S036 COMPLETE / GENERAL GAIN, DENSE GAP WIDER：完整val5000图/36335普通GT（低20981中10109高5245）、11输出、保存3种子。原/实例偏置/系数AP43.698/44.439/44.996、高R7550.829/52.113/52.514，系数高+1.684[1.097,2.277]、AP+1.298；非高−高差距9.846→10.992，扩大1.146[.478,1.800]。条件偏置高+1.284排零，但差距未缩小；系数减偏置高+.400[-.082,.882]。P90高13.365→14.865系数，非高26.475→28.486，仍扩大差距点估计。逐实例校准超过全局常数有证据，但非创新。高组2454GT在三系数输出均失败，不能全归mask。v1空检测dtype错误保留，v2仅零index转int64、4809hash重用+191新图；5000shard/35回执/11套全GT身份通过。所有进程完成，无新训练。结果FROZEN_FULL_VAL_RESULTS_20260912.md及diagnostics/frozen_readouts_fullval_20260912_v2/ANALYSIS.json。下一步在train诊断剩余失败、等覆盖下排序与邻居错误；不能重复S021/S034、继续val调参或恢复CCL。当前只有强基线，密集方法主线未确认。

2026-09-12 14:04 S036 RESUMED v2：原进程13:56在第4,810张（image560371）空检测时停止；官方NMS给空source index默认float，追踪索引报错，4,809张完整输出已保存。v1源码/协议/失败记录完整保留，v2仅对零元素index转int64；非空索引、候选、阈值、权重、评价均不变。4,809逐图SHA验证后硬链接复用，继续剩余191张，不删除空检测图或GT。新输出diagnostics/frozen_readouts_fullval_20260912_v2，包装PID5296/Python19596；CONTINUATION_RECEIPT.json记录复用，progress.json为准。尚无全量评价结论。

2026-09-12 13:28 S036 RUNNING / FROZEN FULL VAL：已从远端取回全部5,000张val2017，仅在本地RTX5060Ti执行新前向。冻结原模型、S032三种子系数头、S035三种子条件偏置/常数及单个全批量常数，共11输出；无新训练。前50图完成、source索引/原decoder重放通过。全部GT官方AP与Mask75匹配后按单实例低/中/高ICI；加入同全局Precision≥90% operating point的分组Recall，不按每组独立调阈值，也不是部署校准。输出diagnostics/frozen_readouts_fullval_20260912，启动包装PID17092、Python16120，进度文件为准；原子逐图保存支持接续。val此前已探索，不能称盲测。FROZEN_FULL_VAL_PROTOCOL_20260912.md。正式CCL/端到端继续暂停；不得重复启动正在运行任务。

2026-09-12 S035 COMPLETE / RANK-PRESERVING CALIBRATION BASELINE：1200fit/7811目标、300transfer/2002GT，同rawCOCO512像素BCE+Dice，常数和实例偏置各3种子15轮90checkpoint，230.94秒；另train-onlyFP64LBFGS常数参考，复用S032系数头。原/常数/条件偏置/系数AP51.902/52.376/52.689/53.313，高R75 51.678/52.685/54.474/54.586。条件偏置高组+2.796[1.010,4.817]、差距缩小.782[-1.198,3.024]未确认；相对系数AP−.624、高R75−.112跨零，不能称等价。条件偏置保持像素排序但高组邻居−1.360、背景−3.402、覆盖−1.693点；不能称归属能力改善。常数参考b−.290788，AP52.349高53.020，不同预算单列。全部原/系数JSON精确重放，90checkpoint齐全，无后台训练。下一步冻结这些已有控制做COCOval确认，不再在同300图调参或堆偏置形式；未启动。READOUT_BIAS_CONTROL_RESULTS_20260912.md。

2026-09-12 S034 COMPLETE / AREA TRANSFER REPRODUCES GAIN：300图/2002GT、3保存种子、73695槽位，148.49秒，无新训练。借用S032新头预测K，在原框内取原logit稳定topK；全部640面积相等、原模型/新头JSON精确重放。原/新/等面积原排序AP51.902/53.313/53.219，高R75 51.678/54.586/55.257；新减控制AP+.094无区间、高R75−.671[-2.726,1.270]，未显示排序优势。高固定IoU差+.016跨零，新增同类邻居错误+.814[.501,1.162]、背景−.728[-1.019,-.447]。控制相对原高R75+3.579[1.360,6.070]，差距缩小1.545[-.923,4.287]未确认；不能作密集胜利。控制仍需新头K、非独立便宜方法。原图恢复面积仅86.26%记录相同，绝对相对差均值.069%，极端值披露。下一步可做同rawCOCO监督标量偏置vs常数/系数强控制，尚未启动。READOUT_EQUAL_AREA_RESULTS_20260912.md；无后台训练。

2026-09-12 S033 COMPLETE / BOUNDARY LABELS, BROADER SHRINKAGE：复用S032全300图，2002GT标签+1815固定匹配目标三种子像素流，124.41秒，无训练。原生独立相对原COCO额外正标签均值12.533%自身GT、缺失.021%；96.946%分歧像素位于2输入像素边界内。类别×大小控制后高减非高额外标签+.135[-2.120,2.459]点，未见密集专属。原COCO头81.763%匹配目标面积缩小；高组删自身2.424/添.897、删背景4.631/添.544点，净覆盖−1.527、背景−4.088完全重放S032，邻居未改善。高组被删自身仅46.044%在2像素内、63.012%在4像素内；删除不只是窄边带且仅8.762%直接对应原生标签多余位置。下一窄诊断是S032新头预测面积K控制下原logit topK排序是否复制AP增益；尚未运行，不重复旧去仿射。v1空标签文件失败保留，v2四张0普通GT图按回执处理，7输出空间重放误差0。READOUT_BOUNDARY_FLOW_RESULTS_20260912.md；无后台训练。

2026-09-12 S032 COMPLETE / POSITIVE ORDINARY BASELINE, DENSE GAP NOT CONFIRMED：同全局头/1200fit+300transfer/三标签×三种子×15轮，135checkpoint，259.72秒。原COCO栅格监督AP53.313、高R75 54.586，原模型51.902/51.678；高组+2.908[.241,5.537]，差距缩小.307[-2.473,3.309]未确认。原生overlap51.771/49.553、原生独立51.897/49.441，故取消overlap不能解释共享收益。全部原生组参数/预测精确重放S026。高组背景错误−4.088点但覆盖−1.527、邻居错误+.244区间跨零，不能宣称邻居分离；属于已探索train2017，非val最终/非新方法。现以原COCO监督全局读出作为强基线，下一步定位标签边界差异与前景/背景收缩代价，不再以退化头衬托提升。无后台训练；报告SHARED_LABEL_CONTROL_RESULTS_20260912.md。

2026-09-12 S030—S031 COMPLETE / LABEL-CONFLICT DIAGNOSIS；S032 RUNNING：同160固定目标，正则λ.01将系数范数中位从314倍限制到2.014倍，未用像素IoU仍+8.18点，但完整原COCO仅+2.808点且区间跨零。随后只替换标签：原生overlap/原生独立/原COCO栅格求解IoU80.811/83.740/86.457%，原模型78.003%；独立−overlap+2.929[.937,5.737]，raw−独立+2.717[1.936,3.473]。4例overlap求解崩至3.75—21.94%，原生独立恢复90.13—94.05%；丢失真阳性95.18—99.67%在overlap标签中被否定。局部单因素证据不等于历史训练错误；独立标签是已有强基线，非创新。所有标签位置/原空间结果重放一致。S032同全局系数头1200/300图×三标签×三种子×15轮已在本地运行，以官方AP/高R75/差距验证共享训练，不恢复正式CCL。详见READOUT_REGULARIZATION_AND_LABELS_20260912.md。

2026-09-12 S027—S029 COMPLETE / CALIBRATION-LIKE FIT, NO_GO FOR REMOVAL：80fit+80transfer图968匹配目标，保存S026三种子重放，无新共享训练。完整头在fit的BCE+Dice0.392747→0.385368但原COCO IoU76.718→76.707%；Dice降/BCE升、未用像素二值IoU持平。160目标同512点GT辅助自由系数明显降低拟合损失，但60触上限、范数中位314倍、未用像素BCE恶化，不能当稳定教师。fit预测拟合的全局正仿射解释完整头约78%transfer未用位置残差平方能量（非因果占比）。固定投影后300图整图分量移除：剩余修正高R75 51.007，原51.678，差−0.671[-1.713,+0.143]；没有被校准遮住的可用空间提升证据。原始及完整预测JSON与S026精确一致。停止去仿射/同头扩展；若下一步做oracle教师，必须先有约束且未用像素有效，不能直接蒸馏极端系数。详见READOUT_FIT_AND_CALIBRATION_RESULTS_20260912.md；正式CCL不恢复。

2026-09-12 S026 COMPLETE / NO_GO：本地复用S024的1200/300原生缓存，完成全局BCE+Dice、标量像素对照、P(x)+h、P(x)+h+预测邻居、空间打乱P五组×三种子×15轮，共225个checkpoint；300图2002普通GT官方评价完成，约632.6秒。完整输入高R75 51.230，原模型51.678，差−0.447[-1.598,+0.563]；标量51.566，空间打乱51.230。完整输入AP51.921，原51.902，标量51.996；没有高组正增益或差距缩小。全局对照本轮退化至49.553，不能仅以超过它声明方法成功。原始官方任务精确重放，原mask像素xor实测0；未完整独立复验。全部组同512原生GT框支持采样像素、同优化预算、最终轮，无GT推理输入。停止扩大本版本；下一步只诊断实际监督像素支持与拟合/空间误差，不重复增加头。结果见RICH_PIXEL_READOUT_RESULTS_20260912.md和diagnostics/rich_pixel_readout_20260912。正式训练未恢复。

2026-09-12 心跳续作 COMPLETE / COMPOSITION CONTROL：复用S024的300图，类别×大小×原框IoU共同分层保留高175/非高523目标；高减非高固定IoU−3.24[−6.54,−0.60]点、同类邻居误分/自身GT+9.67[+5.48,+14.08]点。已匹配目标的官方失败率差−0.62[−8.49,+7.37]，不能声称完整召回差距由掩码解释。两套分层的共同支持不同，不能以差值变化作框的因果贡献。旧14维逐像素own/neighbor MLP已在9月11日做过，不能重复包装成新方法；下一增量为局部原型P(x)+候选h条件读出对比旧MLP和全局BCE+Dice，尚未启动。READOUT_COMPOSITION_CONTROL_20260912.md。统计v1—v4失败保留，v5显式einsum完成，未安装库或用KMP强制绕过。

2026-09-12 S023 COMPLETE、S024快速纯BCE版 COMPLETE / NO_GO：本地RTX5060Ti执行32拟合+64留出小样本，再1200拟合+300留出扩大对照，均三种子。小样本过拟合；扩大后正确3×3空间布局高R75为50.224，原模型51.678，差−1.454点[-3.410,+0.314]；非高−高差距从11.585扩大至13.450点。停止扩大此版本；未完成原计划BCE+Dice强对照，不能声称穷尽S024所有设计。S025错配上下文/全val条件实验未启动。详见READOUT_INPUT_PILOT_RESULTS和READOUT_INPUT_SCALE_RESULTS_20260912.md。

2026-09-12 后续错误定位 COMPLETE / DESCRIPTIVE：同300图2002普通GT，高298中144官方Mask75失败，112已有同类bbox50归属、32未匹配；总差距11.585点中9.884点来自“有框未恢复”状态发生率差，非因果归因。在112固定掩码上GT辅助删除邻居/背景/两者可使27/49/84个达到固定IoU75，仅2个裁切支持本身不足75。所有编辑依赖GT、不是官方Recall提升。全匹配高/非高组邻居暴露66.90%/5.47%，条件误判率20.78%/20.24%（差区间跨零），说明须区分暴露量与辨别难度；尚无组成控制。工作路线选择为自身/邻居/背景联合像素归属，逐像素修正只是下一候选，未开始训练、未宣称新颖。READOUT_FAILURE_ROUTE_20260912.md记录证据及下一对照；正式端到端CCL仍暂停。


2026-09-12 S022 COMPLETE / LIMITED INTERFERENCE：同887目标243图，原生640 mask-only冻结最后1×1系数头梯度诊断，32.65秒。前向缓存误差0，h→c重建及887方向有限差分通过（max3.47e-9）。同图同尺度总mask梯度使目标局部损失上升高39/215=18.14%、其余130/672=19.35%；同类邻居项可使总方向反转仅高13/215、其余15/672。不能将当前最后层冲突当多数失败/密集专属根因；亦不代表历史多batch优化。491/499hash通过，审计WARN、same-family/reused-reviewer/provisional，无实质漏洞。SHARED_HEAD_GRADIENT_RESULTS/AUDIT_20260912.md。网络无更新，GPU释放；下一步共享读出输入/可学习性控制尚未启动。

2026-09-12 S021 COMPLETE / CONDITIONAL READOUT GAP：原生640同887目标/243图/7臂，56.44秒，无网络训练。高组原始/正仿射/辅助阈值/自由32维IoU51.146/52.537/57.419/73.917，达到75数量0/11/18/120；其余57.925/61.341/63.917/73.211，0/84/111/350。自由减正仿射高+21.380[18.814,24.177]点，且自身vs同类邻居AUC（208合格）0.8522→0.9766，支持机会超出简单校准，未证明共享头失败根因。第一次因极端oracle系数的损失浮点重放差停止，失败记录保留；v2显式分析缩放/全图次序差并增加FP64和范数恢复对照，FP64仅3/887变最多1像素、75计数不变。249/257hash通过，审计WARN、same-family/reused-reviewer/provisional，fresh审计创建遇线程上限。READOUT_CALIBRATION_RESULTS/AUDIT_20260912.md。下一步共享头输入/梯度诊断尚未启动；正式训练继续暂停。

2026-09-12关键更正：读取实际官方yolo26m-seg.pt，SHA匹配，train_args已记录mask_ratio=1（640、overlap=True、MuSGD）。S017—S020的160网格为诊断配方，不能倒推官方权重原始低分辨率训练，也不能把clean-run微调的ratio4混为权重历史。已修正S016—S020报告与新颖性定位，PRETRAINED_WEIGHT_CORRECTION_20260912.md保存证据。正式mask_ratio训练未启动；后续创新应定位已有高分辨率监督后仍存在的失败，而不是把已有开关当新方法。

2026-09-12 S020 COMPLETE / CONDITIONAL DIAGNOSTIC：887目标/243图/6臂，701.46秒；实际val loader 3,298条普通ID保留、图像/mask/bbox/类别重放通过，12图与推理缓存图像有差异因此只作路径审计。原生Format在共同推理几何下拟合，原生框/面积160→640：高IoU67.161→73.917（+6.756[5.475,8.111]点）、其余66.166→73.211（+7.045[6.337,7.782]）；75数量79→120/215、210→350/672。旧S019的17.465/19.226点不等于原生流程的差距。5,322系数解码重放和887原生目标梯度通过；1,363次迭代触上限，有限oracle非AP。源码确认用户训练用stock loader且官方已有proto上采样监督，mask_ratio=1不能当方法创新。网络训练未启动，CCL队列仍暂停；用户先同意4vs1后追问新颖性，暂澄清贡献。详见NATIVE_LABEL_PIPELINE_RESULTS_20260912.md、MASK_RATIO_NOVELTY_SCOPE_20260912.md。有限语义审计完成：WARN、same-family/provisional，无fatal，独立复核hash/目标集合/聚合；未独立重做GPU求解。

2026-09-12 S019有限范围终审完成：GRID_SUPPORT_AUDIT_20260912.md/.json，WARN、same-family/provisional、无fatal。审计者独立核验887原COCO身份（155多片）、490回执/501导出hash、10组聚合和主对比；未独立重做优化/COCO解码。1352/3548求解触上限及插值顺序微小差异保留披露。

2026-09-12 S019 COMPLETE / DIAGNOSTIC SUPPORT：固定S018的887目标（高215/其余672）、243图，160/640网格×GT/预测框损失支持，559.19秒。高组A/B/C/D原COCO IoU63.385/66.832/80.850/86.077，达到75数量60/82/161/189；其余64.018/65.724/83.244/88.066，172/202/544/607。纯网格C−A增量高+17.465[14.842,20.336]、其余+19.226[17.852,20.721]点；支持D−C高+5.227、其余+4.822点。主要收益尺度相关，不支持密集专属；高小目标C−A+26.964，中大+6.335。高组覆盖改善而邻居/背景均减少。四臂同纯BCE/GT面积归一化/初始化/参数尺度/预算/最终预测框，无网络训练、无IoU状态选择；非方法AP。640多数达到120迭代上限，有限achieved oracle。3,548项保存系数重解码与梯度幅值通过，一例空支持保留；10项插值顺序1—4像素差，无75判断变化。远端501文件本地hash通过。GRID_SUPPORT_RESULTS_20260912.md记录结果和下一步标签/插值分离计划；正式CCL训练继续暂停。

2026-09-12 S016—S018审计完成：DENSITY_SUPERVISION_AUDIT_20260912.md/.json，WARN、same-family/provisional，无fatal。空间域7例zero-valid面积排除已显式补录；执行远端官方源码已归档、hash匹配。保留一个未用TracedLoss包装器以保持执行脚本原hash。S018仅补充审查，非独立全量重算。结果报告 DENSITY_SUPERVISION_RESULTS_20260912.md，S019网格×支持域分解计划已写、尚未启动。

2026-09-12 S016 COMPLETE / CONDITIONAL SUPPORT：同一探索300图内，将高组317个无合格候选失败GT的S015程序原样用于其余组1,021 GT（ICI≤0.5，非独立低密度图片集）。高/其余可拟合254/742，固定最终匹配237/707；其余组5个crowd排除后无有效像素单列。原始COCO固定目标IoU：高50.197→85.312，其余56.795→85.412；超阈值控制的增益27.834/20.072点，未调整交互+7.763点，2,000次图像聚类bootstrap CI[5.439,10.033]。共同类别×面积×原始IoU粗分层仅保留124/249目标，交互缩至约4.013点，无调整CI。仍是GT同图oracle，不是新方法AP、密集因果或独立验证。结果 low_density_readout1021_20260912；463回执文件本地hash通过。

2026-09-12 S017 COMPLETE / DIAGNOSTIC SUPPORT：300图3,635 GT，精确缓存候选调用已安装官方TaskAlignedAssigner及mask loss，只求系数输出梯度，网络无更新。显式COCO原始annToMask独立mask160标签、无增强、overlap_mask=False，不能冒充历史训练复现。高组317无合格候选失败中237最终框匹配，215自身GT正样本/18非正/4别的GT；多数失败非“当前分配缺监督”。高组215自身GT正样本中197有邻居错误，错误像素在GT框外的实例平均比例25.680%，在监督proto空间支持外17.890%，不能全归因框外未监督。最初新前向smoke超过既有2e-4重放阈值，失败保留；正式使用精确旧缓存，未放宽阈值。304回执文件本地hash通过，语义审计进行中。

2026-09-12 S018 COMPLETE / DIAGNOSTIC SUPPORT：沿用S017实际mask BCE（160网格、GT框裁切/面积归一化），冻结P/最终预测框，对当前自身GT正样本且可拟合的887个失败目标做临时系数LBFGS，243图135.98秒。高215：原IoU51.146→官方目标63.395（59达到75）→旧full-input oracle85.388（187达到75）；其余672：57.925→64.014（171）→86.292（579）。高55/215、其余233/672在官方损失降低时最终IoU反而降低；不能把旧oracle缺口全归系数头。两种oracle同时改变多因素，尚非监督失配单因素因果或密集专属结论。v1空GT裁切失败保留，v2一例保持原系数并计入；97目标触120迭代上限。官方单目标与完整梯度方向最小cos0.999949，490回执文件hash通过。S016—S018详见 DENSITY_SUPERVISION_RESULTS_20260912.md；下一步S019同目标2×2网格/支持域分解，尚未启动。模型训练继续暂停。

2026-09-12 S015 COMPLETE / DIAGNOSTIC SUPPORT：锁定S014高组317个无合格mask候选GT（152图），36无raw框50、9类别断开、18分数断开、17有阈值后框但最终未匹配、237最终框已匹配。固定P/预测框拟合254目标，原始COCO RLE域IoU≥0.75数量：原0、阈值29、自由32维系数220、系数＋偏置220；原最终框237子组纯系数204成功。crowd排除域对应9/36/228/228，不能混口径。全输入支持GT同图oracle，无空间留出、非方法AP、非表达上界。26有效域剩余失败增加240＋240迭代后无新增原始域75成功，仍不可宣称原型容量不足。原系数范数恢复后254掩码0像素差异。原始域/汇总/逐图文件已归档，318远端文件清单本地hash通过；详见 NO_CANDIDATE_READOUT_RESULTS_20260912.md。正式训练与CCL继续停止；下一步追实际训练assigner监督、当前条件输入与共享头可拟合性。

2026-09-12 后续：S014 COMPLETE / DIAGNOSTIC SUPPORT，无训练，详见 CANDIDATE_LINEAGE_RESULTS_20260912.md。300图、3,635 GT，阈值后全部候选掩码解码；NMS顺序、候选身份、最终掩码、原官方Mask75/Box75逐GT命中均重放一致。高组50个GT在NMS失去合格mask候选，最佳候选的压制关系42同GT/8歧义/0明确跨GT。GT辅助只换系数：Mask AP40.452→41.599，高组R75 53.046→56.472；高组IoU+2.758点、覆盖+1.574点、同类邻居错误−4.608点、背景−1.127点。所有替代保持预测槽位和分数并重跑整图评价，重复来源及FP已记录。GT选候选不可当作方法成绩；下一命题是实例读出质量与保留机制的失配，未证实可学习性、密集专属性、新颖性。已下载主结果与全部逐图数据，远端最终1,819文件清单本地hash通过；正式训练继续停止。历史路线设计保留在 FAILURE_TO_MAINLINE_ROADMAP_20260912.md。

日期：2026-09-11，第一轮结果更新。没有重启正式CCL训练；TODO不是已开始运行。结果见 STRUCTURE_RESULTS_20260911.md。

| ID | 实验 | 数据 | 状态 | 输出/门槛 |
|---|---|---|---|---|
| S000 | 运行时预测路径见证 | COCO val图1000 | DONE | one-to-many＋NMS，FP32重放coeff/box/proto均0误差 |
| S001 | 全候选索引追踪冒烟 | val4图＋train32图 | DONE | main300亦全部重放通过；首次train缺图失败保留，v2按实际文件池选择 |
| S002 | 全候选失败分解 | 300图、3,635 GT、1,174相邻对 | DONE first pass | 原始source index、非独占候选可用率及官方同IoU框/掩码匹配；未完成组成控制及NMS干预 |
| S003 | 精简逐层空间可读出性 | 同一300图，16层，66,960探针行 | PARTIAL | 已有32维投影、空间块、坐标和打乱标签；原维数、等容量空间及类别/面积/距离控制待补 |
| S004 | 系数选择与原型可表达性 | 同一176/167个合格高相邻对 | PARTIAL | 已有固定P oracle、实际系数、响应差的配对AUC；共同覆盖/背景及等预算干预未完成 |
| S005 | 共同支持与外扩环归属 | S002固定图；COCO CCL权重未就绪 | CONDITIONAL | 不同模型不直接交换P与c；无干净CCL权重则只做官方模型诊断 |
| S006 | 一个对应机制的方法验证 | 预先冻结协议＋三种子 | NOT STARTED | 机制定位后另定方法与正式训练预算 |
| S007 | 预测邻居相对响应、固定面积重排 | 160 train开发＋600 val锁定评价 | COMPLETE / NO-GO | 非零干预减少邻居错误但增背景，三类选择均为0；val为相同预测一致性检查，不是非零方法效果比较 |
| S008 | 既有开发干预像素流向 | 同一160 train及0.25/0.5强度 | COMPLETE / descriptive | 复算自身/邻居/背景新增删除，解释错误转移；没有重选参数或修改val |
| S009 | 自身/邻居/背景三区域空间留出诊断 | 原train160＋哈希val200；val高组正常30/外扩46目标 | COMPLETE / conditional oracle | 差分损伤背景判别；原型GT辅助读出提高排序但覆盖下降；非可部署增益，非密集专属证据 |
| S010 | 共同响应与差异响应的代数分解 | 与S009完全相同像素和折；11,300条主读数零误差重放 | COMPLETE / post-hoc | m含前景信息、d偏向归属；解释性诊断，不作独立确认或CCL因果结论 |
| S011 | 跨图线性响应读出与训练集覆盖校准 | 新train320拟合＋160校准＋哈希val400；采样种子0/1/2 | COMPLETE / NO-GO | 高组正常覆盖−0.043pp、邻居−0.210pp、背景+0.079pp、IoU−0.018pp；AP+0.031点，R75/对恢复持平。外扩小收益不能替代正常门槛 |

最新计划见同目录EXPERIMENT_PLAN.md。旧全量AP、阈值和CCL结果仍在各自目录，不覆写。

S012（DONE / PARTIAL SUPPORT）：原始错误空间距离及空间交叉拟合读出机会诊断。固定160 train＋400 val、5,194普通GT、3,534完整合格读出目标。val高组原型oracle相对阈值的留出IoU +2.965点[1.685,4.279]，局部相对原型 +0.932点[0.157,1.734]；同类邻居内部侵入存在。但空间过拟合明显、GT oracle依赖及非等容量限制仍在，不能转成方法成绩或系数相似因果结论。详见FAILURE_LOCALITY_RESULTS_20260911.md。正式训练未重启。

S013（COMPLETE / NO_GO，2026-09-12）：重叠候选的像素归属监督验证，已获用户“开始”授权。协议 ownership_ranking_protocol_20260912.json 在主结果前冻结：全部原始 train ID 哈希选 1,200 图，8,322 个固定框匹配且有支持的目标；6 个同容量全局系数残差头损失组 × 3 种子 × 15 轮，每组 2,235 次更新，保留 270 个 checkpoint。统一官方解码，不使用提案旧版邻居相减。500 张哈希 val、3,616 普通 GT（高组 607）完成评价。排序＋覆盖相对 BCE＋Dice：AP −0.218 点，高 R75 +0.165 点 [−0.812,+1.209]，高组相邻对 −0.522 点，覆盖 +2.327 点、邻居错误 +1.844 点、背景 +2.207 点；高组空间 IoU +0.182 点、区间跨零。相对同像素 pair-BCE＋覆盖的高组 IoU −0.045 点。当前版本不扩大 5,000 图、不调参补救；旧端到端队列继续停止。原始、逐种子、均值和配对区间见 OWNERSHIP_RANKING_RESULTS_20260912.md。普通 BCE＋Dice 相对原模型 AP +1.457 点，不转称排序贡献。

S013 缓存 v1 在输入采样数值重放上触发 2e-4 阈值（观测 0.000213146），保留失败目录且未用于主训练。v2 将原型按官方双线性插值扩展到输入网格后直接索引，1,200 图最大绝对误差 7.6293945e-6，64.41 秒完成。v2 未放宽容忍阈值。训练梯度、零残差恢复、配对梯度方向及各损失臂单步更新已通过 smoke witness；完整方法有效性待评估。

结构主推理162.07秒，不包含实现、统计和审查耗时；不是3次端到端训练。结构审计已返回WARN、无fatal，见 STRUCTURE_AUDIT_20260911.md/.json。相对归属最小干预已完成，见 RELATIVE_OWNERSHIP_RESULTS_20260911.md；因开发无收益停止该实现，不启动训练。新干预的独立审查单列 RELATIVE_OWNERSHIP_AUDIT_20260911.md/.json。

三区域及响应分解见THREE_REGION_RESULTS_20260911.md。正式训练继续暂停；原型可读出性不等于无需GT的新方法。下一门槛是跨图、相同覆盖下的前景与归属联合利用，并超过自身校准控制；不因oracle AUC高而自动立项训练。

S011结果见CROSSIMAGE_RESPONSE_RESULTS_20260911.md。跨图覆盖校准基本迁移，真实邻居较打乱控制减少邻居错误，但正常域未提高IoU，预定门槛NO-GO。不要重启该线性方案或CCL正式训练。若继续，应定位原失败像素的实例/空间条件差异，而非在同一val调强度。

S013 独立审计完成：OWNERSHIP_RANKING_AUDIT_20260912.md/.json，WARN、未发现 fatal、same-family/provisional；原始 COCO JSON 内容与官方 AP 未由本地审计者从头重算，不能称为完整独立复现。确定性归档校验通过，NO_GO 结论保持。全部训练与评价进程已结束。
# S058：分层失败分类修订（2026-09-13）

状态：COMPLETE。复用 S048/S056 全量 COCO val2017 固定槽位和关系连接结果，无模型前向、无训练。将最终槽位、Box IoU、框对 GT 支持、Mask IoU、错误表面、关系拓扑和 E(4) 拆成独立字段；输出 36,335 行互斥 `failure_scope`。主要掩码读出池为 4,417 个框好且支持充分但 Mask IoU<0.75 的实例；另有 1,158 个框好但支持不足，不能直接归因于系数/原型。E(4) zero/positive-low/positive-high 下该池占比 11.91/12.38/12.64%，未形成单凭分箱的拥挤因果结论。详见 [LAYERED_FAILURE_TAXONOMY_RESULTS_20260913](LAYERED_FAILURE_TAXONOMY_RESULTS_20260913.md) 和 `diagnostics/layered_failure_taxonomy_20260913_v2`。

# S059：分类层与修复潜力交叉（2026-09-13）

状态：COMPLETE。将 S058 的 36,335 个互斥分类与 S056 的 5,575 个固定槽位 oracle 一对一连接。4,417 个框好且支持充分但掩码差实例中，背景删除/自身补全/同类删除达到 Mask75 的条件比例为 65.20%/42.20%/12.04%；1,158 个框好但支持不足实例为 51.81%/61.14%/5.27%。结果用于机会大小排序，不是可部署方法或 AP 加性分解。详见 `TAXONOMY_REPAIR_POTENTIAL_RESULTS_20260913.md`。
# S060：受守恒候选间归属门控 pilot（2026-09-13）

状态：NO-GO / 实现级诊断。1,200张训练缓存、3个种子，pair holdout BCE下降至约0.5764；20张transfer冒烟图三个种子均0个改变像素，AP与原模型完全一致。训练GT独占有向样本与推理候选共同支持区域分布错位；300图实现因逐候选支持构造过慢停止。停止该具体配方，不将其写成方法收益。详见 `SELECTIVE_OWNERSHIP_GATE_PILOT_20260913.md`。

# S061：支持保持背景风险读出（2026-09-13）

状态：NO-GO。冻结 73D 全局系数头、3 seed、15 epoch、300 张 transfer 图；背景误报约 0.17→0.09，但自身覆盖约 0.91→0.85，Mask AP 51.406–51.689 低于原 51.902，高组 R75 51.007–51.678，低−高差扩大至 12.843–13.036。停止该背景加权配方，确认需避免整体收缩。详见 `SUPPORT_RISK_READOUT_PILOT_RESULTS_20260913.md`。

# S062：前景逐像素信任域读出（2026-09-13）

状态：稳定强基线 / 非密集主线。与 S061 相同预算；逐前景像素约束原始 logit 不低于 `z0−0.25`，避免覆盖率塌缩。Mask AP 52.772–52.805、全部 R75 62.887–62.987、高组 R75 52.685、低组 64.671–64.789；低−高差 11.987–12.104，大于原 11.585。自身覆盖约 0.91、背景约 0.16。保留为支持保持控制，停止将其包装为拥挤专用方法。详见 `MARGIN_PRESERVE_READOUT_PILOT_RESULTS_20260913.md`。

# S063：同类边界暴露加权读出（2026-09-13）

状态：NO-GO / 方向性线索。`1+3E4` 训练权重与 raw COCO 控制各 3 seed，在 300 张 transfer 图评估。density-weighted Mask AP 53.233–53.286，未超过 raw 53.256–53.372；高组 R75 54.362–55.034，低−高差 10.870–11.248，但逐种子高组 bootstrap 均跨零，低组 seed1 明显下降。停止当前配方，不扩完整 val；边界拥挤权重暂不能成为方法主线。详见 `DENSITY_WEIGHTED_READOUT_PILOT_RESULTS_20260913.md`。

# S064：分类 v5 候选身份复核（2026-09-13）

状态：COMPLETE / 诊断纠错。修正 v4 默认分支把框差/支持不足但掩码好的实例错误计入 `mask_success` 的问题；36,335 个 GT 重新互斥分为 8 个状态：框差/掩码差 6,584，框差/掩码好 1,106，支持不足/掩码差 1,158，支持不足/掩码好 1,132，候选身份线索25，掩码读出候选失败4,392，掩码成功18,234，无最终槽位3,704。脚本语法检查通过。详见 `IMPROVED_FAILURE_TAXONOMY_RESULTS_20260913.md`。


# 2026-09-13 S064更正、S065/S066收尾、S067—S069完成

S064旧4,392干净池及v4多计成功的解释撤回：严格池4,260；v5 evidence_tier默认成功错误已在新表显式修复。原始回执不覆盖。S063/S065旧isin bootstrap区间撤回，重复采样计数保留的复算见RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md；S066 AP均值只比raw高0.024点，高R75+0.895的CI跨零。S061/S062亦有损失项缩放混杂，停止继续扫权重。

S067 COMPLETE：完整5,000图36,335 GT，五臂153秒，baseline parity通过。支持充分好框掩码理想修复AP+4.630，支持不足+0.864，框差掩码差+5.629；后者缩小E4差距7.459点。官方失败记账中该状态贡献7.449/8.609点差距，非因果比例。S068为确定性支持/候选再分层；S069 COMPLETE：同6,584槽位只做GT紧矩形输出裁切，全部原TP保留、原IoU逐项对齐，AP+1.659，高E4固定槽位恢复619/2292。事实支持从联合差状态继续区分粗范围、自身缺失与矩形内误报，不能直接认定系数或框为根因。完整报告FAILURE_DIMENSION_AP_RESULTS_20260913.md。


# S070—S073 COMPLETE（2026-09-13）

S070实际640解码GT框，原始RLE与源c逐项重放，7690槽位/2665图；全val AP+2.025，E4差距−1.725，高联合失败830/2292达到Mask75。剩余381原始uncropped自身响应不足75%，说明不能仅修框。初次stale anchors失败保留，v2使用deepcopied执行模型head，310.42秒完成。

S071固定48高/48低条件匹配目标，GT空间位置选内部源系数、等距离镜像、局部均值和预测内部点；高组原框内部−原IoU−2.751，均值/预测内点亦退化。S072等覆盖/排序未显示内部更好。S073当前TAL发现内部 donor31/48给自身而original45/48；二者都own30例仅+0.867点CI跨零。停止这些直接源点配方，不把关联当方法，也不否定所有空间建模。报告JOINT_DECODER_QUERY_RESULTS_20260913.md。无网络训练/环境改动/定时任务更改。

# S078：裁切敏感性与失败状态交叉（2026-09-13）

状态：COMPLETE / 诊断。复用 S077 冻结的 train2017 readout cache，9,626 个已配对目标、1,483 张图；逐实例重放官方 crop/decode 的 0.8--1.4 尺度，并与候选框 IoU、GT-box 支持和 native Mask IoU 交叉。未新增前向、训练或推理规则。判定阈值为候选 Box IoU>=0.50、native Mask IoU>=0.75、GT 像素在候选框内>=0.95。

全体配对目标中，6,097 (63.34%) 为框好且掩码好；970 (10.08%) 为框好但 GT 支持不足且掩码差；2,559 (26.58%) 为框好、GT 支持充分但掩码差。后者的 native IoU 0.611、裁切 oracle 平均增益 +2.751 个 IoU 点 [2.525,2.988]，固定外扩 20% 平均 -4.064 点 [-4.272,-3.832]。高 ICI 配对目标中对应比例为 48.78% 掩码好、15.52% 支持不足且掩码差、35.70% 支持充分但掩码差；支持充分掩码差组 oracle +2.768 点 [2.342,3.247]，固定外扩 20% -4.693 点 [-5.199,-4.212]。

这支持“框已覆盖目标但掩码读出仍失败”的真实失败类，也支持裁切尺度有可恢复余量；不支持固定外扩作为部署方法。由于 S077 cache 只保留已配对候选，box_bad_mask_bad 与 box_bad_mask_good 均为零，不能据此估计框漏检/坏框比例或全端到端 AP。GT 仅用于回溯标签与评价；当前无 GT 自动选择器在 transfer 高组未稳定有效。详见 experiments/coco_clean_20260911/diagnostics/crop_support_failure_cross_20260913/REPORT.md、SUMMARY.json、per_target.csv。

# S079：推理可见裁切尺度选择器（2026-09-13）

状态：COMPLETE / 诊断性 no-go。复用 S078 的 9,626 个已配对 train2017 目标，在 fit 7,811 个目标上以 GT 派生的各尺度 native IoU 训练每尺度 ExtraTrees 回归器；transfer 1,815 个目标只使用 raw 20% 边界阳性率、框内阳性率、边界/内部比、候选分数和预测类别选择 0.8--1.4 crop scale。该结果不是端到端 AP，也没有新增前向或训练模型。

transfer 基线 IoU@1.0 为 0.771219，选择器为 0.771343（+0.0124 IoU 点）。高 ICI 组 n=181 为 -0.1955 点，低 ICI 组 n=1,634 为 +0.0354 点；框好、支持充分但掩码差组 n=468 为 +0.7944 点，支持不足掩码差组 n=177 为 -0.6056 点，掩码好组 n=1,170 为 -0.2069 点。对应 transfer oracle 余量总体 +1.535、支持充分掩码差 +3.493 个 IoU 点。

结论：最优裁切方向依赖失败类型，现有推理可见特征不能稳定识别，不能把尺度选择器作为当前主方法；裁切保留为机制证据，后续主线应直接改善掩码读出或监督。复现脚本 `experiments/coco_clean_20260911/crop_scale_selector_probe_20260913.py`，报告 `experiments/coco_clean_20260911/diagnostics/crop_scale_selector_probe_20260913/REPORT.md`。

# S080：系数机会与固定原型限制归因（2026-09-13）

状态：COMPLETE / S074 复算；无新增前向、训练或 COCO AP。对 S074 的 140 个固定队列实例重算原始系数、固定原型 GT-assisted all-fit 与 IoU=.75 阈值归因。原始 IoU<.75 的 94 个实例中，`requires_target_pixel_recovery` n=48 仅 8 (16.7%) 达到 .75，40 (83.3%) 在最优系数下仍低于 .75；`sufficient_true_pixels_but_residual_false_pixels` n=46 有 25 (54.3%) 达到 .75，21 (45.7%) 仍受限。高/低 ICI 结构分别为 16.7%/54.2% 与 16.7%/54.5%，没有密集独有差异。该结果把主线拆为“系数读出有条件机会”与“固定 P/支持域仍不足”两类，但后者不能直接命名为原型因果；all-fit 增益是 GT oracle，不是方法/AP。源码 `experiments/coco_clean_20260911/coefficient_prototype_attribution_20260913.py`，报告 `experiments/coco_clean_20260911/diagnostics/coefficient_prototype_attribution_20260913/REPORT.md`。

# S081：COCO 图片级迁移的目标/邻居表示判别 probe（2026-09-13）

COMPLETE / 机制诊断；冻结 readout cache，1,500 张 train2017 图片、9,626 个匹配目标，fit/transfer 按图片留出，GT 仅定义像素归属标签。transfer 宏平均目标 AUC：自身 vs 同类邻居中 prototype 0.698、logit 0.937、逐通道 `P⊙c` 0.946；自身 vs 背景中 prototype 0.907、logit 0.951、`P⊙c` 0.951。位置、候选头输入 h、系数单独均约 0.50；加入 h 或坐标对 `P⊙c` 几乎无增益。结果说明实例条件化读出表示已含有较强像素归属信息，不支持“头缺空间坐标”或“系数单独不能区分”作为主根因；应继续查解码分辨率、阈值/校准、支持域和训练监督如何损失已有分离。AUC 不是 AP/方法收益，GT probe 也不是部署方法。源码 `experiments/coco_clean_20260911/representation_ownership_probe_20260913.py`，报告 `diagnostics/representation_ownership_probe_20260913/REPORT.md`。

# S082：表示判别的面积 × 失败状态交叉审计（2026-09-13）

状态：COMPLETE / 机制诊断。修正 S081 面积分层的 annotation_id 类型转换后，使用同一批 image-disjoint transfer 目标与 S078 失败标签，检查目标面积混杂及失败状态是否独立保留。无新增前向、训练或 AP；GT 只用于事后定义像素归属、面积档和失败分组。

面积对 `own_vs_same` 的 `proto_coeff` AUC 有明显影响：small (<1024) n=236 为 0.9237，medium (1024--9216) n=272 为 0.9434，large (>=9216) n=180 为 0.9779；`own_vs_background` 同样为 small n=611 0.9226、medium n=596 0.9619、large n=490 0.9730。因此面积必须作为协变量，不能把总体 AUC 差异直接归因于拥挤。

在同一面积档内，失败状态仍保持一致排序。`own_vs_same/proto_coeff`：large 中 mask_good n=155 为 0.99、support_low_mask_bad n=7 为 0.85、support_sufficient_mask_bad n=18 为 0.92；medium 分别为 n=185/0.98、n=29/0.82、n=58/0.88；small 分别为 n=90/0.98、n=40/0.82、n=106/0.92。`own_vs_background/proto_coeff` 也保持相同方向：large 0.98/0.86/0.91，medium 0.98/0.89/0.93，small 0.97/0.87/0.90（顺序均为 mask_good/support_low_mask_bad/support_sufficient_mask_bad）。

结合 S081，当前最稳妥的机制结论是：高 ICI 本身没有显著降低自身--同类邻居的判别能力（`proto_coeff` 高/低约 0.949/0.945），但高 ICI 下自身--背景判别下降（约 0.933 vs 0.953）；真正稳定的分叉来自 mask-bad 状态，且在面积匹配后仍存在。已有 `P(x)⊙c` 表示包含像素归属信息，失败更像支持域不足、低分辨率采样、阈值/校准、解码或监督目标导致的读出损失，而不是“系数相似”或“候选头缺坐标”单一根因。

证据文件：`experiments/coco_clean_20260911/diagnostics/representation_ownership_probe_20260913_area_stratified/target_auc_stratified.csv`、`STRATIFIED.csv`、`REPORT.md`。后续优先做同一候选在多 mask 解码分辨率及阈值下的可恢复性对照，再决定是否训练方法；不得把本轮 AUC 或面积内差异写成部署收益。

# S082：表示判别的面积 × 失败状态交叉审计（2026-09-13）

COMPLETE / 机制诊断；修正 S081 面积分层的 annotation_id 类型转换后，使用同一批 image-disjoint transfer 目标与 S078 失败标签，检查目标面积混杂及失败状态是否独立保留。面积影响 `own_vs_same/proto_coeff`（small/medium/large：0.9237/0.9434/0.9779），但同一面积档内 mask_good 的 AUC 仍高于 support_low_mask_bad 和 support_sufficient_mask_bad；高 ICI 本身未明显降低 own-vs-same，主要降低 own-vs-background。结论仍限于机制诊断，不是部署收益。证据：`experiments/coco_clean_20260911/diagnostics/representation_ownership_probe_20260913_area_stratified/target_auc_stratified.csv`、`STRATIFIED.csv`、`REPORT.md`。

# S083：冻结解码网格与 logit 阈值探针（2026-09-13）

状态：COMPLETE。脚本 `experiments/coco_clean_20260911/decode_grid_threshold_probe_20260913.py`；输出 `experiments/coco_clean_20260911/diagnostics/decode_grid_threshold_probe_20260913/`。复用 1,483 张图、9,626 个 box-good 候选，官方 `input640_t0` 对 S078 逐目标 IoU 复现最大误差 0。transfer n=1,815：官方 IoU 0.771219，统一 logit threshold=0.5 为 0.774435，post-interpolation 1280 为 0.775765；全体 n=9,626 对应 0.808850/0.811139/0.812028。transfer 失败分组中，support-sufficient mask-bad n=468 在 threshold=0.5 下 +0.0153 IoU，support-low mask-bad n=177 为 -0.0134，mask-good n=1,170 为 +0.0009。结论：解码校准/采样是部分损失源，但统一阈值不稳健，仍不是方法或端到端 AP 证据；后续主线应寻找可部署的失败状态估计或训练监督。

# S086：实例边界归属机制端到端训练（2026-09-14）

正式队列已启动：本地 COCO pilot1000 train2017、1,576 张 dense val、官方 yolo26m-seg.pt、640 分辨率、mask_ratio=1、AMP、15 轮、每轮 checkpoint。三臂为 baseline、`self_out`、`self_out_neighbor`，各 3 个随机种子，单 GPU 顺序运行。8 图真实标签 smoke 已完成并通过完整训练/验证链路；正式任务从 baseline_s0 开始，结果待完成。脚本 `experiments/coco_clean_20260911/train_boundary_ownership.py`，日志 `experiments/coco_clean_20260911/runs/boundary_ownership_20260914/queue_logs/`。
