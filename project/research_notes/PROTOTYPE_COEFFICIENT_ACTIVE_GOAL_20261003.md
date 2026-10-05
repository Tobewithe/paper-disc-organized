## 最新收尾：原生语义响应回投冻结筛查完成（2026-10-04）\n\n[native_semantic_projection_20261004_retry4](../../experiments/native_semantic_projection_20261004_retry4/REPORT.md)已完成并回传集合 COLLECT_91dcd3c8105246569d136b75cc997167。253张有效开发图、1816个官方one-to-one候选，原生未融合Proto26语义分支的预测类别响应经固定0.5融合和λ=.003完整640投影。PROJ相对A图片macro Mask IoU −29.2864pp，95%CI[−32.0156,−26.7988]，Mask75为12修复/620损伤；MIX相对A −29.5318pp。PROJ−MIX虽为+0.2454pp CI[+0.1215,+0.3785]，但两者均远低于A，不能算方法收益；PROJ−ROT_PROJ的+3.6364pp只能说明本冻结投影对空间排列敏感，不能证明实例归属。\n\n固定语义logit尺度/直接融合/投影配置停止，不扫权重、温度、gate或训练。当前排除的是原生预测类别语义响应按该固定方式可安全改善正常原图掩码；未排除经过独立尺度对齐机制或训练后语义—mask配合。四次执行尝试中的失败Run均保留，retry4完成后停止本配置。\n\n## 最新收尾：共享原型重复计权诊断完成（2026-10-04；优先于下方执行状态）

[prototype_multiplicity_audit_20261004](../../experiments/prototype_multiplicity_audit_20261004/REPORT.md)两Run已2026-10-03T21:04:32.893760+00:00完成，197文件SHA回传，集合COLLECT_762ed5a4c0e64230b32e266be752bbc4。128计划/127有效fit图、756原one候选、6931原many候选；零训练每图恢复，正式75.61秒。40图的GT候选数有差异；M官方候选均值与E同候选GT均值的原型参数梯度方向cos≈.981，E相对M向one诊断方向对齐仅+.01534 CI[+.00798,+.02388]，低于预设.05。

正常原图E−M全体macro IoU+0.0000667pp CI[−.009150,+.009781]，目标组+.01313pp CI[−.02723,+.05408]，无额外Mask75跨越；重计权没有实用信号。M/E自身GT更新相对A约+.298pp属于同图GT辅助局部改善，不是学到的无GT方法。101/127图通过预定三臂O目标一阶忠实度，未过90%，故角度→效用机制链记未充分检验，不宣布计权已排除。完整候选均保留，未删26张近似失真图。

当前配置停止，不扫步长/权重、不自动训练重计权或PCGrad。身份/官方loss梯度/原输出/BN冻结/每图恢复核验通过；末尾参数数目记录变量覆盖只影响元数据，已用INPUT_MANIFEST作AUDIT_CORRECTIONS补充，保留原Run不重跑。当前无在跑研究Run，下一项尚未登记。总Goal未完成；下一轮仍需真实历史、实现、数学与原始文献的独立缺口审查，不能把本配置补丁式续跑。本线程仅服务器，笔记本另一线程负责。

## 最新执行：共享原型的候选重复计权审计（2026-10-04）

已登记 [prototype_multiplicity_audit_20261004](../../experiments/prototype_multiplicity_audit_20261004/PROTOCOL.md)，Study STUDY_08833cebe2af47e09f0370f255e52bac，3080Ti服务器pipeline PID28148；先核原始PIPELINE_STATUS与实际PID，禁止重复启动。两个Run见RUN_IDS：smoke RUN_08b61f0b6821444799f9a26c3aeef652，evaluation RUN_a33a3889aafd4d49967ec723253713e0。

复用冻结fit前128计划图，零训练、每图恢复原权重。M官方many候选均值、E相同many候选按GT均值、O原one目标对P的诊断梯度：在同一原型卷积白名单上作等范数临时位移，原one系数/框/标签/候选固定，BN参数与buffers均冻结。不是完整官方训练梯度、学习方法或独立泛化；正常原图掩码评价只判断同图GT辅助局部效用。旧系数梯度、原型总梯度及新增监督位置不替代本对照。

2图smoke上限240秒；正式128图1200秒，1,000次整图bootstrap。方向主量cos(E,O)−cos(M,O)，正常原图E−M及修复/损伤共同判断。若局部线性忠实度不足或M/E机制未激活记为未充分检验；不删图、不扫步长、不自动追加重计权训练。17份实现/配置上传SHA已核验，原型复制、损失/梯度、原assignment和每图恢复由服务器预检。本线程仅服务器，笔记本另一线程负责。总Goal未完成。

# 当前目标与持续执行授权



## 当前入口（2026-10-04最新分工，优先于下方执行历史）



**执行分工：2026-10-04用户明确指定本会话只负责3080 Ti服务器 `connect.bjb2.seetacloud.com:33953`。笔记本 `ssh 28358lan` 由另一个线程独立负责，本会话不连接、启动、停止、评估或收集笔记本任务，也不修改其Study。可以只读项目中其已发布记录来去重；不向另一个线程发送任务（除非用户另行明确要求）。本机桌面仍只用于代码、记录和隐藏传输。此前统筹两台机器的历史授权按此分工收紧。**



## 最新收尾：冻结MIX确认未过，保留有限排序与候选收益（2026-10-04）

[flip_fusion_confirmation_20261004](../../experiments/flip_fusion_confirmation_20261004/REPORT.md)已2026-10-03T20:25:12.768689Z completed，两个Run共1607文件SHA核验回传，COLLECT_11a9ac4f88d44c018a952d99813ceab2。前瞻1536计划/1525有效train图、11034固定官方候选，正式690.18秒；方法开发外而非全项目或预训练未见。无训练、仅原A及原封不动MIX。

主要目标组817图2158候选macro IoU+.3330pp CI[−.0225,+.7117]，低于.5pp且CI跨零，未通过预定确认；全体+.0077pp CI[−.0541,+.0664]，146修复133损伤，净Mask75不确定。全体/目标AUC+.0726/+.2871pp区间正，保留可重复排序信息；目标组候选平均IoU+.2707pp CI[+.0758,+.4796]为正，但原成功候选IoU−.0434pp CI[−.0867,−.0017]为负，两者均为次指标且不能替换图片macro主判定。全体coverage−.1636pp、原成功coverage−.1966pp区间负，不能宣称已确认整体mask收益。实际保留且类正确TAL子集10224候选IoU+.0164pp仍不确定，不是完整部署/AP。

当前固定MIX的掩码改进推进结束，不补图/扫alpha/匹配/投影/gate/新训练挽救；PROJ保持停止。并非证明所有多视图无用或完全无信息，普通TTA不作新系数创新。旧2图A/MIX指标0误差、全部A逐像素正常解码一致、本地260均值复算max2.22e−16。当前无研究Run在跑，下一项尚未登记，总Goal未完成。下一轮需独立机制、真实历史/数学/代码/文献依据，不重复已停配置；本线程只3080Ti服务器，笔记本另一线程负责。

## 最新执行：冻结MIX的方法开发外确认（2026-10-04）

已登记[flip_fusion_confirmation_20261004](../../experiments/flip_fusion_confirmation_20261004/PROTOCOL.md)，Study STUDY_6416b6c4f32b474a8c736f4592b34f4c，3080Ti服务器pipeline PID26551启动。独立smoke RUN_8eaabd20c87842eea26d54c675c02630、正式RUN_5602b49993904800bf01180ed6a13274。先核原始PIPELINE_STATUS及真实PID，禁止重复启动。

仅A与冻结普通MIX，不重启投影。375份身份来源保守排除19294个trainID，从剩98993按固定hash前瞻取1536；CONFIRM_IDS SHA a1c99db0f5b4d5cea2234220ee7789e3b22a7094d1a60fc8b5b89bcae4251b07。这是有已知范围的当前方法开发外确认，不是官方预训练或整个项目未见；旧全train训练及2个空历史清单未知均披露。前置身份解析错误已保留并在任何推理前修正，无按效果换图。

smoke旧2图240秒核A/MIX历史等价，通过后1536图正式总1800秒、5000次整图bootstrap。唯一主指标为预定框好mask差组macro IoU；target估计>=.5pp且CIlo>0与all CIlo>−.1pp共同判断，完整列成功组损伤、coverage等代价。只标注候选与top300/conf>.001保留联系，不做完整AP。完成后冻结结论，不根据结果补样/调alpha/匹配/gate。总Goal未完成，本线程仅服务器，笔记本另一线程负责。

## 最新收尾：整图翻转融合有目标组初筛信号，投影配置停止（2026-10-04）

[flip_response_projection_20261004](../../experiments/flip_response_projection_20261004/REPORT.md)两Run已2026-10-03T19:44:18.095541Z完成，56文件SHA核验回传，COLLECT_eb01ec60a8d045f4a2e3820550dae88e。253有效开发图1816固定官方候选，零训练正式120.49秒。MIX普通半幅翻转logit融合在预定框好mask差组141图353候选上macro IoU+0.5483pp [+.0998,+1.0515]，通过目标组初筛；AUC也有正信号。但全体IoU仅+.0852pp区间跨零，29修复20损伤、coverage下降；不能声称总体收益已确认或原成功无损。

PROJ回投原P目标组+.4636pp，保留部分信息，但相对更简单的MIX全体IoU−.0506pp [−.0866,−.0135]、coverage额外下降。当前投影配置结束，不扫ridge/alpha/匹配/变换、不自动训练新头。MIX是标准TTA思路，保留为待独立确认的信息参照，不宣称新系数方法。MATCH单独替换显著弱于MIRROR；同类且框重叠更高不保证mask更好，源低分raw范围已在报告披露，不临时删样。

身份/原模型回放/几何/求解核验通过，报告保留全体与成功组代价及数值限制。当前没有研究Run在运行、下一项未登记；若继续该正信号，先考虑同一冻结方案在未参与方案选择图片的确认，不能先调方案再称独立确认。本研究总Goal未完成；本线程仅3080Ti服务器，笔记本另一线程负责。

## 历史启动：整图翻转响应与全支持投影（2026-10-04；上方收尾优先）

已登记[flip_response_projection_20261004](../../experiments/flip_response_projection_20261004/PROTOCOL.md)，Study STUDY_b2050d6f60a3445b84b4acb02a059f5d，3080Ti服务器pipeline PID25525。两个独立Run见RUN_IDS；2图smoke已17.57秒完成通过，随后固定256dev正式评价。原始状态和实际PID为准，禁止重复启动。

零训练六臂A/ORIG_MATCH/MIRROR/MATCH/MIX/PROJ。整张缓存640canvas精确水平翻转，不裁切放大；镜像raw对照与同层、同原预测类、全raw预测框对应并列。翻转P和c配对、逆翻完整logit；普通半幅集成MIX与把同一差异全640支持FP64投影回原P的PROJ比较。没有GT选源、8×8压缩、阈值调整或参数训练。正常原图IoU和repair/damage等决定价值，投影只有超过MIX才有额外效用证据。

本轮smoke180秒、正式1800秒上限，源/求解/几何容差预定；若无信号结束，不扫变换、alpha、匹配门槛或ridge。翻转集成已有文献，不预宣称原创。区别于旧1.5倍crop-view和同图donor共识已明确。总Goal未完成；仅此服务器，无笔记本操作。
## 最新收尾：跨任务隐藏表示桥接已完成（2026-10-04；以下执行状态为历史）

[cross_task_coefficient_bridge_20261004](../../experiments/cross_task_coefficient_bridge_20261004/REPORT.md)六个Run已于2026-10-03T19:12:25.970691Z全部completed。189文件含三组末轮checkpoint已SHA256核验回传，集合COLLECT_9884b0d6b4b047f793bbfd36e931b44a。1012有效fit图7172官方候选，分类真源T、同形错源M、框真源R各3轮192次更新；旧N直接复用。253开发图1816候选，T−A图片macro IoU−0.1037pp [−0.2962,+0.1055]，T−N−0.0722pp，R−A−0.0742pp；均未建立超过原模型和普通微调的实用收益。

保留T−M的macro Mask75+0.5206pp [+.1346,+1.0308]和BCE次要正信号，严格同预测类别子集仍有Mask75优势，不能概括成真错源完全相同。但T−A仅19修复18损伤，coverage/AUC下降，冻结T后真源相对错源TW也未形成平均IoU优势。身份、原损失/梯度、初始native梯度、两分支非零更新、冻结状态和A/N历史重放核验通过；细分证据以报告为准。

**当前同raw线性桥接、固定检测源、原生cv4联合训练三轮配置停止。** 不扩桥接、扫学习率或追加轮数；这不证明检测表示完全没有mask信息。当前无在跑研究Run，下一项尚未登记，总Goal仍未完成。下一轮需综合真实历史、代码、数学与原始文献确定不同且有判别力的缺口，不能把本配置改名再跑。笔记本另一线程负责，本会话不访问。自动跟进已更新为本轮已完成，防止重复启动/回传。

## 历史启动记录：跨任务隐藏表示桥接（2026-10-04；已由上方收尾覆盖）

已登记[cross_task_coefficient_bridge_20261004](../../experiments/cross_task_coefficient_bridge_20261004/PROTOCOL.md)，服务器pipeline PID24021已启动，六Run见该目录RUN_IDS。以原分类隐藏256维/框隐藏64维，通过零初始化桥接进入全原生cv4联合训练；T真分类、M同形同预测类别跨图错配、R真框，3epoch，同预算N直接复用。先prepare/smoke，再三组训练和正常原图评价。仅本线程3080Ti服务器，笔记本不访问。

新问题不是添加框坐标或重复7N近邻，而是检测任务塑造的隐藏表示能否经mask监督直接兑现；7I/7L/7N已使用过框hidden，旧many结构探针也研究过分类空间，不能说完全首次。CondInst/BlendMask已有回归塔mask参数先例，不预宣称结构新颖性。当前结果尚未知，必须按协议效用/错配/幅度共同判断；不扫宽度/epoch/LR。原始PIPELINE_STATUS与实际PID为准，不重复启动。总Goal仍未完成。

## 最新收尾：掩码监督位置扩展已完成（2026-10-04；优先于下方执行历史）

[mask_supervision_support_20261004](../../experiments/mask_supervision_support_20261004/REPORT.md)的4个Run全部completed，2026-10-03T18:26:13Z结束。原7,172个GT位置扩为46,282位置，原生one2one cv4训练3轮210.77秒；dev253图1816原候选，D−A macro IoU−0.0553pp [−0.1945,+0.0859]，15修复19损伤；D−N−0.0238pp，未建立优势。框好mask差+0.2474pp区间跨零，未达预设继续条件。相对N coverage+0.1263pp伴随FPR+0.2526pp，不作为净方法收益。损失/梯度/身份/冻结检查通过，A/N历史评价0误差重放。

当前配置停止，不扫topk、0.5权重或epochs，不重训N。120文件含末轮checkpoint已SHA核验回传，COLLECT_9a92094f2beb45d296106ef7db77d551；本Study不重复启动。解释限于本配置，不能排除所有多位置监督或归责原历史训练。研究总Goal未完成；下一项尚未登记，仍需历史、源码、数学与文献共同审查不同机制后才启动，不通过延长同一阴性配置堆尝试。笔记本另一线程负责，本线程不访问。
## 最新执行：掩码监督位置扩展（2026-10-04；优先于下方旧状态）

已登记[mask_supervision_support_20261004](../../experiments/mask_supervision_support_20261004/PROTOCOL.md)。服务器pipeline PID22638，2026-10-03T18:19Z启动；原始PIPELINE_STATUS为准，禁止重复启动。新问题是原one2one系数分支能否受益于同GT更多TAL初选位置；检测和评价候选不变，mask每GT总权重1，原赢家与新增位置各分一半（无新增时原权重1）。整个原生cv4可训，其他网络与BN运行buffers固定，无新参数。复用旧N三轮对照，不重训N。

2图smoke已经通过官方损失/梯度、重复位置全参数梯度和冻结状态核验（7.79秒）；1024fit正式支持构建进行中。随后D固定3轮、最多1800秒，再正常原图A/N/D评估fit前128与dev256。具体Run ID与阈值见本Study。不扫描topk/0.5权重/epochs挽救阴性；先完成回传与报告。总Goal未完成，笔记本不访问。
## 最新收尾：外部空间证据检查已完成（2026-10-04；覆盖下方历史进度）

[external_spatial_projection_20261004](../../experiments/external_spatial_projection_20261004/REPORT.md)已在本线程3080Ti服务器完成，正式Run `RUN_bb0160a72b9b4a93a43e5ffc7c4f3346`，2026-10-03T17:34:01Z结束。253图1816候选、零训练、122.28秒；五Run的76文件SHA核验回传。SAM完整输出−A macro IoU为−4.2256pp，经固定OGPS→原P后为−4.9903pp，未达到预设继续信号。当前配置停止，不扫提示/模型/solver参数、不自动训练教师。投影优于粗8格输出不等于超过原模型，详细取舍和限制只引用该报告。原始pipeline已completed，GPU无研究进程，不重复启动或收集。

**下一项尚未登记、没有新训练在运行。** 去重查到[旧Gram网络训练](../../experiments/coefficient_learning_pilot_20260911/COMPLETE.json)已经执行3seed×15轮：原生末层、完整/对角Gram、每实例logit RMS匹配；[评价](../../experiments/coefficient_learning_eval_v2_20260911/summary.csv)AP小幅上升但拥挤指标下降，不能说完全阴性，也不能包装成从未做过。其[真实代码](../../shared/coco_clean_20260911/train_coefficient_pilot.py)SHA与完成记录一致。

当前官方TAL全分辨率、整个原生cv4、动态BCE Hessian的精确组合尚未找到实际Run；但只把Gram换成更精确Hessian没有充分启动依据。系数空间预条件再经网络雅可比反传，不是参数空间Newton，也不保证共享参数目标下降。若以后考虑，必须先有新增证据并控制聚合参数梯度/实际更新尺度；旧静态教师、VarPro和7N离线Newton均需纳入比较，不按改名重复。本轮没有启动它。

总Goal仍未达成。下一轮继续综合历史、数学、原实现和原始文献选择不同且有判别力的短预算机制；不靠更多参数、扫权重或延长同一阴性配置维持实验数量。笔记本仍由另一线程独立负责。

研究目标覆盖“实例特征、原型响应、系数生成、掩码形成”整条链路。AI承担从构思、历史/文献/数学/实现审查、判别对照到执行与解释的完整判断；快速尝试不同机制，但每轮固定问题、预算与停止边界。通用要求统一维护在[规范第0节](../guidelines/EXPERIMENT_SPEC_STANDARD.md#0-有依据的研究头脑风暴)，此处不另建一套规范。



当前已记录的fast-screen、local-view、native-proto P/PC均已完成并停止当前配置；此前[两步反馈读取](../../experiments/prototype_feedback_readout_20261003/runs/RUN_17e5267de70f4364a8fde3ecc4c7a60d/REPORT.md)也已完成，R−F几乎为零，不能再次按“闭环未做”启动同一对照。总Goal尚未达成。这些停止结论限定于各自配置与预算，不证明整类信息或计算方式无效。旧13k队列及bulk传输保持停止；新启动前仍以实际Run/PID核查为准。



眼前已识别的最低成本补充是当前1816个官方候选上的监督—评价同域检查：旧S031–S033标签实验直接复用，只补当前对象的缺失联系，先锁定预算，再决定是否需要一次冻结重放。分别区分overlap归属、栅格边界、GT框支持与逆变换；用同一输出检查标签一致区/冲突区中的实际变化。它是解释现有更新的辅助诊断，不是预设的新主线或自动重训授权。若损伤主要发生在标签一致区，就降低该解释优先级；相关性不能替代机制干预。



该缺口已由[supervision_output_domain_audit_20261003](../../experiments/supervision_output_domain_audit_20261003/REPORT.md)完成并核验回传，2026-10-03T16:08:13Z结案，正式128.25秒。全部1816候选旧指标复现；N/P/PC误删TP位于标签冲突区的比例19.55%/16.38%/19.24%，多数位于一致前景区。冲突区TP删除率更高，但开发图官方BCE没有改善（P/PC明显上升）。不支持“只是评价标签不一致”的解释，不自动改标签重训；也未排除冲突梯度间接影响一致区域。该Study不得重复启动。下一项已完成历史/文献/数学/实现审查，并登记[candidate_response_consensus_20261004](../../experiments/candidate_response_consensus_20261004/PROTOCOL.md)：原模型同图raw候选GT-free响应共识，七臂零训练、2图smoke及256图开发集一次评价，固定180/900秒。2026-10-03T16:32:37Z服务器PID19106启动；先查该Study原始PIPELINE_STATUS与实际进程，不重复启动。FCIS已有mask voting先例，不能称投票为创新；当前新增判别是超过单donor和普通分数加权的共识效用，以及同数量跨分支对照。无用则停止当前配置，不扫K/框门槛/权重、不自动训练选择器。研究总目标未完成。



最新：上述[candidate_response_consensus_20261004](../../experiments/candidate_response_consensus_20261004/REPORT.md)已于2026-10-03T16:34:47Z完成并核验回传。正式101.82秒，253图/1816候选；CONS−A macro IoU−0.0144pp，CI[−0.0946,+0.0637]，16修复/17损伤，未超过普通分数加权。当前配置停止，不扫K/框门槛/权重、不训练选择器；其same-pool平均mask相似度0.9769是描述，不证明所有候选信息冗余。已无在跑Run，下一项尚未登记，先综合历史/代码/数学/文献选择不同有界机制，总Goal未完成。



笔记本的[candidate_relation_readout_20261004](../../experiments/candidate_relation_readout_20261004/PROTOCOL.md)属于另一线程；本会话不跟进其运行或回传，也不在服务器重复该S/T/M机制。之前只读核对仅作为历史去重，执行中审查记录保留，不改变其协议。



本会话服务器下一项已登记[evidence_path_usage_20261004](../../experiments/evidence_path_usage_20261004/PROTOCOL.md)：固定fast-screen U末轮权重，原A/完整/同参数关残差/仅均值/空间roll五路径，fit前128与dev256，两个Run，180秒smoke及900秒正式上限。只补已有读取是否实际发挥作用的缺口，不重训、不修改笔记本Study，不将mean投影叫阈值校准。已核服务器GPU空闲且无重复Run后启动，pipeline PID20140；先smoke再正式评价。完成依据原图效用与幅度共同解释，不自动补更多epoch。



最新收尾：[evidence_path_usage_20261004](../../experiments/evidence_path_usage_20261004/REPORT.md)已于2026-10-03T17:07:36Z完成，105.02秒，60文件核验回传。dev FULL−同参数NATIVE为+0.0096pp，FULL−MEAN为+0.0026pp，两者CI均在±0.1pp内；FULL−A仍−0.0247pp、17修复17损伤。额外路径非零但幅度和效用小，不能归因为完全无空间信息，也不自动延长训练。当前服务器无在跑研究Run；本Study停止，总Goal未完成。



解释边界：local-view没有训练，不能用训练标签错配解释它的退化；P/PC训练轨迹下降也不足以单独证明过拟合，尚无同一最终参数的fit/dev效用对应。下一轮方法由历史、实现、数学和文献共同选择，不为凑试验数量重开反馈、加大输入或连续修补阴性配置。



下一候选的去重/文献/实现审查见[外部空间证据审查](EXTERNAL_SPATIAL_EVIDENCE_REVIEW_20261004.md)：冻结SAM预测框输出→旧OGPS→原YOLO P，明确外部预训练成本及新颖性限制。当前尚未下载、登记或运行；具体同路径自投影对照/输入坐标核验/预算锁定后才能启动，不能以现有旧数字代替。



最新执行：外部证据检查已登记[external_spatial_projection_20261004](../../experiments/external_spatial_projection_20261004/PROTOCOL.md)，只读官方SAM ViT-B框提示→固定OGPS→原P，dev256、五臂零训练、prepare900秒/smoke300秒/formal3600秒预算。prepare Run RUN_90cbcd11433743149fbd3d87f8845ab6 服务器runner PID20714已启动；正式评价脚本准备中。启动前先核实际PID与Run，不重复下载/重复评价。新增外部预训练及计算单列，不能混称等预算改进；笔记本不访问。



传输修复：外部Study两次prepare因GitHub git HTTP2/HTTP1.1传输错误明确失败，Run日志保留；当前改为本机纯文件中转官方HTTPS固定commit与ViT-B权重，exec session49895传输中，不含模型执行。待服务器独立verify Run RUN_adeab845bdbd4cef8e1678f2ee5ecdac完成再启动smoke，不能按旧prepare PID误报运行。科研条件保持。



实际启动：官方资产已下载中转并在服务器verify Run RUN_adeab845bdbd4cef8e1678f2ee5ecdac核验完成，冻结源码commit dca509fe793f601edb92606367a655c15ac00fdf、权重SHA ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912。external_spatial_projection pipeline PID21217已启动smoke RUN_16c8b7c04a4d4fc8aae868ff9e572d23，通过后formal RUN_bb0160a72b9b4a93a43e5ffc7c4f3346。先核livePID，不重复；不读取笔记本。



## 以下为执行历史（其中“当前”“下一项”以本页上方最新入口为准）



2026-10-03用户明确要求：“笔记本和服务器都归你用，你的目标只有一个就是一直针对原型到系数这个链路，进行改进然后有好的效果。”



当前唯一主目标是改进原型到实例系数生成的链路，使未参与训练或方案选择的图像上获得可重复、有实际意义的正常掩码收益。实际原图Mask IoU、Mask75、修复/损伤及作用对象是效用证据；训练目标下降、方向余弦或局部机制对照不能替代效用。探索集结果与独立确认分开记录。



当时计划先完成[选点比较](../../experiments/prototype_guided_evidence_selection_20261003/PROTOCOL.md)，后按用户快速筛选要求改用下述fast-screen；原协议与等待产物保留，不能视为正在运行。每轮参数、预算和停止规则不能在结果出来后修改；收尾后根据正负证据决定下一项有界实验。



资源可统筹使用3080 Ti服务器 `connect.bjb2.seetacloud.com:33953` 与笔记本 `ssh 28358lan`。本地桌面仅用于代码、文件传输与记录，不执行模型训练或评价。服务器公共COCO入口为 `/autodl-pub/data/COCO2017`。



这项持续授权延续了主问题，更新了此前“当前比较后关闭所有自动跟进”的执行范围；并不撤销当前比较的实验停止条件。自动跟进先查Run与实际进程，避免重复启动。无需为已授权范围再次询问。



## 最新执行优先级：快速多方案筛选



2026-10-03T13:54:11.218525+00:00 用户明确提出“为什么要传缓存，另起一条线不行”并要求快速尝试、用多种创意筛选。当前优先 [prototype_readout_fast_screen_20261003](../../experiments/prototype_readout_fast_screen_20261003/PROTOCOL.md)：服务器原图直接构建小型官方TAL对象，1024fit/256dev、每臂3轮，N/U/Q/P/L/B，正常原图统一评价。短筛阳性之后才另定大规模确认；探索dev不能冒充独立测试。



旧大规模Study及迁移产物保留。其无GPU工作的等待coordinator PID6701已核验处于waiting后停止，见旧Study/SCHEDULING_HOLD.json，避免迁移完成时抢占快筛GPU；没有中断任何训练。旧relay随后也已停止，见下段记录，不是新快筛启动前提。不要根据旧PIPELINE_STATUS的running重启，先读hold与实际PID。当前主目标未缩减，改变的是先快速筛选、再投入验证的执行顺序。



## 六组快速筛选收尾后的当前行动



快速筛选已完成：普通微调与五种静态选点均未显示足够新增掩码收益，见[正式汇总](../../experiments/prototype_readout_fast_screen_20261003/REPORT.md)。不扩大该结构、不恢复旧13k队列。旧bulk relay RUN_973a07a938934b81bdd46796fcd203da已核验身份后停止，部分文件/收据保留，不能自动恢复它。



当前在[local_view_coefficient_replay_20261003](../../experiments/local_view_coefficient_replay_20261003/PROTOCOL.md)实施一次零训练短测：同一原模型读取预测框局部放大视图，比较直接空间输出与回投原图固定P，原OGPS solver完全复用。固定现有256dev、最多1小时，无GT输入、无训练/门控/新loss；结果决定是否值得确认，不自动加头救场。此Study先完成真实smoke再启动正式evaluation，查询实际进程与独立Run避免重复。整个研究Goal仍未达成。



## 局部视图短测收尾



2026-10-03局部视图七臂已全部完成并回传，正式评价190.75秒、1816候选。[结论](../../experiments/local_view_coefficient_replay_20261003/REPORT.md)：直接局部视图−A macro IoU为−2.7052pp，投影后−3.6204pp，42修复/169损伤，当前方案停止。原图同规则重选和解码/数值核验未揭示执行错误，不继续改倍率、阈值、solver或训练视图头。该Study不得重复启动。



下一项尚未启动，先做历史去重与最小可行性核对：原生prototype分支能否与系数分支共同适配，而不继续把P固定。候选对照为原生proto-only与proto+one2one系数分支，其他模块冻结，不加新网络/新loss；拟复用fast-screen已有N=cv4-only同数据同预算作为对照。须核对历史是否重复、真实Proto26梯度接口及同目标实现后，才登记独立方案与运行。总Goal仍未完成；不要为保持实验数量强行开展经典滤波或反复修补当前阴性方案。



## 当前唯一在跑的比较：原生原型与系数适配



上述去重与实现已完成；正式Study为[native_proto_coadaptation_screen_20261003](../../experiments/native_proto_coadaptation_screen_20261003/PROTOCOL.md)，服务器/root/native_proto_coadaptation_screen_20261003。2026-10-03T15:29:16Z启动协调进程PID16966，P/PC两组smoke均通过，P正式训练已开始。不要把上一段的准备状态当当前状态。



官方one-to-one proto.detach事实早在OFFICIAL_TRAINING_AUDIT.md记录过，本轮不是重新发现它；新增的是对此路由的实际短训练对照。P只训练原生Proto26；PC训练Proto26及原生one2one_cv4；N复用刚完成的cv4-only。相同1024fit/256dev、seed0、3epochs、原mask BCE、1e-4原生学习率，训练共40分钟上限。此为显式改变one-to-one mask损失到P的梯度路由，不是官方完整双分支训练复现，也未证明历史失败根因。



先跟进既有Run与实际进程，不追加并行干扰。完成后回传并以正常原图IoU/Mask75/修复损伤判断；负结果停止该配置，不用更多epoch/更大head救场。全部状态及Run ID在该Study内。



## 原生原型适配已收尾（最新状态）



2026-10-03T15:37:16Z，P/PC各3轮及评价全部完成并回传；[报告](../../experiments/native_proto_coadaptation_screen_20261003/REPORT.md)。P−A macro IoU−0.1679pp（25修复/34损伤），PC−A−0.2537pp（23/33），相对N也未改善。合计约8分钟。当前配置停止，不追加epochs/学习率搜索或新模块；不重复启动该pipeline。



目前无在跑的研究Run。下一项先查已有记录中“官方overlap-mask训练目标与原始COCO实例评价标签”的差异是否已经量化，以免把各轮一致的coverage下降直接归给读出机制。先历史去重，已有证据直接复用；缺失才锁定最小、无训练的对照，不能未经核对重复旧监督实验。该审查尚不是新的实验结果，也不能先断言当前监督有错误。总研究Goal仍未完成，快速淘汰不同假设与有价值信号后的独立确认仍有效。



### 标签线历史已核对，不重跑旧实验



此问题不是新发现：S031 [READOUT_REGULARIZATION_AND_LABELS_20260912.md](refinements/coco-structure/READOUT_REGULARIZATION_AND_LABELS_20260912.md)已有受控overlap删除标签案例；S032 [SHARED_LABEL_CONTROL_RESULTS_20260912.md](refinements/coco-structure/SHARED_LABEL_CONTROL_RESULTS_20260912.md)已有共享头原COCO栅格监督对照；S033 [READOUT_BOUNDARY_FLOW_RESULTS_20260912.md](refinements/coco-structure/READOUT_BOUNDARY_FLOW_RESULTS_20260912.md)还发现实际像素收缩不能全部归到标签边带。旧对象/版本/读出与当前不兼容，旧AP数字不归入当前结果。



若继续，只补当前固定1816官方候选中确实缺失的同域检查：读取既有mask/owners/ratio_pad和原COCO标签，统计监督分歧并关联已产生N/P/PC指标，不重新训练旧S031–S033。需要“实际删掉的TP”归因时必须有对应二值mask或同权重冻结解码，不能仅凭IoU相关性宣称因果。这是下一项可锁定的最低成本证据缺口，尚未运行；不用经典滤波或继续调参凑实验数。


