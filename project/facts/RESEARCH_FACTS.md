# COCO 密集实例分割：事实沉淀记录

最后更新：**2026-09-13**。维护目的：让后续探索从已有证据出发，保留正结果、负结果和被纠正的前提，减少重复实验。

本文件是事实索引和当前解释的入口，原始测量留在各实验目录。当前覆盖可追溯的旧 COCO 审计及 S000—S077，合并同一数据的复算；编号不等于独立实验次数。S005、S006、S025 等未执行项目不计作成果。首次整理交叉读取既有报告、协议和结果，**没有重新运行所有历史实验**；随后完成S047—S054的输入/分支/候选诊断、S055关系图分类（错误表面已更正）、S056完整COCO的固定失败修复AP与自动竞争，以及S057同权重原生one-to-one完整COCO对照，并完成S075预测交集支持归属门控复验、S076教师残差可预测性探针和S077裁切支持敏感性探针。未找到原始记录的外部模型口头汇报不作为已核实事实收录。


当前优先入口：[S074固定原型留出像素检验](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FIXED_PROTOTYPE_SPLIT_RESULTS_20260913.md)。S074显示固定原型上的系数修正在留出像素仍有效，下一步应检验共享预测器能否学到这种响应修正；不能把GT求解收益直接写成方法或密集因果。框与掩码共同失败的S067—S073仍是样本分层依据，不再继续盲扫加权损失。

更正：F30、S063/S065旧重采样区间及S061—S066的“低组”、checkpoint回执含义以本次更正为准。历史原始文件保留，索引表修复了表头错位。


**S070—S073新增结论**：实际解码GT框恢复高E4联合失败830/2292，仍不能只追框；直接内部源点替换/均值没有优势，当前分配身份说明内部点不一定给自身监督。继续沿裁切后残余响应查空间表示与读出，禁止据源点位置相关性直接加模块。详见[S070—S073](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md)。

**S074新增结论**：固定原型、原框和官方解码不变时，32维系数残差在一半像素上拟合后，对另一半像素的 AUC 仍提高；高密度失败组 split IoU 增益为 +16.40/+24.28 点，低密度失败组为 +19.01/+16.89 点，mask-good 控制仅 +6.06/+4.70 点。该队列支持“存在可迁移的实例响应读出缺口”，但低/高均有收益，且GT辅助、条件抽样、无AP，不能称密集专属或可部署方法。[S074](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FIXED_PROTOTYPE_SPLIT_RESULTS_20260913.md)

**S075新增结论**：在120张拟合图、20张独立transfer图、112个同类预测候选对上，训练和推理均使用同一预测框交集支持域，GT只用于拟合标签；3个种子、15个epoch均完成。虽然每个种子执行7,413,270次软像素更新且留出BCE下降约1.06e-5—1.18e-5，但1,804个transfer预测的二值掩码均0像素改变，Mask AP/AP50/AP75与基线完全相同。说明该候选归属门控的具体配方不能转化为任务收益；S054的GT归属竞争仍只是诊断上限，停止该公式。[S075](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_SUPPORT_OWNERSHIP_PILOT_20260913.md)

**S076新增结论**：以S074相同的GT系数残差作为教师目标，在137个图像划分后的有效实例上做5折外推；预测器只使用推理可见的实例特征（`h`、原系数、预测框几何、原型通道统计），未输入GT、真实邻居或测试掩码。教师目标在该诊断队列的平均IoU机会为+15.508点，但4种特征组合、3个岭回归强度的留出残差余弦均仅0.011—0.039，最佳配置（`coeff_box`, λ=0.1）预测后平均IoU反而下降9.391点，高/低组分别下降11.358/7.337点，最多只有28.5%的实例改善。说明S074的条件修正尚未形成简单稳定的跨图像共享映射；这否定的是当前线性、固定原型、直接残差回归配方，不否定带任务损失、非线性或更合适结构的后续方法。[S076](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TEACHER_DELTA_PREDICTABILITY_20260913.md)

**S077新增结论**：在冻结的COCO train2017 readout缓存上按七个裁切尺度重放官方解码。独立transfer集固定1.2倍裁切使IoU整体下降2.265点，高ICI组下降3.163点；逐实例选择最佳尺度仍有整体+1.047点、高组+1.493点的oracle空间。可是仅用fit集选择的原始边界外响应启发式在transfer整体仅+0.115点，高组−0.078点。该结果把“裁切支持参与失败”与“已有可用无GT方法”分开：前者得到支持，后者尚未成立。[S077](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/crop_support_probe_20260913/REPORT.md)

## 1. 后续探索先记住这些事实


| 编号 | 已观察到的事实 | 适用范围及尚不能得出的结论 | 证据 |
|---|---|---|---|
| F01 | 完整 COCO val 上，原模型高拥挤 Mask R75 为 50.829%，非高为约 60.675%，相差 9.846 点。 | 这是逐实例 GT 分组后的任务缺口；类别、大小等构成仍可能影响差距，不能直接当作拥挤的因果效应。 | [S036](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FROZEN_FULL_VAL_RESULTS_20260912.md) |
| F02 | 高组确有“框已较好、掩码仍失败”的目标。300 图诊断中，高组 Box R75 72.21%、Mask R75 53.05%，有 164 个 Box75 成功、Mask75 失败的 GT。 | 此类失败值得追掩码分支；不代表所有漏检都由系数造成。框/掩码独立匹配的汇总不能冒充同一候选逐项因果分解。 | [S002](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/STRUCTURE_RESULTS_20260911.md) |
| F03 | 固定原型和原预测框，逐实例使用自身 GT 求解系数，能改善一批目标；正则凸版本中，32 个 transfer 高组目标完整 IoU 从 72.482% 到 85.589%，+13.106 [8.852, 17.822] 点。 | 说明这些固定原型仍有可用表达空间；求解访问同图 GT，解除共享预测函数约束。不能据此断言全部原型已足够好、系数相似是根因或新图能自动获得相同收益。 | [S040](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SPATIAL_CONTROL_RESULTS_20260912.md) |
| F04 | 普通冻结特征读出头已有真实任务收益：全 val Mask AP 43.698→44.996，高组 R75 50.829→52.514；但非高−高差距 9.846→10.992，扩大 1.146 [0.478, 1.800] 点。 | 这是整体改进的强对照，尚未实现“主要修复高拥挤并缩小差距”的论文目标。三个种子是小读出头训练，不是三次完整 YOLO 训练。 | [S032](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SHARED_LABEL_CONTROL_RESULTS_20260912.md)、[S036](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FROZEN_FULL_VAL_RESULTS_20260912.md) |
| F05 | 借用新头预测面积、保持原像素排序，能复制多数小队列收益；固定自身覆盖为 90% 时，新系数的高组邻居错误反而增加 1.870 [0.791, 3.138] 点。 | AP 上升与局部像素保真度下降可以共存。现有新头收益包含面积/阈值校准，不能一律解释成实例归属或邻居分离改善。 | [S034](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_EQUAL_AREA_RESULTS_20260912.md)、[S037](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/EQUAL_COVERAGE_RESULTS_20260912.md) |
| F06 | 目标、邻居和背景的改进存在取舍。当前采样中并非普遍漏采邻居；抑制邻居的局部梯度会提高自身 BCE，背景和非高组也有类似现象。 | 排除了“当前配方普遍完全没监督邻居”的简单解释；梯度冲突本身不等于密集特有的失败根因，也不代表完整训练轨迹。 | [S038](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUPPORT_GRADIENT_RESULTS_20260912.md) |
| F07 | 已试普通投影、空间读出、混合教师蒸馏、全局原型统计条件等具体配方，尚未相对各自强对照建立密集任务优势。 | 仅停止这些实现，不证明整个方向不可能。条件 GT 求解上限与实际共享学习之间仍有未定位的差距。 | [S026](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RICH_PIXEL_READOUT_RESULTS_20260912.md)、[S039](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SELECTIVE_DIRECTION_RESULTS_20260912.md)、[S041](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SOLVER_RESPONSE_RESULTS_20260912.md)、[S045](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PROTOTYPE_CONDITION_RESULTS_20260912.md) |
| F08 | 本项目官方权重元数据已经是 `mask_ratio=1`；旧 COCO 转换脚本则确有“一个多边形写一个实例”的问题。 | 两件事必须分开。不能归责用户从未给官方模型高分辨率监督；也不能用旧错误标签训练链路证明干净数据下的 CCL 效果。 | [权重更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PRETRAINED_WEIGHT_CORRECTION_20260912.md)、[旧标签审计](C:/Dpan/codexproject/paper-disc/refine-logs/coco-evaluation/TRAINING_LABEL_AUDIT_20260911.md) |
| F09 | 最新 8 点诊断中，自身＋邻居提示比自身＋背景提示完整 IoU 更好，同时背景错误增加；相对普通均衡提示，完整 IoU +1.292 [-0.306, 3.224] 点，优势未确认。 | 同图 GT 提示可以影响输出，但还不是自动方法、训练缺身份标签或同类邻居外观干扰的证据。 | [S046](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SPARSE_IDENTITY_RESULTS_20260912.md) |
| F10 | 当前仍未建立“系数相似→实例泄漏→高拥挤召回缺口”的严格因果链，也未确认最终密集专用方法主线。 | 已有成果是失败分解、可改进空间、普通改进基线及多条不成立的简单解释。下一项需要增加新的可判别证据，不能再把同类 oracle 改善当作新突破。 | 下文实验索引与问题清单 |
| F11 | 邻居外观替换首轮 64 图完成；同类近邻固定原框、邻居减背景对照，纹理/颜色完整 IoU +1.136/+1.218 点，两区间均跨零。部分实例改善，但正常检出和主要失败子组没有稳定优势。 | 目标自身模型输入严格不变，已获得输入干预证据；但主要失败子组仅 4 个完整对照、背景位置存在失配，不能据此确认或否定所有邻居干扰机制。[S047](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NEIGHBOR_BACKGROUND_RESULTS_20260912.md) | [原始统计](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/neighbor_background_20260912/ANALYSIS.json) |
| F12 | 高 ICI 不保证真实 mask 边界贴近：S047 的一个 ICI=1.14 目标，GT mask 与所选同类邻居最短距离仍约 56 像素。 | ICI 继续作为框拥挤分层，机制诊断另加 mask 距离、边界接近和具体错误位置，不能混同遮挡/接触。 | [S047 案例与清单](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NEIGHBOR_BACKGROUND_RESULTS_20260912.md) |
| F13 | 全 val 同一个归属预测上，Box IoU≥0.75 但 Mask IoU<0.75 有 5,575 个；初筛裁切支持≥95%、没有其他保留好 mask 者仍有 4,260，其中 Box IoU≥0.90 有 1,271。 | 框好而掩码差真实存在；实际 decoder 支持须另验，不等于框完全没有影响。框差而 mask 好另有 1,106，不能混用两个独立赢家填矩阵。 | [S048](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md) |
| F14 | 新 GT 边界接近比例 E(4) 高组，在同一预测框好者中 mask 失败率 26.25%，低组21.43%。明确同类接近、低标注重叠的失败池511个/397图，其中102个Box90。 | E测自身边界靠近邻居mask的比例，不等于遮挡；149个有效边界不可定义者单列。511个按最大错误分区：背景310、同类90、自身遗漏76、其他邻居35，不能统称向同类泄漏。 | [S048 统计及失败池](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/mask_geometry_failure_census_20260912/ANALYSIS.json) |
| F15 | 严格失败32例固定原框，邻居编辑减距离匹配背景编辑：只换系数IoU纹理/颜色+0.305/−0.179点，只换原型+1.402/+2.015，两者同换+1.207/+1.490；各完整IoU区间及两路径差区间均跨零。 | 未支持系数为主因；原型点估计较大也未确认优势。原型路径提高自身覆盖但未稳定减少邻居/背景错误，不能推出原型容量不足。已保存全部c/P/像素修改，可复用追具体区域。 | [S049](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md) |
| F16 | S050复用S049发现：32主失败中29小目标、30背景误报、2自身遗漏，未含同类邻居误报主导者。单换原型邻居−原完整IoU仅−0.054/−0.132点，而背景−原−1.455/−2.147点；原型相对优势包含控制损伤。 | 临邻自身边界净补回+0.569[0.198,1.005]/+0.715[0.306,1.179]自身面积点，支持局部敏感性，尚无完整恢复。不能把队列结论外推90个同类泄漏池，下一步须固定失败/大小组成。 | [S050](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md) |
| F17 | S051明确分型/大小后44例完成。同类误报15、自身遗漏15，各6小6中3大；背景14。正常流程两填充各3seed均Mask75者同类3例、自身2例。自身遗漏只换原型实际IoU+2.541[0.650,4.777]/+1.347[−2.115,4.510]点，临邻自身净补回2.259/2.783点。 | 有具体框好mask差实例可经上下文变化恢复；系数路径不能单独解释它们。仍有邻居错误取舍、背景控制损伤和反例，整组尚未跨填充确认完整IoU优势。不证明原型容量不足或通用方法有效。 | [S051](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md) |
| F18 | S052同44例固定旧系数/旧框，局部特征插入/恢复完成。15同类误报例P3编辑位置插入纹理/颜色IoU +1.925[0.189,3.854]/+2.357[0.556,4.542]点，恢复−2.467[−4.032,−1.027]/−2.581[−4.520,−0.950]；邻居−背景的正反向区间亦排零。插入邻居错误−3.756/−4.841自身面积点。 | 定位到原型P3的一条可逆传播路径；每填充仅seed0，多重比较未校正，不是AP。15/15目标的P3编辑格与目标格重叠，平均占编辑格30.445%，不能称纯邻居特征；自身遗漏组融合目标区域未跨填充确认。仍有受损反例，不证明错误起源、原型容量不足或系数无问题。 | [S052](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md) |
| F19 | S053同15例等格数拆分P3：不与目标投影重合的邻居格插入实际IoU+0.474[−0.060,1.120]/+0.715[−0.043,1.611]点，恢复−1.345[−2.368,−0.488]/−1.056[−2.132,−0.163]；插入相对背景+0.575/+0.813、恢复相对背景−1.362/−1.035，各点态区间排零。 | 邻居独占格也可撤回影响，不能仅归为直接改到目标格。实际插入区间跨零，混合格−独占格四个比较也均跨零，不能判谁更重要；不重合网格仍有重叠感受野，等格数不等范数。15已探索目标，不是新方法或AP。 | [S053](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md) |
| F20 | S054在S053固定的15个同类邻居误报目标上，15/15都找到原图中对应的真实邻居候选。错误占用邻居像素上，目标候选响应更强比例图片等权均值59.7%；目标候选在自身GT区域约95%、邻居GT区域约21%。用真实邻居候选做逐像素竞争诊断，Mask IoU平均+4.755[0.494,8.772]点，12/15例提高；邻居误报−15.204[−25.727,−6.637]个自身面积百分点。 | 说明很多错误像素上已有候选区分信号，但原目标候选的归属排序没有充分利用；这是GT选邻居候选的诊断上限，不是无GT方法或COCO AP。竞争保留原框/原目标正响应支持，不能据此排除系数、特征或候选生成影响，也不能把候选归属问题称为密集场景唯一原因。 | [S054](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md) |
| F21 | S055关系拓扑计数保留：score≥0.10时高E(4)的M/X占21.93%、低组1.64%。旧错误表面和duplicate_like子型撤回：组件预测并集误计邻居正确输出，分母/自身GT重叠处理有误，重复子型错查IoU表。S048固定槽位复查，高组2,073个M/X中546个Mask75已成功；389个框好mask差中186个同类误报主导。 | M/X不是任务失败计数，C也不等于Mask75。全体高组1,526个框好mask差中，背景714、自身遗漏379、同类371、异类62；不能凭组件并集宣称同类泄漏是全体高拥挤主因。S056已完成修复AP与自动竞争评价。 | [S055更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/COCO_RELATION_TAXONOMY_RESULTS_20260913.md) |
| F22 | S056对全val固定5,575个框好mask差槽位做错误修复。同类误报GT删除：AP43.698→44.076，高E(4)R75 52.132→57.167，+5.035[4.582,5.488]点；低−高差距8.609→3.583。背景GT删除AP+2.709、高R75+9.024，但差距+0.261[-0.444,1.042]；全GT替换5,575失败槽位AP+5.770。 | 给定归属的GT输出修复诊断，非自动方法/严格表达上限/可相加AP分解。它支持同类错误修复对缩小差距更有针对性，尚未证明哪一网络模块是根因。自动按分数/框中心竞争AP−1.047/−5.354，高R75−2.698/−8.421，停止这两种硬竞争规则。 | [S056](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_ERROR_AP_RESULTS_20260913.md) |
| F23 | S057同权重原生one-to-one完整val：MaskAP43.686（原路径43.698），BoxAP52.295（52.554），高E(4)R75 55.231（52.132），+3.100[2.543,3.670]；低−高差距9.151（8.609），变化+0.542[-0.152,1.160]。高组P90召回12.684（13.096）。 | 原生路径没有消除拥挤缺口，R75增加未转成AP/P90优势。它更换预测分支及候选集合，不能归因为单独NMS；S056修复数不能搬用。该资产加载后实际end2endFalse，本轮显式True，不归责用户训练/配置。 | [S057](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NATIVE_ONE2ONE_RESULTS_20260913.md) |
| F24 | S058按同一预测槽位划分COCO val2017的36,335个普通GT：未分配同类Box50槽位3,704；框差掩码差6,584；框差掩码好1,106；框好掩码差5,575；框好掩码好19,366。其中4,417个框好掩码差具有≥95%的框支持代理量。 | 支持是几何代理，不能等同精确decoder约束。固定槽位状态与官方任务匹配分开；具体当前表见S067。 | [S067—S069](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md) |
| F25 | 将S058分类与S056固定槽位oracle逐实例连接：在4,417个框好、支持充分、掩码差实例中，完美删除背景误报可使65.20%达到Mask75，补自身遗漏42.20%，删除同类误报12.04%；在1,158个支持不足实例中对应为51.81/61.14/5.27%。 | 这是条件机会大小，不是可部署AP或加性分解；无最终槽位/框受限类没有oracle估计。支持下一方法优先做选择性像素归属和背景抑制，并保留自身支持，不能把全部失败归因于同类系数相似。 | [修复潜力](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TAXONOMY_REPAIR_POTENTIAL_RESULTS_20260913.md) |
| F26 | S060 受守恒候选间归属门控在1,200张训练缓存、3种子上使pair holdout BCE下降至约0.5764，但20张transfer冒烟图三个种子均改变0个像素，Mask AP/AP50/AP75与原模型完全一致。 | 该具体配方停止；训练损失下降没有转化为二值掩码收益，且训练GT独占区域与推理共同支持区域存在分布错位。300图版本因逐候选实现过慢停止，不能称完整评价。 | [S060 pilot](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SELECTIVE_OWNERSHIP_GATE_PILOT_20260913.md) |
| F27 | 2026-09-13 对S048固定槽位表进行分类v3复算：以最终槽位、Box IoU、框对GT支持、Mask IoU定义五个互斥观测状态。36,335普通GT中：无最终槽位3,704；框状态失败7,690；框支持受限2,290；框好且支持充分但掩码失败4,417；同条件掩码成功18,234。错误表面保留连续面积比，只有两个表面均≥5%才标`mixed_substantial`；4,417池中背景单表面1,188、实质混合2,656。 | v3把状态与机制证据分离；4,417只能称掩码读出候选失败，不能直接称系数/原型根因。支持阈值0.90/0.98时该池为5,142/3,320，说明中心切点需做敏感性报告。`no_final_slot`仍无法区分候选、分类、NMS、分数原因。 | [v3分类报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/IMPROVED_FAILURE_TAXONOMY_RESULTS_20260913.md) |
| F28 | S061支持保持背景风险读出：同一冻结73D系数头、raw COCO控制和support-risk各3 seed，在300张transfer图评估。support-risk使逐实例背景误报均值约0.17→0.09，但自身覆盖约0.91→0.85；Mask AP为51.406–51.689，低于原51.902，高组R75为51.007–51.678，低于或接近原51.678，非高ICI−高ICI差扩大至12.843–13.036。 | 背景加权与软支持保持的当前配方停止；它证明“背景机会大”不能直接等同于背景加权损失可部署。raw COCO控制仍有AP提升，但不是新方法。后续需保护二值化自身支持并处理采样/完整crop分布差异，或转向候选归属约束。 | [S061报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUPPORT_RISK_READOUT_PILOT_RESULTS_20260913.md) |
| F29 | S062三种子AP52.772–52.805，低于同预算raw53.256–53.372。旧称信任域的实现实际是软hinge惩罚；非高ICI−高ICI差11.987–12.104，比原11.585更大。 | 背景权重及Dice缩放也变化，不能隔离支持保持机制；停止该配方，不列为更强基线。旧“低组”实际为非高组。 | [统计更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md) |
| F30 | 更正S064/v5：4,417个支持充分的框好掩码差实例中157个有任意保留同类Mask75，严格排除后为4,260。25只计最佳框候选的Mask75，旧4,392池仍混入132个有好mask候选者。 | 撤回“25证明候选问题很小”和4,392干净池解释。v4未多计2,238成功；v5 evidence_tier旧字符串又把9,980状态落入默认成功。新表已显式修正，历史CSV不覆盖。 | [S067—S069](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md) |
| F31 | S063复算：相对raw三种子平均AP−0.061，高ICI R75+0.112点，正确图片簇CI[−0.883,+1.365]；非高−高差变化−0.190[−1.369,+0.797]点。 | 旧isin重采样CI撤回。1+3E4只乘BCE且未归一化，包含损失尺度混杂；停止当前配方，不据此立机制主线。 | [统计更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md) |
| F32 | S065/S066加权pilot相对同容量raw控制三种子平均AP−0.083/+0.024；高ICI R75+0.336/+0.895点，正确图片簇CI均跨零。S065每种子净+1个高组GT，S066净+3/+3/+2。 | 不构成稳定方法。权重尺度、Dice缩放/GT支持采样等未完全隔离；300图已反复探索，停止继续扫权重。实际各90 checkpoint，旧135回执更正。 | [统计更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RECENT_WEIGHTED_PILOT_CORRECTIONS_20260913.md) |
| F33 | S067全val官方低E4−高E4 R75差8.609点，其中7.449点发生在框差掩码差状态，框好掩码差0.911，未分配Box50槽位0.239。类别×大小共同支持标准化后差8.072点，联合差组贡献6.959点。 | 这是共同分母的观测记账，不是框导致的因果比例；支持优先追6,584联合失败，而非只盯系数相似。分组为E4，不是ICI非高/高。 | [S067—S069](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md) |
| F34 | S067各自独立替换理想GT掩码：框好支持充分4,417例AP+4.630，框好支持不足1,158例+0.864，框差掩码差6,584例+5.629。严格4,260例+4.462。联合差组修复使E4差距8.609→1.150，而支持充分好框组仅→8.156。 | GT输出修复非方法，完整掩码可能越过原框，不能称框修复贡献。AP各臂独立且不可相加，严格池为子集。无Box50槽位没有估算，不填0。 | [S067—S069](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md) |
| F35 | S068/S069：高E4联合差2,292例框支持代理中位94.33%，298例低于75%。仅删除原mask在自身GT紧矩形外的像素，高组619例(27.01%)达到固定Mask75；全val AP+1.659，高E4 R75+5.078，差距8.609→7.369。 | 保持所有原TP的GT空间范围干预，非process_mask重裁切或自动方法。粗范围只能修一部分；残余按现有TP/GT<.75必须补自身，或已有足够TP但矩形内误报多继续分流。 | [S067—S069](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md) |
| F36 | S070实际解码裁切干预：7,690槽位重放，完整val仅改6,584联合失败mask，原c/P/分数/框输出不变。MaskAP43.698→45.723，高E4 R7552.132→58.849，差距8.609→6.884。高组830/2292固定Mask75恢复；381个未裁切正确响应本就不足75%。 | GT紧框作用在实际解码640域，不是S069最终mask交矩形。原遗漏拆分高组裁切6.697%、未裁切响应12.736%自身面积；后项不能归为系数根因。GT裁切也损伤20/310掩码成功控制。 | [S070—S073](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md) |
| F37 | S070高E4框差mask差源点中心在自身mask外34.69%，框差mask好10.97%；类别×大小×stride共同支持后33.29/17.51%。S071配对48高/48低失败，在相同P和原框借用GT内部源系数：高组IoU−2.751[−6.470,+0.463]点；预测内部点−3.125、局部均值−6.579。 | 同尺度等距离镜像作控制，不用候选maskIoU选替代。相关性未转换成直接配方收益；样本是空间资格及原源点在外的条件子集，非总体AP。停止直接搬用/平均，不否定所有空间建模。 | [S070—S073](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md) |
| F38 | S072同96目标排序控制：高组内部系数的自身对负类AUC0.8638→0.8486、背景AUC0.8529→0.8312；匹配原自身覆盖后的Precision0.5826→0.5727。 | 没有显示更好的排序，不能用面积/阈值变化掩盖负结果。GT设阈值、640输入域、条件队列，不能作方法AP；需结合S073监督身份混杂。 | [S070—S073](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md) |
| F39 | S073当前官方one-to-many分配：高组48原源点45给自身/3非正；内部替代点31给自身/4给他人/13非正。两者都给自身的30例，内部系数−原系数IoU+0.867[−0.481,+2.407]，共同GT框+0.145[−1.346,+1.653]。 | 内部几何位置不等于自身监督身份；事后子组无稳定优势。是冻结检查点无增强当前分配，非历史训练、反向梯度或缺监督根因。 | [S070—S073](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md) |
| F40 | S074固定S070张量的原型、预测框和解码器，140个有效实例按GT像素一半拟合32维系数残差、另一半留出。高失败组split IoU增益+16.40/+24.28点，留出AUC+11.97/+16.16点；低失败组+19.01/+16.89点，留出AUC+14.95/+8.61点；mask-good控制仅高+6.06、低+4.70点。 | 留出提升说明固定原型对部分实例仍含可迁移的响应信息，不能把失败全归为原型无表达；低/高均有收益，不能称密集专属。GT辅助、条件队列、640诊断IoU，无网络训练/AP；all-fit为条件机会，不是方法结果。 | [S074](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FIXED_PROTOTYPE_SPLIT_RESULTS_20260913.md) |
| F41 | S075在预测同类候选框交集内训练归属门控，3个种子、15个epoch、20张transfer图均完成；软更新数为7,413,270/seed，留出BCE只下降约1.06e-5—1.18e-5，1,804个预测的二值掩码0像素改变，Mask AP/AP50/AP75不变。 | 该具体预测支持门控没有任务收益；它排除了“仅修正训练/推理支持域错位”这一解释，不能把GT标签训练期可用误写成推理期可用，也不能把S054 GT归属上限当作自动方法。 | [S075](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/JOINT_SUPPORT_OWNERSHIP_PILOT_20260913.md) |
| F42 | S076在137个有效实例、5个图像外推折上，以GT教师残差训练推理可见特征的共享岭预测器。教师平均机会+15.508 IoU点；最佳留出预测仍下降9.391点，残差余弦0.0278，正向实例比例28.5%，高/低组均为负。 | 固定原型上的GT响应机会不能由当前简单共享线性映射恢复；这是对该配方的NO-GO，不证明所有非线性、任务感知或结构化方法都不可行。实验为诊断队列、非完整COCO AP，不能把教师上限写成方法收益。 | [S076](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TEACHER_DELTA_PREDICTABILITY_20260913.md) |
| F43 | S077冻结COCO train2017 readout缓存，fit/transfer分别7811/1815个匹配目标，按0.8—1.4七个裁切尺度重放官方解码。transfer IoU@1.0为0.8163，固定1.2倍降至0.7936（−2.265点）；高ICI组0.7259→0.6943（−3.163点），低组0.8190→0.7968（−2.221点）。逐实例oracle最佳尺度相对1.0整体+1.047点[0.884,1.228]，高组+1.493点[1.061,1.947]，低组+1.022点[0.859,1.201]。 | 裁切/支持敏感性是真实因素，高组有略大的可恢复余量，但固定外扩损害正常预测；fit-only的原始边界外响应启发式在transfer整体仅+0.115点、高组−0.078点、低组+0.137点。当前统计量不能形成稳定无GT方法；结果是冻结读出诊断，不是训练或完整AP。 | [S077](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/crop_support_probe_20260913/REPORT.md) |
数值约定：AP 以 0–100 计，R75/IoU/覆盖等写成百分比，差值为**百分点**；区间沿用对应实验定义。`+1 AP 点` 不等于相对增长 `1%`。未给 AP 区间时，不从其他指标的区间推断 AP 显著性。

## 2. 数据与评价口径

| 数据/队列 | 数量和用途 | 后续使用限制 |
|---|---|---|
| 原 COCO val2017 | 5,000 图、36,335 普通 GT；低 20,981、中 10,109、高 5,245。S036 完整任务评价。 | 已用于探索，后续不能称从未看过的盲确认集。 |
| 旧 COCO-Dense val | 1,576 图、20,827 原始 GT；旧多边形展开产生 23,832 行标签。 | 旧模型训练/评价链路单列，不能混入干净主结果。 |
| S002—S017 主要结构队列 | 300 张 dense 富集 val 图、3,635 GT、高组 788。不同子实验另有匹配/可行性筛选。 | 其 R75 不应与完整 val 或后续 300 train 图混用。 |
| S018—S022 固定失败队列 | 243 图、887 个已选择失败目标，高 215、非高 672。 | 按失败预选，不能把其中 IoU 增益当作全数据 AP 增益。 |
| S024 起共享读出队列 | 1,200 张 train fit 图及 300 张 train transfer 图；transfer 2,002 GT，高 298；常用固定匹配子集 1,815、高 266。fit 的有效匹配集后来为 1,187 图、7,811 目标。 | transfer 对新读出训练未用，但被反复探索，且官方预训练本身使用 train2017；不是全模型从未见过的样本。 |
| S039—S040 小诊断 | 128 个目标、97 图；fit/transfer 各 64，各自高/非高各 32。 | GT oracle 对 transfer 目标也使用其自身 GT，不代表求解器跨图泛化。 |
| S044、S046 | 上述 64 个 fit 目标、47 图，高/非高各 32，三个保存头状态。S046 共同可行子组仅 38 个。 | 小样本、重复使用；三保存状态不能替代三份独立数据或完整训练。 |
| S047 输入干预 | 从完整 train 压缩包抽取 64 张图，同类近邻 32、异类近邻 16、同类远邻 16；保守排除已有记录 5,004 图片 ID。 | 几何可修改条件选样，未按干预收益选；官方预训练已见 train。高 ICI 仅同类近邻中 15 个，类别/大小/位置控制不足，不作组间因果差异结论。 |
| S048 几何和失败清点 | 复用原模型全部5,000 val图预测，36,335普通GT、4,952图含普通GT；无新增前向。E(4)低21,391、中5,342、高9,453、undefined149。 | 原始instances.csv中undefined有初始低分组占位；正确入口instances_classified.csv。保留全体GT分母；该val已探索，不作盲确认。 |
| S049 严格队列输入/输出干预 | 从S048固定池选64图：同类失败32、成功16、异类失败16；实际63个，成功对照1例精确支持不足排除、不替补。两填充各3seed、同一模型。 | 同类失败32中18个person、5个Box90。可编辑和背景匹配要求改变组成，不能外推511全池或全COCO；三个填充seed不是训练seed。 |
| S051—S052 分型队列 | S051从原失败池剩余目标固定选44图：同类误报15、自身遗漏15、背景14，前两型小/中/大各6/6/3，背景6/6/2。S052复用全部44图及保存邻居/背景输入。 | S051两填充各3seed；S052仅各seed0。没有按S051收益筛选，但已探索，不是独立确认；类型配额不消除类别等组成差异。S052固定c0/b0，只测原型分支干预，不是正常流程/AP。 |
| S053 格子拆分 | S052全部15同类误报目标，混合格/独占格/背景格每图各取k=1—4格，三个冻结几何子集平均后统计；两填充各seed0。 | 三几何seed不是训练/填充重复；固定c0/b0，仅操作原型P3输入；所有图均可行，无排除。不能由这一组推断全部高ICI/E目标的效果。 |
| S054 候选归属诊断 | 固定S053的15个同类邻居误报目标；提取目标/真实邻居候选原型logit，在目标、leak、邻居、背景区域比较，并做GT选邻居的竞争上限。15/15邻居候选匹配，75条区域记录，68.859秒。 | GT用于邻居候选匹配和评分；竞争不是自动方法/AP。目标候选在leak像素仍胜约59.7%，竞争IoU+4.755点但保留2个反例；下一步需训练时可用GT归属监督、测试时仅用候选。 |
ICI 在当前协议中按**单个 GT 实例**计算：同类其他 GT 框与自身框的相交面积之和，除以自身框面积。它可以大于 1，不等于真实掩码重叠率、遮挡比例或每图对象数量。分组为近零（≤1e-10）、中等（至 0.5+1e-10）、高（>0.5+1e-10）。旧的图片筛选不能代替实例分组。

S048新增E_i(r)：有效自身可见GT边界中，距其他同类/异类/任意GT掩码并集≤r的比例。主r=4个640输入等效像素，保存2/8及大小归一化敏感性；0为低、(0,0.2)中、≥0.2高。保存连续值、独占邻居版本与GT直接重叠；没有可用边界时undefined。它是本次操作性诊断量，未经查新，不包装成独立创新。

任务评价先保留完整 GT、做官方匹配，再按实例分组。不要删掉非高 GT 后把相应正确预测算成假阳性。固定匹配目标的 IoU/错误面积、采样点 IoU、全 GT R75 和标准 COCO AP 是不同量。相同全局 Precision 工作点下的分组 Recall，也不同于各组分别挑阈值后的 Recall。

GT 在训练中正常可用；限制的是最终新图推理不能偷看其 GT。逐实例 GT 求解、GT 删除错误区域、GT 选候选、GT 穷举阈值均列为诊断，不算自动方法结果。

涉及 bootstrap 时，以实验原协议为准；最近实验通常先对同目标的保存头状态取均值，再按图片簇重采样 2,000 次。区间跨零表示当前比较未确认，不等于证明两方法相同。多项探索的点态 CI 不自动构成整条机制的同时置信保证。

## 3. 历史链路与必须保留的更正

| 事项 | 客观记录 | 使用决定与证据 |
|---|---|---|
| 旧多多边形转换 | 脚本逐 polygon 写行；合成一个 annotation、两个 polygon 会生成两个实例。旧 val 日志 23,832 与展开计数吻合，而原始 GT 是 20,827。旧训练日志也存在对应指纹。 | 数值不因此自动变成“伪造”，但其科学比较条件受损。旧服务器缓存未直接取得的部分仍保留证据边界。[审计](C:/Dpan/codexproject/paper-disc/refine-logs/coco-evaluation/TRAINING_LABEL_AUDIT_20260911.md) |
| 干净训练队列 | 已保存 baseline_s0 第 1—8 轮；第 9 轮后段非有限梯度导致停止，其余任务未形成完整三种子主对照。 | 不能把部分训练当完成的三种子实验；也不能由一次失败推出用户所有训练有问题。[训练状态](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/TRAINING_STATUS.md) |
| 官方标签管线 | 原生 loader 核查了 243 图、3,298 annotations；多多边形仍保持一个实例身份。不同栅格化、重叠编码、缩放路径的像素标签并不完全相同。 | “实例身份正确”与“每个像素监督完全相同”分开检查。[S020](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NATIVE_LABEL_PIPELINE_RESULTS_20260912.md) |
| 官方权重配置 | `yolo26m-seg.pt` SHA256：`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；保存参数包含 mask_ratio=1、overlap_mask=True、imgsz=640。 | 元数据足以否定“当前官方模型只接受过 ratio4 监督”的前提，但不是独立复现全部预训练历史。[更正](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PRETRAINED_WEIGHT_CORRECTION_20260912.md) |
| 原型/系数早期推断 | 系数余弦变小或 raw mask 互相重叠变小，不等于相对自身及邻居 GT 的错误变小；此前方向干预已出现这种分离。 | 停止把“推开系数”直接当作“治愈泄漏”的证据。[机制定位记录](C:/Dpan/codexproject/paper-disc/refine-logs/coco-evaluation/MECHANISM_LOCALIZATION.md) |
| 高网格的解释 | S019 同图 GT 求解的高网格收益存在；S020 更接近原生标签管线时数值改变。 | 不改写旧数值，但撤回“证明用户训练分辨率用错”的解释。S019 与 S020 不能拼成同口径增益。 |
| S042 优化程度 | 全批量 LBFGS 到达 100 步预算，未达声明的平稳性容差。 | 不写“已充分优化仍拟合不了，因此输入缺信息”。S043 只补了特定固定表征、固定正则的凸最后层问题。[S042](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TEACHER_FITTING_RESULTS_20260912.md) |
| S044 背景误差区间 | B 折教师高组背景错误变化 -2.963 点，CI [-6.255, 0.514]，跨零。 | 保留下降点估计，但不能把 A/B 每一项都写成已确认改善。[S044](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TEACHER_REPEATABILITY_RESULTS_20260912.md) |
| S048→S049 裁切支持代理 | image538067/ann443449原坐标框覆盖95.34%，实际decoder支持93.38%，未达95%门槛。 | 保留原初筛、记录精确复核后排除，不替补。不能以Box IoU高或浮点框覆盖代理宣称裁切支持充分。[S049](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md) |

旧 Faro 及最早 CCL 外扩表保留在原材料中。本记录当前主线为 COCO；没有用用户转述的旧表替代本地来源核实，也没有把 COCO 标签错误外推给 Faro。

## 4. 已完成实验索引

每行是一个可查证结果或同源结果组，不是一次新的独立复现。精确超参、排除数、预测域及原始文件位置以链接报告及其协议为准。原始结果主要位于 [diagnostics 目录](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics)。

| ID | 做了什么及客观结果 | 对后续的约束 / 报告 |
|---|---|---|
| S000—S001 | 追踪实际 one-to-many、类别 NMS 与原始候选 source index，检查重放一致性。 | 这是本运行路径，不代表全部 YOLO26 推理模式。[结构诊断](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/STRUCTURE_RESULTS_20260911.md) |
| S002—S004 | 已完成部分：量出框/掩码失败缺口；部分层特征探针；高组实际系数 own-vs-neighbor AUC 约 0.934、坐标对照约 0.961、GT 原型读出约 0.993。S003/S004 原计划仍属部分完成。 | 二类排序 AUC 没包含全部背景，不能当完整分割能力；投影特征探针不代表原层容量上限。[结构诊断](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/STRUCTURE_RESULTS_20260911.md) |
| S007—S008 | 相对邻居响应/同面积调整，在 develop 高组邻居错误下降约 0.134 点，同时背景增加约 0.138、IoU 下降约 0.029。开发选择零干预。S008 是同输出像素流复算。 | 当前响应抑制配方不进入方法主线；零干预 holdout 无变化不算新方法有效。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RELATIVE_OWNERSHIP_RESULTS_20260911.md) |
| S009—S010 | 加入自身/邻居/背景三分区，观察抑制邻居与背景错误的取舍；S010 为相同响应的组合分析。 | 不能只量两个实例互相重叠而忽略背景。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/THREE_REGION_RESULTS_20260911.md) |
| S011 | 跨图响应校准，正常高组 IoU -0.018 点、AP +0.031、高 R75 未变。 | 当前配方无明确任务收益；三个采样 seed 不等于完整训练 seed。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROSSIMAGE_RESPONSE_RESULTS_20260911.md) |
| S012 | 160 train/400 val 的条件探针中，高组原型 GT 读出相对阈值对照未用点 IoU +2.965 [1.685, 4.279]；局部增量 +0.932 [0.157, 1.734]。 | 非等容量完整方法对照；保留表达空间线索，不宣布空间信息缺失。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FAILURE_LOCALITY_RESULTS_20260911.md) |
| S013 | 6 个配方×3 seed，共 270 checkpoint；排序＋覆盖项相对 BCE+Dice，AP -0.218，高 R75 +0.165 [-0.812, 1.209]；自身、邻居、背景正像素均增加。 | 停止当前排序损失；不能把普通 BCE+Dice 改善归给该损失。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/OWNERSHIP_RANKING_RESULTS_20260912.md) |
| S014 | 370 个高组 Mask75 失败 GT 中，53 个在 NMS 前有可用好掩码候选；50 个被 NMS 去掉，最佳被抑制候选对应 42 个同 GT 抑制、8 个归属模糊、0 个明确跨 GT 抑制。GT 选系数固定框使高 R75 53.046→56.472。 | 不能据此说不同实例互相 NMS 掉是主因；GT 选候选不是可部署选择器。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CANDIDATE_LINEAGE_RESULTS_20260912.md) |
| S015 | 317 个缺少好掩码候选的高组失败中，237 个仍有最终框候选；254 个可做 GT 系数拟合，其中 220 个达到 Mask IoU75，而阈值对照为 29 个。 | 机会不全来自候选筛选；有限迭代 GT 解不是表达极限，且早期有大范数。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NO_CANDIDATE_READOUT_RESULTS_20260912.md) |
| S016—S017 | GT 拟合也明显改善非高组；控制类别/大小/框质量后差异缩小。高组 237 个最终框候选中，当前分配器把 215 个分给自身 GT、18 个非正样本、4 个分给其他 GT。 | 不能把读出机会或未分配问题概括成密集专属；当前分配不是历史每步训练的回放。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/DENSITY_SUPERVISION_RESULTS_20260912.md) |
| S018 | 887 个固定失败目标，显式低网格 GT 系数拟合；高 IoU 51.146→63.395，55/215 目标出现 loss 下降而 IoU 下降；部分拟合触预算。 | 目标函数与评价可以不一致；不能指称官方原权重使用 ratio4。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/DENSITY_SUPERVISION_RESULTS_20260912.md) |
| S019 | 网格×支持框 2×2：高组 160/GT框、160/预测框、640/GT框、640/预测框 IoU 为 63.385、66.832、80.850、86.077；非高也改善。 | 同图 GT 优化结果，不是训练开关的新图收益。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/GRID_SUPPORT_RESULTS_20260912.md) |
| S020 | 改为实际原生标签路径，高组相关 160→640 对照 IoU 67.161→73.917，+6.756 点；非高 +7.045。 | 标签实现改变数值；必须引用对应协议，不能继续沿用 S019 的 17.47 点解释该结果。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NATIVE_LABEL_PIPELINE_RESULTS_20260912.md) |
| S021 | 同 887 队列：高组原 IoU 51.146、仿射 52.537、GT 阈值 57.419、自由系数 73.917；正比例缩放保持二值掩码。 | 超过单阈值的形状机会存在，不足以证明系数头输入/容量不足。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_CALIBRATION_RESULTS_20260912.md) |
| S022 | 当前共享最后层总 mask 梯度不利者，高 39/215（18.14%）、非高 130/672（19.35%）；删除邻居梯度救回高 13/215。 | 未见该局部冲突频率高组独占，不能解释完整训练失败链。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SHARED_HEAD_GRADIENT_RESULTS_20260912.md) |
| S023—S024 | 小 pilot 后扩至 1200/300，比较读出输入；有序空间输入高 R75 50.224，原模型 51.678，差 -1.454 [-3.410, 0.314]，未形成优势。 | 仅约束当前配方，空间输入并未被证明普遍无效。[pilot](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_INPUT_PILOT_RESULTS_20260912.md)、[扩量](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_INPUT_SCALE_RESULTS_20260912.md) |
| S026 | 原型＋特征＋邻域像素读出，5 臂×3 seed；高 R75 51.230，原 51.678，标量 51.566，打乱邻域 51.230。 | 当前增加空间输入未超过强简单对照，不能用退化的弱控制制造“空间提升”。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/RICH_PIXEL_READOUT_RESULTS_20260912.md) |
| S027—S029 | 复用 S026 保存头，拟合图采样 BCE+Dice 相对原模型 0.392747→0.385368，但完整 IoU 76.718→76.707；没有新增共享头训练。仿射投影解释约 78% 响应能量，删除仿射后高 R75 51.007，低于原 51.678。 | 响应能量比例不是因果贡献比例；去仿射未救回收益。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_FIT_AND_CALIBRATION_RESULTS_20260912.md) |
| S030—S031 | 同 160 目标限制范数仍有 GT 机会；仅替换标签，同图拟合完整 IoU：native overlap 80.811、native independent 83.740、raw COCO 86.457，原 78.003；有少数标签冲突病例。 | 排除部分标签冲突有局部因果证据，不等于所有密集失败来自用户标签错误。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_REGULARIZATION_AND_LABELS_20260912.md) |
| S032 | 同头、同预算、3 标签×3 seed，135 checkpoint。raw COCO 监督使 300 图 AP 51.902→53.313、高 R75 51.678→54.586；差距缩小 0.307 点未确认。 | 真实普通读出改进；仅关闭 overlap 的原生独立标签没有复制它。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SHARED_LABEL_CONTROL_RESULTS_20260912.md) |
| S033 | 96.946% 原生独立与原 COCO 标签分歧在 2 输入像素边界内；raw COCO 头令 81.763% 匹配目标面积缩小，高组背景减少且自身覆盖减少，删除并不限边界。 | 支持标签/收缩的解释线索，不支持“已主要修复邻居归属”。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_BOUNDARY_FLOW_RESULTS_20260912.md) |
| S034—S035 | 原排序＋新头面积 AP 53.219，接近新头 53.313；条件标量偏置 AP 52.689、高 R75 54.474，亦有收益。 | 保序校准是必要强对照；借用新头面积的诊断不等于独立廉价算法。[面积](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_EQUAL_AREA_RESULTS_20260912.md)、[偏置](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/READOUT_BIAS_CONTROL_RESULTS_20260912.md) |
| S036 | 全 val 5,000 图、11 输出：原/条件偏置/系数 AP 43.698/44.439/44.996，高 R75 50.829/52.113/52.514。系数高 +1.684 [1.097, 2.277]，但差距扩大。 | 当前最完整的任务收益证据。v1 空检测索引错误修复后复用 4,809 图并完成剩余 191，未删空图 GT。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FROZEN_FULL_VAL_RESULTS_20260912.md) |
| S037 | 高组 237 个 90% 覆盖可达目标，等覆盖时新头邻居错误 +1.870 点；残余 100 个有框高失败中，82 个三头穷举阈值均不过 Mask75，其中 80 个原框几何支持充分。 | 仍有不能仅调阈值解决的归属排序问题；剩余全部漏检不能统归系数。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/EQUAL_COVERAGE_RESULTS_20260912.md) |
| S038 | fit 高组实际 512 采样平均含自身 245.79、邻居 131.60、背景 105.51 点；自身/邻居梯度 cosine 约 -0.901，背景约 -0.955，非高也负。 | 不是普遍未采邻居；梯度负相关不能独立证明密集专属机制。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUPPORT_GRADIENT_RESULTS_20260912.md) |
| S039 | 自身约束局部方向改善部分 BCE 取舍，但 transfer 高 32 完整 IoU 仅 +0.019 [-0.021, 0.046] 点，各方向两幅度均无 IoU75 救回。 | 停止当前投影/步长配方，不能把采样损失变化当恢复目标。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SELECTIVE_DIRECTION_RESULTS_20260912.md) |
| S040 | 128 目标×3 状态×5 臂，1,920 凸拟合均收敛。transfer 高组全局 32 系数 GT 解 +13.106 点；空间 128 相对同维全局非线性仅 +0.820 [-0.358, 1.966]。 | 有约束 GT 机会重复存在，但空间特异性未确认，停止扩空间基追分。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SPATIAL_CONTROL_RESULTS_20260912.md) |
| S041 | 3 臂×3 seed，同保存模型/优化器状态继续训练，135 checkpoint。solver 蒸馏高 R75 55.257，直接监督 55.369，差 -0.112 [-0.678, 0.518]；AP 差 +0.069 无区间。 | 当前 GT 教师概率混合 KL 没有超过同预算继续监督，停止此配方。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SOLVER_RESPONSE_RESULTS_20260912.md) |
| S042 | 同 7,811 fit 目标，pure Adam 与有限 full LBFGS 都降低教师 KL；LBFGS 降约 20.24%，采样 IoU 81.545→82.388，但未收敛到容差。 | 不能得出充分优化后的容量上限或输入缺信息。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TEACHER_FITTING_RESULTS_20260912.md) |
| S043 | 固定表征的正则二次最后层，12/12 PCG 收敛。教师加权 MSE 保存头 2.278553→固定 hidden 最后层 2.097559→adapted hidden 2.003674。 | 只定位这些固定表征/线性读出/正则下的残差，不是全非线性头的信息论上限。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CONVEX_SHARED_READOUT_RESULTS_20260912.md) |
| S044 | 64 fit 目标、47 图，A/B 教师与 E 坐标互斥，384 拟合均收敛。高组修正 cosine 0.815、E 二值分歧 4.36%，完整 IoU A/B +9.405/+8.784 点。 | GT 机会有可重复成分，但仍有采样差异；B 背景错误改善区间跨零。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/TEACHER_REPEATABILITY_RESULTS_20260912.md) |
| S045 | 全局原型矩条件读出，留折 MSE base 2.553、真实原型 3.315、错配原型 3.320；真实−错配 -0.00535 [-0.07879, 0.06710]。同图共享系数残差修正也未改善原基线。 | 停止全图均值/二阶矩 PCA 双线性交互配方。上游头已见 fit，新增映射分折不等于新图完整泛化；部分增强求解器未收敛须保留。[结果](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PROTOTYPE_CONDITION_RESULTS_20260912.md) |
| S046 | 64 fit 目标中 38 个共同可行，684 拟合均收敛。高可行 28 个，自身＋邻居相对均衡 8 点：E IoU +2.182 [0.771, 3.986]，完整 IoU +1.292 [-0.306, 3.224]。 | 提示类型影响响应，但完整掩码主要控制未过关，且存在背景取舍。单一稀疏提示配方到此结束。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SPARSE_IDENTITY_RESULTS_20260912.md)、[原始统计](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/sparse_identity_probe_20260912/ANALYSIS.json) |
| S047 | 冻结原模型，64 图邻居替换/背景等面积修改、两填充×3 seed、固定 source 四格解码。原框好但 mask 差的同类近邻 6 个，完整对照 4 个；固定框 IoU 配对差纹理 +1.673 [-1.326,4.672]、颜色 -2.311 [-8.865,1.774]。正常检出未稳定改善；64图自身输入零变化，842哈希及零干预/解码通过。 | 部分实例邻居敏感，但填充及背景位置影响明显；应先改善失败选样/空间对照，不立即转方法训练。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NEIGHBOR_BACKGROUND_RESULTS_20260912.md)、[逐图差值](C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/diagnostics/neighbor_background_20260912/paired_image_effects.csv) |
| S048 | 295.75秒完成5,000图/36,335GT的原预测复算，无新增前向；同槽位box好mask差5,575、支持且未恢复4,260；同类真实边界拥挤低重叠失败511个/397图、Box90为102。新高组Mask R75 52.132%、低60.740%；同槽位给定框好时失败26.252% vs21.431%。 | 两级失败均存在，不能把高低构成差当因果。149有效边界无法定义单列；预测身份/缓存重放验证。保留ICI和E两套分组。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_GEOMETRY_FAILURE_RESULTS_20260912.md) |
| S049 | 210.485秒完成64入选/63可评图，两填充×3seed；严格同类失败32。固定原框只换系数相对背景控制纹理+0.305[−0.416,0.897]、颜色−0.179[−1.479,0.924]；只换原型+1.402[−0.420,3.569]/+2.015[−0.299,4.802]。951原产物哈希、原source/原mask、实际自身输入不变、FP64恒等式通过。 | 当前不支持系数主因；两路径之差亦跨零。正常Mask75邻居减背景仅+1.042/−1.042点且跨零。原型覆盖改善有误报取舍；不训练新损失、不将混合c/P解释成自然因果责任份额。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CROWDED_BRANCH_RESULTS_20260912.md) |
| S050 | 复用S04963图，25.516秒/2268解码，全部完整IoU重放、七区域及代数分解通过。32失败的单原型路径局部临邻边界补回0.569/0.715点，但完整IoU未改善；背景对照损伤解释主要相对优势。原型两填充相对有利12、不利9、其他11。 | 同源探索性像素分解，无新模型前向/训练。29小目标、30背景误报，缺失同类泄漏主导类；不能再将其当全部拥挤失败代表。附正例/控制损伤例/反例和未用图片候选清单。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/PIXEL_FLOW_RESULTS_20260912.md) |
| S051 | 同类误报/自身遗漏/背景15/15/14例，错误×大小固定配额，局部邻居16输入像素半径，44/44精确crop/source/input通过。单原型自身遗漏纹理实际IoU+2.541[0.650,4.777]、颜色+1.347区间跨零，目标临邻净恢复两填充排零但邻居误报增加。正常流程6输入全Mask75同类3、自身2。 | 新增有原图绝对恢复的机制个体；不得以相对背景+9.635点全当真实恢复。局部剂量不同于S049整邻居；668个主产物哈希只是产物核验。含完整类型×大小、正例和明显受损例。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/SUBTYPE_NEIGHBOR_RESULTS_20260912.md) |
| S052 | 44图、5632条记录、194.281秒；5层×目标/编辑位置/全图×插入/恢复×邻居/背景×两填充。固定原c0/b0，同类误报P3编辑位置实际IoU+1.925/+2.357、反向−2.467/−2.581，四点态区间排零且通过配对背景对照。99原产物哈希、220自替换、176全图端点检查通过。 | v1在首次直接模块调用时CPU/GPU引用错误，尚无干预结果；v2只改用实际AutoBackend副本并通过重放。P3全部有混合格，5/15满足观察后两填充正反向>1点描述规则。保留82812目标区域可逆恢复及561465受损例；无训练。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md) |
| S053 | 15图、780条记录、32.734秒；P3完整混合/独占分区及等k格对照，三个几何子集先图内平均。独占格实际插入IoU+0.474/+0.715区间跨零，恢复−1.345/−1.056均排零，配对背景正反向亦排零；混合−独占区间均跨零。71原产物哈希、390自替换、120旧掩码逐像素重放通过。 | 缩小了“只改到目标格”的解释空间，但仍是可逆传播，不判自然起源或独占格主导。两填充正反向实际/配对均>1点规则，混合/独占各2/15。非加性总体未确认，保存正反例、所有格子及P3通道值；无训练。[报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md) |
| S054 | 15图、75个区域记录、68.859秒；15/15真实邻居候选匹配。leak目标候选胜率59.7%；GT候选竞争IoU+4.755[0.494,8.772]点、12/15提高，邻居误报−15.204[−25.727,−6.637]点。保存原始logit/两候选掩码/竞争掩码；无输入编辑、无训练。 | 真实邻居由GT bbox匹配，只是诊断上限；logit不是概率，竞争仍依赖原框和目标候选支持。不能写成自动方法收益、全局AP或系数根因；下一步需实现可学习候选间归属竞争并做独立图片验证。 | [报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md) |
| S055—S056 | S055全val关系拓扑保留，组件并集错误表面与duplicate_like撤回。S056同槽位修复9臂、5000图、36335GT、385.641秒；GT同类修复AP+0.378、高E(4)R75+5.035、差距−5.026。自动分数/中心竞争AP均降低。 | 纠错与新增官方AP重算；基线精确重放，不是训练成果。将实例关系、同槽位Box/Mask状态、互斥错误像素分区、输出修复收益四轴分开。[S056报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/MASK_ERROR_AP_RESULTS_20260913.md) |
| S057 | 同权重原生one-to-one完整5000图，330.515秒，555714预测，36335普通GT；MaskAP43.686，高E(4)R7555.231，差距9.151，高P90召回12.684。 | 所有JPEG和实际分支验证通过；两个分支不能互作只换NMS的因果对照。原路径方法结果不可直接外推。[S057报告](C:/Dpan/codexproject/paper-disc/refine-logs/coco-structure/NATIVE_ONE2ONE_RESULTS_20260913.md) |
部分报告合并了多个实验，读取时必须同时对齐实验 ID、队列与 raw 文件，不按相同报告标题合并指标。

## 5. 尚未回答的问题与已计划但未执行的工作

S070精确裁切/响应分解与S071—S073源点、排序、当前分配对照已完成，见F36—F39。下述S067—S069‘下一步’现为历史启动记录，不重复运行。

**当前优先级以S067—S069为准。** 6,584个联合失败槽位中，高E4有2,292个；理想矩形外裁切后619个达到Mask75，743个现有TP/GT不足75%必须补自身，930个TP足够但矩形内误报仍多。743个中的298个框支持代理也不足75%，另445个代理支持足够但自身像素缺失。下一机制实验分别针对这两类残余，先验证精确解码支持，再沿原型/系数/候选来源追查；不得把联合状态直接归因于检测框。以下旧问题表保留历史解释，不再据其建议盲启归属头或损失扫参。

| 问题 | 当前状态 | 下一步怎样增加证据 |
|---|---|---|
| 邻居的外观是否实际干扰目标预测？是否同类相邻尤其明显？ | S051输入恢复、S052/S053传播路径；S054候选响应机会；S056同类GT删除能显著缩小E(4)差距，自动分数/中心竞争却退化；S057原生路径仍有差距。 | 保留实例归属作为目标；后续需学习可信竞争者及像素归属并保护自身，普通硬抢占已否定。S013排序损失和旧邻居MLP也未超过其控制，不能原样重做。新增方法需明确其训练/推理约束与这些旧臂的实质差异；原生路径纳入基线。 |
| 干扰先表现为分类/候选、框，还是框已好但掩码差？ | S048同候选框好mask差、S052/S053固定c0/b0原型局部传播均已完成。 | 原型P3不是已证明的起源；下一看错误像素的候选归属信息及简单重叠消解能否修复，并量自身损伤，连接可自动使用的信息，不把固定路径效果当全部召回解释。 |
| 已有特征未被充分利用，还是头缺少实例空间信息？ | GT系数机会存在；S052/S053说明邻域特征能改变固定系数输出；S054表明候选响应排序存在可利用线索。 | 当前更值得做候选间归属读出/竞争模块；要证明已有特征被利用，需训练时用GT归属监督、测试时只用预测候选，并与同容量头比较。 |
| 如何缩小高拥挤召回差距且保持整体 AP？ | S036 改善整体，但差距扩大；尚无达标方法。 | 新方法须超过普通 raw COCO 读出/保序校准/同预算继续监督，并报告全 GT 高中低、共同 Precision、类别/大小对照。 |

邻居替换S047、全量几何失败清点S048、严格失败输入/输出四格S049、S050像素流、S051分型输入、S052原型局部特征干预、S053格子拆分及**S054候选归属响应诊断均已完成**。S056已跑简单自动竞争，S060已跑归属门控小试，S061—S066已跑损失变体，均尚未建立密集主线；不能再写“自动归属完全未执行”。不得按旧计划重复这些具体配方，不把后续建议写成完成结果。

正式 CCL、mask_ratio 训练对照及新的端到端训练没有因本次整理而恢复。历史执行表较早的“下一步”文字保留为当时记录；当前完成状态以本文件和执行表顶部的最新条目为准，不按旧计划重复启动实验。

## 6. 后续维护方式

每次本任务完成实验后更新本文件，保留稳定实验 ID，追加以下信息：

1. **问题与状态**：计划、运行、完成、失败、仅复算；说明相比已有记录新增了哪项信息。
2. **数据与预算**：split、图片/GT/可行子集数量、是否使用过、训练 seed/保存状态/采样 seed 各是哪一种。
3. **干预与控制**：改动项、不变项、GT 用在训练/选择/评价中的哪个阶段。
4. **客观测量**：完整对照、单位、差值、区间、负结果、覆盖与邻居/背景取舍；不只记最好一项。
5. **来源**：协议、源码快照、逐图/逐实例结果、统计文件、模型标识、错误与校验回执。哈希校验不能冒充独立复现。
6. **解释与决定**：支持什么、不支持什么、停止哪个具体配方、哪个问题仍未解决。若撤回解释，写明日期和原因，保留原测量。
7. **实验的判别作用**：启动前写明与旧实验的区别、待区分的解释、各可能结果分别改变哪个后续决定。仅凭oracle涨分、换权重或更换名称而不增加可判别信息，不再启动。

新增实验前先查这里是否已有同队列、同干预、同指标的结果；若要重复，说明是独立复现、修复实现、扩展样本还是新对照，避免把换名称/换采样当作新机制发现。

本文件维护不涉及定时任务的提示、目标或频率修改。

### 更新记录

- 2026-09-12：首次建立；整理旧 COCO 标签/权重更正与 S000—S046，补齐 S046 报告，单列尚未执行的邻居背景替换方案。此次是事实归档，没有新增模型训练。
- 2026-09-12：实际完成 S047 邻居外观输入干预，加入整组阴性/不确定结果、两个正反案例、ICI 与实际 mask 距离的区别，以及下一步选样/空间对照限制。无新增训练。
- 2026-09-12：完成S048全量同候选失败/GT边界清点、S049严格失败队列的系数/原型路径对照；新增F13—F15、明确511例可追查失败池、保留未确认路径结果及精确crop排除更正。无新增训练和定时任务修改。

- 2026-09-12：S050复用张量做像素流，记录队列组成偏移、背景控制损伤及真实局部边界补回。全量/逐实例负例不删，后续候选仅存清单、未启动新输入实验。

- 2026-09-12：S051固定失败类型×大小、局部邻居输入44例全部完成；记录实际个体恢复、原型输出线索、未编辑邻居副作用及反例。精确正常流程/固定框结果分开，层级注入尚未开始。
- 2026-09-12：S052全44例层级插入/恢复完成，记录同类误报P3可逆传播、邻居错误变化、混合格限制、自身遗漏不确定性与正反例；v1设备引用失败保留、v2精确重放通过。没有启动训练或下一格子拆分。
- 2026-09-12：S053同15例P3混合/独占格拆分完成，记录独占格可撤回响应、实际插入及区域排名的不确定性，保留等格数/背景/非加性对照。更新当前下一问为错误像素的候选归属信息，不继续盲扫格子或启动训练。

- 2026-09-13：S054固定15例候选归属响应完成，15/15真实邻居候选匹配；目标在leak像素胜率59.7%，GT邻居竞争诊断IoU+4.755点、12/15提高。记录其为诊断上限，主线收窄为可学习的候选间实例归属竞争；无训练。
- 2026-09-13：S056完成完整COCO的六个GT修复和两个自动竞争对照，量出同类修复对差距的特异收益；撤回S055组件并集泄漏和重复子型解释。开始同权重原生one-to-one路径完整比较S057，尚不据运行中状态写结论。
- 2026-09-13：S057已完成；原生one-to-one路径AP接近原路径、全分数R75较高，但P90没有改善且拥挤差距仍在。全部本轮进程完成，没有新训练/定时任务修改。


2026-09-13 S067—S069更新：完成失败类型独立GT掩码修复AP、官方差距记账与类别大小共同支持复算、联合失败支持代理和理想矩形范围干预；所有新增结论见F33—F35。同步纠正v5候选池、证据标签、近期pilot bootstrap及元数据。

2026-09-13 S070—S073：实际解码、候选源点查询、等覆盖排序与当前官方分配完成。记录源点直接替换不成立的具体配方及监督身份混杂；原型/系数深层根因未定，方法主线仍未验证。

# S078：裁切敏感性与失败状态交叉（2026-09-13）

状态：COMPLETE / 诊断。复用 S077 冻结的 train2017 readout cache，9,626 个已配对目标、1,483 张图；逐实例重放官方 crop/decode 的 0.8--1.4 尺度，并与候选框 IoU、GT-box 支持和 native Mask IoU 交叉。未新增前向、训练或推理规则。判定阈值为候选 Box IoU>=0.50、native Mask IoU>=0.75、GT 像素在候选框内>=0.95。

全体配对目标中，6,097 (63.34%) 为框好且掩码好；970 (10.08%) 为框好但 GT 支持不足且掩码差；2,559 (26.58%) 为框好、GT 支持充分但掩码差。后者的 native IoU 0.611、裁切 oracle 平均增益 +2.751 个 IoU 点 [2.525,2.988]，固定外扩 20% 平均 -4.064 点 [-4.272,-3.832]。高 ICI 配对目标中对应比例为 48.78% 掩码好、15.52% 支持不足且掩码差、35.70% 支持充分但掩码差；支持充分掩码差组 oracle +2.768 点 [2.342,3.247]，固定外扩 20% -4.693 点 [-5.199,-4.212]。

这支持“框已覆盖目标但掩码读出仍失败”的真实失败类，也支持裁切尺度有可恢复余量；不支持固定外扩作为部署方法。由于 S077 cache 只保留已配对候选，`box_bad_mask_bad` 与 `box_bad_mask_good` 均为零，不能据此估计框漏检/坏框比例或全端到端 AP。GT 仅用于回溯标签与评价；当前无 GT 自动选择器在 transfer 高组未稳定有效。详见 `experiments/coco_clean_20260911/diagnostics/crop_support_failure_cross_20260913/REPORT.md`、`SUMMARY.json`、`per_target.csv`。

# S079：推理可见裁切尺度选择器（2026-09-13）
状态：COMPLETE / 诊断性 no-go。复用 S078 的 9,626 个已配对 train2017 目标，在 fit 7,811 个目标上以 GT 派生的各尺度 native IoU 训练每尺度 ExtraTrees 回归器；transfer 1,815 个目标只使用推理可见的 raw 20% 边界阳性率、框内阳性率、边界/内部比、候选分数和预测类别选择 0.8--1.4 crop scale。该结果不是端到端 AP，也没有新增前向或训练模型。
transfer 基线 IoU@1.0 为 0.771219；选择器为 0.771343，整体仅 +0.0124 个 IoU 点。高 ICI 组 n=181 为 -0.1955 点，低组 n=1,634 为 +0.0354 点；框好、GT 支持充分但掩码差组 n=468 为 +0.7944 点，支持不足掩码差组 n=177 为 -0.6056 点，掩码好组 n=1,170 为 -0.2069 点。对应 transfer oracle 余量分别为总体 +1.535、支持充分掩码差 +3.493、高 ICI +1.742 个 IoU 点。
结论：最优裁切方向在“支持不足”与“支持充分但掩码差”之间相反；现有边界响应/分数/类别特征未能稳定识别，不能将裁切尺度选择器作为当前主方法。该结果保留裁切作为失败机制证据，并把后续主线收窄到能直接改善掩码读出或训练监督的机制。详见 `experiments/coco_clean_20260911/diagnostics/crop_scale_selector_probe_20260913/REPORT.md`、`SUMMARY.json`。

# S080：系数机会与固定原型限制归因（2026-09-13）

状态：COMPLETE / S074 复算。无新增前向、训练或 COCO AP。对 S074 的 140 个固定队列实例，比较原始系数、固定原型下的 split-fit 与 GT-assisted all-fit；仅对原始 IoU<0.75 的 94 个实例做“达到 IoU 0.75”归因。若 all-fit>=0.75，记为条件上的 `coefficient_recoverable`；否则记为 `fixed_prototype_limited`。后者只表示固定 P、框和支持域下的最优系数仍不足，不单独证明原型因果。

总体失败实例中，`requires_target_pixel_recovery` n=48，系数可修复 8/48 (16.7%)，固定原型仍低于 .75 的 40/48 (83.3%)，all-fit 平均增益 +18.79 个 IoU 点；`sufficient_true_pixels_but_residual_false_pixels` n=46，系数可修复 25/46 (54.3%)，固定原型仍受限 21/46 (45.7%)，all-fit 平均增益 +21.91 点。高 ICI 组分别为 4/24 (16.7%) 与 13/24 (54.2%)，低 ICI 组分别为 4/24 (16.7%) 与 12/22 (54.5%)。高低密度结构相同，不能据此宣称系数瓶颈是密集场景特有。

主线含义：掩码失败至少拆成两种机制。自身像素恢复不足大多不是只改系数就能解决；自身覆盖充分但残余误报中约一半存在较大的系数读出机会，因此可以继续研究共享系数修正，但必须在独立图像上训练/测试并报告完整 AP。S080 仍是 GT oracle 条件诊断，不能把 +18--22 点写成方法收益，也不能把剩余失败直接命名为原型失败。脚本 `experiments/coco_clean_20260911/coefficient_prototype_attribution_20260913.py`，报告 `experiments/coco_clean_20260911/diagnostics/coefficient_prototype_attribution_20260913/REPORT.md`。

# S081：COCO 图片级迁移的目标/邻居表示判别 probe（2026-09-13）

状态：COMPLETE / 机制诊断。复用冻结 readout cache 的 1,500 张 train2017 图片、9,626 个匹配目标；fit 图片拟合线性判别器，transfer 图片完全按图片留出。COCO GT 只定义采样像素的自身/同类邻居或自身/背景归属标签，不参与模型输入、训练或 AP 评价。每个目标每类最多平衡采样 32 个像素，共 12,439 个可用目标-任务记录。

transfer 图片级迁移结果（宏平均目标 AUC）：

| 特征 | 自身 vs 同类邻居 | 自身 vs 背景 |
|---|---:|---:|
| 位置坐标 | 0.488 [0.465, 0.510] | 0.499 [0.489, 0.509] |
| 候选头输入 h | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| 原型向量 P(x) | 0.698 [0.682, 0.715] | 0.907 [0.901, 0.914] |
| 实例系数 c | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| 标量 logit P(x)^T c | 0.937 [0.927, 0.947] | 0.951 [0.947, 0.954] |
| 逐通道贡献 P(x)⊙c | 0.946 [0.936, 0.955] | 0.951 [0.947, 0.955] |
| h + 逐通道贡献 | 0.946 [0.936, 0.955] | 0.951 [0.947, 0.955] |

解释：原型本身已含有中等（同类邻居）或较强（背景）像素区分信息；系数单独和同一候选内恒定的 h 按设计接近随机。真正的实例条件化读出 `P(x)⊙c` / `P(x)^T c` 已有很强的跨图像像素归属可分性，加入 h 或坐标几乎没有增益。因此当前证据不支持“候选头缺少空间坐标是主瓶颈”，也不支持“系数无法区分自身与邻居”作为单独根因。更合理的待查方向是：已有 logit 分离在低分辨率采样、阈值校准、支持域和训练损失中是否被损失；S078/S079 的框支持与裁切敏感性仍需与该表示证据按失败状态交叉。

边界：AUC 是冻结表示的可利用信息诊断，不是端到端方法收益；probe 使用 GT 定义标签，尚未满足“不用 GT 的可部署系数利用”门槛。不能据此恢复 CCL 训练或宣称密集场景因果链。脚本 `experiments/coco_clean_20260911/representation_ownership_probe_20260913.py`，报告 `experiments/coco_clean_20260911/diagnostics/representation_ownership_probe_20260913/REPORT.md`，摘要 `SUMMARY.json`。

# S082：表示判别的面积 × 失败状态交叉审计（2026-09-13）

状态：COMPLETE / 机制诊断。修正 S081 面积分层的 annotation_id 类型转换后，使用同一批 image-disjoint transfer 目标与 S078 失败标签，检查目标面积混杂及失败状态是否独立保留。无新增前向、训练或 AP；GT 只用于事后定义像素归属、面积档和失败分组。

面积对 `own_vs_same` 的 `proto_coeff` AUC 有明显影响：small (<1024) n=236 为 0.9237，medium (1024--9216) n=272 为 0.9434，large (>=9216) n=180 为 0.9779；`own_vs_background` 同样为 small n=611 0.9226、medium n=596 0.9619、large n=490 0.9730。因此面积必须作为协变量，不能把总体 AUC 差异直接归因于拥挤。

在同一面积档内，失败状态仍保持一致排序。`own_vs_same/proto_coeff`：large 中 mask_good n=155 为 0.99、support_low_mask_bad n=7 为 0.85、support_sufficient_mask_bad n=18 为 0.92；medium 分别为 n=185/0.98、n=29/0.82、n=58/0.88；small 分别为 n=90/0.98、n=40/0.82、n=106/0.92。`own_vs_background/proto_coeff` 也保持相同方向：large 0.98/0.86/0.91，medium 0.98/0.89/0.93，small 0.97/0.87/0.90（顺序均为 mask_good/support_low_mask_bad/support_sufficient_mask_bad）。

结合 S081，当前最稳妥的机制结论是：高 ICI 本身没有显著降低自身--同类邻居的判别能力（`proto_coeff` 高/低约 0.949/0.945），但高 ICI 下自身--背景判别下降（约 0.933 vs 0.953）；真正稳定的分叉来自 mask-bad 状态，且在面积匹配后仍存在。已有 `P(x)⊙c` 表示包含像素归属信息，失败更像支持域不足、低分辨率采样、阈值/校准、解码或监督目标导致的读出损失，而不是“系数相似”或“候选头缺坐标”单一根因。

证据文件：`experiments/coco_clean_20260911/diagnostics/representation_ownership_probe_20260913_area_stratified/target_auc_stratified.csv`、`STRATIFIED.csv`、`REPORT.md`。后续优先做同一候选在多 mask 解码分辨率及阈值下的可恢复性对照，再决定是否训练方法；不得把本轮 AUC 或面积内差异写成部署收益。

# S083：冻结解码网格与 logit 阈值探针（2026-09-13）

状态：COMPLETE / 解码机制诊断。复用 1,483 张 train2017 图片、9,626 个 box-good 配对候选的冻结候选框、原型和系数，只改变后处理采样网格及 logit 阈值。官方 `input640_t0` 对 S078 的逐目标 IoU 复现最大绝对误差为 0；因此后续差异不是评价器或候选错位造成的。GT 仅用于事后 IoU、覆盖率、邻居/背景错误和失败状态分组，没有训练或端到端 AP。

在 image-disjoint transfer 子集（1,815 个目标）上，官方解码 IoU 均值为 0.771219。统一提高 logit 阈值到 0.5 后为 0.774435（+0.003216），将后插值网格提高到 1280、阈值 0 后为 0.775765（+0.004546）。全体配对目标上，官方为 0.808850，阈值 0.5 为 0.811139，1280 网格为 0.812028。高 ICI 组的阈值 0.5 增益约 +0.0041，与低 ICI 组相近，不能称为密集特异收益。

按失败状态，transfer 中“支持充分但掩码失败” n=468 在阈值 0.5 下平均提高 +0.0153 IoU，而“支持不足且掩码失败” n=177 下降 -0.0134；掩码正常 n=1,170 仅 +0.0009。因而部分失败确实可由后处理阈值/采样恢复，但统一阈值会伤害支持不足的小目标，且后插值不增加新特征信息。当前机制定位更准确地表述为：已有实例条件化表示包含可用像素归属信息，但低分辨率采样、logit 校准和支持域共同造成读出损失；仍有一部分 mask-bad 失败无法由冻结解码重放恢复。

证据：`experiments/coco_clean_20260911/diagnostics/decode_grid_threshold_probe_20260913/per_target_arm.csv`、`SUMMARY.csv`、`REPORT.md`。边界：这不是可部署方法收益，也不能据此宣称阈值调整解决 COCO AP；后续若形成方法，必须用不依赖测试 GT 的状态估计或训练监督，并与全量标准 AP 对照。

# S084：创新 2--4 与 Full-Synergy 独立复验（2026-09-13）

状态：COMPLETE / 探索性方法复验。审计发现旧 `ortho_center` 的正交损失只作用于冻结缓存原型，对训练参数无梯度；其三个 checkpoint 与 BSR 逐元素完全一致，所以旧“正交增益”实际来自推理中心先验。旧 contrastive 也没有在相同空间坐标比较两个实例。复验重写同坐标、同图同类相邻对损失，并为可训练 32x32 基变换建立 active-orth 与无正交 adapter 精确控制；共 8 模式 × 3 种子 × 15 轮，所有 checkpoint 保留，梯度见证非零。

在固定 1,200 图 fit、300 图 image-disjoint transfer、统一官方等价解码和 COCOeval 下，S032 为 Mask AP 53.313 / High ICI R75 54.586 / gap 11.279。三种子均值：corrected Contrast 53.300/55.034/10.909，Ada 53.365/55.369/10.828，Adapter 53.407/55.145/11.071，Active Orth 53.398/55.369/10.965，Full-noorth+center 53.547/55.705/10.923，Full+orth+center 53.529/55.817/10.713，Ada+center 53.575/56.152/10.710。

精确消融不支持强机制断言：Contrast vs BSR ΔAP -0.006、High +0.224；Orth vs Adapter ΔAP -0.009、High +0.224；Full vs Full-noorth ΔAP -0.018、High +0.112。三者 gap 的 paired image-bootstrap 95% 区间均跨零。Ada vs BSR 仅 ΔAP +0.060、High +0.559，High CI [-0.421,1.754]；中心先验在 BSR/Ada 上分别 ΔAP +0.177/+0.210，但 High 与 gap CI 仍跨零。S080 中 48 个需要恢复目标真像素的失败实例，所有新方法三个种子均 0 个达到 IoU .75。结论：撤回旧“原型正交有效”；保留 Ada+center 为待验证的简洁候选，其现象更符合读出校准/支持域改善，尚未证明密集专项或目标像素恢复机制。transfer 集已被反复探索，不能作为最终测试；下一阶段应锁定 S032、center-only、Ada+center 后在未探索 COCO val2017 做端到端三种子验证。完整报告 `experiments/coco_clean_20260911/runs/innovation_reaudit_20260913/FINAL_REPORT.md`。

# S085：外溢区域监督对照（2026-09-14）

在冻结官方特征/原型的 1,200 张 train2017 图像上，只更新最后系数头；300 张图像作 image-disjoint transfer，训练 15 轮、3 个种子。普通 BCE 的 Mask AP 为 41.396、AP75 为 44.463、高 E4 R75 为 55.076；同类邻居加权为 41.430/44.279/54.188；背景加权为 41.426/44.284/53.934。普通 BCE 优于两种显式外溢加权，说明简单邻居或背景负监督不能作为方法结论。该 pilot 不是完整 COCO AP，后续应追查标签/采样支持和解码边界。

# S086：实例边界归属机制端到端训练已启动（2026-09-14）

直接使用本地 COCO pilot1000 train2017 与 1,576 张 dense val 图像、官方 yolo26m-seg.pt、640 分辨率、mask_ratio=1、AMP、15 轮、每轮保存 checkpoint。三臂为 baseline、`self_out`（自身覆盖+框外背景抑制）和 `self_out_neighbor`（再加同类邻居排他），每臂三个随机种子，单 GPU 顺序运行。8 图真实标签 smoke 已通过；正式队列从 baseline_s0 开始。结果尚未完成，不提前写入 AP 结论。

补充（2026-09-14）：本地 `baseline_s0` 保持运行；其余 8 个配置已迁移到远端 RTX 4090（`/root/autodl-tmp/boundary_coco_20260914`），由 `2590.boundary_train` screen 队列顺序执行。远端当前已进入 `baseline_s1`，首个 epoch 的 `results.csv` 已生成；远端每 epoch 当前约 225 秒，完整远端队列预计约 7--8 小时。该时间估计不构成结果结论。

# S087：边界归属 pilot 队列状态复核（2026-09-14）

本地 `baseline_s0` 已完成 15/15 轮，`BOUNDARY_TRAINING_COMPLETE.json` 报告 `finite=true`，训练集 1,000 张、验证集 1,576 张，官方 YOLO26m-seg、640、mask_ratio=1、batch=2、AMP；`best.pt`、`last.pt` 和 epoch checkpoint 均存在。该 pilot 的验证 CSV 最高 Mask AP50-95 为 0.31028（epoch 1），第 15 轮为 0.29476；这是 pilot 训练曲线事实，不作为完整 COCO 结论。

远端 RTX 4090 队列无异常，当前运行 `baseline_s1`，已完成 8 个验证行（epoch 0--7），当前最佳 Mask AP50-95 为 0.30077（epoch 1）；screen 名称 `boundary_train`，远端路径 `/root/autodl-tmp/boundary_coco_20260914`。远端仍将按队列继续运行剩余配置。

# S088：边界归属方法改为单种子筛选（2026-09-14）

按探索阶段成本控制，停止原定全量三种子队列。远端 `baseline_s1` 已完整完成 15 轮；原队列自动进入 `baseline_s2` 后在完成 3 个验证 epoch 时被主动终止，其目录仅为不完整运行，不纳入比较。新 screen `boundary_pilot` 仅顺序执行 `self_out_s0` 与 `self_out_neighbor_s0`，两者结束即停止。是否补 s1/s2 由单种子 Mask AP、高拥挤 R75、拥挤 gap 和外溢指标共同决定。

# S089：方法特异邻居反事实机制审计（2026-09-14）

状态：COMPLETE / 机制诊断。复用 S047 已冻结的 64 个 train2017 同类近邻、异类近邻和同类远邻图像对；邻居纹理与匹配背景控制使用相同填充值，目标候选沿用原图 8,400 网格的一对多 raw source index，编辑图不重新选择候选。由于当前 Ultralytics 默认启用 end2end 一对一输出，实验显式关闭 `head.end2end` 以重现 S047 的 source-index 路径；未改变模型权重。比较 S032、Ada、Center、Ada+Center 四个读出臂，各 3 个已训练种子；统一采用 640 双线性 logit 解码、原候选框裁切和阈值 0。共得到 3,684 条逐实例记录，41 对具有完整邻居/背景配对，配对效应 1,476 条。

在高 ICI 的 same_near 原图（14 个有共同记录的图像）上，相对 S032 的静态 logit margin 变化为：Ada 自身--邻居 +0.860、Ada+Center +2.360；自身--背景分别 +1.251、+3.753。对应的 mask IoU 变化分别为 -0.0013、-0.0003，Ada+Center 覆盖率变化 -0.0204；Center 的 mask IoU +0.0018、覆盖率 -0.0054。邻居减去匹配背景的配对交互中，same_near 全部 20 对的 Ada+Center 相对 S032 mask-IoU 差为 -0.00050，图像 bootstrap 95% CI [-0.00302, +0.00160]；覆盖率差 +0.00340，CI [-0.00116, +0.01112]；高 ICI 的 7 对中方向相同但样本更少。不同近邻和远邻的交互也没有形成稳定的掩码改善。

结论：Ada/Center 确实把输出 logit 间隔推开，说明它改变了读出校准/背景抑制；但这组固定候选反事实不支持“邻居存在时掩码退化显著减小”或“margin 提高必然带来覆盖改善”的因果链。高 ICI 掩码机制仍应表述为“更强的决策间隔伴随覆盖权衡”，不能把本实验写成密集专项掩码收益或邻居排他性证明。该实验不是 COCO AP，GT 仅用于固定空间区域评价；原始数据和协议见 `experiments/coco_clean_20260911/diagnostics/mechanism_counterfactual_20260914/metrics.csv`、`paired_effects.csv`、`summary.csv`、`protocol.json`，执行脚本为 `experiments/coco_clean_20260911/mechanism_counterfactual.py`。

# S090：Ada+Center 单种子端到端候选筛选（2026-09-14）

状态：COMPLETE / 单种子筛选。为决定是否值得补做三种子确认，在预先固定的 300 张 COCO train2017 transfer 图像（2,002 个普通 GT）上，复用相同的冻结 backbone 候选缓存，比较 S032、Ada、Center、Ada+Center 四个读出臂。所有臂使用相同的候选框、类别、置信度、原型和系数；仅改变读出，采用 640 双线性 logit 解码、原候选框裁切、阈值 0，并由官方 `pycocotools COCOeval` 计算分割 AP。该 transfer 划分已在项目中用于探索，不是最终盲测集；本轮只有 seed 0，不作为三种子最终结论。

修正后的结果为：S032 Mask AP 53.256%、AP50 74.666%、AP75 57.743%，高 ICI R75 54.698%，低 ICI R75 65.786%，高低 gap 11.088 个百分点；Ada 分别为 53.343%、74.692%、58.291%、55.369%、66.021%、10.652；Center 为 53.440%、74.273%、58.232%、55.705%、66.021%、10.316；Ada+Center 为 53.615%、74.429%、59.022%、56.711%、66.725%、10.014。

相对 S032，Ada+Center 的单种子变化为 Mask AP +0.359 AP 点、AP75 +1.280 点、高 ICI R75 +2.013 点、低 ICI R75 +0.939 点，高低 gap 缩小 1.075 点；AP50 下降 0.237 点。Center-only 也有正向 AP/R75 变化，但 Ada+Center 在 AP、AP75、高 ICI R75 和 gap 四项主指标均为四臂最佳。该结果支持把 Ada+Center 作为下一轮三种子确认候选，但不能单凭一个已探索 transfer 划分证明泛化、密集专项因果机制或最终 COCO 结论。结果文件：`experiments/coco_clean_20260911/diagnostics/final_candidate_s0_20260914/transfer_evaluation_summary.csv`；评价代码已修正 `Center-only` 分支，修改见 `experiments/coco_clean_20260911/eval_innovations_suite.py`。

补充：尝试将该候选直接扩展到全量 5,000 张 val2017 时发现，已完成的 S036 全量缓存只保存解码后的 mask、框、分数和类别，没有保存 Ada+Center 重放所需的原型/系数张量，因此不能从该缓存无损重算新读出臂。完整 COCO seed-0 确认改为重新执行 backbone 捕获；当前已生成 4,809/5,000 张逐图收据，期间发现并修正两个 Ultralytics 空索引边界（浮点索引、零候选 reshape），未将不完整结果作为结论。

# S091：Ada+Center 全量 COCO val2017 单种子确认（2026-09-14）

状态：COMPLETE / 全量端到端读出确认（seed 0）。重新运行官方 `yolo26m-seg.pt` backbone，在完整 COCO val2017 的 5,000 张图像、36,335 个普通 GT 上捕获原型、系数、候选框、类别和置信度；四个读出臂共享同一候选集合，GT 只用于官方评价和 ICI 分层。采用 S032、Ada、Center、Ada+Center 四臂，统一 640 双线性解码、候选框裁切和官方 `pycocotools COCOeval`。推理阶段通过 4,809 张断点收据续跑；期间修正了 Ultralytics 空 NMS 索引的 float-to-int 和零候选 reshape 两个边界问题，最终 5,000 张全部完成。

全量 seed-0 结果：S032 Mask AP 45.002%、AP50 67.061%、AP75 48.824%、High ICI R75 52.526%、Low ICI R75 62.371%、gap 10.977 pts；Ada 为 44.978%、67.042%、48.869%、52.698%、62.375%、10.850；Center 为 45.112%、66.988%、49.093%、53.022%、62.685%、10.809；Ada+Center 为 45.124%、66.952%、49.204%、53.194%、62.809%、10.692。

相对 S032，Ada+Center 的变化为 Mask AP +0.122 AP 点、AP50 -0.109 点、AP75 +0.381 点、High ICI R75 +0.667 点、Low ICI R75 +0.438 点、高低 gap 缩小 0.285 点。Center-only 已贡献主要增益（AP +0.109、High R75 +0.496）；Ada-only 的 AP -0.024，说明 Ada 不能作为独立主方法。当前最稳妥的方法定位是：利用预测框的软几何边界先验，在读出 logit 的框边缘和框外施加连续衰减，减少掩码外溢；Ada 自适应校准只作为可选辅助项。该结果是完整 val2017 的单种子确认，足以筛选最终候选，但仍需在同一训练预算下补 seed 1/2 才能作为三种子论文主结果。结果文件：`experiments/coco_clean_20260911/diagnostics/final_candidate_fullval_s0_20260914/summary.csv`、`COMPLETE.json`；执行脚本：`experiments/coco_clean_20260911/eval_final_candidate_fullval_s0.py`。

# S089：远端中断恢复状态（2026-09-14）

复核时发现远端 `boundary_pilot` screen 已死亡，`self_out_s0` 在第 7 个验证 epoch（结果 CSV 行 7）后停止；日志无 OOM、NaN、Traceback 或其他内部错误，GPU 已空闲。已有 `last.pt`（epoch 7 完成后的 checkpoint）和 epoch0--6 checkpoint 均保留。已向训练脚本加入 `--resume-path`，从该 `last.pt` 恢复训练；恢复进程当前已进入 epoch 8/15，GPU 正常使用约 17.6 GiB。独立 watcher 会在 `self_out_s0/BOUNDARY_TRAINING_COMPLETE.json` 出现后启动 `self_out_neighbor_s0`，避免 SSH 断开导致队列丢失。

# S090：YOLO26 推理分支核验及对现有实验的影响（2026-09-14）

在本地实际环境 Ultralytics 8.4.100 与官方 `yolo26m-seg.pt` 上核验：模型 head 的 `end2end=True`。`YOLO.predict` 新建 predictor 时，默认参数、`nms=False`、`nms=True` 三种调用均保持 `head.end2end=True`；显式 `end2end=False` 才切换到 one-to-many，并由后处理执行常规 NMS。当前版本的 `nms` 主要是导出相关配置，不能作为推理分支开关。远端训练环境此前也核验为 Ultralytics 8.4.100。

影响范围：边界归属训练的 `results.csv` 未传 `end2end=False`，因此训练过程中的验证指标来自默认 one-to-one/NMS-free 路径；各 arm 之间仍是公平成对比较，但不能直接与主协议的 one-to-many+NMS COCO 结果或基于该路径的失败分类混用。现有 `failure_dimension_analysis`、`canonical_eval_pipeline`、`probe_pixel_leakage` 等机制诊断显式设置 `head.end2end=False` 或 `end2end=False`，其 15,863 失败归因及 4,417 等计数属于 one-to-many+NMS 主路径。后续边界方法筛选必须对每个 checkpoint 统一补做两条路径的验证，并在结果中记录 `head.end2end`、参数和后处理。

已修正 `experiments/coco_clean_20260911/EXPERIMENT_SPEC_STANDARD.md` 第 4 节，按 8.4.100 的实际行为将主评估写为 `end2end=False` 的 one-to-many+NMS，将 one-to-one/NMS-free 作为结构对照；旧的 8.4.143 泛化描述不再用于当前环境。

# S091：YOLO 参数四步决策门槛（2026-09-14）

后续所有参数先核对当前锁定版本的官方默认值；官方默认值不改 baseline。随后判断参数是否为研究中的因果变量；不是因果变量则在 baseline、方法和对照中统一固定。任何修改还必须区分单纯提高分数的 hyperparameter tuning 与检验明确假设的 ablation/diagnostic。只有已有实验证明某机制造成稳定失败后，才允许修改该机制并将其作为方法创新。参数变更需记录默认值、当前值、因果变量性、实验类型、假设、指标和停止条件。

# S092：配置规范修正：官方 YAML 默认值与运行环境覆盖（2026-09-14）

修正原 S091 的过宽表述：数据集 YAML 和训练/验证配置分开管理。数据集 YAML 只允许按机器调整 `path`/图片清单路径，COCO `names`、类别顺序、类别数和 split 语义保持官方定义。训练/验证配置从当前锁定版本 Ultralytics 8.4.100 的官方 `default.yaml` 起步；`batch` 可因显存覆盖，但它仍影响梯度估计、每轮更新次数和优化轨迹，必须在成对实验中相同并报告有效 batch、`nbs` 和 step 数。

同类运行环境覆盖包括 `device`、`workers`、`cache`、结果目录参数（`project`/`name`/`exist_ok`）和记录参数（`save`/`save_period`/`plots`/`verbose`/`profile`）。这些不作为方法变量，但需成对固定和记录；`cache` 变更标签后必须清除重建。`amp`、`deterministic`、`seed` 不属于纯运行参数，分别影响数值过程或复现控制；`imgsz`、优化器/LR、epoch/预算、增强、`rect`、`multi_scale`、`overlap_mask`、`mask_ratio`、`end2end`/`nms`/阈值等均影响训练、推理或评价，必须保持官方默认或预注册为因果变量。

规范文件已更新为 1.1：`experiments/coco_clean_20260911/EXPERIMENT_SPEC_STANDARD.md` 第 3.1 节。当前边界归属 pilot 的显式覆盖（如 `batch=2`、`workers=0`、`mask_ratio=1`、`epochs=15`、AdamW 与学习率设置）应作为该 pilot 的实验协议记录，不能回写成官方 baseline 默认值。

# S093：YOLO26-seg 全系列发布权重的 mask_ratio 核验（2026-09-14）

当前 Ultralytics 8.4.100 的全局 `default.yaml` 将 `mask_ratio` 设为 4，但这不是官方发布 COCO 权重的历史训练参数。直接下载 Ultralytics assets v8.4.0 的 `yolo26n/s/l/x-seg.pt`，并连同项目现有 `yolo26m-seg.pt` 读取 checkpoint `train_args`：五个尺度均记录 `mask_ratio=1`、`overlap_mask=True`、`imgsz=640`；权重内版本均为 8.3.222。记录的 epoch 分别为 n=245、s=50、m=80、l=60、x=40。官方 YOLO26 论文正文未检索到 `mask_ratio` 字段，因此实验复现应以发布 checkpoint 元数据和实际归档 `args.yaml` 为配置证据，不能从通用默认 YAML 反推权重训练历史。

当前边界归属正式 pilot 的训练脚本和已生成各运行目录 `args.yaml` 均显式使用 `mask_ratio=1`，所以 baseline 与方法臂在该变量上公平，并与官方发布权重元数据一致。此前 `aligned_benchmark`、`pilot_dual_branch` 的运行目录明确使用 `mask_ratio=4`；这些比较在各自组间仍公平，但属于 ratio4 微调实验，不能与官方 ratio1 预训练配方混称。冻结权重的纯推理不读取 GT，`mask_ratio` 不参与预测；涉及训练标签、验证 loss 或基于 loader GT mask 的诊断时仍必须记录该值。

# S094：按失败对象进行局部重推理的适用边界（2026-09-14）

状态：COMPLETE / 诊断。六类失败与成功对象各 48 例，比较 640 全图、1280 全图和 GT 定义的两种 ROI 上下文。只有无最终槽位组出现稳定改善：1280 的 Mask IoU +0.1721 [0.0736,0.2637]、Mask75 恢复 7/48；四倍上下文 ROI +0.1024 [0.0282,0.1818]、恢复 8/48。其余失败类无稳定净改善，成功对照在 1280 下平均 -0.0548。当前同类最佳框候选无法精确重放历史 `prediction_slot`，因此结论仅适用于当前配对诊断。停止把通用 ROI 重推理用于所有失败类。[报告](C:/Dpan/codexproject/paper-disc/diagnostics/failure_object_roi_probe_20260914/REPORT.md)

# S095：无最终槽位的候选生命周期与旧支路停止结论（2026-09-14）

状态：COMPLETE / 完整 COCO val2017 诊断。36,335 个普通 GT 中当前无最终槽位 3,699 个；第一次失败位置为原始几何 1,459、正确类别 1,033、0.001 分数门槛 745、NMS 184、eval100 135、top300 93、分配竞争 50。87.5% 在 NMS 前死亡；原始几何失败中 1,429/1,459 为 small。Proto26 语义残差对无槽位目标的 target/background AUC 为 0.644，但前10自动峰命中21.9%，未超过控制所需的可选择性；语义纠错在192个类别失败上最好top1仅8.3%。停止语义峰ROI和跨头类别融合。[生命周期报告](C:/Dpan/codexproject/paper-disc/diagnostics/no_final_slot_lineage_20260914/REPORT.md)、[语义报告](C:/Dpan/codexproject/paper-disc/diagnostics/semantic_residual_missing_instance_20260914/REPORT.md)、[跨头报告](C:/Dpan/codexproject/paper-disc/diagnostics/cross_head_class_rescue_20260914/REPORT.md)

# S096：小目标原始几何失败的配对表型和分辨率干预（2026-09-14）

状态：COMPLETE / 384 对同类别、最近面积匹配、图像互斥对象。失败组 P3 局部正确类分数 0.1267 对控制 0.2367，局部框 IoU 0.2105 对 0.5760，中心归一化误差 0.4602 对 0.1719；分类与框几何在同一目标局部共同退化。失败关联低 Lab 对比（配对差 -10.34）、输入短边更小（-1.87 像素）、邻近暴露/重叠/背景梯度更高；18表型分组CV AUC 0.705，只作描述。1280 相对640使失败 Box IoU +0.2025 [0.1767,0.2271]，恢复178/384 Box50；控制仅+0.0275。清除保留检测中心后的P3剩余峰对可恢复失败前20命中仅3.37%，停止直接峰值路由。[报告](C:/Dpan/codexproject/paper-disc/diagnostics/small_raw_geometry_origin_20260914/REPORT.md)

# S097：低可分性与近邻干扰的受控干预证据及 P3 层定位（2026-09-14）

状态：COMPLETE / GT 辅助机制诊断，不是推理方法。384 个 small 原始几何失败中，增强目标/干净背景对比使最佳原始框 IoU +0.0700 [0.0570,0.0837]、恢复102个Box50；移除近邻 +0.0369、恢复82个；联合 +0.1022 [0.0862,0.1188]、恢复146个，成功控制联合仅+0.0067且区间跨零。剂量7/14/28/42对应恢复46/67/102/141；软轮廓仍恢复61，平移伪干预仅23。近邻模糊/均值化恢复124/143，成功控制平均-0.0005/-0.0077。层内供体替换中P3全图/局部复现+0.1009/+0.1005，P4+0.0197，P5+0.0045；主要变化在框几何而非类别分数。支持该失败类的目标局部 stride-8 表征受弱目标证据和邻居竞争共同影响，不支持泛化为所有拥挤或掩码失败。[报告](C:/Dpan/codexproject/paper-disc/diagnostics/small_object_causal_interventions_20260914/REPORT.md)

# S098：受控干预引导 P3 定位的首个训练原型（2026-09-14）

状态：RUNNING / seed 0 成对筛选。方法训练期从真实 GT 中选择脆弱小目标，构造减弱近邻并增强目标对比的特权教师视图；固定教师只选择比学生明显更好的 P3 单元，学生在原图上接受该单元的辅助框监督。推理路径不改变、无 GT、无二次前向。16图 smoke 已完成8批，选择16个目标、11个通过教师改善门槛，峰值显存约9.5GB且损失有限。正式首轮按官方权重可由8.4.100执行的保存训练参数，在相同1,000图、1,576图验证、seed0和1 epoch上顺序运行 baseline 与方法；未看结果前门槛为正常指标与目标失败转移共同改善，否则停止该配方。[协议](C:/Dpan/codexproject/paper-disc/experiments/counterfactual_p3_distillation_20260914/protocol.json)

# S099：干预引导 P3 监督的一轮筛选与门控收缩（2026-09-14）

状态：ONE-SEED FEASIBILITY COMPLETE / 三轮复验运行中。相同官方起点、seed 0、1,000 张训练图、1,576 张/20,827 实例验证集、batch 2 和 1 epoch 下，Baseline 的 Box/Mask AP 为 42.554/34.156。宽门控版本（729/980 个目标生效，权重 2.0）为 42.069/33.672，分别下降 0.485/0.484 AP 点，因此拒绝该门控。收缩版本只在学生原图位置 Box IoU<.50、教师干预位置 Box IoU≥.50 且教师至少高 .10 时生效（316/980，权重 .5），Box/Mask AP 为 42.485/34.113，相对 Baseline -0.069/-0.043；Mask AP50 +0.003，视为一轮筛选中的全量近似持平，不能称提升。

在冻结的 384 个 small 原始几何失败上，收缩版相对配对 Baseline 的 raw best Box IoU +0.00691，图像 bootstrap 95% CI [0.00292,0.01115]；最终同类 Box IoU +0.00932 [0.00390,0.01490]，Mask IoU +0.01648 [0.00733,0.02568]。raw Box50 为 12 恢复/7 损失（净 +5，但二值差区间跨零）。384 个类别/面积匹配 small 对照的 raw Box50 为 2/2（净 0），最终 Box50 为 6/7（净 -1）；对照 Mask IoU 也提高 +0.02302，因此不得把 mask 改善写成失败特异效应。

当前只支持“受控干预发现的 P3 定位缺陷可以转化为选择性训练信号，且连续失败质量在一轮筛选中改善”的可行性结论；尚未证明标准 AP、密集特异性或多种子泛化。远端 RTX 4090 已启动相同环境内的 3 epoch Baseline/收缩方法 seed-0 顺序配对，路径 `/root/autodl-tmp/counterfactual_p3_20260914`、screen `cfp3_pair`。只有该轮继续保持失败组增益且无实质 AP 代价，才补 seed 1/2。[完整报告](C:/Dpan/codexproject/paper-disc/experiments/counterfactual_p3_distillation_20260914/REPORT.md)

# S100：干预引导 P3 监督三轮 seed-0 复验（2026-09-14）

状态：COMPLETE / eligible for seed 1/2。远端 RTX 4090 使用相同 1,000/1,576 split、batch 2、Ultralytics 8.4.100、Torch 2.8.0 运行 3 epoch Baseline/救援门控方法。rescue-only 相对 Baseline 的每轮 Mask AP 差为 -0.071、+0.170、+0.326 AP 点，Box AP 差为 -0.091、+0.129、+0.238；三轮均值分别 +0.142/+0.092，曲线非单调，因此是可行性证据而非最终主结果。

远端初次评价使用转换标注时因实例 ID 重编号而正确失败；随后改用原始 COCO `instances_val2017.json`，并限制到远端实际存在的 438 个目标（239 个 raw-geometry-small 失败、199 个匹配控制）。失败组 rescue-only 的 raw best Box IoU +0.02067，最终 Box IoU +0.01712，最终 Mask IoU +0.02021；raw Box50 19 恢复/10 丢失（净+9），最终 Box50 10/5（净+5）。控制组 raw Box50 3/3（净0），最终 Box50 8/3（净+5）；两组均无 Mask75 新恢复。因此方法改善了诊断失败组的连续几何质量，但掩码改善不是纯失败特异，不能夸大为密集专项 AP 增益。

结论：rescue-only 通过单种子短程 feasibility gate，允许补 seed 1/2；尚未证明多种子稳定性、最终 COCO AP 或密集特异性。[报告](C:/Dpan/codexproject/paper-disc/experiments/counterfactual_p3_distillation_20260914/REPORT.md)

补充（2026-09-14）：S100 闸门通过后，远端 screen cfp3_confirm 已启动 seed 1/2 的 Baseline 与 rescue-only 三轮配对确认；所有种子固定 batch=2、epochs=3、同一 COCO pilot 与 1,576 图验证集。

# S101：远端实例中断与 checkpoint 恢复（2026-09-14）

远端实例曾在 rescue-only seed 1 的第 2/3 轮训练中断，screen `cfp3_confirm` 进入 dead 状态，GPU 空闲；目录和 `runs/cfp3r_s1/weights/last.pt` 保留。检查确认 Baseline seed 1 已完成 3/3，rescue-only seed 1 在中断前完成第 1 轮并进入第 2 轮约 82%。已上传支持 `--resume-path` 的训练脚本，从该 `last.pt` 成功恢复为总 epoch 3；恢复后自动继续 seed 2 Baseline 与 rescue-only。当前新 screen 为 `cfp3_confirm_resume`，远端目录仍为 `/root/autodl-tmp/counterfactual_p3_20260914`。此次恢复不重跑已完成的训练，也不改变配对协议。

# S102：干预引导 P3 监督三种子短程确认（2026-09-14）

状态：COMPLETE / 主线候选通过短程确认。相同 1,000 图训练子集、1,576 图验证集、3 epoch 固定预算下，rescue-only 相对成对 Baseline 的最终轮 Mask AP 在 seed 0/1/2 分别为 +0.326/+0.576/+0.247 AP 点，均值 +0.383、样本标准差 0.172；Box AP 分别 +0.238/+0.835/+0.101，均值 +0.391。按每个 arm 独立选择最佳 Mask AP checkpoint 时，三个种子差值为 +0.170/-0.064/-0.075，因此不能只凭固定第 3 轮宣称最终 AP 稳定领先。

使用各臂 `best.pt` 在相同 239 个 raw-geometry-small 失败和 199 个匹配控制上评价。失败组 raw best Box IoU 三种子为 +0.02067/+0.00969/+0.00742，raw Box50 为 +0.03766/+0.02929/+0.00837，最终 Mask IoU 为 +0.02021/+0.03570/+0.00284，三个种子方向均正；最终 Box50 在 seed 2 为 -0.00418，三种子均无 Mask75 新恢复。控制组部分最终指标也提高，所以方法改善尚非纯失败特异。

结论：从失败生命周期、像素干预、P3 移植到选择性训练的链条已达到可作为论文机制主线和首选方法候选的强度。当前失败链及定向评价走 one-to-many + NMS，训练 `results.csv` 是默认 one-to-one 验证；正式实验必须显式并列或统一分支。当前结果仍是短程 pilot；最终主结果必须使用固定的充分训练预算、预声明 checkpoint 规则及完整/未探索 COCO 评价。完整证据见 `experiments/counterfactual_p3_distillation_20260914/THREE_SEED_RESULTS.md`。

# S103：丰富评价修正方法判断（2026-09-14）

状态：COMPLETE / 机制主指标通过、当前方法保留为首选候选。新增官方 COCOeval、AP50/AP75/APS、AR、固定 Precision Recall、raw P3/P4/P5 分层、中心/尺度误差、候选生命周期、Mask coverage/purity、同类邻居/异类/背景外溢、边界 F1、失败减控制交互、阈值转移、失败状态转移、像素总量 micro-average 及面积/ICI/E4/对比度分层。原始 COCO 标注上的 `best.pt` 官方 Mask AP 三种子变化为 +0.205/-0.063/-0.176 点，均值 -0.011±0.196；当前短程 pilot 的总体标准分割效果近似中性。

one-to-many + NMS 的固定失败组中，raw P3 best Box IoU 三种子 +2.171/+1.024/+0.555 点，失败减控制交互 +2.270/+1.502/+0.718，raw center error 三种子均下降；raw Box50 恢复/退化为 19/10、21/14、9/7。该训练信号确实作用于预期 P3 定位环节。Mask IoU 的失败减控制交互为 +1.396/-1.194/+0.038，Mask75 无稳定恢复；coverage、purity 与泄漏指标存在种子差异。像素 micro-average 也显示背景误差总体下降，但同类邻居误差方向不一致。

结论：论文主线收敛为“由失败生命周期和受控层内干预定位 small-object P3 几何失败”。当前 rescue-only P3 监督保留为首选方法候选，因为与假设直接对应的 raw P3 Box IoU、center error、raw Box50 及失败减控制交互在三个种子均改善，最终 Box/Mask IoU、纯度和背景泄漏提供支持。评价按机制主指标、支持指标和总体性能护栏组织，不要求全部指标同时提高。下一轮按 `experiments/counterfactual_p3_distillation_20260914/EVALUATION_PROTOCOL.md` 做充分预算确认。证据见 `RICH_EVALUATION_RESULTS.md`、`METRIC_VALUE_ASSESSMENT.md` 及其 CSV。这里的“干预”指固定模型上的受控输入/特征修改，不表示已经定义或估计严格的反事实因果效应。

# S104：跨种子实例配对效应（2026-09-14）

将同一实例的三个种子方法减基线效应先求均值，再对239个失败实例 bootstrap：raw P3 best Box IoU +1.250点，95% CI [+0.705,+1.829]；raw center error -1.709[-2.877,-0.623]；最终 Box IoU +1.597[+0.819,+2.406]；最终 Mask IoU +1.959[+1.237,+2.722]；target coverage +3.966[+1.991,+5.953]；prediction purity +1.983[+1.044,+2.933]；boundary F1 +3.869[+2.490,+5.320]。135个完整失败/控制匹配对上的失败减控制交互：raw P3 Box IoU +1.496[+0.490,+2.562]、center error -2.450[-4.068,-1.012]、boundary F1 +2.417[+0.069,+4.692]。这支持“指定P3定位失败被特异修复，并传导到若干最终掩码质量维度”；Mask IoU的失败减控制交互仍跨零，不能称全部掩码改善均为失败特异。证据：`experiments/counterfactual_p3_distillation_20260914/rich_across_seed_target_effects.csv`。

# S105：受控干预引导 P3 救援的完整 COCO 单种子筛选（2026-09-14）

状态：RUNNING。远端完整 COCO2017 已通过 Ultralytics 8.4.100 官方 `convert_coco(use_segments=True)` 转换和全量审计：118,287/5,000 张训练/验证图，849,942/36,335 个实例标签，非法类别或坐标为 0，bbox 回退标签为 0；`merge_multi_segment` 保证一个多段 COCO annotation 仍对应一个训练实例。训练集比原始非 crowd 正面积 annotation 少 5 行，精确对应原标注中 5 个相同图像、类别和 bbox 的重复项，由官方转换器去重。

seed 0 的一轮全量成对筛选已在 RTX 4090 的 `fullcoco_pair` screen 中启动，顺序为 Baseline、rescue-only P3 方法；二者使用相同官方预训练权重、权重保存的可执行训练参数、数据、随机种子、epoch、batch 2 和 workers 8。batch 4 起跑在 72/29,572 step 时因 Baseline 密集批次已达 19.6 GiB 而主动废弃，未进入结果。方法教师改为在学生可微前向前只生成 P3 位置和教师 IoU，释放教师特征后再执行学生前向；救援门槛与辅助损失不变。训练结束后将自动运行完整 COCO val 的 one-to-one 官方预测/COCOeval，以及冻结失败/控制对象的 one-to-many+NMS 丰富评价。评价允许单个有科学价值且与机制对齐的指标构成正面结果，其他指标用于界定作用范围和代价，不要求全部提高。协议与状态见 `experiments/counterfactual_p3_fullcoco_20260914/`。

# S106：反事实术语降级为受控干预（2026-09-14）

当前输入编辑只是在相同观测图像上弱化邻居、替换局部背景或增强目标对比，并观察固定模型的输出变化。项目没有定义结构因果模型、潜在结果、可识别条件或 direct causal effect，因此论文不再把该方法称为严格的 counterfactual/causal learning。当前统一名称为 **受控干预引导的 P3 救援（Controlled-intervention-guided P3 rescue）**。历史目录、脚本函数与运行 ID 中的 `counterfactual` 只作为溯源标识保留。规范见 `project/guidelines/INTERVENTION_TERMINOLOGY.md`。

# S107：小目标 P3 失败的邻居特异性对照（2026-09-15）

状态：COMPLETE / 机制主张门槛通过。对冻结的 384 个 small 原始几何失败和 384 个同类别、最近面积匹配成功对照，按输入空间暴露量预先选择近邻，不查看模型响应。302 个失败目标同时具有至少 8 个可干预近邻像素及等形背景对照。目标 GT 像素在全部干预中逐像素保持不变。

失败组中，预选近邻平坦化相对原图使 raw P3 最佳 Box IoU 提高 +13.975 点；对等形、等面积背景作同色平坦化仅 +0.111 点。预注册主对照“近邻平坦化−背景平坦化”为 **+13.864 点，95% CI [+11.674,+16.141]**，P3 中心误差净下降 12.455 个归一化百分点，P3 Box50 净恢复比例 +39.404 点。近邻模糊−等形背景模糊仍为 +10.438 [+8.460,+12.465]；近邻平坦化−等像素远端实例平坦化为 +13.493 [+11.300,+15.736]。背景和远端平坦化的平均像素编辑能量均高于近邻平坦化，却近似无恢复，排除了“任意等量图像简化”解释。

在原有 pair_id 完整匹配的 214 个失败/成功对上，主对照的失败减成功交互为 +14.604 [+11.591,+17.662] 点；按预注册严格定义有 105/302 个邻居特异恢复实例。探索性分组中，异类邻居与同类邻居的净效应分别为 +15.650 和 +11.797 点，均排除零。因此当前机制对象是**特定近邻内容敏感的小目标 P3 定位失败**，不是“同类系数相似”或笼统的高 ICI 失败。该结果是 GT 辅助受控诊断，不是推理方法或 AP 结果。[报告](C:/Dpan/codexproject/paper-disc/diagnostics/neighbor_specific_recovery_20260915/RESULTS.md)

## 论文撰写与口径补充（2026-09-15；非新增实验）

已根据 S095—S107 的现存记录写成[干预引导 P3 监督中英文完整初稿](C:/Dpan/codexproject/paper-disc/paper/manuscripts/neighbor_sensitive_p3/README.md)，配有同源生成的五组表格、三幅图和[证据对应表](C:/Dpan/codexproject/paper-disc/paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_MAP.md)。当前方法名称采用 IG-P3（Intervention-Guided P3 Supervision）；不改历史运行 ID。

写稿核对代码后补充：训练 gate 检查教师选定位置上的学生失败，并不要求学生所有候选都失败；教师看联合编辑视图，GT 仍是回归目标。S103/S104 的 raw_center_error_norm 使用全部层级最佳框，不能直接称为 P3 中心误差。最终掩码质量通过 GT 辅助选同类最佳框关联，不是官方一对一召回。P3 特征替换供体来自联合编辑，不能与 S107 单邻居实验视为同一因子设计；“目标像素不变”限原始分辨率，背景控制未匹配目标距离。S101 恢复时 `_setup_train` 会从恢复学生重新创建教师，因此无中断固定教师的最终重复仍需另行完成。原始数值未更改，以上限定已写入正文，避免把真实改善扩大为尚未测得的结论。
