# 评价指标最小化与 GT–TAL 分类语义（2026-10-12）

**性质：**跨研究线评价定义修正与指标精简。保持所有已完成 Run、历史分类和数值不变。本文件优先于早期“每个新方法必须同时报告全部五层”的泛化要求；单个 Study 的预注册主指标与停止门槛仍按其原协议执行，不得在看过验证结果后替换。

## 0. 科学问题先于指标

新实验只要求一项直接检验研究命题的主指标、必要的对照/损伤护栏和可复现的数据身份。辅助指标必须写出：①研究对象；②回答的不可替代问题；③若改变将作出的决策；④新增代价；⑤若删除会失去什么。**如果与现有指标数学等价、或其值不影响任何判断，就不进入通用必报集合。**

指标、筛选规则和对象身份必须分开。BoxIoU≥0.5 最大基数几何 Hungarian 配对是一个**GT 辅助的候选关联算子**，不是官方评价指标，也不是官方 TAL。它不应成为通用失败选样前置条件；只在明确研究“几何位置的潜在读出能力”时作为独立实验臂保留。

## 1. 官方训练与评价是两个不同过程

- **官方训练：** 对指定 one-to-many / one-to-one 分支通过官方数据处理、标签和 TAL 得到正样本并计算相关 Loss。重放当前冻结 checkpoint 的 TAL 只是“当前无增强输入下的分配”，不能假定等同历史增强训练每一步的分配。
- **官方验证/输出：** 正常无 GT 推理结果经选定分支、置信筛选、预算与 COCO 匹配输出 Mask AP/AR。**TAL 不参与推理后的 COCOeval GT 匹配**。
- Ultralytics 当前 YOLO26 官方文档明确：**默认推理/验证使用 one-to-many + NMS；显式 nms=False 使用 one-to-one 无 NMS**。过去项目将“官方推理通常是 one-to-one”写作默认规则不准确。研究对象若是固定 one-to-one 的合法独立实验，需明确记录 `nms=False / one_to_one` 及该版本实际的模型/代码调用；不能用当前在线文档反推旧 Run 的真实开关。验证必须显式指定 COCO 数据集，避免未找到数据时回退到小样本数据集。
- 官方训练 overlap owner mask（用于监督 Loss）和最终原图 COCO annToMask（用于实例 Mask IoU/COCOeval）不同；它们各有目的，不能把损失换成官方 overlap 就称完成了官方验证。

官方文档：
https://docs.ultralytics.com/guides/end2end-detection
https://docs.ultralytics.com/tasks/segment
版本固定的分配代码：
https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/utils/tal.py

## 2. 默认最小指标矩阵（并非每次全量必报）

| 科学问题 | 首选主要观测 | 条件成立时才扩展 | 不应混用 |
|---|---|---|---|
| 一个实际方法是否提升部署分割？ | 正常输出 `COCO segm AP@[.50:.95]`，与同设置基线配对 | Box AP、Mask AP75、APsmall、原成功损伤/耗时（与声明有关时） | TAL 正样本 mask IoU ≠ COCO AP |
| 某 GT 是否在真实输出失败？ | 正常输出所选分支在预定阈值（如 Mask75）下的 GT 任务匹配与 FP 竞争 | 关联预测的 Box/Mask IoU、类别/分数/保留事件 | 不能用 GT 挑最佳 raw 代替最终结果 |
| 当前 TAL 正样本掩码是否差？ | 每个 GT 的 TAL 分配数、被分配 raw 的 Box/Mask IoU | 相关的官方 Mask BCE/梯度、pixel TP/FP/FN、面积/尺度 | TAL 关联不等于历史训练轨迹/推理匹配 |
| 有没有其他 raw 能提供好掩码？ | 在固定分支 `exists_joint75` + 其 raw 身份 | 仅当判断类别路径时增加 argmax 条件；当分辨框/掩码错位时增加独立 best box/mask | raw GT-oracle ≠ 无 GT 改进 |
| Mask 错误应补自身还是删误报？ | `coverage=TP/GT` 和 `false_area_ratio=FP/GT`（以同一输出 mask、GT、区域定义） | 区分 background/neighbor FP、support、Boundary IoU 或 logit AUC（有机制假设时） | purity、IoU 从这些量可推导，不能全部当独立机制证据 |
| 共享修正是否能拟合/泛化？ | 固定同候选的 dev/val image-macro ΔMask IoU 与图像簇 CI | fit 上 Loss、gradient_inf、oracle gap（仅为优化/可达性解释） | 不跨旧 160/官方 640 直接比较 Loss 数值 |

**精简默认值：** 一次“方法效果”Run 不必强制生成 raw_all/argmax/TAL/oracle/image_exact_n/boundary/AUC/FPR 全套数据；若某模块与研究假设无关则不算本 Run 的缺失项。“论文总证据图”可以跨已完成 Run 聚合不同对象，但不可混为同一分母。

`IoU = coverage / (1+false_area_ratio)`（GT 面积非零，背景/FP 范围一致），`purity=coverage/(coverage+false_area_ratio)`；这是数学上重复的质量描述，可按解释力选用，不应各自重复统计显著性。`image_macro` 和 `candidate_mean` 不等价，选择哪项为主需由科学总体（图片还是候选）决定；非主指标只在检验图像目标数差异影响时附报。

## 3. GT 分类可以映射到 TAL，但不是直接重贴标签

**主键保持** `(dataset, image_id, annotation_id)`，并记录 checkpoint、分支和真实输入几何。对于每个 GT g：

- `T(g)` = 当前无增强 checkpoint 官方 TAL 正样本中 `target_gt_idx` 指向 g 的 raw **集合**，按官方标注顺序正确还原 annotation_id。不能假定一定有正样本，也不能让多个正样本不经处理隐式地平均为单个对象。
- 每个 (g,r) 仍用本 raw 的原生预测框、原型、系数，按固定解码与原始 COCO annToMask 计算 `B(r,g)=BoxIoU`、`M(r,g)=MaskIoU`。这是**对 TAL 分配位置的诊断性评估**；训练 Loss 的 overlap owner GT 单独计算。
- 首先记录 `n_tal(g)=|T(g)|`。无 TAL 正样本为 `unassigned_at_this_replay`，**不是“模型根本没有 good raw”**。
- 有 TAL 正样本时：`joint_good` = ∃同一 r 满足 Box75 ∧ Mask75；`box_only` = ∃Box75，但不存在 Mask75；`mask_only` = ∃Mask75，但不存在 Box75；`separate_good` = 分别存在 Box75/Mask75、但没有同一 raw 双达标；`neither` = 都不存在。五类在 T(g) 非空条件下互斥穷尽，阈值可为预注册的其他值但不可事后改。
- 训练监督位置上得出的“box_only / mask_only / neither”只能描述观测质量。是否因为 TAL 的位置选择而导致 Mask 失败，仍需要同 GT、同 branch、同形状的固定候选对照；不允许直接推断修改 TAL 会提高 AP。

**注意：** one-to-one 的一次 TAL 回放通常每 GT 至多保留一个正样本；one-to-many 往往多个。统一用集合定义，不假设空集或多候选不可能出现。GT 身份来自标注与分配记录，不能用数组行号冒充 annotation_id。

## 4. 怎样把 TAL 分类与真实失败连接起来

为每个 GT 只保留两个基本可观测字段：

- `output_state(g)`：所选分支按预定 COCOeval/Mask75 任务设置的真实结果：`success` / `failure` / `ignored` / `unknown`。
- `tal_state(g)`：`unassigned` / `joint_good` / `box_only` / `mask_only` / `separate_good` / `neither`。

配对交叉表与 GT 身份一一对齐，再回答：同一 GT 真实输出失败时，其 TAL 正样本是否具有好 Mask？正常输出成功时 TAL 是否同样具有好 Mask？两者不一致说明“训练位置质量 ≠ 任务成功”，不自动提供因果解释。

仅当发现 `output=failure` 且 `tal=box_only/neither`，并提出“其他 raw 可能更适合作 Mask 监督”的假设时，才展开 `R_all`/`R_argmax` 并记录可用 donor；若研究仅针对当前 TAL 梯度或固定头优化，不必扫描全 raw。

历史旧 Hungarian 选择如需比较，必须在同一 GT、同一 branch、同一 forward 下与 TAL 位置成对记录：选中位置、GT-free 分类/分数/输出状态和 Mask 差。它是独立的反事实候选策略，**不是**迁移后 GT 分类的默认条件。

## 5. 通过标准 / 停止条件

- **对象 Gate：** 验证实际模型版本、权重哈希、输入几何、分支、dataset/annotation ID、TAL owner→annotation_id 映射；同名字段跨分支/尺寸不得视为相同候选。
- **评估 Gate：** baseline 冻结预测与当前输出一致，GT 使用原始 COCO；条件 GT 子集不得冒充完整数据 COCO AP；旧失败标签不能直接复用，必须重算。
- **因果 Gate：** 仅重新打标签可以发现相关性或总体差异，不能证明 TAL 造成错误。要提出修正方法，必须预注册同总量对照，检查任务评价与修复/损伤；无积极机制证据直接结束研究支线。
- **不扩张 Gate：** 现成官方 TAL manifest 和 Mask 解码缓存足够时，只做 GT identity join + 重新统计（无需训练）；缺少原始 raw 的情况下，不为了补齐所有可选指标全面重推理。
- **停止：** 输出 `PER_GT_TAL_STATE` 与 `OUTPUT_TAL_CROSSTAB`、运行 manifest，并核实互斥分类总计和引用的 baseline；否则明确 `unknown/insufficient_inputs`，不追加模块搜索。

## 6. 对旧规范和研究结论的生效范围

此文**废止“所有方法必须固定报告五层”作为每个新 Run 的硬性要求**，改为按问题选最小指标。旧 `EVALUATION_SYSTEMS_20261005.md` 是历史指标分类与已发表数字的来源，不删除、不改算历史表；其“官方正常推理通常看 one-to-one”仅能理解为当时某些研究特意固定的分支，不符合当前 Ultralytics 默认的通用说法。

每个项目结论仍由具体 Run 的分支、数据、标注、候选身份与评估器决定，而不是规范文件的标题或当前软件的默认值。
