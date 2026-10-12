# 评价指标最小化与 GT–TAL 分类（TAL 为可选专题）

**统一版：2026-10-13。** 跨研究线只强制“**科学命题 → 对象身份 → 一项必要主量 → 必要风险护栏 → 决策**”，不强制所有 Run 产出 raw/TAL/oracle/图片级/边界的完整五层矩阵。历史原始数值和冻结的 Study Gate 不因本文更新而改变。

**本轮用户明确主线**：[固定预训练模型的 Prototype→Coefficient 后训练修复](../ACTIVE_PLAN.md)。GT 任务失败和模型最终输出身份必须先正确；**TAL 不是后训练的必要指标、训练对象或前置 Gate**。

## 1. 一项主指标必须回答什么？

| 命题/对象 | 只需优先测量 | 必要护栏和解释边界 |
|---|---|---|
| 正常方法是否提升实例分割 | **完整 COCO segm AP@[.50:.95]**（同权重基线、同预测集合、同解码和 evaluator，方法仅动 Mask 时） | 原成功 GT 的 Mask75→失败、Box/Class/score/顺序未改、AP75/APsmall、耗时；按 Stage protocol 选择必要项目 |
| 哪些 GT 真实失败 | **最终输出 segm COCOeval@75 未匹配的普通 GT**，固定 `area=all`、`maxDets=100`、`useCats=1` | 同 GT bbox75 是否匹配与其检测 ID；COCO bbox/segm 匹配独立；成功数/普通 GT 数**不是 AP75** |
| 某一个输出框好但 Mask 差 | **固定同一条最终检测**的 `BoxIoU(d,g)` 与 `MaskIoU(d,g)` | ID 来自冻结配对，不拼不同预测，不 GT 选另一个 raw 冒充正常输出 |
| 固定 P、c0 的系数 oracle 是否还有补救机会 | 在同一原生最终检测上，有限优化 `Δc` 后恢复 Mask75 的 GT 占比 | 这是 GT-assisted **找到的机会下界**，不是表达能力最大值/正常方法收益；记录有效样本与收敛 |
| 后训练 adapter 是否可泛化 | 不读评价 GT 的 `delta_cθ` 对原任务失败的固定预测修复率 / image-macro Mask IoU | 对全部原成功实例的损伤率、全量 AP 和冻结模型输出身份；不能只在 GT 失败子集开 GT gate |
| 完整 raw 是否存在好系数/Mask | `raw_all` 或 `raw_argmax_correct` 中 `exists_joint75` 及相应唯一 raw 身份 | 原始候选机会 ≠ 正常输出；argmax-correct 不等于官方 O2O 最高分多类别组合输出 |
| 仅在研究历史训练监督分配时 | `n_tal(g)`、指定时间标签身份、TAL 正样本 Box/Mask 质量 | checkpoint TAL replay ≠ 历史 actual train assignment；**纯后训练不要测** |

**不增加没有决策用途的指标**：对每个补充指标需说明它回答哪个尚不能由现有指标回答的问题，以及结果如何改变下一步。像素级 `IoU = coverage/(1+false_area_ratio)`（GT 面积非零、FP 按同区域定义），`purity=coverage/(coverage+false_area_ratio)`，若由这些量精确推出，不可作为独立统计证据反复计权。

## 2. 统计单位、原生协议与可比性

- `COCO AP` 是类别、IoU 阈值和评分排序聚合；不等于某一实例 IoU、GT 成功比例、或每图 AP 的算术平均。全量 AP 必须对合并的完整预测 JSON 重新执行 COCOeval，不能平均不同图片分片/类别的 AP。
- `candidate_mean`、`unique_GT_mean`、`image_macro`、`class_macro_AP`、`image_exact_n` 是不同总体，不互换分母；需要区间时以自然样本单元（通常图像，外部视频以 scene/sequence）成簇，AP CI 须每次重算完整 COCOeval。历史开发中反复看过的 COCO val2017 不是新盲测。
- `Mask50`/`Mask75`、`Box50`/`Box75` 对应 IoU 阈值；COCO `dtMatches`/`gtMatches` 是 score 排序+类别+GT竞争后的任务匹配，独立 bbox/segm；“有 Mask75 raw”是另一个候选存在性量。
- 只有同 `weight SHA / image list / branch / actual input_hw / letterbox / score+TopK+conf / decoder / category map / prediction JSON / COCO area policy` 的数值可以称同协议配对。固定 8.4.100 `end2end` 才控制 O2O；`nms=False` 不是此版本验证分支开关，`agnostic_nms` 会传到 Head 改变类别输出。
- 官方公布 YOLO26m-seg Box/Mask AP `52.5/44.1` 与项目早期自定义 `51.980/43.518` **不一致**，即使权重 SHA 相同。一个原生配置的 `GT ledger validity` 可以通过而 `published AP parity` 未过，但不得用这份 GT 账本替代“官网数值已重现”。见 [基线协议](../../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md)。

## 3. 训练 TAL 正样本：有专题命题时才用的独立附录

对于原始标注 `g=(dataset,image_id,annotation_id)`，冻结权重在明确无增强配置下的官方 TAL replay `T(g)` 是选中 `target_gt_idx=g` 的 raw **集合**，必须记录完整 owner→annotation 映射、branch/input_hw、criterion、GT label/preprocess、对应实际 mask 监督格式；无正样本记 `unassigned_at_this_replay`，**不是原始网络根本没有好 Mask，也不是历史没收到梯度**。

若 `T(g)` 非空，可按 **同 raw** `Box75`、`Mask75` 派生 `joint_good`、`box_only`、`mask_only`、`separate_good`、`neither` 五类；前提是各类互斥并规定优先次序，不能把不同 raw 的独立好框/Mask 拼成一个 `joint_good`。这仅属于“**当前已冻结 checkpoint 的训练关联位置质量**”，不能凭它解释最终任务未匹配原因。

`actual_train_assignment_event` 需要训练时按真实增强、owner、时间、Loss criterion 另行记录，冻结重放不能反推。只有用户明确要研究 TAL 训练机制且有必要训练预算时阅读 [训练轨迹协议](../audits/candidate_eval_20261012/TAL_TRAINING_TRACE_PROTOCOL.md)。当前 **Prototype→Coefficient 的后训练**只需要在正常最终检测上按冻结 `class+BoxIoU` 规则关联 train2017 GT 作训练标签，**不是 TAL**，不要求这些 TAL 指标与 Run 产物。

## 4. 门槛、统计与停止

1. **主命题**先冻结选择/停止所必需的一个主指标与风险护栏；记录每个决策阈值来源、候选集合身份、预算及使用数据。具体 Stage protocol 的冻结门槛**优先于后出的通用举例**。
2. **方法收益**必须基于正常无 GT 输出：全量 COCO segm AP 与同配置基线比较；例如当前后训练 proposal 用相同检测集合、`zero-delta` 等价、原成功损伤和推理成本作护栏。只靠 GT oracle、训练 Loss 下降、TAL 或个别 Mask75 修复不能晋级。
3. **单次配置阴性**按预登记停机；不因 `argmax` 面板与 normal output 不一致自动推翻旧研究，只保留各自科学边界，不重训补指标。
4. **不重复计算全 raw**：当本研究唯一必需的新量是最终检测的 MaskIoU 或有限 c-only oracle，先读旧全量 raw/预测缓存；身份不一致才做最小重放，而不是扫描 30828840 raw。

原历史的 [EVALUATION_SYSTEMS_20261005](../research_notes/EVALUATION_SYSTEMS_20261005.md) 和独立 [候选身份/官方 COCO 标准](候选身份与评估口径.md) 保留其适用数值及源码依据，不自动上升为本 Study 的额外 Gate。

