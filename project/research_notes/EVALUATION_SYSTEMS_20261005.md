# 当前评价体系总览

更新时间：2026-10-05

本项目同时研究模型的标准性能、raw 候选能力、训练监督位置、最终输出保留和图片级实际可用性。它们回答的是不同问题，不能把一个层次的好结果替代另一个层次的失败。

## 一、统一对象和数据范围

当前主诊断对象是官方 COCO 预训练 `yolo26m-seg.pt`、Ultralytics 8.4.100、COCO val2017 5,000 张图片、36,335 个普通 GT。

评价单位有四种：

| 单位 | 主要回答的问题 |
|---|---|
| raw 候选 | 模型在所有预测位置上是否已经产生了可用框和 mask |
| GT 实例 | 某个目标能否找到正确类别、框和掩码 |
| 最终预测 | 经过分数、top-k、NMS、预算和 COCO 匹配后留下了什么 |
| 图片 | 一张图是否能完整、低误报地交付 |

主要阈值：Box50/Box75、Mask50/Mask75。当前机制诊断主要使用 75，粗质量和敏感性分析补充使用 50。高置信框阈值必须事先固定；现有图片级诊断使用 `score≥0.700615`，它来自既有 Box75 Precision=90% 的全局阈值。

## 二、标准 COCO 性能体系

**对象：**正常最终输出。

**指标：**

- detection AP、AP50、AP75、AP small/medium/large、AR1/10/100；
- segmentation AP、AP50、AP75、AP small/medium/large、AR1/10/100；
- COCO Mask75 正常匹配数量。

当前原模型基线：Box AP 51.980、Box AP75 56.977、Mask AP 43.518、Mask AP75 47.026、Mask75 正常匹配 22,767/36,335。

**作用：**标准论文和模型性能基线。

**限制：**不能说明是 prototype、coefficient、TAL、top-k 还是 NMS 造成失败；AP 也不等于“整张图全部可用”。

来源：[COCO raw 主报告](../../experiments/coco_raw_capability_5000_20260916/REPORT.md)。

## 三、完整 raw 候选几何能力

**对象：**30,828,840 个 raw 候选，读取在类别分数、head top-k、conf 和最终输出之前。

**GT 级指标：**

- `max Box IoU`、`max Mask IoU`；
- 是否存在 Box75 候选；
- 是否存在 Mask75 候选；
- 是否存在同一个 raw 同时 Box75+Mask75；
- Box75/Mask75 的一对一最大匹配；
- small/medium/large 分层；
- 框内前景占比、密度和形状描述。

**分数无关五类：**

| 状态 | 判据 |
|---|---|
| joint good | 同一 raw 有 Box75+Mask75 |
| box good / mask unavailable | 有 Box75，但整个 raw 池没有 Mask75 |
| mask good / box unavailable | 有 Mask75，但没有 Box75 |
| separate good | 好框和好 mask 存在，但不在同一 raw |
| neither good | 两者都没有 |

当前结果：joint good 23,805（65.52%），box good/mask unavailable 6,688（18.41%），separate good 531（1.46%），neither good 4,824（13.28%）。

**作用：**回答“模型是否已经在 raw 候选池中产生了可恢复能力”。

**限制：**GT 选择 raw 是能力诊断，不是模型实际选择结果，也不是部署召回。

## 四、argmax 条件的 raw 能力体系

**对象：**只保留预测类别 argmax 与 GT 类别一致的 raw。

这是对完整 raw 几何评价增加了语义条件，用来回答：

> 在模型已经正确识别类别后，框和 mask formation 还失败多少？

互斥规则：

1. `Semantic fail`：没有 argmax-correct raw；
2. `Box fail`：有 argmax-correct raw，但没有 argmax-correct Box75；
3. `Success`：同一 argmax-correct raw 有 Box75+Mask75；
4. `Box/Mask misaligned`：argmax-correct 集合中分别有好框和好 mask，但没有同一 raw 双达标；
5. `Mask fail`：有 argmax-correct Box75，但整个 argmax-correct 集合没有 Mask75。

当前结果：Semantic fail 44（0.12%）、Box fail 6,335（17.43%）、Mask fail 6,459（17.78%）、misaligned 484（1.33%）、Success 23,013（63.34%）。

**主线意义：**6,459 个 `argmax-correct + Box75 + no Mask75` 是 prototype→coefficient→mask formation 的主要诊断对象。它先排除了 raw 候选层面的主要语义错误。

来源：[argmax 条件报告](../../experiments/coco_raw_argmax_conditioned_20261005/REPORT.md)。

## 五、正常输出阶段记账体系

**对象：**raw 候选到正常最终输出的整个筛选链。

按顺序记账：

1. raw 中是否存在几何合格 mask；
2. GT 类别分数是否达到阈值；
3. 是否进入 head top-k；
4. 是否通过 conf；
5. 是否导出；
6. 是否被每类 maxDets、NMS 或 COCO 匹配竞争排除。

对正常 Mask75 失败的 13,568 个 GT，已有分解：

| 失败位置 | GT数 |
|---|---:|
| raw 池没有 Mask75 | 11,512 |
| 有好 mask，但 GT 类别分数≤0.001 | 1,797 |
| 有好 mask，但未进 head top-300 | 153 |
| 被每类 maxDets=100 排除 | 103 |
| COCO 配对竞争 | 3 |

**作用：**区分 mask formation 失败和输出保留失败。

## 六、图片级实际使用体系

**对象：**一张图是否能直接交付。

设图片中有 (n) 个普通 GT，固定 score 阈值后定义：

- `all-GT coverage`：所有 GT 都被 Box75 预测覆盖，允许额外框；
- `exact-n Box`：恰好保留 n 个框，且 n 个全部 Box75 正确；
- `FP/image`：额外框数量；
- `miss/image`：未覆盖 GT 数量；
- 后续可定义 `exact-n Mask`：同一 n 个框的 mask 也全部达到 Mask75。

当前有 4,952 张含普通 GT 的图片。在 score≥0.700615 下：

- all-GT Box coverage：1,086/4,952 = 21.93%；
- exact-n Box：1,051/4,952 = 21.22%。

粗略扫阈值后，exact-n Box 的最高约 24.64%，说明全局阈值无法同时解决完整召回和额外框。

**作用：**这是面向实际使用的主评价候选，特别适合计数、巡检和“整图可交付”场景。

**限制：**全图指标是全有或全无，严格程度高；不能替代 GT 级 mask 机制指标，也不能直接等同 COCO AP。

来源：[图片级框报告](../../experiments/coco_image_exact_box_20261005/REPORT.md)。

## 七、实例 mask 内容质量体系

**对象：**固定 raw 或固定最终预测与对应 GT 的 mask。

核心指标：

- Mask IoU、Mask50、Mask75；
- GT coverage：预测前景覆盖 GT 的比例；
- purity：预测前景中属于 GT 的比例；
- Boundary IoU / Boundary F1；
- AUC：连续 mask logit 的像素排序；
- FPR：框内或背景错误前景比例；
- background FP、neighbor FP、same-neighbor FP；
- self FN、crop support；
- image macro、candidate mean、GT paired difference。

修复/损伤统计：

- repair：方法把失败实例提升到目标阈值；
- damage：方法损伤原本成功实例；
- net repair = repair − damage；
- 不能只报告一个上涨指标而忽略 coverage、FPR 或原成功实例损伤。

**作用：**回答 prototype、coefficient、空间读出和边界形成的内容质量问题。

## 八、oracle 与可恢复性体系

**对象：**GT 辅助诊断，不是可部署方法。

包括：

- 逐实例有限 oracle coefficient (c^*)；
- (Delta c=c^*-c_0)；
- 像素 logit 修正 (Delta z=PDelta c)；
- oracle gap；
- oracle 前后 AUC、coverage、FPR、Mask IoU；
- coefficient/logit direction cosine；
- 理想 FP 删除、FN 填充、边界修复；
- oracle repair/damage 上界。

**作用：**判断“好解是否存在”和“现有模型离好解有多远”。

**限制：**不能把 GT oracle 的收益写成学习方法收益，也不能直接证明特征或网络一定能够预测 oracle。

## 九、TAL 与训练链路体系

**对象：**训练期间的监督位置和优化状态。

记录：

- one-to-one / one-to-many TAL 正样本；
- `raw_id`、GT identity、pyramid level、target owner；
- 官方 mask BCE、框支持、面积归一化和增益；
- BCE、teacher loss、总 loss；
- (dL/dc)、(dL/dP)、指定模块梯度；
- fit/dev/val 目标差异；
- shared affine 可达性、独立 oracle 差距和数值误差；
- prototype 与 coefficient 是否冻结或联合更新。

**作用：**回答“监督是否作用到了指定训练分支”和“固定目标是否优化充分”。

**限制：**TAL GT 绑定是训练监督关系，不是推理输出关系。训练损失下降也不等于正常图片输出改善。

## 十、分层与对象选择体系

这些不是单独的主结果，而是解释方法作用对象的分层：

- small/medium/large；
- Box75/Mask75 状态；
- argmax 条件状态；
- 框内前景占比；
- mask density、crowding、同类邻居；
- original success / original failure；
- P3/P4/P5，但不能把 pyramid level 自动等同目标大小；
- image exact-n 状态。

分层必须在主指标之外报告，不能挑最有利子组代替全体结果。

## 十一、未来方法的统一报告顺序

每个新方法固定报告以下五层：

1. **正常输出实用性**：COCO AP/AR 作为标准护栏，图片级 exact-n、coverage、FP/image 作为用户目标；
2. **输出阶段**：哪些 raw 候选被分数、top-k、NMS 或预算丢掉；
3. **argmax 条件 raw 能力**：正确语义后是否已有 Box75/Mask75；
4. **mask 内容质量**：IoU、coverage、purity、boundary、AUC、FPR、repair/damage；
5. **训练和机制证据**：TAL 位置、loss、梯度、oracle gap、参数和计算代价。

当前 prototype→coefficient 主线的主要机制对象是第 3、4 层中的 6,459 个 argmax-correct Mask fail；图片级 exact-n 是最终实用护栏；输出保留对象中的 1,233 个 argmax-correct joint-good GT 单独作为排序和筛选问题，不与 mask formation 混合。

## 十二、最容易发生的口径错误

- raw 候选存在好 mask ≠ 最终输出会保留好 mask；
- TAL 正样本 ≠ 推理阶段对应预测；
- GT oracle ≠ 学到的方法；
- candidate mean ≠ image macro；
- Mask75 逐 GT 比例 ≠ COCO AP75；
- exact-n 图片成功率 ≠ 标准检测 AP；
- 不同 raw 的好框和好 mask不能拼成同一个成功实例；
- 分数阈值改变输出集合，但不能改变固定 raw 候选的几何能力；
- 任何新训练都必须同时报告训练目标和正常原图输出。

## 十三、raw 候选诊断的正式分层规范（2026-10-05 补充）

本节把 raw 诊断从“候选池里是否存在好结果”细化为一条可追踪的链路。它与前文的标准 COCO 评测、TAL 监督记账和图片级实际输出相互衔接，但不互相替代。后续 raw 报告以本节的候选身份、集合定义和统计表为准。

### 13.1 先固定 raw 身份，不能只用候选序号

一个 raw 预测位置必须至少由下列字段唯一标识：

```text
image_id
branch              # one-to-one 或 one-to-many
pyramid_level       # P3/P4/P5
raw_id              # 该分支、该尺度下的原始位置索引
grid_y, grid_x
stride
input_shape         # 本次前向的 H×W
```

“同一个 raw”指上述身份全部相同，不能只因为它们对应同一个 GT、处在同一张图或拥有相同候选序号就合并。one-to-one 和 one-to-many 即使网格位置相同，也必须作为两个 raw 分支分别记账。官方正常推理通常看 one-to-one；TAL 监督分析同时保留官方实际使用的分支信息。

输入尺寸也必须随 raw 记录。`imgsz=640` 是验证缩放尺度，不代表每张图的张量都是 `640×640`，因此候选数不固定为 8400。

### 13.2 为每个 GT 建立四个候选集合

对图片中的 GT 实例 (g)，类别为 (y_g)，在同一图、同一分支、同一输入形状内定义：

| 集合 | 定义 | 作用 |
|---|---|---|
| \(R_{\mathrm{all}}(g)\) | 该图该分支的全部 raw 位置 | raw 几何能力上界 |
| \(R_{\mathrm{arg}}(g)\) | \(r\in R_{\mathrm{all}}\)，且 `argmax(class_r)=y_g` | 排除类别 argmax 错误后的能力 |
| \(R_{\mathrm{TAL}}(g)\) | 官方冻结流程为 \(g\) 分配的 raw 位置 | 训练监督实际覆盖的位置 |
| \(R_{\mathrm{out}}(g)\) | 能沿原 raw 身份追溯到正常最终输出的候选 | 输出阶段保留情况 |

`argmax` 只表示每个 raw 的预测类别最大类与 GT 类别相同，不附带 `conf`、top-k 或 NMS 条件。低分但类别 argmax 正确的 raw 仍属于 \(R_{\mathrm{arg}}\)，它的置信分数在输出阶段单独记录。

如果最终后处理无法可靠回溯到原始 raw 身份，\(R_{\mathrm{out}}\) 对应字段必须记为 `unknown`，不能把“未能映射”当作“被筛掉”。

### 13.3 每个候选集合都要同时测独立能力和同 raw 联合能力

对任意集合 \(R\)，固定阈值 \(\tau_B=\tau_M=0.75\)，同时计算：

```text
best_box_iou(R)       = max_r BoxIoU(r, g)
best_mask_iou(R)      = max_r MaskIoU(r, g)
exists_box75(R)       = 是否存在 BoxIoU >= 0.75 的 raw
exists_mask75(R)      = 是否存在 MaskIoU >= 0.75 的 raw
exists_joint75(R)     = 是否存在同一个 raw 同时满足 Box75 和 Mask75
```

同 raw 联合条件严格写成：

\[
\exists r\in R:\quad \mathrm{BoxIoU}(r,g)\geq\tau_B
\ \land\ \mathrm{MaskIoU}(r,g)\geq\tau_M.
\]

不能把 `best_box_iou` 所在 raw 和 `best_mask_iou` 所在 raw 拼成一个“成功实例”。如果二者都存在但没有同一 raw 双达标，记录为 `separate_good` 或 `Box/Mask misaligned`。

除 75 外，可用 50 做敏感性分析，但必须在同一张表中标明阈值，不能用 Mask50 的结果替代 Mask75 的主诊断。

### 13.4 argmax 条件下使用互斥且穷尽的失败分类

在 \(R_{\mathrm{arg}}\) 上按以下顺序分类，每个 GT 只进入一类：

1. **Semantic fail**：\(R_{\mathrm{arg}}(g)=\varnothing\)。
2. **Box fail**：\(R_{\mathrm{arg}}\neq\varnothing\)，但没有 argmax-correct Box75。
3. **Mask fail**：有 argmax-correct Box75，但 \(R_{\mathrm{arg}}\) 中没有任何 Mask75。
4. **Box/Mask misaligned**：argmax-correct 集合中分别存在 Box75 和 Mask75，但不存在同 raw 双达标。
5. **Success**：存在同一个 argmax-correct raw 同时达到 Box75 和 Mask75。

`Box fail` 中如果存在 Mask75 但没有 Box75，另外设置 `mask_good_without_box=true`；这保持主分类互斥，同时不丢失“掩码好、框差”的信息。所有 GT 必须满足：

\[
N_{semantic}+N_{box}+N_{mask}+N_{misaligned}+N_{success}=N_{GT}.
\]

当前已完成的 argmax 报告采用这一口径：Semantic fail 44、Box fail 6,335、Mask fail 6,459、Misaligned 484、Success 23,013；它与完整 \(R_{\mathrm{all}}\) 的统计不能直接互换，因为候选集合不同。

### 13.5 raw 能力与输出保留必须分开记账

对每一个 raw/GT 关系，建议保留以下阶段字段：

```text
raw_geometry_good       # 是否满足 Box/Mask/Joint 阈值
argmax_correct          # 不带分数阈值的类别条件
class_score             # GT 类别原始分数
entered_head_topk       # 是否进入 head top-k
passed_conf             # 是否通过验证/推理 conf
survived_max_det        # 是否留在每图预算内
survived_nms_or_e2e     # 是否通过去重或端到端输出筛选
matched_final_output    # 是否最终与该 GT 配对
```

raw 能力表不先按置信度过滤；`class_score`、`conf`、top-k、max-det、NMS/端到端去重和 COCO 配对只在输出保留表中分析。这样可以区分：

```text
没有好 raw
→ raw 有好结果但类别/分数不够
→ 进入 head 后被 top-k 或预算丢掉
→ 产生最终结果但被匹配竞争排除
```

这些阶段不保证对所有类别扩展形成简单的单调漏斗，所以每个字段必须基于同一个 raw 身份单独统计，不能把不同分母的数量直接相减。

### 13.6 增加图片级 raw 能力，避免只看 GT 平均

设图片 (j) 有 (n_j) 个 GT。除了逐 GT 统计，每张图还应计算：

| 图片级量 | 定义 |
|---|---|
| `raw_all_joint_complete` | 每个 GT 在 (R_{mathrm{all}}) 都有同 raw Joint75，允许存在额外 raw |
| `raw_arg_joint_complete` | 每个 GT 在 (R_{mathrm{arg}}) 都有同 raw Joint75 |
| `raw_arg_one_to_one_complete` | 以 GT 与 raw 的 Joint75 为边做最大二分匹配，匹配数等于 (n_j) |
| `output_exact_n_box` | 正常输出恰好 (n_j) 个合格框 |
| `output_exact_n_mask` | 同一批 (n_j) 个输出同时达到 Mask75 |

最大二分匹配用于防止一个 raw 被多个 GT 重复计数；它与“每个 GT 分别存在候选”的存在性统计必须同时报告。图片级统计按图片 macro 汇总，不能把图片内多个 GT 当成独立图片样本。

### 13.7 raw 诊断的主表和最小交付字段

后续每次 raw 运行至少交付三张表：

**表 A：GT—raw 能力表**

```text
image_id, annotation_id, branch, input_shape
num_raw_all, num_raw_argmax, num_raw_TAL, num_raw_output
best_box_iou_all, best_mask_iou_all, best_joint_all
best_box_iou_argmax, best_mask_iou_argmax, best_joint_argmax
box75_raw_id, mask75_raw_id, joint75_raw_id
semantic_fail, box_fail, mask_fail, misaligned, success
```

**表 B：阶段保留表**

```text
raw_id, pyramid_level, grid_y, grid_x, branch
box_iou, mask_iou, argmax_correct, class_score
entered_topk, passed_conf, survived_max_det
survived_nms_or_e2e, matched_final_output
```

**表 C：图片级交付表**

```text
image_id, n_gt
raw_all_joint_complete, raw_arg_joint_complete
raw_arg_one_to_one_complete
output_exact_n_box, output_exact_n_mask
miss_count, extra_count, raw_reuse_count
```

所有表都要写明模型 checkpoint、Ultralytics 版本、输入尺寸策略、分支、阈值、GT 版本和 raw cache 哈希。

### 13.8 解释顺序和边界

这套 raw 诊断按以下顺序解释：

```text
R_all：模型内部是否有能力
  ↓
R_arg：类别判断正确后是否仍有能力
  ↓
同 raw Joint：框与掩码是否由同一位置共同形成
  ↓
R_TAL：训练监督是否覆盖这些位置
  ↓
R_out：实际推理是否保留这些位置
  ↓
图片级：整张图是否可交付
```

`R_all` 的好结果是能力上界，不代表模型会选择；`R_arg` 是语义条件诊断，不代表通过置信度；`R_TAL` 是训练监督关系，不等同推理对应关系；`R_out` 才接近部署输出。GT 参与 raw 几何比较时只能称为诊断或上界，不能称为模型已经学到的收益。

本节新增的是记账和对照层次，不改变既有实验数字的适用范围，也不要求重跑已经完成的 raw 5000 或 argmax 实验。后续如已有缓存缺少某字段，先标记未知；只有在字段确实决定结论且无法从现有产物恢复时，才登记独立补充 Run。
