# 官方 TAL one-to-one 原生共享系数头：实例梯度冲突审计

日期：2026-10-11。研究线：`neighbor_sensitive_p3`。

本协议冻结一个最终机制判别，不启动训练、不重求共享参数、不改变候选身份、不重做标签或 oracle。它只读取官方 TAL one-to-one 冻结缓存、固定原生特征/系数/prototype，以及已经完成的 oracle 记录。执行环境为笔记本 `ssh 28358lan`；解释器和 vendor 沿用 `coefficient_oracle_alignment_20261009` 的已核对环境。

## 1. 唯一问题和判定

问题是：同一尺度的多个实例是否对官方 one-to-one 共享 coefficient predictor 产生相互冲突的实例梯度，并且这种冲突是否解释了“固定 prototype 下可恢复、原生系数却没有实现”的 oracle gap。

主要判定只使用预先指定的 `val_small`，`dev_small` 只作方向复现：

1. 在原生 Mask75 失败实例中，比较 `recoverable` 与 `unrecoverable` 的共享头梯度冲突。`recoverable` 定义为 `iou_A < 0.75` 且既有 oracle `iou_C >= 0.75`；`unrecoverable` 定义为 `iou_A < 0.75` 且 `iou_C < 0.75`。原生成功实例不进入主组间比较，只作描述性对照。
2. 同图实例对与跨图**状态匹配**实例对分别统计，不能把不同尺度的参数块当作零冲突。跨图配对只使用固定缓存中的 `pyramid_level`、原生 `h` 范数和原生 `c` 范数，不使用 target box、oracle、IoU 或 GT 类别，避免结果泄漏。
3. 主要冲突量是实例匹配摘要中的负余弦比例 `mean(1[cos(g_i,g_j)<0])`；同时保存余弦均值和负余弦质量 `mean(max(0,-cos))` 作为连续敏感性读数。
4. 主要预测检验是候选级冲突量与既有 `oracle_gap = objective_A - objective_C` 的 Spearman 相关；image-macro 相关作为同一检验的聚合复现。

只有在 `val_small` 同时满足以下条件时，才授权另立“冲突感知实例条件系数路由”方法协议：

- 每个主要恢复组至少有 50 个候选且至少覆盖 20 张图；同图与跨图两种主冲突量各自都必须有至少 50 个有效候选摘要，否则该口径记为未知而不判阴性；
- recoverable−unrecoverable 的冲突差在同图和跨图两种配对口径下均为正，候选级和 image-macro 两种 bootstrap 的 95% 区间下界均大于 0；
- 跨图冲突量与 oracle gap 的 Spearman 相关在候选级 bootstrap 下界大于 0，且 `dev_small` 点估计保持同号；
- 所有配对和分层的有效计数、有限性和身份核对通过。

上述任一门槛失败，且有效样本门槛已满足，则关闭当前顶会方法线：不启动 teacher、basis、residual、solver 或冲突损失训练，转入现有 “Recoverable but Unrealized” 机制论文。若有效样本门槛不满足，结论为未知，阻塞变量是恢复/不可恢复两组的图像数；最小动作是从同一既有 `val` 清单扩大图像子样本，不新增训练或数据曲线。

## 2. 固定输入和来源

### 2.1 官方冻结 bank

复用 `coefficient_official_tal_affine_20260930` 的官方 cache（笔记本对应路径通常为 `D:/coco_wire/data/official_tal_affine_20260930/runs/official_cache`）。必须存在 `INDEX.json` 和 `images/<image_id:012d>.pt`。每个图像文件中的 `rows`、`raw_id`、`level`、`h`、`coeff`、`proto`、`masks`、`target_boxes`、`owners` 均原样读取；不重新前向模型。

官方 cache 的 one-to-one 身份字段是：`split, image_id, annotation_id, branch=one2one, raw_id, pyramid_level, target_gt_idx`。身份顺序必须与已有官方 manifest 一致；任何不一致都使 Run 失败。

### 2.2 既有 oracle 记录

读取项目中已回传的 [official TAL 每候选记录](../coefficient_official_tal_affine_20260930/PER_CANDIDATE.jsonl) 或笔记本上对应的不可修改副本。只连接以下已有字段：`iou_A, iou_C, objective_A, objective_C` 以及身份字段。不得重新求 oracle、重新 decode、按本轮结果改变阈值或组别。

`oracle_gap = objective_A - objective_C`；若该值非有限，候选只保留在身份审计，不进入主要相关检验。既有 oracle 是 GT 辅助诊断，不是可部署结果；本审计不作 AP 声明。

### 2.3 版本和数值

- YOLO26m-seg 官方权重 SHA256 沿用 `16b636f0…` 的已记录版本；只核对 `one2one_cv4` 最后一层形状为三组 `32×64` weight、`32` bias，不更新参数。
- Ultralytics `8.4.100`；输入 640、FP32 cache、`mask_ratio=1`、`overlap_mask=true`；官方分割增益 `seg_gain=9.83241`。
- 梯度计算可以在笔记本 CUDA 上运行，但逐实例 BCE 梯度用 FP64 累积；TF32 关闭。计算是纯读数，没有 optimizer、backward 更新或权重写回。

## 3. 小样本面板和独立性

正式 Run 使用已有官方 split 清单，不增加图片：`fit_small`、`dev_small`、`val_small` 每个从各自既有清单确定性抽取 40 张图。每个 split 的抽取键为 `SHA256("native-conflict-v1/{split}/{image_id}")`，按 digest、再按 image_id 升序取前 40 张；保存完整清单及 SHA256。三份清单必须互不重叠。

`val_small` 是主要独立 split，`dev_small` 是方向复现，`fit_small` 只作描述性参考。这里的“独立”指不参与本审计的配对、分组和阈值选择；这些是历史研究图，不称为 COCO 盲测。每张入选图片的全部官方 one-to-one 候选都保留；不按分数、top-k、NMS、IoU、oracle gap 或最终冲突筛选候选。正式 Run 前允许每个 split 4 张图 smoke，只做实现核对，不写入正式统计。

## 4. 实例梯度定义

对候选 `i`（尺度 `l`）从固定 cache 取 prototype 像素矩阵 `P_i`、原生系数 `c_i`、原生 coefficient-head 输入 `h_i∈R^{64}`、官方 ROI 内标签 `y_i` 和 GT 框面积 `a_i`。令

\[
z_i=P_i c_i,\qquad u_i=\frac{9.83241}{a_i}(\sigma(z_i)-y_i),\qquad g^c_i=P_i^\top u_i.
\]

官方 one-to-one 共享末层为 `W_l ∈ R^{32×64}`、`b_l ∈ R^{32}`。实例对该共享头的梯度向量固定为

\[
g_i=\operatorname{vec}(g^c_i h_i^\top)\;\Vert\;g^c_i,
\]

并记录 `gradient_norm`。这就是 mask BCE 对原生共享 coefficient head 的实例梯度；不加入 oracle 正则、共享求解梯度、分类/框损失、上游参数或跨尺度零块。

每个 pair 只在相同 `pyramid_level` 内计算余弦 `cos(g_i,g_j)`。不同尺度的 pair 被排除并单独报告计数，不记为余弦 0。解析公式必须在 4 张图 smoke 上与 autograd 有限差分核对，绝对误差门槛 `3e-5`。

## 5. Pair 定义

### 5.1 同图实例对

在同一 split、同一 `image_id`、同一 `pyramid_level` 内取无序全部实例对 `i<j`。若某图某尺度少于两个候选，计数为零而不补造 pair。主统计为每实例对的余弦和负余弦指标，并另外生成每实例的 pair 均值摘要。

### 5.2 跨图状态匹配实例对

跨图 pair 不使用全量笛卡尔积。对每个 split、每个 level，以固定候选的 `||h||_2` 和 `||c||_2` 各分成四个经验分位区间；在同一个 4×4 单元内按 `(image_id, annotation_id, raw_id)` 排序，每次贪心配对两个不同 image 的候选，每个候选最多出现一次。若相邻候选来自同图，则与该单元内下一个不同图候选交换；仍无法配对则保留为 unmatched。分位点和 pair manifest 必须保存，不能看结果后改变分箱。

这种 pair 只控制原生特征/系数幅度，不使用 GT、target box、oracle 或失败标签；因此它是跨图“状态匹配”，不是语义匹配。报告 pair 数、未匹配数和匹配距离，避免把配对不足误读为无冲突。

## 6. 统计输出

每候选保存：身份、level、split、`iou_A/iou_C`、`recoverable/unrecoverable`、`oracle_gap`、gradient norm、同图 pair 数/余弦均值/负比例/负质量、跨图 matched pair 数/余弦均值/负比例/负质量。

每 pair 保存：两端身份、pair type、split、level、余弦、分箱、匹配距离和两端恢复标签。输出至少包含：

1. `PER_INSTANCE.jsonl` 与 `PER_PAIR.jsonl`；
2. 按 split、level、原生失败分组的 candidate-mean 表；
3. 同图与跨图 matched 两种口径的 recoverable−unrecoverable 差；
4. candidate-level 冲突−oracle-gap Spearman 相关及 image-macro 相关；
5. `BOOTSTRAP.json`、`SUMMARY.json`、`DECISION.json` 和输入/源码/环境 hash。

Bootstrap 固定 `seed=20261011`、`B=2000`、percentile 95% 区间：

- candidate-level：以每候选的匹配 pair 均值摘要为统计单元重采样；不把同一候选的多个 pair 当作独立实例；
- image-macro：先在图内聚合，再以 image 为簇重采样；
- pair 原始表仅用于描述和 identity 核对，不把它冒充独立 image-level 证据。

## 7. 运行和停止

建议 Run 链：`RUN_CONFLICT_SMOKE_S0`（每 split 4 图）→ `RUN_CONFLICT_AUDIT_S0`（每 split 40 图）→ `RUN_CONFLICT_VERIFY_S0`（独立脚本重算身份、梯度、pair manifest 和一组 bootstrap）。正式计算脚本只读取 cache 和既有 oracle，禁止调用训练入口、optimizer 或保存模型权重。

完成 `DECISION.json` 后停止。若 gate 阳性，另立方法协议再讨论冲突消除；本协议本身不授权任何正式训练。若 gate 阴性，沿既有门槛关闭 A/C/F/R、Q-basis、teacher、basis、residual 和 solver 方向。

