# TAL 动态监督图记录协议（设计稿，2026-10-12）

性质：**先纠正研究对象，不代表已经记录了任何真实训练轨迹。** 本协议只指定观测，不批准新训练或修改官方 assigner。与项目 `科研项目通用规范/指标最小化与TAL分类.md` 配合；历史 E0 是固定 checkpoint 的 TAL 重放，**不是**真实训练历史。

## 1. 核心问题与明确结论

若研究 **当前权重在固定 GT 上的正样本位置/损失**，使用冻结 checkpoint 重新执行官方 TAL 已足够，结论只能称 `checkpoint_assignment_replay`。

若研究 **训练动态中是否反复漏掉某 GT、标签在哪些 raw 来回跳转、这些动态是否先于 Mask 失败出现**，必须在真实训练的每次 batch 前向 **同一次官方 loss 内**记录动态分配。单独用最终模型重放不能恢复过去已消失的分配。

**不需要每一步持久化所有 (GT,raw) 稠密笛卡尔积矩阵。** TAL 只给正样本 raw 一个当前 GT owner；非正样本并没有单独的最终 GT owner。将关系分为：
- `eligible(g,r,t)`：原始位置是否在当前 GT 的有效空间候选集合；
- `ranked(g,r,t)`：该对符合候选筛选/进入 top-k 竞争，记录对齐分数与排名；
- `assigned(g,r,t)`：冲突消解与 `topk2` 后的最终监督边，只有此层进入 mask/box 正样本监督；
- `emitted(g,p,t)`：正常推理输出到 GT 的后验评价关系，**不属于 TAL**。

这四种边不能压成同一“GT 关联图”。

## 2. 基于 Ultralytics 8.4.100 的真实读取位置

`v8DetectionLoss.get_assigned_targets_and_loss` 在实际 train-forward 中构造：预测类分数 sigmoid 和预测框（都 detach）、anchor 中心、batch 增强后的 GT 类/框，传给 `self.assigner`；该函数返回每 raw 的 `fg_mask, target_gt_idx` 及相关 target_bboxes/target_scores。`TaskAlignedAssigner.forward` 的输出为 `(target_labels,target_bboxes,target_scores,fg_mask,target_gt_idx)`。

`E2ELoss` 的 one-to-one TAL 是 `topk=7,topk2=1`，one-to-many 是 `topk=10`，`alpha=0.5,beta=6`；仓库中冻结权重审计也核实过。最终分配可以因当前权重、batch GT、增强/输入几何而变。每个微批真正执行的分配，不可由不带增强的 checkpoint replay 冒充。

**记录实现点：** 对训练时**真正被调用的**两套 `criterion.one2one.assigner` / `criterion.one2many.assigner` 分别只读 hook，直接从其输入/输出读取而不重新调用 TAL。不允许额外 `model.init_criterion()` 创建另一份模拟 criterion 以冒充训练真实对象。只允许 detach/read；不得改输出、梯度、随机状态、loss 或训练逻辑。DDP/AMP/compiled 环境需单独校验 hook 确实触发且记录对应 rank，未触发即中止审计。

版本源码：
https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/utils/loss.py
https://github.com/ultralytics/ultralytics/blob/v8.4.100/ultralytics/utils/tal.py

## 3. 真实训练的最小稀疏流：每个实际 batch 记录正样本边

记录时机为 **当前 forward 中 TAL 返回之后、对应 optimizer step 之前**。同一参数状态下可能有多次 gradient-accumulation microbatch，须分别记录：
- `run_id, epoch, global_microstep, optimizer_step, grad_accum_index, ddp_rank, model_version_or_step, branch`；
- `input_hw, feature_hw_by_level, aug_view_id, augmentation_seed_or_recipe_hash, batch_local_image_index`；
- 每个增强后 GT 的 `aug_gt_index, class, augmented_xyxy, validity, source_image_id, original_annotation_id, occurrence_id, lineage_status`；
- `n_raw, n_valid_gt, per_gt_n_positive, per_gt_n_eligible`（后者只有在附加选位 hook 已核实时才可给数，否则 unknown）；
- **仅最终正样本边**：`(batch_index, aug_gt_index, raw_id_global, pyramid_level, grid_y, grid_x, stride, target_score_for_gt_class)`；另外记录所属预测类分数、box 的 CIoU/IoU（仅在该项确有问题时采集）；
- 整个原始 raw 集合的 `n_raw`、输入形状、三尺度 offset、正样本总数与负样本总数；**不为所有负样本编造 GT owner**。

若要解释被忽略的潜在好候选，只在预注册固定面板/特定异常 step 额外记录每 GT 的 eligible count、top-k 前候选及被竞争挤掉的边和对齐分数。完整 dense 矩阵不列为每次训练的必报资产。

## 4. 增强后身份是成败关键

标准 batch 的 `target_gt_idx` 是 **当前 batch 增强并经过标签过滤/重排之后的相对 GT 索引**，不是原始 COCO annotation_id。

必须有贯穿数据管线（包括 Mosaic、MixUp、Copy-Paste、random affine、裁切与 overlap-mask owner 顺序）的 lineage。推荐实例键：

`(run_id, global_microstep, batch_index, aug_view_id, aug_gt_index)`；
并保存 `(source_image_id, original_annotation_id, occurrence_id)` 与实际增强后框/掩码序号。

同一原始标注可能在同一步被复制多次，原标签也可能因裁切消失；不能仅用 source image_id + annotation_id 充当增强后唯一键。做不到可靠 lineage 时标记 `lineage_unknown`，停止跨 epoch 的“同一 GT 被分配不同 raw”结论，不猜一个 ID。

## 5. 两条观察轴，不混成一个时间序列

| 轴 | 数据 | 能下什么结论 | 不得声明 |
|---|---|---|---|
| A **真实训练事件轴** | 当前模型、当前增强输入，逐步的实际 assigned edges | 训练期间实际哪些增强后 GT 获得监督、监督数量/分支/尺度分布、是否存在系统性空分配 | 不同增强视图 raw_id 切换就是模型“不稳定”；任何 GT 训练历史全貌（如采样非穷尽） |
| B **固定检查点重放轴** | 预先冻结同一批无增强图像、相同图像几何，使用选定 checkpoints 重放 TAL | 同一原图 GT 在不同权重阶段的 TAL raw 是否改变、等级及 mask 质量轨迹 | 它等价于当时实际 Mosaic/MixUp batch 所使用的分配 |

轴 B 应固定 `checkpoint_0 / 早期 / 中期 / 最终` 四个已计划且确实保存的权重；不可为“最平滑结果”事后选 epoch。对比相同 GT 的 raw_id 需要确保输入尺寸、分支、网格与标注完全一致；含增强的轴 A 可以按原始 GT 身份聚合，但不同几何位置 raw_id 不直接对照。

## 6. 仅保留回答明确问题的指标

- **必需**：GT 正样本覆盖 `Pr(n_pos>0)`、每 GT 的 `n_pos`、P3/P4/P5 分布、分支分开、增强后来源映射完成率。
- **检验“位置变了”才需要**：固定重放轴上的 GT→raw 身份改变率、被监督位置的 Box/Mask IoU 变化、候选排名差；training axis 仅在增强 view 完全相同可比时才谈绝对 raw 切换。
- **检验“是否应改变 TAL”才需要**：与原始官方位置同图同 GT 的可选 raw / mask 质量差、修改监督策略前后的正常推理 COCO AP 及原成功损伤。GT-best 是诊断 oracle，不是方法增益。
- **不必通报**：每个负 raw 与每个 GT 的全量 IoU/概率、边界/AUC/Purity/准确率全套、每步全量 mask 解码，只为了填五层模板的附带统计。

TAL 正样本的连续 Mask 质量可在固定无增强 checkpoint 面板与原始 COCO Mask 上评测；真实训练的增强后监督 Loss 在其当前 overlap owner 标签上计算。**不能直接将二者的数值差称为训练 Loss gap**。

## 7. 资源预算、前置 Gate 与唯一后续

1. **G0 无改动基线：** 必须固定官方版本/实际 trainer criterion、checkpoint、train/val 清单与种子；只读 hook 关闭/打开，对同一 smoke batch 检查完整 fg/owner、Loss、梯度与 optimizer 更新数值一致（已指定误差预算，优先精确一致）。未通过不正式记录。
2. **G1 identity：** 所有增强后 GT ID 有来源或明确 unknown；gt_index→anno 的映射经重复/裁切/标注重排样本覆盖；正样本 raw 不重复分配冲突、索引不越界，GT 和 raw 总数守恒。
3. **G2 稀疏记录开销：** 只读缓存+批量落盘，记录同步带来的 wall time 与内存开销，不保存所有 raw×GT 笛卡尔积。具体频率先按命题决定：若关心逐步动态，必须每训练 microbatch 持久化稀疏边；若只做阶段性空间变化，在固定四个 checkpoint 重放即可，不重新训练。
4. **G3 研究门槛：** 先明确是否存在稳定、显著的“最终任务失败 GT 在实际训练中监督不足或分配到低 Mask 质量位置”的失配证据。仅相关性不启动 TAL 方法；只有模型保持相同预算、对照正常输出且有可辨认机制时，才独立预注册一次 TAL 修正实验。
5. **终点**：输出 `TRAIN_ASSIGNMENT_EVENTS`（若实际训练已存在/获准）、`CHECKPOINT_TAL_PANEL`（若相应 checkpoint 可得）、`PER_GT_SUMMARY`、`PROVENANCE_CHECKS` 与描述性机制报告。没有训练中真实日志时只能声称固定 checkpoint 结果；不能逆推出历史完整轨迹。

**本文件不是远端执行命令，没有使用训练 GPU，也没有新日志。**
