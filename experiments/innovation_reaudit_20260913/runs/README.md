# 创新方案独立复验

本目录是旁支会话用户明确授权的新实验。原有模型、源码、论文、事实台账和其他运行任务不修改。

## 实验目的与边界

1. 用同一二值化、缩放、原预测框裁切流程重评已有权重，剥离解码变化。
2. 将中心先验与学习模块分开，不能把先验收益归因于原型正交化。
3. 修正对比损失：同图同类、GT box IoU > 0.05 的实例对，在完全相同的空间坐标评价两个实例的 logits；仅在 GT 独占像素上施加间隔。
4. 在冻结原型之外增加可训练 32×32 变换层，让正交损失真正产生梯度，并与同结构无正交损失组比较。这是读出基底相关性正则化，不是原 YOLO 原型网络端到端训练。

数据沿用既有 COCO train2017 1200 图 fit / 300 图 transfer，固定标签、7811 个 fit 目标、3 种子、15 轮、每组3675个优化步。transfer 已经被反复探索，仅用于本轮筛查，不能充当全量 COCO 或未经查看的最终测试。

## 第一阶段

`reexperiment.py` 重评38个分支，保留所有缓存候选与原始分数（空掩码也一致保留）。共同解码顺序：logits bilinear至输入尺寸 → 原预测框裁切 → threshold 0 → 二值掩码缩放到原图 → threshold 0.5。另保留 Original/S032 的延迟阈值分支，直接测量解码顺序效应。

已有3个种子 `ortho_center` 与 BSR 的权重逐元素完全相同：`implementation_audit.json`。冻结原型正交项 `requires_grad=False`。

## 第二阶段

`corrected_suite.py` 等第一阶段完成后串行运行24个读出头训练：

| 模式 | 对照目的 |
|---|---|
| bsr | 配对种子的基础头 |
| ada | BSR + 每实例正标度与偏置 |
| scalar | BSR + 单一可学习全局偏置 |
| contrast | BSR + 同坐标独占像素对比间隔 |
| adapter | BSR + 可训练32×32基底变换，无正交约束 |
| orth | 与adapter同结构、同初始化，加变换后采样通道相关性惩罚 |
| full_noorth | Ada + adapter + 修正对比损失 |
| full | 与full_noorth相同，加正交约束 |

中心先验只在推理时独立开关，不冒称通过训练学得。固定 margin_scale=0.5；不按transfer成绩调参。对比损失权重0.05、间隔1.5、每优化步额外16对；正交权重0.01。不同模式的基本监督目标批次和步数相同，对比分支有额外配对监督与计算成本，不声称计算预算相同。

正交度按单目标采样原型计算后求平均，不把不同图片的原型直接混成一个矩阵。该实验测试可训练读出基底的优化正则化作用，不证明原始P缺少表达能力。

所有训练每轮保存checkpoint。读取和输出均使用本目录冻结的关键旧模块副本。Python为conda pytorch；Ultralytics为既有vendor 8.4.143；不安装或升级依赖。显存分配上限为本进程GPU总显存38%，串行执行。

## 评估和统计

`existing_aligned/`、`corrected_aligned/` 保存官方COCOeval点估计、逐GT与逐对R75记录、每分支RLE预测、解码一致性检查。

`analyze.py` 生成3种子均值和标准差，以及2000次按图片聚类的配对R75、差距变化区间。区间基于3个已训练种子的平均结果，不估计未见种子的完整训练方差，不做多重比较校正；AP仅报告点估计。

`s080_corrected_*` 在同一原图坐标系重新评价既有140例诊断样本，包含48例漏目标像素、46例残留误报及46例原成功对照；必须同时报告救回与损伤。样本分组来自历史诊断，不是随机总体样本。

## 运行

PowerShell，工作目录为本目录，Python路径：`C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe`。

已有任务由隐藏进程运行，PID分别记录在 `reeval.pid` 和 `corrected.pid`。输出为 `reeval.stdout.log`、`reeval.stderr.log`、`corrected.stdout.log`、`corrected.stderr.log`。完成标志为 `existing_aligned/COMPLETE.json` 与 `CORRECTED_COMPLETE.json`。

无需重复启动；不要覆盖正在运行的输出。
