# 原 S1 同检测机会兑现核对：冻结协议 V1

研究线：原型—系数链路（key `neighbor_sensitive_p3`）。用户已授权在笔记本 `ssh 28358lan` 执行这一次零训练核对。当前路由见 [ACTIVE_PLAN](../../project/ACTIVE_PLAN.md)。本协议在任何新增模型运行和机会交叉统计前冻结，冻结时间、源码及输入 SHA 见 `FREEZE_MANIFEST.json`、`INPUT_EXPECTATIONS.json`。不改变既有 S1/S2/S3 的 Gate 或结论。

## 1. 命题、证据与判别动作

S1 原 global32 有限 oracle 在固定 256 条 Box75 最终检测中找到 122 条同检测 MaskIoU≥0.75 的解；S3 现有 soft-IoU checkpoint 加面积门控有正 AP 收益，但 +0.141573 点未达到 +0.3 点的门槛。尚未完成这两份证据在同一检测上的交叉核对。

- H_L：未兑现的已知机会主要在现有 Δc 修复器一侧。预测：oracle 已修好、无门控 Δc 仍未达到 Mask75 的条数 L 多于门控撤回条数 G。
- H_G：未兑现的已知机会主要被原面积选择器撤回。预测：G>L，且这些被撤回的同一检测在无门控 Δc 下确实达到 Mask75。
- 该分解是当前固定 checkpoint、固定面板的确定性诊断；不由某个样例失败推出机制定论，也不把计数差直接换算为全量 AP 缺口。

**事前裁决**：在全部工程验收通过后，L>G 优先推进“以已知可恢复失败为对象的条件修复学习”，G>L 优先推进“有损伤约束的收益选择器”，L=G>0 则没有单侧优势，优先研究“联合修复收益与损伤的条件决策”；L=G=0 则本面板机会全部兑现，优先研究面板覆盖及同检测成功向 COCO 任务收益的转化。该规则只确定研究优先级，不授权训练或改门控；任何下一实验另行冻结。身份/重放/配置/覆盖任何一项失败，科学状态为 blocked，唯一动作是修复有证据的工程问题并保留失败 Run，不据部分样本裁决。

## 2. 唯一面板和来源

- 原 S1：`RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949`，`TARGET_PANEL.json` SHA256 `170676d3017e4be7cd213ff1094ca28065727807c915b7602ecdaec48238cf31`，256 条、243 张图；沿用原顺序、`image_id/annotation_id/output_detection_id`，不重选、不改配对。
- 原 oracle：同 Run 的 `ORACLE_RESULTS.jsonl` SHA256 `6e7ff1c080d934d742a9c4e10dbf3e0bcb3fdfe57fa9dc7d61edba53cc791767`；使用已保存的 FP32 系数、原图 RLE 和 IoU，不追加求解。134 条未找到解只称“原有限 oracle 未找到”，不称不可修复。
- 原生：`RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170` 的完整最终预测、GT 账本及 P/c0/F 缓存。
- 当前修复器：`RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/last.pt` SHA256 `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`。B 架构，原 2000 图、3 轮、1500 步成品；本次新增训练步为 0。
- 既有无门控输出：`RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/B_predictions.json` 及原全局行映射。
- 既有门控输出：`RUN_S3_FULL_670fffd010664a648e096572fc7c476d/predictions_mixed.json`、新图 `DELTA.pt` 和 `GT_CENSUS.jsonl`。原生面积 ≥96²=9216 才应用 Δc，不扫描阈值。
- 原 2000 张训练图不参与新增训练。本次推断覆盖原 S1 的 243 张验证图，全部复用 S0 已有缓存；不能因“沿用缓存”擅自换成训练图或 S2 的另一个评价面板。

## 3. 执行配置和身份护栏

使用冻结 `baseline_profile.json`、Ultralytics 8.4.100、原官方权重 SHA、COCO val2017 图像/标注 SHA、原生 O2O、FP32 CUDA 和原数值策略。完整包版本/原生源码及 evaluator 来源逐项匹配原 soft-IoU `EVAL_CONTRACT.json`。保留实际 torch deterministic flags 与原生参数 `deterministic=True` 的区别。不重跑 YOLO，不改变原 img640/batch16/rect=True 形成的每图实际形状和 ratio_pad，不更改 mask_ratio=4、overlap_mask、decoder、检测顺序、框、类别、score。

每张图将完整 c0 矩阵、P/F 送入同一 B 模型 `eval()`、`inference_mode()`、phase=`eval:0`。原生 decoder 的实际两阶段二值/缩放顺序照旧，不能替换为自行假定的 sigmoid→缩放→0.5。三次解码依次释放：原生零残差、原 S1 保存的 oracle 系数、现有 Δc。重放 oracle 时仅改该图原目标行；相同 rank 唯一。GT 只用于既有配对与评价，绝不进入 adapter 或面积选择器。

必须核验：

1. 全部 256 条一一对应原 Box75 匹配，普通 GT、同图同类、独立计算 BoxIoU≥.75；原生 MaskIoU<.75，原任务失败。原生 IoU 与原面板/账本在 1e-12 内一致。
2. 243 张图所有最终行的 zero-delta RLE 与 S0 逐字一致；oracle 256 条 RLE 与存档逐字一致、IoU 在 1e-12 内一致。
3. 原 S2 评价覆盖图的无门控 RLE 与存档逐行一致；其余图的 Δc 与 S3 保存值逐元素位相同。全部 243 图的完整门控预测与 S3 原混合预测逐行完全一致。Box/Class/score/顺序/输出行身份均不变。
4. 输入文件、图像、cache、checkpoint、模型 state、源码和实际数值配置运行前后保持不变；不得按结果更换输入或容差。

## 4. 主要指标与分母

设 O 为原 oracle IoU≥.75，A 为当前无门控 adapter IoU≥.75，g 为原生原图二值 RLE 面积≥9216。全部 IoU 都是原全尺寸 GT 对**同一固定检测**，不做 Hungarian、不取其它预测最大值。

在 O+ 的固定 122 条上互斥穷尽：

- L：O 且非 A，当前 Δc 尚未兑现 Mask75；另报 g 开/关。
- G：O 且 A 且非 g，已经修好但被门控撤回。
- R：O 且 A 且 g，修好且保留。

要求 L+G+R=122。总 256 条不删；O− 的 134 条单独报告 A/g 修复与改善，不用于否定表达上限。记录每条 native/oracle/ungated/gated IoU、原生面积和开关，报告正/负/零 ΔIoU、均值；“L”不等同于完全没学到或 IoU 一概下降。

辅助表仅用事前固定的原生面积 S<32²、32²≤M<96²、L≥96²，不扫分组阈值。原 S3 全量 COCO Mask75 任务成功和匹配行从存档账本读取，单独列出；不能默认和固定 Box75 检测相同。无门控全量 COCO 成败本轮不重算，标记未测。该诊断不是新的 S2/S3 效用验收，不声称新 AP、损伤 Gate 或独立泛化结论，不额外构造显著性门槛。

## 5. 预算、停止与验收

只在笔记本 YANGLEI 执行。先只读资源/进程/锁与输入核验；存在其它执行所有权则停止，不接管。独立 smoke 为按 image_id 升序的首 8 张图及其全部原目标；正式 Run 为原 243 张图/256 条。smoke 不用来选择样本、阈值或方法。

smoke 最多 900 秒、正式最多 1800 秒；每图检查进程 RSS<6 GiB、可用 RAM≥768 MiB、磁盘剩余≥2 GiB。沿用原线程/设备策略，每图释放大张量；无训练、梯度、优化器、oracle 求解或基础网络 forward。失败保留 ERROR、日志、配置及已有产物，工程恢复需另立 Run/版本说明。

验收包括完整覆盖、上述全部重放/配置护栏、独立 compressed-RLE 区间求交 IoU/面积复算、独立分类与分母复算、smoke/full 重叠逐条一致、笔记本与本机两次独立核验以及全部回传 SHA。发布 `PAIR_RESULTS.jsonl`、`PAIRED_MASKS.jsonl`、SUMMARY/DECISION/CONFIG_AUDIT、输入与源码快照、真实 Run 元数据、REPORT/配置复核。正式完成后给出有依据的肯定研究方向和唯一下一步；不自动启动该下一步、不提交或推送。
