# 跨图冻结响应实验完整性审计

**日期**：2026-09-11  
**审阅者**：Fresh Codex subagent（同家族、只读、临时结论；精确后端型号与推理档位未向审阅者暴露）  
**审阅范围**：S011 跨图冻结响应拟合、训练集校准、锁定评价、汇总、结果报告与本地复核回执  
**总体结论**：**WARN（无 fatal）**  
**完整性状态**：**warn**  
**接受状态**：**provisional / same-family**

本次未发现假 ground truth、评价 GT 进入推理解码、用模型自身极值美化指标、虚构结果文件或把 `NO-GO` 改写成正向方法收益。预设正常支持域门槛确实失败：高 ICI 固定匹配目标的联合均值 mask IoU 相对原解码为 **−0.018 个百分点**，因此 `PAIRED_ANALYSIS.json` 的 `NO_GO` 与结果报告一致。WARN 来自证据边界，而非结果造假：评价使用已探索的 400 张 val 图；旧 val 缓存生成时没有逐文件预提交哈希；只有 3 张固定图做了新鲜前向完全重放；实验只有一个冻结模型、一个读出族和三个像素抽样/打乱种子；空间打乱控制不能排除空间平滑性差异。

## A. Ground Truth 来源：PASS

- 拟合标签来自 COCO train annotation：`crossimage_response_experiment.py:40-51` 将真实 annotation 掩码映射到实际输入网格，`crossimage_response_experiment.py:54-75` 从自身、其他 GT 与背景区域抽样；不是从模型输出生成 reference。
- 校准只在独立的 160 张 train 图上使用真实 GT 覆盖：`crossimage_response_experiment.py:97-139`；拟合 320 图、校准 160 图和评价 400 图由固定哈希选择并分离：`crossimage_response_experiment.py:21-33`。
- 推理解码只接收 `proto/coeff/boxes/detections/input_shape`：`crossimage_response_experiment.py:207-211`；`crossimage_response_decoder.py:33-68` 的特征与阈值路径没有 annotation、mapping 或 ICI 输入。GT mapping 仅在解码后进入固定归属空间测量。
- 任务评价使用 COCO val dataset GT 与 `pycocotools.COCOeval(..., 'segm')`：`summarize_relative_ownership.py:17-31`。同类相邻对分母由全部非 crowd GT、相同类别且 GT bbox IoU>0.05 构成：`summarize_relative_ownership.py:33-45`。
- 判定：没有 GT-at-inference。拟合、校准和评价均属真实数据集 GT；不是 synthetic proxy。

## B. 分数归一化：PASS

- `crossimage_response_decoder.py:49-52` 以拟合得到的正 `own_logit_slope` 把线性分数换算到 own-logit 单位。该正比例变换与训练集校准阈值共同锁定，不参与 AP、IoU、R75 或错误率的事后归一化。
- 拟合阶段显式拒绝非正或非有限 own slope：`crossimage_response_experiment.py:82-91`。没有除以预测自身的 max/min/mean，也没有把评价指标重新缩放到接近 1。
- 原始 AP、逐 GT R75、逐对 R75 和逐实例像素比率均直接保存；结果报告按 0–100 展示时明确标注单位。

## C. 结果存在性与来源链：WARN

- `FIT_COMPLETE.json`、`EVALUATION_COMPLETE.json`、`TASK_COMPLETE.json` 均为 `COMPLETE`；本地 `LOCAL_VERIFICATION.json` 验证三份回执共 28 个哈希条目，并从三个 `fit_samples_s*.npz` 重拟合 ridge，最大参数误差 `7.98e-14`，重算分组/配对均值最大误差 `4.88e-15`。
- 400 图评价分母一致：`evaluation_counts.csv` 为 2,913 个非 crowd GT、2,640 个 bbox50 匹配目标、34,708 个预测；扣除 11 个无有效像素目标后，空间表每臂每域 2,629 个目标。任务表每臂每域 2,913 GT，pair 表每臂每域 614 对。高组正常域为空间 382 目标/116 图、任务 405 GT、pair 432 对。
- 每个联合/打乱臂在正常域和外扩域均有 400 条 intervention 计数记录；每臂有 2,247 个预测邻居合格目标，非零像素改变证明并非零干预假阳性。
- 锁定顺序可验证：`crossimage_response_experiment.py:164-167` 在评价前写 `LOCKED_SETTINGS.json`；评价入口在 `crossimage_response_experiment.py:198-202` 校验拟合回执、源代码哈希与锁；`EVALUATION_COMPLETE.json` 再记录 lock SHA256；汇总在 `summarize_crossimage_response.py:15-22` 校验评价回执和 lock。
- WARN：旧 `full_val_cache_20260911` 的生成回执没有逐文件预提交哈希。补充见证对当前选定 400 个缓存重算哈希，并在固定前三图重新前向得到 proto/coeff/boxes/detections/mapping 全零差，但它只能证明当前一致性与 3 图新鲜重放，不能追溯证明 400 个文件自生成后历史上从未改变。结果报告已在 `CROSSIMAGE_RESPONSE_RESULTS_20260911.md:104-106` 如实披露。

## D. 死代码与实际执行：PASS

- `prepare/features/score/decode` 均由正式评价路径调用：`crossimage_response_experiment.py:209-218`。
- `spatial_and_predictions`、官方任务 `evaluate`、cohort 汇总、三个观测种子平均与 2,000 次图像簇 bootstrap 均有对应结果文件：`crossimage_response_experiment.py:211-227`、`summarize_crossimage_response.py:23-81`。
- own-only ridge 模型没有作为独立任务臂输出，但并非未用死代码：它在校准路径验证正仿射 own-only 阈值与原始 `logit>0` 的逐像素等价，`crossimage_response_experiment.py:127-135`；结果报告也明确说明该控制不等同于更宽松的 IoU 优化阈值搜索。
- 未发现报告中引用了从未调用的指标函数或不存在的 metric key。

## E. 样本、种子与主张范围：WARN

- 实际范围是一个官方冻结 checkpoint、320 张 train 拟合图、160 张不重叠 train 校准图、400 张已探索 val 评价图；三个 seed 是像素抽样和空间打乱 seed，不是三次端到端网络训练。闭式读出每 seed 使用 1,798 个目标、230,144 像素；这些限制在协议和结果报告中均披露。
- 所有拟合 joint/sham seed 均评价，无按 val 结果挑 seed，也无 zero-intervention 选择。任务 CSV 保存每个 seed；报告展示三个 joint AP 和均值，但没有在正文逐项列出每个 seed 的全部空间门槛指标及每个 sham AP。原始机器结果完整，因此是报告完整性 WARN，不是结果缺失。
- 正常域的 high-IoU 门槛没有通过。外扩 20% 的条件性改善不能替代正常域门槛；报告正确将其列为诊断。
- sham 保留 own 响应、邻居支持位置和值分布，但在每个目标的共同支持内打乱邻居 logit：`crossimage_response_decoder.py:33-46`。它同时破坏局部空间连续性，不能单独排除“空间平滑性”解释，也不是邻居身份置换。任何“邻居信息可利用”主张须限定为固定 bbox50 匹配像素诊断、当前空间打乱控制和该冻结 checkpoint。
- 当前数据不足以支持 comprehensive、robust、密集专属、因果根因、新架构有效或可投稿方法提升。结果报告没有提出这些强主张。

## F. 评价类型：PASS — `real_gt`

- **冻结读出拟合/校准**：`real_gt`，使用 COCO train annotation；网络参数冻结，闭式 ridge 只拟合低维读出。
- **任务评价**：`real_gt`，使用 COCO val annotation 与官方 COCOeval segm；AP 是点估计，没有 AP CI。
- **空间错误与 pair recovery**：`real_gt` 的条件性诊断；空间指标条件于原 bbox50 匹配且有有效 GT 像素，任务和 pair 分母保留全部相应 GT。
- **确认级别**：探索性、重复使用 val、单 checkpoint；不能当作未触碰 test 上的确认性结论。

## 关键数字复核

| 检查 | 复核结果 |
|---|---:|
| 正常高组 joint mean − initial：coverage | −0.043 pp，95%区间 [−0.111, +0.014] |
| 正常高组 joint mean − initial：same-neighbor error | −0.210 pp，[−0.314, −0.119] |
| 正常高组 joint mean − initial：background error | +0.079 pp，[+0.008, +0.141] |
| 正常高组 joint mean − initial：mask IoU | −0.018 pp，[−0.069, +0.036] |
| 正常高组 joint mean − sham mean：same-neighbor error | −0.105 pp，[−0.178, −0.047] |
| 正常高组 joint mean − sham mean：mask IoU | +0.012 pp，[−0.022, +0.048] |
| 正常 Mask AP：initial / joint mean | 46.7495 / 46.7804（+0.0309 点，仅点估计） |
| 预设门槛 | `NO_GO`；唯一失败项为 `positive_iou_mean` |

## 主张影响

- **“评价 GT 不进入预测，读出跨图执行”**：支持。
- **“当前冻结联合读出相对原解码及当前 sham 降低固定匹配高组的邻居误分”**：支持，但必须保留固定 cohort、当前 shuffle 和单 checkpoint 限定。
- **“覆盖校准从独立 train 校准图基本迁移到该 val 子集”**：部分支持；只针对宏平均固定匹配高组，不代表每实例严格等覆盖。
- **“正常支持域获得总体掩码或任务恢复提升”**：不支持；预设门槛为 `NO_GO`。
- **“AP 提升 0.031 点”**：作为观测点估计支持；不能称显著、稳健或可投稿提升。
- **“密集场景专属效应、系数相似导致失败、CCL 因果机制”**：不支持。
- **“外扩域存在小的条件性信号”**：可作探索性诊断，不得替代正常域门槛。

## 必须保留的行动项

1. 不启动该线性方案的端到端训练；保持 S011 `COMPLETE / NO-GO`。
2. 若引用正面像素效应，写明固定 bbox50 cohort、当前空间 shuffle、单 checkpoint、已探索 val 与点态区间。
3. 后续缓存生成在创建时写逐文件哈希 manifest；补充见证不得表述为历史不可变性证明。
4. 若形成正式论文表，列出所有 joint/sham seed 的门槛指标，并把三个 seed 明确标为采样/打乱 seed。
5. 若继续验证“邻居身份信息”，增加保留空间自相关而破坏身份对应的主动控制；当前 sham 不足以隔离这一解释。

## 未覆盖项

- 按任务边界未重新运行 GPU 推理、COCOeval 或额外实验。
- 未对 400 张图逐张做新鲜前向；只审阅了 400 文件当前哈希见证与 3 张固定图完全重放回执。
- 未把逐图 RLE 全量下载后独立重跑官方 AP；AP 依赖已哈希的任务表、预测 manifest、正式 COCOeval 代码路径和本地算术复核。
- 未验证硬件、驱动、CUDA 与远端完整环境镜像；协议锁定了权重、主要脚本、Ultralytics ops 与 annotation 哈希。
- 未开展多重性校正、训练数据重抽样 CI、跨 checkpoint 复现或盲测。

审计 trace：`.aris/traces/experiment-audit/2026-09-11_run03/`。
