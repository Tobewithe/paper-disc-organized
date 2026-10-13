# 主假说与候选配置 v2（未在真实数据运行）

2026-10-13。属于 [后训练主协议](POSTTRAIN_RESCOPE_20261013.md) 的待冻结补充，不回填历史 Run。参数是成本可控的预设方案，尚无结果证明其最优；本地通过身份/资源检查并锁定清单后才能执行。

## 唯一方法假说

**H1：最终检测位置的原生系数读出，对本实例预测框内的原型空间结构利用不足；显式提供这项结构信息的新增读出器，能够在固定 P 和检测输出时学到更有效的系数残差。**

“利用不足”是待检验解释；不能假定原生网络完全没有空间信息，也不能从 adapter AP 提升直接认定唯一病因。当前先检验 P 的空间条件信息，不同时增加邻居图、TAL、teacher、门控或原型更新。

| 对照 | 输入 / 操作 | 回答的问题 |
|---|---|---|
| A | 冻结原生模型，delta_c=0 | 基线重放是否一致 |
| B | 同输出位置的 F_i、c0、预测框几何、输出类别/score，以及框内 P 和原始 logit z0 的空间采样 | 新增条件读出在正常推理中是否有用 |
| C | 与 B 同结构、参数数目、初始化种子和训练预算；将 P/z0 网格的空间位置联合打乱，保留数值与通道 | B 的效果是否依赖网格的空间排列，而超出这些数值本身的作用 |

C 使用同一个置换联合移动全部 P 通道及 z0，保留每像素通道向量、各通道边际分布和 c0·P 的关系，不打乱通道，也不置零。它破坏空间排列，但也可能造成输入分布变化；B>C 是空间组织增益证据，不能证明唯一因果机制。F_i 不做 GT ROI，不从 GT 选 raw；同 raw 不同类别输出仍逐行保留，并使用各自类别/score 条件，避免把不同监督对象误合成同一个输入。B/C 的监督样本、优化器、步数和选模规则完全一致。

## 原型读出器候选结构

- 从最终输出行追溯真实 raw 位置，取该位置所属层的真实预训练特征 F_i；层名、通道数、offset 和 hash 在本地实测后冻结，禁止凭假设填数。
- 按**预测框**在 P 网格上作 8×8 双线性采样；z0 由同一 P/c0 计算后使用同一采样网格。坐标变换须使用真实 input_hw/letterbox/输出框，align_corners=False；实现和采样边界在 smoke 时核对。
- 空间输入 K+1 通道 → 两层 3×3 Conv（64 通道，padding=1）+ GELU → 全局平均池化（64 维）。
- F_i 按真实金字塔层分别线性投影至64维（不同层输入通道可不同，B/C 使用相同分层投影），c0 线性投影32维，预测框归一化几何4维、score 1维、输出类别 embedding 16维；拼接181维 → Linear128 + GELU → Linear K。
- 最后一层 weight/bias 全零初始化；不加 sigmoid/tanh 截断系数。仅新增模块可训练，原网络参数、BN running state、P、c0、框/类别/score/行次序均固定。
- 先验证每图所有原生最终输出的 zero-delta RLE，不只验证目标 T。C 也走同样的空间分支计算。

空间采样与 C 的置换实现见 [adapter_primitives.py](scripts/adapter_primitives.py)：8×8 网格中心、bilinear、zero padding、align_corners=False；同一预测框映射 P，z0 由同一采样 P/c0 得到。C 的固定置换键为 ablation_seed=314159、image_id、图内 output_rank 和 phase；训练 phase 为 train:0/1/2，评价固定 eval:0。按 SHA256 为64个位置排序得到一一置换，训练与推理均无 GT 输入；不能在结果后更换种子或选置换。

K、F_i 和模型挂接点仍须由本地实际源码/tensor trace 确认。这些是工程冻结项，不允许把伪造零特征当 B 输入通过测试。

## S1 数值配置与评分

使用 [finite_coefficient_solver.py](scripts/finite_coefficient_solver.py) 的默认配置：float64、zero delta、mean BCE + 0.003×||delta||²/2、LBFGS lr=1、max_iter=120、max_eval=150、history=20、strong_wolfe、梯度验收 infinity norm≤1e−6。梯度容差1e−9、目标变化容差1e−12。单起点、无参数扫描。该文件只解系数，不承担图像投影和评价。

loss ROI 固定为预测框在原型网格中的像素支撑；GT 从 COCO annToRLE 在原图生成，再按该前向真实 resize/pad 记录用 nearest 投到网格。ROI 外不参与此优化 loss，**最终 IoU 的并集必须包含原图完整 GT**，不能裁掉预测框外的漏分部分。最后通过同一原生 decoder 将系数转成原图 RLE，计算全原图 MaskIoU。GT ROI 不作为当前 S1 优化输入。

旧 10/10 求解器可作为数学思路参考，不能原样使用其数据/评分包装：源码使用 cv2.fillPoly，并在预测框支撑内计算交并集；新对象要求 COCO RLE 和完整原图 IoU。保留旧结果及其原适用范围，不在本次改写历史数字。

必须通过两类控制：
1. 同 solver/config 的解析非零最优点正控制（脚本已提供），检查归一化、正则项和优化器实现。
2. 真实缓存 P/c0 的原生 zero-delta 与已知系数掩码重放；GT 栅格化、坐标和原图评分由真实桥接测试核对。解析控制通过不代表这项也通过。

S1 梯度未达标但已经找到的合法系数仍可记录为机会下界；未找到足够机会且有未收敛样本时记 blocked_solver_budget。先排查实现和数值状况，不把求解失败解释成 P 没有表达能力。任何改数值预算必须新版本，不能静默补跑追阳性。

## S2 预设训练预算

候选默认：fit2000、seed=0、3 epoch、图像 batch=4、AdamW lr=1e−3 / weight_decay=1e−4、无 scheduler、无 early stopping、固定最后一 epoch checkpoint；B/C 各一次，不以 dev 结果选 epoch。显存不足时在训练前根据 smoke 冻结 microbatch 与梯度累积，保持有效 batch=4；不得运行中悄悄改预算。

监督：训练框级 join 按 class+BoxIoU≥.50、score 降序、同分按原输出次序；每检测选尚未占用的最高 IoU 普通 GT，同 IoU 按 annotation_id 升序。crowd/ignored 不提供普通 GT 标签；未匹配输出只保留基线，不伪造空 GT。

每实例先在固定预测框 ROI 内对 BCE 像素取均值；每图的 GT 项只在有合法 GT 匹配且 ROI 非空的输出中求均值，保持项在全部 ROI 非空输出中独立求均值：L_image = mean_matched(BCE_GT) + 0.1×mean_all(BCE_baseline)。最后按 batch 图片取均值。两个分母独立，未匹配输出数量不能稀释 GT 项。没有匹配时 GT 项为可微零；没有输出或全部 ROI 为空时两项为可微零，图片仍计入 batch 均值；空 ROI 从两个实例分母排除并记录行号。记录 matched_supervised_instances、preserved_instances、empty_roi_rows 和两个独立 loss。实现见 [adapter_primitives.py](scripts/adapter_primitives.py)。GT 目标与冻结基线 soft probability 都 detach；模型 forward 不接触 GT。后续仍以原图 IoU/AP 与损伤验收。

训练数据清单、特征层、可训练参数名、实际步数、预处理和墙钟资源上限需本地填入冻结合同；本文件不声称这些已确认，也不构成当前阶段的新训练授权。

## 门槛来源与论文结论

- S1 ≥200、≥20% 是筛查预算与机会阈值；同时报告 |T|/|F|、found/selected_T、observed_found/|F|。抽样时最后一项只表示已确认子集的占比，不外推。
- S2 +0.5pp 与 S3 +0.3 AP 点是当前 practical effect 的工作门槛，没有被现有证据证明为自然分界。需要在结果前写明资源/应用理由，不做事后门槛敏感性筛选来挽救配置。
- B>A 仅支持该读出配置有效；B 与 C 的配对差、护栏和正常完整 AP 必须同时报告。未体现 B>C 时，不能声称原型空间条件是增益来源。
- 损伤方法已确定为[有限成功 GT 总体的无放回抽样精确上界](../../shared/evaluation/DAMAGE_ACCEPTANCE.md)：训练前固定预算，B 最后 checkpoint 冻结后抽取 n=min(1024,|S|)，用超几何 CDF 反演单侧95%上界；S3 全量时退化为精确有限总体比例。零事件不能自动放行。该界只覆盖冻结开发总体，不证明新场景安全。
