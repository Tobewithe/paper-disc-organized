# 机制论文证据表（claim → Run，冻结版）

日期：2026-10-09/10。用途：分析/机制论文（`neighbor_sensitive_p3` 线）的每项主张到原始 Run 的映射。
本表只收录**本会话已回原报告/原始产物逐项核验**的事实；未核验、口径不明或无源可考的数字不得进入稿件。
所有面板均为单权重、项目多次探索过的 COCO 图像；oracle 类事实是 GT 辅助诊断，不是方法结果。

## A. 记账事实（全量 5k val）

| # | 主张 | 数字 | 来源（已核验） | 口径边界 |
|---|---|---|---|---|
| A1 | raw 候选池几何能力（36,335 GT，30,828,840 raw） | 同 raw Box75+Mask75 23,805（65.52%）；好框但全池无好 mask 6,688（18.41%）；好框好 mask 分属不同 raw 531（1.46%）；两者都无 4,824（13.28%） | `experiments/coco_raw_capability_5000_20260916/REPORT.md` §3 | 分数无关 raw 能力诊断；不等于最终输出召回或 AP 分解 |
| A2 | argmax 条件失败分类（同 5k） | Semantic fail 44（0.12%）；Box fail 6,335（17.43%）；**Mask fail 6,459（17.78%）**；misaligned 484（1.33%）；Success 23,013（63.34%） | `experiments/coco_raw_argmax_conditioned_20261005/REPORT.md` | 派生统计，无新前向；argmax-correct ≠ 通过 conf/top-k/NMS |
| A3 | 正常输出 Mask75 失败的阶段分解 | 13,568 GT：11,512（84.85%）raw 池无 Mask75；1,797 有好 mask 但类别分≤0.001；153 未进 top-300；103 被 maxDets 排除；3 配对竞争 | `coco_raw_capability_5000_20260916/REPORT.md` §4 | 输出保留诊断；不解释训练根因 |
| A4 | 正常 COCO 基线 | Box AP 51.980 / Mask AP 43.518；Mask75 正常匹配 22,767/36,335（62.659%） | 同上 §2 | 明确 predict/导出协议；不冒称与所有 model.val 默认设置相同 |
| A5 | TIDE 式独立修复上界 | 28,369 固定 Box75 配对中 7,130 mask 失败；只删误报像素 → Mask AP 54.517（+10.999），修复 5,656（79.33%）；完整 GT mask → 61.036（+17.517） | 同上 §5 | 读评价 GT 的理想修复；各 ΔAP 不可加；不是可学习方法 |
| A6 | 前景占比分层 | GT 前景/框面积 <25% 组 62.37% 无 Mask75；≥75% 组 10.46%；同类别+面积一倍区间分层比较差 +30.38pp（109 层，探索性） | 同上 §6 | 探索性关联；可部署门控不能直接用 GT 占比 |
| A7 | 解码对照 | >0.5 二值化 −0.070 AP；连续 logit 缩放再二值化 −0.901 AP | 同上 §7 | 解码控制实验；不声称所有分辨率问题已排除 |
| A8 | 规模分层 | argmax 条件 Mask fail：small 3,888 / medium 1,742 / large 829；raw 池 Mask75 存在率 small 45.15% / medium 82.00% / large 89.53% | `coco_raw_argmax_conditioned_20261005/REPORT.md`；`coco_raw_capability_5000_20260916/REPORT.md` §3 | 两套分层口径不同，不可互换 |

## B. 可恢复性事实（固定 P/系数空间）

| # | 主张 | 数字 | 来源（已核验） | 口径边界 |
|---|---|---|---|---|
| B1 | 严格 Mask fail 总体可恢复率 | **复导：4,843/6,456 = 75.02%**（7D 定义完备复导，7/7 核验）；**记录：4,263/6,459 = 66.00%**（原面板，解码约定仅部分保留）；差 +580 集中于 small；3 个退化 GT 跳过 | `experiments/coefficient_recoverability_rederive_20261010/`（REPORT/SUMMARY/DECISION，核验 RUN_REDERIVE_VERIFY_S0 7/7）；原始记录 `argmax_maskfail_position_oracle_20261005`（PER_GT.jsonl，语义与聚合约精确核验，见其 `VERIFICATION_20261010.md`） | 复导值为主（定义完备可复现）、记录值并列披露；原面板求解脚本未随回传保留，其确切目标四种约定变体均未精确复现；309 行 `fixed_support_limited` 的 oracle 列记支撑上界；稿件核心数字按复导值引用 |
| B2 | 规模分层可恢复率 | small 2,285/3,888（58.77%）；medium 1,237/1,742（71.01%）；large 741/829（89.39%） | 同一 PER_GT.jsonl 复算 | 同 B1 披露 |
| B3 | 位置选择不解释差距 | 同 GT 多 raw 的 oracle 最大 vs 中位平均差 **1.293pp**（全体）；4,504/6,459 有多 raw，其子集均值 1.854pp | 同一 PER_GT.jsonl 复算（oracle_max_minus_median 字段） | 聚合口径已精确核验；oracle 定义披露同 B1 |
| B4 | 独立面板再确认（dev 197 图） | 1,375 个 GT 关联候选中 oracle（λ=0.003 GT 框 BCE）Mask75 达 81.4%；663 个原生失败上 61.99%；oracle Δc 范数 ~4.19 vs 微调臂 ~0.9 | `experiments/coefficient_oracle_alignment_20261009/runs/RUN_COEFF_ALIGN_ANALYSIS_S0/SUMMARY.json` | BCE oracle，与 B1 的 IoU 口径 oracle 不同；dev 面板非盲测 |

## C. 排除链（“为什么原生没实现”的受控证据）

| # | 被排除的解释 | 关键数字 | 来源（已核验） | 口径边界 |
|---|---|---|---|---|
| C1 | 监督量不足（空间 logit teacher） | P−T−P macro IoU +0.013pp CI [+0.005,+0.021]；目标组 +0.016pp CI 跨零；相对原生 −0.017pp 跨零，coverage/FPR 有代价 | `experiments/position_dynamic_prototype_gate_spatial_teacher_20261005/REPORT.md` | 253 dev 图 / 1,816 候选；训练 loss 下降但无效用 |
| C2 | GT-free 归属读出 | 真实空间 ROI 相对同容量 h 对照：IoU −0.0012 / BCE +0.001 / AUC −0.0016，CI 全部跨零（all 组 1,346 实例） | `experiments/coefficient_ownership_readout_20260927/runs/RUN_c0f14ee95ff34304a0e18a2b39768ea4/SUMMARY.json` | 8×8 软归属；196 图已看过；不排除其他归属表示 |
| C3 | 归属/空间证据经 solver | 原路径自投影 −1.3153pp CI 为负；SAM 完整 −4.2256pp；SAM 经 OGPS −4.9903pp | `experiments/external_spatial_projection_20261004/REPORT.md` | 253 图 1,816 候选；结论限于该 SAM/OGPS 配置 |
| C4 | 共享末层可达性（固定特征） | fit 5,872 候选：原生目标 0.419728 vs 精修共享仿射 0.317420 vs 逐实例 oracle 0.201276（回收 46.8%）；val 196 图 GT 框 BCE 降 0.057455 CI [−0.101531,−0.062318]；**Phase B 实掩码**：val 图片 macro IoU 0.753711→0.808549（+0.054838，CI [+0.039546,+0.071756]），Mask75 767→847（修 131/损 51），P3 −0.017023 | `experiments/coefficient_shared_affine_attainability_20260930/REPORT.md`（已逐字核验） | 固定特征 mask-BCE 诊断 + 冻结实掩码确认；不是 COCO AP；收益集中 P4/P5，P3 略损 |
| C5 | 朴素外挂修正（big-global） | 7V：图片 macro IoU +0.092pp CI [+0.022,+0.167]；对单标量阈值 +0.006pp 跨零；504 有效图、3,560 固定 one-to-one 正样本 | 数字经 `project/research_notes/MAINLINE_STATUS_20260930.md` 核验；Run 记录（504 图/3,560 候选/threshold 0.25）见 `coefficient_frozen_confirmation_7V_20260929/runs/RUN_7V_INDEPENDENT_20260929/run.json` | **该 Run 的数值汇总未随产物回传**（本地只有 run.json+脚本）；引用以主线记录为单一来源 |
| C6 | **oracle 目标蒸进原生系数分支**（10-09 新增） | held-out 对齐 B −0.0348 / C −0.0324（P5 −0.118、大目标 −0.129）；C−B = +0.0024（门槛 +0.02）；修复增益 +1.66pp 全来自普通 BCE 微调（C−B = 0）；损伤 6.74%/7.30%；image-macro IoU −1.92/−2.03pp | `experiments/coefficient_oracle_alignment_20261009/`（REPORT/SUMMARY/DECISION，9/9 核验） | 单 seed、3 epoch、796 fit、197 dev；BCE oracle；机制读数无 AP 声明 |
| C7 | **受约束功能空间投影**（10-09/10 新增） | 信赖域 retention 0.302（门槛 0.80）；随机同规模支持 0.588（T−S = −0.286）；保护区 logit 改动 oracle 自身的 6.2% vs 无保护 79.3%；成功候选无损伤（+0.011） | `experiments/coefficient_trust_region_audit_20261009/`（REPORT/SUMMARY/DECISION，7/7 核验） | λ_R 敏感性 0.154–0.491 无一达标；停止为配置级，不否定所有功能空间方法 |

## D. 记录缺口（投稿前必须处理）

1. **B1/B2/B3 无协议记录**：`argmax_maskfail_position_oracle_20261005` 只有 runs/（PER_GT.jsonl 已统计复算，语义与聚合已精确核验，见 `experiments/argmax_maskfail_position_oracle_20261005/VERIFICATION_20261010.md`）。求解脚本未随回传保留，oracle 目标的确切形式（λ/ROI/支撑约束）四种约定变体复算均未精确复现；309 个 `fixed_support_limited` 候选行（27 个 GT）的 oracle 列记的是支撑上界。**建议做定义完备的复导 Run**（7D λ=0.003 目标重解并与 66% 对照）后再作为核心定量主张引用。
2. **“GT ownership 可兑现”分量无本地 Run 引用**：10-05 合成笔记有此转述；只能引用 C2/C3 的阴性部分，或补跑/补引。
3. A1–A4 的原始 NPZ/cache 位置与哈希需在稿件复现附录中固定（原报告 §9 已列 Run）。
4. C5 的数字经项目笔记转引核验（Run 数值汇总未随产物回传，本地只有 run.json 与脚本）；C4 已逐字核验（含 Phase B 实掩码结果）。
5. **环境阻塞**：笔记本 `ssh 28358lan` 于 2026-10-10 不可达（TCP 22 超时）；B1 的数据层复算在本地桌面完成（无模型前向）；复导 Run 需在指定环境恢复后执行。

## E. 不可进入稿件的数字

- 7S/7Q 的 held-out 方向余弦数值（本地未核验，记于 56018 主机）；
- 任何把 oracle/修复写成方法效果、raw 能力写成输出召回、诊断关联写成 COCO 匹配的表述；
- 旧 P3 干预路线（`PAPER_DRAFT*.md`、`DEVELOPMENT_PLAN.md`）的历史数字与本机制线主张混用。
