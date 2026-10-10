# Native coefficient branch oracle-alignment decomposition（协议先行）

日期：2026-10-09。研究线：`neighbor_sensitive_p3`。本协议在执行任何读出前冻结；
不得在看到结果后修改参数、门槛、面板或停止条件。本 Study 不训练模型、不作 AP 声明。

## 1. 问题

在固定官方 prototype 空间下，oracle 目标监督能否通过**原生系数分支**迁移到 held-out 候选？
分解为三个独立读数：

1. **对齐（可学性）**：held-out 候选上，各臂预测系数的修正方向与有限 oracle 方向 Δc* 的余弦；
2. **兑现（ realization）**：在原生 IoU<0.75 的候选上，各臂自身系数（不使用 oracle）解码达到 Mask75 的比率；
3. **效用护栏**：配对候选上的 image-macro IoU、coverage、FPR、Mask75 修复/损伤。

## 2. 臂（零新训练，复用 ACD 已审计拟合产物）

| 臂 | 内容 | 来源 |
|---|---|---|
| A | 原生官方系数 c0 | `D:/coco_wire/models/yolo26m-seg.pt`（SHA256 `16b636f0…`） |
| B | coefficient-only BCE，3 epoch，796 fit 图 | `acd_native_coefficient_20261006/runs/RUN_BASELINE_FEASIBILITY_S0_RETRY1/coeff_final_ema.pt` |
| C | action-guided 辅助项，3 epoch，同预算同数据 | `acd_native_coefficient_20261006/runs/RUN_ACD_FEASIBILITY_S0/coeff_final_ema.pt` |

覆盖层为仅含 `model.23.cv4.*` 与 `model.23.one2one_cv4.*`（72 键）的部分 state_dict，
以 `strict=False` 加载进官方 FP32 模型。加载前后逐张量比对：**差异键集合必须恰好是 24 个可训练参数**
（两个分支各 3 层 × 4 项），否则 Run 失败。上游 backbone/neck/proto/box/class 与所有 BN 不变。

两臂的 ACD 用途限制必须保留：ACD 是单 seed 短预算可行性筛查， gate 未过（相对 official −0.071 AP）；
本 Study 不复评 AP，只读取其拟合系数在 held-out 上的机制表现，不外推方法有效性。

## 3. 面板与身份

- 面板：官方缓存转移至笔记本的 197 图 dev 清单（train2017 图，与 796 fit 图集不相交）。
  ACD 训练期间该清单仅作训练内验证监测；属历史复用研究图，**不是新盲测**。
- 候选：官方 one-to-one 解码位置。GT 关联沿用 native_bank 规则：
  同类、预测框与 GT 框在 640 letterbox 几何下 Hungarian 最大基数后最大 IoU，box IoU ≥ 0.5；
  允许一个 GT 被多个候选匹配——这是诊断关联，不是 COCO 匹配。
- 身份字段（每候选）：`image_id, branch=one2one, pyramid_level ∈ {0,1,2}，raw_id ∈ [0,8400)，grid_y, grid_x, stride ∈ {8,16,32}, input_shape=(640,640)`。
  跨臂按身份元组连接；任一卷身份集合不一致的候选记 `unknown`，不静默丢弃。
- 输入几何：LetterBox((640,640), auto=False, stride=32)，FP32，TF32 关闭（与 bank 构建一致）。

## 4. Oracle 目标

对每个关联候选，在 **GT 框 ROI** 上解 7D 式强凸有限 oracle：

    Δc*_λ = argmin_Δ [ BCE(P(c0+Δ), y)/area + λ‖Δ‖²/2 ]，  LBFGS，120 次迭代

λ 主值 **0.003**（与空间教师线一致）；敏感性 λ ∈ {0.0003, 0.03, 0.3}。
记录每候选的解范数与 stationarity 残差；非有限解记 unknown。
`native-failure`：A 臂在预测框内官方阈值解码后 IoU < 0.75。
`recoverable`：λ=0.003 oracle 解码达到 Mask75。

## 5. 读数与统计

- **R1 对齐**：cos(Δc_arm, Δc*_0.003)，按 level 与 COCO size 分层（small area<1024、medium 1024–9216、large>9216，按 GT area）；
  同时报 A 自身对齐（基线事实）。candidate-mean 与 image-macro 并列。
- **R2 兑现**：native-failure 子集上各臂 Mask75 达成率（用各臂自身系数，不用 oracle）。
- **R3 效用护栏**（配对候选，A 为参照）：image-macro IoU 差、coverage 差、FPR 差、Mask75 修复/损伤计数。
- 区间：5000 次 image-cluster bootstrap（seed 20261009，95% percentile），用于 R2 的 C−A、C−B 与 R3 的 C−A。
  区间条件于单次拟合，不含训练 seed 变异。

## 6. 预登记门槛

- **G-transfer（迁移成立）**：C−B held-out 对齐 ≥ +0.02 余弦 **且** C−A 兑现率 ≥ +0.5pp 且 bootstrap 下界 > 0。
- **G-not-learnable（当前参数化/预算下不迁移）**：C 与 max(A, B) 的对齐差 ≤ +0.01。记为
  “oracle 方向监督在该预算/参数化下不迁移到 held-out”，支持受约束读出路线，不证明整类监督无效。
- **G-damage（护栏）**：任一臂损伤率 > 1% 或 image-macro IoU 下界 < −0.1pp → 只保留机制读数，停止效用声明。
- 三者的机器判定写入 `DECISION.json`；门槛不看结果调整。

## 7. 停止与边界

- 固定预算与 λ 网格结束后即停；不追加 seed、不扫权重/温度/门控。
- 本 Study 的读出是机制诊断：对齐提高不等于 AP 收益；兑现率是固定候选上的 GT 框诊断，不是 COCO AP。
- 面板为历史复用图；结论不得表述为盲测或泛化证明。
- **与 7S 的区别**：`coefficient_spatial_error_supervision_20260929` 曾计划对外部空间修正预测器做
  held-out 方向余弦读数，其数值结果未在本地项目核验（记录在 56018 主机）；本 Study 的对象是
  **原生系数分支**（复用 ACD 拟合产物），不是同一实验的重跑。

## 8. 阶段 II（未授权）

分支族可达性（按 level 拟合/评估 h→c 函数族 vs oracle）需要带分支入口张量的 fit 面板 bank，
另立协议；阶段 I 门槛通过后才可登记。

## 9. 执行安排

笔记本 `ssh 28358lan`；解释器 `C:/Users/28358/anaconda3/envs/pytorch/python.exe`；
vendor `D:/coco_wire/vendor_8.4.100`；执行根 `D:/coco_wire/experiments/coefficient_oracle_alignment_20261009`。
Run 序列：smoke（dev 前 8 图）→ 三臂导出（197 图）→ oracle 对齐 → 读数与 bootstrap → 独立核验。
每个 Run 保存 run.json、源码快照、环境快照、manifest.sha256，完成后回传本项目 runs/ 目录并附 transfer.json。
