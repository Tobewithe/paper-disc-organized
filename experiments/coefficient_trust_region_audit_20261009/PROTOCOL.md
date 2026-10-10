# 功能空间信赖域读出：不训练 oracle 投影审计（协议先行）

日期：2026-10-09。研究线：`neighbor_sensitive_p3`。本协议在执行前冻结；
所有参数、区域定义、指标和门槛不得在看到结果后修改。本 Study 不训练、无新前向，
只用已回传/已缓存的原生候选与 GT 评价；GT 仅用于构造评价指标，不进入任何求解输入。

## 1. 问题

承接 `PROTOTYPE_COEFFICIENT_BREAKTHROUGH_20261005.md` 的首个可证伪动作：
**在不增加网络信息的情况下，功能空间约束（证据区拟合 + 保护区惩罚）能否保留大部分
oracle 收益，同时显著减少对原本可靠区域的改动？** 本审计把 oracle 修正投影到该受约束
子空间，与全 oracle、无保护投影、随机同规模支持对照比较。

## 2. 面板与对象

- 面板：`coefficient_oracle_alignment_20261009` 的 197 图 dev 导出
  （`RUN_COEFF_ALIGN_EXPORT_S0_R1`，1,375 个 GT 关联 one-to-one 候选，663 原生失败 / 712 原生成功）。
- 每候选已有：160 网格 proto（32×160×160）、c0、预测框（640 网格）、GT 160 网格图、
  GT 框 ROI（p, y）。
- oracle：本 Study 自解，λ=0.003 的 GT 框 BCE 强凸有限解（LBFGS 120 次，与对齐研究一致），
  Δc\* = c\*−c0，Δz\* = P·Δc\*（160 网格全图 logit 变化）。

## 3. 区域与四个臂（全部无 GT 构造）

预测框在 160 网格上的 crop 区域记 R_box。z0 = P·c0。

- **证据区 R_E**：R_box 内 |z0| < m_E 的像素（当前不确定处）。**m_E = 1.0 固定。**
- **保护区 R_R**：R_box 内 |z0| ≥ m_R 的像素（当前可靠处）。**m_R = 1.0 固定。**

| 臂 | 修正量 d | 含义 |
|---|---|---|
| O | Δc\* | 全 oracle 上界 |
| U | argmin ‖P_R d − Δz\*_E‖² + λ_c‖d‖² | 无保护投影 |
| T | argmin ‖P_R d − Δz\*_E‖² + λ_R‖P_R d‖² + λ_c‖d‖² | 信赖域（证据拟合+保护） |
| S | 同 U，但 R_E 替换为 R_box 内**同基数随机像素子集**（种子 20261009） | 随机支持对照 |
| Z | 0 | 不改动参照 |

P_R 表示只在 R_E 或 R_R 上取行的原型矩阵。**主配置 λ_c = 1.0、λ_R = 10，运行前固定**；
敏感性 λ_c ∈ {0.1, 10}、λ_R ∈ {1, 100} 一并报告，但门槛只按主配置判定。
R_E 或 R_R 为空的候选：相应指标记 unknown 并报告计数，不填补。

## 4. 指标

每候选、每臂（预测框内官方阈值 >0 解码后）：

- `iou_arm`：与 GT 160 网格图的 IoU；
- `retention_arm = (iou_arm − iou_c0)/(iou_O − iou_c0)`，仅对 oracle 增益 > 0 的候选计算；
- `prot_change_arm = ‖(P_R d)_R‖₂ / ‖(P_R Δc\*)_R‖₂`（保护区 logit 改动相对 oracle 自身改动）；
- `norm_d_arm`：‖d‖₂。

聚合：eligible 子集（原生失败且 oracle 增益>0）上的 candidate-mean retention；
全体候选上的 prot_change candidate-mean；原生成功子集上的 iou_T − iou_c0。

## 5. 预登记门槛（5000 次图像簇 bootstrap，seed 20261009，95% percentile）

- **G1 保留**：主配置 T 的 mean retention ≥ 0.80，且 bootstrap 下界 ≥ 0.75。
- **G2 保护**：mean prot_change_T ≤ 0.80 × mean prot_change_U，即
  (prot_T − 0.8·prot_U) 的点估计 < 0 且 bootstrap 上界 < 0。
- **G3 结构对照**：mean retention_T − mean retention_S > 0，bootstrap 下界 > 0
  （否则收益只是维度压缩，不是区域结构）。
- **G4 成功护栏**：原生成功子集上 mean(iou_T − iou_c0) ≥ −0.005，bootstrap 下界 ≥ −0.01；
  不满足只记录损伤风险，不单独否决 G1–G3。
- **放行**：仅当 G1∧G2∧G3 同时满足，才登记短预算训练协议（网络只预测低维证据，
  固定求解器输出 c'）；否则按 `STOP_ROUTE` 结案并记录具体未达项。
- 若结论只在 GT 选定区域成立、或固定候选解码不成立，记为 oracle 几何现象，不进入方法训练。

## 6. 停止与边界

- 固定求解与 λ 后即停；不追加扫描、不按结果调 m_E/m_R/λ、不挑子集。
- 本审计是机制读数：retention 高不等于 AP 收益；prot_change 低不等于部署安全。
- 面板为历史复用图（非盲测）；诊断关联非同 COCO 匹配；无任何 AP 声明。
- 方法贡献声明仍受文献边界约束（加权最小二乘/信赖域/低秩投影均有先例）；
  本审计只判定该路线是否值得进入训练，不构成新颖性证明。

## 7. 执行安排

笔记本 `ssh 28358lan`；解释器 `C:/Users/28358/anaconda3/envs/pytorch/python.exe`；
vendor `D:/coco_wire/vendor_8.4.100`；执行根 `D:/coco_wire/experiments/coefficient_trust_region_audit_20261009`。
Run 链：smoke（前 64 候选）→ 正式审计（1,375 候选）→ 独立核验（封闭解重算、
bootstrap 确定性、门槛重判）→ 报告与 DECISION。产物按项目约定回传并附 manifest/transfer。
