# 系数分支可学性分解结案：oracle 监督不迁移（2026-10-09）

Study：`experiments/coefficient_oracle_alignment_20261009`（STUDY_COEFFICIENT_ORACLE_ALIGNMENT_20261009，
研究线 `neighbor_sensitive_p3`），笔记本 `28358lan` 执行，零新训练（复用 ACD 两个已审计拟合覆盖层）。

## 决定

机器判定 **`not_learnable_at_this_budget`**（协议预登记门槛，执行前冻结，独立核验 9/9 通过）：

- 197 图 dev 面板、1,375 个 GT 关联 one-to-one 候选（663 原生失败）；
- 两个 3-epoch 拟合臂与 held-out oracle 方向总体负相关（B −0.0348 / C −0.0324 余弦）；
- oracle 目标臂与同预算普通 BCE 微调臂无实际差异（对齐差 +0.0024 < +0.02 门槛）；
- 修复增益 +1.66pp 全来自普通系数微调（C−B=0），仅为 oracle 可达（61.99%）的 ~2.7%；
- 损伤 6.74%/7.30%（护栏 1%）、image-macro IoU −1.92/−2.03pp。

## 对主线的意义

- “把 oracle/动作目标蒸进原生系数分支”这一族（含 ACD）在当前 seed/预算/参数化下被机制读数关闭，
  并与 ACD 全 5k 门槛失败（−0.071 Mask AP）互相印证；
- 拟合头仅移动 oracle 距离约 22% 且方向相反（P5/大目标最强负相关）——支持“受约束读出/参数化”
  路线（见 `PROTOTYPE_COEFFICIENT_BREAKTHROUGH_20261005.md` 的功能空间信赖域候选，须先过其
  不训练 oracle 投影审计），或按记录转向表示层；
- 阶段 II（分支族可达性）未授权；总目标未完成；本结论限于本配置，不证明整类方法无效。

事实与数字以 Study  REPORT.md、SUMMARY.json、DECISION.json 与 VERIFICATION.json 为准；
per-image 缓存按收集约定保留在笔记本 `D:/coco_wire/.../runs/*/images/`。
