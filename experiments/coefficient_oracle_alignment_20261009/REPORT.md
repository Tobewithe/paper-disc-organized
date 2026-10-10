# 原生系数分支 oracle 对齐分解：有限机会未迁移，当前配置结案

日期：2026-10-09（Asia/Shanghai）。研究线：`neighbor_sensitive_p3`。
协议先行见 [PROTOCOL.md](../PROTOCOL.md)；所有读数、门槛和停止条件在执行前冻结。
本 Study 不训练模型、不作 AP 声明；它是机制读数。

## 结论

**在该预算与参数化下，oracle 方向监督不能通过原生系数分支迁移到 held-out 候选。**
机器判定 `not_learnable_at_this_budget`（预登记门槛 G_not_learnable 通过；G_transfer 未通过；G_damage 触发）：

- 两个 3-epoch 拟合臂的系数修正方向与 held-out oracle 方向总体呈**负相关**（B −0.0348、C −0.0324 余弦，native 定义为 0）；
- oracle/动作目标臂（C）与同预算普通 BCE 微调臂（B）的差异在实际意义上为零（对齐差 +0.0024，CI [+0.0013,+0.0035]，门槛 +0.02）；
- 微调带来的 Mask75 修复增益（663 个原生失败候选上 +1.66pp，CI [+0.77,+2.74]）**全部由普通系数微调产生**（C−B = 0.000，CI [−0.60,+0.59]），仅为 oracle 可达修复（61.99%）的约 2.7%；
- 两臂对原成功候选的损伤率 6.74%/7.30%（CI 均不包括 1% 护栏），image-macro IoU −1.92/−2.03pp（CI 为负）。

这与 ACD 在全 5k 上的门槛失败（相对 official −0.071 Mask AP、配对 mean IoU 略负）在机制上一致：
**短预算系数微调产生的是通用漂移，不是向可恢复方向的学习。**

## 面板与执行

| 项 | 值 |
|---|---|
| 面板 | 197 图 dev 清单（train2017，与 796 fit 图集不相交；历史复用研究图，非盲测） |
| 候选 | 1,375 个 GT 关联 one-to-one 候选（native_bank 规则：同类框 IoU≥0.5 Hungarian；诊断关联，非 COCO 匹配） |
| 原生失败 / 成功 | 663 / 712 |
| 臂 | A 原生官方；B ACD coefficient-only BCE 3-epoch overlay；C ACD action-guided 3-epoch overlay（零新训练，复用已审计拟合产物） |
| oracle | GT 框 ROI 内 BCE+λ‖Δ‖²/2 强凸有限解，LBFGS 120 次，主 λ=0.003 |

Run 链：smoke → 导出 R1（分析输入；S0 因缺 truth160 保留不分析）→ 分析 → 独立核验（9/9 通过，
含换初始化重解 oracle、对齐重算、A 臂逐张量 parity、分层计数、bootstrap 确定性与门槛重判）。

## 主表

### Oracle 可恢复性（本面板，固定 P）

| 读数 | 值 |
|---|---|
| oracle Δc 平均范数 | 4.192（B/C 仅 0.913/0.951，约为 oracle 距离的 22%） |
| Mask75：native → oracle | 51.78% → 81.38% |
| 原生失败子集 Mask75：native → oracle | 0% → 61.99% |

### Held-out 对齐 cos(Δc_arm, Δc*)（candidate-mean）

| 分层 | B（纯 BCE 微调） | C（oracle 目标微调） | n |
|---|---:|---:|---:|
| 全体 | −0.0348 | −0.0324 | 1375 |
| level 0（P3） | +0.0071 | +0.0089 | 391 |
| level 1（P4） | +0.0061 | +0.0080 | 527 |
| level 2（P5） | −0.1179 | −0.1143 | 457 |
| small | +0.0181 | +0.0207 | 532 |
| medium | −0.0216 | −0.0197 | 477 |
| large | −0.1290 | −0.1260 | 366 |
| 原生失败子集 | +0.0118 | +0.0171 | 663 |

### 兑现与效用（原生失败子集 663；区间为 5000 次图像簇 bootstrap，seed 20261009）

| 比较 | 点估计 | 95% CI |
|---|---:|---|
| 兑现 B−A（Mask75 pp） | +1.66 | [+0.77, +2.74] |
| 兑现 C−A（Mask75 pp） | +1.66 | [+0.77, +2.74] |
| 兑现 C−B（Mask75 pp） | 0.00 | [−0.60, +0.59] |
| 对齐 C−B（余弦） | +0.0024 | [+0.0013, +0.0035] |
| image-macro IoU B−A（pp） | −1.92 | [−2.46, −1.27] |
| image-macro IoU C−A（pp） | −2.03 | [−2.58, −1.38] |
| 损伤率 B / C（原成功转失败） | 6.74% / 7.30% | [4.94, 8.70] / [5.34, 9.42] |

## 解释与边界

1. **“把 oracle 蒸进原生分支”这一族（含 ACD）在当前预算/参数化下被本读数关闭**：
   目标换成 oracle 派生并不改变 held-out 方向；增益与损伤都与普通系数微调相同。
2. 拟合头只移动了 oracle 距离的约 22%，且总体方向与可恢复方向相反；大目标与 P5 上负相关最强。
3. 边界：单 seed、3 epoch、796 图预算；oracle 为 GT 框 BCE（λ=0.003）而非 IoU-max；
   面板为历史复用图（非盲测）；诊断关联非同 COCO 匹配；不证明整类监督或全部预算无效；
   本 Study 不作任何 AP 声明。
4. 与 7S 的区别：`coefficient_spatial_error_supervision_20260929` 计划的 held-out 方向余弦读数的
   对象是外部空间修正预测器，其数值未在本地核验（56018 主机记录）；本 Study 对象是原生系数分支，
   不是同一实验的重跑。

## 下一步（按协议门槛）

- G_transfer 未通过 → 阶段 II（分支族可达性）**不予授权**。
- 证据支持的路线：固定 prototype 几何约束的读出（`PROTOTYPE_COEFFICIENT_BREAKTHROUGH_20261005.md`
  预注册的功能空间信赖域候选，先做不训练的 oracle 投影审计），或按记录转向表示层。
- ords 记录：研究线总目标未完成；本配置结束不等于整类方法无效。

## 产物

- `runs/RUN_COEFF_ALIGN_EXPORT_S0_R1`：三臂导出（601 文件，bundle SHA `2966d641…`）；per-image 缓存按约定留笔记本。
- `runs/RUN_COEFF_ALIGN_ANALYSIS_S0`：SUMMARY/DECISION/PER_CANDIDATE（bundle SHA `473de1a3…`）。
- `runs/RUN_COEFF_ALIGN_VERIFY_S0`：VERIFICATION.json，9/9 通过（bundle SHA `acb5fad5…`）。
- 历史：`RUN_COEFF_ALIGN_EXPORT_SMOKE_S0`（smoke）、`RUN_COEFF_ALIGN_EXPORT_S0`（缺 truth160，保留）。
