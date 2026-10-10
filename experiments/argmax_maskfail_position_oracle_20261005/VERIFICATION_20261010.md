# `argmax_maskfail_position_oracle_20261005` 定义核验记录（2026-10-10）

背景：机制论文证据表中 B1/B2/B3（严格 Mask fail 的 66% 可恢复、33.58% 固定 P 受限、
位置选择 1.293pp）原先只以 10-05 合成笔记转述为据。本记录是回原始产物的逐项核验结果。
**执行环境说明**：数据层核验与定义复算在本地桌面（CPU）完成——指定环境笔记本
`ssh 28358lan` 不可达（192.168.10.2:22 TCP 连接失败，2026-10-10 多次重试确认）。
本记录不含任何模型前向或训练；笔记本恢复后复算应在指定环境复核一次。

## 1. 已精确核验（数据层，零不符）

对 `runs/RUN_full_gpu_20261005/PER_CANDIDATE.jsonl`（28,426 行？实为 28,446 行）与
`PER_GT.jsonl`（6,459 行）全量复算：

| 检查 | 结果 |
|---|---|
| `coefficient_recoverable` ⇔ `oracle_best ≥ 0.75` | 6,459/6,459 一致（语义定义成立） |
| per-GT 聚合（baseline_best=max(baseline)、oracle_best=max(oracle)、oracle_median=median(oracle)、oracle_max_minus_median=oracle_best−median） | 6,459/6,459 零不符 |
| `oracle_gain = oracle_mask_iou − baseline_mask_iou` | 28,137 可解行零违例 |
| `oracle_mask_iou ≤ support_upper_bound` | 28,446 行零违例 |
| 求解收敛 | stationarity ≤ 4.56e-06，LBFGS ≤100 次；exit_reason：solver_termination 28,090 / iteration_limit 47 / fixed_support_limited 309 |

## 2. 账目口径（比转述更精确）

- 4,263（66.00%）recoverable + 2,169（33.58%）fixed-P/decoder-limited + **27 support-limited = 6,459**。
- 转述中的“固定 P/解码仍受限约 33.58%”对应 **2,169 个 flag**；完整不可恢复集合为 2,196（33.99%），
  含 27 个**支撑不可行** GT。
- **口径警告**：309 个 `fixed_support_limited` 候选行（属 27 个 GT）的 oracle 列记录的是
  **支撑上界**而非求解值（delta_norm=0、iterations=0、oracle==support_upper_bound）。
  这些行的 per-GT 统计把上界混入 oracle_best/oracle_median；不影响 4,263 的 recoverable 计数，
  但稿件引用 oracle_best 分布时须披露。

## 3. 未能复现（定义层，如实记录）

原始求解脚本**未随回传保留**（实验目录无 scripts/，项目内无任何脚本产出这些字段）。
用已记录的 7D 约定（GT 框 BCE+λ‖Δ‖²/2，λ=0.003，LBFGS 120）重建单个候选
（image 139 / annotation 26547 / raw 1710；存储值 oracle 0.49079755、objective 0.31697536、
delta_norm 3.94057798、baseline 0.34901532）：

| 变体（栅格化×ROI） | baseline | oracle | objective | ‖Δc‖ |
|---|---:|---:|---:|---:|
| 两步 nearest × GT 框 | 0.3214 | 0.5238 | 0.3723 | 4.78 |
| 两步 nearest × 预测框 | 0.3214 | 0.5250 | 0.3531 | 4.72 |
| 直接 nearest × GT 框 | 0.3636 | 0.5476 | 0.3785 | 4.27 |
| 直接 nearest × 预测框 | 0.3636 | 0.4762 | 0.3589 | 4.35 |
| **存储值** | **0.3490** | **0.4908** | **0.3170** | **3.94** |

无一精确复现；存储解的 ‖Δc‖ 小于全部重建变体，提示其目标含**支撑约束**（与
`fixed_support_limited`/`support_upper_bound` 字段的存在一致），但具体形式无记录可考。
letterbox 几何已独立确认（gain 1.0、pad top 11、输入 448×640、网格 112×160，
与 `box_original`/`box_input` 差 11 像素互证）。

## 4. 对稿件的影响与建议

- B1 的**语义**（“固定 P 下存在可达 Mask75 的系数修正”）已精确成立；
  **“66.00%”这个数可按语义定义引用**，但定义记录须引用本文件并披露 §2/§3。
- 若稿件要把 B1 作为核心定量主张，建议做一个**定义完备的复导 Run**（协议先行）：
  以已记录的 7D 目标（λ=0.003，GT 框 ROI）在 argmax 条件严格 Mask fail 面板上重解，
  与 66% 对照；本地缓存（2,462 图 NPZ）+ 标注俱在，单候选 LBFGS 轻量。
  需在指定环境（笔记本恢复）或服务器线程执行，按项目 Run 约定落盘。

## 5. 复导已完成（2026-10-10 闭环）

上述复导已执行完毕：`experiments/coefficient_recoverability_rederive_20261010`
（协议先行、detached 计划任务、独立核验 7/7）。**复导值 4,843/6,456 = 75.02%**
（7D λ=0.003 定义），记录值 66.00%，差 +580 集中于阈值附近的小目标；按协议不调参、
两值并列报告；3 个退化 GT 跳过并有据。稿件以复导值为主、记录值披露。
本文件 §2/§3 的口径警告（309 行支撑上界、27 个支撑不可行）继续有效。
