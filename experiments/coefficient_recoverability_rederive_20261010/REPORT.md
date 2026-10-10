# 固定原型可恢复性：定义完备复导报告

日期：2026-10-10。研究线：`neighbor_sensitive_p3`。协议先行见 [PROTOCOL.md](../PROTOCOL.md)；
执行根 `D:/coco_wire/experiments/coefficient_recoverability_rederive_20261010`（笔记本，detached
计划任务运行，不受 ssh 断开影响）。

## 结果

对 `argmax_maskfail_position_oracle_20261005` 的完整面板（6,459 个严格 Mask fail GT、
28,446 个 raw 事件、2,462 图），用项目已记录的 7D 有限 oracle 定义（GT 框 BCE + λ=0.003 位移
正则、LBFGS 120 次双精度、从零出发；ROI 与 GT 栅格化/letterbox 几何从保留缓存逐项重建）完整重解。
**复导可恢复：4,843 / 6,456 = 75.02%**；原面板记录值：4,263 / 6,459 = 66.00%（差 +580）。

| 分层 | 复导 recoverable/GT | 复导 share | 记录 recoverable/GT | 记录 share |
|---|---:|---:|---:|---:|
| small | 2,817 / 3,885 | 72.5% | 2,285 / 3,888 | 58.8% |
| medium | 1,266 / 1,742 | 72.7% | 1,237 / 1,742 | 71.0% |
| large | 760 / 829 | 91.7% | 741 / 829 | 89.4% |
| all | **4,843 / 6,456** | **75.02%** | 4,263 / 6,459 | 66.00% |

- 差异 +580 全部来自 small（+532）与 medium/large（+29/+19）少量；集中在 0.75 阈值附近，
  与原面板解码/ROI 约定仅部分保留一致（不同于 oracle 族别差异——两者的量级在四种约定重建中一致）。
- 3 个 GT（0.05%）的 GT 框在解码网格上退化（ROI 为空），按协议记为跳过，已在核验中逐个
  从原始记录证实。
- 按协议 **不调参** 使两者一致；两个值都如实报告。稿件以复导值为主（定义完备可复现），
  记录值并列披露。

## 独立核验（7/7 通过）

`RUN_REDERIVE_VERIFY_S0`：每第 7 个候选换固定种子随机起点重解（范数/目标值与存储一致）；
per-GT 聚合与分层计数从逐候选产物独立重算（4,843/6,456 完全一致）；身份与 MANIFEST 核对
（3 个缺口逐个证实为退化 GT 框）；事件数 28,443+3=28,446；headline share 内部一致。
独立核验器的两个实现 bug（按 annotation 取 NPZ 行未匹配 raw_id、退化检查读不存在的字段）
在记录中保留修正轨迹：第一次运行 2 项未过、修正后全部通过，失败输出保留。

## 对稿件的影响

- B1 升级为“定义完备、可复现”的定量事实：`EVIDENCE_MAP_MECHANISM_20261009.md` B1 已更新；
- 论文第 4 节与表 2 同时报告两个值及披露（`paper/manuscripts/neighbor_sensitive_p3/main.tex`，
  编译通过 4 页）；
- 排除链（C1–C7）不依赖具体用哪个 recoverable 口径。

## 边界

- 复导值是“已记录 7D 定义”下的可恢复性；原面板目标未能从保留记录精确复现（其求解脚本未随
  回传保留，见 `argmax_maskfail_position_oracle_20261005/VERIFICATION_20261010.md` §3）；
- oracle 为 GT 辅助诊断，非方法结果；面板为历史 val2017（非盲测）；无 AP 声明。

## 产物

- `runs/RUN_REDERIVE_S0_R1`（主运行；含 SCHEDULER_LAUNCH/TASK_EXIT/PER_CANDIDATE/PER_GT/SUMMARY）；
- `runs/RUN_REDERIVE_VERIFY_S0`（VERIFICATION.json，7/7）；
- `runs/RUN_REDERIVE_S0`（中断记录：harness 重启致 ssh 中断于 800/2,462 图，无科学产物）；
- per-image 大缓存按收集约定保留笔记本；汇总与逐 GT 表回传本项目。
