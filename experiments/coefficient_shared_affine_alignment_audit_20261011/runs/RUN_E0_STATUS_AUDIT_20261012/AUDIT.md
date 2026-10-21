# E0 只读状态审计（2026-10-12）

对象：`coefficient_shared_affine_alignment_audit_20261011` 的交叉审计队列
`RUN_CROSS_QUEUE_20261011_053905_129`。只读核查本地仓库副本与已授权笔记本
（`ssh 28358lan`）上的实际状态；未重启任何任务、未修改任何产物。

## 1. 队列真实状态

| 项 | 事实 |
|---|---|
| 队列状态 | `running`/`solvers_running` 冻结于 2026-10-11T05:39:05Z；worker（PID 12128）与两个求解器进程（10792/18312）已不存在，笔记本当前无 python 进程 |
| evaluator | **从未启动**；`RUN_CROSS_EVAL_20261011_053905_129` 不存在；`evaluate_cross.py` 在 scripts 中 |

## 2. 两个求解器 map 的真实状态

| map | 状态 | 收敛 | 产物 |
|---|---|---|---|
| old160 | `RUN_CROSS_OLD160_20261011_053905_129` completed | **gradient_inf = 9.91e-07 < 1e-6 ✓**，fit_images=796 ✓ | run.json / SUMMARY.json / SHARED_old160.pt（model SHA `45a2c5a8…`）/ 日志 |
| official640 | 被打断 | 最后记录 after_block_1，**gradient_inf = 0.00154，未达 1e-6** | **无 SUMMARY/run.json/model** |
| official640 重试 | `RUN_CROSS_OFFICIAL640_FULL_20261011` | 建索引后静默停止 | 仅日志 |
| old160 重试 | `RUN_CROSS_OLD160_FULL_20261011` | 在 `_old_gt_and_support` 崩溃（letterbox padding (0,-1,106,107)） | 仅日志；不用于任何判定 |

## 3. 身份核对

- 笔记本上 `solve_cross.py` / `cross_objectives.py` / `evaluate_cross.py` 的 SHA256 与仓库冻结副本**逐字节一致**（`a77205ce…` / `72091388…` / `38982969…`）；
- 缓存 `INDEX.json` SHA256 `c8c528b1…` 与 old160 run.json 记录的 `cache_index_sha256` **一致**；
- 记录缺口：原队列的 run.json **未存脚本哈希**；本审计用“笔记本现存脚本 = 仓库副本”确立代码身份（脚本 mtime 显示队列启动前 5 分钟定稿后未变）；
- old160 的 letterbox padding 崩溃是**该模式的已知边缘 bug**，不影响 official640 路径（后者用缓存内官方 TAL masks/target_boxes，不走 letterbox 重建）。

## 4. 按预注册判词的路由决定

`DECISION_ADDENDUM_20261011.md` 规定：任一 map 未达 `gradient_inf < 1e-6`、数量不符或评估输出不完整 →
**“输入未达判定状态”**，唯一动作是完成当前交叉审计，不作科学结论。

**E0 判定：`BLOCKED_INPUT`。** 不应用 dev/val 判定规则（官方 640 map 缺失且未收敛）；
不得据 GitHub 无 Run 推断从未启动；不得重复启动未知状态队列——队列已被确认死亡
（无进程、状态冻结、缺口精确定位）。

## 5. 唯一被授权的下一步

按 `GATES_SPEC.json` 的 `gate_routing.e0_unknown_or_invalid` =
`collect_or_complete_existing_cross_audit_only`：**完成现有交叉审计**，即

1. 以**与原队列完全相同的参数**（同脚本、同缓存、`--seed 20261011`、`--mode official640`）
   补跑被打断的 official640 求解器实例，落**新 Run ID**
   `RUN_CROSS_OFFICIAL640_COMPLETE_20261012`（不覆盖、不改写历史 Run）；
2. 求解器收敛（`gradient_inf < 1e-6`、产物完整）后，以**现有 old160 map + 新 official640 map**
   运行 evaluator，落新 Run `RUN_CROSS_EVAL_COMPLETE_20261012`；
3. evaluator 输出完整后，才按 `DECISION_ADDENDUM_20261011.md` 的 dev 选 map + dev/val 双 CI 下界 > 0
   规则作唯一科学判定；在那之前新 Study（`mask_readout_mechanism_v2_20261012`）的真实 G0
   保持 `NOT_AUTHORIZED_BY_GATE`。

## 6. 成本与风险（如实记录）

- official640 的 block 1 实测约 2.9 小时（old160 全程 4 blocks 约 67 分钟）；总 blocks 上限 10，
  实际收敛所需块数未知——这是完成审计的真实成本，不是科学结论；
- 补跑用笔记本 detached 计划任务执行（断连不失进程）；失败重试遵循“同一故障最多两个修复重试”。