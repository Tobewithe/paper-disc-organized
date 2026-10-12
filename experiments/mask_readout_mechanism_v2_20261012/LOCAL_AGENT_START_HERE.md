# 本地 Codex 执行入口 — 机制证据新 Study

**目标**：在真实数据与冻结计算预算内，完成 `PROTOCOL.md` 规定的“可恢复性 → 无 GT 学习 → 同容量机制归因 → 前瞻确认”证据链，或者在已定义的 Gate 上得到完整、可复核的停止结论。你是执行与科研判断 Agent，不是继续添加临时模块的探索代理。

## 第一次进入（务必按顺序）

1. 在 `C:\Dpan\codexproject\paper-disc-organized` 检查 `git status`、当前分支与已拉取的最新提交。优先阅读根目录 `AGENTS.md`、`project/科研项目通用规范/` 的全部规范、`paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md` 和本 Study 全部文件。不要覆盖用户未提交改动。
2. 读取 `PROTOCOL.md` 全文及 `GATES_SPEC.json`，校验原协议 SHA256；登记本地当前依据的 Git commit、Study ID、执行环境/数据路径/版本、预算和未解决阻塞。**草案不是正式启动许可证**；开始任何科学 Run 之前必须冻结新 Study 的协议/门槛并保存精确 SHA 和 commit。正式训练只在 E0 和 E1 条件满足后授权。
3. **仅只读核查 E0**：研究 `experiments/coefficient_shared_affine_alignment_audit_20261011/` 原 `PROTOCOL.md`、`DECISION_ADDENDUM_20261011.md`、`queue_cross_audit.ps1`；核对本地及已获授权机器上现有 `RUN_CROSS_QUEUE_*`、`RUN_CROSS_OLD160_*`、`RUN_CROSS_OFFICIAL640_*`、`RUN_CROSS_EVAL_*` 的真实状态、产物和来源哈希。**不得把 GitHub 没有 Run 当作从未启动，也不得重复开启未知状态队列。**
4. 根据旧预注册判词做唯一决定：无效/缺失 → 只收集或完成已有交叉审计；有效阳性 → 走原共享末层对齐训练路线，新 Study 的 20k 阻塞；有效非阳性 → 进入本 Study E1。对未达收敛、缺分片、缺原图评估不能强行科学判定。
5. 当 E0 允许时，先审计既有 `representation_reachability_20261010` 的 PROTOCOL、报告与脚本。原 F/R smoke 用 RGB pooling/人工候选/代理 loss，**只算模块级 smoke，不当成真实 G0**。实现真实YOLO特征—预测框ROI—两分支TAL/官方mask损失—原图mask/RLE/COCOeval的完整链，并加入本计划的 `F_mean` 和 `R_parallel`。
6. 真实 G0 在 32 fit/8 val 通过所有源锁、候选身份、BN冻结、梯度、零修正、真实COCOeval和模型持久化检查后，才依冻结版启动五训练臂 `C/F_mean/F/R_parallel/R` × 两 seed × 20k/8epoch；A 只作为冻结参考。不得提前启动大训练、私加 seed、调 rank/超参、改历史门槛、改变模型或交换数据。
7. E2 两 seed 后按 Gate 做科学判定。无需新训练的 E3 冻结checkpoint定向机制读出要按协议完成；E4 二值可行性证书、第三seed、封存确认集、外部域与第二尺度均服从原文条件路由，不能先看结果再挑额外样本、挑最好的 seed 或换确认集。

## 实验执行要求

- 每个新 Study/Run 放到其所属实验目录，保留 `study.json`、`run.json`、`PROTOCOL.md`、`REPORT.md`、`SUMMARY.json`、`DECISION.json`、`VERIFICATION.json` 和真实可用的清单/哈希；工程失败和中断 Run 不覆盖。分离 `execution_status`、`artifact_status`、`transfer_status` 与 `scientific_status`。
- 每轮操作前把“作用对象、对照和容量、训练/验证/确认集身份、评价口径、Gate、允许改变范围、资源上限与 Stop 条件”记录清楚。已冻结 Study 不静默改写；确实要修改，则另存版本，先说明它为何不同于已关闭的路线。
- 当服务器/笔记本环境未明确授权、出现缺失权重/数据/依赖、资源不足、旧任务状态未知时，标记 `BLOCKED` 并报告唯一最小排障动作；不猜 SSH 主机/凭据，不越权操作另一任务。
- 正常完整 5k COCO AP 不等于逐图 AP 平均；bootstrap 以图像为簇，重复抽中的图像必须独立重新标识；不要把 GT 辅助 oracle、官方 TAL 正样本、诊断配对和正常 COCO 匹配混成同一评价。
- 不上传图像、模型权重、.npz/.npy、含敏感数据的完整大表；只同步轻量的真实 Run 证据和 SHA/保存位置。独立核验要明确“重新运行了什么”和“只复算了什么”，不声称未执行的计算完成。
- 不把 E1 G0 成功、loss下降、oracle 可恢复率或现有旧 RCMC 分数冒充新方法 AP；不得预填 `DECISION.json` 阳性字段。所有未知使用 `null`/明确状态。
- 一次执行周期结束，主动生成“**结论 → 原始 Run 依据 → 已达到/未达到 Gate → 可信边界 → 唯一下一步**”；保留阴性结果，不持续试模块救阳性。

## 当前最小交付

先交付 **E0 的可审计状态核对与路由**（如果已有结果就登记其数值/原始路径/判定；如果仍在跑只记录真正的任务状态）。只有 E0 允许才进入 E1 代码工作。初次不必生成空的正式 20k Run、伪造计算结果或额外确认数据。

## 研究资源

- 新研究：`PROTOCOL.md`、`GATES_SPEC.json`
- 原交叉审计：`../coefficient_shared_affine_alignment_audit_20261011/`
- 已有表示工程：`../representation_reachability_20261010/`
- 论文证据地图：`../../paper/manuscripts/neighbor_sensitive_p3/EVIDENCE_MAP_MECHANISM_20261009.md`
- 已有强对照：`../frozen_native_mask_comparison/REPORT.md`
