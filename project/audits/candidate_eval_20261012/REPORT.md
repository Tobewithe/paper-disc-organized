# 候选身份与评估口径历史审计：阶段性裁决（2026-10-12）

## 0. 唯一决策

**停止基于旧 GT-几何 raw 配对继续扩展共享系数修正方法；保留已取得的几何候选诊断数据。新机制研究先以正常输出失败为任务入口，再分别追踪 raw 能力和官方 TAL 监督。** 不重新训练全部历史实验，不撤销已有合规 COCOeval 的阴性结果。

这不是对全部历史研究的终审：仓库镜像缺失部分原始缓存和逐候选结果，无法从文字报告补造重算。针对会改变科研结论的缺口，只允许最小重放。

## 1. 已经做完的审计工作

- 仓库递归树盘点：333 个含 study.json 的实验目录，78 个顶层 REPORT.md，86 个顶层 PROTOCOL.md，1,015 个 Run 目录。全部保留在 STUDY_INVENTORY.json（路径+文件存在+blob hash）；此层 **仅文件审计，不是 333 项科研结论已经复核**。
- 78/78 个顶层 REPORT.md 已逐文件读取并完成关键词信号扫描（0 个读取失败），细节见 REPORT_SIGNAL_SCAN.json。共有 15 篇提及几何关联等信号、20 篇提及官方 TAL、26 篇提及 COCOeval、30 篇提及适用范围警告；标签可重叠，仅用于排查，不是已判定违规数量。
- 对最可能改变当前论文主张的 23 条证据进行了人工上下文复核，结论与追溯路径在 CLAIM_AUDIT.json：verified 1，scoped 16，misattributed 1，needs_replay 4，invalid 1。每条状态只限定到记录指明的事实/Run。
- 新增项目通用规范 `候选身份与评估口径.md`，从项目规范 README 和 AGENTS.md 强制引用。增加 E0 同槽历史官方基线一致性程序、可阻塞的 E0 裁决入口、合成测试和补充 Gate 文档；未修改可能在本地排队执行的原求解/评估脚本。

## 2. 主要实证判断

### A. 旧几何 raw 确实与官方 one-to-one TAL 严重不对齐

2026-09-24 原生几何匹配的 1,301 个 val raw 中：311 个仍是官方 TAL 给同一 GT 的正样本，985 个在冻结 TAL 回放中非正样本，另 5 个分给其他 GT。旧匹配是 GT 与全 8,400 raw 的 BoxIoU≥0.5 最大基数/最大框 IoU，**不是** TAL 正样本重放、也不是最终部署预测。

在同样原图 COCO GT + 正常 mask decoder + image-macro IoU 下：
- 几何位置用旧共享仿射：0.753711→0.808549，+5.4838pp；
- **同一份旧仿射参数** 回放在官方 TAL 位置：0.798922→0.797101，−0.1821pp（区间跨 0）；
- 用官方目标在 TAL 位置求解：0.79904→0.78919，−0.985pp，但求解器梯度约 0.00676，不能称已达到统一精修门槛。

问题主要出在候选人群/特征/预测框随位置改变，并非两份 IoU 最后用了不同的 COCO GT；311 个精确同 raw 上的 native IoU 两套代码平均绝对差约 1.4e-5。

### B. 2026-10-04 已做过位置 × 目标的补格，不应再“从零设计 2×2”

position_objective_complement 的已完成报告：
- 官方 TAL 位置 + 旧目标（O-GB）：val −0.692pp；
- 几何位置 + 官方目标（G-OFF）：val +5.269pp，但 L-BFGS stationarity 约 0.045，非严格最优解；
- 几何位置 + 每层常数偏置（G-BIAS）：+5.483pp，stationarity 5.48e-8；
- G-BIAS 直接迁移到官方位置：+0.086pp，区间跨 0。

这组结果进一步支持“候选位置是主要差异来源”，**但不是精确因果份额或新部署方法成功**。E0 仅需复用这些历史证据并补齐严格收敛的官方同位 B/C 与独立基线 Gate；不再另建一个重复的 2×2 项目。

### C. 所有训练阴性都归因于旧几何规则，是错误的

coefficient_data_scaling_20260924、coefficient_direction_scaling_20260925 等协议已明确使用官方 one-to-one 正样本；运行级输入 hash 尚需在论文直接依赖时核对，不能据旧几何历史自动推翻。ACD 2026-10-06 采用 5,000 图 COCO 官方输出与同预算 baseline，Mask AP 方法-baseline +0.037pp，未过 +0.3pp 预注册门槛；这项方法结果也不能靠几何配对问题自动翻盘。

相反，coefficient_oracle_alignment_20261009 的主面板明确是 1,375 个 native_bank 几何 GT 关联候选。其方向学习的阴性结论应保留在该面板与预算下，**原研究对“官方 TAL 位置可学性”的外推不成立**。coefficient_trust_region_audit_20261009 也是 1,375 候选面板，需补上明确来源追踪才能外推。

### D. 还存在其他独立的口径失配

- P3 gate 分配审计中，baseline 89 个 gate 位置仅 32 个归属预期 GT，44 个是背景、13 个归属其他 GT；“位置跨阈值”不能写成“目标获救”。
- 小目标 Mask AR +1.396pp 的表面收益约 94% 来自极少数类别中的两个 GT；官方类平均是合法指标，但不能解释为普遍小目标修复。
- recoverability_rederive_20261010 复导 4,843/6,456=75.02%，旧记录为 4,263/6,459=66.00%；两种定义/ROI/解码不严格一致，不能只挑其中一个覆盖另一个。
- neighbor_sensitive_p3 论文证据图记录旧全量评分曾有类别 ID / 图片范围错误，纠正来源 `experiments/counterfactual_p3_fullcoco_20260914/RESULTS_20260915.md` 未出现在 GitHub 镜像；需本地补齐证据而不是改写论文结论。

## 3. 下一步的严格唯一执行顺序

| 先后 | 最小动作 | 结束凭据 | 不能做 |
|---|---|---|---|
| P0 | 找到官方同缓存的历史逐候选 PER_CANDIDATE.jsonl，以及 E0 B/C 本地真正的 Run（含 .pt、SUMMARY、run.json、PER_CANDIDATE） | 同身份 A 原始基线差 ≤1e-6、B/C gradient_inf<1e-6、完整分片、E0_GATE.json pass | 只凭 9/30/E0 汇总值继续判定 |
| P1 | 若 P0 通过，执行 `adjudicate_e0_with_gate.py`，沿用 10/11 dev 选 B/C、val CI 下界判定 | 有可追溯的 E0_GATE、DECISION、Run/来源 hash | 用 val 重新搜索候选/监督方式 |
| P2 | 审核 10/9 Oracle 对齐与 trust-region 的 1,375 候选，先把论文结论限定到几何面板；只有论文确实需要 TAL 可学性结论时才在官方候选做最小无训练对照 | 修订 claim 适用域或新的同位审计 Run | 为“挽救阴性”盲目增大模型或预算 |
| P3 | 恢复全量 COCO 正确评分的源报告、具体 Run、SHA、映射/图片列表，并逐一核验论文引用 | 本地-仓库 evidence map 无失链 | 将失效评分回填为正确评分 |

P0 有效前，E0 科研终态为 **insufficient_inputs**；并不意味着已经发生错误、也不意味着审计任务必须新跑训练。必要中间产物存于本地/远端，按最小清单回传，不上传 COCO、权重或全量张量。

## 4. 产物入口与重要边界

- `STUDY_INVENTORY.json`：333 个 Study 的全量路径/哈希元数据；未逐 Run 验证。
- `REPORT_SIGNAL_SCAN.json`：78 个顶层报告全文关键词扫描、原位置例句；是排查索引而非自动科学裁决。
- `CLAIM_AUDIT.json`：23 条高后果主张的精确证据范围/风险/最小动作。
- `project/科研项目通用规范/候选身份与评估口径.md`：新 Run 身份与 Gate 规范；配合 AGENTS.md。
- `validate_e0_receipt.py`、`adjudicate_e0_with_gate.py`：只读旧 Run/官方引用，输出附加 E0_GATE 和 DECISION；5 个合成测试已提交，但当前对话环境无法运行 GitHub 仓库及本地缺失的逐候选文件，**不声称已执行测试或 E0 正式复算**。
- `experiments/coefficient_shared_affine_alignment_audit_20261011/E0_BASELINE_PARITY_ADDENDUM_20261012.md`：新完整性门槛。

**仓库镜像问题：** `shared/coco_clean_20260911/EXPERIMENT_SPEC_STANDARD.md` 是指向 `../../project/guidelines/EXPERIMENT_SPEC_STANDARD.md` 的 Git symlink，目标文件不在当前 GitHub tree。未尝试冒充找回历史标准或覆盖用户本地可能存在的目标文件；新有效规范已独立写入并从 AGENTS 链接，后续本地合并时需检查 legacy symlink 来源。

**科学结论边界：** 本轮完成仓库文件/报告级审计、重点 claim 归因与防误判入口；没有存储已验证的全部 1,015 个 Run，不能宣布 333 Study 全部科学终审通过。数据不足要记录未知而非填零或改写成功/失败状态。

## 2026-10-12 指标最小化与 GT–TAL 迁移补充

此前“每个方法必须固定报告五层”已被精简的科学问题驱动规则取代，详见 `project/科研项目通用规范/指标最小化与TAL分类.md`。对每个 GT，官方 TAL 的冻结回放归属形成训练监督视角的独立分类；不能把它覆盖到真正推理的 COCOeval GT 任务状态上。当前 Ultralytics YOLO26 官方文档默认 one-to-many + NMS，`nms=False` 选 one-to-one；旧研究中有意固定 one-to-one 的结果不受当前默认说法否定，但跨 Run 必须核对实际分支。

已新增只读缓存重分类 `summarize_gt_tal_cached.py` 与合成测试 `test_summarize_gt_tal_cached.py`。脚本只使用官方 TAL val 缓存已有的 `all_annotation_ids`、`rows`、`box_iou` 和 `initial_iou` 来生成 GT 级监督位置表；不会重新前向、训练或计算新的 COCO AP。此轮未在本地大缓存运行脚本，**没有新的 GT–TAL 分类数值结果**。缺少官方 GT 全集或原始 Mask IoU 时停止，不能从 TAL 正样本数假定全部 GT 已经覆盖。

## 2026-10-12 TAL 真实训练轨迹与静态重放区分

[动态 TAL 记录协议](TAL_TRAINING_TRACE_PROTOCOL.md)已追加：过去缓存的官方 TAL 正样本是冻结 checkpoint 在当前无增强输入上的重放，绝非权重预训练各步的历史分配。若论文命题涉及“训练过程为何未学习到某些 GT”，必须使用实际训练 criterion 中的正样本事件流或承认历史不可识别；不要要求每一步存储所有 GT×raw 的稠密关系。未来按用户指定新训练预算和 G0/G1 Gate 另行执行，不得把新增设计写成已运行 Run。
