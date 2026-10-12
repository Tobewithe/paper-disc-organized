# 科研项目通用规范：唯一入口

**2026-10-13 整理版。** 本目录提供跨研究线的*行为与评价契约*；**不宣布新的科学结论、不直接启动实验，也不取代已经冻结的 Study protocol**。当前主线、阶段和历史路由由 [project/ACTIVE_PLAN.md](../ACTIVE_PLAN.md) 唯一管理。

## 规范职责（根据任务读取，不默认全部运算）

| 通用文件 | 适用范围 | 不允许推导 |
|---|---|---|
| [候选身份与评估口径](候选身份与评估口径.md) | **涉及 raw、最终预测、GT、COCOeval、冻结输出、跨 Run 对照时必须读**；固定 8.4.100 O2O Head Top-K / 类别展开 / area 与 matcher 口径 | 不把 TAL/GT-oracle/raw 最大值混同真实推理 |
| [指标最小化与 TAL 分类](指标最小化与TAL分类.md) | 一切科学评估选指标需读前半部分；**只有真正研究训练监督/历史 TAL 时**才使用后半部分的 TAL 标签与训练轨迹链接 | 不把 TAL 分配当作新后训练主线的前置必修实验 |
| [训练规模与结论边界](训练规模与结论边界.md) | 拟启动训练、后训练、小样本筛查、扩计算或论文效用声称时 | 不因“训练有希望”自动跳过 Gate 直接20k/多 seed |

**通用最低要求**：每个 Study 固定唯一的科学命题、观察对象（final output / fixed detection / raw / TAL）、数据+权重+代码/分支+输入尺寸+decoder、主指标及必要护栏、资源预算、可证伪门槛和失败停止动作。只交付当前命题必须的数据字段，**不得强制所有实验重算全 raw/TAL/边界指标**。

## 唯一当前计划、优先级和历史关系

1. **用户当次明确任务**；若要求改变科学命题，需要显式标注旧草案被新版本取代，已冻结的旧 Run 不改写。
2. [当前执行总控](../ACTIVE_PLAN.md) 只管 `neighbor_sensitive_p3` 这次用户指定的执行**顺序与依赖状态**，其他研究线按其 `RESEARCH_RULES.md` 和用户当前任务运行。
3. **所属 Study 冻结 protocol 的数值参数/Gate/停止条件在该 Run 内优先**；通用规范仍约束真实身份、防泄漏、审计可追溯，不允许通过新总控偷偷修改旧 Run。新命题用新版本协议，不回填先验门槛。
4. `project/research_notes/`、`project/hypotheses/`、旧 `DEVELOPMENT_PLAN` 和已过时的 `PROTOCOL`、`GATES_SPEC` 都是历史科研资料。不能因标题有“最新/当前”而自动抢占第 2、3 条。

## 本轮特别重要：固定版本输出与官方 COCO

- 对 Ultralytics **8.4.100** 显式确认 `head.end2end=True/False`、`head.agnostic_nms`、`max_det`、`conf`、输入 H×W；`nms=False` 在该版本并非原生验证 O2O 选择器。通常 O2O Head 先 Top-K raw，后 Top-K raw×类别，`conf` 对输出组合行过滤，**同一 raw 可展开多类别**。
- 官方 COCO `bbox` 与 `segm` 各自匹配：主任务 GT 失败来自最终输出的 `segm@.75`，固定 bbox 配对预测自身的 IoU 才能称“框好、掩码差”。统一 `output_detection_id`，绝不拼接不同检测预测。
- 当前权重 SHA 与官方 `yolo26m-seg.pt` 发布资产相同，**公开 Box/Mask AP `52.5/44.1` 与历史自定义 `51.980/43.518` 未对齐**。`published_baseline_parity` 和 `native_gt_ledger_validity` 独立判定；完整细则在 [固定版本来源核验](../audits/candidate_eval_20261012/O2O_TOPK_OFFICIAL_COCO_BASELINE_20261013.md) 与 [本次 GT 任务账本协议](../../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md)。

## 算力与运行事实

先预检数据/模型/依赖/预算，再做全链路小样本 smoke；各 Gate 用冻结输入和代码执行。`planned` 不代表 `completed`；新 `run.json` 记录真实 Run 与命令，工程失败保留。服务器、本地设备、回传状态分开，不因 GitHub 镜像缺少权重/图/NPZ/旧 Run 而推断实验未执行。`project/structure.json` 和自动索引不属于科研修改目标。

**专用 TAL 附录**：只有提出并预注册“冻结 TAL 正样本/真实训练轨迹”的独立命题，才继续读 [TAL 真实训练轨迹协议](../audits/candidate_eval_20261012/TAL_TRAINING_TRACE_PROTOCOL.md)；普通推理后训练修复**不走 TAL**。

