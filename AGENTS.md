# 科研与工作台 Agent 统一规则

**首先读 [当前研究总控](project/ACTIVE_PLAN.md)。** 本页只定义**当前用户指明的研究线**的执行优先级；若用户明确选择另一研究线，则先看 `project/research_lines.json` 和对应 `paper/manuscripts/<line>/RESEARCH_RULES.md`，不可把本线 Gate 强行施加给其它独立论文线。所有科研任务按 [通用规范索引](project/科研项目通用规范/README.md) 选择相关规范；**不要求为每个普通问答重读无关研究文件**。

## 1. 科研决策职责

对科研路线、Gate 是否达标、停止与是否扩大训练，给出**唯一有依据的结论、支撑 Run、可信范围、可推翻条件及唯一下一步**；未知写明具体缺什么，不得拿笼统可能性替代决策。拒绝运行后临时换数据、样本、Loss、候选关系、指标、阈值或训练预算来追阳性。完成一个被预注册否决的实现后，应关闭**该配置**并保留阴性，不宣称数学上全类别永远无效。

## 2. 文档唯一权威与状态

- **当前路由只看 `project/ACTIVE_PLAN.md`。** 新动作、旧计划是否生效、当前唯一 Gate 只能在这里登记一次；不要在旧长日志反复追加“最新下一步”。
- **数值规则以启动前冻结的所属 Study protocol 为准。** 通用规范规定对象和真实性约束，研究线 `RESEARCH_RULES` 补充；当前总控的摘要不是修改旧 Run 冻结 Gate 的许可。用户若改变研究命题，明确 `superseded_by` 并保留旧计划、创建新版本，不静默覆盖已冻结协议。
- 区分 `plan_status` / `execution_status` / `scientific_status` / `artifact_status` / `transfer_status`。`planned`、`draft`、`smoke_passed`、`GitHub 已提交` 都不等于 `completed` 或 `scientifically_supported`；是否已运行必须查看本地真实任务与 `run.json`，**不得把 GitHub 无 Run 当作本地没跑**。
- 老研究线、历史审计、TAL/几何 Hungarian、反事实 GT oracle 与最终 COCOeval 各有身份与范围；不向新方法自动转移旧结论。旧 `EVALUATION_SYSTEMS`/`DEVELOPMENT_PLAN`/`PROTOTYPE_COEFFICIENT_ACTIVE_GOAL` 等出现的“当前”均受当前总控的时间及研究对象约束。

## 3. 本轮核心路线

固定 `yolo26m-seg.pt` 预训练权重，聚焦 **Prototype→Coefficient 的后训练纠错**。先获得合法的真实 O2O 最终输出 GT 失败账本，然后对**同一 Box75 官方匹配检测**只改系数做有限 oracle，机会 Gate 通过才训练新的轻量 `delta_c` 模块。所有基础模型参数、Box/Class/score/排序/检测行身份、原生 P 和 c0 均冻结。**不将 TAL 训练历史审计设置成主线前置 Gate；不启动原旧 5 臂×两 seed×20k 草案；不重跑 30,828,840 raw。** 具体来源及停止条件见 [当前研究总控](project/ACTIVE_PLAN.md) 和 [后训练主协议](experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md)。

## 4. 模型版本与评价语义：只在相关任务检查

- 固定 Ultralytics **8.4.100**：`head.end2end=True` 为 one-to-one，`end2end=False` 为 one-to-many+NMS；该版本 `cfg/default.yaml` 的 `nms=False` 是导出融合 NMS 选项，**不能**当验证分支开关。Head `agnostic_nms=False` 两轮 Top-K 先选 raw 再从 raw×类别选检测行，`conf` 对检测行过滤，故一个 raw 可能贡献多个类别输出；详情见 [候选身份与评估口径](project/科研项目通用规范/候选身份与评估口径.md)。
- 必须区分 `raw_id`、`output_detection_id`、`annotation_id`。COCOeval 的 bbox/segm 是**独立 GT 匹配**，Box75/Mask75 成功不保证是同一预测。诊断“框好 Mask 差”必须固定具体最终预测并计算自己的 Box/Mask IoU；GT 辅助最大 raw 不替代真实部署输出。
- 官方发布 Box/Mask AP `52.5/44.1`，历史自定义导出 `51.980/43.518`；权重 SHA 一致不等于原生验证 parity。**`published_baseline_parity` 和 `native_gt_ledger_validity` 是两项不同 Gate**，按 [真实输出协议](experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) 独立记录。新方法和基线必须严格同配置、同检测集合与评价器。

## 5. 运行前后最小合规

- **执行前：** 核权重/图像/标注/版本源码哈希、实验对象与输出身份、拆分、防泄漏、decoder、主评价、对照、预算、停止、拟更改参数白名单。先资源预检/≤约定 smoke，过 Gate 再正式执行；数据缺失不能偷偷换图或缩小评价集合。
- **输出：** `experiments/<study>/study.json`；每次真实执行独立 `runs/<RUN_ID>/run.json`、`EVAL_CONTRACT` 或同等合同、原命令/环境日志、SUMMARY/REPORT/DECISION、哈希和验证报告。程序中断、未回传和负结果保留，不覆盖伪装成成功。大文件只在本地/服务器，用 SHA/清单关联。
- **训练规模：** 零训练评价与烟雾测试、2k 后训练试验、20k 确认是不同 Gate；不得因为某旧全项目规模指导就自动跳到20k、多 seed 或继续预训练。
- **环境：** 实际环境须由当前会话或用户授权。本地笔记本 `ssh 28358lan` 只是历史已知设备名；租用服务器必须由用户提供当次有效连接信息，不猜 SSH 凭据，不擅自连接/重启另一个会话任务。
- **工作台：** 不直接修改 `project/structure.json`、工作台数据库、`.research/` 或自动索引，不擅自运行 `sync/validate`；需要时按用户指示。当前操作只允许修改约定的仓库文件，不远程启动训练。

## 6. 科研结束时交付

按“**已得结论 → 源 Run/参数身份 → Gate 通过或阻断 → 可信边界 → 单个下一步**”汇报。未实际执行的结果不填数字，历史 AP 与新 AP 不串表；若已经有旧 Run 支撑当前命题，复用它，不重新跑相同实验。确实需要变更当前任务，仅更新总控的路由和新版本 protocol，并留下旧文件完整证据。

