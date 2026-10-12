# 算力受限的 TAL / Mask 监督机制研究：执行决策（2026-10-12）

**决策：不进行从随机权重开始的完整 COCO 预训练，也不要求自己恢复官方预训练轨迹。** 直接固定官方 YOLO26m-seg COCO pretrained checkpoint、Ultralytics 8.4.100 与同一正式评估配置，研究已训练模型中的 *TAL 位置 vs Mask 形成质量*，仅在有足够正向机制证据时从官方权重做有限、配对的微调。

## 1. 研究命题改写

可识别、低成本命题：**“在固定的官方训练后权重上，官方 TAL 选择的位置是否与可用的 Mask 读出位置失配；这种失配能否预测无 GT 的正常推理失败；使用训练 GT 的受控 Mask 监督位置调整，能否在同一官方初始化和训练预算下提高正常输出？”**

禁止越界命题：*“我们证明了官方模型在原始从零预训练中因为长期 TAL 漂移而失败。”* 没有原始训练事件、增强后 GT lineage 和完整 checkpoint 序列就无法回溯这个历史过程。

## 2. 重要源码事实及资源后果

Ultralytics 8.4.100 的 `v8DetectionLoss.get_assigned_targets_and_loss` 传给 TAL 的是 **detach 后的类别预测、预测框、anchor/grid 和当前训练 GT**；没有输入 prototype、mask coefficient、mask IoU。该版本 `E2ELoss` 在分割中对应两套 one-to-many / one-to-one 分配器。

因而如果仅更新 Mask coefficient cv4 / one2one_cv4，**且分类/框及上游特征、BN 状态和输入保持不变**，则同一输入上 TAL 分配是恒定的。每步记录 raw 选择不反映 Mask 分支学习。这一设置下优先只记录首次 TAL 并做同输入静态一致性校验，训练动态 recorder 不进入主实验。

若改变 backbone/neck 或 bbox/class 分支、TAL 配方本身、或增强样本，则 TAL 可变；但增强变化与模型训练变化不能混为一项因果归因。若真的需要轨迹，使用同一批固定输入的 checkpoint replay 与真实训练稀疏事件两条独立时间轴。

## 3. 三阶段最小证据链

**P0：不训练，复用已有官方 checkpoint/缓存。** 统一 COCO GT 身份，产生 `output_state(g) × tal_state(g)` 交叉表；关键是任务真实 Mask75 失败与官方 TAL 选择位置 Box/Mask 质量的关系。已完成的旧几何/TAL 位置 2×2 与 E0 证据必须复用，不可改名重跑。若无可靠 GT 关联、候选身份不一致、Mask decoder 复现失败则停止，不添加训练。

**P1：只对条件样本做最小离线干预。** 固定原模型 class/box/prototype 与同一个 GT，比较当前 TAL 位置、保留真实输出位置和有条件 raw 对照的 Mask IoU/训练目标。GT 优选位置只提供上界；必须同时检查该候选是否有可观察的无 GT 分数/特征信号，否则得不到可部署的选择器。若没有稳定失配或只能通过 GT oracle 选位，终止 TAL 改进假设，不展开全训练。

**P2：仅在 P0/P1 有确实区分性证据时，做从官方权重初始化的双臂微调。**

| 项目 | 标准基线 | 新监督方案 |
|---|---|---|
| 初始化 | 同一未修改官方 COCO pretrained 权重 | **精确相同** |
| 数据与次序 | 相同 train2017 fit 清单、batch、增强与 seed | 相同 |
| 模型参数 | 共同的冻结范围和可训练范围 | 相同 |
| 训练预算 | 同 optimizer、LR/scheduler、有效更新、训练图数量 | 相同 |
| 唯一干预 | 官方 Mask 正样本/损失策略 | 预先声明的 Mask 分配或 Loss 变化 |
| 评价 | 相同无 GT、正常推理/COCOeval | 相同 |

方法-同预算基线是“引入新机制的增量”；方法-**未微调官方原权重**是实用护栏。只好于已经退化的微调基线、却仍差于原官方权重，不得宣传为官方模型分割性能突破。必须报告原成功候选损伤、完整 COCO Mask AP 与标准分支身份。即使在这个预算下成功，也只能证明“从官方 pretrained 初始化的局部适配效果”，**不是**“从零训练改变 TAL 一定更好”。

初筛用已有约 796/197 train2017 fit/dev 分片或可直接复用的更大训练清单，检查 Loss/修复与数值稳定；对有证据的方案再按资源评估扩至约 20,000 张代表性 train 图的固定同预算确认，仍无需从头预训练。若计算资源不足，保留 `budget-limited pilot`，不扩大结论。最终评价应使用完整 COCO val2017 5,000 图；已反复用于开发的 val 不能称为新盲测。

## 4. 复用与避免重复

- `coefficient_shared_affine_alignment_audit_20261011/REPORT.md`：旧几何 raw 与 TAL 仅 311/1301 个 raw 交集；交叉回放的旧修正收益消失。不能用此诊断直接证明“替代 TAL 会提高推理 AP”。
- `position_objective_complement_20261004/REPORT.md`：已完成 2×2；官方 TAL + 旧目标 −0.692pp，而几何位置 + 官方目标 +5.269pp；不得重跑相同四格。
- `coefficient_official_tal_affine_20260930/REPORT.md`：官方 TAL fit 显示共享读出能降低训练 Loss，却未迁移到 held-out；求解器收敛有缺口，仍需按 E0 的现有 Gate 做裁决。
- `acd_native_coefficient_20261006/REPORT.md`：预训练初始化的官方 COCO val 5k 配对微调已有严格先例。ACD 相对同预算 baseline Mask AP +0.037pp，但未过 +0.3pp 门槛；此阴性不能归因于没完成从头预训练。
- `tal_training_assignment_trace_20261012`：真实 TAL recorder **是工程草稿**，没有真实训练 smoke、hook parity、增强 GT lineage 或完成 Run；暂停，把它视为将来必要时的可复用工具，不开启为了采集日志而采集日志的训练。

## 5. 唯一下一步

使用已有官方 TAL val 缓存做 **静态 GT–TAL × 正常输出 GT 交叉审计**。先核实历史 `initial_iou` 与官方原图 COCO GT 的等价性；未有真实正常输出逐 GT outcome 记录时，只生成 TAL 质量表并把任务状态保留 `unknown`，不可自行把 TAL 失败称为部署失败。成果是可追溯的关系统计与明确的失配对象群；无可信新增机制信号则停止，而不启动预训练。

**不支持自动启动新训练，不使用外部远端连接，不向 GitHub 上传模型权重或 COCO 图像。**
