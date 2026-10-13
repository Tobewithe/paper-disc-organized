# 后训练执行合同：对象、阶段和冻结清单

2026-10-13；适用于尚未冻结的 POSTTRAIN_RESCOPE 规划。本文是其操作性补充，与主协议共同冻结；不修改既有 Run。当前阶段只由 [ACTIVE_PLAN](../../project/ACTIVE_PLAN.md) 决定。本文没有执行结果，不代表训练授权。

**实现与候选默认：** [工具说明](../../shared/evaluation/POSTTRAIN_TOOLS.md) 与 [机制假说/默认配置](MECHANISM_AND_DEFAULTS.md)。前者明确已实现组件和未联调部分；后者提供 oracle/adapter 候选参数，仍须在实际 Run 前与本地输入和资源一起冻结。

## 1. 阶段编号只作映射，不混用 Gate

| 总控阶段 | 方法协议 | 进入条件 | 动作 / 离开条件 |
|---|---|---|---|
| S0 | 前序 GT Study；方法 G0 接收其结果 | 当前环境、输入和预算合法 | 完成有效原生 GT 账本；官网 AP parity 独立记录 |
| S1 | G0 身份桥接 + G1 finite oracle | S0 账本有效，能重放同一检测 | 固定 P 的有限系数修复机会检查；通过才进入 S2 |
| S2 | G2 | S1 机会门槛通过，训练合同冻结 | 2k、单 seed pilot + 一个等容量输入消融；通过才进入 S3 |
| S3 | G3 | S2 通过 | **先用 S2 已选定 checkpoint** 做全 5000 图配对 AP 验收，不先扩大训练 |
| S4 | 独立确认 | S3 AP/损伤验收通过 | 才规划 20k / 多 seed / 开发外确认；新冻结预算、拆分与停止规则 |

同名 G0–G5 在前序 Study 与方法 Study 中含义不同。机器记录必须携带 study_id + stage_id + protocol_sha256。S4 不是本次已批准任务；未获新增训练预算时可完成只读准备，不启动训练。用户已明确授权的环境和阶段不重复索要许可。

## 2. 唯一对象与分母

从冻结基线 A 的普通 GT 定义：

- F：segm COCOeval@.75 未匹配的 GT，即真实任务失败总体。
- T：F 中有 bbox75 匹配检测 d_b，且该检测自身 MaskIoU(d_b,g)<.75 的 GT。**S1 和 S2 目标效果的唯一主分母是 T。**
- U：F 中有 bbox75 匹配，但该 d_b 自身 MaskIoU≥.75；作为竞争/预算/匹配诊断，不混入 c-only Mask 质量失败分母。
- S：基线 segm@.75 已成功的全部普通 GT；**任务损伤分母是 S**，不只统计“恰好也有 bbox75 配对”的一部分。

每条记录绑定 image_id、annotation_id、bbox75_match_output_id。G1 至少 200 个不同 GT 是有效样本要求，不是从结果中挑 200 个；在看 oracle 结果前冻结全部入选 ID 或确定性抽样规则。报告 eligible/attempted/completed/numerical_failure 与排除理由。求解失败不得静默删除后提高恢复率：若合同规定的数值有效性检查失败则 Gate blocked；对合法但未找到解的样本按未恢复计入分母。

G1 主量 = T 中同一 d_b 的 oracle MaskIoU≥.75 的数量 / 预冻结 T 数量；门槛沿用主协议 ≥20%，且样本数≥200。这是有限优化找到的机会，**不是部署 GT 恢复率**。需要任务恢复标签时，必须将修改后的全部预测重新送入 COCOeval；不能由单对 IoU 达标直接推断任务匹配成功。

S2 固定配对的 image-macro ΔIoU：先在每张含 T 的图中对 T 的 ΔIoU 求均值，再对这些图平均；不把无 T 图片当作零加入。总体 IoU 护栏使用基线 bbox75 配对的全部普通 GT（成功与失败均含），沿用同一 image-macro 定义。方法产生的新匹配不得替换这个固定配对表。

任务损伤 = 基线 S 中方法完整输出经同口径 segm COCOeval@.75 后变为失败的数目 / |S|。同检测 IoU 变差是另一个诊断字段，不能替代任务损伤。

## 3. S0 到 S1：COCO JSON 不含 P/c 的桥接

predictions.json 及 RLE 足够重建 S0 账本，但**不足以单独恢复 P、c0 和网络特征**。S1 启动前检查本地缓存是否含：

- 基线 Run、代码/权重/图像/预处理/实际 input_hw 哈希，批次与图像次序；
- output_detection_id → 原生输出行 → 实际 head Top-K 返回的 raw_id 的确定性映射；
- 对应 P、c0、预测框坐标变换与 decoder；S2 另需真实可用 F；
- zero-delta 重放的 RLE/二值 Mask 与该基线行完全一致。

[native_output_bridge.py](../../shared/evaluation/native_output_bridge.py) 已实现固定版本原生 validator 的观察式导出与两阶段原生 decoder 重放：记录实际 Top-K 返回索引、P/c0、逐输出 F、真实层/shape/offset、batch/ratio_pad 和哈希；按图保留全部输出，在写缓存和离线重放时均核 RLE。源码哈希固定于 v8.4.100，接口漂移直接阻断。代码已提供，真实模型 smoke 仍待执行。

COCO 检测 ID 不能当 raw_id。框相同或四舍五入后的 score 相同不能作为唯一 join 键。同 raw 多类别输出仍保留独立检测行；若新模块依赖类别，其 delta_c 可以逐输出行不同，须明确记录。

如 S0 没缓存，S1 单独预登记**所需图片的最小特征重放**，保留原 batch/rect 几何与输出次序并核对 zero-delta；不改写 S0 的“一次全量验证”预算，也不把缓存缺失变成重新跑全量 AP / raw 的理由。桥接不一致则 blocked_identity_bridge，不能借用旧 raw 的近似特征继续。

## 4. 正式运行前必须补齐的参数

下表是有意保留的执行冻结项，不假装已经知道本地代码和资源。运行前写入该 Run 的合同并锁 SHA；缺项只阻断相应阶段。

| 阶段 | 必须冻结 |
|---|---|
| S1 | T 的选取清单与顺序；oracle 代码版本、损失、ROI、正则、初始化、优化器、学习率、最大步数/重启数、数值检查；dtype、解码阈值、资源上限 |
| S2 | fit/dev 图清单及历史使用审计；真实 F 层/shape、网络结构与 trainable 白名单；GT join 同分/同 IoU tie-break；BCE 区域及保持项权重；optimizer、lr、batch、epoch/step、seed、checkpoint 选择规则与预算 |
| S2 对照 | A=原生/zero-delta；B=完整条件 adapter；C=同结构同参数量、P/z0 联合空间置换的消融，保留相同训练预算、数值与通道；置换规则/seed/训练及推理 phase 按 MECHANISM_AND_DEFAULTS 事先冻结 |
| S3 | S2 checkpoint SHA；5000 图完整清单；相同输出顺序/字段/area policy；AP 不确定性方法、重采样次数与随机种子、成本测量方式 |
| S4 | 扩量必要性、20k/seed 总预算、确认集来源和隔离规则；确认失败的关闭动作 |

G1 的 20%、G2 的 +0.5pp/损伤 1%、G3 的 +0.3 AP 点为当前 proposal 门槛；只有随完整合同在结果前冻结才是正式预注册。不能把旧草案 GATES_SPEC.json 交给新 runner。C 消融用于检验所选条件信息是否有增量证据；B 相对 A 达标本身不能证明 P→c 的唯一因果机制，未胜过 C 时应限制机制主张。

## 5. 统计冻结与小样本处理

- IoU 内部用 0–1；+0.5pp = +0.005，总体 −0.1pp = −0.001。AP 点用 0–100；+0.3 AP 点 = +0.003（若 evaluator 返回 0–1）。所有 JSON 明写 unit。
- G2 image-macro ΔIoU 用配对图像簇 bootstrap 双侧95%区间，下界须>0；方法与基线使用相同抽样图索引。提前冻结次数/seed/分位数算法。
- G2/G3 任务损伤按 [DAMAGE_ACCEPTANCE](../../shared/evaluation/DAMAGE_ACCEPTANCE.md) 实施：总体为冻结 S0 的全部 S，训练前固定 n=min(1024,|S|)；固定 B 最后 checkpoint 和推理合同后、方法评价前，对 GT 简单随机无放回抽样。完整评价抽中 GT 所在图片，只统计抽中 GT 的损伤；图内相关不需要额外独立假设，随机性来自有限总体抽样设计。
- 单侧95%上界由超几何 CDF 精确反演，U≤1%才通过。n=N 的 S3 全量验收报告精确有限总体损伤率，不能由此推断未来图片风险。零事件不使用全零 bootstrap 放行；U>1%但点估计≤1%为 blocked_insufficient_precision，点估计>1%为 failed。不得观察方法输出后换 seed/样本量、补样、选 checkpoint；离线工具必须有真实运行证据支撑。
- 损伤抽样与512张效果开发图的设计分开记录，不能把按图挑出的 GT 假装成简单随机 GT 样本。若基线标签显示效果组不足，只能在方法输出前按主协议规则冻结补充效果图，不改变损伤抽样设计。
- G3 ΔAP 的配对图像 bootstrap 每次重算 COCOeval。重复抽中的图片须复制并赋予新的 image/annotation/detection ID，不能被 COCO 的 ID 去重。冻结双侧95% ΔAP 区间并要求下界>0，同时点估计≥+0.3 AP 点。
- 开发集上反复选模的 CI 只作研发门槛，不能称盲测显著性。其余尚未冻结的重采样参数/资源预算写入 freeze blocker，由本地准备阶段补齐后再开科学 Run。

## 6. 最小验收顺序

输入合法 → zero-delta 同输出复现 → 冻结样本与参数 → 正式执行 → 独立核对统计单位/分母 → DECISION → 更新 ACTIVE_PLAN。
工程失败、证据不足、科学门槛失败分别写 error / blocked / failed，不互相替代。
