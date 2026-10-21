# LCF服务器迁移启动配置复核

2026-10-20，Slurm **154 / LCFS-train-ABCD** 已实际开始四卡训练，每臂一块A6000。四臂分别真实记录12/13/15/14步（14:48台北时间快照）；尚无方法效用、拓扑或AP结论。

| 核验项 | 证据与结果 |
|---|---|
| 科学配置 | 模型、loss、目标构建、原生输入/解码等源码逐文件SHA与笔记本冻结相同；模型71,361参数、原20k×3轮、每臂15k步、sourcebatch16/effectivebatch4、AdamW、精度和全部科学Gate不变。 |
| 数据 | 20k训练/5k验证图片逐SHA通过，原三轮顺序及完整标注SHA一致。由原COCO逐实例标注构建GT监督；Ultralytics的空YOLO标签提示不表示本训练没有GT。 |
| 初始化 | 发现跨平台同seed生成的13/17张量不同，已保留阴性；执行包装器现精确复制笔记本零步initial.pt的17张量。真实四臂initial.pt均逐张量一致，优化器为新建空状态；未载入训练后权重。 |
| 环境 | Python3.12.3、Torch2.5.1+cu124、Ultralytics8.4.100、numpy2.4.3、scipy1.15.2；仅本次私有overlay补SciPy和私有Ultralytics副本。原共享环境未改。 |
| 真实smoke | Slurm153四臂各4步，梯度与zero-delta通过，四GPU原生P/c0/F/行身份精确一致；首16图2349条原生与旧G完全重放；最大reserved约4.22GiB。四臂保守估时18.13—19.08小时，并行后预计约19小时，仅为smoke外推。 |
| 原生参照 | 同服务器Ns Mask AP44.0138358220、旧G44.1523932108。相对笔记本差异在既有0.02AP点迁移护栏内；最终同环境评价必须用服务器固定身份、完整5000图和原成功22958分母，不能沿用笔记本22957。官方公布parity仍是独立状态。 |
| 预算 | 笔记本失败A542/B541/C541/D541，共2165步保留。服务器从原零步权重重新各15000步，共60000；累计正式尝试上限62165。笔记本smoke16步；服务器151未更新，152和153各16步，共32步单列丢弃。 |
| 实际启动 | 四臂首个更新均为同四图564657/564743/564982/565042，均176连接点+55分离点；A/C对照匹配231/231，无fallback。实际优化器配置逐项相等。此项说明代码正在执行，不是效用阳性。 |
| 产物保护 | 原失败Run及151/152工程证据保留；每500步保存非选择恢复checkpoint，正式评价仍只用15000步last.pt。157项启动记录回传SHA通过。 |

[迁移协议](MIGRATION_PROTOCOL.md) · [初始化修订](INITIALIZATION_RECOVERY.md) · [四卡smoke](SERVER_SMOKE_GATE.json) · [实际启动审计](FORMAL_STARTUP_AUDIT.json) · [Slurm登记](SLURM_SUBMISSION.json) · [回传核验](LOCAL_TRANSFER_VERIFICATION.json)。原科学协议为[LCF V2](../mask_local_connectivity_factorial_20261020/PROTOCOL.md)。

## 接续边界

当前已完成迁移、冻结、smoke和正式启动，训练尚未完成；不设置定时任务/目标模式，不自动提交推送。服务器正式优化启动后不自动再次重启。后台作业在训练完成后退出并释放GPU；后续完整评价/统计按原Gate接续，不能把训练完成当作机制成立。当前提交仅包含train，不宣称已部署自动全量评价接续。

接续先查Slurm154、四个Run/run.json及training.jsonl。统计步数以逐条日志/最终checkpoint为准；run.json批次开头的步数快照可能短暂滞后。评价前需核四臂TRAIN_POINTER/最后checkpoint及全部训练身份记录，汇总为同一完整实验；保持旧G参考、原生身份和最终同D投影/等幅机制对照。
