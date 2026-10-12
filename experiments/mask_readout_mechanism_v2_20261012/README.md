# YOLO26-Seg Prototype→Coefficient 后训练修复（2026-10-13 已重定向）

> **阶段路由只看 [project/ACTIVE_PLAN.md](../../project/ACTIVE_PLAN.md)**；本 Study 当前方法文件为 [POSTTRAIN_RESCOPE_20261013.md](POSTTRAIN_RESCOPE_20261013.md)，旧 PROTOCOL/GATES 为保留的草案，不是新 Run 的执行权来源。

**当前主线**：在官方预训练模型权重、框/分类及基础 Proto/系数均冻结的前提下，只训练轻量的实例条件 `delta_c` 模块，使正常最终输出中的分割失败得到可迁移修复，并约束原成功实例的误伤。**不研究历史 TAL 训练分配**；前期全量 raw 和 GT 失败账本校正只是评价基础设施。

**当前唯一入口**：[POSTTRAIN_RESCOPE_20261013.md](POSTTRAIN_RESCOPE_20261013.md) → [LOCAL_AGENT_START_HERE.md](LOCAL_AGENT_START_HERE.md)。

执行顺序：取得并校验 [原生 GT 任务失败账本](../official_gt_failure_rebaseline_20261013/PROTOCOL.md) → 在同一个最终预测身份下做有限 `c`-only oracle → 通过机会门槛后才允许在 train2017 对最终输出检测进行后训练 → 安全性与完整 COCO AP 验证。候选集合为 `inference_output`，训练 GT 的**框级标签关联**不是 TAL；推理无 GT，成功/失败分组不输入 adapter。

**历史草案保留（不再是主动执行顺序）**：旧 [PROTOCOL.md](PROTOCOL.md) 与 [GATES_SPEC.json](GATES_SPEC.json) 记录 2026-10-12 尚未冻结的 E0 TAL/共享仿射前置和五臂双种子 20k 方案；现已按用户明确要求被更窄的 post-training 路线取代。它们的 `protocol_sha256` 仍属于历史草案，**不要拿来校验新的重定向文件**，也不默默篡改旧配置。任何旧 Run 和先前已登记阴性证据不覆盖。

现 `study.json` 状态仍为规划期；本对话没有启动训练、GPU 原生 COCO 验证或同最终检测 oracle Run。新 Run 的实际来源、参数、数据清单、哈希、Gate 和结果由本地 Codex 按真实执行记录，不向 GitHub 上传权重/COCO/大型缓存。
