# LCF-v1/V2：正式启动与实际配置复核

2026-10-20，笔记本YANGLEI。**正式四臂训练已开始，结果尚未产生。** Run为`RUN_LCF_TRAIN_ced92f98a1aa4d8898fe5a24c0b13a39`，独立进程PID24120；回传快照确认四臂各36/15000步，真实更新日志包含连接/分离监督，不是只有计划或模拟测试。无定时任务、目标模式或自动提交推送。训练后同一有限进程按冻结协议执行完整5000图评价与统计，遇异常保留并退出。

## 已通过的输入与工程核验

| 项目 | 实际证据 |
|---|---|
| 训练标注 | 通用目录原JSON仅993图，未用于本轮。完整118287图/860001实例的原文件SHA与历史合同相同：`610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`，复制到本Study独立目录；旧子集保留。 |
| 图片/顺序 | 原20k训练与5000验证逐图SHA全核；同历史sourcebatch16成员/几何及三轮microgroup顺序；有效batch4。没有缩分母、替换图片或恢复SSR。 |
| 模型/原生配置 | 原yolo26m-seg权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，Ultralytics8.4.100私有_site、torch2.5.1/CUDA12.4。FP32、matmul TF32=False、cuDNN TF32=True及其它profile开关逐项恢复。原生batch16/rect/imgsz640/O2O/候选集合/decoder不变。 |
| 数值/参数控制 | 四臂完全同结构、同71361参数，17个参数张量均有有效梯度；同seed0新初始化，不加载旧Adapter初始权重。实际initial.pt张量SHA四臂一致：`b1e7d076a6cd4dbcabe22bc6eeb916062dfe2b84673cc8e3efdd61358b139fa0`。整份checkpoint SHA因臂身份元数据不同，不用于判断张量是否相同。 |
| 实际优化器 | 从四份ACTUAL_OPTIMIZER与initial.pt核AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd1e-4和全部flags，逐项与冻结CONFIG相等。无AMP/EMA/调度/增强/裁梯度或best checkpoint选择。 |
| 真smoke | `RUN_LCF_SMOKE_dce16fdd86a14ed5a9236815bbccdd36`，最大rect训练批16图，各4步；验证首16图几何/P/c0等与历史缓存精确一致。四臂零残差RLE完全相同、所有参数有效梯度。smoke权重丢弃，正式initial为全新零残差。 |
| 独立身份护栏 | `RUN_LCF_NATIVE_GUARD_75bf135207b045b8bcd89598ba680ebd`：16张历史val图，2345条原生及旧G完整JSON/RLE精确相同，2345项point-F张量精确相同；原基模型state不变、优化步0。补充只读核验源码单独SHA记录，不改冻结科学源。 |
| 真连接信号 | 正式第一步同四图，连接点176、分离点55；A/C在同类同难度非关键点成功匹配231/231，无fallback。B/D监督直接回传真实mask logit，没有只优化辅助头的绕行。该记录证明实现生效，不是机制或效用阳性。 |
| 资源/预算 | smoke峰值CUDAallocated2575928832字节、reserved4559208448字节，均低于6GiB；保守训练估时79.92小时含25%余量，低于冻结96小时。预算仍为每臂20k×3轮15000步、总60000步。 |
| 传输/版本 | 启动122项及正式训练起始13项文件回传SHA核验通过，源码与笔记本freeze相同。f70c9a0a是基准HEAD，新增LCF源尚未提交；真实版本以Run source_snapshot与SHA识别，不能假称该commit已包含新实现。 |

## 冻结配置的明确解释

- V2协议SHA：`689fe66ff5d6e72008a56346b794ed976cbfa675d3aa316f8f881537bb43a3aa`；CONFIG SHA：`46c15235154f726f4ac60767154feb04dc7018786f54d505b1fef37d1088f939`；FREEZE SHA：`4072d34afbb2106b674869e8e8e593c934fdf2e51c61552f05ba00135e0f9a3e`。
- V1仅准备冻结与合成测试，未执行真实smoke或正式优化。执行前发现±2硬截幅会挡住高置信错误修正，V2取消硬截幅并重新核输入/控制；V1 freeze与逐文件源码均保留。见[执行前修订](PRE_SMOKE_REVISION.md)。正式训练与smoke均使用V2，运行中没有换loss、幅度限制或预算。
- CONFIG继承的`adapter_initial`只是历史来源路径，未用于新模型初始化；`gates`为旧SSR闲置门槛，LCF统计未读取，不用于本轮裁决；实际初始权重与优化器已经从真实checkpoint检查。**不在运行后清理或改写这份冻结CONFIG**；字段角色和实际值记录在[配置字段使用表](CONFIG_FIELD_USAGE.json)。本轮数值规则以冻结协议第5节及对应统计源码为准，旧S2/S3阈值不被追溯更改。
- `mask_ratio=4/overlap_mask=True`属于保留的原生参数；本轮新增头的监督从原COCO每实例GT投影到原生输入网格，不使用YOLO训练器的重叠标签栅格。四臂一致。训练连接锚点按固定ROI内可见GT部件构建，最终评价完整原图GT；预测框裁剪或栅格化造成的限制不能被误归因为P本身。
- 全部新方法处理所有最终输出；9216面积门控只用于精确重放旧G参照。A/B的ridge读出、C/D局部field和额外监督位置，是预声明干预；未改变基础权重、Box/Class/score、排序或匹配分母。
- 旧G是原2k soft-IoU+门控，不是服务器G_budget。D−旧G报告实际效用，但训练规模和架构不同；D−B/C及同D投影/等幅干预用于机制判断。官方发布AP parity与原生GT账本有效性仍是独立状态，本轮不宣称重现了官方44.1。
- 训练尚未结束，AP/拓扑/损伤没有新结果。自动统计门槛是结案所需证据，仍须按已冻结协议核漏部件/覆盖率、控制有效性及范围，不能靠少画对象换拓扑分数，也不能把自动bool替代完整科学解释。

入口：[冻结协议](PROTOCOL.md)、[实际启动](LAUNCH_RECEIPT.json)、[smoke摘要](runs/RUN_LCF_SMOKE_dce16fdd86a14ed5a9236815bbccdd36/SUMMARY.json)、[正式Run](runs/RUN_LCF_TRAIN_ced92f98a1aa4d8898fe5a24c0b13a39/run.json)、[实际初始化/优化器核验](runs/RUN_LCF_TRAIN_ced92f98a1aa4d8898fe5a24c0b13a39/STARTUP_INITIALIZATION_AUDIT.json)、[正式更新证据](runs/RUN_LCF_TRAIN_ced92f98a1aa4d8898fe5a24c0b13a39/TRAINING_START_EVIDENCE.json)。
