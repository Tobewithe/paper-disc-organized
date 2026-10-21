# LEA-v1 配置与证据核验

结论：正式计算、配置、产物回传及两机独立复核通过；科学门槛未通过。两项状态不混用。

- 冻结协议SHA：`0fd12efa1393a4619ba000e54d77db53f243be925068dc8529491ed31cbb1f46`。执行恢复只解决Python同名模块导入，原协议不改。
- 版本与实际环境按原生缓存合同一致：Ultralytics8.4.100、torch2.5.1/CUDA12.4、FP32、原runtime_policy。没有重新运行YOLO或Adapter，原batch16/rect几何直接复用；本次不改变mask_ratio、原生decoder、检测集合或评价分母。
- 输入525项哈希与247个正式/烟雾缓存图身份通过；正式256配对/243图，49,976行零残差精确重放，保存的256个C系数重放通过。
- 固定tau1.5，8个确定性等面积随机域，2000次图像bootstrap、seed20261017、linear分位数；没有阈值搜索、重新挑配对或追加求解。
- 完整3,610普通GT和1,735原成功GT；N原生账本精确复现。G/G_B完整输出独立COCO匹配；非Mask字段与原输出排序不变，域外改变像素0。
- [笔记本核验](runs/RUN_LEA_FULL_b383a8db922c451cbc77f2069d385cd7/INDEPENDENT_LAPTOP.json)：256配对、1536个配对RLE检查、2048次随机域重建、3臂完整COCO匹配、4506个缓存原生域重放，全部通过。
- [本机核验](runs/RUN_LEA_FULL_b383a8db922c451cbc77f2069d385cd7/INDEPENDENT_LOCAL.json)：独立重算相同IoU、区间、随机域、完整匹配及门槛，全通过；本机不重跑GPU几何。
- [回传核验](TRANSFER_VERIFICATION.json)：242项文件哈希/尺寸逐项通过，ZIP SHA `2dafaefb78f4deaaa5b6b9c045e0ceed5334287037efd5fad8edd27ba4f8b187`。[原执行端清单](TRANSFER_MANIFEST.json)和[执行端Study快照](EXECUTOR_STUDY_AT_TRANSFER.json)保留；本地结案文件属于回传后的派生产物。
- smoke原失败 `RUN_LEA_SMOKE_6b56df7df3d64803bc21c2824fbced70` 保留error状态。恢复 `RUN_LEA_SMOKE_RECOVERY_1e9fcd1acb274521b4a12cd0ba635e02` 哈希复制既有数据、重算与GPU重放通过；正式 Run不复用其科研结果。
- controller自动停用任务返回1已显式核实处理，随后仅停用本实验两项任务，状态均为Disabled。没有修改其它会话任务；后继LRP使用独立隐藏进程、不新增定时任务。

本次总训练更新、基础模型forward、Adapter forward、Oracle优化均为0。用户后续授权的真实训练是新Study LRP-v1，不能计入或改写本次零训练协议。
