# 第114条恢复后的配置复核

2026-10-20。**原生重新生成精确复现；配置无漂移；完整幅度修复未通过。** 原正式科学实验仍为工程中止，不把本次三个工程Run完成写成256条完成。

| 核验项 | 实际状态 |
|---|---|
| 官方权重/版本 | 与PREPARATION一致：模型SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，Ultralytics8.4.100、torch2.5.1、原冻结Python/依赖/官方源码pins。 |
| 原生输入 | 原batch_index14的16张成员及位置不变；目标270883在position15。imgsz640、batch16、rect=True，实际416×672；重新生成不是batch1方图推理。原数据label cache只读，缺失即拒绝写回。 |
| 实际前向 | 记录3次：后端步长探测1×3×32×32、warmup1×3×640×640、真实batch16×3×416×672。只有这一个真实batch重新生成；Adapter前向0、训练0。 |
| 张量/身份/解码 | 16图P/c0/点F、框/类/score及元数据精确相同；1851个原生RLE完全一致。新缓存另存并回传，不覆盖旧cache；序列化文件SHA不同不代表张量不同，具体新旧SHA在身份回执分别保留。 |
| 精度与配置 | 原生FP32，CUDA matmul TF32=False、cuDNN TF32=True及其余运行值按原合同；没有改为FP64原生评价。FP64只用于已有几何求解及工程筛选。原505项输入及21份科学源码在Run中前后校验，未改科学脚本。 |
| 目标/阈值 | 原0.10*C_RMS、5e-5容差、原方向/随机种子不变。新工程算法只求同一目标幅度；没有按GT/IoU/AP选择结果。C/W候选、原Mask75分母、95%覆盖与统计Gate未变。 |
| 数值界限 | 旧方向幅度与真实FP32前后logit差分开记录。W旧标准22/24、本次双读数16/24，未达完整24；C24/24。不能把仅方向幅度通过称为完整实际压力一致。有限搜索失败不声称数学全域不可能。 |
| 新科学评价 | 无。新smoke与正式256重跑均未启动，因用户要求的“原精度下修复成功”前提未满足。G_budget同面板指标继续null，原完整AP和损伤未重评。 |
| 回传/核验 | 三个Run、157项文件及新缓存SHA通过；独立GPU核验1851个原生RLE和117个保存的压力候选；本机核Run/artifact散列及接受计数，不冒称本机重新跑GPU。 |

原PROTOCOL SHA `fc18e379f8679920ec8f472af2ad8ac955859ee449f7d99d32519d44d09dad12`、CONFIG SHA `d9773c5d0db940dbd9857bab6cc82ceb5ad57215f5d19c0bbc20c06639d14987`保持不变，执行源码仍V3；恢复辅助脚本和两份工程合同均另行冻结并纳入新Run快照。原Run、上一轮报告和其它会话产物未改。

证据：[冻结回执](RECOVERY_FREEZE_20261020.json) · [重新生成Run](runs/RUN_RSM_RECOVER114_f22b016c00f5482eb7551a4acbe3dad9/run.json) · [独立配置核验](runs/RUN_RSM_VERIFY114_1924036e4cb543169c4701a45efcc76b/CONFIG_AUDIT.json) · [本机回执](preflight/RECOVERY_INDEPENDENT_LOCAL_20261020.json) · [回传核验](preflight/RECOVERY_TRANSFER_VERIFIED_20261020.json)。
