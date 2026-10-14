# Proto 同 ROI 二值 IoU 一致性审计结案

**采用“多数异常已在同 ROI 的 BCE→二值质量之间出现”的定位；停止追加当前 S2 配置的训练。** 已知 3950 个 BCE 下降、原图 IoU 下降的监督配对中，2159 个（54.6582%）在原 proto 网格、原监督 ROI 上的二值 IoU 已经下降，超过运行前冻结的严格多数门槛 50%。本结论定位观测到的分歧区间，不把全部实例归因于一个算子，也不宣称已经找到了实现缺陷。

## 直接证据

笔记本 YANGLEI 正式 Run [RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f](runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f/run.json)，2026-10-14 01:00:48—01:03:38（UTC+8），169.812 秒，exit_code=0。原 2000 图全部处理，218209 个最终输出保留原矩阵形状与次序，14268 个有效监督配对全部覆盖。新增优化步数为 0。

| 已知 BCE↓且原图 IoU↓ 的配对，在 proto 同 ROI 的二值 IoU 变化 | 个数 / 3950 | 占比 | 本次支持的定位 |
|---|---:|---:|---|
| 下降 | 2159 | 54.6582% | 在任何本次后续原生上采样/原图范围比较之前，同 ROI 二值质量已经与 BCE 反向 |
| 持平 | 284 | 7.1899% | BCE 降低尚未改变该级 IoU，原图端出现下降；不能当作 proto 改善 |
| 改善 | 1507 | 38.1519% | 正向 proto ROI 变化到负向原图变化之间发生反转；该区间包含原生解码、尺度与评价范围转换，尚未分离到单一操作 |

来源：[SUMMARY](runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f/SUMMARY.json)、[逐对原始计数](runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f/PAIR_PROTO_AUDIT.jsonl)、[DECISION](runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f/DECISION.json)。50% 是本项固定总体的多数定位和下一步优先级门槛，不是新方法效用门槛，也不是开发集外推断。

预测严格取 logit>0；同一 projected GT、同一训练 ROI 内计算 TP/(TP+FP+FN)。47 个配对在该 ROI 内没有 GT 前景，仍全部保留；零残差/最终 B 的空并集分别为 16/24 个，依预注册规则记 IoU=1 并保存标记。原训练排除的 456 个空 ROI 输出与 14 张无有效 GT 监督图片逐图对账一致，没有为得到方向结果改分母。全部 14268 配对的 proto ΔIoU 均值为 +0.0005691547（+0.0569pp），它不代替上述异常总体定位，更不能覆盖旧原图端的阴性结果。

## 配置复核与证据衔接

**运行后配置核验通过，回传后独立算术复核通过。** [CONFIG_AUDIT](runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f/CONFIG_AUDIT.json)、[LOCAL_RESULT_RECHECK](runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f/LOCAL_RESULT_RECHECK.json)。逐对 BCE 以及逐图 GT loss、保持项、总 loss 相对上一诊断的最大绝对误差全部为 **0.0**；配对、投影、ROI 和各项归约不变。24 个实际源码/协议/配置文件保存完整快照，27 个输入文件前后 SHA 复核；全部 2000 图缓存逐文件校验通过。

| 固定项 | 本次实际使用 / 核验 |
|---|---|
| B checkpoint | 原 RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2 的 last.pt；SHA 201ab6a7372c167e772217904eac8f4ba9946c26e5862c2882e3c01bdcd58e5e；保存状态为原 3 epoch / 1500 步，本次不创建优化器，内存状态前后相同、无梯度 |
| 原缓存 | RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec；TRACE SHA fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d；保留 P/c0/F、原 boxes/classes/scores/输出行和几何 |
| 模型与环境 | 官方基础权重 SHA 16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5；Ultralytics 8.4.100、Torch 2.5.1/CUDA 12.4、Python 3.12.3，与原训练及上次诊断一致 |
| 原生配置 | 复用 imgsz=640、batch=16、rect=True 的缓存实际几何；end2end=True、agnostic_nms=False、conf=.001、max_det=300、mask_ratio=4、overlap_mask=True 的配置合同未改。没有重新对图像分批运行基础模型 |
| 实际数值策略 | FP32、AMP=False、matmul=highest、CUDA matmul TF32=False、cuDNN TF32=True；benchmark/deterministic algorithms/cuDNN deterministic 均为 False，CUBLAS=:4096:8。以实际值对账，不把配置字段 deterministic=True 误报为运行时全部确定性开关开启 |
| 原监督 | 保存的 class+BoxIoU≥.5、score 顺序、一对一普通 GT join；与 bbox75/segm75 任务匹配分别保留。原 GT BCE 与 0.1 保持项、逐 ROI/分组/图像归约不变 |
| 原图端点 | 复用 RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926 的逐对原图 IoU、218209 行 zero-delta 精确 RLE 重放与输出不变量证据；原表和完整 B 预测 SHA 均通过。本次新增测量是 proto 二值 IoU，没有重新声称完成一次 COCO AP 或全量 RLE 重放 |

工程 smoke [RUN_PROTO_AUDIT_SMOKE_2f7ace2444ae4f5281be890fbb8e3e69](runs/RUN_PROTO_AUDIT_SMOKE_2f7ace2444ae4f5281be890fbb8e3e69/run.json) 覆盖原清单前 8 图、664 输出、29 个监督配对，27.125 秒；与正式 Run 的源码/输入/环境相同。5 项行为测试在本地和笔记本通过；正式结果不使用 smoke 代替总体。两个 Run 的 37 项文件分别回传并逐文件校验，正式 zip 2593213 字节。

本地独立复核重新从整数 TP/FP/FN/TN 计算每个 IoU、三方向计数、3950 异常配对的分母和多数判定；检查所有原表字段、配对主键及原图 IoU 没有改写。它是独立算术/身份核验，没有在本地再做模型推理。

## 唯一下一步与边界

**下一步只做这 2159 个配对的零训练前景/背景 BCE 贡献分解。** 输入继续是相同 checkpoint、P/c0/F、GT、ROI 与本 Run 的固定配对，逐对核算前景/背景及 0 阈值翻转像素的 BCE 变化，并用现有 TP/FP/FN 计数闭合核验，确定哪些像素贡献降低了 BCE、同时造成二值质量下降。最低验收是分解总和复原原 ΔBCE（绝对误差≤1e-7）、二值计数完全一致，形成对损失下降来源的可判定结论；本次未启动该后继。

这一步优先处理已超过多数门槛的直接证据。另 1507 个改善后反转实例和 284 个持平实例保留完整记录，本 Run 尚未把其差异定位到上采样、裁切或原图 GT 范围中的具体操作。S2 原配置继续关闭，未新增训练、未进入 S3/S4；本地基线有效与官方发表 AP 尚未复现两个状态不变。

推翻条件：后续发现冻结配对、GT/ROI 投影、BCE 闭合或输入/源码核验有误，则撤销本次定位，先修复测量实现并另建零训练 Run；不以增加训练步数补救测量问题。

[EXECUTION_CLOSEOUT](EXECUTION_CLOSEOUT.json) 记录 01:04:41（UTC+8）笔记本无 Python 实验进程、无阶段锁。当前没有尚在运行的本次任务。

## 核验摘要 SHA

- 正式协议：5d0fd73827a295049eb74a625128b422353f7fdfe008b2a888bc48c6a05d0b81。
- 正式合同：f7c9a8ac5cceebf846a084a32a20f5f737ee6ffa4039051bb8c1f8f51b18d230。
- SUMMARY：4d0074970a52cb55ac309362e4e292772a1d623fa62146c700301f6630eb3a76。
- CONFIG_AUDIT：0eef5c6f30af29fea22ebcec8dffe0ea2c17efe49343ba2e3b0eccaccfe83e78。
- 回传 zip：8fed7489cd9e1906d75b4ad1708be432a77b40fedfddc425171f182d87d86f40。
