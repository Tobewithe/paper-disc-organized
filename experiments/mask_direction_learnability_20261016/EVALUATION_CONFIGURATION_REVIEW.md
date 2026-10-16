# CLD-v1 留出评价后配置复核

**结论：生产配置核验、笔记本独立重放、1352项逐文件回传及本机冻结统计复算全部通过；科学判定为机制/效用两轴阴性。** 本次原400图评价新增训练0；以下按生效参数、真实来源和最终输出核对，不把框架未使用的默认键当作本次配置。

| 检查项 | 冻结配置与实际证据 |
|---|---|
| 科学与执行版本 | PROTOCOL SHA `053aaefb1cbb0fd5f87dad2dabdf9e961660201ae81bae6c8e42e67d61d6d453`；11项科学文件及36项执行文件/Run快照保持。原evaluate_cld.py、cld_statistics.py、verify_evaluation.py、verify_cld.py未改。新增控制脚本只负责调用、预算、日志和任务停用。 |
| 操作授权历史 | 原训练授权SHA `e21a827549beda75323193f7d46364e2c1a7b91f87b06966d43511c2fd3b0aa2`按字节归档；当前授权仅评价及独立复核，training/smoke=false，最大新更新0。旧FORMAL_CONTROL_FREEZE和传输记录中的历史授权应与formal_control_snapshot或归档字节核对，不能拿本次新授权误报旧Run漂移。 |
| REAL last | `RUN_CLD_FULL_REAL_bc58d9f5253847e588eb01ce2a309d7b`，SHA `869d44ca30876cec7a243f774d3b82994d1a7bdcad85347cb646f88015003ff8`，1200步/3轮；模型状态哈希与训练SUMMARY一致。 |
| PERM last | `RUN_CLD_FULL_PERM_a131a5d6626845878b5b93b9b51a7bfa`，SHA `8aa44343099384990324c7a1e9acae5beb982a3d2fe948c685216df68f467d20`，1200步/3轮；模型状态哈希与训练SUMMARY一致。两个last均在第一次留出forward前固定，没有挑轮或替换。 |
| 框架与实际环境 | Python3.12.3、torch2.5.1、CUDA12.4、Ultralytics8.4.100、torchvision0.20.1、numpy2.4.3、pycocotools2.0.11、OpenCV4.13.0.92、Pillow11.1.0、PyYAML6.0.2；8项Ultralytics源码SHA及完整environment与原继承合同逐字段一致。 |
| 数值执行 | Adapter设eval、requires_grad=False，torch.inference_mode、FP32，无AMP或optimizer。matmul highest，CUDA matmul TF32=false、cuDNN TF32=true、benchmark=false、cuDNN deterministic=false、deterministic_algorithms=false，CUBLAS `:4096:8`；OMP=1、MKL未设置，离线环境与PYTHONPATH保持。 |
| 基础模型和有效推理几何 | 权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。缓存imgsz640、batch16、rect=True、end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300；逐图input_hw/ratio_pad、P/c0/F和输出行冻结。没有重新用batch1推理。 |
| mask及默认键 | mask_ratio4、overlap_mask=True、retina_masks=False等登记值保持。实际掩码来自原native decoder及缓存；没有新训练GT下采样步骤。原val配置中epochs100、optimizer=auto、amp=True等训练默认键未被执行，本轮不存在新的AdamW或梯度更新。 |
| 数据与历史参照 | 启动前全部2000缓存/原图、权重、预测、教师、完整train标注及2000图子集标注各自SHA通过；真实评价读取缓存Run的training_annotations.json。SPLIT、CONDITION_MAP及400图42167输出身份未变。旧G只复用同400图的固定配对/完整任务结果，不重新训练或选样。 |
| GT辅助方向 | 全部581条/208留出图；尺度只用本条已存教师，R/P/J同尺度、无面积门控，无新oracle或alpha扫描。拟合2510条/867图按原协议作描述。它们不参与模型拟合或checkpoint选择，也不是正常部署成绩。 |
| 正常无GT输出 | M/S均使用唯一拟合幅度a=2.1651711144258443和`d/max(1,RMS_ROI(d@P))`，空ROI归零；只对原生原图mask面积≥9216行应用修正，其余复制原生RLE。GT/教师尺度没有进入M/S计算。 |
| 解码与输出身份 | 完整同图FP32系数矩阵、原native decoder、既有裁剪/插值/二值/scale_masks byte路径；非mask字段不变。400图全部42167行保留原JSON顺序及score并列顺序，不先删门控外检测。原生账本与历史逐GT身份精确复现。 |
| 匹配与分母 | bbox、segm分别COCO匹配；T490/203图和全部2246固定配对/394图计算同检测IoU。1808原Mask75成功GT以完整混合预测重新匹配全查，不用教师子集或Box75检测替代。Box匹配与ignore身份护栏通过生产检查。 |
| 统计与Gate | 图内实例均值后图等权；seed1729、2000次配对图像bootstrap、linear百分位；R−P/R−J/R−N交集与M−N/M−G/总体/损伤分开判。未改.005、−.001、≤18等阈值，未用62条gate-on教师子组替换主分母。 |
| 权限及外推 | 当前是内部开发筛查，旧G训练过全部2000图，400图对G是训练内历史参照。GT辅助方向、正常无GT效用与新全量AP分开；COCO账本助手内部产生的400图AP日志不作为论文全量AP成绩。本次不替代原S2/S3或独立确认。 |
| 原生与官方状态 | 保持native_config_scoped。官方刊载AP parity未通过、原生GT账本有效是两个历史状态；本次没有改写其中任一项。 |

## 核验职责和工程记录

生产评价以原冻结源码执行，原Run的CONFIG_AUDIT及输入复核通过。独立脚本已重新forward、逐实例核系数公式、完整混合RLE及方向IoU，再用同一固定官方COCO引擎复建任务账本和统计，全部通过。共享decoder/COCO引擎不称第二套独立算法；本机只复算字节、身份和冻结统计，不运行另一份模型。

本机独立统计环境为Python3.12.14/numpy2.3.5，模型执行端仍是Python3.12.3/numpy2.4.3。本机复算前已保存[EVALUATION_LOCAL_AUDIT_FREEZE](EVALUATION_LOCAL_AUDIT_FREEZE.json)：调用原冻结纯统计函数，全部重采样索引精确一致、点值/区间在原绝对1e−12容差内一致；没有因版本不同放宽门槛。详见[EVALUATION_LOCAL_VERIFICATION](EVALUATION_LOCAL_VERIFICATION.json)。

首次递归SCP回传中断，笔记本核验持续正常。改用单个归档完成生产端1328项文件回传；归档SHA `43fccc9c59ac90926f1e59d3ebc9b2baa5459ba33091e2b4b205aca95ca54ed5`一致，原Run和代码未修改，见[EVALUATION_ARCHIVE_RECEIPT](EVALUATION_ARCHIVE_RECEIPT.json)。独立日志/回执和控制记录补齐后，最终清单为1352项、157356070字节，全部逐项大小及SHA通过，不仅检查归档。源端清单SHA `73d32c858a99d966d52183b8c305e56a0f5813d52368c7b01f8b278a78ddfacc`在本机一致，见[传输回执](EVALUATION_TRANSFER_RECEIPT.json)。

控制器自停用返回1、原始输出乱码保留；模型评价和独立复核两个子进程均exit0。通过已授权SSH显式停用本次任务后，状态Disabled、XML无触发器、无Python/锁/claim，见[设备收尾](EVALUATION_REMOTE_CLOSURE.json)。该收尾问题未改变科学执行或判定，没有触发重跑。

原训练报告及配置审计保留；本次生产forward2534、独立重放2534，合计5068；基础模型forward、新oracle及训练步数均0。Study累计训练仍为2404步（正式2400、smoke4），累计Adapter forward14748。主评价360.812秒、独立复核800.734秒，在原预算内完成。证据：[整体报告](REPORT.md) · [原科学协议](PROTOCOL.md) · [执行登记](EVALUATION_EXECUTION.md) · [启动复核](EVALUATION_PRELAUNCH_CHECK.json) · [生产配置审计](runs/RUN_CLD_EVALUATION_3bf05f04bb6e47c9abf9bb46cc4e1f47/CONFIG_AUDIT.json) · [笔记本独立核验](runs/RUN_CLD_EVALUATION_3bf05f04bb6e47c9abf9bb46cc4e1f47/INDEPENDENT.json)。
