# 本次零训练评价配置复核

结论：通过。正式Run `RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770`，笔记本和本机独立核验均通过；结果有效范围见[REPORT](REPORT.md)。

| 项目 | 冻结且实际核对的配置/证据 |
|---|---|
| 原生基线合同 | `baseline_profile.json` SHA `ae8133e371d121c642e0a32070b3743f6f7657ded587d6ab0491456329198e45`；原生权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5` |
| checkpoint | 现有soft-IoU最后一轮，SHA `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`；原167872参数，模型内存state前后SHA相同 |
| 原训练身份 | 原2000图、同TRAINING_JOIN、3轮1500步、seed0、有效batch4、AdamW lr=.001、weight_decay=.0001；这些仅是所读取checkpoint的历史，本次没有optimizer或训练 |
| 本次前向 | 原B架构、eval:0、eval模式、requires_grad=False、inference_mode、FP32、AMP关闭；原每图完整系数矩阵，基础网络不实例化 |
| 输入几何与decoder | 直接使用原batch16/rect=True缓存P/c0/F、input_hw/resize/pad；imgsz640、end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det=300；没有重新拼批或图像预处理 |
| mask配置 | 原生合同mask_ratio=4、overlap_mask=True、retina_masks=False；原始COCO GT与固定原生decoder。官方预训练元数据的mask_ratio=1未替代本次验证合同 |
| 固定门控 | `area_gate.mix_row`原代码；原生原图二值mask面积>=96²=9216应用现有Δc；其余保持原生；没有GT门控、倍率、阈值扫描或删除输出 |
| 数据与评价分母 | train2000与dev512不重叠，训练输出218209；T2669/655，Box75固定配对12097/2925，原成功9748/2355；完整预测与GT未缩小 |
| loss核验 | 原soft-IoU epsilon=1e-6、原ROI及分母、全部非空ROI保真BCE×.1；只计算固定checkpoint的loss，不反传；有效GT监督14268、保真实例217753 |
| 数值环境 | Ultralytics8.4.100、Python3.12.3、torch2.5.1/CUDA12.4、numpy2.4.3、pycocotools2.0.11及导入源码哈希与原训练/评价严格相同 |
| 实际数值开关 | matmul_precision=highest、matmul TF32=False、cuDNN TF32=True、benchmark=False、cuDNN deterministic=False、deterministic_algorithms=False、warn_only=False、CUBLAS=:4096:8；不是仅根据args的deterministic字段推断 |
| COCO评价 | 原固定`build_native_ledger.evaluate_ledger`；完整bbox+segmentation JSON；bbox/segm独立匹配、useCats=1、maxDets=[1,10,100]、area=all；所有Box75输出身份保持 |
| 源码/输入 | 26项源码/协议、52项输入开跑前冻结并在结案前逐SHA复核；训练缓存2000文件核对。旧训练/评价计算源码逐字节相同；新代码只做本Study评价/统计 |
| 重放与独立验证 | 218209行zero-delta原生RLE精确一致；独立检查218209行混合规则与非Mask字段，12097配对×3份RLE IoU最大误差0；开发集原均值/区间复算误差<1e-12 |
| 启动前工程修订 | 独立核验器接受GT RLE bytes和JSON string；7项工程用例通过。旧冻结在prelaunch_archive保留，修订在smoke及任何训练结果计算前完成，科学协议未改 |
| 回传与本机复算 | 140文件SHA一致，4个大预测保留笔记本；本机Python3.12.14运行原冻结verify.py通过，统计容差1e-12预先固定 |

记录：[正式CONFIG_AUDIT](runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770/CONFIG_AUDIT.json)、[FROZEN_MODEL_STATE](runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770/FROZEN_MODEL_STATE.json)、[FREEZE_MANIFEST](FREEZE_MANIFEST.json)、[INPUT_EXPECTATIONS](INPUT_EXPECTATIONS.json)、[两机核验](INDEPENDENT_LOCAL.json)、[回传收据](TRANSFER_VERIFICATION.json)。

区别保留：训练2000图AP是附带描述量，不与val5000 AP合成一个表或Gate；dev512完整成功GT损伤census不替代旧1024随机样本的有限总体上界。配置通过不等于S3重新通过，也不等于官方发表AP复现。没有新训练、基础模型forward、oracle求解、阈值扫描或checkpoint选择。
