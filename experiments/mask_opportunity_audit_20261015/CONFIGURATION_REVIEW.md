# 同检测机会核对：运行后配置复核

实际Run为 `RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd`，笔记本YANGLEI。**模型、缓存、配对、decoder、面积门控和实际运行配置均与冻结来源一致。** 本次只新增同检测机会分解，未产生新训练checkpoint或新的AP验收。

| 项目 | 冻结值与实证核验 |
|---|---|
| 官方权重 | `yolo26m-seg.pt`，SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；不作基础网络forward |
| 当前adapter | 原soft-IoU B，2000图/3轮/1500步的last.pt；SHA `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`；eval:0、eval/inference_mode，state前后不变 |
| 包和源码 | Ultralytics8.4.100私有overlay；torch2.5.1、torchvision0.20.1、CUDA12.4、numpy2.4.3、pycocotools2.0.11、faster-coco-eval1.8.0；完整包/源码实际值与旧EVAL_CONTRACT逐项相同 |
| 原生配置 | imgsz640、batch16、rect=True、end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300；classes=None、single_cls/augment/compile=False |
| mask配置 | mask_ratio4、overlap_mask=True、retina_masks=False；实际decoder源码保持原生保存JSON路径，没有替换阈值/缩放顺序 |
| 几何 | 复用243图原S0缓存的P/c0/F、input_hw、original_hw、ratio_pad和完整检测矩阵；没有以本次逐图执行改变原batch/rect几何 |
| 数值策略 | FP32/highest、matmulTF32=False、cudnnTF32=True、benchmark=False、cudnn_deterministic=False、deterministic_algorithms=False、warn_only=False、CUBLAS=:4096:8、线程1 |
| deterministic语义 | 原生请求参数deterministic=True与上述实际torch开关分别记录；沿用既有实效配置，未将名义字段当作实际状态 |
| 数据身份 | 原S1面板SHA `170676d3017e4be7cd213ff1094ca28065727807c915b7602ecdaec48238cf31`；256条/243图、原普通GT及Box75官方匹配输出不变；图像/标注/缓存哈希通过 |
| oracle身份 | 原S1存档global32系数/RLE；SHA `6e7ff1c080d934d742a9c4e10dbf3e0bcb3fdfe57fa9dc7d61edba53cc791767`；256条重放精确相同，无重新求解 |
| 门控 | 原生原图二值掩码面积≥9216；未改阈值、不扫阈值、不使用GT或变化后面积作为门控输入 |
| 输出身份 | 全49,976行zero-delta精确相同、完整门控预测与S3精确相同；17,821行无门控预测与原S2完全相同，另32,155行Δc与原S3张量位相同 |
| 评价 | 原图同检测完整RLE IoU，阈值.75；256条不删，122条原oracle阳性分解100/10/12；S3 COCO任务账本另列，没有用独立匹配替代同检测IoU |
| 预算 | smoke8图/10条、正式243图/256条；无训练、无oracle求解、无基础模型forward；正式单次完成，无模型运行恢复 |

冻结协议SHA `5a5329375c2e42402319f28a40732828a0f55322cb87e08b9b3104f012c0192d`，冻结时间 `2026-10-15T16:09:12.277446+00:00`，早于所有真实Run。19项冻结代码/合同与两机快照保持字节一致，26项源文件及243图/缓存运行前后核验通过。

## 独立核验和首次本机求和差异

笔记本用独立compressed-RLE区间求交实现验证全部256条四种mask的IoU/面积、原始文件身份、分类与分母，最大IoU误差为0。本机同样独立复算，最终全部通过。smoke与正式10条重叠结果完全相同。

本机首次调用Python3.10.19，冻结核验脚本在描述性均值的 `value == recorded` 断言失败。此前所有逐条RLE、身份、分类和决策断言已通过。复查8个分组/臂均值，最大绝对差为 `6.938893903907228e-18`；`math.fsum` 与8个笔记本记录均值全部精确相等。

使用已安装的本机Python3.12.14（笔记本为3.12.3）执行同一冻结脚本后，包括严格均值相等在内全部通过。未更改模型环境、Run、脚本、数值容差或门槛；本机仅作只读整数RLE及统计复算。首次失败见 `INDEPENDENT_LOCAL_INITIAL_FAILURE.json`，通过及解释见 `INDEPENDENT_LOCAL.json`、`INDEPENDENT_LOCAL_RUNTIME.json`。没有把这次描述性求和差异隐去或写成模型配置变化。

80文件回传SHA全部通过，原始包哈希为 `729e71e2e1f12e5881b13c196240e221ff36fe007e4ff953b6ac9675ca91777d`。原S0发表AP parity与原生GT账本有效性仍分别保留旧状态；本核对没有重新宣称官方发布AP复现，也没有重新判定S3晋级。
