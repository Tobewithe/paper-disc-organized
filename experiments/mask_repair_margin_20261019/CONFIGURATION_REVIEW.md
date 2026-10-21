# RSM-v1结案配置复核

2026-10-20。结论：原输入、环境、科学协议及冻结配置未漂移；保留113条的掩码与幅度可复核；**第114条扰动幅度精度失败，完整实验无效，不能以配置一致代替科学或数值门槛通过。**

| 项目 | 实际核验与边界 |
|---|---|
| 执行设备 | 用户当轮授权的笔记本`28358lan / YANGLEI`，Python `C:/Users/28358/anaconda3/envs/pytorch/python.exe`，原RTX4060 Laptop。未转到服务器、未停其它会话视频服务。 |
| 官方模型身份 | `yolo26m-seg.pt`，SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。本轮基础网络和Adapter前向均0，只读已保存原生缓存。 |
| 固定版本 | Ultralytics8.4.100；torch2.5.1、torchvision0.20.1、CUDA构建12.4、Python3.12.3、numpy2.4.3；pycocotools2.0.11。Clarabel0.11.1、scipy1.15.2，沿用已安装环境，无安装或升级。源码8项官方pins与项目依赖随冻结清单核验。 |
| 缓存预处理 | 原native full的imgsz640、batch16、rect=True、end2end=True、conf=.001、iou=.7、max_det300、agnostic_nms=False；保持原batch决定的图像尺寸。mask_ratio4、overlap_mask=True、retina_masks=False保留原配置。本轮没有重新组batch、导出缓存或模型推理。 |
| 实际精度 | 原生解码FP32；CUDA matmul TF32=False，cuDNN TF32=True，matmul_precision=highest；其余运行值与原合同完全一致。不是把全部“TF32”笼统说成关闭。有限几何求解FP64，输出回FP32检验。原args里的amp/optimizer等训练字段不代表本轮执行过训练或半精度解码。 |
| 数据与身份 | 输入清单505项在执行与结案核验时检查；原256条/243图，最新正式保存113条/107图。raw_id、最终检测行、GT、c0、缓存SHA与原C来源一致；不重做bbox/segm匹配。COCO标注SHA `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。 |
| 正常输出复现 | 原完整N×32 GEMM、原输入尺度>0、框裁剪、原图缩放与byte路径不变；保存113条N/C/W及压力由完整原生路径复核5763次。113条C/W首阈值掩码、原图RLE完全相同。 |
| 品质和尺度控制 | W不得改变C的二值掩码；只恢复C logit RMS，不能靠放大logits增加余量。保存行最大中心RMS相对误差2.886755e-6≤5e-5。一个尺度失败候选已退回C。 |
| 压力控制 | 复用CET拟合2510条/867张train2017图库；与本次val图不交叉。每条S/R/M各8次，共24个方向；C/W配对方向相同，目标始终0.10*C_RMS。保存行最大实际压力RMS误差4.915486e-5≤5e-5。第114条超过原要求即退出，具体超差值未保存，不能补写。 |
| 有限求解控制 | 原Clarabel100迭代、24轮、每轮加512约束、1e-8求解容差、2e-7完整可行性容差、1e-12条件比例未改。14条矩阵病态回退，没有临时截断秩或正则；另1条线性代理越限，科学有效性不通过。 |
| 评价分母与统计 | 没有以113条替代256条，没有丢弃回退/越限行，没有产生正式STATISTICS、完整CI或把部分结果写成正/负机制结论。AP与全原成功GT损伤均未测。 |
| 训练与预算 | 网络训练0，基础模型/Adapter前向0；仅有限系数求解和读数。最新正式Run165.657秒后按精度条件退出。旧失败、smoke和工程复现开销独立登记，没有覆盖或拼接。 |
| G_budget | 最终checkpoint SHA `cff530a9c38f4b70c4e54c93326307aa4264ab1b7928680c8b80daf2a8be67bc` 尚不可用；本轮同面板指标null。旧G仅历史辅助，不作为替代。 |

科学协议SHA `fc18e379f8679920ec8f472af2ad8ac955859ee449f7d99d32519d44d09dad12`、CONFIG SHA `d9773c5d0db940dbd9857bab6cc82ceb5ad57215f5d19c0bbc20c06639d14987`保持不变。V1/V2/V3执行清单及逐Run源码快照保留。V2只修smoke空见证bootstrap的报告异常；V3只对固定幅度目标加入最多8次舍入校正，保存因子并独立构造重放，未改变精度标准或科学强度。第二次正式精度失败后停止恢复。见[工程记录](ENGINEERING_RECOVERY.md)。

12项工程测试通过；smoke的204次独立完整重放通过。最新正式错误Run没有伪造`CONFIG_AUDIT=passed`；单独结案核验Run明确记录`partial_artifacts_verified=true`、`full_comparison_passed=false`。505项输入和21份冻结源码核验、5876项本机RLE复算、292文件回传均通过，但不改变科学`INVALID_ENGINEERING`状态。

证据：[PREPARATION](PREPARATION.json) · [冻结V3](FREEZE_MANIFEST_V3.json) · [工程测试](preflight/MATH_TESTS_V3.json) · [结案核验合同](runs/RUN_RSM_VERIFY_8f24ea1564ee4176b0317079f8737382/EVAL_CONTRACT.json) · [笔记本配置审计](runs/RUN_RSM_VERIFY_8f24ea1564ee4176b0317079f8737382/CONFIG_AUDIT.json) · [本机核验](preflight/INDEPENDENT_LOCAL_RETAINED.json) · [回传校验](preflight/TRANSFER_FINAL_VERIFIED.json)。
