# LRP-v1 配置复核

正式LOCAL与UNIFORM均完成1200步/4800次图像访问；各2步烟雾训练单列，正式从同一初始状态重新开始。没有追加epoch、seed、阈值搜索或挑选中间checkpoint。

## 哪些配置保持，哪些是预先声明的新方法

| 项目 | 实际生效状态及比较权限 |
|---|---|
| 基础模型/原型/系数 | `yolo26m-seg.pt` SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`，P/c0与缓存不变；基础模型forward和原型训练均为0。 |
| 原生推理 | Ultralytics8.4.100、原batch16/rect=True/imgsz640、O2O、conf=.001、iou=.7、max_det300、mask_ratio4、overlap_mask=True、retina_masks=False；逐图input_hw/ratio_pad及原生decoder原样复用，不把训练microbatch1当原生batch1。实际输入尺寸不要求每张都是640×640。 |
| 运行环境 | Python3.12.3、torch2.5.1/CUDA12.4、numpy2.4.3、pycocotools2.0.11及源码pins逐项与原缓存相同。FP32/AMP关闭、matmul TF32=False、cudnn TF32=True，线程1；不依赖未记录框架默认值。 |
| 新可训练对象 | 两臂同一3297参数、6个参数张量的逐像素MLP；只有net内三层Linear权重/bias可更新。与旧167872参数CoefficientAdapter是不同参数化，不能宣称全部训练配置与旧G相同。 |
| 新监督与解码 | 原图固定B点上的GT BCE+.1原生锚点概率保持；signed native anchor保证零残差原生一致。新方法只在B内做最终二值合成；域外不变。mask_ratio等原框架键不支配这个新训练loss，实际监督分辨率/采样在新协议明示。 |
| 严格训练对照 | LOCAL/UNIFORM唯一功能差异为逐像素残差或固定采样点残差均值。两臂同数据、初始张量、损失、优化器、幅度上界、1200步及三轮顺序。没有等实际RMS控制，不作该层独立因果归因。 |
| 原有G比较 | G为同原生检测上的历史输出参照，训练过2000图；新两臂仅拟合1600图。必须分别报告对N的绝对收益、对G的实际差值，不能将其差值单独归因于某个模块或loss。 |

AdamW实际参数和最终checkpoint中的param_groups一致：lr=.001、betas=(.9,.999)、eps=1e−8、weight_decay=.0001、amsgrad=False、foreach=False、maximize=False、capturable=False、differentiable=False、fused=False。没有AMP、scheduler、EMA、clip或增强。

## 输入、运行与独立验证

原2000张图像和2000份缓存逐文件SHA核验通过；原标注、权重、预测JSON和53项源码均通过启动及结束检查。1600拟合图生成16,183个可编辑输出的固定8,284,486采样点；其中2800个输出有原规则GT join。准备阶段176,042行原生RLE精确重放；400图评价再次重放完整42,167行。GT只进入监督/评价，没有进入模型特征、B、面积门控或采样选择。

两臂初始张量SHA均为 `ff6baf57c5198c4365f794ca5bb15bc2c1021f9bef303cf3b1eb7480bd20eaed`。LOCAL final checkpoint SHA `193b88783cc520448103d626c2db4f99a4a5dcf1d171a8660b1cedc8de70480a`；UNIFORM SHA `ea29aa294206ffa7c1bfddc72cefbb8e1f2471f6883295c32178f4ec3b471239`。独立审计核6个优化器状态步数全部1200、三轮顺序/4800次访问、所有参数FP32且确实改变、1600份训练张量哈希与留出不交叉。

统计和评价冻结：T490/203图、全2246配对/394图、全1808原成功GT；图像宏平均、2000次配对bootstrap、seed1729。原始bbox与segm分别匹配，全部检测保留，不使用仅门控内分母。独立核验用完整默认COCO阈值/面积网格重算三臂bbox与segm，选择.75/all核对实际账本；域外改动像素0，原生非Mask字段及bbox匹配不变。

工程合成8项、两个真实smoke均通过。正式运行无工程失败；前置LEA原smoke失败保留于其所属Study，不混为本次训练失败。

全部Run和1600份派生训练张量共2145项原产物（2,553,061,194字节）回传，ZIP SHA `be6420de3d4267507340e7f75876b31a160c0c1f44981b0461e19296d60f4209`。最终状态见[回传核验](TRANSFER_VERIFICATION.json)、[本机独立核验](runs/RUN_LRP_EVALUATION_f89d8084795244a985b37031a6b395a9/INDEPENDENT_LOCAL.json)和[独立配置审计](INDEPENDENT_CONFIGURATION.json)。执行端Study原始状态另存快照，本地结案状态不反写旧Run字节。

此次通过WMI启动独立隐藏进程，没有新建Windows计划任务、Codex自动化或goal；前置LEA两项任务已停用。完成后停止在本Study评价/结案范围，不自动S2/S3/S4、不提交或推送。
