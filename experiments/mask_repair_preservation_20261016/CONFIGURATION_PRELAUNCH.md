# 冻结前配置核对

2026-10-16，笔记本YANGLEI。以[预检](PREPARATION.json)、[冻结清单](FREEZE_MANIFEST.json)、[实际参数](CONFIG.json)及[协议](PROTOCOL.md)为准。本机已逐项核对21个冻结文件及PREPARATION字节SHA，与笔记本一致。

## 当前实现和证据层级

11项合成工程测试通过，见[测试回执](engineering/TESTS_V1.json)。17项历史源文件、权重、原始GT、源预测和面板哈希通过；247张输入图片及缓存逐项核验。正式面板256条/243图，smoke为另外4图，具体固定记录见PREPARATION。测试与预检不是科学通过证据。

原生账本不存类别字段，首次预检错误假定该字段存在，已在冻结前改为按annotation_id从同SHA原COCO标注取得类别，并核图ID相同。选样规则未变，未发生真实样本求解，见[保留的错误记录](engineering/PREFLIGHT_V1_FAILURE.json)。正式冻结后不再修改目标、门槛或求解配置。

## 实际配置差异白名单

| 项目 | 当前决定与核验 |
|---|---|
| 基础模型/权重 | 同YOLO26m-seg官方SHA；不加载网络做新forward，不更新权重。固定P/c0来自原S0缓存 |
| 框/类别/分数/顺序 | 继续原最终输出，保持完整N×32矩阵形状；只改面板指定行的临时系数 |
| 输入几何 | 原imgsz640、batch16、rect=True实际缓存尺寸/ratio_pad；不重新分batch |
| 原生推理/评价参数 | end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300、mask_ratio4、overlap=True、retina=False等从原profile及源Run实际环境完整核对；权重train_args的mask_ratio1不回灌 |
| 实际解码 | 固定版本原算子、原始全矩阵FP32 GEMM、原生分块边界、输入尺度先阈值裁框，二值缩原图再byte；没有最终0.5阈值替换 |
| CPU优化端数值 | 新增且三臂共有：真实FP32 z0锚点＋FP64双线性P设计矩阵；源ops内部float转换不冒充FP64。逐项实测代理误差受冻结容差约束 |
| 科学操作白名单 | C/E/EK目标差异；E-equal由两臂实际logit RMS一次计算，无阈值/权重/步数扫描 |
| 训练 | 网络训练步数0、已有checkpoint均不修改。最多768次32维有限优化，与Adapter训练区分 |
| GT/ROI | 原COCO RLE经缓存真实resize/pad投影到首次阈值网格；原预测框ROI；E/K在c0处冻结。原图IoU始终完整GT作为分母 |
| 参照 | 同网格新C为主增量控制；原生N、旧soft-IoU+面积门控R_G和原S1 R_S1同时保留。同256条描述性收益不称新AP或部署优势 |
| 统计与Gate | 243图宏平均，固定2000次图像配对bootstrap/seed1729；+.005增量、25%保护、等幅IoU≥−.001、95%C见证保留及数值有效性，全部事前冻结 |

笔记本原生实际环境与源机会审计EVAL_CONTRACT逐字段相等，包括Torch/Ultralytics/依赖、源码SHA、FP32/TF32/cuDNN/确定性开关和相关环境变量。运行后须再核并完成独立RLE与统计复算。本配置只支撑native_config_scoped研究；官方发布AP复现未通过的状态保持。
