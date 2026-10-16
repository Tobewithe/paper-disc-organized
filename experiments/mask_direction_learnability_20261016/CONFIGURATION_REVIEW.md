# CLD-v1 工程阶段配置复核

结论：**工程阶段配置通过；正式训练和留出评价未执行。** 核验依据为三项真实Run的执行合同、环境/配置回执、保存的梯度/optimizer/checkpoint、独立复核及本机回传核验，不能由本结论推导方法有效。

| 项目 | 冻结要求与实际证据 |
|---|---|
| 版本、源码、运行时 | Ultralytics8.4.100、torch2.5.1/CUDA12.4、Python3.12.3及完整包pins；8项框架源码、TF32/cuDNN/矩阵精度/环境变量逐项与继承环境相等。三份合同environment相同。 |
| 权重和缓存 | `yolo26m-seg.pt` SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。2000原图/2000缓存、预测文件及完整/子集两种标注的各自SHA均复核；没有重导出或基础模型forward。 |
| 原生有效推理配置 | 缓存仍为imgsz640、batch16、rect=True、end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300、mask_ratio4、overlap_mask=True、retina_masks=False及逐图input_hw/ratio_pad。176042行原生RLE精确复现。 |
| Adapter实际配置 | 原B结构167872参数、17参数白名单、相同原initial、FP32、无AMP/增强/EMA/scheduler/早停；AdamW lr=.001、betas(.9,.999)、eps1e−8、weight_decay=.0001，amsgrad/foreach/maximize/capturable/differentiable/fused均false。microbatch1、累积4，不代表基础模型改为batch1。 |
| 框架默认值的权限 | 原val入口快照里的epochs100、optimizer=auto、amp=True等未被本Adapter执行器采用；本轮以显式Adapter合同、实际optimizer序列化状态及AMP关闭为准。没有把训练默认值误当生效配置。 |
| 拆分及数据使用 | 原2000图固定1600/400；smoke是原序拟合清单前8图，两臂均只访问4条拟合教师；原400图教师不进梯度、不进a、不进行模型评价。原2000图初始化与历史soft-IoU last身份分离。 |
| 两臂唯一差异 | 本机对完整EVAL_CONTRACT删去arm后逐字段比较完全一致。REAL保持实例对应；PERM按冻结同图双射重排行d，使用接收实例P/c0/ROI/目标。每臂实际图序、初始化、容量、损失、共同a、AdamW、2步预算相同。 |
| 方向目标与幅度 | 2510拟合教师c*及c0/P缓存精确重放。独立NumPy FP64 RMS最大差1.7763568394002505e−15；共同a=2.1651711144258443。无面积筛选求a、无阈值搜索、无额外教师求解。 |
| loss与保护 | 方向MSE逐实例ROI均值后图均值；所有非空输出ROI的native probability BCE权重.1。原生面积9216门控仅用于正常输出，不缩小训练保护分母；GT/teacher不进入forward输入。 |
| 数值及优化 | 9合成测试通过；独立REAL/PERM梯度最大绝对差4.6566e−10/9.3132e−10，均小于预冻结工程容差。显式AdamW两步参数复算通过；17参数均得到非零梯度。未改变科学门槛或运行后调容差。 |
| 输出、身份与解码 | 两臂各681行zero-delta及结束混合输出经独立重放；Box/Class/score/排序/global index/原生非门控掩码保持。完整同图FP32系数矩阵、原native decoder，不切换ROI尺度或二值阈值。 |
| 实际预算 | 每臂8图2更新，合计4工程更新；正式训练0。Adapter共80次forward含32次独立复算；基础模型0、oracle求解0、留出评价0。三Run均在协议时间预算内完成。 |
| 科研状态 | K_dir/K_use未执行；新方法相对原生及旧G的IoU/AP/损伤均未测。本内部实验未替代原S2/S3。官方刊载AP parity状态保留未通过，native账本合法性保留已通过。 |

执行冻结SHA `56e9f26f4ebf2f56df92de068044ccaafdae60278b6b91b9a204751058fa12a8`；科学协议SHA `053aaefb1cbb0fd5f87dad2dabdf9e961660201ae81bae6c8e42e67d61d6d453`未变。36项执行文件、11项科学文件均未修改；新增本机回传核验脚本仅用于产物审计，不参与模型计算，也未回写冻结源码。

三Run原始ARTIFACT_MANIFEST及运行元数据原样保留。新增独立复核回执一并纳入177文件传输清单，总22178335字节，本机逐项SHA相等。笔记本承担真实张量/梯度独立复算，本机承担字节、合同、预算与产物登记核验；二者范围明确。

证据：[工程阶段报告](ENGINEERING_REPORT.md) · [执行合同](EXECUTION.md) · [本机核验回执](ENGINEERING_VERIFICATION.json) · [完整传输清单](ENGINEERING_TRANSFER_MANIFEST.json)。三个原始Run及其CONFIG_AUDIT/INDEPENDENT路径见工程报告。
