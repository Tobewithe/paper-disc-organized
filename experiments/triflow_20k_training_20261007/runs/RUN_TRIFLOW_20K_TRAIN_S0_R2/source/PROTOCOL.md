# TriFlow 2 万图训练结论实验

2026-10-07 用户明确训练规模：几百图用于调试，2 万图用于形成当前配置结论，十几万全量在结论确认后再验证。已停止原完整训练队列，尚未发生完整集正式训练；旧 pilot、工程测试、停止及失败记录按原目录保留。本实验独立建 Run，不把工程子集冒充正式结论。

使用已授权的离线笔记本 `ssh 28358lan`，解释器 `C:/Users/28358/anaconda3/envs/pytorch/python.exe`，Ultralytics 8.4.100。冻结官方 YOLO26m-seg，权重 SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；TriFlow 核心与冻结输入源码分别沿用 SHA256 `1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e`、`cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293`，不改变方法、势场损失、求解器或信赖域。

输入来自用户已有的 `C:/Dpan/document/model_datasets/datasets/coco/downloads/train2017.zip` 及原始训练 JSON，后者 SHA256 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`。在原始全部 train2017 图像 ID 上按 `SHA256('triflow_train20k_seed0:' + str(image_id))`、image ID 作为并列规则排序，取前 20,000，再按 image ID 排序固化清单；不按 pilot 收益、预测成功、拥挤度或验证 GT 挑选训练图片。只抽取这些图像并校验 ZIP CRC、实际 JPEG SHA 与原 JSON 身份及尺寸，不重新下载或复制全量训练集。

正式预算 seed0、全新模块初始化、20,000 张图每轮完整遍历、固定 8 轮，不按验证集选最好轮次。在线逐图生成冻结 FP32 特征与原始 GT 监督，不落盘巨大 P/F 缓存；first64 post-conf native rows、class-free best GT box IoU≥.5、raw 去重、GT 轮转、每图最多12训练实例、microchunk4每chunk更新、AdamW lr=3e-4、weight_decay=1e-4、clip10沿 pilot。全部训练图像覆盖不等于全部 GT 实例都得到直接 mask-task 监督；无可用候选的图像遍历但不更新，原因计数保留。

每 300 秒及轮次边界原子保存 head、AdamW、全部 RNG 和下一图像/chunk 游标，恢复另建 Run 并核对来源、预算、配置、数据身份。只恢复已持久化前缀，崩溃后的未提交更新可能回滚重放。工程32图只验证输入、梯度及恢复正确性，保持 smoke-only，不用于方法能力结论。

完成后原始 COCO val2017 全 5,000 图评价 Mask/Box AP、AP75、APsmall及原配对诊断；one-to-one、640、FP32、conf=.001、max_det300，COCOeval maxDets=[1,10,100]。GT不进入推理；基线复用必须重新全图核验原生候选身份与RLE字节。配对为固定native detection、同类GT最大BoxIoU≥.5、允许重复GT匹配，IoU=.75定义修复/损伤，不能冒称raw几何评价或COCO匹配。未计算raw五类几何、AUC/FPR、裁切支持、AP置信区间及seed不确定性保持未知。

当前配置参考收益门槛沿用 +0.3 AP点、baseline成功损伤率≤1%、AP75/APsmall不同时下降、成功队列平均IoU增量≥−.005。训练覆盖、预算充分性、执行有效性与实际收益分开报告；20,000是本项目的结论阶段约定，不保证所有问题的统计充分性，不把单seed或门槛未达推广成方法能力上限。结论确认后再决定十几万全量验证。
