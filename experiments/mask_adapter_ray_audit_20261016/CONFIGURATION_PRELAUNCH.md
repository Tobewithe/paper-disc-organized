# 运行前配置核对

2026-10-16，执行端YANGLEI，笔记本原生环境。冻结清单与本机23项源码/协议逐字节一致。

| 项目 | 本轮固定值与证据 |
|---|---|
| 正式对象 | 原S1全256条/243图；原TARGET_PANEL SHA不变；C159仅为已有分层 |
| checkpoint | 原soft-IoU B、3 epochs的last.pt；SHA 15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce |
| 原生输入 | 复用8.4.100、640/batch16/rect=True生成的P/c0/F及geometry；没有重分batch或基础模型forward |
| 原生输出约束 | 同一检测/GT、框/类/score/排序不变；只调整面板目标系数；其余掩码逐像素核对 |
| 本次Adapter执行 | eval、requires_grad=False、phase=eval:0；按图缓存前向，参数哈希运行前后相同才可结案 |
| 唯一科学变量 | alpha非负且≤16；至多6个开区间代表加0/1/16，共至多9个；真实原生解码选有限见证 |
| GT权限 | 只用于离线事件评分和选幅度，不提供可部署AP或全原成功GT损伤保证 |
| 参照 | N原生、U原Adapter、G原面积门控、C已存BCE解；C不重求解 |
| 解码 | 完整N×32 FP32 GEMM、首次>0、预测框裁剪、二值scale_masks后byte；不改为0.5阈值 |
| 数值 | 代理误差5e-4+5e-5*abs(actual)，原图IoU误差≤.001；原生二值转换表要求精确 |
| 训练与基础网络 | 新训练0、基础网络forward0，原模型与checkpoint字节哈希不变 |
| 核验 | 7项合成测试通过；28项源输入及247图/缓存核验；正式结果尚待Run完成 |

协议SHA：`068ca36039aec19d33610909bb6328d5c4f8dc433b0315fadd28e48bd2a31c48`。冻结时间：`2026-10-16T10:16:00.219236+00:00`。启动smoke：`RUN_ADAPTER_RAY_SMOKE_4fe9912dd6dd4422bc9f0c079f6267e3`，Python PID3004；执行完成与科学判定须以真实Run产物为准。
