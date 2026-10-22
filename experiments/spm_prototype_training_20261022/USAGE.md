# 训练好的原型如何使用

本轮交付两件正式产物：`projector.pt`（实际学习后的128维投影）和`memory.pt`（训练token聚类后的猪体/背景双源记忆）。正式位置由`TRAIN_POINTER.json`指定，smoke目录中的2个均值中心不是正式原型。

编码器固定为权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5` 的官方YOLO26m-Seg。特征取自Segment26之前的P3；RGB固定640方形letterbox与0–1预处理，冻结模型eval/FP32。不将该库用于另一个微调过的Neck或不同投影空间。

读取用法（将本Study的`scripts`放入Python导入路径）：

```python
import torch
from memory_query import FrozenPrototypeMemory

payload = torch.load("memory.pt", map_location="cpu", weights_only=True)
memory = FrozenPrototypeMemory(payload).eval()
# p3是上面固定编码器的原始P3张量[B,C,H,W]，未自行做通道标准化。
tokens = p3.permute(0, 2, 3, 1).reshape(-1, p3.shape[1]).cpu()
with torch.no_grad():
    responses = memory(tokens)
# foreground/background各为[N,128]；同时返回Top-k索引与最大余弦。
```

`memory_query`内部执行已冻结的train均值/std、W/b、L2归一化和Top-k。不要重复归一化输入，不要用dev/test重新拟合统计，不在线更新中心。部署时不需要GT或Blender图；来源清单与原型卡用于离线追踪。

此处输出是可供后续残差融合的检索响应；本轮没有训练Wo/gate或增强版YOLO，不把“文件可用”写成“Mask AP已提高”。
