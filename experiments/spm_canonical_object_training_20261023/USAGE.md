# 已训练对象与推理用法

当前交付为**规范身体对应＋实例框条件可见归属对象**，正式6000步末EMA，707237个可训练参数。身体对应要求通过；完整可见归属未达标。没有接入真实YOLO分割微调，不能当作已获+5AP的部署模型。

## 权重

- 首选：[canonical_step_06000.pt](runs/RUN_SPM_V15_TRAIN_f327d1e43cbf/canonical_step_06000.pt)
- 大小：3481958字节。
- SHA256：`57070cf0e7e6ca12b4488487095995b2ff8cc4261df3450cec474aeee6ff5d8f`。
- [1000步](runs/RUN_SPM_V15_TRAIN_f327d1e43cbf/canonical_step_01000.pt)、[3000步](runs/RUN_SPM_V15_TRAIN_f327d1e43cbf/canonical_step_03000.pt)为保留过程检查点，不作为新选择。

文件包含对象头、35维固定身体身份键、283×283图距离、原生共享视觉标准化/PCA、规范网格及冻结来源。它是本项目PyTorch对象包，使用[object_model.py](object_model.py)的`load_object`，不直接传给Ultralytics的YOLO权重加载器。冻结原生视觉编码器另用已有N权重，SHA为`cdea4fb8e837dbbda092bbd5743abe28a8b01731817d9dc326133debd61fe40a`。

## 已登记服务器的最小调用

使用该Study冻结的PyTorch2.5.1+cu124、Ultralytics8.4.100环境。以下组合就是实际RGB复核使用的函数；输入只含RGB和实例框。正常任务的框应来自原生预测；本轮合成验收用GT可见框，权限见[报告](REPORT.md)。RGB须按原生配置处理为640×640；本轮128×128合成画布也被冻结特征桥支持。框为同画布归一化`[x1,y1,x2,y2]`。

```python
from pathlib import Path
import sys
import numpy as np
import torch

root = Path('/srv/gpu-lab/users/a25023104/projects/spm-canonical-object-v15-20261023')
sys.path.insert(0, str(root))
sys.path.append('/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023/scripts')
from runtime import read
from task_model import load
from object_model import load_object
from legacy_surface import bridge, queries
from prepare import roi_context

cfg = read(root / 'CONFIG.json')
checkpoint = root / 'runs/RUN_SPM_V15_TRAIN_f327d1e43cbf/canonical_step_06000.pt'
object_model, package = load_object(checkpoint)
object_model.requires_grad_(False)
native = load(Path(cfg['native_checkpoint']), steps=3000).eval().requires_grad_(False)
statistics = {k: v.cuda() for k, v in package['metadata']['visual_statistics'].items()}
helper = bridge(cfg['visual_bridge'])

@torch.inference_mode()
def predict_body(rgb_uint8, predicted_box_xyxy):
    box = np.asarray(predicted_box_xyxy, dtype=np.float32)
    q = queries(native, statistics, helper, rgb_uint8)
    roi = roi_context(q, box)
    axis = (np.arange(128, dtype=np.float32) + .5) / 128
    xx, yy = np.meshgrid(axis, axis, indexing='xy')
    xy = torch.from_numpy(np.stack([xx, yy], -1).reshape(1, -1, 2)).cuda()
    out = object_model(q.reshape(1, -1, 128), xy,
                       torch.from_numpy(box)[None].cuda(),
                       torch.from_numpy(roi)[None].cuda())
    vertices = out['logits'][0].argmax(-1).reshape(128, 128)
    visible_ownership = out['ownership'][0].sigmoid().reshape(128, 128)
    return vertices.cpu().numpy(), visible_ownership.cpu().numpy()
```

`vertices`是每个查询像素的规范身体顶点身份；仅在目标可见身体区域内验收过定位。`visible_ownership`是实例条件归属分数，本轮完整Mask IoU仅57.20%，不能替代现有强原生Mask。没有显式的新3D姿态/相机解算输出。

实际12次RGB重载均与缓存/训练末点顶点预测一致，见[复核记录](runs/RUN_SPM_V15_EVALUATE_4a9ce44bd9ee/RGB_RELOADS.json)。本地完整权重SHA与源包关联见[LOCAL_ACCEPTANCE](LOCAL_ACCEPTANCE.json)。
