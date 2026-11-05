# V20 已训练对象词表与模型使用

正式10000步末EMA已导出并本地重载，科学结果见[报告](REPORT.md)。该配置真实Box/Mask AP均下降，保留为研究产物；本轮实用参照仍是原生模型。

## 文件

- [weights.pt](trained_model/weights.pt)：四类词向量、检索器、组合/检测/分割/规范解码器、固定投影及完整配置；SHA `f8b6531b7e246969925a6597ce84572c18b3871aeca053ce1ddf1ed52bef72f4`。
- [VOCABULARY.json](trained_model/VOCABULARY.json)：512槽位的类型、身份、训练支持、来源与计数；493支持、19未启用。
- [CANONICAL_REFERENCES.pt](trained_model/CANONICAL_REFERENCES.pt)：训练参考、有效掩码及初始化特征。
- [deploy.py](trained_model/deploy.py)、[model.py](trained_model/model.py)、[MANIFEST.json](trained_model/MANIFEST.json)：可加载代码和实际依赖版本。
- 独立原生视觉权重已在本地：[native_candidate.pt](../spm_object_memory_training_20261024/weights/native_candidate.pt)，SHA `0e40602f1dbf0697fd49fa8e9f887c068970e3c5c39bd618f66e483c8712c8d6`；不把它误当新原型权重。

## 正常图像推理

服务器已冻结并验证的运行环境为本Study `.venv/bin/python`，Torch2.5.1+cu124、Ultralytics8.4.100。正常评价为单图、CUDA BF16任务头、FP32原生前端、TF32关闭。

把`trained_model`加入模块路径后：

```python
import sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("C:/Dpan/codexproject/paper-disc-organized/experiments/spm_semantic_dense_task_20261025")
sys.path.insert(0, str(root / "trained_model"))
from deploy import PrototypeSegmenter

native = root.parent / "spm_object_memory_training_20261024/weights/native_candidate.pt"
segmenter = PrototypeSegmenter(root / "trained_model/weights.pt", native, device="cuda")
rgb = np.array(Image.open("image_640.png").convert("RGB"))
prediction = segmenter(rgb, confidence=0.25)
```

输入必须为与评价一致的640×640 RGB uint8坐标系；不会偷偷缩放图片或读取GT。输出包括当前帧框、分数、二值Mask、查询ID及四类Top-2词条读入。完整AP用score>.001，.25仅是固定展示/工作阈值。词条读入权重不是置信度。

服务器路径相应替换为当前执行根、`runs/RUN_SPM_V20_EXPORT_MODEL_c4288d2c13fc/MODEL_EXPORT/weights.pt`及配置登记的原生权重。CPU可做权重检查和规范解码；完整正常RGB精确核验已在CUDA环境完成。

## 单独取出向量或解码形状

```python
import torch
from model import SemanticTaskModel

saved = torch.load(root / "trained_model/weights.pt", map_location="cpu", weights_only=False)
model = SemanticTaskModel(saved["model_config"])
model.load_state_dict(saved["state_dict"], strict=True)
model.eval()
part_vectors = model.banks["part"].words.detach()  # 256 x 128
whole_vectors = model.banks["whole"].words.detach()  # 128 x 128
with torch.inference_mode():
    canonical = model.canonical_logits().sigmoid()  # 512 x 2 x 32 x 32
```

每个128维向量需配所属类型和训练得到的共享解码器解释。规范图是固定训练参考复现；不是向量自动包含任意真实猪的全部纹理与三维细节。原始词表保留`prone`名称，但该源姿态实际是屈腿代理，不能据此声称真实腹卧能力。未启用词条即使能由共享解码器输出图案，也不算已训练合格对象。
