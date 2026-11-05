# V16 权重、正常RGB推理与复核

本轮已完成训练；**V16在同配置真实开发集上的Mask AP为52.6420，低于V13原生57.3585，当前不选作部署首选。** 权重完整保存用于复现阴性和后续研究。实际首选仍是[V13原生候选](../spm_geometry_score_preservation_20261023/USAGE.md)。本文件不是再次提交训练的命令。

## 三份完整权重

| 组成 | 文件 | 字节数 / SHA256 |
|---|---|---|
| 原生YOLO候选，冻结 | [native_candidate.pt](../spm_geometry_score_preservation_20261023/runs/RUN_SPM_V13_EVALUATE_3e710949e5b0/native_candidate.pt) | 108413518 / `0e40602f1dbf0697fd49fa8e9f887c068970e3c5c39bd618f66e483c8712c8d6` |
| 已训练身体对象，冻结 | [canonical_step_06000.pt](../spm_canonical_object_training_20261023/runs/RUN_SPM_V15_TRAIN_f327d1e43cbf/canonical_step_06000.pt) | 3481958 / `57070cf0e7e6ca12b4488487095995b2ff8cc4261df3450cec474aeee6ff5d8f` |
| 本轮10817参数读出，末4000步EMA | [readout_step_04000.pt](runs/RUN_SPM_V16_TRAIN_7bff2ba6a386/readout_step_04000.pt) | 48566 / `ccde387da1828789da6f32e43f6c54d9b6c2ab3f2b2bd59c5c5b8ee5de45b7a4` |

读出权重含完整`state_dict`、冻结配置、step及附加信息；它是组合模型的一个组件，不能当作标准Ultralytics单模型文件直接调用。1000/2000步文件亦保留，正式候选没有按AP择优。训练优化器状态`RAW_STATE.pt`和1704份特征缓存留服务器，不是推理必需输入；冻结视觉标准化/PCA包含在身体对象包中。

## 推理合同

已核环境为Torch2.5.1+cu124、Ultralytics8.4.100、CUDA、NumPy2.4.3，另需OpenCV和pycocotools。服务器环境解释器为`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python`。本地交付机没有该Torch运行时，本地做文件SHA/文档验收；实际RGB一致性在已登记服务器GPU作业277完成。

输入必须遵循已有数据的640×640 RGB画布及预处理。下例说明已验证的组合函数调用；不读GT、GT框或对应标签。若换到其它图片尺寸，应先独立定义预处理和坐标回映，不能直接声称保持本次评价口径。

```python
from pathlib import Path
import torch
from runtime import read
from pipeline import load_models, native_forward, records_for, corrected, predictions
from readout import load

# 在本Study目录使用已核环境。cfg中的原生/对象路径指向已登记服务器文件。
root = Path('/srv/gpu-lab/users/a25023104/projects/spm-instance-body-v16-20261024')
cfg = read(root / 'CONFIG.json')
native, body, statistics = load_models(cfg)
readout, package = load(root / 'runs/RUN_SPM_V16_TRAIN_7bff2ba6a386/readout_step_04000.pt')

# image: 调用方提供的640×640×3 uint8 RGB；image_id: 调用方的图像标识。
with torch.no_grad():
    forward = native_forward(native, statistics, image, cfg)
    records = records_for(body, forward, range(len(forward['out']))) if len(forward['out']) else []
    inputs = dict(base=forward['base'].cpu(), out=forward['out'].cpu(), records=records)
    logits, masks = corrected(readout, inputs)
    outputs = predictions(
        dict(image_id=image_id), forward['out'].cpu().numpy(),
        masks.cpu().numpy(), forward['ids'], forward['ordinals'])
```

`native_forward`保留原生O2O检测集合；`records_for`在每个正常预测框上查询身体、聚合当前实例视觉；`corrected`仅改Mask logit；`predictions`保留原生raw_id、检测ordinal、output_detection_id、框、分数和类别。所有输出均参加评价，包括空Mask，不按GT选择应用。输入特征FP16量化是冻结推理的一部分，原生logit为FP32。

解码为160网格残差叠加→双线性640→预测框裁剪→>0。这与本次真实安装版本的`process_mask(..., upsample=True)`零残差输出一致。上例为接口说明；已经运行的同调用链及逐项断言见[accept_rgb.py](accept_rgb.py)和[6次重载明细](runs/RUN_SPM_V16_RGB_ACCEPTANCE_423e63c0e598/RGB_RELOADS.json)。

## 已保存的证据

- [准备Run](runs/RUN_SPM_V16_PREPARE_0ac6c66c1423/run.json)：数据/环境/关联、原生输出与缓存索引；缓存本体留服务器。
- [训练Run](runs/RUN_SPM_V16_TRAIN_7bff2ba6a386/run.json)：4000步及三份EMA、训练日志、1000/2000/4000完整预测/指标/GT账本。
- [RGB Run](runs/RUN_SPM_V16_RGB_ACCEPTANCE_423e63c0e598/run.json)：实际重新读图前向、固定检测账本、6个技术样例NPZ。
- [独立验收](runs/RUN_SPM_V16_ACCEPTANCE_453b1b1c388b/SUMMARY.json)：48项AP统计最大差0，2863GT/12906输出身份通过。
- [本地验收与存储范围](LOCAL_ACCEPTANCE.json)、[图集](VISUAL_GUIDE.md)、[完整报告](REPORT.md)。

所有原始配置、Run及冻结源码保留。搬迁文件时在单独的调用配置中改路径并重新校验源权重，不覆盖本次已冻结CONFIG或重新运行训练脚本来“验证加载”。本轮未推送GitHub，GPU已释放。
