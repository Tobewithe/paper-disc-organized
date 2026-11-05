# V21实际末模型用法

这是已实际完成8000步候选学习和8epoch决策学习的末模型。432真实dev的Mask AP为60.4165，比同配置原生+3.0730点；+5未达。正常RGB推理保持原生框/类别/分数/检测身份，只修正Mask。

## 文件与环境

交付目录`final_delivery/db838c03da4e`内：

| 文件 | 含义及SHA256 |
|---|---|
| `weights/SPM_FINAL.pt` | 完整空间分支/决策/记忆及配置，`4b8f85a4602ca62b11941c8ac95d8ba67e3e3d8bfaa387aa81afc4ece1ed1d9a` |
| `weights/native.pt` | 必需的冻结原生模型，`0e40602f1dbf0697fd49fa8e9f887c068970e3c5c39bd618f66e483c8712c8d6` |
| `weights/PROPOSAL_LAST.pt` | 第8000步固定末EMA/优化状态，`1e6041d5103fabae579398dc1fe81d59257d08287c0d745cc4ceb59031ae10ed` |
| `weights/POLICY_LAST.pt` | 第8epoch固定末策略，`5e83740494ef2b6d3956c5ebb906df2b86ce16773d7d5e42d0fd88fecbd108c3` |

服务器实际环境为Python3.12.3、Torch2.5.1+cu124、Ultralytics8.4.100、NumPy2.4.3、pycocotools2.0.11；完整51包和关键源码SHA见`local_reproduction_evidence/ENVIRONMENT_CAPTURE.json`及`requirements.installed.txt`。CUDA12.4，FP32，TF32关闭。训练原解释器为`/srv/gpu-lab/users/a25023104/projects/spm-semantic-dense-v20-20261025/.venv/bin/python`，未修改共享环境。

## 正常RGB推理

在交付目录执行，输入已经是640×640 RGB图：

```bash
python source/predict.py --checkpoint weights/SPM_FINAL.pt --native weights/native.pt --image /absolute/path/image640.png --output /absolute/path/output.npz --device cuda
```

该入口不读取标注，不隐式改尺寸，输出`masks/detections/raw_ids/ordinals/actions/posterior`。mask为640画布；detections前四维是xyxy，后续字段沿冻结原生。`actions=0`为KEEP，其它按`1+k*24+strength_index`编码；posterior是检索分布，**不是最终选用模板**。最终动作应单独读取。

普通新图可使用附加入口：

```bash
python source/predict_raw.py --checkpoint weights/SPM_FINAL.pt --native weights/native.pt --image /absolute/path/new_image.jpg --output /absolute/path/output.npz --device cuda
```

原640输入逐像素保持；非640按最长边缩放、居中114补边，输出原尺寸框和`valid_rect/resize_xy/original_hw`，mask仍在640画布，需裁有效区域后反变换。它不重复JPEG压缩。这是后补便利入口，未作为主评价路径；当前未宣称其非640端到端精确验收。科学成绩仍对应原固定640输入及冻结预处理。

## 复现实验与实际核验范围

原正式根为`/srv/gpu-lab/users/a25023104/projects/spm-spatial-object-memory-v21-20261025/formal_training_v1_db838c03da4e`。`source/TRAINING_PROTOCOL_V1.md`及v2/v3/v4附录固定训练、目标和成本；`source/CODE_FREEZE.json`、`TRAINING_CONFIG_V1.json`、`data/`清单、`runs/`的原命令/Run/日志及`slurm/`保留原执行证据。大RGB、几何及成本缓存未全部复制本地，复训需按SHA清单准备，不能静默换图或缩小分母。

历史`source/infer_final.py`包含已记录的作用域和bbox编码入口问题，保留它是为了追溯原冻结源码。**完整科学推理实际使用`entry_repair_sources/1/entry_repaired.py`及同目录`run_entry.py`**，字段核验使用`entry_repair_sources/2/`；原`slurm`命令和修复manifest定位实际服务器路径。迁移这些完整实验入口须同时映射冻结根目录，不能直接把历史入口当作已修复版本执行。日常正常推理用上面的`source/predict.py`。

实际通过的核验：服务器12张正常RGB严格序列化重载、完整432原生输出重放、独立48项统计误差0；本地用交付源严格加载69项有限状态、1467732参数，候选与固定末EMA逐值一致、策略与固定末8epoch逐值一致。本地CPU核验没有额外前向，不能当成新设备完整推理成绩。

原服务器`DELIVERY_MANIFEST.json`及原`LOCAL_ADDITIONS_MANIFEST.json`保持；新增原生依赖、报告和已看图记录由`FINAL_ADDENDUM_MANIFEST.json`单独核验。旧`visual_QA_pending`是生成时快照，最终实际完成状态以`FINAL_ITERATION_DELIVERY_VERIFIED.json`为准。
