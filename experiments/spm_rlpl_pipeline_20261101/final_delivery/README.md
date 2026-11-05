# RLPL实际交付包

完整有限流程已完成；**本轮推荐保留N**。B2/C2/D2均为已完成2000更新的阴性实验末权重，供复核/研究归档，不建议替换N。详细结论见[完整报告](../FINAL_REPORT.md)与[冻结规则裁决](../FINAL_DECISION.json)。

## 正常推理

在登记的兼容环境中，从本目录运行：

```bash
python cli/predict_native.py --weights weights/N_NATIVE.pt --image /path/to/prepared_640_rgb.png --output predictions.json --device cuda --image-id 1
```

- `cli/`两个文件逐字节来自已通过1036部署核验的冻结源码，而非另写的推理包装。
- 本权重是**原生Ultralytics分割结构的yaml+state_dict格式**，不包含辅助原型/投影头或后置修补；通过随包CLI严格构造/加载。不要假定它是可直接交给`yolo predict model=...`的官方pickle格式。
- 输入必须是共同协议预处理后的640×640 RGB uint8；没有隐式resize/letterbox。输出正常O2O检测与Mask RLE；conf>.001、Top-K300、无NMS。评价器maxDets100是另一层配置。
- 类别id=1；不读取GT，不进行检索或Mask修补。GT及私有原图不附在此公开入口中。
- 环境记录：`RUNTIME_ENVIRONMENT_V1.json`。已在登记服务器的Python3.12.3、Torch2.5.1+cu124、Ultralytics8.4.100、NumPy2.4.3、Pillow11.1.0、pycocotools2.0.11下实测；本机仅完成字节/SHA回传核验，没有冒称本地CUDA运行通过。
- 先核`PACKAGE_MANIFEST.json`及`NATIVE_WEIGHTS_RETURN_VERIFIED.json`，仅加载可信且SHA相符的权重。

## 包内内容

- `weights/`：N/B2/C2/D2四个完整原生权重，共433644436字节。
- `cli/`：通过实际部署资格的纯原生加载/推理代码。
- `evidence/`：三臂训练摘要、完整2000步日志、评价/GT账本/区间/分组成绩、独立统计及部署摘要的原字节副本。
- `figures/`：架构、完整AP、全程原始训练曲线、实测正常推理成本四张技术图。
- `FORMAL_TRAINING_V1.json`和`FORMAL_TRAINING_PROTOCOL_V1.md`：实际冻结的机器/正文合同。
- `SOURCE_PROVENANCE.json`与`PACKAGE_MANIFEST.json`：每个副本的原来源和哈希；没有覆盖旧Run。

私有真实效果图在`../private_data/declared_previews_1038/`，其9图已实际审阅。600张合成图及大型Blend/EXR/NPZ、恢复状态和训练期辅助权重保留服务器，清单见原Run；不把本包误认为全部原始训练数据已迁移。

重现完整训练需使用原来源、真实Run的`source/`/`CODE_FREEZE.json`、日程/坐标/数据SHA及明确新预算，见[复现说明](../REPRODUCTION_README.md)。禁止删除收据以绕过重复训练保护；无自动恢复/后继，test426继续封存。
