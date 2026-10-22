# SPM-YOLO V4权重与推理

优先使用`runs/RUN_V4_EXPORT_LITE_dba5bbf80522/best.pt`：它保留F训练得到的原生YOLO参数，部署无需原型检索，54图dev Mask AP=50.9272。完整研究版F为50.9267。二者均FP32；导出增加训练步数0，5张图与关alpha的完整F输出最大绝对差为0，已独立重载并完整评价。

| 文件 | 内容 |
|---|---|
| `runs/RUN_V4_EXPORT_LITE_dba5bbf80522/best.pt` | 推荐：SPM训练权重、原生head推理 |
| `runs/RUN_V4_SPM_2f2740fc801a/best.pt` | 完整SPM分支/原型/投影研究模型 |
| `runs/RUN_V4_BASELINE_013ffd51f526/best.pt` | 所选real-only B*基线 |
| `runs/RUN_V4_CONTINUED_a7f79eba1da2/best.pt` | 同数据/同步数继续训练对照 |
| `runs/RUN_V4_ADAPTER_209c3f995a24/best.pt` | 普通Adapter对照 |
| `runs/RUN_V4_PROTOTYPE_00bab48aef0f/memory.pt` | 32×128中心、投影、归一化、辅助头及来源 |

这些`.pt`采用显式`yaml + state_dict`格式；完整F另嵌入memory，需用提供的加载器，不能直接当Ultralytics官方pickle格式传给`YOLO(path)`。只加载本项目可信来源权重。依赖固定Ultralytics 8.4.100、PyTorch及OpenCV、pycocotools，版本见ENVIRONMENT.json。推理脚本支持CPU/CUDA，CPU样例已实际运行，输出见examples。

从Study目录运行：

```powershell
python scripts/infer.py --checkpoint runs/RUN_V4_EXPORT_LITE_dba5bbf80522/best.pt --source examples/real_833.png --output examples/my_prediction.png --device cpu
```

服务器GPU执行应放入一次性Slurm作业，使用登记解释器与环境；将`--device cpu`改为`--device cuda`。输入RGB与训练/评价一致转换为640画布；输出PNG和带输入/权重SHA、检测身份、score及可见Mask RLE的JSON。`.25`只控制样例展示，底层保留`.001`以上原生O2O检测；主成绩不使用展示阈值。

Python使用：把scripts目录加入模块路径后，调用`infer.load_checkpoint(path, device='cpu')`返回原生模型及checkpoint元数据。模型names固定为`{0: 'pig'}`。原始input、letterbox及mask解码均已在脚本中提供。

本地6项大文件与服务器SHA逐项一致，记录见TRANSFER_RECEIPT.json；原始压缩包分别为SPM_V4_WEIGHTS.tar.gz和SPM_V4_RESULTS_LIGHT.tar.gz。最后一步权重和训练特征仍在服务器各自Run目录，未覆盖选优模型或阴性结果。
