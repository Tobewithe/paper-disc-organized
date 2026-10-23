# 已训练对象模型与独立推断

两份实际4000步检查点均已回传本地；[SHA与训练来源](MODEL_MANIFEST.json)关联正式训练Run和固定评价Run。它们是单给定猪资产、已知相机的对象学习v0，当前整体稳定性Gate未通过，真实分割效用见[结案报告](../REPORT.md)。

| 文件 | 含义 |
|---|---|
| [structured_last.pt](structured_last.pt) | 已训练的结构对象模型，453384个可训练参数 |
| [free_last.pt](free_last.pt) | 相同预算和容量的普通对象对照 |
| [template_inference.npz](template_inference.npz) | 共享类别几何/蒙皮/形变先验；不含按样本GT姿态或目标顶点 |
| [demo_occluded.png](demo_occluded.png) | 固定留出样本f024_p3_v1_a0_o1的遮挡RGB输入 |
| [demo_camera.json](demo_camera.json) | 对应已知源相机，仅矩阵送入网络 |
| [预测三维对象](demo_inference_232/object_prediction.obj) | 由已训练S从上述输入推断的几何，不是直接复制资产网格 |
| [完整推断数组](demo_inference_232/object_prediction.npz) | 283顶点、16维体形、24骨骼状态和64维表面特征 |
| [推断凭据](demo_inference_232/INFERENCE.json) | 模型/输入/相机SHA和实际输入角色 |

在已登记服务器的本Study目录下，可用现有解释器直接复用入口，无需训练：

```bash
/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python \
  presentation_tools/infer_object.py \
  --model-dir artifacts \
  --image artifacts/demo_occluded.png \
  --camera artifacts/demo_camera.json \
  --output artifacts/my_object_prediction
```

输出目录需尚不存在，程序不会静默覆盖。当前输入必须是相机标定匹配的128×128单对象图像，入口不自行裁剪/缩放或猜相机。输入该样本的GT Mask、GT体形与GT姿态均不需要；跨真实域和未标定相机的推断尚未验证。

权重/数组不提交Git，保存在本地与服务器 `/srv/gpu-lab/users/a25023104/projects/spm-object-prototype-20261023/artifacts`；源码、报告和SHA清单可在GitHub复核。
