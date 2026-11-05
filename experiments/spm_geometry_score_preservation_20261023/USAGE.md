# 当前首选模型的权重与使用

当前首选是**原生YOLO的纯框几何组合**：保留最强N的分类、分数、排序、P/c0；采用V11真实250步的框回归。模型没有新增原型分支，总参数26,969,951。原型几何候选因Box75护栏失败未选用，+5目标没有完成。

权重：[native_candidate.pt](runs/RUN_SPM_V13_EVALUATE_3e710949e5b0/native_candidate.pt)

SHA256：`0e40602f1dbf0697fd49fa8e9f887c068970e3c5c39bd618f66e483c8712c8d6`

该文件采用仓库统一的FP32 state_dict格式，包含yaml/config/全部权重。复现使用已登记服务器的Torch2.5.1+cu124、Ultralytics8.4.100环境；下面的仓库加载器使用CUDA：

```python
import sys
from pathlib import Path

repo = Path("/path/to/paper-disc-organized")
sys.path.insert(0, str(repo / "experiments/spm_detection_only_prototype_20261023"))
sys.path.append(str(repo / "experiments/spm_v4_scale_20261023/scripts"))
from models import load_checkpoint

weights = repo / "experiments/spm_geometry_score_preservation_20261023/runs/RUN_SPM_V13_EVALUATE_3e710949e5b0/native_candidate.pt"
model = load_checkpoint(weights)
```

正常RGB推理及Mask解码使用上述目录的 `evaluation.predict`，输入是与训练相同的640×640 letterbox、RGB、[0,1]、BCHW；conf0.001、maxdet300，保留空Mask预测用于同口径AP。原始执行合同与全部命令在[Run](runs/RUN_SPM_V13_EVALUATE_3e710949e5b0/run.json)。它是单类pig模型，输出类索引0对应COCO类别1。

实际来源：N权重SHA `cdea4fb8e837dbbda092bbd5743abe28a8b01731817d9dc326133debd61fe40a`；框回归来自V11原生250步实存检查点。原生与原型两份几何候选各自完成6图重载，12次预测逐框/score/raw/Mask一致；不是只导出一个未核验的网络结构。

服务器完整首选仍位于 `/srv/gpu-lab/users/a25023104/projects/spm-geometry-v13-20261023/runs/RUN_SPM_V13_EVALUATE_3e710949e5b0/native_candidate.pt`。未选F及恢复状态保留服务器并有SHA；[本地验收](LOCAL_ACCEPTANCE.json)区分已回传和服务器保留项。
