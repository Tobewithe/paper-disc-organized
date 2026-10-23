# V4.1权重与原型加载

仅当对应Run真实完成后，下列指针才存在：BASELINE、CONTINUED、ADAPTER、RANDOM_MEMORY、TASK_SPM、PROTOTYPE_FINAL、EVALUATE_TASK_POINTER.json。查看run.json和SUMMARY核实际完成状态，不把本说明当交付已完成。

五个分割best.pt保存FP32模型state_dict、yaml、模式和所需冻结memory；F/G/R由原32基底/系数扩为64，须用本Study的task_model.load恢复。不要直接将自定义字典交给Ultralytics YOLO(path)。所有部署输入只需RGB。

```python
import sys
sys.path.insert(0, '/absolute/path/to/spm_v4_scale_20261023/scripts')
from task_model import load
model = load('/absolute/path/to/best.pt', device='cpu').eval()
```

推理入口scripts/infer_task.py支持--checkpoint、--image、--output、--device，进行640 letterbox、原生O2O可见Mask解码并把Mask还原到原图大小。conf.001为AP评价候选集合，展示可再按.25过滤，不应把展示分数改作评价配置。

PROTOTYPE_FINAL Run的memory.pt单独保存512维归一化、512→256→128投影、32个单位中心、训练来源/exemplar、B*哈希；PROJECTOR_STATE.pt保留20000步投影/辅助头及优化器。特征features.pt/embedding.pt、全部训练状态和数据缓存留服务器，不把token缓存误当模型。

固定环境为Ultralytics8.4.100、Torch2.5.1+cu124，原始COCO yolo26m-seg.pt的SHA为16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5；具体运行环境和源代码以各Run SOURCE及冻结清单为准。

轻量结果包与权重包分别附SHA清单，本地接收后逐文件验证；只有验收完成才记artifact/transfer completed。科学目标是否完成读取总评价DECISION，不由包名、文件存在或loss下降决定。
