"""Record this experiment's execution environment and immutable inputs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006')
VENDOR = Path('D:/coco_wire/vendor_8.4.100')
sys.path.insert(0, str(VENDOR))
os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
import torch
import ultralytics
from ultralytics import YOLO

assert ultralytics.__version__ == '8.4.100'
weight = Path('D:/coco_wire/models/yolo26m-seg.pt')
weight_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
assert weight_sha == '16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5'
model = YOLO(str(weight)).model.float().eval()
prefixes = ('model.23.cv4.', 'model.23.one2one_cv4.', 'model.23.proto.cv2.', 'model.23.proto.cv3.')
bn_names = {name + '.' + key for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for key in module.state_dict()}
parameters = {name: param for name, param in model.named_parameters()
              if name.startswith(prefixes) and name not in bn_names}
counts = {prefix: sum(value.numel() for name, value in parameters.items() if name.startswith(prefix))
          for prefix in prefixes}
assert sum(counts.values()) == 2306240, counts
for module in ('yaml', 'pycocotools', 'numpy', 'scipy'):
    __import__(module)
gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                      '--format=csv,noheader'], capture_output=True, text=True)
receipt = {'status': 'PASS', 'interpreter': sys.executable, 'torch': torch.__version__,
           'ultralytics': ultralytics.__version__, 'ultralytics_source': ultralytics.__file__,
           'device': torch.cuda.get_device_name(0), 'weights_sha256': weight_sha,
           'trainable_parameter_counts': counts, 'trainable_total': sum(counts.values()),
           'gpu_compute_processes_observed': gpu.stdout.strip().splitlines(),
           'scope': 'Environment/model/input read-only check; no training result'}
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'PREFLIGHT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
