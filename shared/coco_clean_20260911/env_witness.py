"""Seeded real-model GPU forward/backward and official weight provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torchvision
import ultralytics
from ultralytics import YOLO

ROOT = Path('/root/autodl-tmp/coco_clean_20260911')
assert torch.__version__ == '2.8.0+cu128'
assert torchvision.__version__ == '0.23.0+cu128'
assert ultralytics.__version__ == '8.4.143'
assert np.__version__ == '2.2.6'
torch.set_num_threads(4)
torch.manual_seed(1729)
torch.cuda.manual_seed_all(1729)
weights = ROOT / 'weights/yolo26m-seg.pt'
model = YOLO(str(weights)).model.cuda().train()
model.requires_grad_(True)
x = torch.rand(2, 3, 128, 128, device='cuda')
out = model(x)
def leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values(): yield from leaves(v)
    elif isinstance(value, (tuple, list)):
        for v in value: yield from leaves(v)
tensors = [v for v in leaves(out) if v.requires_grad]
loss = sum(v.float().square().mean() for v in tensors)
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert tensors and grads and torch.isfinite(loss)
assert all(torch.isfinite(g).all() for g in grads)
assert sum(float(g.abs().sum()) for g in grads) > 0
result = dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
              gpu=torch.cuda.get_device_name(), outputs=[list(v.shape) for v in tensors],
              loss=float(loss.detach()), gradients=len(grads),
              weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
              weights_bytes=weights.stat().st_size)
(ROOT/'audits').mkdir(exist_ok=True)
(ROOT/'audits/env_witness.json').write_text(json.dumps(result, indent=2))
print('WITNESS_PASS', json.dumps(result), flush=True)
