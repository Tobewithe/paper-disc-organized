"""One seeded CUDA gradient plus exact imported source/version record."""
import argparse
import importlib.metadata
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics
from readout_input_probe import write_json,sha

ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
root=Path(__file__).resolve().parent
source=Path(ultralytics.__file__).resolve()
assert source.is_relative_to(root/'local_readout_runtime_20260912/vendor')
assert ultralytics.__version__=='8.4.143'
torch.manual_seed(23)
x=torch.randn(32,16,device='cuda',requires_grad=True)
loss=(x@x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','opencv-python','pycocotools','pillow']}
versions['ultralytics_imported']=ultralytics.__version__
write_json(args.out,dict(status='PASS',python=sys.version,executable=sys.executable,versions=versions,
           source=str(source),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
           loss=float(loss.detach()),gradient_norm=float(x.grad.norm()),
           weight_sha256=sha(root/'weights/yolo26m-seg.pt'),
           scope='CUDA execution and vendored-source identity; actual model/data witness occurs inline in cache builder'))
print(args.out.read_text(encoding='utf-8'),flush=True)
