import torch
import sys
from pathlib import Path

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

model = load_model(Path('C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt'), 'cuda')

img = torch.rand(1, 3, 640, 640, device='cuda')
preds = model.model(img)

def print_shape(obj, indent=''):
    if isinstance(obj, tuple):
        print(indent + 'tuple len ' + str(len(obj)))
        for x in obj:
            print_shape(x, indent + '  ')
    elif isinstance(obj, list):
        print(indent + 'list len ' + str(len(obj)))
        for x in obj:
            print_shape(x, indent + '  ')
    elif isinstance(obj, torch.Tensor):
        print(indent + 'tensor ' + str(obj.shape))
    else:
        print(indent + str(type(obj)))

print_shape(preds)
