import sys
from pathlib import Path
sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

model = load_model(Path('C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt'), 'cpu')

for i, m in enumerate(model.model.model):
    print(f"Layer {i}: {m.__class__.__name__}, f: {m.f}")
