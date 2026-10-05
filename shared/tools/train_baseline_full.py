import sys
import os
from pathlib import Path

sys.path.insert(0, 'C:/Dpan/codexproject/pigcv_research')
from models.yolo26seg.loader import load_model

def main():
    # Load base model for fine-tuning
    model = load_model(Path('C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02/yolo26_task05_final.pt'), 'cuda')
    
    data_yaml = 'experiments/baseline_finetune/faro_baseline_train.yaml'
    Path('experiments/baseline_finetune').mkdir(parents=True, exist_ok=True)
    Path(data_yaml).write_text(
        "path: C:/Dpan/document/model_datasets/datasets/FaroPigSeg\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: pig\n"
    )
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    model.train(
        data=data_yaml,
        epochs=15,
        imgsz=640,
        batch=8,
        device='cuda',
        project='experiments/baseline_finetune',
        name='run_baseline_15e',
        workers=0,
        optimizer='AdamW',
        lr0=0.001
    )

if __name__ == '__main__':
    main()
