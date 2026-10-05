"""Dual-branch training script for YOLO26-seg with Dilated Mask Supervision (DMS).
Compliant with 2026-09-11 Research Experiment Specifications:
- True dual-branch joint training (One-to-Many + One-to-One)
- Proper dynamic weight schedule preservation (no criterion resets)
- Clean GT assignment reuse
- Verified official starting weights (SHA256 check)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from dilated_dual_loss import DualBranchResilientLoss

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def make_resilient_criterion(model):
    alpha_train = float(os.environ.get('ALPHA_TRAIN', '0.0'))
    return DualBranchResilientLoss(model, alpha_train=alpha_train)

from ultralytics.nn.tasks import SegmentationModel
SegmentationModel.init_criterion = make_resilient_criterion

class ResilientSegmentationTrainer(SegmentationTrainer):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["dilated_0.1", "dilated_0.2", "baseline_check"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    weights_path = root / "weights/yolo26m-seg.pt"
    data_yaml = root / "data/pilot_1000/coco_pilot_1000.yaml"
    project_dir = root / "runs/pilot_dual_branch"
    
    # Verify starting checkpoint hash
    expected_sha = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
    actual_sha = sha256_file(weights_path)
    assert actual_sha == expected_sha, f"Weight SHA256 mismatch! Got {actual_sha}, expected {expected_sha}"
    print(f"Pretrained weights verified: SHA256={actual_sha}")
    
    if args.arm == "dilated_0.1":
        alpha_train = 0.1
        exp_name = f"dilated_0.1_s{args.seed}_e{args.epochs}"
    elif args.arm == "dilated_0.2":
        alpha_train = 0.2
        exp_name = f"dilated_0.2_s{args.seed}_e{args.epochs}"
    elif args.arm == "baseline_check":
        alpha_train = 0.0
        exp_name = f"baseline_check_s{args.seed}_e{args.epochs}"
    else:
        raise ValueError(f"Unknown arm {args.arm}")
        
    os.environ['ALPHA_TRAIN'] = str(alpha_train)
    print(f"=== Running Arm: {args.arm} | ALPHA_TRAIN={alpha_train} ===")
    
    save_dir = project_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "arm": args.arm,
        "alpha_train": alpha_train,
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "base_weights_sha256": actual_sha,
        "dataset": str(data_yaml)
    }
    (save_dir / "run_spec.json").write_text(json.dumps(spec, indent=2))
    
    yolo = YOLO(str(weights_path))
    yolo.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        seed=args.seed,
        workers=0,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        lr0=0.0001,
        warmup_bias_lr=0.0,
        optimizer="AdamW",
        save=True,
        save_period=1,
        plots=False,
    )
    print(f"Training for {exp_name} completed successfully!")

if __name__ == "__main__":
    main()
