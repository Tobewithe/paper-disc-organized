"""Dual-branch aligned training script for YOLO26-seg on COCO pilot dataset.

Supports:
- arm=baseline: Official E2ELoss baseline (alpha=0.0, lambda=0.0).
- arm=control_weight: Mask loss weight scaling control (alpha=0.0, weight_scale=2.0).
- arm=dms_unconfounded: Decoupled margin supervision (alpha=0.1, lambda_margin=1.0).
- Multi-seed execution: seed=0, 1, 2.
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
from unconfounded_dilated_loss import DualBranchUnconfoundedLoss

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def make_unconfounded_criterion(model):
    alpha_train = float(os.environ.get('ALPHA_TRAIN', '0.0'))
    alpha_guard = float(os.environ.get('ALPHA_GUARD', '0.0'))
    lambda_margin = float(os.environ.get('LAMBDA_MARGIN', '0.0'))
    crit = DualBranchUnconfoundedLoss(model, alpha_train=alpha_train, lambda_margin=lambda_margin, alpha_guard=alpha_guard)
    weight_scale = float(os.environ.get('WEIGHT_SCALE', '1.0'))
    crit.one2many.weight_scale = weight_scale
    crit.one2one.weight_scale = weight_scale
    return crit

from ultralytics.nn.tasks import SegmentationModel
SegmentationModel.init_criterion = make_unconfounded_criterion

class AlignedSegmentationTrainer(SegmentationTrainer):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "control_weight", "dms_unconfounded", "dms_guardband"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_ratio", type=int, default=4)
    parser.add_argument("--alpha_guard", type=float, default=0.02)
    parser.add_argument("--lambda_margin", type=float, default=None)
    args = parser.parse_args()
    
    root = Path(r"C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911")
    weights_path = root / "weights/yolo26m-seg.pt"
    data_yaml = root / "data/pilot_1000/coco_pilot_1000.yaml"
    project_dir = root / "runs/aligned_benchmark"
    
    # Verify starting checkpoint hash
    expected_sha = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
    actual_sha = sha256_file(weights_path)
    assert actual_sha == expected_sha, f"Weight SHA256 mismatch! Got {actual_sha}, expected {expected_sha}"
    print(f"Pretrained weights verified: SHA256={actual_sha}")
    
    alpha_guard = 0.0
    if args.arm == "baseline":
        alpha_train = 0.0
        lambda_margin = 0.0
        weight_scale = 1.0
    elif args.arm == "control_weight":
        alpha_train = 0.0
        lambda_margin = 0.0
        weight_scale = 2.0
    elif args.arm == "dms_unconfounded":
        alpha_train = 0.1
        lambda_margin = 1.0 if args.lambda_margin is None else args.lambda_margin
        weight_scale = 1.0
    elif args.arm == "dms_guardband":
        alpha_train = 0.1
        alpha_guard = args.alpha_guard
        lambda_margin = 0.3 if args.lambda_margin is None else args.lambda_margin
        weight_scale = 1.0
    else:
        raise ValueError(f"Unknown arm {args.arm}")
        
    exp_name = f"{args.arm}_s{args.seed}_e{args.epochs}_mr{args.mask_ratio}"
    os.environ['ALPHA_TRAIN'] = str(alpha_train)
    os.environ['ALPHA_GUARD'] = str(alpha_guard)
    os.environ['LAMBDA_MARGIN'] = str(lambda_margin)
    os.environ['WEIGHT_SCALE'] = str(weight_scale)
    
    print(f"\n=======================================================")
    print(f"Running Experiment Arm: {args.arm} (Seed={args.seed})")
    print(f"  alpha_train={alpha_train}, lambda_margin={lambda_margin}, weight_scale={weight_scale}")
    print(f"  mask_ratio={args.mask_ratio}, epochs={args.epochs}, batch={args.batch}")
    print(f"=======================================================\n")
    
    save_dir = project_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "arm": args.arm,
        "alpha_train": alpha_train,
        "lambda_margin": lambda_margin,
        "weight_scale": weight_scale,
        "mask_ratio": args.mask_ratio,
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
        val=False,
        plots=False,
        mask_ratio=args.mask_ratio
    )
    print(f"\nTraining completed for {exp_name}. Weights saved in {save_dir}")

if __name__ == "__main__":
    main()
