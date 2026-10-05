"""Dual-branch training script for YOLO26-seg on COCO dense pilot dataset.
Compliant with 2026-09-11 Research Experiment Specifications:
- True dual-branch joint training (One-to-Many + One-to-One)
- Proper dynamic weight schedule preservation (no criterion resets)
- Clean GT assignment reuse without intra-GT candidate repulsion
- Multi-head checkpoint preservation (no pruning/fusing)
"""
import os
# Windows local conda environment compatibility (Intel MKL + LLVM OpenMP DLL coexistence)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ccl_dual_loss import DualBranchCCLLoss

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def make_dual_branch_criterion(model):
    ccl_o2m = float(os.environ.get('CCL_WEIGHT_O2M', '0.0'))
    ccl_o2o = float(os.environ.get('CCL_WEIGHT_O2O', '0.0'))
    return DualBranchCCLLoss(model, ccl_weight_o2m=ccl_o2m, ccl_weight_o2o=ccl_o2o)

from ultralytics.nn.tasks import SegmentationModel
SegmentationModel.init_criterion = make_dual_branch_criterion

class DualBranchSegmentationTrainer(SegmentationTrainer):
    """Guarded Segmentation Trainer using the patched DualBranchCCLLoss."""
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "ccl_o2m", "ccl_both"], required=True)
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
    
    # Configure CCL weights based on arm
    if args.arm == "baseline":
        ccl_o2m, ccl_o2o = 0.0, 0.0
    elif args.arm == "ccl_o2m":
        ccl_o2m, ccl_o2o = 0.1, 0.0
    elif args.arm == "ccl_both":
        ccl_o2m, ccl_o2o = 0.1, 0.1
    else:
        raise ValueError(f"Unknown arm: {args.arm}")
        
    run_name = f"{args.arm}_s{args.seed}_e{args.epochs}"
    print(f"\n=======================================================")
    print(f"STARTING EXPERIMENT: Arm={args.arm} | Seed={args.seed} | Run={run_name}")
    print(f"CCL Weights: O2M={ccl_o2m}, O2O={ccl_o2o}")
    print(f"Dataset: {data_yaml}")
    print(f"=======================================================")
    
    yolo = YOLO(str(weights_path))
    
    os.environ['CCL_WEIGHT_O2M'] = str(ccl_o2m)
    os.environ['CCL_WEIGHT_O2O'] = str(ccl_o2o)

    train_args = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        workers=0,  # Windows safe
        project=str(project_dir),
        name=run_name,
        seed=args.seed,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_bias_lr=0.0,
        save_period=1,
        save=True,
        plots=False,
        verbose=True,
        overlap_mask=True,
        mask_ratio=4
    )
    
    yolo.train(trainer=DualBranchSegmentationTrainer, **train_args)
    print(f"\nTraining for {run_name} completed successfully!")

if __name__ == "__main__":
    main()
