"""Direct COCO pilot training for the instance-boundary ownership mechanism."""
import argparse, json, os, random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("WANDB_MODE","disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")

import numpy as np
import torch
from ultralytics import YOLO, settings
from ultralytics.models.yolo.segment.train import SegmentationTrainer
import ultralytics.engine.trainer as _trainer_module
from boundary_loss import BoundaryOwnershipLoss, install_boundary


class BoundaryTrainer(SegmentationTrainer):
    """Stock Ultralytics trainer with only the segmentation criterion replaced."""
    pass


def finite_model(model):
    return all(torch.isfinite(v).all().item() for v in model.state_dict().values()
               if isinstance(v, torch.Tensor) and v.is_floating_point())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline","self_out","self_out_neighbor"], required=True)
    ap.add_argument("--seed", type=int, choices=[0,1,2], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-path", type=Path, default=ROOT/"boundary_ownership_data.yaml")
    ap.add_argument("--model-path", type=Path, default=ROOT/"weights/yolo26m-seg.pt")
    ap.add_argument("--project-path", type=Path, default=ROOT/"runs/boundary_ownership_20260914")
    ap.add_argument("--resume-path", type=Path, default=None,
                    help="Resume a partially completed run from an Ultralytics last.pt checkpoint.")
    args=ap.parse_args()
    settings.update({k:False for k in ["wandb","comet","mlflow","clearml","neptune"] if k in settings})
    # Fixed, pre-registered weights; only the loss arm varies.
    if args.arm == "baseline":
        ls=(0.0,0.0,0.0)
    elif args.arm == "self_out":
        ls=(0.25,0.25,0.0)
    else:
        ls=(0.25,0.25,0.25)
    os.environ["BOUNDARY_LAMBDA_SELF"]=str(ls[0])
    os.environ["BOUNDARY_LAMBDA_OUT"]=str(ls[1])
    os.environ["BOUNDARY_LAMBDA_NBR"]=str(ls[2])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    install_boundary()
    # Local official weights are already pinned; avoid Ultralytics' optional
    # network download of yolo26n.pt during its AMP self-check.
    _trainer_module.check_amp = lambda model: True
    model=YOLO(str(args.resume_path if args.resume_path is not None else args.model_path))
    project=args.project_path
    name=f"{args.arm}_s{args.seed}"
    data_path=args.data_path
    if args.smoke:
        import yaml
        smoke=yaml.safe_load(data_path.read_text(encoding="utf-8"))
        smoke["train"]=str(ROOT/"boundary_smoke_train.txt")
        smoke["val"]=str(ROOT/"boundary_smoke_val.txt")
        data_path=ROOT/"boundary_smoke_data.yaml"
        data_path.write_text(yaml.safe_dump(smoke,sort_keys=False),encoding="utf-8")
    cfg=dict(data=str(data_path), imgsz=640, epochs=1 if args.smoke else args.epochs,
             batch=2, workers=0, device=0, optimizer="AdamW", lr0=1e-4, lrf=0.01, cos_lr=True,
             momentum=0.9, weight_decay=5e-4, warmup_epochs=1.0, warmup_bias_lr=0.0,
             nbs=16, amp=True, deterministic=True, patience=0, save=True, save_period=1,
             val=True, plots=False, cache=False, rect=False, overlap_mask=True, mask_ratio=1,
             mosaic=1.0, close_mosaic=5, mixup=0.0, copy_paste=0.0, degrees=0.0,
             translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
             project=str(project), name=name, exist_ok=False, seed=args.seed)
    if args.resume_path is not None:
        cfg["resume"] = str(args.resume_path)
    model.train(trainer=BoundaryTrainer, **cfg)
    save_dir=Path(model.trainer.save_dir)
    criterion=model.trainer.model.criterion
    receipt={"status":"COMPLETE","arm":args.arm,"seed":args.seed,"epochs":cfg["epochs"],
             "loss_weights":{"self":ls[0],"outside":ls[1],"neighbor":ls[2]},
             "train_images":1000,"val_images":1576,"mask_ratio":1,"batch":2,
             "criterion":type(criterion).__name__,"finite":finite_model(model.trainer.model)}
    (save_dir/"BOUNDARY_TRAINING_COMPLETE.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print("BOUNDARY_TRAINING_COMPLETE",json.dumps(receipt),flush=True)


if __name__=="__main__": main()
