import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics.utils import YAML
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--method", action="store_true")
    ap.add_argument("--base-weights", default=None,
                    help="official base weights used to construct the model before restoring a custom quality-head checkpoint")
    ap.add_argument("--quality-alpha", type=float, default=1.0,
                    help="exponent used by the quality score multiplier during inference")
    args = ap.parse_args()
    # A custom quality head is not present in the original YOLO YAML.  Loading
    # the custom checkpoint with YOLO(args.weights) first silently drops the
    # extra quality_head keys, and attaching it afterwards creates a random
    # head.  Construct the official model, attach the head, then restore the
    # full checkpoint state so evaluation uses the trained parameters.
    if args.method and args.base_weights:
        model = YOLO(args.base_weights)
        from train_boundary_quality import _attach_quality_head_to_segmodel
        _attach_quality_head_to_segmodel(model.model, alpha=args.quality_alpha)
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint.get("model") or checkpoint.get("ema")
        state = checkpoint_model.state_dict() if hasattr(checkpoint_model, "state_dict") else checkpoint.get("state_dict")
        if state is None:
            raise RuntimeError("custom checkpoint has no model/ema state_dict")
        missing, unexpected = model.model.load_state_dict(state, strict=False)
        print({"restored_custom_checkpoint": args.weights,
               "missing": list(missing), "unexpected": list(unexpected)}, flush=True)
    else:
        model = YOLO(args.weights)
    if args.method:
        from train_boundary_quality import _attach_quality_head_to_segmodel
        _attach_quality_head_to_segmodel(model.model, alpha=args.quality_alpha)
    data = YAML.load(args.data)
    root = Path(data["path"]).resolve()
    data["path"] = str(root)
    data["train"] = str(root / "images" / "train2017")
    data["val"] = str(root / "images" / "val2017")
    print({"ultralytics": __import__("ultralytics").__version__, "method": args.method,
           "quality_alpha": args.quality_alpha,
           "val": data["val"], "label_dir": str(root / "labels" / "val2017")}, flush=True)
    metrics = model.val(data=args.data, imgsz=640, batch=2, workers=0, device=0, project=args.project, name=args.name,
                        plots=False, verbose=False, split="val", cache=False)
    payload = {
        "images": 5000,
        "box_map50_95": float(metrics.box.map),
        "box_map50": float(metrics.box.map50),
        "mask_map50_95": float(metrics.seg.map),
        "mask_map50": float(metrics.seg.map50),
    }
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    print("STANDARD_VAL_COMPLETE", flush=True)

if __name__ == "__main__":
    main()

