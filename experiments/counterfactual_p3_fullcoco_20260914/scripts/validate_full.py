"""Save paired stock one-to-one predictions on all COCO val2017 images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    args = ap.parse_args()
    receipt = {}
    for arm, checkpoint in (("baseline_s0", args.baseline), ("cfp3r_s0", args.method)):
        model = YOLO(str(checkpoint))
        metrics = model.val(data=str(args.data), split="val", imgsz=640, batch=8, workers=8,
                            device=0, conf=0.001, iou=0.7, max_det=300, nms=False,
                            save_json=True, plots=False, project=str(args.project), name=arm,
                            exist_ok=False, verbose=True)
        receipt[arm] = {"checkpoint": str(checkpoint), "save_dir": str(metrics.save_dir)}
    (args.project / "VALIDATION_COMPLETE.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
