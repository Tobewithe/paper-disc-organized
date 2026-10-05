"""Compare the public predictor adapter with official predict on one real image."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--package-root",required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    sys.path.insert(0,args.package_root)
    import torch
    from ultralytics import YOLO
    from mask_calibration import make_calibrated_predictor
    torch.set_num_threads(4)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    model=YOLO(args.weights)
    common=dict(source=args.source,device=0,imgsz=640,conf=.001,max_det=300,verbose=False,save=False,retina_masks=False)
    baseline=model.predict(**common)[0].cpu()
    model.predictor=None
    disabled=model.predict(**common,predictor=make_calibrated_predictor(threshold=0))[0].cpu()
    assert torch.equal(baseline.boxes.data,disabled.boxes.data),"Zero mode changes boxes/scores"
    assert torch.equal(baseline.masks.data,disabled.masks.data),"Zero mode changes masks"
    model.predictor=None
    calibrated=model.predict(**common,predictor=make_calibrated_predictor())[0].cpu()
    baseline[baseline.boxes.conf>=.25].save(str(out/"baseline_preview.jpg"))
    calibrated[calibrated.boxes.conf>=.25].save(str(out/"calibrated_preview.jpg"))
    summary=dict(run_id=os.environ.get("RESEARCH_RUN_ID"),zero_threshold_pixel_parity=True,
        boxes_scores_parity=True,source=args.source,baseline_instances=len(baseline.boxes),
        calibrated_instances=len(calibrated.boxes),mode="default one-to-one, non-retina",
        limitations=["single-image adapter smoke, not AP", "previews filter score .25 for readability only"])
    (out/"SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
