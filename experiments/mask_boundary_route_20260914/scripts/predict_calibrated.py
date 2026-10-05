"""Actual one-image/stream inference entry for a frozen mask calibration rule."""
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--mode", choices=["gated","smooth","global","none"], default="gated")
    p.add_argument("--area-gate", type=float, default=2304)
    p.add_argument("--threshold", type=float, default=.75)
    p.add_argument("--project", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.package_root)
    from ultralytics import YOLO
    from mask_calibration import make_calibrated_predictor
    predictor = make_calibrated_predictor(args.mode,args.area_gate,args.threshold)
    for result in YOLO(args.weights).predict(source=args.source, predictor=predictor,
            device=args.device,imgsz=640,retina_masks=False,stream=True,save=True,
            project=args.project,name=args.name):
        print(f"{result.path}: {len(result.boxes)} instances",flush=True)


if __name__=="__main__":
    main()
