import argparse, os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", action="store_true")
    args = ap.parse_args()
    model = YOLO(args.weights)
    if args.method:
        from train_boundary_quality import _attach_quality_head_to_segmodel
        _attach_quality_head_to_segmodel(model.model, alpha=1.0)
    print({"ultralytics": __import__("ultralytics").__version__, "method": args.method}, flush=True)
    model.val(data=args.data, imgsz=640, batch=2, workers=0, device=0, project=args.project, name=args.name,
             plots=False, verbose=False, split="val")
    print("STANDARD_VAL_COMPLETE", flush=True)

if __name__ == "__main__":
    main()

