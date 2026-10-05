import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="official_baseline")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=0,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=False,
        verbose=False,
        cache=False,
        pretrained=args.weights,
        save=True,
        val=False,
        seed=0,
    )


if __name__ == "__main__":
    main()
