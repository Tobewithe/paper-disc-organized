"""Evaluate Ultralytics post-training best.pt predictions with official COCO metrics."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def restricted_coco(source: COCO, image_ids: set[int]) -> COCO:
    dataset = {
        "info": source.dataset.get("info", {}),
        "licenses": source.dataset.get("licenses", []),
        "images": [source.imgs[i] for i in sorted(image_ids)],
        "annotations": [a for a in source.dataset["annotations"] if int(a["image_id"]) in image_ids],
        "categories": source.dataset["categories"],
    }
    target = COCO()
    target.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        target.createIndex()
    return target


def recall_at_precision(ev: COCOeval, iou: float, precision_target: float) -> float:
    t = int(np.abs(ev.params.iouThrs - iou).argmin())
    values = ev.eval["precision"][t, :, :, 0, -1]  # recall x category, all area, maxDet=100
    mean_precision = np.array([
        row[row >= 0].mean() if np.any(row >= 0) else np.nan for row in values
    ])
    valid = np.flatnonzero(mean_precision >= precision_target)
    return float(ev.params.recThrs[valid[-1]]) if len(valid) else 0.0


def per_category(ev: COCOeval, gt: COCO, arm: str, kind: str) -> list[dict[str, object]]:
    precision = ev.eval["precision"][:, :, :, 0, -1]
    recall = ev.eval["recall"][:, :, 0, -1]
    t50 = int(np.abs(ev.params.iouThrs - .50).argmin())
    t75 = int(np.abs(ev.params.iouThrs - .75).argmin())
    rows = []
    for k, category_id in enumerate(ev.params.catIds):
        def avg(x: np.ndarray) -> float:
            good = x[x >= 0]
            return float(good.mean()) if len(good) else float("nan")
        rows.append({
            "arm": arm,
            "task": kind,
            "category_id": category_id,
            "category_name": gt.cats[category_id]["name"],
            "ap": avg(precision[:, :, k]),
            "ap50": avg(precision[t50, :, k]),
            "ap75": avg(precision[t75, :, k]),
            "r50": float(recall[t50, k]) if recall[t50, k] >= 0 else float("nan"),
            "r75": float(recall[t75, k]) if recall[t75, k] >= 0 else float("nan"),
        })
    return rows


def evaluate(gt: COCO, predictions: list[dict[str, object]], arm: str, kind: str):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, kind)
        ev.params.imgIds = sorted(gt.imgs)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = ["ap", "ap50", "ap75", "aps", "apm", "apl", "ar_max1", "ar_max10", "ar_max100", "ar_small", "ar_medium", "ar_large"]
    row = {"arm": arm, "task": kind, "images": len(gt.imgs), "predictions": len(predictions)}
    row.update({name: float(value) for name, value in zip(names, ev.stats)})
    for iou in (.50, .75):
        for p in (.80, .90):
            row[f"recall_at_p{int(p*100)}_iou{int(iou*100)}"] = recall_at_precision(ev, iou, p)
    return row, per_category(ev, gt, arm, kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        source = COCO(str(args.annotation))
    summaries: list[dict[str, object]] = []
    categories: list[dict[str, object]] = []
    prediction_paths = sorted(args.runs.glob("*/predictions.json"))
    if not prediction_paths:
        raise FileNotFoundError(f"No predictions.json under {args.runs}")
    for path in prediction_paths:
        arm = path.parent.name
        predictions = json.loads(path.read_text(encoding="utf-8"))
        image_ids = {int(row["image_id"]) for row in predictions}
        gt = restricted_coco(source, image_ids)
        for kind in ("bbox", "segm"):
            summary, rows = evaluate(gt, predictions, arm, kind)
            summaries.append(summary)
            categories.extend(rows)
            print(arm, kind, json.dumps(summary, ensure_ascii=False), flush=True)
    write_csv(args.out / "official_metrics.csv", summaries)
    write_csv(args.out / "per_category_metrics.csv", categories)
    (args.out / "COMPLETE.json").write_text(json.dumps({
        "status": "complete",
        "checkpoint": "best.pt; Ultralytics post-training final_eval overwrote predictions.json after explicitly validating best.pt",
        "inference_branch": "default one-to-one",
        "runs": [p.parent.name for p in prediction_paths],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
