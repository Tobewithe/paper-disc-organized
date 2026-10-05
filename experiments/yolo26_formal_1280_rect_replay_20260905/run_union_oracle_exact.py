"""Compute a small multi-candidate mask-union oracle from retained exact raw outputs."""

from __future__ import annotations

import csv
import json
import zipfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "full_exact_8427_20260905"
SAMPLES = ROOT / "yolo_dense_failure_samples_reconstructed_20260905.csv"
MANIFEST = Path(r"C:\Dpan\document\model_datasets\datasets\piglife\derived\task05_v1\pig_coco_test_task05_v1.json")
IMAGE_ZIP = Path(r"C:\Dpan\document\model_datasets\datasets\piglife\Image\test.zip")
OUTPUT = ROOT / "union_oracle_exact_20260905"


def gt_mask(annotation: dict, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    seg = annotation["segmentation"]
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w), intersect=False)
    else:
        rle = dict(seg)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def scores(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    area = gt.sum()
    return float(inter / union) if union else 0.0, float(inter / area) if area else 0.0


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in manifest["images"]}
    anns = {}
    for a in manifest["annotations"]:
        if int(a.get("iscrowd", 0)) == 0 and int(a.get("category_id", 1)) == 1:
            anns[int(a["id"])] = a
    with SAMPLES.open(encoding="utf-8-sig", newline="") as f:
        targets = [r for r in csv.DictReader(f) if r["sample_role"] == "failure"]
    with (RUN / "yolo26_gt_candidate_trace.csv").open(encoding="utf-8-sig", newline="") as f:
        trace = {}
        for row in csv.DictReader(f):
            if row["annotation_id"] in {r["annotation_id"] for r in targets}:
                trace.setdefault(row["annotation_id"], []).append(row)

    rows = []
    with zipfile.ZipFile(IMAGE_ZIP) as archive:
        for image_id in sorted({int(r["image_id"]) for r in targets}):
            meta = images[image_id]
            member = str(meta["file_name"]).replace("\\", "/")
            image = cv2.imdecode(np.frombuffer(archive.read(member), np.uint8), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            raw = np.load(RUN / "raw" / f"image_{image_id}_candidates.npz")
            boxes = torch.from_numpy(raw["boxes"])
            coeff = torch.from_numpy(raw["mask_coefficients"])
            proto = torch.from_numpy(raw["prototype"])
            target_rows = [r for r in targets if int(r["image_id"]) == image_id]
            for target in target_rows:
                aid = target["annotation_id"]
                candidates = trace.get(aid, [])
                raw_ids = sorted({int(r["candidate_id"]) for r in candidates if r["associated"] == "True"})
                top_ids = sorted({int(r["candidate_id"]) for r in candidates if r["associated"] == "True" and r["pass_conf"] == "True" and r["selected_top300"] == "True"})
                gt = gt_mask(anns[int(aid)], h, w)

                def make_masks(ids: list[int]) -> dict[int, np.ndarray]:
                    from ultralytics.utils import ops

                    result = {}
                    for start in range(0, len(ids), 64):
                        part = ids[start : start + 64]
                        masks = ops.process_mask_native(proto, coeff[part], boxes[part], (h, w)).numpy().astype(bool)
                        result.update({c: masks[i] for i, c in enumerate(part)})
                    return result

                top_masks = make_masks(top_ids)
                raw_masks = make_masks(raw_ids) if raw_ids != top_ids else top_masks
                best_raw_iou, best_raw_cov = max((scores(m, gt) for m in raw_masks.values()), default=(0.0, 0.0), key=lambda x: (x[0], x[1]))
                best_top_iou, best_top_cov = max((scores(m, gt) for m in top_masks.values()), default=(0.0, 0.0), key=lambda x: (x[0], x[1]))
                best_pair_iou, best_pair_cov, best_pair_ids = 0.0, 0.0, ""
                for a, b in combinations(top_ids, 2):
                    iou, cov = scores(np.logical_or(top_masks[a], top_masks[b]), gt)
                    if (iou, cov) > (best_pair_iou, best_pair_cov):
                        best_pair_iou, best_pair_cov, best_pair_ids = iou, cov, f"{a}+{b}"
                all_union = np.zeros_like(gt)
                for mask in top_masks.values():
                    all_union |= mask
                all_iou, all_cov = scores(all_union, gt)
                rows.append({"image_id": image_id, "annotation_id": int(aid), "state": target["state"], "raw_candidate_count": len(raw_ids), "top300_candidate_count": len(top_ids), "best_raw_iou": best_raw_iou, "best_raw_coverage": best_raw_cov, "best_top300_iou": best_top_iou, "best_top300_coverage": best_top_cov, "best_top300_pair_iou": best_pair_iou, "best_top300_pair_coverage": best_pair_cov, "best_pair_ids": best_pair_ids, "all_top300_union_iou": all_iou, "all_top300_union_coverage": all_cov})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "union_oracle_gt.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for state in ("O", "M", "X", "ALL"):
        group = rows if state == "ALL" else [r for r in rows if r["state"] == state]
        summary[state] = {"n": len(group), "top300_iou_ge_0.50": sum(r["best_top300_iou"] >= 0.50 for r in group), "pair_iou_ge_0.50": sum(r["best_top300_pair_iou"] >= 0.50 for r in group), "pair_coverage_ge_0.50": sum(r["best_top300_pair_coverage"] >= 0.50 for r in group), "union_iou_ge_0.50": sum(r["all_top300_union_iou"] >= 0.50 for r in group), "median_pair_iou": float(np.median([r["best_top300_pair_iou"] for r in group])) if group else 0.0, "median_pair_coverage": float(np.median([r["best_top300_pair_coverage"] for r in group])) if group else 0.0}
    (OUTPUT / "union_oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# YOLO26 exact-runtime multi-candidate union oracle", "", "Reconstructed-sample, retained-raw-output analysis; no model forward or training.", "", "| State | N | Top-300 best single IoU >= .50 | Best pair IoU >= .50 | Best pair coverage >= .50 | All Top-300 union IoU >= .50 | Median pair IoU | Median pair coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for state in ("O", "M", "X", "ALL"):
        s = summary[state]
        report.append(f"| {state} | {s['n']} | {s['top300_iou_ge_0.50']} | {s['pair_iou_ge_0.50']} | {s['pair_coverage_ge_0.50']} | {s['union_iou_ge_0.50']} | {s['median_pair_iou']:.3f} | {s['median_pair_coverage']:.3f} |")
    report += ["", "The pair and union columns are oracle upper bounds over associated Top-300 candidates, not a deployable postprocessor. Sampling is reconstructed and the original `SPLIT_TYPE` labels are unavailable."]
    (OUTPUT / "union_oracle_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"targets": len(rows), "output": str(OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
