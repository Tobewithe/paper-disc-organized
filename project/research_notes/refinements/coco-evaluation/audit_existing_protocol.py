"""Read-only audit of existing COCO inputs; writes only into this directory.

No inference, training, credentials, or changes to published experiment files.
Run with the conda pytorch Python interpreter.
"""
import ast
import contextlib
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def ici_bin(value):
    if value == 0:
        return "I0_zero"
    if value <= 0.5:
        return "I1_(0,0.5]"
    if value <= 1:
        return "I2_(0.5,1]"
    return "I3_>1"


def summarize(rows):
    total = len(rows)
    return {
        "instances": total,
        "same_class_ici_bins": dict(Counter(r["ici_bin"] for r in rows)),
        "ici_le_0_5_count": sum(r["ici_same"] <= 0.5 for r in rows),
        "ici_le_0_5_fraction": sum(r["ici_same"] <= 0.5 for r in rows) / total if total else None,
        "tiny_bbox_lt100": sum(r["bbox_area"] < 100 for r in rows),
        "mask_size_counts": dict(Counter(r["mask_size"] for r in rows)),
        "category_counts": dict(Counter(r["category_id"] for r in rows)),
    }


def check_bootstrap():
    source = ROOT / "gemini/scripts/run_fast_bootstrap.py"
    parsed = ast.parse(source.read_text(encoding="utf-8-sig"))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name in {"preindex_eval", "fast_accumulate_preindexed"}]
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {
            "info": {}, "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "detected"}, {"id": 2, "name": "missed"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10], "area": 100, "iscrowd": 0},
            ],
        }
        gt.createIndex()
        dt = gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .9}])
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        pre, rec, nc = namespace["preindex_eval"](ev)
        fast = namespace["fast_accumulate_preindexed"](pre, rec, nc, np.array([0]))
    assert abs(ev.stats[0] - .5) < 1e-10
    return {"fixture": "Two GT categories, one perfect detection, the other entirely missed",
            "official_ap": float(ev.stats[0]), "existing_fast_ap": fast[0],
            "bug_reproduced": bool(abs(fast[0] - ev.stats[0]) > 1e-10),
            "source_sha256": sha256(source),
            "scope": "Synthetic correctness failure; effect on existing COCO intervals not quantified."}


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = ROOT / "datasets/coco/annotations/instances_val2017.json"
    subset_path = ROOT / "gemini/data/coco_dense/coco_dense_val_gt.json"
    quartile_path = ROOT / "gemini/data/coco_dense/coco_dense_quartiles.csv"
    original, subset = read_json(original_path), read_json(subset_path)
    with quartile_path.open(encoding="utf-8-sig", newline="") as handle:
        quartiles = {int(r["image_id"]): r for r in csv.DictReader(handle)}
    dense_ids = {im["id"] for im in subset["images"]}
    orig_anns = {a["id"]: a for a in original["annotations"]}
    expected = {a["id"] for a in original["annotations"] if a["image_id"] in dense_ids}
    actual = {a["id"] for a in subset["annotations"]}
    report = {"created_local": datetime.now().isoformat(), "inputs": {
        str(p.relative_to(ROOT)): sha256(p) for p in [original_path, subset_path, quartile_path]},
        "gt_integrity": {
            "full_images": len(original["images"]), "dense_images": len(dense_ids),
            "annotation_ids_match_original_subset": expected == actual,
            "all_annotation_fields_unchanged": all(orig_anns[a["id"]] == a for a in subset["annotations"]),
            "quartile_membership_matches": set(quartiles) == dense_ids,
            "noncrowd_annotations": sum(not a.get("iscrowd", 0) for a in subset["annotations"]),
            "crowd_annotations": sum(bool(a.get("iscrowd", 0)) for a in subset["annotations"]),
            "multipolygon_noncrowd_instances": sum(not a.get("iscrowd", 0) and isinstance(a.get("segmentation"), list)
                                                    and len(a["segmentation"]) > 1 for a in subset["annotations"]),
        }}
    grouped = defaultdict(list)
    for a in original["annotations"]:
        if not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0:
            grouped[a["image_id"]].append(a)
    rows = []
    mismatch = []
    for iid, anns in grouped.items():
        boxes = np.array([a["bbox"] for a in anns], dtype=float)
        areas = boxes[:, 2] * boxes[:, 3]
        ends = boxes[:, :2] + boxes[:, 2:]
        wh = np.maximum(0, np.minimum(ends[:, None], ends[None, :]) - np.maximum(boxes[:, None, :2], boxes[None, :, :2]))
        overlaps = wh.prod(axis=-1)
        np.fill_diagonal(overlaps, 0)
        cats = np.array([a["category_id"] for a in anns])
        same = (overlaps * (cats[:, None] == cats[None, :])).sum(axis=1) / areas
        all_class = overlaps.sum(axis=1) / areas
        if iid in quartiles:
            legacy_max = float(same[areas >= 100].max()) if np.any(areas >= 100) else 0.0
            if abs(legacy_max - float(quartiles[iid]["max_ici"])) > 1e-9:
                mismatch.append(iid)
        for a, area, si, ai in zip(anns, areas, same, all_class):
            rows.append({"image_id": iid, "annotation_id": a["id"], "category_id": a["category_id"],
                         "mask_area": a["area"], "bbox_area": float(area),
                         "mask_size": "small" if a["area"] < 1024 else "medium" if a["area"] < 9216 else "large",
                         "ici_same": float(si), "ici_all_categories": float(ai), "ici_bin": ici_bin(si),
                         "dense_image": iid in dense_ids, "image_quartile": quartiles.get(iid, {}).get("quartile", "outside_dense")})
    report["gt_integrity"]["legacy_image_max_ici_mismatch_ids"] = mismatch
    report["instance_density"] = {"all_val5000": summarize(rows), "dense1576": summarize([r for r in rows if r["dense_image"]])}
    for q in ("Q1", "Q2", "Q3", "Q4", "outside_dense"):
        report["instance_density"][q] = summarize([r for r in rows if r["image_quartile"] == q])
    report["density_definition"] = "Same-category sum of bbox intersections / reference bbox area; noncrowd valid GT. Tiny references retained for instance bins, unlike historical image selection. Bins: zero, (0,.5], (.5,1], >1."
    manifest_path = OUT / f"COCO_INSTANCE_MANIFEST_{stamp}.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["instance_manifest"] = str(manifest_path.relative_to(ROOT))
    report["predictions"] = {}
    for model in ("baseline", "ccl01", "ccl05"):
        path = ROOT / f"gemini/results/predictions_{model}_stage2.json"
        preds = read_json(path)
        per_image = Counter(p["image_id"] for p in preds)
        score_groups = defaultdict(list)
        for p in preds:
            score_groups[(p["image_id"], p["category_id"])].append(p["score"])
        categories = {c["id"] for c in original["categories"]}
        report["predictions"][model] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "detections": len(preds),
            "images_with_detections": len(per_image), "expected_images_without_detections": len(dense_ids - set(per_image)),
            "out_of_subset_detections": sum(p["image_id"] not in dense_ids for p in preds),
            "invalid_category_detections": sum(p["category_id"] not in categories for p in preds),
            "missing_segmentation": sum("segmentation" not in p for p in preds),
            "both_bbox_and_segmentation": sum("bbox" in p and "segmentation" in p for p in preds),
            "images_at_extraction_cap300": sum(n == 300 for n in per_image.values()),
            "within_image_category_repeated_scores": sum(len(scores) - len(set(scores)) for scores in score_groups.values()),
        }
        del preds
    report["bootstrap_regression"] = check_bootstrap()
    report_path = OUT / f"PROTOCOL_AUDIT_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "gt": report["gt_integrity"],
                      "density": {k: {x: v[x] for x in ("instances", "same_class_ici_bins", "ici_le_0_5_fraction")} for k, v in report["instance_density"].items()},
                      "bootstrap": report["bootstrap_regression"], "predictions": report["predictions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
