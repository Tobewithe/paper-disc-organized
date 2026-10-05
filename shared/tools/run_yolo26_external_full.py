"""Run the existing PigLife-only checkpoint on all Faro and Bama splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "pigcv_research"
DATA = Path("C:/Dpan/document/model_datasets/datasets")
WEIGHTS = Path("C:/Dpan/document/model_datasets/artifacts/pigcv-task05/remote-runs/task05-train-yolo-full-20260810-02")
INFER = LEGACY / "workflows/inference/run_yolo26seg.py"
CONFIG = LEGACY / "configs/yolo26seg/infer.yaml"
PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_write(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_module(name, path):
    sys.path.insert(0, str(LEGACY / "scripts"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_annotation(ann, image, new_id):
    polygons = ann["segmentation"]
    if not isinstance(polygons, list) or not polygons:
        raise ValueError(f"Missing source polygons: {ann['id']}")
    if any(len(p) < 6 or len(p) % 2 or not all(math.isfinite(v) for v in p) for p in polygons):
        raise ValueError(f"Invalid source polygons: {ann['id']}")
    rle = mask_utils.merge(mask_utils.frPyObjects(polygons, image["height"], image["width"]))
    area = float(mask_utils.area(rle))
    if area <= 0 or ann.get("iscrowd", 0) or ann["category_id"] != 1:
        raise ValueError(f"Unexpected empty/crowd/category annotation: {ann['id']}")
    return {"id": new_id, "source_annotation_id": ann["id"], "image_id": image["id"],
            "category_id": 1, "segmentation": polygons, "bbox": mask_utils.toBbox(rle).tolist(),
            "area": area, "source_area": ann.get("area"), "iscrowd": 0}


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    faro = DATA / "FaroPigSeg"
    bama = next(DATA.glob("*/BamaPig2D/BamaPig2D/annotations/train_pig_cocostyle.json")).parent.parent
    inventory, specs, hashes, duplicates = [], [], {}, defaultdict(list)
    expected = {"FaroPigSeg": {"train": 1039, "val": 319, "test": 160},
                "BamaPig2D": {"train": 3008, "eval": 332}}
    for dataset, image_root in (("FaroPigSeg", faro), ("BamaPig2D", bama / "images")):
        full = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "pig"}]}
        seen_files = set()
        for split, expected_images in expected[dataset].items():
            images, annotations = [], []
            if dataset == "FaroPigSeg":
                local_annotation_id = 0
                paths = sorted((faro / split / "images").iterdir())
                paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                if {p.stem for p in paths} != {p.stem for p in (faro / split / "labels").glob("*.txt")}:
                    raise ValueError(f"Image/label pairing mismatch: {split}")
                for source_id, path in enumerate(paths, 1):
                    with Image.open(path) as opened:
                        width, height = opened.size
                    image = {"id": len(full["images"]) + len(images) + 1, "source_image_id": source_id,
                             "file_name": path.relative_to(image_root).as_posix(), "width": width,
                             "height": height, "source_split": split}
                    images.append(image)
                    label = faro / split / "labels" / f"{path.stem}.txt"
                    hashes[str(label)] = sha(label)
                    for line in label.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        values = [float(x) for x in line.split()]
                        if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0:
                            raise ValueError(f"Invalid YOLO polygon: {label}")
                        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values[1:]):
                            raise ValueError(f"Out-of-range polygon: {label}")
                        polygon = [v * (width if i % 2 == 0 else height) for i, v in enumerate(values[1:])]
                        local_annotation_id += 1
                        ann = {"id": local_annotation_id, "category_id": 1, "segmentation": [polygon]}
                        annotations.append(normalize_annotation(ann, image, len(full["annotations"]) + len(annotations) + 1))
            else:
                source_path = bama / "annotations" / f"{split}_pig_cocostyle.json"
                hashes[str(source_path)] = sha(source_path)
                source = read(source_path)
                if len({x["id"] for x in source["images"]}) != len(source["images"]):
                    raise ValueError("Duplicate source image IDs")
                if len({x["id"] for x in source["annotations"]}) != len(source["annotations"]):
                    raise ValueError("Duplicate source annotation IDs")
                image_map = {}
                for original in source["images"]:
                    image = {**original, "source_image_id": original["id"], "source_split": split,
                             "id": len(full["images"]) + len(images) + 1}
                    image_map[original["id"]] = image
                    images.append(image)
                for ann in source["annotations"]:
                    annotations.append(normalize_annotation(ann, image_map[ann["image_id"]], len(full["annotations"]) + len(annotations) + 1))
            if len(images) != expected_images:
                raise ValueError(f"Unexpected image count: {dataset}/{split}: {len(images)}")
            for image in images:
                path = image_root / image["file_name"]
                if image["file_name"] in seen_files:
                    raise ValueError(f"Overlapping split paths: {path}")
                seen_files.add(image["file_name"])
                with Image.open(path) as opened:
                    if opened.size != (image["width"], image["height"]):
                        raise ValueError(f"Image dimensions differ: {path}")
                    opened.verify()
                digest = sha(path)
                hashes[str(path)] = digest
                duplicates[digest].append(str(path))
                inventory.append({"dataset": dataset, "split": split, **image, "sha256": digest})
            manifest = output / "manifests" / f"{dataset}_{split}.json"
            write(manifest, {"images": images, "annotations": annotations, "categories": full["categories"]})
            hashes[str(manifest)] = sha(manifest)
            specs.append({"dataset": dataset, "split": split, "manifest": str(manifest),
                          "image_root": str(image_root), "images": len(images), "gt": len(annotations)})
            full["images"].extend(images)
            full["annotations"].extend(annotations)
            print(json.dumps({"event": "prepared", **specs[-1]}), flush=True)
        full_path = output / "manifests" / f"{dataset}_all.json"
        write(full_path, full)
        hashes[str(full_path)] = sha(full_path)
    for path in (WEIGHTS / "yolo26_task05_final.pt", WEIGHTS / "training_config.json", WEIGHTS / "config.json",
                 WEIGHTS / "piglife_task05.yaml", CONFIG, INFER,
                 LEGACY / "models/yolo26seg/inferencer.py", LEGACY / "models/yolo26seg/loader.py",
                 LEGACY / "models/yolo26seg/types.py", LEGACY / "scripts/analyze_yolo26_diagnostic_cache_full.py",
                 LEGACY / "scripts/classify_public_test_instance_failures.py", Path(__file__)):
        hashes[str(path)] = sha(path)
    write(output / "plan.json", {"splits": specs, "total_images": len(inventory), "inference_config": str(CONFIG),
        "checkpoint": str(WEIGHTS / "yolo26_task05_final.pt"), "training_scope": "PigLife only; user confirmed and training archives agree",
        "gt_normalization": "Positive unique evaluation IDs; original IDs retained; unchanged polygons; raster mask area and bbox",
        "training_or_tuning": False, "raw_candidate_export": False, "all_original_splits_are_inference_only": True})
    write(output / "input_sha256.json", hashes)
    write(output / "exact_duplicate_files.json", [paths for paths in duplicates.values() if len(paths) > 1])
    csv_write(output / "image_inventory.csv", inventory)


def witness(output):
    import torch
    import ultralytics
    plan = read(output / "plan.json")
    if torch.__version__ != "2.9.1+cu128" or ultralytics.__version__ != "8.4.100":
        raise RuntimeError("Runtime differs from frozen diagnostic environment")
    torch.manual_seed(20260809)
    x = torch.randn(16, 16, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()) or float(y.abs().sum()) == 0:
        raise RuntimeError("GPU kernel witness failed")
    sys.path.insert(0, str(LEGACY))
    from models.yolo26seg.inferencer import load_inference_config, predict_image, final_outputs_from_result, set_seed
    from models.yolo26seg.loader import load_model
    config, _ = load_inference_config(CONFIG)
    if str(config.checkpoint) != plan["checkpoint"] and config.checkpoint.resolve() != Path(plan["checkpoint"]).resolve():
        raise RuntimeError("Checkpoint mismatch")
    set_seed(config.seed)
    model = load_model(config.checkpoint, config.device)
    receipts = []
    for spec in (plan["splits"][0], plan["splits"][3]):
        image = read(spec["manifest"])["images"][0]
        result = predict_image(model, Path(spec["image_root"]) / image["file_name"], config)
        predictions, _ = final_outputs_from_result(result, image["id"], image["width"], image["height"], config)
        if not predictions:
            raise RuntimeError(f"Empty smoke prediction: {spec['dataset']}")
        receipts.append({"dataset": spec["dataset"], "image_id": image["id"], "predictions": len(predictions)})
    write(output / "environment_witness.json", {"status": "PASS", "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(), "kernel_shape": list(y.shape), "model_forward": receipts})
    print("WITNESS_PASS " + json.dumps(receipts), flush=True)


def classify(fixed, coco, cache):
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    rows = []
    for image in coco["images"]:
        anns = annotations[image["id"]]
        preds = cache[image["id"]]["predictions"]
        gt_rles = [fixed.normalize_gt(a, image["height"], image["width"]) for a in anns]
        iou, coverage, purity = fixed.matrices(gt_rles, [fixed.rle(p["mask_rle"]) for p in preds])
        labels = {}
        for gs, ps in fixed.graph_components((iou >= .50) | (coverage >= .50)):
            if len(gs) == len(ps) == 1:
                g, p = gs[0], ps[0]
                label = "C" if coverage[g, p] >= .75 and purity[g, p] >= .75 else "I" if coverage[g, p] < .75 and purity[g, p] >= .75 else "L" if coverage[g, p] >= .75 else "S"
            else:
                label = "O" if len(gs) == 1 else "M" if len(ps) == 1 else "X"
            labels.update({g: label for g in gs})
        for g, ann in enumerate(anns):
            rows.append({"image_id": image["id"], "annotation_id": ann["id"], "source_image_id": image["source_image_id"],
                         "source_annotation_id": ann["source_annotation_id"], "file_name": image["file_name"],
                         "primary_class": labels.get(g, "MISS")})
    return rows


def evaluate(output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    plan = read(output / "plan.json")
    fixed = load_module("fixed_external_classifier", LEGACY / "scripts/analyze_yolo26_diagnostic_cache_full.py")
    summaries, all_preds, all_gt = [], defaultdict(list), defaultdict(list)
    historical = {}
    old_csv = LEGACY / "artifacts/analysis/yolo26seg_diagnostic_full_single_forward_20260901_final02/gt_analysis.csv"
    with old_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            historical[(row["dataset"], int(row["image_id"]), int(row["annotation_id"]))] = row["primary_class"]
    comparisons = []
    for spec in plan["splits"]:
        run = output / "inference" / f"{spec['dataset']}_{spec['split']}"
        if read(run / "run_status.json")["status"] != "completed":
            raise RuntimeError(f"Incomplete inference: {run}")
        coco = read(spec["manifest"])
        with (run / "inference_cache.jsonl").open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        cache = {row["image_id"]: row for row in records}
        if len(cache) != len(records) or set(cache) != {i["id"] for i in coco["images"]}:
            raise RuntimeError("Incomplete or duplicate prediction image coverage")
        predictions = read(run / "predictions.json")
        if len(predictions) != sum(len(r["predictions"]) for r in records):
            raise RuntimeError("Cache/export prediction accounting mismatch")
        images = {i["id"]: i for i in coco["images"]}
        for pred in predictions:
            image = images[pred["image_id"]]
            if pred["category_id"] != 1 or not .05 < pred["score"] <= 1 or pred["segmentation"]["size"] != [image["height"], image["width"]]:
                raise RuntimeError("Invalid prediction category, score or dimensions")
        segm = [{k: p[k] for k in ("image_id", "category_id", "score", "segmentation")} for p in predictions]
        gt_rows = classify(fixed, coco, cache)
        csv_write(run / "gt_analysis.csv", gt_rows)
        all_preds[spec["dataset"]].extend(segm)
        all_gt[spec["dataset"]].extend(gt_rows)
        if spec["split"] in ("test", "eval"):
            old_name = f"{spec['dataset']}_{spec['split']}"
            for row in gt_rows:
                key = (old_name, row["source_image_id"], row["source_annotation_id"])
                comparisons.append({"dataset": old_name, **row, "previous_class": historical.get(key),
                                    "same": historical.get(key) == row["primary_class"]})
        write(run / "segmentation_predictions.json", segm)
    csv_write(output / "historical_subset_comparison.csv", comparisons)
    for dataset in all_preds:
        write(output / f"{dataset}_all_predictions.json", all_preds[dataset])
        csv_write(output / f"{dataset}_all_gt_analysis.csv", all_gt[dataset])
    for dataset in all_preds:
        selections = [s for s in plan["splits"] if s["dataset"] == dataset] + [{"dataset": dataset, "split": "all", "manifest": str(output / "manifests" / f"{dataset}_all.json")}]
        for spec in selections:
            if spec["split"] == "all":
                preds, rows = all_preds[dataset], all_gt[dataset]
            else:
                run = output / "inference" / f"{dataset}_{spec['split']}"
                preds = read(run / "segmentation_predictions.json")
                with (run / "gt_analysis.csv").open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            gt = COCO(spec["manifest"])
            if preds:
                dt = gt.loadRes(preds)
            else:
                dt = COCO()
                dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
                dt.createIndex()
            evaluator = COCOeval(gt, dt, "segm")
            evaluator.params.imgIds = sorted(gt.getImgIds())
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = [float(x) for x in evaluator.stats]
            if not all(math.isfinite(x) for x in stats):
                raise RuntimeError("Non-finite COCO metrics")
            counts = Counter(r["primary_class"] for r in rows)
            summary = {"dataset": dataset, "split": spec["split"], "images": len(gt.imgs), "gt": len(rows),
                       "predictions": len(preds), "mask_ap": stats[0], "mask_ap50": stats[1], "mask_ap75": stats[2],
                       "mask_ar100": stats[8], "failed_gt": len(rows) - counts["C"],
                       "failure_rate": (len(rows) - counts["C"]) / len(rows), **{k: counts[k] for k in PRIMARY}}
            summaries.append(summary)
            write(output / "metrics" / f"{dataset}_{spec['split']}.json", {**summary, "coco_stats": stats})
            print(json.dumps({"event": "evaluated", **summary}), flush=True)
    hashes = read(output / "input_sha256.json")
    changed = [p for p, digest in hashes.items() if sha(p) != digest]
    if changed:
        raise RuntimeError(f"Inputs changed during execution: {changed}")
    csv_write(output / "summary.csv", summaries)
    write(output / "summary.json", {"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_images": plan["total_images"], "total_gt": sum(len(r) for r in all_gt.values()), "results": summaries,
        "input_hashes_verified": len(hashes), "historical_gt_compared": len(comparisons),
        "historical_gt_changed": sum(not x["same"] for x in comparisons),
        "protocol": "PigLife-finetuned checkpoint; fixed 1024/rect=false/conf=.05/Top300/native masks; COCO maxDets100",
        "scope": "All target splits are inference-only; prior diagnostic exposure retained; not a trained intervention or untouched confirmation set"})


def run(output):
    plan = read(output / "plan.json")
    for path, digest in read(output / "input_sha256.json").items():
        if sha(path) != digest:
            raise RuntimeError(f"Pre-run input changed: {path}")
    for spec in plan["splits"]:
        target = output / "inference" / f"{spec['dataset']}_{spec['split']}"
        if (target / "run_status.json").exists() and read(target / "run_status.json").get("status") == "completed":
            print(f"RESUME completed split {target.name}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(INFER), "--config", str(CONFIG), "--manifest", spec["manifest"],
               "--image-root", spec["image_root"], "--output-dir", str(target)]
        with (output / f"{target.name}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError(f"Inference failed: {target.name}")
    evaluate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "witness", "run", "evaluate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    {"prepare": prepare, "witness": witness, "run": run, "evaluate": evaluate}[args.mode](output)


if __name__ == "__main__":
    main()
