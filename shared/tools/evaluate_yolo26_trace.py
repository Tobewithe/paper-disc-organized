"""Export one completed YOLO26 diagnostic trace and run public COCO mask evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("public_coco_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def trace_version_metadata(config: dict[str, Any]) -> dict[str, str]:
    """Keep the actual trace runtime separate from checkpoint provenance."""
    return {
        "trace_inference_ultralytics_version": config["model_meta"]["ultralytics_version"],
        "checkpoint_original_legacy_ultralytics_version": config["config_source"]["provenance"]["original_runtime"]["ultralytics"],
    }


def export_predictions(trace: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    config = load_json(trace / "inference_config.json")
    status = load_json(trace / "run_status.json")
    summary = load_json(trace / "summary.json")
    if status.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError(f"Trace is not completed: {trace}")

    manifest = Path(config["manifest"])
    coco = load_json(manifest)
    manifest_ids = {image["id"] for image in coco["images"]}
    if len(manifest_ids) != len(coco["images"]):
        raise ValueError("Manifest has duplicate image IDs")
    source_class_id = config["config"]["source_class_id"]
    output_category_id = config["config"]["output_category_id"]
    seen_ids: set[int] = set()
    predictions: list[dict[str, Any]] = []

    with (trace / "diagnostic_cache.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id not in manifest_ids or image_id in seen_ids:
                raise ValueError(f"Invalid or duplicate trace image ID at line {line_number}: {image_id}")
            seen_ids.add(image_id)
            for prediction in row["final_predictions"]:
                if prediction["class"] != source_class_id:
                    raise ValueError(f"Unexpected source class for image {image_id}")
                score = float(prediction["score"])
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"Invalid score for image {image_id}")
                rle = prediction["mask_rle"]
                if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
                    raise ValueError(f"Invalid mask RLE for image {image_id}")
                # Deliberately omit bbox: COCO derives area from this segmentation RLE.
                predictions.append({"image_id": image_id, "category_id": output_category_id, "score": score, "segmentation": rle})

    if seen_ids != manifest_ids:
        raise ValueError(f"Trace image coverage differs from manifest: {len(seen_ids)} != {len(manifest_ids)}")
    expected_predictions = summary.get("final_prediction_records")
    if expected_predictions != len(predictions):
        raise ValueError(f"Prediction count differs from trace summary: {len(predictions)} != {expected_predictions}")

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "coco_predictions.json"
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=True) + "\n", encoding="utf-8")
    category_counts = Counter(annotation["category_id"] for annotation in coco["annotations"])
    validation = {
        "trace": str(trace),
        "trace_status": status["status"],
        "manifest": str(manifest),
        "manifest_images": len(manifest_ids),
        "trace_images": len(seen_ids),
        "exact_manifest_id_coverage": True,
        "predictions": len(predictions),
        "trace_summary_predictions": expected_predictions,
        "source_class_id": source_class_id,
        "output_category_id": output_category_id,
        "all_scores_in_unit_interval": True,
        "coco_prediction_has_bbox": False,
        **trace_version_metadata(config),
        "gt_taxonomy": [
            {**category, "annotation_count": category_counts[category["id"]]}
            for category in coco["categories"]
        ],
    }
    (output / "provenance.json").write_text(json.dumps(validation, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return predictions_path, validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-metadata", action="store_true", help="Print trace and checkpoint runtime metadata without writing outputs.")
    args = parser.parse_args()
    if args.check_metadata:
        print(json.dumps(trace_version_metadata(load_json(args.trace / "inference_config.json")), ensure_ascii=True, indent=2))
        return 0
    if args.evaluator is None or args.output is None:
        parser.error("--evaluator and --output are required unless --check-metadata is used")
    predictions_path, validation = export_predictions(args.trace, args.output)
    metrics = load_evaluator(args.evaluator)(Path(validation["manifest"]), predictions_path)
    metrics["coco_stat_names"] = [
        "mask_ap", "mask_ap50", "mask_ap75", "mask_ap_small", "mask_ap_medium", "mask_ap_large",
        "recall_ar1", "recall_ar10", "recall_ar100", "recall_ar_small", "recall_ar_medium", "recall_ar_large",
    ]
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "metrics": metrics}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
