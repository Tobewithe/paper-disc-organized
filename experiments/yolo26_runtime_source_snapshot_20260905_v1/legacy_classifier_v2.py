"""Full GT, scene, and failure-stage analysis from the immutable YOLO26 diagnostic cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from classify_public_test_instance_failures import matrices, normalize_gt, rle
from prepare_task05_full_public_gt_geometry import compute_rows


DATASETS = (
    {
        "name": "PigLife_public_test",
        "manifest": Path(r"C:\Dpan\document\model_datasets\datasets\piglife\derived\task05_v1\pig_coco_test_task05_v1.json"),
        "cache": "piglife_public_test",
        "images": 426,
        "gt": 4474,
    },
    {
        "name": "FaroPigSeg_test",
        "manifest": Path(r"C:\Users\Public\pigcv-task05\external-inference\faropigseg_v1\test\manifest.json"),
        "cache": "faropigseg_test",
        "images": 160,
        "gt": 1752,
    },
    {
        "name": "BamaPig2D_eval",
        "manifest": Path(r"C:\Users\Public\pigcv-task05\external-inference\bamapig2d_v1\eval\manifest.json"),
        "cache": "bamapig2d_eval",
        "images": 332,
        "gt": 1148,
    },
)
NPZ_KEYS = {
    "source_candidate_id", "boxes_xyxy", "scores", "mask_coefficients", "prototype",
    "top_indices", "top_scores", "global_rank", "feature_level", "grid_x", "grid_y", "stride",
}
PRIMARY = ("C", "I", "L", "S", "O", "M", "X", "MISS")
STAGES = ("RAW_BOX", "TOPK_BOX", "CONF_BOX", "MASK", "FINAL_SET", "UNRESOLVED")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def reuse_deterministic_results(source: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required = ("gt_analysis.csv", "component_analysis.csv", "prediction_analysis.csv", "scene_analysis.csv", "basic_summary.csv", "component_summary.csv", "geometry_validation.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Reuse source missing required files: {missing}")
    for name in required:
        shutil.copy2(source / name, output / name)
    gt_rows = read_csv_rows(source / "gt_analysis.csv")
    for row in gt_rows:
        for key in ("image_id", "annotation_id", "component_gt_count", "component_prediction_count", "core_degree", "weak_degree", "total_failure", "geometry_failure", "relation_failure", "miss", "near_neighbor_count", "contact_count", "image_gt_count"):
            row[key] = int(row[key])
        for key in ("best_iou", "best_gt_coverage", "best_prediction_purity", "nearest_boundary_distance_norm", "near_other_ratio", "gt_area", "border_distance"):
            row[key] = float(row[key]) if row[key] else None
        row["component_id"] = row["component_id"] or None
    component_rows = read_csv_rows(source / "component_analysis.csv")
    prediction_rows = read_csv_rows(source / "prediction_analysis.csv")
    def numeric_rows(name: str) -> list[dict[str, Any]]:
        rows = read_csv_rows(source / name)
        for row in rows:
            for key, value in list(row.items()):
                if key != "dataset" and key != "scene_label" and key != "component_type" and value not in {"", None}:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
        return rows
    return gt_rows, component_rows, prediction_rows, numeric_rows("basic_summary.csv"), numeric_rows("scene_analysis.csv"), read_json(source / "geometry_validation.json")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def graph_components(edges: np.ndarray) -> list[tuple[list[int], list[int]]]:
    ng, npred = edges.shape
    seen_g: set[int] = set()
    seen_p: set[int] = set()
    out: list[tuple[list[int], list[int]]] = []
    for start in range(ng):
        if start in seen_g or not edges[start].any():
            continue
        gs: set[int] = set()
        ps: set[int] = set()
        queue: deque[tuple[str, int]] = deque([("g", start)])
        while queue:
            kind, index = queue.popleft()
            if kind == "g":
                if index in seen_g:
                    continue
                seen_g.add(index); gs.add(index)
                queue.extend(("p", int(x)) for x in np.flatnonzero(edges[index]))
            else:
                if index in seen_p:
                    continue
                seen_p.add(index); ps.add(index)
                queue.extend(("g", int(x)) for x in np.flatnonzero(edges[:, index]))
        out.append((sorted(gs), sorted(ps)))
    return out


def audit_cache(cache_root: Path, output: Path) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    from pycocotools import mask as mask_utils

    all_records: dict[str, dict[int, dict[str, Any]]] = {}
    report: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(), "datasets": {}, "issues": []}
    totals = Counter()
    for spec in DATASETS:
        name = spec["name"]
        manifest = read_json(spec["manifest"])
        image_by_id = {int(x["id"]): x for x in manifest["images"]}
        cache_dir = cache_root / spec["cache"]
        config = read_json(cache_dir / "inference_config.json")
        rows = load_jsonl(cache_dir / "diagnostic_cache.jsonl")
        records: dict[int, dict[str, Any]] = {}
        issues: list[dict[str, Any]] = []
        score_warnings: list[dict[str, Any]] = []
        same_forward_mapping = True
        final_count = 0
        raw_count = 0
        for row_number, row in enumerate(rows, 1):
            image_id = int(row["image_id"])
            image = image_by_id.get(image_id)
            if image is None or image_id in records:
                issues.append({"row": row_number, "image_id": image_id, "issue": "unknown_or_duplicate_image_id"})
                continue
            records[image_id] = row
            raw_path = cache_dir / row["raw_cache"]
            if not raw_path.is_file():
                issues.append({"image_id": image_id, "issue": "missing_raw_npz", "path": str(raw_path)})
                continue
            with np.load(raw_path, allow_pickle=False) as raw:
                missing = sorted(NPZ_KEYS - set(raw.files))
                n = int(len(raw["source_candidate_id"]))
                top = raw["top_indices"].astype(np.int64)
                ranks = raw["global_rank"].astype(np.int64)
                if missing:
                    issues.append({"image_id": image_id, "issue": "missing_npz_keys", "keys": missing})
                if n != int(row["raw_candidate_count"]) or n != 21504:
                    issues.append({"image_id": image_id, "issue": "raw_count_mismatch", "npz": n, "jsonl": row["raw_candidate_count"]})
                if not np.array_equal(raw["source_candidate_id"], np.arange(n)):
                    issues.append({"image_id": image_id, "issue": "candidate_ids_not_contiguous"})
                if len(top) != int(row["top_k_count"]) or len(np.unique(top)) != len(top) or np.any((top < 0) | (top >= n)):
                    issues.append({"image_id": image_id, "issue": "invalid_topk"})
                if len(ranks) != n or int(ranks.min()) != 1 or int(ranks.max()) != n or len(np.unique(ranks)) != n:
                    issues.append({"image_id": image_id, "issue": "invalid_global_rank"})
                predictions = row["final_predictions"]
                pred_ids = [int(p["pred_id"]) for p in predictions]
                mappings = {int(x["pred_id"]): x for x in row["final_pred_to_source_candidate"]}
                if pred_ids != list(range(1, len(predictions) + 1)) or len(set(pred_ids)) != len(pred_ids):
                    issues.append({"image_id": image_id, "issue": "invalid_pred_ids"})
                if set(mappings) != set(pred_ids) or len(mappings) != len(row["final_pred_to_source_candidate"]):
                    issues.append({"image_id": image_id, "issue": "mapping_pred_id_mismatch"})
                mapped_sources: list[int] = []
                h, w = int(image["height"]), int(image["width"])
                for pred in predictions:
                    pid = int(pred["pred_id"])
                    mapping = mappings.get(pid)
                    if mapping is None:
                        continue
                    same_forward_mapping &= "top_k_position" in mapping and "score_delta" in mapping
                    source = int(mapping["source_candidate_id"])
                    mapped_sources.append(source)
                    if not 0 <= source < n or float(mapping["box_iou"]) < 0.99:
                        issues.append({"image_id": image_id, "pred_id": pid, "issue": "invalid_source_mapping", "mapping": mapping})
                    else:
                        score_delta = abs(float(raw["scores"][source]) - float(pred["score"]))
                        if score_delta > 2e-5:
                            score_warnings.append({"dataset": name, "image_id": image_id, "pred_id": pid, "issue": "separate_forward_mapped_score_mismatch", "absolute_score_delta": score_delta, "box_iou": float(mapping["box_iou"])})
                    box = np.asarray(pred["box_xyxy"], dtype=float)
                    if box.shape != (4,) or not np.isfinite(box).all() or box[0] < -1e-3 or box[1] < -1e-3 or box[2] > w + 1e-3 or box[3] > h + 1e-3 or box[2] <= box[0] or box[3] <= box[1]:
                        issues.append({"image_id": image_id, "pred_id": pid, "issue": "box_outside_original_coordinates", "box": box.tolist()})
                    encoded = rle(pred["mask_rle"])
                    if list(encoded["size"]) != [h, w]:
                        issues.append({"image_id": image_id, "pred_id": pid, "issue": "mask_size_mismatch", "size": encoded["size"]})
                    else:
                        decoded = mask_utils.decode(encoded)
                        if decoded.shape != (h, w) or not decoded.any():
                            issues.append({"image_id": image_id, "pred_id": pid, "issue": "mask_decode_or_empty"})
                if len(mapped_sources) != len(set(mapped_sources)):
                    issues.append({"image_id": image_id, "issue": "duplicate_final_source_candidate"})
                raw_count += n
                final_count += len(predictions)
        missing_ids = sorted(set(image_by_id) - set(records))
        extra_ids = sorted(set(records) - set(image_by_id))
        if missing_ids:
            issues.append({"issue": "missing_image_ids", "ids": missing_ids})
        if extra_ids:
            issues.append({"issue": "extra_image_ids", "ids": extra_ids})
        if len(rows) != int(spec["images"]):
            issues.append({"issue": "cache_row_count", "expected": spec["images"], "actual": len(rows)})
        model_meta = config.get("model_meta", {})
        audit = model_meta.get("checkpoint_key_audit", {})
        if audit.get("missing_keys") or audit.get("unexpected_keys"):
            issues.append({"issue": "checkpoint_key_audit", "value": audit})
        report["datasets"][name] = {
            "manifest_images": len(image_by_id), "manifest_gt": len(manifest["annotations"]),
            "cache_rows": len(rows), "raw_cache_files": len(list((cache_dir / "raw_candidates").glob("*.npz"))),
            "raw_candidates": raw_count, "final_predictions": final_count,
            "run_level_model_meta_present": bool(model_meta), "per_image_preprocess_meta_present": all(bool(x.get("preprocess_meta")) for x in rows),
            "issues": issues, "mapping_score_warning_count": len(score_warnings),
            "same_forward_mapping": same_forward_mapping,
        }
        report["issues"].extend({"dataset": name, **issue} for issue in issues)
        report.setdefault("mapping_score_warnings", []).extend(score_warnings)
        totals.update(images=len(rows), gt=len(manifest["annotations"]), raw=raw_count, predictions=final_count)
        all_records[name] = records
    report["totals"] = dict(totals)
    report["passed"] = not report["issues"] and totals["images"] == 918 and totals["gt"] == 7374
    deltas = [float(x["absolute_score_delta"]) for x in report.get("mapping_score_warnings", [])]
    same_forward = all(bool(x.get("same_forward_mapping")) for x in report["datasets"].values())
    report["mapping_limitation"] = {
        "cause": ("none: final predictions and raw tensors came from one hooked forward" if same_forward else "final predictions and cached raw tensors were collected by separate forward calls; mapping is box-IoU based"),
        "same_forward_mapping": same_forward,
        "warning_count": len(deltas),
        "absolute_score_delta_quantiles": ({str(q): float(np.quantile(deltas, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)} if deltas else {}),
        "policy": "do not modify cache; accept a stage only when the raw branch reproduces the formal primary class, otherwise UNRESOLVED",
    }
    mapping_warning_rows = report.pop("mapping_score_warnings", [])
    write_json(output / "cache_integrity.json", report)
    if mapping_warning_rows:
        write_csv(output / "cache_mapping_score_warnings.csv", mapping_warning_rows)
    if not report["passed"]:
        write_csv(output / "cache_integrity_issues.csv", report["issues"])
        raise RuntimeError(f"Diagnostic cache audit failed with {len(report['issues'])} issue(s); see cache_integrity.json")
    return report, all_records


def classify_all(cache_root: Path, records_by_dataset: dict[str, dict[int, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from pycocotools import mask as mask_utils

    gt_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for spec in DATASETS:
        dataset = spec["name"]
        manifest = read_json(spec["manifest"])
        images = {int(x["id"]): x for x in manifest["images"]}
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in manifest["annotations"]:
            if int(ann.get("category_id", 1)) == 1 and not ann.get("iscrowd", 0):
                anns_by_image[int(ann["image_id"])].append(ann)
        for image_id in sorted(images):
            image = images[image_id]
            anns = sorted(anns_by_image[image_id], key=lambda x: int(x["id"]))
            cache_record = records_by_dataset[dataset][image_id]
            preds = cache_record["final_predictions"]
            mappings = {int(x["pred_id"]): x for x in cache_record["final_pred_to_source_candidate"]}
            gt_rles = [normalize_gt(a, int(image["height"]), int(image["width"])) for a in anns]
            pred_rles = [rle(p["mask_rle"]) for p in preds]
            iou, coverage, purity = matrices(gt_rles, pred_rles)
            core = (iou >= 0.50) | (coverage >= 0.50)
            weak = (~core) & (purity >= 0.50) & (coverage >= 0.10)
            comps = graph_components(core)
            gt_component: dict[int, tuple[str, list[int], list[int], str]] = {}
            for local_id, (gs, ps) in enumerate(comps, 1):
                component_id = f"{dataset}:{image_id}:{local_id}"
                relation = "1:1" if len(gs) == len(ps) == 1 else "O" if len(gs) == 1 else "M" if len(ps) == 1 else "X"
                component_rows.append({
                    "dataset": dataset, "image_id": image_id, "component_id": component_id,
                    "component_type": relation, "gt_count": len(gs), "prediction_count": len(ps),
                    "annotation_ids": json.dumps([int(anns[g]["id"]) for g in gs]),
                    "pred_ids": json.dumps([int(preds[p]["pred_id"]) for p in ps]),
                    "core_edge_count": int(core[np.ix_(gs, ps)].sum()),
                    "weak_edge_count_inside_component": int(weak[np.ix_(gs, ps)].sum()),
                })
                for g in gs:
                    gt_component[g] = (component_id, gs, ps, relation)
            for g, ann in enumerate(anns):
                component = gt_component.get(g)
                if component is None:
                    primary, component_id, gs, ps = "MISS", None, [], []
                else:
                    component_id, gs, ps, relation = component
                    if relation != "1:1":
                        primary = relation
                    else:
                        p = ps[0]
                        primary = "C" if coverage[g, p] >= 0.75 and purity[g, p] >= 0.75 else "I" if coverage[g, p] < 0.75 and purity[g, p] >= 0.75 else "L" if coverage[g, p] >= 0.75 else "S"
                gt_rows.append({
                    "dataset": dataset, "image_id": image_id, "annotation_id": int(ann["id"]), "file_name": image["file_name"],
                    "primary_class": primary, "component_id": component_id,
                    "component_gt_count": len(gs), "component_prediction_count": len(ps),
                    "core_degree": int(core[g].sum()), "weak_degree": int(weak[g].sum()),
                    "best_iou": float(iou[g].max()) if len(preds) else 0.0,
                    "best_gt_coverage": float(coverage[g].max()) if len(preds) else 0.0,
                    "best_prediction_purity": float(purity[g].max()) if len(preds) else 0.0,
                    "total_failure": int(primary != "C"), "geometry_failure": int(primary in {"I", "L", "S"}),
                    "relation_failure": int(primary in {"O", "M", "X"}), "miss": int(primary == "MISS"),
                })
            pred_areas = np.asarray(mask_utils.area(pred_rles), dtype=float) if pred_rles else np.zeros(0)
            for p, pred in enumerate(preds):
                mapping = mappings[int(pred["pred_id"])]
                box = pred["box_xyxy"]
                prediction_rows.append({
                    "dataset": dataset, "image_id": image_id, "pred_id": int(pred["pred_id"]),
                    "source_candidate_id": int(mapping["source_candidate_id"]), "score": float(pred["score"]), "class": int(pred["class"]),
                    "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3], "mask_area": float(pred_areas[p]),
                    "core_gt_degree": int(core[:, p].sum()), "weak_gt_degree": int(weak[:, p].sum()), "is_fp": int(not core[:, p].any()),
                })
    return gt_rows, component_rows, prediction_rows


def add_scene_geometry(gt_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(r["dataset"], int(r["image_id"]), int(r["annotation_id"])): r for r in gt_rows}
    validations: dict[str, Any] = {}
    for spec in DATASETS:
        print(f"[scene] computing GT geometry: {spec['name']}", flush=True)
        manifest = read_json(spec["manifest"])
        geometry, validation = compute_rows(
            manifest, spec["manifest"], category_id=1, scope=spec["name"], file_prefix=None,
            expected_image_count=int(spec["images"]), expected_gt_count=int(spec["gt"]),
            expected_width=None, expected_height=None, near_boundary_delta_px=3.0,
        )
        validations[spec["name"]] = validation
        for geo in geometry:
            row = by_key[(spec["name"], int(geo["image_id"]), int(geo["annotation_id"]))]
            nearest = geo["nearest_boundary_distance_norm"]
            contact = int(geo["contact_count_d0"])
            near_ratio = geo["gt_boundary_near_other_ratio_euclidean_d3px"]
            scene = "Touching" if contact > 0 else "Near/Dense" if (nearest is not None and nearest <= 0.05) or (near_ratio is not None and near_ratio > 0) else "Isolated"
            row.update({
                "scene_label": scene, "occlusion_status": "Unknown",
                "nearest_boundary_distance_norm": nearest,
                "near_neighbor_count": int(geo["near_neighbor_count_euclidean_d3px"]),
                "contact_count": contact, "near_other_ratio": near_ratio,
                "image_gt_count": int(geo["image_gt_count"]), "gt_area": int(geo["gt_mask_area_px"]),
                "border_distance": float(geo["image_border_distance_norm"]),
            })
        print(f"[scene] completed GT geometry: {spec['name']} ({len(geometry)} GT)", flush=True)
    return gt_rows, validations


def aggregate_results(gt_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    basic: list[dict[str, Any]] = []
    scene: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for dataset in [x["name"] for x in DATASETS] + ["ALL"]:
        gs = gt_rows if dataset == "ALL" else [r for r in gt_rows if r["dataset"] == dataset]
        ps = prediction_rows if dataset == "ALL" else [r for r in prediction_rows if r["dataset"] == dataset]
        cs = component_rows if dataset == "ALL" else [r for r in component_rows if r["dataset"] == dataset]
        counts = Counter(r["primary_class"] for r in gs)
        row: dict[str, Any] = {"dataset": dataset, "images": 918 if dataset == "ALL" else next(int(x["images"]) for x in DATASETS if x["name"] == dataset), "gt": len(gs), "predictions": len(ps), "fp": sum(int(r["is_fp"]) for r in ps), "components": len(cs)}
        for label in PRIMARY:
            row[f"{label}_count"] = counts[label]
            row[f"{label}_rate"] = counts[label] / len(gs) if gs else None
        row.update(total_failure_rate=sum(r["total_failure"] for r in gs) / len(gs), geometry_failure_rate=sum(r["geometry_failure"] for r in gs) / len(gs), relation_failure_rate=sum(r["relation_failure"] for r in gs) / len(gs), miss_rate=sum(r["miss"] for r in gs) / len(gs))
        basic.append(row)
        for scene_label in ("Isolated", "Near/Dense", "Touching"):
            sub = [r for r in gs if r["scene_label"] == scene_label]
            n = len(sub)
            sc = Counter(r["primary_class"] for r in sub)
            scene.append({
                "dataset": dataset, "scene_label": scene_label, "n_gt": n,
                "failure_rate": sum(r["total_failure"] for r in sub) / n if n else None,
                "geometry_failure_rate": sum(r["geometry_failure"] for r in sub) / n if n else None,
                "relation_failure_rate": sum(r["relation_failure"] for r in sub) / n if n else None,
                "miss_rate": sc["MISS"] / n if n else None,
                **{f"{label}_rate": sc[label] / n if n else None for label in PRIMARY},
            })
        cc = Counter(r["component_type"] for r in cs)
        components.extend({"dataset": dataset, "component_type": label, "component_count": cc[label], "gt_count": sum(int(r["gt_count"]) for r in cs if r["component_type"] == label), "prediction_count": sum(int(r["prediction_count"]) for r in cs if r["component_type"] == label)} for label in ("1:1", "O", "M", "X"))
    return basic, scene, components


def _quartile_bin(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def controlled_analysis(gt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simple stratified control without a numerical optimizer.

    Each comparison only pools strata containing both groups. Strata are
    dataset + quartiles of log(area), image GT count and border distance.
    """
    out: list[dict[str, Any]] = []
    for dataset in [x["name"] for x in DATASETS] + ["ALL"]:
        selected = gt_rows if dataset == "ALL" else [r for r in gt_rows if r["dataset"] == dataset]
        if not selected:
            continue
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_dataset[row["dataset"]].append(row)
        strata: dict[int, tuple[str, int, int, int]] = {}
        for dataset_name, rows in by_dataset.items():
            area = np.asarray([math.log(float(r["gt_area"])) for r in rows])
            count = np.asarray([float(r["image_gt_count"]) for r in rows])
            border = np.asarray([float(r["border_distance"]) for r in rows])
            edges = [np.quantile(values, [0, .25, .5, .75, 1]) for values in (area, count, border)]
            for row, a, c, b in zip(rows, area, count, border):
                strata[id(row)] = (dataset_name, _quartile_bin(a, edges[0]), _quartile_bin(c, edges[1]), _quartile_bin(b, edges[2]))
        comparisons = [("scene_label", "Isolated", "Near/Dense", selected), ("scene_label", "Isolated", "Touching", selected)]
        for variable in ("contact_count", "nearest_boundary_distance_norm", "near_other_ratio"):
            eligible = [r for r in selected if r.get(variable) is not None and math.isfinite(float(r[variable]))]
            values = np.asarray([float(r[variable]) for r in eligible], dtype=float)
            threshold = float(np.median(values))
            for row in eligible:
                row[f"_control_{variable}"] = "High" if float(row[variable]) > threshold else "Low"
            comparisons.append((f"_control_{variable}", "Low", "High", eligible))
        for outcome in ("total_failure", "relation_failure"):
            for variable, reference, exposed, comparison_rows in comparisons:
                print(f"[control] {dataset} {outcome}: {variable} {exposed} vs {reference}", flush=True)
                buckets: dict[tuple[str, int, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
                for row in comparison_rows:
                    buckets[strata[id(row)]][str(row[variable])].append(row)
                weighted_exposed = weighted_reference = total_weight = 0.0
                used = 0
                for groups in buckets.values():
                    left, right = groups.get(reference, []), groups.get(exposed, [])
                    if not left or not right:
                        continue
                    weight = float(min(len(left), len(right)))
                    weighted_reference += weight * (sum(int(r[outcome]) for r in left) / len(left))
                    weighted_exposed += weight * (sum(int(r[outcome]) for r in right) / len(right))
                    total_weight += weight; used += 1
                out.append({
                    "dataset": dataset, "outcome": outcome,
                    "scene_or_geometry_variable": variable.removeprefix("_control_"),
                    "comparison": f"{exposed} vs {reference}",
                    "method": "stratified_rate_difference",
                    "controls": "dataset + quartiles(log(gt_area), image_gt_count, border_distance)",
                    "matched_strata": used, "weighted_n_per_group": total_weight,
                    "adjusted_exposed_rate": weighted_exposed / total_weight if total_weight else None,
                    "adjusted_reference_rate": weighted_reference / total_weight if total_weight else None,
                    "adjusted_rate_difference": (weighted_exposed - weighted_reference) / total_weight if total_weight else None,
                    "status": "estimable" if total_weight else "not_estimable",
                })
        for row in selected:
            for variable in ("contact_count", "nearest_boundary_distance_norm", "near_other_ratio"):
                row.pop(f"_control_{variable}", None)
    return out


def rectangle_metrics(masks: list[np.ndarray], boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, g = len(boxes), len(masks)
    intersections = np.zeros((g, n), dtype=np.float64)
    x0 = np.clip(np.ceil(boxes[:, 0]).astype(np.int64), 0, masks[0].shape[1])
    y0 = np.clip(np.ceil(boxes[:, 1]).astype(np.int64), 0, masks[0].shape[0])
    x1 = np.clip(np.ceil(boxes[:, 2]).astype(np.int64), 0, masks[0].shape[1])
    y1 = np.clip(np.ceil(boxes[:, 3]).astype(np.int64), 0, masks[0].shape[0])
    rect_area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    gt_area = np.asarray([m.sum() for m in masks], dtype=np.float64)
    for index, mask in enumerate(masks):
        integral = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        intersections[index] = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    union = gt_area[:, None] + rect_area[None, :] - intersections
    return intersections / np.maximum(union, 1), intersections / gt_area[:, None], intersections / np.maximum(rect_area[None, :], 1)


def box_feasibility_edges(primary: str, box_iou: np.ndarray, box_cov: np.ndarray) -> np.ndarray:
    return box_cov >= 0.75 if primary in {"I", "L", "S"} else (box_iou >= 0.50) | (box_cov >= 0.50)


def covers_targets(edges: np.ndarray, target_indices: list[int], candidates: np.ndarray) -> tuple[bool, dict[int, int]]:
    candidate_set = set(int(x) for x in candidates)
    chosen: dict[int, int] = {}
    used: set[int] = set()
    options = {g: [int(c) for c in np.flatnonzero(edges[g] & (edges.sum(axis=0) == 1)) if int(c) in candidate_set] for g in target_indices}
    for g in sorted(target_indices, key=lambda x: len(options[x])):
        available = [c for c in options[g] if c not in used]
        if not available:
            return False, {}
        chosen[g] = available[0]; used.add(available[0])
    return True, chosen


def reconstruct_candidate_edges(raw: Any, candidate_ids: np.ndarray, gt_masks: list[np.ndarray], device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from ultralytics.utils import ops

    g = len(gt_masks)
    result_iou = np.zeros((g, len(candidate_ids)), dtype=np.float64)
    result_cov = np.zeros_like(result_iou)
    result_pur = np.zeros_like(result_iou)
    proto = torch.from_numpy(raw["prototype"]).to(device)
    gt_area = np.asarray([m.sum() for m in gt_masks], dtype=np.float64)
    for start in range(0, len(candidate_ids), 2):
        ids = candidate_ids[start:start + 2]
        coeff = torch.from_numpy(raw["mask_coefficients"][ids]).to(device)
        boxes = torch.from_numpy(raw["boxes_xyxy"][ids]).to(device)
        decoded = ops.process_mask_native(proto, coeff, boxes, gt_masks[0].shape).cpu().numpy().astype(bool)
        pred_area = decoded.reshape(len(ids), -1).sum(1).astype(np.float64)
        for gi, gt in enumerate(gt_masks):
            inter = np.logical_and(decoded, gt[None]).reshape(len(ids), -1).sum(1).astype(np.float64)
            result_cov[gi, start:start + len(ids)] = inter / gt_area[gi]
            result_pur[gi, start:start + len(ids)] = inter / np.maximum(pred_area, 1)
            result_iou[gi, start:start + len(ids)] = inter / np.maximum(gt_area[gi] + pred_area - inter, 1)
    return result_iou, result_cov, result_pur, (result_iou >= 0.50) | (result_cov >= 0.50)


def analyze_failure_stages(cache_root: Path, gt_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]], records_by_dataset: dict[str, dict[int, dict[str, Any]]], device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from pycocotools import mask as mask_utils

    gt_lookup = {(r["dataset"], int(r["image_id"]), int(r["annotation_id"])): r for r in gt_rows}
    comp_lookup = {r["component_id"]: r for r in component_rows}
    stage_rows: list[dict[str, Any]] = []
    o_rows: list[dict[str, Any]] = []
    for spec in DATASETS:
        dataset = spec["name"]
        manifest = read_json(spec["manifest"])
        images = {int(x["id"]): x for x in manifest["images"]}
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in manifest["annotations"]:
            anns_by_image[int(ann["image_id"])].append(ann)
        rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in gt_rows:
            if row["dataset"] == dataset and row["primary_class"] != "C":
                rows_by_image[int(row["image_id"])].append(row)
        cache_dir = cache_root / spec["cache"]
        for image_id in sorted(rows_by_image):
            image = images[image_id]
            anns = sorted(anns_by_image[image_id], key=lambda x: int(x["id"]))
            ann_index = {int(a["id"]): i for i, a in enumerate(anns)}
            gt_rles = [normalize_gt(a, int(image["height"]), int(image["width"])) for a in anns]
            gt_masks = [mask_utils.decode(x).astype(bool) for x in gt_rles]
            record = records_by_dataset[dataset][image_id]
            preds = record["final_predictions"]
            pred_rles = [rle(x["mask_rle"]) for x in preds]
            final_iou, final_cov, final_pur = matrices(gt_rles, pred_rles)
            final_core = (final_iou >= 0.50) | (final_cov >= 0.50)
            final_weak = (~final_core) & (final_pur >= 0.50) & (final_cov >= 0.10)
            mapping = {int(x["pred_id"]): int(x["source_candidate_id"]) for x in record["final_pred_to_source_candidate"]}
            with np.load(cache_dir / record["raw_cache"], allow_pickle=False) as raw:
                boxes = raw["boxes_xyxy"].astype(np.float64)
                scores = raw["scores"].astype(np.float64)
                ranks = raw["global_rank"].astype(np.int64)
                top_ids = raw["top_indices"].astype(np.int64)
                conf_ids = top_ids[scores[top_ids] >= 0.05]
                box_iou, box_cov, _ = rectangle_metrics(gt_masks, boxes)
                units: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows_by_image[image_id]:
                    key = row["component_id"] or f"{dataset}:{image_id}:MISS:{row['annotation_id']}"
                    units[key].append(row)
                for unit_id, unit_rows in units.items():
                    primary = unit_rows[0]["primary_class"]
                    if primary in {"O", "M", "X"}:
                        comp = comp_lookup[unit_rows[0]["component_id"]]
                        target_annotations = json.loads(comp["annotation_ids"])
                        pred_ids = json.loads(comp["pred_ids"])
                    else:
                        target_annotations = [int(unit_rows[0]["annotation_id"])]
                        pred_ids = [] if primary == "MISS" else json.loads(comp_lookup[unit_rows[0]["component_id"]]["pred_ids"])
                    targets = [ann_index[int(x)] for x in target_annotations]
                    edge = box_feasibility_edges(primary, box_iou, box_cov)
                    raw_ok, raw_choice = covers_targets(edge, targets, np.arange(len(boxes)))
                    top_ok, top_choice = covers_targets(edge, targets, top_ids)
                    conf_ok, conf_choice = covers_targets(edge, targets, conf_ids)
                    relevant = np.asarray(sorted({int(c) for g in targets for c in np.flatnonzero(edge[g]) if int(c) in set(conf_ids.tolist())}), dtype=np.int64)
                    mask_ok = False
                    mask_choice: dict[int, int] = {}
                    mask_edge = np.zeros((len(gt_masks), 0), dtype=bool)
                    mask_cov = np.zeros((len(gt_masks), 0), dtype=float)
                    mask_pur = np.zeros((len(gt_masks), 0), dtype=float)
                    if conf_ok and len(relevant):
                        _, mask_cov, mask_pur, mask_edge = reconstruct_candidate_edges(raw, relevant, gt_masks, device)
                        mask_edge_for_stage = (mask_cov >= 0.75) & (mask_pur >= 0.75) if primary in {"I", "L", "S"} else mask_edge
                        local_ok, local_choice = covers_targets(mask_edge_for_stage, targets, np.arange(len(relevant)))
                        mask_ok = local_ok
                        mask_choice = {g: int(relevant[c]) for g, c in local_choice.items()}
                    diagnostic_primary = "MISS"
                    target_component = next(((gs, ps) for gs, ps in graph_components(mask_edge) if targets[0] in gs), None)
                    if target_component is not None:
                        dgs, dps = target_component
                        if len(dgs) == len(dps) == 1:
                            dg, dp = dgs[0], dps[0]
                            diagnostic_primary = "C" if mask_cov[dg, dp] >= 0.75 and mask_pur[dg, dp] >= 0.75 else "I" if mask_cov[dg, dp] < 0.75 and mask_pur[dg, dp] >= 0.75 else "L" if mask_cov[dg, dp] >= 0.75 else "S"
                        else:
                            diagnostic_primary = "O" if len(dgs) == 1 else "M" if len(dps) == 1 else "X"
                    raw_reproduces_primary = diagnostic_primary == primary
                    if primary == "O":
                        component_pred_indices = [i for i, p in enumerate(preds) if int(p["pred_id"]) in set(pred_ids)]
                        cross = any(final_weak[g, p] for g in range(len(anns)) if g not in targets for p in component_pred_indices)
                        pred_masks = [mask_utils.decode(pred_rles[p]).astype(bool) for p in component_pred_indices]
                        max_pair_iou = 0.0
                        for a in range(len(pred_masks)):
                            for b in range(a + 1, len(pred_masks)):
                                inter = np.logical_and(pred_masks[a], pred_masks[b]).sum()
                                union = np.logical_or(pred_masks[a], pred_masks[b]).sum()
                                max_pair_iou = max(max_pair_iou, float(inter / union) if union else 0.0)
                        target_mask = gt_masks[targets[0]]
                        individual_cov = [np.logical_and(m, target_mask).sum() / target_mask.sum() for m in pred_masks]
                        union_mask = np.logical_or.reduce(pred_masks)
                        union_cov = float(np.logical_and(union_mask, target_mask).sum() / target_mask.sum())
                        subtype = "CROSS_INSTANCE_ASSOCIATION" if cross else "TRUE_DUPLICATE" if max_pair_iou >= 0.50 else "FRAGMENTATION" if max_pair_iou < 0.10 and union_cov >= 0.75 and union_cov - max(individual_cov, default=0) >= 0.10 else "UNRESOLVED"
                        stage = "MASK" if subtype in {"CROSS_INSTANCE_ASSOCIATION", "FRAGMENTATION"} else "FINAL_SET" if subtype == "TRUE_DUPLICATE" else "UNRESOLVED"
                        o_rows.append({"dataset": dataset, "image_id": image_id, "component_id": unit_id, "annotation_id": target_annotations[0], "o_subtype": subtype, "prediction_count": len(pred_masks), "max_prediction_pair_iou": max_pair_iou, "union_gt_coverage": union_cov, "weak_cross_instance_relation": int(cross)})
                    elif not raw_ok:
                        stage = "RAW_BOX"
                    elif not top_ok:
                        stage = "TOPK_BOX"
                    elif not conf_ok:
                        stage = "CONF_BOX"
                    elif not mask_ok:
                        stage = "MASK"
                    else:
                        stage = "FINAL_SET"
                    stage_hypothesis = stage
                    if not raw_reproduces_primary:
                        stage = "UNRESOLVED"
                    all_candidate_ids = sorted(set(raw_choice.values()) | set(top_choice.values()) | set(conf_choice.values()) | set(mask_choice.values()))
                    best_id = max(all_candidate_ids, key=lambda c: float(scores[c])) if all_candidate_ids else None
                    scene_counts = Counter(r["scene_label"] for r in unit_rows)
                    stage_row = {
                        "dataset": dataset, "image_id": image_id, "analysis_unit_id": unit_id, "primary_class": primary,
                        "annotation_ids": json.dumps(target_annotations), "pred_ids": json.dumps(pred_ids), "gt_count": len(targets), "prediction_count": len(pred_ids),
                        "scene_label": scene_counts.most_common(1)[0][0], "failure_stage": stage,
                        "diagnostic_stage_hypothesis": stage_hypothesis, "diagnostic_primary_class": diagnostic_primary,
                        "raw_branch_reproduces_formal_primary": int(raw_reproduces_primary),
                        "raw_box_feasible": int(raw_ok), "topk_box_feasible": int(top_ok), "conf_box_feasible": int(conf_ok), "mask_separable": int(mask_ok),
                        "box_feasibility_rule": "GT coverage >= 0.75" if primary in {"I", "L", "S"} else "box IoU >= 0.50 OR GT coverage >= 0.50",
                        "best_candidate_id": best_id, "best_candidate_score": float(scores[best_id]) if best_id is not None else None,
                        "best_candidate_rank": int(ranks[best_id]) if best_id is not None else None,
                        "reconstructed_conf_candidate_count": int(len(relevant)),
                    }
                    stage_rows.append(stage_row)
                    for row in unit_rows:
                        row["failure_stage"] = stage
                        row["failure_analysis_unit_id"] = unit_id
    for row in gt_rows:
        if row["primary_class"] == "C":
            row["failure_stage"] = None
            row["failure_analysis_unit_id"] = None
    return stage_rows, o_rows


def stage_summaries(stage_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_failure: list[dict[str, Any]] = []
    by_scene: list[dict[str, Any]] = []
    for dataset in [x["name"] for x in DATASETS] + ["ALL"]:
        rows = stage_rows if dataset == "ALL" else [r for r in stage_rows if r["dataset"] == dataset]
        for primary in ("I", "L", "S", "O", "M", "X", "MISS", "ALL_FAILURE"):
            sub = rows if primary == "ALL_FAILURE" else [r for r in rows if r["primary_class"] == primary]
            counts = Counter(r["failure_stage"] for r in sub)
            for stage in STAGES:
                by_failure.append({"dataset": dataset, "primary_class": primary, "failure_stage": stage, "analysis_units": counts[stage], "rate": counts[stage] / len(sub) if sub else None})
        for scene in ("Isolated", "Near/Dense", "Touching"):
            sub = [r for r in rows if r["scene_label"] == scene]
            counts = Counter(r["failure_stage"] for r in sub)
            for stage in STAGES:
                by_scene.append({"dataset": dataset, "scene_label": scene, "failure_stage": stage, "analysis_units": counts[stage], "rate": counts[stage] / len(sub) if sub else None})
    return by_failure, by_scene


def make_figures(output: Path, scene_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    pooled = [r for r in scene_rows if r["dataset"] == "ALL"]
    labels = [r["scene_label"] for r in pooled]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, [r["failure_rate"] for r in pooled], 0.36, label="Any failure")
    ax.bar(x + 0.18, [r["relation_failure_rate"] for r in pooled], 0.36, label="O/M/X")
    ax.set_xticks(x, labels); ax.set_ylabel("Rate"); ax.set_ylim(0, 1); ax.legend(); fig.tight_layout()
    fig.savefig(output / "failure_by_scene.png", dpi=180); plt.close(fig)
    pooled_stage = [r for r in stage_rows if r["dataset"] == "ALL" and r["primary_class"] == "ALL_FAILURE"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["failure_stage"] for r in pooled_stage], [r["analysis_units"] for r in pooled_stage])
    ax.set_ylabel("Failure analysis units"); ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    fig.savefig(output / "failure_stage_distribution.png", dpi=180); plt.close(fig)


def make_summary(output: Path, integrity: dict[str, Any], basic: list[dict[str, Any]], scene: list[dict[str, Any]], controls: list[dict[str, Any]], stage_summary: list[dict[str, Any]], o_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = next(r for r in basic if r["dataset"] == "ALL")
    stage_counts = {r["failure_stage"]: r["analysis_units"] for r in stage_summary if r["dataset"] == "ALL" and r["primary_class"] == "ALL_FAILURE"}
    o_counts = Counter(r["o_subtype"] for r in o_rows)
    principal = max(stage_counts, key=stage_counts.get) if stage_counts else "UNRESOLVED"
    summary = {
        "cache_integrity_passed": integrity["passed"], "images": pooled["images"], "gt": pooled["gt"], "predictions": pooled["predictions"], "fp": pooled["fp"],
        "primary_counts": {label: pooled[f"{label}_count"] for label in PRIMARY},
        "rates": {"total_failure": pooled["total_failure_rate"], "geometry_failure": pooled["geometry_failure_rate"], "relation_failure": pooled["relation_failure_rate"], "miss": pooled["miss_rate"]},
        "failure_stage_counts": stage_counts, "o_subtypes": dict(o_counts), "principal_observed_failure_stage": principal,
        "unresolved_analysis_units": stage_counts.get("UNRESOLVED", 0),
        "evidence_limits": [
            "Associations are descriptive and not causal.",
            "Touching is computed from visible GT masks and is not an occlusion label.",
            "RAW_BOX/TOPK_BOX/CONF_BOX are box-feasibility gates; only MASK uses reconstructed candidate masks. UNRESOLVED is retained where evidence is insufficient.",
            "The 8.4.100 diagnostic runtime differs from the archived 8.4.27 training-era runtime, although checkpoint/config key audit passed.",
        ],
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# YOLO26-seg 全量诊断缓存正式分析", "", f"完成时间：{datetime.now().astimezone().isoformat()}", "",
        "## 数据与缓存验收", "", f"- 诊断缓存：{summary['images']} 张图，验收：{'通过' if summary['cache_integrity_passed'] else '失败'}。",
        f"- GT：{summary['gt']}；最终 Prediction：{summary['predictions']}；无核心关系 FP：{summary['fp']}。", "",
        "## GT 主分类", "",
        "| 类别 | 数量 | 占全部 GT |", "|---|---:|---:|",
        *[f"| {label} | {pooled[f'{label}_count']} | {pooled[f'{label}_rate']:.2%} |" for label in PRIMARY], "",
        f"总失败率：{pooled['total_failure_rate']:.2%}；I/L/S：{pooled['geometry_failure_rate']:.2%}；O/M/X：{pooled['relation_failure_rate']:.2%}；MISS：{pooled['miss_rate']:.2%}。", "",
        "## 场景条件失败率（全数据集合并）", "", "| 场景 | GT | 总失败率 | O/M/X率 | MISS率 |", "|---|---:|---:|---:|---:|",
    ]
    for row in [r for r in scene if r["dataset"] == "ALL"]:
        lines.append(f"| {row['scene_label']} | {row['n_gt']} | {row['failure_rate']:.2%} | {row['relation_failure_rate']:.2%} | {row['miss_rate']:.2%} |")
    lines += ["", "## 首次失败阶段（分析单元）", "", "RAW_BOX/TOPK_BOX/CONF_BOX 只表示 box feasibility；MASK 才表示重建 mask 的失败。", "", "| 阶段 | 数量 |", "|---|---:|", *[f"| {stage} | {stage_counts.get(stage, 0)} |" for stage in STAGES], "", f"当前数量最多的已观察阶段：`{principal}`。这是缓存证据下的定位，不是因果结论。", "", "## O 类细分", "", "| 子型 | 数量 |", "|---|---:|", *[f"| {key} | {value} |" for key, value in sorted(o_counts.items())], "", "## 控制分析说明", "", "`controlled_analysis.csv` 分别控制 log(GT面积)、图像GT数量和归一化图像边缘距离。结果只描述条件关联；拟合不稳定的行已明确标记。", "", "## 证据边界", "", *[f"- {x}" for x in summary["evidence_limits"]], "", "## 最值得验证的干预", "", f"优先对数量最多且可定位的 `{principal}` 样本做单变量 oracle/bypass；保持 checkpoint、输入尺度、Top-K、conf 和掩码生成以外的条件不变。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts" / "inference_cache" / "yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "analysis" / "yolo26seg_diagnostic_full_single_forward_20260901")
    parser.add_argument("--reuse-deterministic-dir", type=Path, help="Reuse already completed cache-derived classification and GT geometry tables.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/6] auditing diagnostic cache", flush=True)
    integrity, records = audit_cache(args.cache_root.resolve(), args.output_dir)
    if args.reuse_deterministic_dir:
        print("[2-4/6] reusing completed deterministic classification and geometry tables", flush=True)
        gt_rows, component_rows, prediction_rows, basic, scene, geometry_validation = reuse_deterministic_results(args.reuse_deterministic_dir.resolve(), args.output_dir)
    else:
        print("[2/6] classifying final predictions", flush=True)
        gt_rows, component_rows, prediction_rows = classify_all(args.cache_root.resolve(), records)
        print("[3/6] computing full-GT scene geometry", flush=True)
        gt_rows, geometry_validation = add_scene_geometry(gt_rows)
        print("[4/6] aggregating results", flush=True)
        basic, scene, component_summary = aggregate_results(gt_rows, component_rows, prediction_rows)
        write_csv(args.output_dir / "gt_analysis.csv", gt_rows)
        write_csv(args.output_dir / "component_analysis.csv", component_rows)
        write_csv(args.output_dir / "prediction_analysis.csv", prediction_rows)
        write_csv(args.output_dir / "scene_analysis.csv", scene)
        write_csv(args.output_dir / "basic_summary.csv", basic)
        write_csv(args.output_dir / "component_summary.csv", component_summary)
        write_json(args.output_dir / "geometry_validation.json", geometry_validation)
    print("[4/6] running controlled analysis", flush=True)
    controls = controlled_analysis(gt_rows)
    print("[5/6] locating internal failure stages", flush=True)
    stage_rows, o_rows = analyze_failure_stages(args.cache_root.resolve(), gt_rows, component_rows, records, args.device)
    print("[6/6] writing result tables and figures", flush=True)
    stage_summary, scene_stage = stage_summaries(stage_rows)
    write_csv(args.output_dir / "controlled_analysis.csv", controls)
    write_csv(args.output_dir / "failure_stage_analysis.csv", stage_rows)
    write_csv(args.output_dir / "failure_stage_summary.csv", stage_summary)
    write_csv(args.output_dir / "scene_failure_stage.csv", scene_stage)
    write_csv(args.output_dir / "o_subtype_analysis.csv", o_rows)
    make_figures(args.output_dir, scene, stage_summary)
    summary = make_summary(args.output_dir, integrity, basic, scene, controls, stage_summary, o_rows)
    print(json.dumps({"status": "completed", "output": str(args.output_dir), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
