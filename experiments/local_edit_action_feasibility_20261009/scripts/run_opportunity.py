"""Fresh-native engineering or predeclared finite-action train opportunity.

Engineering parses no GT and uses exactly the panel's engineering image IDs.
Opportunity requires all locked512 train images and original known-hash COCO
annotations. GT scores already generated actions on the original pixel grid.
The paired image-cluster interval is a finite-action investment gate, not AP,
a deployable method result, or an upper bound on the method family.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

import local_actions
import native_opportunity
from native_opportunity import (ADAPTER_VERSION, GLOBAL_ACTIONS, NATIVE_SCRIPTS, OFFICIAL_SHA256,
                                FiniteNativeAdapter, FrozenYOLO, Timer, prepare_native, sha256, timed_native_forward)

VERSION = "local_edit_finite_opportunity_v1"
PANEL_SCHEMA = "local_edit_opportunity_panel_v1"
LOCK_SCHEMA = "local_edit_opportunity_source_lock_v1"
BOOTSTRAP_SEED = 20261009
BOOTSTRAP_SAMPLES = 5000
GATE = .002
POOL_SOURCE_SHA256 = "ef3453e65b80a2d07997c882b0624a88f6d8161fcd32e2d629cdff3285ba3c63"
CALIBRATION_SOURCE_SHA256 = "1b75dfea2ca011dc43aaa8a7b863900c3bae913472b849f76934361366d9d8ce"
SAMPLING_ALGORITHM = "numpy.default_rng(seed).choice(sorted_eligible_ids,512,replace=False); sort selected IDs"
GROUPS = ("G_old", "G_eq", "G_eq_plus_L")
VENDOR_FILES = {"vendor_ops": "ultralytics/utils/ops.py", "vendor_nms": "ultralytics/utils/nms.py",
                "vendor_augment": "ultralytics/data/augment.py", "vendor_segment_validator": "ultralytics/models/yolo/segment/val.py",
                "vendor_head": "ultralytics/nn/modules/head.py"}
FROZEN_DEPENDENCIES = ("mask_calibration", "local_features", "risk_calibration", "portable_risk")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
    handle.flush()


def read_locked(path, expected, label):
    path = Path(path).resolve()
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Locked {label} path/bytes differ: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_panel(panel, mode):
    if panel.get("schema_version") != PANEL_SCHEMA:
        raise ValueError("Unknown panel schema")
    if panel.get("split") != "train2017" or panel.get("device") not in ("cuda", "cuda:0"):
        raise ValueError("This fixed panel requires desktop CUDA and train2017")
    if Path(panel["interpreter"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Actual interpreter differs from panel declaration")
    if panel["weights"].get("sha256") != OFFICIAL_SHA256 or sha256(panel["weights"]["path"]) != OFFICIAL_SHA256:
        raise ValueError("Official weight bytes differ")
    if panel["vendor"].get("version") != "8.4.100":
        raise ValueError("Fixed native vendor version differs")
    images = panel.get("images", [])
    ids = [item["image_id"] for item in images]
    paths = [str(Path(item["path"]).resolve()) for item in images]
    if len(images) != 512 or len(set(ids)) != 512 or len(set(paths)) != 512 or ids != sorted(ids):
        raise ValueError("Panel must contain sorted unique512 image identities and paths")
    if any(not isinstance(iid, int) or isinstance(iid, bool) for iid in ids):
        raise ValueError("Image IDs must be integers")
    for item in images:
        if not Path(item["path"]).is_absolute() or not Path(item["path"]).is_file() or sha256(item["path"]) != item["sha256"]:
            raise ValueError("Panel image path/bytes differ: "+str(item["image_id"]))
    selected = panel.get("engineering_image_ids", [])
    if selected != ids[:4]:
        raise ValueError("Engineering must use the predeclared first4 sorted panel image IDs")
    if mode == "opportunity":
        if not panel.get("annotations") or not panel.get("calibration_exclusion") or not panel.get("selection") or not panel.get("population"):
            raise ValueError("Complete formal annotation/sampling/exclusion declarations required")
        if panel["selection"].get("kind") != "uniform_without_replacement" or panel["selection"].get("image_count") != 512 or panel["selection"].get("algorithm") != SAMPLING_ALGORITHM:
            raise ValueError("Uniform512 sampling declaration differs")
    return images if mode == "opportunity" else [next(item for item in images if item["image_id"] == iid) for iid in selected]


def validate_source_lock(lock, panel, study):
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("Unknown source-lock schema")
    expected = {"run_opportunity": Path(__file__).resolve(), "native_opportunity": Path(native_opportunity.__file__).resolve(),
                "local_actions": Path(local_actions.__file__).resolve(), "frozen_io": NATIVE_SCRIPTS / "frozen_io.py",
                "native_mask_adapter": NATIVE_SCRIPTS / "native_mask_adapter.py", "protocol": study / "PROTOCOL.md"}
    expected.update({key: Path(panel["vendor"]["path"]).resolve()/relative for key, relative in VENDOR_FILES.items()})
    expected.update({key: NATIVE_SCRIPTS/(key+".py") for key in FROZEN_DEPENDENCIES})
    if set(lock.get("files", {})) != set(expected):
        raise ValueError("Source-lock keys differ from exact required execution source set")
    for key, path in expected.items():
        declared = lock["files"][key]
        if Path(declared["path"]).resolve() != path.resolve() or not path.is_file() or sha256(path) != declared["sha256"]:
            raise ValueError("Runtime source path/bytes differ: "+key)
    return {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in expected.items()}


def archive_inputs(run, panel_path, lock_path, source):
    target = run / "source"
    target.mkdir(exist_ok=True)
    for key, item in source.items():
        filename = (key.removeprefix("vendor_")+".py") if key.startswith("vendor_") else ("PROTOCOL.md" if key == "protocol" else key+".py")
        shutil.copy2(item["path"], target/filename)
        if sha256(target/filename) != item["sha256"]:
            raise RuntimeError("Archived source bytes differ: "+key)
    shutil.copy2(panel_path, run / "PANEL.json")
    shutil.copy2(lock_path, run / "DECLARED_SOURCE_LOCK.json")
    dump(run / "SOURCE_LOCK.json", {"schema_version": LOCK_SCHEMA, "files": source,
                                     "snapshot_directory": str(target.resolve()), "locked_at": now()})


def load_formal_annotations(panel):
    annotation = panel["annotations"]
    if sha256(annotation["path"]) != annotation["sha256"]:
        raise ValueError("Original COCO train annotation bytes differ")
    exclusion = panel["calibration_exclusion"]
    if sha256(exclusion["path"]) != exclusion["sha256"]:
        raise ValueError("Independent calibration-list bytes differ")
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(annotation["path"])
    excluded_ids = exclusion.get("image_ids")
    if not isinstance(excluded_ids, list) or len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("Explicit unique calibration exclusion IDs required")
    excluded_document = read_locked(exclusion["path"], exclusion["sha256"], "fixed calibration exclusion list")
    if excluded_document != sorted(excluded_ids) or len(excluded_ids) != 2000:
        raise ValueError("Locked exclusion file must equal unique sorted2000 declared IDs")
    calibration_source = exclusion.get("source", {})
    if calibration_source.get("sha256") != CALIBRATION_SOURCE_SHA256:
        raise ValueError("Original independent calibration split hash differs")
    calibration_document = read_locked(calibration_source["path"], CALIBRATION_SOURCE_SHA256, "original calibration split")
    fit, select = calibration_document.get("fit_ids", []), calibration_document.get("selection_ids", [])
    if len(fit) != 1500 or len(select) != 500 or set(fit)&set(select) or sorted(set(fit)|set(select)) != sorted(excluded_ids):
        raise ValueError("Exclusion2000 must equal original fit1500 plus select500")
    population = panel["population"]
    if population["source"]["sha256"] != POOL_SOURCE_SHA256:
        raise ValueError("Original declared TriFlow20k source pool SHA differs")
    eligible_document = read_locked(population["path"], population["sha256"], "eligible image pool")
    source_document = read_locked(population["source"]["path"], population["source"]["sha256"], "original20k pool")
    def image_ids(document):
        if isinstance(document, list):
            return document
        if isinstance(document, dict) and isinstance(document.get("image_ids"), list):
            return document["image_ids"]
        raise ValueError("Pool source must be JSON list or {image_ids:list}")
    source_ids = image_ids(source_document)
    eligible = image_ids(eligible_document)
    if len(source_ids) != 20000 or len(set(source_ids)) != 20000:
        raise ValueError("Original pool must retain unique20000 source image IDs")
    if sorted(set(source_ids)-set(excluded_ids)) != eligible or len(eligible) != 19638:
        raise ValueError("Eligible19638 pool differs from original20k minus locked calibration2000")
    if any(iid not in coco.imgs for iid in eligible):
        raise ValueError("Eligible pool contains image absent from original train annotations")
    if population.get("image_ids") is not None and population["image_ids"] != eligible:
        raise ValueError("Optional inline eligible IDs differ from locked population file")
    selection = panel["selection"]
    if selection.get("seed") != BOOTSTRAP_SEED:
        raise ValueError("Predeclared uniform512 sampling seed differs")
    if selection.get("population_count") != len(eligible) or selection.get("population_sha256") != canonical_sha(eligible):
        raise ValueError("Declared eligible training universe differs")
    replay = sorted(int(value) for value in np.random.default_rng(selection["seed"]).choice(np.asarray(eligible, dtype=np.int64), 512, replace=False))
    if replay != [item["image_id"] for item in panel["images"]]:
        raise ValueError("Panel IDs do not replay the predeclared uniform sample")
    if set(replay)&set(excluded_ids):
        raise ValueError("Training opportunity overlaps independent calibration")
    for item in panel["images"]:
        if Path(item["path"]).name != coco.imgs[item["image_id"]]["file_name"]:
            raise ValueError("Image ID/COCO filename identity differs")
    return coco


def associate(candidate, coco, image_id, cache):
    """Final8 fixed output association: original COCO order and first argmax."""
    same_class = [ann for ann in coco.imgToAnns.get(image_id, [])
                  if not ann.get("iscrowd", 0) and ann["category_id"] == candidate.identity["category_id"]]
    identity = candidate.binding.metadata()
    if not same_class:
        return identity | {"matched": False, "labelable": False, "unknown_reason": "no_noncrowd_same_category_GT",
                           "annotation_id": None, "box_iou": None}, None
    boxes = np.asarray([[ann["bbox"][0], ann["bbox"][1], ann["bbox"][0]+ann["bbox"][2], ann["bbox"][1]+ann["bbox"][3]] for ann in same_class], dtype=np.float64)
    box = np.asarray(candidate.identity["box_xyxy"], dtype=np.float64)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])-np.maximum(box[:2], boxes[:, :2])).prod(1)
    union = np.maximum(0, box[2:]-box[:2]).prod()+np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(1)-intersection
    overlaps = intersection/np.maximum(union, 1e-12)
    index = int(np.argmax(overlaps))
    if overlaps[index] < .5:
        return identity | {"matched": False, "labelable": False, "unknown_reason": "maximum_same_category_box_iou_below_0.5",
                           "annotation_id": None, "box_iou": float(overlaps[index])}, None
    ann, gid = same_class[index], int(same_class[index]["id"])
    if gid not in cache:
        try:
            mask = coco.annToMask(ann).astype(np.bool_)
            cache[gid] = (mask, None)
        except Exception as exc:
            cache[gid] = (None, "GT_decode_failed: "+repr(exc))
    mask, error = cache[gid]
    if error is None and tuple(mask.shape) != tuple(candidate.baseline_original.shape):
        error = "GT_original_shape_differs"
    if error is None and not mask.any():
        error = "GT_has_zero_original_area"
    return identity | {"matched": True, "labelable": error is None, "unknown_reason": error,
                       "annotation_id": gid, "category_id": int(ann["category_id"]), "box_iou": float(overlaps[index]),
                       "gt_annotation_area": float(ann.get("area", 0)), "gt_original_pixel_area": None if mask is None else int(mask.sum()),
                       "gt_bbox_xywh": ann["bbox"], "gt_iscrowd": int(ann.get("iscrowd", 0)),
                       "gt_ignore": ann.get("ignore"), "original_COCO_same_class_order_index": index}, mask if error is None else None


def pixel_metrics(mask, gt):
    if gt is None:
        return {key: None for key in ("pixel_TP", "pixel_FP", "pixel_FN", "mask_iou", "coverage", "purity")}
    tp = int(np.count_nonzero(mask & gt))
    fp = int(np.count_nonzero(mask & ~gt))
    fn = int(np.count_nonzero(~mask & gt))
    return {"pixel_TP": tp, "pixel_FP": fp, "pixel_FN": fn, "mask_iou": tp/(tp+fp+fn),
            "coverage": tp/(tp+fn), "purity": tp/(tp+fp) if tp+fp else None}


def rle_encode(mask, mask_utils):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def bootstrap(per_image):
    """Ratio of image-cluster sums; retain no-match images in every draw."""
    values = np.asarray(per_image, dtype=np.float64)
    totals = values.sum(0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for start in range(0, BOOTSTRAP_SAMPLES, 100):
        batch = min(100, BOOTSTRAP_SAMPLES-start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        draws.append(values[indices].sum(1))
    draws = np.concatenate(draws)
    result = {"bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
              "bootstrap_unit": "paired supplied image clusters; ratio of candidate sums; no-match images retained",
              "images": len(values), "images_with_labelable_candidates": int((values[:, 0]>0).sum()),
              "labelable_matched_candidates": int(totals[0]), "baseline_success_count": int(totals[3]),
              "baseline_failure_count": int(totals[4])}
    definitions = {"local_marginal_mask_iou": (1, 0), "equal_direction_vs_old_mask_iou": (2, 0),
                   "old_oracle_mean_mask_iou": (5, 0), "equal_oracle_mean_mask_iou": (6, 0),
                   "local_oracle_mean_mask_iou": (7, 0), "old_oracle_repair_rate": (8, 4),
                   "equal_oracle_repair_rate": (9, 4), "local_oracle_repair_rate": (10, 4),
                   "local_additional_repair_rate": (11, 4)}
    for name, (num, den) in definitions.items():
        valid = draws[:, den] > 0
        distribution = draws[valid, num]/draws[valid, den]
        result[name] = {"value": float(totals[num]/totals[den]) if totals[den] else None,
                        "ci95": np.quantile(distribution, [.025, .975]).tolist() if len(distribution) else None,
                        "valid_bootstrap_draws": int(valid.sum())}
    return result


def gate_result(statistics):
    interval = statistics["local_marginal_mask_iou"]["ci95"]
    decision = "insufficient_evidence" if interval is None else "invest_in_separate_method_comparison" if interval[0] >= GATE else "stop_current_configuration" if interval[1] < GATE else "insufficient_evidence"
    return {"decision": decision, "threshold": GATE, "metric": "best(G_eq+L)-best(G_eq), matched candidate mean",
            "rule": "lower95CI>=.002 invest; upper95CI<.002 stop; otherwise insufficient",
            "meaning": "artificial finite-action route investment gate; not AP prediction, deployment evidence, or method upper bound"}


def peak_memory(torch):
    rss = peak = None
    try:
        import psutil
        memory = psutil.Process().memory_info()
        rss, peak = memory.rss, getattr(memory, "peak_wset", None)
    except ImportError:
        pass
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cpu_current_rss_bytes": rss, "cpu_peak_working_set_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "run-id", "panel", "panel-sha256", "source-lock", "source-lock-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--mode", choices=("engineering", "opportunity"), required=True)
    args = parser.parse_args()
    study = Path(args.study_root).resolve()
    if study != Path(__file__).resolve().parents[1]:
        raise ValueError("Execution study root differs from current script's Study")
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("Run ID must be one path component")
    run = study / "runs" / args.run_id
    # Run metadata is registered by runner/root, never fabricated here.
    if not (run / "run.json").is_file():
        raise ValueError("Register a distinct Run through the workbench runner before execution")
    for name in ("SUMMARY.json", "CANDIDATES.jsonl", "ACTIONS.jsonl", "FAILURE.json", "EXECUTION_INPUTS.json"):
        if (run/name).exists():
            raise FileExistsError("Execution history exists; retry requires a new Run ID")
    started, began, extractor = now(), time.perf_counter(), None
    try:
        panel = read_locked(args.panel, args.panel_sha256, "panel")
        lock = read_locked(args.source_lock, args.source_lock_sha256, "source lock")
        images = validate_panel(panel, args.mode)
        source = validate_source_lock(lock, panel, study)
        archive_inputs(run, args.panel, args.source_lock, source)
        coco = load_formal_annotations(panel) if args.mode == "opportunity" else None
        configuration = {"version": VERSION, "adapter_version": ADAPTER_VERSION, "mode": args.mode,
                         "panel_sha256": args.panel_sha256, "declared_source_lock_sha256": args.source_lock_sha256,
                         "source_sha256": {key: value["sha256"] for key, value in source.items()},
                         "image_ids": [item["image_id"] for item in images], "interpreter": sys.executable,
                         "weights": panel["weights"], "vendor": panel["vendor"], "device": panel["device"],
                         "config": {"input_shape": [640, 640], "fp32": True, "tf32": False, "conf": .001,
                                    "max_det": 300, "batch": 1, "scaleup": False, "rect": False,
                                    "native_one2one": True, "native_chunk_size": 78,
                                    "decode": "process_mask_native input640; scale_preds binary explicit ratio_pad to original byte"},
                         "actions": {"groups": list(GROUPS), "global": [[name, tau] for name, tau in GLOBAL_ACTIONS],
                                     "smooth": ".75/(1+(original_baseline_pixel_area/2304)^2)",
                                     "band_radius": 4, "connectivity": 8, "max_components_per_sign": 8,
                                     "single_region_only": True, "combinations": False, "empty_actions_preserved": True},
                         "GT_parsed": coco is not None, "GT_used_for_action_generation": False,
                         "association": None if coco is None else "noncrowd same COCO category in original imgToAnns order; highest original box IoU>=.5; first np.argmax tie; duplicate GT matches allowed",
                         "statistics": None if coco is None else {"bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                                                                   "gate_threshold": GATE, "no_match_images_retained": True}}
        dump(run / "EXECUTION_INPUTS.json", {"configuration": configuration, "configuration_sha256": canonical_sha(configuration), "started_at": started})
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Predeclared desktop CUDA device is unavailable")
        extractor = FrozenYOLO(panel["weights"]["path"], panel["vendor"]["path"], panel["device"])
        from pycocotools import mask as mask_utils
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        validator = SegmentationValidator(args={"conf": .001, "max_det": 300, "save_json": True, "plots": False,
                                               "imgsz": 640, "half": False, "rect": False})
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        adapter = FiniteNativeAdapter(panel["device"], engineering=args.mode == "engineering")
        torch.cuda.reset_peak_memory_stats()
        native_rows = matched = labelable = actions_count = empty_count = 0
        gt_seconds = rle_seconds = io_seconds = extract_seconds = model_seconds = native_seconds = 0.0
        unique_gt, per_image, individual_controls = set(), [], {}
        (run / "sparse").mkdir(exist_ok=True)
        with (run / "CANDIDATES.jsonl").open("w", encoding="utf-8") as candidates_file, \
             (run / "ACTIONS.jsonl").open("w", encoding="utf-8") as actions_file, \
             (run / "IMAGES.jsonl").open("w", encoding="utf-8") as images_file, \
             (run / "GT_ASSOCIATIONS.jsonl").open("w", encoding="utf-8") as associations_file, \
             torch.inference_mode():
            for number, item in enumerate(images, 1):
                iid, image_start = int(item["image_id"]), time.perf_counter()
                before_times, before_checks = dict(adapter.timings), dict(adapter.checks)
                extracted, forward = timed_native_forward(extractor, item["path"])
                if extracted["image_sha256"] != item["sha256"]:
                    raise RuntimeError("Image bytes changed between preflight and model extraction")
                extract_seconds += forward["frozen_extract_wall_seconds"]
                model_seconds += forward["native_model_seconds"]
                with Timer(panel["device"]) as timer:
                    prepared = prepare_native(extracted, validator, panel["device"])
                native_seconds += timer.seconds
                reference_decode_seconds = timer.seconds
                native_rows += len(prepared.coefficients)
                gt_cache, sparse = {}, {}
                counts = np.zeros(12, dtype=np.float64)
                image_matched = image_labelable = image_actions = image_empty = image_candidates = 0
                image_gt_seconds = image_rle_seconds = 0.0
                for candidate in adapter.candidates(extracted, prepared, iid, categories):
                    image_candidates += 1
                    score_start = time.perf_counter()
                    association, gt = (associate(candidate, coco, iid, gt_cache) if coco is not None else
                                       (candidate.binding.metadata() | {"matched": None, "labelable": False, "unknown_reason": "engineering_no_GT", "annotation_id": None}, None))
                    baseline_metrics = pixel_metrics(candidate.baseline_original, gt)
                    baseline_iou = baseline_metrics["mask_iou"]
                    success = None if baseline_iou is None else baseline_iou >= .75
                    association.update(baseline_metrics | {"baseline_success": success})
                    elapsed = time.perf_counter()-score_start
                    gt_seconds += elapsed
                    image_gt_seconds += elapsed
                    if association["matched"]:
                        matched += 1
                        image_matched += 1
                    if gt is not None:
                        labelable += 1
                        image_labelable += 1
                        unique_gt.add((iid, association["annotation_id"]))
                    best = {group: None for group in GROUPS}
                    action_ids, candidate_empties = [], 0
                    for metadata, original, indices in adapter.export_actions(candidate):
                        action_ids.append(metadata["action_id"])
                        actions_count += 1
                        image_actions += 1
                        if metadata["original_empty"]:
                            empty_count += 1
                            image_empty += 1
                            candidate_empties += 1
                        score_start = time.perf_counter()
                        metrics = pixel_metrics(original, gt)
                        elapsed = time.perf_counter()-score_start
                        gt_seconds += elapsed
                        image_gt_seconds += elapsed
                        iou = metrics["mask_iou"]
                        repair = None if iou is None else not success and iou >= .75
                        damage = None if iou is None else success and iou < .75
                        if iou is not None:
                            for group in metadata["available_in"]:
                                if best[group] is None or iou > best[group]["mask_iou"]:
                                    best[group] = {"action_id": metadata["action_id"], "mask_iou": iou}
                            control_name = metadata["action_id"] if metadata["family"] != "local" else "local_"+metadata["sparse"]["operation"]
                            control = individual_controls.setdefault(control_name, {"labelable_action_count": 0, "baseline_success_action_count": 0,
                                                                                  "baseline_failure_action_count": 0, "damage_action_count": 0, "repair_action_count": 0})
                            control["labelable_action_count"] += 1
                            control["baseline_success_action_count"] += int(success)
                            control["baseline_failure_action_count"] += int(not success)
                            control["damage_action_count"] += int(damage)
                            control["repair_action_count"] += int(repair)
                        encoding_start = time.perf_counter()
                        rle = rle_encode(original, mask_utils)
                        elapsed = time.perf_counter()-encoding_start
                        rle_seconds += elapsed
                        image_rle_seconds += elapsed
                        if indices is not None and indices.size:
                            key = f"d{candidate.binding.detection_index:03d}_a{len(action_ids)-1:02d}"
                            sparse[key] = indices
                            metadata["sparse"]["indices_artifact"] = {"path": f"sparse/{iid:012d}.npz", "key": key}
                        write_start = time.perf_counter()
                        write_line(actions_file, candidate.binding.metadata() | metadata | metrics | {
                            "annotation_id": association["annotation_id"], "labelable": gt is not None,
                            "baseline_success": success, "repair": repair, "damage": damage, "segmentation": rle})
                        io_seconds += time.perf_counter()-write_start
                    if gt is not None:
                        old, eq, local = (best[group]["mask_iou"] for group in GROUPS)
                        if local < eq or eq < old or old < baseline_iou:
                            raise RuntimeError("Nested finite oracle sets lost zero action or monotonicity")
                        local_delta, direction_delta = local-eq, eq-old
                        counts += [1, local_delta, direction_delta, int(success), int(not success), old, eq, local,
                                   int(not success and old >= .75), int(not success and eq >= .75),
                                   int(not success and local >= .75), int(not success and eq < .75 and local >= .75)]
                        opportunity = {"local_marginal_mask_iou": local_delta, "equal_direction_vs_old_mask_iou": direction_delta,
                                       "best_actions": best, "zero_increment_retained": local_delta == 0,
                                       "oracle_repair": {group: not success and best[group]["mask_iou"] >= .75 for group in GROUPS},
                                       "oracle_damage": {group: success and best[group]["mask_iou"] < .75 for group in GROUPS}}
                    else:
                        opportunity = {"local_marginal_mask_iou": None, "equal_direction_vs_old_mask_iou": None,
                                       "best_actions": best, "zero_increment_retained": None, "oracle_repair": None, "oracle_damage": None}
                    write_start = time.perf_counter()
                    write_line(candidates_file, candidate.identity | {"association": association, "baseline_original_area": int(candidate.baseline_original.sum()),
                                                                      "baseline_empty": not bool(candidate.baseline_original.any()),
                                                                      "components": candidate.component_counts, "action_ids": action_ids,
                                                                      "action_count": len(action_ids), "original_empty_actions": candidate_empties,
                                                                      "generation_seconds": candidate.generation_seconds, "opportunity": opportunity})
                    write_line(associations_file, association | opportunity)
                    io_seconds += time.perf_counter()-write_start
                if image_candidates != len(prepared.coefficients):
                    raise RuntimeError("Finite adapter did not preserve every normal native output candidate")
                if sparse:
                    write_start = time.perf_counter()
                    np.savez_compressed(run / "sparse" / f"{iid:012d}.npz", **sparse)
                    io_seconds += time.perf_counter()-write_start
                per_image.append(counts.tolist())
                image_record = {"image_id": iid, "source_image_sha256": item["sha256"], "native_candidate_count": len(prepared.coefficients),
                                "matched_candidate_count": image_matched if coco is not None else None, "labelable_candidate_count": image_labelable if coco is not None else None,
                                "unmatched_candidate_count": len(prepared.coefficients)-image_matched if coco is not None else None,
                                "action_count": image_actions, "original_empty_actions": image_empty,
                                "cluster_sums": counts.tolist() if coco is not None else None,
                                "cluster_columns": ["labelable", "local_delta_sum", "direction_delta_sum", "baseline_success", "baseline_failure",
                                                    "old_iou_sum", "equal_iou_sum", "local_iou_sum", "old_repair", "equal_repair", "local_repair", "local_additional_repair"] if coco is not None else None,
                                "checks": {key: adapter.checks[key]-before_checks[key] for key in adapter.checks},
                                "timing": forward | {"native_reference_decode_seconds": reference_decode_seconds,
                                                      "adapter_stages": {key: adapter.timings[key]-before_times[key] for key in adapter.timings},
                                                      "GT_association_scoring_seconds": image_gt_seconds if coco is not None else None,
                                                      "RLE_encoding_seconds": image_rle_seconds,
                                                      "total_image_elapsed_seconds": time.perf_counter()-image_start}}
                write_line(images_file, image_record)
                print(f"IMAGE {number}/{len(images)} id={iid} native={len(prepared.coefficients)} actions={image_actions} labelable={image_labelable if coco is not None else 'unknown'}", flush=True)
                del extracted, prepared, gt_cache, sparse
        frozen = extractor.verify_frozen()
        if len(per_image) != len(images) or args.mode == "opportunity" and len(per_image) != 512:
            raise RuntimeError("Complete locked panel did not finish")
        source_after = validate_source_lock(lock, panel, study)
        if source_after != source:
            raise RuntimeError("Runtime source changed during execution")
        statistics = bootstrap(per_image) if coco is not None else None
        if coco is not None:
            dump(run / "OPPORTUNITY_STATISTICS.json", statistics | {"investment_gate": gate_result(statistics)})
            dump(run / "ACTION_CONTROLS.json", {"definition": "each retained generated action against its own fixed baseline; local operation pools have action-level denominators", "controls": individual_controls})
        summary = {"status": "engineering_complete" if coco is None else "opportunity_complete", "passed": True,
                   "mode": args.mode, "image_count": len(images), "native_candidates": native_rows, "actions": actions_count,
                   "matched_candidates": matched if coco is not None else None, "labelable_matched_candidates": labelable if coco is not None else None,
                   "unmatched_candidates": native_rows-matched if coco is not None else None,
                   "matched_but_unlabelable_candidates": matched-labelable if coco is not None else None,
                   "unique_matched_GT_instances": len(unique_gt) if coco is not None else None,
                   "empty_actions_preserved_count": empty_count, "engineering_checks": adapter.checks,
                   "zero_native_input_original_parity_passed": True, "frozen_integrity": frozen,
                   "source_unchanged": True, "source_lock_sha256": sha256(run / "SOURCE_LOCK.json"),
                   "configuration_sha256": canonical_sha(configuration), "GT_parsed": coco is not None, "GT_used_for_action_generation": False,
                   "statistics": statistics, "investment_gate": gate_result(statistics) if statistics is not None else None,
                   "timing": {"native_model_seconds": model_seconds, "frozen_extract_wall_seconds": extract_seconds,
                              "native_reference_decode_seconds": native_seconds, "adapter_stages": adapter.timings,
                              "GT_association_scoring_seconds": gt_seconds if coco is not None else None,
                              "RLE_encoding_seconds": rle_seconds, "artifact_write_seconds": io_seconds,
                              "total_elapsed_seconds": time.perf_counter()-began,
                              "note": "synchronized device stage wall times; CPU generation/export/GT scoring measured separately; shared native work charged once"},
                   "peak_memory": peak_memory(torch), "started_at": started, "completed_at": now(),
                   "limitations": ["normal native filtered output candidates, not complete raw candidate geometry or AP",
                                   "many detections may share a GT; candidate mean and image-cluster interval use this fixed scope",
                                   "GT-selected finite action opportunity is artificial; no learned RGB method or deployable selector evaluated",
                                   "4-input-pixel bands can cover small objects; single connected component need not be a small perturbation",
                                   "this fixed finite configuration does not bound other local-edit method families"]}
        dump(run / "SUMMARY.json", summary)
        dump(run / "COMPLETE.json", {"status": summary["status"], "summary_sha256": sha256(run / "SUMMARY.json"), "completed_at": now()})
        print(f"COMPLETE {run / 'SUMMARY.json'}", flush=True)
        return 0
    except Exception as exc:
        dump(run / "FAILURE.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "time": now(),
                                  "metrics": None, "opportunity_gate": None, "partial_artifacts_retained": True})
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    raise SystemExit(main())
