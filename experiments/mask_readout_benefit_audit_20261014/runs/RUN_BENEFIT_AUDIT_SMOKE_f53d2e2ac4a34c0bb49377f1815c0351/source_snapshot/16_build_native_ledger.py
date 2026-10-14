"""Offline ledger from an unchanged native COCO prediction JSON; never runs YOLO."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import importlib.metadata
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared" / "evaluation"))
from posttrain_audit import finite, population, read_json, require, sha256, write_new

MODEL_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
GT_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
METRICS = ("AP", "AP50", "AP75", "APS", "APM", "APL",
           "AR1", "AR10", "AR100", "ARS", "ARM", "ARL")


def output_id(source_sha, index):
    return f"{source_sha}:{index}"  # zero-based ORIGINAL JSON row, not raw_id


def validate_predictions(predictions, images, categories):
    require(isinstance(predictions, list), "predictions must be a list")
    for i, p in enumerate(predictions):
        require(p["image_id"] in images and p["category_id"] in categories, f"row {i}: unknown image/category")
        finite(p["score"], 0, 1)
        require(len(p["bbox"]) == 4, f"row {i}: invalid bbox")
        for value in p["bbox"]:
            finite(value)
        require(p["bbox"][2] >= 0 and p["bbox"][3] >= 0, f"row {i}: negative box extent")
        seg = p["segmentation"]
        im = images[p["image_id"]]
        require(isinstance(seg, dict) and isinstance(seg.get("counts"), str),
                f"row {i}: expected compressed native RLE")
        require(seg["size"] == [im["height"], im["width"]], f"row {i}: wrong RLE shape")


def load_results(coco, predictions):
    # COCO.loadRes mutates rows and fails on []; never hand it the source objects.
    if predictions:
        return coco.loadRes(copy.deepcopy(predictions))
    from pycocotools.coco import COCO
    result = COCO()
    result.dataset = {"images": copy.deepcopy(coco.dataset["images"]),
                      "categories": copy.deepcopy(coco.dataset["categories"]), "annotations": []}
    result.createIndex()
    return result


def extract_matches(evaluator):
    """Use actual evaluator thresholds and IDs, never positional cross-task joins."""
    ts = [float(x) for x in evaluator.params.iouThrs]
    t75 = next(i for i, x in enumerate(ts) if abs(x - .75) < 1e-12)
    t50 = next(i for i, x in enumerate(ts) if abs(x - .50) < 1e-12)
    rows = {}
    for item in evaluator.evalImgs:
        if item is None or list(item["aRng"]) != list(evaluator.params.areaRng[0]):
            continue
        require(item["maxDet"] == 100, "unexpected evaluation budget")
        taken = set()
        for j, aid in enumerate(item["gtIds"]):
            pair = (int(item["image_id"]), int(aid))
            require(pair not in rows, "duplicate ordinary/ignored GT evaluation row")
            ignored = bool(item["gtIgnore"][j])
            d75 = int(item["gtMatches"][t75][j])
            if d75 and not ignored:
                require(d75 not in taken, "detection matched multiple ordinary GT")
                taken.add(d75)
                di = list(item["dtIds"]).index(d75)
                require(not bool(item["dtIgnore"][t75][di]), "ordinary match uses ignored detection")
                require(int(item["dtMatches"][t75][di]) == aid, "match arrays disagree")
            rows[pair] = {"ignore": ignored, "d75": d75,
                          "d50": int(item["gtMatches"][t50][j])}
    return rows


def evaluate_ledger(annotations, predictions, image_ids, source_sha):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks

    coco = COCO(str(annotations))
    id_set = set(image_ids)
    require(len(id_set) == len(image_ids) and bool(image_ids), "duplicate/empty image manifest")
    require(set(image_ids) <= set(coco.imgs), "unknown manifest image")
    require(all(p["image_id"] in id_set for p in predictions), "prediction outside image manifest")
    validate_predictions(predictions, coco.imgs, set(coco.cats))
    evaluations, matches = {}, {}
    for task in ("bbox", "segm"):
        # Both tasks load the SAME complete JSON including bbox: detection area policy stays bbox.
        gt = COCO(str(annotations))
        result = load_results(gt, predictions)
        require(len(result.anns) == len(predictions), "COCO loadRes changed row count")
        for index, p in enumerate(predictions):
            q = result.anns[index + 1]
            for field in ("image_id", "category_id", "score", "bbox", "segmentation"):
                require(p[field] == q[field], f"COCO row mapping changed: {index}/{field}")
        ev = COCOeval(gt, result, task)
        ev.params.imgIds = list(image_ids)
        ev.params.useCats = 1
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        evaluations[task] = dict(zip(METRICS, [100 * float(v) if v >= 0 else None for v in ev.stats]))
        matches[task] = extract_matches(ev)

    expected = {(a["image_id"], a["id"]) for a in coco.dataset["annotations"] if a["image_id"] in id_set}
    require(set(matches["bbox"]) == expected == set(matches["segm"]), "GT coverage mismatch")
    grouped = defaultdict(list)
    for index, p in enumerate(predictions):
        grouped[(p["image_id"], p["category_id"])].append(index)
    by_group = defaultdict(list)
    for pair in sorted(expected):
        ann = coco.anns[pair[1]]
        by_group[(ann["image_id"], ann["category_id"])].append(ann)

    ledger = []
    for group, anns in by_group.items():
        indexes = grouped[group]
        # Stable score sort, ties retain original JSON order; maxDets is per image AND category.
        ranked = sorted(indexes, key=lambda i: -predictions[i]["score"])
        budget = set(ranked[:100])
        gt_rle = [coco.annToRLE(a) for a in anns]
        pred = [predictions[i] for i in indexes]
        crowd = [int(a.get("iscrowd", 0)) for a in anns]
        miou = masks.iou([p["segmentation"] for p in pred], gt_rle, crowd) if pred else np.zeros((0, len(anns)))
        biou = masks.iou([p["bbox"] for p in pred], [a["bbox"] for a in anns], crowd) if pred else np.zeros((0, len(anns)))
        loc = {index + 1: j for j, index in enumerate(indexes)}
        for j, ann in enumerate(anns):
            pair = (ann["image_id"], ann["id"])
            b, s = matches["bbox"][pair], matches["segm"][pair]
            require(b["ignore"] == s["ignore"], "bbox/segm ordinary GT domains differ")
            bid, sid, ignored = b["d75"], s["d75"], b["ignore"]
            def oid(dt):
                if not dt:
                    return None
                require(dt in loc, "matched detection has wrong image/category")
                return output_id(source_sha, dt - 1)
            row = dict(image_id=pair[0], annotation_id=pair[1],
                       gt_category_id=ann["category_id"], gt_area=float(ann["area"]),
                       is_ignored=ignored,
                       task_mask75="ignored" if ignored else "success" if sid else "failure",
                       task_mask50="ignored" if ignored else "success" if s["d50"] else "failure",
                       task_box75="ignored" if ignored else "success" if bid else "failure",
                       bbox75_match_output_id=oid(bid), segm75_match_output_id=oid(sid),
                       box_iou_of_bbox_match=float(biou[loc[bid], j]) if bid else None,
                       mask_iou_of_bbox_match=float(miou[loc[bid], j]) if bid else None,
                       bbox_match_score=predictions[bid - 1]["score"] if bid else None,
                       bbox_match_category_id=predictions[bid - 1]["category_id"] if bid else None,
                       exists_same_output_joint75_for_gt=any(
                           miou[k, j] >= .75 and biou[k, j] >= .75 for k in range(len(indexes))),
                       correct_class_mask75_present_pre_coco_budget=any(
                           miou[k, j] >= .75 for k in range(len(indexes))),
                       correct_class_mask75_present_within_coco_maxdet100=any(
                           miou[k, j] >= .75 and index in budget for k, index in enumerate(indexes)),
                       source_prediction_sha=source_sha, raw_id=None)
            row["population"] = population(row)
            ledger.append(row)
    ledger.sort(key=lambda r: (r["image_id"], r["annotation_id"]))
    return ledger, evaluations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "predictions", "image-ids", "contract", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--scope", choices=("smoke", "full"), required=True)
    ap.add_argument("--native-summary", type=Path)
    args = ap.parse_args()
    ids, predictions, contract = read_json(args.image_ids), read_json(args.predictions), read_json(args.contract)
    require(isinstance(ids, list), "image-ids must contain a JSON integer list")
    hashes = {k: sha256(getattr(args, k)) for k in ("annotations", "predictions", "image_ids", "contract")}
    for k in ("annotations", "predictions", "image_ids"):
        require(contract.get(k + "_sha256") == hashes[k], f"contract {k} hash mismatch")
    for k, v in {"ultralytics_version": "8.4.100", "model_sha256": MODEL_SHA,
                 "end2end": True, "agnostic_nms": False, "conf": .001, "max_det": 300,
                 "candidate_policy": "inference_output", "gt_assisted_inference": False}.items():
        require(k in contract and contract[k] == v, f"contract mismatch: {k}")
    for k in ("run_id", "code_commit", "decoder", "input_geometry_manifest_sha256"):
        require(bool(contract.get(k)), f"contract missing: {k}")
    if args.scope == "full":
        require(len(ids) == 5000 and hashes["annotations"] == GT_SHA, "full COCO identity mismatch")
        require(set(ids) == {r["id"] for r in read_json(args.annotations)["images"]}, "incomplete COCO image set")
    else:
        require(0 < len(ids) <= 16, "smoke requires 1..16 images")
    require(not args.out.exists(), "output directory exists; use a new run/subdirectory")
    args.out.mkdir(parents=True)
    with (args.out / "cocoeval.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            ledger, metrics = evaluate_ledger(args.annotations, predictions, ids, hashes["predictions"])
    ordinary = [r for r in ledger if not r["is_ignored"]]
    if args.scope == "full":
        require(len(ordinary) == 36335, "ordinary GT denominator differs from frozen protocol")
    for row in ledger:
        row["eval_contract_sha"] = hashes["contract"]
        row["branch"] = "one2one"
    with (args.out / "PER_GT_FAILURE.jsonl").open("x", encoding="utf-8") as out:
        for row in ledger:
            out.write(json.dumps(row, allow_nan=False) + "\n")
    with (args.out / "OUTPUT_ID_MAP.jsonl").open("x", encoding="utf-8") as out:
        for i, p in enumerate(predictions):
            out.write(json.dumps({"output_detection_id": output_id(hashes["predictions"], i),
                "source_pred_json_index": i, "coco_dt_id": i + 1, "raw_id": None,
                "image_id": p["image_id"], "category_id": p["category_id"]}) + "\n")
    backend = "unadjudicated_missing_native_summary"
    if args.native_summary:
        native = read_json(args.native_summary)
        require(native["unit"] == "AP_points" and native["backend"] == "COCOeval_faster",
                "native summary needs AP_points and COCOeval_faster")
        for k in ("annotations", "predictions", "image_ids"):
            require(native.get(k + "_sha256") == hashes[k], "native summary source mismatch: " + k)
        diffs = []
        for task in ("bbox", "segm"):
            for metric in METRICS:
                value = metrics[task][metric]
                other = native[task][metric]
                require((value is None) == (other is None), "native backend metric availability differs")
                if value is not None:
                    diffs.append(abs(value - finite(other, 0, 100)))
        backend = "passed" if max(diffs, default=0) <= .02 else "failed"
    parity = "not_applicable_smoke"
    if args.scope == "full":
        # Decimal display parity only; not a certificate that all unpublished settings match.
        parity = "passed" if (round(metrics["bbox"]["AP"], 1), round(metrics["segm"]["AP"], 1)) == (52.5, 44.1) else "failed"
    summary = {"environment": {"python": sys.version, "numpy": importlib.metadata.version("numpy"),
               "pycocotools": importlib.metadata.version("pycocotools"), "script_sha256": sha256(__file__)},
               "scope": args.scope, "inputs_sha256": hashes, "metrics_unit": "AP_points",
               "metrics": metrics, "images": len(ids), "ordinary_gt": len(ordinary),
               "mask75_success": sum(r["task_mask75"] == "success" for r in ordinary),
               "offline_identity_and_ledger_checks": "passed", "coco_backend_parity": backend,
               "published_baseline_parity": parity,
               "native_gt_ledger_validity": "unadjudicated_pending_native_run_and_smoke_review",
               "limits": ["This utility checks files, not whether YOLO really executed with the declared contract.",
                          "Native run logs, source/geometry receipts and required smoke tests remain separate evidence."]}
    write_new(args.out / "SUMMARY.json", summary)
    write_new(args.out / "ARTIFACT_HASHES.json", {
        p.name: sha256(p) for p in args.out.iterdir() if p.is_file()})
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 1 if backend == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
