"""S056: fixed-slot error repair ceilings and prediction-only competition.

The oracle arms use evaluation GT explicitly. They estimate recoverable AP for
the fixed attribution, not an achievable model or additive AP decomposition.
"""
import os
import sys
from pathlib import Path

if os.name == "nt":
    os.environ["PATH"] = str(Path(sys.prefix) / "Library/bin") + os.pathsep + os.environ.get("PATH", "")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import argparse
import contextlib
import csv
import gc
import gzip
import hashlib
import io
import json
import shutil
import time
from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "diagnostics/frozen_readouts_fullval_20260912_v2"
CENSUS = ROOT / "diagnostics/mask_geometry_failure_census_20260912/instances_classified.csv"
RELATION = ROOT / "diagnostics/coco_relation_taxonomy_score010_20260913/gt_relation_taxonomy.csv"
ANNOTATION = ROOT.parents[1] / "datasets/coco/annotations/instances_val2017.json"
ARMS = ["original", "oracle_same_remove", "oracle_background_remove",
        "oracle_other_remove", "oracle_own_complete", "oracle_all_fp_remove",
        "oracle_goodbox_replace", "score_competition", "center_competition"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def save_gz(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=1) as handle:
        json.dump(value, handle, separators=(",", ":"), allow_nan=False)
    temp.replace(path)


def encode(mask):
    result = mu.encode(np.asfortranarray(mask, dtype=np.uint8))
    result["counts"] = result["counts"].decode("ascii")
    return result


def error_partition(pred, own, same, different, crowd):
    # Own GT always wins. Ambiguous same/different overlap is an explicit bin.
    valid = ~crowd
    fp = pred & ~own & valid
    parts = {
        "target_fn": ~pred & own & valid,
        "same_only": fp & same & ~different,
        "other_only": fp & different & ~same,
        "ambiguous_neighbor": fp & same & different,
        "background": fp & ~same & ~different,
    }
    counts = {k: int(v.sum()) for k, v in parts.items()}
    assert sum(counts[k] for k in counts if k != "target_fn") == int(fp.sum())
    return parts, counts


def compete(predictions, boxes, decoded, score_min, kind):
    """No GT, GT matching, density or failure-state inputs are accepted."""
    groups = defaultdict(list)
    for slot, pred in enumerate(predictions):
        if pred["score"] >= score_min:
            groups[pred["category_id"]].append(slot)
    patches = {}
    for slots in groups.values():
        if len(slots) < 2:
            continue
        for slot in slots:
            if slot not in decoded:
                decoded[slot] = mu.decode(predictions[slot]["segmentation"]).astype(bool)
        shape = decoded[slots[0]].shape
        best = np.full(shape, -np.inf, np.float32)
        winner = np.full(shape, -1, np.int32)
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        for slot in slots:
            if kind == "score":
                value = np.float32(predictions[slot]["score"])
            else:
                x, y, w, h = boxes[slot]["bbox"]
                value = -(((xx - x - w / 2) / max(w / 2, 1)) ** 2
                          + ((yy - y - h / 2) / max(h / 2, 1)) ** 2).astype(np.float32)
            better = decoded[slot] & (value > best)
            best[better] = value if np.ndim(value) == 0 else value[better]
            winner[better] = slot
        for slot in slots:
            edited = decoded[slot] & (winner == slot)
            if np.any(edited != decoded[slot]):
                patches[str(slot)] = encode(edited)
    return patches


def evaluate(gt, predictions, ids, metadata, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, "segm")
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    t = int(np.flatnonzero(np.isclose(ev.params.iouThrs, .75))[0])
    records, scores, tp = [], [], []
    for item in ev.evalImgs:
        if item is None or item["aRng"] != [0, 1e10] or item["maxDet"] != 100:
            continue
        score_map = dict(zip(item["dtIds"], item["dtScores"]))
        valid = ~item["dtIgnore"][t].astype(bool)
        scores.extend(np.asarray(item["dtScores"])[valid].tolist())
        tp.extend((item["dtMatches"][t][valid] > 0).tolist())
        for j, aid in enumerate(item["gtIds"]):
            if item["gtIgnore"][j]:
                continue
            meta = metadata[int(aid)]
            match = int(item["gtMatches"][t, j])
            records.append(dict(annotation_id=int(aid), image_id=item["image_id"],
                                category_id=int(meta["category_id"]), area_bin=meta["area_bin"],
                                density=meta["mask_density"], ici_high=meta["ici_high"] == "True",
                                hit75=bool(match), score75=score_map[match] if match else None))
    selected_ids = set(ids)
    expected = {aid for aid, row in metadata.items() if int(row["image_id"]) in selected_ids}
    assert len(records) == len(expected) and {r["annotation_id"] for r in records} == expected
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    ctp = np.cumsum(np.asarray(tp, np.int64)[order])
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    eligible = ends[(ctp[ends] / (ends + 1) >= .9) & (ctp[ends] > 0)]
    threshold = float(scores[eligible[-1]]) if len(eligible) else None
    for row in records:
        row["hit90"] = bool(threshold is not None and row["hit75"] and row["score75"] >= threshold)
    summary = dict(arm=arm, mask_ap=float(ev.stats[0]), mask_ap50=float(ev.stats[1]),
                   mask_ap75=float(ev.stats[2]), gt=len(records), predictions=len(predictions),
                   p90_threshold=threshold)
    for name in ("all", "low", "middle", "high", "undefined", "ici_high"):
        subset = [r for r in records if name == "all" or (r["ici_high"] if name == "ici_high" else r["density"] == name)]
        summary["n_" + name] = len(subset)
        for key in ("hit75", "hit90"):
            summary[key + "_" + name] = float(np.mean([r[key] for r in subset])) if subset else None
    summary["gap_low_high"] = summary["hit75_low"] - summary["hit75_high"]
    return summary, records


def summarize_relations(metadata, relations):
    rows = []
    for aid, meta in metadata.items():
        relation = relations[aid]["relation"]
        rows.append(dict(annotation_id=aid, image_id=int(meta["image_id"]), relation=relation,
                         density=meta["mask_density"], state=meta["state"],
                         fixed_slot=int(meta["prediction_slot"]), score=float(meta["score"]) if meta["score"] else None,
                         dominant_error=meta["dominant_error"],
                         eligible_goodbox_badmask=meta["state"] == "box_good__mask_bad"))
    result = {}
    for group in ("all", "low", "middle", "high", "undefined"):
        selected = [r for r in rows if group == "all" or r["density"] == group]
        mx = [r for r in selected if r["relation"] in ("M", "X")]
        eligible = [r for r in selected if r["eligible_goodbox_badmask"]]
        result[group] = dict(n=len(selected), states=dict(Counter(r["state"] for r in selected)),
                             mx_n=len(mx), mx_fixed_slot_states=dict(Counter(r["state"] for r in mx)),
                             eligible_n=len(eligible),
                             eligible_dominant=dict(Counter(r["dominant_error"] for r in eligible)),
                             mx_eligible_dominant=dict(Counter(r["dominant_error"] for r in mx if r["eligible_goodbox_badmask"])))
    return rows, result


def paired_stats(original, new, count=2000):
    lookup = {r["annotation_id"]: r for r in original}
    image_ids = sorted({r["image_id"] for r in original})
    image_idx = {iid: i for i, iid in enumerate(image_ids)}
    numer = np.zeros((len(image_ids), 2))
    denom = np.zeros_like(numer)
    result = {}
    for group in ("all", "low", "middle", "high"):
        values = np.zeros((len(image_ids), 2))
        for row in new:
            if group != "all" and row["density"] != group:
                continue
            ii = image_idx[row["image_id"]]
            values[ii] += [int(row["hit75"]) - int(lookup[row["annotation_id"]]["hit75"]), 1]
        rng = np.random.default_rng(20260913)
        # Image clusters preserve correlation between multiple instances.
        draws = rng.integers(len(image_ids), size=(count, len(image_ids)))
        sums = values[draws].sum(1)
        delta = sums[:, 0] / np.maximum(sums[:, 1], 1)
        result[group] = dict(delta=float(values[:, 0].sum() / max(values[:, 1].sum(), 1)),
                             ci95=np.quantile(delta, [.025, .975]).tolist())
        if group in ("low", "high"):
            col = 0 if group == "low" else 1
            numer[:, col], denom[:, col] = values[:, 0], values[:, 1]
    draws = np.random.default_rng(20260913).integers(len(image_ids), size=(count, len(image_ids)))
    ns, ds = numer[draws].sum(1), denom[draws].sum(1)
    gap = ns[:, 0] / np.maximum(ds[:, 0], 1) - ns[:, 1] / np.maximum(ds[:, 1], 1)
    result["gap_low_high"] = dict(delta=result["low"]["delta"] - result["high"]["delta"],
                                   ci95=np.quantile(gap, [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    with CENSUS.open(encoding="utf-8-sig") as handle:
        metadata = {int(r["annotation_id"]): r for r in csv.DictReader(handle)}
    with RELATION.open(encoding="utf-8-sig") as handle:
        relations = {int(r["annotation_id"]): r for r in csv.DictReader(handle)}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANNOTATION))
    ids = sorted(gt.imgs)[:args.limit or None]
    byimage = defaultdict(list)
    for row in metadata.values():
        byimage[int(row["image_id"])].append(row)
    protocol = dict(experiment="S056_FIXED_SLOT_ERROR_AP", images=ids, arms=ARMS,
                    script_sha256=sha(__file__), census_sha256=sha(CENSUS), relation_sha256=sha(RELATION),
                    annotation_sha256=sha(ANNOTATION), receipts_sha256=sha(SOURCE / "image_receipts.json"),
                    candidate_scope="All original final slots, scores/classes/order retained, including masks emptied by intervention.",
                    oracle_scope="Only S048 SAME-slot BoxIoU>=.75 and MaskIoU<.75, original official bbox50 one-to-one ownership. All other predictions unchanged.",
                    oracle_pixels="Exclusive same-only/other-only/shared-neighbor/background FP bins, own GT wins, crowd pixels untouched. same_remove includes shared-neighbor FP; other_remove only different-exclusive. Own completion adds full visible own GT, possibly outside original crop; goodbox_replace is exact GT and not a feasible decoder promise.",
                    automatic="Only original predictions score>=.10 compete among same class positive masks. Score priority or smallest normalized squared distance to predicted box center. All choices use predictions alone; no GT/ICI/failure gating; no tuning.",
                    metrics="Official COCOeval masks all image/category/size/IoU thresholds; all ordinary GT matched before E(4) strata, one pooled P90. Paired image bootstrap2000 for R75/gap only, no AP significance claim.",
                    taxonomy_correction="S055 union-of-component predictions measures neighbors' correct masks as target leakage; withdraw error-surface interpretation. Join relation topology to S048 fixed-slot errors instead. Equal denominators; no C/I/L/S to IoU75 equivalence.",
                    limitations="Explored COCO val, oracle scores are descriptive attainable repairs for fixed assignments, neither deployable nor additive; automatic controls are standard baselines, not novelty.",
                    training=False, model_inference=False)
    if (out / "protocol.json").exists():
        assert json.loads((out / "protocol.json").read_text()) == protocol, "Protocol changed; use new output"
    else:
        dump(out / "protocol.json", protocol)
        shutil.copy2(__file__, out / Path(__file__).name)
    joined, counts = summarize_relations(metadata, relations)
    save_csv(out / "corrected_relation_join.csv", joined)
    dump(out / "corrected_relation_summary.json", counts)
    receipts = json.loads((SOURCE / "image_receipts.json").read_text())
    start = time.monotonic()
    details, edit_counts = [], Counter()
    for number, iid in enumerate(ids, 1):
        path = out / "images" / f"{iid}.json.gz"
        if path.exists():
            saved = read_gz(path)
            details.extend(saved["details"])
            edit_counts.update({k: len(v) for k, v in saved["patches"].items()})
            continue
        source_path = SOURCE / "images" / f"{iid}.json.gz"
        if sha(source_path) != receipts[str(iid)]["sha256"]:
            raise RuntimeError("Frozen prediction cache changed")
        source = read_gz(source_path)
        preds, boxes = source["original"], source["original_boxes"]
        assert len(preds) == len(boxes)
        assert all(p["category_id"] == b["category_id"] and p["score"] == b["score"] for p, b in zip(preds, boxes))
        shape = (gt.imgs[iid]["height"], gt.imgs[iid]["width"])
        decoded = {}
        patches = {arm: {} for arm in ARMS if arm != "original"}
        patches["score_competition"] = compete(preds, boxes, decoded, .1, "score")
        patches["center_competition"] = compete(preds, boxes, decoded, .1, "center")
        targets = [r for r in byimage[iid] if r["state"] == "box_good__mask_bad"]
        image_details = []
        if targets:
            anns = gt.imgToAnns[iid]
            masks = {ann["id"]: gt.annToMask(ann).astype(bool) for ann in anns}
            crowd = np.zeros(shape, bool)
            for ann in anns:
                if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                    crowd |= masks[ann["id"]]
            used = set()
            for row in targets:
                aid, slot = int(row["annotation_id"]), int(row["prediction_slot"])
                assert slot not in used
                used.add(slot)
                assert abs(preds[slot]["score"] - float(row["score"])) < 1e-12, (iid, aid, slot, preds[slot]["score"], row["score"])
                assert preds[slot]["category_id"] == gt.anns[aid]["category_id"]
                own = masks[aid]
                if slot not in decoded:
                    decoded[slot] = mu.decode(preds[slot]["segmentation"]).astype(bool)
                pred = decoded[slot]
                iou = int((pred & own).sum()) / max(int((pred | own).sum()), 1)
                assert abs(iou - float(row["mask_iou"])) < 1e-10
                same, different = np.zeros(shape, bool), np.zeros(shape, bool)
                for ann in anns:
                    if ann["id"] == aid or ann.get("iscrowd", 0) or ann.get("ignore", 0):
                        continue
                    if ann["category_id"] == gt.anns[aid]["category_id"]:
                        same |= masks[ann["id"]]
                    else:
                        different |= masks[ann["id"]]
                part, errors = error_partition(pred, own, same, different, crowd)
                valid_area = max(int((own & ~crowd).sum()), 1)
                assert abs((errors["same_only"] + errors["ambiguous_neighbor"]) / valid_area - float(row["same_neighbor_error"])) < 1e-10
                variants = {
                    "oracle_same_remove": pred & ~(part["same_only"] | part["ambiguous_neighbor"]),
                    "oracle_other_remove": pred & ~part["other_only"],
                    "oracle_background_remove": pred & ~part["background"],
                    "oracle_own_complete": pred | part["target_fn"],
                    "oracle_all_fp_remove": pred & (own | crowd),
                    "oracle_goodbox_replace": own,
                }
                rr = dict(annotation_id=aid, image_id=iid, slot=slot, density=row["mask_density"],
                          category_id=int(row["category_id"]), area_bin=row["area_bin"],
                          relation=relations[aid]["relation"], box_iou=float(row["box_iou"]),
                          original_iou=iou, valid_gt_pixels=valid_area,
                          dominant_disjoint_error=max(errors, key=errors.get), **errors)
                for arm, edited in variants.items():
                    rr[arm + "_iou"] = int((edited & own).sum()) / max(int((edited | own).sum()), 1)
                    if np.any(edited != pred):
                        patches[arm][str(slot)] = encode(edited)
                image_details.append(rr)
        saved = dict(original=preds, patches=patches, details=image_details)
        save_gz(path, saved)
        details.extend(image_details)
        edit_counts.update({k: len(v) for k, v in patches.items()})
        if number % 100 == 0 or number == len(ids):
            state = dict(stage="PREPARE", images=number, total=len(ids), eligible=len(details),
                         seconds=round(time.monotonic() - start, 2), pid=os.getpid())
            dump(out / "progress.json", state)
            print(json.dumps(state), flush=True)
    save_csv(out / "fixed_slot_repairs.csv", details)
    dump(out / "PREPARED.json", dict(images=len(ids), eligible=len(details), changed_slots=dict(edit_counts)))
    if args.prepare_only:
        return
    task, all_records = [], {}
    for arm in ARMS:
        receipt = out / f"{arm}.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            with (out / f"{arm}_gt.csv").open() as handle:
                records = list(csv.DictReader(handle))
            for r in records:
                for key in ("annotation_id", "image_id", "category_id"):
                    r[key] = int(r[key])
                for key in ("ici_high", "hit75", "hit90"):
                    r[key] = r[key] == "True"
        else:
            predictions = []
            for iid in ids:
                saved = read_gz(out / "images" / f"{iid}.json.gz")
                for slot, pred in enumerate(saved["original"]):
                    patch = saved["patches"].get(arm, {}).get(str(slot))
                    predictions.append(dict(pred, segmentation=patch) if patch is not None else pred)
            print(json.dumps(dict(stage="EVALUATE", arm=arm, predictions=len(predictions))), flush=True)
            summary, records = evaluate(gt, predictions, ids, metadata, arm)
            if arm == "original" and len(ids) == 5000:
                reference = json.loads((SOURCE / "evaluation/original.json").read_text())["summary"]
                assert abs(summary["mask_ap"] - reference["mask_ap"]) < 1e-12
            value = dict(summary=summary)
            save_csv(out / f"{arm}_gt.csv", records)
            dump(receipt, value)
            del predictions
            gc.collect()
        task.append(value["summary"])
        all_records[arm] = records
        save_csv(out / "task_summary.csv", task)
        print(json.dumps(value["summary"]), flush=True)
        dump(out / "progress.json", dict(stage="EVALUATE", complete=len(task), total=len(ARMS), pid=os.getpid()))
    stats = {arm: paired_stats(all_records["original"], all_records[arm]) for arm in ARMS[1:]}
    dump(out / "paired_stats.json", stats)
    dump(out / "COMPLETE.json", dict(status="COMPLETE", images=len(ids), eligible=len(details),
                                      arms=ARMS, seconds=round(time.monotonic() - start, 3),
                                      protocol_sha256=sha(out / "protocol.json"),
                                      task_summary_sha256=sha(out / "task_summary.csv")))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
