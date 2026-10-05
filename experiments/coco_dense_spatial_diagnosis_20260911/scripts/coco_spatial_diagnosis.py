"""GT-grounded spatial error diagnosis on existing COCO prediction RLEs.

Primary ownership is official same-category bbox IoU>=.5 matching. Secondary
ownership is the previous official mask IoU>=.5 matching. Unmatched GT are
retained as missing diagnostic observations, never rewarded for zero leakage.
"""
import argparse
import gc
import inspect
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

from coco_instance_metrics import ROOT, digest, load_metadata, make_dt, write_csv, write_json

PIXEL_METRICS = ["coverage", "neighbor", "same_neighbor", "background", "purity", "mask_iou", "official_mask_iou", "box_iou", "gt_inside_box", "box_center_error"]
DECOMP = ["coverage_contribution", "neighbor_contribution", "background_contribution"]


def area(rle):
    return int(mu.area(rle))


def intersection(*rles):
    return area(mu.merge(list(rles), intersect=True))


def spatial_metrics(pred, own, union, same_union, crowd=None):
    """Disjoint pixel accounting, with crowd regions excluded from all sets."""
    ga, pa = area(own), area(pred)
    tp, in_all, in_same = intersection(pred, own), intersection(pred, union), intersection(pred, same_union)
    if crowd is not None:
        ga -= intersection(own, crowd)
        pa -= intersection(pred, crowd)
        tp -= intersection(pred, own, crowd)
        in_all -= intersection(pred, union, crowd)
        in_same -= intersection(pred, same_union, crowd)
    if ga == 0:
        return None
    neighbor, same_neighbor, background = in_all - tp, in_same - tp, pa - in_all
    assert min(tp, neighbor, same_neighbor, background, pa, ga) >= 0
    assert same_neighbor <= neighbor and tp <= ga
    assert tp + neighbor + background == pa
    coverage, n, b = tp / ga, neighbor / ga, background / ga
    iou = tp / (ga + pa - tp)
    assert abs(iou - coverage / (1 + n + b)) < 1e-12
    return {"gt_valid_pixels": ga, "pred_valid_pixels": pa, "target_pixels": tp,
            "neighbor_pixels": neighbor, "same_neighbor_pixels": same_neighbor, "background_pixels": background,
            "coverage": coverage, "neighbor": n, "same_neighbor": same_neighbor / ga, "background": b,
            "purity": tp / pa if pa else None, "mask_iou": iou}


def box_metrics(pred_box, gt_box, own, crowd, shape):
    p, g = np.array(pred_box, dtype=float), np.array(gt_box, dtype=float)
    p2, g2 = p[:2] + p[2:], g[:2] + g[2:]
    wh = np.maximum(0, np.minimum(p2, g2) - np.maximum(p[:2], g[:2]))
    overlap = wh.prod()
    iou = overlap / (p[2:].prod() + g[2:].prod() - overlap)
    center = np.linalg.norm((p[:2] + p[2:] / 2) - (g[:2] + g[2:] / 2)) / np.sqrt(g[2:].prod())
    polygon = [p[0], p[1], p2[0], p[1], p2[0], p2[1], p[0], p2[1]]
    rect = mu.merge(mu.frPyObjects([polygon], *shape))
    gt_a, inside = area(own), intersection(own, rect)
    if crowd is not None:
        gt_a -= intersection(own, crowd)
        inside -= intersection(own, rect, crowd)
    return {"box_iou": float(iou), "box_center_error": float(center),
            "gt_inside_box": inside / gt_a if gt_a else None}


def decompose(base, ccl):
    """Exact symmetric finite-change identity, not a causal mediation estimate."""
    rb, rc = base["coverage"], ccl["coverage"]
    fb, fc = base["neighbor"] + base["background"], ccl["neighbor"] + ccl["background"]
    coverage = .5 * (1/(1+fb) + 1/(1+fc)) * (rc-rb)
    multiplier = -.5 * (rb+rc) / ((1+fb)*(1+fc))
    neighbor = multiplier * (ccl["neighbor"]-base["neighbor"])
    background = multiplier * (ccl["background"]-base["background"])
    assert abs(coverage + neighbor + background - (ccl["mask_iou"] - base["mask_iou"])) < 1e-12
    return dict(zip(DECOMP, [coverage, neighbor, background]))


def box_assignment(gt, preds, aids):
    ev = COCOeval(gt, make_dt(gt, preds, "bbox"), "bbox")
    ev.params.imgIds = sorted(gt.imgs)
    ev.params.iouThrs = np.array([.5])
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [100]
    ev.evaluate()
    result = np.zeros(len(aids), dtype=np.int64)
    index = {int(a): i for i, a in enumerate(aids)}
    seen = set()
    for r in ev.evalImgs:
        if r is None:
            continue
        for j, aid in enumerate(r["gtIds"]):
            if r["gtIgnore"][j]:
                continue
            assert aid not in seen
            seen.add(aid)
            result[index[aid]] = int(r["gtMatches"][0, j])
    assert seen == set(index)
    selected = result[result > 0]
    assert len(selected) == len(set(selected))
    return result


def sample_stats(rows, field, selected, image_lookup, weights):
    valid = [r for r in rows if selected(r) and r.get("base_"+field) is not None and r.get("ccl_"+field) is not None]
    if not valid:
        return None
    b = np.array([r["base_"+field] for r in valid])
    c = np.array([r["ccl_"+field] for r in valid])
    ids = np.array([image_lookup[r["image_id"]] for r in valid])
    n = len(image_lookup)
    count = np.bincount(ids, minlength=n)
    num = np.bincount(ids, weights=c-b, minlength=n)
    denom = weights @ count
    active = denom > 0
    boot = (weights[active] @ num) / denom[active]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": len(valid), "images": len(set(ids)), "baseline": float(b.mean()), "ccl": float(c.mean()),
            "delta": float((c-b).mean()), "ci95": [float(lo), float(hi)], "valid_draws": int(active.sum())}


def component_stats(rows, field, selected, image_lookup, weights):
    valid = [r for r in rows if selected(r) and r.get(field) is not None]
    if not valid:
        return None
    ids = np.array([image_lookup[r["image_id"]] for r in valid])
    values = np.array([r[field] for r in valid])
    count = np.bincount(ids, minlength=len(image_lookup))
    sums = np.bincount(ids, weights=values, minlength=len(image_lookup))
    denom = weights @ count
    boot = (weights[denom > 0] @ sums) / denom[denom > 0]
    return {"n": len(valid), "mean": float(values.mean()), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}


def summarize(rows, meta, out, previous_run):
    draws = np.load(previous_run / "bootstrap_draws.npz", allow_pickle=False)
    np.testing.assert_array_equal(draws["image_ids"], meta["image_ids"])
    weights = np.asarray([np.bincount(r, minlength=len(meta["image_ids"])) for r in draws["sampled_image_indices"]], dtype=np.int32)
    image_lookup = {int(i): k for k, i in enumerate(meta["image_ids"])}
    groups = {"All": lambda r: True, "H": lambda r: r["ici"] > .5, "L": lambda r: r["ici"] <= .5}
    for state in ("gained", "lost", "retained", "unrecovered"):
        groups["H_"+state] = lambda r, s=state: r["ici"] > .5 and r["outcome75"] == s
    result, coverage, tables = {}, [], []
    for assignment in ("box50", "mask50"):
        assigned = [r for r in rows if r["assignment"] == assignment]
        result[assignment] = {}
        for label, select in groups.items():
            group_rows = [r for r in assigned if select(r)]
            coverage.append({"assignment": assignment, "group": label, "total_gt": len(group_rows),
                             "both_matched": sum(r["base_detection"] > 0 and r["ccl_detection"] > 0 for r in group_rows),
                             "only_baseline": sum(r["base_detection"] > 0 and r["ccl_detection"] == 0 for r in group_rows),
                             "only_ccl": sum(r["base_detection"] == 0 and r["ccl_detection"] > 0 for r in group_rows),
                             "neither": sum(r["base_detection"] == 0 and r["ccl_detection"] == 0 for r in group_rows),
                             "both_pixel_valid": sum(r.get("coverage_contribution") is not None for r in group_rows),
                             "same_output_as_mask50_both": sum(r["agrees_with_mask50_both"] for r in group_rows),
                             "selected_transition_agrees": sum(r.get("selected_transition_agrees") is True for r in group_rows)})
            metrics = {}
            for field in PIXEL_METRICS:
                stat = sample_stats(group_rows, field, lambda r: True, image_lookup, weights)
                metrics[field] = stat
                if stat:
                    tables.append({"assignment": assignment, "group": label, "metric": field, **{k: v for k, v in stat.items() if k != "ci95"}, "ci_low": stat["ci95"][0], "ci_high": stat["ci95"][1]})
            metrics["decomposition"] = {f: component_stats(group_rows, f, lambda r: True, image_lookup, weights) for f in DECOMP}
            result[assignment][label] = metrics
        print(f"Summarized {assignment} diagnostics", flush=True)
    # Composition is described over all GT outcomes, not only diagnostic matches.
    composition = []
    high = [r for r in rows if r["assignment"] == "box50" and r["ici"] > .5]
    for state in ("gained", "lost", "retained", "unrecovered"):
        selected = [r for r in high if r["outcome75"] == state]
        for cid in sorted({r["category_id"] for r in selected}):
            for size in ("small", "medium", "large"):
                c = sum(r["category_id"] == cid and r["size"] == size for r in selected)
                if c:
                    composition.append({"outcome75": state, "category_id": cid, "size": size, "gt": c})
    write_csv(out / "diagnostic_matching_coverage.csv", coverage)
    write_csv(out / "spatial_metric_summary.csv", tables)
    write_csv(out / "outcome_category_size.csv", composition)
    write_json(out / "summary.json", {"groups": result, "matching_coverage": coverage,
                                      "scope": "cropped cached masks, conditional paired spatial diagnostics; not raw coefficient causality",
                                      "bootstrap": 2000, "draws_source": str(previous_run / "bootstrap_draws.npz")})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-run", type=Path, default=ROOT / "gemini/results/coco_instance_eval/20260911_034617")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gt_path = ROOT / "gemini/data/coco_dense/coco_dense_val_gt.json"
    manifest = ROOT / "refine-logs/coco-evaluation/COCO_INSTANCE_MANIFEST_20260911_032703.csv"
    data, anns, meta = load_metadata(gt_path, manifest)
    prior_meta = json.loads((args.previous_run / "run.json").read_text(encoding="utf-8"))
    assert digest(gt_path) == prior_meta["gt_sha256"] and digest(manifest) == prior_meta["manifest_sha256"]
    masks, boxes, predictions = {}, {}, {}
    run = {"started": datetime.now().isoformat(), "script_sha256": digest(__file__), "gt_sha256": digest(gt_path),
           "previous_run": str(args.previous_run), "previous_summary_sha256": digest(args.previous_run / "summary.json"),
           "protocol": "All GT retained. Official bbox IoU50 assignment primary, mask IoU50 secondary; full image/category matching maxDets100. No mask75 success filter. Crowd pixels excluded, own-GT raster area denominator. Outcome-conditioned comparisons are descriptive."}
    write_json(args.out / "run.json", run)
    gt = COCO(str(gt_path))
    for name in ("baseline", "ccl01"):
        pred_path = ROOT / "gemini/results" / f"predictions_{name}_stage2.json"
        info = json.loads((args.previous_run / f"matches_{name}.json").read_text(encoding="utf-8"))
        assert digest(pred_path) == info["fingerprint"]["predictions"]
        predictions[name] = json.loads(pred_path.read_text(encoding="utf-8"))
        masks[name] = np.load(args.previous_run / f"matches_{name}.npz", allow_pickle=False)["matches"]
        print(f"[{name}] Official box50 ownership", flush=True)
        boxes[name] = box_assignment(gt, predictions[name], meta["annotation_ids"])
        np.save(args.out / f"box50_assignment_{name}.npy", boxes[name])
        gc.collect()
    base75, ccl75 = masks["baseline"][:, 5] > 0, masks["ccl01"][:, 5] > 0
    outcomes = np.select([base75 & ccl75, ~base75 & ccl75, base75 & ~ccl75], ["retained", "gained", "lost"], default="unrecovered")
    by_image = defaultdict(list)
    for i, a in enumerate(anns):
        by_image[a["image_id"]].append(i)
    rows, started = [], time.perf_counter()
    for completed, (iid, indexes) in enumerate(sorted(by_image.items()), 1):
        gt_all = gt.imgToAnns[iid]
        raster = {a["id"]: gt.annToRLE(a) for a in gt_all}
        ordinary = [a for a in gt_all if not a.get("iscrowd", 0)]
        crowd_rles = [raster[a["id"]] for a in gt_all if a.get("iscrowd", 0)]
        crowd = mu.merge(crowd_rles) if crowd_rles else None
        union = mu.merge([raster[a["id"]] for a in ordinary])
        same_union = {cid: mu.merge([raster[a["id"]] for a in ordinary if a["category_id"] == cid]) for cid in {a["category_id"] for a in ordinary}}
        shape = (gt.imgs[iid]["height"], gt.imgs[iid]["width"])
        metric_cache = {}
        for i in indexes:
            a = anns[i]
            aid, cid = a["id"], a["category_id"]
            for assignment in ("box50", "mask50"):
                assigned = boxes if assignment == "box50" else {n: masks[n][:, 0] for n in masks}
                db, dc = int(assigned["baseline"][i]), int(assigned["ccl01"][i])
                row = {"annotation_id": aid, "image_id": iid, "category_id": cid, "size": "small" if a["area"] < 1024 else "medium" if a["area"] < 9216 else "large",
                       "ici": float(meta["crowd"][i]), "outcome75": outcomes[i], "assignment": assignment,
                       "base_detection": db, "ccl_detection": dc,
                       "agrees_with_mask50_both": bool(db > 0 and dc > 0 and db == masks["baseline"][i, 0] and dc == masks["ccl01"][i, 0])}
                values = {}
                for name, prefix, did in (("baseline", "base", db), ("ccl01", "ccl", dc)):
                    value = None
                    if did:
                        key = (name, aid, did)
                        if key not in metric_cache:
                            pred = predictions[name][did-1]
                            assert pred["image_id"] == iid and pred["category_id"] == cid
                            assert pred["segmentation"]["size"] == list(shape)
                            value = spatial_metrics(pred["segmentation"], raster[aid], union, same_union[cid], crowd)
                            if value is not None:
                                value.update(box_metrics(pred["bbox"], a["bbox"], raster[aid], crowd, shape))
                                value["official_mask_iou"] = float(mu.iou([pred["segmentation"]], [raster[aid]], [0])[0, 0])
                                value["score"] = pred["score"]
                            metric_cache[key] = value
                        value = metric_cache[key]
                    values[prefix] = value
                    for field in PIXEL_METRICS + ["score", "gt_valid_pixels", "pred_valid_pixels", "target_pixels", "neighbor_pixels", "same_neighbor_pixels", "background_pixels"]:
                        row[prefix+"_"+field] = value.get(field) if value else None
                if values["base"] and values["ccl"]:
                    row.update(decompose(values["base"], values["ccl"]))
                    pb, pc = values["base"]["official_mask_iou"] >= .75, values["ccl"]["official_mask_iou"] >= .75
                    selected_state = "retained" if pb and pc else "lost" if pb else "gained" if pc else "unrecovered"
                    row["selected_outcome75"] = selected_state
                    row["selected_transition_agrees"] = selected_state == outcomes[i]
                else:
                    row.update({f: None for f in DECOMP})
                    row["selected_outcome75"] = None
                    row["selected_transition_agrees"] = None
                rows.append(row)
        if completed % 200 == 0 or completed == len(by_image):
            print(f"Spatial pixels: {completed}/{len(by_image)} images ({time.perf_counter()-started:.1f}s)", flush=True)
    write_csv(args.out / "per_gt_spatial.csv", rows)
    result = summarize(rows, meta, args.out, args.previous_run)
    run["completed"] = datetime.now().isoformat()
    run["rows"] = len(rows)
    write_json(args.out / "run.json", run)
    print(json.dumps({"output": str(args.out), "high_box50": result["box50"]["H"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
