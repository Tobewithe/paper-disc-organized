"""Official COCO matching followed by GT-defined crowding diagnostics.

No model inference or training. Cached matches retain all ordinary GT, including
misses, and the official crowd/category/score-order matching semantics.
"""
import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import inspect
import io
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pycocotools.cocoeval as cocoeval_module
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
IOU75 = 5
GROUPS = ["All", "I0", "I1", "I2", "I3", "L", "H"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dt(gt, predictions, kind):
    field = "segmentation" if kind == "segm" else "bbox"
    filtered = [{key: copy.deepcopy(p[key]) for key in ("image_id", "category_id", "score", field)} for p in predictions]
    if filtered:
        return gt.loadRes(filtered)
    dt = COCO()
    dt.dataset = {"images": copy.deepcopy(gt.dataset["images"]), "categories": copy.deepcopy(gt.dataset["categories"]), "annotations": []}
    dt.createIndex()
    return dt


def official_eval(gt, predictions, kind="segm"):
    dt = make_dt(gt, predictions, kind)
    ev = COCOeval(gt, dt, kind)
    ev.params.imgIds = sorted(gt.imgs)
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def extract_matches(ev, annotation_ids, image_ids):
    """Read official area=all/maxDets=100 matches, without rematching a subgroup."""
    index = {int(a): i for i, a in enumerate(annotation_ids)}
    image_index = {int(a): i for i, a in enumerate(image_ids)}
    matches = np.zeros((len(annotation_ids), len(ev.params.iouThrs)), dtype=np.int64)
    scores = np.full(matches.shape, -np.inf, dtype=float)
    seen = set()
    dt_scores, dt_image, dt_tp, dt_gt, dt_ids = [], [], [], [], []
    for record in ev.evalImgs:
        if record is None or record["aRng"] != ev.params.areaRng[0]:
            continue
        score_lookup = dict(zip(record["dtIds"], record["dtScores"]))
        for col, (aid, ignored) in enumerate(zip(record["gtIds"], record["gtIgnore"])):
            if ignored:
                continue
            assert aid in index, f"Unlisted nonignored GT: {aid}"
            assert aid not in seen, f"Repeated GT: {aid}"
            seen.add(aid)
            row = index[aid]
            matches[row] = record["gtMatches"][:, col].astype(np.int64)
            for t, did in enumerate(matches[row]):
                if did:
                    scores[row, t] = score_lookup[did]
        for j, did in enumerate(record["dtIds"]):
            if record["dtIgnore"][IOU75, j]:
                continue
            matched_aid = int(record["dtMatches"][IOU75, j])
            dt_scores.append(record["dtScores"][j])
            dt_image.append(image_index[record["image_id"]])
            dt_tp.append(bool(matched_aid))
            dt_gt.append(index[matched_aid] if matched_aid else -1)
            dt_ids.append(did)
    assert seen == set(index), "Missing nonignored GT in official matches"
    for t in range(matches.shape[1]):
        positive = matches[:, t][matches[:, t] > 0]
        assert len(positive) == len(np.unique(positive)), "One detection matched multiple ordinary GT"
    assert int((matches[:, IOU75] > 0).sum()) == sum(dt_tp)
    return {"matches": matches, "match_scores": scores,
            "dt_scores": np.asarray(dt_scores), "dt_image": np.asarray(dt_image, dtype=np.int32),
            "dt_tp": np.asarray(dt_tp, dtype=bool), "dt_gt": np.asarray(dt_gt, dtype=np.int32),
            "dt_ids": np.asarray(dt_ids, dtype=np.int64)}


def load_metadata(gt_path, manifest):
    data = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    image_ids = np.asarray(sorted(im["id"] for im in data["images"]), dtype=np.int64)
    image_index = {int(i): k for k, i in enumerate(image_ids)}
    anns = sorted((a for a in data["annotations"] if not a.get("iscrowd", 0)), key=lambda a: a["id"])
    assert all(a["bbox"][2] > 0 and a["bbox"][3] > 0 for a in anns)
    with Path(manifest).open(encoding="utf-8-sig", newline="") as handle:
        rows = {int(r["annotation_id"]): r for r in csv.DictReader(handle)}
    aid = np.asarray([a["id"] for a in anns], dtype=np.int64)
    crowd = np.asarray([float(rows[a]["ici_same"]) for a in aid])
    crowd_all = np.asarray([float(rows[a]["ici_all_categories"]) for a in aid])
    b = np.select([crowd == 0, crowd <= .5, crowd <= 1], [0, 1, 2], default=3)
    ba = np.select([crowd_all == 0, crowd_all <= .5, crowd_all <= 1], [0, 1, 2], default=3)
    masks = np.column_stack([np.ones(len(aid), bool)] + [b == k for k in range(4)] + [crowd <= .5, crowd > .5])
    return data, anns, {"annotation_ids": aid, "image_ids": image_ids,
                        "gt_image": np.asarray([image_index[a["image_id"]] for a in anns], dtype=np.int32),
                        "category": np.asarray([a["category_id"] for a in anns]),
                        "area": np.asarray([a["area"] for a in anns]),
                        "crowd": crowd, "crowd_all": crowd_all, "bins": b, "all_bins": ba, "group_masks": masks}


def pair_indices(anns, meta):
    grouped = defaultdict(list)
    for i, a in enumerate(anns):
        grouped[a["image_id"]].append(i)
    pairs = []
    for ids in grouped.values():
        boxes = np.asarray([anns[i]["bbox"] for i in ids], dtype=float)
        ends = boxes[:, :2] + boxes[:, 2:]
        wh = np.maximum(0, np.minimum(ends[:, None], ends[None, :]) - np.maximum(boxes[:, None, :2], boxes[None, :, :2]))
        intersection = wh.prod(axis=2)
        areas = boxes[:, 2] * boxes[:, 3]
        iou = intersection / (areas[:, None] + areas[None, :] - intersection)
        for i, j in zip(*np.nonzero(np.triu(iou > .05, 1))):
            a, b = ids[i], ids[j]
            pairs.append((a, b, meta["gt_image"][a], int(meta["category"][a] == meta["category"][b]), iou[i, j]))
    return np.asarray(pairs, dtype=float).reshape(-1, 5)


def image_sums(values, img, n_images):
    arr = np.zeros((n_images,) + values.shape[1:], dtype=float)
    np.add.at(arr, img, values)
    return arr


def ci(observed, samples):
    v = np.asarray(samples)
    valid = np.isfinite(v)
    return {"estimate_pp": float(observed * 100), "ci95_pp": [float(x * 100) for x in np.percentile(v[valid], [2.5, 97.5])] if valid.any() else None,
            "valid_draws": int(valid.sum())}


def precision_curve(cache):
    unique = np.unique(cache["dt_scores"])[::-1]
    bucket = len(unique) - 1 - np.searchsorted(unique[::-1], cache["dt_scores"])
    return unique, bucket


def precision_point(cache, curve, target, weights=None):
    """Choose a realizable whole-score tie block, not an interpolated partial tie."""
    unique, bucket = curve
    if not len(unique):
        return None
    w = np.ones(len(bucket)) if weights is None else weights[cache["dt_image"]]
    tp = np.cumsum(np.bincount(bucket, weights=w * cache["dt_tp"], minlength=len(unique)))
    fp = np.cumsum(np.bincount(bucket, weights=w * ~cache["dt_tp"], minlength=len(unique)))
    precision = tp / np.maximum(tp + fp, 1)
    valid = np.flatnonzero((precision >= target) & (tp + fp > 0))
    if not len(valid):
        return None
    # Maximum total recall first, then higher precision, then higher threshold.
    best_tp = tp[valid].max()
    choices = valid[tp[valid] == best_tp]
    chosen = choices[np.argmax(precision[choices])]
    return {"threshold": float(unique[chosen]), "precision": float(precision[chosen]),
            "tp": int(tp[chosen]), "fp": int(fp[chosen])}


def standardized_support(meta, n_images):
    size = np.select([meta["area"] < 1024, meta["area"] < 9216], [0, 1], default=2)
    cells = meta["category"] * 3 + size
    high = meta["crowd"] > .5
    records, masks, weights = [], [], []
    for cell in sorted(set(cells)):
        lo, hi = (cells == cell) & ~high, (cells == cell) & high
        nl, nh = int(lo.sum()), int(hi.sum())
        il, ih = len(set(meta["gt_image"][lo])), len(set(meta["gt_image"][hi]))
        if min(nl, nh) >= 20 and min(il, ih) >= 10:
            masks.extend([lo, hi])
            weights.append(nl + nh)
            records.append({"category_id": int(cell // 3), "size": ["small", "medium", "large"][cell % 3], "low_gt": nl, "high_gt": nh, "low_images": il, "high_images": ih})
    assert masks, "No category/size common support"
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    for r, value in zip(records, w):
        r["weight"] = float(value)
    return np.column_stack(masks), w, records


def evaluate_model(name, pred_path, gt_path, meta, out):
    cache_path = out / f"matches_{name}.npz"
    info_path = out / f"matches_{name}.json"
    fingerprint = {"gt": digest(gt_path), "predictions": digest(pred_path),
                   "cocoeval_source": digest(cocoeval_module.__file__),
                   "matching_code": hashlib.sha256((inspect.getsource(make_dt) + inspect.getsource(official_eval) + inspect.getsource(extract_matches)).encode()).hexdigest(),
                   "annotation_ids": hashlib.sha256(meta["annotation_ids"].tobytes()).hexdigest()}
    if cache_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["fingerprint"] == fingerprint:
            print(f"[{name}] Verified match cache", flush=True)
            return dict(np.load(cache_path, allow_pickle=False)), info
    print(f"[{name}] Official segmentation evaluation started", flush=True)
    gt = COCO(str(gt_path))
    predictions = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    start = time.perf_counter()
    ev = official_eval(gt, predictions)
    matches = extract_matches(ev, meta["annotation_ids"], meta["image_ids"])
    info = {"fingerprint": fingerprint, "source_prediction": str(pred_path), "segm_stats": [float(x) for x in ev.stats],
            "elapsed_segm_seconds": time.perf_counter() - start, "num_predictions": len(predictions)}
    del ev
    gc.collect()
    print(f"[{name}] Official bbox evaluation started", flush=True)
    ev = official_eval(gt, predictions, "bbox")
    info["box_stats"] = [float(x) for x in ev.stats]
    info["elapsed_total_seconds"] = time.perf_counter() - start
    np.savez_compressed(cache_path, **matches)
    write_json(info_path, info)
    return matches, info


def run_analysis(caches, infos, meta, pairs, out, B, seed):
    n_images = len(meta["image_ids"])
    img = meta["gt_image"]
    masks = meta["group_masks"]
    counts = image_sums(masks.astype(float), img, n_images)
    denom = counts.sum(axis=0)
    rng = np.random.RandomState(seed)
    draws = rng.randint(n_images, size=(B, n_images), dtype=np.int32)
    weights = np.asarray([np.bincount(row, minlength=n_images) for row in draws], dtype=np.int32)
    np.savez_compressed(out / "bootstrap_draws.npz", image_ids=meta["image_ids"], sampled_image_indices=draws)
    boot_denom = weights @ counts
    results, rows, boot = {}, [], {}
    support, sw, support_records = standardized_support(meta, n_images)
    support_count = image_sums(support.astype(float), img, n_images)
    support_boot_count = weights @ support_count
    write_csv(out / "standardization_support.csv", support_records)
    pair_a, pair_b, pair_img = [pairs[:, col].astype(int) for col in range(3)]
    same = pairs[:, 3].astype(bool)
    high = meta["crowd"] > .5
    pair_masks = {"same_all": same, "same_both_high": same & high[pair_a] & high[pair_b],
                  "same_mixed": same & (high[pair_a] != high[pair_b]), "same_both_low": same & ~high[pair_a] & ~high[pair_b], "cross_class": ~same}
    pair_rows = []
    for name, cache in caches.items():
        hit = cache["matches"] > 0
        per_img_hit = image_sums(masks[:, :, None] * hit[:, None, :], img, n_images)
        recall = per_img_hit.sum(axis=0) / denom[:, None]
        boot_r75 = (weights @ per_img_hit[:, :, IOU75]) / boot_denom
        boot[name] = {"r75": boot_r75}
        curve = precision_curve(cache)
        operating = {}
        for target in (.8, .9, .95):
            point = precision_point(cache, curve, target)
            if point:
                accepted = cache["match_scores"][:, IOU75] >= point["threshold"]
                point["recall_by_group"] = {g: float(accepted[masks[:, k]].mean()) for k, g in enumerate(GROUPS)}
                assert accepted.sum() == point["tp"]
            operating[str(target)] = point
        pair_result = {}
        pair_hit = hit[pair_a, IOU75] & hit[pair_b, IOU75]
        assert np.all(cache["matches"][pair_a[pair_hit], IOU75] != cache["matches"][pair_b[pair_hit], IOU75])
        for label, mask in pair_masks.items():
            pc = np.bincount(pair_img[mask], minlength=n_images)
            ph = np.bincount(pair_img[mask], weights=pair_hit[mask], minlength=n_images)
            if not pc.sum():
                continue
            point = float(ph.sum() / pc.sum())
            samples = (weights @ ph) / (weights @ pc)
            boot[name]["pair_" + label] = samples
            im_has = pc > 0
            image_macro = float((ph[im_has] / pc[im_has]).mean())
            pair_result[label] = {"pairs": int(pc.sum()), "images": int(im_has.sum()), "recall75": point, "image_macro_recall75": image_macro}
            pair_rows.append({"model": name, "pair_group": label, **pair_result[label]})
        sh = image_sums(support * hit[:, IOU75, None], img, n_images)
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = sh.sum(axis=0) / support_count.sum(axis=0)
            sb = (weights @ sh) / support_boot_count
        standard = (sr.reshape(-1, 2) * sw[:, None]).sum(axis=0)
        sboot = (sb.reshape(B, -1, 2) * sw[None, :, None]).sum(axis=1)
        boot[name]["standardized"] = sboot
        per_category = {}
        for k, group in enumerate(GROUPS):
            values = [float(hit[masks[:, k] & (meta["category"] == c), IOU75].mean()) for c in sorted(set(meta["category"][masks[:, k]]))]
            per_category[group] = float(np.mean(values))
            rows.append({"model": name, "group": group, "gt": int(denom[k]),
                         "images": int((counts[:, k] > 0).sum()), "R50": float(recall[k, 0]), "R75": float(recall[k, IOU75]),
                         "mean_R50_95": float(recall[k].mean()), "R75_category_macro": per_category[group],
                         "R75_at_P90": operating["0.9"]["recall_by_group"][group] if operating["0.9"] else None})
        all_category = {}
        for k in range(4):
            sel = meta["all_bins"] == k
            all_category[f"I{k}"] = {"gt": int(sel.sum()), "R75": float(hit[sel, IOU75].mean())}
        results[name] = {"standard_ap": infos[name], "recall_by_group": {g: {"gt": int(denom[k]), "R75": float(recall[k, IOU75]), "recall_curve": recall[k].tolist()} for k, g in enumerate(GROUPS)},
                         "operating_points": operating, "pair_recall": pair_result,
                         "standardized_R75": {"L": float(standard[0]), "H": float(standard[1])}, "all_category_ici_secondary": all_category}
        # Re-select the global precision operation in every image bootstrap draw.
        bp = np.full((B, len(GROUPS)), np.nan)
        gt_scores = cache["match_scores"][:, IOU75]
        start = time.perf_counter()
        for b in range(B):
            point = precision_point(cache, curve, .9, weights[b])
            if point:
                accepted = gt_scores >= point["threshold"]
                numerator = ((weights[b, img] * accepted)[:, None] * masks).sum(axis=0)
                bp[b] = numerator / boot_denom[b]
            if (b + 1) % 500 == 0:
                print(f"[{name}] P90 bootstrap {b+1}/{B} ({time.perf_counter()-start:.1f}s)", flush=True)
        boot[name]["P90"] = bp
        write_csv(out / f"gt_matches_{name}.csv", [{"annotation_id": int(aid), "image_id": int(meta["image_ids"][img[i]]), "category_id": int(meta["category"][i]),
                  "ici": float(meta["crowd"][i]), "group": f"I{meta['bins'][i]}", "hit50": int(hit[i, 0]), "hit75": int(hit[i, IOU75]),
                  "matched_detection75": int(cache["matches"][i, IOU75]), "matched_score75": float(gt_scores[i]) if np.isfinite(gt_scores[i]) else None} for i, aid in enumerate(meta["annotation_ids"])])
    contrasts = {}
    for name in caches:
        if name == "baseline":
            continue
        delta = boot[name]["r75"] - boot["baseline"]["r75"]
        observed = np.asarray([results[name]["recall_by_group"][g]["R75"] - results["baseline"]["recall_by_group"][g]["R75"] for g in GROUPS])
        record = {g: ci(observed[k], delta[:, k]) for k, g in enumerate(GROUPS)}
        record["D_H_minus_L"] = ci(observed[6] - observed[5], delta[:, 6] - delta[:, 5])
        # Exact GT-micro recall decomposition, not an AP decomposition.
        np.testing.assert_allclose(observed[0], (denom[5] * observed[5] + denom[6] * observed[6]) / denom[0], atol=1e-14)
        sd = boot[name]["standardized"] - boot["baseline"]["standardized"]
        so = np.asarray([results[name]["standardized_R75"][g] - results["baseline"]["standardized_R75"][g] for g in ("L", "H")])
        record["standardized"] = {g: ci(so[k], sd[:, k]) for k, g in enumerate(("L", "H"))}
        record["standardized"]["D_H_minus_L"] = ci(so[1] - so[0], sd[:, 1] - sd[:, 0])
        pp = {}
        if results[name]["operating_points"]["0.9"] and results["baseline"]["operating_points"]["0.9"]:
            pd = boot[name]["P90"] - boot["baseline"]["P90"]
            po = np.asarray([results[name]["operating_points"]["0.9"]["recall_by_group"][g] - results["baseline"]["operating_points"]["0.9"]["recall_by_group"][g] for g in GROUPS])
            pp = {g: ci(po[k], pd[:, k]) for k, g in enumerate(GROUPS)}
            pp["D_H_minus_L"] = ci(po[6] - po[5], pd[:, 6] - pd[:, 5])
        record["P90"] = pp
        record["pairs"] = {g: ci(results[name]["pair_recall"][g]["recall75"] - results["baseline"]["pair_recall"][g]["recall75"], boot[name]["pair_"+g] - boot["baseline"]["pair_"+g]) for g in results[name]["pair_recall"]}
        contrasts[name + "_minus_baseline"] = record
    pair_export = []
    for k, (a, b, im) in enumerate(zip(pair_a, pair_b, pair_img)):
        row = {"image_id": int(meta["image_ids"][im]), "annotation_a": int(meta["annotation_ids"][a]), "annotation_b": int(meta["annotation_ids"][b]), "same_category": bool(same[k]), "box_iou": float(pairs[k, 4])}
        for name, cache in caches.items():
            row[name + "_both_hit75"] = bool(cache["matches"][a, IOU75] and cache["matches"][b, IOU75])
        pair_export.append(row)
    write_csv(out / "gt_pairs.csv", pair_export)
    write_csv(out / "recall_by_density.csv", rows)
    write_csv(out / "pair_recall.csv", pair_rows)
    np.savez_compressed(out / "bootstrap_metrics.npz", **{name + "__" + k: v for name, values in boot.items() for k, v in values.items()})
    summary = {"images": n_images, "gt_instances": len(img), "bootstrap_B": B, "bootstrap_seed": seed,
               "interval_scope": "paired image resampling of fixed seed-0 checkpoints; exploratory cached predictions, not training-seed uncertainty",
               "groups": GROUPS, "standardization_support": support_records, "models": results, "contrasts": contrasts}
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=ROOT / "gemini/data/coco_dense/coco_dense_val_gt.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refine-logs/coco-evaluation/COCO_INSTANCE_MANIFEST_20260911_032703.csv")
    parser.add_argument("--pred-dir", type=Path, default=ROOT / "gemini/results")
    parser.add_argument("--models", nargs="+", default=["baseline", "ccl01", "ccl05"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _, anns, meta = load_metadata(args.gt, args.manifest)
    run = {"started": datetime.now().isoformat(), "script_sha256": digest(__file__), "gt_sha256": digest(args.gt),
           "manifest_sha256": digest(args.manifest), "models": args.models, "bootstrap": args.bootstrap, "seed": args.seed,
           "protocol": "COCOeval maxDets=100; GT strata after complete matching; tie-complete global P90 selection; masks-only loadRes for segm; cached scores retain original five-decimal rounding."}
    write_json(args.out / "run.json", run)
    caches, infos = {}, {}
    for name in args.models:
        caches[name], infos[name] = evaluate_model(name, args.pred_dir / f"predictions_{name}_stage2.json", args.gt, meta, args.out)
        gc.collect()
    pairs = pair_indices(anns, meta)
    print(f"Matched {len(anns)} GT across {len(meta['image_ids'])} images; {len(pairs)} adjacent GT pairs", flush=True)
    summary = run_analysis(caches, infos, meta, pairs, args.out, args.bootstrap, args.seed)
    run["completed"] = datetime.now().isoformat()
    write_json(args.out / "run.json", run)
    print(json.dumps({"output": str(args.out), "contrasts": {k: {g: v[g] for g in ("All", "L", "H", "D_H_minus_L")} for k, v in summary["contrasts"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
