"""Cross-fitted nearest-direction analysis of the frozen 7L finite-oracle bank."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch


BLOCKS = ("h", "basic", "raw_detection", "local_summary")
MODES = {
    "h": ("h",),
    "h_detection": ("h", "basic", "raw_detection"),
    "h_local": ("h", "local_summary"),
    "h_detection_local": BLOCKS,
}


def load_bank(shards, annotations, manifest):
    obj = json.loads(annotations.read_text(encoding="utf-8"))
    ann = {int(a["id"]): (int(a["category_id"]), float(a["area"]))
           for a in obj["annotations"]}
    del obj
    by_image = {int(iid): fold for fold, images in enumerate(manifest["folds"])
                for iid in images}
    assert len(by_image) == 10000
    arrays = defaultdict(list)
    files = sorted(shards.glob("PART_*.pt"))
    assert len(files) == 100, len(files)
    for count, path in enumerate(files, 1):
        rows = torch.load(path, map_location="cpu", weights_only=True)
        if not rows:
            continue
        for key in BLOCKS + ("delta", "gram", "predicted_box"):
            arrays[key].append(np.stack([r[key].numpy() for r in rows]).astype(np.float32))
        arrays["image_id"].append(np.asarray([int(r["image_id"]) for r in rows], dtype=np.int64))
        arrays["annotation_id"].append(np.asarray([int(r["annotation_id"]) for r in rows], dtype=np.int64))
        arrays["fold"].append(np.asarray([by_image[int(r["image_id"])] for r in rows], dtype=np.int8))
        arrays["category"].append(np.asarray([ann[int(r["annotation_id"])][0] for r in rows], dtype=np.int16))
        arrays["area_bin"].append(np.digitize(
            np.sqrt([ann[int(r["annotation_id"])][1] for r in rows]), [32, 96]).astype(np.int8))
        arrays["level"].append(np.asarray([int(r["level"]) for r in rows], dtype=np.int8))
        arrays["original_iou"].append(np.asarray([float(r["original_iou"]) for r in rows], dtype=np.float32))
        arrays["box_iou"].append(np.asarray([float(r["box_iou"]) for r in rows], dtype=np.float32))
        del rows
        if count % 20 == 0:
            print(json.dumps({"loaded_shards": count, "total_shards": len(files)}), flush=True)
    bank = {key: np.concatenate(parts) for key, parts in arrays.items()}
    assert len(bank["image_id"]) == 71269
    assert len(set(zip(bank["image_id"], bank["annotation_id"]))) == 71269
    assert np.isfinite(bank["delta"]).all() and np.isfinite(bank["gram"]).all()
    bank["radius"] = np.linalg.norm(bank["delta"], axis=1)
    bank["direction"] = bank["delta"] / np.maximum(bank["radius"][:, None], 1e-12)
    return bank


def normalized_blocks(bank, fit):
    result = {}
    for key in BLOCKS:
        x = bank[key]
        mean = x[fit].mean(axis=0)
        std = x[fit].std(axis=0)
        z = (x - mean) / np.maximum(std, .01)
        z = np.nan_to_num(z, nan=0., posinf=0., neginf=0.)
        result[key] = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return {mode: np.concatenate([result[key] for key in keys], axis=1).astype(np.float32)
            / np.sqrt(len(keys)) for mode, keys in MODES.items()}


def matching_groups(bank, fit, query):
    exact, category_area, category = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in fit:
        c, a, l = (int(bank[key][i]) for key in ("category", "area_bin", "level"))
        exact[(c, a, l)].append(i)
        category_area[(c, a)].append(i)
        category[c].append(i)
    for lookup in (exact, category_area, category):
        for key in lookup:
            lookup[key] = np.asarray(lookup[key], dtype=np.int32)
    pools, groups = {}, defaultdict(list)
    fallback = np.zeros(len(bank["image_id"]), dtype=np.int8)
    for i in query:
        c, a, l = (int(bank[key][i]) for key in ("category", "area_bin", "level"))
        choices = ((0, (c, a, l), exact.get((c, a, l))),
                   (1, (c, a), category_area.get((c, a))),
                   (2, (c,), category.get(c)), (3, (), fit))
        for rank, key, eligible in choices:
            if eligible is not None and len(eligible) >= 5:
                group = (rank,) + key
                pools[group] = eligible
                groups[group].append(i)
                fallback[i] = rank
                break
        else:
            raise AssertionError(i)
    return pools, groups, fallback


def retrieve(features, pools, groups, total):
    top = np.full((total, 5), -1, dtype=np.int32)
    similarity = np.full(total, np.nan, dtype=np.float32)
    for group, rows in groups.items():
        eligible = pools[group]
        bank = features[eligible].T.copy()
        for lo in range(0, len(rows), 256):
            qi = np.asarray(rows[lo:lo + 256], dtype=np.int32)
            sim = features[qi] @ bank
            take = np.argpartition(sim, sim.shape[1] - 5, axis=1)[:, -5:]
            sort = np.argsort(np.take_along_axis(sim, take, axis=1), axis=1)[:, ::-1]
            take = np.take_along_axis(take, sort, axis=1)
            top[qi] = eligible[take]
            similarity[qi] = np.take_along_axis(sim, take[:, :1], axis=1)[:, 0]
    return top, similarity


def effect_cos(vectors, target, gram):
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 32)
    target = np.asarray(target, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    gt = gram @ target
    numerator = vectors @ gt
    va = np.einsum("ni,ij,nj->n", vectors, gram, vectors)
    ta = float(target @ gt)
    return np.clip(numerator / np.sqrt(np.maximum(va * ta, 1e-20)), -1, 1)


def summarize(rows, boot_count, seed):
    masks = {
        "all": lambda r: True,
        "failure": lambda r: r["original_iou"] < .75,
        "success": lambda r: r["original_iou"] >= .75,
        "failure_good_box": lambda r: r["original_iou"] < .75 and r["box_iou"] >= .75,
        "success_good_box": lambda r: r["original_iou"] >= .75 and r["box_iou"] >= .75,
        "failure_bad_box": lambda r: r["original_iou"] < .75 and r["box_iou"] < .75,
        "mild_failure": lambda r: .5 <= r["original_iou"] < .75,
        "severe_failure": lambda r: r["original_iou"] < .5,
    }
    output = {}
    for name, predicate in masks.items():
        subset = [r for r in rows if predicate(r)]
        if not subset:
            continue
        image_ids = np.asarray([r["image_id"] for r in subset])
        images, inverse = np.unique(image_ids, return_inverse=True)
        rng = np.random.default_rng(seed + len(name))
        draws = rng.integers(0, len(images), size=(boot_count, len(images)), dtype=np.int32)
        counts = np.bincount(inverse, minlength=len(images))
        denominators = counts[draws].sum(axis=1)
        metrics = {"h_top1_minus_random1": np.asarray([r["h_top1"] - r["random1"] for r in subset]),
                   "h_top5_minus_random5": np.asarray([r["h_top5"] - r["random5"] for r in subset])}
        for mode in MODES:
            if mode != "h":
                metrics[mode + "_top5_minus_h"] = np.asarray(
                    [r[mode + "_top5"] - r["h_top5"] for r in subset])
        paired = {}
        for metric, values in metrics.items():
            by_image = np.bincount(inverse, weights=values, minlength=len(images))
            boot = by_image[draws].sum(axis=1) / denominators
            paired[metric] = {"mean": float(values.mean()),
                              "ci95": np.quantile(boot, [.025, .975]).tolist()}
        means = {key: float(np.mean([r[key] for r in subset]))
                 for key in ("random1", "random5", "h_top1", "h_top5",
                             "h_detection_top5", "h_local_top5", "h_detection_local_top5",
                             "h_feature_cos", "box_iou", "original_iou")}
        output[name] = {
            "images": len(images), "candidates": len(subset),
            "fallback": np.bincount([r["fallback"] for r in subset], minlength=4).tolist(),
            "eligible_median": float(np.median([r["eligible"] for r in subset])),
            "radius_median": float(np.median([r["radius"] for r in subset])),
            "h_top1_nonpositive_fraction": float(np.mean([r["h_top1"] <= 0 for r in subset])),
            "means": means, "paired": paired,
        }
    return output


def main(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank = load_bank(args.targets / "shards", args.annotations, manifest)
    rng = np.random.default_rng(args.seed)
    output = []
    for fold in range(5):
        if args.fold >= 0 and fold != args.fold:
            continue
        fit = np.flatnonzero(bank["fold"] != fold).astype(np.int32)
        query = np.flatnonzero(bank["fold"] == fold).astype(np.int32)
        if args.max_queries:
            query = query[:args.max_queries]
        assert len(set(bank["image_id"][fit]) & set(bank["image_id"][query])) == 0
        feature = normalized_blocks(bank, fit)
        pools, groups, fallback = matching_groups(bank, fit, query)
        retrieved = {}
        for mode in MODES:
            retrieved[mode] = retrieve(feature[mode], pools, groups, len(bank["image_id"]))
        for step, i in enumerate(query, 1):
            c, a, l = (int(bank[key][i]) for key in ("category", "area_bin", "level"))
            group = ((0, c, a, l) if fallback[i] == 0 else
                     (1, c, a) if fallback[i] == 1 else
                     (2, c) if fallback[i] == 2 else (3,))
            eligible = pools[group]
            random = rng.choice(eligible, size=(8, 5), replace=True)
            target, gram = bank["direction"][i], bank["gram"][i]
            random1 = float(effect_cos(bank["direction"][random[:, 0]], target, gram).mean())
            random5 = float(effect_cos(bank["direction"][random].mean(axis=1), target, gram).mean())
            row = dict(image_id=int(bank["image_id"][i]), annotation_id=int(bank["annotation_id"][i]),
                       fold=fold, category=c, area_bin=a, level=l, eligible=len(eligible),
                       fallback=int(fallback[i]), radius=float(bank["radius"][i]),
                       original_iou=float(bank["original_iou"][i]), box_iou=float(bank["box_iou"][i]),
                       random1=random1, random5=random5)
            for mode in MODES:
                top, similarity = retrieved[mode]
                neighbors = top[i]
                row[mode + "_top1"] = float(effect_cos(bank["direction"][neighbors[:1]], target, gram)[0])
                row[mode + "_top5"] = float(effect_cos(
                    bank["direction"][neighbors].mean(axis=0), target, gram)[0])
                row[mode + "_feature_cos"] = float(similarity[i])
            output.append(row)
            if step % 5000 == 0:
                print(json.dumps({"fold": fold, "scored": step, "queries": len(query)}), flush=True)
        print(json.dumps({"fold_completed": fold, "queries": len(query)}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "ROWS.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = summarize(output, args.bootstrap, args.seed)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    complete = dict(images=len(set(r["image_id"] for r in output)), candidates=len(output),
                    folds=sorted(set(r["fold"] for r in output)),
                    target_source=str(args.targets), seed=args.seed, bootstrap=args.bootstrap,
                    diagnostic_gt_controls_only=True)
    (args.out / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "targets", "annotations", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    main(parser.parse_args())
