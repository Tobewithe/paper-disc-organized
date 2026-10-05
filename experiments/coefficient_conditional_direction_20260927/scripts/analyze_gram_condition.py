"""Exploratory true-versus-shuffled prototype-Gram retrieval control."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_conditional_direction import load_bank, matching_groups, retrieve, effect_cos


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def transform(x, fit):
    mean = x[fit].mean(axis=0)
    std = x[fit].std(axis=0)
    return unit(np.nan_to_num((x - mean) / np.maximum(std, .01)))


def shuffled_gram_indices(bank):
    groups = defaultdict(list)
    by_category_area, by_category = defaultdict(list), defaultdict(list)
    for i in range(len(bank["image_id"])):
        c, a, l = (int(bank[key][i]) for key in ("category", "area_bin", "level"))
        group = (c, a, l)
        groups[group].append(i)
        by_category_area[(c, a)].append(i)
        by_category[c].append(i)
    perm = np.arange(len(bank["image_id"]), dtype=np.int32)
    fallback = 0
    for (c, a, _), indices in groups.items():
        if len(indices) > 1:
            indices = np.asarray(indices, dtype=np.int32)
            perm[indices] = np.roll(indices, 1)
        else:
            i = indices[0]
            eligible = [j for j in by_category_area[(c, a)] if j != i]
            if not eligible:
                eligible = [j for j in by_category[c] if j != i]
            assert eligible, (c, a, i)
            perm[i] = eligible[0]
            fallback += 1
    assert np.all(perm != np.arange(len(perm)))
    return perm, fallback


def summarize(rows, boot_count):
    tests = {
        "all": lambda r: True,
        "failure": lambda r: r["original_iou"] < .75,
        "success": lambda r: r["original_iou"] >= .75,
        "failure_good_box": lambda r: r["original_iou"] < .75 and r["box_iou"] >= .75,
        "success_good_box": lambda r: r["original_iou"] >= .75 and r["box_iou"] >= .75,
        "severe_failure": lambda r: r["original_iou"] < .5,
    }
    result = {}
    for name, check in tests.items():
        sub = [r for r in rows if check(r)]
        images, inv = np.unique([r["image_id"] for r in sub], return_inverse=True)
        draw = np.random.default_rng(20260927 + len(name)).integers(
            0, len(images), size=(boot_count, len(images)), dtype=np.int32)
        count = np.bincount(inv, minlength=len(images))
        denom = count[draw].sum(1)
        pairs = {}
        for label, first, second in (
            ("true_gram_minus_h", "true_gram_top5", "h_top5"),
            ("true_gram_minus_wrong_gram", "true_gram_top5", "wrong_gram_top5"),
            ("diag_gram_minus_h", "diag_gram_top5", "h_top5"),
        ):
            diff = np.asarray([r[first] - r[second] for r in sub])
            by_image = np.bincount(inv, weights=diff, minlength=len(images))
            boot = by_image[draw].sum(1) / denom
            pairs[label] = dict(mean=float(diff.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())
        result[name] = dict(images=len(images), candidates=len(sub),
            means={key: float(np.mean([r[key] for r in sub]))
                   for key in ("h_top5", "true_gram_top5", "diag_gram_top5", "wrong_gram_top5")},
            pairs=pairs)
    return result


def main(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank = load_bank(args.targets / "shards", args.annotations, manifest)
    with gzip.open(args.reference, "rt", encoding="utf-8") as stream:
        reference = {(int(r["image_id"]), int(r["annotation_id"])): r
                     for r in map(json.loads, stream)}
    assert len(reference) == 71269
    perm, wrong_match_fallback = shuffled_gram_indices(bank)
    upper_index = np.triu_indices(32)
    upper = bank["gram"][:, upper_index[0], upper_index[1]]
    diag = np.diagonal(bank["gram"], axis1=1, axis2=2).copy()
    output = []
    for fold in range(5):
        fit = np.flatnonzero(bank["fold"] != fold).astype(np.int32)
        query = np.flatnonzero(bank["fold"] == fold).astype(np.int32)
        h = transform(bank["h"], fit)
        g = transform(upper, fit)
        d = transform(diag, fit)
        # The wrong Gram uses the same fit-fold standardizer and its source is
        # another instance of the same category, area group and pyramid level.
        wrong_mean = upper[fit].mean(axis=0)
        wrong_std = upper[fit].std(axis=0)
        wrong = unit(np.nan_to_num((upper[perm] - wrong_mean) / np.maximum(wrong_std, .01)))
        features = {
            "true_gram": np.concatenate((h, g), axis=1) / np.sqrt(2),
            "diag_gram": np.concatenate((h, d), axis=1) / np.sqrt(2),
            "wrong_gram": np.concatenate((h, wrong), axis=1) / np.sqrt(2),
        }
        pools, groups, _ = matching_groups(bank, fit, query)
        neighbor = {name: retrieve(x.astype(np.float32), pools, groups, len(bank["image_id"]))[0]
                    for name, x in features.items()}
        for i in query:
            key = (int(bank["image_id"][i]), int(bank["annotation_id"][i]))
            old = reference[key]
            row = dict(image_id=key[0], annotation_id=key[1], fold=fold,
                       original_iou=float(bank["original_iou"][i]),
                       box_iou=float(bank["box_iou"][i]),
                       h_top5=float(old["h_top5"]))
            for name in features:
                direction = bank["direction"][neighbor[name][i]].mean(axis=0)
                row[name + "_top5"] = float(effect_cos(
                    direction, bank["direction"][i], bank["gram"][i])[0])
            output.append(row)
        print(json.dumps({"fold": fold, "scored": len(output)}), flush=True)
    assert len(output) == 71269
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "ROWS.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.out / "SUMMARY.json").write_text(json.dumps(summarize(output, args.bootstrap), indent=2))
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(images=9924, candidates=len(output),
        folds=[0, 1, 2, 3, 4], exploratory=True, wrong_gram_matched=True,
        wrong_match_exact_fallback=wrong_match_fallback), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "targets", "annotations", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    main(parser.parse_args())
