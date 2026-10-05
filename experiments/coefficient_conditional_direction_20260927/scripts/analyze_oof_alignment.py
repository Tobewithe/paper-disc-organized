"""Join frozen 7L five-fold correction outputs to finite oracle directions."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch


def key(row):
    return (int(row["image_id"]), int(row["annotation_id"]), int(row["original_raw_id"]))


def effect_cos(left, right, gram):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    lr = float(left @ gram @ right)
    ll = max(float(left @ gram @ left), 1e-20)
    rr = max(float(right @ gram @ right), 1e-20)
    return float(np.clip(lr / np.sqrt(ll * rr), -1, 1))


def grouped(rows, mask):
    chosen = [r for r in rows if mask(r)]
    if not chosen:
        return {}
    eff = np.asarray([r["effect_cos"] for r in chosen])
    benefit = np.asarray([r["benefit_iou"] for r in chosen])
    return dict(images=len({r["image_id"] for r in chosen}), candidates=len(chosen),
                effect_cos_mean=float(eff.mean()), effect_cos_median=float(np.median(eff)),
                effect_cos_nonpositive_fraction=float(np.mean(eff <= 0)),
                coefficient_cos_mean=float(np.mean([r["coefficient_cos"] for r in chosen])),
                mean_benefit_iou=float(benefit.mean()),
                repair_over_001=int(np.sum(benefit > .01)),
                damage_below_negative_001=int(np.sum(benefit < -.01)),
                mask75_repair=int(np.sum([r["mask75_repair"] for r in chosen])),
                mask75_damage=int(np.sum([r["mask75_damage"] for r in chosen])),
                oracle_radius_median=float(np.median([r["oracle_radius"] for r in chosen])),
                predictor_radius_median=float(np.median([r["predictor_radius"] for r in chosen])),
                effect_benefit_spearman=float(spearmanr(eff, benefit).statistic))


def cluster_difference(rows, select_a, select_b, field, bootstrap):
    ids = np.asarray([r["image_id"] for r in rows])
    images, inverse = np.unique(ids, return_inverse=True)
    a = np.asarray([select_a(r) for r in rows], dtype=bool)
    b = np.asarray([select_b(r) for r in rows], dtype=bool)
    values = np.asarray([r[field] for r in rows], dtype=np.float64)
    assert a.sum() and b.sum()
    asum = np.bincount(inverse, weights=np.where(a, values, 0), minlength=len(images))
    acount = np.bincount(inverse, weights=a.astype(np.int32), minlength=len(images))
    bsum = np.bincount(inverse, weights=np.where(b, values, 0), minlength=len(images))
    bcount = np.bincount(inverse, weights=b.astype(np.int32), minlength=len(images))
    rng = np.random.default_rng(20260927)
    draw = rng.integers(0, len(images), size=(bootstrap, len(images)), dtype=np.int32)
    delta = (asum[draw].sum(1) / np.maximum(acount[draw].sum(1), 1) -
             bsum[draw].sum(1) / np.maximum(bcount[draw].sum(1), 1))
    return dict(mean=float(values[a].mean() - values[b].mean()),
                ci95=np.quantile(delta, [.025, .975]).tolist(), bootstrap=bootstrap)


def main(args):
    oracle = {}
    for path in sorted((args.targets / "shards").glob("PART_*.pt")):
        for row in torch.load(path, map_location="cpu", weights_only=True):
            k = key(row)
            assert k not in oracle
            oracle[k] = (row["delta"].numpy(), row["gram"].numpy(),
                         float(row["original_iou"]), float(row["box_iou"]))
    assert len(oracle) == 71269
    retrieval = {}
    with gzip.open(args.retrieval, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            k = (int(row["image_id"]), int(row["annotation_id"]))
            assert k not in retrieval
            retrieval[k] = (float(row["h_feature_cos"]), float(row["h_top1"]))
    assert len(retrieval) == 71269
    rows, used = [], set()
    for fold, (score_dir, model_dir) in enumerate(zip(args.score_folds, args.model_folds)):
        cp = torch.load(model_dir / "BEST.pt", map_location="cpu", weights_only=True)
        assert cp["fold"] == fold
        trained_images = set(map(int, cp["fit_image_ids"]))
        for path in sorted((score_dir / "shards").glob("PART_*.pt")):
            for row in torch.load(path, map_location="cpu", weights_only=True):
                k = key(row)
                assert k not in used and k in oracle
                used.add(k)
                assert int(row["fold"]) == fold and k[0] not in trained_images
                target, gram, original_iou, box_iou = oracle[k]
                assert abs(float(row["original_iou"]) - original_iou) < 1e-5
                assert abs(float(row["box_iou"]) - box_iou) < 1e-5
                delta = row["correction"].numpy()
                coeff_cos = float(np.dot(target, delta) /
                                  max(np.linalg.norm(target) * np.linalg.norm(delta), 1e-20))
                hsim, htarget = retrieval[(k[0], k[1])]
                rows.append(dict(image_id=k[0], annotation_id=k[1], raw_id=k[2], fold=fold,
                                 original_iou=original_iou, box_iou=box_iou,
                                 oracle_radius=float(np.linalg.norm(target)),
                                 predictor_radius=float(np.linalg.norm(delta)),
                                 effect_cos=effect_cos(target, delta, gram),
                                 coefficient_cos=coeff_cos,
                                 benefit_iou=float(row["benefit_iou"]),
                                 mask75_repair=bool(row["mask75_repair"]),
                                 mask75_damage=bool(row["mask75_damage"]),
                                 h_nearest_similarity=hsim,
                                 h_nearest_target_effect_cos=htarget))
        print(json.dumps({"fold": fold, "joined": len(rows)}), flush=True)
    assert len(used) == len(oracle) == len(rows) == 71269
    groups = {
        "all": lambda r: True,
        "failure": lambda r: r["original_iou"] < .75,
        "success": lambda r: r["original_iou"] >= .75,
        "failure_good_box": lambda r: r["original_iou"] < .75 and r["box_iou"] >= .75,
        "success_good_box": lambda r: r["original_iou"] >= .75 and r["box_iou"] >= .75,
        "severe_failure": lambda r: r["original_iou"] < .5,
        "failure_good_box_high_h_similarity": lambda r: r["original_iou"] < .75 and r["box_iou"] >= .75 and r["h_nearest_similarity"] >= .9,
        "failure_good_box_high_h_but_wrong_direction": lambda r: r["original_iou"] < .75 and r["box_iou"] >= .75 and r["h_nearest_similarity"] >= .9 and r["h_nearest_target_effect_cos"] <= 0,
    }
    summary = {name: grouped(rows, test) for name, test in groups.items()}
    select_f = groups["failure_good_box"]
    select_s = groups["success_good_box"]
    summary["failure_minus_success_good_box_effect_cos"] = cluster_difference(
        rows, select_f, select_s, "effect_cos", args.bootstrap)
    summary["failure_minus_success_good_box_benefit_iou"] = cluster_difference(
        rows, select_f, select_s, "benefit_iou", args.bootstrap)
    summary["origin"] = dict(targets=str(args.targets), score_folds=[str(x) for x in args.score_folds],
                             model_folds=[str(x) for x in args.model_folds],
                             held_images=9924, candidates=len(rows), bootstrap=args.bootstrap)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with gzip.open(args.out / "ROWS.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(images=9924, candidates=len(rows),
        joined_without_duplicates=True, image_disjoint_check=True, replay_iou_check=True), indent=2))
    print(json.dumps({"complete": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--score-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--model-folds", type=Path, nargs=5, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    main(parser.parse_args())
