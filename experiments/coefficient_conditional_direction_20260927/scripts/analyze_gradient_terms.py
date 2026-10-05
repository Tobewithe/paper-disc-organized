"""Decompose official GT-box coefficient gradients for high-h neighbor pairs."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

from analyze_conditional_direction import (
    load_bank, matching_groups, retrieve, effect_cos,
)


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20))


def selected_pairs(bank, manifest, reference, per_group):
    ids = {(int(bank["image_id"][i]), int(bank["annotation_id"][i])): i
           for i in range(len(bank["image_id"]))}
    assert len(ids) == 71269
    with gzip.open(reference, "rt", encoding="utf-8") as stream:
        ref = {(int(row["image_id"]), int(row["annotation_id"])): row
               for row in map(json.loads, stream)}
    assert set(ref) == set(ids)
    eligible = [(ids[key], row) for key, row in ref.items()
                if row["original_iou"] < .75 and row["box_iou"] >= .75
                and row["h_feature_cos"] >= .9]
    candidates = {
        "discordant": sorted([i for i, r in eligible if r["h_top1"] <= 0]),
        "aligned": sorted([i for i, r in eligible if r["h_top1"] >= .5]),
    }
    rng = np.random.default_rng(20260927)
    chosen = {name: np.sort(rng.choice(values, size=per_group, replace=False))
              for name, values in candidates.items()}
    pairs = []
    for fold in range(5):
        fit = np.flatnonzero(bank["fold"] != fold).astype(np.int32)
        query = np.asarray([int(i) for group in chosen.values() for i in group
                            if bank["fold"][i] == fold], dtype=np.int32)
        if not len(query):
            continue
        x = bank["h"]
        mean, std = x[fit].mean(0), x[fit].std(0)
        x = (x - mean) / np.maximum(std, .01)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        pools, groups, _ = matching_groups(bank, fit, query)
        top, similarity = retrieve(x.astype(np.float32), pools, groups, len(bank["image_id"]))
        for i in query:
            j = int(top[i, 0])
            key = (int(bank["image_id"][i]), int(bank["annotation_id"][i]))
            assert abs(float(similarity[i]) - ref[key]["h_feature_cos"]) < 1e-5
            transferred = float(effect_cos(
                bank["direction"][j], bank["direction"][i], bank["gram"][i])[0])
            assert abs(transferred - ref[key]["h_top1"]) < 1e-4
            label = "discordant" if i in chosen["discordant"] else "aligned"
            pairs.append(dict(group=label, query=int(i), neighbor=j,
                              image_id=key[0], h_cos=float(similarity[i]),
                              oracle_effect_cos=transferred))
    assert len(pairs) == 2 * per_group
    assert all(bank["image_id"][p["query"]] != bank["image_id"][p["neighbor"]]
               for p in pairs)
    return pairs, candidates


def moments_for_pairs(bank, pairs, cache_dir, official_source):
    sys.path.insert(0, str(official_source / "scripts"))
    from official_pipeline import target_rois

    wanted = defaultdict(set)
    for pair in pairs:
        for i in (pair["query"], pair["neighbor"]):
            wanted[int(bank["image_id"][i])].add(int(bank["annotation_id"][i]))
    index = {(int(bank["image_id"][i]), int(bank["annotation_id"][i])): i
             for i in range(len(bank["image_id"]))}
    result = {}
    for step, (iid, aids) in enumerate(sorted(wanted.items()), 1):
        path = cache_dir / f"{iid:012d}.pt"
        cache = torch.load(path, map_location="cpu", weights_only=True)
        encoded = np.frombuffer(zlib.decompress(cache["mask_zlib"]), dtype="<u2").copy()
        mask = torch.from_numpy(encoded.reshape(cache["mask_shape"]).astype(np.int64))
        image = dict(cache, masks=mask, proto=cache["proto"].float())
        positions = {int(row["annotation_id"]): j for j, row in enumerate(cache["rows"])}
        assert aids <= set(positions)
        sorted_aids = sorted(aids)
        rois = target_rois(image, [positions[aid] for aid in sorted_aids])
        with torch.no_grad():
            for aid, (pixels, y, area) in zip(sorted_aids, rois):
                pos = positions[aid]
                i = index[(iid, aid)]
                c0 = cache["coeff"][pos].cuda().float()
                oracle = torch.from_numpy(bank["delta"][i]).cuda()
                z0 = pixels @ c0
                probability = z0.sigmoid()
                s = pixels.T @ probability / area
                t = pixels.T @ y / area
                g = s - t
                weight = probability * (1 - probability)
                hessian = pixels.T @ (pixels * weight[:, None]) / area
                hessian = hessian + .003 * torch.eye(32, device="cuda")
                newton = torch.linalg.solve(hessian, -g)
                final_grad = pixels.T @ ((pixels @ (c0 + oracle)).sigmoid() - y) / area
                final_grad = final_grad + .003 * oracle
                result[i] = dict(s=s.cpu().numpy(), t=t.cpu().numpy(), g=g.cpu().numpy(),
                                 area=float(area), positives=int(y.sum()),
                                 s_t_cos=float(torch.dot(s, t) / (s.norm() * t.norm()).clamp_min(1e-8)),
                                 cancellation_ratio=float((s.norm() + t.norm()) / g.norm().clamp_min(1e-8)),
                                 newton_effect_cos=float(effect_cos(newton.cpu().numpy(),
                                     oracle.cpu().numpy(), bank["gram"][i])[0]),
                                 stationarity_abs=float(final_grad.norm()),
                                 stationarity_relative=float(final_grad.norm() /
                                                             g.norm().clamp_min(1e-8)))
        del cache, image, rois
        if step % 100 == 0:
            print(json.dumps({"images_processed": step, "total_images": len(wanted)}), flush=True)
    assert set(result) == {i for pair in pairs for i in (pair["query"], pair["neighbor"])}
    return result


def bootstrap_mean(rows, field, repetitions=2000):
    images, inv = np.unique([r["image_id"] for r in rows], return_inverse=True)
    values = np.asarray([r[field] for r in rows])
    count = np.bincount(inv, minlength=len(images))
    sums = np.bincount(inv, weights=values, minlength=len(images))
    rng = np.random.default_rng(20260927)
    draw = rng.integers(0, len(images), size=(repetitions, len(images)), dtype=np.int32)
    boot = sums[draw].sum(1) / count[draw].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())


def main(args):
    torch.set_num_threads(6)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank = load_bank(args.targets / "shards", args.annotations, manifest)
    pairs, candidate_count = selected_pairs(bank, manifest, args.reference, args.per_group)
    moments = moments_for_pairs(bank, pairs, args.cache, args.official_source)
    rows = []
    for pair in pairs:
        q, n = moments[pair["query"]], moments[pair["neighbor"]]
        row = dict(pair, cos_s=cosine(q["s"], n["s"]),
                   cos_t=cosine(q["t"], n["t"]), cos_g=cosine(q["g"], n["g"]),
                   q_s_t_cos=q["s_t_cos"], n_s_t_cos=n["s_t_cos"],
                   q_cancellation_ratio=q["cancellation_ratio"],
                   n_cancellation_ratio=n["cancellation_ratio"],
                   q_newton_effect_cos=q["newton_effect_cos"],
                   n_newton_effect_cos=n["newton_effect_cos"],
                   q_stationarity_abs=q["stationarity_abs"],
                   n_stationarity_abs=n["stationarity_abs"],
                   q_stationarity_relative=q["stationarity_relative"],
                   n_stationarity_relative=n["stationarity_relative"])
        row["s_minus_t_cos"] = row["cos_s"] - row["cos_t"]
        rows.append(row)
    result = {}
    for label in ("discordant", "aligned"):
        sub = [r for r in rows if r["group"] == label]
        result[label] = dict(pairs=len(sub), images=len({r["image_id"] for r in sub}),
                             cos_s=bootstrap_mean(sub, "cos_s"),
                             cos_t=bootstrap_mean(sub, "cos_t"),
                             cos_g=bootstrap_mean(sub, "cos_g"),
                             s_minus_t_cos=bootstrap_mean(sub, "s_minus_t_cos"),
                             query_s_t_cos=bootstrap_mean(sub, "q_s_t_cos"),
                             query_cancellation_ratio_median=float(np.median(
                                 [r["q_cancellation_ratio"] for r in sub])),
                             query_cancellation_ratio_p90=float(np.quantile(
                                 [r["q_cancellation_ratio"] for r in sub], .9)),
                             query_newton_effect_cos=bootstrap_mean(sub, "q_newton_effect_cos"),
                             oracle_effect_cos=float(np.mean([r["oracle_effect_cos"] for r in sub])),
                             h_cos=float(np.mean([r["h_cos"] for r in sub])))
    stations = np.asarray([x["stationarity_abs"] for x in moments.values()])
    relative = np.asarray([x["stationarity_relative"] for x in moments.values()])
    result["audit"] = dict(selected_per_group=args.per_group,
                           available={k:len(v) for k,v in candidate_count.items()},
                           unique_instances=len(moments), unique_images=len({int(bank["image_id"][i]) for i in moments}),
                           cached_prototype_dtype="float16_to_float32",
                           oracle_stationarity_abs_median=float(np.median(stations)),
                           oracle_stationarity_abs_p95=float(np.quantile(stations, .95)),
                           oracle_stationarity_relative_median=float(np.median(relative)),
                           oracle_stationarity_relative_p95=float(np.quantile(relative, .95)))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with gzip.open(args.out / "PAIRS.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.out / "COMPLETE.json").write_text(json.dumps(dict(pairs=len(rows),
        unique_instances=len(moments), unique_images=result["audit"]["unique_images"],
        exploratory=True), indent=2))
    print(json.dumps(result["audit"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "targets", "annotations", "reference", "cache",
                 "official_source", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=400)
    main(parser.parse_args())
