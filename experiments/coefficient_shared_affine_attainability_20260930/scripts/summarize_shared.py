"""Summarize fixed-cache original versus frozen shared-affine outputs."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import shared_affine_audit as audit


def values(rows, A, stats):
    out = []
    for r in rows:
        h = r["h"]
        mean, std = stats[r["level"]]
        hs = (h - mean) / std
        d = torch.cat((hs, torch.ones(1, dtype=torch.float64))) @ A[r["level"]]
        z0 = r["p"] @ r["c"]
        z1 = r["p"] @ (r["c"] + d)
        b0 = float(F.binary_cross_entropy_with_logits(z0, r["y"], reduction="none").sum() / r["area"])
        b1 = float(F.binary_cross_entropy_with_logits(z1, r["y"], reduction="none").sum() / r["area"])
        reg = float(0.003 * d.square().sum() / 2)
        out.append(dict(image_id=r["image_id"], annotation_id=r["annotation_id"], level=r["level"],
                        original_bce=b0, shared_bce=b1, shared_objective=b1 + reg, regularizer=reg))
    return out


def bootstrap(rows, field, seed=20260930, draws=2000):
    groups = {}
    for r in rows:
        groups.setdefault(r["image_id"], []).append(r[field] - r["original_bce"] if field == "shared_bce" else r[field] - r["original_objective"])
    # fields are either shared_bce/shared_objective; make explicit below.
    keys = sorted(groups)
    vals = np.asarray([np.mean(groups[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    samples = vals[rng.integers(0, len(vals), size=(draws, len(vals)))].mean(1)
    return {"images": len(keys), "mean": float(vals.mean()),
            "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}


def main(a):
    obj = torch.load(a.affine, map_location="cpu", weights_only=False)
    A, stats = obj["A"], obj["stats"]
    result = {"lambda": .003, "splits": {}}
    for name, path in (("fit", a.fit), ("dev", a.dev), ("val", a.val)):
        rows = audit.load_rows(path)
        rr = values(rows, A, stats)
        for r in rr:
            r["original_objective"] = r["original_bce"]
        mean_orig = float(np.mean([r["original_bce"] for r in rr]))
        mean_shared = float(np.mean([r["shared_bce"] for r in rr]))
        mean_obj = float(np.mean([r["shared_objective"] for r in rr]))
        levels = {}
        for lv in range(3):
            q = [r for r in rr if r["level"] == lv]
            levels[str(lv)] = {"records": len(q), "original_bce": float(np.mean([r["original_bce"] for r in q])),
                               "shared_bce": float(np.mean([r["shared_bce"] for r in q])),
                               "shared_objective": float(np.mean([r["shared_objective"] for r in q]))}
        # image-clustered bootstrap of shared minus original BCE and objective.
        gb = {}
        go = {}
        for r in rr:
            gb.setdefault(r["image_id"], []).append(r["shared_bce"] - r["original_bce"])
            go.setdefault(r["image_id"], []).append(r["shared_objective"] - r["original_objective"])
        rng = np.random.default_rng(20260930)
        keys = sorted(gb)
        db = np.asarray([np.mean(gb[k]) for k in keys])
        do = np.asarray([np.mean(go[k]) for k in keys])
        idx = rng.integers(0, len(keys), size=(2000, len(keys)))
        result["splits"][name] = {
            "records": len(rr), "images": len(keys), "original_bce": mean_orig,
            "shared_bce": mean_shared, "shared_objective": mean_obj,
            "delta_bce": mean_shared - mean_orig, "delta_objective": mean_obj - mean_orig,
            "bootstrap_image_delta_bce": [float(np.quantile(db[idx].mean(1), .025)), float(np.quantile(db[idx].mean(1), .975))],
            "bootstrap_image_delta_objective": [float(np.quantile(do[idx].mean(1), .025)), float(np.quantile(do[idx].mean(1), .975))],
            "levels": levels,
        }
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--affine", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
