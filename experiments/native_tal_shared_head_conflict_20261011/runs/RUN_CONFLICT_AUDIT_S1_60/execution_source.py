"""Read-only audit of instance gradient conflicts in native one-to-one cv4.

The script consumes the frozen official TAL cache and an existing PER_CANDIDATE
oracle/evaluation table.  It does not import a model, train, optimize, or write
weights.  It is intentionally self contained so the same source can be copied
to the laptop run directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ultralytics.utils import ops
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ultralytics is required for the official ROI crop") from exc

SEG_GAIN = 9.83241
BOOT_SEED = 20261011
BOOT_B = 2000
SPLITS = ("fit", "dev", "val")
ID_FIELDS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, with deterministic finite input handling."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x[keep]), rankdata(y[keep]))[0, 1])


def bootstrap_ci(values_a, values_b, stat, seed=BOOT_SEED, B=BOOT_B):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_a": int(len(a)), "n_b": int(len(b))}
    rng = np.random.default_rng(seed)
    estimate = float(stat(a, b))
    vals = np.empty(B, dtype=float)
    for k in range(B):
        vals[k] = stat(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
    return {"estimate": estimate, "lo": float(np.quantile(vals, .025)), "hi": float(np.quantile(vals, .975)), "n_a": int(len(a)), "n_b": int(len(b))}


def bootstrap_corr(rows, group_key, x_key, y_key, seed=BOOT_SEED, B=BOOT_B):
    """Bootstrap Spearman correlation either over candidates or image clusters."""
    valid = [r for r in rows if math.isfinite(float(r.get(x_key, float("nan")))) and math.isfinite(float(r.get(y_key, float("nan"))))]
    if group_key is None:
        groups = {str(i): [r] for i, r in enumerate(valid)}
    else:
        groups = defaultdict(list)
        for r in valid:
            groups[str(r[group_key])].append(r)
        groups = {k: v for k, v in groups.items()}
    keys = sorted(groups)
    if len(keys) < 3:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(keys)}
    if group_key is None:
        agg = [(float(v[0][x_key]), float(v[0][y_key])) for v in groups.values()]
    else:
        agg = [(float(np.mean([r[x_key] for r in v])), float(np.mean([r[y_key] for r in v]))) for v in groups.values()]
    arr = np.asarray(agg, dtype=float)
    est = spearman(arr[:, 0], arr[:, 1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        s = arr[rng.integers(0, len(arr), len(arr))]
        vals.append(spearman(s[:, 0], s[:, 1]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return {"estimate": float(est), "lo": float(np.quantile(vals, .025)) if len(vals) else float("nan"), "hi": float(np.quantile(vals, .975)) if len(vals) else float("nan"), "n": int(len(arr))}


def choose_images(index, max_images):
    chosen = {}
    for split in SPLITS:
        ids = [int(x["image_id"]) for x in index[split]]
        ids = sorted(ids, key=lambda iid: (hashlib.sha256(f"native-conflict-v1/{split}/{iid}".encode()).hexdigest(), iid))
        chosen[split] = ids[:max_images]
    return chosen


def load_rows(cache: Path, oracle_map, selected, smoke=False):
    rows = []
    manifest = []
    formula_errors = []
    for split in SPLITS:
        for iid in selected[split]:
            path = cache / "images" / f"{iid:012d}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            image = torch.load(path, map_location="cpu", weights_only=False)
            expected_source = "val" if split == "val" else "train"
            if str(image.get("source_split", expected_source)) != expected_source:
                raise RuntimeError(f"source split mismatch for image {iid}: {image.get('source_split')} != {expected_source}")
            proto = image["proto"].float()
            up = F.interpolate(proto[None], (640, 640), mode="bilinear", align_corners=False)[0]
            masks = image["masks"].float()
            for k, r in enumerate(image["rows"]):
                ident = {"split": split, "image_id": iid, "annotation_id": int(r["annotation_id"]), "branch": "one2one", "raw_id": int(r["raw_id"]), "pyramid_level": int(r["level"]), "target_gt_idx": int(r["gt_index"])}
                key = tuple(ident[f] for f in ID_FIELDS)
                if key not in oracle_map:
                    raise RuntimeError(f"oracle identity missing: {key}")
                o = oracle_map[key]
                box = image["target_boxes"][k].float()
                support = ops.crop_mask(torch.ones((1, 640, 640), dtype=torch.float32), box[None])[0].bool()
                if not bool(support.any()):
                    raise RuntimeError(f"empty ROI: {key}")
                owner = int(image["owners"][k])
                p = up[:, support].T.contiguous().double()
                y = (masks[support] == owner + 1).double()
                c = image["coeff"][int(r["raw_id"])].double()
                h = image["h"][int(r["raw_id"])].double()
                area = float(((box[2:] - box[:2]) / 640.0).prod() * 640.0 * 640.0)
                z = p @ c
                gc = (SEG_GAIN * (torch.sigmoid(z) - y) / area) @ p
                if smoke and len(formula_errors) < 4:
                    cc = c.detach().clone().requires_grad_(True)
                    loss = SEG_GAIN * (F.softplus(p @ cc) - y * (p @ cc)).sum() / area
                    ag = torch.autograd.grad(loss, cc)[0]
                    formula_errors.append(float((ag - gc).abs().max()))
                grad = torch.cat((torch.outer(gc, h).reshape(-1), gc)).numpy()
                norm = float(np.linalg.norm(grad))
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f"non-finite/zero gradient: {key}")
                rec = dict(ident)
                rec.update({"level": int(r["level"]), "area": area, "h_norm": float(h.norm()), "c_norm": float(c.norm()), "gradient_norm": norm, "gradient": (grad / norm).astype(np.float32), "iou_A": float(o["iou_A"]), "iou_C": float(o["iou_C"]), "objective_A": float(o["objective_A"]), "objective_C": float(o["objective_C"]), "oracle_gap": float(o["objective_A"] - o["objective_C"]), "recoverable": bool(float(o["iou_A"]) < .75 and float(o["iou_C"]) >= .75), "unrecoverable": bool(float(o["iou_A"]) < .75 and float(o["iou_C"]) < .75)})
                rows.append(rec)
                manifest.append(ident)
    return rows, manifest, formula_errors


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def make_pairs(rows):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["split"], r["level"], r["image_id"])].append(i)
    pair_rows = []
    per = {i: {"same_count": 0, "same_cos_sum": 0.0, "same_neg_sum": 0.0, "same_neg_mass_sum": 0.0, "cross_count": 0, "cross_cos_sum": 0.0, "cross_neg_sum": 0.0, "cross_neg_mass_sum": 0.0} for i in range(len(rows))}
    for (split, level, iid), inds in sorted(by.items()):
        for i, j in combinations(inds, 2):
            c = cosine(rows[i]["gradient"], rows[j]["gradient"])
            q = {"split": split, "level": level, "pair_type": "same_image", "i": i, "j": j, "cosine": c, "negative": int(c < 0), "negative_mass": max(0.0, -c), "image_i": iid, "image_j": iid}
            pair_rows.append(q)
            for u in (i, j):
                per[u]["same_count"] += 1; per[u]["same_cos_sum"] += c; per[u]["same_neg_sum"] += int(c < 0); per[u]["same_neg_mass_sum"] += max(0.0, -c)
    bylevel = defaultdict(list)
    quantiles = []
    for i, r in enumerate(rows):
        bylevel[(r["split"], r["level"])].append(i)
    for (split, level), inds in sorted(bylevel.items()):
        hn = np.asarray([rows[i]["h_norm"] for i in inds]); cn = np.asarray([rows[i]["c_norm"] for i in inds])
        hq = np.quantile(hn, [0.25, .5, .75]); cq = np.quantile(cn, [0.25, .5, .75])
        quantiles.append({"split": split, "level": level, "h_quantiles": [float(x) for x in hq], "c_quantiles": [float(x) for x in cq], "n": len(inds)})
        hb = np.searchsorted(hq, hn, side="right")
        cb = np.searchsorted(cq, cn, side="right")
        cells = defaultdict(list)
        for i, a, b in zip(inds, hb, cb): cells[(int(a), int(b))].append(i)
        for cell, vals in sorted(cells.items()):
            vals = sorted(vals, key=lambda i: (rows[i]["image_id"], rows[i]["annotation_id"], rows[i]["raw_id"]))
            used = set(); k = 0
            while k < len(vals):
                i = vals[k]
                if i in used: k += 1; continue
                jpos = next((p for p in range(k + 1, len(vals)) if vals[p] not in used and rows[vals[p]]["image_id"] != rows[i]["image_id"]), None)
                if jpos is None: k += 1; continue
                j = vals[jpos]; used.update((i, j))
                d = float(abs(hn[inds.index(i)] - hn[inds.index(j)]) + abs(cn[inds.index(i)] - cn[inds.index(j)]))
                c = cosine(rows[i]["gradient"], rows[j]["gradient"])
                q = {"split": split, "level": level, "pair_type": "cross_image_matched", "i": i, "j": j, "cosine": c, "negative": int(c < 0), "negative_mass": max(0.0, -c), "image_i": rows[i]["image_id"], "image_j": rows[j]["image_id"], "cell_h": cell[0], "cell_c": cell[1], "match_distance": d}
                pair_rows.append(q)
                for u in (i, j):
                    per[u]["cross_count"] += 1; per[u]["cross_cos_sum"] += c; per[u]["cross_neg_sum"] += int(c < 0); per[u]["cross_neg_mass_sum"] += max(0.0, -c)
                k += 1
    for i, r in enumerate(rows):
        s = per[i]
        r["same_pair_count"] = s["same_count"]; r["same_cos_mean"] = s["same_cos_sum"] / s["same_count"] if s["same_count"] else float("nan"); r["same_neg_frac"] = s["same_neg_sum"] / s["same_count"] if s["same_count"] else float("nan"); r["same_neg_mass"] = s["same_neg_mass_sum"] / s["same_count"] if s["same_count"] else float("nan")
        r["cross_pair_count"] = s["cross_count"]; r["cross_cos_mean"] = s["cross_cos_sum"] / s["cross_count"] if s["cross_count"] else float("nan"); r["cross_neg_frac"] = s["cross_neg_sum"] / s["cross_count"] if s["cross_count"] else float("nan"); r["cross_neg_mass"] = s["cross_neg_mass_sum"] / s["cross_count"] if s["cross_count"] else float("nan")
        r.pop("gradient", None)
    return pair_rows, quantiles


def group_metrics(rows, metric):
    out = {}
    for group in ("recoverable", "unrecoverable"):
        vals = [r[metric] for r in rows if r[group] and math.isfinite(float(r.get(metric, float("nan"))))]
        out[group] = {"mean": float(np.mean(vals)) if vals else float("nan"), "n": len(vals)}
    return out


def run(args):
    cache = Path(args.cache); oracle_path = Path(args.oracle); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = json.loads((cache / "INDEX.json").read_text(encoding="utf-8"))
    selected = choose_images(index, args.max_images)
    oracle = {}
    for o in read_jsonl(oracle_path):
        key = tuple(o.get(f) for f in ID_FIELDS)
        oracle[key] = o
    rows, manifest, formula_errors = load_rows(cache, oracle, selected, smoke=args.smoke)
    if args.smoke and len(rows) == 0: raise RuntimeError("smoke produced no rows")
    pairs, quantiles = make_pairs(rows)
    pair_out = []
    for p in pairs:
        q = dict(p); q["i_identity"] = {f: rows[p["i"]][f] for f in ID_FIELDS}; q["j_identity"] = {f: rows[p["j"]][f] for f in ID_FIELDS}; q.pop("i"); q.pop("j"); pair_out.append(q)
    per_out = []
    for r in rows:
        q = {k: v for k, v in r.items() if k != "gradient"}; per_out.append(q)
    with (out / "PER_INSTANCE.jsonl").open("w", encoding="utf-8") as f:
        for r in per_out: f.write(json.dumps(r, sort_keys=True, allow_nan=True) + "\n")
    with (out / "PER_PAIR.jsonl").open("w", encoding="utf-8") as f:
        for p in pair_out: f.write(json.dumps(p, sort_keys=True, allow_nan=True) + "\n")
    (out / "PAIR_QUANTILES.json").write_text(json.dumps(quantiles, indent=2), encoding="utf-8")
    (out / "INPUT_MANIFEST.json").write_text(json.dumps({"selected_images": selected, "identity_count": len(manifest), "identity_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), "oracle_sha256": sha256_file(oracle_path), "cache_index_sha256": sha256_file(cache / "INDEX.json")}, indent=2), encoding="utf-8")
    metrics = ["same_neg_frac", "cross_neg_frac", "same_cos_mean", "cross_cos_mean", "same_neg_mass", "cross_neg_mass"]
    summary = {"counts": {"rows": len(rows), "pairs": len(pairs), "images": {s: len(selected[s]) for s in SPLITS}, "pairs_by_type": {t: sum(1 for p in pairs if p["pair_type"] == t) for t in ("same_image", "cross_image_matched")}}, "selected_images": selected, "groups": {}, "formula_check": {"max_abs_error": float(max(formula_errors)) if formula_errors else None, "errors": formula_errors, "threshold": 3e-5}}
    boot = {"seed": BOOT_SEED, "B": BOOT_B, "metrics": {}}
    for split in SPLITS:
        sr = [r for r in rows if r["split"] == split and float(r["iou_A"]) < .75]
        summary["groups"][split] = {"recoverable": sum(r["recoverable"] for r in sr), "unrecoverable": sum(r["unrecoverable"] for r in sr), "recoverable_images": len({r["image_id"] for r in sr if r["recoverable"]}), "unrecoverable_images": len({r["image_id"] for r in sr if r["unrecoverable"]})}
        for metric in metrics:
            gm = group_metrics(sr, metric); summary.setdefault("means", {}).setdefault(split, {})[metric] = gm
            a = [r[metric] for r in sr if r["recoverable"]]; b = [r[metric] for r in sr if r["unrecoverable"]]
            stat = lambda x, y: float(np.mean(x) - np.mean(y))
            ci = bootstrap_ci(a, b, stat)
            # image-macro: aggregate each metric within image and group, then resample image clusters.
            ia = defaultdict(list); ib = defaultdict(list)
            for r in sr:
                if not math.isfinite(float(r.get(metric, float("nan")))): continue
                (ia if r["recoverable"] else ib if r["unrecoverable"] else defaultdict(list))[r["image_id"]].append(r[metric])
            image_a = [float(np.mean(v)) for v in ia.values()]; image_b = [float(np.mean(v)) for v in ib.values()]
            ici = bootstrap_ci(image_a, image_b, stat, seed=BOOT_SEED + 17)
            boot["metrics"].setdefault(split, {})[metric] = {"candidate": ci, "image_macro": ici}
    for split in SPLITS:
        sr = [r for r in rows if r["split"] == split and float(r["iou_A"]) < .75 and r["cross_pair_count"] > 0]
        boot.setdefault("correlation", {}).setdefault(split, {})["candidate_cross_conflict_vs_oracle_gap"] = bootstrap_corr(sr, None, "cross_neg_frac", "oracle_gap")
        boot["correlation"][split]["image_macro_cross_conflict_vs_oracle_gap"] = bootstrap_corr(sr, "image_id", "cross_neg_frac", "oracle_gap", seed=BOOT_SEED + 31)
        boot["correlation"][split]["point_dev_cross"] = spearman(np.asarray([r["cross_neg_frac"] for r in sr if r["split"] == "dev" and math.isfinite(r["cross_neg_frac"]) and math.isfinite(r["oracle_gap"])]), np.asarray([r["oracle_gap"] for r in sr if r["split"] == "dev" and math.isfinite(r["cross_neg_frac"]) and math.isfinite(r["oracle_gap"])])) if split == "dev" else None
    # Strict pre-registered gate on val; positive iff every required lower bound is >0.
    v = summary["groups"]["val"]; gate_counts = min(v["recoverable"], v["unrecoverable"], v["recoverable_images"], v["unrecoverable_images"]) >= 0
    gate_counts = (v["recoverable"] >= 50 and v["unrecoverable"] >= 50 and v["recoverable_images"] >= 20 and v["unrecoverable_images"] >= 20)
    req = [boot["metrics"]["val"][m][k]["lo"] for m in ("same_neg_frac", "cross_neg_frac") for k in ("candidate", "image_macro")]
    corr_lo = boot["correlation"]["val"]["candidate_cross_conflict_vs_oracle_gap"]["lo"]
    dev_est = boot["correlation"]["dev"]["candidate_cross_conflict_vs_oracle_gap"]["estimate"]
    gate_positive = bool(gate_counts and all(math.isfinite(x) and x > 0 for x in req) and math.isfinite(corr_lo) and corr_lo > 0 and math.isfinite(dev_est) and dev_est > 0)
    decision = {"status": "positive_authorize_conflict_routing" if gate_positive else "negative_close_method_line" if gate_counts else "unknown_insufficient_groups", "gate_counts": gate_counts, "required_lower_bounds": req, "val_correlation_lower": corr_lo, "dev_correlation_point": dev_est, "overturn": "If an independently rerun val_small with the frozen protocol makes every required lower bound positive and dev correlation positive, reopen conflict-aware routing; otherwise remain closed."}
    (out / "BOOTSTRAP.json").write_text(json.dumps(boot, indent=2, allow_nan=True), encoding="utf-8")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2, allow_nan=True), encoding="utf-8")
    run = {"run_id": out.name, "status": "complete", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": " ".join(sys.argv), "cache": str(cache), "oracle": str(oracle_path), "max_images": args.max_images, "smoke": bool(args.smoke), "torch": torch.__version__, "python": platform.python_version(), "script_sha256": sha256_file(Path(__file__))}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out / "COMPLETE.json").write_text(json.dumps({"status": "complete", "decision": decision["status"]}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows), "pairs": len(pairs), "decision": decision["status"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args())
