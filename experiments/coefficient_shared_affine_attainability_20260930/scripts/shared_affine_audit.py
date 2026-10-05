"""Convex reachability audit for the frozen native coefficient last layer.

The script consumes native-bank records (h, c, p, y, factor) and never runs a
GT-dependent correction at inference.  It solves an additive shared affine
map per feature-pyramid level and independent finite coefficient oracles under
the same ROI BCE + lambda displacement objective.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


LAMBDA = 0.003


def load_rows(path: Path, limit: int = 0):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        out.append({
            "level": int(r["meta"]["level"]),
            "h": r["h"].double(),
            "c": r["c"].double(),
            "p": r["p"].double(),
            "y": r["y"].double(),
            "area": float(len(r["y"]) / max(float(r["factor"]), 1e-12)),
            "image_id": int(r["meta"]["image_id"]),
            "annotation_id": int(r["meta"]["annotation_id"]),
        })
    return out


def batch_arrays(rows):
    """Pad one independent batch; returns tensors and valid-pixel mask."""
    b = len(rows)
    m = max(len(r["y"]) for r in rows)
    p = torch.zeros((b, m, 32), dtype=torch.float64)
    y = torch.zeros((b, m), dtype=torch.float64)
    valid = torch.zeros((b, m), dtype=torch.float64)
    c = torch.stack([r["c"] for r in rows])
    areas = torch.tensor([r["area"] for r in rows], dtype=torch.float64)
    for k, r in enumerate(rows):
        n = len(r["y"])
        p[k, :n] = r["p"]
        y[k, :n] = r["y"]
        valid[k, :n] = 1.0
    return p, y, valid, c, areas


def independent_objective(delta, p, y, valid, c0, areas):
    logits = torch.bmm(p, (c0 + delta).unsqueeze(-1)).squeeze(-1)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * valid).sum(1) / areas
    reg = LAMBDA * delta.square().sum(1) / 2.0
    return (bce + reg).mean(), bce.mean(), reg.mean()


def solve_independent(rows, max_iter=60, batch_size=32, save_delta=False):
    total = bce = reg = 0.0
    deltas = []
    stationarity = []
    start = time.monotonic()
    for lo in range(0, len(rows), batch_size):
        chunk = rows[lo:lo + batch_size]
        p, y, valid, c0, areas = batch_arrays(chunk)
        delta = torch.zeros((len(chunk), 32), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=max_iter,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            obj, _, _ = independent_objective(delta, p, y, valid, c0, areas)
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            obj, cb, cr = independent_objective(delta, p, y, valid, c0, areas)
        grad = torch.autograd.grad(independent_objective(delta, p, y, valid, c0, areas)[0], delta)[0]
        total += float(obj) * len(chunk)
        bce += float(cb) * len(chunk)
        reg += float(cr) * len(chunk)
        stationarity.append(float(grad.norm(dim=1).max()))
        if save_delta:
            deltas.append(delta.detach().float())
        if (lo // batch_size) % 10 == 0:
            print(json.dumps({"stage": "independent", "records": min(lo + len(chunk), len(rows)),
                              "total": len(rows), "elapsed": time.monotonic() - start}), flush=True)
    result = {"objective": total / len(rows), "bce": bce / len(rows), "regularizer": reg / len(rows),
              "max_batch_stationarity": max(stationarity), "records": len(rows)}
    return result, (torch.cat(deltas) if deltas else None)


def feature_stats(rows):
    stats = {}
    for level in range(3):
        hs = torch.stack([r["h"] for r in rows if r["level"] == level])
        stats[level] = (hs.mean(0), hs.std(0).clamp_min(1e-6))
    return stats


def affine_objective(A, rows, chunk_size=64, stats=None):
    totals = 0.0
    bces = 0.0
    regs = 0.0
    n = len(rows)
    for lo in range(0, n, chunk_size):
        chunk = rows[lo:lo + chunk_size]
        H = []
        for r in chunk:
            h = r["h"]
            if stats is not None:
                mean, std = stats[int(r["level"])]
                h = (h - mean) / std
            H.append(torch.cat((h, torch.ones(1, dtype=torch.float64))))
        H = torch.stack(H)
        levels = torch.tensor([r["level"] for r in chunk], dtype=torch.long)
        delta = torch.stack([H[k] @ A[int(levels[k])] for k in range(len(chunk))])
        p = torch.cat([r["p"] for r in chunk])
        y = torch.cat([r["y"] for r in chunk])
        idx = torch.cat([torch.full((len(r["y"]),), k, dtype=torch.long) for k, r in enumerate(chunk)])
        areas = torch.tensor([r["area"] for r in chunk], dtype=torch.float64)
        c0 = torch.stack([r["c"] for r in chunk])
        logits = (p * (c0[idx] + delta[idx])).sum(1)
        vals = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        bce_each = torch.zeros(len(chunk), dtype=torch.float64)
        bce_each.scatter_add_(0, idx, vals)
        bce_each = bce_each / areas
        reg_each = LAMBDA * delta.square().sum(1) / 2.0
        bces = bces + bce_each.sum()
        regs = regs + reg_each.sum()
        totals = totals + (bce_each + reg_each).sum()
    value = (totals / n, bces / n, regs / n)
    return value


def solve_shared(rows, max_iter=50, chunk_size=64, init_scale=0.0):
    stats = feature_stats(rows)
    A = [torch.randn((65, 32), dtype=torch.float64) * init_scale for _ in range(3)]
    for a in A:
        a.requires_grad_(True)
    opt = torch.optim.LBFGS(A, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)
    start = time.monotonic()

    def closure():
        opt.zero_grad()
        value, _, _ = affine_objective(A, rows, chunk_size=chunk_size, stats=stats)
        value.backward()
        return value

    opt.step(closure)
    value, bce, reg = affine_objective(A, rows, chunk_size=chunk_size, stats=stats)
    grads = torch.autograd.grad(value, A, allow_unused=True)
    grad_norm = max(float(g.norm()) if g is not None else 0.0 for g in grads)
    return [a.detach() for a in A], stats, {"objective": float(value), "bce": float(bce),
                                      "regularizer": float(reg), "stationarity_norm": grad_norm,
                                      "elapsed": time.monotonic() - start,
                                      "iterations": int(opt.state[A[0]].get("n_iter", -1))}


def eval_shared(rows, A, stats, chunk_size=64):
    with torch.no_grad():
        v, b, r = affine_objective(A, rows, chunk_size=chunk_size, stats=stats)
    return {"objective": float(v), "bce": float(b), "regularizer": float(r), "records": len(rows)}


def main(args):
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    fit = load_rows(args.fit, args.limit)
    dev = load_rows(args.dev, args.dev_limit)
    val = load_rows(args.val, args.val_limit)
    all_fit = {"records": len(fit), "images": len({r["image_id"] for r in fit}),
               "levels": {str(k): sum(r["level"] == k for r in fit) for k in range(3)}}
    print(json.dumps({"stage": "loaded", "fit": all_fit, "dev": len(dev), "val": len(val)}), flush=True)
    A, stats, shared_fit = solve_shared(fit, args.shared_iter, args.chunk_size, args.init_scale)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {"config": config, "lambda": LAMBDA, "fit": all_fit,
               "shared_fit": shared_fit, "original_fit": {"objective": float(affine_objective([torch.zeros((65,32),dtype=torch.float64)]*3, fit)[0]), "records": len(fit)}}
    summary["shared_dev"] = eval_shared(dev, A, stats, args.chunk_size)
    summary["shared_val"] = eval_shared(val, A, stats, args.chunk_size)
    if args.independent_limit:
        ind_rows = fit[:args.independent_limit]
        ind, deltas = solve_independent(ind_rows, args.independent_iter, args.independent_batch, save_delta=False)
        base = {"objective": float(affine_objective([torch.zeros((65,32),dtype=torch.float64)]*3, ind_rows)[0]), "records": len(ind_rows)}
        summary["independent_fit_prefix"] = ind
        summary["original_fit_prefix"] = base
        summary["decomposition_prefix"] = {"J0_minus_Jshared": base["objective"] - shared_fit["objective"],
                                             "Jshared_minus_Jind": shared_fit["objective"] - ind["objective"],
                                             "J0_minus_Jind": base["objective"] - ind["objective"]}
    torch.save({"A": A, "stats": stats}, args.out / "SHARED_AFFINE.pt")
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", "summary": str(args.out / 'SUMMARY.json')}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--shared-iter", type=int, default=50)
    p.add_argument("--independent-iter", type=int, default=60)
    p.add_argument("--independent-limit", type=int, default=0)
    p.add_argument("--independent-batch", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--init-scale", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=12)
    main(p.parse_args())
