"""Solve one frozen shared affine objective on a common official cache.

The solver is deliberately full-candidate and analytic.  It reads rows made by
``cross_objectives.load_image_rows`` and never changes candidate identity,
boxes, prototypes, or labels.  Pixels are transferred in chunks, so a CUDA run
does not retain the complete pixel bank on an 8 GB device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .cross_objectives import LAMBDA, load_coco_for_split, load_image_rows
except ImportError:  # direct ``python scripts/solve_cross.py`` execution
    from cross_objectives import LAMBDA, load_coco_for_split, load_image_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_items(index: dict[str, Any], count: int | None, seed: int) -> list[dict[str, Any]]:
    items = list(index["fit"])
    if count is None or count <= 0 or count >= len(items):
        return sorted(items, key=lambda x: int(x["image_id"]))
    rng = random.Random(seed)
    return sorted(rng.sample(items, count), key=lambda x: int(x["image_id"]))


def prepare_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for level in range(3):
        hs = [torch.as_tensor(r["h"], dtype=torch.float64) for r in rows if int(r["pyramid_level"]) == level]
        if not hs:
            raise RuntimeError(f"fit subset has no rows at pyramid level {level}")
        stacked = torch.stack(hs)
        stats[level] = (stacked.mean(0), stacked.std(0).clamp_min(1e-12))
    prepared: list[dict[str, Any]] = []
    for row in rows:
        level = int(row["pyramid_level"])
        mean, std = stats[level]
        h = torch.as_tensor(row["h"], dtype=torch.float64)
        h_aug = torch.cat(((h - mean) / std, torch.ones(1, dtype=torch.float64)))
        prepared.append({**row, "h_aug": h_aug})
    return prepared, stats


def zeros(stats: dict[int, tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> list[torch.Tensor]:
    width = next(iter(stats.values()))[0].numel() + 1
    return [torch.zeros((width, 32), dtype=torch.float64, device=device, requires_grad=True) for _ in range(3)]


@torch.no_grad()
def evaluate(
    A: list[torch.Tensor],
    rows: list[dict[str, Any]],
    device: torch.device,
    chunk_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], float]:
    """Return exact candidate-mean objective/BCE/reg and analytic A gradients."""
    if not rows:
        raise ValueError("empty fit rows")
    total = torch.zeros((), dtype=torch.float64, device=device)
    bce_total = torch.zeros_like(total)
    reg_total = torch.zeros_like(total)
    gradients = [torch.zeros_like(a) for a in A]
    max_row_grad = 0.0
    denominator = float(len(rows))
    for row in rows:
        level = int(row["pyramid_level"])
        h_aug = row["h_aug"].to(device=device)
        c = torch.as_tensor(row["c"], dtype=torch.float64, device=device)
        p_cpu = torch.as_tensor(row["p"], dtype=torch.float32)
        y_cpu = torch.as_tensor(row["y"], dtype=torch.float32)
        if p_cpu.ndim != 2 or p_cpu.shape[1] != 32 or y_cpu.numel() != p_cpu.shape[0]:
            raise ValueError(f"bad p/y shape for {row.get('annotation_id')}: {tuple(p_cpu.shape)} {tuple(y_cpu.shape)}")
        area = float(row["area"])
        gain = float(row["gain"])
        if not math.isfinite(area) or area <= 0 or not math.isfinite(gain):
            raise ValueError(f"invalid area/gain for row {row.get('annotation_id')}: {area}/{gain}")
        delta = h_aug @ A[level]
        coeff = c + delta
        row_bce = torch.zeros((), dtype=torch.float64, device=device)
        row_gd = torch.zeros((32,), dtype=torch.float64, device=device)
        for lo in range(0, p_cpu.shape[0], chunk_pixels):
            hi = min(lo + chunk_pixels, p_cpu.shape[0])
            p = p_cpu[lo:hi].to(device=device, dtype=torch.float64)
            y = y_cpu[lo:hi].to(device=device, dtype=torch.float64)
            logits = p @ coeff
            err = gain * (torch.sigmoid(logits) - y) / area
            row_bce = row_bce + gain * (F.softplus(logits) - y * logits).sum() / area
            row_gd = row_gd + err @ p
        row_reg = LAMBDA * delta.square().sum() / 2.0
        row_value = row_bce + row_reg
        total = total + row_value / denominator
        bce_total = bce_total + row_bce / denominator
        reg_total = reg_total + row_reg / denominator
        # The fitted objective includes the per-row coefficient regularizer.
        # Its exact gradient is lambda * h_aug outer delta; omitting it would
        # optimize a different function while reporting the regularized loss.
        row_gd = row_gd + LAMBDA * delta
        row_gnorm = float(row_gd.abs().max().cpu())
        max_row_grad = max(max_row_grad, row_gnorm)
        gradients[level].add_(torch.outer(h_aug, row_gd) / denominator)
    gradient_inf = max(float(g.abs().max().cpu()) for g in gradients)
    return total, bce_total, reg_total, gradients, gradient_inf


def solve(rows: list[dict[str, Any]], stats: dict[int, tuple[torch.Tensor, torch.Tensor]], device: torch.device, args: argparse.Namespace) -> tuple[list[torch.Tensor], dict[str, Any]]:
    A = zeros(stats, device)
    trajectory: list[dict[str, Any]] = []
    start = time.monotonic()

    def run_eval(stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], float]:
        value, bce, reg, gradients, grad_inf = evaluate(A, rows, device, args.chunk_pixels)
        trajectory.append({
            "stage": stage,
            "objective": float(value.cpu()),
            "bce": float(bce.cpu()),
            "regularizer": float(reg.cpu()),
            "gradient_inf": grad_inf,
            "elapsed_s": time.monotonic() - start,
        })
        return value, bce, reg, gradients, grad_inf

    initial, initial_bce, initial_reg, _, initial_grad = run_eval("initial")
    converged = False
    blocks_used = 0
    for block in range(args.max_blocks):
        blocks_used = block + 1
        optimizer = torch.optim.LBFGS(
            A,
            lr=args.lr,
            max_iter=args.block_iters,
            max_eval=args.block_iters * 2,
            tolerance_grad=args.tolerance,
            tolerance_change=args.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            value, bce, reg, gradients, grad_inf = evaluate(A, rows, device, args.chunk_pixels)
            for parameter, gradient in zip(A, gradients):
                parameter.grad = gradient
            trajectory.append({
                "stage": f"lbfgs_block_{block + 1}",
                "objective": float(value.cpu()),
                "bce": float(bce.cpu()),
                "regularizer": float(reg.cpu()),
                "gradient_inf": grad_inf,
                "elapsed_s": time.monotonic() - start,
            })
            return value

        optimizer.step(closure)
        block_value, _, _, _, grad_inf = run_eval(f"after_block_{block + 1}")
        print(json.dumps({"stage": f"after_block_{block + 1}", "objective": float(block_value.cpu()), "gradient_inf": grad_inf, "elapsed_s": time.monotonic() - start}), flush=True)
        if grad_inf <= args.tolerance:
            converged = True
            break
    final, final_bce, final_reg, _, final_grad = run_eval("final")
    result = {
        "initial": {"objective": float(initial.cpu()), "bce": float(initial_bce.cpu()), "regularizer": float(initial_reg.cpu()), "gradient_inf": initial_grad},
        "final": {"objective": float(final.cpu()), "bce": float(final_bce.cpu()), "regularizer": float(final_reg.cpu()), "gradient_inf": final_grad},
        "converged": converged,
        "gradient_inf": final_grad,
        "blocks_used": blocks_used,
        "trajectory": trajectory,
        "elapsed_s": time.monotonic() - start,
    }
    return [a.detach().cpu() for a in A], result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO annotations directory or instances_*.json")
    parser.add_argument("--mode", choices=("old160", "official640"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chunk-pixels", type=int, default=16384)
    parser.add_argument("--block-iters", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--tolerance-change", type=float, default=1e-12)
    args = parser.parse_args()
    if args.chunk_pixels <= 0 or args.block_iters <= 0 or args.max_blocks <= 0:
        raise ValueError("chunk-pixels, block-iters and max-blocks must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.cache / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    items = choose_items(index, args.fit_images, args.seed)
    coco = load_coco_for_split(args.annotations, "train")
    rows: list[dict[str, Any]] = []
    for item in items:
        iid = int(item["image_id"])
        image = torch.load(args.cache / "images" / f"{iid:012d}.pt", map_location="cpu", weights_only=False)
        image["image_id"] = iid
        rows.extend(load_image_rows(image, coco, args.mode))
    if not rows:
        raise RuntimeError("no candidate rows loaded")
    prepared, stats = prepare_rows(rows)
    A, solver = solve(prepared, stats, device, args)
    out_model = args.out / f"SHARED_{args.mode}.pt"
    torch.save({"A": A, "stats": stats, "mode": args.mode, "gradient_inf": solver["gradient_inf"], "converged": solver["converged"], "fit_rows": len(prepared), "fit_images": len(items)}, out_model)
    summary = {
        "status": "complete",
        "mode": args.mode,
        "device": str(device),
        "fit_images": len(items),
        "fit_rows": len(prepared),
        "fit_image_ids": [int(x["image_id"]) for x in items],
        "target": {"segmentation_gain": float(prepared[0]["gain"]), "lambda": LAMBDA, "target_mode": args.mode},
        "solver": solver,
        "reference": str(out_model),
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    run = {
        "run_id": args.out.name,
        "status": "completed",
        "command": sys.argv,
        "created_at": time.time(),
        "mode": args.mode,
        "cache": str(args.cache),
        "cache_index_sha256": sha256(index_path),
        "annotations": str(args.annotations),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "config": vars(args) | {"out": str(args.out)},
        "summary": "SUMMARY.json",
    }
    run["model_sha256"] = sha256(out_model)
    (args.out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"completed": True, "mode": args.mode, "fit_rows": len(prepared), "gradient_inf": solver["gradient_inf"], "converged": solver["converged"], "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
