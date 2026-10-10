"""Small official-TAL-aligned training pilot for a frozen native coefficient head.

The script intentionally trains only the additive 65->32 affine increment over
the cached native coefficient output.  Its loss is the same official-TAL ROI
loss used by coefficient_official_tal_affine_20260930: segmentation gain
9.83241, candidate-mean GT-box BCE, and an explicit lambda=0.003 output
displacement penalty.  No assignment, box, prototype, or upstream feature is
changed.  The FP64 L-BFGS reference and FP32 Adam/SGD arms use the same rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


SEG_GAIN = 9.83241
LAMBDA = 0.003
DEFAULT_SEED = 20261011


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def choose_items(cache: Path, group: str, count: int, seed: int) -> list[dict]:
    index = json.loads((cache / "INDEX.json").read_text(encoding="utf-8"))
    items = list(index[group])
    if count <= 0 or count >= len(items):
        return items
    rng = random.Random(seed + {"fit": 0, "dev": 1, "val": 2}[group])
    selected = rng.sample(items, count)
    return sorted(selected, key=lambda x: int(x["image_id"]))


def load_rows(cache: Path, items: Iterable[dict]) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    image_ids: list[int] = []
    for item in items:
        image_id = int(item["image_id"])
        image_ids.append(image_id)
        image = torch.load(cache / "images" / f"{image_id:012d}.pt", map_location="cpu", weights_only=False)
        proto = image["proto"].float()
        proto_640 = F.interpolate(proto[None], (640, 640), mode="bilinear", align_corners=False)[0]
        masks = image["masks"].float()
        for k, record in enumerate(image["rows"]):
            box = image["target_boxes"][k].float()
            # Exact official support convention: x/y >= lower bound and < upper bound.
            yy, xx = torch.meshgrid(torch.arange(640), torch.arange(640), indexing="ij")
            support = ((xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3]))
            if not bool(support.any()):
                raise RuntimeError(f"empty official support: {group_name(item)} {image_id} {record['annotation_id']}")
            owner = int(image["owners"][k])
            rows.append({
                "image_id": image_id,
                "annotation_id": int(record["annotation_id"]),
                "branch": "one2one",
                "raw_id": int(record["raw_id"]),
                "pyramid_level": int(record["level"]),
                "target_gt_idx": int(record["gt_index"]),
                "h": image["h"][int(record["raw_id"])].float(),
                "c": image["coeff"][int(record["raw_id"])].float(),
                "p": proto_640[:, support].T.contiguous(),
                "y": (masks[support] == owner + 1).float().contiguous(),
                "area": float(((box[2:] - box[:2]) / 640.0).prod() * 640.0 * 640.0),
                "box_iou": float(record.get("box_iou", float("nan"))),
            })
        del image
    return rows, image_ids


def group_name(item: dict) -> str:
    return str(item.get("split", "official"))


def feature_stats(rows: list[dict]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for level in range(3):
        level_rows = [row["h"] for row in rows if row["pyramid_level"] == level]
        if not level_rows:
            raise RuntimeError(f"fit subset has no official candidates at pyramid level {level}")
        h = torch.stack(level_rows).double()
        result[level] = (h.mean(0), h.std(0).clamp_min(1e-6))
    return result


def prepare_rows(rows: list[dict], stats: dict[int, tuple[torch.Tensor, torch.Tensor]], dtype: torch.dtype, device: torch.device) -> list[dict]:
    prepared = []
    for row in rows:
        mean, std = stats[row["pyramid_level"]]
        h = torch.cat(((row["h"].double() - mean) / std, torch.ones(1, dtype=torch.float64))).to(dtype=dtype, device=device)
        prepared.append({**row, "h_aug": h, "p": row["p"].to(dtype=dtype, device=device), "y": row["y"].to(dtype=dtype, device=device), "c": row["c"].to(dtype=dtype, device=device)})
    return prepared


def values_for_rows(A: list[torch.Tensor], rows: list[dict], need_parts: bool = True):
    total = rows[0]["h_aug"].new_zeros(())
    bce_sum = total.clone()
    reg_sum = total.clone()
    per_image: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        level = row["pyramid_level"]
        delta = row["h_aug"] @ A[level]
        logits = row["p"] @ (row["c"] + delta)
        bce = SEG_GAIN * F.binary_cross_entropy_with_logits(logits, row["y"], reduction="sum") / row["area"]
        reg = LAMBDA * delta.square().sum() / 2.0
        value = bce + reg
        total = total + value
        bce_sum = bce_sum + bce
        reg_sum = reg_sum + reg
        if need_parts:
            per_image[row["image_id"]].append(float(value.detach().cpu()))
    n = max(len(rows), 1)
    return {
        "objective": total / n,
        "bce": bce_sum / n,
        "regularizer": reg_sum / n,
        "per_image": {str(k): sum(v) / len(v) for k, v in per_image.items()},
    }


def batch_loss(A: list[torch.Tensor], rows: list[dict]) -> torch.Tensor:
    # Keep every candidate equally weighted, matching the official solver.
    return values_for_rows(A, rows, need_parts=False)["objective"]


def zeros_like_A(dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
    return [torch.zeros((65, 32), dtype=dtype, device=device, requires_grad=True) for _ in range(3)]


def rows_to_device(rows: list[dict], device: torch.device) -> list[dict]:
    """Move only tensor fields needed by the fixed-cache loss to the train device."""
    out = []
    for row in rows:
        out.append({**row, **{k: row[k].to(device) for k in ("h_aug", "p", "y", "c")}})
    return out


def solve_reference(rows: list[dict], max_iter: int, device: torch.device, chunk_rows: int = 16) -> tuple[list[torch.Tensor], list[float]]:
    A = zeros_like_A(torch.float64, device)
    trace: list[float] = []
    optimizer = torch.optim.LBFGS(A, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), dtype=torch.float64, device=device)
        gradients = [torch.zeros_like(a) for a in A]
        for lo in range(0, len(rows), chunk_rows):
            chunk = rows[lo:lo + chunk_rows]
            value = batch_loss(A, chunk)
            grad = torch.autograd.grad(value, A, retain_graph=False, allow_unused=True)
            weight = len(chunk) / max(len(rows), 1)
            total = total + value.detach() * weight
            for dst, src in zip(gradients, grad):
                if src is not None:
                    dst.add_(src, alpha=weight)
        for parameter, grad in zip(A, gradients):
            parameter.grad = grad
        trace.append(float(total.detach().cpu()))
        return total

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    return [a.detach().cpu() for a in A], trace


def make_optimizer(A: list[torch.Tensor], name: str, lr: float, momentum: float):
    params = [{"params": A}]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0.0, nesterov=momentum > 0)
    raise ValueError(f"unsupported optimizer: {name}")


def bootstrap_delta(per_a: dict[str, float], per_b: dict[str, float], seed: int, draws: int) -> dict:
    keys = sorted(set(per_a).intersection(per_b))
    if not keys:
        return {"n_images": 0, "delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    a = torch.tensor([per_a[k] for k in keys], dtype=torch.float64).numpy()
    b = torch.tensor([per_b[k] for k in keys], dtype=torch.float64).numpy()
    import numpy as np
    rng = np.random.default_rng(seed)
    delta = b - a
    index = rng.integers(0, len(keys), size=(draws, len(keys)))
    boot = delta[index].mean(1)
    return {"n_images": len(keys), "delta": float(delta.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def arm_summary(A: list[torch.Tensor], rows: list[dict], native: list[torch.Tensor], seed: int, draws: int) -> dict:
    device = rows[0]["h_aug"].device
    dtype = rows[0]["h_aug"].dtype
    A_eval = [a.to(device=device, dtype=dtype) for a in A]
    native_eval = [a.to(device=device, dtype=dtype) for a in native]
    result = values_for_rows(A_eval, rows)
    base = values_for_rows(native_eval, rows)
    return {
        "records": len(rows),
        "images": len(result["per_image"]),
        "objective": float(result["objective"].detach().cpu()),
        "bce": float(result["bce"].detach().cpu()),
        "regularizer": float(result["regularizer"].detach().cpu()),
        "delta_vs_native": float((result["objective"] - base["objective"]).detach().cpu()),
        "image_macro_delta_vs_native": bootstrap_delta(base["per_image"], result["per_image"], seed, draws),
        "per_image_objective": result["per_image"],
    }


def merge_check(A: list[torch.Tensor], rows: list[dict], stats: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Check standardized affine output equals its raw-h Conv2d form."""
    maxima = []
    for row in rows:
        level = row["pyramid_level"]
        mean, std = stats[level]
        aa = A[level].double()
        U, bias = aa[:-1], aa[-1]
        dw = U / std[:, None]
        db = bias - (mean / std) @ U
        d_direct = row["h_aug"].double().cpu() @ aa.cpu()
        d_merged = row["h"].double().cpu() @ dw.cpu() + db.cpu()
        maxima.append(float((d_direct - d_merged).abs().max()))
    return {"max_abs_coeff_error": max(maxima) if maxima else 0.0, "records": len(rows), "atol": 1e-4, "rtol": 1e-5}


def train_arm(name: str, train_rows: list[dict], eval_rows: dict[str, list[dict]], epochs: int, batch_images: int, lr: float, momentum: float, seed: int, out: Path, device: torch.device, draws: int):
    torch.manual_seed(seed)
    random.seed(seed)
    A = zeros_like_A(torch.float32, device)
    optimizer = make_optimizer(A, name, lr, momentum)
    train_rows_device = rows_to_device(train_rows, device)
    eval_rows_device = {group: rows_to_device(rows, device) for group, rows in eval_rows.items()}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for row in train_rows_device:
        by_image[row["image_id"]].append(row)
    image_ids = sorted(by_image)
    trace_path = out / f"TRACE_{name}.jsonl"
    history = []
    with trace_path.open("w", encoding="utf-8") as trace:
        for epoch in range(epochs):
            order = list(image_ids)
            random.Random(seed + epoch).shuffle(order)
            total = 0.0
            steps = 0
            for lo in range(0, len(order), batch_images):
                batch = [row for image_id in order[lo:lo + batch_images] for row in by_image[image_id]]
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(A, batch)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(A, float("inf")))
                optimizer.step()
                total += float(loss.detach().cpu())
                steps += 1
                trace.write(json.dumps({"epoch": epoch + 1, "step": steps, "objective": float(loss.detach().cpu()), "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"]}) + "\n")
            train_eval = values_for_rows(A, train_rows_device)
            rec = {"epoch": epoch + 1, "steps": steps, "trajectory_objective_mean": total / max(steps, 1), "fit_objective": float(train_eval["objective"].detach().cpu())}
            history.append(rec)
            print(json.dumps({"stage": "train", "optimizer": name, **rec}), flush=True)
    A_cpu = [a.detach().cpu() for a in A]
    torch.save({"optimizer": name, "A": A_cpu, "history": history}, out / f"TRAINED_{name}.pt")
    summary = {"optimizer": name, "history": history, "fit": arm_summary(A_cpu, train_rows, [torch.zeros_like(a) for a in A_cpu], seed, draws)}
    for group, rows in eval_rows.items():
        summary[group] = arm_summary(A_cpu, rows, [torch.zeros_like(a) for a in A_cpu], seed, draws)
    return A_cpu, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-images", type=int, default=64)
    ap.add_argument("--dev-images", type=int, default=32)
    ap.add_argument("--val-images", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-images", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--momentum", type=float, default=.9)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--reference-iter", type=int, default=100)
    ap.add_argument("--optimizers", nargs="+", choices=("adam", "sgd"), default=("adam", "sgd"))
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fit_items = choose_items(args.cache, "fit", args.fit_images, args.seed)
    dev_items = choose_items(args.cache, "dev", args.dev_images, args.seed)
    val_items = choose_items(args.cache, "val", args.val_images, args.seed)
    fit, fit_ids = load_rows(args.cache, fit_items)
    dev, dev_ids = load_rows(args.cache, dev_items)
    val, val_ids = load_rows(args.cache, val_items)
    stats = feature_stats(fit)
    fit_p = prepare_rows(fit, stats, torch.float32, device)
    dev_p = prepare_rows(dev, stats, torch.float32, device)
    val_p = prepare_rows(val, stats, torch.float32, device)
    reference_rows = prepare_rows(fit, stats, torch.float64, device)
    zero32 = [torch.zeros((65, 32), dtype=torch.float32, device=device) for _ in range(3)]
    zero64 = [torch.zeros((65, 32), dtype=torch.float64, device=device) for _ in range(3)]
    ref_rows_device = rows_to_device(reference_rows, device)
    ref, ref_trace = solve_reference(ref_rows_device, args.reference_iter, device)
    torch.save({"A": ref, "stats": stats, "trace": ref_trace}, args.out / "REFERENCE.pt")

    config = {
        "cache": str(args.cache), "cache_index_sha256": sha256(args.cache / "INDEX.json"), "device": str(device),
        "seed": args.seed, "fit_images": len(fit_ids), "dev_images": len(dev_ids), "val_images": len(val_ids),
        "fit_image_ids": fit_ids, "dev_image_ids": dev_ids, "val_image_ids": val_ids,
        "epochs": args.epochs, "batch_images": args.batch_images, "lr": args.lr, "momentum": args.momentum,
        "optimizers": list(args.optimizers), "reference_iter": args.reference_iter, "seg_gain": SEG_GAIN, "lambda": LAMBDA,
        "target": "official TAL GT-box support BCE with official segmentation gain plus explicit output displacement penalty",
        "freeze": ["h", "prototype", "boxes", "labels", "assignment", "backbone", "neck", "all branches except additive coefficient affine"],
        "optimizer_weight_decay": 0.0, "git_commit": git_commit(), "python": sys.version,
    }
    write_json(args.out / "CONFIG.json", config)
    write_json(args.out / "run.json", {"run_id": args.out.name, "status": "running", "created_at": time.time(), "command": sys.argv, "config": config})

    native_fit = arm_summary(zero32, fit_p, zero32, args.seed, args.bootstrap_draws)
    native_dev = arm_summary(zero32, dev_p, zero32, args.seed, args.bootstrap_draws)
    native_val = arm_summary(zero32, val_p, zero32, args.seed, args.bootstrap_draws)
    ref_fit = arm_summary(ref, reference_rows, zero64, args.seed, args.bootstrap_draws)
    eval_rows = {"dev": dev_p, "val": val_p}
    arms = {}
    for name in args.optimizers:
        _, arms[name] = train_arm(name, fit_p, eval_rows, args.epochs, args.batch_images, args.lr, args.momentum, args.seed, args.out, device, args.bootstrap_draws)

    j0 = native_fit["objective"]
    jr = ref_fit["objective"]
    best = None
    for name in args.optimizers:
        fit_value = arms[name]["fit"]["objective"]
        recovery = (j0 - fit_value) / max(j0 - jr, 1e-12)
        dev_ci = arms[name]["dev"]["image_macro_delta_vs_native"]["ci95"][1]
        val_ci = arms[name]["val"]["image_macro_delta_vs_native"]["ci95"][1]
        arms[name]["fit_reference_recovery"] = recovery
        arms[name]["merge_check"] = merge_check(torch.load(args.out / f"TRAINED_{name}.pt", map_location="cpu")["A"], fit_p, stats)
        arms[name]["decision"] = {"fit_recovery_ge_90pct": recovery >= .90, "dev_macro_upper_le_zero": dev_ci <= 0.0, "val_macro_upper_le_zero": val_ci <= 0.0, "finds_shared_rule": recovery >= .90 and dev_ci <= 0.0 and val_ci <= 0.0}
        if best is None or recovery > best["recovery"]:
            best = {"optimizer": name, "recovery": recovery}

    summary = {
        "status": "complete", "protocol": {"seg_gain": SEG_GAIN, "lambda": LAMBDA, "official_target": True, "candidate_mean": True},
        "native": {"fit": native_fit, "dev": native_dev, "val": native_val},
        "reference": {"fit": ref_fit, "trace_length": len(ref_trace), "merge_check": merge_check(ref, fit_p, stats)}, "arms": arms,
        "primary": {"best_optimizer_by_fit_recovery": best, "decision_rule": "fit recovery >=90% and dev/val macro bootstrap upper CI <=0"},
    }
    write_json(args.out / "SUMMARY.json", summary)
    write_json(args.out / "COMPLETE.json", {"status": "completed", "run_id": args.out.name, "summary": "SUMMARY.json"})
    write_json(args.out / "run.json", {"run_id": args.out.name, "status": "completed", "created_at": time.time(), "command": sys.argv, "config": config, "summary": summary})
    print(json.dumps({"stage": "complete", "out": str(args.out), "best": best}), flush=True)


if __name__ == "__main__":
    main()
