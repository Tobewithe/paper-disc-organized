"""Cross-evaluate frozen official TAL candidates under three coefficient maps.

This is an evaluation-only script.  It consumes the complete official cache,
the original COCO annotations, and two converged shared affine maps.  It never
reassigns candidates, changes prototypes/boxes, or trains a model.  The three
reported arms are A (native coefficients), B (old-160 map), and C (official
640 map).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops


GROUPS = ("fit", "dev", "val")
ARMS = ("A", "B", "C")
METRICS = ("iou", "coverage", "mask75")
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261011


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_json(v: Any) -> Any:
    """Convert torch/numpy scalar containers to ordinary JSON values."""
    if isinstance(v, dict):
        return {str(k): _as_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_as_json(x) for x in v]
    if isinstance(v, torch.Tensor):
        if v.ndim == 0:
            return _as_json(v.item())
        return _as_json(v.detach().cpu().tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def _numeric_values(v: Any) -> list[float]:
    if isinstance(v, torch.Tensor):
        return [float(x) for x in v.detach().cpu().reshape(-1).tolist()]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.reshape(-1).tolist()]
    if isinstance(v, dict):
        out: list[float] = []
        for x in v.values():
            out.extend(_numeric_values(x))
        return out
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_numeric_values(x))
        return out
    try:
        return [float(v)]
    except (TypeError, ValueError):
        return []


def _level_value(d: Any, level: int) -> Any:
    if isinstance(d, dict):
        if level in d:
            return d[level]
        if str(level) in d:
            return d[str(level)]
    if isinstance(d, (list, tuple)) and len(d) == 3:
        return d[level]
    raise KeyError(f"missing level {level} in map field")


def load_map(path: Path, name: str) -> dict[str, Any]:
    """Load and strictly validate a solved affine map before any evaluation."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"{name}: map must be a dictionary, got {type(obj)!r}")
    for key in ("A", "stats", "converged", "gradient_inf"):
        if key not in obj:
            raise KeyError(f"{name}: required map field {key!r} is absent")
    if not bool(obj["converged"]):
        raise AssertionError(f"{name}: map is not marked converged")
    grad_values = _numeric_values(obj["gradient_inf"])
    if not grad_values or any(not math.isfinite(x) for x in grad_values):
        raise AssertionError(f"{name}: gradient_inf is missing, nonnumeric, or nonfinite")
    max_grad = max(abs(x) for x in grad_values)
    if not max_grad < 1e-6:
        raise AssertionError(f"{name}: gradient_inf={max_grad:.8g} is not < 1e-6")
    A_raw = obj["A"]
    if not isinstance(A_raw, (list, tuple)) or len(A_raw) != 3:
        raise AssertionError(f"{name}: A must contain three level matrices")
    A: list[torch.Tensor] = []
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for level in range(3):
        a = torch.as_tensor(A_raw[level], dtype=torch.float64, device="cpu")
        if tuple(a.shape) != (65, 32):
            raise AssertionError(f"{name}: A[{level}] has shape {tuple(a.shape)}, expected (65, 32)")
        if not bool(torch.isfinite(a).all()):
            raise AssertionError(f"{name}: A[{level}] contains nonfinite values")
        A.append(a.contiguous())
        raw_stats = _level_value(obj["stats"], level)
        if not isinstance(raw_stats, (list, tuple)) or len(raw_stats) != 2:
            raise AssertionError(f"{name}: stats[{level}] must be (mean, std)")
        mu = torch.as_tensor(raw_stats[0], dtype=torch.float64, device="cpu").reshape(-1)
        sd = torch.as_tensor(raw_stats[1], dtype=torch.float64, device="cpu").reshape(-1)
        if mu.numel() != 64 or sd.numel() != 64:
            raise AssertionError(f"{name}: stats[{level}] must each have 64 entries")
        if not bool(torch.isfinite(mu).all() and torch.isfinite(sd).all() and (sd > 0).all()):
            raise AssertionError(f"{name}: invalid stats[{level}] mean/std")
        stats[level] = (mu.contiguous(), sd.contiguous())
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "A": A,
        "stats": stats,
        "gradient_inf_max": max_grad,
        "converged": True,
    }


def _resolve_annotation(annotations: Path, split: str) -> Path:
    """Resolve a COCO annotation file from a file, annotations dir, or data root."""
    if annotations.is_file():
        # If a caller points at one split's JSON while evaluating both train
        # and val, follow its sibling file when available rather than silently
        # scoring the wrong COCO images.
        expected = "instances_val2017.json" if split == "val" else "instances_train2017.json"
        if annotations.name != expected:
            sibling = annotations.with_name(expected)
            if sibling.is_file():
                return sibling
        return annotations
    filename = "instances_val2017.json" if split == "val" else "instances_train2017.json"
    candidates = [annotations / filename, annotations / "annotations" / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot find {filename} below --annotations {annotations}")


def _load_index(cache: Path) -> dict[str, list[dict[str, Any]]]:
    index_path = cache / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for group in GROUPS:
        values = index.get(group, [])
        if not isinstance(values, list):
            raise TypeError(f"cache INDEX.json group {group!r} is not a list")
        out[group] = values
    return out


def _map_delta(map_data: dict[str, Any], level: int, h: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Evaluation may process thousands of rows; keep the tiny map tensors on
    # the decode device instead of copying them for every candidate.
    if map_data.get("_device") == str(device):
        mu, sd = map_data["stats_device"][level]
        A = map_data["A_device"][level]
    else:
        mu, sd = map_data["stats"][level]
        A = map_data["A"][level]
    hh = h.to(device=device, dtype=torch.float64)
    mu = mu.to(device=device)
    sd = sd.to(device=device)
    A = A.to(device=device)
    if hh.numel() != 64:
        raise AssertionError(f"cache h has {hh.numel()} features, expected 64")
    z = torch.cat(((hh.reshape(-1) - mu) / sd, torch.ones(1, device=device, dtype=torch.float64)))
    return z @ A


def _decode(proto: torch.Tensor, coeff: torch.Tensor, boxes: torch.Tensor, original_shape: Any, ratio_pad: Any) -> torch.Tensor:
    """Official TAL decode: process_mask at 640, then scale_masks with bilinear."""
    padded = ops.process_mask(proto, coeff.reshape(1, -1), boxes.reshape(1, 4), (640, 640), upsample=True)
    # scale_masks uses bilinear interpolation internally in the pinned Ultralytics version.
    return ops.scale_masks(padded[None], original_shape, ratio_pad=ratio_pad)[0, 0] > 0.5


def _original_gt(coco: COCO, annotation_id: int, device: torch.device) -> torch.Tensor:
    ann = coco.anns[int(annotation_id)]
    return torch.from_numpy(coco.annToMask(ann).astype(bool, copy=False)).to(device=device)


def _candidate_iou(mask: torch.Tensor, gt: torch.Tensor) -> tuple[float, float, int]:
    inter = int((mask & gt).sum().item())
    union = int((mask | gt).sum().item())
    gt_area = int(gt.sum().item())
    return inter / max(union, 1), inter / max(gt_area, 1), int(inter / max(union, 1) >= 0.75)


def _iter_group_images(index: dict[str, list[dict[str, Any]]], groups: Iterable[str]):
    for split in groups:
        for item in index[split]:
            yield split, item


def evaluate(args: argparse.Namespace, run_meta: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = args.cache
    index = _load_index(cache)
    map_b = load_map(args.old_map, "old160")
    map_c = load_map(args.official_map, "official640")
    groups = tuple(args.groups) if args.groups else GROUPS
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown --groups {unknown}; choose from {GROUPS}")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    for map_data in (map_b, map_c):
        map_data["_device"] = str(device)
        map_data["A_device"] = [a.to(device=device) for a in map_data["A"]]
        map_data["stats_device"] = [
            (map_data["stats"][level][0].to(device=device), map_data["stats"][level][1].to(device=device))
            for level in range(3)
        ]
    torch.set_grad_enabled(False)
    coco_by_split = {
        split: COCO(str(_resolve_annotation(args.annotations, split)))
        for split in groups
    }
    # Keep only the current image resident.  The full cache is about 11.7 GB;
    # retaining every loaded tensor would exhaust notebook RAM.  The index is
    # image keyed, so reloading a repeated key is cheaper than unbounded cache
    # growth and does not alter any identity row.
    rows: list[dict[str, Any]] = []
    current_key: tuple[str, int] | None = None
    current_image: dict[str, Any] | None = None
    started = time.monotonic()
    for n_image, (split, item) in enumerate(_iter_group_images(index, groups), 1):
        image_id = int(item["image_id"])
        key_image = (split, image_id)
        if key_image != current_key:
            current_image = torch.load(cache / "images" / f"{image_id:012d}.pt", map_location="cpu", weights_only=False)
            current_key = key_image
        x = current_image
        if x is None:
            raise RuntimeError("current image failed to load")
        proto = x["proto"].to(device=device, dtype=torch.float32)
        coeff0 = x["coeff"].to(device=device, dtype=torch.float32)
        boxes = x["boxes"].to(device=device, dtype=torch.float32)
        h_all = x["h"]
        original_shape = x["original_shape"]
        ratio_pad = x["ratio_pad"]
        coco = coco_by_split[split]
        for row_idx, r in enumerate(x["rows"]):
            raw_id = int(r["raw_id"])
            level = int(r.get("level", r.get("pyramid_level")))
            c0 = coeff0[raw_id]
            d_b = _map_delta(map_b, level, h_all[raw_id], device)
            d_c = _map_delta(map_c, level, h_all[raw_id], device)
            c_b = (c0.double() + d_b).float()
            c_c = (c0.double() + d_c).float()
            # Keep FP64 through c0+delta; only the official decoder receives FP32.
            masks = {
                "A": _decode(proto, c0, boxes[raw_id], original_shape, ratio_pad),
                "B": _decode(proto, c_b, boxes[raw_id], original_shape, ratio_pad),
                "C": _decode(proto, c_c, boxes[raw_id], original_shape, ratio_pad),
            }
            annotation_id = int(r["annotation_id"])
            gt = _original_gt(coco, annotation_id, device)
            rr: dict[str, Any] = {
                "split": split,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "branch": str(r.get("branch", "one2one")),
                "raw_id": raw_id,
                "pyramid_level": level,
                "target_gt_idx": int(r.get("gt_index", r.get("target_gt_idx", -1))),
                "box_iou": float(r.get("box_iou", float("nan"))),
                "candidate_index_in_image": row_idx,
            }
            for arm in ARMS:
                iou, coverage, mask75 = _candidate_iou(masks[arm], gt)
                rr[f"iou_{arm}"] = float(iou)
                rr[f"coverage_{arm}"] = float(coverage)
                rr[f"mask75_{arm}"] = int(mask75)
            rr["delta_l2_B"] = float((d_b.detach().double().square().sum().sqrt()).cpu())
            rr["delta_l2_C"] = float((d_c.detach().double().square().sum().sqrt()).cpu())
            rr["delta_max_B"] = float(d_b.detach().abs().max().cpu())
            rr["delta_max_C"] = float(d_c.detach().abs().max().cpu())
            rows.append(rr)
        if n_image % 50 == 0 or n_image == sum(len(index[g]) for g in groups):
            print(json.dumps({"groups": groups, "images": n_image, "elapsed_s": time.monotonic() - started}), flush=True)
    run_meta.update({"n_candidates": len(rows), "n_images": len({(r['split'], r['image_id']) for r in rows}), "device": str(device), "elapsed_s": time.monotonic() - started})
    return rows, {"groups": groups, "map_b": map_b, "map_c": map_c}


def _finite_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _bootstrap_summary(rr: list[dict[str, Any]], metric: str, rng: np.random.Generator) -> dict[str, Any]:
    ids = sorted({(r["split"], int(r["image_id"])) for r in rr})
    by_image = [[r for r in rr if (r["split"], int(r["image_id"])) == iid] for iid in ids]
    image_vals = {
        arm: np.asarray([_finite_mean(r[f"{metric}_{arm}"] for r in group) for group in by_image], dtype=np.float64)
        for arm in ARMS
    }
    candidate = {
        arm: float(np.nanmean([r[f"{metric}_{arm}"] for r in rr]))
        for arm in ARMS
    }
    macro = {arm: float(np.nanmean(image_vals[arm])) for arm in ARMS}
    draw = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids))) if ids else np.empty((0, 0), dtype=np.int64)
    out: dict[str, Any] = {
        "n_candidates": len(rr),
        "n_images": len(ids),
        "candidate": candidate,
        "image_macro": macro,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "cluster": "image"},
    }
    # Candidate means are computed as a ratio of sums/counts under image-cluster resampling.
    sums = {arm: np.asarray([np.nansum([r[f"{metric}_{arm}"] for r in group]) for group in by_image]) for arm in ARMS}
    counts = {arm: np.asarray([np.isfinite([r[f"{metric}_{arm}"] for r in group]).sum() for group in by_image]) for arm in ARMS}
    for left, right, label in (("B", "A", "B_minus_A"), ("C", "A", "C_minus_A"), ("C", "B", "C_minus_B")):
        point = candidate[left] - candidate[right]
        if ids:
            left_draw = sums[left][draw].sum(axis=1) / np.maximum(counts[left][draw].sum(axis=1), 1)
            right_draw = sums[right][draw].sum(axis=1) / np.maximum(counts[right][draw].sum(axis=1), 1)
            candidate_boot = left_draw - right_draw
            image_boot = np.nanmean((image_vals[left] - image_vals[right])[draw], axis=1)
            cci = np.quantile(candidate_boot, [0.025, 0.975]).tolist()
            ici = np.quantile(image_boot, [0.025, 0.975]).tolist()
        else:
            cci = [float("nan"), float("nan")]
            ici = [float("nan"), float("nan")]
        out["candidate"][label] = {"delta": float(point), "ci95": cci, "scope": "image_cluster_candidate_mean"}
        out["image_macro"][label] = {"delta": float(macro[left] - macro[right]), "ci95": ici, "scope": "image_cluster_image_macro"}
    return out


def aggregate(rows: list[dict[str, Any]], groups: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tables: dict[str, Any] = {}
    image_rows: list[dict[str, Any]] = []
    for split in groups:
        rr = [r for r in rows if r["split"] == split]
        if not rr:
            continue
        tables[split] = {metric: _bootstrap_summary(rr, metric, rng) for metric in METRICS}
        by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for r in rr:
            by_key.setdefault((split, int(r["image_id"])), []).append(r)
        for (sp, iid), group in sorted(by_key.items()):
            one: dict[str, Any] = {"split": sp, "image_id": iid, "n_candidates": len(group)}
            for metric in METRICS:
                for arm in ARMS:
                    one[f"{metric}_{arm}"] = _finite_mean(r[f"{metric}_{arm}"] for r in group)
            image_rows.append(one)
    summary = {
        "tables": tables,
        "arms": {"A": "native coefficient", "B": "old160 shared affine map", "C": "official640 shared affine map"},
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_scope": "image-cluster; candidate mean and image macro are both reported",
        "identity_policy": "fixed official cache rows; no candidate deletion or re-assignment",
        "decode": "ultralytics process_mask at 640 followed by scale_masks bilinear; original COCO mask",
    }
    return summary, image_rows


def _identity_key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r.get("split"), int(r.get("image_id")), int(r.get("annotation_id")), int(r.get("raw_id")), int(r.get("pyramid_level", r.get("level", -1))))


def baseline_check(rows: list[dict[str, Any]], path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    existing: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                existing[_identity_key(r)] = r
    checked = missing = 0
    max_abs: dict[str, float] = {}
    if rows:
        max_abs = {f"iou_{arm}": 0.0 for arm in ARMS if f"iou_{arm}" in rows[0]}
    mismatches: list[dict[str, Any]] = []
    for r in rows:
        old = existing.get(_identity_key(r))
        if old is None:
            missing += 1
            continue
        checked += 1
        for key in tuple(max_abs):
            if key not in old or not math.isfinite(float(old[key])) or not math.isfinite(float(r[key])):
                continue
            err = abs(float(old[key]) - float(r[key])); max_abs[key] = max(max_abs[key], err)
            if err > 1e-6 and len(mismatches) < 20:
                mismatches.append({"identity": list(_identity_key(r)), "field": key, "new": r[key], "existing": old[key], "abs_error": err})
    return {"path": str(path.resolve()), "sha256": _sha256(path), "existing_rows": len(existing), "checked": checked, "missing": missing, "max_abs_error": max_abs, "mismatches_first20": mismatches}


def verify_weights(weights: Path, rows: list[dict[str, Any]], map_b: dict[str, Any], map_c: dict[str, Any], cache: Path) -> dict[str, Any]:
    """Optional algebraic checkpoint merge verification; no forward/training is run."""
    from ultralytics import YOLO
    model = YOLO(str(weights)).model.cpu().float().eval()
    head = model.model[-1]
    if not hasattr(head, "one2one_cv4"):
        raise AssertionError("weights do not expose official one2one coefficient branches")
    native_params: list[tuple[torch.Tensor, torch.Tensor]] = []
    for level, branch in enumerate(head.one2one_cv4):
        conv = branch[-1]
        w0 = conv.weight.detach().cpu().float().reshape(conv.out_channels, conv.in_channels)
        b0 = conv.bias.detach().cpu().float()
        native_params.append((w0, b0))
    native_max = map_b_max = map_c_max = 0.0
    # Group identities by image so optional verification loads one image at a
    # time instead of retaining the complete cache in notebook RAM.
    rows_by_image: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_image.setdefault(int(row["image_id"]), []).append(row)
    for iid, image_rows in rows_by_image.items():
        x = torch.load(cache / "images" / f"{iid:012d}.pt", map_location="cpu", weights_only=False)
        for r in image_rows:
            lev = int(r["pyramid_level"]); h = x["h"][int(r["raw_id"])].float(); c0 = x["coeff"][int(r["raw_id"])].float()
            w0, b0 = native_params[lev]
            native = w0 @ h + b0
            native_max = max(native_max, float((native - c0).abs().max()))
            for map_data, slot in ((map_b, "b"), (map_c, "c")):
                mu, sd = map_data["stats"][lev]; z = torch.cat(((h.double() - mu) / sd, torch.ones(1, dtype=torch.float64)))
                direct = c0.double() + z @ map_data["A"][lev]
                U, v = map_data["A"][lev][:-1], map_data["A"][lev][-1]
                wm = w0.double() + (U / sd[:, None]).T; bm = b0.double() + v - (mu / sd) @ U
                err = float((wm @ h.double() + bm - direct).abs().max())
                if slot == "b": map_b_max = max(map_b_max, err)
                else: map_c_max = max(map_c_max, err)
    return {"weights": str(weights.resolve()), "sha256": _sha256(weights), "native_head_max_error": native_max, "merged_old160_max_error": map_b_max, "merged_official640_max_error": map_c_max, "passed_atol": max(native_max, map_b_max, map_c_max) <= 1e-4}


def write_manifest(out: Path) -> None:
    entries = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            entries.append(f"{_sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "manifest.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True, help="COCO annotation file, annotations directory, or data root")
    ap.add_argument("--old-map", type=Path, required=True)
    ap.add_argument("--official-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    ap.add_argument("--official-per-candidate", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=None, help="optional checkpoint for algebraic last-conv merge verification")
    ap.add_argument("--device", default=None, help="torch device; default cuda when available, otherwise cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = args.out.name
    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cache": str(args.cache.resolve()),
        "cache_index_sha256": _sha256(args.cache / "INDEX.json"),
        "annotations": str(args.annotations.resolve()),
        "old_map": str(args.old_map.resolve()),
        "old_map_sha256": _sha256(args.old_map),
        "official_map": str(args.official_map.resolve()),
        "official_map_sha256": _sha256(args.official_map),
        "groups": list(args.groups),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    try:
        import ultralytics
        run_meta["ultralytics_version"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:
        run_meta["ultralytics_version"] = f"unavailable:{type(exc).__name__}"
    _json_dump(args.out / "run.json", run_meta)
    try:
        rows, context = evaluate(args, run_meta)
        run_meta["map_validation"] = {
            "old160": {"converged": True, "gradient_inf_max": context["map_b"]["gradient_inf_max"]},
            "official640": {"converged": True, "gradient_inf_max": context["map_c"]["gradient_inf_max"]},
        }
        with (args.out / "PER_CANDIDATE.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(_as_json(r), allow_nan=True, sort_keys=True) + "\n")
        summary, image_rows = aggregate(rows, context["groups"])
        _json_dump(args.out / "SUMMARY.json", summary)
        with (args.out / "PER_IMAGE.jsonl").open("w", encoding="utf-8") as f:
            for r in image_rows:
                f.write(json.dumps(_as_json(r), allow_nan=True, sort_keys=True) + "\n")
        checks = baseline_check(rows, args.official_per_candidate)
        if checks is not None:
            _json_dump(args.out / "BASELINE_CHECK.json", checks)
        if args.weights is not None:
            _json_dump(args.out / "WEIGHTS_MERGE_CHECK.json", verify_weights(args.weights, rows, context["map_b"], context["map_c"], args.cache))
        run_meta.update({"status": "complete", "completed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "outputs": ["PER_CANDIDATE.jsonl", "PER_IMAGE.jsonl", "SUMMARY.json"]})
        _json_dump(args.out / "run.json", run_meta)
        write_manifest(args.out)
        print(json.dumps({"completed": True, "n_candidates": len(rows), "out": str(args.out.resolve())}), flush=True)
    except Exception as exc:
        run_meta.update({"status": "failed", "failed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "error": repr(exc)})
        _json_dump(args.out / "run.json", run_meta)
        raise


if __name__ == "__main__":
    main()
