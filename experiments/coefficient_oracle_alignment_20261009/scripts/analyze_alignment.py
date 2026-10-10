"""Stage 2: oracle solve, alignment / realization / utility readouts, gated decision.

Consumes the stage-1 export Run directory (images/<iid>.pt shared objects and
arms/<arm>/<iid>.pt per-arm coefficient tables). Per protocol:
  - finite regularized oracle in the GT-box ROI, LBFGS 120 iterations, double precision
    (experiments/coefficient_finite_oracle_20260925/scripts/finite_oracle.py convention);
  - primary lambda 0.003, sensitivity 0.0003 / 0.03 / 0.3;
  - alignment cos(delta_c_arm, delta_c*) overall / per level / per COCO size stratum;
  - realization (Mask75 via each arm's own coefficients) on native-failure candidates;
  - utility guardrails: image-macro IoU, coverage, box-FPR, repair/damage vs arm A;
  - 5000 image-cluster bootstrap resamples, seed 20261009, 95% percentile intervals;
  - pre-registered gates -> DECISION.json. No parameter or gate changes after results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils import ops

LAMBDAS = (0.003, 0.0003, 0.03, 0.3)
PRIMARY = 0.003
ARMS = ("native", "baseline", "acd")
BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20261009


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def solve(c0, roi, penalty, iterations=120):
    """Strongly convex GT-box BCE oracle with L2 penalty (7D convention, double precision)."""
    p, y, area = roi
    p, y, c0, area = p.double().cuda(), y.double().cuda(), c0.double().cuda(), float(area)

    def objective(delta):
        logits = p @ (c0 + delta)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / area
        return bce + penalty * delta.square().sum() / 2

    delta = torch.zeros_like(c0, requires_grad=True)
    optimizer = torch.optim.LBFGS([delta], lr=1, max_iter=iterations,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                  tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = objective(delta)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = objective(delta)
    gradient = torch.autograd.grad(value, delta)[0]
    return delta.detach().float().cpu(), dict(objective=float(value.detach()),
                                              stationary_norm=float(gradient.norm()),
                                              iterations=int(optimizer.state[delta]["n_iter"]))


def cosine(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na == 0 or nb == 0:
        return None
    return float((a @ b) / (na * nb))


def predicted_box_iou(proto, c, box, truth):
    z = (c.cuda() @ proto.cuda().flatten(1)).reshape(160, 160)
    cropped = ops.crop_mask(z[None].clone(), box.cuda()[None] / 4)[0] > 0
    truth = truth.cuda()
    inter = int((cropped & truth).sum())
    union = int((cropped | truth).sum())
    return inter / max(union, 1)


def size_stratum(area):
    if area < 1024:
        return "small"
    if area < 9216:
        return "medium"
    return "large"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(rows_by_image, statistic, seed=BOOTSTRAP_SEED, n=BOOTSTRAP):
    """Percentile CI of a statistic over image-cluster resampling of candidate rows."""
    generator = torch.Generator().manual_seed(seed)
    images = list(rows_by_image)
    values = []
    for _ in range(n):
        picks = torch.randint(0, len(images), (len(images),), generator=generator).tolist()
        sampled = [row for i in picks for row in rows_by_image[images[i]]]
        value = statistic(sampled)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return None, None, 0
    values = sorted(values)
    lo = values[max(0, int(0.025 * len(values)) - 1)]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    return lo, hi, len(values)


def main(a):
    torch.manual_seed(0)
    a.out.mkdir(parents=True, exist_ok=True)
    image_files = sorted((a.export / "images").glob("*.pt"))
    assert image_files, "no export images found"

    table = []
    for path in image_files:
        image_id = int(path.stem)
        shared = torch.load(path, weights_only=False, map_location="cpu")
        arms = {}
        for arm in ARMS:
            arm_path = a.export / "arms" / arm / path.name
            arms[arm] = torch.load(arm_path, weights_only=False, map_location="cpu")
        coeff_of = {}
        for arm in ARMS:
            ids = arms[arm]["ids"]
            coeff_of[arm] = {rid: arms[arm]["coeff"][i] for i, rid in enumerate(ids)}
        metrics_of = {}
        for arm in ARMS:
            metrics_of[arm] = {(r["annotation_id"], r["raw_id"]): r for r in arms[arm]["rows"]}
        for row in shared["rows"]:
            key = (row["annotation_id"], row["raw_id"])
            c0 = coeff_of["native"][row["raw_id"]]
            entry = dict(image_id=image_id, annotation_id=row["annotation_id"],
                         raw_id=row["raw_id"], level=row["level"], area=row["area"],
                         fill=row["fill"], box_iou=row["box_iou"],
                         gt_class_score=row["gt_class_score"],
                         size=size_stratum(row["area"]),
                         roi=(row["p"], row["y"], row["area_px"]))
            for arm in ARMS:
                metrics = metrics_of[arm][key]
                entry[f"c_{arm}"] = coeff_of[arm][row["raw_id"]]
                entry[f"iou_{arm}"] = metrics["iou"]
                entry[f"coverage_{arm}"] = metrics["coverage"]
                entry[f"fpr_{arm}"] = metrics["fpr"]
                entry[f"mask75_{arm}"] = metrics["mask75"]
            # parity: recompute native predicted-box IoU from stored tensors
            recomputed = predicted_box_iou(shared["proto"], c0, row["pred_box"], row["truth160"])
            entry["iou_A_recomputed"] = recomputed
            entry["oracle"] = {}
            table.append(entry)

    parity = [abs(e["iou_native"] - e["iou_A_recomputed"]) for e in table]
    parity_max = max(parity)

    # oracle solves
    for entry in table:
        p, y, area_px = entry.pop("roi")
        c0 = entry["c_native"]
        solutions = {}
        for penalty in LAMBDAS:
            delta, state = solve(c0, (p, y, area_px), penalty)
            solutions[penalty] = (delta, state)
        entry["oracle"] = {}
        for penalty in LAMBDAS:
            delta, state = solutions[penalty]
            entry["oracle"][str(penalty)] = dict(
                norm=float(delta.norm()), objective=state["objective"],
                stationary_norm=state["stationary_norm"], iterations=state["iterations"])
        primary = solutions[PRIMARY][0]
        entry["delta_star"] = primary
        for arm in ("baseline", "acd"):
            entry[f"align_{arm}"] = cosine(entry[f"c_{arm}"] - c0, primary)
        entry["align_native"] = 0.0
        entry["delta_norm_baseline"] = float((entry["c_baseline"] - c0).norm())
        entry["delta_norm_acd"] = float((entry["c_acd"] - c0).norm())

    # oracle predicted-box realization per entry
    image_cache = {}
    for entry in table:
        if entry["image_id"] not in image_cache:
            image_cache[entry["image_id"]] = torch.load(
                a.export / "images" / f"{entry['image_id']:012d}.pt", weights_only=False, map_location="cpu")
    for entry in table:
        shared = image_cache[entry["image_id"]]
        proto = shared["proto"]
        row_truth = next(r for r in shared["rows"] if r["raw_id"] == entry["raw_id"])
        box, truth = row_truth["pred_box"], row_truth["truth160"]
        c_star = entry["c_native"] + entry["delta_star"]
        entry["iou_oracle"] = predicted_box_iou(proto, c_star, box, truth)
        entry["mask75_oracle"] = int(entry["iou_oracle"] >= .75)

    valid = [e for e in table if e["align_baseline"] is not None and e["align_acd"] is not None]
    native_fail = [e for e in valid if e["iou_native"] < .75]
    native_success = [e for e in valid if e["iou_native"] >= .75]

    def strat(rows, keyf):
        groups = {}
        for row in rows:
            groups.setdefault(keyf(row), []).append(row)
        return groups

    def block(rows):
        if not rows:
            return None
        return dict(
            n=len(rows),
            align_native=mean([r["align_native"] for r in rows]),
            align_baseline=mean([r["align_baseline"] for r in rows]),
            align_acd=mean([r["align_acd"] for r in rows]),
            delta_norm_baseline=mean([r["delta_norm_baseline"] for r in rows]),
            delta_norm_acd=mean([r["delta_norm_acd"] for r in rows]),
            oracle_norm=mean([float(r["delta_star"].norm()) for r in rows]),
            mask75_native=mean([r["mask75_native"] for r in rows]),
            mask75_baseline=mean([r["mask75_baseline"] for r in rows]),
            mask75_acd=mean([r["mask75_acd"] for r in rows]),
            mask75_oracle=mean([r["mask75_oracle"] for r in rows]),
            iou_native=mean([r["iou_native"] for r in rows]),
            iou_baseline=mean([r["iou_baseline"] for r in rows]),
            iou_acd=mean([r["iou_acd"] for r in rows]),
        )

    rows_by_image = {}
    for row in valid:
        rows_by_image.setdefault(row["image_id"], []).append(row)

    def stat_align_cb(sampled):
        return mean([r["align_acd"] - r["align_baseline"] for r in sampled])

    def stat_align_c(sampled):
        return mean([r["align_acd"] for r in sampled])

    def stat_realize_ca(sampled):
        sub = [r for r in sampled if r["iou_native"] < .75]
        return mean([r["mask75_acd"] - r["mask75_native"] for r in sub])

    def stat_realize_cb(sampled):
        sub = [r for r in sampled if r["iou_native"] < .75]
        return mean([r["mask75_acd"] - r["mask75_baseline"] for r in sub])

    def stat_realize_ba(sampled):
        sub = [r for r in sampled if r["iou_native"] < .75]
        return mean([r["mask75_baseline"] - r["mask75_native"] for r in sub])

    def _by_image(sampled):
        per_image = {}
        for row in sampled:
            per_image.setdefault(row["image_id"], []).append(row)
        return per_image

    def stat_image_iou_ca(sampled):
        diffs = [mean([r["iou_acd"] - r["iou_native"] for r in rs]) for rs in _by_image(sampled).values()]
        return mean(diffs)

    def stat_damage_c(sampled):
        sub = [r for r in sampled if r["iou_native"] >= .75]
        return mean([1 - r["mask75_acd"] for r in sub])

    def stat_damage_b(sampled):
        sub = [r for r in sampled if r["iou_native"] >= .75]
        return mean([1 - r["mask75_baseline"] for r in sub])

    def stat_damage_diff(sampled):
        sub = [r for r in sampled if r["iou_native"] >= .75]
        return mean([(1 - r["mask75_acd"]) - (1 - r["mask75_native"]) for r in sub])

    estimates = {}
    for name, fn in (("align_acd_minus_baseline", stat_align_cb), ("align_acd", stat_align_c),
                     ("realize_acd_minus_native", stat_realize_ca),
                     ("realize_acd_minus_baseline", stat_realize_cb),
                     ("realize_baseline_minus_native", stat_realize_ba),
                     ("image_macro_iou_acd_minus_native", stat_image_iou_ca),
                     ("image_macro_iou_baseline_minus_native", lambda s: mean(
                         [mean([r["iou_baseline"] - r["iou_native"] for r in rs])
                          for rs in _by_image(s).values()])),
                     ("damage_rate_acd", stat_damage_c),
                     ("damage_rate_baseline", stat_damage_b),
                     ("damage_rate_acd_minus_native", stat_damage_diff)):
        point = fn([r for rs in rows_by_image.values() for r in rs])
        lo, hi, n = bootstrap_ci(rows_by_image, fn)
        estimates[name] = dict(point=point, ci95=[lo, hi], bootstrap_samples=n)

    gates = {}
    align_cb = estimates["align_acd_minus_baseline"]
    realize_ca = estimates["realize_acd_minus_native"]
    gates["G_transfer"] = dict(
        pass_=bool(align_cb["point"] is not None and align_cb["point"] >= 0.02
                   and realize_ca["point"] is not None and realize_ca["point"] >= 0.005
                   and realize_ca["ci95"][0] is not None and realize_ca["ci95"][0] > 0),
        alignment_delta_point=align_cb["point"], alignment_delta_ci95=align_cb["ci95"],
        realization_delta_point=realize_ca["point"], realization_delta_ci95=realize_ca["ci95"],
        rule="align(C-B) >= +0.02 cosine AND realization(C-A) >= +0.005 with bootstrap lower bound > 0")
    align_b = block(valid)["align_baseline"]
    align_c = block(valid)["align_acd"]
    gates["G_not_learnable"] = dict(
        pass_=bool(align_c - max(0.0, align_b) <= 0.01),
        align_acd=align_c, align_baseline=align_b,
        rule="align(C) within +0.01 of max(native=0, B)")
    damage = estimates["damage_rate_acd"]
    damage_b = estimates["damage_rate_baseline"]
    image_iou = estimates["image_macro_iou_acd_minus_native"]
    gates["G_damage"] = dict(
        triggered=bool(damage["point"] is not None and damage["point"] > 0.01
                       or damage_b["point"] is not None and damage_b["point"] > 0.01
                       or (image_iou["ci95"][0] is not None and image_iou["ci95"][0] < -0.001)),
        damage_rate_acd_point=damage["point"], damage_rate_baseline_point=damage_b["point"],
        image_macro_iou_delta=image_iou,
        rule="damage rate of any arm > 1% or image-macro IoU CI lower < -0.1pp")

    summary = dict(
        candidates=len(table), valid=len(valid), native_failure=len(native_fail),
        native_success=len(native_success), native_iou_parity_max_abs=parity_max,
        overall=block(valid),
        by_level={str(l): block(rows) for l, rows in strat(valid, lambda r: r["level"]).items()},
        by_size={s: block(rows) for s, rows in strat(valid, lambda r: r["size"]).items()},
        native_failure_block=block(native_fail),
        estimates=estimates, gates=gates,
        oracle_objective_mean=mean([e["oracle"]["0.003"]["objective"] for e in valid]),
    )
    write_json(a.out / "SUMMARY.json", summary)
    write_json(a.out / "DECISION.json", dict(study="STUDY_COEFFICIENT_ORACLE_ALIGNMENT_20261009",
                                             gates=gates, estimates=estimates,
                                             decision=("transfer" if gates["G_transfer"]["pass_"]
                                                       else "not_learnable_at_this_budget"
                                                       if gates["G_not_learnable"]["pass_"] else "undetermined")))
    with (a.out / "PER_CANDIDATE.jsonl").open("w", encoding="utf-8") as sink:
        for entry in table:
            record = {k: v for k, v in entry.items() if not k.startswith("c_") and k != "delta_star"}
            record["delta_star_norm"] = float(entry["delta_star"].norm())
            record["align_baseline"] = entry["align_baseline"]
            record["align_acd"] = entry["align_acd"]
            sink.write(json.dumps(record, allow_nan=False) + "\n")
    write_json(a.out / "COMPLETE.json", {"status": "complete", "candidates": len(table)})
    print("ANALYSIS_COMPLETE " + json.dumps(dict(candidates=len(table), valid=len(valid),
                                                 native_failure=len(native_fail),
                                                 decision="transfer" if gates["G_transfer"]["pass_"] else
                                                 ("not_learnable_at_this_budget" if gates["G_not_learnable"]["pass_"] else "undetermined"))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
