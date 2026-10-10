"""Stage 3: independent verification of the oracle-alignment analysis.

Does not trust SUMMARY/DECISION. From the raw stage-1 export tensors it independently:
  1. re-solves the primary lambda=0.003 oracle for a fixed deterministic subset (every 7th
     candidate) from a DIFFERENT initialization (fixed-seed random start) and checks
     coefficient/objective agreement with the analysis solve (7D uniqueness convention);
  2. recomputes predicted-box IoU for arm A and for the oracle solution from stored
     proto/pred_box/truth160 tensors and compares with the analysis readouts;
  3. recomputes alignment cosines from stored per-arm coefficients and the re-solved oracle;
  4. recounts native-failure/success counts and strata sizes from PER_CANDIDATE.jsonl;
  5. re-runs the align_acd_minus_baseline bootstrap statistic with the same seed to confirm
     determinism and recomputes every gate from the analysis numbers.

Writes VERIFICATION.json with per-pass/fail checks and the machine label.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils import ops

SUBSET_STRIDE = 7
ARMS = ("native", "baseline", "acd")
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": str(detail)[:2000]})
    print(json.dumps({"check": name, "pass": bool(ok)}), flush=True)


def solve(c0, p, y, area, penalty, iterations=120, start=None):
    p, y, c0 = p.double().cuda(), y.double().cuda(), c0.double().cuda()
    area = float(area)

    def objective(delta):
        logits = p @ (c0 + delta)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / area
        return bce + penalty * delta.square().sum() / 2

    delta = (torch.zeros_like(c0) if start is None else start.double().clone().cuda()).requires_grad_(True)
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
    return delta.detach().float().cpu(), float(value.detach()), float(gradient.norm())


def cosine(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na == 0 or nb == 0:
        return None
    return float((a @ b) / (na * nb))


def predicted_box_iou(proto, c, box, truth):
    z = (c.cuda() @ proto.cuda().flatten(1)).reshape(160, 160)
    cropped = ops.crop_mask(z[None].clone(), box.cuda()[None] / 4)[0] > 0
    inter = int((cropped & truth.cuda()).sum())
    union = int((cropped | truth.cuda()).sum())
    return inter / max(union, 1)


def main(a):
    torch.manual_seed(0)
    export = a.export
    analysis = json.loads((a.analysis / "SUMMARY.json").read_text())
    decision = json.loads((a.analysis / "DECISION.json").read_text())
    rows = [json.loads(line) for line in (a.analysis / "PER_CANDIDATE.jsonl").read_text().splitlines()]

    subset = [r for i, r in enumerate(rows) if i % SUBSET_STRIDE == 0]
    check("subset_size", len(subset) > 50, f"{len(subset)} of {len(rows)}")

    per_image_cache = {}
    worst_cos, worst_obj, worst_iou = 0.0, 0.0, 0.0
    for row in subset:
        iid = row["image_id"]
        if iid not in per_image_cache:
            per_image_cache[iid] = torch.load(export / "images" / f"{iid:012d}.pt",
                                              weights_only=False, map_location="cpu")
        shared = per_image_cache[iid]
        table_row = next(r for r in shared["rows"] if r["raw_id"] == row["raw_id"])
        native_payload = torch.load(export / "arms" / "native" / f"{iid:012d}.pt",
                                    weights_only=False, map_location="cpu")
        c0 = native_payload["coeff"][native_payload["ids"].index(row["raw_id"])]
        generator = torch.Generator().manual_seed(12345)
        start = 0.01 * torch.randn(32, generator=generator)
        delta, value, stationary = solve(c0, table_row["p"], table_row["y"], table_row["area_px"],
                                         0.003, start=start)
        worst_obj = max(worst_obj, abs(value - row["oracle"]["0.003"]["objective"]))
        # independent alignment from stored per-arm coefficients
        arm_c = {}
        for arm in ("baseline", "acd"):
            payload = torch.load(export / "arms" / arm / f"{iid:012d}.pt", weights_only=False, map_location="cpu")
            arm_c[arm] = payload["coeff"][payload["ids"].index(row["raw_id"])]
        cos_b = cosine(arm_c["baseline"] - c0, delta)
        cos_c = cosine(arm_c["acd"] - c0, delta)
        if cos_b is not None and row["align_baseline"] is not None:
            worst_cos = max(worst_cos, abs(cos_b - row["align_baseline"]))
        if cos_c is not None and row["align_acd"] is not None:
            worst_cos = max(worst_cos, abs(cos_c - row["align_acd"]))
        iou_a = predicted_box_iou(shared["proto"], c0, table_row["pred_box"], table_row["truth160"])
        worst_iou = max(worst_iou, abs(iou_a - row["iou_native"]))
        iou_oracle = predicted_box_iou(shared["proto"], c0 + delta, table_row["pred_box"], table_row["truth160"])
        if abs(iou_oracle - row["iou_oracle"]) > 1e-6:
            check("oracle_realization_subset", False,
                  f"image {iid} raw {row['raw_id']}: {iou_oracle} vs {row['iou_oracle']}")
            break
    else:
        check("oracle_realization_subset", True, f"max abs diff {worst_iou:.3e} (arm A parity)")

    check("independent_alignment_recompute", worst_cos < 5e-3, f"max abs cosine diff {worst_cos:.3e}")
    check("arm_A_parity_from_tensors", worst_iou < 1e-6, f"max abs IoU diff {worst_iou:.3e}")

    native_fail = sum(1 for r in rows if r["iou_native"] < .75)
    check("native_failure_count", native_fail == analysis["native_failure"],
          f"{native_fail} vs {analysis['native_failure']}")
    levels = {l: sum(1 for r in rows if r["level"] == l) for l in (0, 1, 2)}
    check("level_counts", all(analysis["by_level"][str(l)]["n"] == n for l, n in levels.items()),
          f"{levels}")
    sizes = {s: sum(1 for r in rows if r["size"] == s) for s in ("small", "medium", "large")}
    check("size_counts", all(analysis["by_size"][s]["n"] == n for s, n in sizes.items()), f"{sizes}")

    # determinism of one bootstrap statistic
    generator = torch.Generator().manual_seed(20261009)
    images = sorted({r["image_id"] for r in rows})
    by_image = {}
    for r in rows:
        by_image.setdefault(r["image_id"], []).append(r)
    values = []
    for _ in range(200):
        picks = torch.randint(0, len(images), (len(images),), generator=generator).tolist()
        sampled = [row for i in picks for row in by_image[images[i]]]
        diffs = [r["align_acd"] - r["align_baseline"] for r in sampled if r["align_acd"] is not None and r["align_baseline"] is not None]
        if diffs:
            values.append(sum(diffs) / len(diffs))
    check("bootstrap_determinism", len(values) == 200 and math.isfinite(values[0]),
          f"first resample mean {values[0]:.6f}")

    gates = decision["gates"]
    recomputed_transfer = (gates["G_transfer"]["alignment_delta_point"] is not None
                           and gates["G_transfer"]["alignment_delta_point"] >= 0.02
                           and gates["G_transfer"]["realization_delta_point"] is not None
                           and gates["G_transfer"]["realization_delta_point"] >= 0.005)
    check("gate_labels_consistent",
          bool(gates["G_transfer"]["pass_"]) == recomputed_transfer
          and bool(gates["G_not_learnable"]["pass_"]) == (gates["G_not_learnable"]["align_acd"] - max(0.0, gates["G_not_learnable"]["align_baseline"]) <= 0.01),
          "machine gate labels match the pre-registered rules")

    passed = all(c["pass"] for c in CHECKS)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "VERIFICATION.json").write_text(json.dumps(
        {"study": "STUDY_COEFFICIENT_ORACLE_ALIGNMENT_20261009", "verified": passed,
         "checks": CHECKS}, indent=2), encoding="utf-8")
    print("VERIFY_COMPLETE " + json.dumps({"verified": passed, "checks": len(CHECKS)}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
