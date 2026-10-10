"""Trust-region oracle projection audit (no training, no new forward).

Inputs: the stage-1 export of STUDY_COEFFICIENT_ORACLE_ALIGNMENT_20261009
(RUN_COEFF_ALIGN_EXPORT_S0_R1): per-image proto / boxes / rows (GT-box ROI, GT 160-grid map,
predicted box) and per-arm native coefficients.

Per candidate:
  - oracle d* = argmin BCE_GTbox(P(c0+d))/area + 0.003*||d||^2/2   (LBFGS 120, double)
  - evidence region E = inside predicted box AND |z0| < m_E ; protected region R = inside box AND |z0| >= m_R
  - U  d = argmin ||A_E d - b_E||^2 + lc*||d||^2
  - T  d = argmin ||A_E d - b_E||^2 + lr*||A_R d||^2 + lc*||d||^2
  - S  same as U with a seeded random same-cardinality pixel subset of the box
  - O  d = d* ; Z d = 0
  - decode at the predicted box (official >0 threshold) -> IoU vs GT; retention vs oracle gain;
    protected-zone logit change relative to the oracle's own.

Gates (pre-registered in PROTOCOL.md) are evaluated on the primary configuration only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils import ops

import sys
_vendor_parser = argparse.ArgumentParser(add_help=False)
_vendor_parser.add_argument("--vendor", type=Path)
_vendor_args, _ = _vendor_parser.parse_known_args()
if _vendor_args.vendor:
    sys.path.insert(0, str(_vendor_args.vendor.resolve()))

LAMBDA_ORACLE = 0.003
PRIMARY = (1.0, 10.0)  # (lambda_c, lambda_R)
CONFIGS = [(1.0, 10.0), (0.1, 10.0), (10.0, 10.0), (1.0, 1.0), (1.0, 100.0)]
M_E, M_R = 1.0, 1.0
BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20261009
RANDOM_SEED = 20261009


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def solve_oracle(c0, p, y, area, penalty=LAMBDA_ORACLE, iterations=120):
    p, y, c0 = p.double().cuda(), y.double().cuda(), c0.double().cuda()
    area = float(area)

    def objective(delta):
        logits = p @ (c0 + delta)
        return F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / area + penalty * delta.square().sum() / 2

    delta = torch.zeros_like(c0, requires_grad=True)
    optimizer = torch.optim.LBFGS([delta], lr=1, max_iter=iterations,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = objective(delta)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = objective(delta)
    gradient = torch.autograd.grad(value, delta)[0]
    return delta.detach().float().cpu(), float(value.detach()), float(gradient.norm())


def box_support(box, device="cuda"):
    x1, y1, x2, y2 = (box / 4).tolist()
    rr = torch.arange(160, device=device)[None, :]
    cc = torch.arange(160, device=device)[:, None]
    return (rr >= x1) & (rr < x2) & (cc >= y1) & (cc < y2)


def decode_iou(proto, c, box, truth):
    z = (c.cuda() @ proto.cuda().flatten(1)).reshape(160, 160)
    cropped = ops.crop_mask(z[None].clone(), box.cuda()[None] / 4)[0] > 0
    inter = int((cropped & truth.cuda()).sum())
    union = int((cropped | truth.cuda()).sum())
    return inter / max(union, 1)


def constrained_solve(A_E, b_E, A_R=None, lr=0.0, lc=1.0):
    """min_d ||A_E^T d - b_E||^2 (+ lr*||A_R^T d||^2) + lc*||d||^2 ; A_* are 32 x n."""
    G = A_E @ A_E.T + lc * torch.eye(32, device=A_E.device, dtype=A_E.dtype)
    if A_R is not None and lr:
        G = G + lr * (A_R @ A_R.T)
    return torch.linalg.solve(G, A_E @ b_E)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(rows_by_image, statistic, seed=BOOTSTRAP_SEED, n=BOOTSTRAP):
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
    return values[max(0, int(0.025 * len(values)) - 1)], values[min(len(values) - 1, int(0.975 * len(values)))], len(values)


def main(a):
    torch.manual_seed(0)
    a.out.mkdir(parents=True, exist_ok=True)
    export = a.export
    image_files = sorted((export / "images").glob("*.pt"))
    assert image_files

    rows_all = []
    for path in image_files:
        iid = int(path.stem)
        shared = torch.load(path, weights_only=False, map_location="cpu")
        native = torch.load(export / "arms" / "native" / path.name, weights_only=False, map_location="cpu")
        c0_by_id = {rid: native["coeff"][i] for i, rid in enumerate(native["ids"])}
        for row in shared["rows"]:
            c0 = c0_by_id[row["raw_id"]]
            delta, obj_value, stationary = solve_oracle(c0, row["p"], row["y"], row["area_px"])
            proto = shared["proto"].cuda()
            c0_g = c0.cuda()
            z0 = (c0_g @ proto.flatten(1)).reshape(160, 160)
            z_oracle = ((c0_g + delta.cuda()) @ proto.flatten(1)).reshape(160, 160)
            delta_z = (z_oracle - z0).flatten()
            support = box_support(row["pred_box"])
            evidence = (support & (z0.abs() < M_E)).flatten()
            protected = (support & (z0.abs() >= M_R)).flatten()
            record = dict(image_id=iid, annotation_id=row["annotation_id"], raw_id=row["raw_id"],
                          level=row["level"], area=row["area"],
                          n_box=int(support.sum()), n_evidence=int(evidence.sum()),
                          n_protected=int(protected.sum()),
                          oracle_norm=float(delta.norm()), oracle_objective=obj_value,
                          oracle_stationary=stationary)
            # native iou: recompute from stored tensors for consistency
            record["iou_native"] = decode_iou(shared["proto"], c0, row["pred_box"], row["truth160"])
            record["iou_oracle"] = decode_iou(shared["proto"], c0 + delta, row["pred_box"], row["truth160"])
            proto_flat = proto.flatten(1)  # 32 x 25600
            A_E = proto_flat[:, evidence].double()
            A_R = proto_flat[:, protected].double()
            b_E = (delta_z[evidence]).double()
            oracle_gain = record["iou_oracle"] - record["iou_native"]
            record["oracle_gain"] = oracle_gain
            if record["n_evidence"] > 0:
                generator = torch.Generator(device="cuda").manual_seed(RANDOM_SEED)
                pool = support.flatten().nonzero(as_tuple=True)[0]
                perm = torch.randperm(len(pool), generator=generator, device="cuda")[: record["n_evidence"]]
                random_support = torch.zeros(25600, dtype=torch.bool, device="cuda")
                random_support[pool[perm]] = True
                A_S = proto_flat[:, random_support].double()
                b_S = (delta_z[random_support]).double()
            else:
                A_S = b_S = None
            for name, (lc, lr) in zip([f"lc{c}_lr{r}" for c, r in CONFIGS], CONFIGS):
                if record["n_evidence"] > 0:
                    d = constrained_solve(A_E, b_E, None, 0.0, lc)
                    dT = constrained_solve(A_E, b_E, A_R, lr, lc)
                else:
                    d = dT = torch.zeros(32, device="cuda", dtype=torch.float64)
                arm = {}
                arm["iou_U"] = decode_iou(shared["proto"], c0.cuda() + d.float(), row["pred_box"], row["truth160"])
                arm["iou_T"] = decode_iou(shared["proto"], c0.cuda() + dT.float(), row["pred_box"], row["truth160"])
                arm["norm_U"] = float(d.norm())
                arm["norm_T"] = float(dT.norm())
                if record["n_protected"] > 0:
                    oracle_prot = float((A_R.T @ delta.double().cuda()).norm())
                    arm["prot_U"] = float((A_R.T @ d).norm()) / max(oracle_prot, 1e-12)
                    arm["prot_T"] = float((A_R.T @ dT).norm()) / max(oracle_prot, 1e-12)
                else:
                    arm["prot_U"] = arm["prot_T"] = None
                record[name] = arm
            if A_S is not None:
                dS = constrained_solve(A_S, b_S, None, 0.0, PRIMARY[0])
                record["iou_S"] = decode_iou(shared["proto"], c0.cuda() + dS.float(), row["pred_box"], row["truth160"])
                record["norm_S"] = float(dS.norm())
            else:
                record["iou_S"] = None
            rows_all.append(record)
            if a.limit and len(rows_all) >= a.limit:
                break
        if a.limit and len(rows_all) >= a.limit:
            break
        if len(rows_all) % 200 == 0:
            print(json.dumps({"candidates": len(rows_all)}), flush=True)

    # retention per config: only for candidates with a positive oracle gain AND a non-empty
    # evidence region (protocol: empty-region candidates are unknown, never imputed)
    for record in rows_all:
        eligible = record["oracle_gain"] > 0 and record["n_evidence"] > 0
        for name, _ in zip([f"lc{c}_lr{r}" for c, r in CONFIGS], CONFIGS):
            arm = record[name]
            arm["retention_U"] = ((arm["iou_U"] - record["iou_native"]) / record["oracle_gain"]) if eligible else None
            arm["retention_T"] = ((arm["iou_T"] - record["iou_native"]) / record["oracle_gain"]) if eligible else None
        record["retention_S"] = ((record["iou_S"] - record["iou_native"]) / record["oracle_gain"]
                                 if eligible and record.get("iou_S") is not None else None)

    primary = f"lc{PRIMARY[0]}_lr{PRIMARY[1]}"
    by_image = {}
    for row in rows_all:
        by_image.setdefault(row["image_id"], []).append(row)

    def agg(rows, name=primary):
        return dict(
            n=len(rows),
            retention_U=mean([r[name]["retention_U"] for r in rows]),
            retention_T=mean([r[name]["retention_T"] for r in rows]),
            retention_S=mean([r.get("retention_S") for r in rows]),
            iou_T_minus_native=mean([r[name]["iou_T"] - r["iou_native"] for r in rows]),
            iou_U_minus_native=mean([r[name]["iou_U"] - r["iou_native"] for r in rows]),
            prot_U=mean([r[name]["prot_U"] for r in rows]),
            prot_T=mean([r[name]["prot_T"] for r in rows]),
            norm_T=mean([r[name]["norm_T"] for r in rows]),
            norm_U=mean([r[name]["norm_U"] for r in rows]),
        )

    eligible = [r for r in rows_all if r["oracle_gain"] > 0]
    native_failure = [r for r in rows_all if r["iou_native"] < .75]
    native_success = [r for r in rows_all if r["iou_native"] >= .75]

    estimates = {}
    def add(name, fn, rows_by_img):
        flat = [r for rs in rows_by_img.values() for r in rs]
        point = fn(flat)
        lo, hi, n = bootstrap_ci(rows_by_img, fn)
        estimates[name] = dict(point=point, ci95=[lo, hi], bootstrap_samples=n)

    add("retention_T", lambda rs: mean([r[primary]["retention_T"] for r in rs]),
        {i: [r for r in rs if r["oracle_gain"] > 0] for i, rs in by_image.items() if any(r["oracle_gain"] > 0 for r in rs)})
    add("retention_S", lambda rs: mean([r.get("retention_S") for r in rs]),
        {i: [r for r in rs if r["oracle_gain"] > 0] for i, rs in by_image.items() if any(r["oracle_gain"] > 0 for r in rs)})
    add("prot_U", lambda rs: mean([r[primary]["prot_U"] for r in rs]), by_image)
    add("prot_T", lambda rs: mean([r[primary]["prot_T"] for r in rs]), by_image)
    add("prot_diff_T_minus_0.8U", lambda rs: mean(
        [r[primary]["prot_T"] for r in rs if r[primary]["prot_T"] is not None]) - 0.8 * mean(
        [r[primary]["prot_U"] for r in rs if r[primary]["prot_U"] is not None]), by_image)
    add("retention_T_minus_S", lambda rs: mean([r[primary]["retention_T"] for r in rs]) - mean(
        [r.get("retention_S") for r in rs]),
        {i: [r for r in rs if r["oracle_gain"] > 0 and r.get("retention_S") is not None]
         for i, rs in by_image.items() if any(r["oracle_gain"] > 0 and r.get("retention_S") is not None for r in rs)})
    add("success_iou_T_minus_native", lambda rs: mean(
        [r[primary]["iou_T"] - r["iou_native"] for r in rs if r["iou_native"] >= .75]),
        {i: [r for r in rs if r["iou_native"] >= .75] for i, rs in by_image.items() if any(r["iou_native"] >= .75 for r in rs)})

    def gate(name, ok, rule, **extra):
        return {"gate": name, "pass_": bool(ok), "rule": rule, **extra}

    g1 = gate("G1_retention",
              estimates["retention_T"]["point"] is not None and estimates["retention_T"]["point"] >= 0.80
              and (estimates["retention_T"]["ci95"][0] or -1) >= 0.75,
              "mean retention_T >= 0.80 with bootstrap lower >= 0.75",
              estimate=estimates["retention_T"])
    pd = estimates["prot_diff_T_minus_0.8U"]
    g2 = gate("G2_protection", pd["point"] is not None and pd["point"] < 0 and (pd["ci95"][1] or 1) < 0,
              "prot_change_T <= 0.80 * prot_change_U with bootstrap upper < 0",
              estimate=pd, prot_U=estimates["prot_U"], prot_T=estimates["prot_T"])
    g3 = gate("G3_control", estimates["retention_T_minus_S"]["point"] is not None
              and estimates["retention_T_minus_S"]["point"] > 0
              and (estimates["retention_T_minus_S"]["ci95"][0] or -1) > 0,
              "retention_T > retention_S with bootstrap lower > 0",
              estimate=estimates["retention_T_minus_S"])
    g4 = gate("G4_success_guard", estimates["success_iou_T_minus_native"]["point"] is not None
              and estimates["success_iou_T_minus_native"]["point"] >= -0.005
              and (estimates["success_iou_T_minus_native"]["ci95"][0] or -1) >= -0.01,
              "mean(iou_T - iou_native) >= -0.005 on native-success, bootstrap lower >= -0.01",
              estimate=estimates["success_iou_T_minus_native"])

    proceed = g1["pass_"] and g2["pass_"] and g3["pass_"]
    decision = "PROCEED_TO_TRAINING" if proceed else "STOP_ROUTE"

    summary = dict(
        candidates=len(rows_all), eligible_oracle_gain=len(eligible),
        native_failure=len(native_failure), native_success=len(native_success),
        empty_evidence=sum(1 for r in rows_all if r["n_evidence"] == 0),
        empty_protected=sum(1 for r in rows_all if r["n_protected"] == 0),
        primary_config={"lambda_c": PRIMARY[0], "lambda_R": PRIMARY[1], "m_E": M_E, "m_R": M_R},
        overall=agg(rows_all),
        native_failure_block=agg(native_failure),
        native_success_block=agg(native_success),
        estimates=estimates,
        gates=[g1, g2, g3, g4],
        decision=decision,
        sensitivity={f"lc{lc}_lr{lr}": agg(rows_all, f"lc{lc}_lr{lr}") for lc, lr in CONFIGS},
    )
    write_json(a.out / "SUMMARY.json", summary)
    write_json(a.out / "DECISION.json", dict(study="STUDY_COEFFICIENT_TRUST_REGION_AUDIT_20261009",
                                             decision=decision, gates=[g1, g2, g3, g4],
                                             estimates=estimates,
                                             note="Gates and parameters were frozen in PROTOCOL.md before execution"))
    with (a.out / "PER_CANDIDATE.jsonl").open("w", encoding="utf-8") as sink:
        for record in rows_all:
            sink.write(json.dumps(record, allow_nan=False) + "\n")
    write_json(a.out / "COMPLETE.json", {"status": "complete", "candidates": len(rows_all), "decision": decision})
    print("AUDIT_COMPLETE " + json.dumps(dict(candidates=len(rows_all), decision=decision,
                                              g1=g1["pass_"], g2=g2["pass_"], g3=g3["pass_"], g4=g4["pass_"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
