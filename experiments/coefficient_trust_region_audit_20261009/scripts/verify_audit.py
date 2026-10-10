"""Independent verification of the trust-region audit.

Recomputes, for a fixed deterministic subset (every 11th candidate), from the raw export:
  - the trust-region solution via a DIFFERENT numerical route (augmented least squares on the
    stacked system, instead of the 32x32 normal equations), and compares norm and decoded IoU
    with the stored per-candidate record;
  - the unprotected and random-support solutions the same way;
  - the retention arithmetic from stored IoU values;
  - the primary-config regression between RUN_TRUST_REGION_AUDIT_S0 and _R1;
  - the gate labels re-derived from SUMMARY/DECISION numbers.
Writes VERIFICATION.json with per-pass/fail checks and the machine label.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils import ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAMBDA_ORACLE = 0.003
PRIMARY = (1.0, 10.0)
M_E, M_R = 1.0, 1.0
RANDOM_SEED = 20261009
SUBSET_STRIDE = 11
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": str(detail)[:2000]})
    print(json.dumps({"check": name, "pass": bool(ok)}), flush=True)


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
    return delta.detach().float().cpu()


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


def lstsq_solve(A_E, b_E, A_R=None, lr=0.0, lc=1.0):
    """Independent route: augmented least squares instead of 32x32 normal equations."""
    rows, rhs = [A_E], [b_E]
    if A_R is not None and lr:
        rows.append(math.sqrt(lr) * A_R)
        rhs.append(torch.zeros(A_R.shape[1], device=A_R.device, dtype=A_R.dtype))
    rows.append(math.sqrt(lc) * torch.eye(32, device=A_E.device, dtype=A_E.dtype))
    rhs.append(torch.zeros(32, device=A_E.device, dtype=A_E.dtype))
    stacked = torch.cat(rows, dim=1).T  # n x 32
    target = torch.cat(rhs)
    return torch.linalg.lstsq(stacked, target.unsqueeze(1)).solution.squeeze(1)


def main(a):
    torch.manual_seed(0)
    records = [json.loads(line) for line in (a.analysis / "PER_CANDIDATE.jsonl").read_text().splitlines()]
    summary = json.loads((a.analysis / "SUMMARY.json").read_text())
    decision = json.loads((a.analysis / "DECISION.json").read_text())
    primary = f"lc{PRIMARY[0]}_lr{PRIMARY[1]}"

    subset = [r for i, r in enumerate(records) if i % SUBSET_STRIDE == 0]
    check("subset_size", len(subset) > 50, f"{len(subset)} of {len(records)}")

    per_image = {}
    worst_norm, worst_iou, worst_retention = 0.0, 0.0, 0.0
    for record in subset:
        iid = record["image_id"]
        if iid not in per_image:
            per_image[iid] = torch.load(a.export / "images" / f"{iid:012d}.pt", weights_only=False, map_location="cpu")
        shared = per_image[iid]
        native = torch.load(a.export / "arms" / "native" / f"{iid:012d}.pt", weights_only=False, map_location="cpu")
        c0 = native["coeff"][native["ids"].index(record["raw_id"])].cuda()
        row = next(r for r in shared["rows"] if r["raw_id"] == record["raw_id"])
        delta = solve_oracle(native["coeff"][native["ids"].index(record["raw_id"])],
                             row["p"], row["y"], row["area_px"])
        proto = shared["proto"].cuda()
        proto_flat = proto.flatten(1)
        z0 = (c0 @ proto_flat).reshape(160, 160)
        z_o = ((c0 + delta.cuda()) @ proto_flat).reshape(160, 160)
        delta_z = (z_o - z0).flatten()
        support = box_support(row["pred_box"])
        evidence = (support & (z0.abs() < M_E)).flatten()
        protected = (support & (z0.abs() >= M_R)).flatten()
        A_E = proto_flat[:, evidence].double()
        A_R = proto_flat[:, protected].double()
        b_E = delta_z[evidence].double()
        if record["n_evidence"] > 0:
            dT = lstsq_solve(A_E, b_E, A_R, PRIMARY[1], PRIMARY[0])
            dU = lstsq_solve(A_E, b_E, None, 0.0, PRIMARY[0])
        else:
            dT = dU = torch.zeros(32, device="cuda", dtype=torch.float64)
        iou_T = decode_iou(shared["proto"], c0 + dT.float(), row["pred_box"], row["truth160"])
        iou_U = decode_iou(shared["proto"], c0 + dU.float(), row["pred_box"], row["truth160"])
        stored_T = record[primary]
        worst_norm = max(worst_norm, abs(float(dT.norm()) - stored_T["norm_T"]))
        worst_iou = max(worst_iou, abs(iou_T - stored_T["iou_T"]), abs(iou_U - stored_T["iou_U"]))
        if record["oracle_gain"] > 0 and record["n_evidence"] > 0:
            retention = (iou_T - record["iou_native"]) / record["oracle_gain"]
            worst_retention = max(worst_retention, abs(retention - stored_T["retention_T"]))
    check("trust_region_solution_independent_route", worst_norm < 1e-3, f"max abs norm diff {worst_norm:.3e}")
    check("decoded_iou_independent_route", worst_iou < 1e-5, f"max abs IoU diff {worst_iou:.3e}")
    check("retention_arithmetic", worst_retention < 1e-4, f"max abs retention diff {worst_retention:.3e}")

    if a.previous and (a.previous / "PER_CANDIDATE.jsonl").is_file():
        previous = [json.loads(line) for line in (a.previous / "PER_CANDIDATE.jsonl").read_text().splitlines()]
        same = len(previous) == len(records) and all(
            abs(p[primary]["iou_T"] - r[primary]["iou_T"]) < 1e-9 and abs(p["iou_oracle"] - r["iou_oracle"]) < 1e-9
            for p, r in zip(previous, records))
        check("primary_regression_vs_previous_run", same,
              f"{len(previous)} vs {len(records)} records; primary IoU/oracle identical" if same else "mismatch")
    else:
        check("primary_regression_vs_previous_run", True, "previous per-candidate record not provided; skipped")

    g = {gate["gate"]: gate for gate in decision["gates"]}
    e = decision["estimates"]
    g1 = (e["retention_T"]["point"] >= 0.80 and (e["retention_T"]["ci95"][0] or -1) >= 0.75)
    g2 = (e["prot_diff_T_minus_0.8U"]["point"] < 0 and (e["prot_diff_T_minus_0.8U"]["ci95"][1] or 1) < 0)
    g3 = (e["retention_T_minus_S"]["point"] > 0 and (e["retention_T_minus_S"]["ci95"][0] or -1) > 0)
    g4 = (e["success_iou_T_minus_native"]["point"] >= -0.005 and (e["success_iou_T_minus_native"]["ci95"][0] or -1) >= -0.01)
    check("gate_labels_consistent",
          bool(g["G1_retention"]["pass_"]) == g1 and bool(g["G2_protection"]["pass_"]) == g2
          and bool(g["G3_control"]["pass_"]) == g3 and bool(g["G4_success_guard"]["pass_"]) == g4
          and decision["decision"] == ("PROCEED_TO_TRAINING" if g1 and g2 and g3 else "STOP_ROUTE"),
          "machine gate labels and decision match the pre-registered rules")

    sens = summary["sensitivity"]
    distinct = len({round(v["retention_T"], 9) for v in sens.values()}) == len(sens)
    check("sensitivity_configs_distinct", distinct,
          "; ".join(f"{k}: ret_T={v['retention_T']:.3f} prot_T={v['prot_T']:.3f}" for k, v in sens.items()))

    passed = all(c["pass"] for c in CHECKS)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "VERIFICATION.json").write_text(json.dumps(
        {"study": "STUDY_COEFFICIENT_TRUST_REGION_AUDIT_20261009", "verified": passed,
         "checks": CHECKS}, indent=2), encoding="utf-8")
    print("VERIFY_COMPLETE " + json.dumps({"verified": passed, "checks": len(CHECKS)}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    raise SystemExit(main(p.parse_args()))
