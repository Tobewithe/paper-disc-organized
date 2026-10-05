"""Solve the shared native coefficient head on the frozen official TAL bank.

The bank is produced by official_pipeline.py, which calls the native
YOLODataset/TAL criterion before saving each one-to-one positive identity.
This script only optimizes the three native affine last layers and finite
per-instance coefficient oracles under the same ROI BCE + displacement target.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.utils import ops

LAMBDA = 0.003


def load_rows(cache: Path, group: str):
    index = json.loads((cache / "INDEX.json").read_text(encoding="utf-8"))
    rows = []
    for item in index[group]:
        iid = int(item["image_id"])
        image = torch.load(cache / "images" / f"{iid:012d}.pt", map_location="cpu", weights_only=False)
        proto = image["proto"].float()
        up = F.interpolate(proto[None], (640, 640), mode="bilinear", align_corners=False)[0]
        masks = image["masks"].float()
        # target_boxes and owners are saved in the same compact one-to-one
        # positive order as rows.  r["gt_index"] is the full-image label
        # index and must never be used as an index into target_boxes.
        for k, r in enumerate(image["rows"]):
            raw = int(r["raw_id"])
            box = image["target_boxes"][k].float()
            # crop_mask is an in-place operation in Ultralytics 8.4.100;
            # allocate a fresh support canvas for every instance.
            full = torch.ones((1, 640, 640), dtype=torch.float32)
            support = ops.crop_mask(full, box[None])[0].bool()
            if not bool(support.any()):
                raise RuntimeError(f"empty official support: {group} {iid} {r['annotation_id']}")
            p = up[:, support].T.contiguous()
            owner = int(image["owners"][k])
            y = (masks[support] == owner + 1).float()
            area = float(((box[2:] - box[:2]) / 640.0).prod() * (640.0 * 640.0))
            rows.append({
                "image_id": iid,
                "annotation_id": int(r["annotation_id"]),
                "raw_id": raw,
                "pyramid_level": int(r["level"]),
                "target_gt_idx": int(r["gt_index"]),
                "h": image["h"][raw].float(),
                "c": image["coeff"][raw].float(),
                "p": p,
                "y": y,
                "area": area,
                "box_iou": float(r.get("box_iou", float("nan"))),
            })
    return rows


def stats(rows):
    out = {}
    for level in range(3):
        h = torch.stack([r["h"] for r in rows if r["pyramid_level"] == level]).double()
        out[level] = (h.mean(0), h.std(0).clamp_min(1e-6))
    return out


def objective(A, rows, st, device, seg_gain, chunk=32, history=None):
    total = torch.zeros((), dtype=torch.float64, device=device)
    bce_sum = torch.zeros((), dtype=torch.float64, device=device)
    reg_sum = torch.zeros((), dtype=torch.float64, device=device)
    for lo in range(0, len(rows), chunk):
        q = rows[lo:lo + chunk]
        vals = []
        bces = []
        regs = []
        for r in q:
            mu, sd = st[r["pyramid_level"]]
            h = ((r["h"].double() - mu) / sd).to(device)
            h = torch.cat((h, torch.ones(1, dtype=torch.float64, device=device)))
            d = h @ A[r["pyramid_level"]]
            p = r["p"].to(device=device, dtype=torch.float64)
            y = r["y"].to(device=device, dtype=torch.float64)
            c = r["c"].to(device=device, dtype=torch.float64) + d
            # This is the exact scalar gain applied by Ultralytics' native
            # segmentation loss (model.args.box).  The displacement penalty
            # remains outside that official loss, as specified by 7D.
            b = seg_gain * F.binary_cross_entropy_with_logits(p @ c, y, reduction="sum") / r["area"]
            reg = LAMBDA * d.square().sum() / 2.0
            vals.append(b + reg); bces.append(b); regs.append(reg)
        total = total + torch.stack(vals).sum()
        bce_sum = bce_sum + torch.stack(bces).sum()
        reg_sum = reg_sum + torch.stack(regs).sum()
    val = total / len(rows)
    if history is not None and not val.requires_grad:
        history.append(float(val.detach().cpu()))
    return val, bce_sum / len(rows), reg_sum / len(rows)


def solve_shared(rows, device, seg_gain, max_iter=50, chunk=32, init_scale=0.0):
    st = stats(rows)
    A = [(torch.randn((65, 32), dtype=torch.float64, device=device) * init_scale).requires_grad_(True) for _ in range(3)]
    history = []
    last_stationarity = float("nan")
    opt = torch.optim.LBFGS(A, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-12)
    start = time.monotonic()
    def closure():
        nonlocal last_stationarity
        opt.zero_grad()
        total = 0.0
        grads = [torch.zeros_like(a) for a in A]
        scale = 1.0 / len(rows)
        # Build and release one small autograd graph at a time.  This is
        # algebraically the exact full-batch gradient, without retaining all
        # 6k instance graphs simultaneously on the GPU.
        for lo in range(0, len(rows), chunk):
            q = rows[lo:lo + chunk]
            with torch.enable_grad():
                v, _, _ = objective(A, q, st, device, seg_gain, len(q))
                g = torch.autograd.grad(v, A, retain_graph=False, allow_unused=True)
            weight = len(q) * scale
            total += float(v.detach().cpu()) * weight
            for k, gg in enumerate(g):
                if gg is not None:
                    grads[k].add_(gg.detach(), alpha=weight)
        for a, g in zip(A, grads):
            a.grad = g
        last_stationarity = max(float(g.norm().detach().cpu()) for g in grads)
        value = torch.tensor(total, dtype=torch.float64, device=device)
        history.append(total)
        return value
    opt.step(closure)
    with torch.no_grad():
        v, b, reg = objective(A, rows, st, device, seg_gain, chunk)
    return [a.detach().cpu() for a in A], st, {
        "objective": float(v.cpu()), "bce": float(b.cpu()), "regularizer": float(reg.cpu()),
        "stationarity_norm": last_stationarity,
        "iterations": int(opt.state[A[0]].get("n_iter", -1)), "elapsed_s": time.monotonic() - start,
        "objective_trace": history,
    }


def independent_objective(delta, p, y, valid, c, areas, seg_gain):
    logits = torch.bmm(p, (c + delta).unsqueeze(-1)).squeeze(-1)
    bce = seg_gain * (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * valid).sum(1) / areas
    reg = LAMBDA * delta.square().sum(1) / 2.0
    return (bce + reg).mean(), bce.mean(), reg.mean()


def solve_independent(rows, device, seg_gain, max_iter=60, batch_size=32):
    vals = {"objective": 0.0, "bce": 0.0, "regularizer": 0.0}
    stationarity = []
    start = time.monotonic()
    for lo in range(0, len(rows), batch_size):
        q = rows[lo:lo + batch_size]
        m = max(len(r["y"]) for r in q)
        p = torch.zeros((len(q), m, 32), dtype=torch.float64, device=device)
        y = torch.zeros((len(q), m), dtype=torch.float64, device=device)
        valid = torch.zeros((len(q), m), dtype=torch.float64, device=device)
        c = torch.stack([r["c"].double() for r in q]).to(device)
        areas = torch.tensor([r["area"] for r in q], dtype=torch.float64, device=device)
        for k, r in enumerate(q):
            n = len(r["y"]); p[k, :n] = r["p"].double().to(device); y[k, :n] = r["y"].double().to(device); valid[k, :n] = 1.0
        delta = torch.zeros((len(q), 32), dtype=torch.float64, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); v, _, _ = independent_objective(delta, p, y, valid, c, areas, seg_gain); v.backward(); return v
        opt.step(closure)
        with torch.no_grad(): v, b, reg = independent_objective(delta, p, y, valid, c, areas, seg_gain)
        g = torch.autograd.grad(independent_objective(delta, p, y, valid, c, areas, seg_gain)[0], delta)[0]
        vals["objective"] += float(v.cpu()) * len(q); vals["bce"] += float(b.cpu()) * len(q); vals["regularizer"] += float(reg.cpu()) * len(q)
        stationarity.append(float(g.norm(dim=1).max().cpu()))
        if lo == 0 or (lo // batch_size) % 20 == 0:
            print(json.dumps({"stage":"independent","records":min(lo+len(q),len(rows)),"total":len(rows),"elapsed_s":time.monotonic()-start}), flush=True)
    n = len(rows)
    return {k: v/n for k,v in vals.items()} | {"records":n,"max_batch_stationarity":max(stationarity),"elapsed_s":time.monotonic()-start}


def eval_rows(rows, A, st, device, seg_gain, chunk=32):
    z = [torch.zeros((65,32), dtype=torch.float64, device=device) for _ in range(3)]
    with torch.no_grad():
        v0,b0,r0 = objective(z, rows, st, device, seg_gain, chunk)
        v,b,r = objective([a.to(device) for a in A], rows, st, device, seg_gain, chunk)
    return {"original":{"objective":float(v0.cpu()),"bce":float(b0.cpu()),"regularizer":float(r0.cpu()),"records":len(rows)},"shared":{"objective":float(v.cpu()),"bce":float(b.cpu()),"regularizer":float(r.cpu()),"records":len(rows)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shared-iter", type=int, default=50)
    ap.add_argument("--independent-iter", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--independent-batch", type=int, default=32)
    ap.add_argument("--seg-gain", type=float, required=True,
                    help="model.args.box from the frozen checkpoint; official native segmentation gain")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = {g: load_rows(args.cache, g) for g in ("fit", "dev", "val")}
    counts = {g:{"records":len(r),"images":len({x["image_id"] for x in r},),"levels":{str(l):sum(x["pyramid_level"]==l for x in r) for l in range(3)}} for g,r in groups.items()}
    print(json.dumps({"stage":"loaded","device":str(device),"counts":counts}), flush=True)
    A, st, shared = solve_shared(groups["fit"], device, args.seg_gain, args.shared_iter, args.chunk)
    independent = solve_independent(groups["fit"], device, args.seg_gain, args.independent_iter, args.independent_batch)
    original = eval_rows(groups["fit"], [torch.zeros((65,32),dtype=torch.float64) for _ in range(3)], st, device, args.seg_gain, args.chunk)
    summary = {"protocol":{"lambda":LAMBDA,"segmentation_gain":args.seg_gain,"official_loss":"official_pipeline cache equivalence; ROI BCE expression reused","device":str(device),"shared_fit_only":True},"counts":counts,"shared_fit":shared,"independent_fit":independent,"fit":original,"dev":eval_rows(groups["dev"],A,st,device,args.seg_gain,args.chunk),"val":eval_rows(groups["val"],A,st,device,args.seg_gain,args.chunk),"decomposition":{"J0_minus_Jshared":original["original"]["objective"]-shared["objective"],"Jshared_minus_Jind":shared["objective"]-independent["objective"],"J0_minus_Jind":original["original"]["objective"]-independent["objective"]}}
    torch.save({"A":A,"stats":st}, args.out/"SHARED_AFFINE_OFFICIAL.pt")
    (args.out/"SOLVER_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"stage":"complete","summary":str(args.out/"SOLVER_SUMMARY.json")}),flush=True)


if __name__ == "__main__": main()
