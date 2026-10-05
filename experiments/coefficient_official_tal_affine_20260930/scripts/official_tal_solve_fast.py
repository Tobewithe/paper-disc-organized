"""Exact official-TAL shared affine audit with analytic BCE gradients.

The objective is identical to official_tal_solve.py.  The implementation
stores the fixed prototype pixels once on the GPU and accumulates the exact
coefficient gradients analytically, avoiding a retained autograd graph over
all 6,058 candidates.
"""
from __future__ import annotations

import argparse
import json
import time
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
        for k, r in enumerate(image["rows"]):
            box = image["target_boxes"][k].float()
            support = ops.crop_mask(torch.ones((1, 640, 640), dtype=torch.float32), box[None])[0].bool()
            if not bool(support.any()):
                raise RuntimeError(f"empty official support: {group} {iid} {r['annotation_id']}")
            owner = int(image["owners"][k])
            rows.append({
                "image_id": iid, "annotation_id": int(r["annotation_id"]),
                "raw_id": int(r["raw_id"]), "pyramid_level": int(r["level"]),
                "target_gt_idx": int(r["gt_index"]), "h": image["h"][int(r["raw_id"])].float(),
                "c": image["coeff"][int(r["raw_id"])].float(),
                "p": up[:, support].T.contiguous(), "y": (masks[support] == owner + 1).float(),
                "area": float(((box[2:] - box[:2]) / 640.0).prod() * 640.0 * 640.0),
                "box_iou": float(r.get("box_iou", float("nan"))),
            })
    return rows


def stats(rows):
    return {
        level: (
            torch.stack([r["h"] for r in rows if r["pyramid_level"] == level]).double().mean(0),
            torch.stack([r["h"] for r in rows if r["pyramid_level"] == level]).double().std(0).clamp_min(1e-6),
        )
        for level in range(3)
    }


class Bank:
    def __init__(self, rows, st, device, seg_gain):
        self.rows, self.st, self.device, self.seg_gain = rows, st, device, float(seg_gain)
        self.n = len(rows)
        self.level = torch.tensor([r["pyramid_level"] for r in rows], dtype=torch.long, device=device)
        self.area = torch.tensor([r["area"] for r in rows], dtype=torch.float64, device=device)
        self.c = torch.stack([r["c"] for r in rows]).double().to(device)
        hs = []
        for r in rows:
            mu, sd = st[r["pyramid_level"]]
            hs.append(torch.cat(((r["h"].double() - mu) / sd, torch.ones(1, dtype=torch.float64))))
        self.h = torch.stack(hs).to(device)
        offsets = [0]
        for r in rows:
            offsets.append(offsets[-1] + len(r["y"]))
        self.offsets = offsets
        total = offsets[-1]
        self.p = torch.empty((total, 32), dtype=torch.float32, device=device)
        self.y = torch.empty((total,), dtype=torch.float32, device=device)
        for i, r in enumerate(rows):
            lo, hi = offsets[i], offsets[i + 1]
            self.p[lo:hi].copy_(r["p"].to(device))
            self.y[lo:hi].copy_(r["y"].to(device))
        for r in rows:
            r.pop("p", None); r.pop("y", None)

    def _one(self, i, d, need_grad):
        lo, hi = self.offsets[i], self.offsets[i + 1]
        p = self.p[lo:hi].double()
        y = self.y[lo:hi].double()
        c = self.c[i] + d
        z = p @ c
        # softplus(z)-y*z is BCE-with-logits, with the official gain.
        bce = self.seg_gain * (F.softplus(z) - y * z).sum() / self.area[i]
        reg = LAMBDA * d.square().sum() / 2.0
        if not need_grad:
            return bce + reg, bce, reg, None
        err = self.seg_gain * (torch.sigmoid(z) - y) / self.area[i]
        gd = err @ p + LAMBDA * d
        return bce + reg, bce, reg, gd

    @torch.no_grad()
    def shared_value_grad(self, A, want_grad=True):
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        bsum = torch.zeros_like(total); rsum = torch.zeros_like(total)
        grads = [torch.zeros_like(a) for a in A] if want_grad else None
        for i in range(self.n):
            lev = int(self.level[i]); d = self.h[i] @ A[lev]
            v, b, r, gd = self._one(i, d, want_grad)
            total += v; bsum += b; rsum += r
            if want_grad:
                grads[lev].add_(torch.outer(self.h[i], gd) / self.n)
        return total / self.n, bsum / self.n, rsum / self.n, grads

    @torch.no_grad()
    def independent_value_grad(self, ids, delta, want_grad=True):
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        bsum = torch.zeros_like(total); rsum = torch.zeros_like(total)
        grads = torch.zeros_like(delta) if want_grad else None
        for k, i in enumerate(ids):
            v, b, r, gd = self._one(i, delta[k], want_grad)
            total += v; bsum += b; rsum += r
            if want_grad: grads[k] = gd
        n = len(ids)
        return total / n, bsum / n, rsum / n, (grads / n if want_grad else None)


def solve_shared(bank, max_iter, init_scale=0.0):
    torch.manual_seed(20260930)
    A = [(torch.randn((bank.h.shape[1], bank.c.shape[1]), dtype=torch.float64, device=bank.device).mul_(init_scale).requires_grad_()) for _ in range(3)]
    trace, last_stationarity = [], float("nan")
    opt = torch.optim.LBFGS(A, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-12)
    start = time.monotonic()
    def closure():
        nonlocal last_stationarity
        opt.zero_grad()
        v, _, _, g = bank.shared_value_grad(A, True)
        for a, gg in zip(A, g): a.grad = gg
        last_stationarity = max(float(gg.norm().detach().cpu()) for gg in g)
        value = float(v.detach().cpu()); trace.append(value)
        print(json.dumps({"stage":"shared", "closure":len(trace), "objective":value, "gradient_norm":last_stationarity, "elapsed_s":time.monotonic()-start}), flush=True)
        return torch.tensor(value, dtype=torch.float64, device=bank.device)
    opt.step(closure)
    closure()
    with torch.no_grad(): v, b, reg, _ = bank.shared_value_grad(A, False)
    return [a.detach().cpu() for a in A], {
        "objective": float(v.cpu()), "bce": float(b.cpu()), "regularizer": float(reg.cpu()),
        "stationarity_norm": last_stationarity, "iterations": int(opt.state[A[0]].get("n_iter", -1)),
        "elapsed_s": time.monotonic() - start, "objective_trace": trace,
        "initialization_scale":init_scale,"initialization_seed":20260930,
        "gradient_norm_definition":"maximum Frobenius norm of the three standardized affine gradients of the equal-candidate mean",
        "exit_reason":"iteration_limit" if int(opt.state[A[0]].get("n_iter",-1)) >= max_iter else "native_LBFGS_tolerance_or_line_search_termination",
        "global_optimality_certified":False,
    }


def solve_independent(bank, max_iter, batch_size):
    vals = {"objective": 0.0, "bce": 0.0, "regularizer": 0.0}; stationarity = []
    start = time.monotonic(); targets=torch.zeros_like(bank.c); records=[]
    for lo in range(0, bank.n, batch_size):
        ids = list(range(lo, min(lo + batch_size, bank.n)))
        delta = torch.zeros((len(ids), 32), dtype=torch.float64, device=bank.device).requires_grad_()
        opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); v, _, _, g = bank.independent_value_grad(ids, delta, True); delta.grad = g; return torch.tensor(float(v.detach().cpu()), dtype=torch.float64, device=bank.device)
        opt.step(closure)
        with torch.no_grad(): v, b, reg, g = bank.independent_value_grad(ids, delta, True)
        targets[ids]=delta.detach()
        for k,i in enumerate(ids):
            vi,bi,ri,gi=bank._one(i,delta[k].detach(),True)
            ident={key:bank.rows[i][key] for key in ('image_id','annotation_id','raw_id','pyramid_level','target_gt_idx')}
            records.append({**ident,'objective':float(vi),'bce':float(bi),'regularizer':float(ri),'gradient_norm':float(gi.norm()),'iterations':int(opt.state[delta].get('n_iter',-1)),'exit_reason':'iteration_limit' if int(opt.state[delta].get('n_iter',-1))>=max_iter else 'native_LBFGS_tolerance_or_line_search_termination'})
        vals["objective"] += float(v.cpu()) * len(ids); vals["bce"] += float(b.cpu()) * len(ids); vals["regularizer"] += float(reg.cpu()) * len(ids)
        stationarity.append(float(g.norm(dim=1).max().cpu()))
        if lo == 0 or (lo // batch_size) % 100 == 0:
            print(json.dumps({"stage":"independent", "records":min(lo+len(ids),bank.n), "total":bank.n, "elapsed_s":time.monotonic()-start}), flush=True)
    return ({k: v / bank.n for k, v in vals.items()} | {"records": bank.n, "max_batch_mean_gradient_norm": max(stationarity), "elapsed_s": time.monotonic()-start}, targets.cpu(), records)


def evaluate(bank, A):
    zero = [torch.zeros((bank.h.shape[1], bank.c.shape[1]), dtype=torch.float64, device=bank.device) for _ in range(3)]
    with torch.no_grad():
        v0, b0, r0, _ = bank.shared_value_grad(zero, False)
        v, b, r, _ = bank.shared_value_grad([a.to(bank.device) for a in A], False)
    return {"original":{"objective":float(v0.cpu()),"bce":float(b0.cpu()),"regularizer":float(r0.cpu()),"records":bank.n}, "shared":{"objective":float(v.cpu()),"bce":float(b.cpu()),"regularizer":float(r.cpu()),"records":bank.n}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shared-iter", type=int, default=50); ap.add_argument("--independent-iter", type=int, default=60)
    ap.add_argument("--independent-batch", type=int, default=4); ap.add_argument("--seg-gain", type=float, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = {g: load_rows(args.cache, g) for g in ("fit", "dev", "val")}
    counts = {g:{"records":len(r), "images":len({x["image_id"] for x in r}), "levels":{str(l):sum(x["pyramid_level"]==l for x in r) for l in range(3)}} for g,r in groups.items()}
    print(json.dumps({"stage":"loaded","device":str(device),"counts":counts}), flush=True)
    st = stats(groups["fit"])
    bank = Bank(groups["fit"], st, device, args.seg_gain)
    print(json.dumps({"stage":"bank_ready","device":str(device),"fit_pixels":bank.offsets[-1]}), flush=True)
    A, shared = solve_shared(bank, args.shared_iter)
    torch.save({"A":A,"stats":st}, args.out/"SHARED_AFFINE_OFFICIAL.pt")
    (args.out/"SHARED_SOLVE.json").write_text(json.dumps(shared,indent=2),encoding="utf-8")
    random_A, random_shared = solve_shared(bank,args.shared_iter,0.001)
    (args.out/'SHARED_RANDOM_INITIALIZATION.json').write_text(json.dumps(random_shared,indent=2),encoding='utf-8')
    shared['random_initialization_objective']=random_shared['objective']
    shared['initialization_final_objective_difference']=random_shared['objective']-shared['objective']
    independent,targets,oracle_rows = solve_independent(bank, args.independent_iter, args.independent_batch)
    torch.save({'delta':targets,'identities':oracle_rows},args.out/'ORACLE_fit.pt')
    fit = evaluate(bank, A)
    del bank
    torch.cuda.empty_cache()
    bank = Bank(groups["dev"], st, device, args.seg_gain); dev = evaluate(bank, A)
    del bank
    torch.cuda.empty_cache()
    bank = Bank(groups["val"], st, device, args.seg_gain); val = evaluate(bank, A)
    summary = {"protocol":{"lambda":LAMBDA,"segmentation_gain":args.seg_gain,"official_loss":"Gate-A reference ROI BCE; analytic gradient implementation","device":str(device),"shared_fit_only":True},"counts":counts,"shared_fit":shared,"independent_fit":independent,"fit":fit,"dev":dev,"val":val,"decomposition":{"J0_minus_Jshared":fit["original"]["objective"]-fit["shared"]["objective"],"Jshared_minus_Jind":fit["shared"]["objective"]-independent["objective"],"J0_minus_Jind":fit["original"]["objective"]-independent["objective"]}}
    torch.save({"A":A,"stats":st}, args.out/"SHARED_AFFINE_OFFICIAL.pt")
    (args.out/"SOLVER_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"stage":"complete","summary":str(args.out/"SOLVER_SUMMARY.json")}),flush=True)


if __name__ == "__main__": main()
