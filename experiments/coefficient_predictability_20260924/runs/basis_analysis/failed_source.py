"""Cross-image prototype geometry diagnostics (hypothesis check, not causal proof)."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def normalize(proto):
    # Remove channel means and normalize each prototype channel. Spatial pixels
    # remain in their common 640-letterbox coordinates.
    p = proto.flatten(1).float()
    p = p - p.mean(1, keepdim=True)
    return p / p.norm(dim=1, keepdim=True).clamp_min(1e-8)


def pair_metrics(a, b):
    a, b = normalize(a).cuda(), normalize(b).cuda()
    # Row-space principal angles: QR is cheaper and more stable than a full
    # 25,600 x 25,600 covariance decomposition.
    qa = torch.linalg.qr(a.T, mode="reduced").Q
    qb = torch.linalg.qr(b.T, mode="reduced").Q
    sv = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1).cpu().numpy()
    corr = (a @ b.T).abs().cpu().numpy()
    ri, ci = linear_sum_assignment(-corr)
    matched = corr[ri, ci]
    cross = (a @ b.T).cpu()
    u, _, vt = torch.linalg.svd(cross)
    rotation = u @ vt
    aligned = (rotation @ b).cuda()
    residual_before = float((a-b).square().mean().sqrt().cpu())
    residual_after = float((a-aligned).square().mean().sqrt().cpu())
    return dict(principal_cos_mean=float(sv.mean()), principal_cos_min=float(sv.min()),
        principal_cos_q10=float(np.quantile(sv,.1)), channel_abs_corr_mean=float(corr.mean()),
        channel_hungarian_mean=float(matched.mean()), channel_hungarian_min=float(matched.min()),
        procrustes_residual_before=residual_before, procrustes_residual_after=residual_after,
        procrustes_relative_reduction=float((residual_before-residual_after)/max(residual_before,1e-8)))


def main(a):
    torch.set_num_threads(6)
    random.seed(a.seed)
    images=sorted((a.bank/"images").glob("*.pt"))
    images=random.sample(images,min(a.images,len(images)))
    protos=[torch.load(p,weights_only=False,map_location="cpu")["proto"] for p in images]
    pairs=[]
    for _ in range(min(a.pairs,len(protos)*(len(protos)-1)//2)):
        i,j=random.sample(range(len(protos)),2)
        pairs.append((i,j))
    rows=[]; start=time.monotonic()
    for n,(i,j) in enumerate(pairs,1):
        rows.append(pair_metrics(protos[i],protos[j]))
        if n%50==0: print(json.dumps(dict(pairs=n,total=len(pairs),elapsed=time.monotonic()-start)),flush=True)
    keys=rows[0].keys()
    summary={k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),
                    q10=float(np.quantile([r[k] for r in rows],.1)),q90=float(np.quantile([r[k] for r in rows],.9))) for k in keys}
    result=dict(images=len(images),pairs=len(rows),summary=summary,
        protocol="Prototype channels are compared in common 640-letterbox pixel coordinates after per-channel centering and normalization. Orthogonal Procrustes is descriptive; it does not establish channel drift or causality. No GT is used.",
        limitations="Content and spatial-layout changes can produce the same geometry; cross-image angles alone cannot prove coefficient-head failure.")
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"SUMMARY.json").write_text(json.dumps(result,indent=2))
    (a.out/"ROWS.json").write_text(json.dumps(rows,indent=2))
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(images=len(images),pairs=len(rows))))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bank",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--images",type=int,default=200);p.add_argument("--pairs",type=int,default=500);p.add_argument("--seed",type=int,default=20260924)
    main(p.parse_args())
