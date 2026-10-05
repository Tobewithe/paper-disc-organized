"""Pairwise antisymmetric ownership correction.

For a same-class candidate pair and a pixel in their joint support, a scalar
gate delta is applied as z_i += delta and z_j -= delta. This conservation
constraint prevents a pairwise module from creating foreground mass and makes
the intervention specifically about ownership. The model sees predictions and
box geometry only at inference; GT is used solely for training labels.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PairGate(nn.Module):
    def __init__(self, dim=23, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # Bounded correction keeps the pair intervention local and numerically
        # stable. The zero initialization exactly reproduces stock logits.
        return 2.0 * torch.tanh(self.net(x).squeeze(-1) / 2.0)


def pair_features(zi, zj, xi, xj, positions, shape):
    """Prediction-only pair features, deterministic and dimension-auditable."""
    h, w = [float(v) for v in shape]
    # x layout is [64 feature, 3 level one-hot, 4 xyxy normalized, 2 wh].
    # Broadcast per-candidate geometry/score scalars to the sampled pixels.
    bi, bj = xi[..., 67:73], xj[..., 67:73]
    if bi.ndim == 1:
        bi = bi.unsqueeze(-1).expand(6, zi.shape[-1]).transpose(0, 1)
        bj = bj.unsqueeze(-1).expand(6, zj.shape[-1]).transpose(0, 1)
    # Cache stores flattened input-grid indices, so recover pixel centres here.
    flat = positions.to(torch.float32)
    px = (torch.remainder(flat, w) + 0.5) / max(w, 1.0) * 2 - 1
    py = (torch.floor(flat / max(w, 1.0)) + 0.5) / max(h, 1.0) * 2 - 1
    margin = zi - zj
    # Candidate branch appearance: the first 64 channels are the source
    # feature at the coefficient-producing location. Symmetric summaries let
    # the gate distinguish similarly shaped boxes with different evidence.
    ai, aj = xi[..., :64], xj[..., :64]
    cos = F.cosine_similarity(ai, aj, dim=-1).expand_as(zi)
    l1 = (ai - aj).abs().mean().expand_as(zi)
    ni = ai.norm().expand_as(zi); nj = aj.norm().expand_as(zi)
    return torch.stack([
        zi.clamp(-10, 10) / 5, zj.clamp(-10, 10) / 5,
        zi.sigmoid(), zj.sigmoid(), margin.clamp(-10, 10) / 5,
        px, py,
        bi[..., 0], bi[..., 1], bi[..., 2], bi[..., 3], bi[..., 4], bi[..., 5],
        bj[..., 0], bj[..., 1], bj[..., 2], bj[..., 3], bj[..., 4], bj[..., 5],
        cos, l1, ni.clamp_max(20) / 20, nj.clamp_max(20) / 20,
    ], -1)


def load_records(cache, device="cuda"):
    selection = json.loads((cache / "selection.json").read_text())
    records=[]
    for iid in selection["train"]:
        with np.load(cache / "train" / f"{iid}.npz") as q:
            if not len(q["pair_source"]): continue
            records.append({
                "pair_p": torch.tensor(q["pair_p"], device=device, dtype=torch.float32),
                "x": torch.tensor(q["x"], device=device, dtype=torch.float32),
                "c": torch.tensor(q["c"], device=device, dtype=torch.float32),
                "pair_source": torch.tensor(q["pair_source"], device=device, dtype=torch.long),
                "pair_neighbor": torch.tensor(q["pair_neighbor"], device=device, dtype=torch.long),
                "pair_positions": torch.tensor(q["pair_positions"], device=device, dtype=torch.float32),
                "input_shape": tuple(int(v) for v in q["input_shape"]),
            })
    if not records: raise RuntimeError("no pair records")
    return records


def flatten_pairs(records):
    # Preserve image boundaries and directed pair identities. Each row is one
    # directed pair with 64 sampled pixels and two GT ownership labels.
    rows=[]
    for r in records:
        p=r["pair_p"]; src=r["pair_source"]; dst=r["pair_neighbor"]
        ci=r["c"][src]; cj=r["c"][dst]
        zi=torch.einsum("psk,pk->ps", p, ci); zj=torch.einsum("psk,pk->ps", p, cj)
        # Pair cache was sampled from GT_i & ~GT_j for the source direction.
        # Reconstruct the directed labels from the fact that the cache stores
        # only the source-exclusive region: y_i=1, y_j=0.
        for k in range(len(src)):
            fi=pair_features(zi[k],zj[k],r["x"][src[k]],r["x"][dst[k]],r["pair_positions"][k],r["input_shape"])
            fr=pair_features(zj[k],zi[k],r["x"][dst[k]],r["x"][src[k]],r["pair_positions"][k],r["input_shape"])
            rows.append((fi, fr, zi[k], zj[k]))
    return rows


def train(records, out, seeds=(0,1,2), epochs=15, lr=1e-4):
    rows=flatten_pairs(records); X=torch.cat([r[0] for r in rows]); Xr=torch.cat([r[1] for r in rows]); Zi=torch.cat([r[2] for r in rows]); Zj=torch.cat([r[3] for r in rows])
    n=len(X); split=max(1,int(.8*n)); perm=np.random.default_rng(20260913).permutation(n); fit=torch.tensor(perm[:split],device="cuda"); hold=torch.tensor(perm[split:],device="cuda")
    out.mkdir(parents=True, exist_ok=False); (out/"checkpoints").mkdir()
    stats=[]
    for seed in seeds:
        torch.manual_seed(seed); model=PairGate().cuda(); opt=torch.optim.Adam(model.parameters(),lr=lr); rng=np.random.default_rng(seed)
        for ep in range(epochs):
            order=rng.permutation(len(fit)); total=0.
            for start in range(0,len(order),4096):
                ids=fit[torch.tensor(order[start:start+4096],device="cuda")]; d=0.5*(model(X[ids])-model(Xr[ids]));
                # Antisymmetric pair logits: source should be positive, rival negative.
                loss=(F.softplus(-(Zi[ids]+d))+F.softplus(Zj[ids]+d)).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();total+=float(loss)*len(ids)
            with torch.no_grad():
                d=0.5*(model(X[hold])-model(Xr[hold])); before=(F.softplus(-Zi[hold])+F.softplus(Zj[hold])).mean(); after=(F.softplus(-(Zi[hold]+d))+F.softplus(Zj[hold]+d)).mean();
            stats.append(dict(seed=seed,epoch=ep+1,train_loss=total/len(fit),hold_before=float(before),hold_after=float(after),hold_delta=float(after-before)))
            torch.save({"model":model.state_dict(),"seed":seed,"epoch":ep+1},out/f"pair_gate_s{seed}_epoch{ep+1:02d}.pt")
    (out/"SUMMARY.json").write_text(json.dumps({"status":"COMPLETE","pairs":len(rows),"pixels":int(n),"fit":int(split),"holdout":int(n-split),"stats":stats},indent=2),encoding="utf-8")
    return stats


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--epochs",type=int,default=15); a=ap.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    records=load_records(a.cache); stats=train(records,a.out,epochs=a.epochs); print(json.dumps(stats[-3:],ensure_ascii=False))


if __name__=="__main__": main()
