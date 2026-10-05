"""S074: fixed-prototype coefficient capacity with pixel-held-out control.

This is a GT-assisted diagnostic, not a deployable method or AP evaluation.
The prototype and predicted box stay fixed. A regularized coefficient residual
is fitted on one deterministic set of pixels and evaluated on disjoint pixels.
"""
from pathlib import Path
import argparse, csv, contextlib, hashlib, json, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "diagnostics/joint_failure_decoder_20260913_v2"
MANIFEST = RUN / "tensor_manifest.json"
ANN = ROOT.parent.parent / "datasets/coco/annotations/instances_val2017.json"


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    values = np.r_[pos, neg]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # Tie correction by average rank.
    for value in np.unique(values):
        ix = np.flatnonzero(values == value)
        ranks[ix] = ranks[ix].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def input_mask(gt, iid, aid, shape=(640, 640)):
    own = gt.annToMask(gt.anns[aid]).astype(bool)
    h, w = own.shape
    gain = min(shape[0] / h, shape[1] / w)
    nh, nw = round(h * gain), round(w * gain)
    top, left = round((shape[0] - nh) / 2 - .1), round((shape[1] - nw) / 2 - .1)
    out = np.zeros(shape, dtype=bool)
    out[top:top + nh, left:left + nw] = cv2.resize(
        own.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST_EXACT) > 0
    return out


def fit_delta(x, z, y, steps=24):
    """Same strictly convex residual objective used by the earlier readout controls."""
    x, z, y = x.double(), z.double(), y.double()
    n, d = x.shape
    w = torch.zeros(d, device=x.device, dtype=x.dtype)
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    for _ in range(steps):
        delta = x @ w
        prob = (z + delta).sigmoid()
        grad = x.T @ ((prob - y + .02 * delta) / n) + .0002 * w
        if float(grad.abs().max()) < 1e-7:
            break
        curvature = (prob * (1 - prob) + .02) / n
        hessian = x.T @ (curvature[:, None] * x) + .0002 * eye
        step = torch.linalg.solve(hessian, grad)
        value = F.binary_cross_entropy_with_logits(z + delta, y) + .01 * delta.square().mean() + .0001 * w.square().sum()
        descent = grad @ step
        accepted = False
        for back in range(18):
            scale = .5 ** back
            trial = w - scale * step
            tdelta = x @ trial
            tvalue = F.binary_cross_entropy_with_logits(z + tdelta, y) + .01 * tdelta.square().mean() + .0001 * trial.square().sum()
            if float(tvalue) <= float(value - 1e-4 * scale * descent):
                w = trial
                accepted = True
                break
        if not accepted:
            break
    return w.float()


def iou_binary(pred, own):
    inter = np.count_nonzero(pred & own)
    union = np.count_nonzero(pred | own)
    return float(inter / max(union, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = COCO(str(ANN))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    started = time.monotonic()
    for number, item in enumerate(manifest, 1):
        aid, iid = int(item["annotation_id"]), int(item["image_id"])
        tensor_path = RUN / "tensor_cache" / f"{iid}.npz"
        cache = np.load(tensor_path)
        where = np.flatnonzero(cache["annotation_ids"].astype(np.int64) == aid)
        if len(where) != 1:
            raise RuntimeError(f"missing unique tensor for {iid}/{aid}")
        k = int(where[0])
        proto = torch.as_tensor(cache["proto"], device=device, dtype=torch.float32)
        coeff = torch.as_tensor(cache["coeff"][k], device=device, dtype=torch.float32)
        box = torch.as_tensor(cache["boxes"][k], device=device, dtype=torch.float32)
        p640 = F.interpolate(proto[None], (640, 640), mode="bilinear", align_corners=False)[0].flatten(1).T.contiguous()
        zfull = p640 @ coeff
        support = torch.ones((1, 640, 640), device=device, dtype=torch.uint8)
        # crop_mask uses the exact decoder integer handling.
        from ultralytics.utils import ops
        support = ops.crop_mask(support, box[None])[0].bool().cpu().numpy()
        own = input_mask(gt, iid, aid)
        crowd = np.zeros_like(own)
        for ann in gt.imgToAnns[iid]:
            if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                crowd |= input_mask(gt, iid, ann["id"])
        valid = support & ~crowd
        pos = np.flatnonzero(valid & own)
        neg = np.flatnonzero(valid & ~own)
        if len(pos) < 8 or len(neg) < 8:
            continue
        rng = np.random.default_rng(int(hashlib.sha256(f"S074:{aid}".encode()).hexdigest()[:8], 16))
        max_each = min(256, len(pos), len(neg))
        pos = rng.choice(pos, max_each, replace=False)
        neg = rng.choice(neg, max_each, replace=False)
        rng.shuffle(pos); rng.shuffle(neg)
        cut = max(4, max_each // 2)
        fit_ix = np.r_[pos[:cut], neg[:cut]]
        hold_ix = np.r_[pos[cut:], neg[cut:]]
        base = (zfull > 0).reshape(640, 640).cpu().numpy()
        xfit = p640[torch.as_tensor(fit_ix, device=device)]
        zfit = zfull[torch.as_tensor(fit_ix, device=device)]
        yfit = torch.as_tensor(np.r_[np.ones(cut), np.zeros(cut)], device=device)
        wall = fit_delta(xfit, zfit, yfit)
        zfit_all = xfit @ (coeff + wall)
        xhold = p640[torch.as_tensor(hold_ix, device=device)]
        zhold_split = zfull[torch.as_tensor(hold_ix, device=device)] + xhold @ wall
        # A second fit on every sampled point estimates the fixed-prototype
        # existence opportunity without pretending it is a held-out result.
        all_ix = np.r_[fit_ix, hold_ix]
        xall = p640[torch.as_tensor(all_ix, device=device)]
        zall = zfull[torch.as_tensor(all_ix, device=device)]
        yall = torch.as_tensor(np.r_[np.ones(cut), np.zeros(cut),
                                     np.ones(max_each - cut), np.zeros(max_each - cut)], device=device)
        wall_all = fit_delta(xall, zall, yall)
        full_split = (zfull + p640 @ wall > 0).reshape(640, 640).cpu().numpy() & support
        full_all = (zfull + p640 @ wall_all > 0).reshape(640, 640).cpu().numpy() & support
        hold_base = (zfull[torch.as_tensor(hold_ix, device=device)] > 0).cpu().numpy()
        hold_split = (zhold_split > 0).cpu().numpy()
        hold_y = np.r_[np.ones(max_each - cut), np.zeros(max_each - cut)].astype(bool)
        row = dict(annotation_id=aid, image_id=iid, density=item["group"].split(":", 1)[0],
                   residual=item["group"].split(":", 1)[1], n_pos=max_each, n_fit=2 * cut,
                   n_hold=2 * (max_each - cut), baseline_iou=iou_binary(base & support, own),
                   split_iou=iou_binary(full_split, own), allfit_iou=iou_binary(full_all, own),
                   baseline_hold_auc=auc(zfull[torch.as_tensor(hold_ix[:max_each-cut], device=device)].detach().cpu().numpy(),
                                         zfull[torch.as_tensor(hold_ix[max_each-cut:], device=device)].detach().cpu().numpy()),
                   split_hold_auc=auc(zhold_split[:max_each-cut].detach().cpu().numpy(), zhold_split[max_each-cut:].detach().cpu().numpy()),
                   baseline_hold_iou=float((hold_base == hold_y).mean()),
                   split_hold_iou=float((hold_split == hold_y).mean()),
                   split_param_norm=float(wall.norm().cpu()), allfit_param_norm=float(wall_all.norm().cpu()))
        row["split_gain_points"] = 100 * (row["split_iou"] - row["baseline_iou"])
        row["allfit_gain_points"] = 100 * (row["allfit_iou"] - row["baseline_iou"])
        row["hold_auc_gain_points"] = 100 * (row["split_hold_auc"] - row["baseline_hold_auc"])
        rows.append(row)
        if number % 24 == 0 or number == len(manifest):
            print(json.dumps(dict(done=number, total=len(manifest), rows=len(rows), seconds=round(time.monotonic()-started, 2))), flush=True)
    pd.DataFrame(rows).to_csv(out / "instances.csv", index=False)
    d = pd.DataFrame(rows)
    summary = []
    for (density, residual), g in d.groupby(["density", "residual"]):
        summary.append(dict(density=density, residual=residual, n=len(g),
            baseline_iou=float(g.baseline_iou.mean()), split_iou=float(g.split_iou.mean()),
            allfit_iou=float(g.allfit_iou.mean()), split_gain_points=float(g.split_gain_points.mean()),
            allfit_gain_points=float(g.allfit_gain_points.mean()), baseline_hold_auc=float(g.baseline_hold_auc.mean()),
            split_hold_auc=float(g.split_hold_auc.mean()), hold_auc_gain_points=float(g.hold_auc_gain_points.mean()),
            split_iou75=int((g.split_iou >= .75).sum()), allfit_iou75=int((g.allfit_iou >= .75).sum())))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    protocol = dict(experiment="S074_FIXED_PROTOTYPE_SPLIT_ORACLE", training=False,
        question="Does a fixed prototype support a coefficient correction that transfers to held-out pixels?",
        population="Frozen S070 tensor manifest: 144 targets, 24 per density x prior residual stratum; selection frozen before S074 outcomes.",
        intervention="Only coefficient residual is optimized. Prototype, predicted box, target class, candidate score and decoder are fixed.",
        fit="Regularized convex logistic residual on balanced 640-grid own-vs-nonown pixels; deterministic half held out by class. Objective BCE + .01 logit residual RMS^2 + .0001 parameter L2, Newton/backtracking, max24 steps.",
        readout="Baseline, split-fit and all-pixel-fit masks use same p640 bilinear logits, >0 threshold, original crop support. IoU is in letterboxed 640-grid against nearest-exact COCO raster, diagnostic only; no AP.",
        interpretation="All-pixel gain is a conditional existence opportunity. Held-out AUC/gain tests whether the fixed prototype correction generalizes across pixels. Neither establishes a deployable method or unique coefficient causality.",
        runtime=dict(torch=torch.__version__, cuda=torch.cuda.is_available(), device=str(device)))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "COMPLETE.json").write_text(json.dumps(dict(status="COMPLETE", rows=len(rows), seconds=time.monotonic()-started), indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
