"""S076: can inference-visible features predict the GT-derived coefficient response?

This is a feasibility diagnostic.  GT is used only to produce per-instance
teacher residuals; the predictor is evaluated on image-disjoint instances and
never receives masks or neighbor identities.
"""
from pathlib import Path
import argparse, hashlib, json, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

from probe_fixed_prototype_split import ANN, MANIFEST, RUN, fit_delta, input_mask, iou_binary


def ridge_fit(x, y, lam):
    x1 = torch.cat([x, torch.ones((len(x), 1), device=x.device, dtype=x.dtype)], dim=1)
    eye = torch.eye(x1.shape[1], device=x.device, dtype=x.dtype)
    eye[-1, -1] = 0
    return torch.linalg.solve(x1.T @ x1 + lam * eye, x1.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pixels-per-class', type=int, default=256)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source').mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt = COCO(str(ANN))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows, feature_rows, labels = [], [], []
    started = time.monotonic()
    from ultralytics.utils import ops
    for number, item in enumerate(manifest, 1):
        aid, iid = int(item['annotation_id']), int(item['image_id'])
        cache = np.load(RUN / 'tensor_cache' / f'{iid}.npz')
        where = np.flatnonzero(cache['annotation_ids'].astype(np.int64) == aid)
        if len(where) != 1:
            continue
        k = int(where[0])
        proto = torch.as_tensor(cache['proto'], device=device, dtype=torch.float32)
        coeff = torch.as_tensor(cache['coeff'][k], device=device, dtype=torch.float32)
        box = torch.as_tensor(cache['boxes'][k], device=device, dtype=torch.float32)
        p640 = F.interpolate(proto[None], (640, 640), mode='bilinear', align_corners=False)[0].flatten(1).T.contiguous()
        zfull = p640 @ coeff
        support = ops.crop_mask(torch.ones((1, 640, 640), device=device, dtype=torch.uint8), box[None])[0].bool().cpu().numpy()
        own = input_mask(gt, iid, aid)
        crowd = np.zeros_like(own)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd', 0) or ann.get('ignore', 0):
                crowd |= input_mask(gt, iid, ann['id'])
        valid = support & ~crowd
        pos = np.flatnonzero(valid & own)
        neg = np.flatnonzero(valid & ~own)
        if len(pos) < 16 or len(neg) < 16:
            continue
        rng = np.random.default_rng(int(hashlib.sha256(f'S076:{aid}'.encode()).hexdigest()[:8], 16))
        n = min(args.pixels_per_class, len(pos), len(neg))
        pos = rng.choice(pos, n, replace=False)
        neg = rng.choice(neg, n, replace=False)
        ix = np.r_[pos, neg]
        y = torch.as_tensor(np.r_[np.ones(n), np.zeros(n)], device=device, dtype=torch.float32)
        xpix = p640[torch.as_tensor(ix, device=device)]
        zpix = zfull[torch.as_tensor(ix, device=device)]
        wall = fit_delta(xpix, zpix, y)
        base = (zfull > 0).reshape(640, 640).cpu().numpy() & support
        teacher = (zfull + p640 @ wall > 0).reshape(640, 640).cpu().numpy() & support
        feat = np.r_[cache['h'][0].astype(np.float32), cache['coeff'][k].astype(np.float32),
                     np.asarray(box.detach().cpu(), dtype=np.float32) / 640.0,
                     proto.mean((1, 2)).detach().cpu().numpy().astype(np.float32),
                     proto.std((1, 2), unbiased=False).detach().cpu().numpy().astype(np.float32)]
        feature_rows.append(feat)
        labels.append(wall.detach().cpu().numpy())
        rows.append(dict(annotation_id=aid, image_id=iid, density=item['group'].split(':', 1)[0],
                         residual=item['group'].split(':', 1)[1], baseline_iou=iou_binary(base, own),
                         teacher_iou=iou_binary(teacher, own), teacher_gain_points=100 * (iou_binary(teacher, own) - iou_binary(base, own)),
                         teacher_norm=float(wall.norm().cpu()), n_pixels=2 * n))
        if number % 24 == 0 or number == len(manifest):
            print(json.dumps(dict(done=number, total=len(manifest), rows=len(rows), seconds=round(time.monotonic() - started, 2))), flush=True)
    if len(rows) < args.folds:
        raise RuntimeError('Too few valid teacher targets')
    X = torch.as_tensor(np.stack(feature_rows), device=device, dtype=torch.float64)
    Y = torch.as_tensor(np.stack(labels), device=device, dtype=torch.float64)
    # Four inference-visible blocks: h, coefficient, box, and prototype stats.
    block_slices = {'h': slice(0, 64), 'coeff': slice(64, 96), 'box': slice(96, 100), 'proto_stats': slice(100, 164)}
    feature_sets = {
        'h': list(range(0, 64)),
        'h_coeff_box': list(range(0, 100)),
        'h_coeff_box_proto': list(range(0, 164)),
        'coeff_box': list(range(64, 100)),
    }
    # Image-disjoint deterministic folds.  There is one selected target per cache file,
    # but image_id remains the split unit in case the manifest changes.
    keys = np.asarray([int(r['image_id']) for r in rows], dtype=np.int64)
    fold = np.asarray([int(hashlib.sha256(f'S076-fold:{x}'.encode()).hexdigest()[:8], 16) % args.folds for x in keys])
    eval_rows = []
    for name, cols in feature_sets.items():
        mu = X[:, cols].mean(0)
        sd = X[:, cols].std(0, unbiased=False).clamp_min(1e-3)
        xn = (X[:, cols] - mu) / sd
        for lam in (1e-3, 1e-2, 1e-1):
            for f in range(args.folds):
                tr = torch.as_tensor(fold != f, device=device)
                te = torch.as_tensor(fold == f, device=device)
                if int(te.sum()) == 0 or int(tr.sum()) == 0:
                    continue
                w = ridge_fit(xn[tr], Y[tr], lam)
                pred = torch.cat([xn[te], torch.ones((int(te.sum()), 1), device=device, dtype=xn.dtype)], dim=1) @ w
                for j, row_i in enumerate(np.flatnonzero(fold == f)):
                    base = rows[row_i]['baseline_iou']
                    own = input_mask(gt, rows[row_i]['image_id'], rows[row_i]['annotation_id'])
                    cache = np.load(RUN / 'tensor_cache' / f"{rows[row_i]['image_id']}.npz")
                    k = int(np.flatnonzero(cache['annotation_ids'].astype(np.int64) == rows[row_i]['annotation_id'])[0])
                    proto = torch.as_tensor(cache['proto'], device=device, dtype=torch.float32)
                    c = torch.as_tensor(cache['coeff'][k], device=device, dtype=torch.float32)
                    b = torch.as_tensor(cache['boxes'][k], device=device, dtype=torch.float32)
                    p = F.interpolate(proto[None], (640, 640), mode='bilinear', align_corners=False)[0].flatten(1).T.contiguous()
                    support = ops.crop_mask(torch.ones((1, 640, 640), device=device, dtype=torch.uint8), b[None])[0].bool().cpu().numpy()
                    z = p @ c
                    pred32 = pred[j].float()
                    pmask = (z + p @ pred32 > 0).reshape(640, 640).cpu().numpy() & support
                    piou = iou_binary(pmask, own)
                    eval_rows.append(dict(feature_set=name, ridge=lam, fold=f, annotation_id=rows[row_i]['annotation_id'],
                                          image_id=rows[row_i]['image_id'], density=rows[row_i]['density'], residual=rows[row_i]['residual'],
                                          baseline_iou=base, teacher_iou=rows[row_i]['teacher_iou'], predicted_iou=piou,
                                          predicted_gain_points=100 * (piou - base), teacher_gain_points=rows[row_i]['teacher_gain_points'],
                                          delta_cosine=float(F.cosine_similarity(pred[j], Y[row_i], dim=0).cpu())))
    pd.DataFrame(rows).to_csv(out / 'teacher_targets.csv', index=False)
    pd.DataFrame(eval_rows).to_csv(out / 'heldout_predictions.csv', index=False)
    d = pd.DataFrame(eval_rows)
    summary = []
    for (name, lam), g in d.groupby(['feature_set', 'ridge']):
        summary.append(dict(feature_set=name, ridge=lam, n=len(g), predicted_gain_points=float(g.predicted_gain_points.mean()),
                            teacher_gain_points=float(g.teacher_gain_points.mean()), predicted_positive_rate=float((g.predicted_gain_points > 0).mean()),
                            delta_cosine=float(g.delta_cosine.mean()), high_predicted_gain_points=float(g[g.density == 'high'].predicted_gain_points.mean()) if (g.density == 'high').any() else float('nan'),
                            low_predicted_gain_points=float(g[g.density == 'low'].predicted_gain_points.mean()) if (g.density == 'low').any() else float('nan')))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    protocol = dict(experiment='S076_TEACHER_DELTA_PREDICTABILITY', training='ridge predictor only',
                    source='Frozen S070 tensor manifest; GT is used to fit per-instance teacher residuals.',
                    inputs='Inference-visible h, original coefficient, predicted box geometry, prototype channel mean/std.',
                    split=f'{args.folds} deterministic image-disjoint folds; no mask or neighbor identity input.',
                    teacher='Regularized convex coefficient residual on balanced own-vs-nonown pixels, same fit_delta as S074; no AP claim.',
                    evaluation='Fixed prototype, original predicted box and official crop; predicted coefficient residual decoded on held-out instances.',
                    limits='Small frozen diagnostic queue; positive result is feasibility, not full-task method evidence.',
                    runtime=dict(torch=torch.__version__, cuda=torch.cuda.is_available(), device=str(device), rows=len(rows)))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out / 'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE', rows=len(rows), evaluations=len(eval_rows), seconds=time.monotonic() - started), indent=2), encoding='utf-8')
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
