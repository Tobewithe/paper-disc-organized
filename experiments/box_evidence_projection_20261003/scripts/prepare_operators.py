"""Frozen, GT-free prototype operators and native point features; laptop only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torchvision.ops import roi_align
from ultralytics import YOLO
from runtime_utils import setup, dump, load, read_image


def fingerprint(cfg):
    fields = {k: cfg[k] for k in ('cache', 'weights', 'projection_lambda', 'solver_jitter', 'roi_side')}
    fields['schema'] = 'box_evidence_full_response_v1'
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def build(x, native, cfg, device):
    n = len(x['raw_ids'])
    fs = [f[None].to(device) for f in x['F']]
    maps = []
    for branch, feature in zip(native, fs):
        for layer in list(branch.children())[:-1]:
            feature = layer(feature)
        maps.append(feature[0])
    ids = x['raw_ids'].to(device)
    h0 = torch.cat([h.flatten(1) for h in maps], 1).T[ids]
    c_check = torch.empty((n, 32), device=device)
    for level, branch in enumerate(native):
        positions = torch.where(x['levels'].to(device) == level)[0]
        if len(positions):
            c_check[positions] = branch[-1](h0[positions].T[None, :, None, :])[0, :, 0].T
    error = float((c_check.cpu() - x['c0']).abs().max()) if n else 0.
    torch.testing.assert_close(c_check.cpu(), x['c0'], atol=3e-5, rtol=3e-5)
    proto = x['proto'].to(device).float()
    boxes = x['boxes'].to(device).clone()
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, 640)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, 640)
    valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
    rois = torch.cat((boxes.new_zeros((n, 1)), boxes), 1)
    side = int(cfg['roi_side'])
    a = roi_align(proto[None], rois, (side, side), spatial_scale=proto.shape[-1]/640,
                  sampling_ratio=2, aligned=True).flatten(2).transpose(1, 2).cpu().double()
    flat = proto.flatten(1).T
    yy, xx = torch.meshgrid(torch.arange(proto.shape[-2], device=device),
                           torch.arange(proto.shape[-1], device=device), indexing='ij')
    xx = xx.flatten(); yy = yy.flatten()
    gram = torch.empty((n, 32, 32), dtype=torch.float64)
    counts = []; fallback = []
    for i, box in enumerate(boxes):
        scaled = box * box.new_tensor([proto.shape[-1]/640, proto.shape[-2]/640]*2)
        support = (xx >= scaled[0]) & (xx < scaled[2]) & (yy >= scaled[1]) & (yy < scaled[3])
        count = int(support.sum()); counts.append(count)
        if count:
            p = flat[support].double()
            gram[i] = (p.T @ p / count).cpu()
            fallback.append(False)
        else:
            # Tiny subpixel boxes have no native-grid sites. Keep the instance;
            # define its response norm using exactly the sampled ROI instead.
            gram[i] = a[i].T @ a[i] / (side*side)
            fallback.append(True)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
    epsilon = float(cfg['solver_jitter']) * scale
    m = a.transpose(1,2) @ a / (side*side) + float(cfg['projection_lambda'])*gram
    m = m + epsilon[:, None, None] * torch.eye(32, dtype=torch.float64)
    rhs = a.transpose(1,2)/(side*side)
    k = torch.linalg.solve(m, rhs)
    residual = (m @ k - rhs).flatten(1).norm(dim=1) / rhs.flatten(1).norm(dim=1).clamp_min(1e-15)
    if not torch.isfinite(k).all() or float(residual.max()) > 1e-8:
        raise ArithmeticError('Projection solve failed its predeclared FP64 residual tolerance')
    k32 = k.float()
    cast_residual = (m @ k32.double()-rhs).flatten(1).norm(dim=1)/rhs.flatten(1).norm(dim=1).clamp_min(1e-15)
    if not torch.isfinite(cast_residual).all() or float(cast_residual.max()) > 5e-3:
        raise ArithmeticError('FP32 operator cast is too inaccurate; stop instead of silently changing solver')
    return dict(schema='box_evidence_full_response_v1', fingerprint=fingerprint(cfg),
                image_id=x['image_id'], source_fingerprint=x['fingerprint'], raw_ids=x['raw_ids'],
                h0=h0.cpu(), A=a.float(), K=k32, G=gram.float(), valid=valid.cpu(),
                support_count=torch.tensor(counts), tiny_box_roi_fallback=torch.tensor(fallback),
                epsilon=epsilon, fp64_relative_residual=residual, fp32_relative_residual=cast_residual,
                original_coefficient_max_error=error, feature_channels=[int(f.shape[0]) for f in x['F']])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));setup(cfg['seed'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    op=Path(cfg['operator_cache']);(op/'images').mkdir(parents=True,exist_ok=True)
    index=json.loads((Path(cfg['cache'])/'INDEX.json').read_text(encoding='utf-8-sig'))
    source=YOLO(cfg['weights']).model.cuda().eval().requires_grad_(False)
    native=source.model[-1].one2one_cv4
    started=time.monotonic();new=0;reused=0;candidate_count=0;max_error=0.;max_residual=0.;max_cast=0.;fallback_count=0
    cache_bytes=sum(p.stat().st_size for p in (op/'images').glob('*.pt'))
    for split,items in index.items():
        positives=[r for r in items if r['n']]
        if args.smoke: positives=positives[:8 if split=='fit' else 3]
        for j,item in enumerate(positives):
            path=op/'images'/f"{item['image_id']:012d}.pt"
            if path.exists():
                cached=load(path)
                if cached['fingerprint']!=fingerprint(cfg):raise RuntimeError('Operator config mismatch')
                reused+=1
            else:
                x=read_image(cfg,item['image_id']);cached=build(x,native,cfg,'cuda')
                tmp=path.with_suffix('.pt.tmp');torch.save(cached,tmp);tmp.replace(path);new+=1
                cache_bytes+=path.stat().st_size
                if cache_bytes>20*1024**3:raise RuntimeError('Fixed 20 GiB incremental cache budget exceeded')
            candidate_count+=int(item['n'])
            max_error=max(max_error,cached['original_coefficient_max_error'])
            max_residual=max(max_residual,float(cached['fp64_relative_residual'].max()))
            max_cast=max(max_cast,float(cached['fp32_relative_residual'].max()))
            fallback_count+=int(cached['tiny_box_roi_fallback'].sum())
            if j%25==0 or j+1==len(positives):
                progress=dict(stage='operators',split=split,images=j+1,total=len(positives),
                              new_images=new,reused_images=reused,candidates=candidate_count,
                              cache_bytes=cache_bytes,elapsed_s=time.monotonic()-started)
                dump(out/'PROGRESS.json',progress); print(json.dumps(progress),flush=True)
    report=dict(completed=True,smoke=args.smoke,new_images=new,reused_images=reused,
                candidates=candidate_count,original_coefficient_max_error=max_error,
                max_fp64_relative_residual=max_residual,max_fp32_relative_residual=max_cast,
                tiny_box_roi_fallback_candidates=fallback_count,elapsed_s=time.monotonic()-started,
                gt_fields_used=False,fingerprint=fingerprint(cfg))
    dump(out/'OPERATOR_AUDIT.json',report);dump(out/'COMPLETE.json',report)
    if not args.smoke:dump(op/'COMPLETE.json',report)


if __name__=='__main__':main()
