from __future__ import annotations
import argparse, gzip, json, math, time
from pathlib import Path
import sys
sys.path.insert(0, '/root/autodl-tmp/prototype_guided_evidence_selection_20261003/py')
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

LAMBDA = 0.003

def load_asset(cache: Path, image_id: int):
    path = cache / 'images' / f'{int(image_id):012d}.pt.gz'
    return torch.load(gzip.open(path, 'rb'), map_location='cpu', weights_only=False)

def prepare_records(image, device):
    proto = image['proto'].float().to(device)
    up = F.interpolate(proto[None], (640, 640), mode='bilinear', align_corners=False)[0]
    masks = image['masks'].float().to(device)
    rows = []
    for k, row in enumerate(image['rows']):
        box = image['target_boxes'][k].float().to(device)
        support = ops.crop_mask(torch.ones((1, 640, 640), device=device), box[None])[0].bool()
        if not bool(support.any()):
            raise RuntimeError(f'empty support {image["image_id"]} {row["annotation_id"]}')
        p = up[:, support].T.contiguous().double()
        y = (masks[support] == int(image['owners'][k]) + 1).double()
        area = float(((box[2:] - box[:2]) / 640.0).prod() * 640.0 * 640.0)
        ident = {
            'image_id': int(image['image_id']),
            'annotation_id': int(row['annotation_id']),
            'raw_id': int(row['raw_id']),
            'pyramid_level': int(row.get('pyramid_level', row.get('level'))),
            'target_gt_idx': int(row.get('target_gt_idx', row.get('gt_index'))),
        }
        rows.append(dict(p=p, y=y, c=image['c0'][k].float().to(device).double(), area=area, ident=ident))
    return rows

def solve_batch(records, device, max_iter):
    n = len(records)
    delta = torch.zeros((n, 32), dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
    def value_grad():
        total = torch.zeros((), dtype=torch.float64, device=device)
        grad = torch.zeros_like(delta)
        for k, r in enumerate(records):
            z = r['p'] @ (r['c'] + delta[k])
            bce = float(r['seg_gain']) * (F.softplus(z) - r['y'] * z).sum() / r['area'] if 'seg_gain' in r else (F.softplus(z) - r['y'] * z).sum() / r['area']
            err = (torch.sigmoid(z) - r['y']) / r['area']
            total = total + bce + LAMBDA * delta[k].square().sum() / 2.0
            grad[k] = err @ r['p'] + LAMBDA * delta[k]
        return total / n, grad / n
    def closure():
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            value, grad = value_grad()
        delta.grad = grad
        return value
    opt.step(closure)
    with torch.no_grad():
        value, grad = value_grad()
    return delta.detach().float().cpu(), float(value.cpu()), float(grad.norm(dim=1).max().cpu()), int(opt.state[delta].get('n_iter', -1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-iter', type=int, default=100)
    ap.add_argument('--deadline', type=float, default=0.0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    split = json.loads(args.split.read_text())
    ids = [int(x) for x in split['fit']]
    deltas = []
    identities = []
    start = time.monotonic(); total_rows = 0; images = 0
    for iid in ids:
        if args.deadline and time.time() >= args.deadline:
            raise TimeoutError('oracle deadline reached')
        image = load_asset(args.cache, iid)
        raw_records = prepare_records(image, device)
        for r in raw_records: r['seg_gain'] = float(image['segmentation_gain'])
        for lo in range(0, len(raw_records), args.batch):
            part = raw_records[lo:lo + args.batch]
            d, v, g, it = solve_batch(part, device, args.max_iter)
            deltas.append(d)
            identities.extend([{**r['ident'], 'objective': None, 'gradient_norm': g, 'iterations': it, 'exit_reason': 'iteration_limit' if it >= args.max_iter else 'native_LBFGS_tolerance_or_line_search_termination'} for r in part])
            total_rows += len(part)
        images += 1
        if images == 1 or images % 32 == 0:
            print(json.dumps({'images': images, 'planned_images': len(ids), 'records': total_rows, 'elapsed_s': time.monotonic() - start}), flush=True)
    delta = torch.cat(deltas, 0)
    torch.save({'delta': delta, 'identities': identities, 'lambda': LAMBDA, 'source': 'fast-screen official one2one cache; full GT-box support', 'images': images}, args.out)
    summary = {'completed': True, 'images': images, 'records': total_rows, 'delta_shape': list(delta.shape), 'lambda': LAMBDA, 'device': str(device), 'max_iter': args.max_iter, 'elapsed_s': time.monotonic() - start, 'max_batch_gradient_norm': max(float(x['gradient_norm']) for x in identities) if identities else None}
    args.out.with_suffix('.SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()

