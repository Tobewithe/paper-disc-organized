"""S018: per-instance coefficient oracle under the installed official mask BCE.

Frozen P, original prediction box and source; only current own-GT-positive,
no-good-mask-candidate failures. Explicit S017 COCO raster recipe, not historical
pretraining reconstruction. No shared network/head training and no method AP.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse, contextlib, io, json, time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from candidate_lineage_probe import ROOT, read, need, save_csv, write_json, sha


def solve(proto, c0, gtmask, box160, area, max_iter):
    support = ops.crop_mask(torch.ones_like(gtmask), box160)[0].bool()
    if not bool(support.any()):
        value = float(v8SegmentationLoss.single_mask_loss(gtmask, c0[None], proto, box160, area))
        need(value == 0., 'Empty crop has nonzero mask loss')
        return c0.clone(), dict(official_loss_before=value, official_loss_after=value,
            scaled_gradient_max=0., iterations=0, evaluations=0, coefficient_norm=float(c0.norm()),
            solve_status='empty_GT_crop_kept_unchanged'), torch.zeros_like(c0)
    scale = proto[:, support].square().mean(1).sqrt().clamp_min(.01)
    w = torch.nn.Parameter((c0 * scale).clone())
    def objective():
        return v8SegmentationLoss.single_mask_loss(gtmask, (w / scale)[None], proto, box160, area)
    before = objective()
    first_grad = torch.autograd.grad(before, w)[0] * scale
    optimizer = torch.optim.LBFGS([w], lr=1., max_iter=max_iter, max_eval=max_iter * 2,
        tolerance_grad=1e-6, tolerance_change=1e-9, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = objective()
        need(bool(torch.isfinite(loss)), 'Nonfinite official-loss solve')
        loss.backward()
        return loss
    optimizer.step(closure)
    after = objective()
    grad = torch.autograd.grad(after, w)[0]
    need(float(after) <= float(before) + 1e-6, 'Official BCE increased')
    c = (w / scale).detach()
    state = optimizer.state[w]
    return c, dict(official_loss_before=float(before.detach()), official_loss_after=float(after.detach()),
        scaled_gradient_max=float(grad.abs().max()), iterations=int(state['n_iter']),
        evaluations=int(state['func_evals']), coefficient_norm=float(c.norm()), solve_status='optimized'), first_grad.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--images', type=int, default=0)
    p.add_argument('--max-iter', type=int, default=120)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'images').mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assignment = ROOT / 'diagnostics/assignment300_20260912'
    high = ROOT / 'diagnostics/no_candidate317_20260912'
    other = ROOT / 'diagnostics/low_density_readout1021_20260912'
    s017 = {int(r['annotation_id']): r for r in read(assignment / 'targets.csv')}
    selected = []
    for density, source in [('high', high), ('other', other)]:
        for r in read(source / 'targets.csv'):
            if r['fit_status'] == 'ok' and s017[int(r['annotation_id'])]['final_source_assignment'] == 'own_gt_positive':
                selected.append(dict(r, density=density))
    ids = sorted({int(r['image_id']) for r in selected})
    if a.images:
        ids = ids[:a.images]
    selected = [r for r in selected if int(r['image_id']) in ids]
    write_json(a.out / 'protocol.json', dict(network_training=False, oracle=True, images=ids,
        target_ids=[int(r['annotation_id']) for r in selected], script_sha256=sha(__file__),
        input_hashes={str(s / f): sha(s / f) for s, f in [(assignment, 'targets.csv'), (high, 'targets.csv'), (other, 'targets.csv')]},
        arms=['original', 'official_BCE160_GTbox', 'previous_fullinput_oracle'],
        selection='S015/S016 fit-eligible fixed-final failures whose raw source is own-GT-positive in S017. Both density groups, all eligible, no success-based selection.',
        objective='Installed v8SegmentationLoss.single_mask_loss, GT160 nearest raster and GT box crop/area exactly as S017. No Dice, bias, added regularization, or IoU checkpoint selection.',
        optimizer=dict(name='LBFGS', max_iter=a.max_iter, line_search='strong_wolfe', initialization='original coefficient', parameter_scaling='RMS within GT crop; no centering'),
        scope='Same-image GT-assisted coefficient fitting, frozen P/box, exact current source. Official segmentation objective with explicit COCO nonoverlap raster recipe; not historical training data or shared-head optimization. A finite solve is not a capacity upper bound.',
        ops_sha256=sha(ops.__file__), loss_source_sha256=sha(__import__('ultralytics.utils.loss', fromlist=['']).__file__),
        annotation_sha256=sha(ROOT / 'data/annotations/instances_val2017.json')))
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ROOT / 'data/annotations/instances_val2017.json'))
    rows = []
    start = time.monotonic()
    for num, iid in enumerate(ids, 1):
        cachepath = ROOT / 'diagnostics/full_val_cache_20260911/val' / f'{iid}.npz'
        with np.load(cachepath) as z:
            cache = {k: z[k] for k in z.files}
        with np.load(assignment / 'images' / f'{iid}.npz') as z:
            supervised = {k: z[k] for k in z.files}
        proto = torch.tensor(cache['proto'], device='cuda')
        gtindex = {int(aid): g for g, aid in enumerate(supervised['annotation_ids'])}
        parameters = {}
        hashes = dict(cache=sha(cachepath), supervision=sha(assignment / 'images' / f'{iid}.npz'))
        for density, source in [('high', high), ('other', other)]:
            rr = [r for r in selected if int(r['image_id']) == iid and r['density'] == density]
            if not rr:
                continue
            previous = json.loads((source / 'images' / f'{iid}.json').read_text())
            need(previous['input_hashes']['cache'] == hashes['cache'], 'Previous oracle cache mismatch')
            with np.load(source / 'images' / f'{iid}.npz') as z:
                old = {k: z[k] for k in z.files}
            hashes[density + '_oracle'] = sha(source / 'images' / f'{iid}.npz')
            for r in rr:
                aid = int(r['annotation_id'])
                g = gtindex[aid]
                src = int(r['source_index'])
                need(bool(supervised['foreground'][src]) and int(supervised['target_gt_index'][src]) == g, 'Not own-GT-positive')
                c0 = torch.tensor(old[f'{aid}_original_coefficient'], device='cuda')
                box = torch.tensor(old[f'{aid}_box'], device='cuda')
                gtmask = torch.tensor(supervised['gt_masks160'][g:g+1], device='cuda').float()
                normalized = torch.tensor(supervised['gt_boxes_normalized'][g:g+1], device='cuda')
                box160 = ops.xywh2xyxy(normalized) * 160
                area = normalized[:, 2:].prod(1)
                c, details, initial_gradient = solve(proto, c0, gtmask, box160, area, a.max_iter)
                recorded_grad = torch.tensor(supervised['coefficient_gradient'][src], device='cuda')
                if bool(initial_gradient.any()):
                    cosine = float(torch.nn.functional.cosine_similarity(initial_gradient, recorded_grad, dim=0))
                    need(cosine > .9999, f'Official gradient direction mismatch: {cosine}')
                else:
                    need(not bool(recorded_grad.any()), 'Zero initial gradient disagrees with full criterion')
                    cosine = None
                previous_c = torch.tensor(old[f'{aid}_free_coefficient_coefficient'], device='cuda')
                previous_b = float(old[f'{aid}_free_coefficient_bias'])
                need(previous_b == 0., 'Unexpected bias in coefficient-only oracle')
                details['previous_oracle_official_loss'] = float(v8SegmentationLoss.single_mask_loss(gtmask, previous_c[None], proto, box160, area))
                row = dict(image_id=iid, annotation_id=aid, density=density, ici=float(r['ici']), category_id=int(r['category_id']),
                    area=float(r['area']), source_index=src, gradient_direction_cosine=cosine, **details)
                truth = gt.annToRLE(gt.anns[aid])
                for arm, cc in [('original', c0), ('official_BCE160_GTbox', c), ('previous_fullinput_oracle', previous_c)]:
                    binary = ops.process_mask(proto, cc[None], box, (640, 640), upsample=True)[0]
                    original = (ops.scale_masks(binary[None, None], tuple(map(int, cache['shape'])))[0, 0] > .5).cpu().numpy()
                    rle = mu.encode(np.asfortranarray(original.astype(np.uint8)))
                    row[arm + '_coco_iou'] = float(mu.iou([rle], [truth], [0])[0, 0])
                need(abs(row['original_coco_iou'] - float(r['anchor_official_mask_iou'])) < 1e-7, 'Original IoU replay mismatch')
                rows.append(row)
                parameters[f'{aid}_official_coefficient'] = c.cpu().numpy()
        np.savez_compressed(a.out / 'images' / f'{iid}.npz', **parameters)
        write_json(a.out / 'images' / f'{iid}.json', dict(image_id=iid, input_hashes=hashes,
            targets=[r for r in rows if r['image_id'] == iid]))
        if num % 10 == 0 or num == len(ids):
            progress = dict(images=num, total=len(ids), targets=len(rows), seconds=time.monotonic()-start)
            print(json.dumps(progress), flush=True)
            write_json(a.out / 'progress.json', progress)
    save_csv(a.out / 'targets.csv', rows)
    summary = []
    for density in ['high', 'other']:
        rr = [r for r in rows if r['density'] == density]
        if not rr:
            continue
        for arm in ['original', 'official_BCE160_GTbox', 'previous_fullinput_oracle']:
            summary.append(dict(density=density, arm=arm, n=len(rr), mean_coco_iou=100*float(np.mean([r[arm+'_coco_iou'] for r in rr])),
                recovered75=sum(r[arm+'_coco_iou'] >= .75 for r in rr), worse_than_original=sum(r[arm+'_coco_iou'] < r['original_coco_iou']-1e-6 for r in rr)))
    write_json(a.out / 'ANALYSIS.json', dict(groups=summary, csv_sha256=sha(a.out / 'targets.csv'),
        gradient_witness_min_cosine=min(r['gradient_direction_cosine'] for r in rows if r['gradient_direction_cosine'] is not None),
        empty_GT_crop_unchanged=sum(r['solve_status']=='empty_GT_crop_kept_unchanged' for r in rows),
        optimizer_hit_iteration_limit=sum(r['iterations'] >= a.max_iter for r in rows)))
    write_json(a.out / 'COMPLETE.json', dict(status='COMPLETE', network_training=False, oracle=True, images=len(ids), targets=len(rows),
        seconds=time.monotonic()-start, hashes={str(q.relative_to(a.out)):sha(q) for q in a.out.rglob('*') if q.is_file()}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
