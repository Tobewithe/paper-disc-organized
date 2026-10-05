"""Fixed original COCO attribution IoU, not official AP or GT-filtered Recall."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from ultralytics.utils import ops

from readout_input_probe import ARMS, Readout, inputs, objective, sha, write_json


def ici(ann, ordinary):
    x, y, w, h = ann['bbox']
    total = 0.
    for other in ordinary:
        if other['id'] == ann['id'] or other['category_id'] != ann['category_id']:
            continue
        u, v, a, b = other['bbox']
        total += max(0., min(x+w, u+a)-max(x, u)) * max(0., min(y+h, v+b)-max(y, v))
    return total / max(w*h, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--trained', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    for directory in [args.cache, args.trained]:
        receipt = json.loads((directory / 'COMPLETE.json').read_text())
        for name, digest in receipt['hashes'].items():
            if sha(directory / name) != digest:
                raise RuntimeError(f'Source changed: {directory / name}')
    protocol = json.loads((args.cache / 'protocol.json').read_text())
    config = protocol['config']
    annotation = Path(protocol.get('data_root',str(root/'data'))) / 'annotations/instances_train2017.json'
    if sha(annotation) != protocol['annotation_sha256']:
        raise RuntimeError('COCO annotation changed')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
    selection = json.loads((args.cache / 'selection.json').read_text())
    norm0 = torch.load(args.trained / 'normalizer.pt', weights_only=True)
    norm = (norm0['mean'].cuda(), norm0['std'].cuda())
    channels = len(norm[0])
    runs = [('original', -1, None)]
    for seed in config['seeds']:
        for mode in ARMS:
            model = Readout(channels, linear=mode == 'linear').cuda()
            checkpoint = args.trained / f'{mode}_s{seed}/checkpoints/epoch{config["epochs"]:03d}.pt'
            model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['model'])
            model.eval()
            runs.append((mode, seed, model))
    for ridge in ([1e-4, 1e-6] if config.get('run_convex',True) else []):
        mode = f'convex_ridge_{ridge:.0e}'
        model = Readout(channels, linear=True, dtype=torch.float64).cuda()
        model.load_state_dict(torch.load(args.trained / f'{mode}.pt', map_location='cuda', weights_only=True)['model'])
        model.eval()
        runs.append((mode, -1, model))
    records = []
    raw_dir = args.out / 'coefficients'
    raw_dir.mkdir()
    start = time.monotonic()
    for split in config.get('eval_splits',['fit','transfer']):
        for number, iid in enumerate(selection[split], 1):
            with np.load(args.cache / 'images' / f'{iid}.npz') as q:
                item = {key: q[key] for key in q.files}
            h = torch.tensor(item['h'], device='cuda')
            region = torch.tensor(item['region'], device='cuda')
            level = torch.tensor(item['level'], device='cuda', dtype=torch.long)
            identities = [(iid, int(src)) for src in item['source_index']]
            c0 = torch.tensor(item['coeff'], device='cuda')
            proto = torch.tensor(item['proto'], device='cuda')
            boxes = torch.tensor(item['boxes'], device='cuda')
            tids, predidx = item['annotation_ids'], item['prediction_indices']
            ordinary = [a for a in gt.imgToAnns[iid] if not a.get('iscrowd', 0)]
            allmasks = {a['id']: gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]}
            crowd = np.zeros(tuple(item['shape']), dtype=bool)
            union = np.zeros_like(crowd)
            for a in gt.imgToAnns[iid]:
                if a.get('iscrowd', 0):
                    crowd |= allmasks[a['id']]
                else:
                    union |= allmasks[a['id']]
            saved = {}
            with torch.inference_mode():
                for mode, seed, model in runs:
                    draws = [0, 1, 2] if mode == 'shuffled' else [0]
                    for draw in draws:
                        if model is None:
                            coeff = c0
                        elif model.linear:
                            x = (h - norm[0]) / norm[1]
                            dtype = next(model.parameters()).dtype
                            coeff = c0 + model(x.to(dtype), level).float()
                        else:
                            x = inputs(h, region, level, mode, identities, seed, 1000000+draw, norm)
                            coeff = c0 + model(x, level)
                        saved[f'{mode}_s{seed}_d{draw}'] = coeff.cpu().numpy()
                        # The function above operates on EVERY original prediction.
                        # Only now use fixed GT attribution for diagnostic metrics.
                        if not len(tids):
                            continue
                        decoded = ops.process_mask(proto, coeff[predidx], boxes[predidx], (640, 640), upsample=True)
                        px = torch.tensor(item['sample_p'], device='cuda')
                        yy = torch.tensor(item['sample_y'], device='cuda', dtype=torch.float32)
                        factor = torch.tensor(item['loss_factor'], device='cuda')
                        losses = []
                        for k in range(len(tids)):
                            losses.append(float(objective(coeff[predidx[k:k+1]]-c0[predidx[k:k+1]],
                                          c0[predidx[k:k+1]], px[k:k+1], yy[k:k+1], factor[k:k+1])))
                        for k, aid0 in enumerate(tids):
                            aid = int(aid0)
                            ann = gt.anns[aid]
                            pred = (ops.scale_masks(decoded[k:k+1, None], tuple(item['shape']))[0, 0] > .5).cpu().numpy()
                            encoded = mask_utils.encode(np.asfortranarray(pred.astype(np.uint8)))
                            iou = float(mask_utils.iou([encoded], [gt.annToRLE(ann)], [0])[0, 0])
                            own = allmasks[aid] & ~crowd
                            valid_pred = pred & ~crowd
                            same = np.zeros_like(own)
                            for other in ordinary:
                                if other['category_id'] == ann['category_id'] and other['id'] != aid:
                                    same |= allmasks[other['id']]
                            area = int(own.sum())
                            ownhit = int((valid_pred & own).sum())
                            neighbor = int((valid_pred & same & ~own).sum())
                            background = int((valid_pred & ~union).sum())
                            density = ici(ann, ordinary)
                            records.append(dict(split=split, image_id=iid, annotation_id=aid, prediction_index=int(predidx[k]),
                                mode=mode, seed=seed, draw=draw, coco_iou=iou, sampled_native_bce=losses[k],
                                category_id=ann['category_id'], area=ann['area'], ici_same=density,
                                density='high' if density > .5+1e-10 else 'other',
                                valid_gt_area=area, coverage=ownhit/area if area else None,
                                same_neighbor=neighbor/area if area else None,
                                background=background/area if area else None))
            np.savez_compressed(raw_dir / f'{iid}.npz', **saved)
            if number % 8 == 0 or number == len(selection[split]):
                print(json.dumps(dict(stage='eval', split=split, images=number, total=len(selection[split]),
                                     seconds=time.monotonic()-start)), flush=True)
    if records:
        with (args.out / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    summary = []
    modes = list(dict.fromkeys(r['mode'] for r in records))
    for split in ['fit', 'transfer']:
        for group in ['all', 'high', 'other']:
            for mode in modes:
                rr = [r for r in records if r['split']==split and r['mode']==mode and (group=='all' or r['density']==group)]
                if not rr:
                    continue
                ids = sorted({r['annotation_id'] for r in rr})
                # First average draws/seeds per target; avoid treating predictions as independent GTs.
                values = {aid:np.mean([r['coco_iou'] for r in rr if r['annotation_id']==aid]) for aid in ids}
                summary.append(dict(split=split, density=group, mode=mode, targets=len(ids),
                                    images=len({r['image_id'] for r in rr}), mean_coco_iou=float(np.mean(list(values.values())))))
    write_json(args.out / 'SUMMARY.json', summary)
    write_json(args.out / 'protocol.json', dict(scope='Fixed original bbox50-attributed target IoU, not AP/Recall. '
               'All ordinary GT status retained in source gt_status.csv. Fit and transfer are small train2017 cohorts; '
               'shuffled draw metrics averaged, no prediction ensemble. Real raw-COCO RLE evaluation.',
               cache_sha256=sha(args.cache/'COMPLETE.json'), trained_sha256=sha(args.trained/'COMPLETE.json'),
               source_sha256=sha(__file__), config=config))
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(records), seconds=time.monotonic()-start,
               hashes={str(p.relative_to(args.out)):sha(p) for p in args.out.rglob('*') if p.is_file()}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
