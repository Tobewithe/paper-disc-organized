"""Train-only calibration then locked evaluation of frozen mask reranking."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as maskutils
from ultralytics import YOLO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT, Capture, ownership, sha, write_json, write_csv
from relative_ownership_decoder import decode

FIELDS = ['coverage', 'same_neighbor', 'neighbor', 'background', 'mask_iou',
          'exclusive_same_neighbor', 'gt_overlap_recall', 'predicted_area_over_gt']


def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def rank(iid, split):
    return hashlib.sha256(f'relative-ownership:20260911:{split}:{iid}'.encode()).hexdigest()


def selection(config):
    old = set(json.loads((ROOT / 'diagnostics/coefficient_pilot_cache_v4_20260911/selection.json').read_text())['train'])
    witness = set(json.loads((ROOT / 'diagnostics/structure_train_witness32_v2_20260911/protocol.json').read_text())['images'])
    available = {int(p.stem) for p in (ROOT / 'data/images/train2017').glob('*.jpg')}
    pools = {True: [], False: []}
    for r in read(ROOT / 'census/train2017_images.csv'):
        iid = int(r['image_id'])
        if iid in available and iid not in old | witness and int(r['n_instances']) > 0:
            pools[int(r['n_high_same']) > 0].append(iid)
    n = config['development_images']
    pool = pools[True] + pools[False]
    assert len(pool) >= n
    train = sorted(pool, key=lambda x:rank(x, 'train'))[:n]
    prior = set(json.loads((ROOT / 'diagnostics/structure_main300_20260911/protocol.json').read_text())['images'])
    allval = {int(p.stem) for p in (ROOT / 'diagnostics/full_val_cache_20260911/val').glob('*.npz')}
    val = sorted(allval - prior, key=lambda x: rank(x, 'val'))[:config['evaluation_images']]
    assert len(val) == config['evaluation_images'] and not set(train) & set(val)
    return dict(development=sorted(train), evaluation=sorted(val), available_train=len(available),
                excluded_train=len(old | witness), excluded_val_structure=len(prior),
                eligible_train_high_images=len(pools[True]), eligible_train_other_images=len(pools[False]))


@torch.inference_mode()
def spatial_and_predictions(gt, iid, item, masks, metadata, save_predictions):
    shape = tuple(map(int, item['shape']))
    det = item['detections']
    mapping = dict(zip(map(int, item['mapping_gt']), map(int, item['mapping_pred'])))
    assert len(mapping.values()) == len(set(mapping.values()))
    aa = gt.imgToAnns[iid]
    raster = {a['id']: torch.tensor(gt.annToMask(a).astype(bool), device='cuda') for a in aa}
    ordinary = [a for a in aa if not a.get('iscrowd', 0)]
    count = sum((raster[a['id']].to(torch.int16) for a in ordinary), torch.zeros(shape, dtype=torch.int16, device='cuda'))
    union = count > 0
    crowd = torch.zeros(shape, dtype=torch.bool, device='cuda')
    for a in aa:
        if a.get('iscrowd', 0): crowd |= raster[a['id']]
    valid = ~crowd
    same = {}
    for cat in {a['category_id'] for a in ordinary}:
        same[cat] = torch.stack([raster[a['id']] for a in ordinary if a['category_id'] == cat]).any(0)
    regions = {}
    skipped = []
    for aid, j in mapping.items():
        own = raster[aid] & valid
        area = int(own.sum())
        if not area:
            skipped.append(dict(image_id=iid, annotation_id=aid, reason='no_valid_pixels'))
            continue
        neighbor = union & ~own & valid
        sn = same[gt.anns[aid]['category_id']] & ~own & valid
        overlap = own & (count > 1)
        regions[j] = (aid, area, int(overlap.sum()), torch.stack([
            own, sn, neighbor, ~union & valid, sn & (count == 1), overlap]).flatten(1).float())
    categories = sorted(gt.cats)
    rows = []
    predictions = {}
    for arm, binary in masks.items():
        preds = []
        for first in range(0, len(binary), 32):
            chunk = binary[first:first+32]
            orig = ops.scale_masks(chunk[:, None], shape)[:, 0] > .5
            nonempty = chunk.flatten(1).any(1)
            for k in range(len(chunk)):
                j = first + k
                if save_predictions and bool(nonempty[k]):
                    rle = maskutils.encode(np.asfortranarray(orig[k].cpu().numpy().astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    preds.append(dict(image_id=iid, category_id=categories[int(det[j, 5])],
                                      score=float(det[j, 4]), segmentation=rle))
                if j in regions:
                    aid, area, overlap_area, reg = regions[j]
                    counts = (orig[k].flatten().float() @ reg.T).double().cpu().tolist()
                    tp, sn, near, bg, exclusive, overlap_tp = counts
                    values = [tp/area, sn/area, near/area, bg/area,
                              tp/(area+near+bg), exclusive/area,
                              overlap_tp/overlap_area if overlap_area else 0.,
                              (tp+near+bg)/area]
                    rows.append(dict(arm=arm, image_id=iid, target_annotation=aid,
                                     target_ici=float(metadata[aid]['ici_same']),
                                     category_id=gt.anns[aid]['category_id'], gt_area=gt.anns[aid]['area'],
                                     gt_overlap_area=overlap_area, **dict(zip(FIELDS, values))))
        predictions[arm] = preds
    return rows, predictions, skipped


def cohort_summary(rows):
    result = []
    for arm in sorted({r['arm'] for r in rows}):
        for group in ['all', 'high', 'low']:
            rr = [r for r in rows if r['arm'] == arm and
                  (group == 'all' or (float(r['target_ici']) > .5+1e-10) == (group == 'high'))]
            if rr:
                result.append(dict(arm=arm, group=group, targets=len(rr),
                                   **{m:float(np.mean([float(r[m]) for r in rr])) for m in FIELDS}))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase', choices=['development', 'evaluation'], required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = args.out
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = Path(__file__).with_name('relative_ownership_protocol.json')
    config = json.loads(config_path.read_text())
    if args.smoke:
        config['development_images'] = 6
        config['evaluation_images'] = 8
    if args.phase == 'development':
        out.mkdir(exist_ok=False, parents=True)
        (out / 'train_cache').mkdir()
        chosen = selection(config)
        write_json(out/'selection.json', chosen)
        write_json(out/'protocol.json', dict(config=config, selection_sha256=sha(out/'selection.json'),
             script_sha256=sha(__file__), decoder_sha256=sha(Path(__file__).with_name('relative_ownership_decoder.py')),
             protocol_source_sha256=sha(config_path), weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),
             annotation_hashes={s:sha(ROOT/f'data/annotations/instances_{s}2017.json') for s in ['train','val']},
             metadata_hashes={s:sha(ROOT/('census/train2017_instances.csv' if s=='train' else 'census/COCO_EVAL_INSTANCE_MANIFEST.csv')) for s in ['train','val']}))
        arms = {'initial': dict(kind='initial', value=0)}
        for kind in ['relative', 'geometry']:
            for alpha in config['alpha_grid']:
                arms[f'{kind}_{alpha}'] = dict(kind=kind, value=alpha)
        for tau in [-.5, .25, .5, 1.]:
            arms[f'threshold_{tau}'] = dict(kind='threshold', value=tau)
        split = 'train'
        ids = chosen['development']
        model = YOLO(str(ROOT/'weights/yolo26m-seg.pt'))
        model.model.eval().requires_grad_(False)
    else:
        protocol = json.loads((out/'protocol.json').read_text())
        assert protocol['script_sha256'] == sha(__file__)
        assert protocol['decoder_sha256'] == sha(Path(__file__).with_name('relative_ownership_decoder.py'))
        assert protocol['protocol_source_sha256'] == sha(config_path)
        lock = json.loads((out/'LOCKED_SETTINGS.json').read_text())
        config = protocol['config']
        chosen = json.loads((out/'selection.json').read_text())
        ids = chosen['evaluation']
        split = 'val'
        arms = lock['arms']
        assert not (out/'EVALUATION_COMPLETE.json').exists()
        (out/'predictions_by_image').mkdir(exist_ok=False)
    ann = ROOT/f'data/annotations/instances_{split}2017.json'
    with contextlib.redirect_stdout(io.StringIO()): gt = COCO(str(ann))
    mp = ROOT/('census/train2017_instances.csv' if split == 'train' else 'census/COCO_EVAL_INSTANCE_MANIFEST.csv')
    metadata = {int(r['annotation_id']):r for r in read(mp)}
    allrows, corrections, skipped, coverage = [], [], [], []
    hashes = {}
    start = time.monotonic()
    for number, iid in enumerate(ids, 1):
        if split == 'train':
            with torch.inference_mode():
                model.predict(str(ROOT/'data/images/train2017'/gt.imgs[iid]['file_name']), predictor=Capture,
                              imgsz=640, conf=.001, iou=.7, max_det=300, rect=False, half=False,
                              retina_masks=False, device=0, verbose=False)
                cap = model.predictor.capture
                mapping = ownership(gt, iid, cap['detections'])
                item = {key:val.cpu().numpy() if isinstance(val, torch.Tensor) else val for key,val in cap.items()}
                item.update(mapping_gt=np.array(list(mapping), dtype=np.int64),
                            mapping_pred=np.array(list(mapping.values()), dtype=np.int64))
                path = out/'train_cache'/f'{iid}.npz'
                np.savez_compressed(path, **item)
        else:
            path = ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz'
            with np.load(path) as q:
                item = {k:q[k] for k in ['proto','coeff','boxes','detections','shape','input_shape','mapping_gt','mapping_pred']}
        hashes[str(path)] = sha(path)
        # Deliberately filter model inputs; decoder cannot see GT or attribution.
        inputs = {k:item[k] for k in ['proto','coeff','boxes','detections','input_shape']}
        masks, changes, xor = decode(inputs, arms, iid, config['eligibility'])
        assert xor == 0
        rows, preds, ss = spatial_and_predictions(gt, iid, item, masks, metadata, split=='val')
        allrows.extend(rows); corrections.extend(changes); skipped.extend(ss)
        coverage.append(dict(image_id=iid, gt=sum(not a.get('iscrowd',0) for a in gt.imgToAnns[iid]),
                             matched=len(item['mapping_gt']), detections=len(item['coeff']), official_replay_xor=xor))
        if split == 'val':
            for arm, pp in preds.items():
                directory = out/'predictions_by_image'/arm
                directory.mkdir(exist_ok=True)
                with gzip.open(directory/f'{iid}.json.gz', 'wt', encoding='utf-8') as f:
                    json.dump(pp, f, separators=(',',':'))
        del masks
        if number % 10 == 0 or number == len(ids):
            progress = dict(phase=args.phase, completed=number, total=len(ids), seconds=round(time.monotonic()-start,1))
            write_json(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    prefix = args.phase
    write_csv(out/f'{prefix}_spatial.csv', allrows)
    write_csv(out/f'{prefix}_corrections.csv', corrections)
    write_csv(out/f'{prefix}_coverage.csv', coverage)
    write_json(out/f'{prefix}_skipped.json', skipped)
    write_json(out/f'{prefix}_cache_hashes.json', hashes)
    summary = cohort_summary(allrows)
    write_csv(out/f'{prefix}_summary.csv', summary)
    if split == 'train':
        high = {r['arm']:r for r in summary if r['group']=='high'}
        base = high['initial']
        selected = {}
        for kind in ['relative','geometry','threshold']:
            choices = [name for name in arms if name=='initial' or arms[name]['kind']==kind]
            eligible = [name for name in choices if high[name]['coverage'] >= base['coverage']-.0025]
            best = sorted(eligible, key=lambda name:(-high[name]['mask_iou'], abs(arms[name]['value']), name))[0]
            selected[kind] = dict(kind=kind, value=arms[best]['value'], selected_arm=best,
                                  development_high=high[best], eligible_arms=eligible)
        evalarms = {'initial':dict(kind='initial',value=0)}
        for kind,item in selected.items():evalarms[kind] = dict(kind=kind,value=item['value'])
        for seed in range(3):evalarms[f'random_s{seed}'] = dict(kind='random',value=selected['relative']['value'],seed=seed)
        write_json(out/'LOCKED_SETTINGS.json',dict(status='LOCKED_BEFORE_EVALUATION',
             selected=selected, arms=evalarms, development_spatial_sha256=sha(out/'development_spatial.csv'),
             selection_sha256=sha(out/'selection.json'), script_sha256=sha(__file__),
             decoder_sha256=sha(Path(__file__).with_name('relative_ownership_decoder.py'))))
        print(json.dumps(selected), flush=True)
    write_json(out/f'{prefix.upper()}_COMPLETE.json',dict(status='COMPLETE',training=False,images=len(ids),
        seconds=time.monotonic()-start, measured_replay_xor=sum(r['official_replay_xor'] for r in coverage),
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    main()
