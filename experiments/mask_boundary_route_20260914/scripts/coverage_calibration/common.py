"""Shared, fixed-version data access for the RCMC mechanism diagnostics."""
from pathlib import Path
import csv
import json
import os
import sys
import time
import importlib.metadata as metadata


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def setup(protocol, output):
    p = json.loads(Path(protocol).read_text(encoding='utf-8-sig'))
    sys.path.insert(0, p['package_root'])
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    import torch
    import numpy as np
    import cv2
    import ultralytics
    actual = {'python': '.'.join(map(str, sys.version_info[:3])), 'ultralytics': ultralytics.__version__,
              'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
              'pycocotools': metadata.version('pycocotools'), 'scipy': metadata.version('scipy'),
              'scikit-learn': metadata.version('scikit-learn')}
    assert actual == p['environment'], (actual, p['environment'])
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    atomic(out/'environment.json', actual | {'ultralytics_file': ultralytics.__file__,
                                           'executable': sys.executable})
    return p, out


def load_slots(p):
    """One fixed original output slot per associated GT; no new matching."""
    decisions = {}
    with Path(p['decisions']).open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['mode'] == 'response':
                decisions[(int(r['image_id']), int(r['candidate_index']))] = r
    rows = []
    with (Path(p['bank'])/'instance_records.csv').open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['variant'] != 'smooth_gated':
                continue
            key = (int(r['image_id']), int(r['candidate_index']))
            d = decisions[key]
            assert int(r['annotation_id']) == int(d['annotation_id'])
            row = {k: int(r[k]) for k in ('image_id', 'candidate_index', 'annotation_id', 'category_id')}
            row.update({k: float(r[k]) for k in ('box_iou', 'gt_area', 'baseline_iou', 'baseline_recall', 'baseline_purity')})
            assert abs(row['baseline_iou'] - float(d['baseline_iou'])) < 1e-9
            row.update(calibrated=int(d['calibrated']), gated_iou=float(d['gated_iou']),
                       fixed_smooth_iou=float(d['fixed_smooth_iou']))
            good_box = row['box_iou'] >= .75
            high_cov = row['baseline_recall'] >= .95
            good_mask = row['baseline_iou'] >= .75
            row['group'] = ('target_failure' if good_box and high_cov and not good_mask else
                            'high_coverage_success' if good_box and high_cov and good_mask else
                            'other_success' if good_mask else
                            'undercoverage_failure' if not high_cov else 'other_failure')
            rows.append(row)
    lookup = {(r['image_id'], r['candidate_index']): r for r in rows}
    with (Path(p['bank'])/'candidate_records.csv').open(newline='', encoding='utf-8') as f:
        for c in csv.DictReader(f):
            row = lookup.get((int(c['image_id']), int(c['candidate_index'])))
            if row is not None:
                row['box'] = [float(c[k]) for k in ('box_x1','box_y1','box_x2','box_y2')]
                row['score'] = float(c['score'])
    assert len(rows) == 30426 and sum(r['group'] == 'target_failure' for r in rows) == 1701
    return rows


def load_predictions(path):
    data = json.loads(Path(path).read_text())
    mapped = {(r['image_id'], r['candidate_index']): r for r in data}
    assert len(data) == len(mapped)
    return data, mapped


def image_regions(coco, image_id):
    import numpy as np
    h, w = coco.imgs[image_id]['height'], coco.imgs[image_id]['width']
    ordinary = np.zeros((h,w), np.uint16)
    categories = {}
    crowd = np.zeros((h,w), bool)
    for ann in coco.imgToAnns[image_id]:
        mask = coco.annToMask(ann).astype(bool)
        if ann.get('iscrowd', 0) or ann.get('ignore', 0):
            crowd |= mask
        else:
            ordinary += mask
            if ann['category_id'] not in categories:
                categories[ann['category_id']] = np.zeros((h,w), np.uint16)
            categories[ann['category_id']] += mask
    return ordinary, categories, crowd


def gt_regions(coco, ann_id, regions):
    import numpy as np
    ordinary, cats, crowd = regions
    ann = coco.anns[ann_id]
    target = coco.annToMask(ann).astype(bool)
    same = (cats[ann['category_id']] - target) > 0
    different = (ordinary - cats[ann['category_id']]) > 0
    outside = ~target
    ambiguous = outside & crowd
    valid = outside & ~crowd
    # Mutually exclusive categories; own GT takes priority over all overlaps.
    labels = np.zeros(target.shape, np.uint8)
    labels[valid & same & ~different] = 1
    labels[valid & different & ~same] = 2
    labels[valid & same & different] = 3
    labels[ambiguous] = 4
    labels[target] = 5
    return target, labels


def mask_counts(mask, target, labels):
    import numpy as np
    counts = np.bincount(labels[mask].ravel(), minlength=6)
    tp = int(counts[5]); g = int(target.sum()); fp = int(counts[:5].sum())
    return {'tp': tp, 'fn': g-tp, 'fp': fp, 'gt_pixels': g,
            'background_fp': int(counts[0]), 'same_neighbor_fp': int(counts[1]),
            'different_neighbor_fp': int(counts[2]), 'mixed_neighbor_fp': int(counts[3]),
            'crowd_ignore_fp': int(counts[4]), 'iou': tp/max(g+fp,1),
            'coverage': tp/max(g,1), 'purity': tp/(tp+fp) if tp+fp else None}


def progress(out, stage, **values):
    status = {'stage': stage, **values}
    atomic(Path(out)/'progress.json', status)
    print(json.dumps(status), flush=True)


def choose_panel(rows, coco, p):
    """Round-robin category/size sampling, without looking at intervention outcomes."""
    import random
    import math
    from collections import defaultdict, Counter
    rng = random.Random(p['panel_seed'])
    strata = defaultdict(list)
    for row in rows:
        if row['group'] == 'target_failure':
            size = 'small' if row['gt_area'] < 1024 else 'medium' if row['gt_area'] < 9216 else 'large'
            strata[(row['category_id'], size)].append(row)
    for bucket in strata.values():
        rng.shuffle(bucket)
    keys = sorted(strata); rng.shuffle(keys)
    selected = []; counts = Counter()
    while len(selected) < p['panel_targets'] and any(strata.values()):
        for key in keys:
            while strata[key]:
                row = strata[key].pop()
                if counts[row['image_id']] < 2:
                    selected.append(dict(row, panel_role='target'))
                    counts[row['image_id']] += 1
                    break
            if len(selected) >= p['panel_targets']:
                break
    used = set(); controls = []; matches = []; unmatched = []
    pool = [r for r in rows if r['group'] == 'high_coverage_success']
    def vector(r):
        box = coco.anns[r['annotation_id']]['bbox']; im = coco.imgs[r['image_id']]
        return (math.log(max(r['gt_area'],1)), r['box_iou'],
                math.log(max(box[2],1)/max(box[3],1)), math.log(im['width']*im['height']))
    for r in selected:
        v = vector(r); options = []
        for c in pool:
            if c['category_id'] != r['category_id'] or c['annotation_id'] in used or counts[c['image_id']] >= 2:
                continue
            w = vector(c)
            if abs(w[0]-v[0]) > math.log(2) or abs(w[1]-v[1]) > .10 or abs(w[2]-v[2]) > math.log(2) or abs(w[3]-v[3]) > math.log(2):
                continue
            distance = ((w[0]-v[0])/math.log(2))**2+((w[1]-v[1])/.1)**2+((w[2]-v[2])/math.log(2))**2+((w[3]-v[3])/math.log(2))**2
            options.append((distance, c['annotation_id'], c))
        if not options:
            unmatched.append(r['annotation_id']); continue
        distance, _, c = min(options, key=lambda x:(x[0],x[1]))
        used.add(c['annotation_id']); counts[c['image_id']] += 1
        controls.append(dict(c, panel_role='control'))
        matches.append({'target_ann':r['annotation_id'],'control_ann':c['annotation_id'],'covariate_distance':distance})
    panel = selected + controls
    images = sorted({r['image_id'] for r in panel}); rng.shuffle(images)
    exploratory = set(images[:len(images)//2])
    for r in panel:
        r['panel_partition'] = 'exploratory' if r['image_id'] in exploratory else 'confirmation'
    return {'rows': panel, 'matches': matches, 'unmatched_target_ids': unmatched,
            'images': sorted(images), 'targets':len(selected),'controls':len(controls),
            'selection': 'Baseline only; no intervention outcome selection. Both partitions are previously explored validation images.'}
