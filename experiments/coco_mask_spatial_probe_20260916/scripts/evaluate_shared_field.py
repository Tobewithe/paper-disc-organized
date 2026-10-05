"""Whole-val COCO evaluation of a training-derived shared spatial shape.

All arms share the detector, candidates, scores, boxes, and prototypes. The
shared-shape control retains each instance's learned correction mean/std.
"""
import argparse
import gc
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

import learn_refinement as original
from component_seed_probe import setup, image_correct


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for key in ['data', 'weights', 'local-weights', 'template', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    setup(); torch.backends.cudnn.benchmark = False; start = time.monotonic()
    coco = COCO(str(args.data / 'annotations/instances_val2017.json'))
    val_ids = sorted(coco.imgs); assert len(val_ids) == 5000
    cat_ids = sorted(coco.cats)
    local = original.Refiner('local4').cuda().eval()
    local.load_state_dict(torch.load(args.local_weights, weights_only=False)['state_dict'])
    shared = torch.tensor(json.loads(args.template.read_text())['field'], device='cuda').reshape(1, 16)
    model = YOLO(str(args.weights)).model.cuda().float().eval(); head = model.model[-1]
    assert head.end2end
    transform = LetterBox((640, 640), auto=True, stride=32)
    names = ['baseline', 'local', 'shared_template']
    original.save(args.out / 'SPLIT.json', {'val_ids': val_ids})
    original.save(args.out / 'SETUP.json', {
        'images': 5000, 'alpha': .5, 'coordinate': 'pixel_center', 'max_det': 300, 'conf': .001,
        'arms': names, 'training': False, 'template_source': str(args.template),
        'shared_control': 'shared training shape, instance learned crop mean/std',
        'selection': 'all normal outputs on all val2017 images; no GT-based cohort or candidate selection',
    })
    streams = {name: gzip.open(args.out / f'predictions_{name}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) for name in names}
    empty_support = 0
    for number, image_id in enumerate(val_ids):
        im = cv2.imread(str(args.data / 'images/val2017' / coco.imgs[image_id]['file_name']))
        assert im is not None
        orig = im.shape[:2]
        resized = transform.apply_image({'img': im}, transform.get_params({'img': im}))['img']
        inp = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
        shape = tuple(inp.shape[-2:]); _, raw = model(inp); p = raw['one2one']
        proto, allc = p['proto'][0], p['mask_coefficient'][0].T
        allboxes = head._get_decode_boxes(p)[0].T
        ts, tc, ti = head.get_topk_index(p['scores'].permute(0, 2, 1).sigmoid(), 300)
        keep = ts[0, :, 0] > .001
        scores, classes, ids = ts[0, keep, 0], tc[0, keep, 0].long(), ti[0, keep, 0]
        boxes, coeff = allboxes[ids], allc[ids]
        low = (allc @ proto.flatten(1)).reshape(-1, *proto.shape[-2:])
        for first in range(0, len(boxes), 16):
            sl = slice(first, first + 16); bb = boxes[sl]
            x, _, _ = original.features(proto, coeff[sl], bb, shape)
            base = F.interpolate(low[ids[sl]][None], shape, mode='bilinear', align_corners=False)[0]
            zero = torch.zeros_like(base)
            field = image_correct(zero, proto, None, bb, shape, local(x) * .5, 'local4', 'pixel_center')
            template = image_correct(zero, proto, None, bb, shape, shared.expand(len(bb), -1), 'local4', 'pixel_center')
            support = ops.crop_mask(torch.ones_like(base), bb).bool()
            counts = support.sum((1, 2)); empty_support += int((counts == 0).sum())
            n = counts.clamp_min(1)
            mean = (field * support).sum((1, 2))
            mean = mean / n
            std = (((field - mean[:, None, None]).square() * support).sum((1, 2)) / n).sqrt()
            tm = (template * support).sum((1, 2)) / n
            tstd = (((template - tm[:, None, None]).square() * support).sum((1, 2)) / n).sqrt()
            template = (template - tm[:, None, None]) * (std / tstd.clamp_min(1e-12))[:, None, None] + mean[:, None, None]
            variants = {'baseline': base, 'local': base + field, 'shared_template': base + template}
            for name, z in variants.items():
                masks = ops.scale_masks(ops.crop_mask(z.clone(), bb).gt(0).byte()[None], orig)[0].byte().cpu().numpy()
                for j, mask in enumerate(masks):
                    if not mask.any():
                        continue
                    k = first + j
                    rle = mu.encode(np.asfortranarray(mask)); rle['counts'] = rle['counts'].decode('ascii')
                    streams[name].write(json.dumps({'image_id': image_id, 'category_id': cat_ids[int(classes[k])],
                        'score': float(scores[k]), 'segmentation': rle, 'raw_id': int(ids[k])}) + '\n')
        if number % 50 == 0 or number + 1 == len(val_ids):
            for stream in streams.values(): stream.flush()
            progress = {'stage': 'predict', 'images': number + 1, 'total': len(val_ids), 'elapsed_s': time.monotonic() - start}
            original.save(args.out / 'progress.json', progress); print(json.dumps(progress), flush=True)
    for stream in streams.values(): stream.close()
    # Free GPU memory before CPU COCO evaluation.
    del model, local, base, field, template, variants, raw, p, proto, low, x
    gc.collect(); torch.cuda.empty_cache()
    ordinary = {ann['id']: ann for ann in coco.anns.values() if not ann.get('iscrowd', 0) and not ann.get('ignore', 0)}
    metric_names = ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL')
    results, matched = {}, {}; baseline_good = set()
    def size(ann): return 'small' if ann['area'] < 1024 else 'medium' if ann['area'] < 9216 else 'large'
    for name in names:
        with gzip.open(args.out / f'predictions_{name}.jsonl.gz', 'rt') as f:
            pred = [json.loads(line) for line in f]
        assert all('bbox' not in row for row in pred)
        dt = coco.loadRes(pred); ev = COCOeval(coco, dt, 'segm'); ev.params.imgIds = val_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        ti = int(np.argmin(abs(ev.params.iouThrs - .75))); good = set()
        for entry in ev.evalImgs:
            if entry is not None and entry['aRng'] == ev.params.areaRng[0]:
                good.update(int(g) for j, g in enumerate(entry['gtIds']) if not entry['gtIgnore'][j] and entry['gtMatches'][ti, j] > 0)
        good &= ordinary.keys()
        if name == 'baseline': baseline_good = good.copy()
        results[name] = {'metrics': dict(zip(metric_names, map(float, ev.stats))), 'predictions': len(pred),
            'matched75': len(good), 'repaired75': len(good - baseline_good), 'damaged75': len(baseline_good - good),
            'size_recall75': {group: {'gt': sum(size(ann) == group for ann in ordinary.values()),
                'matched': sum(size(ordinary[aid]) == group for aid in good)} for group in ['small', 'medium', 'large']}}
        matched[name] = sorted(good)
        original.save(args.out / 'RESULTS.json', results); original.save(args.out / 'MATCHED_GT75.json', matched)
        progress = {'stage': 'score', 'mode': name, 'mask_ap': 100 * ev.stats[0], 'elapsed_s': time.monotonic() - start}
        original.save(args.out / 'progress.json', progress); print(json.dumps(progress), flush=True)
        del pred, dt, ev; gc.collect()
    original.save(args.out / 'COMPLETE.json', {'images': len(val_ids), 'ordinary_gt': len(ordinary),
        'elapsed_s': time.monotonic() - start, 'empty_crop_support_candidates': empty_support, 'no_new_training': True})


if __name__ == '__main__':
    main()
