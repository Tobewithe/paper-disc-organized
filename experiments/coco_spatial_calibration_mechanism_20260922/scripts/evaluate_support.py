"""Normal-output COCO evaluation of the independently trained shared-shape head."""
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
from shared_shape_head import SharedShapeHead, grid_tensor, full_template, correct


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for key in ['train', 'template', 'data', 'weights', 'reference', 'scalar-reference', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--scalar-key', default='scalar_selected')
    parser.add_argument('--fixed-alpha', type=float)
    parser.add_argument('--local-control', type=Path)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    setup(); torch.backends.cudnn.benchmark = False; start = time.monotonic()
    assert json.loads((args.train/'run.json').read_text())['status'] == 'completed'
    selection = json.loads((args.train/'SELECTION.json').read_text()); alpha = selection['alpha'] if args.fixed_alpha is None else args.fixed_alpha
    checkpoint = torch.load(args.train/'epoch8.pt', weights_only=False)
    assert checkpoint['template']['field'] == json.loads(args.template.read_text())['field']
    net = SharedShapeHead().cuda().eval(); net.load_state_dict(checkpoint['state_dict'])
    local_control = None
    if args.local_control:
        local_control = original.Refiner('local4').cuda().eval()
        local_checkpoint = torch.load(args.local_control, weights_only=False)
        assert local_checkpoint['seed'] == selection['seed']
        local_control.load_state_dict(local_checkpoint['state_dict'])
    grid = grid_tensor(args.template)
    coco = COCO(str(args.data/'annotations/instances_val2017.json')); ids = sorted(coco.imgs); assert len(ids) == 5000
    cat_ids = sorted(coco.cats)
    model = YOLO(str(args.weights)).model.cuda().float().eval(); head = model.model[-1]; assert head.end2end
    transform = LetterBox((640, 640), auto=True, stride=32)
    original.save(args.out/'SPLIT.json', {'val_ids': ids})
    original.save(args.out/'SETUP.json', {'images': 5000, 'alpha': alpha, 'checkpoint': str(args.train/'epoch8.pt'),
        'template': str(args.template), 'coordinate': 'pixel_center', 'independent_head': True,
        'candidate_uses_local_head': False, 'separate_local_control': str(args.local_control) if args.local_control else None,
        'gt_used_in_inference': False, 'protocol': 'PROTOCOL.json'})
    total = 0; empty_support = 0; zero_std = 0
    control_stream = gzip.open(args.out/'predictions_local.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) if local_control else None
    with gzip.open(args.out/'predictions_shared2.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as stream:
        for number, image_id in enumerate(ids):
            im = cv2.imread(str(args.data/'images/val2017'/coco.imgs[image_id]['file_name'])); assert im is not None
            orig = im.shape[:2]; resized = transform.apply_image({'img': im}, transform.get_params({'img': im}))['img']
            inp = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None]/255
            shape = tuple(inp.shape[-2:]); _, raw = model(inp); p = raw['one2one']
            proto, allc = p['proto'][0], p['mask_coefficient'][0].T
            allboxes = head._get_decode_boxes(p)[0].T
            ts, tc, ti = head.get_topk_index(p['scores'].permute(0, 2, 1).sigmoid(), 300)
            keep = ts[0, :, 0] > .001
            scores, classes, selected = ts[0, keep, 0], tc[0, keep, 0].long(), ti[0, keep, 0]
            boxes, coeff = allboxes[selected], allc[selected]
            low = (allc@proto.flatten(1)).reshape(-1, *proto.shape[-2:])
            for first in range(0, len(boxes), 16):
                sl = slice(first, first+16); bb = boxes[sl]
                x, _, _ = original.features(proto, coeff[sl], bb, shape)
                base = F.interpolate(low[selected[sl]][None], shape, mode='bilinear', align_corners=False)[0]
                basis, _, std, count = full_template(grid, bb, shape)
                empty_support += int((count == 0).sum()); zero_std += int(((std <= 1e-12) & (count > 0)).sum())
                z = correct(base, basis, net(x)*alpha)
                assert torch.isfinite(z).all()
                masks = ops.scale_masks(ops.crop_mask(z.clone(), bb).gt(0).byte()[None], orig)[0].byte().cpu().numpy()
                for j, mask in enumerate(masks):
                    if not mask.any(): continue
                    k = first+j; rle = mu.encode(np.asfortranarray(mask)); rle['counts'] = rle['counts'].decode('ascii')
                    stream.write(json.dumps({'image_id': image_id, 'category_id': cat_ids[int(classes[k])],
                        'score': float(scores[k]), 'raw_id': int(selected[k]), 'segmentation': rle})+'\n'); total += 1
                if local_control is not None:
                    control_z = image_correct(base, proto, None, bb, shape, local_control(x)*.5, 'local4', 'pixel_center')
                    control_masks = ops.scale_masks(ops.crop_mask(control_z, bb).gt(0).byte()[None], orig)[0].byte().cpu().numpy()
                    for j, mask in enumerate(control_masks):
                        if not mask.any(): continue
                        k = first+j; rle = mu.encode(np.asfortranarray(mask)); rle['counts'] = rle['counts'].decode('ascii')
                        control_stream.write(json.dumps({'image_id': image_id, 'category_id': cat_ids[int(classes[k])],
                            'score': float(scores[k]), 'raw_id': int(selected[k]), 'segmentation': rle})+'\n')
            if number % 100 == 0 or number+1 == len(ids):
                stream.flush(); progress = {'stage': 'predict', 'images': number+1, 'total': len(ids), 'elapsed_s': time.monotonic()-start}
                original.save(args.out/'progress.json', progress); print(json.dumps(progress), flush=True)
    if control_stream: control_stream.close()
    del model, net, raw, p, proto, low, base, basis, z, x; gc.collect(); torch.cuda.empty_cache()
    reference = json.loads((args.reference/'RESULTS.json').read_text())
    refmatch = json.loads((args.reference/'MATCHED_GT75.json').read_text())
    scalar = json.loads((args.scalar_reference/'RESULTS.json').read_text())
    scalar_match = json.loads((args.scalar_reference/'MATCHED_GT75.json').read_text())
    assert abs(reference['baseline']['metrics']['AP']-scalar['baseline']['metrics']['AP']) < 1e-12
    results = {**reference, 'scalar': scalar[args.scalar_key]}
    matched = {**refmatch, 'scalar': scalar_match[args.scalar_key]}
    with gzip.open(args.out/'predictions_shared2.jsonl.gz', 'rt') as stream: pred = [json.loads(line) for line in stream]
    assert all('bbox' not in row for row in pred)
    dt = coco.loadRes(pred); ev = COCOeval(coco, dt, 'segm'); ev.params.imgIds = ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    ti = int(np.argmin(abs(ev.params.iouThrs-.75))); good = set()
    for entry in ev.evalImgs:
        if entry is not None and entry['aRng'] == ev.params.areaRng[0]:
            good.update(int(gt) for j, gt in enumerate(entry['gtIds']) if not entry['gtIgnore'][j] and entry['gtMatches'][ti, j] > 0)
    ordinary = {ann['id']: ann for ann in coco.anns.values() if not ann.get('iscrowd', 0) and not ann.get('ignore', 0)}
    good &= ordinary.keys(); baseline_good = set(matched['baseline'])
    def size(ann): return 'small' if ann['area'] < 1024 else 'medium' if ann['area'] < 9216 else 'large'
    metrics = ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL')
    results['shared2'] = {'metrics': dict(zip(metrics, map(float, ev.stats))), 'predictions': len(pred),
        'matched75': len(good), 'repaired75': len(good-baseline_good), 'damaged75': len(baseline_good-good),
        'size_recall75': {g: {'gt': sum(size(ann) == g for ann in ordinary.values()),
            'matched': sum(size(ordinary[aid]) == g for aid in good)} for g in ['small', 'medium', 'large']},
        'alpha': alpha, 'seed': selection['seed'], 'parameters': selection['parameters']}
    matched['shared2'] = sorted(good)
    gate = {'ap': 100*ev.stats[0], 'threshold_ap': 44.30, 'scalar_ap': 100*results['scalar']['metrics']['AP'],
        'pass': bool(100*ev.stats[0] >= 44.30 and ev.stats[0] > results['scalar']['metrics']['AP']),
        'not_a_significance_test': True, 'scope': 'Historical shared2 engineering reference only; does not control this ablation queue.'}
    if local_control is not None:
        del pred, dt, ev; gc.collect()
        with gzip.open(args.out/'predictions_local.jsonl.gz', 'rt') as stream: pred = [json.loads(line) for line in stream]
        dt = coco.loadRes(pred); ev = COCOeval(coco, dt, 'segm'); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize(); good = set()
        for entry in ev.evalImgs:
            if entry is not None and entry['aRng'] == ev.params.areaRng[0]:
                good.update(int(gt) for j, gt in enumerate(entry['gtIds']) if not entry['gtIgnore'][j] and entry['gtMatches'][ti, j] > 0)
        good &= ordinary.keys()
        results['local'] = {'metrics': dict(zip(metrics, map(float, ev.stats))), 'predictions': len(pred),
            'matched75': len(good), 'repaired75': len(good-baseline_good), 'damaged75': len(baseline_good-good),
            'size_recall75': {g: {'gt': sum(size(ann) == g for ann in ordinary.values()),
                'matched': sum(size(ordinary[aid]) == g for aid in good)} for g in ['small', 'medium', 'large']},
            'alpha': .5, 'seed': selection['seed'], 'coordinate_rule': 'pixel_center', 'weights': str(args.local_control)}
        matched['local'] = sorted(good)
    original.save(args.out/'RESULTS.json', results); original.save(args.out/'MATCHED_GT75.json', matched)
    original.save(args.out/'GATE.json', gate)
    original.save(args.out/'COMPLETE.json', {'images': 5000, 'predictions': total, 'elapsed_s': time.monotonic()-start,
        'empty_crop_support': empty_support, 'nonempty_zero_std': zero_std, 'reference': str(args.reference),
        'scalar_reference': str(args.scalar_reference), 'gate': gate})
    print(json.dumps({'shared2': results['shared2'], 'gate': gate}), flush=True)


if __name__ == '__main__':
    main()
