"""Matched FP32 latency and parameter/MAC accounting for independent shared2."""
import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from component_seed_probe import setup, image_correct, original
from shared_shape_head import SharedShapeHead, grid_tensor, full_template, correct


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for key in ['train', 'template', 'scalar-weights', 'local-weights', 'data', 'weights', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); setup()
    torch.backends.cudnn.benchmark = False
    selection = json.loads((args.train/'SELECTION.json').read_text())
    shared = SharedShapeHead().cuda().eval(); shared.load_state_dict(torch.load(args.train/'epoch8.pt', weights_only=False)['state_dict'])
    nets = {'shared2': shared}
    for name, path in [('scalar', args.scalar_weights), ('local4', args.local_weights)]:
        net = original.Refiner(name).cuda().eval(); net.load_state_dict(torch.load(path, weights_only=False)['state_dict']); nets[name] = net
    grid = grid_tensor(args.template)
    coco = json.loads((args.data/'annotations/instances_val2017.json').read_text())
    files = {im['id']: im['file_name'] for im in coco['images']}
    ids = sorted(files); random.Random(20260916).shuffle(ids); warm, ids = ids[:5], ids[5:55]
    model = YOLO(str(args.weights)).model.cuda().float().eval(); head = model.model[-1]; assert head.end2end
    transform = LetterBox((640, 640), auto=True, stride=32)
    def prepare(image_id):
        im = cv2.imread(str(args.data/'images/val2017'/files[image_id])); assert im is not None
        res = transform.apply_image({'img': im}, transform.get_params({'img': im}))['img']
        return torch.from_numpy(np.ascontiguousarray(res[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None]/255, im.shape[:2]
    def once(inp, orig, mode):
        torch.cuda.synchronize(); start = time.perf_counter()
        _, raw = model(inp); p = raw['one2one']; torch.cuda.synchronize(); forward = time.perf_counter()
        shape = tuple(inp.shape[-2:]); proto = p['proto'][0]; allc = p['mask_coefficient'][0].T
        boxes = head._get_decode_boxes(p)[0].T
        scores, _, raw_ids = head.get_topk_index(p['scores'].permute(0, 2, 1).sigmoid(), 300)
        selected = raw_ids[0, scores[0, :, 0] > .001, 0]; boxes = boxes[selected]; coeff = allc[selected]
        low = (allc@proto.flatten(1)).reshape(-1, *proto.shape[-2:])
        for first in range(0, len(boxes), 16):
            sl = slice(first, first+16); bb = boxes[sl]
            base = F.interpolate(low[selected[sl]][None], shape, mode='bilinear', align_corners=False)[0]
            z = base
            if mode != 'baseline':
                x, _, _ = original.features(proto, coeff[sl], bb, shape)
                if mode == 'shared2':
                    basis, _, _, _ = full_template(grid, bb, shape)
                    z = correct(base, basis, nets[mode](x)*selection['alpha'])
                else:
                    z = image_correct(base, proto, None, bb, shape, nets[mode](x)*.5, mode, 'pixel_center')
            _ = ops.scale_masks(ops.crop_mask(z.clone(), bb).gt(0).byte()[None], orig)[0].byte().cpu().numpy()
        torch.cuda.synchronize(); end = time.perf_counter()
        return {'total_ms': 1000*(end-start), 'forward_ms': 1000*(forward-start),
            'mask_ms': 1000*(end-forward), 'predictions': len(boxes)}
    modes = ['baseline', 'scalar', 'local4', 'shared2']
    for image_id in warm:
        inp, orig = prepare(image_id)
        for mode in modes: once(inp, orig, mode)
    rows = []
    for repeat in range(3):
        for number, image_id in enumerate(ids):
            inp, orig = prepare(image_id); offset = (number+repeat) % 4
            for mode in modes[offset:]+modes[:offset]:
                rows.append({'mode': mode, 'image_id': image_id, 'repeat': repeat, **once(inp, orig, mode)})
        print(json.dumps({'timing_round': repeat+1, 'images': 50}), flush=True)
    summary = {}
    for mode in modes:
        rr = [row for row in rows if row['mode'] == mode]
        summary[mode] = {key: {'mean': float(np.mean([row[key] for row in rr])),
            'median': float(np.median([row[key] for row in rr])), 'p90': float(np.quantile([row[key] for row in rr], .9))}
            for key in ['total_ms', 'forward_ms', 'mask_ms']}
    # Isolate refiner forward from feature sampling and full-resolution decode.
    torch.manual_seed(20260920); x = torch.randn(16, 67, 32, 32, device='cuda'); head_times = {name: [] for name in nets}
    for net in nets.values():
        for _ in range(20): net(x)
    names = list(nets)
    for repeat in range(100):
        for name in names[repeat % 3:]+names[:repeat % 3]:
            begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            begin.record(); nets[name](x); end.record(); end.synchronize()
            head_times[name].append(begin.elapsed_time(end))
    params = {name: sum(p.numel() for p in net.parameters()) for name, net in nets.items()}
    # Conv/Linear multiply-accumulates only: nonlinearities/pooling/sampling excluded.
    fixed_macs = 67*32*3*3*16*16 + 32*32*3*3*8*8 + 512*128
    macs = {name: fixed_macs+128*dim for name, dim in [('scalar', 1), ('local4', 16), ('shared2', 2)]}
    result = {'gpu': torch.cuda.get_device_name(0), 'dtype': 'float32', 'image_batch': 1, 'roi_batch': 16,
        'images': ids, 'warm_images': warm, 'rounds': 3, 'summary': summary, 'parameters': params,
        'head_macs_per_roi': macs, 'mac_scope': 'Conv/Linear MAC only; excludes feature sampling, activations, pooling and full mask decoding.',
        'head_forward_ms_per_16_roi': {name: {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'p90': float(np.quantile(v, .9))} for name, v in head_times.items()},
        'candidate_count_range': [min(row['predictions'] for row in rows), max(row['predictions'] for row in rows)],
        'scope': 'Same machine, eager unfused FP32. Pipeline includes forward/topk/full masks/refinement/crop/resize/CPU transfer. Image loading/preprocessing/RLE excluded. Not optimized deployment FPS.'}
    original.save(args.out/'TIMING.json', result); original.save(args.out/'samples.json', rows)
    original.save(args.out/'head_samples.json', head_times); original.save(args.out/'COMPLETE.json', {'images': 50, 'rounds': 3})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
