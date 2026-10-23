import contextlib
import io
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import ops

from common import dump, read
from data import rle


@torch.no_grad()
def predict(model, image):
    model.eval()
    result = model(image)
    det, proto = result[0]
    if isinstance(proto, tuple):
        proto = proto[0]
    out = det[0]; out = out[out[:, 4] > .001]
    out[:, :4].clamp_(0, 640)
    if not len(out):
        return out.cpu().numpy(), np.zeros((0, 640, 640), np.uint8)
    masks = ops.process_mask(proto[0], out[:, 6:], out[:, :4], (640, 640), upsample=True)
    keep = masks.flatten(1).any(1).bool()
    return out[keep].cpu().numpy(), masks[keep].cpu().numpy().astype(np.uint8)


def score(gt_file, predictions, kind):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_file))
        # Segmentation must derive area from RLE, rather than the detection box.
        source = [{k: v for k, v in p.items() if k != ('bbox' if kind == 'segm' else 'segmentation')} for p in predictions]
        if source:
            dt = gt.loadRes(source)
        else:
            dt = COCO(); dt.dataset = {**gt.dataset, 'annotations': []}; dt.createIndex()
        ev = COCOeval(gt, dt, kind); ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return [float(v)*100 for v in ev.stats]


def evaluate(model, root, store, outdir, tag):
    outdir.mkdir(parents=True, exist_ok=True); start = time.monotonic(); predictions = []
    rows = [r for r in store.rows if r['domain'] == 'real' and r['split'] == 'dev']
    for row in rows:
        batch = store.batch([row], seed=0, augment=False)
        det, masks = predict(model, batch['img'])
        for d, mask in zip(det, masks):
            x0, y0, x1, y1 = d[:4].tolist()
            predictions.append({'image_id': row['image_id'], 'category_id': 1, 'score': float(d[4]),
                                'bbox': [x0, y0, x1-x0, y1-y0], 'segmentation': rle(mask)})
    seg, box = score(root/'DEV_COCO.json', predictions, 'segm'), score(root/'DEV_COCO.json', predictions, 'bbox')
    metrics = {'mask_ap': seg[0], 'mask_ap50': seg[1], 'mask_ap75': seg[2], 'mask_small_ap': seg[3],
               'mask_ar100': seg[8], 'box_ap': box[0], 'box_ap50': box[1], 'box_ap75': box[2],
               'segm_stats': seg, 'bbox_stats': box, 'images': len(rows), 'predictions': len(predictions),
               'elapsed_seconds': time.monotonic()-start, 'tag': tag, 'canvas': 640,
               'selection_split': 'development_not_independent_test'}
    dump(outdir/(tag+'_predictions.json'), predictions); dump(outdir/(tag+'_metrics.json'), metrics)
    print('EVAL', tag, {k: metrics[k] for k in ['mask_ap', 'mask_ap75', 'box_ap']}, flush=True)
    return metrics
