"""Final-grid support and exact restricted-binary edit bounds."""
import hashlib
import numpy as np
from pycocotools import mask as masks


def encode(x):
    r = masks.encode(np.asfortranarray(np.asarray(x, dtype=np.uint8)))
    return {'size': [int(s) for s in r['size']], 'counts': r['counts'].decode('ascii')}


def decode(r):
    return masks.decode(r).astype(bool)


def iou(a, y):
    union = np.count_nonzero(a | y)
    return float(np.count_nonzero(a & y) / union) if union else 0.


def bbox_support(shape, box):
    h, w = shape
    left, top, width, height = [float(x) for x in box]
    xx = np.arange(w, dtype=np.float64)[None]
    yy = np.arange(h, dtype=np.float64)[:, None]
    return (xx >= left) & (xx < left + width) & (yy >= top) & (yy < top + height)


def gate_from_logits(z_input, cache, bbox, tau):
    import torch
    from ultralytics.utils import ops
    with torch.inference_mode():
        z = ops.scale_masks(z_input.to('cuda')[None, None], cache['original_hw'],
                            ratio_pad=cache['ratio_pad'])[0, 0].cpu().numpy()
    roi = bbox_support(z.shape, bbox)
    return (np.abs(z) <= tau) & roi, roi


def compose(native, replacement, support):
    return np.where(support, replacement, native).astype(bool)


def random_seed(seed, image_id, output_id, repeat):
    key = f'{seed}:{image_id}:{output_id}:{repeat}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')


def random_support(roi, count, seed):
    points = np.flatnonzero(roi)
    if not 0 <= count <= len(points):
        raise ValueError('Support budget invalid')
    out = np.zeros(roi.size, dtype=bool)
    if count:
        out[np.random.Generator(np.random.PCG64(seed)).choice(points, count, replace=False)] = True
    return out.reshape(roi.shape)


def counts(native, changed, y, band):
    return dict(changed_pixels=int(np.count_nonzero(native != changed)),
                changed_outside=int(np.count_nonzero((native != changed) & ~band)),
                native_tp=int(np.count_nonzero(native & y)),
                native_fp=int(np.count_nonzero(native & ~y)),
                native_fn=int(np.count_nonzero(~native & y)),
                editable_tp=int(np.count_nonzero(native & y & band)),
                editable_tn=int(np.count_nonzero(~native & ~y & band)),
                editable_fp=int(np.count_nonzero(native & ~y & band)),
                editable_fn=int(np.count_nonzero(~native & y & band)),
                tp_to_fn=int(np.count_nonzero(native & y & ~changed)),
                tn_to_fp=int(np.count_nonzero(~native & ~y & changed)))


def pair_record(target, native, teacher, existing, truth, band, roi, cfg):
    n, c, g, y = map(decode, (native, teacher, existing, truth))
    arms = dict(N=n, C=c, G=g, O_B=compose(n, y, band),
                C_B=compose(n, c, band), G_B=compose(n, g, band))
    repeats = []
    for r in range(cfg['random_support_repeats']):
        seed = random_seed(cfg['seed'], target['image_id'], target['output_detection_id'], r)
        b = random_support(roi, int(band.sum()), seed)
        repeats.append(dict(repeat=r, seed=seed, iou=iou(compose(n, y, b), y)))
    out = dict(**target, native_rle=native, teacher_rle=teacher, existing_rle=existing, GT_rle=truth,
               B_rle=encode(band), ROI_rle=encode(roi), editable_pixels=int(band.sum()), roi_pixels=int(roi.sum()),
               editable_fraction=float(band.sum() / roi.sum()) if roi.any() else 0.,
               original_pixels=int(band.size), empty_roi=not bool(roi.any()),
               random_oracles=repeats, O_R=float(np.mean([r['iou'] for r in repeats])))
    out['arms'] = {k: dict(iou=iou(v, y), rle=encode(v), **counts(n, v, y, band)) for k, v in arms.items()}
    return out
