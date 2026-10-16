"""Finite ray candidates from binary transition events; actual scoring remains native."""
import numpy as np


def conversion_table(cache):
    import torch
    from ultralytics.utils import ops
    ih, iw = cache['native_decode_hw']; oh, ow = cache['original_hw']
    if (ih, iw) == (oh, ow):
        top, left, h, w = 0, 0, ih, iw
    else:
        pw, ph = cache['ratio_pad'][1]
        top, left = round(ph-.1), round(pw-.1)
        h, w = ih-round(ph+.1)-top, iw-round(pw+.1)-left
    ys = np.maximum((np.arange(oh)+.5)*h/oh-.5, 0)
    xs = np.maximum((np.arange(ow)+.5)*w/ow-.5, 0)
    y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    indices = np.zeros((oh*ow, 4), dtype=np.int64)
    present = np.zeros_like(indices, dtype=bool)
    pixels = np.arange(oh*ow)
    for yy in (y0, np.minimum(y0+1, h-1)):
        for xx in (x0, np.minimum(x0+1, w-1)):
            y = np.broadcast_to((yy+top)[:, None], (oh, ow)).ravel()
            x = np.broadcast_to((xx+left)[None, :], (oh, ow)).ravel()
            bit = (y % 2)*2 + x % 2
            indices[pixels, bit] = y*iw+x
            present[pixels, bit] = True
    yy = torch.arange(ih, device='cuda')[:, None]
    xx = torch.arange(iw, device='cuda')[None, :]
    parity = (yy % 2)*2 + xx % 2
    patterns = ((torch.arange(16, device='cuda')[:, None, None] >> parity) & 1).byte()
    with torch.inference_mode():
        table = ops.scale_masks(patterns[None], (oh, ow), ratio_pad=cache['ratio_pad'])[0].byte()
    return dict(indices=indices, present=present, table=table.cpu().numpy().reshape(16, -1).astype(bool),
                output_shape=(oh, ow), input_shape=(ih, iw))


def lookup(binary, transform):
    values = np.asarray(binary, dtype=bool).ravel()[transform['indices']]
    values &= transform['present']
    pattern = (values * np.array([1, 2, 4, 8], dtype=np.uint8)).sum(axis=1)
    return transform['table'][pattern, np.arange(len(pattern))].reshape(transform['output_shape'])


def binary_iou(binary, truth):
    binary, truth = np.asarray(binary, bool), np.asarray(truth, bool)
    return float(np.count_nonzero(binary & truth)/max(1, np.count_nonzero(binary | truth)))


def build_sweep(z0, direction, roi, truth, transform, maximum):
    """Count all open-interval proxy masks without decoding every interval separately."""
    z = np.where(roi, np.asarray(z0, dtype=np.float64), -1.)
    d = np.where(roi, np.asarray(direction, dtype=np.float64), 0.)
    indices, present, table = (transform[k] for k in ('indices', 'present', 'table'))
    zg, dg = z.ravel()[indices].copy(), d.ravel()[indices].copy()
    zg[~present] = -1.; dg[~present] = 0.
    threshold = np.full(zg.shape, np.inf, dtype=np.float64)
    np.divide(-zg, dg, out=threshold, where=dg != 0)
    bits = np.array([1, 2, 4, 8], dtype=np.uint8)
    pixels = np.arange(len(zg)); gt = np.asarray(truth, bool).ravel()
    def output(values):
        return table[(values*bits).sum(axis=1), pixels]
    initial = output((zg > 0) | ((zg == 0) & (dg > 0)))
    event_t, event_a, event_i = [], [], []
    for j in range(4):
        t = threshold[:, j]
        valid = np.isfinite(t) & (t > 0) & (t < maximum)
        if j:
            valid &= (threshold[:, :j] != t[:, None]).all(axis=1)
        before = ((dg > 0) & (threshold < t[:, None])) | ((dg < 0) & (threshold >= t[:, None])) | ((dg == 0) & (zg > 0))
        after = ((dg > 0) & (threshold <= t[:, None])) | ((dg < 0) & (threshold > t[:, None])) | ((dg == 0) & (zg > 0))
        change = output(after).astype(np.int8)-output(before).astype(np.int8)
        valid &= change != 0
        event_t.append(t[valid]); event_a.append(change[valid]); event_i.append(change[valid]*gt[valid])
    raw_t = np.concatenate(event_t)
    if len(raw_t):
        times, inverse = np.unique(raw_t, return_inverse=True)
        da = np.bincount(inverse, weights=np.concatenate(event_a)).astype(np.int64)
        di = np.bincount(inverse, weights=np.concatenate(event_i)).astype(np.int64)
    else:
        times = np.array([], dtype=np.float64); da = di = np.array([], dtype=np.int64)
    return dict(times=times, delta_area=da, delta_intersection=di,
                initial_area=np.int64(initial.sum()), initial_intersection=np.int64((initial & gt).sum()),
                gt_area=np.int64(gt.sum()), alpha_max=np.float64(maximum),
                original_pixels=np.int64(len(gt)), zero_direction=np.bool_(not np.any(d)))


def curve(sweep):
    times = sweep['times']; maximum = float(sweep['alpha_max'])
    edges = np.concatenate(([0.], times, [maximum]))
    alpha = (edges[:-1]+edges[1:])/2
    area = int(sweep['initial_area'])+np.concatenate(([0], np.cumsum(sweep['delta_area'])))
    intersection = int(sweep['initial_intersection'])+np.concatenate(([0], np.cumsum(sweep['delta_intersection'])))
    gt = int(sweep['gt_area'])
    if np.any(area < 0) or np.any(intersection < 0) or np.any(intersection > np.minimum(area, gt)):
        raise ValueError('invalid event-sweep counts')
    valid = (alpha > edges[:-1]) & (alpha < edges[1:])
    return alpha[valid], intersection[valid]/np.maximum(1, gt+area[valid]-intersection[valid])


def candidates(sweep, cfg):
    alpha, scores = curve(sweep)
    order = np.lexsort((alpha, np.abs(alpha-1), -scores))
    selected = [] if bool(sweep['zero_direction']) else alpha[order[:cfg['top_candidates']]].tolist()
    return list(dict.fromkeys(selected+cfg['fixed_candidates']))


def predict(z0, direction, roi, alpha, transform):
    return lookup(((np.asarray(z0)+alpha*np.asarray(direction)) > 0) & roi, transform)


def choose_actual(records):
    return min(range(len(records)), key=lambda i: (-records[i]['original_iou'], abs(records[i]['alpha']-1), records[i]['alpha']))
