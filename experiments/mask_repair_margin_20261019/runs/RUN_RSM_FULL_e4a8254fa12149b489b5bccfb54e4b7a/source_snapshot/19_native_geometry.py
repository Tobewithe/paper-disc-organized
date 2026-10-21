"""Pinned native stages, actual first-threshold logits, and FP64 linear map."""
import torch
import torch.nn.functional as F


def roi_for(cache, rank, device='cpu'):
    h, w = cache['native_decode_hw']
    box = cache['boxes_input'][rank].detach().to(device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=box.dtype)[None, :]
    y = torch.arange(h, device=device, dtype=box.dtype)[:, None]
    return (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])


def target_grid(binary_original, cache):
    from s1_geometry import geometry
    info = geometry(cache)
    y = torch.as_tensor(binary_original.copy(), dtype=torch.float32)
    if list(y.shape) != info['original_hw'] or not ((y == 0) | (y == 1)).all():
        raise ValueError('GT/cache shape or binary identity changed')
    y = F.interpolate(y[None, None], info['resized_hw'], mode='nearest')
    left, top, right, bottom = info['ltrb_padding']
    y = F.pad(y, (left, right, top, bottom))[0, 0]
    if list(y.shape) != list(cache['native_decode_hw']):
        raise ValueError('first-threshold/input-grid dimensions changed')
    return y


def lifted_basis(cache):
    """Exact real-valued bilinear operator, with FP64 rather than ops' FP32 cast."""
    p = cache['proto'].detach().cpu().double()
    shape = list(cache['native_decode_hw'])
    if shape != [4 * v for v in p.shape[-2:]] or shape != list(cache['input_hw']):
        raise ValueError('only the frozen 4x, no-first-stage-padding geometry is supported')
    return F.interpolate(p[None], size=shape, mode='bilinear', align_corners=False)[0]


def native_stages(cache, coefficients, ranks):
    """Keep full N x 32 GEMM shape and native chunk boundaries for every arm."""
    from ultralytics.utils import ops
    p = cache['proto'].to('cuda')
    c = coefficients.to(device='cuda', dtype=cache['c0'].dtype)
    boxes = cache['boxes_input'].to('cuda')
    shape = cache['native_decode_hw']
    if c.shape != cache['c0'].shape or not torch.isfinite(c).all():
        raise ValueError('invalid full-shape coefficient matrix')
    selected = {}
    with torch.inference_mode():
        first = ops.process_mask_native(p, c, boxes, shape)
        final = ops.scale_masks(first[None], cache['original_hw'], ratio_pad=cache['ratio_pad'])[0].byte()
        ph, pw = p.shape[-2:]
        proto_logits = c @ p.float().view(p.shape[0], -1)
        step = max(1, 32_000_000 // (shape[0] * shape[1]))
        for start in range(0, len(c), step):
            wanted = [r for r in ranks if start <= r < start + step]
            if not wanted:
                continue
            up = ops.scale_masks(proto_logits[start:start+step].view(-1, ph, pw)[None], shape)[0]
            for rank in wanted:
                grid = up[rank-start]
                if not torch.equal((grid > 0) & roi_for(cache, rank, 'cuda'), first[rank].bool()):
                    raise ValueError('native first-threshold stage does not match captured logits')
                selected[rank] = grid.cpu().clone()
    return first.cpu(), final.cpu(), selected


def pixel_counts(native_binary, changed_binary, target, roi):
    native, changed, truth = native_binary.bool(), changed_binary.bool(), target.bool()
    tp = native & truth & roi
    tn = ~native & ~truth & roi
    fp = native & ~truth & roi
    fn = ~native & truth & roi
    return dict(native_tp=int(tp.sum()), native_tn=int(tn.sum()), native_fp=int(fp.sum()), native_fn=int(fn.sum()),
        tp_to_fn=int((tp & ~changed).sum()), tn_to_fp=int((tn & changed).sum()),
        fp_to_tn=int((fp & ~changed).sum()), fn_to_tp=int((fn & changed).sum()))
