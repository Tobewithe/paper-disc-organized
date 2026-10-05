"""GT-free, frozen candidate-response relations on the existing 16x16 ROI.

The caller supplies a score-filtered, frozen prediction pool from the SAME
image as A. This module never accesses labels, TAL assignments, or GT classes.
Classes are prediction classes and are used only for descriptive counts.

S retains neighbour boxes/scores/support but substitutes the target's frozen
c0. It is therefore a self-response control with shared neighbour geometry,
not a control that is completely unaware of other candidate boxes. T uses
the neighbours' frozen coefficients. M rolls each T response inside that
neighbour's discrete support rectangle before aggregation. Candidate masks
may overlap, and different raw IDs may describe the same physical instance.

A is the existing ROIAlign(P, clamped target box, 16x16, sampling_ratio=2,
aligned=True) operator. A row averages the four ROIAlign samples; support is
defined at the geometric centre of that SAME ROI bin. We do not substitute
the neighbour's own ROI grid or claim support is an exact decoded mask.
"""
from __future__ import annotations

import torch


SIDE = 16
MAX_NEIGHBOURS = 4
INPUT_SIZE = 640.0
CHANGE_ATOL = 1e-7


def _check_vector(name, value, length, *, integer=False):
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {tuple(value.shape)}")
    if integer and value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer IDs/classes")


def _aggregate(responses, availability):
    """Mean denominator is selected neighbours, including empty support."""
    return torch.stack((responses.amax(0), responses.mean(0), availability))


def _support_roll(responses, support):
    """Preserve each neighbour's support and response multiset exactly.

An empty, 1x1, or constant rectangle can remain unchanged. Such candidates
are retained; changed_fraction reports whether the final relation changed.
Rolling within a rectangle introduces a wrap boundary, so this control is
a spatial-correspondence intervention, not a distribution-free causal test.
"""
    wrong = responses.clone()
    for j in range(len(responses)):
        ys = torch.where(support[j].any(dim=1))[0]
        xs = torch.where(support[j].any(dim=0))[0]
        if not len(ys) or not len(xs):
            continue
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        patch = responses[j, y0:y1, x0:x1]
        wrong[j, y0:y1, x0:x1] = torch.roll(
            patch, shifts=(patch.shape[0] // 2, patch.shape[1] // 2), dims=(0, 1)
        )
    return wrong


@torch.no_grad()
def build_relations(A, c0, target_boxes, target_raw, pool_boxes, pool_c,
                    pool_score, pool_raw, pool_class, target_class):
    """Return three frozen [N,3,16,16] relation tensors and audit vectors.

All arguments must be tensors on the same device. A:[N,256,32], c0:[N,32],
target_boxes:[N,4] and pool_boxes:[M,4] use input-pixel xyxy coordinates.
Coefficients and response calculations use FP32. All outputs stay on the
input device; callers may move them to CPU for their immutable cache.

For each valid target, exclude its raw ID and choose at most four valid
pool rows with positive clipped-box IoU, ranked by score*IoU. Ties use raw
ID ascending. The caller controls the pool's top-k and score threshold.
No class restriction, NMS, GT identity removal, or forced mask exclusion is
applied here. Distinct raw candidates may duplicate the same real object.

The first two channels are maximum and mean of score*sigmoid(A_i*c_j),
cropped by that neighbour's box at ROI-bin centres. Mean divides by the
number of selected neighbours, not by per-pixel support or padded slots.
Channel 3 is the number of supporting neighbours divided by FOUR, even
when fewer than four are selected. All three arms share that third channel.

Invalid/degenerate target boxes return zero relations and counts. Invalid
pool geometry, nonfinite coefficients/scores, and scores outside [0,1] are
excluded. Nonfinite A/c0 on a valid target raises instead of hiding it.
Empty pools and continuous intersections containing no ROI-bin centres are
allowed; the latter still count as selected neighbours. Padding raw IDs=-1.

changed_fraction is the fraction of union-supported bins where either M
response channel differs from T by >1e-7, with zero for no supported bins.
It measures the intervention after aggregation, not mask/GT accuracy.
"""
    tensors = (A, c0, target_boxes, target_raw, pool_boxes, pool_c,
               pool_score, pool_raw, pool_class, target_class)
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("Every build_relations argument must be a tensor")
    device = A.device
    if any(value.device != device for value in tensors):
        raise ValueError("All relation inputs must share one device")
    if A.ndim != 3 or A.shape[1:] != (SIDE * SIDE, 32):
        raise ValueError("A must have shape [N,256,32]")
    n = len(A)
    if pool_boxes.ndim != 2 or pool_boxes.shape[1] != 4:
        raise ValueError("pool_boxes must have shape [M,4]")
    m = len(pool_boxes)
    for name, value, shape in (("c0", c0, (n, 32)),
                               ("target_boxes", target_boxes, (n, 4)),
                               ("pool_c", pool_c, (m, 32))):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, value, length in (("target_raw", target_raw, n),
                                ("target_class", target_class, n),
                                ("pool_raw", pool_raw, m),
                                ("pool_class", pool_class, m)):
        _check_vector(name, value, length, integer=True)
    _check_vector("pool_score", pool_score, m)
    if bool((target_raw < 0).any()) or bool((pool_raw < 0).any()):
        raise ValueError("Input raw IDs must be nonnegative; -1 is output padding only")
    if pool_raw.unique().numel() != m:
        raise ValueError("Prediction pool must contain each raw anchor at most once")

    relations = [torch.zeros((n, 3, SIDE, SIDE), device=device, dtype=torch.float32)
                 for _ in range(3)]
    counts = torch.zeros(n, device=device, dtype=torch.long)
    same = torch.zeros_like(counts)
    changed = torch.zeros(n, device=device, dtype=torch.float32)
    chosen_raw = torch.full((n, MAX_NEIGHBOURS), -1, device=device, dtype=torch.long)
    targets = target_boxes.detach().float().clamp(0.0, INPUT_SIZE)
    boxes = pool_boxes.detach().float().clamp(0.0, INPUT_SIZE)
    coefficients = pool_c.detach().float()
    scores = pool_score.detach().float()
    target_valid = torch.isfinite(target_boxes).all(1)
    target_valid &= ((targets[:, 2:] - targets[:, :2]) > 0).all(1)
    pool_valid = torch.isfinite(pool_boxes).all(1) & torch.isfinite(coefficients).all(1)
    pool_valid &= ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
    pool_valid &= torch.isfinite(scores) & (scores >= 0) & (scores <= 1)
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    axis = (torch.arange(SIDE, device=device, dtype=torch.float32) + 0.5) / SIDE

    with torch.autocast(device_type=device.type, enabled=False):
        for i in torch.where(target_valid)[0].tolist():
            if not bool(torch.isfinite(A[i]).all() & torch.isfinite(c0[i]).all()):
                raise ValueError(f"Nonfinite A/c0 for valid target row {i}")
            box = targets[i]
            intersection = (torch.minimum(box[2:], boxes[:, 2:]) -
                            torch.maximum(box[:2], boxes[:, :2])).clamp_min(0).prod(1)
            area = (box[2:] - box[:2]).prod()
            iou = intersection / (area + areas - intersection).clamp_min(1e-12)
            eligible = pool_valid & (pool_raw != target_raw[i]) & (iou > 0)
            ix = torch.where(eligible)[0]
            if not len(ix):
                continue
            ix = ix[torch.argsort(pool_raw[ix], stable=True)]
            merit = scores[ix] * iou[ix]
            ix = ix[torch.argsort(merit, descending=True, stable=True)[:MAX_NEIGHBOURS]]
            count = len(ix)
            counts[i] = count
            same[i] = (pool_class[ix] == target_class[i]).sum()
            chosen_raw[i, :count] = pool_raw[ix]

            xx = box[0] + axis * (box[2] - box[0])
            yy = box[1] + axis * (box[3] - box[1])
            neighbour_boxes = boxes[ix]
            inside_x = (xx[None, :] >= neighbour_boxes[:, 0, None]) & (
                xx[None, :] < neighbour_boxes[:, 2, None])
            inside_y = (yy[None, :] >= neighbour_boxes[:, 1, None]) & (
                yy[None, :] < neighbour_boxes[:, 3, None])
            support = inside_y[:, :, None] & inside_x[:, None, :]
            weights = scores[ix, None, None]
            a = A[i].detach().float()
            true = (a @ coefficients[ix].T).T.reshape(count, SIDE, SIDE).sigmoid()
            true = true * weights * support
            own = (a @ c0[i].detach().float()).reshape(SIDE, SIDE).sigmoid()
            own = own[None] * weights * support
            wrong = _support_roll(true, support)
            availability = support.float().sum(0) / MAX_NEIGHBOURS
            relations[0][i] = _aggregate(own, availability)
            relations[1][i] = _aggregate(true, availability)
            relations[2][i] = _aggregate(wrong, availability)
            union = support.any(0)
            difference = (relations[1][i, :2] - relations[2][i, :2]).abs().amax(0)
            changed[i] = ((difference > CHANGE_ATOL) & union).float().sum() / union.sum().clamp_min(1)

    return dict(relation_self=relations[0], relation_true=relations[1],
                relation_wrong=relations[2], neighbor_count=counts,
                same_class_count=same, changed_fraction=changed,
                chosen_raw=chosen_raw)
