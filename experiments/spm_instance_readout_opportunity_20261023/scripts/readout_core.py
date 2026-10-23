import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics.utils import ops


def valid_grid(rect, device):
    left, top, width, height = rect
    valid = np.zeros((640, 640), np.uint8)
    valid[top:top+height, left:left+width] = 1
    return torch.as_tensor(cv2.resize(valid.astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA) > .999,
                           device=device)


@torch.no_grad()
def forward_native(model, image, captured):
    captured.clear()
    det, proto = model(image)[0]
    proto = proto[0] if isinstance(proto, tuple) else proto
    det = det[0]
    head_ids = torch.arange(len(det), device=det.device)[det[:, 4] > .001]
    det = det[head_ids].clone()
    det[:, :4].clamp_(0, 640)
    assert len(det), 'No native detection in this frame'
    masks = ops.process_mask(proto[0], det[:, 6:], det[:, :4], (640, 640), upsample=True)
    keep = masks.flatten(1).any(1).bool()
    logits = (det[:, 6:]@proto[0].flatten(1)).reshape(-1, 160, 160)
    logits = ops.crop_mask(logits, det[:, :4]*.25)
    det, masks, logits, head_ids = det[keep], masks[keep], logits[keep], head_ids[keep]
    features = torch.cat([captured[2], F.interpolate(captured[16], (160, 160), mode='bilinear', align_corners=False)], 1)
    return det, masks.bool(), logits, head_ids, features


def representations(features, memory, trained, random):
    raw = ((features-memory['mean'].to(features.device).view(1, 512, 1, 1))/
           memory['std'].to(features.device).view(1, 512, 1, 1)).permute(0, 2, 3, 1)
    # This matches the trained memory coordinate, not a newly fitted normalization.
    return {'T': F.normalize(trained(raw), dim=-1)[0].permute(2, 0, 1).flatten(1),
            'U': F.normalize(random(raw), dim=-1)[0].permute(2, 0, 1).flatten(1),
            'R': F.normalize(raw, dim=-1)[0].permute(2, 0, 1).flatten(1)}


def support_scores(query, supports):
    """Cosine scores with the query pixel excluded from its own support mean."""
    flat = supports.reshape(len(supports), -1).float()
    counts = flat.sum(1)
    sums = flat@query.T
    means = F.normalize(sums, dim=1)
    scores = means@query
    for k in range(len(supports)):
        index = torch.nonzero(flat[k] > 0)[:, 0]
        if len(index) >= 2:
            q = query[:, index].T
            scores[k, index] = (F.normalize(sums[k:k+1]-q, dim=1)*q).sum(1)
    return scores, counts >= 2, counts


def decode_margin(margin, boxes):
    cropped = ops.crop_mask(margin.reshape(-1, 160, 160), boxes*.25)
    return F.interpolate(cropped[None], (640, 640), mode='bilinear', align_corners=False)[0] > 0


def predicted_supports(det, logits, native_masks, valid):
    supports = (logits.sigmoid() > .8) & valid[None]
    confident = det[:, 4] >= .25
    union = (logits[confident] > 0).any(0) if confident.any() else torch.zeros_like(valid)
    background = valid & ~union
    # IoU uses the normal decoded 640 masks; no GT mask/overlap input is available here.
    flat = native_masks.flatten(1).float()
    intersection = flat@flat.T
    area = flat.sum(1)
    iou = intersection/(area[:, None]+area[None]-intersection).clamp_min(1)
    return torch.cat([supports, background[None]]), confident, iou


def predicted_readout(query, supports, confident, iou, boxes, native_masks, wrong=False):
    scores, available, counts = support_scores(query, supports)
    n = len(boxes)
    mapping = torch.arange(n, device=query.device)
    ids = torch.nonzero(available[:n])[:, 0]
    if wrong and len(ids) > 1:
        mapping[ids] = ids.roll(1)
    output = native_masks.clone()
    changed = []
    for k in range(n):
        key = int(mapping[k])
        if not bool(available[key]) or not bool(available[-1]):
            changed.append(False)
            continue
        # Row roles stay fixed while support keys, including negative roles, follow the permutation.
        negative_rows = torch.nonzero(confident & available[:n][mapping] & (iou[k] < .5))[:, 0]
        negative_rows = negative_rows[negative_rows != k]
        negatives = torch.cat([mapping[negative_rows], mapping.new_tensor([n])])
        margin = scores[key]-scores[negatives].amax(0)
        output[k:k+1] = decode_margin(margin[None], boxes[k:k+1])
        changed.append(True)
    return output, dict(available=[bool(x) for x in available[:n]],
                       support_counts=[int(x) for x in counts[:n]],
                       readout_applied=changed, wrong_mapping=mapping.cpu().tolist(),
                       background_support=int(counts[-1]),
                       permutable_instances=int(len(ids)) if len(ids) > 1 else 0)


def gt_supports(gt_masks, rect, device):
    valid = valid_grid(rect, device)
    overlap = gt_masks.sum(0) > 1
    clear = cv2.dilate(overlap.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    valid &= torch.as_tensor(cv2.resize(clear.astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA) > .999,
                             device=device)
    purity = torch.as_tensor(np.stack([cv2.resize(m.astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA)
                                       for m in gt_masks]), device=device)
    union = gt_masks.any(0)
    bg = torch.as_tensor(cv2.resize((~union).astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA) > .99,
                         device=device) & valid
    return torch.cat([(purity >= .5) & valid[None], bg[None]])


def oracle_masks(query, supports, det, native_masks, pairs):
    scores, available, counts = support_scores(query, supports)
    result, audit = {}, {}
    valid_ids = torch.nonzero(available)[:, 0]
    for output_id, gt_local in pairs:
        usable = bool(available[gt_local]) and bool(available[-1])
        mask = native_masks[output_id]
        if usable:
            other = valid_ids[valid_ids != gt_local]
            margin = scores[gt_local]-scores[other].amax(0)
            mask = decode_margin(margin[None], det[output_id:output_id+1, :4])[0]
        result[output_id] = mask
        audit[output_id] = dict(prototype_available=usable, support_count=int(counts[gt_local]),
                                background_support_count=int(counts[-1]))
    return result, audit


def iou_mask(a, b):
    intersection = np.logical_and(a, b).sum()
    return float(intersection/max(1, np.logical_or(a, b).sum()))
