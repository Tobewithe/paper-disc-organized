"""The single preregistered change: same-ROI GT BCE -> smoothed soft IoU."""
import torch
from torch.nn import functional as F

EPSILON = 1e-6
LOSS_SPEC = {
    'name': 'same_ROI_soft_IoU', 'epsilon': EPSILON,
    'formula': '1-(sum_ROI(sigmoid(z)*g)+eps)/(sum_ROI(sigmoid(z)+g-sigmoid(z)*g)+eps)',
    'empty_GT': 'same_formula_in_original_supervised_instance_denominator',
    'empty_ROI': 'exclude_and_count_as_original', 'dtype': 'float32',
    'reduction': 'instance_mean_within_image_then_original_effective_batch4_mean',
    'preservation': 'original_BCE_all_nonempty_ROI_instances', 'preservation_weight': .1,
}


def soft_iou_grouped_loss(logits, baseline, roi, targets):
    if logits.shape != baseline.shape or logits.shape != roi.shape or roi.dtype != torch.bool:
        raise ValueError('loss grid mismatch')
    if not torch.isfinite(logits).all().item() or not torch.isfinite(baseline).all().item():
        raise ValueError('non-finite loss')
    pixels = roi.sum(dim=(1, 2))
    valid = pixels > 0
    zero = logits.sum() * 0
    # Exact same operations and reduction as the original preservation term.
    per_instance = (F.binary_cross_entropy_with_logits(logits, baseline.detach().sigmoid(), reduction='none') * roi).sum((1, 2)) / pixels.clamp_min(1)
    preservation = per_instance[valid].mean() if valid.any().item() else zero
    indices = [i for i in targets if bool(valid[i])]
    empty_gt = 0
    if indices:
        target = torch.stack([targets[i].detach().to(logits) for i in indices])
        if target.shape != logits[indices].shape or not ((target >= 0) & (target <= 1)).all().item():
            raise ValueError('invalid training target')
        probability = logits[indices].sigmoid()
        product = probability * target
        intersection = (product * roi[indices]).sum((1, 2))
        union = ((probability + target - product) * roi[indices]).sum((1, 2))
        gt = (1 - (intersection + EPSILON) / (union + EPSILON)).mean()
        empty_gt = int(((target * roi[indices]).sum((1, 2)) == 0).sum())
    else:
        gt = zero
    return gt + .1 * preservation, {
        'gt_loss': float(gt.detach()), 'preservation_loss': float(preservation.detach()),
        'matched_supervised_instances': len(indices), 'preserved_instances': int(valid.sum()),
        'empty_roi_rows': (~valid).nonzero().flatten().tolist(), 'no_output_image': len(logits) == 0,
        'empty_GT_in_ROI_instances': empty_gt,
    }
