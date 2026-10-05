"""Original CCL formula; stock batches, reused assignment, no random logging.

The source annotation mapping is audited before training. During stock spatial
augmentation CCL uses the official transformed target index, as the original
method does. It never treats individual polygon components as separate targets.
"""
import os
import math
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou

class CCLSegmentationLoss(v8SegmentationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccl_weight = float(os.environ.get('CCL_WEIGHT', '0'))
        if not math.isfinite(self.ccl_weight) or self.ccl_weight < 0:
            raise ValueError('CCL_WEIGHT must be finite and nonnegative')
        self.ccl_margin = 0.1
        self.ccl_iou = 0.05
        self.ccl_stats = dict(calls=0, pairs=0, active_pairs=0, raw_sum=0.0)
        self.last_pairs = 0
        self.last_raw = 0.0

    def get_assigned_targets_and_loss(self, preds, batch):
        result = super().get_assigned_targets_and_loss(preds, batch)
        if self.ccl_weight:
            self._ccl_assignment = (result[0][0].detach(), result[0][1].detach())
        return result

    def pair_penalties(self, coeff, fg_mask, target_gt_idx, batch):
        # Reduce in FP32 even under BF16 model autocast. Otherwise summing many
        # candidate coefficients in BF16 loses precision before cosine similarity.
        coeff = coeff.float()
        penalties = []
        for b in range(coeff.shape[0]):
            sel = batch['batch_idx'].view(-1) == b
            num_gt = int(sel.sum())
            if num_gt<2 or not fg_mask[b].any(): continue
            assigned = target_gt_idx[b][fg_mask[b]]
            unique, inverse, counts = assigned.unique(return_inverse=True, return_counts=True)
            if len(unique)<2: continue
            assert unique.min()>=0 and unique.max()<num_gt
            means = torch.zeros(len(unique), coeff.shape[-1], device=coeff.device, dtype=coeff.dtype)
            means.index_add_(0, inverse, coeff[b][fg_mask[b]])
            means = means / counts[:, None].float()
            boxes = xywh2xyxy(batch['bboxes'][sel][unique].float())
            # Historical formula used epsilon=1e-6 on normalized boxes.
            eligible = torch.triu(box_iou(boxes, boxes, eps=1e-6) > self.ccl_iou, diagonal=1)
            row, col = eligible.nonzero(as_tuple=True)
            if len(row):
                assert (unique[row] != unique[col]).all()
                cosine = F.cosine_similarity(means[row], means[col], dim=1)
                penalties.append(F.relu(cosine-self.ccl_margin))
        return torch.cat(penalties) if penalties else coeff.reshape(-1)[:0]

    def loss(self, preds, batch):
        loss_array, loss_items = super().loss(preds, batch)
        if self.ccl_weight == 0: return loss_array, loss_items
        fg_mask, target_gt_idx = self._ccl_assignment
        self._ccl_assignment = None  # no batch tensor retained into checkpoint/EMA
        coeff = preds['mask_coefficient'].permute(0, 2, 1)
        with torch.autocast(device_type=coeff.device.type, enabled=False):
            values = self.pair_penalties(coeff, fg_mask, target_gt_idx, batch)
        self.ccl_stats['calls'] += 1
        self.last_pairs = len(values)
        self.last_raw = float(values.detach().mean()) if len(values) else 0.0
        if len(values):
            raw = values.mean()
            term = raw*self.ccl_weight
            loss_array[1] = loss_array[1] + term*coeff.shape[0]
            if isinstance(loss_items, dict):
                key = 'seg_loss'
                assert self.loss_names[1] == key and key in loss_items
                loss_items[key] = loss_items[key]+term.detach()
            else: loss_items[1] = loss_items[1]+term.detach()
            self.ccl_stats['pairs'] += len(values)
            self.ccl_stats['active_pairs'] += int((values.detach()>0).sum())
            self.ccl_stats['raw_sum'] += float(values.detach().sum())
        return loss_array, loss_items

def install_ccl():
    import ultralytics.nn.tasks
    import ultralytics.utils.loss
    ultralytics.nn.tasks.v8SegmentationLoss = CCLSegmentationLoss
    ultralytics.utils.loss.v8SegmentationLoss = CCLSegmentationLoss
