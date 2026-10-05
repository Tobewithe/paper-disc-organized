"""Mathematically rigorous dual-branch CCL loss for YOLO26 / E2E instance segmentation.

Key requirements strictly implemented:
1. Reuses official Task-Aligned / Hungarian assignment from both branches.
2. In One-to-Many: candidate coefficients for the same GT instance are aggregated by mean;
   penalties are ONLY applied between different GT instances.
3. In One-to-One: each candidate corresponds to a distinct GT instance; penalties applied between distinct GTs.
4. Preserves E2ELoss dynamic loss weighting schedule (self.o2m from 0.8 to 0.1).
5. Exact zero-weight equivalence: when ccl_weight == 0, loss and gradients are bitwise equivalent to stock E2ELoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou

class CCLBranchSegmentationLoss(v8SegmentationLoss):
    """Subclass of v8SegmentationLoss that intercepts target assignment for CCL computation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccl_weight = 0.0
        self.ccl_margin = 0.1
        self.ccl_iou = 0.05
        self._last_assignment = None

    def get_assigned_targets_and_loss(self, preds, batch):
        result = super().get_assigned_targets_and_loss(preds, batch)
        if self.ccl_weight > 0:
            # result[0] is (fg_mask, target_gt_idx, target_bboxes, target_scores, target_scores_sum)
            fg_mask, target_gt_idx = result[0][0], result[0][1]
            self._last_assignment = (fg_mask.detach(), target_gt_idx.detach())
        return result

    def compute_ccl_penalty(self, coeff, fg_mask, target_gt_idx, batch):
        """Compute pairwise cosine penalty between distinct GT instances."""
        coeff = coeff.float()
        penalties = []
        batch_size = coeff.shape[0]
        
        for b in range(batch_size):
            sel = (batch['batch_idx'].view(-1) == b)
            num_gt = int(sel.sum())
            if num_gt < 2 or not fg_mask[b].any():
                continue
                
            assigned = target_gt_idx[b][fg_mask[b]]
            unique_gt, inverse, counts = assigned.unique(return_inverse=True, return_counts=True)
            if len(unique_gt) < 2:
                continue
                
            # Aggregate coefficients by GT instance (mean pooling of multiple candidates)
            means = torch.zeros(len(unique_gt), coeff.shape[-1], device=coeff.device, dtype=coeff.dtype)
            means.index_add_(0, inverse, coeff[b][fg_mask[b]])
            means = means / counts[:, None].float()
            
            # Check bounding box overlap between GTs in this image
            boxes = xywh2xyxy(batch['bboxes'][sel][unique_gt].float())
            eligible = torch.triu(box_iou(boxes, boxes, eps=1e-6) > self.ccl_iou, diagonal=1)
            row, col = eligible.nonzero(as_tuple=True)
            
            if len(row):
                cosine = F.cosine_similarity(means[row], means[col], dim=1)
                penalties.append(F.relu(cosine - self.ccl_margin))
                
        if penalties:
            return torch.cat(penalties)
        return coeff.new_zeros(0)

    def loss(self, preds, batch):
        loss_array, loss_items = super().loss(preds, batch)
        if self.ccl_weight <= 0 or self._last_assignment is None:
            self._last_assignment = None
            return loss_array, loss_items
            
        fg_mask, target_gt_idx = self._last_assignment
        self._last_assignment = None
        
        coeff = preds['mask_coefficient'].permute(0, 2, 1)
        with torch.autocast(device_type=coeff.device.type, enabled=False):
            penalties = self.compute_ccl_penalty(coeff, fg_mask, target_gt_idx, batch)
            
        if len(penalties) > 0:
            ccl_loss = penalties.mean() * self.ccl_weight
            # Add to seg_loss (index 1) scaled by batch size
            loss_array[1] = loss_array[1] + ccl_loss * coeff.shape[0]
            if isinstance(loss_items, dict) and 'seg_loss' in loss_items:
                loss_items['seg_loss'] = loss_items['seg_loss'] + ccl_loss.detach()
            elif isinstance(loss_items, torch.Tensor) and len(loss_items) > 1:
                loss_items[1] = loss_items[1] + ccl_loss.detach()
                
        return loss_array, loss_items


class DualBranchCCLLoss(E2ELoss):
    """E2ELoss that integrates CCL into One-to-Many and/or One-to-One branches."""
    def __init__(self, model: torch.nn.Module, ccl_weight_o2m: float = 0.1, ccl_weight_o2o: float = 0.1):
        super().__init__(model, loss_fn=CCLBranchSegmentationLoss)
        self.ccl_weight_o2m = float(ccl_weight_o2m)
        self.ccl_weight_o2o = float(ccl_weight_o2o)
        
        self.one2many.ccl_weight = self.ccl_weight_o2m
        self.one2one.ccl_weight = self.ccl_weight_o2o

    def __call__(self, preds, batch):
        # Explicit forward calling with stock schedule
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        
        # Weighted combination according to dynamic schedule
        combined_loss = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o
        return combined_loss, loss_one2one[1]
