"""Unconfounded Dual-Branch Resilient Loss for Instance Segmentation.

Key mathematical formulation:
L_total = L_orig + lambda_margin * L_margin

Where:
1. L_orig: Standard Ultralytics instance segmentation loss evaluated strictly
   inside the ground truth bounding box B_orig. This maintains 100% of the
   intra-box positive/negative supervision strength (no downweighting by 1/(1+2*alpha)^2).
2. L_margin: Negative supervision evaluated exclusively on the expanded boundary
   margin band: M_margin = B_dilated \ B_orig.
   Since ground truth labels outside B_orig are 0 for this instance, any positive
   activation in M_margin is penalized as false-positive leakage.
3. lambda_margin: Explicit hyperparameter controlling margin penalty strength.
4. When alpha_train == 0 or lambda_margin == 0, strictly bitwise/numerically identical to stock Ultralytics E2ELoss.
"""
import os
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import xyxy2xywh, crop_mask

class ResilientUnconfoundedSegmentationLoss(v8SegmentationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_train = float(os.environ.get("ALPHA_TRAIN", "0.0"))
        self.alpha_guard = float(os.environ.get("ALPHA_GUARD", "0.0"))
        self.lambda_margin = float(os.environ.get("LAMBDA_MARGIN", "1.0"))
        self.weight_scale = float(os.environ.get("WEIGHT_SCALE", "1.0"))

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate unconfounded instance segmentation loss."""
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize target bboxes to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea_orig = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Coordinates on prototype mask resolution
        mxyxy_orig = target_bboxes_normalized * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=proto.device
        )

        if self.alpha_train > 0 and self.lambda_margin > 0:
            bw = mxyxy_orig[..., 2] - mxyxy_orig[..., 0]
            bh = mxyxy_orig[..., 3] - mxyxy_orig[..., 1]
            x1 = torch.clamp(mxyxy_orig[..., 0] - self.alpha_train * bw, min=0.0)
            y1 = torch.clamp(mxyxy_orig[..., 1] - self.alpha_train * bh, min=0.0)
            x2 = torch.clamp(mxyxy_orig[..., 2] + self.alpha_train * bw, max=float(mask_w))
            y2 = torch.clamp(mxyxy_orig[..., 3] + self.alpha_train * bh, max=float(mask_h))
            mxyxy_dil = torch.stack([x1, y1, x2, y2], dim=-1)

            # Normalized margin area
            w_dil_norm = (x2 - x1) / float(mask_w)
            h_dil_norm = (y2 - y1) / float(mask_h)
            marea_dil = w_dil_norm * h_dil_norm

            if self.alpha_guard > 0:
                gx1 = torch.clamp(mxyxy_orig[..., 0] - self.alpha_guard * bw, min=0.0)
                gy1 = torch.clamp(mxyxy_orig[..., 1] - self.alpha_guard * bh, min=0.0)
                gx2 = torch.clamp(mxyxy_orig[..., 2] + self.alpha_guard * bw, max=float(mask_w))
                gy2 = torch.clamp(mxyxy_orig[..., 3] + self.alpha_guard * bh, max=float(mask_h))
                mxyxy_guard = torch.stack([gx1, gy1, gx2, gy2], dim=-1)
                gw_norm = (gx2 - gx1) / float(mask_w)
                gh_norm = (gy2 - gy1) / float(mask_h)
                marea_guard = gw_norm * gh_norm
                marea_margin = (marea_dil - marea_guard).clamp(min=1e-6)
            else:
                mxyxy_guard = None
                marea_margin = (marea_dil - marea_orig).clamp(min=1e-6)
        else:
            mxyxy_dil = None
            mxyxy_guard = None
            marea_margin = None

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy_orig, marea_orig)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_orig_i, marea_orig_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                pred_mask = torch.einsum("in,nhw->ihw", pred_masks_i[fg_mask_i], proto_i)
                bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

                # Original stock loss inside B_orig
                loss_orig_crop = crop_mask(bce_loss.clone(), mxyxy_orig_i[fg_mask_i])
                loss_orig = (loss_orig_crop.mean(dim=(1, 2)) / marea_orig_i[fg_mask_i]).sum()
                if self.weight_scale != 1.0:
                    loss_orig = loss_orig * self.weight_scale

                # Unconfounded margin loss
                if mxyxy_dil is not None:
                    mxyxy_dil_i = mxyxy_dil[i][fg_mask_i]
                    marea_margin_i = marea_margin[i][fg_mask_i]
                    loss_dil_crop = crop_mask(bce_loss.clone(), mxyxy_dil_i)
                    if mxyxy_guard is not None:
                        loss_inner_crop = crop_mask(bce_loss.clone(), mxyxy_guard[i][fg_mask_i])
                    else:
                        loss_inner_crop = loss_orig_crop
                    loss_margin_crop = loss_dil_crop - loss_inner_crop
                    loss_margin = (loss_margin_crop.mean(dim=(1, 2)) / marea_margin_i).sum()
                    loss += loss_orig + self.lambda_margin * loss_margin
                else:
                    loss += loss_orig
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()


class DualBranchUnconfoundedLoss(E2ELoss):
    """E2ELoss with ResilientUnconfoundedSegmentationLoss in both branches."""
    def __init__(self, model: torch.nn.Module, alpha_train: float = 0.0, lambda_margin: float = 1.0, alpha_guard: float = 0.0):
        super().__init__(model, loss_fn=ResilientUnconfoundedSegmentationLoss)
        self.alpha_train = float(alpha_train)
        self.alpha_guard = float(alpha_guard)
        self.lambda_margin = float(lambda_margin)
        self.one2many.alpha_train = self.alpha_train
        self.one2many.alpha_guard = self.alpha_guard
        self.one2many.lambda_margin = self.lambda_margin
        self.one2one.alpha_train = self.alpha_train
        self.one2one.alpha_guard = self.alpha_guard
        self.one2one.lambda_margin = self.lambda_margin

    def __call__(self, preds, batch):
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        
        combined_loss = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o
        return combined_loss, loss_one2one[1]
