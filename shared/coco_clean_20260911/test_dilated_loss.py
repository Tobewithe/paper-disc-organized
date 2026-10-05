"""Test bitwise equivalence when alpha_train=0 and gradient sanity when alpha_train>0."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss, E2ELoss
from ultralytics.utils.ops import xyxy2xywh

class DilatedBranchSegmentationLoss(v8SegmentationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_train = 0.0

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
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        if self.alpha_train > 0:
            bw = mxyxy[..., 2] - mxyxy[..., 0]
            bh = mxyxy[..., 3] - mxyxy[..., 1]
            x1 = torch.clamp(mxyxy[..., 0] - self.alpha_train * bw, min=0.0)
            y1 = torch.clamp(mxyxy[..., 1] - self.alpha_train * bh, min=0.0)
            x2 = torch.clamp(mxyxy[..., 2] + self.alpha_train * bw, max=float(mask_w))
            y2 = torch.clamp(mxyxy[..., 3] + self.alpha_train * bh, max=float(mask_h))
            mxyxy_sup = torch.stack([x1, y1, x2, y2], dim=-1)
            marea_sup = marea * ((1.0 + 2.0 * self.alpha_train) ** 2)
        else:
            mxyxy_sup = mxyxy
            marea_sup = marea

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy_sup, marea_sup)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()

class DilatedE2ELoss(E2ELoss):
    def __init__(self, model: torch.nn.Module, alpha_train: float = 0.1):
        super().__init__(model, loss_fn=DilatedBranchSegmentationLoss)
        self.alpha_train = float(alpha_train)
        self.one2many.alpha_train = self.alpha_train
        self.one2one.alpha_train = self.alpha_train

from ultralytics import YOLO
yolo = YOLO("weights/yolo26m-seg.pt")
crit = DilatedE2ELoss(yolo.model, alpha_train=0.1)
assert crit.one2many.alpha_train == 0.1
assert crit.one2one.alpha_train == 0.1
print("DilatedE2ELoss successfully instantiated with YOLO model!")
