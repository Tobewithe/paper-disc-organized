"""Trainable instance-boundary ownership loss for YOLO segmentation."""
import os
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss


class BoundaryOwnershipLoss(v8SegmentationLoss):
    """Stock segmentation loss plus self/outside/neighbor ownership terms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_self = float(os.environ.get("BOUNDARY_LAMBDA_SELF", "0"))
        self.lambda_out = float(os.environ.get("BOUNDARY_LAMBDA_OUT", "0"))
        self.lambda_nbr = float(os.environ.get("BOUNDARY_LAMBDA_NBR", "0"))
        self.stats = {"calls": 0, "targets": 0, "self": 0.0, "out": 0.0, "nbr": 0.0}

    @staticmethod
    def _mean_or_zero(values, device):
        return values.mean() if values.numel() else torch.zeros((), device=device)

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch_idx, proto, pred_masks, imgsz):
        base = super().calculate_segmentation_loss(
            fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
        )
        if self.lambda_self == 0 and self.lambda_out == 0 and self.lambda_nbr == 0:
            return base

        _, _, mh, mw = proto.shape
        total = torch.zeros((), device=proto.device, dtype=proto.dtype)
        sums = {"self": 0.0, "out": 0.0, "nbr": 0.0, "targets": 0}
        all_batch_idx = batch_idx.view(-1).long()
        # target_bboxes are in the model mask coordinate system after normalization.
        norm = imgsz[[1, 0, 1, 0]].to(proto.device, dtype=target_bboxes.dtype)
        boxes = target_bboxes / norm
        boxes = boxes * torch.tensor([mw, mh, mw, mh], device=proto.device, dtype=boxes.dtype)
        yy = torch.arange(mh, device=proto.device).view(mh, 1)
        xx = torch.arange(mw, device=proto.device).view(1, mw)

        for bi, (fg_i, idx_i, pm_i, p_i, bx_i) in enumerate(
            zip(fg_mask, target_gt_idx, pred_masks, proto, boxes)
        ):
            if not fg_i.any():
                continue
            img_gt = all_batch_idx == bi
            if self.overlap:
                n_gt = int(img_gt.sum())
                gt_all = torch.stack([(masks[bi] == (k + 1)).float() for k in range(n_gt)]) if n_gt else masks.new_zeros((0, mh, mw))
            else:
                gt_all = masks[img_gt].float()
            if gt_all.numel() == 0:
                continue
            cls_all = None
            # Class labels are only used for the same-class exclusion region.
            # batch_idx indexes the flattened GT list in the original batch.
            # It is safe to read cls from the caller batch only when attached below.
            cls_all = getattr(self, "_current_cls", None)
            cls_img = cls_all[img_gt].view(-1).long() if cls_all is not None else None
            for local_pos, gi in zip(fg_i.nonzero(as_tuple=False).view(-1), idx_i[fg_i].long()):
                gi = int(gi)
                if gi < 0 or gi >= len(gt_all):
                    continue
                own = gt_all[gi] > 0.5
                logit = torch.einsum("chw,c->hw", p_i, pm_i[local_pos])
                b = bx_i[local_pos]
                x1 = int(torch.floor(b[0]).clamp(0, mw - 1)); y1 = int(torch.floor(b[1]).clamp(0, mh - 1))
                x2 = int(torch.ceil(b[2]).clamp(x1 + 1, mw)); y2 = int(torch.ceil(b[3]).clamp(y1 + 1, mh))
                in_box = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
                union = gt_all.any(0)
                if cls_img is not None and gi < len(cls_img):
                    same = (cls_img == cls_img[gi]); same[gi] = False
                    nbr = gt_all[same].any(0) if same.any() else torch.zeros_like(union)
                else:
                    nbr = torch.zeros_like(union)
                # Self term: preserve positive pixels and the target-box interior.
                self_pos = F.binary_cross_entropy_with_logits(logit[own], torch.ones_like(logit[own])) if own.any() else logit.sum() * 0
                self_neg_mask = in_box & ~own
                self_neg = F.binary_cross_entropy_with_logits(logit[self_neg_mask], torch.zeros_like(logit[self_neg_mask])) if self_neg_mask.any() else logit.sum() * 0
                out_mask = (~in_box) & (~union)
                out_loss = F.binary_cross_entropy_with_logits(logit[out_mask], torch.zeros_like(logit[out_mask])) if out_mask.any() else logit.sum() * 0
                nbr_loss = F.binary_cross_entropy_with_logits(logit[nbr], torch.zeros_like(logit[nbr])) if nbr.any() else logit.sum() * 0
                extra = self.lambda_self * (self_pos + self_neg) + self.lambda_out * out_loss + self.lambda_nbr * nbr_loss
                total = total + extra
                sums["self"] += float((self_pos + self_neg).detach()); sums["out"] += float(out_loss.detach()); sums["nbr"] += float(nbr_loss.detach()); sums["targets"] += 1
        self.stats["calls"] += 1; self.stats["targets"] += sums["targets"]
        for key in ("self", "out", "nbr"): self.stats[key] += sums[key]
        return base + total / max(sums["targets"], 1)

    def loss(self, preds, batch):
        # Make flattened class labels available to calculate_segmentation_loss.
        self._current_cls = batch["cls"].detach().view(-1)
        try:
            return super().loss(preds, batch)
        finally:
            self._current_cls = None


def install_boundary():
    import ultralytics.nn.tasks
    import ultralytics.utils.loss
    ultralytics.nn.tasks.v8SegmentationLoss = BoundaryOwnershipLoss
    ultralytics.utils.loss.v8SegmentationLoss = BoundaryOwnershipLoss
