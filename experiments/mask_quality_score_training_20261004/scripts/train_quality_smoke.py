import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import ultralytics.nn.tasks as tasks
from ultralytics.utils.loss import E2ELoss as OfficialE2ELoss
from ultralytics.utils.ops import crop_mask


class MaskQualityE2ELoss(OfficialE2ELoss):
    """Official E2E loss plus a detached soft-mask-IoU target for class scores."""

    quality_weight = 0.10
    total_quality_seen = 0
    total_quality_batches = 0
    total_quality_sum = 0.0

    def __init__(self, model, loss_fn=None):
        super().__init__(model, loss_fn=loss_fn or tasks.v8SegmentationLoss)
        self.quality_seen = 0
        self.quality_sum = 0.0
        self.quality_count = 0

    @staticmethod
    def _image_gt_classes(batch, image_index, device):
        bi = batch["batch_idx"].view(-1).to(device)
        cls = batch["cls"].view(-1).to(device).long()
        return cls[bi == image_index]

    def quality_loss(self, preds, batch):
        if self.quality_seen == 0:
            print("QUALITY_HOOK_ACTIVE", flush=True)
        assigned, _, _ = self.one2one.get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx, _, _, _ = assigned
        coeff = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]
        if isinstance(proto, tuple):
            proto = proto[0]
        scores = preds["scores"].permute(0, 2, 1).contiguous()
        masks = batch["masks"].to(proto.device).float()
        bsz, _, ph, pw = proto.shape
        if tuple(masks.shape[-2:]) != (ph, pw):
            masks = F.interpolate(masks[:, None], (ph, pw), mode="nearest")[:, 0]
        values = []
        score_values = []
        with torch.no_grad():
            image_classes = [self._image_gt_classes(batch, b, proto.device) for b in range(bsz)]
        # Quality is measured on the cropped binary mask used in prediction,
        # not the uncropped global response. The target is detached.
        with torch.no_grad():
            anchors, strides = assigned[3:]
            pred_boxes = self.one2one.bbox_decode(anchors, preds["boxes"].permute(0, 2, 1)) * strides
            image_hw = torch.tensor(preds["feats"][0].shape[2:], device=proto.device) * self.one2one.stride[0]
            box_scale = torch.tensor([pw, ph, pw, ph], device=proto.device) / image_hw[[1, 0, 1, 0]]
        for b in range(bsz):
            pos = torch.where(fg_mask[b])[0]
            if pos.numel() == 0:
                continue
            gt_idx = target_gt_idx[b, pos].long()
            gt = torch.stack([(masks[b] == (i + 1)).float() for i in gt_idx])
            with torch.no_grad():
                logits = torch.einsum("nc,chw->nhw", coeff[b, pos].float(), proto[b].float())
                binary = crop_mask((logits > 0).float(), pred_boxes[b, pos].float() * box_scale)
                inter = (binary * gt).flatten(1).sum(1)
                union = (binary + gt - binary * gt).flatten(1).sum(1).clamp_min(1e-6)
                quality = (inter / union).clamp(0.0, 1.0).detach()
            classes = image_classes[b][gt_idx]
            logit = scores[b, pos, classes]
            values.append(F.binary_cross_entropy_with_logits(logit, quality, reduction="none"))
            score_values.append(quality)
        if not values:
            return coeff.sum() * 0.0
        loss_vec = torch.cat(values)
        q = torch.cat(score_values)
        self.quality_seen += int(q.numel())
        self.quality_sum += float(q.detach().sum().cpu())
        self.quality_count += 1
        type(self).total_quality_seen += int(q.numel())
        type(self).total_quality_sum += float(q.detach().sum().cpu())
        type(self).total_quality_batches += 1
        return loss_vec.mean()

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            return super().__call__(preds, batch)
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        qloss = self.quality_loss(one2one, batch)
        total = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o
        total = total.clone()
        total[2] += self.quality_weight * qloss * one2one["boxes"].shape[0]
        if not torch.isfinite(total).all():
            raise FloatingPointError("non-finite quality-augmented loss")
        return total, loss_one2one[1]


_official_seg_init_criterion = tasks.SegmentationModel.init_criterion


def _quality_init_criterion(self):
    if getattr(self, "end2end", False):
        return MaskQualityE2ELoss(self, tasks.v8SegmentationLoss)
    return _official_seg_init_criterion(self)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="quality_smoke")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--quality-weight", type=float, default=0.1)
    args = ap.parse_args()

    tasks.E2ELoss = MaskQualityE2ELoss
    MaskQualityE2ELoss.quality_weight = args.quality_weight
    tasks.SegmentationModel.init_criterion = _quality_init_criterion
    from ultralytics import YOLO
    assert __import__("ultralytics").__version__ == "8.4.100", "Fixed runtime is required"

    model = YOLO(args.weights)
    criterion = getattr(model.model, "criterion", None)
    print(json.dumps({
        "ultralytics": __import__("ultralytics").__version__,
        "end2end": bool(getattr(model.model, "end2end", False)),
        "criterion": type(criterion).__name__ if criterion is not None else None,
        "quality_weight": MaskQualityE2ELoss.quality_weight,
    }, ensure_ascii=False), flush=True)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=0,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=False,
        verbose=False,
        cache=False,
        pretrained=args.weights,
        save=True,
        val=False,
    )
    criterion = getattr(model.model, "criterion", None)
    print(json.dumps({
        "quality_seen": int(MaskQualityE2ELoss.total_quality_seen),
        "quality_batches": int(MaskQualityE2ELoss.total_quality_batches),
        "quality_mean_target": (float(MaskQualityE2ELoss.total_quality_sum) / max(int(MaskQualityE2ELoss.total_quality_seen), 1)),
    }, ensure_ascii=False), flush=True)
    print("QUALITY_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
