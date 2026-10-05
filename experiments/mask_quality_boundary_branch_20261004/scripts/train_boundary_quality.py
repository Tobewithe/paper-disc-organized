import argparse
import json
import os
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.ops import crop_mask
from ultralytics.utils.tal import make_anchors
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.head import Segment26
from ultralytics.utils.loss import E2ELoss as OfficialE2ELoss


# Class-level hooks survive the trainer's EMA/deepcopy path. Instance-bound
# closures are unsafe here because a copied head can retain the old bound self.
_ORIGINAL_SEGMENT26_FORWARD = Segment26.forward


def _proto_tensor(proto):
    return proto[0] if isinstance(proto, tuple) else proto


def _decoded_boxes(criterion, preds):
    anchor_points, stride_tensor = make_anchors(preds["feats"], criterion.one2one.stride, 0.5)
    boxes = criterion.one2one.bbox_decode(
        anchor_points, preds["boxes"].permute(0, 2, 1).contiguous()
    ) * stride_tensor
    return boxes


def _quality_features(preds, boxes, scores, proto_override=None):
    coeff = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
    if boxes.shape[1] == 4 and boxes.shape[-1] != 4:
        boxes = boxes.permute(0, 2, 1).contiguous()
    if scores.shape[1] != coeff.shape[1] and scores.shape[-1] == coeff.shape[1]:
        scores = scores.permute(0, 2, 1).contiguous()
    proto_value = proto_override if proto_override is not None else preds.get("proto")
    if proto_value is None:
        raise RuntimeError("prototype tensor is required for boundary-quality features")
    proto = _proto_tensor(proto_value)
    b, _, ph, pw = proto.shape
    img_hw = torch.tensor(preds["feats"][0].shape[2:], device=proto.device, dtype=proto.dtype) * 8
    norm = torch.stack((img_hw[1], img_hw[0], img_hw[1], img_hw[0])).view(1, 1, 4)
    box_feat = (boxes / norm).clamp(-2.0, 2.0)
    small = F.adaptive_avg_pool2d(proto, (8, 8))
    logits = torch.einsum("bac,bchw->bahw", coeff.float(), small.float()).to(coeff.dtype)
    prob = logits.sigmoid()
    entropy = -(prob.clamp(1e-5, 1 - 1e-5) * prob.clamp(1e-5, 1 - 1e-5).log() + (1 - prob).clamp(1e-5, 1 - 1e-5) * (1 - prob).clamp(1e-5, 1 - 1e-5).log()).mean((-1, -2))
    dx = (prob[..., :, 1:] - prob[..., :, :-1]).abs().mean((-1, -2))
    dy = (prob[..., 1:, :] - prob[..., :-1, :]).abs().mean((-1, -2))
    stats = torch.stack((prob.mean((-1, -2)), prob.amax((-1, -2)), entropy, dx, dy, logits.mean((-1, -2))), dim=-1)
    # scores is [B, anchors, classes] here; retain one score descriptor per anchor.
    score_stats = scores.sigmoid().amax(-1, keepdim=True)
    return torch.cat((coeff, box_feat, stats, score_stats), dim=-1)


def _boundary_quality_inference(self, x):
    dbox = self._get_decode_boxes(x)
    scores = x["scores"].sigmoid()
    q_dtype = next(self.quality_head.parameters()).dtype
    features = _quality_features(x, dbox, x["scores"], getattr(self, "_quality_proto", None)).to(q_dtype)
    quality = self.quality_head(features).squeeze(-1).sigmoid()
    scores = scores * (0.5 + 0.5 * quality.unsqueeze(1)).pow(getattr(self, "quality_alpha", 1.0))
    return torch.cat((dbox, scores, x["mask_coefficient"]), 1)


def _boundary_quality_forward(self, x):
    if not self.training:
        self._quality_proto = self.proto(x)
    try:
        return _ORIGINAL_SEGMENT26_FORWARD(self, x)
    finally:
        if not self.training and hasattr(self, "_quality_proto"):
            del self._quality_proto


def _install_class_hooks():
    if not getattr(Segment26, "_boundary_quality_hooks", False):
        Segment26._inference = _boundary_quality_inference
        Segment26.forward = _boundary_quality_forward
        Segment26._boundary_quality_hooks = True


class BoundaryQualityE2ELoss(OfficialE2ELoss):
    quality_weight = 0.10
    total_targets = 0
    epoch_quality_sum = 0.0
    epoch_quality_targets = 0

    def __init__(self, model, loss_fn=None):
        super().__init__(model, loss_fn=loss_fn or tasks.v8SegmentationLoss)
        self.head = model.model[-1]

    def _targets(self, preds, batch, fg_mask, target_gt_idx, boxes):
        proto = _proto_tensor(preds["proto"])
        masks = batch["masks"].to(proto.device).float()
        ph, pw = proto.shape[-2:]
        if tuple(masks.shape[-2:]) != (ph, pw):
            masks = F.interpolate(masks[:, None], (ph, pw), mode="nearest")[:, 0]
        out = []
        for b in range(proto.shape[0]):
            pos = torch.where(fg_mask[b])[0]
            if pos.numel() == 0:
                continue
            ids = target_gt_idx[b, pos].long()
            gt = torch.stack([(masks[b] == (idx + 1)).float() for idx in ids])
            with torch.no_grad():
                logits = torch.einsum("nc,chw->nhw", preds["mask_coefficient"].permute(0, 2, 1)[b, pos].float(), proto[b].float())
                scale = torch.tensor([pw, ph, pw, ph], device=proto.device, dtype=boxes.dtype)
                pred = crop_mask((logits.sigmoid() > 0.5).float(), boxes[b, pos].float() * scale)
                inter = (pred * gt).flatten(1).sum(1)
                union = (pred + gt - pred * gt).flatten(1).sum(1).clamp_min(1.0)
                out.append((inter / union).clamp(0.01, 0.99))
        return torch.cat(out) if out else proto.sum().reshape(1) * 0.0

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            return super().__call__(preds, batch)
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        assigned, _, _ = self.one2one.get_assigned_targets_and_loss(one2one, batch)
        fg_mask, target_gt_idx, _, _, _ = assigned
        boxes = _decoded_boxes(self, one2one)
        scores = one2one["scores"].permute(0, 2, 1).contiguous()
        features = _quality_features(one2one, boxes, scores).detach()
        quality_logits = self.head.quality_head(features).squeeze(-1)
        selected = []
        for b in range(fg_mask.shape[0]):
            pos = torch.where(fg_mask[b])[0]
            if pos.numel():
                selected.append(quality_logits[b, pos])
        if selected:
            selected_logits = torch.cat(selected)
            target = self._targets(one2one, batch, fg_mask, target_gt_idx, boxes).detach()
            qloss = F.binary_cross_entropy_with_logits(selected_logits, target)
            type(self).total_targets += int(target.numel())
            type(self).epoch_quality_sum += float(qloss.detach()) * int(target.numel())
            type(self).epoch_quality_targets += int(target.numel())
        else:
            qloss = quality_logits.sum() * 0.0
        total = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o
        total = total.clone()
        total[1] += self.quality_weight * qloss * one2one["boxes"].shape[0]
        if not torch.isfinite(total).all():
            raise FloatingPointError("non-finite boundary quality loss")
        return total, loss_one2one[1]


def _attach_quality_head_to_segmodel(seg_model, alpha=1.0):
    """Attach the branch to the actual trainer model (trainer rebuilds this model)."""
    head = seg_model.model[-1]
    _install_class_hooks()
    if hasattr(head, "quality_head"):
        return head
    head.quality_head = nn.Sequential(nn.Linear(43, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)).to(next(head.parameters()).device)
    nn.init.constant_(head.quality_head[-1].bias, 0.0)

    head.quality_alpha = alpha
    return head


def install_quality_head(model, alpha=1.0):
    return _attach_quality_head_to_segmodel(model.model, alpha=alpha)


def _quality_epoch_end(trainer):
    """Emit a per-epoch quality-head loss for convergence auditing."""
    cls = BoundaryQualityE2ELoss
    count = int(cls.epoch_quality_targets)
    mean = cls.epoch_quality_sum / count if count else None
    print(json.dumps({"quality_epoch": int(trainer.epoch) + 1,
                      "quality_bce": mean,
                      "quality_targets_epoch": count}, allow_nan=True), flush=True)
    cls.epoch_quality_sum = 0.0
    cls.epoch_quality_targets = 0


_official_init = tasks.SegmentationModel.init_criterion


def _quality_init(self):
    if getattr(self, "end2end", False):
        return BoundaryQualityE2ELoss(self, tasks.v8SegmentationLoss)
    return _official_init(self)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="boundary_quality")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--quality-weight", type=float, default=0.1)
    ap.add_argument("--save-period", type=int, default=1,
                    help="save a checkpoint every N epochs for convergence auditing")
    ap.add_argument("--freeze-original", action="store_true",
                    help="freeze all pretrained detector/segmenter parameters and train only quality_head")
    args = ap.parse_args()
    assert __import__("ultralytics").__version__ == "8.4.100"
    tasks.SegmentationModel.init_criterion = _quality_init
    # Ultralytics rebuilds a SegmentationModel inside SegmentationTrainer.get_model;
    # patch that construction point so the quality branch is present before the
    # optimizer is created and is therefore trained/saved with the backbone.
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    _trainer_get_model = SegmentationTrainer.get_model
    def _get_model_with_quality(self, cfg=None, weights=None, verbose=True):
        seg_model = _trainer_get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        _attach_quality_head_to_segmodel(seg_model, alpha=1.0)
        return seg_model
    SegmentationTrainer.get_model = _get_model_with_quality
    if args.freeze_original:
        _trainer_setup_train = SegmentationTrainer._setup_train
        def _setup_train_frozen(self):
            _trainer_setup_train(self)
            # ``requires_grad=False`` does not freeze BatchNorm running
            # statistics.  Ultralytics calls ``model.train()`` at every
            # epoch, so explicitly reuse its native freeze-layer mechanism
            # to put every original-model BatchNorm layer back in eval mode.
            # The quality head has no BatchNorm layers and remains trainable.
            self.freeze_layer_names = ["model"]
            trainable = []
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = ("quality_head" in name)
                if parameter.requires_grad:
                    trainable.append(name)
            if not trainable:
                raise RuntimeError("freeze-original left no trainable quality head parameters")
            print(json.dumps({"freeze_original": True, "trainable_parameters": trainable}), flush=True)
        SegmentationTrainer._setup_train = _setup_train_frozen
    from ultralytics import YOLO
    model = YOLO(args.weights)
    head = install_quality_head(model)
    BoundaryQualityE2ELoss.quality_weight = args.quality_weight
    model.add_callback("on_train_epoch_end", _quality_epoch_end)
    print(json.dumps({"ultralytics": __import__("ultralytics").__version__, "quality_head": sum(p.numel() for p in head.quality_head.parameters()), "end2end": bool(getattr(model.model, "end2end", False))}), flush=True)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=0, device=0, project=args.project, name=args.name, exist_ok=True, plots=False, verbose=False, cache=False, pretrained=args.weights, save=True, save_period=args.save_period, val=False, seed=0)
    print(json.dumps({"quality_targets": BoundaryQualityE2ELoss.total_targets}), flush=True)
    print("BOUNDARY_QUALITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
