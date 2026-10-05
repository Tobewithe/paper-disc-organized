"""Original Proto26 / one-to-one coefficient branches, with no new network.

P updates only native Proto26 mask-generating parameters, keeping original
candidate c0 fixed. PC also updates the original one2one_cv4. The prefix and
candidate assignment are frozen. All BN running buffers remain unchanged;
BN affine parameters follow the prior N control. Unused semseg is frozen.

This deliberately routes the SAME fixed one-to-one instance-mask objective
into the prototype branch. Official Segment26.forward detaches P for that
branch, so this is a controlled gradient-routing intervention, not a claim
to reproduce the full original one-to-many / one-to-one training objective.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("P", "PC")


class NativeProtoModel(nn.Module):
    """Train native modules on three complete frozen neck feature maps."""

    def __init__(self, replay, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Native prototype mode must be P or PC")
        self.mode = mode
        self.feature_channels = tuple(int(v) for v in replay.feature_channels)
        if len(self.feature_channels) != 3:
            raise ValueError("Require three original P3/P4/P5 feature scales")
        self.native_proto = copy.deepcopy(replay.head.proto)
        self.native_proto.requires_grad_(True)
        # Proto26.semseg is an auxiliary semantic output, not the prototype.
        # No new semantic loss or semseg updates enter this screen.
        if getattr(self.native_proto, "semseg", None) is not None:
            self.native_proto.semseg.requires_grad_(False)
        self.native_cv4 = copy.deepcopy(replay.native_cv4) if mode == "PC" else None
        if self.native_cv4 is not None:
            self.native_cv4.requires_grad_(True)
            for branch in self.native_cv4:
                last = branch[-1]
                if not (isinstance(last, nn.Conv2d) and last.in_channels == 64
                        and last.out_channels == 32 and last.kernel_size == (1, 1)):
                    raise AssertionError("Original native coefficient branch is not 64->32")
        self.train(True)
        self._frozen_buffers = {name: value.detach().cpu().clone()
                                for name, value in self.named_buffers()}

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if getattr(self.native_proto, "semseg", None) is not None:
            self.native_proto.semseg.eval()
        return self

    def assert_frozen_buffers(self):
        current = dict(self.named_buffers())
        if set(current) != set(self._frozen_buffers):
            raise AssertionError("Native buffer identity changed")
        for name, value in current.items():
            if not torch.equal(value.detach().cpu(), self._frozen_buffers[name]):
                raise AssertionError(f"Frozen native buffer changed: {name}")
        if any(module.training for module in self.modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)):
            raise AssertionError("Native BN entered train mode")
        if getattr(self.native_proto, "semseg", None) is not None:
            if any(p.requires_grad or p.grad is not None for p in self.native_proto.semseg.parameters()):
                raise AssertionError("Unused semantic branch received gradients")

    def trainable_parameter_names(self):
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def parameter_counts(self):
        proto_total = sum(p.numel() for p in self.native_proto.parameters())
        proto_trainable = sum(p.numel() for p in self.native_proto.parameters() if p.requires_grad)
        cv4_total = 0 if self.native_cv4 is None else sum(p.numel() for p in self.native_cv4.parameters())
        return dict(total=proto_total+cv4_total, trainable=proto_trainable+cv4_total,
                    native_proto=proto_total, native_proto_trainable=proto_trainable,
                    native_cv4=cv4_total, semseg_frozen=proto_total-proto_trainable,
                    new=0, mode=self.mode)

    def forward(self, features, images):
        if len(features) != 3 or any(f.ndim != 4 for f in features):
            raise ValueError("Expected three complete BCHW frozen neck maps")
        batch = len(images)
        device = features[0].device
        for feature, channels in zip(features, self.feature_channels):
            if feature.shape[:2] != (batch, channels) or feature.device != device:
                raise ValueError("Frozen feature shape/device mismatch")
        # Features are detached by design. Native branches below remain in
        # ordinary autograd even while their BN modules use frozen statistics.
        frozen = [feature.detach().float() for feature in features]
        prototype_batch = self.native_proto(frozen, return_semantic=False)
        if not torch.is_tensor(prototype_batch) or tuple(prototype_batch.shape) != (batch, 32, 160, 160):
            raise AssertionError("Expected native Proto26 tensor [B,32,160,160]")
        if self.mode == "P":
            coefficients = [image["c0"].detach().to(device=device, dtype=torch.float32) for image in images]
        else:
            maps = [branch(feature) for branch, feature in zip(self.native_cv4, frozen)]
            all_coefficients = torch.cat([value.flatten(2) for value in maps], dim=2).transpose(1, 2)
            coefficients = []
            for index, image in enumerate(images):
                raw = image["raw_ids"].detach().to(device=device)
                if raw.ndim != 1 or raw.dtype not in (torch.int32, torch.int64):
                    raise ValueError("Original raw IDs must be an integer vector")
                if bool(((raw < 0) | (raw >= all_coefficients.shape[1])).any()):
                    raise ValueError("Raw candidate identity outside native prediction grid")
                coefficients.append(all_coefficients[index, raw.long()])
        for c, image in zip(coefficients, images):
            if tuple(c.shape) != (len(image["raw_ids"]), 32):
                raise AssertionError("Fixed candidate coefficient identity/shape changed")
        return dict(coefficients=coefficients, prototypes=list(prototype_batch.unbind(0)))

    def forward_details(self, features, images):
        return self.forward(features, images)


def build_model(mode, replay, cfg=None):
    """Compatibility factory; configuration cannot silently change the model."""
    return NativeProtoModel(replay, mode)


@torch.enable_grad()
def loss_and_grad(image, coefficient, prototype, denominator):
    """Return (summed BCE, dL/dc, dL/dP160) for the unchanged screen objective.

    The scalar displayed is the sum of per-candidate official losses. Returned
    gradients divide by the effective image group's candidate denominator,
    exactly like the old fixed-P helper. GT labels stay on the full 640 canvas;
    prototype interpolation precedes official single_mask_loss, with original
    GT-box support, normalized GT-box area and segmentation gain unchanged.

    Independent leaves bound pixel-workspace memory. The caller chains both
    returned derivatives to live native outputs using torch.autograd.backward.
    In P mode c0 does not require grad: chain only the returned prototype grad.
    """
    from ultralytics.utils.loss import v8SegmentationLoss
    if denominator <= 0:
        raise ValueError("Candidate denominator must be positive")
    if tuple(prototype.shape) != (32, 160, 160):
        raise ValueError("Current full prototype, not cached selected points, required")
    if coefficient.ndim != 2 or coefficient.shape[1] != 32:
        raise ValueError("Expected current candidate coefficients [N,32]")
    device = prototype.device
    if coefficient.device != device:
        raise ValueError("Coefficient and prototype devices differ")
    n = len(coefficient)
    masks = image["masks"].to(device)
    owners = image["owners"].to(device)
    boxes = image["target_boxes"].to(device)
    if tuple(masks.shape) != (640, 640) or len(owners) != n or tuple(boxes.shape) != (n, 4):
        raise ValueError("Full official labels/candidate support shape changed")
    gradient_c = torch.zeros_like(coefficient)
    gradient_p = torch.zeros_like(prototype)
    if n == 0:
        return 0., gradient_c, gradient_p
    prototype_leaf = prototype.detach().float().requires_grad_(True)
    expanded = F.interpolate(prototype_leaf[None], (640, 640),
                             mode="bilinear", align_corners=False)[0]
    total = 0.
    for lo in range(0, n, 8):
        hi = min(lo+8, n)
        coefficient_leaf = coefficient[lo:hi].detach().requires_grad_(True)
        gt = (masks[None] == (owners[lo:hi]+1)[:, None, None]).float()
        box = boxes[lo:hi]
        area = ((box[:, 2:]-box[:, :2])/640).prod(1)
        loss = v8SegmentationLoss.single_mask_loss(
            gt, coefficient_leaf, expanded, box, area)*float(image["segmentation_gain"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite original instance mask objective")
        gc, gp = torch.autograd.grad(loss/denominator, (coefficient_leaf, prototype_leaf),
                                     retain_graph=(hi < n))
        if not torch.isfinite(gc).all() or not torch.isfinite(gp).all():
            raise FloatingPointError("Nonfinite native prototype/coefficient derivative")
        gradient_c[lo:hi] = gc
        gradient_p.add_(gp)
        total += float(loss.detach())
    return total, gradient_c, gradient_p


@torch.enable_grad()
def assert_fixed_prototype_loss_equivalence(image, coefficient, denominator,
                                           atol=3e-5, rtol=3e-5):
    """Server smoke helper: same scalar and coefficient gradient as prior N.

    No model or optimizer is constructed. Independently compare both c and P
    derivatives with one unchunked official single_mask_loss autograd call.
    """
    from runtime_utils import loss_and_grad as fixed_loss_and_grad
    device = coefficient.device
    payload = {key: image[key].to(device) if torch.is_tensor(image[key]) else image[key]
               for key in ("proto", "masks", "owners", "target_boxes", "segmentation_gain")}
    old_total, old_gc = fixed_loss_and_grad(payload, coefficient, denominator)
    total, gc, gp = loss_and_grad(payload, coefficient, payload["proto"], denominator)
    torch.testing.assert_close(coefficient.new_tensor(total), coefficient.new_tensor(old_total), atol=atol, rtol=rtol)
    torch.testing.assert_close(gc, old_gc, atol=atol, rtol=rtol)
    from ultralytics.utils.loss import v8SegmentationLoss
    direct_c = coefficient.detach().clone().requires_grad_(True)
    direct_p = payload["proto"].detach().float().clone().requires_grad_(True)
    direct_expanded = F.interpolate(direct_p[None], (640, 640),
                                   mode="bilinear", align_corners=False)[0]
    direct_gt = (payload["masks"][None] == (payload["owners"]+1)[:, None, None]).float()
    direct_boxes = payload["target_boxes"]
    direct_area = ((direct_boxes[:, 2:]-direct_boxes[:, :2])/640).prod(1)
    direct_loss = v8SegmentationLoss.single_mask_loss(
        direct_gt, direct_c, direct_expanded, direct_boxes, direct_area
    )*float(payload["segmentation_gain"])
    direct_gc, direct_gp = torch.autograd.grad(direct_loss/denominator, (direct_c, direct_p))
    torch.testing.assert_close(coefficient.new_tensor(total), direct_loss.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(gc, direct_gc, atol=atol, rtol=rtol)
    torch.testing.assert_close(gp, direct_gp, atol=atol, rtol=rtol)
    return dict(passed=True, atol=atol, rtol=rtol, old_total=float(old_total), new_total=float(total),
                scalar_absolute_error=abs(total-old_total),
                coefficient_gradient_max_absolute_error=float((gc-old_gc).abs().max()) if gc.numel() else 0.,
                prototype_gradient_norm=float(gp.norm()),
                direct_unchunked_total=float(direct_loss.detach()),
                direct_unchunked_scalar_absolute_error=abs(total-float(direct_loss.detach())),
                direct_unchunked_coefficient_gradient_max_absolute_error=float((gc-direct_gc).abs().max()) if gc.numel() else 0.,
                direct_unchunked_prototype_gradient_max_absolute_error=float((gp-direct_gp).abs().max()),
                direct_unchunked_both_gradients_equivalent=True)
