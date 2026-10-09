"""Scale-conditioned mask decoding, with the official input-grid order preserved.

No GT enters this module. The original-image area gate measures the exported
official binary mask, not a GT box or a segmentation annotation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def input_logits(proto, coefficients, boxes, shape):
    channels, height, width = proto.shape
    if not len(coefficients):
        return proto.new_zeros((0, *shape))
    logits = (coefficients @ proto.float().view(channels, -1)).view(-1, height, width)
    logits = F.interpolate(logits[None], shape, mode="bilinear")[0]
    return ops.crop_mask(logits, boxes)


def export_masks(binary, original_shape):
    """Match 8.4.100 validator's scale_preds binary resize and byte conversion."""
    if not len(binary):
        return binary.new_zeros((0, *original_shape), dtype=torch.uint8)
    return ops.scale_masks(binary[None], original_shape)[0].byte()


def apply_threshold(logits, areas, mode="gated", area_gate=2304.0, threshold=0.75):
    """Return binary masks on the same grid as logits, without host transfers."""
    if mode == "gated":
        values = (areas < area_gate).to(logits.dtype) * threshold
    elif mode == "smooth":
        # Fixed smooth comparator: tau = .75 / (1 + (A/2304)^2).
        values = threshold / (1.0 + (areas / area_gate).square())
    elif mode == "global":
        values = torch.full_like(areas, threshold, dtype=logits.dtype)
    elif mode == "none":
        values = torch.zeros_like(areas, dtype=logits.dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if threshold < 0:
        raise ValueError("Nonnegative thresholds required for zero-filled crop exterior")
    return (logits > values[:, None, None]).to(torch.uint8)


def process_mask_calibrated(proto, coefficients, boxes, shape, original_shape,
                            mode="gated", area_gate=2304.0, threshold=0.75):
    """Drop-in input-grid decoder when original_shape is available to predictor.

    The tensor contract is the same as process_mask(..., upsample=True).
    A zero threshold is pixel-identical to that function on the same device.
    Original-grid export is used only to measure predicted area.
    """
    logits = input_logits(proto, coefficients, boxes, shape)
    baseline = (logits > 0).byte()
    if threshold == 0 or mode == "none" or not len(baseline):
        return baseline
    areas = export_masks(baseline, original_shape).sum((1, 2)).float()
    return apply_threshold(logits, areas, mode, area_gate, threshold)


def erode_small(binary, areas, area_gate=2304.0):
    """Fixed 3x3 erosion on the original grid; exterior treated as background."""
    selected = areas < area_gate
    result = binary.clone()
    if selected.any():
        values = binary[selected, None].float()
        padded = F.pad(values, (1, 1, 1, 1), value=0)
        result[selected] = (-F.max_pool2d(-padded, 3, stride=1))[:, 0].byte()
    return result


def make_calibrated_predictor(mode="gated", area_gate=2304.0, threshold=0.75):
    """Use with YOLO(weights).predict(..., predictor=make_calibrated_predictor()).

    This factory follows the default non-retina predictor output contract.
    Threshold/area parameters are fixed before evaluation, never fitted here.
    """
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    class CalibratedSegmentationPredictor(SegmentationPredictor):
        def construct_result(self, pred, img, orig_img, img_path, proto):
            if self.args.retina_masks:
                raise ValueError("This calibrated predictor implements retina_masks=False; use a separately evaluated native decoder for retina masks")
            if not len(pred):
                masks = None
            else:
                masks = process_mask_calibrated(proto, pred[:,6:], pred[:,:4],
                    img.shape[2:], orig_img.shape[:2], mode, area_gate, threshold)
                pred[:,:4] = ops.scale_boxes(img.shape[2:], pred[:,:4], orig_img.shape)
                keep = masks.amax((-2,-1)) > 0
                if not (all(keep) or getattr(self,"_feats",None) is not None):
                    pred, masks = pred[keep], masks[keep]
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:,:6], masks=masks)

    return CalibratedSegmentationPredictor
