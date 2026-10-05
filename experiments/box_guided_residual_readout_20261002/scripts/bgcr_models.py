"""Minimal box-guided residual readout and frozen feature extraction.

The same ResidualReadout is used for the point control and box-difference
arm.  Its inputs are already standardized using that arm's FIT statistics.
Each of the three levels has exactly one zero-initialized 64->32 affine
map.  The caller adds its output to the original frozen coefficients c0.

ROI extraction reads the frozen ONE-TO-ONE coefficient branch's complete
last-layer input H, not an isolated vector or a neck feature substitution.
For each query, ROIAlign samples its prediction box on its own level's H;
the 3x3 result is averaged and r = mean_H_in_box - query_h.  Sampling is
fixed: 640-input xyxy boxes, aligned=True, sampling_ratio=2.  No GT field
is inspected.  Candidate/GT identities and exact raw-h replay are audited
by the main program, outside this module.

Wrong-box control keeps query_h, the sampling level and downstream final
box unchanged.  Donors are other candidates in the same image, preferring
the same PREDICTED class, then closest predicted-box log-area/log-aspect.
They need not be different physical objects; selection does not read GT.
The returned geometric overlap makes that limitation visible.

This module performs no training, file I/O or tensor checks on import.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align


FEATURE_DIM = 64
COEFFICIENT_DIM = 32
LEVEL_COUNT = 3
INPUT_SIZE = 640
ROI_SIZE = 3
SAMPLING_RATIO = 2
EXPECTED_LEVEL_SIDES = (80, 40, 20)


class ResidualReadout(nn.Module):
    """Three independent zero-initialized affine residuals: 6,240 parameters.

    forward(features[N,64], levels[N]) -> delta[N,32].
    levels are 0/1/2 for P3/P4/P5; features are fit-standardized outside.
    Both weight and bias are trainable from the first step.  Zero output at
    initialization guarantees c0+delta == c0, including the box arm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            nn.Linear(FEATURE_DIM, COEFFICIENT_DIM) for _ in range(LEVEL_COUNT)
        )
        for layer in self.levels:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor, levels: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError("features must have shape [N,64]")
        if levels.shape != (len(features),):
            raise ValueError("levels must have shape [N]")
        if levels.device != features.device:
            raise ValueError("features and levels must be on the same device")
        if levels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("levels must be integer indices")
        if bool(((levels < 0) | (levels >= LEVEL_COUNT)).any()):
            raise ValueError("levels must be 0, 1 or 2")
        delta = features.new_zeros((len(features), COEFFICIENT_DIM))
        for level, layer in enumerate(self.levels):
            select = levels == level
            if bool(select.any()):
                delta[select] = layer(features[select])
        return delta


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _box_iou(a: list[float], b: list[float]) -> float:
    intersection = (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@torch.no_grad()
def extract_box_features(raw_full: dict[str, Any],
                         compact: dict[str, Any]) -> dict[str, Any]:
    """Extract true/wrong box residuals for ONE image without any GT inputs.

    raw_full: h[8400,64], levels[8400] in native raw-id order.
    compact: h[N,64], boxes[N,4], levels[N], rows[N].  Every row must contain
             image_id, raw_id, predicted_class_id (prediction argmax).

    Returns FP32 roi_mean/residual/wrong_residual on raw_full['h'].device,
    donor_metadata (JSON-friendly list), and extraction audit information.
    The original compact boxes and all input tensors are left unchanged.

    An empty/inverted box, including one empty after sampling-only clamp,
    is retained with roi_mean=query_h and both residuals zero.  No donor
    gives wrong_residual=residual with donor_valid=False.  The caller must
    exclude these marked controls from a true-versus-wrong paired claim,
    while preserving their identities in all other candidate evaluations.
    """
    full_h = raw_full["h"]
    full_levels = raw_full["levels"]
    query_h = compact["h"]
    boxes_original = compact["boxes"]
    levels = compact["levels"]
    rows = compact["rows"]
    if full_h.shape != (8400, FEATURE_DIM) or full_levels.shape != (8400,):
        raise ValueError("full h/levels must have shapes [8400,64]/[8400]")
    count = len(rows)
    if query_h.shape != (count, FEATURE_DIM) or boxes_original.shape != (count, 4) or levels.shape != (count,):
        raise ValueError("compact h, boxes, levels and rows have inconsistent shapes")
    for name, value in (("full_h", full_h), ("query_h", query_h),
                        ("prediction_boxes", boxes_original)):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        _finite(name, value)
    for name, value in (("full_levels", full_levels), ("levels", levels)):
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"{name} must contain integer level indices")
        if bool(((value < 0) | (value >= LEVEL_COUNT)).any()):
            raise ValueError(f"{name} must contain only 0/1/2")
    # A full feature tensor here represents one image.  Do not accidentally
    # borrow another image's query vectors while sampling this image's H.
    image_ids = [int(row["image_id"]) for row in rows]
    if len(set(image_ids)) > 1:
        raise ValueError("extract_box_features expects one image per call")
    raw_ids = [int(row["raw_id"]) for row in rows]
    predicted_classes = [int(row["predicted_class_id"]) for row in rows]
    if len(set(raw_ids)) != len(raw_ids) or any(raw < 0 or raw >= 8400 for raw in raw_ids):
        raise ValueError("compact raw_ids must be unique and in [0,8400)")

    device = full_h.device
    full_h = full_h.detach().to(dtype=torch.float32)
    full_levels = full_levels.detach().to(device=device, dtype=torch.long)
    query_h = query_h.detach().to(device=device, dtype=torch.float32)
    levels = levels.detach().to(device=device, dtype=torch.long)
    # Only this sampling copy is clamped; donor distance uses original boxes.
    boxes = boxes_original.detach().to(device=device, dtype=torch.float32)
    sample_boxes = boxes.clamp(0.0, float(INPUT_SIZE))
    raw_width_height = boxes[:, 2:] - boxes[:, :2]
    sample_width_height = sample_boxes[:, 2:] - sample_boxes[:, :2]
    raw_valid = (raw_width_height > 0).all(-1)
    valid = raw_valid & (sample_width_height > 0).all(-1)
    empty = ~valid
    h_maps, counts, sides = [], [], []
    for level in range(LEVEL_COUNT):
        # Raw candidates are flattened row-major within each native level.
        level_h = full_h[full_levels == level]
        side = math.isqrt(len(level_h))
        if side * side != len(level_h) or side != EXPECTED_LEVEL_SIDES[level]:
            raise ValueError(f"level {level} has invalid 640-input grid count {len(level_h)}")
        h_maps.append(level_h.transpose(0, 1).reshape(1, FEATURE_DIM, side, side).contiguous())
        counts.append(len(level_h))
        sides.append(side)

    roi_mean = query_h.clone()  # Empty-support policy gives exactly r=0.
    for level, feature_map in enumerate(h_maps):
        indices = torch.nonzero((levels == level) & valid, as_tuple=False).flatten()
        if len(indices):
            rois = torch.cat((sample_boxes.new_zeros((len(indices), 1)), sample_boxes[indices]), dim=1)
            aligned = roi_align(feature_map, rois, output_size=(ROI_SIZE, ROI_SIZE),
                                spatial_scale=sides[level] / INPUT_SIZE,
                                sampling_ratio=SAMPLING_RATIO, aligned=True)
            roi_mean[indices] = aligned.mean(dim=(-2, -1))
    residual = roi_mean - query_h
    wrong_residual = residual.clone()

    boxes_list = boxes.detach().cpu().tolist()
    valid_list = valid.detach().cpu().tolist()
    raw_valid_list = raw_valid.detach().cpu().tolist()
    level_list = levels.detach().cpu().tolist()
    geometry = []
    for box in boxes_list:
        width, height = box[2] - box[0], box[3] - box[1]
        geometry.append((width * height, width / height) if width > 0 and height > 0 else None)
    donors = [-1] * count
    metadata = []
    for i, row in enumerate(rows):
        entry = {
            "image_id": image_ids[i], "query_raw_id": raw_ids[i],
            "query_level": level_list[i], "query_predicted_class_id": predicted_classes[i],
            "donor_valid": False, "donor_raw_id": None, "donor_compact_index": None,
            "donor_level": None, "donor_predicted_class_id": None,
            "same_predicted_class": None, "geometry_distance": None,
            "query_donor_box_iou": None, "boxes_identical": None,
            "empty_roi": not valid_list[i],
            "empty_reason": ("original_nonpositive_extent" if not raw_valid_list[i]
                             else "empty_after_sampling_clamp" if not valid_list[i] else None),
            "sampling_box_was_clamped": boxes_list[i] != sample_boxes[i].detach().cpu().tolist(),
        }
        if not valid_list[i]:
            entry["fallback_reason"] = "empty_query_roi"
            metadata.append(entry)
            continue
        eligible = [j for j in range(count)
                    if j != i and raw_ids[j] != raw_ids[i] and image_ids[j] == image_ids[i]
                    and valid_list[j]]
        same_class = [j for j in eligible if predicted_classes[j] == predicted_classes[i]]
        pool = same_class if same_class else eligible
        if not pool:
            entry["fallback_reason"] = "no_other_valid_same_image_candidate"
            metadata.append(entry)
            continue
        area_i, aspect_i = geometry[i]

        def donor_distance(j: int) -> float:
            area_j, aspect_j = geometry[j]
            return abs(math.log(area_j) - math.log(area_i)) + abs(math.log(aspect_j) - math.log(aspect_i))

        donor = min(pool, key=lambda j: (donor_distance(j), raw_ids[j], j))
        donors[i] = donor
        entry.update({
            "donor_valid": True, "donor_raw_id": raw_ids[donor],
            "donor_compact_index": donor, "donor_level": level_list[donor],
            "donor_predicted_class_id": predicted_classes[donor],
            "same_predicted_class": predicted_classes[i] == predicted_classes[donor],
            "geometry_distance": donor_distance(donor),
            "query_donor_box_iou": _box_iou(boxes_list[i], boxes_list[donor]),
            "boxes_identical": boxes_list[i] == boxes_list[donor],
            "fallback_reason": None,
        })
        metadata.append(entry)

    donor_tensor = torch.tensor(donors, dtype=torch.long, device=device)
    has_donor = donor_tensor >= 0
    for level, feature_map in enumerate(h_maps):
        indices = torch.nonzero((levels == level) & has_donor, as_tuple=False).flatten()
        if len(indices):
            donor_boxes = sample_boxes[donor_tensor[indices]]
            rois = torch.cat((sample_boxes.new_zeros((len(indices), 1)), donor_boxes), dim=1)
            aligned = roi_align(feature_map, rois, output_size=(ROI_SIZE, ROI_SIZE),
                                spatial_scale=sides[level] / INPUT_SIZE,
                                sampling_ratio=SAMPLING_RATIO, aligned=True)
            wrong_residual[indices] = aligned.mean(dim=(-2, -1)) - query_h[indices]
    for name, value in (("roi_mean", roi_mean), ("residual", residual),
                        ("wrong_residual", wrong_residual)):
        _finite(name, value)
    return {
        "roi_mean": roi_mean, "residual": residual, "wrong_residual": wrong_residual,
        "donor_metadata": metadata, "donor_valid": has_donor, "empty_roi": empty,
        "audit": {
            "candidates": count, "level_counts_full": counts, "level_sides": sides,
            "feature_dtype": "float32", "roi_align_output_size": [ROI_SIZE, ROI_SIZE],
            "roi_align_sampling_ratio": SAMPLING_RATIO, "roi_align_aligned": True,
            "input_size": INPUT_SIZE, "empty_roi_count": int(empty.sum()),
            "valid_donor_count": int(has_donor.sum()),
            "wrong_roi_uses_query_level": True,
            "empty_policy": "roi_mean=query_h; true_and_wrong_residual=0",
            "no_donor_policy": "wrong_residual=true_residual; donor_valid=false",
            "donor_priority": "same_predicted_class_then_abs_log_area_plus_abs_log_aspect_then_raw_id",
            "GT_fields_read": [],
        },
    }
