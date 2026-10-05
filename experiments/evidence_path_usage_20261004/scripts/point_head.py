"""Joint native cv4 + identical 64-point evidence readout for U/Q/P.

Arm identity is metadata only. The forward architecture, parameter count and
initialization are identical; frozen indices and prototype K are the only arm
inputs that differ. Arbitrary selected cells are never reshaped as an 8x8 map.
Each cell is independently read with ROIAlign(output_size=1, sampling_ratio=2)
and processed by the same point MLP, preserving its true 16x16 relative XY.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align

from evidence_head import CHANNELS, COEFFICIENTS, INPUT_SIZE, LEVELS, ROI_SIDE, SAMPLING_RATIO, TOKENS
from joint_head import CurrentEvidenceReadout, JointCoefficientReadout

SELECTED_TOKENS = 64
RESIDUAL_BOUND = 4.0


def selected_cell_rois(boxes: Tensor, indices: Tensor, batch_index: int = 0) -> Tensor:
    """Convert selected original-grid cells into N*64 input-coordinate ROIs.

    Input boxes are already clamped exactly as in the old complete 16x16 ROI
    path. Dividing a box into its 16x16 cells and using 1x1 ROIAlign with fixed
    sampling_ratio=2, aligned=True visits the same four interpolation points
    as the corresponding full-ROI output bin (up to arithmetic roundoff).
    No additional clipping or box normalization is introduced here.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4 or indices.shape != (len(boxes), SELECTED_TOKENS):
        raise ValueError("boxes/indices must be [N,4]/[N,64]")
    column = (indices % ROI_SIDE).to(boxes.dtype)
    row = torch.div(indices, ROI_SIDE, rounding_mode="floor").to(boxes.dtype)
    wh = boxes[:, 2:] - boxes[:, :2]
    x1 = boxes[:, 0, None] + column * wh[:, 0, None] / ROI_SIDE
    y1 = boxes[:, 1, None] + row * wh[:, 1, None] / ROI_SIDE
    x2 = boxes[:, 0, None] + (column + 1) * wh[:, 0, None] / ROI_SIDE
    y2 = boxes[:, 1, None] + (row + 1) * wh[:, 1, None] / ROI_SIDE
    cell_boxes = torch.stack((x1, y1, x2, y2), dim=2).reshape(-1, 4)
    image_column = cell_boxes.new_full((len(cell_boxes), 1), float(batch_index))
    return torch.cat((image_column, cell_boxes), dim=1)


class PointEvidenceReadout(CurrentEvidenceReadout):
    """Same frozen F encoder/query FiLM for all arms; only 64 cells are read.

    Required per-image keys: raw_ids [N], boxes [N,4], h_current [N,64],
    c_current [N,32], A_full (or A) [N,256,32], indices [N,64], K [N,32,64].
    Optional valid [N] preserves invalid candidates with zero correction.
    Cached h0/c0 are retained by the caller for audit but not used as live
    inputs. No GT, annotation_id, target_gt_idx or class label is inspected.
    """

    def __init__(self, feature_channels: Sequence[int], config: Mapping[str, Any] | None = None):
        # Reuse the existing F encoder, query, FiLM, zero-init convention and
        # frozen/live tensor distinction. Replace the spatial CNN completely.
        super().__init__(feature_channels, "S", config)
        self.evidence_decoder = nn.Sequential(
            nn.Linear(CHANNELS + 2 + 1, CHANNELS),
            nn.LayerNorm(CHANNELS),
            nn.SiLU(),
            nn.Linear(CHANNELS, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.evidence_decoder[-1].weight)
        nn.init.zeros_(self.evidence_decoder[-1].bias)

    def forward_with_evidence(self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Any]]):
        if len(features) != LEVELS or any(f.ndim != 4 for f in features):
            raise ValueError("features must be three complete BCHW neck maps")
        batch, device = features[0].shape[0], features[0].device
        if len(selected) != batch or any(f.shape[0] != batch or f.device != device for f in features):
            raise ValueError("features/selection batch or device mismatch")
        if device != self.encoder_adapters[0][0].weight.device:
            raise ValueError("model and features must share a device")
        height, width = features[0].shape[-2:]
        if height != width or height <= 0:
            raise ValueError("Expected a nonempty square P3 grid")
        counts = [feature.shape[-2] * feature.shape[-1] for feature in features]
        key_f = self._encode(features, batch)
        outputs, evidence_outputs = [], []

        for image_index, selection in enumerate(selected):
            raw_ids = torch.as_tensor(selection["raw_ids"], device=device).detach()
            if raw_ids.ndim != 1 or raw_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("raw_ids must be an integer vector")
            raw_ids = raw_ids.long()
            if bool(((raw_ids < 0) | (raw_ids >= sum(counts))).any()):
                raise ValueError("raw_id outside complete P3/P4/P5 grid")
            n = len(raw_ids)
            # CurrentEvidenceReadout resolves these two names to live tensors
            # without detach; this is the native/evidence joint gradient path.
            h = self._frozen(selection, "h0", (n, CHANNELS), device)
            c = self._frozen(selection, "c0", (n, COEFFICIENTS), device)
            a_key = "A_full" if "A_full" in selection else "A"
            a = self._frozen(selection, a_key, (n, TOKENS, COEFFICIENTS), device)
            k = self._frozen(selection, "K", (n, COEFFICIENTS, SELECTED_TOKENS), device)
            boxes = self._frozen(selection, "boxes", (n, 4), device).float()
            indices = self._frozen(selection, "indices", (n, SELECTED_TOKENS), device)
            if indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("indices must be integer original-grid IDs")
            indices = indices.long()
            if bool(((indices < 0) | (indices >= TOKENS)).any()):
                raise ValueError("selected grid ID outside 0..255")
            ordered_indices = indices.sort(dim=1).values
            if bool((ordered_indices[:, 1:] == ordered_indices[:, :-1]).any()):
                raise ValueError("Every candidate must use 64 distinct cells")
            sampling_key = "sampling_boxes" if "sampling_boxes" in selection else "boxes"
            sampling = self._frozen(selection, sampling_key, (n, 4), device).float()
            clamped, sampling_clamped = boxes.clamp(0, INPUT_SIZE), sampling.clamp(0, INPUT_SIZE)
            valid = torch.isfinite(boxes).all(1) & torch.isfinite(sampling).all(1)
            valid &= ((clamped[:, 2:] - clamped[:, :2]) > 0).all(1)
            valid &= ((sampling_clamped[:, 2:] - sampling_clamped[:, :2]) > 0).all(1)
            if "valid" in selection:
                valid &= self._frozen(selection, "valid", (n,), device).bool()
            positions = torch.where(valid)[0]
            if positions.numel():
                if not bool(torch.isfinite(a[positions]).all() & torch.isfinite(k[positions]).all()):
                    raise ValueError("A/K nonfinite on a valid candidate")
            levels = (raw_ids >= counts[0]).long() + (raw_ids >= counts[0] + counts[1]).long()
            anchor = self.evidence_decoder[-1].weight.reshape(-1)[0] * 0.0
            residual = c.new_zeros((n, SELECTED_TOKENS), dtype=torch.float32) + anchor.float()
            for start in range(0, len(positions), self.roi_chunk_size):
                rows = positions[start:start + self.roi_chunk_size]
                chosen = indices[rows]
                rois = selected_cell_rois(sampling_clamped[rows].to(key_f.dtype), chosen)
                points = roi_align(key_f[image_index:image_index + 1], rois,
                                   output_size=(1, 1), spatial_scale=height / INPUT_SIZE,
                                   sampling_ratio=SAMPLING_RATIO, aligned=True)
                points = points.reshape(len(rows), SELECTED_TOKENS, CHANNELS)
                query_dtype = self.query_mlp[0].weight.dtype
                query = self.query_mlp(torch.cat((h[rows].to(query_dtype), c[rows].to(query_dtype),
                                                 self.level_embedding(levels[rows]).to(query_dtype)), dim=1))
                gamma, beta = self.query_film(query).to(points.dtype).chunk(2, dim=1)
                conditioned = points * (1.0 + gamma[:, None, :]) + beta[:, None, :]
                sampled_a = a[rows].gather(1, chosen[:, :, None].expand(-1, -1, COEFFICIENTS))
                with torch.autocast(device_type=device.type, enabled=False):
                    z_current = torch.bmm(sampled_a.float(), c[rows].float().unsqueeze(2))
                x = ((chosen % ROI_SIDE).to(points.dtype) + 0.5) * (2.0 / ROI_SIDE) - 1.0
                y = (torch.div(chosen, ROI_SIDE, rounding_mode="floor").to(points.dtype) + 0.5) * (2.0 / ROI_SIDE) - 1.0
                xy = torch.stack((x, y), dim=2)
                point_input = torch.cat((conditioned, xy, z_current.to(conditioned.dtype)), dim=2)
                raw = self.evidence_decoder(point_input).squeeze(2)
                bounded = RESIDUAL_BOUND * torch.tanh(raw.float() / RESIDUAL_BOUND)
                residual = residual.index_copy(0, rows, bounded)
            delta = c.new_zeros((n, COEFFICIENTS), dtype=torch.float32)
            delta = delta + residual.sum(dim=1, keepdim=True) * 0.0
            if positions.numel():
                with torch.autocast(device_type=device.type, enabled=False):
                    correction = torch.bmm(k[positions].float(), residual[positions].unsqueeze(2)).squeeze(2)
                delta = delta.index_copy(0, positions, correction)
            outputs.append(c + delta.to(c.dtype))
            evidence_outputs.append(residual)
        return outputs, evidence_outputs


class JointPointCoefficientReadout(JointCoefficientReadout):
    """Native one-to-one cv4 and new point readout learn jointly.

    Inherits current_selection, forward_details and the BN-buffer freeze from
    the audited joint implementation. mode is not consulted by the point head:
    U/Q/P differ only through the externally frozen selection/operator cache.
    BN affine parameters remain trainable, matching the prior joint protocol.
    """

    def __init__(self, native_cv4: nn.ModuleList, feature_channels: Sequence[int],
                 mode: str, config: Mapping[str, Any] | None = None):
        if mode not in ("U", "Q", "P"):
            raise ValueError("mode must be U, Q, or P")
        super().__init__(native_cv4, feature_channels, "N", config)
        self.mode = mode
        self.evidence = PointEvidenceReadout(feature_channels, config)
        self.requires_grad_(True)
        self.train(True)
