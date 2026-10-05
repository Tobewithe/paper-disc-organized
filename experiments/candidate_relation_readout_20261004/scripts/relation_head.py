"""Single-step coefficient readout with frozen candidate-response context.

N reruns only native cv4. S, T and M share exactly the same architecture and
initialization: their sole difference is the cached relation tensor consumed.
S substitutes the target's frozen c0 into each neighbor support; T uses true
neighbor responses; M uses responses rearranged *within each neighbor support*
by the cache builder. The three channels are score-weighted response maximum,
mean and availability. This module never uses GT to construct neighbors.

The native h/c path remains differentiable. Neck features, prototype operators,
boxes and all relation channels are detached. Final evidence convolution is
zero-initialized, making the initial corrected coefficients exactly native c.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align

from evidence_head import (
    CHANNELS, COEFFICIENTS, INPUT_SIZE, LEVELS, RESIDUAL_BOUND,
    ROI_SIDE, SAMPLING_RATIO, TOKENS, _conv_norm_act,
)
from joint_head import CurrentEvidenceReadout, JointCoefficientReadout


RELATION_KEYS = {"S": "relation_self", "T": "relation_true", "M": "relation_wrong"}
RELATION_CHANNELS = 3


class CandidateRelationEvidence(CurrentEvidenceReadout):
    """Prototype-projected ROI evidence augmented by frozen relation channels."""

    def __init__(self, feature_channels: Sequence[int], mode: str,
                 config: Mapping[str, Any] | None = None) -> None:
        if mode not in RELATION_KEYS:
            raise ValueError("Candidate relation mode must be S, T or M")
        # All relational arms call precisely the same constructors in the same
        # order. A reset RNG seed therefore gives identical state_dict values.
        super().__init__(feature_channels, "S", config)
        self.relation_mode = mode
        self.evidence_decoder = nn.Sequential(
            _conv_norm_act(CHANNELS + 2 + 1 + RELATION_CHANNELS, CHANNELS, 3),
            _conv_norm_act(CHANNELS, 32, 3),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.evidence_decoder[-1].weight)
        nn.init.zeros_(self.evidence_decoder[-1].bias)

    def forward_with_evidence(
        self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if len(features) != LEVELS or any(feature.ndim != 4 for feature in features):
            raise ValueError("features must contain three complete BCHW neck maps")
        batch, device = features[0].shape[0], features[0].device
        if len(selected) != batch or any(feature.shape[0] != batch for feature in features):
            raise ValueError("features and selected disagree on batch size")
        if any(feature.device != device for feature in features):
            raise ValueError("All neck feature maps must share a device")
        if device != self.encoder_adapters[0][0].weight.device:
            raise ValueError("The readout and feature maps must share a device")
        height, width = features[0].shape[-2:]
        if height != width or min(height, width) <= 0:
            raise ValueError("The 640x640 protocol requires a nonempty square P3 grid")
        counts = [feature.shape[-2] * feature.shape[-1] for feature in features]
        total_raw = sum(counts)
        encoded = self._encode(features, batch)
        outputs, evidence_outputs = [], []

        for image_idx, selection in enumerate(selected):
            raw_ids = torch.as_tensor(selection["raw_ids"], device=device).detach()
            if raw_ids.ndim != 1 or raw_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("raw_ids must be a one-dimensional integer tensor")
            raw_ids = raw_ids.long()
            if bool(((raw_ids < 0) | (raw_ids >= total_raw)).any()):
                raise ValueError("raw_id is outside the complete detector candidate grid")
            n = raw_ids.numel()
            # CurrentEvidenceReadout deliberately preserves autograd only for
            # h0/c0, resolving them to native h_current/c_current respectively.
            c = self._frozen(selection, "c0", (n, COEFFICIENTS), device)
            h = self._frozen(selection, "h0", (n, CHANNELS), device)
            a = self._frozen(selection, "A", (n, TOKENS, COEFFICIENTS), device)
            k = self._frozen(selection, "K", (n, COEFFICIENTS, TOKENS), device)
            relation = self._frozen(
                selection, RELATION_KEYS[self.relation_mode],
                (n, RELATION_CHANNELS, ROI_SIDE, ROI_SIDE), device,
            )
            boxes = self._frozen(selection, "boxes", (n, 4), device).float()
            sampling_key = "sampling_boxes" if "sampling_boxes" in selection else "boxes"
            sampling = self._frozen(selection, sampling_key, (n, 4), device).float()
            original_clamped, sampling_clamped = boxes.clamp(0, INPUT_SIZE), sampling.clamp(0, INPUT_SIZE)
            valid = torch.isfinite(boxes).all(1) & torch.isfinite(sampling).all(1)
            valid &= ((original_clamped[:, 2:] - original_clamped[:, :2]) > 0).all(1)
            valid &= ((sampling_clamped[:, 2:] - sampling_clamped[:, :2]) > 0).all(1)
            if "valid" in selection:
                valid &= self._frozen(selection, "valid", (n,), device).bool()
            positions = torch.where(valid)[0]
            if positions.numel() and not bool(torch.isfinite(relation[positions]).all()):
                raise ValueError("Valid candidates contain nonfinite relation channels")
            levels = (raw_ids >= counts[0]).long() + (raw_ids >= counts[0] + counts[1]).long()

            anchor = self.evidence_decoder[-1].weight.reshape(-1)[0] * 0.0
            residual = torch.zeros((n, TOKENS), device=device, dtype=torch.float32) + anchor.float()
            for start in range(0, positions.numel(), self.roi_chunk_size):
                indices = positions[start:start + self.roi_chunk_size]
                roi_boxes = sampling_clamped[indices].to(dtype=encoded.dtype)
                rois = torch.cat((roi_boxes.new_zeros((len(indices), 1)), roi_boxes), dim=1)
                roi = roi_align(
                    encoded[image_idx:image_idx + 1], rois,
                    output_size=(ROI_SIDE, ROI_SIDE), spatial_scale=height / INPUT_SIZE,
                    sampling_ratio=SAMPLING_RATIO, aligned=True,
                )
                query_dtype = self.query_mlp[0].weight.dtype
                query_input = torch.cat((
                    h[indices].to(query_dtype), c[indices].to(query_dtype),
                    self.level_embedding(levels[indices]).to(query_dtype),
                ), dim=1)
                query = self.query_mlp(query_input)
                gamma, beta = self.query_film(query).to(dtype=roi.dtype).chunk(2, dim=1)
                conditioned = roi * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
                with torch.autocast(device_type=device.type, enabled=False):
                    current_z = torch.bmm(a[indices].float(), c[indices].float().unsqueeze(2))
                current_z = current_z.reshape(-1, 1, ROI_SIDE, ROI_SIDE).to(conditioned.dtype)
                xy = self.relative_xy.to(conditioned.dtype).expand(len(indices), -1, -1, -1)
                context = relation[indices].to(conditioned.dtype)
                raw = self.evidence_decoder(torch.cat((conditioned, xy, current_z, context), dim=1))
                bounded = RESIDUAL_BOUND * torch.tanh(raw.float() / RESIDUAL_BOUND)
                residual = residual.index_copy(0, indices, bounded.flatten(1))

            delta = torch.zeros((n, COEFFICIENTS), device=device, dtype=torch.float32)
            delta = delta + residual.sum(dim=1, keepdim=True) * 0.0
            if positions.numel():
                with torch.autocast(device_type=device.type, enabled=False):
                    correction = torch.bmm(k[positions].float(), residual[positions].float().unsqueeze(2)).squeeze(2)
                delta = delta.index_copy(0, positions, correction)
            outputs.append(c + delta.to(dtype=c.dtype))
            evidence_outputs.append(residual)
        return outputs, evidence_outputs


class RelationReadout(JointCoefficientReadout):
    """N: native; S: self response; T: true relation; M: mismatched relation.

    ``selected`` rows additionally carry frozen [N,3,16,16] relation_self,
    relation_true and relation_wrong caches. Only the chosen cache is read;
    N needs none of them. An absent-neighbor row is represented by three zero
    channels, not a removed candidate. No relation tensor receives gradients.
    """

    def __init__(self, native_cv4: nn.ModuleList, feature_channels: Sequence[int],
                 mode: str, config: Mapping[str, Any] | None = None) -> None:
        if mode not in ("N", "S", "T", "M"):
            raise ValueError("mode must be N, S, T or M")
        # Reuse native validation/copying without constructing a discarded
        # evidence head; all non-native arms then follow the same RNG path.
        super().__init__(native_cv4, feature_channels, "N", config)
        self.mode = mode
        if mode != "N":
            self.evidence = CandidateRelationEvidence(feature_channels, mode, config)
        self.requires_grad_(True)
        self.train(True)

    def forward_details(self, features, selected):
        return super().forward_details(features, selected)

    def forward(self, features, selected):
        coefficients, _, _ = self.forward_details(features, selected)
        return coefficients
