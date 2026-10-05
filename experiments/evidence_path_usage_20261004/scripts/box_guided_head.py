"""Trainable native coefficient branches with prediction-box cross-attention.

``CoefficientReadout(native_cv4, mode='R')`` owns a deep copy of the three
official ONE-TO-ONE coefficient branches.  ``features`` are full frozen neck
maps entering cv4, NOT previously cached 64-dimensional h.  Every forward
recomputes both native convolutions to obtain H, gathers query h at the fixed
raw IDs, and applies the original final 1x1 convolution to h + delta_h.

Modes: N/native = native branch; P/point = capacity-matched point control;
R/region = prediction-box ROI.  P and R have identical parameters.  In P,
the 49 content tokens are copies of query h; the same position and level
embeddings and cross-attention are retained.  In R, all 49 ROI tokens remain
separate.  No averaging, extra loss, teacher, gate, or GT input is introduced.

The residual projection starts at zero.  Thus all three modes initially
produce the native coefficients.  On the first backward pass the projection
can receive gradients while its preceding attention parameters have zero
gradient; the latter become trainable after the projection moves.  The native
branches receive their normal mask-loss gradients immediately.

All native parameters, including BatchNorm affine parameters, are trainable.
BatchNorm running statistics ALWAYS stay in evaluation mode, including after
calling train().  Features, raw IDs and boxes cannot backpropagate to the
frozen backbone/neck or detection branch.  This module performs no data I/O,
training, or tensor computation at import time.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align


CHANNELS = 64
LEVELS = 3
ROI_SIDE = 7
TOKENS = ROI_SIDE * ROI_SIDE
SAMPLING_RATIO = 2


class _CrossAttentionBlock(nn.Module):
    """One query attends to 49 spatial tokens, followed by a small FFN."""

    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(CHANNELS)
        self.memory_norm = nn.LayerNorm(CHANNELS)
        self.attention = nn.MultiheadAttention(
            CHANNELS, num_heads=4, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(CHANNELS)
        self.ffn = nn.Sequential(
            nn.Linear(CHANNELS, 128), nn.SiLU(), nn.Linear(128, CHANNELS)
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        update, _ = self.attention(
            self.query_norm(query), normalized_memory, normalized_memory,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class _BoxFusion(nn.Module):
    """Shared spatial fusion with a fixed 2D ROI grid and learned embeddings."""

    def __init__(self) -> None:
        super().__init__()
        # Relative ROI-bin centres: x,y in [-1,1].  Token order matches
        # ROIAlign's flatten(2).transpose(1,2): y outer, x inner.
        axis = (torch.arange(ROI_SIDE, dtype=torch.float32) + 0.5) / ROI_SIDE
        axis = axis * 2.0 - 1.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("positions_xy", torch.stack((xx, yy), -1).reshape(TOKENS, 2))
        self.position_embedding = nn.Linear(2, CHANNELS)
        self.level_embedding = nn.Embedding(LEVELS, CHANNELS)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(), _CrossAttentionBlock()])
        self.output_norm = nn.LayerNorm(CHANNELS)
        self.residual_projection = nn.Linear(CHANNELS, CHANNELS)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, query_h: Tensor, content: Tensor, level: int) -> Tensor:
        level_code = self.level_embedding.weight[level].to(dtype=query_h.dtype)
        position_code = self.position_embedding(self.positions_xy).to(dtype=content.dtype)
        query = (query_h + level_code).unsqueeze(1)
        memory = content + position_code.unsqueeze(0) + level_code
        for block in self.blocks:
            query = block(query, memory)
        return self.residual_projection(self.output_norm(query[:, 0]))


class CoefficientReadout(nn.Module):
    """Recompute native cv4 features and read selected raw coefficients.

    Args:
        native_cv4: Original head.one2one_cv4, three independent sequential
            branches ending in a 64->32, stride-1, groups-1 1x1 Conv2d.
            Deep copied, never mutated; use the same original for N/P/R.
        mode: 'N', 'P', 'R', or aliases 'native', 'point', 'region'.
        input_size: Detector input height,width; default (640,640).

    forward(features, selected) -> list[Tensor[N_image,32]]

    features[l] has shape [B,C_l,H_l,W_l] BEFORE the entire native cv4[l].
    selected has B dictionaries with raw_ids [N] (P3 then P4 then P5,
    row-major within each level) and predicted boxes [N,4] in input-image
    xyxy coordinates.  Optional sampling_boxes [N,4] changes ONLY R's ROI
    extraction, enabling frozen-model wrong-box replay while query raw_ids
    and original final boxes remain fixed.  No GT fields are inspected.

    Boxes are detached; only a sampling copy is clamped to the input image.
    Invalid/empty original or sampling boxes retain the native coefficient
    branch with zero fusion residual.  They are never removed.  The caller
    records such cases and controls the independent final mask crop.
    """

    _ALIASES = {
        "N": "N", "native": "N", "P": "P", "point": "P",
        "R": "R", "region": "R",
    }

    def __init__(
        self, native_cv4: nn.ModuleList, mode: str = "R",
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()
        if mode not in self._ALIASES:
            raise ValueError("mode must be N/native, P/point, or R/region")
        if len(native_cv4) != LEVELS:
            raise ValueError("Exactly three native cv4 branches are required")
        if len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError("input_size must be positive (height,width)")
        self.mode = self._ALIASES[mode]
        self.input_size = tuple(int(x) for x in input_size)
        self.native_cv4 = copy.deepcopy(native_cv4)
        for level, branch in enumerate(self.native_cv4):
            if not isinstance(branch, nn.Sequential) or len(branch) < 2:
                raise TypeError(f"cv4[{level}] must be the native Sequential branch")
            final = branch[-1]
            if (not isinstance(final, nn.Conv2d) or final.in_channels != CHANNELS
                    or final.out_channels != 32 or final.kernel_size != (1, 1)
                    or final.stride != (1, 1) or final.padding != (0, 0)
                    or final.groups != 1):
                raise ValueError(f"cv4[{level}] does not end in the expected 64->32 1x1 readout")
        self.fusion = _BoxFusion() if self.mode != "N" else None
        # The source model may already have requires_grad=False everywhere.
        # These private copies, including BN gamma/beta, must learn again.
        self.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True) -> "CoefficientReadout":
        super().train(mode)
        # eval does not disable gradients of BN's affine gamma/beta.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "native": sum(p.numel() for p in self.native_cv4.parameters()),
            "fusion": 0 if self.fusion is None else sum(p.numel() for p in self.fusion.parameters()),
        }

    def forward(
        self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Any]],
    ) -> list[Tensor]:
        if len(features) != LEVELS or any(f.ndim != 4 for f in features):
            raise ValueError("features must contain three complete BCHW neck maps")
        batch = features[0].shape[0]
        if len(selected) != batch or any(f.shape[0] != batch for f in features):
            raise ValueError("features and selected disagree on batch size")
        if any(f.device != features[0].device for f in features):
            raise ValueError("All feature maps must share a device")
        maps = []
        for branch, frozen_f in zip(self.native_cv4, features):
            h_map = frozen_f.detach()
            for layer in list(branch.children())[:-1]:
                h_map = layer(h_map)
            if h_map.shape[1] != CHANNELS:
                raise ValueError("Native cv4 prefix did not produce 64-channel H")
            maps.append(h_map)
        counts = [h.shape[-2] * h.shape[-1] for h in maps]
        offsets = [0, counts[0], counts[0] + counts[1]]
        total_raw = sum(counts)
        input_h, input_w = self.input_size
        results = []
        for image_idx, selection in enumerate(selected):
            raw_ids = torch.as_tensor(selection["raw_ids"], device=maps[0].device)
            if raw_ids.ndim != 1 or (len(raw_ids) and raw_ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("raw_ids must be a 1D integer tensor")
            raw_ids = raw_ids.detach().long()
            if bool(((raw_ids < 0) | (raw_ids >= total_raw)).any()):
                raise ValueError("raw_id is outside the complete native candidate grid")
            boxes = torch.as_tensor(selection["boxes"], device=maps[0].device).detach()
            if boxes.shape != (len(raw_ids), 4) or not bool(torch.isfinite(boxes).all()):
                raise ValueError("boxes must be finite [N,4] predicted xyxy boxes")
            sampling = selection.get("sampling_boxes", selection["boxes"]) if self.mode == "R" else selection["boxes"]
            sampling = torch.as_tensor(sampling, device=maps[0].device).detach().clone()
            if sampling.shape != boxes.shape or not bool(torch.isfinite(sampling).all()):
                raise ValueError("sampling_boxes must be finite [N,4] boxes")
            sampling[:, (0, 2)] = sampling[:, (0, 2)].clamp(0, input_w)
            sampling[:, (1, 3)] = sampling[:, (1, 3)].clamp(0, input_h)
            valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
            original_clamped = boxes.clone()
            original_clamped[:, (0, 2)] = original_clamped[:, (0, 2)].clamp(0, input_w)
            original_clamped[:, (1, 3)] = original_clamped[:, (1, 3)].clamp(0, input_h)
            valid &= ((original_clamped[:, 2:] - original_clamped[:, :2]) > 0).all(1)
            valid &= ((sampling[:, 2:] - sampling[:, :2]) > 0).all(1)
            coefficients = maps[0].new_zeros((len(raw_ids), 32))
            for level, (h_map, offset, count) in enumerate(zip(maps, offsets, counts)):
                positions = torch.where((raw_ids >= offset) & (raw_ids < offset + count))[0]
                if not len(positions):
                    continue
                local_ids = raw_ids[positions] - offset
                query_h = h_map[image_idx].flatten(1).T[local_ids]
                enriched_h = query_h
                if self.fusion is not None:
                    usable = torch.where(valid[positions])[0]
                    delta_h = torch.zeros_like(query_h)
                    if len(usable):
                        query_valid = query_h[usable]
                        if self.mode == "P":
                            tokens = query_valid.unsqueeze(1).expand(-1, TOKENS, -1)
                        else:
                            height, width = h_map.shape[-2:]
                            scale = width / input_w
                            if abs(scale - height / input_h) > 1e-12:
                                raise ValueError("ROIAlign requires the same feature stride in x and y")
                            roi_boxes = sampling[positions[usable]].to(dtype=h_map.dtype)
                            rois = torch.cat((roi_boxes.new_zeros((len(usable), 1)), roi_boxes), 1)
                            roi = roi_align(
                                h_map[image_idx:image_idx + 1], rois,
                                output_size=(ROI_SIDE, ROI_SIDE), spatial_scale=scale,
                                sampling_ratio=SAMPLING_RATIO, aligned=True,
                            )
                            tokens = roi.flatten(2).transpose(1, 2)
                        delta_h[usable] = self.fusion(query_valid, tokens, level)
                    enriched_h = query_h + delta_h
                # Apply the actual native Conv2d, retaining its weight/bias
                # and autograd path; do not create another coefficient head.
                coefficient = self.native_cv4[level][-1](
                    enriched_h.T.unsqueeze(0).unsqueeze(2)
                )[0, :, 0, :].T
                coefficients[positions] = coefficient
            results.append(coefficients)
        return results

