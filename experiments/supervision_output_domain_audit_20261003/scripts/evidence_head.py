"""Frozen-detector box evidence followed by prototype-aware coefficient projection.

The only trainable modules here are the shared feature encoder, query/FiLM,
evidence decoder and (in D only) a direct 256->32 projection. No detector or
native cv4 module is copied. All cached detector/operator inputs are detached.

S uses the cached K computed from the CURRENT frozen prototype basis; D uses
an independently trainable matrix and has 8,192 extra parameters. These arms
share their evidence architecture, but are not exactly parameter matched.
The zero final evidence convolution makes both arms identity at initialization.
Its gradient is available immediately; upstream gradients become nonzero after
that convolution moves. D's matrix is nonzero orthogonal at initialization so
it does not block the first evidence gradient.

Importing this module performs no data I/O, training, or tensor computation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align


LEVELS = 3
CHANNELS = 64
COEFFICIENTS = 32
LEVEL_CHANNELS = 16
ROI_SIDE = 16
TOKENS = ROI_SIDE * ROI_SIDE
INPUT_SIZE = 640
RESIDUAL_BOUND = 4.0
SAMPLING_RATIO = 2


def _conv_norm_act(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class EvidenceReadout(nn.Module):
    """Generate ROI evidence and update a fixed detector's mask coefficients.

    Args:
        feature_channels: The three actual cached neck channel counts. They
            are intentionally not hard coded to one detector width.
        mode: ``S`` for cached prototype projection or ``D`` for direct control.
        config: Optional mapping. ``roi_chunk_size`` (default 64) bounds decoder
            memory per image. Protocol constants ``input_size``, ``roi_side``
            and ``residual_bound`` may be stated explicitly but not changed.
            Other experiment/training configuration keys are ignored here.

    ``features`` contains three complete [B,C_l,H_l,W_l] frozen neck maps.
    ``selected`` contains one mapping per image with raw_ids [N], boxes [N,4],
    h0 [N,64], c0 [N,32], A [N,256,32], and (for S) K [N,32,256]. G [N,32,32]
    is cache provenance used to construct K and is not used again here. An
    optional bool ``valid`` [N] masks invalid cached operators. Raw IDs index
    the concatenated P3/P4/P5 row-major detector grids.

    ``sampling_boxes`` optionally changes ONLY which image-feature ROI is
    read. Query, A, K, c0 and final mask decoding retain the original object.
    It is intended for a subsequent frozen wrong-box diagnostic. No GT fields
    are inspected. Invalid operators/boxes produce exactly zero evidence and
    coefficient update; candidates are not filtered out.

    Normal forward returns list[Tensor[N,32]]. ``forward_with_evidence`` also
    returns list[Tensor[N,256]] of the actual differentiable bounded residuals
    used by that forward, for smoke tests and frozen diagnostic auditing.
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        mode: str = "S",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("S", "D"):
            raise ValueError("mode must be S (prototype projection) or D (direct)")
        if len(feature_channels) != LEVELS or any(int(c) <= 0 for c in feature_channels):
            raise ValueError("feature_channels must contain three positive channel counts")
        self.feature_channels = tuple(int(c) for c in feature_channels)
        self.mode = mode
        config = {} if config is None else dict(config)
        configured_size = config.get("input_size", INPUT_SIZE)
        if isinstance(configured_size, int):
            configured_size = (configured_size, configured_size)
        if tuple(configured_size) != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError("This protocol requires 640x640 detector input")
        if int(config.get("roi_side", ROI_SIDE)) != ROI_SIDE:
            raise ValueError("This protocol requires 16x16 evidence ROIs")
        if float(config.get("residual_bound", RESIDUAL_BOUND)) != RESIDUAL_BOUND:
            raise ValueError("This protocol fixes r = 4 * tanh(raw / 4)")
        self.roi_chunk_size = int(config.get("roi_chunk_size", 64))
        if self.roi_chunk_size <= 0:
            raise ValueError("roi_chunk_size must be positive")

        self.encoder_adapters = nn.ModuleList(
            [_conv_norm_act(channels, CHANNELS, 1) for channels in self.feature_channels]
        )
        self.encoder_fusion = _conv_norm_act(CHANNELS, CHANNELS, 3)
        self.level_embedding = nn.Embedding(LEVELS, LEVEL_CHANNELS)
        self.query_mlp = nn.Sequential(
            nn.Linear(CHANNELS + COEFFICIENTS + LEVEL_CHANNELS, 128),
            nn.SiLU(),
            nn.Linear(128, CHANNELS),
        )
        self.query_film = nn.Linear(CHANNELS, 2 * CHANNELS)
        self.evidence_decoder = nn.Sequential(
            _conv_norm_act(CHANNELS + 2 + 1, CHANNELS, 3),
            _conv_norm_act(CHANNELS, 32, 3),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.evidence_decoder[-1].weight)
        nn.init.zeros_(self.evidence_decoder[-1].bias)
        axis = (torch.arange(ROI_SIDE, dtype=torch.float32) + 0.5) / ROI_SIDE
        yy, xx = torch.meshgrid(axis * 2.0 - 1.0, axis * 2.0 - 1.0, indexing="ij")
        self.register_buffer("relative_xy", torch.stack((xx, yy), dim=0).unsqueeze(0))

        # Must be constructed AFTER every shared parameter: resetting the RNG
        # to the same seed then gives bitwise-identical shared S/D initial state.
        self.direct_projection = None
        if self.mode == "D":
            self.direct_projection = nn.Linear(TOKENS, COEFFICIENTS, bias=False)
            nn.init.orthogonal_(self.direct_projection.weight, gain=0.25)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def parameter_shapes(self) -> dict[str, list[int]]:
        return {name: list(parameter.shape) for name, parameter in self.named_parameters()}

    def parameter_counts(self) -> dict[str, int]:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        direct = 0 if self.direct_projection is None else count(self.direct_projection)
        return {
            "total": count(self),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "native": 0,
            "shared": count(self) - direct,
            "encoder": count(self.encoder_adapters) + count(self.encoder_fusion),
            "query": count(self.level_embedding) + count(self.query_mlp) + count(self.query_film),
            "evidence_decoder": count(self.evidence_decoder),
            "direct_projection": direct,
        }

    def _encode(self, features: Sequence[Tensor], batch: int) -> Tensor:
        target_size = features[0].shape[-2:]
        encoded = []
        for level, (feature, adapter) in enumerate(zip(features, self.encoder_adapters)):
            if feature.shape[:2] != (batch, self.feature_channels[level]):
                raise ValueError(f"Feature level {level} has an unexpected batch/channel shape")
            # Cached tensors can be FP16 even when parameters and the optimizer
            # use FP32. Autocast may reduce computation after this detached cast.
            value = adapter(feature.detach().to(dtype=adapter[0].weight.dtype))
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            encoded.append(value)
        return self.encoder_fusion((encoded[0] + encoded[1] + encoded[2]) / LEVELS)

    @staticmethod
    def _frozen(
        selection: Mapping[str, Any], key: str, shape: tuple[int, ...], device: torch.device,
    ) -> Tensor:
        value = torch.as_tensor(selection[key], device=device).detach()
        if tuple(value.shape) != shape:
            raise ValueError(f"{key} must have shape {shape}, got {tuple(value.shape)}")
        return value

    def forward(
        self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Any]],
    ) -> list[Tensor]:
        coefficients, _ = self.forward_with_evidence(features, selected)
        return coefficients

    def forward_with_evidence(
        self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if len(features) != LEVELS or any(feature.ndim != 4 for feature in features):
            raise ValueError("features must contain three complete BCHW neck maps")
        batch = features[0].shape[0]
        device = features[0].device
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
        key_f = self._encode(features, batch)
        outputs, evidence_outputs = [], []

        for image_idx, selection in enumerate(selected):
            raw_ids = torch.as_tensor(selection["raw_ids"], device=device).detach()
            if raw_ids.ndim != 1 or (raw_ids.numel() and raw_ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("raw_ids must be a one-dimensional integer tensor")
            raw_ids = raw_ids.long()
            if bool(((raw_ids < 0) | (raw_ids >= total_raw)).any()):
                raise ValueError("raw_id is outside the complete detector candidate grid")
            n = raw_ids.numel()
            c0 = self._frozen(selection, "c0", (n, COEFFICIENTS), device)
            h0 = self._frozen(selection, "h0", (n, CHANNELS), device)
            a = self._frozen(selection, "A", (n, TOKENS, COEFFICIENTS), device)
            k = self._frozen(selection, "K", (n, COEFFICIENTS, TOKENS), device) if self.mode == "S" else None
            boxes = self._frozen(selection, "boxes", (n, 4), device).float()
            sampling_key = "sampling_boxes" if "sampling_boxes" in selection else "boxes"
            sampling = self._frozen(selection, sampling_key, (n, 4), device).float()
            original_clamped = boxes.clamp(0, INPUT_SIZE)
            sampling_clamped = sampling.clamp(0, INPUT_SIZE)
            valid = torch.isfinite(boxes).all(1) & torch.isfinite(sampling).all(1)
            valid &= ((original_clamped[:, 2:] - original_clamped[:, :2]) > 0).all(1)
            valid &= ((sampling_clamped[:, 2:] - sampling_clamped[:, :2]) > 0).all(1)
            if "valid" in selection:
                valid &= self._frozen(selection, "valid", (n,), device).bool()
            positions = torch.where(valid)[0]
            levels = (raw_ids >= counts[0]).long() + (raw_ids >= counts[0] + counts[1]).long()

            # All-invalid/empty images preserve a legitimate zero-gradient path.
            # No invalid A/K/h0 entries enter matrix multiplication or the decoder.
            anchor = self.evidence_decoder[-1].weight.reshape(-1)[0] * 0.0
            residual = torch.zeros((n, TOKENS), device=device, dtype=torch.float32) + anchor.float()
            for start in range(0, positions.numel(), self.roi_chunk_size):
                indices = positions[start:start + self.roi_chunk_size]
                roi_boxes = sampling_clamped[indices].to(dtype=key_f.dtype)
                rois = torch.cat((roi_boxes.new_zeros((len(indices), 1)), roi_boxes), dim=1)
                roi = roi_align(
                    key_f[image_idx:image_idx + 1], rois,
                    output_size=(ROI_SIDE, ROI_SIDE),
                    spatial_scale=height / INPUT_SIZE,
                    sampling_ratio=SAMPLING_RATIO,
                    aligned=True,
                )
                query_dtype = self.query_mlp[0].weight.dtype
                query_input = torch.cat((
                    h0[indices].to(query_dtype), c0[indices].to(query_dtype),
                    self.level_embedding(levels[indices]).to(query_dtype),
                ), dim=1)
                query = self.query_mlp(query_input)
                gamma, beta = self.query_film(query).to(dtype=roi.dtype).chunk(2, dim=1)
                conditioned = roi * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
                # z0 remains the baseline logit on the ORIGINAL object's A grid,
                # even when sampling_boxes reads a different image-feature ROI.
                with torch.autocast(device_type=device.type, enabled=False):
                    z0 = torch.bmm(a[indices].float(), c0[indices].float().unsqueeze(2))
                z0 = z0.reshape(-1, 1, ROI_SIDE, ROI_SIDE).to(dtype=conditioned.dtype)
                xy = self.relative_xy.to(dtype=conditioned.dtype).expand(len(indices), -1, -1, -1)
                raw = self.evidence_decoder(torch.cat((conditioned, xy, z0), dim=1))
                bounded = RESIDUAL_BOUND * torch.tanh(raw.float() / RESIDUAL_BOUND)
                residual = residual.index_copy(0, indices, bounded.flatten(1))

            # Invalid cached operators may contain nonfinite diagnostic entries.
            # Project only valid rows, then scatter zero updates for all others.
            delta = torch.zeros((n, COEFFICIENTS), device=device, dtype=torch.float32)
            delta = delta + residual.sum(dim=1, keepdim=True) * 0.0
            if positions.numel():
                with torch.autocast(device_type=device.type, enabled=False):
                    if self.mode == "S":
                        correction = torch.bmm(k[positions].float(), residual[positions].float().unsqueeze(2)).squeeze(2)
                    else:
                        correction = F.linear(residual[positions].float(), self.direct_projection.weight.float())
                delta = delta.index_copy(0, positions, correction)
            outputs.append(c0 + delta.to(dtype=c0.dtype))
            evidence_outputs.append(residual)
        return outputs, evidence_outputs
