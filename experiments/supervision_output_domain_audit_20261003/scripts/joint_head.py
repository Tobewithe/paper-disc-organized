"""Joint native one-to-one coefficient branch and box-evidence readout.

Only full, frozen neck maps are inputs. Native cv4 is re-run every forward;
cached h0/c0 are provenance/diagnostic references, never a substitute for the
current trainable h/c. The evidence path retains derivatives to BOTH current
h and current c. Its prototype operators, boxes and upstream neck maps remain
frozen. BatchNorm affine parameters learn while running buffers stay fixed.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from evidence_head import EvidenceReadout, CHANNELS, COEFFICIENTS, LEVELS


class CurrentEvidenceReadout(EvidenceReadout):
    """Reuse exactly the old evidence architecture, but consume live h and c.

    The inherited implementation's local variables are called h0/c0. Here
    those two reads explicitly resolve to h_current/c_current, and ONLY those
    values keep autograd. Cached h0/c0 in the input selection are untouched.
    This changes gradient flow, not projection, decoder, bounds or ROI rules.
    """

    @staticmethod
    def _frozen(selection: Mapping[str, Any], key: str, shape: tuple[int, ...],
                device: torch.device) -> Tensor:
        if key in ("h0", "c0"):
            live_key = "h_current" if key == "h0" else "c_current"
            value = selection[live_key]
            if not torch.is_tensor(value) or value.device != device:
                raise ValueError(f"{live_key} must be a live tensor on {device}")
            if tuple(value.shape) != shape:
                raise ValueError(f"{live_key} must have shape {shape}, got {tuple(value.shape)}")
            return value  # Deliberately no detach: evidence -> native branch.
        return EvidenceReadout._frozen(selection, key, shape, device)


class JointCoefficientReadout(nn.Module):
    """N: native only; S/D: native + prototype/direct evidence residual."""

    def __init__(self, native_cv4: nn.ModuleList, feature_channels: Sequence[int],
                 mode: str, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if mode not in ("N", "S", "D"):
            raise ValueError("mode must be N, S, or D")
        if len(native_cv4) != LEVELS or len(feature_channels) != LEVELS:
            raise ValueError("Expected three native one-to-one scales")
        self.mode = mode
        self.feature_channels = tuple(int(c) for c in feature_channels)
        self.native_cv4 = copy.deepcopy(native_cv4)
        for level, branch in enumerate(self.native_cv4):
            if not isinstance(branch, nn.Sequential) or len(branch) < 2:
                raise ValueError(f"Native branch {level} is not Sequential")
            final = branch[-1]
            if (not isinstance(final, nn.Conv2d) or final.in_channels != CHANNELS
                    or final.out_channels != COEFFICIENTS or final.kernel_size != (1, 1)
                    or final.stride != (1, 1) or final.padding != (0, 0)
                    or final.groups != 1):
                raise ValueError(f"Native branch {level} lacks original 64->32 readout")
        self.evidence = None if mode == "N" else CurrentEvidenceReadout(feature_channels, mode, config)
        self.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def trainable_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def parameter_counts(self):
        native = sum(p.numel() for p in self.native_cv4.parameters())
        extra = {} if self.evidence is None else self.evidence.parameter_counts()
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total, trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    native=native, new=total-native, evidence=extra)

    def current_selection(self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Any]]):
        if len(features) != LEVELS or any(f.ndim != 4 for f in features):
            raise ValueError("features must be three complete BCHW neck maps")
        batch, device = features[0].shape[0], features[0].device
        if len(selected) != batch:
            raise ValueError("features/selection batch mismatch")
        h_maps, c_maps = [], []
        for level, (branch, feature) in enumerate(zip(self.native_cv4, features)):
            if feature.device != device or feature.shape[:2] != (batch, self.feature_channels[level]):
                raise ValueError(f"Frozen feature scale {level} shape/device mismatch")
            h = feature.detach().float()
            for layer in list(branch.children())[:-1]:
                h = layer(h)
            if h.shape[1] != CHANNELS:
                raise ValueError("Native branch prefix must produce 64 channels")
            h_maps.append(h)
            c_maps.append(branch[-1](h))
        h_all = torch.cat([h.flatten(2) for h in h_maps], dim=2).transpose(1, 2)
        c_all = torch.cat([c.flatten(2) for c in c_maps], dim=2).transpose(1, 2)
        current = []
        for image_index, selection in enumerate(selected):
            raw = torch.as_tensor(selection["raw_ids"], device=device).detach()
            if raw.ndim != 1 or raw.dtype not in (torch.int32, torch.int64):
                raise ValueError("raw_ids must be a one-dimensional integer tensor")
            if bool(((raw < 0) | (raw >= h_all.shape[1])).any()):
                raise ValueError("raw_id outside native concatenated P3/P4/P5 grid")
            row = dict(selection)
            row["h_current"] = h_all[image_index, raw.long()]
            row["c_current"] = c_all[image_index, raw.long()]
            current.append(row)
        return current

    def forward_details(self, features, selected):
        current = self.current_selection(features, selected)
        if self.mode == "N":
            coefficients = [row["c_current"] for row in current]
            residuals = [c.new_zeros((len(c), 256)) for c in coefficients]
        else:
            coefficients, residuals = self.evidence.forward_with_evidence(features, current)
        return coefficients, residuals, current

    def forward_with_evidence(self, features, selected):
        coefficients, residuals, _ = self.forward_details(features, selected)
        return coefficients, residuals

    def forward(self, features, selected):
        coefficients, _, _ = self.forward_details(features, selected)
        return coefficients
