"""Frozen cross-task features bridge into the live native cv4 representation.

The readout is exactly W(h + bridge(t)) + b.  We evaluate it as the original
convolution plus W bridge(t), preserving the original dense-convolution path
bit-for-bit when the added bridge is zero. No detector parameters live here.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
import torch.nn.functional as F


class BridgeReadout(nn.Module):
    def __init__(self, native_cv4, feature_channels, mode, cfg, stats):
        super().__init__()
        if mode not in ('N', 'T', 'R', 'M'):
            raise ValueError('Expected N/T/R/M')
        if len(native_cv4) != 3 or len(feature_channels) != 3:
            raise ValueError('Exactly three original head scales are required')
        self.mode = mode
        self.feature_channels = tuple(int(x) for x in feature_channels)
        self.native_cv4 = copy.deepcopy(native_cv4)
        self.bridges = nn.ModuleList()
        self.source_kind = 'box' if mode == 'R' else 'cls'
        self.source_dim = 64 if mode == 'R' else 256
        for level, branch in enumerate(self.native_cv4):
            last = branch[-1]
            if (not isinstance(branch, nn.Sequential) or not isinstance(last, nn.Conv2d)
                    or last.in_channels != 64 or last.out_channels != 32
                    or last.kernel_size != (1, 1) or last.stride != (1, 1)
                    or last.padding != (0, 0) or last.groups != 1):
                raise ValueError('Expected actual native 64->32 affine final convolution')
            if mode != 'N':
                stat = stats[self.source_kind][level]
                mean = torch.as_tensor(stat['mean']).detach().clone().float()
                std = torch.as_tensor(stat['std']).detach().clone().float()
                if (mean.shape != (self.source_dim,) or std.shape != mean.shape
                        or int(stat['count']) <= 0 or not bool(torch.isfinite(mean).all())
                        or not bool(torch.isfinite(std).all()) or not bool((std >= 1e-6).all())):
                    raise ValueError('Invalid fixed per-scale fit normalization')
                self.register_buffer(f'source_mean_{level}', mean)
                self.register_buffer(f'source_std_{level}', std)
                bridge = nn.Linear(self.source_dim, 64, bias=True)
                nn.init.zeros_(bridge.weight)
                nn.init.zeros_(bridge.bias)
                self.bridges.append(bridge)
        self.requires_grad_(True)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def trainable_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def parameter_counts(self):
        native = sum(p.numel() for p in self.native_cv4.parameters())
        per_level = [sum(p.numel() for p in b.parameters()) for b in self.bridges]
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total,
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    native=native, new=total-native,
                    bridges={} if self.mode == 'N' else dict(
                        input_dim=self.source_dim, per_level=per_level, total=sum(per_level)))

    def forward_details(self, features, selections):
        if len(features) != 3 or any(f.ndim != 4 for f in features):
            raise ValueError('Expected full BCHW P3/P4/P5 frozen inputs')
        batch, device = features[0].shape[0], features[0].device
        if len(selections) != batch:
            raise ValueError('Input/selection batch length differs')
        coeff_maps, hidden_maps, raw_levels = [], [], []
        for level, (branch, f) in enumerate(zip(self.native_cv4, features)):
            if tuple(f.shape[:2]) != (batch, self.feature_channels[level]) or f.device != device:
                raise ValueError('Wrong frozen scale input')
            h = branch[:-1](f.detach().float())
            hidden_maps.append(h.flatten(2))
            coeff_maps.append(branch[-1](h).flatten(2))
            raw_levels.append(torch.full((h.shape[-2]*h.shape[-1],), level,
                                         dtype=torch.long, device=device))
        coefficients = torch.cat(coeff_maps, 2).transpose(1, 2)
        hidden = torch.cat(hidden_maps, 2).transpose(1, 2)
        all_levels = torch.cat(raw_levels)
        result, residuals, current_rows = [], [], []
        for j, selected in enumerate(selections):
            raw = torch.as_tensor(selected['raw_ids'], device=device).detach()
            levels = torch.as_tensor(selected['levels'], device=device).detach()
            if (raw.ndim != 1 or raw.dtype not in (torch.int32, torch.int64)
                    or levels.shape != raw.shape or levels.dtype not in (torch.int32, torch.int64)
                    or bool(((raw < 0) | (raw >= coefficients.shape[1])).any())):
                raise ValueError('Bad raw candidate indices')
            if not torch.equal(all_levels[raw.long()], levels.long()):
                raise AssertionError('Level/raw identity mismatch')
            current = coefficients[j, raw.long()]
            current_h = hidden[j, raw.long()]
            if self.mode == 'N':
                result.append(current)
                residuals.append(torch.zeros_like(current))
                current_rows.append(dict(h_current=current_h, c_current=current,
                                         h_bridge=torch.zeros_like(current_h)))
                continue
            source = torch.as_tensor(selected['source_features'], device=device).detach().float()
            if source.shape != (len(raw), self.source_dim) or not bool(torch.isfinite(source).all()):
                raise ValueError('Invalid frozen detector source feature')
            residual = torch.zeros_like(current)
            h_bridge = torch.zeros_like(current_h)
            for level in range(3):
                positions = torch.where(levels == level)[0]
                mean = getattr(self, f'source_mean_{level}')
                std = getattr(self, f'source_std_{level}')
                delta_h = self.bridges[level]((source[positions]-mean)/std)
                last = self.native_cv4[level][-1]
                delta_c = F.linear(delta_h, last.weight[:, :, 0, 0], bias=None)
                residual = residual.index_copy(0, positions, delta_c)
                h_bridge = h_bridge.index_copy(0, positions, delta_h)
            result.append(current + residual)
            residuals.append(residual)
            current_rows.append(dict(h_current=current_h, c_current=current, h_bridge=h_bridge))
        return result, residuals, current_rows

    def forward(self, features, selections):
        return self.forward_details(features, selections)[0]
