"""Low-rank position/prototype-conditioned dynamic prototype readout.

The native one-to-one cv4 branch remains the only source of the base
coefficient.  G and P differ by one input to the per-pixel prototype-channel
selection score: G sees query and box-relative coordinates; P additionally
sees the local prototype response.  The final residual is zero at
initialization, so the frozen model is reproduced before training.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import Tensor, nn

LEVELS, CHANNELS, COEFFICIENTS = 3, 64, 32
INPUT_SIZE = 640
PROTO_SIZE = 160


class DynamicPrototypeReadout(nn.Module):
    def __init__(self, native_cv4: nn.ModuleList, feature_channels: Sequence[int], mode: str, rho_init: float = 0.0):
        super().__init__()
        if mode not in ("N", "G", "P"):
            raise ValueError(mode)
        self.mode = mode
        self.rho_init = float(rho_init)
        self.feature_channels = tuple(int(x) for x in feature_channels)
        self.native_cv4 = copy.deepcopy(native_cv4)
        for branch in self.native_cv4:
            final = branch[-1]
            if not isinstance(final, nn.Conv2d) or final.in_channels != CHANNELS or final.out_channels != COEFFICIENTS:
                raise ValueError("unexpected native coefficient branch")
        if mode != "N":
            self.query = nn.Sequential(nn.Linear(CHANNELS, 24), nn.SiLU())
            self.channel_embed = nn.Parameter(torch.empty(COEFFICIENTS, 24))
            # With a non-zero residual amplitude, zero channel scores are needed
            # to preserve the native output exactly at initialization.
            if self.rho_init > 0:
                nn.init.zeros_(self.channel_embed)
            else:
                nn.init.normal_(self.channel_embed, std=0.02)
            self.position = nn.Linear(3, COEFFICIENTS, bias=True)
            self.proto_scale = nn.Linear(24, 1, bias=True)
            self.rho = nn.Linear(24, 1, bias=True)
            # zero residual and zero prototype contribution; G/P both start at native output
            nn.init.zeros_(self.position.weight); nn.init.zeros_(self.position.bias)
            nn.init.zeros_(self.proto_scale.weight); nn.init.zeros_(self.proto_scale.bias)
            nn.init.zeros_(self.rho.weight)
            if self.rho_init > 0:
                if not (0.0 < self.rho_init < 0.25):
                    raise ValueError('rho_init must lie in (0, 0.25)')
                nn.init.constant_(self.rho.bias, float(torch.atanh(torch.tensor(4.0 * self.rho_init))))
            else:
                nn.init.zeros_(self.rho.bias)
        self.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "native": sum(p.numel() for p in self.native_cv4.parameters()),
                "dynamic": 0 if self.mode == "N" else sum(p.numel() for n,p in self.named_parameters() if not n.startswith("native_cv4."))}

    def current(self, features):
        maps_h, maps_c = [], []
        for branch, feature in zip(self.native_cv4, features):
            h = feature.detach().float()
            for layer in list(branch.children())[:-1]:
                h = layer(h)
            maps_h.append(h)
            maps_c.append(branch[-1](h))
        h_all = torch.cat([h.flatten(2) for h in maps_h], 2).transpose(1, 2)
        c_all = torch.cat([c.flatten(2) for c in maps_c], 2).transpose(1, 2)
        return h_all, c_all, [h.shape[-2:] for h in maps_h]

    def forward(self, features: Sequence[Tensor], selected: Sequence[Mapping[str, Tensor]], return_base: bool = False):
        h_all, c_all, _ = self.current(features)
        device = features[0].device
        outputs = []
        bases = []
        # Fixed 160x160 proto grid.  Chunking by candidate controls memory.
        yy, xx = torch.meshgrid(torch.arange(PROTO_SIZE, device=device), torch.arange(PROTO_SIZE, device=device), indexing="ij")
        for b, row in enumerate(selected):
            raw = row["raw_ids"].long().detach()
            boxes = row["boxes"].float().detach()
            proto = row["proto"].float().detach()  # [32,160,160], frozen image-specific basis
            h = h_all[b, raw]
            c = c_all[b, raw]
            if self.mode == "N":
                # The training/evaluation contract is pixel logits for every arm;
                # project the native coefficient output through this image's frozen
                # prototype basis just as the official decoder does.
                native = (proto[None] * c[:, :, None, None]).sum(1)
                outputs.append(native)
                bases.append(native)
                continue
            n = len(raw)
            out_chunks = []
            base_chunks = []
            for start in range(0, n, 8):
                stop = min(n, start + 8)
                hh, cc = h[start:stop], c[start:stop]
                q = self.query(hh)
                base = torch.einsum("nd,kd->nk", q, self.channel_embed) / (24.0 ** 0.5)
                # Candidate-relative coordinates; outside-box pixels are harmless because
                # the official decoder crops with the same predicted box.
                cx = (boxes[start:stop, 0] + boxes[start:stop, 2]) * 0.5
                cy = (boxes[start:stop, 1] + boxes[start:stop, 3]) * 0.5
                bw = (boxes[start:stop, 2] - boxes[start:stop, 0]).clamp_min(1.0)
                bh = (boxes[start:stop, 3] - boxes[start:stop, 1]).clamp_min(1.0)
                rx = (xx[None].float() * INPUT_SIZE / PROTO_SIZE - cx[:, None, None]) / bw[:, None, None]
                ry = (yy[None].float() * INPUT_SIZE / PROTO_SIZE - cy[:, None, None]) / bh[:, None, None]
                phi = torch.stack((rx, ry, rx * ry), dim=-1)
                pos = self.position(phi).permute(0, 3, 1, 2)
                score = base[:, :, None, None] + pos
                if self.mode == "P":
                    pp = proto.permute(1, 2, 0)[None].expand(stop-start, -1, -1, -1)
                    pp = (pp - pp.mean((1, 2), keepdim=True)) / pp.std((1, 2), keepdim=True).clamp_min(1e-4)
                    scale = self.proto_scale(q).view(stop-start, 1, 1, 1)
                    score = score + scale * pp.permute(0, 3, 1, 2)
                weights = torch.softmax(score, dim=1)
                pp = proto[None].expand(stop-start, -1, -1, -1)
                z0 = (pp * cc[:, :, None, None]).sum(1)
                mixed = 32.0 * (weights * pp * cc[:, :, None, None]).sum(1)
                rho = 0.25 * torch.tanh(self.rho(q)).view(stop-start, 1, 1)
                out_chunks.append(z0 + rho * (mixed - z0))
                base_chunks.append(z0)
            outputs.append(torch.cat(out_chunks, dim=0))
            bases.append(torch.cat(base_chunks, dim=0))
        return (outputs, bases) if return_base else outputs
