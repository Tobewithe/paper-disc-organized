"""TriFlow: continuous ownership potentials compiled into prototype coefficients.

This module deliberately has no learned 32-D residual output.  The learned
field is evaluated at *coordinates* (including normal/ray probes); a float32
weighted SPD solve converts its geometric requests into a coefficient update.
Frozen YOLO features, prototypes, coefficients and predicted neighbours are
the only forward inputs.  Ground truth appears exclusively in ``loss``.

Coordinates and distances use prototype-grid pixels.  Foreground has z > 0,
so the outward normal is -grad(z)/|grad(z)|.  At an approximate surface x0,
an outward displacement s requests p(x0).T dc = s*|grad(z)| - z(x0).
The minus sign in the supplied sketch applied to the *inward* normal and is
therefore corrected here.  Potential magnitude is proximity evidence, not an
exact Euclidean distance to an ownership separator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TriFlowConfig:
    token_count: int = 64
    hidden_dim: int = 128
    attention_heads: int = 4
    max_distance: float = 16.0
    max_displacement: float = 4.0
    ray_samples: int = 9
    normal_probe_step: float = 1.0
    boundary_band: float = 1.5
    boundary_temperature: float = 0.75
    competition_temperature: float = 1.0
    leverage_weight: float = 0.5
    boundary_token_fraction: float = 0.5
    competition_token_fraction: float = 0.25
    gram_epsilon: float = 1e-3
    ridge_lambda: float = 0.1
    numerical_ridge_epsilon_multiplier: float = 32.0
    compiler_passes: int = 3
    stiffness_floor: float = 0.1
    stiffness_calibration_weight: float = 0.1
    region_margin_max: float = 3.0
    region_margin_distance: float = 4.0
    max_logit_change: float = 2.0
    max_relative_coefficient_change: float = 0.25
    direction_weight: float = 0.2
    ordering_weight: float = 0.1
    task_weight: float = 1.0
    ordering_margin: float = 0.1
    dice_weight: float = 0.5
    gradient_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.token_count < 1 or self.ray_samples < 3:
            raise ValueError("token_count >= 1 and ray_samples >= 3 required")
        if self.compiler_passes < 1:
            raise ValueError("compiler_passes must be positive")
        if not 0 < self.stiffness_floor < 1:
            raise ValueError("stiffness_floor must lie strictly between 0 and 1")
        if not (0 < self.boundary_token_fraction <= 1 and 0 <= self.competition_token_fraction < 1 and self.boundary_token_fraction + self.competition_token_fraction <= 1):
            raise ValueError("token fractions must reserve a positive boundary quota and sum to <= 1")
        for name in ("max_distance", "max_displacement", "gram_epsilon", "ridge_lambda", "numerical_ridge_epsilon_multiplier", "max_logit_change", "max_relative_coefficient_change"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def spatial_gradient(values: Tensor) -> Tensor:
    """Central x/y differences, with one-sided image-edge differences."""
    if values.shape[-2] < 2 or values.shape[-1] < 2:
        raise ValueError("spatial grids must have H,W >= 2")
    dx = torch.empty_like(values)
    dy = torch.empty_like(values)
    dx[..., 1:-1] = 0.5 * (values[..., 2:] - values[..., :-2])
    dx[..., 0] = values[..., 1] - values[..., 0]
    dx[..., -1] = values[..., -1] - values[..., -2]
    dy[..., 1:-1, :] = 0.5 * (values[..., 2:, :] - values[..., :-2, :])
    dy[..., 0, :] = values[..., 1, :] - values[..., 0, :]
    dy[..., -1, :] = values[..., -1, :] - values[..., -2, :]
    return torch.stack((dx, dy), dim=-3)


def sample_grid(values: Tensor, xy: Tensor, reference_hw: tuple[int, int]) -> Tensor:
    """Sample N,C,h,w at N,...,2 prototype-grid coordinates; returns N,...,C.

    Normalizing against reference_hw allows the actual P3 feature map to have
    a different spatial size.  No dense feature interpolation is materialized.
    """
    n, channels = values.shape[:2]
    h, w = reference_hw
    query_shape = xy.shape[1:-1]
    scale = xy.new_tensor([max(w - 1, 1), max(h - 1, 1)])
    normalized = (xy / scale) * 2.0 - 1.0
    sampled = F.grid_sample(values, normalized.reshape(n, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[..., 0].transpose(1, 2).reshape(n, *query_shape, channels)


def _distance_to_region(region: Tensor) -> Tensor:
    """Exact pixel-centre Euclidean distance, GT-only and non-differentiable.

    SciPy EDT is preferred.  The fallback is exact chunked cdist (slower), not
    an approximation by morphology iterations or a zero-filled missing set.
    Empty regions are a caller error and are represented by validity masks.
    """
    if not bool(region.any()):
        raise ValueError("distance to an empty region is undefined")
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt
        result = distance_transform_edt(~region.detach().cpu().numpy().astype(np.bool_))
        return torch.as_tensor(result, device=region.device, dtype=torch.float32)
    except ImportError:
        h, w = region.shape
        grid = torch.stack(torch.meshgrid(torch.arange(h, device=region.device), torch.arange(w, device=region.device), indexing="ij"), -1).reshape(-1, 2).float()
        members = region.nonzero().float()
        distances = []
        for query in grid.split(256):
            best = query.new_full((query.shape[0],), float("inf"))
            for member in members.split(4096):
                best = torch.minimum(best, torch.cdist(query, member).min(1).values)
            distances.append(best)
        return torch.cat(distances).reshape(h, w)


@torch.no_grad()
def build_ownership_targets(
    self_masks: Tensor,
    neighbor_masks: Tensor | None = None,
    neighbor_valid: Tensor | None = None,
    *,
    background_masks: Tensor | None = None,
    valid_pixels: Tensor | None = None,
    max_distance: float = 16.0,
) -> dict[str, Tensor]:
    """Construct phi=(dN-dS,dB-dS), subtracting before clipping.

    ``neighbor_masks`` is the union of other *GT* instances (or N,K,H,W).
    ``background_masks`` should be supplied when that union excludes distant
    annotated objects: background is the complement of ALL labeled foreground.
    ``valid_pixels`` excludes crowd/ignore pixels.  Self has precedence over
    overlap, then Neighbor; missing N or B is explicitly invalid.  Prediction
    neighbour availability may further mask the N target via neighbor_valid.
    Placeholder zeros for missing fields are never supervised as distances.
    """
    if self_masks.ndim != 3:
        raise ValueError("self_masks must be N,H,W")
    self_masks = self_masks.bool()
    n, h, w = self_masks.shape
    if neighbor_masks is None:
        other = torch.zeros_like(self_masks)
    elif neighbor_masks.ndim == 4:
        other = neighbor_masks.bool().any(1)
    elif neighbor_masks.shape == self_masks.shape:
        other = neighbor_masks.bool()
    else:
        raise ValueError("neighbor_masks must be N,H,W or N,K,H,W")
    valid = torch.ones_like(self_masks) if valid_pixels is None else valid_pixels.bool()
    if valid.shape != self_masks.shape:
        raise ValueError("valid_pixels shape must match self_masks")
    self_region = self_masks & valid
    neighbor_region = other & ~self_masks & valid
    background_region = (~(self_masks | other) if background_masks is None else background_masks.bool()) & ~self_masks & ~other & valid
    if background_region.shape != self_masks.shape:
        raise ValueError("background_masks shape must match self_masks")
    prediction_n_valid = torch.ones(n, device=self_masks.device, dtype=torch.bool)
    if neighbor_valid is not None:
        prediction_n_valid = neighbor_valid.bool()
        if prediction_n_valid.ndim == 2:
            prediction_n_valid = prediction_n_valid.any(1)
        if prediction_n_valid.shape != (n,):
            raise ValueError("neighbor_valid must be N or N,K")
    phi = torch.zeros((n, 2, h, w), device=self_masks.device, dtype=torch.float32)
    field_valid = torch.zeros((n, 2), device=self_masks.device, dtype=torch.bool)
    for i in range(n):
        if not bool(self_region[i].any()):
            raise ValueError(f"instance {i} has no valid self GT pixels")
        d_self = _distance_to_region(self_region[i])
        for channel, region in enumerate((neighbor_region[i], background_region[i])):
            available = bool(region.any()) and (channel == 1 or bool(prediction_n_valid[i]))
            if available:
                # Crucially: do not clip dS/dN/dB separately before subtraction.
                phi[i, channel] = (_distance_to_region(region) - d_self).clamp(-max_distance, max_distance)
                field_valid[i, channel] = True
    return {"phi": phi, "potential_valid": field_valid, "valid_pixels": valid, "self_masks": self_masks, "neighbor_masks": neighbor_region, "background_masks": background_region}


class TriFlowModel(nn.Module):
    """Sparse continuous field + deterministic differentiable 32-D compiler."""

    def __init__(self, feature_channels: int, instance_hidden_channels: int = 32, config: TriFlowConfig | None = None):
        super().__init__()
        self.config = config or TriFlowConfig()
        self.feature_channels = int(feature_channels)
        self.instance_hidden_channels = int(instance_hidden_channels)
        if self.feature_channels < 1 or self.instance_hidden_channels < 1:
            raise ValueError("real visual and hidden channels must be positive")
        dim = self.config.hidden_dim
        # F + P + grad(P) + own/2-neighbour logits + grad(z) + xy_rel + h.
        input_dim = self.feature_channels + 32 + 64 + 3 + 2 + 2 + self.instance_hidden_channels
        self.token_projection = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.ownership_embeddings = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.ownership_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cross_attention = nn.MultiheadAttention(dim, self.config.attention_heads, batch_first=True, dropout=0.0)
        self.interaction_norm = nn.LayerNorm(dim)
        self.field_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.normal_(self.field_head[-1].weight, std=0.01)
        nn.init.zeros_(self.field_head[-1].bias)

    def configuration(self) -> dict[str, Any]:
        return {"feature_channels": self.feature_channels, "instance_hidden_channels": self.instance_hidden_channels, **asdict(self.config)}

    def _numerical_ridge(self, matrix: Tensor, existing_identity_ridge: Tensor | float) -> Tensor:
        """Predefined scale-aware FP32 diagonal floor, never a retry fallback.

        The recorded deterministic addition is part of the actual objective.
        Matrix scale is detached, so its numerical conditioning policy is not
        another learned weight.  Existing explicit identity ridge is retained.
        """
        scale = matrix.diagonal(dim1=-2, dim2=-1).abs().max(-1).values.detach()
        required = self.config.numerical_ridge_epsilon_multiplier * torch.finfo(torch.float32).eps * scale
        existing = torch.as_tensor(existing_identity_ridge, device=matrix.device, dtype=torch.float32).detach()
        return (required - existing).clamp_min(0)

    @staticmethod
    def _checked_cholesky(matrix: Tensor, label: str, identity_ridge: Tensor) -> tuple[Tensor, Tensor]:
        """Require actual FP32 positive definiteness rather than assuming it."""
        factor, info = torch.linalg.cholesky_ex(matrix.float(), check_errors=False)
        invalid = info != 0
        if bool(invalid.any()):
            rows = invalid.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            scales = matrix.diagonal(dim1=-2, dim2=-1).abs().max(-1).values.detach().cpu().tolist()
            floors = identity_ridge.detach().cpu().tolist()
            failure_info = info.detach().cpu().tolist()
            raise RuntimeError(f"{label} not FP32 SPD; failed_instances={rows}; cholesky_info={failure_info}; max_abs_diagonal={scales}; explicit_identity_ridge={floors}; no fallback attempted")
        return factor, info

    @staticmethod
    def _expand_map(values: Tensor, n: int, channels: int, name: str) -> Tensor:
        if values.ndim == 3:
            values = values.unsqueeze(0)
        if values.ndim != 4 or values.shape[1] != channels or values.shape[0] not in (1, n):
            raise ValueError(f"{name} must be {channels},H,W or N,{channels},H,W")
        return values.detach().float().expand(n, -1, -1, -1)

    def _prepare(self, prototypes: Tensor, coefficients: Tensor, visual: Tensor, boxes: Tensor, neighbor_logits: Tensor | None, neighbor_valid: Tensor | None, instance_hidden: Tensor | None) -> dict[str, Tensor | tuple[int, int]]:
        if coefficients.ndim != 2 or coefficients.shape[1] != 32 or coefficients.shape[0] < 1:
            raise ValueError("coefficients must be a nonempty N,32 tensor")
        n = coefficients.shape[0]
        p = self._expand_map(prototypes, n, 32, "prototypes")
        f = self._expand_map(visual, n, self.feature_channels, "visual")
        h, w = p.shape[-2:]
        c = coefficients.detach().float()
        if boxes.shape != (n, 4) or not torch.isfinite(boxes).all():
            raise ValueError("boxes must be finite N,4 grid xyxy")
        boxes = boxes.detach().float()
        if not bool(((boxes[:, 2:] - boxes[:, :2]) > 0).all()):
            raise ValueError("all boxes must have positive grid width and height")
        z = torch.einsum("nchw,nc->nhw", p, c)
        if neighbor_logits is None:
            neighbor_logits = z.new_empty((n, 0, h, w))
        if neighbor_logits.ndim != 4 or neighbor_logits.shape[0] != n or neighbor_logits.shape[-2:] != (h, w):
            raise ValueError("neighbor_logits must be N,K,H,W on the prototype grid")
        neighbors = neighbor_logits.detach().float()
        if neighbor_valid is None:
            neighbor_valid = torch.ones(neighbors.shape[:2], device=z.device, dtype=torch.bool)
        if neighbor_valid.shape != neighbors.shape[:2]:
            raise ValueError("neighbor_valid must be N,K")
        neighbor_valid = neighbor_valid.detach().bool()
        if neighbors.shape[1]:
            ranked = neighbors.masked_fill(~neighbor_valid[..., None, None], -1e4).topk(min(2, neighbors.shape[1]), dim=1).values
            if ranked.shape[1] == 1:
                ranked = torch.cat((ranked, ranked.new_zeros((n, 1, h, w))), 1)
            ranked_valid = torch.arange(2, device=z.device)[None] < neighbor_valid.sum(1)[:, None].clamp(max=2)
            ranked = ranked.masked_fill(~ranked_valid[..., None, None], 0.0)
        else:
            ranked = z.new_zeros((n, 2, h, w))
        hidden = c if instance_hidden is None else instance_hidden.detach().float()
        if hidden.shape != (n, self.instance_hidden_channels):
            raise ValueError("instance_hidden shape does not match constructor")
        yy, xx = torch.meshgrid(torch.arange(h, device=z.device), torch.arange(w, device=z.device), indexing="ij")
        xy = torch.stack((xx, yy), -1).float()[None].expand(n, -1, -1, -1)
        roi = self._inside(xy, boxes, (h, w))
        if not bool(roi.flatten(1).any(1).all()):
            raise ValueError("all boxes must contain at least one prototype pixel centre")
        if not all(bool(torch.isfinite(v).all()) for v in (p, f, c, z, neighbors, hidden)):
            raise ValueError("frozen inputs must be finite")
        return {"p": p, "f": f, "c": c, "z": z, "grad_p": spatial_gradient(p).flatten(1, 2), "grad_z": spatial_gradient(z).reshape(n, 2, h, w), "neighbors": neighbors, "neighbor_valid": neighbor_valid, "ranked_neighbors": ranked, "hidden": hidden, "boxes": boxes, "xy": xy, "roi": roi, "hw": (h, w)}

    @staticmethod
    def _inside(xy: Tensor, boxes: Tensor, hw: tuple[int, int]) -> Tensor:
        h, w = hw
        view = (boxes.shape[0],) + (1,) * (xy.ndim - 2)
        x, y = xy.unbind(-1)
        return (x >= boxes[:, 0].reshape(view)) & (x < boxes[:, 2].reshape(view)) & (y >= boxes[:, 1].reshape(view)) & (y < boxes[:, 3].reshape(view)) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)

    def _token_values(self, inputs: dict, xy: Tensor) -> Tensor:
        hw = inputs["hw"]
        local = [sample_grid(inputs["f"], xy, hw), sample_grid(inputs["p"], xy, hw), sample_grid(inputs["grad_p"], xy, hw), sample_grid(inputs["z"][:, None], xy, hw), sample_grid(inputs["ranked_neighbors"], xy, hw), sample_grid(inputs["grad_z"], xy, hw)]
        query_shape = xy.shape[1:-1]
        broadcast = (xy.shape[0],) + (1,) * len(query_shape) + (2,)
        boxes = inputs["boxes"]
        center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1.0)
        relative = (xy - center.reshape(broadcast)) / size.reshape(broadcast)
        hidden_shape = (xy.shape[0],) + (1,) * len(query_shape) + (self.instance_hidden_channels,)
        hidden = inputs["hidden"].reshape(hidden_shape).expand(*xy.shape[:-1], -1)
        return torch.cat((*local, relative, hidden), -1)

    def _gram_and_tokens(self, inputs: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        p = inputs["p"].flatten(2).transpose(1, 2)
        roi = inputs["roi"].flatten(1)
        count = roi.sum(1).float().clamp_min(1)
        gram = torch.einsum("nlc,nld,nl->ncd", p, p, roi.float()) / count[:, None, None]
        # Positive identity term prevents rank-deficient/zero prototypes failing.
        identity = torch.eye(32, device=p.device, dtype=torch.float32)[None]
        gram = gram + cfg.gram_epsilon * identity
        added_gram_ridge = self._numerical_ridge(gram, cfg.gram_epsilon)
        gram = gram + added_gram_ridge[:, None, None] * identity
        gram_identity_ridge = added_gram_ridge + cfg.gram_epsilon
        gram_factor, gram_info = self._checked_cholesky(gram, "ROI Gram/leverage", gram_identity_ridge)
        inputs["added_gram_numeric_ridge"] = added_gram_ridge
        inputs["gram_identity_ridge"] = gram_identity_ridge
        inputs["gram_cholesky_info"] = gram_info
        inverse_p = torch.cholesky_solve(p.transpose(1, 2), gram_factor).transpose(1, 2)
        leverage = (p * inverse_p).sum(-1).clamp_min(0)
        leverage_mean = (leverage * roi).sum(1) / count
        leverage = leverage / leverage_mean[:, None].clamp_min(cfg.gradient_epsilon)
        z = inputs["z"].flatten(1)
        boundary = torch.exp(-z.abs() / cfg.boundary_temperature)
        neighbor = inputs["ranked_neighbors"][:, 0].flatten(1)
        n_available = inputs["neighbor_valid"].any(1)
        competition = torch.exp(-(z - neighbor).abs() / cfg.competition_temperature) * z.sigmoid() * neighbor.sigmoid() * n_available[:, None]
        score = boundary + competition + cfg.leverage_weight * leverage
        score = score.masked_fill(~roi, -float("inf"))
        # Reserve quotas so high leverage cannot consume every boundary token.
        # Neighbour-free instances reassign the competition quota to boundary.
        # Selection is detached and ties resolve by flattened grid index. Small
        # ROIs pad deterministic repeats with zero observation weight.
        token_count = min(cfg.token_count, score.shape[1])
        index_rows, valid_rows, origin_rows = [], [], []
        for i in range(p.shape[0]):
            available_count = min(token_count, int(roi[i].sum()))
            boundary_quota = max(1, int(token_count * cfg.boundary_token_fraction))
            competition_quota = int(token_count * cfg.competition_token_fraction) if bool(n_available[i]) else 0
            if not bool(n_available[i]):
                boundary_quota += int(token_count * cfg.competition_token_fraction)
            leverage_quota = token_count - boundary_quota - competition_quota
            selected = torch.zeros_like(roi[i])
            chosen, origins = [], []
            for origin, component, quota in ((0, boundary[i], boundary_quota), (1, competition[i], competition_quota), (2, leverage[i], leverage_quota)):
                if quota < 1:
                    continue
                order = component.detach().masked_fill(~roi[i] | selected, -float("inf")).argsort(descending=True, stable=True)
                order = order[roi[i, order] & ~selected[order]][:quota]
                selected[order] = True
                chosen.append(order)
                origins.append(torch.full_like(order, origin))
            row = torch.cat(chosen)
            origin_row = torch.cat(origins)
            if row.numel() < available_count:
                remaining = score[i].detach().masked_fill(~roi[i] | selected, -float("inf")).argsort(descending=True, stable=True)
                remaining = remaining[roi[i, remaining] & ~selected[remaining]][:available_count - row.numel()]
                row = torch.cat((row, remaining))
                origin_row = torch.cat((origin_row, torch.full_like(remaining, 2)))
            validity = torch.arange(token_count, device=p.device) < row.numel()
            if row.numel() < token_count:
                missing = token_count - row.numel()
                row = torch.cat((row, row[:1].expand(missing)))
                origin_row = torch.cat((origin_row, torch.full((missing,), 3, device=p.device, dtype=torch.long)))
            index_rows.append(row)
            valid_rows.append(validity)
            origin_rows.append(origin_row)
        indices = torch.stack(index_rows)
        token_valid = torch.stack(valid_rows)
        inputs["token_origin"] = torch.stack(origin_rows)
        xy = inputs["xy"].flatten(1, 2).gather(1, indices[..., None].expand(-1, -1, 2))
        return gram, xy, token_valid, score.gather(1, indices)

    def _ownership_context(self, inputs: dict, xy: Tensor, token_valid: Tensor) -> tuple[Tensor, Tensor]:
        token = self.token_projection(self._token_values(inputs, xy))
        z = sample_grid(inputs["z"][:, None], xy, inputs["hw"])[..., 0]
        own = z.sigmoid()
        n_available = inputs["neighbor_valid"].any(1)
        neighbor = sample_grid(inputs["ranked_neighbors"][:, :1], xy, inputs["hw"])[..., 0].sigmoid() * n_available[:, None]
        background = 1.0 - torch.maximum(own, neighbor)
        weights = torch.stack((own, neighbor, background), 1) * token_valid[:, None]
        aggregates = torch.einsum("not,ntd->nod", weights, token) / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        context = self.ownership_projection(aggregates) + self.ownership_embeddings[None]
        padding = torch.zeros((token.shape[0], 3), device=token.device, dtype=torch.bool)
        padding[:, 1] = ~n_available
        return context, padding

    def _field(self, inputs: dict, xy: Tensor, context: Tensor, padding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        shape = xy.shape[1:-1]
        tokens = self.token_projection(self._token_values(inputs, xy)).reshape(xy.shape[0], -1, self.config.hidden_dim)
        attended, attention = self.cross_attention(tokens, context, context, key_padding_mask=padding, need_weights=True, average_attn_weights=True)
        features = self.interaction_norm(tokens + attended)
        raw = self.field_head(features).reshape(xy.shape[0], *shape, 3)
        phi = raw[..., :2].tanh() * self.config.max_distance
        stiffness = self.config.stiffness_floor + (1.0 - self.config.stiffness_floor) * raw[..., 2].sigmoid()
        return phi, stiffness, attention.reshape(xy.shape[0], *shape, 3)

    def _compile(self, inputs: dict, xy: Tensor, token_valid: Tensor, gram: Tensor, context: Tensor, padding: Tensor, phi: Tensor, stiffness: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        n, t = xy.shape[:2]
        hw = inputs["hw"]
        initial_z = sample_grid(inputs["z"][:, None], xy, hw)[..., 0]
        initial_gradient = sample_grid(inputs["grad_z"], xy, hw)
        initial_norm = initial_gradient.norm(dim=-1)
        initial_outward = -initial_gradient / initial_norm[..., None].clamp_min(cfg.gradient_epsilon)
        # One Newton step projects a near-band grid sample to the current level
        # set. Remaining z(x0) is explicitly corrected in the shape constraint.
        projection = (initial_z / initial_norm.clamp_min(cfg.gradient_epsilon)).clamp(-cfg.max_displacement, cfg.max_displacement)
        surface_xy = xy + projection[..., None] * initial_outward
        surface_z = sample_grid(inputs["z"][:, None], surface_xy, hw)[..., 0]
        gradient = sample_grid(inputs["grad_z"], surface_xy, hw)
        gradient_norm = gradient.norm(dim=-1)
        outward = -gradient / gradient_norm[..., None].clamp_min(cfg.gradient_epsilon)
        boundary_valid = token_valid & (initial_z.abs() <= cfg.boundary_band) & (initial_norm > cfg.gradient_epsilon) & (gradient_norm > cfg.gradient_epsilon) & self._inside(surface_xy, inputs["boxes"], hw)
        ray_s = torch.linspace(-cfg.max_displacement, cfg.max_displacement, cfg.ray_samples, device=xy.device, dtype=torch.float32)
        ray_xy = surface_xy[:, :, None] + ray_s[None, None, :, None] * outward[:, :, None]
        ray_phi, ray_stiffness, _ = self._field(inputs, ray_xy, context, padding)
        n_available = inputs["neighbor_valid"].any(1)
        phi_n = ray_phi[..., 0].masked_fill(~n_available[:, None, None], float("inf"))
        # Self is the intersection of both nonnegative potentials. Its boundary
        # is the envelope, not an arbitrary N zero inside an already-BG region.
        envelope = torch.minimum(phi_n, ray_phi[..., 1])
        left, right = envelope[..., :-1], envelope[..., 1:]
        ray_inside = self._inside(ray_xy, inputs["boxes"], hw)
        crossing = (left >= 0) & (right <= 0) & (left - right > cfg.gradient_epsilon) & ray_inside[..., :-1] & ray_inside[..., 1:] & boundary_valid[..., None]
        denominator = (left - right).clamp_min(cfg.gradient_epsilon)  # positive for the selected outward crossing
        fraction = (left / denominator).clamp(0.0, 1.0)
        root_s = ray_s[:-1][None, None] + fraction * (ray_s[1:] - ray_s[:-1])[None, None]
        nearest = root_s.detach().abs().masked_fill(~crossing, float("inf")).argmin(-1)
        root_valid = crossing.any(-1)
        displacement = root_s.gather(-1, nearest[..., None])[..., 0]
        displacement = torch.where(root_valid, displacement, torch.zeros_like(displacement))
        root_confidence = (ray_stiffness[..., :-1] * (1 - fraction) + ray_stiffness[..., 1:] * fraction).gather(-1, nearest[..., None])[..., 0]
        surface_p = sample_grid(inputs["p"], surface_xy, hw)
        boundary_b = displacement * gradient_norm - surface_z
        boundary_weight = root_confidence * root_valid

        # A fixed finite active-hinge approximation, not a global KKT solution.
        # Initial satisfied anchors exert no force; subsequent passes activate
        # any newly violated anchors caused by the joint boundary solve.
        valid_phi_n = phi[..., 0].masked_fill(~n_available[:, None], float("inf"))
        winner_n = (valid_phi_n < 0) & (valid_phi_n < phi[..., 1])
        winner_b = (phi[..., 1] < 0) & ~winner_n
        winner_s = ~winner_n & ~winner_b
        potential_magnitude = torch.where(winner_n, valid_phi_n.abs(), torch.where(winner_b, phi[..., 1].abs(), torch.minimum(valid_phi_n.abs(), phi[..., 1].abs())))
        margin = cfg.region_margin_max * torch.tanh(potential_magnitude / cfg.region_margin_distance)
        neighbor_z = sample_grid(inputs["ranked_neighbors"][:, :1], xy, hw)[..., 0]
        anchor_target = torch.where(winner_s, margin, torch.where(winner_n, neighbor_z - margin, -margin))
        # Each pass solves for the absolute delta from frozen c0, not an
        # accumulated incremental correction. The absolute boundary target is
        # unchanged, as is this continuous anchor target.
        anchor_b = anchor_target - initial_z
        anchor_p = sample_grid(inputs["p"], xy, hw)
        a = torch.cat((surface_p, anchor_p), 1)
        b = torch.cat((boundary_b, anchor_b), 1)

        def anchor_violation(candidate_z: Tensor) -> Tensor:
            return torch.where(winner_s, (anchor_target - candidate_z).clamp_min(0), (candidate_z - anchor_target).clamp_min(0)) * token_valid

        initial_violation = anchor_violation(initial_z)
        candidate_z = initial_z
        pass_active, pass_residual, pass_violation = [], [], []
        pass_numeric_ridge, pass_identity_ridge, pass_condition, pass_cholesky_info = [], [], [], []
        identity = torch.eye(32, device=xy.device, dtype=torch.float32)[None]
        for _ in range(cfg.compiler_passes):
            violation = anchor_violation(candidate_z)
            active_anchor = token_valid & (violation.detach() > cfg.gradient_epsilon)
            anchor_weight = stiffness * active_anchor
            weight = torch.cat((boundary_weight, anchor_weight), 1)
            effective_weight = weight.sum(1).clamp_min(1.0)
            normalized_weight = weight / effective_weight[:, None]
            lhs = torch.einsum("ntc,ntd,nt->ncd", a, a, normalized_weight) + cfg.ridge_lambda * gram
            rhs = torch.einsum("ntc,nt,nt->nc", a, b, normalized_weight)
            inherited_identity_ridge = cfg.ridge_lambda * inputs["gram_identity_ridge"]
            added_system_ridge = self._numerical_ridge(lhs, inherited_identity_ridge)
            lhs = lhs + added_system_ridge[:, None, None] * identity
            actual_identity_ridge = inherited_identity_ridge + added_system_ridge
            factor, info = self._checked_cholesky(lhs, f"compiler pass {len(pass_active) + 1}", actual_identity_ridge)
            delta_raw = torch.cholesky_solve(rhs[..., None].float(), factor)[..., 0]
            candidate_z = initial_z + torch.einsum("ntc,nc->nt", anchor_p, delta_raw)
            pass_active.append(active_anchor)
            pass_residual.append((torch.einsum("ncd,nd->nc", lhs, delta_raw) - rhs).norm(dim=1))
            pass_violation.append(anchor_violation(candidate_z))
            pass_numeric_ridge.append(added_system_ridge)
            pass_identity_ridge.append(actual_identity_ridge)
            pass_cholesky_info.append(info)
            with torch.no_grad():
                eigenvalues = torch.linalg.eigvalsh(lhs)
                pass_condition.append(eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-12))
        p = inputs["p"].flatten(2).transpose(1, 2)
        roi = inputs["roi"].flatten(1)
        change_raw = torch.einsum("nlc,nc->nl", p, delta_raw)
        max_raw = change_raw.abs().masked_fill(~roi, 0).max(1).values
        rms_raw = ((change_raw.square() * roi).sum(1) / roi.sum(1).clamp_min(1)).sqrt()
        coefficient_limit = cfg.max_relative_coefficient_change * inputs["c"].norm(dim=1).clamp_min(1e-3)
        coefficient_scale = coefficient_limit / delta_raw.norm(dim=1).clamp_min(cfg.gradient_epsilon)
        logit_scale = cfg.max_logit_change / max_raw.clamp_min(cfg.gradient_epsilon)
        trust_scale = torch.minimum(torch.minimum(coefficient_scale, logit_scale), torch.ones_like(logit_scale))
        delta = delta_raw * trust_scale[:, None]
        coefficients_refined = inputs["c"] + delta
        logits_refined = torch.einsum("nchw,nc->nhw", inputs["p"], coefficients_refined)
        residual = (torch.einsum("ncd,nd->nc", lhs, delta_raw) - rhs).norm(dim=1)
        refined_anchor_z = initial_z + torch.einsum("ntc,nc->nt", anchor_p, delta)
        final_violation = anchor_violation(refined_anchor_z)
        active_history = torch.stack(pass_active, 1)
        oscillation = torch.zeros(n, device=xy.device, dtype=torch.long)
        if cfg.compiler_passes >= 3:
            for i in range(2, cfg.compiler_passes):
                oscillation += ((active_history[:, i] == active_history[:, i - 2]) & (active_history[:, i] != active_history[:, i - 1]) & token_valid).sum(1)
        effective_row_residual = ((torch.einsum("ntc,nc->nt", a, delta) - b).square() * normalized_weight).sum(1).sqrt()
        with torch.no_grad():
            gram_eigenvalues = torch.linalg.eigvalsh(gram)
            system_eigenvalues = torch.linalg.eigvalsh(lhs)
            gram_condition = gram_eigenvalues[:, -1] / gram_eigenvalues[:, 0].clamp_min(1e-12)
            system_condition = system_eigenvalues[:, -1] / system_eigenvalues[:, 0].clamp_min(1e-12)
        field_selected = ray_phi.gather(2, nearest[..., None, None].expand(-1, -1, 1, 2))[..., 0, :]
        transition_neighbor = n_available[:, None] & (field_selected[..., 0] < field_selected[..., 1]) & root_valid
        return {"coefficients_refined": coefficients_refined, "logits_refined": logits_refined, "delta_coefficients": delta, "delta_coefficients_raw": delta_raw, "surface_xy": surface_xy, "outward_normal": outward, "displacement": displacement, "boundary_valid": boundary_valid, "root_valid": root_valid, "transition_neighbor": transition_neighbor, "anchor_active": active_anchor, "anchor_initial_active": pass_active[0], "region_winner": torch.where(winner_s, 0, torch.where(winner_n, 1, 2)), "region_margin": margin, "ray_phi": ray_phi, "ray_envelope": envelope, "diagnostics": {"boundary_valid_count": boundary_valid.sum(1), "root_count": root_valid.sum(1), "neighbor_transition_count": transition_neighbor.sum(1), "active_anchor_count": active_anchor.sum(1), "compiler_pass_count": torch.full((n,), cfg.compiler_passes, device=xy.device, dtype=torch.long), "pass_active_anchor_count": active_history.sum(-1), "pass_solve_residual": torch.stack(pass_residual, 1), "pass_anchor_violation_count": torch.stack(pass_violation, 1).gt(cfg.gradient_epsilon).sum(-1), "before_anchor_violation_count": initial_violation.gt(cfg.gradient_epsilon).sum(1), "after_anchor_violation_count": final_violation.gt(cfg.gradient_epsilon).sum(1), "before_anchor_violation_max": initial_violation.max(1).values, "after_anchor_violation_max": final_violation.max(1).values, "active_set_oscillation_count": oscillation, "solve_residual": residual, "weighted_constraint_residual": effective_row_residual, "added_gram_numeric_ridge": inputs["added_gram_numeric_ridge"], "gram_identity_ridge": inputs["gram_identity_ridge"], "gram_cholesky_info": inputs["gram_cholesky_info"], "added_system_numeric_ridge": torch.stack(pass_numeric_ridge, 1), "effective_system_identity_ridge": torch.stack(pass_identity_ridge, 1), "system_cholesky_info": torch.stack(pass_cholesky_info, 1), "pass_system_condition": torch.stack(pass_condition, 1), "gram_condition": gram_condition, "system_condition": system_condition, "trust_scale": trust_scale, "trust_saturated": trust_scale < 0.999999, "pre_logit_max": max_raw, "post_logit_max": max_raw * trust_scale, "pre_logit_rms": rms_raw, "post_logit_rms": rms_raw * trust_scale, "neighbor_valid": n_available, "stiffness_mean": (stiffness * token_valid).sum(1) / token_valid.sum(1).clamp_min(1), "gram_min_diagonal": gram.diagonal(dim1=-2, dim2=-1).min(1).values}}

    def forward(self, prototypes: Tensor, coefficients: Tensor, visual: Tensor, boxes: Tensor, neighbor_logits: Tensor | None = None, neighbor_valid: Tensor | None = None, instance_hidden: Tensor | None = None) -> dict[str, Any]:
        # AMP outside this module cannot demote either field or SPD arithmetic.
        with torch.autocast(device_type=prototypes.device.type, enabled=False):
            inputs = self._prepare(prototypes, coefficients, visual, boxes, neighbor_logits, neighbor_valid, instance_hidden)
            gram, xy, token_valid, score = self._gram_and_tokens(inputs)
            context, padding = self._ownership_context(inputs, xy, token_valid)
            phi, stiffness, ownership_attention = self._field(inputs, xy, context, padding)
            result = self._compile(inputs, xy, token_valid, gram, context, padding, phi, stiffness)
            result.update({"phi": phi, "stiffness": stiffness, "ownership_attention": ownership_attention, "token_xy": xy, "token_valid": token_valid, "token_origin": inputs["token_origin"], "token_score": score, "logits_initial": inputs["z"], "_inputs": inputs, "_context": context, "_padding": padding})
            return result

    def loss(self, outputs: dict[str, Any], self_masks: Tensor, neighbor_masks: Tensor | None = None, neighbor_valid: Tensor | None = None, *, targets: dict[str, Tensor] | None = None, background_masks: Tensor | None = None, valid_pixels: Tensor | None = None) -> dict[str, Tensor]:
        """Potential, same-head direction/ordering and compiled-mask task loss."""
        cfg = self.config
        inputs, context, padding = outputs["_inputs"], outputs["_context"], outputs["_padding"]
        device = outputs["phi"].device
        if targets is None:
            # Missing predicted neighbour masks remain invalid even if another
            # GT object exists. Ground truth never creates an inference query.
            predicted_valid = inputs["neighbor_valid"].any(1)
            if neighbor_valid is not None:
                supplied_valid = neighbor_valid.bool()
                if supplied_valid.ndim == 2:
                    supplied_valid = supplied_valid.any(1)
                predicted_valid = predicted_valid & supplied_valid.to(device)
            targets = build_ownership_targets(self_masks.to(device), None if neighbor_masks is None else neighbor_masks.to(device), predicted_valid, background_masks=None if background_masks is None else background_masks.to(device), valid_pixels=None if valid_pixels is None else valid_pixels.to(device), max_distance=cfg.max_distance)
        targets = {key: value.to(device) for key, value in targets.items()}
        xy, token_valid = outputs["token_xy"], outputs["token_valid"]
        hw = inputs["hw"]
        target_phi = targets["phi"].float()
        if target_phi.shape != (xy.shape[0], 2, *hw):
            raise ValueError("target potential grid must match prototypes")
        field_valid = targets["potential_valid"].bool().clone()
        field_valid[:, 0] &= inputs["neighbor_valid"].any(1)
        valid_map = targets.get("valid_pixels", torch.ones_like(self_masks, device=device, dtype=torch.bool))
        query_valid = sample_grid(valid_map[:, None].float(), xy, hw)[..., 0] >= 0.999
        supervision = token_valid[..., None] & query_valid[..., None] & field_valid[:, None]
        phi_true = sample_grid(target_phi, xy, hw)

        def masked_mean(value: Tensor, valid: Tensor) -> Tensor:
            return (value * valid).sum() / valid.sum().clamp_min(1)

        loss_potential_values = masked_mean(F.smooth_l1_loss(outputs["phi"], phi_true, reduction="none"), supervision)
        valid_true_n = phi_true[..., 0].abs().masked_fill(~field_valid[:, None, 0], float("inf"))
        valid_true_b = phi_true[..., 1].abs().masked_fill(~field_valid[:, None, 1], float("inf"))
        certainty = torch.minimum(valid_true_n, valid_true_b).clamp(max=cfg.max_distance)
        stiffness_target = cfg.stiffness_floor + (1.0 - cfg.stiffness_floor) * (0.5 + 0.5 * torch.tanh(certainty / 2.0))
        confidence_valid = token_valid & query_valid & field_valid.any(1)[:, None]
        loss_stiffness = masked_mean(F.smooth_l1_loss(outputs["stiffness"], stiffness_target, reduction="none"), confidence_valid)
        loss_potential = loss_potential_values + cfg.stiffness_calibration_weight * loss_stiffness

        # Derivatives query the exact SAME continuous field head at nearby
        # coordinates. They are not differences between unrelated sparse tokens.
        step = cfg.normal_probe_step
        cardinal = xy.new_tensor([[step, 0], [-step, 0], [0, step], [0, -step]])
        directional_xy = xy[:, :, None] + cardinal[None, None]
        predicted_probes, _, _ = self._field(inputs, directional_xy, context, padding)
        predicted_gradient = torch.stack(((predicted_probes[..., 0, :] - predicted_probes[..., 1, :]) / (2 * step), (predicted_probes[..., 2, :] - predicted_probes[..., 3, :]) / (2 * step)), -1)
        target_gradient = sample_grid(spatial_gradient(target_phi).flatten(1, 2), xy, hw).reshape(*xy.shape[:2], 2, 2)
        target_norm = target_gradient.norm(dim=-1)
        predicted_norm = predicted_gradient.norm(dim=-1)
        dir_inside = self._inside(directional_xy, inputs["boxes"], hw).all(-1)
        dir_pixels_valid = sample_grid(valid_map[:, None].float(), directional_xy, hw)[..., 0].ge(0.999).all(-1)
        direction_valid = supervision & (target_norm > 0.05) & (phi_true.abs() < cfg.max_distance - 0.5) & dir_inside[..., None] & dir_pixels_valid[..., None]
        cosine = (predicted_gradient * target_gradient).sum(-1) / (predicted_norm * target_norm).clamp_min(1e-6)
        loss_direction = masked_mean(1 - cosine.clamp(-1, 1), direction_valid)

        gt_outward = -target_gradient / target_norm[..., None].clamp_min(cfg.gradient_epsilon)
        ordering_center = xy[:, :, None].expand_as(gt_outward)
        ordering_xy = torch.stack((ordering_center - step * gt_outward, ordering_center, ordering_center + step * gt_outward), dim=-2)
        # Shape N,T,potential,probe,xy; central coordinate is repeated per field
        # intentionally, while context and head parameters remain identical.
        predicted_order_all, _, _ = self._field(inputs, ordering_xy, context, padding)
        truth_order_all = sample_grid(target_phi, ordering_xy, hw)
        field_index = torch.arange(2, device=device)[None, None, :, None, None].expand(xy.shape[0], xy.shape[1], -1, 3, 1)
        predicted_order = predicted_order_all.gather(-1, field_index)[..., 0]
        truth_order = truth_order_all.gather(-1, field_index)[..., 0]
        truth_decreasing = (truth_order[..., 0] > truth_order[..., 1] + 0.05) & (truth_order[..., 1] > truth_order[..., 2] + 0.05)
        actual_interface = (truth_order[..., 0] > 0) & (truth_order[..., 2] < 0)
        # Self↔N is meaningful only while BG permits Self, and vice versa.
        competing_phi = torch.stack((phi_true[..., 1], phi_true[..., 0]), -1)
        competing_valid = torch.stack((field_valid[:, 1], field_valid[:, 0]), -1)[:, None]
        competitor_permits_self = (~competing_valid) | (competing_phi >= -0.05)
        order_inside = self._inside(ordering_xy, inputs["boxes"], hw).all(-1)
        order_pixels_valid = sample_grid(valid_map[:, None].float(), ordering_xy, hw)[..., 0].ge(0.999).all(-1)
        order_valid = direction_valid & truth_decreasing & actual_interface & competitor_permits_self & order_inside & order_pixels_valid
        ordering_values = F.softplus(cfg.ordering_margin - (predicted_order[..., 0] - predicted_order[..., 1])) + F.softplus(cfg.ordering_margin - (predicted_order[..., 1] - predicted_order[..., 2]))
        loss_ordering = masked_mean(ordering_values, order_valid)

        masks = targets.get("self_masks", self_masks.to(device)).float()
        task_valid = inputs["roi"] & valid_map.bool()
        bce = F.binary_cross_entropy_with_logits(outputs["logits_refined"], masks, reduction="none")
        bce_per_instance = (bce * task_valid).flatten(1).sum(1) / task_valid.flatten(1).sum(1).clamp_min(1)
        probability = outputs["logits_refined"].sigmoid() * task_valid
        truth = masks * task_valid
        dice = 1 - (2 * (probability * truth).flatten(1).sum(1) + 1e-6) / (probability.flatten(1).sum(1) + truth.flatten(1).sum(1) + 1e-6)
        loss_task = (bce_per_instance + cfg.dice_weight * dice).mean()
        total = loss_potential + cfg.direction_weight * loss_direction + cfg.ordering_weight * loss_ordering + cfg.task_weight * loss_task
        return {"loss": total, "loss_potential": loss_potential, "loss_potential_values": loss_potential_values, "loss_stiffness": loss_stiffness, "loss_direction": loss_direction, "loss_ordering": loss_ordering, "loss_task": loss_task, "potential_supervised_count": supervision.sum(), "direction_supervised_count": direction_valid.sum(), "ordering_supervised_count": order_valid.sum(), "stiffness_supervised_count": confidence_valid.sum()}


__all__ = ["TriFlowConfig", "TriFlowModel", "build_ownership_targets", "sample_grid", "spatial_gradient"]
