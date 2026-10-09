"""Explicit runtime diagnostic scopes for the unchanged frozen TriFlow method.

The minimal compiler is a controlled transcription of the SHA-locked original
_compile. It omits only diagnostic-only computations/outputs; actual three-pass
FP32 ridge/Cholesky solves, hinge selection, trust clipping and learned outputs
are retained. Full mode calls the original method. No global torch patching.
Skipped quantities are named as unmeasured, never represented by zero.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch
from torch import Tensor
import triflow_model as original
from triflow_model import TriFlowConfig, TriFlowModel, sample_grid

RUNTIME_VERSION = "triflow_diagnostics_runtime_v1"
ORIGINAL_MODEL_SOURCE_SHA256 = "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"
VALID_SCOPES = ("minimal", "full")
MINIMAL_KEYS = frozenset(("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count", "trust_saturated"))
FULL_DIAGNOSTIC_KEYS = ('boundary_valid_count', 'root_count', 'neighbor_transition_count', 'active_anchor_count', 'compiler_pass_count', 'pass_active_anchor_count', 'pass_solve_residual', 'pass_anchor_violation_count', 'before_anchor_violation_count', 'after_anchor_violation_count', 'before_anchor_violation_max', 'after_anchor_violation_max', 'active_set_oscillation_count', 'solve_residual', 'weighted_constraint_residual', 'added_gram_numeric_ridge', 'gram_identity_ridge', 'gram_cholesky_info', 'added_system_numeric_ridge', 'effective_system_identity_ridge', 'system_cholesky_info', 'pass_system_condition', 'gram_condition', 'system_condition', 'trust_scale', 'trust_saturated', 'pre_logit_max', 'post_logit_max', 'pre_logit_rms', 'post_logit_rms', 'neighbor_valid', 'stiffness_mean', 'gram_min_diagonal')


def source_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def guard_original_source():
    actual = source_sha256(original.__file__)
    if actual != ORIGINAL_MODEL_SOURCE_SHA256:
        raise RuntimeError("Runtime diagnostics require the locked original core bytes; got " + actual)


def _validate_scope(scope):
    if scope not in VALID_SCOPES:
        raise ValueError("diagnostic scope must be 'minimal' or 'full'")
    return scope


class TriFlowRuntimeModel(TriFlowModel):
    """Same parameters/configuration/forward/loss, with an explicit scope."""

    def __init__(self, feature_channels, instance_hidden_channels=32, config=None, *, diagnostic_scope="minimal"):
        guard_original_source()
        super().__init__(feature_channels, instance_hidden_channels, config)
        self.diagnostic_scope = _validate_scope(diagnostic_scope)

    def set_diagnostics(self, scope):
        self.diagnostic_scope = _validate_scope(scope)
        return self

    def _compile(self, *args, **kwargs):
        scope = _validate_scope(self.diagnostic_scope)
        result = super()._compile(*args, **kwargs) if scope == "full" else self._compile_minimal(*args, **kwargs)
        result["diagnostic_scope"] = scope
        result["diagnostics_unmeasured"] = [] if scope == "full" else [key for key in FULL_DIAGNOSTIC_KEYS if key not in MINIMAL_KEYS]
        return result

    def _compile_minimal(self, inputs: dict, xy: Tensor, token_valid: Tensor, gram: Tensor, context: Tensor, padding: Tensor, phi: Tensor, stiffness: Tensor) -> dict[str, Tensor]:
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

        candidate_z = initial_z
        first_active = None
        identity = torch.eye(32, device=xy.device, dtype=torch.float32)[None]
        for pass_index in range(cfg.compiler_passes):
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
            factor, info = self._checked_cholesky(lhs, f"compiler pass {pass_index + 1}", actual_identity_ridge)
            delta_raw = torch.cholesky_solve(rhs[..., None].float(), factor)[..., 0]
            candidate_z = initial_z + torch.einsum("ntc,nc->nt", anchor_p, delta_raw)
            if first_active is None:
                first_active = active_anchor
        p = inputs["p"].flatten(2).transpose(1, 2)
        roi = inputs["roi"].flatten(1)
        change_raw = torch.einsum("nlc,nc->nl", p, delta_raw)
        max_raw = change_raw.abs().masked_fill(~roi, 0).max(1).values
        coefficient_limit = cfg.max_relative_coefficient_change * inputs["c"].norm(dim=1).clamp_min(1e-3)
        coefficient_scale = coefficient_limit / delta_raw.norm(dim=1).clamp_min(cfg.gradient_epsilon)
        logit_scale = cfg.max_logit_change / max_raw.clamp_min(cfg.gradient_epsilon)
        trust_scale = torch.minimum(torch.minimum(coefficient_scale, logit_scale), torch.ones_like(logit_scale))
        delta = delta_raw * trust_scale[:, None]
        coefficients_refined = inputs["c"] + delta
        logits_refined = torch.einsum("nchw,nc->nhw", inputs["p"], coefficients_refined)

        field_selected = ray_phi.gather(2, nearest[..., None, None].expand(-1, -1, 1, 2))[..., 0, :]
        transition_neighbor = n_available[:, None] & (field_selected[..., 0] < field_selected[..., 1]) & root_valid
        return {"coefficients_refined": coefficients_refined, "logits_refined": logits_refined, "delta_coefficients": delta, "delta_coefficients_raw": delta_raw, "surface_xy": surface_xy, "outward_normal": outward, "displacement": displacement, "boundary_valid": boundary_valid, "root_valid": root_valid, "transition_neighbor": transition_neighbor, "anchor_active": active_anchor, "anchor_initial_active": first_active, "region_winner": torch.where(winner_s, 0, torch.where(winner_n, 1, 2)), "region_margin": margin, "ray_phi": ray_phi, "ray_envelope": envelope, "diagnostics": {"boundary_valid_count": boundary_valid.sum(1), "root_count": root_valid.sum(1), "neighbor_transition_count": transition_neighbor.sum(1), "active_anchor_count": active_anchor.sum(1), "trust_saturated": trust_scale < 0.999999}}


def install(module_class, mode="minimal"):
    """Return a scoped class; callers explicitly replace their local binding.

    The original module/class and torch remain unchanged. A mismatched or already
    replaced class is rejected, avoiding hidden stacking of runtime adapters.
    """
    guard_original_source()
    _validate_scope(mode)
    if module_class is not TriFlowModel:
        raise TypeError("install requires the exact original triflow_model.TriFlowModel")

    class ConfiguredTriFlowRuntimeModel(TriFlowRuntimeModel):
        def __init__(self, feature_channels, instance_hidden_channels=32, config=None, *, diagnostic_scope=mode):
            super().__init__(feature_channels, instance_hidden_channels, config, diagnostic_scope=diagnostic_scope)
    return ConfiguredTriFlowRuntimeModel


def set_diagnostics(module, scope):
    if not isinstance(module, TriFlowRuntimeModel):
        raise TypeError("Explicit runtime model required for a diagnostic scope")
    return module.set_diagnostics(scope)


@contextmanager
def diagnostics_scope(module, scope):
    previous = module.diagnostic_scope
    set_diagnostics(module, scope)
    try:
        yield module
    finally:
        set_diagnostics(module, previous)
