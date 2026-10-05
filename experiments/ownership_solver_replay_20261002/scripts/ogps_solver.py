"""Frozen ownership-to-coefficient ridge replay; no learned parameters.

The official study fixes lambda=0.003 and an 8x8 predicted-box grid.  Each
row of A is the 32-channel prototype response pooled into one grid cell.
All *formal* arms call ``solve_ogps`` with probabilities, including soft GT
occupancy.  Probabilities are clipped to [0.01, 0.99], then converted to
logits.  A 7O OwnershipNet output is already a LOGIT: callers must apply
sigmoid exactly once before passing it here.  GT occupancy is already a
probability and must NOT receive a sigmoid.

For K=64, the objective is
    .5/K * ||A(c0+delta)-t||^2 + .5*0.003*||delta||^2.
Thus (A.T A/K + .003 I) delta = A.T(t-A c0)/K.  The regularizer is on the
coefficient OUTPUT INCREMENT, not on c or a network's parameters.  It is
not the official BCE objective, and the GT-soft-occupancy arm is a GT-
assisted diagnostic, NOT a theoretical upper bound on IoU or every solver.

Clipping changes even the baseline pseudo-target sigmoid(A c0), so that
formal arm need not reproduce c0.  ``baseline_clipping_audit`` records this
effect separately.  ``solve_ogps_logits`` exposes the very same linear
solve without probability conversion ONLY for an exact logit identity
check (t=A c0); it is not an additional formal arm or a clipping exception.

All solves and numerical diagnostics are FP64, on the input A's device.
The caller explicitly casts returned coefficients only when required by
the unchanged downstream decoder.  No optimizer, sweep, GT matching,
candidate selection, normal-image decoding, or files are handled here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


RIDGE_LAMBDA = 0.003
PROBABILITY_MIN = 0.01
PROBABILITY_MAX = 0.99
GRID_SIZE = 8
CELL_COUNT = GRID_SIZE * GRID_SIZE
COEFFICIENT_DIM = 32
INPUT_SIZE = 640


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


@torch.no_grad()
def crop_pool_7o(feature: Tensor, box: Tensor) -> Tensor:
    """Reproduce 7O crop_pool, including its minimum-one-cell convention.

    feature is [C,H,W], box is xyxy in the 640x640 letterbox input frame.
    This is floor/ceil indexing followed by adaptive AVERAGE pooling,
    not ROIAlign or interpolation.  GT masks and prototype maps may have
    different H/W, just as in the source 7O builder.  The caller must not
    substitute original-image boxes or rescale a GT box in place of the
    frozen prediction.  Degenerate/out-of-frame finite boxes are retained
    using the source's exact clamping convention, never silently dropped.

    Source: coefficient_ownership_readout_20260927/scripts/
            build_ownership_data.py:crop_pool.
    """
    if feature.ndim != 3 or min(feature.shape) < 1:
        raise ValueError("feature must be a nonempty [C,H,W] tensor")
    if box.shape != (4,):
        raise ValueError(f"box must have shape [4], got {tuple(box.shape)}")
    _require_finite("feature", feature)
    _require_finite("box", box)
    _, height, width = feature.shape
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / INPUT_SIZE)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / INPUT_SIZE)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / INPUT_SIZE)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / INPUT_SIZE)))
    return F.adaptive_avg_pool2d(feature[:, y1:y2, x1:x2], (GRID_SIZE, GRID_SIZE))


@torch.no_grad()
def pooled_design_matrix(proto: Tensor, boxes: Tensor) -> Tensor:
    """One full prototype [32,H,W] plus [4]/[N,4] boxes -> [64,32]/[N,64,32].

    Pooling uses FP32, matching 7O's ``cache['proto'].float()``; only the
    linear solve is then promoted to FP64.  The 64 rows are flattened in
    row-major (y,x) order, matching flattening of 7O [1,8,8] targets.
    Linearity makes A c0 equal pooled prototype logits up to FP32 roundoff
    BEFORE 7O's separate clamp[-30,30].  This function does not apply that
    nonlinear clamp to A or individual prototype channels.
    """
    if proto.ndim != 3 or proto.shape[0] != COEFFICIENT_DIM:
        raise ValueError("proto must have shape [32,H,W]")
    single = boxes.ndim == 1
    boxes_batch = boxes.unsqueeze(0) if single else boxes
    if boxes_batch.ndim != 2 or boxes_batch.shape[1] != 4:
        raise ValueError("boxes must have shape [4] or [N,4]")
    proto = proto.detach().to(dtype=torch.float32)
    matrices = [crop_pool_7o(proto, box).flatten(1).transpose(0, 1).contiguous()
                for box in boxes_batch]
    if not matrices:
        return proto.new_empty((0, CELL_COUNT, COEFFICIENT_DIM))
    result = torch.stack(matrices)
    return result[0] if single else result


def _prepare(A: Tensor, c0: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if A.ndim < 2 or A.shape[-2:] != (CELL_COUNT, COEFFICIENT_DIM):
        raise ValueError(f"A must end in [64,32], got {tuple(A.shape)}")
    prefix = A.shape[:-2]
    if c0.shape != prefix + (COEFFICIENT_DIM,):
        raise ValueError(f"c0 must have shape {prefix + (COEFFICIENT_DIM,)}")
    allowed = (prefix + (CELL_COUNT,), prefix + (GRID_SIZE, GRID_SIZE),
               prefix + (1, GRID_SIZE, GRID_SIZE))
    if target.shape not in allowed:
        raise ValueError(f"target shape {tuple(target.shape)} is not one of {allowed}")
    if A.device != c0.device or A.device != target.device:
        raise ValueError("A, c0 and target must be on the same device")
    if not (A.is_floating_point() and c0.is_floating_point() and target.is_floating_point()):
        raise TypeError("A, c0 and target must be floating-point tensors")
    for name, value in (("A", A), ("c0", c0), ("target", target)):
        _require_finite(name, value)
    return (A.detach().to(torch.float64), c0.detach().to(torch.float64),
            target.detach().reshape(prefix + (CELL_COUNT,)).to(torch.float64))


def _solve_prepared(A: Tensor, c0: Tensor, t: Tensor) -> dict[str, Any]:
    original_logits = (A @ c0.unsqueeze(-1)).squeeze(-1)
    gram = A.transpose(-1, -2) @ A / CELL_COUNT
    identity = torch.eye(COEFFICIENT_DIM, dtype=A.dtype, device=A.device)
    system = gram + RIDGE_LAMBDA * identity
    rhs = (A.transpose(-1, -2) @ (t - original_logits).unsqueeze(-1)).squeeze(-1) / CELL_COUNT
    factor, info = torch.linalg.cholesky_ex(system)
    if bool((info != 0).any()):
        # A fixed positive ridge makes this SPD in exact arithmetic.  Do not
        # silently change lambda, drop rows, add jitter, or invoke a new arm.
        raise RuntimeError(f"FP64 Cholesky failed at fixed lambda={RIDGE_LAMBDA}: {info}")
    delta = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
    coeff = c0 + delta
    fitted_logits = (A @ coeff.unsqueeze(-1)).squeeze(-1)
    residual = fitted_logits - t
    # Gradient of the declared half-MSE plus half-ridge objective.
    gradient = ((A.transpose(-1, -2) @ residual.unsqueeze(-1)).squeeze(-1)
                / CELL_COUNT + RIDGE_LAMBDA * delta)
    algebraic_residual = (system @ delta.unsqueeze(-1)).squeeze(-1) - rhs
    eigenvalues = torch.linalg.eigvalsh(system)
    if bool((eigenvalues[..., 0] <= 0).any()):
        raise RuntimeError("non-positive eigenvalue in fixed-ridge FP64 system")
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1)
    tiny = torch.finfo(torch.float64).tiny
    grad_norm = torch.linalg.vector_norm(gradient, dim=-1)
    data_term = 0.5 * residual.square().mean(-1)
    ridge_term = 0.5 * RIDGE_LAMBDA * delta.square().sum(-1)
    normal_residual_norm = torch.linalg.vector_norm(algebraic_residual, dim=-1)
    backward_scale = (eigenvalues[..., -1] * torch.linalg.vector_norm(delta, dim=-1)
                      + rhs_norm).clamp_min(tiny)
    diagnostics = {
        "residual_rms_before": (original_logits - t).square().mean(-1).sqrt(),
        "residual_rms_after": residual.square().mean(-1).sqrt(),
        "stationarity_l2": grad_norm,
        "stationarity_max_abs": gradient.abs().amax(-1),
        "relative_stationarity_l2": grad_norm / rhs_norm.clamp_min(tiny),
        "normal_equation_residual_l2": normal_residual_norm,
        "normal_equation_relative_backward_error": normal_residual_norm / backward_scale,
        "system_condition_2": eigenvalues[..., -1] / eigenvalues[..., 0],
        "system_min_eigenvalue": eigenvalues[..., 0],
        "system_max_eigenvalue": eigenvalues[..., -1],
        "delta_l2": torch.linalg.vector_norm(delta, dim=-1),
        "data_term": data_term,
        "ridge_term": ridge_term,
        "objective_before": 0.5 * (original_logits - t).square().mean(-1),
        "objective_after": data_term + ridge_term,
    }
    for name, value in diagnostics.items():
        _require_finite(name, value)
    return {"coeff": coeff, "delta": delta, "target_logits": t,
            "fitted_logits": fitted_logits, "diagnostics": diagnostics,
            "lambda": RIDGE_LAMBDA, "grid_size": GRID_SIZE,
            "solve_dtype": "float64"}


@torch.no_grad()
def solve_ogps(A: Tensor, c0: Tensor, probabilities: Tensor) -> dict[str, Any]:
    """Formal common solver: accepts probabilities in [0,1], not logits.

    A [...,64,32]; c0 [...,32]; probabilities [...,64], [...,8,8], or
    [...,1,8,8].  No broadcasting or candidate reordering is performed.
    Result diagnostics are tensors with the batch prefix shape; coeff is
    FP64 [...,32].  The caller owns arm labels and permanent candidate IDs.
    """
    A64, c064, q = _prepare(A, c0, probabilities)
    if bool(((q < 0) | (q > 1)).any()):
        raise ValueError("probabilities must be in [0,1]; convert ownership logits with sigmoid")
    clipped = q.clamp(PROBABILITY_MIN, PROBABILITY_MAX)
    target_logits = torch.log(clipped) - torch.log1p(-clipped)
    result = _solve_prepared(A64, c064, target_logits)
    result["probability_clip"] = [PROBABILITY_MIN, PROBABILITY_MAX]
    result["input_space"] = "probability"
    result["diagnostics"].update({
        "clipped_low_count": (q < PROBABILITY_MIN).sum(-1),
        "clipped_high_count": (q > PROBABILITY_MAX).sum(-1),
        "probability_clip_l1_mean": (clipped - q).abs().mean(-1),
    })
    return result


@torch.no_grad()
def solve_ogps_logits(A: Tensor, c0: Tensor, target_logits: Tensor) -> dict[str, Any]:
    """Diagnostic identity path only; same ridge but no probability clipping.

    Passing t=A c0 gives delta=0 in exact arithmetic.  Using logits avoids
    sigmoid saturation and a lossy logit(sigmoid(z)) round-trip.  This is
    deliberately a separately named API and is NOT a formal-arm override.
    """
    result = _solve_prepared(*_prepare(A, c0, target_logits))
    result["probability_clip"] = None
    result["input_space"] = "logit_identity_diagnostic_only"
    return result


@torch.no_grad()
def baseline_clipping_audit(A: Tensor, c0: Tensor) -> dict[str, Tensor]:
    """Quantify how the same formal clipping changes a baseline pseudo-target.

    The pseudo-probability is sigmoid(A c0), NOT adaptive_pool(sigmoid(Pc0)):
    sigmoid does not commute with average pooling.  7O pool-then-clamp[-30,30]
    and its stored FP32 values should be checked separately by the caller.
    """
    base = (A.detach().to(torch.float64) @ c0.detach().to(torch.float64).unsqueeze(-1)).squeeze(-1)
    formal = solve_ogps(A, c0, base.sigmoid())
    identity = solve_ogps_logits(A, c0, base)
    return {
        "unclipped_identity_coeff_max_abs": identity["delta"].abs().amax(-1),
        "formal_clipped_coeff_delta_l2": formal["diagnostics"]["delta_l2"],
        "formal_clipped_logit_delta_rms": (formal["fitted_logits"] - base).square().mean(-1).sqrt(),
        "target_clip_logit_delta_rms": (formal["target_logits"] - base).square().mean(-1).sqrt(),
        "clipped_low_count": formal["diagnostics"]["clipped_low_count"],
        "clipped_high_count": formal["diagnostics"]["clipped_high_count"],
    }


def mathematical_self_checks(device: str = "cpu") -> dict[str, Any]:
    """Small deterministic mathematical checks, executed ONLY when requested.

    No data/model is loaded and no training occurs.  Intended for execution
    on the authorized laptop/server; importing this module runs no checks.
    Tests check the normal equation, declared objective/gradient, batching,
    rank deficiency, clipping bias, and the source pooling identity.
    """
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    A = torch.randn(3, CELL_COUNT, COEFFICIENT_DIM, generator=generator,
                    dtype=torch.float64).to(device)
    c0 = torch.randn(3, COEFFICIENT_DIM, generator=generator,
                     dtype=torch.float64).to(device)
    target = torch.rand(3, 1, GRID_SIZE, GRID_SIZE, generator=generator,
                        dtype=torch.float64).to(device)
    fitted = solve_ogps(A, c0, target)
    identity = solve_ogps_logits(A, c0, (A @ c0.unsqueeze(-1)).squeeze(-1))
    assert float(identity["delta"].abs().max()) == 0.0
    separate = torch.stack([solve_ogps(A[i], c0[i], target[i])["coeff"] for i in range(3)])
    torch.testing.assert_close(fitted["coeff"], separate, atol=1e-10, rtol=1e-10)
    assert bool((fitted["diagnostics"]["objective_after"]
                 <= fitted["diagnostics"]["objective_before"] + 1e-12).all())
    assert float(fitted["diagnostics"]["stationarity_l2"].max()) < 1e-10
    # Independent autograd checks the gradient of the written objective,
    # not just a second invocation of the linear system formula.
    with torch.enable_grad():
        delta = fitted["delta"].detach().clone().requires_grad_(True)
        residual = (A @ (c0 + delta).unsqueeze(-1)).squeeze(-1) - fitted["target_logits"]
        objective = (0.5 * residual.square().mean(-1)
                     + 0.5 * RIDGE_LAMBDA * delta.square().sum(-1)).sum()
        gradient = torch.autograd.grad(objective, delta)[0]
    assert float(gradient.abs().max()) < 1e-10
    zero = solve_ogps(torch.zeros_like(A), c0, target)
    assert float(zero["delta"].abs().max()) == 0.0
    torch.testing.assert_close(zero["diagnostics"]["system_condition_2"],
                               torch.ones(3, dtype=torch.float64, device=device))
    # Closed-form diagonal system: clipping really must change large logits.
    diagonal = torch.eye(COEFFICIENT_DIM, dtype=torch.float64, device=device).repeat(2, 1)
    large_c = torch.zeros(COEFFICIENT_DIM, dtype=torch.float64, device=device)
    large_c[0] = 8.0
    base = diagonal @ large_c
    clipped = solve_ogps(diagonal, large_c, base.sigmoid())
    expected = (math.log(0.99 / 0.01) - 8.0) / (1.0 + 32 * RIDGE_LAMBDA)
    assert abs(float(clipped["delta"][0]) - expected) < 1e-10
    assert int(clipped["diagnostics"]["clipped_high_count"]) == 2
    # Source pooling vs linear prototype combination, before both clamps.
    proto = torch.randn(COEFFICIENT_DIM, 15, 17, generator=generator).to(device)
    boxes = torch.tensor([[10.2, 21.3, 511.4, 605.6], [-20., 400., 800., 900.]], device=device)
    matrices = pooled_design_matrix(proto, boxes)
    coeff32 = c0[0].float()
    full_logits = (coeff32 @ proto.flatten(1)).reshape(1, *proto.shape[1:])
    direct = torch.stack([crop_pool_7o(full_logits, box).flatten() for box in boxes])
    pooled = (matrices @ coeff32.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(direct, pooled, atol=1e-5, rtol=1e-5)
    return {"passed": True, "device": str(device), "lambda": RIDGE_LAMBDA,
            "max_stationarity_l2": float(fitted["diagnostics"]["stationarity_l2"].max()),
            "max_autograd_gradient_abs": float(gradient.abs().max()),
            "pooling_commutation_max_abs": float((direct - pooled).abs().max()),
            "clipping_changes_baseline": bool(float(clipped["delta"].norm()) > 0),
            "clipped_diagonal_delta0": float(clipped["delta"][0])}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
    else:
        print(json.dumps(mathematical_self_checks(args.device), indent=2))
