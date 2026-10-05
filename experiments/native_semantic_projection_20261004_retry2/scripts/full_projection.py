"""Exact full predicted-box support ridge projection of continuous logits.

For supplied target_delta=t (the caller fixes t=.5*(z_flip-z0)), minimize
    mean_{u in S}(P_up(u)^T delta - t(u))**2 + .003*||delta||**2.
The 32x32 normal equations use FP64, all support pixels and exact chunks.
No GT, probability clipping, low-resolution pooling or optimizer is used.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F


RIDGE = .003
INTERPOLATION_ATOL = 1e-4
INTERPOLATION_RTOL = 3e-5
NORMAL_EQUATION_TOLERANCE = 1e-10


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")


def _deadline(value):
    if value is not None and time.monotonic() >= value:
        raise TimeoutError("Full support projection exceeded its fixed deadline")


@torch.no_grad()
def prepare_prototype(proto):
    if tuple(proto.shape) != (32, 160, 160) or not proto.is_floating_point():
        raise ValueError("Require complete [32,160,160] floating prototype")
    _finite("prototype", proto)
    # Preserve the official FP32 interpolation convention before promoting
    # chunk products to FP64. This allocation can be reused for the whole image.
    result = F.interpolate(proto.detach().float()[None], (640, 640), mode="bilinear",
                           align_corners=False)[0].contiguous()
    _finite("upsampled prototype", result)
    return result


@torch.no_grad()
def interpolation_audit(proto, coefficient, prototype_up=None):
    up = prepare_prototype(proto) if prototype_up is None else prototype_up
    coefficient = coefficient.to(proto.device, dtype=torch.float32)
    ordinary = F.interpolate((coefficient @ proto.float().flatten(1)).reshape(1, 1, 160, 160),
                             (640, 640), mode="bilinear", align_corners=False)[0, 0]
    reordered = (coefficient @ up.float().flatten(1)).reshape(640, 640)
    torch.testing.assert_close(reordered, ordinary, atol=INTERPOLATION_ATOL,
                               rtol=INTERPOLATION_RTOL)
    error = (reordered - ordinary).abs()
    scaled = error / (INTERPOLATION_ATOL + INTERPOLATION_RTOL * ordinary.abs())
    return dict(passed=True, max_absolute_error=float(error.max()),
                max_tolerance_ratio=float(scaled.max()), atol=INTERPOLATION_ATOL,
                rtol=INTERPOLATION_RTOL, ordinary_order="interpolate(P*c)",
                projection_order="interpolate(P)*c")


@torch.no_grad()
def solve_full_projection(proto, c0, box, target_delta, prototype_up=None,
                          chunk_pixels=16384, deadline_monotonic=None):
    _deadline(deadline_monotonic)
    if (tuple(proto.shape) != (32, 160, 160) or tuple(c0.shape) != (32,)
            or tuple(box.shape) != (4,) or tuple(target_delta.shape) != (640, 640)):
        raise ValueError("Full support projection input shapes changed")
    if int(chunk_pixels) <= 0:
        raise ValueError("chunk_pixels must be positive")
    device = proto.device
    if c0.device != device or box.device != device or target_delta.device != device:
        raise ValueError("All projection inputs must be on one device")
    for name, value in (("c0", c0), ("box", box), ("target_delta", target_delta)):
        _finite(name, value)
    up = prepare_prototype(proto) if prototype_up is None else prototype_up
    if tuple(up.shape) != (32, 640, 640) or up.device != device or up.dtype != torch.float32:
        raise ValueError("prototype_up must be FP32 [32,640,640] on the same device")
    _finite("prototype_up", up)
    interpolation = interpolation_audit(proto, c0, up)
    # Exact CUDA crop_mask convention: integer pixel coordinates compared to
    # the unrounded, unclipped frozen prediction. Do not force one empty pixel.
    yy, xx = torch.meshgrid(torch.arange(640, device=device), torch.arange(640, device=device),
                            indexing="ij")
    support = (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
    indices = support.flatten().nonzero(as_tuple=True)[0]
    count = int(indices.numel())
    c0d = c0.detach().double()
    flat = up.flatten(1)
    target = target_delta.detach().flatten()
    gram = torch.zeros((32, 32), device=device, dtype=torch.float64)
    rhs = torch.zeros(32, device=device, dtype=torch.float64)
    target_square = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, count, int(chunk_pixels)):
        _deadline(deadline_monotonic)
        selection = indices[start:start + int(chunk_pixels)]
        design = flat[:, selection].T.double()
        t = target[selection].double()
        gram.add_(design.T @ design)
        rhs.add_(design.T @ t)
        target_square.add_(t.square().sum())
    if count:
        gram.div_(count); rhs.div_(count); target_square.div_(count)
    system = gram + RIDGE * torch.eye(32, device=device, dtype=torch.float64)
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info) != 0:
        raise RuntimeError("Fixed positive-ridge FP64 normal equations failed Cholesky; no jitter allowed")
    delta = torch.cholesky_solve(rhs[:, None], factor)[:, 0]
    coefficients = c0d + delta
    residual_vector = system @ delta - rhs
    denom = torch.linalg.matrix_norm(system) * torch.linalg.vector_norm(delta) + torch.linalg.vector_norm(rhs)
    backward_error = float(torch.linalg.vector_norm(residual_vector) / denom) if float(denom) > 0 else 0.
    if not backward_error <= NORMAL_EQUATION_TOLERANCE:
        raise AssertionError(f"Normal equation relative backward error {backward_error} exceeds fixed tolerance")
    # Explicit second pass avoids cancellation in a nearly exact fit and
    # records the actual full-support error, not merely a quadratic expansion.
    data_residual = torch.zeros((), device=device, dtype=torch.float64)
    effect_square = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, count, int(chunk_pixels)):
        _deadline(deadline_monotonic)
        selection = indices[start:start + int(chunk_pixels)]
        effect = flat[:, selection].T.double() @ delta
        t = target[selection].double()
        data_residual.add_((effect - t).square().sum())
        effect_square.add_(effect.square().sum())
    if count:
        data_residual.div_(count); effect_square.div_(count)
    regularizer = RIDGE * delta.square().sum()
    objective = data_residual + regularizer
    for name, value in (("coefficient", coefficients), ("delta", delta), ("objective", objective)):
        _finite(name, value)
    # No view signal must be an exact identity, including a truly empty crop.
    zero_target = count == 0 or not bool(target[indices].ne(0).any())
    identity = not zero_target or bool(torch.equal(delta, torch.zeros_like(delta)))
    if not identity:
        raise AssertionError("Zero-target projection did not return exact original coefficients")
    projected_interpolation = interpolation_audit(proto, coefficients.float(), up)
    _deadline(deadline_monotonic)
    diagnostics = dict(passed=True, support_pixels=count, empty_support=count == 0,
                       ridge=RIDGE, chunk_pixels=int(chunk_pixels), dtype="float64 normal equations",
                       normal_equation_relative_backward_error=backward_error,
                       normal_equation_absolute_residual=float(torch.linalg.vector_norm(residual_vector)),
                       normal_equation_tolerance=NORMAL_EQUATION_TOLERANCE,
                       objective=float(objective), data_mse=float(data_residual),
                       increment_regularizer=float(regularizer), zero_increment_objective=float(target_square),
                       objective_change=float(objective-target_square),
                       gradient_norm=float(2*torch.linalg.vector_norm(residual_vector)),
                       delta_l2=float(torch.linalg.vector_norm(delta)),
                       target_delta_rms=float(target_square.sqrt()), projected_delta_rms=float(effect_square.sqrt()),
                       coefficient_cast_max_abs=float((coefficients.float().double()-coefficients).abs().max()),
                       zero_target=zero_target, zero_target_identity_passed=identity,
                       baseline_interpolation=interpolation, projected_interpolation=projected_interpolation,
                       loss_support="complete 640 integer grid inside frozen predicted box",
                       target_multiplier="already applied by caller; no extra scaling here")
    return dict(coefficients=coefficients, coefficients_fp32=coefficients.float(), delta=delta,
                diagnostics=diagnostics)

