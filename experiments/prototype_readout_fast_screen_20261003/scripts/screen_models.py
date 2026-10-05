"""Fast screen: unchanged native/point heads, five frozen 64-cell selectors.

U/Q/P call the previously checked implementations without modification. L
removes only the uncertainty multiplier from P. B uses P scores but alternates
the original-logit nonnegative/negative pools; an exhausted pool yields its
remaining quota to the other pool. One shared C is updated after every pick.
No GT, class labels, current learned coefficients, or new loss enter selection.
"""
from __future__ import annotations

import hashlib
import math

import torch

from joint_head import JointCoefficientReadout
from point_head import JointPointCoefficientReadout
from point_selectors import (build_selection_operators, UNIFORM_INDICES,
                             COEFFICIENTS, TOKENS, SELECTED_TOKENS, RIDGE, EPSILON_SCALE)

TRAIN_ARMS = ("N", "U", "Q", "P", "L", "B")
POINT_ARMS = ("U", "Q", "P", "L", "B")


def build_model(mode, replay, cfg):
    """Return a fresh native cv4 copy; the caller resets the paired seed."""
    if mode == "N":
        return JointCoefficientReadout(replay.native_cv4, replay.feature_channels, "N", cfg)
    if mode not in POINT_ARMS:
        raise ValueError(f"Unknown training arm: {mode}")
    model = JointPointCoefficientReadout(replay.native_cv4, replay.feature_channels,
                                        mode if mode in ("U", "Q", "P") else "P", cfg)
    # The inherited forward only distinguishes native-only N from evidence.
    # PointEvidenceReadout itself never consults this metadata attribute.
    model.mode = mode
    return model


@torch.no_grad()
def select_operators(A, G, c0, mode):
    """Low-level batched FP64 selector; U/Q/P remain the exact old function."""
    if mode in ("U", "Q", "P"):
        return build_selection_operators(A, G, c0, mode)
    if mode not in ("L", "B"):
        raise ValueError("Operator arm must be U/Q/P/L/B")
    A = torch.as_tensor(A).detach().to(dtype=torch.float64)
    device = A.device
    G = torch.as_tensor(G, device=device).detach().to(dtype=torch.float64)
    c0 = torch.as_tensor(c0, device=device).detach().to(dtype=torch.float64)
    if A.ndim != 3 or tuple(A.shape[1:]) != (TOKENS, COEFFICIENTS):
        raise ValueError("A must have shape [N,256,32]")
    n = len(A)
    if tuple(G.shape) != (n, 32, 32) or tuple(c0.shape) != (n, 32):
        raise ValueError("G/c0 shape mismatch")
    eye = torch.eye(32, dtype=torch.float64, device=device)
    uniform = torch.tensor(UNIFORM_INDICES, device=device, dtype=torch.long)
    if not n:
        return dict(indices=uniform.expand(0, -1).clone(), K=A.new_zeros((0, 32, 64)),
                    valid=torch.zeros(0, dtype=torch.bool, device=device), diagnostics={})
    finite = torch.isfinite(A).all((1, 2)) & torch.isfinite(G).all((1, 2)) & torch.isfinite(c0).all(1)
    a = torch.where(finite[:, None, None], A, torch.zeros_like(A))
    g = torch.where(finite[:, None, None], G, eye.expand(n, -1, -1))
    c = torch.where(finite[:, None], c0, torch.zeros_like(c0))
    symmetry = (g-g.transpose(1, 2)).abs().amax((1, 2))
    finite &= symmetry <= 1e-6*g.abs().amax((1, 2)).clamp_min(1e-12)
    g = (g+g.transpose(1, 2))*0.5
    epsilon = EPSILON_SCALE*(g.diagonal(dim1=1, dim2=2).sum(1)/32).clamp_min(1e-8)
    initial = RIDGE*g+epsilon[:, None, None]*eye
    _, initial_info = torch.linalg.cholesky_ex(initial, check_errors=False)
    valid = finite & (initial_info == 0)
    safe_initial = torch.where(valid[:, None, None], initial, eye.expand(n, -1, -1))
    a = torch.where(valid[:, None, None], a, torch.zeros_like(a))
    z0 = torch.bmm(a, c.unsqueeze(2)).squeeze(2)
    probability = z0.sigmoid()
    uncertainty = 4*probability*(1-probability)
    positive = z0 >= 0  # Exactly zero belongs to the nonnegative pool.
    working = safe_initial.clone()
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(working))
    indices = torch.empty((n, 64), device=device, dtype=torch.long)
    used = torch.zeros((n, 256), device=device, dtype=torch.bool)
    rows = torch.arange(n, device=device)
    minimum_quadratic = A.new_full((n,), float("inf"))
    fallback_steps = torch.zeros(n, device=device, dtype=torch.long)
    for step in range(64):
        quadratic = (torch.bmm(a, inverse)*a).sum(2)
        minimum_quadratic = torch.minimum(minimum_quadratic, quadratic.amin(1))
        score = torch.log1p(quadratic.clamp_min(0)/64)
        if mode == "B":
            score *= 0.1+uncertainty
            preferred = positive if step % 2 == 0 else ~positive
            eligible = preferred & ~used
            exhausted = ~eligible.any(1)
            fallback_steps += exhausted.long()
            eligible = torch.where(exhausted[:, None], ~used, eligible)
        else:
            eligible = ~used
        pick = score.masked_fill(~eligible, -torch.inf).argmax(1)
        indices[:, step] = pick  # First argmax gives lowest original grid ID.
        used[rows, pick] = True
        selected_a = a[rows, pick]
        v = torch.bmm(inverse, selected_a.unsqueeze(2)).squeeze(2)
        denominator = 64+(selected_a*v).sum(1)
        inverse -= v.unsqueeze(2)*v.unsqueeze(1)/denominator[:, None, None]
        working += selected_a.unsqueeze(2)*selected_a.unsqueeze(1)/64
        if (step+1) % 8 == 0 and step+1 < 64:
            _, info = torch.linalg.cholesky_ex(working, check_errors=False)
            valid &= info == 0
            safe_working = torch.where(valid[:, None, None], working, eye.expand(n, -1, -1))
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(safe_working))
    sampled = a.gather(1, indices[:, :, None].expand(-1, -1, 32))
    rhs = sampled.transpose(1, 2)/64
    system = safe_initial+torch.bmm(sampled.transpose(1, 2), sampled)/64
    _, solve_info = torch.linalg.cholesky_ex(system, check_errors=False)
    valid &= solve_info == 0
    safe_system = torch.where(valid[:, None, None], system, eye.expand(n, -1, -1))
    safe_rhs = torch.where(valid[:, None, None], rhs, torch.zeros_like(rhs))
    k = torch.cholesky_solve(safe_rhs, torch.linalg.cholesky(safe_system))
    denominator = safe_rhs.square().sum((1, 2)).sqrt().clamp_min(1e-30)
    residual64 = (torch.bmm(safe_system, k)-safe_rhs).square().sum((1, 2)).sqrt()/denominator
    residual32 = (torch.bmm(safe_system, k.float().double())-safe_rhs).square().sum((1, 2)).sqrt()/denominator
    indices = torch.where(valid[:, None], indices, uniform.expand(n, -1))
    k = torch.where(valid[:, None, None], k, torch.zeros_like(k))
    positive_count = positive.sum(1)
    target_positive = positive_count.clamp(max=32)+(32-(256-positive_count)).clamp_min(0)
    actual_positive = positive.gather(1, indices).sum(1)
    if mode == "B" and bool((valid & (actual_positive != target_positive)).any()):
        raise AssertionError("Balanced selection violated fixed 32/32 plus exhaustion fallback")
    return dict(indices=indices, K=k, valid=valid, diagnostics=dict(
        epsilon=epsilon, gram_symmetry_error=symmetry,
        initial_cholesky_info=initial_info, final_cholesky_info=solve_info,
        relative_residual_fp64=residual64, relative_residual_fp32=residual32,
        minimum_greedy_quadratic=minimum_quadratic,
        mean_selected_uncertainty=uncertainty.gather(1, indices).mean(1),
        original_nonnegative_cells=positive_count, selected_nonnegative_cells=actual_positive,
        target_nonnegative_cells=target_positive if mode == "B" else torch.full_like(target_positive, -1),
        exhausted_preferred_pool_steps=fallback_steps))


def tensor_sha(value):
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode()+str(list(x.shape)).encode()+x.numpy().tobytes()).hexdigest()


@torch.no_grad()
def build_operators(image, mode, cfg, device):
    """Return the existing forward_details selection dict plus JSON diagnostics.

    Call once before learning for each arm/image. The caller caches these fixed
    tensor values; neither the current trainable h/c nor labels are read here.
    Invalid original rows are retained with zero evidence and are counted.
    """
    if mode not in TRAIN_ARMS:
        raise ValueError("Unknown arm")
    for key, expected in dict(roi_side=16, sampled_cells=64, projection_lambda=0.1,
                              solver_jitter=1e-6, uncertainty_floor=0.1).items():
        if key in cfg and cfg[key] != expected:
            raise ValueError(f"Protocol fixes {key}={expected}")
    op = image.get("_operator", image.get("transfer_operator"))
    if op is None:
        raise KeyError("Original frozen prototype operator inputs are missing")
    c0 = torch.as_tensor(image["c0"], device=device).float().detach()
    selected = dict(raw_ids=torch.as_tensor(image["raw_ids"], device=device).long().detach(),
                    boxes=torch.as_tensor(image["boxes"], device=device).float().detach(),
                    h0=torch.as_tensor(op["h0"], device=device).float().detach(), c0=c0)
    if mode == "N":
        selected["_operator_diagnostics"] = dict(mode=mode, candidates=len(c0), native_only=True)
        return selected
    a = torch.as_tensor(op.get("A_full", op.get("A")), device=device).float().detach()
    g = torch.as_tensor(op["G"], device=device).detach()
    built = select_operators(a, g, c0, mode)
    original_valid = torch.as_tensor(op.get("valid", torch.ones(len(c0), dtype=torch.bool)), device=device).bool()
    if bool((original_valid & ~built["valid"]).any()):
        raise RuntimeError("Numerical operator failure on an originally valid candidate; do not filter")
    valid = built["valid"] & original_valid
    diag = built["diagnostics"]
    maxima = {}
    for key, tolerance in (("relative_residual_fp64", 1e-8), ("relative_residual_fp32", 1e-4)):
        value = float(diag[key][valid].max()) if bool(valid.any()) else 0.
        if not math.isfinite(value) or value > tolerance:
            raise AssertionError(f"{mode}: {key}={value} exceeds predeclared {tolerance}")
        maxima[key] = value
    ids = built["indices"].detach()
    if ids.shape != (len(c0), 64) or bool((ids.sort(1).values[:, 1:] == ids.sort(1).values[:, :-1]).any()):
        raise AssertionError("Each candidate must retain exactly 64 distinct cells")
    k = torch.where(valid[:, None, None], built["K"].float(), torch.zeros_like(built["K"], dtype=torch.float32))
    selected.update(A_full=a, indices=ids, K=k.detach(), valid=valid.detach())
    summary = dict(mode=mode, candidates=len(c0), invalid_original=int((~original_valid).sum()),
                   invalid_total=int((~valid).sum()), indices_sha256=tensor_sha(ids), K_sha256=tensor_sha(k),
                   selected_cells=64, fixed_original_c0=True, **maxima)
    for name in ("original_nonnegative_cells", "selected_nonnegative_cells", "target_nonnegative_cells",
                 "exhausted_preferred_pool_steps"):
        if name in diag:
            summary[name] = diag[name].cpu().tolist()
    selected["_operator_diagnostics"] = summary
    return selected
