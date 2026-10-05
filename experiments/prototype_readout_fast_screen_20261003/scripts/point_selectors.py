"""Frozen, GT-free 64-of-256 selection and prototype projection.

U: fixed uniform grid; Q: original-logit uncertainty; P: uncertainty-weighted
greedy log-determinant gain. All arms use the same final projection formula.
No model inference, optimization, or tensors are created at module import.
"""
from __future__ import annotations

from typing import Any

ROI_SIDE = 16
TOKENS = 256
SELECTED_TOKENS = 64
COEFFICIENTS = 32
RIDGE = 0.1
EPSILON_SCALE = 1e-6
UNIFORM_AXIS = (0, 2, 4, 6, 9, 11, 13, 15)
UNIFORM_INDICES = tuple(y * ROI_SIDE + x for y in UNIFORM_AXIS for x in UNIFORM_AXIS)


def build_selection_operators(A: Any, G: Any, c0: Any, mode: str) -> dict[str, Any]:
    """Batched FP64 construction on the input tensor's device.

    A [N,256,32] and G [N,32,32] use the frozen predicted-box grid/support;
    c0 [N,32] is the ORIGINAL frozen coefficient, never the current trainable
    coefficient. The caller freezes/saves these operators before training.

    Returns indices [N,64], K [N,32,64] (FP64), valid [N], and per-row tensor
    diagnostics. Nonfinite/non-SPD inputs retain their row, receive zero K and
    valid=False; shape/configuration mistakes raise. Invalid rows are NEVER
    silently dropped. The caller must report them before formal training.
    """
    import torch

    if mode not in ("U", "Q", "P"):
        raise ValueError("mode must be U, Q, or P")
    A = torch.as_tensor(A).detach().to(dtype=torch.float64)
    device = A.device
    G = torch.as_tensor(G, device=device).detach().to(dtype=torch.float64)
    c0 = torch.as_tensor(c0, device=device).detach().to(dtype=torch.float64)
    if A.ndim != 3 or tuple(A.shape[1:]) != (TOKENS, COEFFICIENTS):
        raise ValueError("A must be [N,256,32]")
    n = A.shape[0]
    if tuple(G.shape) != (n, COEFFICIENTS, COEFFICIENTS) or tuple(c0.shape) != (n, COEFFICIENTS):
        raise ValueError("G/c0 shape mismatch")
    eye = torch.eye(COEFFICIENTS, dtype=torch.float64, device=device)
    uniform = torch.tensor(UNIFORM_INDICES, dtype=torch.long, device=device)
    if n == 0:
        return dict(indices=uniform.expand(0, -1).clone(),
                    K=A.new_zeros((0, COEFFICIENTS, SELECTED_TOKENS)),
                    valid=torch.zeros(0, dtype=torch.bool, device=device), diagnostics={})
    valid_input = torch.isfinite(A).all(dim=(1, 2)) & torch.isfinite(G).all(dim=(1, 2))
    valid_input &= torch.isfinite(c0).all(dim=1)
    clean_a = torch.where(valid_input[:, None, None], A, torch.zeros_like(A))
    clean_g = torch.where(valid_input[:, None, None], G, eye.expand(n, -1, -1))
    clean_c = torch.where(valid_input[:, None], c0, torch.zeros_like(c0))
    symmetry_error = (clean_g - clean_g.transpose(1, 2)).abs().amax(dim=(1, 2))
    gscale = clean_g.abs().amax(dim=(1, 2)).clamp_min(1e-12)
    # A Gram matrix can have last-bit asymmetry in a serialized FP32 cache.
    valid_input &= symmetry_error <= 1e-6 * gscale
    clean_g = (clean_g + clean_g.transpose(1, 2)) * 0.5
    eps = EPSILON_SCALE * (clean_g.diagonal(dim1=1, dim2=2).sum(1) / COEFFICIENTS).clamp_min(1e-8)
    initial = RIDGE * clean_g + eps[:, None, None] * eye
    chol, initial_info = torch.linalg.cholesky_ex(initial, check_errors=False)
    valid = valid_input & (initial_info == 0)
    safe_initial = torch.where(valid[:, None, None], initial, eye.expand(n, -1, -1))
    clean_a = torch.where(valid[:, None, None], clean_a, torch.zeros_like(clean_a))
    probabilities = torch.sigmoid(torch.bmm(clean_a, clean_c.unsqueeze(2)).squeeze(2))
    uncertainty = 4.0 * probabilities * (1.0 - probabilities)
    minimum_quadratic = A.new_full((n,), float("inf"))

    if mode == "U":
        indices = uniform.expand(n, -1).clone()
    elif mode == "Q":
        # Stable descending sort makes the smaller original grid ID win ties.
        indices = torch.argsort(uncertainty, dim=1, descending=True, stable=True)[:, :SELECTED_TOKENS]
    else:
        working = safe_initial.clone()
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(working))
        indices = torch.empty((n, SELECTED_TOKENS), dtype=torch.long, device=device)
        used = torch.zeros((n, TOKENS), dtype=torch.bool, device=device)
        rows = torch.arange(n, device=device)
        for step in range(SELECTED_TOKENS):
            quadratic = (torch.bmm(clean_a, inverse) * clean_a).sum(2)
            minimum_quadratic = torch.minimum(minimum_quadratic, quadratic.amin(1))
            # A negative value here can only be roundoff for an SPD inverse.
            score = (0.1 + uncertainty) * torch.log1p(quadratic.clamp_min(0) / SELECTED_TOKENS)
            score = score.masked_fill(used, -torch.inf)
            pick = score.argmax(dim=1)  # First maximum = lowest grid ID.
            indices[:, step] = pick
            used[rows, pick] = True
            a = clean_a[rows, pick]
            v = torch.bmm(inverse, a.unsqueeze(2)).squeeze(2)
            denominator = SELECTED_TOKENS + (a * v).sum(1)
            inverse = inverse - v.unsqueeze(2) * v.unsqueeze(1) / denominator[:, None, None]
            working = working + a.unsqueeze(2) * a.unsqueeze(1) / SELECTED_TOKENS
            # Refresh avoids cumulative Sherman-Morrison roundoff; it does not
            # change the selected objective or introduce an additional ridge.
            if (step + 1) % 8 == 0 and step + 1 < SELECTED_TOKENS:
                refresh_chol, refresh_info = torch.linalg.cholesky_ex(working, check_errors=False)
                valid &= refresh_info == 0
                safe_working = torch.where(valid[:, None, None], working, eye.expand(n, -1, -1))
                inverse = torch.cholesky_inverse(torch.linalg.cholesky(safe_working))

    sampled = clean_a.gather(1, indices[:, :, None].expand(-1, -1, COEFFICIENTS))
    rhs = sampled.transpose(1, 2) / SELECTED_TOKENS
    system = safe_initial + torch.bmm(sampled.transpose(1, 2), sampled) / SELECTED_TOKENS
    solve_chol, solve_info = torch.linalg.cholesky_ex(system, check_errors=False)
    valid &= solve_info == 0
    safe_system = torch.where(valid[:, None, None], system, eye.expand(n, -1, -1))
    safe_rhs = torch.where(valid[:, None, None], rhs, torch.zeros_like(rhs))
    K = torch.cholesky_solve(safe_rhs, torch.linalg.cholesky(safe_system))
    denominator = safe_rhs.square().sum(dim=(1, 2)).sqrt().clamp_min(1e-30)
    residual64 = (torch.bmm(safe_system, K) - safe_rhs).square().sum(dim=(1, 2)).sqrt() / denominator
    K32 = K.float().double()
    residual32 = (torch.bmm(safe_system, K32) - safe_rhs).square().sum(dim=(1, 2)).sqrt() / denominator
    indices = torch.where(valid[:, None], indices, uniform.expand(n, -1))
    K = torch.where(valid[:, None, None], K, torch.zeros_like(K))
    if mode != "P":
        minimum_quadratic = torch.full_like(minimum_quadratic, float("nan"))
    diagnostics = dict(epsilon=eps, gram_symmetry_error=symmetry_error,
                       initial_cholesky_info=initial_info, final_cholesky_info=solve_info,
                       relative_residual_fp64=residual64, relative_residual_fp32=residual32,
                       minimum_greedy_quadratic=minimum_quadratic,
                       mean_selected_uncertainty=uncertainty.gather(1, indices).mean(1))
    return dict(indices=indices, K=K, valid=valid, diagnostics=diagnostics)


def select_and_project(A: Any, G: Any, c0: Any, mode: str) -> dict[str, Any]:
    """Single-instance wrapper; preserves device and returns FP64 K."""
    import torch
    A = torch.as_tensor(A)
    if tuple(A.shape) != (TOKENS, COEFFICIENTS):
        raise ValueError("A must be [256,32]")
    result = build_selection_operators(A.unsqueeze(0), torch.as_tensor(G).unsqueeze(0),
                                       torch.as_tensor(c0).unsqueeze(0), mode)
    return {key: ({name: value[0] for name, value in item.items()} if key == "diagnostics" else item[0])
            for key, item in result.items()}
