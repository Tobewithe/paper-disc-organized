"""Finite fixed-P solver component. Native feature/decoder bridge is intentionally external.

This module does NOT turn old raw caches into final-output evidence and does NOT
compute IoU. Feed the frozen, documented design matrix and score the returned
coefficients through the verified native decoder in original-image coordinates.
"""
from __future__ import annotations

DEFAULTS = {
    "penalty": .003, "max_iter": 120, "max_eval": 150, "history_size": 20,
    "lr": 1.0, "tolerance_grad": 1e-9, "tolerance_change": 1e-12,
    "gradient_acceptance": 1e-6, "line_search_fn": "strong_wolfe",
    "dtype": "float64", "initialization": "zero_delta",
    "loss_normalization": "mean_over_frozen_roi_pixels",
}


def solve_coefficients(design, targets, c0, config=None):
    """design [pixels,K], targets [pixels] in [0,1], c0 [K]; no trainable base model."""
    import torch
    import torch.nn.functional as F
    cfg = dict(DEFAULTS if config is None else config)
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Freeze the complete solver config; missing/extra keys are not accepted")
    if cfg["dtype"] != "float64" or cfg["initialization"] != "zero_delta":
        raise ValueError("Only the declared float64 zero-delta solver is implemented")
    if cfg["loss_normalization"] != "mean_over_frozen_roi_pixels":
        raise ValueError("Unsupported loss normalization")
    if cfg["penalty"] <= 0 or cfg["max_iter"] <= 0 or cfg["max_eval"] <= 0:
        raise ValueError("Invalid solver budget or penalty")
    p, y, base = (x.detach().to(device=design.device, dtype=torch.float64) for x in (design, targets, c0))
    if p.ndim != 2 or y.shape != (p.shape[0],) or base.shape != (p.shape[1],) or not p.shape[0]:
        raise ValueError("Expected nonempty [pixels,K], [pixels], [K]")
    if not all(bool(torch.isfinite(x).all()) for x in (p, y, base)) or bool(((y < 0) | (y > 1)).any()):
        raise ValueError("Nonfinite inputs or targets outside [0,1]")
    delta = torch.zeros_like(base, requires_grad=True)
    def objective():
        return F.binary_cross_entropy_with_logits(p @ (base + delta), y) + cfg["penalty"] * delta.square().sum() / 2
    opt = torch.optim.LBFGS([delta], lr=cfg["lr"], max_iter=cfg["max_iter"],
        max_eval=cfg["max_eval"], history_size=cfg["history_size"],
        tolerance_grad=cfg["tolerance_grad"], tolerance_change=cfg["tolerance_change"],
        line_search_fn=cfg["line_search_fn"])
    def closure():
        opt.zero_grad()
        value = objective()
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite objective")
        value.backward()
        if not bool(torch.isfinite(delta.grad).all()):
            raise FloatingPointError("Nonfinite gradient")
        return value
    opt.step(closure)
    value = objective()
    grad = torch.autograd.grad(value, delta)[0]
    if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(grad).all()):
        raise FloatingPointError("Nonfinite solution")
    residual = float(grad.detach().abs().max())
    state = opt.state[delta]
    return {"coefficients": (base + delta).detach(), "delta": delta.detach(),
            "objective": float(value.detach()), "gradient_inf_norm": residual,
            "delta_norm": float(delta.detach().norm()),
            "iterations": int(state.get("n_iter", 0)), "function_evals": int(state.get("func_evals", 0)),
            "solver_status": "converged" if residual <= cfg["gradient_acceptance"] else "budget_exhausted"}


def positive_control(config=None):
    """Known nonzero optimum of the SAME regularized objective, solved on CPU.

Identity design and soft labels make c_star analytically stationary. This tests
the optimizer/regularizer/normalization; native mask rasterization and decoding
still need separate zero-delta and known-mask controls on actual cached P.
"""
    import torch
    cfg = dict(DEFAULTS if config is None else config)
    p = torch.eye(8, dtype=torch.float64)
    star = torch.tensor([.2, -.2] * 4, dtype=torch.float64)
    c0 = -star
    y = torch.sigmoid(star) + p.shape[0] * cfg["penalty"] * (star - c0)
    result = solve_coefficients(p, y, c0, cfg)
    error = float((result["coefficients"] - star).abs().max())
    return {"known_optimum_max_error": error, "gradient_inf_norm": result["gradient_inf_norm"],
            "passed": error <= 1e-4 and result["solver_status"] == "converged",
            "scope": "optimizer algebra only; not native decoder validation"}


if __name__ == "__main__":
    import json
    result = positive_control()
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
