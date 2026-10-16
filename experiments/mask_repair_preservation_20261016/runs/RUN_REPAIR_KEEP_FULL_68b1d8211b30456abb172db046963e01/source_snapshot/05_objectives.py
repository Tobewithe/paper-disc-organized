"""Frozen, zero-network-training coefficient objectives and bounded solver.

z0 is the actual native FP32 input-grid logit, lifted to FP64. The design is
the real-valued bilinear map of P; the affine surrogate is z0 + design @ dc.
Final masks must always be evaluated with the original FP32 decoder.
"""
import time


def groups(z0, target):
    import torch
    if z0.shape != target.shape or z0.ndim != 1:
        raise ValueError('expected equal flattened ROI arrays')
    if not torch.isfinite(z0).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('invalid logits or binary targets')
    sign = 2 * target - 1
    error = (z0 > 0) != target.bool()
    keep = ~error
    margin = torch.minimum(torch.ones_like(z0), sign * z0)
    if (margin[keep] < 0).any():
        raise ValueError('negative correct-pixel margin')
    return sign, error, keep, margin


def objective(delta, design, z0, target, grouping, arm, cfg):
    import torch
    import torch.nn.functional as F
    z = z0 + design @ delta
    sign, error, keep, margin = grouping
    zero = delta.sum() * 0
    if arm == 'C':
        data = F.binary_cross_entropy_with_logits(z, target) if len(z) else zero
    elif arm in ('E', 'EK'):
        data = F.relu(cfg['error_margin'] - sign[error] * z[error]).square().mean() if error.any() else zero
        if arm == 'EK' and keep.any():
            data = data + cfg['protection_weight'] * F.relu(margin[keep] - sign[keep] * z[keep]).square().mean()
    else:
        raise ValueError('unknown objective arm')
    return data + cfg['penalty'] * delta.square().sum() / 2


def solve(design, z0, target, arm, cfg, budget_check=lambda: None):
    import torch
    p, z, y = [v.detach().to(device='cpu', dtype=torch.float64) for v in (design, z0, target)]
    if p.ndim != 2 or z.shape != (p.shape[0],) or y.shape != z.shape:
        raise ValueError('invalid design dimensions')
    if not all(torch.isfinite(v).all() for v in (p, z, y)):
        raise ValueError('nonfinite solver input')
    grouping = groups(z, y)
    delta = torch.zeros(p.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([delta], lr=cfg['lr'], max_iter=cfg['max_iter'],
        max_eval=cfg['max_eval'], history_size=cfg['history_size'],
        tolerance_grad=cfg['tolerance_grad'], tolerance_change=cfg['tolerance_change'],
        line_search_fn=cfg['line_search_fn'])
    evaluations, capped = 0, False
    best_value, best_delta = float('inf'), delta.detach().clone()
    started = time.monotonic()

    class EvaluationLimit(Exception):
        pass

    def closure():
        nonlocal evaluations, best_value, best_delta
        budget_check()
        if evaluations >= cfg['max_eval']:
            raise EvaluationLimit()
        evaluations += 1
        opt.zero_grad()
        value = objective(delta, p, z, y, grouping, arm, cfg)
        if not torch.isfinite(value):
            raise FloatingPointError('nonfinite objective')
        value.backward()
        if not torch.isfinite(delta.grad).all():
            raise FloatingPointError('nonfinite objective gradient')
        if value.item() < best_value:
            best_value, best_delta = value.item(), delta.detach().clone()
        return value

    if len(z):
        try:
            opt.step(closure)
        except EvaluationLimit:
            capped = True
            with torch.no_grad():
                delta.copy_(best_delta)
    value = objective(delta, p, z, y, grouping, arm, cfg)
    gradient = torch.autograd.grad(value, delta)[0]
    residual = float(gradient.abs().max())
    if not torch.isfinite(delta).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError('nonfinite final solution')
    return delta.detach(), dict(arm=arm, objective=float(value.detach()),
        gradient_inf_norm=residual, iterations=int(opt.state[delta].get('n_iter', 0)),
        function_evals=evaluations, hard_evaluation_cap_hit=capped,
        solver_status=('converged' if residual <= cfg['gradient_acceptance'] else 'budget_exhausted'),
        roi_pixels=len(z), error_pixels=int(grouping[1].sum()), correct_pixels=int(grouping[2].sum()),
        empty_roi_no_action=not len(z), elapsed_seconds=time.monotonic() - started)


def rms(x):
    return float(x.double().square().mean().sqrt()) if x.numel() else 0.0


def equal_scale(error_rms, protected_rms):
    if error_rms == 0:
        return (0.0, 'both_zero') if protected_rms == 0 else (0.0, 'undefined_zero_direction')
    return protected_rms / error_rms, 'defined'
