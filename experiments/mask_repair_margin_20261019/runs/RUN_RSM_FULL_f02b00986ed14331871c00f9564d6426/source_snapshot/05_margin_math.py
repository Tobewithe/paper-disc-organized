"""Finite fixed-sign SOCP. No network or GT enters this solver."""
import hashlib
import time
import numpy as np


def geometry(B, coefficients, labels, cfg):
    H = B.T @ B / len(B)
    eigenvalues, vectors = np.linalg.eigh(H)
    if eigenvalues[0] <= eigenvalues[-1] * cfg['eigenvalue_ratio_min']:
        raise ArithmeticError('ill_conditioned_metric')
    T = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    S = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    X = B @ T
    rownorm = np.linalg.norm(X, axis=1)
    usable = rownorm > 1e-10
    if np.any(~usable & labels):
        raise ArithmeticError('positive_zero_response')
    scale = np.sqrt(np.mean((B @ coefficients)**2))
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ArithmeticError('zero_center_scale')
    A = X[usable] * (2 * labels[usable].astype(float) - 1)[:, None] / rownorm[usable, None]
    u0 = S @ coefficients / scale
    return A, T, u0, scale, rownorm, usable, eigenvalues


def recenter(B, coefficients, labels, cfg, budget_check=lambda: None):
    import clarabel
    from scipy import sparse
    start = time.monotonic()
    try:
        A, T, u0, scale, rownorm, usable, eigenvalues = geometry(B, coefficients, labels, cfg)
    except ArithmeticError as e:
        return coefficients.copy(), {'status': str(e), 'converged': False}
    initial = A @ u0
    active = np.unique(np.concatenate((np.argsort(initial, kind='stable')[:cfg['add_constraints']],
                        np.linspace(0, len(A)-1, min(cfg['initial_extra'], len(A)), dtype=int))))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = cfg['max_iter']
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = cfg['tol']
    settings.time_limit = cfg['time_limit']
    history = []
    converged = False
    u = u0.copy()
    for iteration in range(cfg['max_rounds']):
        budget_check()
        if time.monotonic()-start > cfg['pair_time_limit']:
            break
        # x=(u,t); A_conic x + slack = b, slack in nonnegative cone / SOC.
        linear = np.column_stack((-A[active], np.ones(len(active))))
        soc = np.zeros((33, 33)); soc[1:, :32] = -np.eye(32)
        matrix = sparse.csc_matrix(np.vstack((linear, soc)))
        rhs = np.r_[np.zeros(len(active)), 1., np.zeros(32)]
        q = np.r_[np.zeros(32), -1.]
        solver = clarabel.DefaultSolver(sparse.csc_matrix((33, 33)), q, matrix, rhs,
                      [clarabel.NonnegativeConeT(len(active)), clarabel.SecondOrderConeT(33)], settings)
        solution = solver.solve()
        if solution.x is None or not np.isfinite(solution.x).all():
            history.append({'round': iteration, 'status': str(solution.status)})
            break
        u = np.asarray(solution.x[:32]); t = float(solution.x[-1])
        values = A @ u
        violation = float(max(0., t-values.min(), np.linalg.norm(u)-1.))
        history.append({'round': iteration, 'constraints': len(active), 'status': str(solution.status),
                        't': t, 'full_violation': violation, 'iterations': solution.iterations,
                        'primal_residual': solution.r_prim, 'dual_residual': solution.r_dual})
        if str(solution.status) != 'Solved':
            break
        if violation <= cfg['full_feasibility_tol']:
            converged = True
            break
        worst = np.argsort(values, kind='stable')[:cfg['add_constraints']]
        new_active = np.unique(np.r_[active, worst])
        if len(new_active) == len(active):
            break
        active = new_active
    norm = np.linalg.norm(u)
    if not np.isfinite(norm) or norm <= 1e-10:
        return coefficients.copy(), {'status': 'zero_solution', 'converged': False, 'history': history}
    candidate = T @ (u/norm) * scale
    return candidate, {'status': 'converged' if converged else 'finite_budget_unresolved',
        'converged': converged, 'history': history, 'elapsed_seconds': time.monotonic()-start,
        'C_geometric_margin': float(initial.min()), 'candidate_geometric_margin': float((A @ (u/norm)).min()),
        'metric_min_eigenvalue': float(eigenvalues[0]), 'metric_max_eigenvalue': float(eigenvalues[-1]),
        'rows': len(B), 'nonzero_rows': int(usable.sum())}


def geometric_margin(B, coefficients, labels, cfg):
    try:
        A, _, u, _, _, _, _ = geometry(B, coefficients, labels, cfg)
        return float((A @ u).min())
    except ArithmeticError:
        return None


def draw(errors, mean, image_ids, image_id, rank, config):
    if image_id in image_ids:
        raise ValueError('pressure donor image leakage')
    seed = int.from_bytes(hashlib.sha256(f"RSM-v1|{config['seed']}|{image_id}|{rank}".encode()).digest()[:8], 'little')
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = {int(k): np.flatnonzero(image_ids == k) for k in np.unique(image_ids)}
    ids = sorted(groups)
    vectors, labels = [], []
    for i in range(config['donors_per_pair']):
        iid = ids[int(rng.integers(len(ids)))]; donor = int(rng.choice(groups[iid]))
        real = errors[donor]-mean
        random = rng.normal(size=32)
        permutation = rng.permutation(32)
        while np.any(permutation == np.arange(32)):
            permutation = rng.permutation(32)
        for arm, value in [('S',real), ('R',random), ('M',real[permutation])]:
            for sign in [1, -1]:
                vectors.append(sign*value)
                labels.append({'arm':arm,'draw':i,'sign':sign,'donor_index':donor,'donor_image_id':iid,
                               'permutation':permutation.tolist() if arm=='M' else None})
    return np.stack(vectors), labels
