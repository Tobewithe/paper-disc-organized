"""Analytic-gradient refinement of the shared affine convex objective."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
import shared_affine_audit as audit


LAM = .003


def solve_level(rows, start, mean, std, maxiter):
    n = len(rows)
    H = [np.concatenate([((r["h"].numpy() - mean.numpy()) / std.numpy()).astype(np.float64), np.ones(1)]) for r in rows]
    P = [r["p"].numpy().astype(np.float64) for r in rows]
    Y = [r["y"].numpy().astype(np.float64) for r in rows]
    C = [r["c"].numpy().astype(np.float64) for r in rows]
    areas = [r["area"] for r in rows]

    def fg(x):
        A = x.reshape(65, 32)
        value = 0.0
        grad = np.zeros_like(A)
        for h, p, y, c, area in zip(H, P, Y, C, areas):
            d = h @ A
            z = p @ (c + d)
            bce = np.logaddexp(0.0, z) - y * z
            value += bce.sum() / area + LAM * np.dot(d, d) / 2
            gd = (p.T @ (expit(z) - y)) / area + LAM * d
            grad += np.outer(h, gd)
        return value / n, (grad / n).ravel()

    result = minimize(fg, start.reshape(-1), jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50, "maxcor": 50})
    A = torch.from_numpy(result.x.reshape(65, 32)).double()
    return A, {"success": bool(result.success), "message": str(result.message), "nit": int(result.nit),
               "nfev": int(result.nfev), "objective": float(result.fun), "gradient_inf": float(np.max(np.abs(result.jac)))}


def main(a):
    torch.set_num_threads(a.threads)
    fit = audit.load_rows(a.fit)
    old = torch.load(a.affine, map_location="cpu", weights_only=False)
    stats = old["stats"]
    Anew, reports = [], []
    for lv in range(3):
        rows = [r for r in fit if r["level"] == lv]
        t0 = time.monotonic()
        A, rep = solve_level(rows, old["A"][lv], stats[lv][0], stats[lv][1], a.maxiter)
        rep["level"] = lv; rep["records"] = len(rows); rep["elapsed"] = time.monotonic() - t0
        print(json.dumps(rep), flush=True)
        Anew.append(A); reports.append(rep)
    value, bce, reg = audit.affine_objective(Anew, fit, stats=stats, chunk_size=a.chunk_size)
    out = {"levels": reports, "fit": {"objective": float(value), "bce": float(bce), "regularizer": float(reg)},
           "lambda": LAM}
    torch.save({"A": Anew, "stats": stats}, a.out / "SHARED_AFFINE_REFINED.pt")
    (a.out / "REFINEMENT.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fit", type=Path, required=True); p.add_argument("--affine", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--chunk-size", type=int, default=128); p.add_argument("--threads", type=int, default=8)
    main(p.parse_args())
