"""Bounded, zero-training audit of OGPS target-error propagation.

Inputs are one candidate's FP64 A[64,32], G[32,32], and e[64], where
    G = P_full_support.T @ P_full_support / M
is the MEAN-normalized prototype Gram over the caller's frozen full-image
support.  If M=0, pass G=0, preserve the candidate, and record M=0 outside
this module.  We cannot infer M from G; an all-zero G need not imply M=0.
e = clipped_target_logit_pred - clipped_target_logit_GT, NOT probability
error.  This module never receives IoU, labels, probabilities, or images.

With K=(A.T A/64 + .003 I)^-1 A.T/64 and Q=K.T G K, e.T Q e is the full-
support mean squared logit error relative to the GT-soft ridge solution.
It is not an IoU loss and the GT-soft solution is not a segmentation upper
bound.  The pooled response operator A K is a contraction; the full-support
response has a different geometry.

Exactly eight cells are replaced by their GT targets in every repair arm:
H8 greedy exact output-energy reduction; E8 largest absolute input error;
R8 uniform random eight; EM8 closest removed INPUT energy to H8 among a
fixed pool of 4096 random eight-subsets.  EM8 selection reads only e and
H8's scalar removed input energy, never Q, G, or IoU.  Its pool is random,
so energy matching may fail; no retry or enlarged search is permitted.

Twenty signed-value-preserving error permutations are mathematical
perturbations, not claims about realizable clipped probability predictions.
No additional clipping is applied (that would destroy equal-error matching).
No calculations run on import.  The self-check entry point is intended for
the authorized laptop/server, not execution on the desktop.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor


RIDGE_LAMBDA = 0.003
CELL_COUNT = 64
COEFFICIENT_DIM = 32
REPAIR_COUNT = 8
MATCH_POOL_SIZE = 4096
PERMUTATION_COUNT = 20
BASE_SEED = 20261002
MATCH_RELATIVE_TOLERANCE = 0.05


def _seed(candidate_key: Any, purpose: str) -> int:
    """Stable across processes; never use Python's randomized hash()."""
    payload = json.dumps([BASE_SEED, candidate_key, purpose], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8],
                          "little") % (2**63 - 1)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _finite(name: str, x: Tensor) -> None:
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} has non-finite values")


def _validate(A: Tensor, G: Tensor, e: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    if A.shape != (CELL_COUNT, COEFFICIENT_DIM) or G.shape != (COEFFICIENT_DIM, COEFFICIENT_DIM) or e.shape != (CELL_COUNT,):
        raise ValueError("expected A[64,32], G[32,32], e[64]")
    if any(x.dtype != torch.float64 for x in (A, G, e)):
        raise TypeError("A, G and e must already be FP64")
    if A.device != G.device or A.device != e.device:
        raise ValueError("A, G and e must be on the same device")
    for name, x in (("A", A), ("G", G), ("e", e)):
        _finite(name, x)
    A, G, e = A.detach(), G.detach(), e.detach()
    scale = max(1.0, float(G.abs().max()))
    asymmetry = float((G - G.T).abs().max())
    if asymmetry > 1e-10 * scale:
        raise ValueError("G is not symmetric within the predeclared FP64 tolerance")
    G = (G + G.T) * 0.5
    smallest = float(torch.linalg.eigvalsh(G)[0])
    if smallest < -1e-10 * scale:
        raise ValueError("G is not positive semidefinite within the FP64 tolerance")
    return A, G, e, {"G_asymmetry_max_abs": asymmetry,
                      "G_min_eigenvalue": smallest,
                      "G_all_zero": bool(torch.count_nonzero(G) == 0)}


def _nonnegative_energy(x: Tensor, Q: Tensor) -> Tensor:
    """Allow only small negative roundoff in a theoretically PSD quadratic."""
    value = torch.einsum("...i,ij,...j->...", x, Q, x)
    scale = Q.abs().max() * x.abs().sum(-1).square()
    tolerance = 1e-10 * scale.clamp_min(1.0)
    if bool((value < -tolerance).any()):
        raise RuntimeError("negative propagated energy exceeds FP64 tolerance")
    _finite("propagated energy", value)
    return value.clamp_min(0)


def _arm(e: Tensor, Q: Tensor, indices: Tensor) -> dict[str, Any]:
    if indices.shape != (REPAIR_COUNT,) or len(torch.unique(indices)) != REPAIR_COUNT:
        raise RuntimeError("repair arm must select exactly eight unique cells")
    corrected = e.clone()
    corrected[indices] = 0.0  # Direct GT replacement, not partial interpolation.
    energy = float(_nonnegative_energy(corrected, Q))
    removed = float(e[indices].square().sum())
    return {"e_new": corrected,
            "selected_idx": indices.detach().cpu().tolist(),
            "full_output_energy": energy,
            "full_output_rms": math.sqrt(energy),
            "removed_input_energy": removed,
            "remaining_input_energy": float(corrected.square().sum()),
            "repair_count": REPAIR_COUNT}


@torch.no_grad()
def analyze_error_geometry(A: Tensor, G: Tensor, e: Tensor,
                           candidate_key: Any) -> dict[str, Any]:
    """Return four repair arms plus 20 permutations for one frozen candidate.

    candidate_key is a JSON-serializable permanent identity (e.g. a list of
    split/image_id/annotation_id/branch/raw_id), never its order in a batch.
    e_new tensors remain on the input device; all statistics are JSON-safe.
    There are deliberately no hyperparameter or random-budget overrides.
    """
    A, G, e, audit = _validate(A, G, e)
    system = A.T @ A / CELL_COUNT + RIDGE_LAMBDA * torch.eye(
        COEFFICIENT_DIM, dtype=torch.float64, device=A.device)
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info) != 0:
        raise RuntimeError("fixed-ridge FP64 Cholesky failed; no jitter/retry is allowed")
    K = torch.cholesky_solve(A.T / CELL_COUNT, factor)
    Q = K.T @ G @ K
    Q = (Q + Q.T) * 0.5
    _finite("Q", Q)
    ar_norm = float(torch.linalg.matrix_norm(A @ K, ord=2))
    if ar_norm > 1.0 + 1e-10:
        raise RuntimeError("A K violated its theoretical contraction bound")
    original_energy = float(_nonnegative_energy(e, Q))
    input_energy = float(e.square().sum())

    # Greedy selection recalculates the exact marginal gain after EACH zero.
    working = e.clone()
    unavailable = torch.zeros(CELL_COUNT, dtype=torch.bool, device=e.device)
    h_indices, greedy_trace = [], []
    for step in range(REPAIR_COUNT):
        gains = 2 * working * (Q @ working) - working.square() * Q.diagonal()
        gains[unavailable] = -torch.inf
        j = int(torch.argmax(gains))  # First/lower cell index wins an exact tie.
        before = float(_nonnegative_energy(working, Q))
        predicted_drop = float(gains[j])
        working[j] = 0.0
        after = float(_nonnegative_energy(working, Q))
        greedy_trace.append({"step": step + 1, "index": j,
                             "predicted_energy_drop": predicted_drop,
                             "actual_energy_drop": before - after})
        unavailable[j] = True
        h_indices.append(j)
    H = torch.tensor(h_indices, dtype=torch.long, device=e.device)
    # Stable sort resolves equal magnitudes by the original grid-cell index.
    E = torch.argsort(e.abs(), descending=True, stable=True)[:REPAIR_COUNT]
    seeds = {purpose: _seed(candidate_key, purpose)
             for purpose in ("R8", "EM8_pool", "permutations")}
    R = torch.randperm(CELL_COUNT, generator=_generator(seeds["R8"]))[:REPAIR_COUNT].to(e.device)

    pool_generator = _generator(seeds["EM8_pool"])
    pool_cpu = torch.stack([torch.randperm(CELL_COUNT, generator=pool_generator)[:REPAIR_COUNT]
                            for _ in range(MATCH_POOL_SIZE)])
    pool = pool_cpu.to(e.device)
    pool_input_energy = e[pool].square().sum(-1)
    desired_input_energy = e[H].square().sum()
    # The only matching target from H8 is this SCALAR input energy.  No Q/G.
    pool_row = int(torch.argmin((pool_input_energy - desired_input_energy).abs()))
    EM = pool[pool_row]
    target_energy = float(desired_input_energy)
    matched_energy = float(pool_input_energy[pool_row])
    absolute_mismatch = abs(matched_energy - target_energy)
    if target_energy > 0:
        relative_mismatch = absolute_mismatch / target_energy
    elif absolute_mismatch == 0:
        relative_mismatch = 0.0
    else:
        relative_mismatch = None  # Undefined ratio; JSON-safe, explicitly unmatched.
    matching = {"pool_size": MATCH_POOL_SIZE, "selected_pool_row": pool_row,
                "target_removed_input_energy": target_energy,
                "matched_removed_input_energy": matched_energy,
                "absolute_mismatch": absolute_mismatch,
                "relative_mismatch": relative_mismatch,
                "relative_tolerance": MATCH_RELATIVE_TOLERANCE,
                "within_5pct": (relative_mismatch is not None
                                and relative_mismatch <= MATCH_RELATIVE_TOLERANCE),
                "selection_uses_Q_or_G_or_IoU": False,
                "target_input_energy_zero": target_energy == 0.0}
    arms = {name: _arm(e, Q, idx) for name, idx in
            (("H8", H), ("E8", E), ("R8", R), ("EM8", EM))}
    for arm in arms.values():
        arm["full_output_energy_reduction"] = original_energy - arm["full_output_energy"]
    arms["H8"]["greedy_trace"] = greedy_trace
    arms["EM8"]["matching"] = matching

    perm_generator = _generator(seeds["permutations"])
    permutations = torch.stack([torch.randperm(CELL_COUNT, generator=perm_generator)
                                 for _ in range(PERMUTATION_COUNT)]).to(e.device)
    permuted_e = e[permutations]
    perm_energy = _nonnegative_energy(permuted_e, Q)
    perm_rms = perm_energy.sqrt()
    original_rms = math.sqrt(original_energy)
    mean_random_rms = float(perm_rms.mean())
    random_rms = perm_rms.cpu().tolist()
    per_ratio = [original_rms / value if value > 0 else None for value in random_rms]
    return {
        "K": K,
        "candidate_key": candidate_key,
        "lambda": RIDGE_LAMBDA, "G_normalization": "P_full_support.T_P_full_support_div_M",
        "original_input_energy": input_energy,
        "original_input_rms": math.sqrt(input_energy / CELL_COUNT),
        "original_full_energy": original_energy,
        "original_full_rms": original_rms,
        "zero_input_error": input_energy == 0.0,
        "AR_spectral_norm": ar_norm,
        "arms": arms, "energy_matching": matching,
        "permutation": {
            "count": PERMUTATION_COUNT,
            "full_energy": perm_energy.cpu().tolist(),
            "full_rms": random_rms,
            "mean_full_rms": mean_random_rms,
            "original_over_mean_random_rms": original_rms / mean_random_rms if mean_random_rms > 0 else None,
            "original_over_each_random_rms": per_ratio,
            "ratio_defined": mean_random_rms > 0,
            "input_energy_max_abs_difference": float((permuted_e.square().sum(-1) - input_energy).abs().max()),
            "indices": permutations.cpu().tolist(),
        },
        "seeds": seeds, "numerical_audit": audit,
    }


def math_self_checks(device: str = "cpu") -> dict[str, Any]:
    """Explicit laptop/server self-check; no data loading or training."""
    gen = _generator(BASE_SEED)
    A = torch.randn(CELL_COUNT, COEFFICIENT_DIM, generator=gen, dtype=torch.float64).to(device)
    P = torch.randn(97, COEFFICIENT_DIM, generator=gen, dtype=torch.float64).to(device)
    e = torch.randn(CELL_COUNT, generator=gen, dtype=torch.float64).to(device)
    G = P.T @ P / len(P)
    result = analyze_error_geometry(A, G, e, ["self_check", 17, 9, "one2one", 800])
    K = torch.linalg.solve(A.T @ A / CELL_COUNT + RIDGE_LAMBDA * torch.eye(
        COEFFICIENT_DIM, dtype=torch.float64, device=A.device), A.T / CELL_COUNT)
    direct = float((P @ K @ e).square().mean())
    assert abs(result["original_full_energy"] - direct) < 1e-10
    for arm in result["arms"].values():
        idx = arm["selected_idx"]
        assert len(idx) == len(set(idx)) == REPAIR_COUNT
        torch.testing.assert_close(arm["e_new"][idx], torch.zeros(REPAIR_COUNT, dtype=torch.float64, device=device))
        keep = torch.ones(CELL_COUNT, dtype=torch.bool, device=device)
        keep[idx] = False
        torch.testing.assert_close(arm["e_new"][keep], e[keep], atol=0, rtol=0)
        energy = float((P @ K @ arm["e_new"]).square().mean())
        assert abs(energy - arm["full_output_energy"]) < 1e-10
    # Verify every greedy step was globally best among its remaining cells.
    current = e.clone()
    available = set(range(CELL_COUNT))
    Q = K.T @ G @ K
    for chosen in result["arms"]["H8"]["selected_idx"]:
        before = float(current @ Q @ current)
        alternatives = []
        for j in sorted(available):
            proposed = current.clone()
            proposed[j] = 0
            alternatives.append((before - float(proposed @ Q @ proposed), j))
        selected_gain = next(gain for gain, j in alternatives if j == chosen)
        assert selected_gain >= max(gain for gain, _ in alternatives) - 1e-10
        current[chosen] = 0
        available.remove(chosen)
    perm_indices = torch.tensor(result["permutation"]["indices"], device=device)
    torch.testing.assert_close(torch.sort(e[perm_indices], dim=-1).values,
                               torch.sort(e).values.expand(PERMUTATION_COUNT, -1), atol=0, rtol=0)
    # Reconstruct the fixed 4096-subset pool to check input-energy matching.
    pool_gen = _generator(result["seeds"]["EM8_pool"])
    pool = torch.stack([torch.randperm(CELL_COUNT, generator=pool_gen)[:REPAIR_COUNT]
                        for _ in range(MATCH_POOL_SIZE)]).to(device)
    differences = (e[pool].square().sum(-1) - result["energy_matching"]["target_removed_input_energy"]).abs()
    assert result["energy_matching"]["selected_pool_row"] == int(torch.argmin(differences))
    zero = analyze_error_geometry(A, G, torch.zeros_like(e), ["zero_error"])
    assert zero["original_full_energy"] == 0 and zero["zero_input_error"]
    assert zero["energy_matching"]["within_5pct"]
    assert zero["permutation"]["original_over_mean_random_rms"] is None
    zero_G = analyze_error_geometry(A, torch.zeros_like(G), e, ["zero_support_or_zero_proto"])
    assert zero_G["original_full_energy"] == 0
    assert all(arm["full_output_energy"] == 0 for arm in zero_G["arms"].values())
    assert _seed([1, 2, "x"], "R8") == _seed([1, 2, "x"], "R8")
    assert _seed([1, 2, "x"], "R8") != _seed([1, 2, "x"], "EM8_pool")
    return {"passed": True, "device": str(device),
            "energy_direct_error": abs(result["original_full_energy"] - direct),
            "AR_spectral_norm": result["AR_spectral_norm"],
            "greedy_steps_checked": REPAIR_COUNT,
            "energy_match_pool_checked": MATCH_POOL_SIZE,
            "zero_error_and_zero_G_checked": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(math_self_checks(args.device), indent=2))
    else:
        parser.print_help()
