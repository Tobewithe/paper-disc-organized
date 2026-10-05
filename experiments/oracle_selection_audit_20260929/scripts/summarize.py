import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")

def boot_indices(image_ids, seed=72929, reps=2000):
    ids, inv = np.unique(image_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    return ids, inv, np.stack([np.bincount(rng.integers(len(ids), size=len(ids), dtype=np.int64), minlength=len(ids)) for _ in range(reps)])

def summarize(values, image_ids, names, arms, mask, weights):
    ids, inv, W = weights
    x = values[mask]
    invx = inv[mask]
    # Candidate-level means and image-cluster bootstrap means.
    sums = np.zeros((len(ids), x.shape[1], x.shape[2]), dtype=np.float64)
    counts = np.zeros_like(sums)
    for j, image_index in enumerate(invx):
        finite = np.isfinite(x[j])
        sums[image_index][finite] += x[j][finite]
        counts[image_index] += finite
    wb = torch.from_numpy(W).double()
    bs = (wb @ torch.from_numpy(sums.reshape(len(ids), -1))).numpy().reshape(len(W), x.shape[1], x.shape[2])
    bc = (wb @ torch.from_numpy(counts.reshape(len(ids), -1))).numpy().reshape(len(W), x.shape[1], x.shape[2])
    means = np.divide(bs, bc, out=np.full_like(bs, np.nan), where=bc > 0)
    out = {"instances": int(mask.sum()), "images": int(len(np.unique(image_ids[mask]))), "arms": {}}
    for ai, arm in enumerate(arms):
        st = {}
        for mi, name in enumerate(names):
            z = means[:, ai, mi]
            z = z[np.isfinite(z)]
            st[name] = {"mean": float(np.nanmean(x[:, ai, mi])), "ci95": np.nanquantile(z, [0.025, 0.975]).tolist(), "n_valid": int(np.isfinite(x[:, ai, mi]).sum())}
        # U* hit is a ratio of counts, not a mean of per-instance ratios.
        numer = sums[:, ai, names.index("u_hit")]
        denom = sums[:, ai, names.index("u_star")]
        ratio = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
        st["u_hit_ratio_count_weighted"] = {"mean": float(np.nansum(x[:, ai, names.index("u_hit")]) / max(np.nansum(x[:, ai, names.index("u_star")]), 1)), "ci95": np.nanquantile(ratio[np.isfinite(ratio)], [0.025, 0.975]).tolist(), "u_star_total": int(np.nansum(x[:, ai, names.index("u_star")]))}
        st["selector_nonnegative_fraction"] = {"mean": float(np.mean(x[:, ai, names.index("delta_select")] >= -1e-8)), "ci95": np.nanquantile((means[:, ai, names.index("delta_select")] >= -1e-8).astype(float), [0.025, 0.975]).tolist()}
        out["arms"][arm] = st
    return out

def main(a):
    d = np.load(a.source / "SELECTION.npz")
    values = d["values"].astype(np.float64)
    image_ids = d["image_ids"]
    names = d["names"].tolist(); arms = d["arms"].tolist()
    primary = d["primary"].astype(bool); size = d["size"]
    weights = boot_indices(image_ids)
    groups = {"all": np.ones(len(values), bool), "good_box_failure": primary,
              "mask75_success": values[:, 0, names.index("iou0")] >= .75,
              "small": size < 32**2, "medium": (size >= 32**2) & (size < 96**2), "large": size >= 96**2}
    results = {k: summarize(values, image_ids, names, arms, m, weights) for k, m in groups.items()}
    report = ["# Perfect error-location selection audit", "", "This is a GT-assisted upper-bound diagnostic using frozen 7Q p-spatial outputs; it is not a deployable method or COCO AP.", "", "## Main results", ""]
    for group in ("all", "good_box_failure", "mask75_success"):
        report += [f"### {group} (n={results[group]['instances']:,})", "", "| arm | M1−M0 IoU pp | perfect-selector−M0 pp | oracle−M0 pp | U* hit rate | selector nonnegative |", "|---|---:|---:|---:|---:|---:|"]
        for arm in arms:
            st = results[group]["arms"][arm]
            f = lambda key, scale=100: f"{st[key]['mean']*scale:.3f} [{st[key]['ci95'][0]*scale:.3f}, {st[key]['ci95'][1]*scale:.3f}]"
            report.append(f"| {arm} | {f('delta1')} | {f('delta_select')} | {f('delta_oracle')} | {st['u_hit_ratio_count_weighted']['mean']*100:.2f}% | {st['selector_nonnegative_fraction']['mean']*100:.1f}% |")
        report += [""]
    report += ["## Pixel transition accounting on good-box failures", "", "`a`=FN→TP, `b`=FP→TN, `c`=TP→FN, `d`=TN→FP. A wrong binary pixel cannot switch from FN to FP with a fixed GT; it either remains wrong or becomes correct.", "", "| arm | a | b | c | d | FN remaining | FP remaining | U* pixels | U* hits |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        st = results["good_box_failure"]["arms"][arm]
        report.append(f"| {arm} | {st['a_fn_tp']['mean']:.1f} | {st['b_fp_tn']['mean']:.1f} | {st['c_tp_fn']['mean']:.1f} | {st['d_tn_fp']['mean']:.1f} | {st['fn_remain']['mean']:.1f} | {st['fp_remain']['mean']:.1f} | {st['u_hit_ratio_count_weighted']['u_star_total']:,} | {st['u_hit_ratio_count_weighted']['mean']*st['u_hit_ratio_count_weighted']['u_star_total']:.0f} |")
    report += ["", "## Size strata (exploratory)", "", "These strata are descriptive and were selected after the 7R audit; they do not establish a new independent test.", "", "| group | arm | selector−M0 pp | U* hit rate |", "|---|---|---:|---:|"]
    for group in ("small", "medium", "large"):
        for arm in arms:
            st = results[group]["arms"][arm]
            report.append(f"| {group} | {arm} | {st['delta_select']['mean']*100:.3f} | {st['u_hit_ratio_count_weighted']['mean']*100:.2f}% |")
    report += ["", "## Decision", "", "A large selector gain would show that existing binary corrections contain useful changes masked by damage, and would justify studying a GT-free expected-benefit gate. A small selector gain would rule out selection as the main fix for the frozen correction. U* hit rate measures how much of the finite-oracle opportunity the current correction actually reaches; it does not measure whether a future gate is predictable without GT.", "", "The selector uses GT only to choose between M0 and M1 and is therefore an upper bound. The current test set was already used by 7Q/7R, so intervals are exploratory image-cluster bootstrap intervals."]
    (a.out / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    write(a.out / "RESULTS.json", {"source": str(a.source), "groups": results, "names": names, "arms": arms, "recorded_at": datetime.now(timezone.utc).isoformat()})
    write(a.out / "COMPLETE.json", {"status": "completed", "source": str(a.source), "instances": len(values), "groups": list(groups)})
    print(json.dumps({"status": "completed", "instances": len(values), "groups": list(groups)}), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--out", type=Path, required=True); main(p.parse_args())
