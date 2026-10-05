"""Recompute paired seed results without losing bootstrap multiplicities.

Training seeds are paired, not independent GT observations. CIs resample image
clusters and average over the three fixed seeds; they do not quantify all
possible training-seed uncertainty or correct adaptive experiment selection.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "S063": ("density_weighted_readout_pilot_20260913", "density_raw"),
    "S065": ("uncertainty_weighted_readout_pilot_20260913", "uncertainty_raw"),
    "S066": ("pixel_uncertainty_readout_pilot_20260913", "pixel_uncertainty"),
}


def truth(x):
    assert x.notna().all()
    assert set(x.astype(str).str.lower()) <= {"true", "false", "1", "0"}
    return x.astype(str).str.lower().isin(["true", "1"])


def cluster_estimate(numerator, denominator, draws):
    # Repeated indices are counted repeatedly: image 0 sampled twice carries
    # twice its original contribution. Filtering with isin() is NOT valid.
    den = denominator[draws].sum(1)
    num = numerator[draws].sum(1)
    return 100 * num / np.where(den > 0, den, np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    selection = json.loads((ROOT / "diagnostics/readout_input_scale1200_20260912/cache/selection.json").read_text())
    image_ids = np.array(selection["transfer"], int)
    assert len(image_ids) == len(set(image_ids)) == 300
    image_pos = {int(x): i for i, x in enumerate(image_ids)}
    draws = np.random.default_rng(20260913).integers(300, size=(2000, 300))
    # Minimal semantic witness: the duplicated cluster changes the statistic.
    assert np.isclose(cluster_estimate(np.array([1., -1.]), np.array([1., 1.]), np.array([[0, 0, 1]]))[0], 100/3)
    rows, means, details, provenance = [], [], {}, {}
    for experiment, (folder, mode) in CONFIGS.items():
        run = ROOT / "diagnostics" / folder
        assert (run / "COMPLETE.json").exists()
        task = pd.read_csv(run / "task_summary.csv").set_index("arm")
        gt = pd.read_csv(run / "gt_recovery.csv")
        gt["hit75"] = truth(gt.hit75).astype(float)
        seed_arrays = []
        checkpoints = list(run.glob("*/checkpoints/epoch*.pt"))
        assert len(checkpoints) == 90
        provenance[experiment] = {"actual_checkpoints": len(checkpoints),
            "receipt_claims_checkpoints": json.loads((run / "COMPLETE.json").read_text()).get("checkpoints"),
            "raw_receipts_preserved": True,
            "correct_experiment": experiment, "modes": ["raw_coco", mode],
            "intervention_is_loss_weighting_not_label_switch": True}
        for seed in range(3):
            control, method = f"raw_coco_s{seed}_d0", f"{mode}_s{seed}_d0"
            b = gt[gt.arm == control].set_index("annotation_id").sort_index()
            m = gt[gt.arm == method].set_index("annotation_id").sort_index()
            assert len(b) == len(m) == 2002 and b.index.equals(m.index)
            assert b[["image_id", "ici", "category_id", "area"]].equals(m[["image_id", "ici", "category_id", "area"]])
            diff = m.hit75 - b.hit75
            high = b.ici > .5 + 1e-10
            nums, dens = {}, {}
            for group, select in {"all": np.ones(len(b), bool), "high": high, "nonhigh": ~high}.items():
                numer, denom = np.zeros(300), np.zeros(300)
                idx = np.array([image_pos[int(x)] for x in b.loc[select, "image_id"]])
                np.add.at(numer, idx, diff[select].to_numpy())
                np.add.at(denom, idx, 1)
                nums[group], dens[group] = numer, denom
            seed_arrays.append((nums, dens))
            row = dict(experiment=experiment, seed=seed,
                raw_ap=100*task.loc[control, "mask_ap"], method_ap=100*task.loc[method, "mask_ap"],
                delta_ap=100*(task.loc[method, "mask_ap"]-task.loc[control, "mask_ap"]),
                delta_ap75=100*(task.loc[method, "mask_ap75"]-task.loc[control, "mask_ap75"]),
                recovered_high=int(((m.hit75 == 1) & (b.hit75 == 0) & high).sum()),
                lost_high=int(((m.hit75 == 0) & (b.hit75 == 1) & high).sum()))
            for group in nums:
                boot = cluster_estimate(nums[group], dens[group], draws)
                row[f"delta_r75_{group}"] = 100*nums[group].sum()/dens[group].sum()
                row[f"ci_low_{group}"], row[f"ci_high_{group}"] = np.nanquantile(boot, [.025,.975])
            row["delta_gap_nonhigh_high"] = row["delta_r75_nonhigh"]-row["delta_r75_high"]
            rows.append(row)
        pooled = {}
        boots = {}
        for group in ["all", "high", "nonhigh"]:
            numer = np.mean([x[0][group] for x in seed_arrays], axis=0)
            denom = seed_arrays[0][1][group]
            boots[group] = cluster_estimate(numer, denom, draws)
            pooled[group] = {"delta_points": float(100*numer.sum()/denom.sum()),
                             "ci95_points": np.nanquantile(boots[group], [.025,.975]).tolist()}
        pooled["gap_nonhigh_high"] = {"delta_points": pooled["nonhigh"]["delta_points"]-pooled["high"]["delta_points"],
            "ci95_points": np.nanquantile(boots["nonhigh"]-boots["high"], [.025,.975]).tolist()}
        details[experiment] = pooled
        sub = pd.DataFrame([r for r in rows if r["experiment"] == experiment])
        means.append(dict(experiment=experiment, mean_raw_ap=sub.raw_ap.mean(), mean_method_ap=sub.method_ap.mean(),
            std_method_ap=sub.method_ap.std(ddof=1), mean_delta_ap=sub.delta_ap.mean(), std_delta_ap=sub.delta_ap.std(ddof=1),
            mean_delta_ap75=sub.delta_ap75.mean(), mean_delta_high=sub.delta_r75_high.mean(),
            mean_delta_nonhigh=sub.delta_r75_nonhigh.mean(), mean_delta_gap=sub.delta_gap_nonhigh_high.mean()))
    pd.DataFrame(rows).to_csv(a.out / "paired_seed_results.csv", index=False)
    pd.DataFrame(means).to_csv(a.out / "seed_means.csv", index=False)
    (a.out / "paired_image_ci.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (a.out / "provenance_corrections.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (a.out / "README.md").write_text(
        "# Recent pilot statistical correction\n\nS063 and S065 earlier isin-based bootstrap intervals are withdrawn. The paired image bootstrap here preserves resampling multiplicities and all 300 image IDs. Point estimates use the full sample, not the mean of bootstrap draws. Three fixed training seeds are paired; pooled CIs average their responses within each resampled image, not pool GT as independent replicates. CIs do not include multiple-testing correction or unseen training seed variance. All runs reused an extensively explored transfer set from train2017. Historical r75_low means ICI nonhigh (<=.5), not genuinely low density. No AP confidence intervals computed. Checkpoint receipts inherited 135; actual count is 90. Do not overwrite historical receipts.\n\nLoss caveats: S063 reweights BCE only, with unnormalized 1+3E4, so global BCE/Dice scale changes. S065 normalizes an instance BCE weight; samples are selected in GT-box support. S066 also applies factor to Dice, unlike raw, and changes mean BCE scale. Thus none identifies uncertainty or density as a causal mechanism in isolation.\n",
        encoding="utf-8")
    shutil.copy2(__file__, a.out / Path(__file__).name)
    print(pd.DataFrame(means).round(4).to_string(index=False))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
