"""Render the frozen failure-gate risk–coverage and repair–damage curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(a):
    data = json.loads((a.results / "RESULTS.json").read_text())
    gate = data["gates"]["h_raw_detection_failure"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    styles = (("7JN_h_only", "h-only correction", "#1b6ca8"),
              ("7JN_true_local", "neck-ROI correction", "#d65b27"))
    for arm, label, color in styles:
        points = [gate[str(rate)]["arms"][arm] for rate in (1, 3, 5, 10)]
        coverage = np.array([p["coverage"] * 100 for p in points])
        net = np.array([p["mask75_delta"] * 100 for p in points])
        lower = net - np.array([p["mask75_delta_ci95"][0] * 100 for p in points])
        upper = np.array([p["mask75_delta_ci95"][1] * 100 for p in points]) - net
        axes[0].errorbar(coverage, net, yerr=[lower, upper], fmt="o-", color=color,
                         capsize=2.5, label=label)
        all_on = data["all_on"][arm]
        axes[0].scatter(100, all_on["mask75_delta"] * 100, marker="s", color=color, s=45)
        for rate, x, y in zip((1, 3, 5, 10), coverage, net):
            axes[0].annotate(f"{rate}%", (x, y), xytext=(3, 4), textcoords="offset points",
                             color=color, fontsize=8)
        axes[1].plot([p["damages"] for p in points], [p["repairs"] for p in points],
                     "o-", color=color, label=label)
        axes[1].scatter(all_on["damages"], all_on["repairs"], marker="s", color=color, s=45)
        for rate, p in zip((1, 3, 5, 10), points):
            axes[1].annotate(f"{rate}%", (p["damages"], p["repairs"]),
                             xytext=(3, 4), textcoords="offset points", color=color, fontsize=8)
    axes[0].axhline(0, color="0.35", linewidth=.8)
    axes[0].set(xlabel="Candidates corrected (%)", ylabel="Net Mask75 change (pp)",
                title="Risk–coverage on fixed val candidates")
    axes[0].set_xlim(0, 106)
    axes[0].grid(alpha=.18)
    axes[1].plot([0, 25], [0, 25], ":", color="0.5", linewidth=1, label="repairs = damages")
    axes[1].set(xlabel="Originally successful candidates damaged", ylabel="Original failures repaired",
                title="Repair–damage trade-off")
    axes[1].grid(alpha=.18)
    axes[1].set_xlim(-.5, 25)
    axes[1].set_ylim(-.5, 28)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.text(.5, -.015, "Circles: dev target good-instance FPR; squares: correction applied to all.",
             ha="center", va="top", fontsize=8)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "RISK_COVERAGE.png", dpi=240, bbox_inches="tight")
    fig.savefig(a.out / "RISK_COVERAGE.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
