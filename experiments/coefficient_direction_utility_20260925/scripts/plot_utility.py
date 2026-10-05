"""Publication-style static figure from the audited 7E.1 summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RHO = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


def main(args):
    data = json.loads(args.summary.read_text())["groups"]["all"]
    metrics = (("image_iou_delta", "Mean mask IoU change"),
               ("auc_delta", "Mean in-box AUC change"),
               ("mask75_net", "Net Mask75 fraction change"))
    colors = {"h_only": "#4a6fa5", "true_local": "#c25242"}
    labels = {"h_only": r"$h$ only", "true_local": r"$h+P_{local}$"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        ax.axhline(0, color="#555555", linewidth=.8, linestyle=":")
        for arm in ("h_only", "true_local"):
            arm_data = data[arm]
            y = np.array([arm_data[f"rho_{rho:g}"]["metrics"][metric]["mean"] for rho in RHO])
            ci = np.array([arm_data[f"rho_{rho:g}"]["metrics"][metric]["ci95"] for rho in RHO])
            ax.plot(RHO, y, marker="o", markersize=3.5, linewidth=1.6,
                    color=colors[arm], linestyle="-" if arm == "true_local" else "--",
                    label=labels[arm] + " controlled")
            if arm == "true_local":
                ax.fill_between(RHO, ci[:, 0], ci[:, 1], color=colors[arm], alpha=.15, linewidth=0)
            raw = arm_data["raw"]
            ax.scatter(raw["metrics"]["effect_cos"]["mean"], raw["metrics"][metric]["mean"],
                       marker="X", s=55, color=colors[arm], edgecolor="white", linewidth=.5,
                       zorder=5, label=labels[arm] + " actual")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Functional direction cosine to oracle")
        ax.grid(alpha=.18)
        ax.set_xlim(-.02, 1.02)
    axes[0].set_ylabel("Change from original candidate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, -.04))
    fig.suptitle("GT-assisted controlled directions and held-out predictor outputs", fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, .94))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "DIRECTION_UTILITY.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.out / "DIRECTION_UTILITY.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
