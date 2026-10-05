"""Publication-style risk/coverage figure for locked 7L results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(a):
    result = json.loads(a.results.read_text())
    keys = ("X1_huber", "X2_huber", "X3_huber", "X4_huber")
    colors = {"X1_huber":"#8595a6", "X2_huber":"#348c9c",
              "X3_huber":"#ce672e", "X4_huber":"#7c61a3"}
    labels = {"X1_huber":"Box / class / score", "X2_huber":"+ h and raw detection",
              "X3_huber":"+ correction statistics", "X4_huber":"+ local features"}
    coverage = np.array([.01,.03,.05,.10,.20,.50,1.])
    baseline = result["all_on_mean_benefit"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=180,
                             gridspec_kw={"wspace":.29})
    for key in keys:
        data = result["models"][key]["risk_coverage"]
        conditional = np.array([data[str(q)]["mean_benefit"] for q in coverage])
        axes[0].plot(coverage*100, conditional*100, marker="o", ms=3.4,
                     lw=2.2 if key=="X3_huber" else 1.35,
                     color=colors[key], label=labels[key])
        axes[1].plot(coverage*100, coverage*conditional*100,
                     marker="o", ms=3.4, lw=2.2 if key=="X3_huber" else 1.35,
                     color=colors[key])
    for ax in axes:
        ax.axvline(10, color="#687481", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(.9, 105)
        ax.set_xticks([1,3,10,20,50,100])
        ax.set_xticklabels(["1","3","10","20","50","100"])
        ax.set_xlabel("Candidates corrected (%)")
        ax.grid(alpha=.17, lw=.6)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Mean mask IoU gain among corrected (pp)")
    axes[1].set_ylabel("Mean mask IoU gain over all candidates (pp)")
    axes[1].axhline(baseline*100, color="#333333", ls="--", lw=1.4,
                    label="Correct all candidates")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    selected = result["models"]["X3_huber"]
    axes[0].scatter([10],[selected["top10"]["mean_benefit"]*100],
                    color=colors["X3_huber"], s=36, zorder=5)
    ci = selected["top10_cluster_ci"]["mean_benefit_95ci"]
    axes[0].errorbar([10],[selected["top10"]["mean_benefit"]*100],
        yerr=[[selected["top10"]["mean_benefit"]*100-ci[0]*100],
              [ci[1]*100-selected["top10"]["mean_benefit"]*100]],
        fmt="none", color=colors["X3_huber"], capsize=3, lw=1.2)
    fig.suptitle("7L: Predictable benefit does not yet imply a better gate",
                 fontsize=11.5, y=1.01)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out/"FIGURE_7L_RISK_COVERAGE.png", bbox_inches="tight")
    fig.savefig(a.out/"FIGURE_7L_RISK_COVERAGE.pdf", bbox_inches="tight")
    plt.close(fig)
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(figure="FIGURE_7L_RISK_COVERAGE.png",
        source=str(a.results), axes="conditional and population mean mask IoU gain"),indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
