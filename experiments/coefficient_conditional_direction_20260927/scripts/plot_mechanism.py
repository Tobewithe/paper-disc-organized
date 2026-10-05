"""Publication-oriented static figure for the 7N diagnostic results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    oof = json.loads(args.oof.read_text(encoding="utf-8"))
    gradient = json.loads(args.gradient.read_text(encoding="utf-8"))
    order = ["success_good_box", "failure_good_box", "severe_failure"]
    labels = ["Good box, mask success", "Good box, mask failure", "Severe mask failure"]
    colors = ["#26749A", "#D56B35", "#9F383F"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.17]})
    y = np.arange(3)
    values = np.asarray([retrieval[k]["paired"]["h_top5_minus_random5"]["mean"] for k in order])
    intervals = np.asarray([retrieval[k]["paired"]["h_top5_minus_random5"]["ci95"] for k in order])
    axes[0].barh(y, values, color=colors, height=.58)
    axes[0].errorbar(values, y, xerr=np.vstack((values-intervals[:, 0], intervals[:, 1]-values)),
                     fmt="none", color="black", capsize=3, lw=1)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, .15)
    axes[0].set_xlabel("Direction agreement: h-neighbor − matched random")
    axes[0].set_title("A. Cross-image retrieval", loc="left", fontweight="bold")
    axes[0].axvline(0, color="#777", lw=.8)

    effect = np.asarray([oof[k]["effect_cos_mean"] for k in order])
    axes[1].barh(y, effect, color=colors, height=.58)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, .56)
    axes[1].set_xlabel("OOF predictor vs oracle effect cosine")
    axes[1].set_title("B. Learned correction", loc="left", fontweight="bold")
    for yy, xx in zip(y, effect):
        axes[1].text(xx+.012, yy, f"{xx:.3f}", va="center", fontsize=9)

    groups = ["discordant", "aligned"]
    x = np.arange(2)
    width = .23
    for offset, metric, label, color in [(-width, "cos_s", "Model response s", "#6288A6"),
                                         (0, "cos_t", "GT moment t", "#67A081"),
                                         (width, "cos_g", "Residual gradient g", "#C45A49")]:
        means = np.asarray([gradient[k][metric]["mean"] for k in groups])
        bounds = np.asarray([gradient[k][metric]["ci95"] for k in groups])
        axes[2].bar(x+offset, means, width=width, color=color, label=label)
        axes[2].errorbar(x+offset, means,
                         yerr=np.vstack((means-bounds[:, 0], bounds[:, 1]-means)),
                         fmt="none", color="black", capsize=2, lw=.8)
    axes[2].set_xticks(x, ["Opposite oracle\ndirections", "Aligned oracle\ndirections"])
    axes[2].set_ylim(-.5, 1.02)
    axes[2].axhline(0, color="#777", lw=.8)
    axes[2].set_ylabel("Cross-instance cosine")
    axes[2].set_title("C. Gradient decomposition*", loc="left", fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=7.8, frameon=False)
    fig.text(.01, -.02, "Five image-held-out COCO train folds, 71,269 GT-conditioned candidates. Error bars: image-cluster 95% CI.", fontsize=8)
    fig.text(.01, -.085, "*Panel C: exploratory 400+400 high-h-similarity pairs selected by oracle-direction extremes; not a population prevalence estimate.", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, 1), w_pad=2.2)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "FIGURE_7N_MECHANISM.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.out / "FIGURE_7N_MECHANISM.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("retrieval", "oof", "gradient", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
