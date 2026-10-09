"""Plot fixed diagnostic probe records; no model imports or step selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import fmean
import sys


PROBES = (("A", 0.0), ("tiny", 1 / 4096), ("small", 1 / 128),
          ("train_inner", 1 / 8), ("train_outer", 1 / 4), ("first_nominal", 1 / 2))
IDENTITY = ("image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
TZ = timezone(timedelta(hours=8), "Asia/Shanghai")


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    out = (args.out or study / "runs" / "RUN_damage_curve_20261006").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    source = study / "runs" / "RUN_damage_gradient_local_20261006"
    inputs = [{"path": str(source / name), "sha256": sha(source / name)}
              for name in ("PER_CANDIDATE.jsonl", "SUMMARY.json", "run.json")]
    run = {
        "schema_version": 2,
        "run_id": out.name,
        "study_id": json.loads((study / "study.json").read_text(encoding="utf-8"))["study_id"],
        "source_kind": "derived_diagnostic",
        "status": "running",
        "started_at": now(),
        "timezone": "Asia/Shanghai",
        "inputs": inputs,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "command": [sys.executable, str(Path(__file__).resolve()), "--study", str(study), "--out", str(out)],
        "working_directory": str(Path.cwd()),
        "execution_environment": "local CPU statistics and plotting",
        "model_execution": False,
        "parameter_updates": False,
        "step_selection": False,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "scope": "Candidate-micro paired mean diagnostic curves for baseline-success candidates from fixed first 64 DEV images; no formal method benefit claim.",
    }
    save(out / "run.json", run)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullLocator

        run["environment"]["matplotlib"] = matplotlib.__version__
        metadata = json.loads((source / "SUMMARY.json").read_text(encoding="utf-8"))
        source_run = json.loads((source / "run.json").read_text(encoding="utf-8"))
        if source_run["status"] != "completed" or metadata["metadata"]["requested_images"] != 64:
            raise AssertionError("Plot requires completed fixed 64-image diagnostic")
        rows = [json.loads(line) for line in (source / "PER_CANDIDATE.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len({tuple(row[key] for key in IDENTITY) for row in rows}) != len(rows):
            raise AssertionError("Candidate identities are not unique")
        if set(row["image_id"] for row in rows) != set(metadata["metadata"]["image_ids"]):
            raise AssertionError("Source rows do not cover fixed 64 DEV image IDs")
        success = [row for row in rows if row["iou_A"] >= .75]
        if len(success) != 323 or len(success) != metadata["groups"]["baseline_success"]["candidates"]:
            raise AssertionError("Unexpected baseline-success count")
        rho = float(metadata["metadata"]["rho"])
        points = []
        for name, fraction in PROBES:
            point = {"probe": name, "fraction_of_rho": fraction, "displacement_norm": fraction * rho, "candidate_pairs": len(success)}
            for metric in ("q", "soft", "iou"):
                value = fmean(row[f"{metric}_{name}"] - row[f"{metric}_A"] for row in success)
                point[f"mean_delta_{metric}"] = value
            point["mean_delta_training_soft_iou_pp"] = point["mean_delta_soft"] * 100
            point["mean_delta_original_hard_iou_pp"] = point["mean_delta_iou"] * 100
            if name != "A":
                for metric, sourcekey in (("soft", "mean_delta_soft_iou"), ("iou", "mean_delta_hard_iou")):
                    expected = metadata["groups"]["baseline_success"][sourcekey][name]
                    if abs(point[f"mean_delta_{metric}"] - expected) > 1e-12:
                        raise AssertionError("Source aggregate and plotted values differ")
            points.append(point)
        summary = {
            "source_kind": "derived_diagnostic",
            "created_at": now(),
            "parameter_updates": False,
            "model_execution": False,
            "step_selection": False,
            "source_sha256": {Path(item["path"]).name: item["sha256"] for item in inputs},
            "code_sha256": run["code"]["sha256"],
            "requested_dev_images": 64,
            "source_candidates": len(rows),
            "baseline_success_candidates": len(success),
            "baseline_success_images": len({row["image_id"] for row in success}),
            "cohort": "same 323 baseline normal-original-image MaskIoU >= .75 candidates for every point; all classes/box qualities included",
            "aggregation": "candidate micro mean of paired changes relative to c0; not image macro; no CI",
            "rho": rho,
            "success_training_max_radius": rho / 4,
            "actual_first_step": rho / 2,
            "x_axis_scale": "symlog fraction-of-rho axis, linear threshold rho/4096; exact labeled fixed probe fractions",
            "points": points,
            "interpretation_limits": "Fixed-direction mechanism diagnostic; no step/radius selection or claim of method benefit. Success-training radius does not bound all Failure training states. Training SoftIoU uses actual cropped overlap target; HardIoU uses normal original-image decode, hence their label scope differs. Connected probe points guide the eye and do not establish unobserved extrema.",
            "checks": {"identity_unique": True, "same_cohort_each_point": True, "source_soft_hard_aggregate_match_at_1e_minus12": True},
        }
        save(out / "SUMMARY.json", summary)

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.titlesize": 11, "axes.labelsize": 10,
                             "legend.fontsize": 9, "pdf.fonttype": 42,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
        fig.subplots_adjust(left=.12, right=.97, top=.83, bottom=.18, hspace=.34)
        x = [point["fraction_of_rho"] for point in points]
        axes[0].plot(x, [point["mean_delta_q"] for point in points], "o-", color="#275D9F", linewidth=2, markersize=5, label="Mean change in Q")
        axes[0].set_ylabel("Change in Q (score)")
        axes[0].set_title("a  Learned quality score", loc="left")
        axes[1].plot(x, [point["mean_delta_training_soft_iou_pp"] for point in points], "o-", color="#B76A16", linewidth=2, markersize=5, label="Training SoftIoU")
        axes[1].plot(x, [point["mean_delta_original_hard_iou_pp"] for point in points], "s-", color="#277F67", linewidth=2, markersize=5, label="Original-image HardIoU")
        axes[1].set_ylabel("Change in IoU (pp)")
        axes[1].set_title("b  Training surrogate and normal original-image decode", loc="left")
        axes[1].set_xlabel("Displacement along fixed initial Q-gradient direction (symlog axis)", labelpad=11)
        for ax in axes:
            ax.axhline(0, color="#9AA1A8", linewidth=.8, zorder=0)
            ax.axvline(.25, color="#686868", linestyle="--", linewidth=1.2,
                       label="Success training max: rho/4" if ax is axes[0] else None)
            ax.axvline(.5, color="#A5373A", linestyle=":", linewidth=1.6,
                       label="Actual first step: rho/2" if ax is axes[0] else None)
            ax.set_xscale("symlog", linthresh=1 / 4096, linscale=.6)
            ax.set_xlim(-.00003, .64)
            ax.grid(axis="y", color="#E5E7EB", linewidth=.6)
            ax.set_axisbelow(True)
            ax.xaxis.set_minor_locator(NullLocator())
        axes[1].set_xticks(x, ["0", "rho/4096", "rho/128", "rho/8", "rho/4", "rho/2"])
        axes[0].legend(loc="upper left", frameon=False)
        axes[1].legend(loc="lower left", frameon=False)
        small = points[2]
        axes[1].annotate(f"rho/128: Soft {small['mean_delta_training_soft_iou_pp']:+.3f} pp\nHard {small['mean_delta_original_hard_iou_pp']:+.3f} pp",
                         xy=(small["fraction_of_rho"], small["mean_delta_training_soft_iou_pp"]),
                         xytext=(.10, .28), textcoords="axes fraction", fontsize=9,
                         arrowprops={"arrowstyle": "-", "color": "#7A7A7A", "lw": .8})
        fig.suptitle("Fixed-direction QCR displacement diagnostic", x=.12, ha="left", y=.965, fontsize=15, fontweight="bold")
        fig.text(.12, .918, "Fixed first 64 DEV images | baseline success n = 323 (63 images)", fontsize=10)
        fig.text(.12, .889, "Candidate micro means of paired changes from c0 | diagnostic only; no step selection", fontsize=9, color="#555555")
        fig.text(.12, .045, "Same candidates at all six probes. Lines connect observed points; they do not imply measured intermediate states.\nSuccess training max refers to Success perturbations only. No method benefit or generalization claim.", fontsize=8.3, color="#555555", linespacing=1.5)
        png, pdf = out / "QCR_SUCCESS_DISPLACEMENT.png", out / "QCR_SUCCESS_DISPLACEMENT.pdf"
        fig.savefig(png, dpi=200, facecolor="white")
        fig.savefig(pdf, facecolor="white", metadata={"Title": "Fixed-direction QCR diagnostic", "Subject": "Fixed 64 DEV; baseline success 323; paired micro means; no step selection"})
        plt.close(fig)
        run.update(status="completed", finished_at=now(), return_code=0, artifact_completeness="complete",
                   artifacts=[{"path": str(out / name), "sha256": sha(out / name)} for name in ("SUMMARY.json", png.name, pdf.name)])
        save(out / "run.json", run)
        print(json.dumps({"png": str(png), "pdf": str(pdf), "status": "completed"}))
    except Exception as error:
        run.update(status="failed", finished_at=now(), return_code=1, artifact_completeness="partial", error=f"{type(error).__name__}: {error}")
        save(out / "run.json", run)
        raise


if __name__ == "__main__":
    main()
