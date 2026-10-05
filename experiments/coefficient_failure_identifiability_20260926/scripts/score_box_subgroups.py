"""Exploratory conditional check: does the gate work when the box is already good?"""
import argparse
import json
from pathlib import Path

import numpy as np

from score_identifiability import ARMS, binary_metrics, cluster_contrast


def main(a):
    scores = json.loads((a.scores / "SCORES.json").read_text(encoding="utf-8"))
    summary = json.loads((a.scores / "SUMMARY.json").read_text(encoding="utf-8"))
    evaluated = json.loads((a.evaluation / "ROWS.json").read_text(encoding="utf-8"))
    meta = {(r["image_id"], r["annotation_id"]): r for r in evaluated}
    assert len(scores) == len(evaluated) == len(meta)
    groups = {
        "good_box": [r for r in scores if meta[(r["image_id"], r["annotation_id"])]["box_iou"] >= .75],
        "poor_box": [r for r in scores if meta[(r["image_id"], r["annotation_id"])]["box_iou"] < .75],
    }
    output = {}
    for group, rows in groups.items():
        output[group] = dict(images=len({r["image_id"] for r in rows}), instances=len(rows), tasks={})
        for task in ("failure", "high_gap"):
            labels = np.array([r[task] for r in rows], dtype=int)
            arm_metrics = {}
            for arm in ARMS:
                values = np.array([r["scores"][arm][task] for r in rows])
                threshold = summary["results"][arm][task]["val"]["threshold"]
                arm_metrics[arm] = binary_metrics(labels, values, threshold)
            output[group]["tasks"][task] = dict(arms=arm_metrics)
            if group == "good_box":
                contrasts = {}
                for other in ("box_score", "h", "h_raw_detection"):
                    a_scores = np.array([r["scores"]["h_local_space"][task] for r in rows])
                    b_scores = np.array([r["scores"][other][task] for r in rows])
                    contrasts["h_local_space_minus_" + other] = cluster_contrast(
                        rows, labels, a_scores, b_scores,
                        summary["results"]["h_local_space"][task]["val"]["threshold"],
                        summary["results"][other][task]["val"]["threshold"])
                output[group]["tasks"][task]["contrasts"] = contrasts
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "SUBGROUPS.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({g: {t: {arm: {m: output[g]["tasks"][t]["arms"][arm][m]
                      for m in ("prevalence", "auroc", "auprc", "threshold_recall", "threshold_fpr")}
                      for arm in ARMS} for t in ("failure", "high_gap")} for g in output}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("scores", "evaluation", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    main(ap.parse_args())
