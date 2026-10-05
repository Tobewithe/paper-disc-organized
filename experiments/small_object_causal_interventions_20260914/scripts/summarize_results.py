"""Summarize completed cached interventions without rerunning inference."""
from pathlib import Path
import json
from causal_image_probe import read_csv, write_csv, bootstrap

out = Path(__file__).resolve().parents[1]
rows = read_csv(out / "per_target_intervention.csv")
by_key = {(r["annotation_id"], r["intervention"]): r for r in rows}
metrics = ["best_any_box_iou", "best_any_true_score", "best_true_box_iou", "best_true_score",
           "local_true_score", "local_box_iou", "local_top1", "raw_box50"]
interventions = ["contrast_increase", "contrast_decrease", "neighbor_remove", "contrast_plus_neighbor_remove"]
summary = []
for cohort in sorted({r["cohort"] for r in rows}):
    ids = sorted({r["annotation_id"] for r in rows if r["cohort"] == cohort})
    for intervention in interventions:
        record = {"cohort": cohort, "intervention": intervention, "n": len(ids),
                  "box50_recovered": "", "box50_lost": ""}
        for metric in metrics:
            delta = [float(by_key[(aid, intervention)][metric]) - float(by_key[(aid, "original")][metric]) for aid in ids]
            mean, lo, hi = bootstrap(delta)
            record[metric + "_delta"] = mean
            record[metric + "_ci_low"] = lo
            record[metric + "_ci_high"] = hi
        if cohort == "raw_geometry_small":
            record["box50_recovered"] = sum(int(by_key[(aid, "original")]["raw_box50"]) == 0 and int(by_key[(aid, intervention)]["raw_box50"]) == 1 for aid in ids)
            record["box50_lost"] = sum(int(by_key[(aid, "original")]["raw_box50"]) == 1 and int(by_key[(aid, intervention)]["raw_box50"]) == 0 for aid in ids)
        summary.append(record)
write_csv(out / "summary.csv", summary)
(out / "COMPLETE.json").write_text(json.dumps({"status": "complete", "targets": len(rows) // 5,
    "inferences": len(rows), "ultralytics": "8.4.100", "end2end": False,
    "gt_role": "controlled image intervention and outcome measurement only"}, ensure_ascii=False, indent=2), encoding="utf-8")
