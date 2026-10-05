"""S068 descriptive triage of S067's joint box/mask failure population.

Does not equate floating-box support with the exact decoder support. This
table decides whether the next controlled probe must prioritize crop support,
retained-candidate geometry, or masks within an already adequate support.
"""
from pathlib import Path
import argparse
import json
import shutil
import pandas as pd
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    d = pd.read_csv(a.input)
    d = d[d.primary_state.eq("box_bad_mask_bad")].copy()
    assert len(d) == 6584
    # Presence does not imply jointly feasible assignment, but prevents saying
    # the model never produced an adequate retained box/mask.
    d["retained_box75"] = d.best_kept_box_iou >= .75
    d["proxy_support_band"] = pd.cut(d.box_gt_pixel_coverage, [-np.inf,.75,.95,np.inf],
        right=False, labels=["below75", "75to95", "atleast95"])
    d["box_band"] = pd.cut(d.box_iou, [-np.inf,.6,.7,.75], right=False,
        labels=["below60", "60to70", "70to75"])
    d["mask_band"] = pd.cut(d.mask_iou, [-np.inf,.5,.7,.75], right=False,
        labels=["below50", "50to70", "70to75"])
    d["oversize_ratio"] = (d.box_iou*0+np.nan) # Box areas are not inferred from IoU.
    rows = []
    for group in ["all", "low", "middle", "high", "undefined"]:
        z = d if group == "all" else d[d.mask_density.eq(group)]
        rec = dict(group=group, n=len(z), official_miss=int(z.task_mask75.eq("miss").sum()),
            any_retained_mask75=int(z.alternate_sameclass_mask75.sum()),
            any_retained_box75=int(z.retained_box75.sum()),
            support_median=float(z.box_gt_pixel_coverage.median()),
            box_iou_median=float(z.box_iou.median()), mask_iou_median=float(z.mask_iou.median()))
        for threshold in [.75,.90,.95]:
            rec[f"support_below_{threshold}"] = int((z.box_gt_pixel_coverage < threshold).sum())
        for size in ["small", "medium", "large"]:
            rec[f"n_{size}"] = int(z.area_bin.eq(size).sum())
        rows.append(rec)
    d.drop(columns="oversize_ratio").to_csv(a.out / "joint_failure_instances.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(a.out / "summary.csv", index=False)
    crossings = ["mask_density", "proxy_support_band", "retained_box75", "alternate_sameclass_mask75", "task_mask75"]
    d.groupby(crossings, observed=True, dropna=False).size().reset_index(name="n").to_csv(a.out / "support_candidate_task.csv",index=False)
    for band in ["box_band", "mask_band", "area_bin"]:
        d.groupby(["mask_density",band], observed=True, dropna=False).size().reset_index(name="n").to_csv(a.out/f"{band}.csv",index=False)
    (a.out / "protocol.json").write_text(json.dumps(dict(experiment="S068_JOINT_FAILURE_DESCRIPTIVE_TRIAGE",
        population="All 6584 S048 fixed-box50 assigned BoxIoU<.75 AND MaskIoU<.75",
        intervention="none; deterministic reaggregation of previously measured per-GT fields",
        question="Within the newly prioritized state, how often is target support geometrically inadequate or an adequate retained candidate available?",
        actionable="Choose support intervention versus candidate selection/mask quality probe, retain every selected case regardless of later improvement",
        caution="Support is floating-box geometric proxy, not actual decoder upper bound. Box/mask candidate availability may not be jointly assignable. Descriptive associations do not establish a cause.",
        source=str(a.input.resolve())),indent=2),encoding="utf-8")
    shutil.copy2(__file__,a.out/Path(__file__).name)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
