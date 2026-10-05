"""Make a sensitivity-audited COCO failure taxonomy from the fixed-slot census.

The output separates observable states from mechanism hypotheses.  It reports
the mutually exclusive primary state, continuous margins, and a robust error
surface profile whose ``mixed`` label requires two *substantial* surfaces.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ERR = ["self_fn", "same_neighbor_fp", "other_neighbor_fp", "background_fp"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve(); out.mkdir(parents=False, exist_ok=False)
    d = pd.read_csv(a.input)
    required = ["annotation_id", "matched", "box_iou", "mask_iou", "box_gt_pixel_coverage",
                "same_neighbor_error", "other_only_error", "background_error", "target_fn"]
    miss = [x for x in required if x not in d]
    if miss: raise RuntimeError(f"missing columns: {miss}")
    if d.annotation_id.nunique() != len(d): raise RuntimeError("annotation_id is not unique")
    b = d["matched"].astype(str).str.lower().isin(["true", "1"])
    box = pd.to_numeric(d.box_iou, errors="coerce")
    mask = pd.to_numeric(d.mask_iou, errors="coerce")
    cov = pd.to_numeric(d.box_gt_pixel_coverage, errors="coerce")
    d["box_iou_margin75"] = box - .75
    d["mask_iou_margin75"] = mask - .75
    d["support_margin95"] = cov - .95
    d["mask_gap_box_minus_mask"] = box - mask
    d["mask_band"] = pd.cut(mask, [-np.inf, .50, .75, .90, np.inf], labels=["severe_<.50", "moderate_.50-.75", "near_.75-.90", "good_>=.90"], right=False).astype(str)
    d["alternate_sameclass_mask75"] = d["any_kept_mask75"].astype(str).str.lower().isin(["true", "1"]) if "any_kept_mask75" in d else False
    alt_box_mask = pd.to_numeric(d.get("bestbox_slot_mask_iou", pd.Series(np.nan, index=d.index)), errors="coerce")
    d["best_box_slot_mask75"] = alt_box_mask >= .75
    if not np.isfinite(d.loc[b, ["box_iou", "mask_iou", "box_gt_pixel_coverage"]].to_numpy(float)).all():
        raise RuntimeError("Matched slots have missing geometry; do not silently classify as success")
    # Five observable states. Support and alternate candidates are independent
    # dimensions, not additional causes or replacements for task matching.
    d["primary_state"] = np.select(
        [~b,
         b & (box < .75) & (mask < .75),
         b & (box < .75) & (mask >= .75),
         b & (box >= .75) & (mask < .75),
         b & (box >= .75) & (mask >= .75)],
        ["no_bbox50_assignment", "box_bad_mask_bad", "box_bad_mask_good",
         "box_good_mask_bad", "box_good_mask_good"], default="unresolved")
    assert not d.primary_state.eq("unresolved").any()
    d["support_proxy"] = np.select([~b, cov >= .95], ["not_measured", "at_least_95pct"], default="below_95pct")
    d["supported_mask_failure"] = b & (box >= .75) & (cov >= .95) & (mask < .75)
    official_hit = d.official_mask75.astype(str).str.lower().isin(["true", "1"])
    d["strict_no_available_mask_failure"] = d.supported_mask_failure & ~d.alternate_sameclass_mask75 & ~official_hit
    d["task_mask75"] = np.where(official_hit, "hit", "miss")
    d["evidence_tier"] = np.select(
        [~b, mask >= .75, d.strict_no_available_mask_failure,
         d.supported_mask_failure & d.alternate_sameclass_mask75,
         b & (box >= .75) & (cov < .95) & (mask < .75),
         b & (box < .75) & (mask < .75)],
        ["upstream_unresolved", "fixed_mask_success_not_necessarily_task_success",
         "mask_branch_investigation_candidate", "retained_candidate_availability_check",
         "crop_geometry_check", "localization_and_mask_unresolved"], default="unresolved")
    assert not d.evidence_tier.eq("unresolved").any()
    # Use a 5% area threshold per surface. Tiny nonzero numerical remnants do
    # not turn a single-surface failure into a spurious 'mixed' mechanism.
    d["surface_profile"] = "none_or_below_5pct"
    e = pd.DataFrame({
        "self_fn": pd.to_numeric(d.target_fn, errors="coerce").fillna(0),
        "same_neighbor_fp": pd.to_numeric(d.same_neighbor_error, errors="coerce").fillna(0),
        "other_neighbor_fp": pd.to_numeric(d.other_only_error, errors="coerce").fillna(0),
        "background_fp": pd.to_numeric(d.background_error, errors="coerce").fillna(0),
    })
    active = e >= .05
    for i in d.index:
        names = list(e.columns[active.loc[i]])
        if len(names) == 1: d.at[i, "surface_profile"] = names[0]
        elif len(names) > 1: d.at[i, "surface_profile"] = "mixed_substantial"
    d.loc[~b, "surface_profile"] = "not_measured"
    # S048 allocated same/different ambiguous pixels to same. Preserve that
    # convention, but additionally expose a fully disjoint five-bin partition.
    ambiguous = pd.to_numeric(d.same_other_gt_overlap_fp, errors="coerce") / pd.to_numeric(d.valid_gt_pixels, errors="coerce").clip(lower=1)
    d["ambiguous_neighbor_fp_fraction"] = ambiguous
    d["same_only_fp_fraction"] = d.same_neighbor_error - ambiguous
    assert (d.loc[b, "same_only_fp_fraction"] >= -1e-9).all()
    # Sensitivity of the central readout-failure count to pre-registered,
    # nearby thresholds. This is a robustness diagnostic, not model tuning.
    sensitivity = []
    for bt in [.50, .75, .90]:
        for st in [.90, .95, .98]:
            for mt in [.50, .75]:
                readout = b & (box >= bt) & (cov >= st) & (mask < mt)
                sensitivity.append({"box_iou_threshold": bt, "support_threshold": st, "mask_iou_threshold": mt,
                                    "n": int(readout.sum()), "rate_all_gt": float(readout.mean()),
                                    "n_without_available_mask75": int((readout & ~d.alternate_sameclass_mask75 & ~official_hit).sum())})
    def counts(cols):
        return json.loads(d.groupby(cols, dropna=False).size().reset_index(name="n").to_json(orient="records"))
    summary = {
        "protocol": {"population": "COCO val2017 ordinary GT from fixed-slot census",
                     "primary_state": "Same-class bbox50 one-to-one assigned slot; Box75 x Mask75 on that SAME slot",
                     "central_thresholds": {"box_iou": .75, "support": .95, "mask_iou": .75},
                     "surface_threshold": 0.05,
                     "causal_language": "states and evidence tiers are descriptive; no state is itself a cause",
                     "no_bbox50_assignment": "No assigned same-class bbox50 slot, NOT absence of all final or raw candidates",
                     "support": "Original-image floating box raster is a geometric proxy, not exact pipeline crop support or a proof of mask expressivity",
                     "density": "Reuse S048 same-class E4 low=0, middle=(0,.2), high>=.2; ICI and median-split density_q remain separately named",
                     "candidate": "Any retained same-class Mask75 availability is an oracle edge, not guaranteed jointly assignable or automatically selectable",
                     "task": "Official Mask75 matching independent from the diagnostic bbox50 assignment"},
        "counts": {"gt": int(len(d)), "supported_mask_failure": int(d.supported_mask_failure.sum()),
                   "supported_with_retained_mask75": int((d.supported_mask_failure & d.alternate_sameclass_mask75).sum()),
                   "supported_with_bestbox_mask75": int((d.supported_mask_failure & d.best_box_slot_mask75).sum()),
                   "strict_no_available_mask_failure": int(d.strict_no_available_mask_failure.sum())},
        "tables": {"primary_state": counts(["primary_state"]),
                   "candidate_evidence": counts(["primary_state", "alternate_sameclass_mask75", "best_box_slot_mask75"]),
                   "state_by_mask_band": counts(["primary_state", "mask_band"]),
                   "surface_by_state": counts(["primary_state", "surface_profile"]),
                   "evidence_tier": counts(["evidence_tier"]),
                   "state_by_e4": counts(["mask_density", "primary_state"]),
                   "state_by_task": counts(["primary_state", "task_mask75"]),
                   "state_by_support": counts(["primary_state", "support_proxy"]),
                   "strict_pool_by_e4": counts(["mask_density", "strict_no_available_mask_failure"])},
        "sensitivity": sensitivity,
    }
    d.to_csv(out / "instances_improved.csv", index=False)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text("# COCO failure dimensions\n\nFive primary SAME-slot outcome states; support proxy, retained-candidate availability, official task success, pixel-error surfaces, relation graph and GT density are independent dimensions. Neither a state nor an oracle repair proves a network cause. Unmatched slots have unmeasured surfaces, not zero errors. `strict_no_available_mask_failure` excludes ANY retained same-class Mask75 and official Mask75 success. Previous best-box-only exclusion left 132 available-mask cases in its alleged clean pool. Historical artifacts remain unchanged.\n", encoding="utf-8")
    print(json.dumps({"status":"COMPLETE", "gt":len(d), "out":str(out)}, ensure_ascii=False))

if __name__ == "__main__": main()
