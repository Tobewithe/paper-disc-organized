"""Build a layered, mutually-auditable failure taxonomy from S048/S056 outputs.

The taxonomy keeps four questions separate:
1) Was a same-class final slot assigned? 2) Was its box good?
3) Did the box provide enough support for the GT? 4) Where did mask error go?
Relation topology and density are context fields, never failure causes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def density_bin(x: float, source: str, positive_median: float) -> str:
    """Stable descriptive bins; zero is kept separate from positive exposure."""
    if source == "undefined" or not np.isfinite(x):
        return "undefined"
    if x <= 1e-10:
        return "zero"
    return "positive_low" if x <= positive_median else "positive_high"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--relations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    d = pd.read_csv(args.instances)
    r = pd.read_csv(args.relations, usecols=["annotation_id", "relation"])
    d = d.merge(r, on="annotation_id", how="left", validate="one_to_one")
    if len(d) != 36335 or d.annotation_id.nunique() != len(d):
        raise RuntimeError("GT denominator or relation join is not one-to-one")

    # Density is a continuous covariate. Quartiles are descriptive only and
    # prevent a threshold selected after observing AP from defining a result.
    e = pd.to_numeric(d["same_boundary_exposure4"], errors="coerce")
    valid = e.notna() & (d["mask_density"] != "undefined") & (e > 1e-10)
    positive_median = float(e[valid].median()) if valid.any() else 0.0
    d["density_q"] = [density_bin(float(x), str(s), positive_median) for x, s in zip(e, d["mask_density"])]
    d["density_e4"] = e

    d["candidate_state"] = np.where(d["matched"], "final_same_class_slot", "no_final_same_class_slot")
    d["box_state"] = np.where(d["matched"], np.where(d["box_iou"] >= .75, "box_good", "box_bad"), "unresolved")
    d["mask_state"] = np.where(d["matched"], np.where(d["mask_iou"] >= .75, "mask_good", "mask_bad"), "unresolved")
    d["support_state"] = np.where(
        d["matched"] & (d["box_iou"] >= .75),
        np.where(d["box_gt_pixel_coverage"] >= .95, "support_sufficient", "support_limited"),
        "not_applicable",
    )

    # Raw error profiles are retained as continuous fractions. A primary label
    # is supplied only for compact tables; it is not treated as causal.
    err_cols = {
        "self_fn": "target_fn",
        "same_neighbor_fp": "same_neighbor_error",
        "other_neighbor_fp": "other_only_error",
        "background_fp": "background_error",
    }
    for name, col in err_cols.items():
        d[name] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    arr = d[list(err_cols)].to_numpy(float)
    labels = np.array(list(err_cols))
    d["error_surface_primary"] = [labels[i] if row.sum() > 0 else "unresolved" for row, i in zip(arr, arr.argmax(axis=1))]
    d["error_surface_n_active"] = (arr > 0).sum(axis=1)
    d["error_surface"] = np.where(
        d["error_surface_n_active"] == 0,
        "none_or_unresolved",
        np.where(d["error_surface_n_active"] == 1, d["error_surface_primary"], "mixed"),
    )
    # Only call a surface substantial when it occupies >=5% of valid target
    # area; this avoids turning one stray pixel into a mechanism claim.
    d["substantial_surface"] = np.where(
        arr.max(axis=1) >= .05, d["error_surface_primary"], "none_below_5pct"
    )

    matched = d["matched"]
    box_good = matched & (d["box_iou"] >= .75)
    support_good = box_good & (d["box_gt_pixel_coverage"] >= .95)
    mask_good = matched & (d["mask_iou"] >= .75)
    d["failure_scope"] = np.select(
        [~matched,
         matched & ~box_good & ~mask_good,
         matched & ~box_good & mask_good,
         box_good & ~support_good & ~mask_good,
         box_good & ~support_good & mask_good,
         support_good & ~mask_good,
         support_good & mask_good],
        ["no_final_slot",
         "box_limited_mask_bad",
         "box_limited_mask_good",
         "box_good_support_limited_mask_bad",
         "box_good_support_limited_mask_good",
         "box_good_support_sufficient_mask_bad",
         "box_good_support_sufficient_mask_good"],
        default="unresolved",
    )

    # Compact summaries, all denominators explicit.
    def tab(cols: list[str]) -> list[dict]:
        g = d.groupby(cols, dropna=False).size().reset_index(name="n")
        return json.loads(g.to_json(orient="records"))

    summary = {
        "protocol": {
            "population": "COCO val2017 ordinary GT, crowd/ignore excluded",
            "density": "continuous same-class E4 plus zero/positive-low/positive-high descriptive bins; density never defines cause",
            "failure_scope": "final slot -> box quality -> box GT support -> mask quality (mutually exclusive)",
            "surface_threshold": "raw fractions retained; substantial_surface is >=5% of valid GT area",
            "relation": "final candidate topology joined as context only",
            "limitations": [
                "no_final_slot cannot distinguish raw candidate absence, classification, NMS, or score filtering",
                "primary/substantial surface is descriptive and can hide mixed errors; use raw fractions for claims",
                "fixed-slot IoU is an attribution diagnostic, not official AP assignment",
            ],
        },
        "counts": {
            "gt": int(len(d)),
            "undefined_density": int((d.density_q == "undefined").sum()),
            "positive_e4_median": positive_median,
        },
        "tables": {
            "failure_scope": tab(["failure_scope"]),
            "scope_by_density_q": tab(["density_q", "failure_scope"]),
            "surface_by_density_q": tab(["density_q", "substantial_surface"]),
            "state_by_density_q": tab(["density_q", "state"]),
            "relation_by_scope": tab(["relation", "failure_scope"]),
            "surface_by_relation": tab(["relation", "substantial_surface"]),
        },
    }
    d.to_csv(out / "instances_layered.csv", index=False)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(
        "# Layered COCO failure taxonomy\n\n"
        "Outcome, support, error surface, candidate topology and density are separate fields. "
        "Use `failure_scope` for mutually exclusive accounting and the four raw error fractions "
        "for mechanism analysis. `substantial_surface` only flags a descriptive 5% area threshold.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "gt": len(d), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
