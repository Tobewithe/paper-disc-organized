"""Compute the candidate-stage oracle/gate from the retained scalar DB."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DB = Path(r"C:\Dpan\codexproject\pigcv_research\data\analysis\pigcv_analysis.db")
OUT = Path(__file__).resolve().parents[1] / "experiments" / "yolo26_candidate_gate_20260905"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        rows = pd.read_sql_query(
            """
            WITH c AS (
              SELECT dataset, image_id, annotation_id,
                     MAX(CASE WHEN gt_mask_iou >= 0.50 THEN 1 ELSE 0 END) raw_good,
                     MAX(CASE WHEN gt_mask_iou >= 0.50 AND in_topk = 1 THEN 1 ELSE 0 END) topk_good,
                     MAX(CASE WHEN gt_mask_iou >= 0.50 AND survives_to_final = 1 THEN 1 ELSE 0 END) final_good,
                     MAX(CASE WHEN gt_mask_iou >= 0.50 AND in_topk = 1 AND survives_to_final = 0 THEN 1 ELSE 0 END) topk_dropped_good
              FROM candidates GROUP BY dataset, image_id, annotation_id
            )
            SELECT g.dataset, g.primary_class, COUNT(*) AS failed_gt,
                   COALESCE(SUM(c.raw_good), 0) AS raw_good,
                   COALESCE(SUM(c.topk_good), 0) AS topk_good,
                   COALESCE(SUM(c.final_good), 0) AS final_good,
                   COALESCE(SUM(c.topk_dropped_good), 0) AS topk_dropped_good
            FROM gt_instances g LEFT JOIN c USING(dataset, image_id, annotation_id)
            WHERE g.total_failure = 1
            GROUP BY g.dataset, g.primary_class
            ORDER BY g.dataset, g.primary_class
            """, conn)
    rows.to_csv(OUT / "candidate_gate_by_dataset_class.csv", index=False, encoding="utf-8-sig")
    totals = rows.groupby("dataset", as_index=False)[["failed_gt", "raw_good", "topk_good", "final_good", "topk_dropped_good"]].sum()
    totals["oracle_recovery_rate"] = totals["topk_good"] / totals["failed_gt"]
    totals["selection_loss_rate"] = totals["topk_dropped_good"] / totals["failed_gt"]
    totals.to_csv(OUT / "candidate_gate_by_dataset.csv", index=False, encoding="utf-8-sig")
    overall = totals[["failed_gt", "raw_good", "topk_good", "final_good", "topk_dropped_good"]].sum()
    gate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(DB),
        "threshold": {"mask_iou": 0.50, "topk": True},
        "overall": {k: int(overall[k]) for k in overall.index},
        "oracle_recovery_rate": float(overall["topk_good"] / overall["failed_gt"]),
        "selection_loss_rate": float(overall["topk_dropped_good"] / overall["failed_gt"]),
        "datasets_with_direction": int((totals["oracle_recovery_rate"] >= 0.40).sum()),
        "candidate_classes_present": sorted(rows.loc[rows["raw_good"] > 0, "primary_class"].unique().tolist()),
        "gate_status": "not_passed_candidate_route" if int((totals["oracle_recovery_rate"] >= 0.40).sum()) < 2 else "candidate_route_supported_descriptively",
        "interpretation_boundary": "Oracle and selection-loss rates are upper-bound/descriptive evidence from retained scalar tables; they do not prove causality or justify training a scorer without perturbation controls.",
    }
    (OUT / "summary.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == "__main__":
    main()
