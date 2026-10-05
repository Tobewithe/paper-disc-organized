"""Audit retained candidate coverage across failure classes (R006)."""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB = Path(r"C:\Dpan\codexproject\pigcv_research\data\analysis\pigcv_analysis.db")
OUT = Path(__file__).resolve().parents[1] / "experiments" / "yolo26_candidate_gate_20260905"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    query = """
    WITH ca AS (
      SELECT dataset, image_id, annotation_id,
             COUNT(*) AS candidate_rows,
             MAX(CASE WHEN gt_mask_iou >= 0.50 THEN 1 ELSE 0 END) AS any_mask_good,
             MAX(CASE WHEN gt_mask_iou >= 0.50 AND in_topk = 1 THEN 1 ELSE 0 END) AS topk_mask_good,
             MAX(CASE WHEN gt_mask_iou >= 0.50 AND survives_to_final = 1 THEN 1 ELSE 0 END) AS final_mask_good
      FROM candidates
      GROUP BY dataset, image_id, annotation_id
    )
    SELECT g.dataset, g.primary_class,
           COUNT(*) AS failed_gt,
           SUM(CASE WHEN ca.candidate_rows IS NOT NULL THEN 1 ELSE 0 END) AS gt_with_candidate,
           SUM(CASE WHEN COALESCE(ca.candidate_rows, 0) = 0 THEN 1 ELSE 0 END) AS gt_without_candidate,
           COALESCE(SUM(ca.candidate_rows), 0) AS candidate_rows,
           COALESCE(SUM(ca.any_mask_good), 0) AS gt_with_mask_iou_ge_050,
           COALESCE(SUM(ca.topk_mask_good), 0) AS gt_with_topk_mask_iou_ge_050,
           COALESCE(SUM(ca.final_mask_good), 0) AS gt_with_final_mask_iou_ge_050
    FROM gt_instances AS g
    LEFT JOIN ca USING (dataset, image_id, annotation_id)
    WHERE g.total_failure = 1
    GROUP BY g.dataset, g.primary_class
    ORDER BY g.dataset, g.primary_class
    """
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(query).fetchall()]
        columns = [d[0] for d in conn.execute(query).description]
    out_csv = OUT / "candidate_coverage_by_class.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(DB),
        "rows": rows,
        "interpretation_boundary": "Coverage is limited to retained scalar candidates; missing rows indicate an audit blind spot, not proof that the runtime generated no candidate.",
    }
    (OUT / "candidate_coverage_by_class.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
