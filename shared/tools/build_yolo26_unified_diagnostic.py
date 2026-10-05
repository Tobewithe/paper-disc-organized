"""Build a provenance-aware, machine-readable YOLO26 diagnostic bundle."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(r"C:\Dpan\codexproject\pigcv_research\artifacts\analysis\yolo26seg_diagnostic_full_single_forward_20260901_final")
COCO = Path(r"C:\Dpan\document\model_datasets\datasets\piglife\derived\task05_v1\pig_coco_test_task05_v1.json")
OUT = ROOT / "experiments" / "yolo26_unified_diagnostic_20260905"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    basic = read_csv(LEGACY / "basic_summary.csv")
    scene = read_csv(LEGACY / "scene_analysis.csv")
    onset = read_csv(LEGACY / "failure_stage_analysis.csv")
    coco = json.loads(COCO.read_text(encoding="utf-8"))
    piglife_basic = next(row for row in basic if row["dataset"] == "PigLife_public_test")
    label_audit = {
        "path": str(COCO),
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": coco.get("categories", []),
        "matches_legacy_gt": int(piglife_basic["gt"]) == len(coco.get("annotations", [])),
        "matches_legacy_images": int(piglife_basic["images"]) == len(coco.get("images", [])),
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "YOLO26 diagnostic cache, fixed evaluator; no new forward or training",
        "legacy_root": str(LEGACY),
        "datasets": basic,
        "label_audit": label_audit,
        "status": "audit_complete" if all(label_audit[k] for k in ("matches_legacy_gt", "matches_legacy_images")) else "alignment_failed",
        "interpretation_boundary": "Cache replay and inherited tables establish reproducible descriptive evidence, not causal attribution or deployable intervention performance.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT / "baseline_summary.csv", basic)
    write_csv(OUT / "scene_failure_taxonomy.csv", scene)
    write_csv(OUT / "failure_onset_taxonomy.csv", onset)
    (OUT / "provenance_audit.json").write_text(json.dumps({"label_audit": label_audit, "inputs": [str(LEGACY / name) for name in ("basic_summary.csv", "scene_analysis.csv", "failure_stage_analysis.csv")]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "status": summary["status"], "datasets": len(basic), "piglife_gt": label_audit["annotations"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
