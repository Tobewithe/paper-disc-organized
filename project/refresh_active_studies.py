"""Register disk study folders in the project-level Workbench catalog.

The catalog keeps one semantic record per experiment/diagnostic directory.  The
Workbench file index remains responsible for individual files, so large output
directories are not expanded into thousands of asset rows here.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "project" / "structure.json"


def representative_records(folder: Path) -> list[str]:
    preferred = (
        "REPORT.md", "FINAL_REPORT.md", "PILOT_RESULTS.json", "SUMMARY.json",
        "COMPLETE.json", "TARGETED_COMPLETE.json", "protocol.json", "study.json",
    )
    return [p.relative_to(ROOT).as_posix() for name in preferred if (p := folder / name).is_file()]


def main() -> None:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    if data.get("generated_by") == "researchctl":
        # Preserve old callers while keeping the new journal/index as the sole catalog writer.
        try:
            from workbench.research.cli import main as research_main
        except ImportError as exc:
            raise SystemExit("目录已由 researchctl 管理；请在工作台环境运行 researchctl.ps1 --project paper-disc sync") from exc
        raise SystemExit(research_main(["--project", data.get("project_id", "paper-disc"), "sync"]))
    studies = data.setdefault("studies", [])
    by_path = {row["path"]: row for row in studies}
    changed = []
    for parent in (ROOT / "diagnostics", ROOT / "experiments"):
        for marker in sorted(parent.glob("*/study.json")):
            folder = marker.parent
            rel = folder.relative_to(ROOT).as_posix()
            meta = json.loads(marker.read_text(encoding="utf-8"))
            kind = meta.get("kind", "diagnostic" if parent.name == "diagnostics" else "experiment")
            files = [p for p in folder.rglob("*") if p.is_file()]
            code = [p.relative_to(ROOT).as_posix() for p in files if p.suffix.lower() in {".py", ".ps1", ".sh"}]
            row = by_path.get(rel)
            if row is None:
                row = {
                    "study_id": f"paper-disc:{kind}:{folder.name}",
                    "name": folder.name,
                    "kind": kind,
                    "path": rel,
                    "original_path": rel,
                    "scope": meta.get("scope", "project"),
                    "status": meta.get("status", "documented"),
                    "provenance": "目录级研究对象；结论以 REPORT/事实沉淀为准",
                    "file_count": len(files),
                    "code_paths": code,
                    "record_paths": representative_records(folder),
                    "shared_code_path": "shared/coco_clean_20260911",
                    "classification_status": "由 study.json 明确登记",
                }
                studies.append(row)
                by_path[rel] = row
                changed.append(rel)
            else:
                row.update(status=meta.get("status", row.get("status", "documented")),
                           file_count=len(files), code_paths=code,
                           record_paths=representative_records(folder))
    CATALOG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registered_new": changed, "catalog_studies": len(studies)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
