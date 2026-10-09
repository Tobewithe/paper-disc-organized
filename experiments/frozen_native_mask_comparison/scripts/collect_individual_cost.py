"""Package completed cost Runs and exact panel references; no inference."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone

study = Path(sys.argv[1]).resolve()
engine_id = "RUN_FROZEN_NATIVE_COST_ENGINEERING_S0_01"
formal_id = "RUN_FROZEN_NATIVE_COST_PANEL_S0_01"
entries = []
per_run = {}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def add(path, destination):
    if any(record["path"] == destination for record in entries):
        raise ValueError("Duplicate archive destination")
    entries.append({"path": destination, "source_path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})


for run_id, count, children in ((engine_id, 20, 5), (formal_id, 480, 15)):
    run = study / "runs" / run_id
    recorded = json.loads((run / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "completed" and recorded["return_code"] == 0 and recorded["artifact_completeness"] == "complete"
    assert summary["sample_count"] == count and summary["child_processes"] == children
    assert summary["all_output_parity_passed"] and summary["all_isolation_passed"]
    for path in sorted(run.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == ".run.claim":
            continue
        if path.suffix == ".zip" or path.name.startswith("COST_PANEL_RETURN") or path.name in ("transfer.json", "manifest.sha256"):
            continue
        add(path, path.relative_to(study).as_posix())
    per_run[run_id] = {"source_status": recorded["status"], "return_code": 0,
                       "artifact_completeness": recorded["artifact_completeness"], "samples": count}

formal = study / "runs" / formal_id
inputs = json.loads((formal / "COST_INPUTS.json").read_text(encoding="utf-8"))
native, triflow = Path(inputs["reference_run"]), Path(inputs["triflow_reference"])
arms = inputs["arm_names"]
for iid in inputs["image_ids"]:
    for arm in arms:
        origin = triflow / "triflow" if arm == "TriFlow_final8" else native / arm
        add(origin / "images" / f"{iid:012d}.json", f"runs/{formal_id}/reference_panel/{arm}/images/{iid:012d}.json")
    native_image = json.loads((native / "baseline" / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))
    add(Path(native_image["image_path"]), f"assets/cost_panel/images/{iid:012d}.jpg")
for name in ("SUMMARY.json", "BASELINE_PARITY.json", "BASELINE_PARITY_IMAGES.jsonl", "EVALUATION_INPUTS.json"):
    add(native / name, f"runs/{formal_id}/reference_panel/native_provenance/{name}")
add(triflow / "HEAD_PROVENANCE.json", f"runs/{formal_id}/reference_panel/triflow_provenance/HEAD_PROVENANCE.json")

bundle = {"version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
          "scope": "complete original engineering/formal cost artifacts plus original32 JPEGs and five-arm reference JSON copies",
          "source_host": "28358lan", "source_study": str(study), "runs": per_run,
          "source_file_count": len(entries), "source_bytes": sum(record["bytes"] for record in entries), "files": entries}
package = formal / "COST_PANEL_COMPLETE_RETURN.zip"
with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
    for record in entries:
        archive.write(record["source_path"], record["path"])
    archive.writestr(f"runs/{formal_id}/COST_RETURN_BUNDLE.json", json.dumps(bundle, indent=2) + "\n")
    for run_id in per_run:
        prefix = f"runs/{run_id}/"
        records = [record for record in entries if record["path"].startswith(prefix)]
        manifest = "".join(record["sha256"] + "  " + record["path"][len(prefix):] + "\n" for record in records)
        archive.writestr(prefix + "manifest.sha256", manifest)
        archive.writestr(prefix + "COST_SOURCE_PACKAGE_RECEIPT.json", json.dumps({"run_id": run_id, "scope": bundle["scope"],
                           "files": records, "source_status": "completed", "return_code": 0}, indent=2) + "\n")
receipt = {"package": str(package), "bytes": package.stat().st_size, "sha256": digest(package),
           "source_files": len(entries), "source_bytes": bundle["source_bytes"], "captured_at": bundle["captured_at"]}
(formal / "COST_PANEL_RETURN_PACKAGE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt))
