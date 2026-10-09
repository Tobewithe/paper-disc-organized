"""Return already completed pipeline metadata; no experiment execution."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
run = json.loads((root / "run.json").read_text(encoding="utf-8"))
assert run["status"] == "completed" and run["return_code"] == 0
assert run["artifact_completeness"] == "complete"
names = ["run.json", "PIPELINE_STATUS.json", "PIPELINE_CONFIG.json",
         "stdout.log", "stderr.log",
         "launcher.stdout.log", "launcher.stderr.log"]
files = [root / name for name in names]
assert all(path.is_file() for path in files)
files += sorted((root / "snapshots").rglob("*"))
files = [path for path in files if path.is_file()]
assert len(files) == 11 and sum(path.stat().st_size for path in files) < 5_000_000
records = [{"path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in files]
receipt = {"schema_version": 1, "run_id": run["run_id"],
           "scope": "terminal_pipeline_metadata_and_original_snapshots",
           "captured_at": datetime.now(timezone.utc).isoformat(),
           "source_run_status": run["status"], "source_return_code": 0,
           "source_artifact_completeness": run["artifact_completeness"],
           "files": records}
manifest = "".join(f'{item["sha256"]}  {item["path"]}\n' for item in records)
package = root / "PIPELINE_TERMINAL_RETURN.zip"
with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())
    archive.writestr("TERMINAL_BUNDLE_RECEIPT.json",
                     json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    archive.writestr("manifest.sha256", manifest)
output = {"package": str(package), "bytes": package.stat().st_size,
          "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
          "files": len(records)}
(root / "PIPELINE_TERMINAL_PACKAGE.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output))
