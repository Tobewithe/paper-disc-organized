"""Original-byte seal for extra finite Run products (no scientific computation)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import zipfile
from target_common import dump, now, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, action="append")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "BUNDLE_MANIFEST.json").exists():
        raise FileExistsError("Seal retry requires new Run")
    began, inventory, runs = time.perf_counter(), [], []
    for value in a.run:
        root = Path(value).resolve()
        metadata = json.loads((root / "run.json").read_text())
        if metadata["status"] not in ("completed", "failed"):
            raise ValueError("Cannot seal an active Run")
        runs.append(dict(path=str(root), status=metadata["status"], return_code=metadata["return_code"], artifact_completeness=metadata["artifact_completeness"]))
        lines = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in ("manifest.sha256", "transfer.json") or path.name.endswith(".tmp"):
                continue
            relative, digest = path.relative_to(root).as_posix(), sha(path)
            lines.append(digest + "  " + relative)
            inventory.append(dict(source=str(path), archive=root.name + "/" + relative, sha256=digest, bytes=path.stat().st_size))
        manifest = root / "manifest.sha256"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        inventory.append(dict(source=str(manifest), archive=root.name + "/manifest.sha256", sha256=sha(manifest), bytes=manifest.stat().st_size))
    dump(out / "BUNDLE_MANIFEST.json", dict(source_runs=runs, files=inventory, failure_facts_preserved=True))
    archive = out / "EXTRA_RUNS_BUNDLE.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for info in inventory:
            if sha(info["source"]) != info["sha256"]:
                raise ValueError("Source changed before sealing")
            z.write(info["source"], info["archive"])
        z.write(out / "BUNDLE_MANIFEST.json", "BUNDLE_MANIFEST.json")
    for info in inventory:
        if sha(info["source"]) != info["sha256"]:
            raise ValueError("Source changed during sealing")
    summary = dict(passed=True, status="sealed", source_runs=runs, source_files=len(inventory), source_bytes=sum(i["bytes"] for i in inventory),
                   bundle_sha256=sha(archive), bundle_bytes=archive.stat().st_size, manifest_sha256=sha(out / "BUNDLE_MANIFEST.json"),
                   elapsed_seconds=time.perf_counter()-began, completed_at=now(), transfer_complete=False, source_unchanged=True)
    dump(out / "SUMMARY.json", summary)
    dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), bundle_sha256=summary["bundle_sha256"]))


if __name__ == "__main__":
    main()
