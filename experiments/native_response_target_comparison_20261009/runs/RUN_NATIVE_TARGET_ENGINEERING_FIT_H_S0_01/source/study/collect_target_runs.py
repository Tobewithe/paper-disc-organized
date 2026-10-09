"""Seal completed producer Runs into a complete original-byte return package."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import zipfile

from target_common import dump, now, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--preflight", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    pipeline, preflight, out = Path(args.pipeline).resolve(), Path(args.preflight).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "BUNDLE_MANIFEST.json").exists():
        raise FileExistsError("Collection retry requires new Run")
    began = time.perf_counter()
    complete = json.loads((pipeline / "PIPELINE_COMPLETE.json").read_text())
    if not complete["passed"] or complete["summary_sha256"] != sha(pipeline / "SUMMARY.json"):
        raise ValueError("Finite pipeline is incomplete")
    roots = [preflight, pipeline]
    for phase in complete["stages"].values():
        roots.extend(Path(phase[key]) for key in ("extract", "fit_I", "fit_H", "inference"))
    inventory = []
    for root in roots:
        metadata = json.loads((root / "run.json").read_text())
        if metadata["status"] != "completed" or metadata["return_code"] != 0 or metadata["artifact_completeness"] != "complete":
            raise ValueError("Source Run not runner-complete: " + str(root))
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("manifest.sha256", "transfer.json") and not p.name.endswith(".tmp"))
        manifest_lines = []
        for path in files:
            digest = sha(path)
            relative = path.relative_to(root).as_posix()
            manifest_lines.append(digest + "  " + relative)
            inventory.append(dict(source=str(path), archive=root.name + "/" + relative, sha256=digest, bytes=path.stat().st_size))
        (root / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        path = root / "manifest.sha256"
        inventory.append(dict(source=str(path), archive=root.name + "/manifest.sha256", sha256=sha(path), bytes=path.stat().st_size))
    dump(out / "BUNDLE_MANIFEST.json", dict(source_runs=[str(root) for root in roots], files=inventory,
                                          scope="every source Run file; only transfer.json, transient .tmp and self-manifest excluded from original manifest"))
    archive = out / "PRODUCTION_BUNDLE.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for info in inventory:
            path = Path(info["source"])
            if sha(path) != info["sha256"]:
                raise ValueError("Source changed before packaging")
            z.write(path, info["archive"])
        z.write(out / "BUNDLE_MANIFEST.json", "BUNDLE_MANIFEST.json")
    for info in inventory:
        if sha(info["source"]) != info["sha256"]:
            raise ValueError("Source changed during packaging")
    summary = dict(passed=True, status="sealed", source_runs=len(roots), source_files=len(inventory),
                   source_bytes=sum(i["bytes"] for i in inventory), bundle_sha256=sha(archive), bundle_bytes=archive.stat().st_size,
                   manifest_sha256=sha(out / "BUNDLE_MANIFEST.json"), source_unchanged=True,
                   elapsed_seconds=time.perf_counter()-began, completed_at=now(),
                   actual_execution_environment="local_laptop_28358lan", transfer_complete=False)
    dump(out / "SUMMARY.json", summary)
    dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), bundle_sha256=summary["bundle_sha256"]))


if __name__ == "__main__":
    main()
