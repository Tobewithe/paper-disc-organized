"""Verify a complete returned original-byte bundle; preserve remote run.json."""
import argparse
import json
from pathlib import Path
import zipfile
from target_common import dump, now, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--bundle-name", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    collection, destination, out = Path(a.collection).resolve(), Path(a.destination).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((collection / "SUMMARY.json").read_text())
    metadata = json.loads((collection / "run.json").read_text())
    if metadata["status"] != "completed" or metadata["return_code"] != 0 or metadata["artifact_completeness"] != "complete" or not summary["passed"]:
        raise ValueError("Actual source seal is not complete")
    archive = collection / a.bundle_name
    inventory = json.loads((collection / "BUNDLE_MANIFEST.json").read_text())
    if sha(archive) != summary["bundle_sha256"] or sha(collection / "BUNDLE_MANIFEST.json") != summary["manifest_sha256"]:
        raise ValueError("Received package/source inventory differs")
    roots = {}
    with zipfile.ZipFile(archive) as z:
        for info in inventory["files"]:
            relative = Path(info["archive"])
            path = (destination / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(destination):
                raise ValueError("Archive file escapes intended Study Run root")
            data = z.read(info["archive"])
            import hashlib
            if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError("Original returned source bytes differ")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Existing local source artifact differs; preserve instead of overwrite")
            if not path.exists():
                path.write_bytes(data)
            roots[relative.parts[0]] = roots.get(relative.parts[0], 0) + 1
    for name in roots:
        root = destination / name
        for line in (root / "manifest.sha256").read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                expected, relative = line.split(None, 1)
                path = (root / relative.strip().lstrip("*")).resolve()
                if not path.is_relative_to(root) or sha(path) != expected:
                    raise ValueError("Original source Run manifest differs")
        original = json.loads((root / "run.json").read_text())
        dump(root / "transfer.json", dict(status="complete_scientific_consumption", all_manifest_sha256_verified=True,
             source_run_directory=original["locations"][0]["path"], destination_run_directory=str(root),
             source_status=original["status"], source_return_code=original["return_code"],
             source_artifact_completeness=original["artifact_completeness"], source_run_json_preserved=True,
             package_sha256=summary["bundle_sha256"], package_inventory_sha256=summary["manifest_sha256"],
             verified_source_files=roots[name], completed_at=now()))
    result = dict(passed=True, source_collection=str(collection), roots=roots, verified_source_files=len(inventory["files"]),
                  package_sha256=summary["bundle_sha256"], package_inventory_sha256=summary["manifest_sha256"],
                  source_run_json_preserved=True, completed_at=now(), gpu_used=False, gt_parsed=False)
    dump(out / "SUMMARY.json", result)
    dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json")))


if __name__ == "__main__":
    main()
