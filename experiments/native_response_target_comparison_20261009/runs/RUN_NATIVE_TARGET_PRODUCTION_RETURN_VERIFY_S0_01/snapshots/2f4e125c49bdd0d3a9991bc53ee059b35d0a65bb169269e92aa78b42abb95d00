"""Fail-closed CPU receipt wrapper for the complete target production/cost bundle.

Keep the original byte receiver unchanged. Verify declared source Run sets,
sealed collection completion, full archive inventory and every source manifest
before calling that pinned receiver. No GT, model or scientific execution.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import sys
import time
import traceback
import zipfile

BASE_RECEIVER_SHA = "e92af1da7959398050c2d815566eae425d882754da1c3191c4a663747a571454"
COMMON_SHA = "4de7929a7278038de66c0daacba17403217db71643d8b77ccd806658bb1f011f"
PRODUCTION = {"RUN_NATIVE_TARGET_PREFLIGHT_S0_01", "RUN_NATIVE_TARGET_PIPELINE_S0_01"}
PRODUCTION.update("RUN_NATIVE_TARGET_" + phase + "_" + stage + "_S0_01"
                  for phase in ("ENGINEERING", "FORMAL") for stage in ("EXTRACT", "FIT_I", "FIT_H", "INFERENCE"))
COST = {"RUN_NATIVE_TARGET_COST_ENGINEERING_S0_01", "RUN_NATIVE_TARGET_COST_PANEL_S0_01"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-receiver", "collection", "destination", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--kind", choices=("production", "cost"), required=True)
    a = p.parse_args()
    base_path = Path(a.base_receiver).resolve()
    require(sha(base_path) == BASE_RECEIVER_SHA and sha(base_path.with_name("target_common.py")) == COMMON_SHA,
            "Pinned original receiver/common source changed")
    from target_common import dump, now
    collection, out = Path(a.collection).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "BASE_RECEIVER").exists() and not (out / "COMPLETE.json").exists(), "Receipt retry requires a new Run")
    began = time.perf_counter()
    watched = {}
    try:
        def watch(path, expected=None):
            actual = sha(path)
            require(expected is None or actual == expected, "Received source digest differs: " + str(path))
            watched[str(path)] = actual
            return actual
        watch(base_path, BASE_RECEIVER_SHA)
        watch(base_path.with_name("target_common.py"), COMMON_SHA)
        watch(__file__)
        metadata, summary, complete, inventory = [read(collection / n) for n in
                    ("run.json", "SUMMARY.json", "COMPLETE.json", "BUNDLE_MANIFEST.json")]
        for name in ("run.json", "SUMMARY.json", "COMPLETE.json", "BUNDLE_MANIFEST.json"):
            watch(collection / name)
        require(metadata["source_kind"] == "runner_observed" and metadata["status"] == "completed" and
                metadata["return_code"] == 0 and metadata["artifact_completeness"] == "complete", "Actual collection runner gate differs")
        require(summary["passed"] is True and summary["status"] == "sealed" and summary["source_unchanged"] is True and
                complete["passed"] is True and complete["summary_sha256"] == sha(collection / "SUMMARY.json") and
                complete["bundle_sha256"] == summary["bundle_sha256"], "Original complete collection seal differs")
        expected_roots = PRODUCTION if a.kind == "production" else COST
        names = [PureWindowsPath(v if isinstance(v, str) else v["path"]).name for v in inventory["source_runs"]]
        require(len(names) == len(set(names)) and set(names) == expected_roots, "Declared source Run set differs")
        require(summary["source_runs"] == len(expected_roots) if a.kind == "production" else
                [PureWindowsPath(v["path"]).name for v in summary["source_runs"]] == names, "Collection summary source Runs differ")
        bundle_name = "PRODUCTION_BUNDLE.zip" if a.kind == "production" else "EXTRA_RUNS_BUNDLE.zip"
        archive = collection / bundle_name
        watch(archive, summary["bundle_sha256"])
        watch(collection / "BUNDLE_MANIFEST.json", summary["manifest_sha256"])
        require(archive.stat().st_size == summary["bundle_bytes"], "Received archive byte count differs")
        rows = inventory["files"]
        require(len(rows) == summary["source_files"] and len(rows) > 0 and
                sum(v["bytes"] for v in rows) == summary["source_bytes"], "Complete source file/byte denominators differ")
        paths = [v["archive"] for v in rows]
        require(len(paths) == len(set(paths)), "Archive inventory duplicates source file")
        by_root = {name: set() for name in expected_roots}
        with zipfile.ZipFile(archive) as z:
            require(set(z.namelist()) == set(paths) | {"BUNDLE_MANIFEST.json"} and
                    len(z.namelist()) == len(paths)+1, "Actual archive file set differs from full original inventory")
            require(z.read("BUNDLE_MANIFEST.json") == (collection / "BUNDLE_MANIFEST.json").read_bytes(),
                    "In-archive original inventory bytes differ")
            for item in rows:
                relative = PureWindowsPath(item["archive"])
                require(not relative.is_absolute() and ".." not in relative.parts and
                        relative.parts[0] in expected_roots and len(relative.parts) > 1, "Source inventory path escapes expected Run")
                root, rest = relative.parts[0], "/".join(relative.parts[1:])
                source_path = PureWindowsPath(item["source"])
                require(tuple(source_path.parts[-len(relative.parts):]) == relative.parts,
                        "Retained remote source path differs from archive identity")
                by_root[root].add(rest)
                data = z.read(item["archive"])
                require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                        "Full original source file bytes differ")
            for root, files in by_root.items():
                require({"run.json", "manifest.sha256", "SUMMARY.json"}.issubset(files), "Source Run lacks minimum original receipts")
                rm = json.loads(z.read(root + "/run.json"))
                require(rm["source_kind"] == "runner_observed" and rm["run_id"] == root and rm["status"] == "completed" and
                        rm["return_code"] == 0 and rm["artifact_completeness"] == "complete", "Source Run actual execution completion differs")
                lines = z.read(root + "/manifest.sha256").decode("utf-8-sig").splitlines()
                manifest = {}
                for line in lines:
                    if line.strip():
                        expected, name = line.split(None, 1)
                        name = name.strip().lstrip("*").replace("\\", "/")
                        require(name not in manifest and not PureWindowsPath(name).is_absolute() and ".." not in PureWindowsPath(name).parts,
                                "Source manifest path duplicate/escape")
                        manifest[name] = expected
                require(set(manifest) == files - {"manifest.sha256"}, "Full source manifest coverage differs")
                for name, expected in manifest.items():
                    require(hashlib.sha256(z.read(root + "/" + name)).hexdigest() == expected, "Original source manifest digest differs")
        spec = importlib.util.spec_from_file_location("target_complete_byte_receiver", base_path)
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        old_argv = sys.argv
        try:
            sys.argv = [str(base_path), "--collection", str(collection), "--destination", a.destination,
                        "--bundle-name", bundle_name, "--output", str(out / "BASE_RECEIVER")]
            base.main()
        finally:
            sys.argv = old_argv
        for path, expected in watched.items():
            require(sha(path) == expected, "Collection changed during complete original-byte consumption")
        result = read(out / "BASE_RECEIVER" / "SUMMARY.json")
        result.update(kind=a.kind, inventory_and_source_sets_verified=True, collection_seal_verified=True,
                      original_receiver_sha256=BASE_RECEIVER_SHA, verifier_sha256=sha(__file__),
                      elapsed_seconds=time.perf_counter()-began, completed_at=now())
        dump(out / "SUMMARY.json", result)
        dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json")))
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
