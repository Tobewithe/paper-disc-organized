"""Refresh and fully verify current local manifests for collected ACD Runs.

The default mode is read-only verification. --refresh updates only each Run's
manifest.sha256, then verifies every entry and complete local-file coverage.
_source_manifest.sha256 is preserved byte-for-byte and included in the current
manifest. collection.zip, manifest.sha256 and mutable transfer.json are excluded
by filename, as in collect_run.py. No execution/transfer metadata is rewritten.

Examples:
  python scripts/refresh_manifests.py --refresh
  python scripts/refresh_manifests.py --verify --run RUN_ACD_PAIRED_EVAL_S0
  python scripts/refresh_manifests.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

WORKSPACE = Path("C:/Dpan/codexproject/paper-disc-organized")
EXPERIMENT_ROOT = WORKSPACE / "experiments" / "acd_proto_tail_unfreeze_20261006"
EXCLUDED_NAMES = frozenset({"collection.zip", "manifest.sha256", "transfer.json"})
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(RuntimeError):
    """An incomplete, unsafe or inconsistent current local manifest."""


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def no_link(path: Path) -> None:
    info = path.lstat()
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
        raise ManifestError(f"Link/reparse point is unsupported: {path}")


def scoped_run(run: Path, root: Path) -> Path:
    root = root.resolve(strict=True)
    # Check links before resolving a selected Run; a junction must not lead the
    # enumeration into another checkout or an external artifact directory.
    no_link(root / "runs")
    no_link(run)
    run = run.resolve(strict=True)
    if run.parent != root / "runs" or not run.is_dir():
        raise ManifestError(f"Run is not a direct child of this experiment's runs/: {run}")
    return run


def sha_file(path: Path) -> str:
    no_link(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ManifestError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def local_files(run: Path) -> dict[str, Path]:
    """Enumerate all regular local files; never follow a link or junction."""
    found: dict[str, Path] = {}
    directories = [run]
    while directories:
        directory = directories.pop()
        for path in directory.iterdir():
            no_link(path)
            if not path.resolve(strict=True).is_relative_to(run):
                raise ManifestError(f"Local path escapes Run: {path}")
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                directories.append(path)
            elif stat.S_ISREG(mode):
                if path.name.lower() in EXCLUDED_NAMES:
                    continue
                relative = path.relative_to(run).as_posix()
                if "\n" in relative or "\r" in relative:
                    raise ManifestError(f"Filename cannot be represented in SHA256 manifest: {relative!r}")
                found[relative] = path
            else:
                raise ManifestError(f"Non-regular local file is unsupported: {path}")
    return dict(sorted(found.items()))


def collection_identity(run: Path) -> tuple[bytes, dict[str, Any]]:
    """A verified transfer plus its immutable source-manifest marker is required."""
    marker, transfer_path = run / "_source_manifest.sha256", run / "transfer.json"
    for path in (marker, transfer_path):
        try:
            no_link(path)
        except FileNotFoundError as error:
            raise ManifestError("Run is not collected: _source_manifest.sha256 or transfer.json is missing") from error
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ManifestError(f"Collection marker/receipt is not a regular file: {path.name}")
    original = marker.read_bytes()
    try:
        transfer = json.loads(transfer_path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise ManifestError(f"Cannot read transfer receipt: {error}") from error
    if not isinstance(transfer, dict) or transfer.get("status") != "verified":
        raise ManifestError("Run is not a verified collected transfer")
    declared = transfer.get("source_manifest_sha256")
    if not isinstance(declared, str) or not HASH_RE.fullmatch(declared):
        raise ManifestError("Transfer receipt lacks the verified source_manifest_sha256")
    if digest_bytes(original) != declared.lower():
        raise ManifestError("Immutable source manifest differs from the verified transfer receipt")
    return original, transfer


def manifest_entries(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as error:
        raise ManifestError(f"Manifest is not UTF-8: {error}") from error
    entries: dict[str, str] = {}
    for index, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        if "  " not in line:
            raise ManifestError(f"Manifest line {index} lacks SHA256 double-space separator")
        digest, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        if not HASH_RE.fullmatch(digest):
            raise ManifestError(f"Invalid SHA256 on manifest line {index}")
        if (not relative or path.is_absolute() or PureWindowsPath(relative).drive
                or "\\" in relative or ".." in path.parts or path.as_posix() != relative):
            raise ManifestError(f"Unsafe or noncanonical relative path on manifest line {index}: {relative!r}")
        if path.name.lower() in EXCLUDED_NAMES:
            raise ManifestError(f"Excluded mutable/bundle file is listed in manifest: {relative}")
        if relative in entries:
            raise ManifestError(f"Duplicate manifest entry: {relative}")
        entries[relative] = digest.lower()
    if not entries or "_source_manifest.sha256" not in entries:
        raise ManifestError("Current manifest must include _source_manifest.sha256")
    return entries


def verify_run(run: Path, root: Path) -> dict[str, Any]:
    run = scoped_run(run, root)
    original, transfer = collection_identity(run)
    manifest_path = run / "manifest.sha256"
    if not manifest_path.is_file():
        raise ManifestError("Current manifest.sha256 is missing")
    no_link(manifest_path)
    raw = manifest_path.read_bytes()
    entries = manifest_entries(raw)
    files = local_files(run)
    missing, unlisted = sorted(set(entries) - set(files)), sorted(set(files) - set(entries))
    if missing or unlisted:
        raise ManifestError(f"Incomplete manifest coverage: missing={missing}; unlisted={unlisted}")
    mismatches = [relative for relative, path in files.items() if sha_file(path) != entries[relative]]
    if mismatches:
        raise ManifestError(f"SHA256 mismatch: {mismatches}")
    # Inventory and source bytes must still agree after the full verification.
    if set(local_files(run)) != set(files):
        raise ManifestError("Local file inventory changed during verification")
    if (run / "_source_manifest.sha256").read_bytes() != original:
        raise ManifestError("Source manifest changed during verification")
    if manifest_path.read_bytes() != raw:
        raise ManifestError("Current manifest changed during verification")
    return {"run_id": run.name, "verified": True, "verified_files": len(entries),
            "verified_bytes": sum(path.stat().st_size for path in files.values()),
            "manifest_sha256": digest_bytes(raw), "source_manifest_sha256": digest_bytes(original),
            "source_manifest_preserved": True, "transfer_status": transfer["status"]}


def refresh_run(run: Path, root: Path) -> dict[str, Any]:
    run = scoped_run(run, root)
    original, _ = collection_identity(run)
    # Include source snapshots, reports, run.json, DECISION.json and every local
    # extra receipt, even when they were added after the original collection.
    files = local_files(run)
    entries = {relative: sha_file(path) for relative, path in files.items()}
    if (run / "_source_manifest.sha256").read_bytes() != original:
        raise ManifestError("Source manifest changed before refresh; no manifest written")
    manifest_path = run / "manifest.sha256"
    if manifest_path.exists():
        no_link(manifest_path)
    raw = "".join(f"{digest}  {relative}\n" for relative, digest in entries.items()).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=run, prefix="manifest.sha256.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
        temporary = None
    finally:
        if temporary is not None and temporary.parent.resolve() == run and temporary.is_file():
            temporary.unlink()
    if (run / "_source_manifest.sha256").read_bytes() != original:
        raise ManifestError("Source manifest changed during refresh")
    result = verify_run(run, root)
    result["refreshed"] = True
    return result


def select_runs(root: Path, names: list[str] | None) -> tuple[list[Path], list[dict[str, str]]]:
    run_root = root / "runs"
    if not run_root.is_dir():
        raise ManifestError(f"Experiment runs directory is missing: {run_root}")
    no_link(run_root)
    selected, skipped = [], []
    if names:
        for name in dict.fromkeys(names):
            if not name or name in (".", "..") or "/" in name or "\\" in name or PureWindowsPath(name).drive:
                raise ManifestError(f"Expected a Run ID, not a path: {name!r}")
            run = scoped_run(run_root / name, root)
            collection_identity(run)
            selected.append(run)
    else:
        for run in sorted(run_root.iterdir()):
            no_link(run)
            if not run.is_dir():
                continue
            try:
                collection_identity(run)
            except (OSError, ManifestError) as error:
                skipped.append({"run_id": run.name, "reason": str(error)})
                continue
            selected.append(run)
    return selected, skipped


def self_test(root: Path) -> dict[str, Any]:
    """Small temporary fixture under this workspace; never refresh real Runs."""
    scratch_parent = root / "scripts"
    no_link(scratch_parent)
    if not scratch_parent.resolve(strict=True).is_relative_to(root):
        raise ManifestError("Self-test scratch parent escapes experiment")
    with tempfile.TemporaryDirectory(prefix=".manifest-selftest-", dir=scratch_parent) as temporary:
        fixture = Path(temporary).resolve()
        # Verify the absolute recursive-cleanup target before entering fixture
        # work; TemporaryDirectory removes only this checked task-owned path.
        if fixture.parent != scratch_parent.resolve() or not fixture.is_relative_to(root):
            raise ManifestError("Self-test cleanup target escapes experiment")
        run = fixture / "runs" / "RUN_MANIFEST_SELF_TEST"
        (run / "source").mkdir(parents=True)
        (run / "run.json").write_bytes(b'{"status":"failed"}\r\n')
        (run / "REPORT.md").write_text("source report\n", encoding="utf-8")
        (run / "source" / "train_acd.py").write_bytes(b"# actual execution snapshot\n")
        original = (f"{sha_file(run / 'run.json')}  run.json\r\n"
                    f"{sha_file(run / 'REPORT.md')}  REPORT.md\r\n").encode("utf-8")
        (run / "_source_manifest.sha256").write_bytes(original)
        receipt = {"status": "verified", "source_manifest_sha256": digest_bytes(original)}
        transfer_before = json.dumps(receipt).encode("utf-8")
        (run / "transfer.json").write_bytes(transfer_before)
        (run / "collection.zip").write_bytes(b"untouched bundle fixture")
        (run / "manifest.sha256").write_bytes(original)
        (run / "run.json").write_bytes(b'{"status":"failed","local_note":"retained"}\n')
        (run / "REPORT.md").write_text("current local report\n", encoding="utf-8")
        (run / "DECISION.json").write_bytes(b'{"outcome":"unknown"}\n')
        (run / "LOCAL_RECEIPT.json").write_bytes(b'{"available":true}\n')
        (run / "\u9644\u52a0\u8bc1\u636e.txt").write_text("Unicode filename is covered\n", encoding="utf-8")
        payload_before = {key: path.read_bytes() for key, path in local_files(run).items()}
        refreshed = refresh_run(run, fixture)
        assert refreshed["verified_files"] == 7
        assert {key: path.read_bytes() for key, path in local_files(run).items()} == payload_before
        assert (run / "_source_manifest.sha256").read_bytes() == original
        assert (run / "transfer.json").read_bytes() == transfer_before
        assert (run / "collection.zip").read_bytes() == b"untouched bundle fixture"
        entries = manifest_entries((run / "manifest.sha256").read_bytes())
        assert not any(PurePosixPath(name).name in EXCLUDED_NAMES for name in entries)
        (run / "extra.json").write_bytes(b"{}\n")
        try:
            verify_run(run, fixture)
        except ManifestError as error:
            assert "unlisted" in str(error)
        else:
            raise AssertionError("Unlisted local receipt was not detected")
        assert refresh_run(run, fixture)["verified_files"] == 8
        receipt["mutable_note"] = "transfer is intentionally excluded"
        (run / "transfer.json").write_text(json.dumps(receipt), encoding="utf-8")
        assert verify_run(run, fixture)["verified"]
        (run / "REPORT.md").write_text("tampered after refresh\n", encoding="utf-8")
        try:
            verify_run(run, fixture)
        except ManifestError as error:
            assert "SHA256 mismatch" in str(error)
        else:
            raise AssertionError("Modified report was not detected")
        for bad in ("../escape.json", "C:/escape.json", "source/../../escape.json"):
            try:
                manifest_entries(f"{'0' * 64}  {bad}\n".encode())
            except ManifestError:
                pass
            else:
                raise AssertionError("Unsafe manifest path was accepted")
        assert (run / "_source_manifest.sha256").read_bytes() == original
    return {"self_test": "passed", "fixture_files_verified": [7, 8],
            "checks": ["complete local coverage", "immutable CRLF source-manifest bytes",
                       "reports/run.json/DECISION/source/extra receipts included", "Unicode filenames",
                       "bundle/manifest/mutable transfer excluded", "unlisted and modified files rejected",
                       "unsafe paths rejected"], "temporary_fixture_removed": True,
            "real_runs_refreshed": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--run", action="append", dest="runs", help="Run ID; repeat to select multiple collected Runs")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refresh", action="store_true", help="Refresh current local manifest, then fully verify")
    modes.add_argument("--verify", action="store_true", help="Read-only full verification (default)")
    modes.add_argument("--self-test", action="store_true", help="Quick isolated fixture under this experiment's scripts/")
    args = parser.parse_args()
    # Reject another root lexically before resolving/accessing that directory.
    if args.root.absolute() != EXPERIMENT_ROOT.absolute():
        parser.error(f"This tool is limited to {EXPERIMENT_ROOT}")
    root = args.root.resolve(strict=True)
    expected = EXPERIMENT_ROOT.resolve(strict=True)
    if root != expected or not root.is_relative_to(WORKSPACE.resolve(strict=True)):
        parser.error(f"This tool is limited to {expected}")
    if args.self_test:
        if args.runs:
            parser.error("--self-test cannot select real Runs")
        print(json.dumps(self_test(root), ensure_ascii=False))
        return 0
    try:
        runs, skipped = select_runs(root, args.runs)
    except (OSError, ManifestError) as error:
        print(json.dumps({"mode": "refresh" if args.refresh else "verify", "error": str(error)}, ensure_ascii=False))
        return 1
    results, errors = [], []
    for run in runs:
        try:
            results.append(refresh_run(run, root) if args.refresh else verify_run(run, root))
        except (OSError, ManifestError) as error:
            errors.append({"run_id": run.name, "error": str(error)})
    print(json.dumps({"mode": "refresh" if args.refresh else "verify", "root": str(root),
                      "runs": results, "errors": errors, "uncollected_runs_skipped": skipped,
                      "excluded_names": sorted(EXCLUDED_NAMES), "other_metadata_modified": False},
                     ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
