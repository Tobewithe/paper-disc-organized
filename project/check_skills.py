#!/usr/bin/env python3
"""Check portable Skill sources; explicitly refresh derived manifests when requested.

Default is read-only. Never installs Skills, runs their code, or contacts services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "skills"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(files: dict[str, str]) -> str:
    return digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def file_hashes(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        rel = path.relative_to(directory)
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Review symlink before recording: {path}")
        if path.is_file():
            result[rel.as_posix()] = digest(path.read_bytes())
    return result


def version(directory: Path) -> str | None:
    text = (directory / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1] if text.startswith("---") else ""
    match = re.search(r'^\s+version:\s*[\"\']?([^\"\'\s]+)', frontmatter, re.M)
    return match.group(1) if match else None


def package_names(root: Path) -> set[str]:
    return {p.parent.name for p in root.glob("*/SKILL.md") if p.is_file()}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def refresh(root: Path) -> None:
    """Recalculate integrity metadata after intentional source edits, without claiming validation."""
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_names = manifest.setdefault("initial_import_skill_names", sorted(manifest["skills"]))
    entries = {}
    for name in sorted(package_names(root)):
        directory = root / name
        v = version(directory)
        local_manifest = directory / "MANIFEST.json"
        if v is not None or local_manifest.exists():
            old = json.loads(local_manifest.read_text(encoding="utf-8")) if local_manifest.exists() else {}
            data = dict(old)
            files = {k: h for k, h in file_hashes(directory).items() if k != "MANIFEST.json"}
            data.update(schema_version=1, name=name, version=v, files=files, content_sha256=content_hash(files))
            if not old:
                data["content_hash_encoding"] = "UTF-8 canonical JSON of files, sorted keys, compact separators"
                data["purpose"] = "Portable documentation integrity; no installation or runtime role."
            if data != old:
                save_json(local_manifest, data)
        files = file_hashes(directory)
        entry = dict(manifest["skills"].get(name, {}))
        entry.update(version=v, files=files, content_sha256=content_hash(files))
        entry.setdefault("origin", "existing_user_installation" if name in initial_names else "project_authored")
        entries[name] = entry
    manifest["skills"] = entries
    manifest["recorded_at"] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path, manifest)


def compare(label: str, expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    errors = []
    for path in sorted(expected.keys() - actual.keys()):
        errors.append(f"{label}: missing {path}")
    for path in sorted(actual.keys() - expected.keys()):
        errors.append(f"{label}: unrecorded {path}")
    for path in sorted(expected.keys() & actual.keys()):
        if expected[path] != actual[path]:
            errors.append(f"{label}: changed {path}")
    return errors


def check(root: Path, installed_root: Path | None) -> tuple[list[str], int, int]:
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["skills"]
    if any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) for name in entries):
        raise ValueError("Invalid Skill directory name in manifest")
    errors = []
    names = package_names(root)
    if names != set(entries):
        errors.append(f"Skill inventory differs: missing={sorted(set(entries)-names)}, unrecorded={sorted(names-set(entries))}")
    total = 0
    for name, entry in entries.items():
        directory = root / name
        files = file_hashes(directory)
        total += len(files)
        errors.extend(compare(f"source/{name}", entry["files"], files))
        if content_hash(entry["files"]) != entry["content_sha256"]:
            errors.append(f"source/{name}: invalid inventory digest")
        if (directory / "SKILL.md").is_file() and version(directory) != entry["version"]:
            errors.append(f"source/{name}: version differs from SKILL.md")
        local_manifest = directory / "MANIFEST.json"
        if local_manifest.is_file():
            data = json.loads(local_manifest.read_text(encoding="utf-8"))
            payload = {k: h for k, h in files.items() if k != "MANIFEST.json"}
            errors.extend(compare(f"package/{name}", data["files"], payload))
            if data.get("name") != name or data.get("version") != entry["version"]:
                errors.append(f"package/{name}: identity or version mismatch")
            if content_hash(data["files"]) != data["content_sha256"]:
                errors.append(f"package/{name}: invalid package digest")
        if installed_root is not None:
            errors.extend(compare(f"installed/{name}", entry["files"], file_hashes(installed_root / name)))
    return errors, len(entries), total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--installed-root", type=Path, help="Optional read-only comparison with an installed/restored copy")
    parser.add_argument("--refresh-manifests", action="store_true", help="Rebuild derived hashes after reviewed source edits; does not edit content or install")
    args = parser.parse_args()
    try:
        if args.refresh_manifests:
            refresh(args.source_root)
        errors, count, files = check(args.source_root, args.installed_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    if errors:
        print("FAIL:\n" + "\n".join(errors))
        return 1
    comparison = "; installed/restored copy matches" if args.installed_root is not None else ""
    print(f"PASS: {count} Skills, {files} portable files, manifests and versions{comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
