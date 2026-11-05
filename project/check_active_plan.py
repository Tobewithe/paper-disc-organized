#!/usr/bin/env python3
"""Read-only check for AGENTS -> guides -> active plan and local links.

Run from any directory: python project/check_active_plan.py
This does not evaluate experiments, contact servers, or bind routing to a historical Study.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CORE = (
    "README.md",
    "AGENTS.md",
    "project/WORKBENCH_GUIDE.md",
    "project/科研项目通用规范/通用规范.md",
    "project/ACTIVE_PLAN.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")


def check() -> tuple[list[str], int]:
    problems: list[str] = []
    links_checked = 0
    linked_paths: dict[str, set[Path]] = {}
    for rel in CORE:
        linked_paths[rel] = set()
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"missing active document: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            dest = match.group(1).strip().split(" ", 1)[0]
            if dest.startswith(("https://", "http://", "#", "mailto:")):
                continue
            local = unquote(dest.split("#", 1)[0].split("?", 1)[0])
            if not local:
                continue
            links_checked += 1
            target = (path.parent / local).resolve()
            linked_paths[rel].add(target)
            if not target.is_relative_to(ROOT):
                problems.append(f"link escapes repository: {rel} -> {local}")
            elif not target.exists():
                problems.append(f"missing active link: {rel} -> {local}")
    for rel, required in (
        ("AGENTS.md", "project/WORKBENCH_GUIDE.md"),
        ("AGENTS.md", "project/科研项目通用规范/通用规范.md"),
        ("project/WORKBENCH_GUIDE.md", "project/ACTIVE_PLAN.md"),
    ):
        if (ROOT / rel).is_file() and (ROOT / required).resolve() not in linked_paths[rel]:
            problems.append(f"missing reading entry: {rel} -> {required}")
    catalog_path = ROOT / "project/research_notes/maintenance/navigation/PLAN_CATALOG.json"
    if not catalog_path.is_file():
        problems.append("missing plan catalog")
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("canonical_route") != "project/ACTIVE_PLAN.md":
            problems.append("catalog does not point at the canonical route")
        for entry in catalog.get("entries", []):
            target = (ROOT / entry["path"]).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                problems.append(f"invalid catalog path: {entry['path']}")
    return problems, links_checked


def main() -> int:
    errors, nlinks = check()
    if errors:
        print(f"FAIL: {len(errors)} governance issue(s), {nlinks} active links inspected")
        for error in errors:
            print(f" - {error}")
        return 1
    print(f"PASS: {len(CORE)} active docs, {nlinks} local links, current routing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
