#!/usr/bin/env python3
"""Read-only sanity check for active research routing, linked docs, and Study status.

Run from any directory: python project/check_active_plan.py
This intentionally does NOT evaluate experiments, contact servers, or inspect old Run validity.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CORE = (
    "shared/evaluation/POSTTRAIN_TOOLS.md",
    "experiments/mask_readout_mechanism_v2_20261012/MECHANISM_AND_DEFAULTS.md",
    "experiments/mask_readout_mechanism_v2_20261012/EXECUTION_CONTRACT.md",
    "project/科研项目通用规范/运行登记与本地协同.md",
    "README.md",
    "AGENTS.md",
    "project/ACTIVE_PLAN.md",
    "project/科研项目通用规范/README.md",
    "project/科研项目通用规范/候选身份与评估口径.md",
    "project/科研项目通用规范/指标最小化与TAL分类.md",
    "project/科研项目通用规范/训练规模与结论边界.md",
    "paper/manuscripts/neighbor_sensitive_p3/RESEARCH_RULES.md",
    "experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md",
    "experiments/mask_readout_mechanism_v2_20261012/README.md",
    "experiments/mask_readout_mechanism_v2_20261012/LOCAL_AGENT_START_HERE.md",
    "experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md",
)
HISTORICAL = (
    "project/hypotheses/ACTIVE_FAILURE_ROUTE.md",
    "project/research_notes/PROTOTYPE_COEFFICIENT_ACTIVE_GOAL_20261003.md",
    "project/research_notes/MAINLINE_STATUS_20260930.md",
    "project/research_notes/PROTOTYPE_COEFFICIENT_BREAKTHROUGH_20261005.md",
    "paper/manuscripts/neighbor_sensitive_p3/DEVELOPMENT_PLAN.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")


def check() -> tuple[list[str], int]:
    problems: list[str] = []
    links_checked = 0
    for rel in CORE:
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
            if not target.is_relative_to(ROOT):
                problems.append(f"link escapes repository: {rel} -> {local}")
            elif not target.exists():
                problems.append(f"missing active link: {rel} -> {local}")
    for rel in HISTORICAL:
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"historical pointer missing: {rel}")
        elif "历史导航提示（2026-10-13）" not in path.read_text(encoding="utf-8")[:1000]:
            problems.append(f"historical-plan disclaimer missing: {rel}")
    method_path = ROOT / "experiments/mask_readout_mechanism_v2_20261012/study.json"
    baseline_path = ROOT / "experiments/official_gt_failure_rebaseline_20261013/study.json"
    if method_path.is_file():
        method = json.loads(method_path.read_text(encoding="utf-8"))
        if method.get("protocol") != "POSTTRAIN_RESCOPE_20261013.md":
            problems.append("method Study no longer points at the post-training protocol")
        if "E0" in str(method.get("current_stage", "")):
            problems.append("method Study wrongly routes to historical TAL/E0 prerequisite")
        if method.get("entrypoint") != "LOCAL_AGENT_START_HERE.md":
            problems.append("method Study local entrypoint is not canonical")
    else:
        problems.append("missing method study.json")
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if "PROTOCOL.md" not in baseline.get("record_paths", []):
            problems.append("baseline Study has no referenced GT-ledger protocol")
    else:
        problems.append("missing baseline study.json")
    for rel in ("README.md", "AGENTS.md"):
        path = ROOT / rel
        if path.is_file() and "project/ACTIVE_PLAN.md" not in path.read_text(encoding="utf-8"):
            problems.append(f"no canonical plan in {rel}")
    catalog_path = ROOT / "project/PLAN_CATALOG.json"
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
    if method_path.is_file() and "EXECUTION_CONTRACT.md" not in method.get("protocol_companions", []):
        problems.append("method Study must reference its execution contract")
    return problems, links_checked


def main() -> int:
    errors, nlinks = check()
    if errors:
        print(f"FAIL: {len(errors)} governance issue(s), {nlinks} active links inspected")
        for error in errors:
            print(f" - {error}")
        return 1
    print(f"PASS: {len(CORE)} active docs, {nlinks} local links, "
          f"{len(HISTORICAL)} archived-plan warnings, Study routing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
