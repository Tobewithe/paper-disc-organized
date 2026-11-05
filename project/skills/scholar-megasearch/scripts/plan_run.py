#!/usr/bin/env python3
"""Build a scholar-megasearch run plan from topic, field, goal, and depth.

Use interactively when the user has not specified enough search intent:

    python3 plan_run.py "graph neural networks" --field cs-ml --goal survey --depth 3

If field, goal, or depth is omitted and stdin is interactive, this script asks for
them. It emits a JSON plan that the agent can pass into the fan-out workflow.
"""
import argparse
import json
import re
import sys


DEPTH = {
    1: {"label": "L1 Quick", "facets": 3, "buckets": 4, "cap": 15, "pdfs": 10, "waves": ["initial"]},
    2: {"label": "L2 Standard", "facets": 5, "buckets": 5, "cap": 25, "pdfs": 30, "waves": ["initial"]},
    3: {"label": "L3 Deep", "facets": 6, "buckets": 6, "cap": 30, "pdfs": 50, "waves": ["initial", "snowball"]},
    4: {"label": "L4 Exhaustive", "facets": 8, "buckets": 7, "cap": 40, "pdfs": 100, "waves": ["initial", "snowball", "critic"]},
    5: {"label": "L5 Total", "facets": 8, "buckets": 7, "cap": 40, "pdfs": "all", "waves": ["initial", "snowball", "critic-loop"]},
}

FIELDS = {
    "cs-ml": ["A", "B", "F", "C", "G", "E", "D"],
    "biomed": ["D", "B", "C", "E", "G", "A", "F"],
    "physics": ["A", "B", "C", "E", "G", "F", "D"],
    "chem-materials": ["B", "C", "D", "E", "A", "G", "F"],
    "crypto-security": ["A", "F", "B", "G", "C", "E", "D"],
    "econ-social-law": ["F", "B", "C", "G", "E", "D", "A"],
    "math": ["A", "B", "C", "F", "G", "E", "D"],
    "interdisciplinary": ["A", "B", "C", "D", "E", "G", "F"],
}

GOALS = {
    "survey": "balanced survey / literature map",
    "systematic": "high-recall systematic-review corpus",
    "newest": "recent frontier and latest papers",
    "seminal": "canonical and highly cited works",
    "implementation": "methods, code, datasets, and reproducibility",
    "pdf-corpus": "maximize downloadable full text",
}


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s or "literature-search"


def ask_choice(label, choices, default):
    print(f"\n{label}:")
    for i, item in enumerate(choices, 1):
        desc = GOALS.get(item, "")
        suffix = f" - {desc}" if desc else ""
        print(f"  {i}. {item}{suffix}")
    raw = input(f"Choose 1-{len(choices)} or type a value [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1]
    return raw


def ask_depth(default=2):
    while True:
        raw = input(f"\nDepth 1-5 [{default}]: ").strip()
        if not raw:
            return default
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Enter a numeric depth from 1 to 5.")


def facets_for(topic, goal, count):
    base = [
        topic,
        f"{topic} review survey",
        f"{topic} methods",
        f"{topic} applications",
        f"{topic} benchmark dataset",
        f"{topic} limitations open problems",
        f"{topic} recent advances",
        f"{topic} key authors",
    ]
    extras = {
        "systematic": [f"{topic} systematic review", f"{topic} meta analysis"],
        "newest": [f"{topic} 2025 2026", f"{topic} latest arxiv"],
        "seminal": [f"{topic} highly cited", f"{topic} foundational"],
        "implementation": [f"{topic} github code", f"{topic} reproducibility benchmark"],
        "pdf-corpus": [f"{topic} open access pdf", f"{topic} full text"],
    }
    ordered = []
    for item in base + extras.get(goal, []):
        if item not in ordered:
            ordered.append(item)
    return ordered[:count]


def build_plan(topic, field, goal, depth):
    settings = DEPTH[depth]
    field = field if field in FIELDS else "interdisciplinary"
    goal = goal if goal in GOALS else "survey"
    buckets = FIELDS[field][:settings["buckets"]]
    return {
        "topic": topic,
        "slug": slug(topic),
        "field": field,
        "goal": goal,
        "depth": depth,
        "depth_label": settings["label"],
        "settings": settings,
        "buckets": buckets,
        "facets": facets_for(topic, goal, settings["facets"]),
        "recovery_policy": [
            "try preferred MCP tools first",
            "try alternate MCP tools within the bucket",
            "fall back to resilient_search.py for arxiv/semanticscholar/ddg where applicable",
            "write status for failed or empty sources and continue the run",
        ],
        "merge_args": ["--ranking", "layered", "--goal", goal, "--topic", topic],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic", nargs="?")
    ap.add_argument("--field", choices=sorted(FIELDS))
    ap.add_argument("--goal", choices=sorted(GOALS))
    ap.add_argument("--depth", type=int, choices=range(1, 6), metavar="1-5")
    ap.add_argument("-o", "--out")
    ap.add_argument("--no-input", action="store_true", help="fail instead of prompting for missing values")
    args = ap.parse_args()

    topic = args.topic
    if not topic and not args.no_input and sys.stdin.isatty():
        topic = input("Topic: ").strip()
    if not topic:
        sys.exit("topic is required")

    field = args.field
    goal = args.goal
    depth = args.depth
    if not args.no_input and sys.stdin.isatty():
        if not field:
            field = ask_choice("Field", sorted(FIELDS), "interdisciplinary")
        if not goal:
            goal = ask_choice("Goal", sorted(GOALS), "survey")
        if not depth:
            depth = ask_depth(2)
    if not field or not goal or not depth:
        sys.exit("--field, --goal, and --depth are required in non-interactive mode")
    depth = int(depth)

    payload = json.dumps(build_plan(topic, field, goal, depth), indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"wrote plan -> {args.out}", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
