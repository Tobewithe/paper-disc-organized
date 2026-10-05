"""Bootstrap uncertainty for the existing YOLO26 diagnostic cache.

This is a read-only audit: it does not run inference or alter predictions.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


KEYS = ("total_failure", "relation_failure", "miss")
DATASETS = ("PigLife_public_test", "FaroPigSeg_test", "BamaPig2D_eval")
SCENES = ("Isolated", "Near/Dense", "Touching")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_rate(rows: list[dict[str, str]], key: str, draws: int, seed: int) -> tuple[float, float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(int(row[key]))
    numerators = np.asarray([sum(values) for values in grouped.values()], dtype=float)
    denominators = np.asarray([len(values) for values in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerators), size=(draws, len(numerators)))
    rates = numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    point = float(numerators.sum() / denominators.sum())
    low, high = np.quantile(rates, [0.025, 0.975])
    return point, float(low), float(high)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--nms-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=3000)
    args = parser.parse_args()

    rows = read_rows(args.gt_csv)
    lines = [
        "# YOLO26 mechanism-gate uncertainty audit",
        "",
        "Read-only bootstrap audit of the existing fixed cache; no model forward, training, or GT edits.",
        "Bootstrap unit is image (all GT rows from a sampled image stay together). Seed: 20260905.",
        "",
        "## GT-level rates (95% image-cluster bootstrap CI)",
        "",
        "| Scope | N GT | Total failure | Relation failure | Miss |",
        "|---|---:|---:|---:|---:|",
    ]
    scopes = [("ALL", rows)]
    scopes += [(dataset, [row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS]
    for name, scoped in scopes:
        values = [bootstrap_rate(scoped, key, args.draws, 20260905 + index) for index, key in enumerate(KEYS)]
        formatted = [f"{pct(point)} [{pct(low)}, {pct(high)}]" for point, low, high in values]
        lines.append(f"| {name} | {len(scoped)} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")

    lines += [
        "",
        "## Scene-stratified total failure",
        "",
        "| Dataset | Scene | N GT | Rate (95% CI) |",
        "|---|---|---:|---:|",
    ]
    for dataset in DATASETS:
        for scene in SCENES:
            scoped = [row for row in rows if row["dataset"] == dataset and row["scene_label"] == scene]
            if not scoped:
                continue
            point, low, high = bootstrap_rate(scoped, "total_failure", max(1000, args.draws // 2), 20260906)
            lines.append(f"| {dataset} | {scene} | {len(scoped)} | {pct(point)} [{pct(low)}, {pct(high)}] |")

    with args.nms_csv.open(encoding="utf-8-sig", newline="") as handle:
        nms_rows = list(csv.DictReader(handle))
    lines += [
        "",
        "## Existing NMS threshold sensitivity",
        "",
        "The following values are copied from the cache-only comparison; they are not re-estimated here.",
        "",
        "| Method | Threshold | Total failure | Relation failure | O -> C | C -> failure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in nms_rows:
        if row["method"] not in {"BASELINE", "MASK_NMS", "SOURCE_AWARE"}:
            continue
        threshold = row["threshold"] or "-"
        lines.append(
            f"| {row['method']} | {threshold} | {pct(float(row['total_failure_rate']))} | "
            f"{pct(float(row['relation_failure_rate']))} | {row['O_to_C']} | {row['C_to_failure']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The intervals quantify uncertainty for this fixed checkpoint/cache and do not establish causality or a deployable postprocessor. The threshold scan remains a control: it shows duplicate suppression changes the error mix, but does not identify the mechanism responsible for residual low-overlap failures.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
