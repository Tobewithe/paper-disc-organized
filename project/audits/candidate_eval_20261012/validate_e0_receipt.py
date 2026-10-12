#!/usr/bin/env python3
"""Fail-closed audit of E0 cross-objective evaluation receipts.

Read-only for original runs. Compare ONLY iou_A against the historic official
per-candidate table. Historic official iou_B is a different intervention and
must not be compared with the E0 B/C columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED = {"fit": (796, 6058), "dev": (197, 1402), "val": (196, 1346)}
TOL_IOU = 1e-6
GRADIENT_LIMIT = 1e-6
FIELDS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(row: dict):
    missing = [k for k in FIELDS if k not in row]
    if missing:
        raise ValueError(f"missing identity fields: {missing}")
    return (
        str(row["split"]), int(row["image_id"]), int(row["annotation_id"]),
        str(row["branch"]), int(row["raw_id"]), int(row["pyramid_level"]),
    )


def rows_by_identity(path: Path, require_arms: bool):
    rows = {}
    by_split = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = identity(row)
            if key in rows:
                raise ValueError(f"{path} duplicate identity on line {line_no}: {key}")
            if key[0] not in EXPECTED:
                raise ValueError(f"unknown split {key[0]} in {path}")
            if key[3] != "one2one":
                raise ValueError(f"wrong branch {key[3]} in {path}")
            cols = ("iou_A", "iou_B", "iou_C") if require_arms else ("iou_A",)
            for col in cols:
                if col not in row:
                    raise ValueError(f"{path}:{line_no} missing {col}")
                value = float(row[col])
                if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                    raise ValueError(f"{path}:{line_no} invalid {col}={value}")
            rows[key] = row
            by_split[key[0]].append(row)
    return rows, by_split


def check(name: str, func, checks: dict, errors: list):
    try:
        result = func()
        checks[name] = {"passed": True, "detail": result}
        return result
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        checks[name] = {"passed": False, "error": reason}
        errors.append(f"{name}: {reason}")
        return None


def require(condition: bool, reason: str):
    if not condition:
        raise ValueError(reason)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--official-per-candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    checks, errors = {}, []
    eval_dir = args.eval_dir
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "gate": "E0_BASELINE_AND_IDENTITY", "checks": checks,
            "sources": {"eval_dir": str(eval_dir), "reference": str(args.official_per_candidate)}}

    def load_core():
        run = load_json(eval_dir / "run.json")
        summary = load_json(eval_dir / "SUMMARY.json")
        require(run.get("status") == "complete", "E0 evaluator run is not complete")
        maps = run.get("map_validation", {})
        for mode in ("old160", "official640"):
            record = maps.get(mode, {})
            require(record.get("converged") is True, f"{mode} lacks convergence receipt")
            g = float(record.get("gradient_inf_max", float("nan")))
            require(math.isfinite(g) and g < GRADIENT_LIMIT,
                    f"{mode} gradient_inf={g} does not meet strict <1e-6")
        require("tables" in summary, "missing E0 tables")
        return run, summary

    core = check("G0_eval_complete_and_converged", load_core, checks, errors)

    def load_e0():
        path = eval_dir / "PER_CANDIDATE.jsonl"
        rows, split_rows = rows_by_identity(path, require_arms=True)
        for split, (images, candidates) in EXPECTED.items():
            block = split_rows.get(split, [])
            nimages = len({int(r["image_id"]) for r in block})
            require(len(block) == candidates,
                    f"{split}: candidates {len(block)} != {candidates}")
            require(nimages == images, f"{split}: images {nimages} != {images}")
        return rows, split_rows

    e0 = check("G1_e0_identity_counts", load_e0, checks, errors)

    def parity():
        require(e0 is not None, "E0 rows unavailable")
        ref, _ = rows_by_identity(args.official_per_candidate, require_arms=False)
        current = e0[0]
        missing = sorted(set(current) - set(ref))
        extra = sorted(set(ref) - set(current))
        require(not missing and not extra,
                f"identity mismatch: missing_in_reference={len(missing)}, extra_in_reference={len(extra)};"
                f" first_missing={missing[:2]}, first_extra={extra[:2]}")
        differences = [(k, abs(float(r["iou_A"]) - float(ref[k]["iou_A"])))
                       for k, r in current.items()]
        bad = sorted(((k, v) for k, v in differences if v > TOL_IOU),
                     key=lambda item: item[1], reverse=True)
        require(not bad, f"native baseline disagreement {len(bad)} / {len(current)};"
                f" worst={bad[:3]}")
        return {"identities": len(current), "max_abs_iou_A_error":
                max((v for _, v in differences), default=0),
                "reference_sha256": digest(args.official_per_candidate),
                "native_only_comparison": True}

    check("G2_official_native_A_parity", parity, checks, errors)

    def summary_parity():
        require(core is not None and e0 is not None, "summary or rows unavailable")
        summary = core[1]
        for split, (images, candidates) in EXPECTED.items():
            values = e0[1][split]
            rec = summary["tables"][split]["iou"]
            require(rec["n_images"] == images and rec["n_candidates"] == candidates,
                    f"{split}: SUMMARY count mismatch")
            by_img = defaultdict(list)
            for row in values:
                by_img[int(row["image_id"])].append(row)
            means = {}
            for arm in "ABC":
                image_mean = [sum(float(x[f"iou_{arm}"]) for x in rows) / len(rows)
                              for rows in by_img.values()]
                means[arm] = sum(image_mean) / len(image_mean)
                actual = float(rec["image_macro"][arm])
                require(abs(means[arm] - actual) <= 1e-9,
                        f"{split} image_macro {arm}: {means[arm]} != {actual}")
            for a, b, key in (("B", "A", "B_minus_A"), ("C", "A", "C_minus_A"),
                              ("C", "B", "C_minus_B")):
                actual = float(rec["image_macro"][key]["delta"])
                require(abs((means[a] - means[b]) - actual) <= 1e-9,
                        f"{split}: delta {key} disagrees with per-candidate records")
        return {"confirmed_splits": list(EXPECTED), "image_macro_and_point_deltas_recomputed": True,
                "ci_recomputed": False}

    check("G3_summary_points", summary_parity, checks, errors)
    for filename in ("run.json", "SUMMARY.json", "PER_CANDIDATE.jsonl"):
        f = eval_dir / filename
        if f.is_file():
            data["sources"][filename + "_sha256"] = digest(f)
    data.update({"passed": not errors, "status": "valid_for_E0_adjudication" if not errors
                 else "insufficient_inputs", "errors": errors})
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": data["status"], "out": str(out),
                      "failed_checks": list(k for k, v in checks.items() if not v["passed"])},
                     ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
