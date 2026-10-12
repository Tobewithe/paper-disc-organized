#!/usr/bin/env python3
"""Reclassify GTs on an EXISTING official TAL cache; no forward or training.

Requires cache/INDEX.json, cache/images/{image_id:012d}.pt with:
  all_annotation_ids: every GT annotation_id retained by official loader;
  rows: one per TAL positive with annotation_id, raw_id, level,
        box_iou, initial_iou (normal-mask vs original COCO GT).

The input's existing initial_iou must have been validated in its original Run.
This script does not prove its equivalence to a new evaluator. VAL ONLY by
default because the official cache computed initial_iou on val positions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path


STATES = (
    "unassigned", "joint_good", "box_only",
    "mask_only", "separate_good", "neither",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state(rows: list[dict], box_threshold: float = 0.75,
          mask_threshold: float = 0.75) -> str:
    if not rows:
        return "unassigned"
    has_box = any(float(r["box_iou"]) >= box_threshold for r in rows)
    has_mask = any(float(r["mask_iou"]) >= mask_threshold for r in rows)
    joint = any(float(r["box_iou"]) >= box_threshold and
                float(r["mask_iou"]) >= mask_threshold for r in rows)
    if joint:
        return "joint_good"
    if has_box and has_mask:
        return "separate_good"
    if has_box:
        return "box_only"
    if has_mask:
        return "mask_only"
    return "neither"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_task_outcomes(path: Path):
    """Optional GT table from a separately verified *normal-output* COCOeval.

    Mandatory columns: image_id, annotation_id, state (success/failure/
    ignored/unknown). It is not generated from TAL or raw-oracle flags here.
    """
    if path is None:
        return {}
    outcomes = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            k = (int(obj["image_id"]), int(obj["annotation_id"]))
            require(k not in outcomes, f"duplicated output GT {k}, line {n}")
            result = str(obj["state"])
            require(result in ("success", "failure", "ignored", "unknown"),
                    f"invalid output GT state {result}")
            outcomes[k] = result
    return outcomes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="val", choices=["val"])
    p.add_argument("--branch", default="one2one", choices=["one2one"])
    p.add_argument("--task-outcomes", type=Path, default=None,
                   help="optional verified COCOeval GT-state JSONL; never derived from TAL")
    args = p.parse_args(argv)
    import torch  # only when actually reading trusted local torch caches

    index_file = args.cache / "INDEX.json"
    index = json.loads(index_file.read_text(encoding="utf-8"))
    require(isinstance(index.get("val"), list), "cache has no val index")
    targets = index["val"]
    outcomes = load_task_outcomes(args.task_outcomes)
    results = []
    candidate_rows = []
    all_ids = set()
    for item in targets:
        iid = int(item["image_id"])
        require(iid not in all_ids, f"duplicate val image_id {iid}")
        all_ids.add(iid)
        path = args.cache / "images" / f"{iid:012d}.pt"
        require(path.exists(), f"missing {path}")
        x = torch.load(path, map_location="cpu", weights_only=False)
        require("all_annotation_ids" in x and isinstance(x["all_annotation_ids"], (list,tuple)),
                f"{iid} no official all_annotation_ids: cannot construct full GT denominator")
        gts = [int(v) for v in x["all_annotation_ids"]]
        require(len(set(gts)) == len(gts), f"{iid} duplicate all_annotation_ids")
        grouped = collections.defaultdict(list)
        seen_raw = set()
        for r in x["rows"]:
            gid = int(r["annotation_id"])
            require(gid in gts, f"{iid}: TAL row annotation {gid} not in official kept GT list")
            rid = int(r["raw_id"])
            require(rid not in seen_raw, f"{iid}: duplicated positive raw_id {rid}")
            seen_raw.add(rid)
            level = int(r.get("level", r.get("pyramid_level", -1)))
            require(level in (0, 1, 2), f"{iid}: invalid pyramid level {level}")
            if "branch" in r:
                require(str(r["branch"]) == args.branch, f"{iid} wrong cached branch")
            require("initial_iou" in r, f"{iid}:{rid} no initial_iou; don't use loss as proxy")
            biou, miou = float(r["box_iou"]), float(r["initial_iou"])
            require(all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (biou, miou)),
                    f"{iid}:{rid} invalid Box/Mask IoU")
            row = {"image_id": iid, "annotation_id": gid, "branch": args.branch,
                   "raw_id": rid, "pyramid_level": level, "box_iou": biou,
                   "mask_iou": miou}
            grouped[gid].append(row)
            candidate_rows.append(row)
        for gid in gts:
            rr = grouped.get(gid, [])
            label = state(rr)
            key = (iid, gid)
            obs = outcomes.get(key, "unknown")
            results.append({
                "image_id": iid, "annotation_id": gid, "branch": args.branch,
                "n_tal": len(rr), "tal_state": label,
                "output_state": obs, "output_state_source": "external_output_table" if
                    key in outcomes else "not_provided",
                "tal_raw_ids": [r["raw_id"] for r in rr],
                "best_tal_box_iou": max((r["box_iou"] for r in rr), default=None),
                "best_tal_mask_iou": max((r["mask_iou"] for r in rr), default=None),
            })
    require(len({(r["image_id"],r["annotation_id"]) for r in results}) == len(results),
            "duplicate output GT key")
    total_positive = sum(r["n_tal"] for r in results)
    require(total_positive == len(candidate_rows),
            "candidate count disagreement between GT and raw ledger")
    unknown_outcomes = set(outcomes) - {(r["image_id"],r["annotation_id"]) for r in results}
    require(not unknown_outcomes,
            f"task outcomes include GT not in this TAL cohort: {len(unknown_outcomes)}")

    # Create outputs only after all identity gates pass; no partial success.
    args.out.mkdir(parents=True, exist_ok=True)
    def dump_json(name, obj):
        (args.out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    def dump_lines(name, rows):
        with (args.out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_lines("PER_GT_TAL_STATE.jsonl", results)
    dump_lines("PER_TAL_CANDIDATE.jsonl", candidate_rows)
    status = collections.Counter(r["tal_state"] for r in results)
    cross = collections.Counter(
        (r["output_state"], r["tal_state"]) for r in results)
    summary = {
        "status": "completed", "scope": "official TAL positive positions, not deployment AP",
        "data_source": "cached original Mask IoU and cached official annotation ids",
        "cocoeval_task_state": "externally_provided" if args.task_outcomes else "not_provided",
        "thresholds": {"box": 0.75, "mask": 0.75},
        "split": args.split, "branch": args.branch,
        "n_images": len(targets), "n_official_gt": len(results),
        "n_tal_positive_rows": len(candidate_rows),
        "tal_state_counts": {k: status[k] for k in STATES},
        "cross_tab": [{"output_state": a, "tal_state": b, "n_gt": n}
                      for (a,b),n in sorted(cross.items())],
        "identity_gates_passed": True,
        "interpretation_limit": "frozen checkpoint label assignment does not reconstruct training history",
    }
    require(sum(status.values()) == len(results), "taxonomy not exhaustive")
    dump_json("SUMMARY.json", summary)
    run = {"kind": "gt_tal_taxonomy_from_existing_cache", "status": "completed",
           "index_sha256": sha256(index_file), "cache": str(args.cache.resolve()),
           "task_outcomes_sha256": sha256(args.task_outcomes) if args.task_outcomes else None,
           "summary": "SUMMARY.json", "source_used_gt": True,
           "source_used_existing_mask_iou": True, "model_forward_executed": False,
           "training_executed": False}
    dump_json("run.json", run)
    print(json.dumps({"n_gt": len(results), "n_tal_positive": len(candidate_rows),
                      "counts": summary["tal_state_counts"], "out": str(args.out)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
