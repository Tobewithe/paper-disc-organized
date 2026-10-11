#!/usr/bin/env python3
"""Apply the frozen cross-audit decision rule to an evaluator output.

The evaluator has already produced one fixed-candidate table for arms A
(native), B (old160 map), and C (official640 map).  This script only reads
those outputs and writes a machine-readable decision plus a short Chinese
brief.  Selection is deliberately frozen from the dev split before the val
split is inspected.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any


RULE_VERSION = "cross_audit_decision_addendum_2026-10-11_v1"
GRADIENT_LIMIT = 1e-6
EXPECTED = {
    "fit": {"n_candidates": 6058, "n_images": 796},
    "dev": {"n_candidates": 1402, "n_images": 197},
    "val": {"n_candidates": 1346, "n_images": 196},
}
ARMS = ("A", "B", "C")


class InputError(ValueError):
    """The evaluator output is not at the registered adjudication state."""


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"missing required file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error text is user-facing
        raise InputError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"{path.name} must contain a JSON object")
    return value


def _map_gradients(run: dict[str, Any]) -> dict[str, float]:
    mv = run.get("map_validation")
    if not isinstance(mv, dict):
        raise InputError("run.json lacks map_validation")
    out: dict[str, float] = {}
    for name in ("old160", "official640"):
        item = mv.get(name)
        if not isinstance(item, dict):
            raise InputError(f"map_validation lacks {name}")
        if item.get("converged") is not True:
            raise InputError(f"{name} map is not marked converged")
        raw = item.get("gradient_inf_max", item.get("gradient_inf"))
        if isinstance(raw, (list, tuple, dict)):
            vals: list[float] = []
            if isinstance(raw, dict):
                raw = raw.values()
            for v in raw:
                if not _finite(v):
                    raise InputError(f"{name} gradient is nonfinite")
                vals.append(abs(float(v)))
            if not vals:
                raise InputError(f"{name} gradient is empty")
            value = max(vals)
        else:
            if not _finite(raw):
                raise InputError(f"{name} gradient is nonfinite or missing")
            value = abs(float(raw))
        if not value < GRADIENT_LIMIT:
            raise InputError(f"{name} gradient {value:.8g} is not < {GRADIENT_LIMIT:g}")
        out[name] = value
    return out


def _table(summary: dict[str, Any], split: str) -> dict[str, Any]:
    tables = summary.get("tables")
    if not isinstance(tables, dict) or split not in tables:
        raise InputError(f"SUMMARY.json lacks tables.{split}")
    table = tables[split]
    if not isinstance(table, dict) or not isinstance(table.get("iou"), dict):
        raise InputError(f"SUMMARY.json lacks tables.{split}.iou")
    return table


def _counts(summary: dict[str, Any], split: str) -> dict[str, int]:
    metric = _table(summary, split)["iou"]
    try:
        got = {"n_candidates": int(metric["n_candidates"]), "n_images": int(metric["n_images"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"tables.{split}.iou has no valid n_candidates/n_images") from exc
    if got != EXPECTED[split]:
        raise InputError(f"{split} counts {got} do not equal {EXPECTED[split]}")
    return got


def _contrast(table: dict[str, Any], label: str) -> dict[str, Any]:
    metric = table["iou"]
    value = metric.get("image_macro", {}).get(label)
    if not isinstance(value, dict):
        raise InputError(f"missing iou.image_macro.{label}")
    point = value.get("delta")
    ci = value.get("ci95")
    if not _finite(point) or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        raise InputError(f"invalid contrast record {label}")
    if not all(_finite(x) for x in ci):
        raise InputError(f"nonfinite CI in contrast record {label}")
    low, high = float(ci[0]), float(ci[1])
    if low > high:
        raise InputError(f"reversed CI in contrast record {label}")
    return {"delta": float(point), "ci95": [low, high], "scope": value.get("scope")}


def _candidate_identity_check(eval_dir: Path) -> dict[str, int]:
    path = eval_dir / "PER_CANDIDATE.jsonl"
    if not path.is_file():
        raise InputError("missing PER_CANDIDATE.jsonl")
    seen: set[tuple[Any, ...]] = set()
    counts = {split: 0 for split in EXPECTED}
    images: dict[str, set[int]] = {split: set() for split in EXPECTED}
    required = {"split", "image_id", "annotation_id", "raw_id", "iou_A", "iou_B", "iou_C"}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise InputError(f"invalid PER_CANDIDATE.jsonl line {line_no}: {exc}") from exc
            if not isinstance(row, dict) or not required.issubset(row):
                raise InputError(f"candidate row {line_no} lacks fixed identity/arm fields")
            split = str(row["split"])
            if split not in EXPECTED:
                raise InputError(f"unknown candidate split {split!r}")
            try:
                key = (split, int(row["image_id"]), int(row["annotation_id"]), int(row["raw_id"]), int(row.get("pyramid_level", row.get("level", -1))))
            except (TypeError, ValueError) as exc:
                raise InputError(f"invalid candidate identity on line {line_no}") from exc
            if key in seen:
                raise InputError(f"duplicate candidate identity {key}")
            seen.add(key)
            counts[split] += 1
            images[split].add(key[1])
            if not all(_finite(row[k]) for k in ("iou_A", "iou_B", "iou_C")):
                raise InputError(f"nonfinite candidate IoU on line {line_no}")
    for split, expected in EXPECTED.items():
        got = {"n_candidates": counts[split], "n_images": len(images[split])}
        if got != expected:
            raise InputError(f"PER_CANDIDATE {split} counts {got} do not equal {expected}")
    return counts


def _direct_bundle(summary: dict[str, Any]) -> dict[str, Any] | None:
    dev = _contrast(_table(summary, "dev"), "C_minus_B")
    val = _contrast(_table(summary, "val"), "C_minus_B")
    if dev["ci95"][0] > 0 and val["ci95"][0] > 0:
        return {"winner": "C", "dev": dev, "val": val, "direction": "C_minus_B_positive"}
    if dev["ci95"][1] < 0 and val["ci95"][1] < 0:
        return {"winner": "B", "dev": dev, "val": val, "direction": "C_minus_B_negative"}
    return None


def _brief(decision: dict[str, Any]) -> str:
    status = decision["status"]
    if status == "insufficient_inputs":
        return (
            "当前输入未达到交叉审计判定状态；唯一下一步是完成当前交叉审计。\n"
            f"阻塞：{decision.get('blocking_reason', 'unknown')}。"
        )
    winner = decision["winner"]
    label = "old160 (B)" if winner == "B" else "official640 (C)"
    dev = decision["winner_evidence"]["dev"]
    val = decision["winner_evidence"]["val"]
    if status == "proceed_aligned_training":
        return (
            f"交叉审计支持 {label} 在固定官方 TAL 候选上形成跨图像共享修正："
            f"dev Δ={dev['delta']:.6g}, CI95=[{dev['ci95'][0]:.6g}, {dev['ci95'][1]:.6g}]；"
            f"val Δ={val['delta']:.6g}, CI95=[{val['ci95'][0]:.6g}, {val['ci95'][1]:.6g}]。\n"
            "唯一下一步：启动固定候选、固定特征、同目标的系数头 minibatch 训练 pilot，匹配该 affine solution。"
        )
    return (
        f"交叉审计估计 {label} 未达到 dev 与 val 同时为正的 image-macro 迁移门槛。"
        f" dev Δ={dev['delta']:.6g}, CI95=[{dev['ci95'][0]:.6g}, {dev['ci95'][1]:.6g}]；"
        f" val Δ={val['delta']:.6g}, CI95=[{val['ci95'][0]:.6g}, {val['ci95'][1]:.6g}]。\n"
        "唯一下一步：把完整 fit-to-heldout gap 整理进机制报告并停止 affine 方法扩展。"
    )


def decide(eval_dir: Path, out: Path | None = None) -> dict[str, Any]:
    """Return and persist the single prospective decision for ``eval_dir``."""
    eval_dir = eval_dir.resolve()
    out = (out or eval_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    decision: dict[str, Any] = {
        "rule_version": RULE_VERSION,
        "rule_type": "prospective_addendum_not_original_preregistration",
        "gradient_limit": GRADIENT_LIMIT,
        "expected_counts": EXPECTED,
        "eval_dir": str(eval_dir),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    try:
        run = _read_json(eval_dir / "run.json")
        summary = _read_json(eval_dir / "SUMMARY.json")
        # The evaluator may still report ``evaluating``; map_validation plus
        # complete outputs is the adjudication gate.
        decision["evaluator_status"] = run.get("status")
        decision["map_gradients"] = _map_gradients(run)
        # Freeze selection using dev only.  Held-out val fields are not read
        # until this winner is fixed.
        _counts(summary, "fit")
        _counts(summary, "dev")
        dev_b = _contrast(_table(summary, "dev"), "B_minus_A")
        dev_c = _contrast(_table(summary, "dev"), "C_minus_A")
        winner = "B" if dev_b["delta"] >= dev_c["delta"] else "C"
        decision["winner"] = winner
        decision["selection"] = {"criterion": "larger dev image_macro delta vs A; exact tie selects B", "B_minus_A": dev_b, "C_minus_A": dev_c}
        decision["selection_frozen_before_val"] = True
        # Now validate held-out structure and read only the selected contrast.
        _counts(summary, "val")
        _candidate_identity_check(eval_dir)
        val_selected = _contrast(_table(summary, "val"), f"{winner}_minus_A")
        winner_selected = dev_b if winner == "B" else dev_c
        decision["winner_evidence"] = {"dev": winner_selected, "val": val_selected}
        decision["direct_objective_bundle_superiority"] = _direct_bundle(summary)
        qualifies = winner_selected["ci95"][0] > 0 and val_selected["ci95"][0] > 0
        decision["qualification"] = {"dev_lower_ci_gt_zero": winner_selected["ci95"][0] > 0, "val_lower_ci_gt_zero": val_selected["ci95"][0] > 0, "qualifies": qualifies}
        if qualifies:
            decision["status"] = "proceed_aligned_training"
            decision["next_step"] = "fixed_candidate_fixed_feature_same_objective_minibatch_coefficient_head_training_pilot"
        else:
            decision["status"] = "stop_affine_method_expansion"
            decision["next_step"] = "synthesize_full_fit_to_heldout_gap_in_mechanism_report"
    except InputError as exc:
        decision["status"] = "insufficient_inputs"
        decision["blocking_reason"] = str(exc)
        decision["next_step"] = "complete_current_audit"
    decision["brief"] = _brief(decision)
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "DECISION.md").write_text("# 交叉审计唯一判定\n\n" + decision["brief"] + "\n", encoding="utf-8")
    return decision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    decision = decide(args.eval_dir, args.out)
    print(json.dumps({"status": decision["status"], "next_step": decision["next_step"], "out": str((args.out or args.eval_dir).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
