"""Build evidence-based Chinese reports for the locked coefficient screen.

This stdlib-only script reads local Run evidence, writes REPORT.md per Run and
at the experiment root, and writes DECISION.json in the formal evaluation Run.
It never changes run.json, study.json, training artifacts, or remote state.
Incomplete or unverified evidence remains unknown. Run with --root pointing
to the experiment directory (the project root is also accepted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

EXPERIMENT = "acd_native_coefficient_20261006"
EVAL_RUN = "RUN_ACD_PAIRED_EVAL_S0"
BRANCHES = ("one2many", "one2one")
MODEL_NAMES = ("official", "baseline", "acd")
AP_KEYS = ("AP", "AP75", "APsmall")
EXPECTED_VERSION = "8.4.100"
EXPECTED_WEIGHT_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
PROTOCOL_FRAGMENTS = (
    "Ultralytics 8.4.100", "796", "5,000", "3 epochs", "third-epoch EMA",
    "0.003", "1%", ">= -0.005", "cv4", "one2one_cv4",
    "every gradient actually applied", "optimizer-attempt identities",
)
UNKNOWN = "未知"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def at(obj: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def truth(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def equal(value: Any, expected: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(expected, bool):
        return truth(value) is expected if truth(value) is not None else None
    return value == expected


def combine(values: list[bool | None]) -> bool | None:
    # Known invalid evidence stays invalid; missing evidence never becomes true.
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def verdict(value: bool | None) -> str:
    return "通过" if value is True else "未通过" if value is False else UNKNOWN


def fmt(value: Any, *, points: bool = False, signed: bool = False, digits: int = 4) -> str:
    value = number(value)
    if value is None:
        return UNKNOWN
    if points:
        value *= 100
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def cell(value: Any) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value).replace("|", "\\|").replace("\n", " ")


def clean(value: Any) -> Any:
    """Non-finite JSON numbers are unknown, never a serializable result."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content.rstrip() + "\n", encoding="utf-8")
    temp.replace(path)


def write_json(path: Path, content: Any) -> None:
    write_text(path, json.dumps(clean(content), ensure_ascii=False, indent=2, allow_nan=False))


class Evidence:
    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[Path, Any] = {}
        self.errors: list[str] = []
        self.sources: dict[str, str] = {}

    def label(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def digest(self, path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            self.sources[self.label(path)] = digest
            return digest
        except OSError as error:
            self.errors.append(f"{self.label(path)} 无法读取：{error}")
            return None

    def read(self, path: Path) -> Any:
        if path in self.cache:
            return self.cache[path]
        if not path.is_file():
            self.cache[path] = None
            return None
        try:
            raw = path.read_bytes()
            value = clean(json.loads(raw.decode("utf-8-sig")))
            self.sources[self.label(path)] = hashlib.sha256(raw).hexdigest()
        except (OSError, UnicodeError, ValueError) as error:
            self.errors.append(f"{self.label(path)} 无法读取：{type(error).__name__}: {error}")
            value = None
        self.cache[path] = value
        return value

    def stream(self, path: Path) -> list[Any] | None:
        if not path.is_file():
            return None
        try:
            raw = path.read_bytes()
            rows = [clean(json.loads(line)) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            self.sources[self.label(path)] = hashlib.sha256(raw).hexdigest()
            return rows
        except (OSError, UnicodeError, ValueError) as error:
            self.errors.append(f"{self.label(path)} 无法读取：{type(error).__name__}: {error}")
            return None


class Run:
    def __init__(self, path: Path, evidence: Evidence):
        self.path, self.rid, self.evidence = path, path.name, evidence
        self.meta = evidence.read(path / "run.json") or {}
        self.complete = evidence.read(path / "TRAINING_COMPLETE.json") or {}
        self.setup = evidence.read(path / "TRAINING_SETUP.json") or {}
        self.audit = self.complete.get("audit") or evidence.read(path / "TRAINING_AUDIT.json") or {}
        self.failed = evidence.read(path / "TRAINING_FAILED.json") or evidence.read(path / "FAILED.json") or {}
        self.summary = evidence.read(path / "SUMMARY.json") or {}
        self.progress = evidence.read(path / "PROGRESS.json") or {}
        self.transfer = evidence.read(path / "transfer.json") or {}
        self.pipeline_complete = evidence.read(path / "COMPLETE.json") or {}
        self.source = evidence.read(path / "SOURCE.json") or {}

    def ref(self, filename: str) -> str:
        return self.evidence.label(self.path / filename)


def select_training(runs: dict[str, Run], arm: str, stage: str) -> Run | None:
    base = f"RUN_{arm.upper()}_{stage}_S0"
    candidates = [r for rid, r in runs.items() if rid == base or rid.startswith(base + "_RETRY")]
    # Failed attempts stay in the reports, but never become a scientific arm.
    # For progress reports prefer an active retry to its failed predecessor.
    def failed(run: Run) -> bool:
        return bool(run.failed) or run.meta.get("status") == "failed" or run.meta.get("execution_status") == "failed"

    def completed(run: Run) -> bool:
        return run.complete.get("status") == "complete" and not failed(run)

    if base in runs and completed(runs[base]):
        return runs[base]
    complete = [r for r in candidates if completed(r)]
    if complete:
        return sorted(complete, key=lambda r: (str(r.complete.get("timestamp_utc", "")), r.rid))[-1]
    active = [r for r in candidates if not failed(r)]
    if active:
        return sorted(active, key=lambda r: r.rid)[-1]
    return runs.get(base) or (sorted(candidates, key=lambda r: r.rid)[-1] if candidates else None)


def branch_health(run: Run | None, key: str) -> bool | None:
    if run is None:
        return None
    detail = at(run.audit, "branch_updates", key)
    count = number(at(detail, "changed_parameter_count"))
    steps = number(at(detail, "gradients", "nonzero_steps"))
    applied = number(at(detail, "gradients", "applied_steps"))
    raw_nonfinite = number(at(detail, "gradients", "observed_nonfinite_gradient_steps"))
    raw_skipped = number(at(detail, "gradients", "observed_nonfinite_skipped_overflow_steps"))
    # Never reinterpret the old all_finite (AMP-scaled) flag as an applied
    # gradient audit. The explicit fields are required by the revised audit.
    return combine([
        truth(at(detail, "gradients", "all_applied_gradients_finite")),
        truth(at(detail, "gradients", "all_observed_nonfinite_steps_skipped")),
        count > 0 if count is not None else None,
        steps > 0 if steps is not None else None,
        applied > 0 if applied is not None else None,
        0 <= raw_nonfinite == raw_skipped if raw_nonfinite is not None and raw_skipped is not None else None,
    ])


def optimizer_health(run: Run | None) -> bool | None:
    if run is None:
        return None
    audit = run.audit.get("optimizer_audit")
    if not isinstance(audit, dict):
        return None
    counts = {key: number(audit.get(key)) for key in (
        "attempted_steps", "applied_steps", "skipped_overflow_steps", "observed_nonfinite_gradient_steps",
        "observed_nonfinite_skipped_overflow_steps", "unsafe_step_prevented_count", "unclassified_skipped_steps")}
    valid_counts = all(v is not None and v >= 0 and v.is_integer() for v in counts.values())
    attempted, applied, skipped = (counts[k] for k in ("attempted_steps", "applied_steps", "skipped_overflow_steps"))
    observed, confirmed = (counts[k] for k in ("observed_nonfinite_gradient_steps", "observed_nonfinite_skipped_overflow_steps"))
    arithmetic = (attempted > 0 and applied > 0 and attempted == applied + skipped
                  and observed == confirmed == skipped) if valid_counts else None
    values = [arithmetic, truth(audit.get("all_applied_gradients_finite")),
              truth(audit.get("all_observed_nonfinite_steps_skipped")),
              equal(audit.get("observed_all_gradients_finite"), observed == 0) if observed is not None else None,
              equal(audit.get("unsafe_step_prevented_count"), 0), equal(audit.get("unclassified_skipped_steps"), 0),
              equal(audit.get("ema_updates"), attempted) if attempted is not None else None,
              truth(run.audit.get("audit_passed")), truth(run.audit.get("live_state_finite")),
              truth(run.audit.get("ema_state_finite")), truth(run.audit.get("ema_frozen_state_unchanged"))]
    # If the step log is available locally, independently corroborate the
    # aggregate receipt. A missing local log does not rewrite remote evidence.
    step_path = run.path / "optimizer_step_audit.jsonl"
    if step_path.is_file():
        rows = run.evidence.stream(step_path)
        if rows is None:
            values.append(None)
        else:
            valid_rows = []
            observed_steps = 0
            for row in rows:
                unscaled = at(row, "unscaled")
                finite = combine([truth(at(unscaled, branch, "all_gradients_finite")) for branch in BRANCHES])
                observed_steps += int(finite is False)
                row_applied = truth(at(row, "optimizer_applied"))
                row_skipped = truth(at(row, "skipped_overflow_step"))
                if row_skipped is True:
                    before, after = number(at(row, "scale_before")), number(at(row, "scale_after"))
                    valid_rows.append(combine([
                        truth(at(row, "amp")), truth(at(row, "scaler_enabled")),
                        equal(at(row, "optimizer_invoked"), False), equal(row_applied, False),
                        finite is False if finite is not None else None,
                        after < before if after is not None and before is not None else None,
                    ]))
                elif row_applied is True:
                    valid_rows.append(combine([
                        equal(row_skipped, False), truth(at(row, "optimizer_invoked")),
                        *[truth(at(row, "at_optimizer_call", branch, "all_gradients_finite")) for branch in BRANCHES],
                    ]))
                else:
                    valid_rows.append(None)
                valid_rows.append(equal(at(row, "unsafe_step_prevented"), False))
                ema_before, ema_after = number(at(row, "ema_updates_before")), number(at(row, "ema_updates_after"))
                valid_rows.append(ema_after == ema_before + 1 if ema_before is not None and ema_after is not None else None)
            values.extend([combine(valid_rows), len(rows) == attempted if attempted is not None else None,
                           sum(at(row, "optimizer_applied") is True for row in rows) == applied if applied is not None else None,
                           sum(at(row, "skipped_overflow_step") is True for row in rows) == skipped if skipped is not None else None,
                           observed_steps == observed if observed is not None else None])
    return combine(values)


def optimizer_comparison(baseline: Run | None, acd: Run | None) -> dict[str, Any]:
    keys = ("attempted_steps", "applied_steps", "skipped_overflow_steps", "observed_nonfinite_gradient_steps", "ema_updates")
    result: dict[str, Any] = {}
    for name, run in (("baseline", baseline), ("acd", acd)):
        audit = run.audit.get("optimizer_audit", {}) if run else {}
        result[name] = {"run": run.rid if run else None, **{key: audit.get(key) for key in keys}}
    b, a = number(at(result, "baseline", "skipped_overflow_steps")), number(at(result, "acd", "skipped_overflow_steps"))
    result["skipped_overflow_steps_delta_acd_minus_baseline"] = a - b if a is not None and b is not None else None
    result["same_skipped_count"] = a == b if a is not None and b is not None else None
    skip_identities = {}
    for name, run in (("baseline", baseline), ("acd", acd)):
        if run and (run.path / 'optimizer_step_audit.jsonl').is_file():
            rows = run.evidence.stream(run.path / 'optimizer_step_audit.jsonl')
            if rows is not None:
                skip_identities[name] = [{k: row[k] for k in ('attempt', 'epoch', 'input_batches_seen')}
                                         for row in rows if row.get('skipped_overflow_step') is True]
    result['overflow_step_identities'] = skip_identities
    result['same_overflow_skip_identity'] = (skip_identities['baseline'] == skip_identities['acd']
                                             if set(skip_identities) == {'baseline', 'acd'} else None)
    if result["same_skipped_count"] is False:
        result["limitation"] = "两臂 GradScaler overflow-skip 次数不同；相同输入/epoch 预算不等于相同实际 optimizer 更新量。结果适用于各臂实际 AMP 轨迹，不自动把差异归因于 ACD。"
    elif result['same_overflow_skip_identity'] is True:
        result['limitation'] = '已直接核验两臂 overflow-skip 的 attempt、epoch 和 batch 位置相同；损失与梯度值仍由各自训练条件决定。'
    elif result["same_skipped_count"] is True:
        result["limitation"] = "skip 总数一致不能证明跳过的是同一批次；不从计数推断两臂数值轨迹完全相同。"
    else:
        result["limitation"] = "实际 applied/overflow-skip 计数尚未齐；不能由末端 GradScaler scale 推算精确跳步数量。"
    return result


def optimizer_schedule_parity(evidence: Evidence, runs: dict[str, Run], baseline: Run | None,
                              acd: Run | None) -> dict[str, Any]:
    result: dict[str, Any] = {"verified": None, "sources": [],
                              "baseline": baseline.rid if baseline else None, "acd": acd.rid if acd else None}
    if baseline is None or acd is None:
        return result
    keys = ("epoch", "attempt", "input_batches_seen", "learning_rates", "accumulate", "amp", "scaler_enabled")
    result["schedule_keys"] = list(keys)
    values: list[bool | None] = []
    av = evidence.stream(baseline.path / "optimizer_step_audit.jsonl")
    bv = evidence.stream(acd.path / "optimizer_step_audit.jsonl")
    if av is not None and bv is not None:
        result["sources"].extend([baseline.ref("optimizer_step_audit.jsonl"), acd.ref("optimizer_step_audit.jsonl")])
        complete = all(isinstance(row, dict) and all(key in row and row[key] is not None for key in keys) for row in av + bv)
        if complete:
            schedules = [[{key: row[key] for key in keys} for row in rows] for rows in (av, bv)]
            values.append(bool(av) and schedules[0] == schedules[1])
            result["directly_compared_steps"] = {"baseline": len(av), "acd": len(bv)}
            for name, run, rows in (("baseline", baseline, av), ("acd", acd, bv)):
                attempts = number(at(run.audit, "optimizer_audit", "attempted_steps"))
                values.append(len(rows) == attempts if attempts is not None else None)
        else:
            values.append(None)
            result["limitation"] = "逐 step 记录未包含完整的锁定 attempt schedule 字段"
    for run in runs.values():
        receipt = evidence.read(run.path / "OPTIMIZER_SCHEDULE_PARITY.json")
        if not isinstance(receipt, dict) or receipt.get("baseline") != baseline.rid or receipt.get("acd") != acd.rid:
            continue
        result["sources"].append(run.ref("OPTIMIZER_SCHEDULE_PARITY.json"))
        values.append(truth(receipt.get("same_attempt_schedule")))
        result["receipts"] = result.get("receipts", []) + [receipt]
        for name, training_run in (("baseline", baseline), ("acd", acd)):
            for receipt_key, audit_key in (("attempts", "attempted_steps"), ("applied", "applied_steps"), ("overflow_skips", "skipped_overflow_steps")):
                observed = number(at(receipt, receipt_key, name))
                audited = number(at(training_run.audit, "optimizer_audit", audit_key))
                values.append(observed == audited if observed is not None and audited is not None else None)
    result["verified"] = combine(values)
    return result


def source_snapshot(run: Run) -> dict[str, Any]:
    result: dict[str, Any] = {"metadata": run.source, "metadata_source": run.ref("SOURCE.json")}
    filename = str(run.source.get("executed_snapshot", "")).replace("\\", "/").rsplit("/", 1)[-1]
    for kind, path, declared_key in (
        ("script", run.path / "source" / filename, "script_sha256"),
        ("protocol", run.path / "source" / "PROTOCOL.md", "protocol_sha256"),
    ):
        declared = run.source.get(declared_key)
        observed = None
        if path.is_file():
            try:
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
                run.evidence.sources[run.evidence.label(path)] = observed
            except OSError as error:
                run.evidence.errors.append(f"{run.evidence.label(path)} 无法读取：{error}")
        result[kind] = {"source": run.evidence.label(path), "declared_sha256": declared,
                        "observed_sha256": observed, "verified": observed == declared if observed and declared else None}
    result["verified"] = combine([result["script"]["verified"], result["protocol"]["verified"]])
    return result


def data_parity(evidence: Evidence, runs: dict[str, Run], baseline: Run | None,
                acd: Run | None, filename: str) -> dict[str, Any]:
    if baseline is None or acd is None:
        return {"verified": None, "reason": "配对训练 Run 尚未发现", "sources": []}
    values: list[bool | None] = []
    sources = []
    result: dict[str, Any] = {"baseline": baseline.rid, "acd": acd.rid, "sources": sources}
    av = evidence.stream(baseline.path / "input_batch_hashes.jsonl")
    bv = evidence.stream(acd.path / "input_batch_hashes.jsonl")
    if av is not None and bv is not None:
        values.append(bool(av) and av == bv)
        result.update(baseline_steps=len(av), acd_steps=len(bv), compared_json_rows=True)
        sources.extend([baseline.ref("input_batch_hashes.jsonl"), acd.ref("input_batch_hashes.jsonl")])
    receipts = []
    for run in runs.values():
        receipt = evidence.read(run.path / filename)
        if isinstance(receipt, dict) and receipt.get("baseline") == baseline.rid and receipt.get("acd") == acd.rid:
            receipts.append(receipt)
            sources.append(run.ref(filename))
            steps = number(receipt.get("steps"))
            values.append(combine([truth(receipt.get("same_transformed_inputs")),
                                   steps > 0 if steps is not None else None]))
            if av is None or bv is None:
                result.update(baseline_steps=receipt.get("steps"), acd_steps=receipt.get("steps"))
    # Audited complete stream length must agree with every supplied parity
    # receipt. Matching an incomplete prefix cannot verify paired training.
    ba = number(baseline.audit.get("input_batches"))
    aa = number(acd.audit.get("input_batches"))
    values.append(ba == aa and ba > 0 if ba is not None and aa is not None else None)
    for count_key, audited in (("baseline_steps", ba), ("acd_steps", aa)):
        observed = number(result.get(count_key))
        values.append(observed == audited if observed is not None and audited is not None else None)
    for receipt in receipts:
        count = number(receipt.get("steps"))
        values.append(count == ba == aa if count is not None and ba is not None and aa is not None else None)
    result["verified"] = combine(values) if sources else None
    result["reason"] = "完整 transformed-input 流/配对 receipt 与两臂审计步数核对" if sources else "尚无完整输入流或对应 pipeline parity receipt"
    return result


def protocol_lock(evidence: Evidence) -> dict[str, Any]:
    path = evidence.root / "PROTOCOL.md"
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        return {"verified": None, "reason": f"PROTOCOL.md 无法读取：{error}", "source": "PROTOCOL.md"}
    digest = hashlib.sha256(raw).hexdigest()
    evidence.sources["PROTOCOL.md"] = digest
    missing = [fragment for fragment in PROTOCOL_FRAGMENTS if fragment not in content]
    return {"verified": not missing, "sha256": digest, "source": "PROTOCOL.md",
            "missing_lock_fragments": missing,
            "thresholds": {"segm_AP_delta_min": .003, "damage_rate_max": .01,
                           "baseline_success_mean_iou_delta_min": -.005,
                           "AP75_and_APsmall_may_not_both_decrease": True}}


def criterion(name: str, value: bool | None, details: Any, sources: list[str],
              group: str = "verification") -> dict[str, Any]:
    return {"name": name, "group": group, "passed": value,
            "status": "pass" if value is True else "fail" if value is False else "unknown",
            "details": details, "sources": sources}


def training_checks(run: Run | None, arm: str, paired_source: dict[str, Any]) -> list[dict[str, Any]]:
    if run is None:
        return [criterion(f"{arm} 正式训练及审计", None, "尚未发现 Run", [])]
    refs = [run.ref("TRAINING_COMPLETE.json"), run.ref("TRAINING_SETUP.json"), run.ref("SOURCE.json")]
    complete = combine([equal(run.complete.get("status"), "complete"), not bool(run.failed),
                        run.meta.get("status") != "failed", run.meta.get("execution_status") != "failed"])
    initial = at(run.audit, "frozen_state_initial", "sha256")
    final = at(run.audit, "frozen_state_final", "sha256")
    digest_equal = initial == final if isinstance(initial, str) and initial and isinstance(final, str) and final else None
    lock = combine([
        complete, equal(run.complete.get("arm"), arm), equal(run.complete.get("epochs"), 3),
        equal(run.complete.get("seed"), 0), equal(run.complete.get("runtime_ultralytics"), EXPECTED_VERSION),
        equal(run.setup.get("base_weights_sha256"), EXPECTED_WEIGHT_SHA),
        equal(run.setup.get("train_images"), 796), equal(run.setup.get("validation_images"), 196),
        equal(run.setup.get("batch"), 2), equal(run.setup.get("all_batchnorm_eval"), True),
        equal(run.setup.get("all_batchnorm_affine_frozen"), True),
        equal(run.audit.get("input_batches"), 1194),
        truth(paired_source.get("verified")),
    ])
    return [
        criterion(f"{arm} 固定训练预算/版本/数据/BN", lock,
                  {"run": run.rid, "status": run.complete.get("status"), "epochs": run.complete.get("epochs"),
                   "seed": run.complete.get("seed"), "ultralytics": run.complete.get("runtime_ultralytics"),
                   "train_images": run.setup.get("train_images"), "validation_images": run.setup.get("validation_images"),
                   "batch": run.setup.get("batch"), "base_weights_sha256": run.setup.get("base_weights_sha256"),
                   "paired_source_snapshots": paired_source}, refs),
        criterion(f"{arm} 有限 live loss", truth(run.audit.get("finite_training_loss")),
                  run.audit.get("finite_training_loss"), [run.ref("TRAINING_AUDIT.json")]),
        criterion(f"{arm} 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过", optimizer_health(run),
                  {"optimizer_audit": run.audit.get("optimizer_audit"), "audit_passed": run.audit.get("audit_passed"),
                   "audit_failures": run.audit.get("audit_failures"), "live_state_finite": run.audit.get("live_state_finite"),
                   "ema_state_finite": run.audit.get("ema_state_finite"), "ema_frozen_state_unchanged": run.audit.get("ema_frozen_state_unchanged")},
                  [run.ref("TRAINING_AUDIT.json"), run.ref("optimizer_step_audit.jsonl")]),
        criterion(f"{arm} frozen 参数及 BN buffers 不变", combine([truth(run.audit.get("frozen_state_unchanged")), digest_equal]),
                  {"initial": run.audit.get("frozen_state_initial"), "final": run.audit.get("frozen_state_final")},
                  [run.ref("TRAINING_AUDIT.json")]),
        *[criterion(f"{arm} {branch} 实际应用梯度有限且系数更新", branch_health(run, branch),
                    at(run.audit, "branch_updates", branch), [run.ref("TRAINING_AUDIT.json")]) for branch in BRANCHES],
    ]


def direct_parity(summary: dict[str, Any]) -> bool | None:
    pair = at(summary, "paired", "acd")
    p = at(pair, "parity")
    return combine([
        truth(at(pair, "paired_readout_valid")), truth(at(p, "frozen_state_exact_equal")),
        truth(at(p, "all_images_box_class_exact_equal")), truth(at(p, "detection_index_equal")),
        equal(at(p, "image_count"), 5000), equal(at(p, "detection_count_mismatch_images"), 0),
        equal(at(p, "class_mismatch_detections"), 0), equal(at(p, "box_max_abs_error"), 0),
        equal(at(p, "confidence_max_abs_error"), 0),
    ])


def build_decision(evidence: Evidence, runs: dict[str, Run]) -> dict[str, Any]:
    lock = protocol_lock(evidence)
    baseline = select_training(runs, "baseline", "FEASIBILITY")
    acd = select_training(runs, "acd", "FEASIBILITY")
    eval_run = runs.get(EVAL_RUN)
    summary = eval_run.summary if eval_run else {}
    ref = f"runs/{EVAL_RUN}/SUMMARY.json"
    snapshots = {arm: source_snapshot(run) if run else None
                 for arm, run in (("baseline", baseline), ("acd", acd), ("evaluation", eval_run))}
    script_hashes = [at(snapshots, arm, "script", "observed_sha256") for arm in ("baseline", "acd")]
    protocol_hashes = [at(snapshots, arm, "protocol", "observed_sha256") for arm in ("baseline", "acd")]
    paired_source = {
        "script_sha256": dict(zip(("baseline", "acd"), script_hashes)),
        "protocol_sha256": dict(zip(("baseline", "acd"), protocol_hashes)),
        "verified": combine([
            truth(at(snapshots, "baseline", "verified")), truth(at(snapshots, "acd", "verified")),
            script_hashes[0] == script_hashes[1] if all(script_hashes) else None,
            all(value == lock.get("sha256") for value in protocol_hashes) if all(protocol_hashes) and lock.get("sha256") else None,
        ]),
    }
    checks = [criterion("协议锁", truth(lock.get("verified")), lock, ["PROTOCOL.md"])]
    checks.extend(training_checks(baseline, "baseline", paired_source))
    checks.extend(training_checks(acd, "acd", paired_source))
    parity = data_parity(evidence, runs, baseline, acd, "FEASIBILITY_PARITY.json")
    checks.append(criterion("配对 transformed-input 完整流一致", parity["verified"], parity, parity["sources"]))
    schedule = optimizer_schedule_parity(evidence, runs, baseline, acd)
    checks.append(criterion("两臂 optimizer attempt 的 batch/LR/accumulate/AMP schedule 一致", schedule["verified"], schedule, schedule["sources"]))
    complete_full = combine([equal(summary.get("status"), "complete"), equal(summary.get("image_count"), 5000),
                             truth(summary.get("covers_all_annotation_images")), equal(summary.get("baseline"), "baseline")])
    configuration = at(summary, "configuration") or {}
    data_receipt = evidence.read(evidence.root / "data" / "DATA_RECEIPT.json") or {}
    image_ids = configuration.get("image_ids")
    complete_full = combine([
        complete_full,
        equal(at(configuration, "annotations", "sha256"), data_receipt.get("annotation_sha256")),
        equal(at(configuration, "images_list", "sha256"), at(data_receipt, "lists", "val_full.txt")),
        equal(evidence.digest(evidence.root / "data" / "val_full.txt"), at(data_receipt, "lists", "val_full.txt")),
        len(image_ids) == len(set(image_ids)) == 5000 if isinstance(image_ids, list) else None,
    ])
    readout_links = {}
    for arm in MODEL_NAMES:
        receipt = at(summary, "prediction_receipts", arm) or {}
        metrics = at(summary, "metrics", arm) or {}
        fingerprint = receipt.get("fingerprint", {})
        common_fingerprint = {key: value for key, value in fingerprint.items() if key != "weights"}
        local_receipt = evidence.read(eval_run.path / arm / "COMPLETE.json") if eval_run else None
        local_metrics = evidence.read(eval_run.path / arm / "COCO_METRICS.json") if eval_run else None
        predictions_sha = evidence.digest(eval_run.path / arm / "predictions.json") if eval_run else None
        readout_links[arm] = {
            "predictions_sha256": predictions_sha,
            "verified": combine([
                equal(receipt.get("status"), "prediction_complete"), equal(receipt.get("image_count"), 5000),
                common_fingerprint == configuration if common_fingerprint and configuration else None,
                local_receipt == receipt if local_receipt and receipt else None,
                local_metrics == metrics if local_metrics and metrics else None,
                equal(metrics.get("image_ids"), sorted(image_ids)) if isinstance(image_ids, list) else None,
                equal(metrics.get("backend"), "pycocotools.COCOeval"),
                equal(receipt.get("predictions_sha256"), predictions_sha) if predictions_sha else None,
                equal(metrics.get("predictions_sha256"), predictions_sha) if predictions_sha else None,
            ]),
        }
    complete_full = combine([complete_full, *[item["verified"] for item in readout_links.values()]])
    independent = evidence.read(eval_run.path / "READOUT_VERIFICATION.json") if eval_run else None
    independent = independent or {}
    independent_valid = combine([
        truth(independent.get("passed")), equal(independent.get("status"), "passed"),
        equal(independent.get("errors"), []), equal(independent.get("missing"), []),
        equal(at(independent, "evidence", "SUMMARY.json", "sha256"), evidence.sources.get(ref)),
    ])
    complete_full = combine([complete_full, independent_valid])
    checks.append(criterion("完整 5,000 张原始 COCO val2017 独立评估", complete_full,
                            {"evaluation": {key: summary.get(key) for key in ("status", "image_count", "covers_all_annotation_images", "evaluation_scope")},
                             "annotations": configuration.get("annotations"), "images_list": configuration.get("images_list"),
                             "per_arm_readout_links": readout_links, "independent_aggregate_readout_verified": independent_valid},
                            [ref, "data/DATA_RECEIPT.json", "data/val_full.txt", f"runs/{EVAL_RUN}/READOUT_VERIFICATION.json"]))
    config = configuration.get("config", {})
    model_receipts = at(summary, "prediction_receipts") or {}
    reconstruction = [equal(configuration.get("ultralytics"), EXPECTED_VERSION),
                      equal(config.get("half"), False), equal(config.get("imgsz"), 640),
                      equal(config.get("conf"), .001), equal(config.get("max_det"), 300),
                      equal(config.get("native_one2one"), True), equal(config.get("tf32"), False),
                      equal(config.get("rect"), False), equal(config.get("scaleup"), False),
                      equal(config.get("augment"), False), equal(config.get("coco_maxDets"), [1, 10, 100]),
                      equal(config.get("decode"), "SegmentationValidator.save_json=True: process_mask_native; scale_preds.byte()")]
    coefficient_hashes = {}
    for arm in ("baseline", "acd"):
        training_run = baseline if arm == "baseline" else acd
        coefficient_hashes[arm] = evidence.digest(training_run.path / "coeff_final_ema.pt") if training_run else None
        reconstruction.extend([
            truth(at(model_receipts, arm, "model", "reconstructed_coefficient_only")),
            equal(at(model_receipts, arm, "model", "coefficient_epoch"), 3),
            equal(at(model_receipts, arm, "model", "coefficient_kind"), "coefficient_only_ema_final"),
            equal(at(model_receipts, arm, "fingerprint", "weights", "base_weights", "sha256"), EXPECTED_WEIGHT_SHA),
            equal(at(model_receipts, arm, "fingerprint", "weights", "coefficients", "sha256"), coefficient_hashes[arm])
            if coefficient_hashes[arm] else None,
        ])
    reconstruction.append(equal(at(model_receipts, "official", "fingerprint", "weights", "checkpoint", "sha256"), EXPECTED_WEIGHT_SHA))
    reconstruction.extend([
        truth(at(snapshots, "evaluation", "verified")),
        equal(at(snapshots, "evaluation", "protocol", "observed_sha256"), lock.get("sha256")),
        equal(configuration.get("evaluator_sha256"), at(snapshots, "evaluation", "script", "observed_sha256")),
    ])
    checks.append(criterion("固定第 3 epoch EMA coefficient overlay 官方 FP32 推理", combine(reconstruction),
                            {"config": config, "baseline": at(model_receipts, "baseline", "model"),
                             "acd": at(model_receipts, "acd", "model"),
                             "selected_formal_coefficient_sha256": coefficient_hashes,
                             "evaluation_source_snapshot": snapshots["evaluation"]}, [ref, f"runs/{EVAL_RUN}/SOURCE.json"]))
    hashes = {arm: at(model_receipts, arm, "frozen_state_sha256") for arm in MODEL_NAMES}
    frozen_ref = len(set(hashes.values())) == 1 if all(isinstance(v, str) and v for v in hashes.values()) else None
    checks.append(criterion("official/baseline/acd frozen FP32 tensors digest 一致", frozen_ref, hashes, [ref]))
    checks.append(criterion("ACD/baseline box/class/confidence/detection identity 直接逐图一致", direct_parity(summary),
                            at(summary, "paired", "acd", "parity"), [ref]))
    metrics = at(summary, "metrics") or {}
    deltas: dict[str, float | None] = {}
    for key in AP_KEYS:
        b = number(at(metrics, "baseline", "segm", key))
        a = number(at(metrics, "acd", "segm", key))
        deltas[key] = a - b if a is not None and b is not None and 0 <= a <= 1 and 0 <= b <= 1 else None
    ap_delta = deltas["AP"]
    checks.append(criterion("Mask AP：ACD − baseline >= +0.003", ap_delta >= .003 if ap_delta is not None else None,
                            {"delta_fraction": ap_delta, "delta_percentage_points": 100 * ap_delta if ap_delta is not None else None,
                             "threshold_fraction": .003}, [ref], "science"))
    d75, ds = deltas["AP75"], deltas["APsmall"]
    checks.append(criterion("Mask AP75/APsmall 不同时下降", not (d75 < 0 and ds < 0) if d75 is not None and ds is not None else None,
                            {"AP75_delta_fraction": d75, "APsmall_delta_fraction": ds}, [ref], "science"))
    stats = at(summary, "paired", "acd", "statistics") or {}
    success = number(stats.get("baseline_success_count"))
    matched = number(stats.get("matched_detection_count"))
    damage = number(at(stats, "damage_rate_of_baseline_success", "value"))
    success_delta = number(at(stats, "baseline_success_mean_iou_delta", "value"))
    damage_count = number(stats.get("damage_count"))
    damage_detail = at(stats, "damage_rate_of_baseline_success") or {}
    success_detail = at(stats, "baseline_success_mean_iou_delta") or {}
    ci_available = []
    for detail in (damage_detail, success_detail):
        ci = detail.get("ci95")
        ci_available.append(isinstance(ci, list) and len(ci) == 2 and all(number(v) is not None for v in ci)
                            if ci is not None else None)
        valid_draws = number(detail.get("valid_bootstrap_draws"))
        ci_available.append(valid_draws > 0 if valid_draws is not None else None)
    rate_consistent = (0 <= damage_count <= success and math.isclose(damage_count / success, damage, abs_tol=1e-12)
                       if damage_count is not None and success is not None and success > 0 and damage is not None else None)
    sufficient = combine([success > 0 if success is not None else None,
                          matched > 0 if matched is not None else None, equal(stats.get("images"), 5000),
                          equal(stats.get("bootstrap_samples"), 5000), equal(stats.get("bootstrap_seed"), 0),
                          rate_consistent, *ci_available])
    checks.append(criterion("配对损伤诊断有 baseline-success 样本且图像 bootstrap 完整", sufficient,
                            {key: stats.get(key) for key in ("matched_detection_count", "baseline_success_count", "images", "bootstrap_samples", "bootstrap_seed", "bootstrap_unit")}, [ref]))
    checks.append(criterion("baseline-success 损伤率 <= 1%", 0 <= damage <= .01 if damage is not None and sufficient is True else None,
                            at(stats, "damage_rate_of_baseline_success"), [ref], "science"))
    checks.append(criterion("baseline-success 平均 mask IoU 变化 >= −0.005", success_delta >= -.005 if success_delta is not None and sufficient is True else None,
                            at(stats, "baseline_success_mean_iou_delta"), [ref], "science"))
    verification = [c["passed"] for c in checks if c["group"] == "verification"]
    science = [c["passed"] for c in checks if c["group"] == "science"]
    # A damaged/partial execution cannot establish a scientific transfer result.
    # It does close the valid-result gate, but the scientific conclusion is unknown.
    verified = combine(verification)
    if verified is not True or any(v is None for v in science):
        outcome = "unknown"
        gate = "fail" if verified is False else "unknown"
        reason = "有效性核验未全部通过或指标未齐；无法判定正/负 training-transfer。"
    elif all(v is True for v in science):
        outcome, gate = "positive", "pass"
        reason = "固定预算下所有有效性核验与预定义科学阈值通过。"
    else:
        outcome, gate = "negative", "fail"
        reason = "有效性核验通过，但至少一个预定义科学阈值未通过；保留本次未达门槛的结果并停止调参。"
    return {"generated_at": timestamp(), "experiment": EXPERIMENT, "evaluation_run": EVAL_RUN,
            "outcome": outcome, "gate_status": gate, "reason": reason,
            "comparison": "acd minus coefficient-only baseline", "protocol": lock,
            "training_runs": {"baseline": baseline.rid if baseline else None, "acd": acd.rid if acd else None},
            "optimizer_step_comparison": optimizer_comparison(baseline, acd),
            "optimizer_schedule_parity": schedule,
            "source_snapshots": snapshots,
            "independent_readout_verification": {"passed": independent_valid, "source": f"runs/{EVAL_RUN}/READOUT_VERIFICATION.json",
                                                 "limitations": independent.get("limitations")},
            "excluded_failed_training_runs": [run.rid for run in runs.values()
                                                if "FEASIBILITY" in run.rid and (run.failed or run.meta.get("execution_status") == "failed")],
            "selected_epoch": 3, "seed": 0, "fit_images": 796, "final_validation_images": 5000,
            "blind_test": False, "AP_confidence_intervals_computed": False,
            "criteria": checks, "segm_delta": deltas,
            "read_errors": evidence.errors, "evidence_sha256": dict(sorted(evidence.sources.items())),
            "next_action": "Stop at the fixed budget. Do not add seeds, tune auxiliary settings, or expand training automatically."}


def metric_table(summary: dict[str, Any]) -> list[str]:
    lines = ["COCO AP 均以百分数展示；差值为百分点。未知表示指标缺失、未回传或非有限值。", "",
             "| 评价 | 模型 | AP50:95 | AP75 | APsmall |", "|---|---|---:|---:|---:|"]
    metrics = summary.get("metrics", {})
    for iou_type, label in (("segm", "Mask"), ("bbox", "Box")):
        for name in MODEL_NAMES:
            row = [fmt(at(metrics, name, iou_type, key), points=True, digits=3) for key in AP_KEYS]
            lines.append(f"| {label} | {name} | " + " | ".join(row) + " |")
        delta = []
        for key in AP_KEYS:
            a, b = number(at(metrics, "acd", iou_type, key)), number(at(metrics, "baseline", iou_type, key))
            delta.append(fmt(a - b if a is not None and b is not None else None, points=True, signed=True, digits=3))
        lines.append(f"| {label} | ACD − baseline | " + " | ".join(delta) + " |")
    lines.extend(["", "这里不提供 AP 置信区间；本实验的 bootstrap 只用于图像配对的 IoU/损伤诊断。"])
    return lines


def confidence_interval(value: Any, *, rate: bool = False) -> str:
    if not isinstance(value, list) or len(value) != 2 or any(number(v) is None for v in value):
        return UNKNOWN
    return "[" + ", ".join(fmt(v, points=rate, digits=4 if rate else 6) for v in value) + "]"


def pair_report(summary: dict[str, Any]) -> list[str]:
    pair = at(summary, "paired", "acd") or {}
    stats = pair.get("statistics", {})
    lines = ["配对定义：每个 baseline detection 以同类 box IoU ≥ 0.5 匹配最高重叠的 noncrowd GT；",
             "baseline 原图 mask IoU ≥ 0.75 为成功，ACD 降至 < 0.75 为损伤，反向越过该阈值为修复。",
             "允许多个 detection 匹配同一 GT，计数是 detection identity 的诊断，不能当成 COCO AP 一对一匹配。", "",
             f"配对可用性：{verdict(truth(pair.get('paired_readout_valid')))}。", "",
             "| 计数 | 数值 |", "|---|---:|"]
    count_labels = {"matched_detection_count": "配对匹配 detections", "baseline_success_count": "baseline-success",
                    "damage_count": "成功组损伤", "baseline_failure_count": "baseline-failure", "repair_count": "失败组修复",
                    "images": "评价图像", "images_with_matched_detections": "含匹配 detection 的图像"}
    for key, label in count_labels.items():
        lines.append(f"| {label} | {cell(stats.get(key))} |")
    lines.append(f"| unique matched GT | {cell(pair.get('unique_matched_gt_instances'))} |")
    lines.extend(["", "| 图像簇配对诊断 | 点估计 | 95% bootstrap 区间 |", "|---|---:|---:|"])
    for key, label, rate in (
        ("mean_mask_iou_delta", "所有匹配 detection 平均 mask IoU 变化", False),
        ("baseline_success_mean_iou_delta", "baseline-success 平均 mask IoU 变化", False),
        ("damage_rate_of_baseline_success", "baseline-success 损伤率（%）", True),
        ("repair_rate_of_baseline_failure", "baseline-failure 修复率（%）", True),
        ("baseline_mean_mask_iou", "baseline 平均 mask IoU", False),
        ("method_mean_mask_iou", "ACD 平均 mask IoU", False),
    ):
        detail = stats.get(key, {})
        lines.append(f"| {label} | {fmt(at(detail, 'value'), points=rate, digits=4 if rate else 6)} | {confidence_interval(at(detail, 'ci95'), rate=rate)} |")
    lines.extend(["", f"bootstrap：{cell(stats.get('bootstrap_samples'))} 次，seed={cell(stats.get('bootstrap_seed'))}；",
                  f"单位为图像簇，两臂配对抽样，以簇总和之比统计。记录定义：{cell(stats.get('bootstrap_unit'))}。",
                  "区间描述当前固定模型与该验证集的诊断变化，不量化不同训练 seed 的不确定性。"])
    if pair.get("limitation"):
        lines.extend(["", "配对限制：" + cell(pair["limitation"])])
    return lines


def training_report(run: Run) -> list[str]:
    lines = ["## 训练审计", "", f"TRAINING_COMPLETE receipt：{'已存在' if run.complete else '未发现'}；",
             f"receipt arm={cell(run.complete.get('arm'))}，epoch={cell(run.complete.get('epochs'))}，",
             f"seed={cell(run.complete.get('seed'))}，Ultralytics={cell(run.complete.get('runtime_ultralytics'))}。", "",
             f"有限 loss：{verdict(truth(run.audit.get('finite_training_loss')))}；",
             f"frozen 参数及 BN buffers 不变：{verdict(truth(run.audit.get('frozen_state_unchanged')))}；",
             f"transformed-input batches={cell(run.audit.get('input_batches'))}。", "",
             f"梯度数值的记录单位：{cell(run.audit.get('gradient_magnitude_units'))}。", "",
             "原始 AMP-scaled 梯度、unscale 后/clip 前观测梯度、实际 optimizer 调用处的梯度分别记录。",
             "观察到 Inf/NaN 的 attempt 保留为非有限；只有被 GradScaler 实际跳过的 attempt 才可排除出 applied 集合。",
             "实际应用梯度 gate：" + verdict(optimizer_health(run)) + "。", "",
             "| optimizer / GradScaler 全局审计 | 数值 |", "|---|---|"]
    optimizer = run.audit.get("optimizer_audit", {})
    for key, label in (
        ("attempted_steps", "尝试更新 step"), ("applied_steps", "实际应用 step"),
        ("skipped_overflow_steps", "确认 AMP overflow skip step"),
        ("ema_updates", "官方 EMA updates（包括 overflow-skip attempt）"),
        ("observed_all_gradients_finite", "unscale 后/clip 前梯度全部有限"),
        ("observed_nonfinite_gradient_steps", "原始 unscaled 非有限 attempt"),
        ("observed_nonfinite_skipped_overflow_steps", "非有限且确认由 scaler 跳过的 attempt"),
        ("all_observed_nonfinite_steps_skipped", "所有原始非有限 attempt 均确认被跳过"),
        ("all_applied_gradients_finite", "实际应用梯度全部有限"),
        ("unsafe_step_prevented_count", "危险 optimizer 调用被拦截次数"),
        ("unclassified_skipped_steps", "未分类 skip 次数"),
    ):
        lines.append(f"| {label} | {cell(optimizer.get(key))} |")
    if not optimizer:
        lines.extend(["", "此 Run 使用旧梯度审计，缺少直接 optimizer invocation / GradScaler skip 记录。",
                      "旧 `all_finite` 记录的是 AMP-scaled 梯度，不能转写成实际应用梯度有限；",
                      "旧审计的 `all_finite=false` 保持为 false。实际 applied/skip 数量未知，末端 scaler scale 不能证明精确跳步数量。"])
    lines.extend(["", "| coefficient branch | 更新参数数 | scaled 全部有限 | scaled 非有限 attempts | unscaled 全部有限 | unscaled 非有限 attempts | 实际应用全部有限 | 实际应用 steps | 实际应用非零 steps | 应用梯度及更新 gate |",
                  "|---|---:|---|---:|---|---:|---|---:|---:|---|"])
    for branch in BRANCHES:
        updates = at(run.audit, "branch_updates", branch) or {}
        gradients = updates.get("gradients", {})
        raw_scaled_finite = gradients.get("observed_all_scaled_gradients_finite") if optimizer else gradients.get("all_finite")
        values = [branch, cell(updates.get("changed_parameter_count")), cell(raw_scaled_finite),
                  cell(gradients.get("observed_nonfinite_scaled_gradient_steps")), cell(gradients.get("observed_all_gradients_finite")),
                  cell(gradients.get("observed_nonfinite_gradient_steps")), cell(gradients.get("all_applied_gradients_finite")),
                  cell(gradients.get("applied_steps")), cell(gradients.get("nonzero_steps") if optimizer else None),
                  verdict(branch_health(run, branch))]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "| ACD loss branch | loss calls | candidates | accepted actions | FP pixel terms | aux sum |",
                  "|---|---:|---:|---:|---:|---:|"])
    stats = run.complete.get("stats") or run.failed.get("stats") or {}
    for branch in BRANCHES:
        s = stats.get(branch, {})
        values = [branch, cell(s.get("loss_calls")), cell(s.get("candidate_count")),
                  cell(s.get("action_positive")), cell(s.get("pixel_terms")), fmt(s.get("aux_sum"))]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "baseline 关闭 action 项；其 accepted actions=0 不代表训练失败。ACD 的候选/接受数分别保留两个 loss branch 的实际记录。",
                  "冻结审计针对 live tensors；最终评价必须用 FP32 EMA coefficient 文件覆盖原始官方 FP32 模型。"])
    internal = run.complete.get("trainer_final_metrics") or run.evidence.read(run.path / "trainer_final_epoch_metrics.json")
    if internal:
        lines.extend(["", "## 内部末 epoch 指标", "", "这些指标来自训练时的转换标签验证范围；smoke/196 张内部验证指标与原始 COCO 全 5,000 张独立评价范围不同。", "",
                      "```json", json.dumps(internal, ensure_ascii=False, indent=2, allow_nan=False), "```"])
    return lines


def optimizer_comparison_report(comparison: dict[str, Any]) -> list[str]:
    lines = ["| 正式训练臂 | 选用 Run | attempted | applied | confirmed overflow skips | observed unscaled nonfinite attempts | EMA updates |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ("baseline", "acd"):
        detail = comparison.get(arm, {})
        lines.append("| " + " | ".join([arm, cell(detail.get("run")),
                     *[cell(detail.get(key)) for key in ("attempted_steps", "applied_steps", "skipped_overflow_steps", "observed_nonfinite_gradient_steps", "ema_updates")]]) + " |")
    lines.extend(["", "overflow-skip 次数差（ACD − baseline）："
                  + fmt(comparison.get("skipped_overflow_steps_delta_acd_minus_baseline"), signed=True, digits=0) + "。",
                  cell(comparison.get("limitation")),
                  "非有限梯度观测不改称 finite；计数差异不自动归因于 ACD。",
                  "attempt 的 epoch/batch/LR/accumulate/AMP schedule 必须一致；applied/overflow-skip 数量允许按官方 AMP 行为分别记录。",
                  "EMA 保持官方时机：每次 optimizer attempt 都更新，包括 scaler 跳过 actual optimizer step 的 attempt。"])
    return lines


def decision_report(decision: dict[str, Any]) -> list[str]:
    labels = {"positive": "本次可行性门槛通过", "negative": "本次可行性门槛未通过", "unknown": "未知 / 尚不能判定"}
    lines = [f"结论：**{labels[decision['outcome']]}**。{decision['reason']}", "",
             "主比较为 ACD − coefficient-only baseline，official 为额外参考。",
             "negative 表示未达到预设门槛，不能据此断言方法整体无效。", "",
             "| 协议 gate | 判定 |", "|---|---|"]
    for check in decision["criteria"]:
        lines.append(f"| {cell(check['name'])} | {verdict(check['passed'])} |")
    lines.extend(["", "未知不能当成通过。执行或 frozen/data/parity 核验无效时，不将 AP 差解释为训练迁移结论。",
                  "固定预算结束后停止；不自动加 seed、修改 auxiliary 设置或扩展训练。"])
    return lines


def failure_report(run: Run) -> list[str]:
    reason = run.meta.get("failure_observation") or run.failed.get("exception") or run.failed.get("traceback")
    lines = []
    if reason or run.meta.get("status") == "failed" or run.failed:
        lines.extend(["## 失败/重试事实", "", f"失败记录：{cell(reason)}。",
                      "失败及部分产物保留；未观测的退出码或原因保持未知。"])
    if "RETRY" in run.rid:
        lines.extend(["", f"这是独立重试 Run：{run.rid}；它不覆盖原尝试的状态或路径。"])
    return lines


def run_report(run: Run, decision: dict[str, Any]) -> str:
    lines = [f"# {run.rid} 报告", "", f"生成于 {decision['generated_at']}（Asia/Shanghai）。", "",
             "本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。", "",
             "| 记录项 | 当前值 |", "|---|---|",
             *[f"| {label} | {cell(run.meta.get(key))} |" for key, label in (
                 ("status", "Run status"), ("execution_status", "执行状态"), ("artifact_status", "产物状态"),
                 ("transfer_status", "回传状态"), ("exit_code", "exit code"), ("started_at", "开始时间"),
                 ("ended_at", "结束时间"), ("remote_directory", "远端实际目录"), ("interpreter", "解释器"))],
             "", f"本地 Run 目录：`{run.path.as_posix()}`。",
             f"transfer.json：{'可读' if run.transfer else '未发现/不可读'}；manifest.sha256：{'存在' if (run.path / 'manifest.sha256').is_file() else '未发现'}。"]
    lines.extend(["", "| 关键文件 | 当前本地可用性 |", "|---|---|"])
    files = ("TRAINING_COMPLETE.json", "TRAINING_AUDIT.json", "optimizer_step_audit.jsonl", "input_batch_hashes.jsonl",
             "coeff_final.pt", "coeff_final_ema.pt", "trainer/weights/last.pt", "SUMMARY.json")
    if run.rid == EVAL_RUN or run.summary:
        files += tuple(f"{arm}/predictions.json" for arm in MODEL_NAMES)
    for filename in files:
        lines.append(f"| `{filename}` | {'存在' if (run.path / filename).is_file() else '本地未发现'} |")
    lines.extend(["", "本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。"])
    if run.meta.get("unverified"):
        lines.extend(["", "run.json 的待验证项：" + "; ".join(cell(v) for v in run.meta["unverified"]) + "。"])
    lines.extend(failure_report(run))
    if run.setup or run.audit or run.complete or (run.failed and "stats" in run.failed):
        lines.extend(["", *training_report(run)])
    if run.summary:
        lines.extend(["", "## 独立评价", "", f"summary status={cell(run.summary.get('status'))}；",
                      f"图像={cell(run.summary.get('image_count'))}；覆盖全部 annotation images={cell(run.summary.get('covers_all_annotation_images'))}。",
                      f"范围：{cell(run.summary.get('evaluation_scope'))}。", "", *metric_table(run.summary)])
        if at(run.summary, "paired", "acd"):
            lines.extend(["", "## 实例配对诊断", "", *pair_report(run.summary)])
    if run.rid == EVAL_RUN:
        lines.extend(["", "## 预定义可行性判定", "", *decision_report(decision),
                      "", "## 配对训练数值轨迹", "", *optimizer_comparison_report(decision["optimizer_step_comparison"]),
                      "", "完整判定及来源 SHA256：`DECISION.json`。"])
    if run.progress:
        lines.extend(["", "## Pipeline 进度", "", "```json", json.dumps(run.progress, ensure_ascii=False, indent=2), "```"])
    if run.pipeline_complete:
        lines.extend(["", "Pipeline COMPLETE receipt：", "", "```json", json.dumps(run.pipeline_complete, ensure_ascii=False, indent=2), "```"])
    if not run.summary and not run.complete:
        lines.extend(["", "当前未发现完整训练/评价结果 receipt；任何未回传结果保持未知。"])
    lines.extend(["", "主要原始证据："])
    for name in ("run.json", "TRAINING_SETUP.json", "TRAINING_AUDIT.json", "TRAINING_COMPLETE.json", "TRAINING_FAILED.json",
                 "SUMMARY.json", "optimizer_step_audit.jsonl", "SOURCE.json", "OPTIMIZER_SCHEDULE_PARITY.json", "PROGRESS.json", "COMPLETE.json", "FAILED.json", "transfer.json", "manifest.sha256", "stdout.log"):
        if (run.path / name).is_file():
            lines.append(f"- [{name}]({name})")
    if run.source:
        snapshot = source_snapshot(run)
        lines.extend(["", "## 实际执行源码与协议快照", "",
                      f"执行 snapshot：`{cell(run.source.get('executed_snapshot'))}`；记录时间={cell(run.source.get('captured_at'))}。", "",
                      "| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |", "|---|---|---|"])
        for kind, label in (("script", "实际执行脚本"), ("protocol", "运行前协议")):
            detail = snapshot[kind]
            lines.append(f"| [{label}]({Path(detail['source']).relative_to(Path('runs') / run.rid).as_posix()}) | `{cell(detail['declared_sha256'])}` | {verdict(detail['verified'])} |")
        lines.extend(["", "此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。"])
    return "\n".join(lines)


def experiment_report(evidence: Evidence, runs: dict[str, Run], decision: dict[str, Any]) -> str:
    summary = runs[EVAL_RUN].summary if EVAL_RUN in runs else {}
    lines = ["# ACD native coefficient feasibility experiment", "", f"更新于 {decision['generated_at']}（Asia/Shanghai）。", "",
             *decision_report(decision), "", "## 固定范围与限制", "",
             "Ultralytics 8.4.100，官方 COCO `yolo26m-seg.pt`；仅训练原生 `cv4` / `one2one_cv4` coefficient heads，",
             "其余网络及所有 BatchNorm 冻结。baseline 与 ACD 从相同官方权重重启，seed=0，batch=2，workers=0，输入 640。",
             "两臂先各 1 epoch smoke，再各固定 3 epoch 正式训练；smoke 不是主 AP 结果。",
             "正式 fit 使用已有 796 张 train2017 cohort，训练内验证使用已有 196 张转换标签验证图像。",
             "最终采用固定第 3 epoch 的 EMA coefficients 覆盖原始官方 FP32 模型，保留官方 FP32 frozen 权重；",
             "不选 best epoch，不在推理时使用 action 或 GT，不修改 native one-to-one 后处理。",
             "独立原始 COCO val2017 全 5,000 张评价：FP32、640 square letterbox、conf=0.001、max_det=300、",
             "官方 mask decoder / 原图 scaling、COCO80→91 mapping、pycocotools。绝对 AP 可能不同于 stock rectangular-batch 验证器。", "",
             "fit cohort 和全验证集均复用历史研究图像，包含其选择限制；全 5,000 张验证集不是新的 blind test。",
             "这是一 seed、短预算的 coefficient-only feasibility screen，不能替代充分训练的结果，也不能量化训练 seed 的不确定性。",
             "ACD auxiliary 固定 weight=0.05、tau=0.50、min gain=0.01、min coverage=0.80、每图每 branch 最多 12 个候选；",
             "观察 paired 结果后不改变超参数。完整规则见 [PROTOCOL.md](PROTOCOL.md)。", "",
             "## 原始 COCO 全验证集结果", "", *metric_table(summary), "", "## 配对损伤与修复", "", *pair_report(summary),
             "", "## 配对训练数值轨迹", "", *optimizer_comparison_report(decision["optimizer_step_comparison"]),
             "", "## Run 记录", "", "| Run | arm / purpose | 执行状态 | 训练/评价 receipt | 回传状态 |", "|---|---|---|---|---|"]
    for rid, run in sorted(runs.items()):
        receipts = []
        if run.complete:
            receipts.append("训练 " + cell(run.complete.get("status")))
        if run.summary:
            receipts.append("评价 " + cell(run.summary.get("status")) + f" / {cell(run.summary.get('image_count'))} 图")
        if run.pipeline_complete:
            receipts.append("pipeline COMPLETE")
        lines.append(f"| [{rid}](runs/{rid}/REPORT.md) | {cell(run.meta.get('arm'))} / {cell(run.meta.get('purpose'))} | "
                     f"{cell(run.meta.get('execution_status'))} | {', '.join(receipts) if receipts else UNKNOWN} | {cell(run.meta.get('transfer_status'))} |")
    smoke = select_training(runs, "baseline", "SMOKE")
    lines.extend(["", "首次 detached SSH 尝试的 `RUN_BASELINE_SMOKE_S0` 保留为失败记录。该尝试在 TRAINING_SETUP 前终止，",
                  "没有训练指标；精确终止原因与退出码未观测。首次成功的 baseline smoke 是独立重试 Run `RUN_BASELINE_SMOKE_S0_RETRY1`；",
                  f"审计修正后本次配对流程采用的 baseline smoke：`{smoke.rid if smoke and smoke.complete else UNKNOWN}`。",
                  "上述失败事实以对应 run.json 的 failure_observation 为来源，报告不把重试成功回填为首次尝试成功。"])
    if decision["excluded_failed_training_runs"]:
        lines.extend(["", "未作为科学比较臂的失败正式训练 Run："
                      + ", ".join(f"`{rid}`" for rid in decision["excluded_failed_training_runs"]) + "。",
                      "失败审计及已生成的 checkpoint/部分产物继续保留；工程重试保持原 AMP/超参数与固定预算，不回填旧 Run 为成功。"])
    lines.extend(["", "## 核验与门槛判定", ""])
    independent = decision.get("independent_readout_verification", {})
    lines.extend([f"独立汇总核验：{verdict(truth(independent.get('passed')))}；"
                  f"[READOUT_VERIFICATION.json](runs/{EVAL_RUN}/READOUT_VERIFICATION.json)。",
                  "该核验从 COCO accumulated precision 重算 AP，并从逐图/逐 detection 记录重算配对计数与点估计；",
                  "不重新执行模型、GT 匹配或 bootstrap 区间计算。", ""])
    incomplete = [c for c in decision["criteria"] if c["passed"] is not True]
    if incomplete:
        for check in incomplete:
            lines.append(f"- {check['name']}：{verdict(check['passed'])}；证据 " + ", ".join(f"`{v}`" for v in check["sources"]) + "。")
    else:
        lines.append("所有预定义 gate 已有可读证据且通过；不开展额外训练。")
    if evidence.errors:
        lines.extend(["", "文件读取问题：", "", *[f"- {cell(error)}" for error in evidence.errors]])
    lines.extend(["", f"机器可读判定：[DECISION.json](runs/{EVAL_RUN}/DECISION.json)；",
                  "其中保留逐项三值判定、所选 Run、数值阈值、原始证据相对路径和 SHA256。",
                  "生成脚本：[build_report.py](scripts/build_report.py)。报告生成不会修改任何 Run metadata。"])
    return "\n".join(lines)


def generate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "PROTOCOL.md").is_file() and (root / "experiments" / EXPERIMENT).is_dir():
        root = root / "experiments" / EXPERIMENT
    if not root.is_dir():
        raise FileNotFoundError(root)
    evidence = Evidence(root)
    run_root = root / "runs"
    runs = {path.name: Run(path, evidence) for path in sorted(run_root.iterdir()) if path.is_dir()} if run_root.is_dir() else {}
    decision = build_decision(evidence, runs)
    # Reports may load additional sources (internal metrics); put their hashes
    # into the final decision only after rendering every report.
    for run in runs.values():
        write_text(run.path / "REPORT.md", run_report(run, decision))
    write_text(root / "REPORT.md", experiment_report(evidence, runs, decision))
    decision["read_errors"] = evidence.errors
    decision["evidence_sha256"] = dict(sorted(evidence.sources.items()))
    write_json(run_root / EVAL_RUN / "DECISION.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    decision = generate(args.root)
    print(json.dumps({"outcome": decision["outcome"], "gate_status": decision["gate_status"],
                      "reason": decision["reason"], "criteria": len(decision["criteria"]),
                      "read_errors": decision["read_errors"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
