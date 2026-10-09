"""Build evidence-based Chinese reports for the locked prototype-tail screen.

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
import posixpath
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

EXPERIMENT = "acd_proto_tail_unfreeze_20261006"
WORKSPACE = Path("C:/Dpan/codexproject/paper-disc-organized")
EXPERIMENT_ROOT = WORKSPACE / "experiments" / EXPERIMENT
EVAL_RUN = "RUN_PROTO_TAIL_PAIRED_EVAL_S0"
BRANCHES = ("one2many", "one2one", "proto_tail")
LOSS_BRANCHES = ("one2many", "one2one")
PREFIXES = ("model.23.cv4.", "model.23.one2one_cv4.", "model.23.proto.cv2.", "model.23.proto.cv3.")
GROUP_PREFIXES = {"one2many": PREFIXES[:1], "one2one": PREFIXES[1:2], "proto_tail": PREFIXES[2:]}
GROUP_PARAMETER_COUNTS = {"one2many": 854112, "one2one": 854112, "proto_tail": 598016}
PREFIX_STATE_COUNTS = dict(zip(PREFIXES, (42, 42, 6, 6)))
TRAINABLE_PARAMETER_COUNT = 2306240
PARENT_EXPERIMENT = "acd_native_coefficient_20261006"
PARENT_EVAL_RUN = "RUN_ACD_PAIRED_EVAL_S0"
NUMERICAL_RUN = "RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0"
LOADER_RUN = "RUN_PROTO_TAIL_LOADER_VERIFY_S0"
MODEL_NAMES = ("official", "baseline", "acd")
AP_KEYS = ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge")
EXPECTED_VERSION = "8.4.100"
EXPECTED_WEIGHT_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
PROTOCOL_FRAGMENTS = (
    "Ultralytics 8.4.100", "796", "5,000", "3 epochs", "third-epoch EMA",
    "0.003", "1%", ">= -0.005", "cv4", "one2one_cv4",
    "every gradient actually applied", "optimizer-attempt", "2,306,240",
    "model.23.proto.cv2", "model.23.proto.cv3", "one2one branch receives detached prototypes",
    "mask_coefficient_proto_tail_ema_final", "raw five-state taxonomy",
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


def display_time(value: Any) -> str:
    if value is None:
        return UNKNOWN
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone(timedelta(hours=8))).isoformat() if parsed.tzinfo else cell(value) + "（时区未知）"
    except ValueError:
        return cell(value)


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
        self.digest_cache: dict[Path, tuple[Any, str]] = {}

    def label(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def digest(self, path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            info = path.stat()
            signature = (info.st_size, info.st_mtime_ns, info.st_ino)
            cached = self.digest_cache.get(path)
            if cached and cached[0] == signature:
                return cached[1]
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ino) != signature:
                self.errors.append(f"{self.label(path)} 在哈希期间改变，保持未知")
                return None
            self.sources[self.label(path)] = digest
            self.digest_cache[path] = (signature, digest)
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
    base = f"RUN_PROTO_TAIL_{arm.upper()}_{stage}_S0"
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
    prefix_values = []
    for prefix in GROUP_PREFIXES[key]:
        prefix_update = at(detail, "prefix_updates", prefix)
        changed = number(at(prefix_update, "changed_parameter_count"))
        names = at(prefix_update, "changed_parameters")
        prefix_values.extend([
            changed > 0 if changed is not None else None,
            len(names) == changed and all(isinstance(name, str) and name.startswith(prefix) for name in names)
            if isinstance(names, list) and changed is not None else None,
        ])
    return combine([
        truth(at(detail, "gradients", "all_applied_gradients_finite")),
        truth(at(detail, "gradients", "all_observed_nonfinite_steps_skipped")),
        count > 0 if count is not None else None,
        steps > 0 if steps is not None else None,
        applied > 0 if applied is not None else None,
        0 <= raw_nonfinite == raw_skipped if raw_nonfinite is not None and raw_skipped is not None else None,
        *prefix_values,
    ])


def trainable_scope_health(run: Run) -> dict[str, Any]:
    names = run.setup.get("optimizer_parameters")
    groups = run.setup.get("trainable_parameter_groups", {})
    values = [equal(run.setup.get("trainable_parameter_count"), TRAINABLE_PARAMETER_COUNT),
              equal(run.setup.get("scope"), list(PREFIXES)), equal(run.audit.get("scope"), list(PREFIXES)),
              truth(run.setup.get("all_batchnorm_eval")), truth(run.setup.get("all_batchnorm_affine_frozen"))]
    group_names = []
    for group, prefixes in GROUP_PREFIXES.items():
        detail = groups.get(group, {})
        gn = detail.get("names")
        values.extend([equal(detail.get("prefixes"), list(prefixes)),
                       equal(detail.get("parameter_count"), GROUP_PARAMETER_COUNTS[group]),
                       all(isinstance(name, str) and name.startswith(prefixes) and ".bn." not in name for name in gn)
                       if isinstance(gn, list) and gn else None])
        if isinstance(gn, list):
            group_names.extend(gn)
    if isinstance(names, list) and names:
        audited = run.audit.get("trainable_parameters")
        initial = run.setup.get("initial_trainable_sha256")
        values.extend([len(names) == len(set(names)) == len(group_names) and set(names) == set(group_names),
                       set(names) == set(audited) if isinstance(audited, list) else None,
                       set(names) == set(initial) if isinstance(initial, dict) else None,
                       all(isinstance(name, str) and name.startswith(PREFIXES) and ".bn." not in name for name in names),
                       *[any(name.startswith(prefix) for name in names) for prefix in PREFIXES]])
    else:
        values.append(None)
    return {"verified": combine(values), "expected_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "actual_parameter_count": run.setup.get("trainable_parameter_count"),
            "scope": run.setup.get("scope"), "parameter_groups": groups,
            "optimizer_parameters": names, "all_batchnorm_eval": run.setup.get("all_batchnorm_eval"),
            "all_batchnorm_affine_frozen": run.setup.get("all_batchnorm_affine_frozen")}


def overlay_health(model: Any, run: Run | None) -> bool | None:
    keys = at(model, "overlay_state_keys")
    mutable = at(model, "overlay_non_bn_parameter_keys")
    bn = at(model, "overlay_bn_state_keys")
    actual_parameters = run.setup.get("optimizer_parameters") if run else None
    values = [truth(at(model, "reconstructed_mask_tail_overlay")), truth(at(model, "overlay_audit_passed")),
              truth(at(model, "all_overlay_bn_exact_official")), equal(at(model, "overlay_kind"), "mask_coefficient_proto_tail_ema_final"),
              equal(at(model, "overlay_epoch"), 3), equal(at(model, "overlay_base_weights_sha256"), EXPECTED_WEIGHT_SHA),
              equal(at(model, "overlay_scope_prefixes"), list(PREFIXES))]
    if all(isinstance(items, list) for items in (keys, mutable, bn)):
        values.extend([len(keys) == len(set(keys)) == 96, len(mutable) == len(set(mutable)) == 26,
                       len(bn) == len(set(bn)) == 70, set(keys) == set(mutable) | set(bn), not set(mutable) & set(bn),
                       set(mutable) == set(actual_parameters) if isinstance(actual_parameters, list) else None,
                       all(isinstance(key, str) and key.startswith(PREFIXES) and ".bn." in key for key in bn),
                       *[sum(key.startswith(prefix) for key in keys) == count for prefix, count in PREFIX_STATE_COUNTS.items()]])
    else:
        values.append(None)
    return combine(values)


def diagnostic_evidence(evidence: Evidence, runs: dict[str, Run], paired_source: dict[str, Any]) -> dict[str, Any]:
    run = runs.get(NUMERICAL_RUN)
    receipt = evidence.read(run.path / "TRAINING_VERIFICATION.json") if run else None
    receipt = receipt or {}
    snapshot = source_snapshot(run) if run else {}
    many = at(receipt, "checks", "one2many", "gradient_reachability") or {}
    one = at(receipt, "checks", "one2one", "gradient_reachability") or {}
    proto = at(many, "proto_tail", "nonzero_parameter_names")
    values = [equal(receipt.get("status"), "PASS"), equal(receipt.get("trainable_parameter_count"), TRAINABLE_PARAMETER_COUNT),
              equal(receipt.get("scope_prefixes"), list(PREFIXES)), equal(receipt.get("weight_sha256"), EXPECTED_WEIGHT_SHA),
              equal(receipt.get("production_acd_constants_changed"), False), equal(receipt.get("one2one_detach_changed"), False),
              truth(receipt.get("official_one2one_proto_detached")), truth(receipt.get("all_bn_eval_and_affine_frozen")),
              truth(receipt.get("frozen_live_and_ema_exact")), truth(receipt.get("audited_optimizer_matches_official")),
              truth(receipt.get("trainable_ema_matches_official")), truth(receipt.get("overlay_full_scope_exact")),
              truth(receipt.get("overlay_includes_frozen_bn_original_values")), truth(receipt.get("overlay_strict_reconstruction")),
              equal(receipt.get("overlay_training_claim"), False), truth(snapshot.get("verified")),
              equal(receipt.get("verification_script_sha256"), at(snapshot, "script", "observed_sha256")),
              bool(at(many, "one2many", "nonzero_parameter_names")) if at(many, "one2many", "nonzero_parameter_names") is not None else None,
              bool(at(one, "one2one", "nonzero_parameter_names")) if at(one, "one2one", "nonzero_parameter_names") is not None else None,
              equal(at(many, "one2one", "connected_parameter_names"), []),
              equal(at(one, "one2many", "connected_parameter_names"), []),
              equal(at(one, "proto_tail", "connected_parameter_names"), []),
              *[any(name.startswith(prefix) for name in proto) if isinstance(proto, list) else None for prefix in PREFIXES[2:]]]
    intrinsic = combine(values)
    formal_link = equal(receipt.get("training_script_sha256"), at(paired_source, "script_sha256", "baseline")) if at(paired_source, "script_sha256", "baseline") else None
    return {"verified": combine([intrinsic, formal_link]), "intrinsic_verified": intrinsic, "formal_training_source_link": formal_link,
            "run": NUMERICAL_RUN, "source": f"runs/{NUMERICAL_RUN}/TRAINING_VERIFICATION.json",
            "receipt": receipt, "snapshot": snapshot,
            "limitation": "真实官方模型的 fixture GT/assignment + 一步 FP32 诊断，只证明梯度/optimizer/EMA/overlay 合同；不等于正式训练或 AP 结果。"}


def loader_evidence(evidence: Evidence, runs: dict[str, Run], evaluation_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    run = runs.get(LOADER_RUN)
    receipt = evidence.read(run.path / "LOADER_VERIFICATION.json") if run else None
    receipt = receipt or {}
    snapshot = source_snapshot(run) if run else {}
    evaluator_sha = evidence.digest(run.path / "source" / "evaluate.py") if run else None
    expected_checks = ("valid_full_scope_fp32_overlay", "reject_legacy_kind", "reject_wrong_epoch", "reject_failed_audit",
                       "reject_wrong_base_hash", "reject_missing_key", "reject_extra_frozen_key", "reject_fp16",
                       "reject_nan", "reject_bn_change", "digest_detects_other_proto_semantic_and_bn_changes")
    values = [equal(receipt.get("status"), "PASS"), equal(receipt.get("training_claim"), False),
              equal(receipt.get("overlay_keys"), 96), equal(receipt.get("mutable_keys"), 26),
              equal(receipt.get("base_weights_sha256"), EXPECTED_WEIGHT_SHA), truth(snapshot.get("verified")),
              equal(receipt.get("evaluator_sha256"), evaluator_sha) if evaluator_sha else None,
              *[truth(at(receipt, "checks", key)) for key in expected_checks]]
    intrinsic = combine(values)
    formal_link = equal(receipt.get("evaluator_sha256"), at(evaluation_snapshot, "script", "observed_sha256")) if at(evaluation_snapshot, "script", "observed_sha256") else None
    return {"verified": combine([intrinsic, formal_link]), "intrinsic_verified": intrinsic, "formal_evaluation_source_link": formal_link,
            "run": LOADER_RUN, "source": f"runs/{LOADER_RUN}/LOADER_VERIFICATION.json",
            "receipt": receipt, "snapshot": snapshot, "evaluator_sha256": evaluator_sha,
            "limitation": "synthetic loader fixture 的 epoch=3 是合同字段，不是三轮训练事实；只验证正确/错误 payload 的接受与拒绝。"}


def collection_health(evidence: Evidence, run: Run | None) -> dict[str, Any]:
    if run is None:
        return {"verified": None, "run": None}
    marker = evidence.digest(run.path / "_source_manifest.sha256")
    receipt_status, meta_status = run.transfer.get("status"), run.meta.get("transfer_status")
    # Planned/running/not-yet-collected evidence is pending, not a failed transfer.
    ready = True if receipt_status == meta_status == "verified" else False if receipt_status in ("failed", "error") else None
    return {"verified": combine([ready, equal(run.transfer.get("source_manifest_sha256"), marker) if marker else None]),
            "run": run.rid, "source_manifest_sha256": marker, "transfer_status": run.transfer.get("status"),
            "sources": [run.ref("transfer.json"), run.ref("_source_manifest.sha256"), run.ref("run.json")]}


def data_receipt_health(evidence: Evidence, receipt: dict[str, Any]) -> dict[str, Any]:
    values = [equal(at(receipt, "sources", "fit", "count"), 796), equal(at(receipt, "sources", "val", "count"), 196),
              equal(receipt.get("smoke_train_images"), 32), equal(receipt.get("smoke_val_images"), 8),
              equal(receipt.get("final_evaluation_images"), 5000)]
    lists = {}
    for name, expected_count in (("fit.txt", 796), ("val.txt", 196), ("smoke_train.txt", 32), ("smoke_val.txt", 8), ("val_full.txt", 5000)):
        path = evidence.root / "data" / name
        digest = evidence.digest(path)
        count = None
        if path.is_file():
            try:
                count = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines())
            except (OSError, UnicodeError) as error:
                evidence.errors.append(f"{evidence.label(path)} 无法计数：{error}")
        lists[name] = {"sha256": digest, "images": count, "expected_images": expected_count}
        values.extend([equal(at(receipt, "lists", name), digest) if digest else None,
                       equal(count, expected_count)])
    return {"verified": combine(values), "lists": lists, "receipt": receipt,
            "source": "data/DATA_RECEIPT.json"}


def dataset_overlap(evidence: Evidence) -> dict[str, Any]:
    """List-based identities only; never open a listed remote image path."""
    receipt = evidence.read(evidence.root / "data" / "DATA_RECEIPT.json") or {}
    identities, lists = {}, {}
    for name in ("fit.txt", "dev.txt", "val.txt", "val_full.txt", "smoke_train.txt", "smoke_val.txt"):
        path = evidence.root / "data" / name
        rows = None
        if path.is_file():
            try:
                rows = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            except (OSError, UnicodeError) as error:
                evidence.errors.append(f"{evidence.label(path)} 无法读取列表交集：{error}")
        digest = evidence.digest(path)
        paths, ids = None, None
        if rows is not None:
            paths = {posixpath.normpath(row.replace("\\", "/")).casefold() for row in rows}
            stems = [Path(row.replace("\\", "/")).stem for row in rows]
            ids = {int(stem) for stem in stems} if all(stem.isdigit() for stem in stems) else None
        identities[name] = (paths, ids)
        lists[name] = {"source": evidence.label(path), "sha256": digest,
                       "receipt_sha256": at(receipt, "lists", name),
                       "receipt_hash_match": equal(at(receipt, "lists", name), digest) if digest else None,
                       "rows": len(rows) if rows is not None else None,
                       "unique_paths": len(paths) if paths is not None else None,
                       "unique_numeric_image_ids": len(ids) if ids is not None else None}
    full_paths, full_ids = identities["val_full.txt"]
    for name, (paths, ids) in identities.items():
        lists[name]["fullval_path_overlap"] = len(paths & full_paths) if paths is not None and full_paths is not None else None
        lists[name]["fullval_image_id_overlap"] = len(ids & full_ids) if ids is not None and full_ids is not None else None
        lists[name]["overlap_id_examples"] = sorted(ids & full_ids)[:8] if ids is not None and full_ids is not None else None
    pairs = []
    for left, right in (("fit.txt", "dev.txt"), ("fit.txt", "val.txt"), ("dev.txt", "val.txt")):
        lp, li = identities[left]
        rp, ri = identities[right]
        pairs.append({"sets": [left, right], "image_id_overlap": len(li & ri) if li is not None and ri is not None else None,
                      "path_overlap": len(lp & rp) if lp is not None and rp is not None else None})
    return {"source": "data/DATA_RECEIPT.json", "lists": lists, "fit_dev_internalval_pairs": pairs,
            "method": "Numeric COCO image ID from filename stem; path strings normalize slash/dot segments and Windows case; no image-byte comparison",
            "fullval_not_in_internalval_ids": len(full_ids - identities["val.txt"][1]) if full_ids is not None and identities["val.txt"][1] is not None else None,
            "limitation": "列表交集是派生披露，不改变固定 gate。内部验证复用不等于 fit/eval 图像泄漏；完整 5000 是原始 COCO 读出流程，不能声明新的 blind heldout。"}


def dataset_overlap_report(overlap: dict[str, Any]) -> list[str]:
    labels = {"fit.txt": "fit / 训练", "dev.txt": "dev / 未用于拟合", "val.txt": "internal-val / 每 epoch 验证",
              "smoke_train.txt": "smoke train", "smoke_val.txt": "smoke val"}
    lines = ["## 数据列表交集与评价限制", "",
             "按实际列表的 COCO 文件名数值 ID 和规范化路径字符串分别计算；未读取远端图片，也未核对图片字节。", "",
             "| 集合 | 条目数 | 与 full-val 5000 的图像 ID 交集 | 路径字符串交集 | DATA_RECEIPT SHA256 匹配 |",
             "|---|---:|---:|---:|---|"]
    for name, label in labels.items():
        detail = overlap["lists"][name]
        lines.append(f"| [{label}]({detail['source']}) | {cell(detail['rows'])} | {cell(detail['fullval_image_id_overlap'])} | "
                     f"{cell(detail['fullval_path_overlap'])} | {verdict(detail['receipt_hash_match'])} |")
    fit, internal, smoke, full = (overlap["lists"][name] for name in ("fit.txt", "val.txt", "smoke_val.txt", "val_full.txt"))
    full_confirmed = full["rows"] == full["unique_numeric_image_ids"] == 5000 and full["receipt_hash_match"] is True
    if fit["fullval_image_id_overlap"] == 0 and fit["rows"] == fit["unique_numeric_image_ids"] == 796 and fit["receipt_hash_match"] is True and full_confirmed:
        lines.extend(["", "现有列表可确认：fit 的 796 个图像 ID 与最终评估的 5000 个 ID 没有交集；",
                      "本轮列表没有显示把这些评估图像用于 fit 的证据。"])
    if internal["fullval_image_id_overlap"] == 196 and smoke["fullval_image_id_overlap"] == 8 and full_confirmed:
        lines.extend(["", "内部验证的 196 张全部包含于 full-val（3.92%）；smoke 验证的 8 张也包含于其中。",
                      "converted-cache 与原始 COCO 图像目录不同，因此路径交集为 0 不能证明这些图像独立；同一 COCO ID 应视为重复图像身份。",
                      "这是内部验证复用，不等于 fit 泄漏。固定第 3 epoch、不挑 best 保留原设计，但 full-val 不能声称新的 blind heldout。"])
    lines.extend(["", "“独立评价”指原始 COCO annotations/decoder 的单独测量流程，不表示与内部验证、历史研究信息独立。",
                  "缺失列表或不可解析的 ID 保持未知，不把缺失计为零交集。交集数、列表 SHA256 与来源路径保存在派生 DECISION.json；",
                  "没有创建新的划分列表，也没有增加训练或评估。"])
    return lines


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
        equal(run.setup.get("optimizer"), "MuSGD"), equal(run.setup.get("amp"), True),
        equal(run.complete.get("workers"), 0),
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
        criterion(f"{arm} 四个 trainable prefixes/2,306,240 参数/全 BN 冻结", trainable_scope_health(run)["verified"],
                  trainable_scope_health(run), [run.ref("TRAINING_SETUP.json"), run.ref("TRAINING_AUDIT.json")]),
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
        *[criterion(f"{arm} {branch} 实际应用梯度有限且每个 prefix 更新", branch_health(run, branch),
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


def arm_readout_links(evidence: Evidence, run: Run | None, summary: dict[str, Any]) -> dict[str, Any]:
    configuration = summary.get("configuration", {})
    ids = configuration.get("image_ids")
    links = {}
    for arm in MODEL_NAMES:
        receipt, metrics = at(summary, "prediction_receipts", arm) or {}, at(summary, "metrics", arm) or {}
        fingerprint = receipt.get("fingerprint", {})
        common = {key: value for key, value in fingerprint.items() if key != "weights"}
        local_receipt = evidence.read(run.path / arm / "COMPLETE.json") if run else None
        local_metrics = evidence.read(run.path / arm / "COCO_METRICS.json") if run else None
        digest = evidence.digest(run.path / arm / "predictions.json") if run else None
        links[arm] = {"predictions_sha256": digest, "verified": combine([
            equal(receipt.get("status"), "prediction_complete"), equal(receipt.get("image_count"), 5000),
            common == configuration if common and configuration else None,
            local_receipt == receipt if local_receipt and receipt else None,
            local_metrics == metrics if local_metrics and metrics else None,
            equal(metrics.get("image_ids"), sorted(ids)) if isinstance(ids, list) else None,
            equal(metrics.get("backend"), "pycocotools.COCOeval"),
            equal(receipt.get("predictions_sha256"), digest) if digest else None,
            equal(metrics.get("predictions_sha256"), digest) if digest else None,
        ])}
    return links


def independent_link(evidence: Evidence, run: Run | None) -> dict[str, Any]:
    receipt = evidence.read(run.path / "READOUT_VERIFICATION.json") if run else None
    receipt = receipt or {}
    summary_sha = evidence.digest(run.path / "SUMMARY.json") if run else None
    values = [truth(receipt.get("passed")), equal(receipt.get("status"), "passed"),
              equal(receipt.get("errors"), []), equal(receipt.get("missing"), []),
              equal(at(receipt, "evidence", "SUMMARY.json", "sha256"), summary_sha) if summary_sha else None]
    bound = {}
    for relative, item in (receipt.get("evidence") or {}).items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not run or not (run.path / path).resolve().is_relative_to(run.path.resolve()):
            bound[relative] = {"verified": False, "reason": "evidence path escapes evaluation Run"}
            values.append(False)
            continue
        observed = evidence.digest(run.path / path)
        verified = equal(at(item, "sha256"), observed) if observed else None
        bound[relative] = {"observed_sha256": observed, "verified": verified}
        values.append(verified)
    if not bound:
        values.append(None)
    return {"verified": combine(values), "receipt": receipt, "current_evidence_links": bound,
            "source": evidence.label(run.path / "READOUT_VERIFICATION.json") if run else None}


def parent_comparison(evidence: Evidence, summary: dict[str, Any], execution_valid: bool | None) -> dict[str, Any]:
    parent_root = evidence.root.parent / PARENT_EXPERIMENT
    parent_run = Run(parent_root / "runs" / PARENT_EVAL_RUN, evidence)
    old = parent_run.summary
    old_data = evidence.read(parent_root / "data" / "DATA_RECEIPT.json") or {}
    new_data = evidence.read(evidence.root / "data" / "DATA_RECEIPT.json") or {}
    old_links = arm_readout_links(evidence, parent_run, old)
    old_independent = independent_link(evidence, parent_run)
    old_snapshot = source_snapshot(parent_run)
    parent_valid = combine([equal(old.get("status"), "complete"), equal(old.get("image_count"), 5000),
                            truth(old.get("covers_all_annotation_images")), old_independent["verified"],
                            truth(old_snapshot.get("verified")),
                            equal(at(old, "configuration", "evaluator_sha256"), at(old_snapshot, "script", "observed_sha256")),
                            *[item["verified"] for item in old_links.values()]])
    comparisons = []
    def same(name: str, new: Any, previous: Any) -> None:
        known = new is not None and previous is not None
        comparisons.append({"name": name, "same": new == previous if known else None,
                            "new": new, "coefficient_only": previous})
    for arm in MODEL_NAMES:
        key = "checkpoint" if arm == "official" else "base_weights"
        new_weight = at(summary, "prediction_receipts", arm, "fingerprint", "weights", key, "sha256")
        old_weight = at(old, "prediction_receipts", arm, "fingerprint", "weights", key, "sha256")
        same(f"{arm} official initialization SHA256", new_weight, old_weight)
    for name in ("fit.txt", "val.txt", "val_full.txt"):
        new_hash, old_hash = at(new_data, "lists", name), at(old_data, "lists", name)
        same(f"{name} list SHA256", new_hash, old_hash)
        same(f"{name} current local files match receipt",
             evidence.digest(evidence.root / "data" / name) == new_hash if new_hash else None,
             evidence.digest(parent_root / "data" / name) == old_hash if old_hash else None)
        if new_hash:
            comparisons.append({"name": f"new {name} verified", "same": equal(evidence.digest(evidence.root / "data" / name), new_hash)})
        if old_hash:
            comparisons.append({"name": f"parent {name} verified", "same": equal(evidence.digest(parent_root / "data" / name), old_hash)})
    for key in ("config", "image_ids", "vendor_source_sha256", "ultralytics", "torch"):
        same(f"evaluation {key}", at(summary, "configuration", key), at(old, "configuration", key))
    for key in ("annotations", "images_list"):
        same(f"evaluation {key} SHA256", at(summary, "configuration", key, "sha256"), at(old, "configuration", key, "sha256"))
    same("original official prediction SHA256", at(summary, "prediction_receipts", "official", "predictions_sha256"),
         at(old, "prediction_receipts", "official", "predictions_sha256"))
    comparable = combine([execution_valid, parent_valid, *[item["same"] for item in comparisons]])
    deltas = {}
    for key in AP_KEYS:
        nb, na = number(at(summary, "metrics", "baseline", "segm", key)), number(at(summary, "metrics", "acd", "segm", key))
        pb, pa = number(at(old, "metrics", "baseline", "segm", key)), number(at(old, "metrics", "acd", "segm", key))
        current = na - nb if na is not None and nb is not None else None
        previous = pa - pb if pa is not None and pb is not None else None
        deltas[key] = {"prototype_tail_ACD_effect": current, "coefficient_only_ACD_effect": previous,
                       "ACD_effect_interaction": current - previous if comparable is True and current is not None and previous is not None else None}
    return {"comparable": comparable, "parent_readout_verified": parent_valid, "checks": comparisons,
            "parent_run": PARENT_EVAL_RUN, "source": evidence.label(parent_run.path / "SUMMARY.json"),
            "parent_metrics": old.get("metrics", {}), "parent_readout_links": old_links,
            "parent_independent_readout": old_independent, "segm_effects": deltas,
            "limitation": "仅同一原始 COCO 全验证集、同初始化/列表/推理配置的描述性交互；单 seed，非显著性或独立因果分解。不纳入 best.pt 或其它历史改法的收益。"}


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
    numerical = diagnostic_evidence(evidence, runs, paired_source)
    loader = loader_evidence(evidence, runs, snapshots["evaluation"])
    checks.append(criterion("实际模型 one2many 原型尾部梯度/one2one detach/官方单步诊断", numerical["verified"], numerical, [numerical["source"]]))
    checks.append(criterion("完整 prototype-tail FP32 overlay 的独立 loader 合同", loader["verified"], loader, [loader["source"]]))
    collected = {arm: collection_health(evidence, run) for arm, run in (
        ("baseline", baseline), ("acd", acd), ("evaluation", eval_run),
        ("numerical", runs.get(NUMERICAL_RUN)), ("loader", runs.get(LOADER_RUN)))}
    checks.append(criterion("正式训练/评估/诊断已完整回传且原 source manifest 保留", combine([item["verified"] for item in collected.values()]),
                            collected, [source for item in collected.values() for source in item.get("sources", [])]))
    parity = data_parity(evidence, runs, baseline, acd, "FEASIBILITY_PARITY.json")
    checks.append(criterion("配对 transformed-input 完整流一致", parity["verified"], parity, parity["sources"]))
    schedule = optimizer_schedule_parity(evidence, runs, baseline, acd)
    checks.append(criterion("两臂 optimizer attempt 的 batch/LR/accumulate/AMP schedule 一致", schedule["verified"], schedule, schedule["sources"]))
    complete_full = combine([equal(summary.get("status"), "complete"), equal(summary.get("image_count"), 5000),
                             truth(summary.get("covers_all_annotation_images")), equal(summary.get("baseline"), "baseline")])
    configuration = at(summary, "configuration") or {}
    data_receipt = evidence.read(evidence.root / "data" / "DATA_RECEIPT.json") or {}
    data_health = data_receipt_health(evidence, data_receipt)
    checks.append(criterion("DATA_RECEIPT 与 fit/internal-val/smoke/full-val 列表哈希及数量", data_health["verified"], data_health,
                            ["data/DATA_RECEIPT.json", *[f"data/{name}" for name in data_health["lists"]]]))
    image_ids = configuration.get("image_ids")
    complete_full = combine([
        complete_full,
        equal(at(configuration, "annotations", "sha256"), data_receipt.get("annotation_sha256")),
        equal(at(configuration, "images_list", "sha256"), at(data_receipt, "lists", "val_full.txt")),
        equal(evidence.digest(evidence.root / "data" / "val_full.txt"), at(data_receipt, "lists", "val_full.txt")),
        len(image_ids) == len(set(image_ids)) == 5000 if isinstance(image_ids, list) else None,
    ])
    readout_links = arm_readout_links(evidence, eval_run, summary)
    complete_full = combine([complete_full, *[item["verified"] for item in readout_links.values()]])
    independent_chain = independent_link(evidence, eval_run)
    independent, independent_valid = independent_chain["receipt"], independent_chain["verified"]
    complete_full = combine([complete_full, independent_valid])
    checks.append(criterion("完整 5,000 张原始 COCO val2017 独立评估", complete_full,
                            {"evaluation": {key: summary.get(key) for key in ("status", "image_count", "covers_all_annotation_images", "evaluation_scope")},
                             "annotations": configuration.get("annotations"), "images_list": configuration.get("images_list"),
                             "per_arm_readout_links": readout_links, "independent_aggregate_readout_verified": independent_valid},
                            [ref, "data/DATA_RECEIPT.json", "data/val_full.txt", f"runs/{EVAL_RUN}/READOUT_VERIFICATION.json"]))
    config = configuration.get("config", {})
    model_receipts = at(summary, "prediction_receipts") or {}
    reconstruction = [equal(configuration.get("ultralytics"), EXPECTED_VERSION),
                      equal(configuration.get("overlay_scope_prefixes"), list(PREFIXES)),
                      equal(config.get("half"), False), equal(config.get("imgsz"), 640),
                      equal(config.get("conf"), .001), equal(config.get("max_det"), 300),
                      equal(config.get("native_one2one"), True), equal(config.get("tf32"), False),
                      equal(config.get("rect"), False), equal(config.get("scaleup"), False),
                      equal(config.get("augment"), False), equal(config.get("coco_maxDets"), [1, 10, 100]),
                      equal(config.get("decode"), "SegmentationValidator.save_json=True: process_mask_native; scale_preds.byte()")]
    coefficient_hashes = {}
    for arm in ("baseline", "acd"):
        training_run = baseline if arm == "baseline" else acd
        coefficient_hashes[arm] = evidence.digest(training_run.path / "mask_final_ema.pt") if training_run else None
        reconstruction.extend([
            overlay_health(at(model_receipts, arm, "model"), training_run),
            equal(at(model_receipts, arm, "model", "coefficient_epoch"), 3),
            equal(at(model_receipts, arm, "model", "coefficient_kind"), "mask_coefficient_proto_tail_ema_final"),
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
    checks.append(criterion("固定第 3 epoch EMA 四-prefix mask-state overlay 官方 FP32 推理", combine(reconstruction),
                            {"config": config, "baseline": at(model_receipts, "baseline", "model"),
                             "acd": at(model_receipts, "acd", "model"),
                             "selected_formal_mask_sha256": coefficient_hashes,
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
    comparison = parent_comparison(evidence, summary, verified)
    return {"generated_at": timestamp(), "experiment": EXPERIMENT, "evaluation_run": EVAL_RUN,
            "outcome": outcome, "gate_status": gate, "reason": reason,
            "comparison": "acd minus the coefficient-plus-prototype-tail baseline", "protocol": lock,
            "training_runs": {"baseline": baseline.rid if baseline else None, "acd": acd.rid if acd else None},
            "optimizer_step_comparison": optimizer_comparison(baseline, acd),
            "optimizer_schedule_parity": schedule,
            "source_snapshots": snapshots,
            "training_gradient_diagnostic": numerical, "loader_contract_diagnostic": loader,
            "data_receipt_verification": data_health, "collection_verification": collected,
            "dataset_overlap": dataset_overlap(evidence),
            "coefficient_only_comparison": comparison,
            "unmeasured_diagnostics": {"full_raw_five_state_taxonomy": None, "pixel_AUC_FPR": None, "crop_support_ceiling": None},
            "independent_readout_verification": {"passed": independent_valid, "source": f"runs/{EVAL_RUN}/READOUT_VERIFICATION.json",
                                                 "limitations": independent.get("limitations"), "current_evidence_links": independent_chain["current_evidence_links"]},
            "excluded_failed_training_runs": [run.rid for run in runs.values()
                                                if "FEASIBILITY" in run.rid and (run.failed or run.meta.get("execution_status") == "failed")],
            "selected_epoch": 3, "seed": 0, "fit_images": 796, "final_validation_images": 5000,
            "blind_test": False, "AP_confidence_intervals_computed": False,
            "criteria": checks, "segm_delta": deltas,
            "read_errors": evidence.errors, "evidence_sha256": dict(sorted(evidence.sources.items())),
            "next_action": "Stop at the fixed budget. Do not add seeds, tune auxiliary settings, or expand training automatically."}


def metric_table(summary: dict[str, Any]) -> list[str]:
    lines = ["COCO AP 以百分数展示，差值为百分点；未知表示缺失、未回传或非有限值。", "",
             "| 评价 | 模型/比较 | AP50:95 | AP50 | AP75 | APsmall | APmedium | APlarge |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    metrics = summary.get("metrics", {})
    for iou_type, label in (("segm", "Mask"), ("bbox", "Box")):
        for name in MODEL_NAMES:
            lines.append(f"| {label} | {name} | " + " | ".join(fmt(at(metrics, name, iou_type, key), points=True, digits=3) for key in AP_KEYS) + " |")
        for method, reference in (("acd", "baseline"), ("acd", "official"), ("baseline", "official")):
            values = []
            for key in AP_KEYS:
                a, b = number(at(metrics, method, iou_type, key)), number(at(metrics, reference, iou_type, key))
                values.append(fmt(a-b if a is not None and b is not None else None, points=True, signed=True, digits=3))
            lines.append(f"| {label} | {method} − {reference} | " + " | ".join(values) + " |")
    lines.extend(["", "不提供 AP 置信区间；bootstrap 仅用于图像簇配对的 IoU/损伤诊断。",
                  "原型尾部学习和 baseline 普通微调都可能改变 Mask AP；ACD − official 也保留以显示微调是否退化。"])
    return lines


def comparison_report(comparison: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    lines = [f"与 coefficient-only 原始 COCO Run 的可比性：{verdict(comparison.get('comparable'))}。",
             f"原始 Run={comparison['parent_run']}；其独立读出链核验={verdict(comparison.get('parent_readout_verified'))}。", "",
             "原始数据来源：`" + comparison["source"] + "`。这里只使用该固定末 epoch 的原始 COCO 结果。",
             "不纳入 best.pt 或其他历史方法的 AP 收益。", "",
             "| 条件与模型 | Mask AP | AP50 | AP75 | APsmall | APmedium | APlarge |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scope, metrics in (("coefficient-only", comparison.get("parent_metrics", {})), ("coefficient+proto-tail", summary.get("metrics", {}))):
        for arm in MODEL_NAMES:
            values = [fmt(at(metrics, arm, "segm", key), points=True, digits=3) for key in AP_KEYS]
            lines.append(f"| {scope} / {arm} | " + " | ".join(values) + " |")
    for field, label in (("coefficient_only_ACD_effect", "coefficient-only Δ_ACD=ACD−baseline"),
                         ("prototype_tail_ACD_effect", "proto-tail Δ_ACD=ACD−baseline"),
                         ("ACD_effect_interaction", "Δ_ACD 交互=proto-tail效应−coefficient-only效应")):
        values = [fmt(at(comparison, "segm_effects", key, field), points=True, signed=True, digits=4) for key in AP_KEYS]
        lines.append(f"| {label}（百分点） | " + " | ".join(values) + " |")
    lines.extend(["", "可比性要求初始化 SHA256、fit/internal-val/full-val 列表及实际本地列表哈希、原始 annotations、图像 ID、",
                  "完整推理配置、版本/官方 decoder 源码与 original-official predictions SHA256 一致；",
                  "且本轮执行有效、原实验三个臂的 prediction/COCO metrics/独立读出链均已核验。",
                  "缺失或不一致时交互保持未知；上表各自原始数值不自动构成跨条件可比结论。", comparison['limitation']])
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
    for branch in LOSS_BRANCHES:
        s = stats.get(branch, {})
        values = [branch, cell(s.get("loss_calls")), cell(s.get("candidate_count")),
                  cell(s.get("action_positive")), cell(s.get("pixel_terms")), fmt(s.get("aux_sum"))]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "baseline 关闭 action 项；其 accepted actions=0 不代表训练失败。ACD 的候选/接受数分别保留两个 loss branch 的实际记录。",
                  "proto_tail 是参数/梯度组，不是第三个 ACD loss branch；one2one prototype 保持 detach。",
                  "冻结审计覆盖 live/EMA 所有其它 state 与所有 BN；最终评价用 FP32 EMA coefficient+proto-tail state 覆盖原始官方 FP32 模型。"])
    lines.extend(["", "## 可训练范围及每个 prefix 的实际变化", "",
                  f"参数总数：实际={cell(run.setup.get('trainable_parameter_count'))}，官方 m 预期={TRAINABLE_PARAMETER_COUNT}。", "",
                  "| 参数组 | prefix | changed parameters | 该 prefix 预期完整 overlay state keys |",
                  "|---|---|---:|---:|"])
    for group, prefixes in GROUP_PREFIXES.items():
        for prefix in prefixes:
            changed = at(run.audit, "branch_updates", group, "prefix_updates", prefix, "changed_parameter_count")
            lines.append(f"| {group} | `{prefix}` | {cell(changed)} | {PREFIX_STATE_COUNTS[prefix]} |")
    lines.extend(["", "正式 payload 四 prefix 共 96 个 state keys：26 个可变 non-BN 参数与 70 个原始冻结 BN state；",
                  "训练 count 和 names 与 loader 的 exact keys/dtype/BN 原始值合同分别核验。", "",
                  "| 初始 optimizer group | LR | Muon | 参数数 | proto-tail prefixes |", "|---|---:|---|---:|---|"])
    for group in run.setup.get("optimizer_parameter_groups", []):
        names = group.get("names", [])
        tail = [prefix for prefix in PREFIXES[2:] if any(name.startswith(prefix) for name in names)]
        lines.append(f"| {cell(group.get('index'))} / {cell(group.get('parameter_group'))} | {fmt(group.get('learning_rate'), digits=8)} | "
                     f"{cell(group.get('use_muon'))} | {len(names)} | {cell(', '.join(tail))} |")
    lines.extend(["", "保留官方 MuSGD 名称规则：prototype cv3 获得 3× group LR，cv2 为普通 group LR。",
                  "官方全参数 clip_grad_norm_(..., 10) 的联合范数会随可训练原型尾部增大；可能影响系数梯度缩放及 AMP overflow 轨迹。",
                  "这些是本次可训练范围干预的一部分，不能把结果解释成纯粹的 prototype 表示能力证书。"])
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
             "主比较为 ACD − 同范围 coefficient+proto-tail baseline；official 为额外参考。",
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
             *[f"| {label} | {display_time(run.meta.get(key)) if key in ('started_at', 'ended_at') else cell(run.meta.get(key))} |" for key, label in (
                 ("status", "Run status"), ("execution_status", "执行状态"), ("artifact_status", "产物状态"),
                 ("transfer_status", "回传状态"), ("exit_code", "exit code"), ("started_at", "包裹层开始时间"),
                 ("ended_at", "包裹层结束时间"), ("remote_directory", "远端实际目录"), ("interpreter", "解释器"))],
             "", f"本地 Run 目录：`{run.path.as_posix()}`。",
             f"transfer.json：{'可读' if run.transfer else '未发现/不可读'}；manifest.sha256：{'存在' if (run.path / 'manifest.sha256').is_file() else '未发现'}。"]
    lines.extend(["", "| 关键文件 | 当前本地可用性 |", "|---|---|"])
    files = ("TRAINING_COMPLETE.json", "TRAINING_AUDIT.json", "optimizer_step_audit.jsonl", "input_batch_hashes.jsonl",
             "mask_final.pt", "mask_final_ema.pt", "trainer/weights/last.pt", "SUMMARY.json")
    if run.rid == EVAL_RUN or run.summary:
        files += tuple(f"{arm}/predictions.json" for arm in MODEL_NAMES)
    for filename in files:
        lines.append(f"| `{filename}` | {'存在' if (run.path / filename).is_file() else '本地未发现'} |")
    lines.extend(["", "本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。"])
    if run.meta.get("unverified"):
        lines.extend(["", "run.json 的待验证项：" + "; ".join(cell(v) for v in run.meta["unverified"]) + "。"])
    if run.meta.get("metadata_reconciliation_reason"):
        lines.extend(["", "## 元数据恢复与仍未知的时间", "", cell(run.meta["metadata_reconciliation_reason"]),
                      "诊断状态依据 observed exit code 与 preserved PASS receipt 恢复；精确 wrapper 开始/结束仍保持未知。",
                      f"receipt 自身诊断开始={display_time(run.meta.get('diagnostic_started_at'))}；诊断完成={display_time(run.meta.get('diagnostic_completed_at'))}（Asia/Shanghai）。"])
    diagnostic = None
    for filename in ("TRAINING_VERIFICATION.json", "LOADER_VERIFICATION.json"):
        receipt = run.evidence.read(run.path / filename)
        if receipt:
            diagnostic = receipt
            lines.extend(["", "## 独立诊断 receipt", "", f"[{filename}]({filename})：status={cell(receipt.get('status'))}。",
                          "这是梯度/optimizer/EMA 或 synthetic loader 合同诊断；不是正式三轮训练、不是 AP 结果。",
                          f"receipt 开始={display_time(receipt.get('started_at_utc'))}；结束={display_time(receipt.get('completed_at_utc') or receipt.get('completed_at'))}（Asia/Shanghai）。",
                          "包裹层时间和 receipt 时间分别保留，缺失保持未知。"])
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
    if not run.summary and not run.complete and not diagnostic:
        lines.extend(["", "当前未发现完整训练/评价结果 receipt；任何未回传结果保持未知。"])
    lines.extend(["", "主要原始证据："])
    for name in ("run.json", "TRAINING_SETUP.json", "TRAINING_AUDIT.json", "TRAINING_COMPLETE.json", "TRAINING_FAILED.json",
                 "SUMMARY.json", "TRAINING_VERIFICATION.json", "LOADER_VERIFICATION.json", "optimizer_step_audit.jsonl", "SOURCE.json", "OPTIMIZER_SCHEDULE_PARITY.json", "PROGRESS.json", "COMPLETE.json", "FAILED.json", "transfer.json", "manifest.sha256", "stdout.log"):
        if (run.path / name).is_file():
            lines.append(f"- [{name}]({name})")
    if run.source:
        snapshot = source_snapshot(run)
        lines.extend(["", "## 实际执行源码与协议快照", "",
                      f"SOURCE.json 记录的执行版本：`{cell(run.source.get('executed_snapshot'))}`；记录时间={display_time(run.source.get('captured_at'))}（Asia/Shanghai）。", "",
                      "| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |", "|---|---|---|"])
        for kind, label in (("script", "实际执行脚本"), ("protocol", "运行前协议")):
            detail = snapshot[kind]
            lines.append(f"| [{label}]({Path(detail['source']).relative_to(Path('runs') / run.rid).as_posix()}) | `{cell(detail['declared_sha256'])}` | {verdict(detail['verified'])} |")
        lines.extend(["", "此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。"])
        if run.source.get("execution_mode"):
            lines.extend(["", "执行方式记录：" + cell(run.source["execution_mode"])])
    return "\n".join(lines)


def experiment_report(evidence: Evidence, runs: dict[str, Run], decision: dict[str, Any]) -> str:
    summary = runs[EVAL_RUN].summary if EVAL_RUN in runs else {}
    lines = ["# ACD prototype-tail unfreezing feasibility experiment", "", f"更新于 {decision['generated_at']}（Asia/Shanghai）。", "",
             *decision_report(decision), "", "## 固定范围及机制限制", "",
             "本轮 baseline 和 ACD 均从相同官方 COCO yolo26m-seg.pt 重启，Ultralytics 8.4.100。",
             "更新完整原生 cv4/one2one_cv4 两个系数头，以及 model.23.proto.cv2/cv3 的 non-BN 参数；",
             "原型前端、upsample、refinement/fusion、semantic、backbone/neck、box/class 与全部 BN affine/running state 冻结。",
             "官方 m 预期可训练参数：系数 1,708,224 + prototype tail 598,016 = 2,306,240，实际 names/count 必须核验。", "",
             "保留官方 one2one prototype detach。原型尾部仅接收 one2many mask/ACD 路径的梯度；",
             "one2one ACD 直接训练其系数头。三轮 one2many/one2one 权重为 0.8/0.2、0.45/0.55、0.1/0.9。",
             "这项梯度不对称和后期衰减是当前配置的限制，不能由短屏结果推断 prototype 或 ACD 普遍无效。", "",
             "保留 MuSGD 的官方参数名称分组：prototype cv3 为 3× group LR，cv2 为普通 group LR。",
             "保留官方 global clip_grad_norm_(model.parameters(), 10)：新增可训练原型参数会改变联合裁剪范数，",
             "可同时影响系数梯度缩放和 AMP overflow 轨迹。可训练范围、LR 分组及联合裁剪共同构成本次干预；",
             "结果不能单独证明 prototype 表示能力的充分性或不足。", "",
             "先做独立梯度/optimizer 与 loader 合同诊断，再各做 1 epoch smoke；有效后从官方权重重启各固定 3 epochs。",
             "seed=0、batch=2、workers=0、imgsz=640，固定官方支持的超参数、AMP、augmentation 和 overlap-mask 语义。",
             "正式 fit=796、内部转换标签 val=196；dev=197 不用于拟合或挑设置。1,194 input batches 不等于 1,194 optimizer updates。",
             "fit cohort 及全 val2017 均为历史研究复用，包含选择限制；全验证集不是新 blind test。", "",
             "ACD 不变：weight=0.05、tau=0.50、min gain=0.01、min coverage=0.80、每图每 loss branch 首 12 candidates。",
             "接受候选中所有 positive-logit 背景像素参与 softplus penalty，限 assigned GT support box；",
             "不只学习 action 删除的像素，亦不是完整 tightened-mask/logit distillation。训练 GT 选择辅助项，不进入推理。", "",
             "最终插入固定第 3 epoch 的完整 coefficient+prototype-tail FP32 EMA state（mask_final_ema.pt），",
             "kind=mask_coefficient_proto_tail_ema_final、epoch=3、audit_passed=true；四 prefix 共 96 state keys，BN 仍必须逐值等于官方。",
             "不选 best.pt。全 5,000 原始 COCO val2017 独立评价采用 native one2one、FP32、tf32=false、square 640、scaleup=false、",
             "conf=0.001、max_det=300，官方 JSON decoder/原图 scale_preds.byte()、COCO80→91、pycocotools maxDets=[1,10,100]。", "",
             "完整协议：[PROTOCOL.md](PROTOCOL.md)。固定屏结束即停止，不扩 seed、不改 detach/auxiliary/backbone/neck、不延长训练。",
             "有效但未达门槛为当前配置的负屏；缺失或无效证据保持未知。", "", "## 原始 COCO 全验证集结果", "",
             *metric_table(summary), "", "## 输出层配对损伤与修复", "", *pair_report(summary), "",
             "raw score-independent 五状态分类、pixel AUC/FPR、crop-support ceiling：**未知，本屏未测量**。",
             "conf/max_det 筛选后的输出配对不等于 raw GT geometry、最大一对一匹配、COCO recall 或 AP 不确定性。", "",
             "## 与 coefficient-only 条件的描述性比较", "", *comparison_report(decision["coefficient_only_comparison"], summary), "",
             "## 配对训练数值轨迹", "", *optimizer_comparison_report(decision["optimizer_step_comparison"]), "",
             *dataset_overlap_report(decision["dataset_overlap"]), "",
             "## 独立诊断及未知包裹层时间", ""]
    for name, label in (("training_gradient_diagnostic", "真实模型梯度/optimizer/EMA"), ("loader_contract_diagnostic", "synthetic loader 合同")):
        diagnostic = decision[name]
        lines.extend([f"- {label}：原 receipt status={cell(at(diagnostic, 'receipt', 'status'))}，",
                      f"  本地诊断证据核验={verdict(diagnostic.get('intrinsic_verified'))}，与正式源码链接={verdict(diagnostic.get('formal_training_source_link', diagnostic.get('formal_evaluation_source_link')))}。",
                      f"  [{diagnostic['run']}](runs/{diagnostic['run']}/REPORT.md)；{diagnostic['limitation']}"])
    reconciled = [r for r in runs.values() if r.meta.get("metadata_reconciliation_reason")]
    if reconciled:
        lines.extend(["", "诊断 Run 的 planned 模板曾覆盖 wrapper metadata；随后依据 observed exit=0/PASS receipt 恢复 completed 状态。",
                      "精确 wrapper started_at/ended_at 保持未知，receipt 自身诊断时间另行记录；本报告不填造包裹层时间，也不将其未知解释成诊断失败。"])
    lines.extend(["", "## Run 记录", "", "| Run | arm / purpose | 执行状态 | 训练/评价/诊断 receipt | 回传状态 |", "|---|---|---|---|---|"])
    for rid, run in sorted(runs.items()):
        receipts = []
        if run.complete:
            receipts.append("训练 " + cell(run.complete.get("status")))
        if run.summary:
            receipts.append("评价 " + cell(run.summary.get("status")) + f" / {cell(run.summary.get('image_count'))} 图")
        for filename in ("TRAINING_VERIFICATION.json", "LOADER_VERIFICATION.json"):
            receipt = evidence.read(run.path / filename)
            if receipt:
                receipts.append("诊断 " + cell(receipt.get("status")))
        if run.pipeline_complete:
            receipts.append("pipeline COMPLETE")
        lines.append(f"| [{rid}](runs/{rid}/REPORT.md) | {cell(run.meta.get('arm'))} / {cell(run.meta.get('purpose'))} | "
                     f"{cell(run.meta.get('execution_status'))} | {', '.join(receipts) if receipts else UNKNOWN} | {cell(run.meta.get('transfer_status'))} |")
    failures = [r for r in runs.values() if r.failed or r.meta.get("execution_status") == "failed"]
    lines.extend(["", "## 失败尝试", ""])
    if failures:
        for run in failures:
            lines.append(f"- [{run.rid}](runs/{run.rid}/REPORT.md)：{cell(run.meta.get('failure_observation') or run.failed.get('exception') or run.failed.get('traceback'))}。")
        lines.append("失败及部分 payload/checkpoint 保留；未通过审计的 payload 不进入正式 scientific 比较。")
    else:
        lines.append("当前本地可读记录未发现失败尝试；未回传状态不据此推断成功。")
    lines.extend(["", "## 核验与未完成项", ""])
    incomplete = [c for c in decision["criteria"] if c["passed"] is not True]
    for check in incomplete:
        lines.append(f"- {check['name']}：{verdict(check['passed'])}；证据 " + ", ".join(f"`{v}`" for v in check["sources"]) + "。")
    if not incomplete:
        lines.append("所有执行/evidence gate 已核验；按上述预定义科学阈值解释结果，固定预算结束后不开展额外训练。")
    lines.extend(["", f"独立读出：[READOUT_VERIFICATION.json](runs/{EVAL_RUN}/READOUT_VERIFICATION.json)；",
                  "其 AP/AP75/APsmall 来自 COCO accumulated precision 复算，配对 counts/点估计来自逐图/逐 detection 记录复算。",
                  "它不重做 COCO matching、GT/mask decoder 或 bootstrap CI；当前文件哈希与 audit evidence 再次链接。"])
    if evidence.errors:
        lines.extend(["", "文件读取问题：", "", *[f"- {cell(error)}" for error in evidence.errors]])
    lines.extend(["", f"机器可读三值判定与原始证据 SHA256：[DECISION.json](runs/{EVAL_RUN}/DECISION.json)。",
                  "报告脚本：[build_report.py](scripts/build_report.py)。仅写 REPORT/DECISION，不改 study/run metadata 或原实验。"])
    return "\n".join(lines)


def generate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "PROTOCOL.md").is_file() and (root / "experiments" / EXPERIMENT).is_dir():
        root = root / "experiments" / EXPERIMENT
    allowed = EXPERIMENT_ROOT.resolve()
    if root != allowed and not root.is_relative_to(allowed / "scripts"):
        raise ValueError("Report writes are limited to the new proto-tail experiment and its temporary test fixtures")
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
    parser.add_argument("--root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()
    if args.root.absolute() not in (EXPERIMENT_ROOT.absolute(), WORKSPACE.absolute()):
        parser.error("--root must be the organized workspace or this proto-tail experiment")
    decision = generate(args.root)
    print(json.dumps({"outcome": decision["outcome"], "gate_status": decision["gate_status"],
                      "reason": decision["reason"], "criteria": len(decision["criteria"]),
                      "read_errors": decision["read_errors"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
