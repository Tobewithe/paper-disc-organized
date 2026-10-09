"""Build a factual, three-valued TriFlow report from immutable local evidence.

Only REPORT.md, per-Run REPORT.md, and formal-evaluation DECISION.json are
written. Missing/partial/unreturned evidence is unknown; historical failed
Runs stay failed and never replace the explicitly selected successful retry.
No Torch import, model loading, inference, GT matching, or remote operation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT = "triflow_potential_coefficient_20261006"
DEFAULT_ROOT = Path("C:/Dpan/codexproject/paper-disc-organized/experiments") / EXPERIMENT
RUNS = {
    "preflight": "RUN_TRIFLOW_PREFLIGHT_S0_R1",
    "numerical": "RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1",
    "smoke_cache": "RUN_TRIFLOW_SMOKE_CACHE_S0",
    "smoke_train": "RUN_TRIFLOW_SMOKE_TRAIN_S0_R1",
    "fit_cache": "RUN_TRIFLOW_FIT_CACHE_S0",
    "train": "RUN_TRIFLOW_TRAIN_S0_R1",
    "eval_contract": "RUN_TRIFLOW_EVAL_CONTRACT_S0_R1",
    "eval": "RUN_TRIFLOW_COCO5000_EVAL_S0",
    "pipeline": "RUN_TRIFLOW_PIPELINE_S0_R1",
}
OFFICIAL_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
TRAIN_ANNOTATIONS_SHA = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
VAL_ANNOTATIONS_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
GROUPS = ("token_projection", "ownership_embeddings", "ownership_projection", "cross_attention", "interaction_norm", "field_head")
TRAIN_SOURCES = {"train_triflow.py", "frozen_io.py", "triflow_model.py"}
CACHE_SOURCES = {"prepare_cache.py", "frozen_io.py", "triflow_model.py"}
EVAL_SOURCES = {"evaluate_triflow.py", "readout_support.py", "frozen_io.py", "triflow_model.py"}
UNKNOWN = "未知"


def at(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def eq(value: Any, expected: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(expected, bool) and not isinstance(value, bool):
        return False
    return value == expected


def same(a: Any, b: Any) -> bool | None:
    return None if a is None or b is None else a == b


def conjunction(values: list[bool | None]) -> bool | None:
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def verdict(value: bool | None) -> str:
    return "通过" if value is True else "未通过" if value is False else UNKNOWN


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest_dict(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def markdown_link(path: Path, label: str | None = None) -> str:
    target = path.resolve().as_posix()
    if path.drive and not target.startswith("/"):
        target = "/" + target
    return f"[{label or path.name}](<{target}>)"


def cell(value: Any) -> str:
    if value is None:
        return UNKNOWN
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt(value: Any, points: bool = False, signed: bool = False, precision: int = 4) -> str:
    value = finite(value)
    if value is None:
        return UNKNOWN
    if points:
        value *= 100
    return f"{value:+.{precision}f}" if signed else f"{value:.{precision}f}"


class Evidence:
    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[Path, Any] = {}
        self.errors: list[str] = []
        self.files: dict[str, dict] = {}
        self.hash_cache: dict[Path, str] = {}

    def path(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Evidence path escapes the experiment")
        return path

    def file_sha(self, relative: str | Path) -> str | None:
        path = self.path(relative)
        if not path.is_file():
            return None
        if path not in self.hash_cache:
            self.hash_cache[path] = sha(path)
            self.files[path.relative_to(self.root).as_posix()] = {"sha256": self.hash_cache[path], "bytes": path.stat().st_size}
        return self.hash_cache[path]

    def read(self, relative: str | Path) -> dict | None:
        path = self.path(relative)
        if path not in self.cache:
            if not path.is_file():
                self.cache[path] = None
            else:
                try:
                    value = json.loads(path.read_text(encoding="utf-8-sig"))
                    if not isinstance(value, dict):
                        raise ValueError("Expected JSON object")
                    self.file_sha(relative)
                    self.cache[path] = value
                except (OSError, ValueError) as error:
                    self.errors.append(f"{path.relative_to(self.root)}: {error}")
                    self.cache[path] = None
        return self.cache[path]

    def binding(self, relative: str | Path, expected_sha: Any) -> bool | None:
        actual = self.file_sha(relative)
        if actual is None or expected_sha is None:
            return None
        return is_sha(expected_sha) and actual == expected_sha

    def sources(self, run_relative: Path, values: Any, required: set[str]) -> bool | None:
        if values is None:
            return None
        if not isinstance(values, dict) or set(values) != required:
            return False
        return conjunction([self.binding(run_relative / "source" / name, values[name]) for name in sorted(required)])


def run_relative(name: str) -> Path:
    return Path("runs") / RUNS[name]


def execution(evidence: Evidence, name: str) -> bool | None:
    meta = evidence.read(run_relative(name) / "run.json")
    if meta is None:
        return None
    if meta.get("status") in ("failed", "cancelled") or meta.get("execution_status") in ("failed", "cancelled"):
        return False
    if meta.get("status") != "completed" or meta.get("execution_status") != "completed":
        return None
    return conjunction([eq(meta.get("exit_code"), 0), eq(meta.get("run_id"), RUNS[name])])


def transferred(evidence: Evidence, name: str) -> bool | None:
    relative = run_relative(name)
    transfer = evidence.read(relative / "transfer.json")
    meta = evidence.read(relative / "run.json")
    if transfer is None or meta is None:
        return None
    if transfer.get("status") != "verified":
        return None
    return conjunction([eq(meta.get("transfer_status"), "verified"), eq(transfer.get("run_id"), RUNS[name]),
                        evidence.binding(relative / "_source_manifest.sha256", transfer.get("source_manifest_sha256"))])


def frozen_integrity(value: Any) -> bool | None:
    if value is None:
        return None
    return conjunction([eq(at(value, "passed"), True), eq(at(value, "all_state_exact"), True),
                        same(at(value, "initial_state_sha256"), at(value, "final_state_sha256")),
                        is_sha(at(value, "initial_state_sha256")) if at(value, "initial_state_sha256") is not None else None])


def model_defaults(evidence: Evidence, relative: Path) -> dict | None:
    path = evidence.path(relative)
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TriFlowConfig")
        return {node.target.id: ast.literal_eval(node.value) for node in klass.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None}
    except (ValueError, SyntaxError, StopIteration) as error:
        evidence.errors.append(f"Cannot read immutable TriFlow defaults: {error}")
        return None


def independent_binding(evidence: Evidence, relative: Path, audit: dict | None) -> bool | None:
    if audit is None:
        return None
    records = audit.get("evidence")
    if not isinstance(records, dict) or not records:
        return False
    checks = [eq(audit.get("passed"), True), eq(audit.get("status"), "passed"), eq(audit.get("audit_version"), "triflow_readout_verify_v1")]
    normalized_names = set()
    alias_names = set()
    for filename, record in records.items():
        if not isinstance(filename, str):
            checks.append(False)
            continue
        normalized = filename.replace("\\", "/")
        path = Path(normalized)
        canonical = path.as_posix()
        if path.is_absolute() or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in path.parts or canonical.casefold() in alias_names or not isinstance(record, dict):
            checks.append(False)
            continue
        normalized_names.add(canonical)
        alias_names.add(canonical.casefold())
        checks.append(evidence.binding(relative / path, record.get("sha256")))
        actual = evidence.path(relative / path)
        checks.append(same(actual.stat().st_size if actual.is_file() else None, record.get("bytes")))
    required = {"SUMMARY.json", "HEAD_PROVENANCE.json", "BASELINE_REPLAY_VERIFICATION.json", "BASELINE_REPLAY_IMAGES.jsonl", "COMPILER_IMAGES.jsonl",
                "baseline/COCO_segm_ACCUMULATED.npz", "baseline/COCO_bbox_ACCUMULATED.npz",
                "triflow/COCO_segm_ACCUMULATED.npz", "triflow/COCO_bbox_ACCUMULATED.npz",
                "paired_triflow_vs_baseline/IMAGES.jsonl", "paired_triflow_vs_baseline/INSTANCES.jsonl"}
    checks.append(required.issubset(normalized_names))
    checks.append(eq(audit.get("missing"), []))
    checks.append(eq(audit.get("errors"), []))
    return conjunction(checks)


def flatten_numbers(value: Any) -> list:
    if isinstance(value, list):
        return [number for item in value for number in flatten_numbers(item)]
    return [value]


def jsonl_records(path: Path):
    """Stream potentially large trace files without loading them into RAM."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path.name}:{number}: blank JSONL row")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{number}: expected object")
            yield record


def actual_replay_chain(evidence: Evidence, relative: Path, summary: dict | None) -> bool | None:
    """Bind replay rows to currently present individual baseline image bytes."""
    if at(summary, "status") != "complete":
        return None
    path = evidence.path(relative / "BASELINE_REPLAY_IMAGES.jsonl")
    if not path.is_file():
        return None
    expected = at(summary, "configuration", "image_ids")
    ids, chain = [], []
    try:
        for record in jsonl_records(path):
            iid = record["image_id"]
            image = relative / "baseline" / "images" / f"{iid:012d}.json"
            actual = evidence.file_sha(image)
            if actual is None:
                return None
            if actual != record.get("source_image_cache_sha256"):
                return False
            ids.append(iid)
            chain.append(actual)
        return ids == expected and len(ids) == 5000 and hashlib.sha256("".join(chain).encode()).hexdigest() == at(summary, "baseline_replay", "baseline_image_cache_sha256_chain")
    except (OSError, ValueError, KeyError, TypeError) as error:
        evidence.errors.append(f"Actual per-image baseline replay binding: {error}")
        return False


def compiler_diagnostics(evidence: Evidence, relative: Path, summary: dict | None) -> tuple[bool | None, dict | None]:
    """Verify recorded actual SPD info/numericridge/trust for every deploy row."""
    if at(summary, "status") != "complete":
        return None, None
    path = evidence.path(relative / "COMPILER_IMAGES.jsonl")
    if not path.is_file():
        return None, None
    totals = {"images": 0, "solver_instances": 0, "gram_numeric_ridge_instances": 0, "system_numeric_ridge_instances": 0,
              "maximum_added_gram_ridge": None, "maximum_added_system_ridge": None, "maximum_recorded_gram_condition": None,
              "maximum_recorded_system_condition": None, "maximum_system_residual": None, "maximum_post_logit_change": None,
              "solver_nonzero_cholesky_info_count": 0}
    valid = True
    image_ids = []
    try:
        for record in jsonl_records(path):
            image_ids.append(record["image_id"])
            totals["images"] += 1
            for chunk in record["chunks"]:
                diag = chunk["diagnostics"]
                count = len(chunk["native_output_rows"])
                totals["solver_instances"] += count
                valid &= 1 <= count <= 4
                scalar_keys = ("gram_cholesky_info", "added_gram_numeric_ridge", "gram_identity_ridge", "gram_condition", "post_logit_max", "trust_scale", "compiler_pass_count")
                pass_keys = ("system_cholesky_info", "added_system_numeric_ridge", "effective_system_identity_ridge", "pass_system_condition", "pass_solve_residual", "pass_anchor_violation_count", "pass_active_anchor_count")
                shape_valid = all(isinstance(diag.get(key), list) and len(diag[key]) == count and all(finite(v) is not None for v in diag[key]) for key in scalar_keys)
                shape_valid &= all(isinstance(diag.get(key), list) and len(diag[key]) == count and all(isinstance(row, list) and len(row) == 3 and all(finite(v) is not None for v in row) for row in diag[key]) for key in pass_keys)
                valid &= shape_valid
                if not shape_valid:
                    # Missing/empty arrays are invalid evidence, never a zero
                    # measurement or a vacuous all(...) success.
                    continue
                all_info = flatten_numbers(diag.get("gram_cholesky_info", [])) + flatten_numbers(diag.get("system_cholesky_info", []))
                totals["solver_nonzero_cholesky_info_count"] += sum(value != 0 for value in all_info)
                valid &= len(all_info) == 4 * count and all(value == 0 for value in all_info)
                valid &= diag.get("compiler_pass_count") == [3] * count
                for key in ("added_gram_numeric_ridge", "added_system_numeric_ridge"):
                    valid &= all(finite(v) is not None and v >= 0 for v in flatten_numbers(diag.get(key, [])))
                valid &= all(finite(v) is not None and v > 0 for v in flatten_numbers(diag.get("effective_system_identity_ridge", [])))
                valid &= all(v > 0 for v in diag["gram_identity_ridge"])
                valid &= all(v >= 1 - 1e-5 for v in diag["gram_condition"])
                valid &= all(v >= 1 - 1e-5 for v in flatten_numbers(diag["pass_system_condition"]))
                valid &= all(v >= 0 for v in flatten_numbers(diag["pass_solve_residual"]))
                valid &= all(finite(v) is not None and 0 < v <= 1 + 1e-6 for v in diag.get("trust_scale", []))
                valid &= all(finite(v) is not None and 0 <= v <= 2 + 1e-4 for v in diag.get("post_logit_max", []))
                gram_added = diag.get("added_gram_numeric_ridge", [])
                system_added = diag.get("added_system_numeric_ridge", [])
                totals["gram_numeric_ridge_instances"] += sum(v > 0 for v in gram_added)
                totals["system_numeric_ridge_instances"] += sum(any(v > 0 for v in values) for values in system_added)
                for diagnostic, target in (("added_gram_numeric_ridge", "maximum_added_gram_ridge"), ("added_system_numeric_ridge", "maximum_added_system_ridge"),
                                           ("gram_condition", "maximum_recorded_gram_condition"), ("pass_system_condition", "maximum_recorded_system_condition"),
                                           ("pass_solve_residual", "maximum_system_residual"), ("post_logit_max", "maximum_post_logit_change")):
                    values = flatten_numbers(diag.get(diagnostic, []))
                    valid &= all(finite(v) is not None for v in values)
                    if values:
                        measured = max(v for v in values if finite(v) is not None)
                        totals[target] = max(totals[target], measured) if totals[target] is not None else measured
        valid &= image_ids == at(summary, "configuration", "image_ids") and len(image_ids) == 5000 and totals["solver_instances"] == summary.get("refined_instances")
        evidence.file_sha(relative / "COMPILER_IMAGES.jsonl")
        return valid, totals
    except (OSError, ValueError, KeyError, TypeError) as error:
        evidence.errors.append(f"Actual deployment compiler diagnostic: {error}")
        return False, totals


def collect_decision(evidence: Evidence) -> dict:
    criteria: list[dict] = []

    def add(key: str, description: str, value: bool | None, detail: Any = None) -> None:
        criteria.append({"key": key, "description": description, "passed": value, "status": verdict(value), "detail": clean(detail)})

    pf = evidence.read(run_relative("preflight") / "PREFLIGHT.json")
    diagnostic = evidence.read(run_relative("numerical") / "TRIFLOW_VERIFICATION.json")
    cache_rel, train_rel, ev_rel = run_relative("fit_cache"), run_relative("train"), run_relative("eval")
    cache = evidence.read(cache_rel / "CACHE_RECEIPT.json")
    cache_complete = evidence.read(cache_rel / "CACHE_COMPLETE.json")
    inputs = evidence.read(train_rel / "TRAINING_INPUTS.json")
    train = evidence.read(train_rel / "TRAINING_AUDIT.json")
    completed = evidence.read(train_rel / "TRAINING_COMPLETE.json")
    task = evidence.read(train_rel / "TASK_GRADIENT_EVIDENCE.json")
    epochs = evidence.read(train_rel / "EPOCHS.json")
    summary = evidence.read(ev_rel / "SUMMARY.json")
    ev_complete = evidence.read(ev_rel / "EVALUATION_COMPLETE.json")
    head = evidence.read(ev_rel / "HEAD_PROVENANCE.json")
    ev_inputs = evidence.read(ev_rel / "EVALUATION_INPUTS.json")
    replay = evidence.read(ev_rel / "BASELINE_REPLAY_VERIFICATION.json")
    original = evidence.read(ev_rel / "ORIGINAL_BASELINE_RECEIPT.json")
    readout = evidence.read(ev_rel / "READOUT_VERIFICATION.json")
    contract_relative = run_relative("eval_contract") / "EVALUATOR_VERIFICATION.json"
    contract = evidence.read(contract_relative)

    for name in ("preflight", "numerical", "smoke_cache", "smoke_train", "fit_cache", "train", "eval_contract", "eval", "pipeline"):
        add("execution_" + name, RUNS[name] + " 执行完成且 exit0", execution(evidence, name))
        add("transfer_" + name, RUNS[name] + " 回传验证及原始 manifest 保留", transferred(evidence, name))
    add("preflight", "实际环境、原始训练/验证标注及 native 输入预检", conjunction([
        eq(at(pf, "status"), "PASS"), eq(at(pf, "ultralytics"), "8.4.100"), eq(at(pf, "native_replay_exact"), True),
        eq(at(pf, "train_annotations_sha256"), TRAIN_ANNOTATIONS_SHA), eq(at(pf, "val_annotations_sha256"), VAL_ANNOTATIONS_SHA),
        eq(at(pf, "lists", "fit", "count"), 796), eq(at(pf, "lists", "val_full", "count"), 5000), frozen_integrity(at(pf, "frozen_state"))]))
    numeric_checks = at(diagnostic, "checks")
    add("numerical", "20 项数值、task-only 梯度与真实冻结 native 重放均通过", conjunction([
        eq(at(diagnostic, "passed"), True), eq(at(diagnostic, "numerical_only"), False), eq(at(diagnostic, "actual_frozen_forward_required"), True),
        eq(at(diagnostic, "check_count"), 20), eq(at(diagnostic, "failed_checks"), []),
        all(isinstance(c, dict) and c.get("passed") is True for c in numeric_checks) if isinstance(numeric_checks, list) and len(numeric_checks) == 20 else None if numeric_checks is None else False]))
    numeric_sources = {k: v.get("sha256") for k, v in at(diagnostic, "source_files").items()} if isinstance(at(diagnostic, "source_files"), dict) else None
    add("numerical_sources", "通过诊断的实际 source archive 与训练核心/提取器一致", conjunction([
        evidence.sources(run_relative("numerical"), numeric_sources, {"verify_triflow.py", "triflow_model.py", "frozen_io.py"}),
        same(at(numeric_sources, "triflow_model.py"), at(inputs, "sources", "triflow_model.py")),
        same(at(numeric_sources, "frozen_io.py"), at(inputs, "sources", "frozen_io.py"))]))
    add("cache_complete", "796 张正式冻结 cache 完成并绑定 receipt 字节", conjunction([
        eq(at(cache, "status"), "completed"), eq(at(cache, "images"), 796), eq(at(cache_complete, "images"), 796),
        evidence.binding(cache_rel / "CACHE_RECEIPT.json", at(cache_complete, "receipt_sha256")),
        eq(at(cache, "base_weights_sha256"), OFFICIAL_SHA), eq(at(cache, "annotation_sha256"), TRAIN_ANNOTATIONS_SHA),
        same(at(cache, "images_list_sha256"), at(pf, "lists", "fit", "sha256")),
        evidence.binding("data/fit.txt", at(cache, "images_list_sha256")),
        eq(at(cache, "all_cache_float_tensors"), "finite FP32"), eq(at(cache, "all_native_replay_exact"), True), frozen_integrity(at(cache, "frozen_integrity"))]))
    inventory = at(cache, "cache_files")
    inventory_check = None
    if isinstance(inventory, list):
        inventory_check = len(inventory) == 796 and len({item.get("image_id") for item in inventory if isinstance(item, dict)}) == 796 and all(isinstance(item, dict) and is_sha(item.get("sha256")) and is_sha(item.get("image_sha256")) and is_sha(item.get("input_sha256")) and isinstance(item.get("bytes"), int) and item["bytes"] > 0 and item.get("native_replay_exact") is True for item in inventory)
        if inventory_check:
            inventory_check = sum(item["bytes"] for item in inventory) == cache.get("cache_bytes") and sum(item.get("instances", 0) for item in inventory) == cache.get("instances") and [item["image_id"] for item in inventory] == cache.get("image_ids")
    add("cache_inventory", "逐图 FP32 cache/image/input SHA、数量与大小 inventory 一致", inventory_check,
        {"tensors_local": "large caches may be retained remote; actual trainer validates every payload byte before use", "inventory_images": len(inventory) if isinstance(inventory, list) else None})
    add("cache_source", "cache source archive、正式训练源与通过诊断源一致", conjunction([
        evidence.sources(cache_rel, at(cache, "sources"), CACHE_SOURCES),
        same(at(cache, "sources", "triflow_model.py"), at(inputs, "sources", "triflow_model.py")),
        same(at(cache, "sources", "frozen_io.py"), at(inputs, "sources", "frozen_io.py"))]))
    selection = at(cache, "selection")
    add("cache_candidate_domain", "训练 first64、raw 去重、class-free boxIoU≥.5、GT round-robin≤12", conjunction([
        eq(at(selection, "native_confidence"), .001), eq(at(selection, "first_native_post_conf_rows"), 64),
        eq(at(selection, "max_instances_per_image"), 12), eq(at(selection, "class_free_best_gt_box_iou"), .5),
        eq(at(selection, "deduplicate_raw"), True), eq(at(selection, "tie"), "annotation_id ascending"),
        eq(at(selection, "order"), "round-robin matched GT groups ordered by first native output row")]))
    add("cache_ownership", "Self 优先、crowd 忽略、BG 独立、N 缺失监督有 validity mask", conjunction([
        eq(at(cache, "ownership", "self_priority"), True), eq(at(cache, "ownership", "gt_neighbor"), "union all other noncrowd GT minus Self"),
        eq(at(cache, "ownership", "ignore"), "union crowd minus Self"), eq(at(cache, "ownership", "background"), "outside all ordinary GT and crowd"),
        eq(at(cache, "ownership", "neighbor_supervision_missing"), "phi_N masked if GT Neighbor empty OR no eligible predicted neighbor")]))
    add("train_fixed_budget", "正式训练固定8 epoch、seed0、chunk4、FP32无AMP/增广、固定AdamW", conjunction([
        eq(at(inputs, "epochs"), 8), eq(at(inputs, "seed"), 0), eq(at(inputs, "instance_chunk"), 4), eq(at(inputs, "augmentation"), False),
        eq(at(inputs, "precision"), "FP32, TF32 disabled, no AMP"), eq(at(inputs, "optimizer"), {"class": "AdamW", "lr": .0003, "weight_decay": .0001, "clip_grad_norm": 10}),
        eq(at(train, "epochs"), 8), eq(at(completed, "epochs"), 8)]))
    configuration = at(inputs, "configuration")
    defaults = model_defaults(evidence, train_rel / "source" / "triflow_model.py")
    cfg_check = None
    if isinstance(configuration, dict) and isinstance(defaults, dict):
        cfg_check = set(configuration) == set(defaults) | {"feature_channels", "instance_hidden_channels"} and all(configuration.get(k) == v for k, v in defaults.items())
        cfg_check = conjunction([cfg_check, eq(configuration.get("feature_channels"), at(cache, "dimensions", "feature_channels")),
                                 eq(configuration.get("instance_hidden_channels"), at(cache, "dimensions", "instance_hidden_channels")),
                                 eq(digest_dict(configuration), at(inputs, "configuration_sha256")),
                                 same(at(inputs, "configuration_sha256"), at(train, "configuration_sha256"))])
    add("train_configuration", "实际 native feature/hidden 维度、所有锁定默认值与 configSHA", cfg_check)
    add("train_input_binding", "正式训练绑定实际796 cache、原始训练标注与列表SHA", conjunction([
        evidence.binding(cache_rel / "CACHE_RECEIPT.json", at(inputs, "cache_receipt_sha256")),
        same(at(inputs, "cache_receipt_sha256"), at(train, "cache_receipt_sha256")), eq(at(inputs, "base_weights_sha256"), OFFICIAL_SHA),
        same(at(inputs, "annotation_sha256"), at(cache, "annotation_sha256")), same(at(inputs, "images_list_sha256"), at(cache, "images_list_sha256")),
        evidence.sources(train_rel, at(inputs, "sources"), TRAIN_SOURCES), same(at(inputs, "sources"), at(train, "sources"))]))
    epoch_list = at(epochs, "epochs")
    epoch_check = None
    if isinstance(epoch_list, list):
        epoch_check = len(epoch_list) == 8 and [x.get("epoch") for x in epoch_list] == list(range(1, 9))
        if epoch_check:
            epoch_check = all(x.get("instances") == at(cache, "instances") and len(x.get("image_order", [])) == 796 and set(x.get("image_order", [])) == set(at(cache, "image_ids") or []) for x in epoch_list)
            epoch_check = conjunction([epoch_check, same(sum(x.get("applied_steps", 0) for x in epoch_list), at(train, "optimizer_applied"))])
    add("epoch_history", "实际8轮796图/instances顺序与 applied step 计数一致", epoch_check)
    attempts, applied = finite(at(train, "optimizer_attempts")), finite(at(train, "optimizer_applied"))
    add("training_audit", "实际更新有限、无跳过且6个模块组确实改变", conjunction([
        eq(at(train, "passed"), True), same(at(train, "optimizer_attempts"), at(train, "optimizer_applied")), applied > 0 if applied is not None else None,
        eq(at(train, "skipped_updates"), 0), eq(at(train, "all_applied_gradients_finite"), True), eq(at(train, "all_parameters_finite"), True),
        eq(at(train, "all_groups_really_changed"), True),
        all(at(train, "gradient_finite_by_group", g) is True and (finite(at(train, "parameter_updates_by_group", g, "changed_tensors")) or 0) > 0 and (finite(at(train, "nonzero_gradient_steps_by_group", g)) or 0) > 0 for g in GROUPS) if train is not None else None]))
    evidence_list = at(task, "evidence")
    task_check = None
    if isinstance(evidence_list, list):
        task_check = len(evidence_list) == 8 and [x.get("epoch") for x in evidence_list] == list(range(1, 9)) and all(x.get("all_task_gradients_finite") is True for x in evidence_list)
        task_check = conjunction([task_check, eq(at(task, "any_observed_nonzero"), True), eq(at(task, "no_hidden_warmup"), True),
                                  eq(at(train, "compiler_task_gradient_observed_nonzero"), True), eq(at(train, "no_hidden_warmup"), True)])
    if isinstance(evidence_list, list):
        task_check = conjunction([task_check, any((finite(at(row, "group_task_gradient_norms", "field_head")) or 0) > 0 and
                                                  (finite(at(row, "group_task_gradient_norms", "cross_attention")) or 0) > 0 for row in evidence_list)])
        phi_attention_observations = [(finite(row.get("phi_task_gradient_norm")), finite(row.get("cross_attention_task_gradient_norm"))) for row in evidence_list]
        observed_values = all(phi is not None and attention is not None for phi, attention in phi_attention_observations)
        observed_joint = any(phi > 0 and attention > 0 for phi, attention in phi_attention_observations) if observed_values else None
        task_check = conjunction([task_check, observed_joint, eq(at(task, "any_observed_phi_attention_same_probe_nonzero"), True),
                                  eq(at(train, "compiler_task_phi_and_attention_same_probe_observed_nonzero"), True)])
        if observed_values:
            phi_count = sum(phi > 0 for phi, _ in phi_attention_observations)
            attention_count = sum(attention > 0 for _, attention in phi_attention_observations)
            joint_count = sum(phi > 0 and attention > 0 for phi, attention in phi_attention_observations)
            task_check = conjunction([task_check, all(row.get("task_gradient_phi_attention_same_probe_nonzero") is (phi > 0 and attention > 0) for row, (phi, attention) in zip(evidence_list, phi_attention_observations)),
                                      eq(at(train, "compiler_task_phi_nonzero_probe_count"), phi_count),
                                      eq(at(train, "compiler_task_attention_nonzero_probe_count"), attention_count),
                                      eq(at(train, "compiler_task_phi_attention_same_probe_count"), joint_count),
                                      same(at(head, "phi_attention_same_probe_count"), joint_count),
                                      evidence.binding(train_rel / "TASK_GRADIENT_EVIDENCE.json", at(head, "task_gradient_evidence_sha256"))])
    add("compiled_task_gradient", "同一真实mask task-only probe经Compiler回到φ前2行与attention；无隐藏warm-up", task_check,
        {"epochs_probed": len(evidence_list) if isinstance(evidence_list, list) else None,
         "diagnostic_task_only_gradient_and_update": next((c.get("facts") for c in numeric_checks if "task_only_gradient" in c.get("name", "")), None) if isinstance(numeric_checks, list) else None})
    add("frozen_training_inputs", "仅训练新模块；每轮首chunk输入hash不变，cache实际逐字节验过", conjunction([
        eq(at(train, "frozen_yolo_instantiated"), False), eq(at(train, "frozen_cache_receipt_integrity_passed"), True),
        eq(at(train, "frozen_input_mutation_checks_passed"), True), eq(at(train, "frozen_input_mutation_checks"), 8)]))
    add("final_head_bytes", "唯一正式final8 head字节/大小绑定训练completion与已加载head", conjunction([
        evidence.binding(train_rel / "head_final.pt", at(completed, "head_sha256")),
        same(at(completed, "head_sha256"), at(head, "head_sha256")), same(at(completed, "head_bytes"), at(head, "head_bytes")),
        eq(at(completed, "audit_passed"), True), eq(at(head, "kind"), "triflow_module_final"), eq(at(head, "epoch"), 8),
        eq(at(head, "smoke_only"), False), eq(at(head, "all_state_finite_fp32"), True), eq(at(head, "audit_passed"), True),
        same(at(head, "loaded_state_sha256"), at(train, "final_head_state_sha256")),
        evidence.binding(train_rel / "TRAINING_AUDIT.json", at(head, "training_audit_sha256"))]))
    add("eval_head_provenance", "评估确实加载该final8 head及正式配置/cache/source链", conjunction([
        same(head, at(summary, "head_provenance")), same(head, at(ev_inputs, "head")),
        same(at(head, "configuration_sha256"), at(inputs, "configuration_sha256")), same(at(head, "configuration"), configuration),
        same(at(head, "cache_receipt_sha256"), at(inputs, "cache_receipt_sha256")), same(at(head, "training_sources"), at(inputs, "sources")),
        eq(at(head, "base_weights_sha256"), OFFICIAL_SHA), eq(at(head, "no_gt_forward"), True)]))
    contract_images = at(contract, "images")
    contract_checks = [eq(at(contract, "passed"), True), eq(at(contract, "status"), "passed"),
                       eq(at(contract, "verification_version"), "triflow_four_image_native_decoder_verification_v1"),
                       eq(at(contract, "gt_opened"), False), eq(at(contract, "successful_images"), 4), eq(at(contract, "sample_count_required"), 4),
                       eq(at(contract, "strict_loader_smoke_rejection", "passed"), True), eq(at(contract, "strict_loader_smoke_rejection", "actual_epoch"), 1),
                       eq(at(contract, "strict_loader_smoke_rejection", "actual_smoke_only"), True),
                       eq(at(contract, "strict_loader_smoke_rejection", "formal_positive_checkpoint_fabricated"), False),
                       eq(at(contract, "formal_epoch8_positive_loader_verified"), False), eq(at(contract, "ap_measured"), False),
                       frozen_integrity(at(contract, "frozen_after")),
                       eq(at(contract, "inputs", "weights", "sha256"), OFFICIAL_SHA),
                       evidence.binding("data/val_full.txt", at(contract, "inputs", "images_list", "sha256"))]
    if isinstance(contract_images, list):
        contract_checks.append(len(contract_images) == 4 and all(row.get("passed") is True and at(row, "parity", "passed") is True and at(row, "parity", "all_fields_canonical_json_exact") is True and at(row, "parity", "rle_size_and_ascii_counts_exact") is True for row in contract_images))
    else:
        contract_checks.append(None)
    contract_sources = at(contract, "sources")
    expected_contract_sources = {"verify_evaluator.py", *EVAL_SOURCES}
    if isinstance(contract_sources, dict):
        contract_checks.append(set(contract_sources) == expected_contract_sources)
        for name, source_record in contract_sources.items():
            contract_checks.extend([eq(source_record.get("executed_in_captured_snapshot"), True),
                                    same(source_record.get("executed_sha256"), source_record.get("captured_sha256")),
                                    evidence.binding(run_relative("eval_contract") / "source" / name, source_record.get("executed_sha256"))])
            if name in EVAL_SOURCES:
                contract_checks.append(same(source_record.get("executed_sha256"), at(summary, "configuration", "method_source_sha256", name)))
    else:
        contract_checks.append(None)
    add("eval_contract", "4图official c0 native identity/RLE与真实smoke拒绝；不是formal正例/AP/5000证明", conjunction(contract_checks),
        {"file": contract_relative.as_posix(), "scope": "four-image decoder contract and real smoke rejection; formal positive loading belongs to final evaluator"})
    cfg = at(summary, "configuration")
    add("eval_complete", "完整original COCO5000评估及completion->SUMMARY SHA", conjunction([
        eq(at(summary, "status"), "complete"), eq(at(summary, "image_count"), 5000), eq(at(summary, "covers_all_annotation_images"), True),
        eq(at(ev_complete, "status"), "completed"), evidence.binding(ev_rel / "SUMMARY.json", at(ev_complete, "summary_sha256")),
        eq(at(ev_complete, "images"), 5000), eq(at(cfg, "annotations", "sha256"), VAL_ANNOTATIONS_SHA),
        same(at(cfg, "images_list", "sha256"), at(pf, "lists", "val_full", "sha256")), evidence.binding("data/val_full.txt", at(cfg, "images_list", "sha256"))]))
    fixed_eval = {"imgsz": 640, "shape": [640, 640], "batch": 1, "half": False, "conf": .001, "max_det": 300,
                  "coco_maxDets": [1, 10, 100], "rect": False, "scaleup": False, "augment": False, "native_one2one": True, "tf32": False,
                  "decode": "SegmentationValidator.save_json=True: process_mask_native; scale_preds.byte()"}
    add("eval_scope_configuration", "官方8.4.100/native one2one/FP32/640/原始mask-area AP协议", conjunction([
        eq(at(cfg, "ultralytics"), "8.4.100"), eq(at(cfg, "evaluator_version"), "triflow_eval_v1"),
        *[eq(at(cfg, "config", key), value) for key, value in fixed_eval.items()],
        len(at(cfg, "image_ids")) == 5000 and len(set(at(cfg, "image_ids"))) == 5000 if isinstance(at(cfg, "image_ids"), list) else None]))
    add("eval_sources", "实际evaluation/readout/core/extraction source archive及训练源一致", conjunction([
        evidence.sources(ev_rel, at(cfg, "method_source_sha256"), EVAL_SOURCES),
        evidence.binding(ev_rel / "source" / "evaluate_triflow.py", at(cfg, "evaluator_sha256")),
        same(at(cfg, "method_source_sha256", "triflow_model.py"), at(inputs, "sources", "triflow_model.py")),
        same(at(cfg, "method_source_sha256", "frozen_io.py"), at(inputs, "sources", "frozen_io.py"))]))
    method = at(cfg, "method")
    add("gt_free_deployment_domain", "部署first64预测域、top2预测邻居、无GT/Direct32；unsupported保持c0", conjunction([
        eq(at(method, "first_post_conf_native_rows"), 64), eq(at(method, "eligible_confidence"), .001), eq(at(method, "instance_chunk"), 4),
        eq(at(method, "gt_used_in_forward"), False), eq(at(method, "learned_coefficient_residual"), False),
        eq(at(method, "predicted_neighbors"), "top2 positive predicted-box IoU; stable output row tie; same raw id excluded"),
        eq(at(method, "unsupported_native_roi"), "first64 budget retained; unsupported rows keep c0 exactly, no replacement row")]))
    add("baseline_replay", "原baseline逐5000图fresh官方forward/native decode与identity/RLE完全相同", conjunction([
        same(replay, at(summary, "baseline_replay")), eq(at(replay, "status"), "passed"), eq(at(replay, "images"), 5000),
        eq(at(replay, "all_5000_images_replayed"), True), eq(at(replay, "all_native_forward_exact"), True),
        eq(at(replay, "all_baseline_identity_and_rle_exact"), True), frozen_integrity(at(replay, "current_frozen_model_integrity")),
        evidence.binding(ev_rel / "ORIGINAL_BASELINE_RECEIPT.json", at(replay, "original_baseline_receipt_sha256")),
        same(at(replay, "original_frozen_state_sha256"), at(original, "frozen_state_sha256")),
        isinstance(at(replay, "original_digest_scope"), str) and "not equal" in replay["original_digest_scope"] if replay is not None else None]))
    arm_checks = []
    for name in ("baseline", "triflow"):
        arm = evidence.read(ev_rel / name / "COMPLETE.json")
        arm_checks.extend([same(arm, at(summary, "prediction_receipts", name)), eq(at(arm, "image_count"), 5000), eq(at(arm, "status"), "prediction_complete"),
                           eq(at(arm, "model", "official_base_sha256"), OFFICIAL_SHA), eq(at(arm, "model", "official_weights_unchanged"), True),
                           eq(at(arm, "model", "learned_module_loaded"), name == "triflow"), eq(at(arm, "model", "no_gt_forward"), True),
                           eq(at(arm, "frozen_digest_scope"), "all actual frozen official model state_dict tensors"), frozen_integrity(at(arm, "frozen_integrity")),
                           same(at(arm, "frozen_state_sha256"), at(replay, "current_frozen_model_integrity", "final_state_sha256")),
                           evidence.binding(ev_rel / name / "predictions.json", at(arm, "predictions_sha256"))])
    add("both_arm_frozen_identity", "两arm同一完整冻结official模型digest且predictions字节绑定", conjunction(arm_checks))
    add("actual_baseline_image_bytes", "5000个baseline逐图文件当前SHA与fresh replay链逐个绑定", actual_replay_chain(evidence, ev_rel, summary))
    compiler_valid, compiler_summary = compiler_diagnostics(evidence, ev_rel, summary)
    add("actual_compiler_numerics", "每部署instance/3pass实际SPD info0、finite数值ridge、trust≤2logit", compiler_valid, compiler_summary)
    add("independent_readout", "NumPy独立NPZ AP/bbox数组/paired点估计/5000域重聚合且所有引用文件SHA未过期", independent_binding(evidence, ev_rel, readout),
        {"checks": at(readout, "checks"), "scope": "aggregate only; no GT mask match or bootstrap CI recomputation"})
    pair = at(summary, "paired", "triflow")
    add("paired_identity", "same native detection index / bbox / class / confidence，5000图完整配对", conjunction([
        eq(at(pair, "paired_readout_valid"), True), eq(at(pair, "parity", "image_count"), 5000),
        *[eq(at(pair, "parity", key), True) for key in ("frozen_state_exact_equal", "detection_index_equal", "all_images_box_class_exact_equal")],
        *[eq(at(pair, "parity", key), 0) for key in ("detection_count_mismatch_images", "class_mismatch_detections", "box_max_abs_error", "confidence_max_abs_error")],
        eq(at(pair, "statistics", "bootstrap_samples"), 5000), eq(at(pair, "statistics", "bootstrap_seed"), 0), eq(at(pair, "statistics", "images"), 5000)]))
    add("evaluation_state_immutable", "评估期间learned module state不变", eq(at(summary, "learned_module_state_exact"), True))
    add("primary_receipts_readable", "所有当前必要receipt可解析，无错误伪装为缺失", not evidence.errors, evidence.errors.copy())

    execution_valid = conjunction([item["passed"] for item in criteria])
    base = at(summary, "metrics", "baseline", "segm")
    method_metrics = at(summary, "metrics", "triflow", "segm")
    metrics = {}
    for key in ("AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge"):
        baseline_value, method_value = finite(at(base, key)), finite(at(method_metrics, key))
        metrics[key] = {"baseline": baseline_value, "triflow": method_value, "delta_fraction": method_value - baseline_value if baseline_value is not None and method_value is not None else None,
                        "delta_points": 100 * (method_value - baseline_value) if baseline_value is not None and method_value is not None else None}
    statistics = at(pair, "statistics")
    damage_rate = finite(at(statistics, "damage_rate_of_baseline_success", "value"))
    success_delta = finite(at(statistics, "baseline_success_mean_iou_delta", "value"))
    ap_delta = metrics["AP"]["delta_fraction"]
    ap75_delta, small_delta = metrics["AP75"]["delta_fraction"], metrics["APsmall"]["delta_fraction"]
    scientific = [
        {"key": "mask_ap", "description": "Mask AP增加至少0.3 AP点", "passed": ap_delta >= .003 if ap_delta is not None else None, "value": ap_delta, "unit": "AP fraction", "threshold": .003},
        {"key": "ap75_small", "description": "AP75与APsmall不能同时下降", "passed": not (ap75_delta < 0 and small_delta < 0) if ap75_delta is not None and small_delta is not None else None, "value": {"AP75": ap75_delta, "APsmall": small_delta}},
        {"key": "damage", "description": "baseline成功实例damage比例≤1%", "passed": damage_rate <= .01 if damage_rate is not None else None, "value": damage_rate, "threshold": .01},
        {"key": "success_iou", "description": "baseline成功实例平均IoU增量≥−.005", "passed": success_delta >= -.005 if success_delta is not None else None, "value": success_delta, "threshold": -.005},
    ]
    science_passed = conjunction([item["passed"] for item in scientific]) if execution_valid is True else None
    gate = conjunction([execution_valid, science_passed])
    outcome = "valid_positive" if execution_valid is True and science_passed is True else "valid_negative" if execution_valid is True and science_passed is False else "invalid_execution" if execution_valid is False else "unknown"
    return {"report_version": "triflow_report_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "study": EXPERIMENT, "selected_runs": RUNS, "execution_valid": execution_valid, "scientific_gate_passed": science_passed,
            "gate_passed": gate, "outcome": outcome, "execution_criteria": criteria, "scientific_criteria": scientific,
            "mask_metrics": metrics, "paired_statistics": statistics, "compiler_counts": at(summary, "compiler_counts"),
            "selected_instances": at(summary, "selected_instances"), "refined_instances": at(summary, "refined_instances"),
            "unsupported_roi_instances": at(summary, "unsupported_roi_instances"), "timing": at(summary, "timing"), "compiler_numeric_summary": compiler_summary,
            "head_provenance": head, "training_counts": {"epochs": at(train, "epochs"), "attempts": attempts, "applied": applied,
                                                        "instances_per_epoch": at(cache, "instances"), "trainable_parameters": at(train, "trainable_parameters")},
            "read_errors": evidence.errors, "evidence_files": evidence.files,
            "unmeasured": ["full raw candidate five-state metrics", "AUC/FPR domain", "Ucrop ceiling", "AP confidence interval", "novelty/publication suitability"],
            "limitations": ["single seed, fixed8 epochs, historical research cohort; not a new blind test",
                            "different module, loss, and budget from ACD; no single-variable causal attribution",
                            "fixed3pass active-hinge approximation plus recorded numericalridge and trust, not a globally certified inequality optimum",
                            "first64 post-conf native rows only; unsupportedROI and unselected rows keep c0; frozen box crop limits reachable foreground",
                            "paired diagnostic permits duplicate same-class GT matches, unlike COCO AP",
                            "independent verifier checks NPZ aggregates and paired point estimates; it does not recompute GT match, maskIoU, or bootstrap CI",
                            "extra module time includes Python/adapters/transfers; not optimized deployment throughput"]}


def render_report(evidence: Evidence, decision: dict) -> str:
    summary = evidence.read(run_relative("eval") / "SUMMARY.json")
    cache = evidence.read(run_relative("fit_cache") / "CACHE_RECEIPT.json")
    train = evidence.read(run_relative("train") / "TRAINING_AUDIT.json")
    numeric = evidence.read(run_relative("numerical") / "TRIFLOW_VERIFICATION.json")
    state = {"valid_positive": "本次执行有效，达到预先锁定的可行性门槛。", "valid_negative": "本次执行有效，未达到预先锁定的可行性门槛。",
             "invalid_execution": "必要执行或证据链未通过，当前不能作有效科学结论。", "unknown": "实验或证据回传/独立核验尚未完成，科学结论未知。"}[decision["outcome"]]
    lines = ["# TriFlow 首次可行性报告", "", state, "", f"生成时间（UTC）：{decision['generated_at_utc']}。执行有效性：**{verdict(decision['execution_valid'])}**；科学门槛：**{verdict(decision['scientific_gate_passed'])}**。",
             "", "| 完整original COCO5000 Mask指标 | 冻结官方 | 冻结官方 + TriFlow | 增量（AP点） |", "|---|---:|---:|---:|"]
    for key, values in decision["mask_metrics"].items():
        lines.append(f"| {key} | {fmt(values['baseline'], points=True)} | {fmt(values['triflow'], points=True)} | {fmt(values['delta_points'], signed=True)} |")
    lines.extend(["", "以上AP展示乘100；gate使用原始AP fraction，0.003等于0.3个AP点。未知不填写0。数值oracle/20项数学诊断通过不能证明AP改善。", "", "## 实际方法与预算", "",
                  "冻结官方YOLO26m-seg（Ultralytics 8.4.100），官方权重SHA256：`" + OFFICIAL_SHA + "`。所有官方参数与BN保持冻结；仅训练新TriFlow模块。输入是真实P3特征、32通道prototype、原生one2one系数、最后系数卷积的真实hidden query及预测邻居。未使用ACD微调checkpoint。",
                  "", "1. Conflict Tokenizer：64个预测位置，默认32边界/16实例竞争/16prototype leverage；无预测邻居时48边界/16leverage。选择不读GT，小ROI补位为零权重。",
                  "2. Tri-Ownership Interaction：Self/Neighbor/BG cross-attention预测连续φN、φB和stiffness，无Direct32输出和ownership CE。φ为GT距离差先相减再裁剪到±16网格像素，未除以16。GT仅用于训练监督，Self优先，crowd忽略；空GT区域以及缺失预测邻居使用validity mask。",
                  "3. Ownership Flow：同一连续场头查询法线射线和邻近probe，Self envelope=min(valid φN,φB)的交点决定带符号位移。z>0为前景，外法线−∇z/||∇z||；约束b=δs||∇z||−z(x0)。无根/零梯度不制造flow。",
                  "4. Prototype Jacobian Compiler：固定3pass active-hinge weighted32×32 SPD求解；已满足anchor初始零force，后续pass重新激活违反项。保留ROI Gram/行权重归一化和原ridge，加预定义FP32 scale-aware numericalridge，并验证实际Cholesky info。求解具有数值ridge与logit/系数trust bound；有限pass不保证全局KKT或所有约束满足。",
                  "", "损失为Potential SmoothL1、同场head邻近查询的Direction cosine、真实接口法线上的Ordering，以及Compiler输出mask的task BCE+Dice。stiffness有floor，并以GT势幅度定义其监督目标。",
                  "", f"正式fit：796张，实际匹配instance/epoch：{cell(at(cache, 'instances'))}；预算8epoch，seed0，每图chunk4，FP32无AMP/增广，AdamW lr=.0003/weight_decay=.0001，梯度裁剪10。实际attempt/applied：{cell(at(train, 'optimizer_attempts'))}/{cell(at(train, 'optimizer_applied'))}；可训练参数：{cell(at(train, 'trainable_parameters'))}。dev197不用作选择，保存固定final8，不自动延长/增加seed。",
                  "", "训练采用class-free最大boxIoU≥.5、first64 native post-conf候选、raw index去重及每GT round-robin选最多12。部署在GT-free的first64预测域运行top2预测邻居；候选预算内无法执行的空native ROI保持c0且不补位，其他候选保持c0。官方bbox支持区固定，超框前景不可恢复。",
                  "", f"数值诊断：{cell(at(numeric, 'check_count'))}项，诊断状态{cell(at(numeric, 'status'))}；包含oracle扩缩/邻居/ignore/空区域、实际task-only梯度和优化器更新、FP32高尺度rank1、真实官方冻结native重放。诊断效应不是正式训练或AP效应。",
                  "", "## 回放、读出与配对范围", "",
                  "完整5000 val2017、640 square LetterBox/scaleup=false、native one2one FP32、conf=.001/max_det300；原生process_mask_native后scale_preds.byte()。COCOeval maxDets=[1,10,100]，segm读出剔除bbox字段以保留mask area语义。",
                  "", "官方baseline复用先前COCO输出字节，但本次必须逐5000图fresh冻结模型forward与原生decoder，证明每图检测identity与RLE完全相同。原baseline的legacy部分state digest与本次完整official state digest范围不同，不制造相等；本次两arm均绑定同一个完整冻结状态。",
                  "", "独立NumPy读出复核重新聚合COCO NPZ的AP/AP75/APsmall并比较bbox累计数组，核对5000图、first64/unsupported/预测邻居/no-GT域及paired点估计；不重新计算GT mask匹配、maskIoU或bootstrap置信区间。",
                  "", "paired诊断是same native detection index/class对应的同类非crowd GT最大boxIoU≥.5，允许多个预测匹配同一GT。baseline maskIoU≥.75算成功，掉到.75以下为damage，失败变成功为repair；它和COCO AP匹配、原始候选五状态几何不是同一指标。Image-cluster bootstrap5000/seed0仅用于paired IoU/damage；未计算AP CI。",
                  "", "| 配对点估计 | 值 |", "|---|---:|"])
    statistics = decision["paired_statistics"]
    for key in ("matched_detection_count", "baseline_success_count", "damage_count", "repair_count"):
        lines.append(f"| {key} | {cell(at(statistics, key))} |")
    for key in ("mean_mask_iou_delta", "damage_rate_of_baseline_success", "repair_rate_of_baseline_failure", "baseline_success_mean_iou_delta"):
        lines.append(f"| {key} | {fmt(at(statistics, key, 'value'), signed='delta' in key, precision=6)} |")
    lines.extend(["", "## 推理域与开销", "", "| 项目 | 实测 |", "|---|---:|"])
    for key in ("selected_instances", "refined_instances", "unsupported_roi_instances"):
        lines.append(f"| {key} | {cell(decision[key])} |")
    for key in ("boundary_valid_count", "root_count", "invalid_boundary_root_count", "neighbor_transition_count", "active_anchor_count", "trust_saturated_instances"):
        lines.append(f"| {key} | {cell(at(decision, 'compiler_counts', key))} |")
    for key in ("solver_instances", "solver_nonzero_cholesky_info_count", "gram_numeric_ridge_instances", "system_numeric_ridge_instances", "maximum_recorded_system_condition", "maximum_system_residual"):
        lines.append(f"| {key} | {cell(at(decision, 'compiler_numeric_summary', key))} |")
    for key in ("actual_frozen_forward_seconds", "extra_module_seconds", "extra_module_ms_per_image", "total_elapsed_seconds"):
        lines.append(f"| {key} | {fmt(at(decision, 'timing', key), precision=3)} |")
    lines.extend(["", "时间包含Python、adapter和数据转移，不能当成优化部署吞吐。具体numericalridge、逐pass条件数/违反次数/oscillation及trust scale保留于TRAIN_TRACE/TASK_GRADIENT_EVIDENCE和COMPILER_IMAGES。invalid_boundary_root是无可用root的边界row，不等于solver失败；非SPD/非有限solver会使Run明确失败。", "", "## 预先锁定的科学门槛", "", "| 门槛 | 原始读出条件 |", "|---|---|"])
    for criterion in decision["scientific_criteria"]:
        lines.append(f"| {criterion['description']} | {verdict(criterion['passed'])} |")
    lines.extend(["", "只有以下全部必要执行/证据链通过，才把上述数值条件解释为有效正/负结果。", "", "## 执行与证据链", "", "| 检查 | 状态 |", "|---|---|"])
    for criterion in decision["execution_criteria"]:
        lines.append(f"| {criterion['description']} | {verdict(criterion['passed'])} |")
    lines.extend(["", "## Run记录与已知失败", "", "| Run | 执行 | exit | 产物 | 回传 |", "|---|---|---:|---|---|"])
    directories = set(evidence.path("runs").glob("*")) if evidence.path("runs").exists() else set()
    directories |= {evidence.path(Path("runs") / run) for run in RUNS.values()}
    for directory in sorted(directories):
        if not directory.is_dir() and directory.name not in RUNS.values():
            continue
        meta = evidence.read(directory.relative_to(evidence.root) / "run.json")
        lines.append(f"| {markdown_link(directory / 'REPORT.md', directory.name) if directory.is_dir() else directory.name} | {cell(at(meta, 'execution_status'))} | {cell(at(meta, 'exit_code'))} | {cell(at(meta, 'artifact_status'))} | {cell(at(meta, 'transfer_status'))} |")
    lines.extend(["", "首次preflight与numerical S0失败历史保留，不被R1成功覆写。最初anchor fixture λ·ε=1e−8在单位尺度共线FP32系统里被舍掉；R1用正式ridge，另加大尺度rank1数值ridge验证。旧Pipeline/Train S0在加强真实φ-task/attention同probe证据前保留中断/失败记录，新pipeline、smoke、contract和正式train使用实际R1 ID；它们不延长训练预算。正式核心已锁，未把diagnostic oracle或warm-up算成正式训练。", "", "## 结论限制与未测项", "",
                  "本次为历史cohort、单seed、固定8轮新模块试验；与ACD的可训练范围、目标和训练预算不同，不能作单变量因果归因。最终可说明此固定部署域/预算是否达到门槛；不能据此证明论文新颖性、普遍有效或普遍无效。raw五状态/AUC/FPR域/Ucrop/AP CI及论文发表适用性保持未知。", "", "关键产物：",
                  "", "- " + markdown_link(evidence.path("PROTOCOL.md"), "锁定协议"),
                  "- " + markdown_link(evidence.path(run_relative("eval") / "DECISION.json"), "三值门槛决定"),
                  "- " + markdown_link(evidence.path(run_relative("eval") / "SUMMARY.json"), "正式评估原始SUMMARY（若已回传）"),
                  "- " + markdown_link(evidence.path(run_relative("eval") / "READOUT_VERIFICATION.json"), "独立aggregate核验（若已完成）")])
    if decision["read_errors"]:
        lines.extend(["", "文件读取/解析错误（不作通过处理）：", "", *["- " + cell(error) for error in decision["read_errors"]]])
    return "\n".join(lines)


def render_run(evidence: Evidence, directory: Path) -> str:
    relative = directory.relative_to(evidence.root)
    meta = evidence.read(relative / "run.json")
    lines = ["# " + directory.name, "", f"执行：{cell(at(meta, 'execution_status'))}；exit：{cell(at(meta, 'exit_code'))}；产物：{cell(at(meta, 'artifact_status'))}；回传：{cell(at(meta, 'transfer_status'))}。",
             "", f"开始：{cell(at(meta, 'started_at'))}；结束：{cell(at(meta, 'ended_at'))}。未回传/未记录时保持未知。", ""]
    for filename, label in (("PREFLIGHT.json", "预检"), ("TRIFLOW_VERIFICATION.json", "方法数值/冻结诊断"), ("CACHE_RECEIPT.json", "冻结cache"),
                            ("TRAINING_AUDIT.json", "新模块训练审计"), ("SUMMARY.json", "正式读出"), ("READOUT_VERIFICATION.json", "独立aggregate核验")):
        value = evidence.read(relative / filename)
        if value is None:
            continue
        lines.append(f"{label}：{markdown_link(directory / filename)}；状态：{cell(value.get('status', value.get('passed')))}。")
        if filename == "TRIFLOW_VERIFICATION.json":
            lines.append(f"诊断checks：{cell(value.get('check_count'))}；failed：{cell(value.get('failed_checks'))}；oracle和诊断优化器不构成正式AP结果。")
        elif filename == "CACHE_RECEIPT.json":
            lines.append(f"images：{cell(value.get('images'))}；instances：{cell(value.get('instances'))}；cache bytes：{cell(value.get('cache_bytes'))}。大FP32 tensors可留远端，以receipt inventory SHA与训练逐payload验证为据。")
        elif filename == "TRAINING_AUDIT.json":
            lines.append(f"epochs：{cell(value.get('epochs'))}；attempt/applied：{cell(value.get('optimizer_attempts'))}/{cell(value.get('optimizer_applied'))}；actual task-gradient：{cell(value.get('compiler_task_gradient_observed_nonzero'))}。smoke1轮与formal8轮独立初始化，smoke不得作为formal结果。")
    for filename in ("FAILED.json", "TRAINING_FAILURE.json", "CACHE_FAILURE.json", "EVALUATION_FAILURE.json"):
        value = evidence.read(relative / filename)
        if value is not None:
            lines.append("保留的失败：" + markdown_link(directory / filename) + "；" + cell(value.get("error", value.get("reason"))))
    lines.extend(["", markdown_link(evidence.root / "REPORT.md", "完整实验报告") + "；Run结论须结合正式执行、source/config/hash、回传和独立核验。"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-run", help="Actual immutable formal-training retry ID, if different from S0")
    parser.add_argument("--pipeline-run", help="Actual immutable pipeline retry ID, if different from S0")
    parser.add_argument("--eval-run", help="Actual immutable evaluation retry ID, if different from S0")
    args = parser.parse_args()
    for role in ("train", "pipeline", "eval"):
        selected = getattr(args, role + "_run")
        if selected is not None:
            if not re.fullmatch(r"RUN_TRIFLOW_[A-Z0-9_]+", selected):
                raise ValueError("Explicit actual Run ID must belong to TriFlow, without path components")
            RUNS[role] = selected
    root = args.root.resolve()
    if root.name != EXPERIMENT and (root / "experiments" / EXPERIMENT).is_dir():
        root = root / "experiments" / EXPERIMENT
    if root.name != EXPERIMENT or not (root / "study.json").is_file():
        raise ValueError("Expected the actual TriFlow experiment directory with study.json")
    evidence = Evidence(root)
    decision = collect_decision(evidence)
    report = render_report(evidence, decision)
    atomic_text(root / "REPORT.md", report)
    for directory in sorted((root / "runs").iterdir()):
        if directory.is_dir():
            atomic_text(directory / "REPORT.md", render_run(evidence, directory))
    # Only derived files are overwritten. Evidence receipts/source/meta stay exact.
    decision["read_errors"] = evidence.errors
    decision["evidence_files"] = evidence.files
    atomic_text(evidence.path(run_relative("eval") / "DECISION.json"), json.dumps(clean(decision), ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"report": str(root / "REPORT.md"), "outcome": decision["outcome"], "execution_valid": decision["execution_valid"],
                      "scientific_gate_passed": decision["scientific_gate_passed"], "criteria": len(decision["execution_criteria"]), "read_errors": evidence.errors}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
