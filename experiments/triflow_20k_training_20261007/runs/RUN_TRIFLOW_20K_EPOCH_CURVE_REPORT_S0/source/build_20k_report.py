"""Compact factual declared20k TriFlow report; incomplete evidence stays pending.

Training coverage, execution integrity, and benefit are separate decisions.
Only derived REPORT.md and evaluation DECISION.json are written; source,
checkpoint, receipts, run metadata and historical pilot evidence are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT = "triflow_20k_training_20261007"
ROOT = Path("C:/Dpan/codexproject/paper-disc-organized/experiments") / EXPERIMENT
TRAIN_RUN = "RUN_TRIFLOW_20K_TRAIN_S0"
EVAL_RUN = "RUN_TRIFLOW_20K_COCO5000_EVAL_S0"
OFFICIAL = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
TRAIN_ANNOTATIONS = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
DECLARED_IMAGES = 20000


def at(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def eq(value, expected):
    return None if value is None else value == expected


def same(a, b):
    return None if a is None or b is None else a == b


def all_known(values):
    return False if False in values else True if values and all(v is True for v in values) else None


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, scale=1, signed=False):
    value = number(value)
    return "未知" if value is None else format(value * scale, "+.5f" if signed else ".5f")


def label(value):
    return "通过" if value is True else "未通过" if value is False else "待完成/未知"


class Reader:
    def __init__(self, root):
        self.root = root
        self.errors, self.evidence = [], {}
        self.hashes = {}

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Evidence path escapes the experiment")
        return path

    def digest(self, name):
        path = self.path(name)
        if not path.is_file():
            return None
        if path not in self.hashes:
            self.hashes[path] = sha(path)
            self.evidence[path.relative_to(self.root).as_posix()] = {"sha256": self.hashes[path], "bytes": path.stat().st_size}
        return self.hashes[path]

    def read(self, name):
        path = self.path(name)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object")
            self.digest(name)
            return value
        except (ValueError, OSError) as error:
            self.errors.append(f"{name}: {error}")
            return None

    def binding(self, name, digest):
        actual = self.digest(name)
        return None if actual is None or digest is None else bool(re.fullmatch(r"[0-9a-f]{64}", str(digest))) and actual == digest


def audit_binding(reader, run, audit):
    if audit is None:
        return None
    evidence = audit.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        return False
    checks = [eq(audit.get("passed"), True), eq(audit.get("status"), "passed"), eq(audit.get("errors"), []), eq(audit.get("missing"), [])]
    names, aliases = set(), set()
    for filename, record in evidence.items():
        if not isinstance(filename, str) or not isinstance(record, dict):
            checks.append(False)
            continue
        normalized = filename.replace("\\", "/")
        path = Path(normalized)
        canonical = path.as_posix()
        if path.is_absolute() or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in path.parts or canonical.casefold() in aliases:
            checks.append(False)
            continue
        aliases.add(canonical.casefold())
        names.add(canonical)
        checks.append(reader.binding(run / path, record.get("sha256")))
    required = {"SUMMARY.json", "HEAD_PROVENANCE.json", "BASELINE_REPLAY_VERIFICATION.json", "COMPILER_IMAGES.jsonl",
                "baseline/COCO_segm_ACCUMULATED.npz", "triflow/COCO_segm_ACCUMULATED.npz",
                "baseline/COCO_bbox_ACCUMULATED.npz", "triflow/COCO_bbox_ACCUMULATED.npz",
                "head_provenance/STREAM_RECEIPT.json", "head_provenance/STREAM_INPUTS.json", "head_provenance/EPOCHS.json"}
    checks.append(required.issubset(names))
    return all_known(checks)


def generate(root, train_id, eval_id, verification=None):
    reader = Reader(root)
    train_dir, eval_dir = Path("runs") / train_id, Path("runs") / eval_id
    inputs = reader.read(train_dir / "TRAINING_INPUTS.json")
    training = reader.read(train_dir / "TRAINING_AUDIT.json")
    stream = reader.read(train_dir / "STREAM_RECEIPT.json")
    epoch_receipt = reader.read(train_dir / "EPOCHS.json")
    summary = reader.read(eval_dir / "SUMMARY.json")
    head = reader.read(eval_dir / "HEAD_PROVENANCE.json")
    verification_path = Path(verification) if verification is not None else eval_dir / "READOUT_VERIFICATION.json"
    readout = reader.read(verification_path)
    completed = reader.read(eval_dir / "EVALUATION_COMPLETE.json")
    epoch_rows = at(epoch_receipt, "epochs")
    coverage_checks = [eq(at(stream, "complete_declared_train_subset"), True), eq(at(stream, "complete_original_train_split"), False), eq(at(stream, "listed_images"), DECLARED_IMAGES),
                       eq(at(stream, "original_split_images"), 118287), eq(at(stream, "observed_unique_images"), DECLARED_IMAGES),
                       eq(at(stream, "all_listed_images_observed"), True), eq(at(stream, "engineering_subset"), False),
                       eq(at(stream, "annotation_sha256"), TRAIN_ANNOTATIONS), eq(at(training, "complete_declared_train_subset"), True), eq(at(training, "complete_original_train_split"), False),
                       eq(at(training, "smoke_only"), False), eq(at(training, "completed_epochs"), 8),
                       eq(at(training, "every_epoch_declared_images_complete"), True), eq(at(training, "every_epoch_actual_unique_images_complete"), True)]
    if isinstance(epoch_rows, list):
        coverage_checks.append(len(epoch_rows) == 8 and [row.get("epoch") for row in epoch_rows] == list(range(1, 9)) and all(row.get("images") == DECLARED_IMAGES for row in epoch_rows))
        coverage_checks.append(all(row.get("visited_unique_images") == DECLARED_IMAGES and row.get("visited_image_identity_sha256") == at(stream, "image_identity_sha256") and row.get("actual_completed_image_ids_match_declared") is True for row in epoch_rows))
    else:
        coverage_checks.append(None)
    coverage = all_known(coverage_checks)
    criteria = [
        {"key": "training_coverage", "description": "锁定hash-ranked20k身份及实际8轮覆盖", "passed": coverage},
        {"key": "training", "description": "正式训练审计/官方权重/同probeφ与attention task梯度", "passed": all_known([
            eq(at(training, "passed"), True), eq(at(training, "base_weights_sha256"), OFFICIAL), eq(at(training, "epochs"), 8),
            eq(at(training, "all_applied_gradients_finite"), True), eq(at(training, "all_parameters_finite"), True),
            eq(at(training, "compiler_task_phi_and_attention_same_probe_observed_nonzero"), True)])},
        {"key": "loaded_head", "description": "实际加载20k final8、online receipt与epoch/history/source字节绑定", "passed": all_known([
            eq(at(head, "kind"), "triflow_20k_module_final"), eq(at(head, "epoch"), 8), eq(at(head, "smoke_only"), False),
            eq(at(head, "complete_declared_train_subset"), True), eq(at(head, "complete_original_train_split"), False), eq(at(head, "engineering_subset"), False),
            eq(at(head, "all_epochs_declared_subset_coverage_verified"), True), eq(at(head, "declared_train_images"), DECLARED_IMAGES),
            reader.binding(train_dir / "head_final.pt", at(head, "head_sha256")),
            reader.binding(train_dir / "TRAINING_AUDIT.json", at(head, "training_audit_sha256")),
            reader.binding(train_dir / "STREAM_RECEIPT.json", at(head, "stream_receipt_sha256")),
            reader.binding(train_dir / "STREAM_INPUTS.json", at(head, "stream_inputs_sha256")),
            reader.binding(train_dir / "EPOCHS.json", at(head, "epochs_sha256")), same(head, at(summary, "head_provenance"))])},
        {"key": "evaluation", "description": "original5000 evaluation完成且SUMMARY bytes绑定", "passed": all_known([
            eq(at(summary, "status"), "complete"), eq(at(summary, "image_count"), 5000), eq(at(summary, "covers_all_annotation_images"), True),
            reader.binding(eval_dir / "SUMMARY.json", at(completed, "summary_sha256")), eq(at(summary, "learned_module_state_exact"), True)])},
        {"key": "baseline_replay", "description": "fresh5000冻结官方前向/原生decoder identity/RLE同一对照", "passed": all_known([
            eq(at(summary, "baseline_replay", "status"), "passed"), eq(at(summary, "baseline_replay", "images"), 5000),
            eq(at(summary, "baseline_replay", "all_native_forward_exact"), True), eq(at(summary, "baseline_replay", "all_baseline_identity_and_rle_exact"), True)])},
        {"key": "independent_readout", "description": "独立NumPy AP/paired聚合与当前引用文件SHA", "passed": audit_binding(reader, eval_dir, readout)},
    ]
    # A legitimately unfinished epoch list/provider counter is progress, not
    # invalid scientific execution. Final audits or explicit failure metadata
    # distinguish missing work from a completed-but-insufficient experiment.
    train_meta = reader.read(train_dir / "run.json")
    eval_meta = reader.read(eval_dir / "run.json")
    active_failure = any(at(meta, "execution_status") in ("failed", "cancelled") or at(meta, "status") in ("failed", "cancelled") for meta in (train_meta, eval_meta))
    if training is None and not active_failure:
        coverage = None
        criteria[0]["passed"] = None
    if summary is None and not active_failure:
        for row in criteria:
            if row["key"] in ("loaded_head", "evaluation", "baseline_replay", "independent_readout"):
                row["passed"] = None
    if active_failure:
        criteria.append({"key": "selected_execution_failed", "description": "当前选定Run没有显式failed/cancelled", "passed": False})
    execution_valid = all_known([row["passed"] for row in criteria])
    metrics = {arm: at(summary, "metrics", arm, "segm") for arm in ("baseline", "triflow")}
    deltas = {key: number(at(metrics, "triflow", key)) - number(at(metrics, "baseline", key))
              if number(at(metrics, "triflow", key)) is not None and number(at(metrics, "baseline", key)) is not None else None for key in ("AP", "AP75", "APsmall")}
    statistics = at(summary, "paired", "triflow", "statistics")
    damage = number(at(statistics, "damage_rate_of_baseline_success", "value"))
    success_delta = number(at(statistics, "baseline_success_mean_iou_delta", "value"))
    scientific = [
        {"description": "MaskAP增量≥0.3 AP点", "passed": deltas["AP"] >= .003 if deltas["AP"] is not None else None},
        {"description": "AP75/APsmall不同时下降", "passed": not (deltas["AP75"] < 0 and deltas["APsmall"] < 0) if deltas["AP75"] is not None and deltas["APsmall"] is not None else None},
        {"description": "baseline成功damage≤1%", "passed": damage <= .01 if damage is not None else None},
        {"description": "baseline成功平均IoU增量≥−.005", "passed": success_delta >= -.005 if success_delta is not None else None},
    ]
    science_passed = all_known([row["passed"] for row in scientific]) if execution_valid is True else None
    outcome = "valid_positive" if execution_valid is True and science_passed is True else "valid_negative" if execution_valid is True and science_passed is False else "invalid_execution" if execution_valid is False else "pending"
    decision = {"report_version": "triflow_20k_report_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "training_run": train_id,
                "evaluation_run": eval_id, "verification_path": str(reader.path(verification_path)), "training_coverage_sufficient": coverage, "execution_valid": execution_valid,
                "scientific_gate_passed": science_passed, "outcome": outcome, "execution_criteria": criteria, "scientific_criteria": scientific,
                "metrics": metrics, "delta_fraction": deltas, "paired_statistics": statistics, "timing": at(summary, "timing"),
                "read_errors": reader.errors, "evidence_files": reader.evidence,
                "unknown": ["raw five-state geometry", "AUC/FPR", "crop ceiling", "AP confidence interval", "training seed uncertainty", "paper novelty"]}
    if reader.errors:
        decision["execution_valid"] = False
        decision["scientific_gate_passed"] = None
        decision["outcome"] = "invalid_execution"
    title = {"pending": "实验或证据尚未完成，收益未知。", "valid_positive": "当前20k配置执行有效并达到预先门槛。", "valid_negative": "当前20k配置执行有效，未达到预先门槛。", "invalid_execution": "存在必要证据不通过，不能形成有效收益结论。"}[decision["outcome"]]
    lines = ["# TriFlow2万图训练报告", "", title, "", f"训练覆盖：{label(coverage)}；执行完整性：{label(decision['execution_valid'])}；收益门槛：{label(decision['scientific_gate_passed'])}。",
             "", "方法核心/冻结official/source/hash不变，从原始118287图按锁定SHA排名选20000图在线FP32、fixed8epoch、seed0、microchunk4每chunk更新、AdamW3e−4/WD1e−4/clip10。读取实际provider/训练/audit证据；不能把工程子集或旧796图称20k结论训练。",
             "", "| Mask指标 | 官方 | TriFlow | 增量AP点 |", "|---|---:|---:|---:|"]
    for key in ("AP", "AP75", "APsmall"):
        lines.append(f"| {key} | {fmt(at(metrics, 'baseline', key), 100)} | {fmt(at(metrics, 'triflow', key), 100)} | {fmt(deltas[key], 100, True)} |")
    lines.extend(["", f"damage：{fmt(damage, 100)}%；baseline成功平均IoU增量：{fmt(success_delta, signed=True)}。AP显示乘100，门槛0.003为0.3 AP点。", "", "| 必要证据/覆盖 | 状态 |", "|---|---|"])
    for row in criteria:
        lines.append(f"| {row['description']} | {label(row['passed'])} |")
    lines.extend(["", "| 收益门槛 | 原始条件 |", "|---|---|"])
    for row in scientific:
        lines.append(f"| {row['description']} | {label(row['passed'])} |")
    lines.extend(["", "完整5000 COCO val2017、native one2one/640/FP32/conf.001/max_det300、COCOeval maxDets[1,10,100]；segm使用mask area。GT不进入部署forward，bbox/score/class/order保持冻结身份；first64预算和fixedbox crop支持范围沿原方法。",
                  "", "训练覆盖指声明的20000图集合，不冒称原始118287图全量。实例监督仍按first64预测候选、class-free boxIoU≥.5、raw去重与GT round-robin每图最多12选取；没有可用候选的图像仍须实际遍历，但没有mask-task更新。没有进入Self候选的GT不能冒称获得了完整实例监督；miss/exclusion和crowd状态保留于provider计数。",
                  "", "baseline复用需fresh5000逐图原生identity/RLE一致。legacy部分state digest与当前whole-model digest不能混为相同范围。独立读出仅重聚合NPZ与paired点估计，不重算GT mask匹配/IoU/bootstrap CI或训练模型。",
                  "", "paired诊断为同native detection、同类GT最大boxIoU≥.5，允许重复GT匹配；IoU=.75定义修复/损伤，与COCO AP和raw几何不同。额外时延包含Python/传输；不是优化部署吞吐。",
                  "", "旧796图只是pilot。本次按用户约定使用20k作结论阶段：从原始118287图按sha256('triflow_train20k_seed0:'+str(image_id))排名取20000，再按imageID排序，选择不读取GT类别、mask或图像内容。20k规模不是对所有方法/配置的万能充分性定理；单seed结果仍只约束当前固定配置。未测raw五状态/AUC/FPR/Ucrop/AP CI及论文新颖性保持未知，不把门槛未达推广为方法能力上限，也不把扩大数据预设为收益保证。",
                  "", "运行、恢复、中断和失败历史按原Run保留；checkpoint恢复只读取已提交模型/optimizer/RNG/游标和来源契约，未提交更新可回滚重放，不能伪造额外已完成轮次。",
                  "", f"生成时间：{decision['generated_at_utc']}。"])
    return decision, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train-run", default=TRAIN_RUN)
    parser.add_argument("--eval-run", default=EVAL_RUN)
    parser.add_argument("--run-id", help="Optional actual report-execution Run; its receipt binds the derived files")
    parser.add_argument("--verification", type=Path, help="Actual independent audit JSON, within this experiment; missing remains pending")
    args = parser.parse_args()
    for selected in (args.train_run, args.eval_run, *([args.run_id] if args.run_id is not None else [])):
        if not re.fullmatch(r"RUN_TRIFLOW_20K_[A-Z0-9_]+", selected):
            raise ValueError("Expected an explicit 20k Run ID without path components")
    root = args.root.resolve()
    if root.name != EXPERIMENT or not (root / "study.json").is_file():
        raise ValueError("Expected the actual 20k experiment directory")
    decision, report = generate(root, args.train_run, args.eval_run, args.verification)
    for path, content in ((root / "REPORT.md", report), (root / "runs" / args.eval_run / "DECISION.json", json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content + "\n", encoding="utf-8")
        temporary.replace(path)
    if args.run_id is not None:
        report_run = root / "runs" / args.run_id
        report_run.mkdir(parents=True, exist_ok=True)
        run_report = report_run / "REPORT.md"
        run_report.write_text(report + "\n", encoding="utf-8")
        receipt = {"status": "completed", "report_generation_only": True, "scientific_outcome": decision["outcome"],
                   "training_coverage_sufficient": decision["training_coverage_sufficient"], "execution_valid": decision["execution_valid"],
                   "report_file": str(root / "REPORT.md"), "report_sha256": sha(root / "REPORT.md"),
                   "decision_file": str(root / "runs" / args.eval_run / "DECISION.json"), "decision_sha256": sha(root / "runs" / args.eval_run / "DECISION.json"),
                   "run_report_sha256": sha(run_report), "verification_path": decision["verification_path"],
                   "verification_sha256": sha(Path(decision["verification_path"])) if Path(decision["verification_path"]).is_file() else None,
                   "script_sha256": sha(Path(__file__)), "generated_at_utc": decision["generated_at_utc"]}
        temporary = report_run / "REPORT_RECEIPT.json.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(report_run / "REPORT_RECEIPT.json")
    print(json.dumps({"outcome": decision["outcome"], "coverage": decision["training_coverage_sufficient"], "execution_valid": decision["execution_valid"]}))


if __name__ == "__main__":
    main()
