"""Read existing sources/logs/receipts only; never import or execute model code.

This is a document audit, not a scientific Run. Online epoch means are descriptive
and must not be interpreted as fixed-checkpoint evaluation or convergence tests.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = {}


def record(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    key = path.relative_to(ROOT).as_posix()
    if key not in SOURCES:
        data = path.read_bytes()
        SOURCES[key] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return SOURCES[key]


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    record(path)
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    record(path)
    with (ROOT / path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


TRAIN = {
    "B": "mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2",
    "C": "mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_C_81f47e540104405bba84e1f1f29e2261",
    "SOFT": "mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb",
    "CLD_REAL": "mask_direction_learnability_20261016/runs/RUN_CLD_FULL_REAL_bc58d9f5253847e588eb01ce2a309d7b",
    "CLD_PERM": "mask_direction_learnability_20261016/runs/RUN_CLD_FULL_PERM_a131a5d6626845878b5b93b9b51a7bfa",
    "LRP_LOCAL": "mask_local_residual_pilot_20261017/runs/RUN_LRP_FULL_LOCAL_564e54cb99764cd894b856bbc2f3fe47",
    "LRP_UNIFORM": "mask_local_residual_pilot_20261017/runs/RUN_LRP_FULL_UNIFORM_cc6633af9a1840188dfa899cf7c1a811",
}
EVIDENCE = {
    "BC": "mask_readout_mechanism_v2_20261012/runs/RUN_S2_EVAL_a24a564568c943eab362784abfbe2c54",
    "S1": "mask_readout_mechanism_v2_20261012/runs/RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949",
    "FINITE_C": "mask_repair_preservation_20261016/runs/RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01",
    "FIT": "mask_readout_fit_diagnostic_20261014/runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926",
    "PROTO": "mask_readout_proto_consistency_20261014/runs/RUN_PROTO_AUDIT_FULL_2a0b0e0afeb64cd7a1300a10aac1a30f",
    "SOFT_FIT": "mask_soft_fit_audit_20261016/runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770",
    "S3": "mask_gate_s3r_20261015/runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d",
    "OPPORTUNITY": "mask_opportunity_audit_20261015/runs/RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd",
    "RAY": "mask_adapter_ray_audit_20261016/runs/RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b",
    "AMT": "mask_attainable_teacher_20261016/runs/RUN_AMT_S2_FULL_5dbbed12ed4b4f0989826ff4eabf3357",
    "CLD": "mask_direction_learnability_20261016/runs/RUN_CLD_EVALUATION_3bf05f04bb6e47c9abf9bb46cc4e1f47",
    "CET": "mask_coefficient_error_transfer_20261017/runs/RUN_CET_FULL_3d2de9164a2f434eba4ff3a756690c45",
    "LEA": "mask_local_editability_20261017/runs/RUN_LEA_FULL_b383a8db922c451cbc77f2069d385cd7",
    "LRP": "mask_local_residual_pilot_20261017/runs/RUN_LRP_EVALUATION_f89d8084795244a985b37031a6b395a9",
}
CODE_NAMES = {
    "adapter_model.py", "adapter_data.py", "adapter_primitives.py", "s1_geometry.py",
    "native_output_bridge.py", "run_s2_training.py", "run_training.py", "soft_iou_loss.py",
    "cld_math.py", "train_cld.py", "lrp_math.py", "lrp_prepare.py", "lrp_train.py",
}


def run_summary(folder):
    summary = read(folder / "SUMMARY.json")
    decision = read(folder / "DECISION.json")
    expected = decision.get("summary_sha256", decision.get("evidence_refs", {}).get("SUMMARY.json"))
    actual = record(folder / "SUMMARY.json")["sha256"]
    if expected is not None and expected != actual:
        raise ValueError(f"summary pin mismatch: {folder}")
    metadata = read(folder / "run.json")
    return summary, dict(path=folder.as_posix(), summary_sha256=actual,
                         decision_pin_present=expected is not None,
                         decision_pin_matches=expected == actual if expected else None,
                         original_decision=decision, original_run_metadata=metadata)


def source_checks(folder):
    manifest = read(folder / "SOURCE_MANIFEST.json")["files"]
    map_path = folder / "SOURCE_SNAPSHOT_MAP.json"
    mapping = read(map_path) if (ROOT / map_path).exists() else {}
    result = []
    for name, expected in manifest.items():
        if Path(name).name not in CODE_NAMES:
            continue
        current = record(name)["sha256"]
        candidates = [ROOT / folder / mapping[name]] if name in mapping else list(
            (ROOT / folder / "source_snapshot").glob("*_" + Path(name).name))
        matched = [p for p in candidates if record(p)["sha256"] == expected]
        if current != expected or not matched:
            raise ValueError(f"code/source snapshot mismatch: {folder}: {name}")
        result.append(dict(path=name, expected_sha256=expected, current_matches=True,
                           snapshot=matched[0].relative_to(ROOT).as_posix(), snapshot_matches=True))
    return result


def main():
    train = {}
    for label, relative in TRAIN.items():
        folder = Path("experiments") / relative
        summary, receipt = run_summary(folder)
        log = rows(folder / "training.jsonl")
        contract = read(folder / "EVAL_CONTRACT.json")
        read(folder / "ACTUAL_OPTIMIZER.json")
        read(folder / "CONFIG_AUDIT.json")
        for name in ("INDEPENDENT_CONFIG_AUDIT.json", "INDEPENDENT.json", "LOCAL_INDEPENDENT_RECHECK.json"):
            if (ROOT / folder / name).exists():
                record(folder / name)
        checkpoint_sha = record(folder / "last.pt")["sha256"]
        if checkpoint_sha != summary["checkpoint_sha256"]:
            raise ValueError(f"checkpoint pin mismatch: {label}")
        loss_fields = [key for key in log[0] if "loss" in key]
        epochs = []
        for epoch in sorted({r["epoch"] for r in log}):
            group = [r for r in log if r["epoch"] == epoch]
            if not all(math.isfinite(float(r[k])) for r in group for k in loss_fields):
                raise ValueError("nonfinite stored log")
            epochs.append(dict(epoch=epoch, image_visits=len(group), unique_images=len({r["image_id"] for r in group}),
                               online_mean={k: math.fsum(r[k] for r in group) / len(group) for k in loss_fields}))
        expected_visits = contract.get("expected_visits", contract.get("expected_image_visits", contract.get("fit_images", 0) * contract.get("epochs", 0)))
        if len(log) != expected_visits:
            raise ValueError(f"image visit mismatch: {label}")
        train[label] = dict(**receipt, summary=summary, code_checks=source_checks(folder),
                            checkpoint_pin_matches=True, contract=contract, epochs=epochs,
                            log_fields=list(log[0]), log_rows=len(log),
                            interpretation="online losses only; no new checkpoint evaluation, gradient norm or convergence evidence")

    evidence = {}
    for label, relative in EVIDENCE.items():
        folder = Path("experiments") / relative
        summary, receipt = run_summary(folder)
        evidence[label] = dict(**receipt, summary=summary)
        # Capture relevant existing receipts, without executing their verifiers.
        for name in ("EVAL_CONTRACT.json", "CONFIG_AUDIT.json", "INDEPENDENT_LAPTOP.json", "INDEPENDENT_LOCAL.json",
                     "INDEPENDENT.json", "CONFIGURATION_REVIEW.md", "LOCAL_RESULT_RECHECK.json"):
            if (ROOT / folder / name).exists():
                record(folder / name)

    cache = Path("experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec")
    cache_summary, cache_receipt = run_summary(cache)
    trace_path = cache / "trace/TRACE_MANIFEST.json"
    trace = read(trace_path)
    if record(trace_path)["sha256"] != cache_summary["trace_manifest_sha256"]:
        raise ValueError("cache manifest pin mismatch")
    actual_shapes = Counter(str(r["input_hw"]) for r in trace["images"])
    cache_info = dict(**cache_receipt, images=len(trace["images"]), input_shapes=dict(actual_shapes),
                      source_model_sha256=trace["model_sha256"], trace_sha256=record(trace_path)["sha256"],
                      source_environment=trace["environment"], native_args=trace["native_args"],
                      observed_channels=sorted({tuple(s[0] for s in r["head_feature_shapes"]) for r in trace["images"]}),
                      observed_strides=sorted({tuple(r["input_hw"][0] // s[1] for s in r["head_feature_shapes"]) for r in trace["images"]}),
                      feature_sources=trace["images"][0]["feature_sources"],
                      traced_predictions=sum(len(r["prediction_indices"]) for r in trace["images"]),
                      raw_cache_files_present_locally=sum((ROOT / cache / "trace" / r["cache"]).is_file() for r in trace["images"]),
                      cache_scope="existing manifests and prior receipts; no current raw-cache tensor replay")
    pins = read("shared/evaluation/native_source_pins.json")["sha256_normalized_utf8"]
    vendor = {}
    for module in ("ultralytics.nn.modules.head", "ultralytics.utils.ops", "ultralytics.models.yolo.segment.val"):
        path = Path("shared/vendor/ultralytics_8_4_100") / (module.replace(".", "/") + ".py")
        record(path)
        normalized = hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        if normalized != pins[module]:
            raise ValueError(f"vendor source pin mismatch: {module}")
        vendor[module] = dict(path=path.as_posix(), normalized_sha256=normalized, matches_native_pin=True)

    extra = [
        "experiments/mask_readout_sphere_recovery_20261015/scripts/spherical_model.py",
        "experiments/mask_readout_aligned_repulsion_20261015/scripts/run_training.py",
        "experiments/mask_readout_aligned_repulsion_20261015/scripts/aligned_loss.py",
        "experiments/mask_attainable_teacher_20261016/scripts/training.py",
        "experiments/mask_readout_mechanism_v2_20261012/S2_REPORT_20261013.md",
        "experiments/mask_local_residual_pilot_20261017/INDEPENDENT_CONFIGURATION.json",
        "experiments/mask_local_residual_pilot_20261017/CONFIGURATION_REVIEW.md",
        "experiments/mask_local_residual_pilot_20261017/TRANSFER_VERIFICATION.json",
        "experiments/mask_local_editability_20261017/CONFIGURATION_REVIEW.md",
        "experiments/mask_direction_learnability_20261016/EVALUATION_CONFIGURATION_REVIEW.md",
        "experiments/mask_direction_learnability_20261016/EVALUATION_LOCAL_VERIFICATION.json",
    ]
    studies = sorted({Path(p).parts[0] for p in EVIDENCE.values()})
    for study in studies:
        for name in ("REPORT.md", "PROTOCOL.md"):
            path = Path("experiments") / study / name
            if (ROOT / path).exists():
                extra.append(path)
    for path in extra:
        record(path)
    record(__file__)
    record(OUT / "SCOPE.md")
    result = dict(schema_version=1, reviewed_at=datetime.now(timezone.utc).isoformat(),
                  audit_type="static_source_and_existing_artifact_review_not_scientific_run",
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  no_model_import_forward_training_evaluation_or_new_intervention=True,
                  training=train, evidence=evidence, cache=cache_info, vendor_source_pins=vendor,
                  sources=SOURCES,
                  integrity=dict(source_snapshot_checks=sum(len(t["code_checks"]) for t in train.values()),
                                 checkpoints_checked=len(train), summary_pins_checked=len(train)+len(evidence)+1,
                                 original_artifacts_modified=False))
    target = OUT / "EXISTING_ARTIFACT_REVIEW.json"
    # This owned review receipt is regenerable; original experiment files are read-only.
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(target), sources=len(SOURCES), **result["integrity"])))


if __name__ == "__main__":
    main()
