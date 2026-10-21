"""Read-only recheck of existing receipts; no model, training, or new AP draws."""
from pathlib import Path
from collections import Counter
from array import array
import ast
import datetime as dt
import hashlib
import json
import math
import statistics
import struct
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
ROOT = STUDY.parents[1]
ARMS = ("G_budget", "C_native", "PC_native")
METHODS = ("Ns", "G_old_s", *ARMS)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def close(a, b, tolerance=1e-10):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) <= tolerance, (a, b)


def load_npy(path, dtype, shape):
    with path.open("rb") as stream:
        assert stream.read(6) == b"\x93NUMPY"
        major, minor = stream.read(2)
        assert major in (1, 2, 3)
        n = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(stream.read(n).decode("utf-8" if major == 3 else "latin1"))
        assert header["descr"] == dtype and header["shape"] == shape
        assert not header["fortran_order"]
        values = array("d" if dtype == "<f8" else "i")
        values.frombytes(stream.read())
        if sys.byteorder != "little":
            values.byteswap()
        assert len(values) == math.prod(shape)
        return values


def percentile(values, percent):
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main():
    cfg = read(STUDY / "CONFIG.json")
    frozen = read(STUDY / "FROZEN_MANIFEST.json")
    pointers = read(HERE / "remote_snapshot/POINTERS.json")
    paths = {key: STUDY / "runs" / v["run_id"] for key, v in pointers.items()}
    inventory = read(HERE / "REMOTE_ARTIFACT_INVENTORY.json")
    remote_files = {x["path"]: x for x in inventory["files"]}
    transfer = read(HERE / "TRANSFER_MANIFEST.json")
    source_pins = read(ROOT / "shared/evaluation/native_source_pins.json")["sha256_normalized_utf8"]
    environment_records = 0
    for item in transfer["files"]:
        path = STUDY / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert sha(path) == item["sha256"], item["path"]
    for item in frozen["files"]:
        assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
    assert sha(STUDY / "CONFIG.json") == frozen["config_sha256"]
    assert sha(STUDY / "PROTOCOL.md") == frozen["protocol_sha256"]
    artifact_count = 0
    for stage, path in paths.items():
        meta = read(path / "run.json")
        assert meta["execution_status"] == "completed", stage
        assert meta["config_sha256"] == frozen["config_sha256"]
        assert meta["protocol_sha256"] == frozen["protocol_sha256"]
        assert meta["frozen_manifest_sha256"] == sha(STUDY / "FROZEN_MANIFEST.json")
        assert read(path / "CONFIG_FROZEN.json") == cfg
        assert sha(path / "SUMMARY.json") == pointers[stage]["summary_sha256"]
        for name, digest in read(path / "ARTIFACT_MANIFEST.json").items():
            relative = str((path / name).relative_to(STUDY)).replace("\\", "/")
            assert remote_files[relative]["sha256"] == digest, relative
            if (path / name).exists():
                assert sha(path / name) == digest, relative
            artifact_count += 1
        for relative, digest in read(path / "SOURCE_MANIFEST.json").items():
            assert sha(path / "source_snapshot" / relative) == digest
            assert sha(ROOT / relative) == digest
        if (path / "ENVIRONMENT.json").exists():
            env = read(path / "ENVIRONMENT.json")
            assert env["python_version"].startswith(cfg["expected_python"] + " ")
            assert env["torch_version"] == cfg["expected_packages"]["torch"]
            assert env["ultralytics_version"] == cfg["expected_packages"]["ultralytics"]
            assert env["source_sha256"] == source_pins
            for key, value in env["packages"].items():
                assert value == cfg["expected_packages"][key]
            for key, value in cfg["runtime_policy"].items():
                assert env[key] == value
            environment_records += 1

    prepared = paths["PREPARE"]
    contract = read(prepared / "DATA_CONTRACT.json")
    orders = read(prepared / "IMAGE_ORDERS.json")
    assert len(set(contract["train_ids"])) == 20000
    assert len(set(contract["val_ids"])) == 5000
    assert not set(contract["train_ids"]) & set(contract["val_ids"])
    assert len(contract["original_fit_ids"]) == 2000
    assert set(contract["original_fit_ids"]) <= set(contract["train_ids"])
    assert len(contract["train_batches"]) == 1250 and len(contract["val_batches"]) == 313
    assert len(contract["train_image_hashes"]) == 20000 and len(contract["val_image_hashes"]) == 5000
    for split in ("train", "val"):
        assert contract["inputs"][split + "_annotations"]["sha256"] == cfg["expected_annotations"][split]
    expected = []
    step = 0
    for epoch, sequence in enumerate(orders):
        assert len(sequence) == 1250
        for item in sequence:
            batch = contract["train_batches"][item["source_batch_index"]]
            assert sorted(x for group in item["microgroups"] for x in group) == list(range(16))
            for group in item["microgroups"]:
                assert len(group) == 4
                step += 1
                expected.extend((epoch, step, batch["image_ids"][position]) for position in group)
    assert step == 15000 and len(expected) == 60000 and len(orders) == 3
    assert set(Counter(x[2] for x in expected).values()) == {3}
    train_records = {}
    actual_optimizers = []
    for arm, param_count in zip(ARMS, (167872, 854496, 3158016)):
        path = paths["TRAIN_" + arm]
        summary = read(path / "SUMMARY.json")
        audit = read(path / "CONFIG_AUDIT.json")
        environment = read(path / "ENVIRONMENT.json")
        assert summary["optimizer_steps"] == 15000 and summary["images_seen"] == 60000
        assert summary["epochs"] == 3 and summary["checkpoint_selection"] == "last_only"
        assert summary["parameters"]["trainable_parameter_count"] == param_count
        assert summary["initial_state_sha256"] != summary["final_state_sha256"]
        assert set(summary["parameters_with_nonzero_gradient"]) == set(summary["parameters"]["trainable_whitelist"])
        assert audit["status"] == "passed" and audit["frozen_base"]["status"] == "passed"
        assert audit["frozen_base"]["external_native_state"]["status"] == "passed"
        assert audit["same_cross_arm_native_inputs_and_GT_join"]
        assert audit["runtime_policy"] == cfg["runtime_policy"]
        assert sha(path / "IMAGE_ORDERS.json") == sha(prepared / "IMAGE_ORDERS.json")
        actual = read(path / "ACTUAL_OPTIMIZER.json")
        actual_optimizers.append(actual)
        for key, value in cfg["optimizer"].items():
            if key != "name":
                assert actual[key] == value
        assert all(actual[key] is False for key in ("amsgrad", "maximize", "capturable", "differentiable"))
        assert environment["python_version"].startswith(cfg["expected_python"] + " ")
        assert environment["torch_version"] == cfg["expected_packages"]["torch"]
        assert environment["ultralytics_version"] == cfg["expected_packages"]["ultralytics"]
        for key, value in environment["packages"].items():
            assert value == cfg["expected_packages"][key]
        for key, value in cfg["runtime_policy"].items():
            assert environment[key] == value
        for key, value in environment["source_sha256"].items():
            assert value == source_pins[key]
        args = read(path / "native_config/NATIVE_ARGS.json")
        required_args = {"imgsz": 640, "batch": 16, "rect": True, "conf": .001,
                         "iou": .7, "max_det": 300, "end2end": True, "agnostic_nms": False,
                         "retina_masks": False, "mask_ratio": 4, "overlap_mask": True,
                         "augment": False}
        for key, value in required_args.items():
            assert args[key] == value, (key, args[key])
        count = 0
        for i, row in enumerate(lines(path / "training.jsonl")):
            assert (row["epoch"], row["step"], row["image_id"]) == expected[i]
            assert all(math.isfinite(row[k]) for k in ("loss", "gt_loss", "preservation_loss"))
            count += 1
        assert count == 60000
        cp_relative = str((path / "last.pt").relative_to(STUDY)).replace("\\", "/")
        assert remote_files[cp_relative]["sha256"] == summary["checkpoint_sha256"] == audit["final_checkpoint_sha256"]
        assert read(paths["EVAL_" + arm] / "SUMMARY.json")["checkpoint_sha256"] == summary["checkpoint_sha256"]
        assert max(summary["max_cuda_allocated"], summary["max_cuda_reserved"]) <= 44 * 2 ** 30
        train_records[arm] = {
            "run_id": path.name, "optimizer_steps": 15000, "logged_images": count,
            "parameters": param_count, "checkpoint_sha256": summary["checkpoint_sha256"],
            "parameter_update_counts": summary["parameter_update_counts"],
            "max_reserved_GiB": summary["max_cuda_reserved"] / 2 ** 30,
        }
    assert actual_optimizers[0] == actual_optimizers[1] == actual_optimizers[2]
    for name in ("TRAINING_NATIVE_IDENTITY.json", "TRAINING_JOIN.json"):
        digests = {remote_files[str((paths["TRAIN_" + arm] / name).relative_to(STUDY)).replace("\\", "/")]["sha256"] for arm in ARMS}
        assert len(digests) == 1, name

    native = paths["NATIVE"]
    stat_path = paths["STATISTICS"]
    stat = read(stat_path / "SUMMARY.json")
    assert stat == read(stat_path / "DECISION.json")
    native_summary = read(native / "SUMMARY.json")
    assert native_summary["migration_status"] == "passed"
    assert native_summary["native_gt_ledger_validity"] == "passed"
    assert native_summary["published_parity"] == "failed"
    geometry = read(native / "HISTORICAL_GEOMETRY_REPLAY.json")
    assert len(geometry) == 5000 and all(x["historical_input_and_geometry_exact"] for x in geometry)
    baseline = {x["annotation_id"]: x for x in lines(native / "Ns_PER_GT.jsonl")}
    assert sum(not x["is_ignored"] for x in baseline.values()) == 36335
    success = {k for k, v in baseline.items() if not v["is_ignored"] and v["task_mask75"] == "success"}
    assert len(success) == 22958
    census = read(stat_path / "DAMAGE_CENSUS.json")
    method_success = {}
    damage = {}
    metrics = {}
    inference_costs = {}
    for method in METHODS:
        path = native if method in METHODS[:2] else paths["EVAL_" + method]
        metric = read(path / (method + "_METRICS.json"))
        metrics[method] = metric
        assert metric["images"] == 5000 and metric["outputs"] == 539125
        assert metric["ordinary_gt"] == 36335
        close(metric["metrics"]["bbox"]["AP"], 52.34795017624022)
        for kind in ("bbox", "segm"):
            close(metric["metrics"][kind]["AP"], metric["faster_coco_eval_AP"][kind])
        pred_relative = str((path / (method + "_predictions.json")).relative_to(STUDY)).replace("\\", "/")
        assert remote_files[pred_relative]["sha256"] == metric["predictions_sha256"]
        current = {x["annotation_id"]: x for x in lines(path / (method + "_PER_GT.jsonl"))}
        assert current.keys() == baseline.keys()
        for aid, row in current.items():
            old = baseline[aid]
            assert (row["image_id"], row["is_ignored"], row["task_box75"]) == (old["image_id"], old["is_ignored"], old["task_box75"])
            old_id, new_id = old["bbox75_match_output_id"], row["bbox75_match_output_id"]
            assert (old_id.rsplit(":", 1)[-1] if old_id else None) == (new_id.rsplit(":", 1)[-1] if new_id else None)
        current_success = {k for k, v in current.items() if not v["is_ignored"] and v["task_mask75"] == "success"}
        damaged, repaired = sorted(success - current_success), sorted(current_success - success)
        assert damaged == census[method]["damaged_annotation_ids"]
        assert repaired == census[method]["repaired_annotation_ids"]
        assert len(current_success) == metric["mask75_success"]
        assert len(damaged) == census[method]["damaged"] and len(repaired) == census[method]["repaired"]
        close(len(damaged) / len(success), census[method]["damage_fraction"])
        method_success[method] = current_success
        damage[method] = {"damaged": len(damaged), "repaired": len(repaired),
                          "net_repaired": len(repaired) - len(damaged), "damage_fraction": len(damaged) / len(success)}
        if method in ARMS:
            summary = read(path / "SUMMARY.json")
            assert summary["full_row_identity_guard"] == "passed"
            assert read(path / "FROZEN_BASE_CHECK.json")["status"] == "passed"
            times = read(path / "BATCH_TIMINGS.json")
            eligible = [x for x in times[1:] if x["images"] == 16]
            assert len(eligible) == 311
            ratio = statistics.median(x["method_seconds"] for x in eligible) / statistics.median(x["native_only_seconds"] for x in eligible)
            close(ratio, summary["inference_cost"]["method_over_native_ratio"])
            inference_costs[method] = ratio

    values = load_npy(stat_path / "BOOTSTRAP_AP_POINTS.npy", "<f8", (2000, 5))
    assert all(math.isfinite(x) for x in values)
    draws = load_npy(stat_path / "PAIRED_IMAGE_DRAWS.npy", "<i4", (2000, 5000))
    assert min(draws) >= 0 and max(draws) < 5000
    assert sha(stat_path / "PAIRED_IMAGE_DRAWS.npy") == stat["bootstrap"]["common_draws_sha256"]
    seen = set()
    for row in lines(stat_path / "BOOTSTRAP_PROGRESS.jsonl"):
        i = row["replicate"]
        assert i not in seen and 0 <= i < 2000
        seen.add(i)
        for j, value in enumerate(row["AP_points"]):
            close(value, values[i * 5 + j], 0)
    assert len(seen) == 2000
    comparisons = {}
    for section in ("relative_to_Ns_and_existing_G", "primary_comparisons"):
        for name, saved in stat[section].items():
            method, reference = name.split("_minus_")
            a, b = METHODS.index(method), METHODS.index(reference)
            delta = metrics[method]["metrics"]["segm"]["AP"] - metrics[reference]["metrics"]["segm"]["AP"]
            distribution = [values[i * 5 + a] - values[i * 5 + b] for i in range(2000)]
            ci = [percentile(distribution, p) for p in (2.5, 97.5)]
            close(delta, saved["AP_points"])
            for x, y in zip(ci, saved["ci95"]):
                close(x, y)
            comparisons[name] = {"AP_points": delta, "ci95": ci}
            if section == "primary_comparisons":
                threshold = {"Ns": .5, "G_old_s": .3, "G_budget": .2, "C_native": .2}[reference]
                assert saved["minimum_AP_points"] == threshold
                assert saved["passed"] == (delta >= threshold and ci[0] > 0)
    # Descriptive comparison from the same saved replicates, not a new selection gate.
    gc_distribution = [values[i * 5 + 2] - values[i * 5 + 3] for i in range(2000)]
    descriptive_gc = {"AP_points": stat["Mask_AP_points"]["G_budget"] - stat["Mask_AP_points"]["C_native"],
                      "ci95": [percentile(gc_distribution, p) for p in (2.5, 97.5)],
                      "scope": "descriptive post-hoc reduction of existing paired samples; no new gate"}
    pairwise_success = {}
    for method in ARMS:
        pairwise_success[method] = {
            "lost_vs_G_old": len(method_success["G_old_s"] - method_success[method]),
            "gained_vs_G_old": len(method_success[method] - method_success["G_old_s"]),
            "difference_in_native_damage_count": damage[method]["damaged"] - damage["G_old_s"]["damaged"],
        }
    assert sorted(method_success["G_old_s"] - method_success["PC_native"]) == stat["PC_vs_G_old_task_success_change"]["lost_annotation_ids"]
    assert sorted(method_success["PC_native"] - method_success["G_old_s"]) == stat["PC_vs_G_old_task_success_change"]["gained_annotation_ids"]
    assert stat["primary_gate_passed"] is False
    assert all(v is True for v in stat["guards"].values())
    assert damage["PC_native"]["damage_fraction"] <= .01 and inference_costs["PC_native"] <= 4
    literal = read(stat_path / "BOOTSTRAP_LITERAL_VERIFICATION.json")
    assert all(x["status"] == "passed" and x["exact_precision_recall_scores"] for x in literal.values())
    result = {
        "status": "passed", "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "independent standard-library recomputation from saved AP draws, GT ledgers, timings, and training records; no COCO/model rerun",
        "additional_training_steps": 0, "new_model_forwards": 0, "new_bootstrap_replicates": 0,
        "python": sys.version, "script_sha256": sha(Path(__file__)),
        "frozen_files_sha_verified_locally": len(frozen["files"]),
        "transferred_files_sha_verified": len(transfer["files"]),
        "original_manifest_artifacts_reconciled_with_fresh_remote_sha": artifact_count,
        "remote_inventory_files": len(remote_files), "large_files_retained_on_server": 27,
        "runtime_environment_records_checked": environment_records,
        "runs": {k: p.name for k, p in paths.items()}, "formal_training": train_records,
        "data": {"fit_images": 20000, "val_images": 5000, "ordinary_gt": 36335, "native_success_gt": 22958,
                 "outputs_per_method": 539125, "cross_arm_training_identity_and_join_sha_equal": True},
        "bootstrap": {"saved_replicates_verified": 2000, "seed": 20261018,
                      "common_draws_shape": [2000, 5000], "interval_reduction": "independent linear percentile 2.5/97.5",
                      "COCO_accumulation_not_rerun_locally": True},
        "Mask_AP_points": stat["Mask_AP_points"], "comparisons": comparisons,
        "damage": damage, "success_change_vs_existing_G": pairwise_success,
        "descriptive_G_budget_minus_C_native": descriptive_gc,
        "inference_median_ratios": inference_costs,
        "primary_gate_passed": False, "guards": stat["guards"],
        "decision": stat["decision"], "published_baseline_parity": "failed",
        "native_gt_ledger_validity": "passed",
        "notes": [
            "Native validator args amp=True/epochs=100/optimizer=auto are inactive trainer defaults, not this custom training loop; FP32 backend and frozen actual AdamW used.",
            "G adapter parameter eligible-update counts 14996/14999/15000 preserve grad=None skips; global optimizer calls remain 15000.",
            "PC primary rejection is retained; highest observed G_budget is an existing-method control result, not PC contribution or independent confirmation.",
        ],
    }
    output = HERE / "LOCAL_INDEPENDENT_RECHECK.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "frozen_files_sha_verified_locally", "transferred_files_sha_verified", "damage", "descriptive_G_budget_minus_C_native", "primary_gate_passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
