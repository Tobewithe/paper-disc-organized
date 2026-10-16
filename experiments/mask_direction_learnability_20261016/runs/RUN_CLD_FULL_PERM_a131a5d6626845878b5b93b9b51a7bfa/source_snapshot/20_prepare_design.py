"""Build/verify deterministic design metadata. No tensors, model runs, or training.

After SCIENTIFIC_FREEZE.json exists, outputs must be byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
CACHE = "experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec"
INITIAL = "experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2"
TEACHER = "experiments/mask_attainable_teacher_20261016/runs/RUN_AMT_TEACHER_FULL_24a4d184c8e54f45bf4327b0ab4710ca"
FIT = "experiments/mask_soft_fit_audit_20261016/runs/RUN_SOFT_FIT_FULL_14971e618ba8498a9fa8efcd19d21770"
OLD = "experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def lines(relative):
    with (ROOT / relative).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def publish(name, value, compact=False):
    data = canonical(value) if compact else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path = STUDY / name
    if path.exists():
        if path.read_bytes() != data and (STUDY / "SCIENTIFIC_FREEZE.json").exists():
            raise RuntimeError(f"Refusing to revise existing design artifact: {name}")
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    return {"path": name, "sha256": digest(data), "size_bytes": len(data)}


def main():
    source_files = {
        f"{CACHE}/SPLIT_CONTRACT.json": "original_fit2000_and_dev512_identity",
        f"{CACHE}/trace/TRACE_MANIFEST.json": "native_2000_cache_geometry_and_row_identity",
        f"{INITIAL}/initial.pt": "only_train_initialization",
        f"{INITIAL}/EVAL_CONTRACT.json": "original_optimizer_architecture_runtime_contract",
        f"{TEACHER}/TEACHER_ROWS.jsonl": "train2017_only_per_instance_teacher_coefficients",
        f"{TEACHER}/TEACHER_MANIFEST.json": "teacher_tensor_packages_reference_not_student_inputs",
        f"{TEACHER}/SUMMARY.json": "historical_teacher_validity",
        f"{FIT}/FIT_FIXED_PAIRS.jsonl": "fixed_bbox75_pairs_and_existing_method_IoU",
        f"{FIT}/BASELINE_GT_LEDGER.jsonl": "native_task_denominators",
        f"{FIT}/FIT_TASKS.jsonl": "historical_native_U_G_task_results",
        f"{OLD}/last.pt": "historical_comparator_only_never_feature_extractor",
        "experiments/mask_readout_mechanism_v2_20261012/scripts/adapter_model.py": "unchanged_167872_parameter_architecture",
        "experiments/mask_readout_mechanism_v2_20261012/scripts/adapter_primitives.py": "unchanged_ROI_sampling",
        "experiments/mask_readout_mechanism_v2_20261012/scripts/adapter_data.py": "cached_data_access_reference",
        "shared/evaluation/native_output_bridge.py": "native_decoder_identity_reference",
        "shared/evaluation/native_source_pins.json": "ultralytics_source_pins",
        "paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json": "native_validation_profile",
        "experiments/mask_adapter_ray_audit_20261016/runs/RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b/DECISION.json": "direction_priority_evidence_not_training_data",
    }
    records = []
    for relative, role in source_files.items():
        data = (ROOT / relative).read_bytes()
        records.append({"path": relative, "sha256": digest(data), "size_bytes": len(data), "role": role, "local_read_verified": True})
    split = read(f"{CACHE}/SPLIT_CONTRACT.json")
    trace = read(f"{CACHE}/trace/TRACE_MANIFEST.json")
    teachers = lines(f"{TEACHER}/TEACHER_ROWS.jsonl")
    pairs = lines(f"{FIT}/FIT_FIXED_PAIRS.jsonl")
    ledger = lines(f"{FIT}/BASELINE_GT_LEDGER.jsonl")
    images = {int(item["image_id"]): item for item in trace["images"]}
    ids = split["fit_ids"]
    assert len(ids) == len(set(ids)) == 2000
    assert set(ids) == set(images)
    assert set(row["image_id"] for row in teachers).issubset(images)  # Some images have no valid GT join.
    assert not set(ids).intersection(split["dev_ids"])
    ranked = sorted(ids, key=lambda i: (digest(f"direction-learnability-v1|image_id={i}".encode("ascii")), i))
    held = set(ranked[:400])
    fit = set(ranked[400:])
    fit_order = [i for i in ids if i in fit]
    artifacts = []
    artifacts.append(publish("SPLIT.json", {
        "algorithm": "sha256 ASCII direction-learnability-v1|image_id=<decimal>; ascending digest then id; first 400 heldout",
        "fit_image_ids": fit_order, "heldout_image_ids": [i for i in ids if i in held],
        "original_fit_count": 2000, "fit_count": 1600, "heldout_count": 400,
        "overlap_count": 0, "validation_images_used_for_training": 0,
        "scope": "new_learner_image_holdout_within_previously_exposed_train2000_not_independent_confirmation",
    }))
    rng = random.Random(0)
    orders = []
    for _ in range(3):
        order = list(fit_order)
        rng.shuffle(order)
        orders.append(order)
    artifacts.append(publish("IMAGE_ORDERS.json", {"seed": 0, "algorithm": "persistent Python random.Random(0), fresh original-order filtered fit list each epoch", "epochs": orders, "image_visits_per_arm": 4800, "optimizer_steps_per_arm": 1200}))
    maps = []
    singletons = []
    for image_id in ids:
        indices = images[image_id]["prediction_indices"]
        order = sorted(range(len(indices)), key=lambda r: (digest(f"direction-condition-v1|image_id={image_id}|global_index={indices[r]}".encode("ascii")), r))
        mapping = list(range(len(indices)))
        if len(order) > 1:
            for k, recipient in enumerate(order):
                mapping[recipient] = order[(k + 1) % len(order)]
            assert all(r != donor for r, donor in enumerate(mapping))
        else:
            singletons.append(image_id)
        assert sorted(mapping) == list(range(len(indices)))
        maps.append({"image_id": image_id, "recipient_rank_to_donor_rank": mapping})
    artifacts.append(publish("CONDITION_MAP.json", {
        "algorithm": "within-image SHA256 ordering of ranks using direction-condition-v1|image_id=<id>|global_index=<index>; cyclic next rank donor; n<=1 identity",
        "selection_uses_GT": False, "same_map_all_epochs_and_intervention": True, "images": maps,
    }, compact=True))
    subsets = {}
    for name, selected in (("fit", fit), ("heldout", held)):
        rows = [r for r in teachers if r["image_id"] in selected]
        accepted = [r for r in rows if r["accepted"]]
        gate = [r for r in accepted if r["native_area"] >= 9216]
        fixed = [r for r in pairs if r["image_id"] in selected]
        tasks = [r for r in ledger if r["image_id"] in selected]
        subsets[name] = {
            "images": len(selected), "outputs": sum(len(images[i]["prediction_indices"]) for i in selected),
            "training_join_rows": len(rows), "accepted_teachers": len(accepted), "accepted_teacher_images": len({r["image_id"] for r in accepted}),
            "gate_on_accepted_teachers": len(gate), "gate_on_accepted_teacher_images": len({r["image_id"] for r in gate}),
            "fixed_bbox75_pairs": len(fixed), "fixed_bbox75_pair_images": len({r["image_id"] for r in fixed}),
            "fixed_pair_population_counts": dict(Counter(r["population"] for r in fixed)),
            "fixed_pair_population_images": {p: len({r["image_id"] for r in fixed if r["population"] == p}) for p in sorted({r["population"] for r in fixed})},
            "ordinary_GT": sum(not r["is_ignored"] for r in tasks),
            "native_success_GT": sum(r["task_mask75"] == "success" for r in tasks),
            "singleton_or_empty_output_images": sorted(selected.intersection(singletons)),
        }
        for row in rows:
            item = images[row["image_id"]]
            assert row["global_index"] == item["prediction_indices"][row["rank"]]
            assert row["cache_sha256"] == item["cache_sha256"]
            assert row["output_detection_id"] == f'{trace["predictions_sha256"]}:{row["global_index"]}'
            if row["accepted"]:
                assert len(row["coefficients_fp32"]) == 32
                assert row["solver_status"] == "converged" and row["oracle_iou"] - row["baseline_iou"] >= .01
    assert (subsets["fit"]["accepted_teachers"], subsets["heldout"]["accepted_teachers"]) == (2510, 581)
    assert subsets["heldout"]["outputs"] == 42167 and subsets["heldout"]["native_success_GT"] == 1808
    artifacts.append(publish("DATA_COUNTS.json", {"kind": "deterministic_existing_metadata_audit_not_model_experiment", "subsets": subsets, "all_teacher_rows_identity_verified": len(teachers), "all_output_condition_map_rows": sum(len(x["prediction_indices"]) for x in images.values()), "teacher_amplitude_value": None, "teacher_amplitude_status": "fit_only_deterministic_derivation_pending_tensor_preflight"}))
    artifacts.append(publish("INHERITED_CONTRACT.json", {
        "source_training_contract": read(f"{INITIAL}/EVAL_CONTRACT.json"),
        "actual_cached_native_args": trace["native_args"], "source_environment": trace["environment"],
        "overrides": {"fit_images": 1600, "epochs": 3, "expected_optimizer_steps": 1200, "image_visits": 4800, "arms": ["REAL", "PERM"], "objective": "PROTOCOL.md section 4", "parameter_count": 167872, "preservation_weight": .1},
    }))
    artifacts.append(publish("INPUT_MANIFEST.json", {
        "files": records, "source_trace_cache_files": 2000, "source_trace_cache_bytes": trace["cache_bytes"],
        "native_train_prediction_sha256": trace["predictions_sha256"],
        "evaluation_annotations": {"executor_relative_path": f"{CACHE}/dataset/training_annotations.json", "sha256": trace["annotations_sha256"], "scope": "frozen_train2000_subset"},
        "source_annotations": {"executor_path": split["source_train_annotations"], "sha256": split["source_train_annotations_sha256"], "scope": "original_full_train2017_source_not_the_evaluation_file"},
        "executor_root": "D:\\coco_wire\\baseline_native_20261013", "executor_host": "YANGLEI / ssh 28358lan",
        "local_manifest_hashes_do_not_assert_remote_tensor_availability": True,
    }))
    print(json.dumps({"status": "PASS", "scope": "design_metadata_only", "new_training_steps": 0, "new_model_forwards": 0, "artifacts": artifacts, "counts": subsets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
