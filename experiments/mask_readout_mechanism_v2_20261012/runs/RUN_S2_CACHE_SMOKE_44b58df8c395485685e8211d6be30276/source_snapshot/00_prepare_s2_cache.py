"""Freeze train/dev IDs and export frozen native train2017 features, with explicit COCO API aliases."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from posttrain_run import ROOT, STUDY, RegisteredRun, read_json, require, sha256, write_new, configure_native

BASE_ID = "RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170"
BASE = ROOT / "experiments/official_gt_failure_rebaseline_20261013/runs" / BASE_ID
S1_ID = "RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949"
FULL_GT = Path("D:/coco_wire/bgcr_native_20261002/data/annotations/instances_train2017.json")
IMAGE_ROOT = Path("D:/coco_wire/data/images/train2017")
ARCHIVE = ROOT / "assets/dependencies/baseline_20261013/coco2017labels-segments.zip"
WEIGHTS = ROOT / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
CODE_FILES = [
    "experiments/mask_readout_mechanism_v2_20261012/scripts/prepare_s2_cache.py",
    "experiments/mask_readout_mechanism_v2_20261012/scripts/posttrain_run.py",
    "experiments/mask_readout_mechanism_v2_20261012/S2_EXECUTION_20261013.md",
    "shared/evaluation/native_output_bridge.py", "shared/evaluation/native_source_pins.json",
    "shared/evaluation/posttrain_audit.py", "paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json",
    "shared/vendor/ultralytics_8_4_100/ultralytics/cfg/datasets/coco.yaml",
]


def ranked(ids, prefix):
    return sorted(ids, key=lambda value: (hashlib.sha256(f"{prefix}:0:{value}".encode()).hexdigest(), value))


def history_lists(train_ids):
    excluded, known_fit, records = set(), set(), []
    path = Path("D:/coco_wire/data/annotations/calibration2000_split.json")
    split = read_json(path)
    for key, role in (("fit_ids", "fit"), ("selection_ids", "selection")):
        ids = set(split[key]) & train_ids
        (known_fit if role == "fit" else excluded).update(ids)
        records.append({"path": str(path), "sha256": sha256(path), "field": key, "role": role,
                        "train2017_images": len(ids), "image_ids": sorted(ids)})
    for name, role in (("fit_win.txt", "fit"), ("dev_win.txt", "selection")):
        path = Path("D:/coco_wire/acd_native_20261006") / name
        ids = {int(Path(line.strip()).stem) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
        ids &= train_ids
        (known_fit if role == "fit" else excluded).update(ids)
        records.append({"path": str(path), "sha256": sha256(path), "role": role,
                        "train2017_images": len(ids), "image_ids": sorted(ids)})
    return excluded, known_fit, records


def prepare(run, scope, profile):
    import yaml
    import psutil
    from PIL import Image
    require(psutil.virtual_memory().available >= 5 * 1024**3, "need 5 GiB available RAM for full annotation read")
    require(shutil.disk_usage(run.path).free >= 20 * 1024**3, "need 20 GiB for bounded train cache and pilot")
    require(sha256(WEIGHTS) == profile["model_sha256"], "weight changed")
    archive_sha = sha256(ARCHIVE)
    require(archive_sha == "039d39b6548fa9dbcc1bb84e722953d25ebc04935772805e5d5c7badefcb349b", "official label archive changed")
    full_gt_sha = sha256(FULL_GT)
    data = read_json(FULL_GT)
    require(len(data["images"]) == 118287 and len(data["annotations"]) == 860001 and len(data["categories"]) == 80,
            "training source is not the complete declared COCO train2017 inventory")
    image_info = {image["id"]: image for image in data["images"]}
    train_ids = set(image_info)
    excluded, known_fit, historical = history_lists(train_ids)
    available = {int(p.stem): p for p in IMAGE_ROOT.glob("*.jpg") if p.stem.isdigit() and int(p.stem) in train_ids}
    eligible = set(available) - excluded
    fit_ids = ranked(eligible, "s2-fit-v1")[:2000]
    require(len(fit_ids) == 2000, "less than 2000 eligible training images; no substitution")
    val_manifest = read_json(BASE / "trace/TRACE_MANIFEST.json")
    val_ids = set(val_manifest["image_ids"])
    require(not set(fit_ids) & val_ids, "train/dev image overlap")
    sys.path.insert(0, str(ROOT / "shared/evaluation"))
    from posttrain_audit import read_jsonl, population
    baseline_rows = read_jsonl(BASE / "ledger/PER_GT_FAILURE.jsonl")
    target_counts = Counter(row["image_id"] for row in baseline_rows if population(row) == "T")
    dev_order = ranked(val_ids, "s2-dev-v1")
    dev_ids = dev_order[:512]
    cursor = 512
    while sum(target_counts[i] for i in dev_ids) < 200:
        require(cursor < len(dev_order), "insufficient target population for dev")
        dev_ids.append(dev_order[cursor])
        cursor += 1
    membership = {"fit_ids": fit_ids, "dev_ids": dev_ids, "fit_dataset_split": "COCO_train2017",
                  "dev_dataset_split": "COCO_val2017_development_only", "fit_selection": "s2-fit-v1:0:image_id_SHA256",
                  "dev_selection": "s2-dev-v1:0:image_id_SHA256_then_baseline_T_count_only",
                  "source_train_annotations": str(FULL_GT), "source_train_annotations_sha256": full_gt_sha,
                  "source_train_images": len(train_ids), "available_images": len(available),
                  "excluded_known_selection": len(set(available) & excluded), "eligible_images": len(eligible),
                  "known_prior_fit_overlap": len(set(fit_ids) & known_fit),
                  "known_prior_selection_overlap": len(set(fit_ids) & excluded),
                  "history_scope": "listed calibration2000 and ACD manifests; no claim of exhaustive historical non-use",
                  "historical_manifests": historical, "dev_T": sum(target_counts[i] for i in dev_ids)}
    write_new(run.path / "SPLIT_CONTRACT.json", membership)
    fit_set = set(fit_ids)
    subset = {"info": data["info"], "licenses": data["licenses"], "categories": data["categories"],
              "images": [image_info[i] for i in fit_ids],
              "annotations": [row for row in data["annotations"] if row["image_id"] in fit_set]}
    del data, image_info, baseline_rows
    gc.collect()
    dataset = run.path / "dataset"
    (dataset / "images/train2017").mkdir(parents=True)
    (dataset / "labels/train2017").mkdir(parents=True)
    (dataset / "annotations").mkdir()
    write_new(dataset / "training_annotations.json", subset)
    # This is an API filename alias, explicitly recorded as train2017; not val data.
    os.link(dataset / "training_annotations.json", dataset / "annotations/instances_val2017.json")
    records = []
    info = {row["id"]: row for row in subset["images"]}
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = set(archive.namelist())
        require("coco/train2017.txt" in members, "wrong official label archive structure")
        for image_id in fit_ids:
            run.check_budget()
            source = available[image_id]
            target = dataset / "images/train2017" / source.name
            digest = sha256(source)
            with Image.open(source) as im:
                require(list(im.size) == [info[image_id]["width"], info[image_id]["height"]], "image/annotation size mismatch")
            os.link(source, target)
            label_name = "coco/labels/train2017/" + source.with_suffix(".txt").name
            label_sha = None
            if label_name in members:
                label = dataset / "labels/train2017" / source.with_suffix(".txt").name
                label.write_bytes(archive.read(label_name))
                label_sha = sha256(label)
            records.append({"image_id": image_id, "file_name": source.name, "source_image": str(source),
                            "image_sha256": digest, "label_sha256": label_sha})
    export_ids = fit_ids[:8] if scope == "smoke" else fit_ids
    list_path = dataset / "val2017.txt"
    list_path.write_text("".join(str((dataset / "images/train2017" / f"{i:012d}.jpg").resolve()) + "\n" for i in export_ids), encoding="utf8")
    template = yaml.safe_load((ROOT / "shared/vendor/ultralytics_8_4_100/ultralytics/cfg/datasets/coco.yaml").read_text(encoding="utf8"))
    yaml_path = dataset / "coco.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(dataset.resolve()), "train": str(list_path.resolve()),
                        "val": str(list_path.resolve()), "names": template["names"]}, sort_keys=False), encoding="utf8")
    write_new(run.path / "image_ids.json", export_ids)
    write_new(run.path / "IMAGE_MANIFEST.json", records)
    write_new(dataset / "NATIVE_ENTRY_ALIAS.json", {"actual_dataset_split": "COCO_train2017",
              "native_entry_mode": "val", "list_alias": str(list_path),
              "annotations_alias": "annotations/instances_val2017.json",
              "reason": "pinned native COCO category mapping/serializer filename recognition",
              "source": str(FULL_GT), "source_sha256": full_gt_sha,
              "subset_sha256": sha256(dataset / "training_annotations.json"),
              "never_interpret_as": "COCO_val2017_baseline_or_blind_test"})
    return membership, yaml_path, dataset / "training_annotations.json", records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    parent = STUDY / "runs" / S1_ID
    require(read_json(parent / "DECISION.json")["status"] == "passed", "S1 opportunity gate not passed")
    kind = "S2_CACHE_SMOKE" if args.scope == "smoke" else "S2_CACHE_FULL"
    with RegisteredRun(kind, "S2", parent, CODE_FILES, 900 if args.scope == "smoke" else 3600) as run:
        profile, environment = configure_native()
        split, yaml_path, gt_path, records = prepare(run, args.scope, profile)
        contract = {"stage": "S2_train_cache", "scope": args.scope, "actual_dataset_split": "COCO_train2017",
                    "images": 8 if args.scope == "smoke" else 2000,
                    "source_baseline": BASE_ID, "opportunity_run": S1_ID,
                    "split_sha256": sha256(run.path / "SPLIT_CONTRACT.json"),
                    "image_manifest_sha256": sha256(run.path / "IMAGE_MANIFEST.json"),
                    "gt_subset_sha256": sha256(gt_path), "profile_sha256": sha256(ROOT / "paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json"),
                    "base_weights_frozen": True, "max_cache_bytes": 16 * 1024**3,
                    "native_alias_disclosed": "dataset/NATIVE_ENTRY_ALIAS.json", "environment": environment}
        run.freeze(contract)
        if args.scope == "full":
            require(args.smoke_run is not None and read_json(args.smoke_run / "DECISION.json")["status"] == "passed", "cache smoke not passed")
            require(read_json(args.smoke_run / "SPLIT_CONTRACT.json") == split, "input split changed after cache smoke")
            require(read_json(args.smoke_run / "SOURCE_MANIFEST.json")["files"] == run.sources, "cache source changed after smoke")
        from native_output_bridge import make_trace_validator
        from ultralytics import YOLO
        import torch
        trace = run.path / "trace"
        (trace / "cache").mkdir(parents=True)
        requested = {**profile["native_args"], "data": str(yaml_path.resolve())}
        write_new(run.path / "REQUESTED_ARGS.json", requested)
        write_new(run.path / "ENVIRONMENT.json", environment)
        ids = read_json(run.path / "image_ids.json")
        observer = make_trace_validator(trace, ids, "COCO_train2017_cache_" + args.scope,
                  contract["max_cache_bytes"], sha256(gt_path), dict(requested), sha256(run.path / "image_ids.json"), profile["runtime_policy"])
        model = YOLO(str(WEIGHTS))
        run.update(phase="native_frozen_train_feature_export", input_manifest_sha256=contract["image_manifest_sha256"])
        with torch.inference_mode():
            model.val(validator=observer, **requested, project=str(trace.resolve()), name="native", exist_ok=False)
        manifest = read_json(trace / "TRACE_MANIFEST.json")
        require(manifest["capture_zero_delta_rle_equal"] and manifest["base_state_unchanged"], "train cache identity failed")
        require(sha256(WEIGHTS) == profile["model_sha256"], "base weight file changed")
        for record in records:
            require(sha256(Path(record["source_image"])) == record["image_sha256"], "source image changed")
        run.check_budget()
        write_new(run.path / "CONFIG_AUDIT.json", {
            "status": "passed", "scope": "frozen native training input export",
            "checks": {"pinned_packages_and_sources": True, "effective_native_args_and_numerics": True,
                       "native_zero_delta_all_exported_rows": True, "base_state_unchanged": True,
                       "source_images_unchanged": True, "weight_sha256_unchanged": True,
                       "fit_dev_disjoint": True},
            "evidence": {"native_manifest": sha256(trace / "TRACE_MANIFEST.json"),
                         "requested_args": sha256(run.path / "REQUESTED_ARGS.json"),
                         "split": sha256(run.path / "SPLIT_CONTRACT.json")},
            "intentional_differences_from_S0": ["COCO train2017 selected images and original annotations", "dataset/output paths"],
            "training_optimizer_started": False})
        run.finish({"actual_dataset_split": "COCO_train2017", "fit_images": 2000,
                    "exported_images": len(ids), "cache_bytes": manifest["cache_bytes"],
                    "trace_manifest_sha256": sha256(trace / "TRACE_MANIFEST.json"),
                    "source_images_and_weight_unchanged": True, "zero_delta": "exact_native_RLE",
                    "dev_images": len(split["dev_ids"]), "dev_T": split["dev_T"]},
                   scientific="input_cache_verified_not_method_effect", next_action="S2 adapter engineering smoke then B/C training")


if __name__ == "__main__":
    main()
