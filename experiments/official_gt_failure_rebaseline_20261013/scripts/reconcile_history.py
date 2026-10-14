"""S0 G4: compare two historical GT definitions using saved files only, without YOLO."""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/official_gt_failure_rebaseline_20261013"
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, require, sha256, write_new


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    args = parser.parse_args()
    summary = read_json(args.parent / "SUMMARY.json")
    require(summary["native_gt_ledger_validity"] == "passed", "G3 must pass before G4")
    run = STUDY / "runs" / ("RUN_HISTORY_RECONCILIATION_" + uuid.uuid4().hex)
    lock = STUDY / ".S0.lock"
    write_new(lock, {"run_id": run.name, "stage_id": "S0", "gate_id": "G4",
                     "host": socket.gethostname(), "pid": os.getpid(), "started_at": now()})
    run.mkdir(parents=True, exist_ok=False)
    meta = {"run_id": run.name, "study_id": read_json(STUDY / "study.json")["study_id"],
            "parent_run_id": args.parent.name, "stage_id": "S0", "gate_id": "G4",
            "host": socket.gethostname(), "interpreter": sys.executable,
            "command": sys.argv, "started_at": now(), "execution_status": "running",
            "scientific_status": "unadjudicated", "artifact_status": "partial",
            "transfer_status": "local_only", "scope": "saved_predictions_only_no_forward"}
    meta["code_commit"] = read_json(args.parent / "run.json")["code_commit"]
    meta["protocol_path"] = str(STUDY / "PROTOCOL.md")
    write_new(run / "START.json", meta)
    shutil.copyfile(__file__, run / "reconcile_history.py")
    shutil.copyfile(STUDY / "PROTOCOL.md", run / "PROTOCOL.md")
    print("HISTORY_RUN", run, flush=True)
    try:
        import csv
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import importlib.metadata
        import zlib

        official_index = ROOT / "assets/dependencies/baseline_20261013/COCO_OFFICIAL_ZIP_INDEX.json"
        require(sha256(official_index) == "b61befccd9184e6192ae118b5fb796768397607b0e2e2394ca8e4854b6d30bb7", "ZIP index receipt changed")
        index = read_json(official_index)
        trace = read_json(args.parent / "trace/TRACE_MANIFEST.json")
        native_image_hashes = {r["image_id"]: r["image_sha256"] for r in trace["images"]}
        require({int(Path(r["file_name"]).stem) for r in index["images"]} == set(native_image_hashes), "official ZIP image set differs")
        mismatches = []
        for entry in index["images"]:
            image = ROOT / "assets/datasets/coco_native_s0_20261013/images/val2017" / entry["file_name"]
            content = image.read_bytes()
            require(hashlib.sha256(content).hexdigest() == native_image_hashes[int(image.stem)], "image changed since forward")
            if len(content) != entry["size_bytes"] or f"{zlib.crc32(content):08x}" != entry["crc32"]:
                mismatches.append(entry["file_name"])
        write_new(run / "OFFICIAL_IMAGE_ARCHIVE_CHECK.json", {
            "source_url": index["url"], "index_sha256": sha256(official_index), "images": len(index["images"]),
            "post_forward_sha_match": True, "size_crc32_mismatch_count": len(mismatches), "mismatch_files": mismatches,
            "status": "passed" if not mismatches else "failed",
            "scope": "All image sizes and CRC32 match the official archive central directory; CRC32 is not an official per-image SHA certificate."})
        require(not mismatches, "local images differ from official ZIP entries")

        config_sources = read_json(args.parent / "trace/CONFIG_SOURCES.json")
        effective_args = read_json(args.parent / "trace/NATIVE_ARGS.json")
        layers = {**{k: config_sources[k] for k in ("framework_default", "checkpoint_train_args", "model_overrides", "requested_native_args")},
                  "effective_native_args": effective_args}
        fields = sorted(set().union(*(set(v) for v in layers.values())))
        write_new(run / "CONFIG_PARAMETER_AUDIT.json", {
            "source_run_id": args.parent.name, "checkpoint_version": config_sources["checkpoint_version"],
            "profile_sha256": config_sources["profile_sha256"], "parameter_counts": {k: len(v) for k, v in layers.items()},
            "source_sha256": {f: sha256(args.parent / "trace" / f) for f in ("CONFIG_SOURCES.json", "NATIVE_ARGS.json", "ENVIRONMENT.json", "MODEL_RECEIPT.json")},
            "parameters": [{"name": name, "layers": {layer: {"present": name in values, "value": values.get(name)}
                            for layer, values in layers.items()}} for name in fields],
            "runtime": read_json(args.parent / "trace/ENVIRONMENT.json"),
            "model": read_json(args.parent / "trace/MODEL_RECEIPT.json"),
            "interpretation": "Values present in val args include inactive train/predict/export options. Stage-specific effects are documented in BASELINE_CONFIGURATION.md; presence alone does not establish activity."})

        annotations = ROOT / "assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json"
        require(sha256(annotations) == "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f", "wrong GT")
        new_path = args.parent / "ledger/PER_GT_FAILURE.jsonl"
        new, populations = {}, Counter()
        matched_in_both = different_detection_in_both = ignored = 0
        for line in new_path.read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            populations[row["population"]] += 1
            ignored += int(row["is_ignored"])
            if not row["is_ignored"]:
                key = (row["image_id"], row["annotation_id"], row["gt_category_id"])
                require(key not in new, "duplicate new GT")
                new[key] = row["task_mask75"] == "success"
                if row["task_box75"] == "success" and row["task_mask75"] == "success":
                    matched_in_both += 1
                    different_detection_in_both += int(row["bbox75_match_output_id"] != row["segm75_match_output_id"])
        require(len(new) == 36335, "wrong new denominator")
        native = read_json(args.parent / "trace/native_eval_summary.json")
        differences = {task: {metric: abs(value - native[task][metric])
                       for metric, value in summary["metrics"][task].items() if value is not None}
                       for task in ("bbox", "segm")}
        write_new(run / "NATIVE_LEDGER_DIAGNOSTICS.json", {
            "source_run_id": args.parent.name, "source_ledger_sha256": sha256(new_path),
            "ordinary_gt": len(new), "ignored_gt": ignored, "populations": dict(populations),
            "success_in_both_bbox75_and_segm75": matched_in_both,
            "different_detection_ids_among_both_success": different_detection_in_both,
            "backend_abs_differences_ap_points": differences,
            "backend_max_abs_difference_ap_points": max(v for d in differences.values() for v in d.values()),
            "S1_finite_oracle_status": "not_started; population T is only the baseline-defined candidate set"})
        inputs = {str(new_path): sha256(new_path), str(annotations): sha256(annotations), str(official_index): sha256(official_index)}

        # Source 1 already contains actual COCO matching, unlike fixed-slot geometry labels.
        raw_path = args.legacy / "raw_gt_joined.jsonl"
        raw = {}
        for line in raw_path.read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            key = (row["image_id"], row["annotation_id"], row["category_id"])
            require(key not in raw and isinstance(row["segm_coco75"], bool), "invalid legacy raw row")
            raw[key] = row["segm_coco75"]
        inputs[str(raw_path)] = sha256(raw_path)
        require(set(raw) == set(new) and sum(raw.values()) == 22767, "legacy raw identity/count mismatch")
        old_raw_summary = read_json(args.legacy / "raw_summary.json")
        old_alignment_summary = read_json(args.legacy / "alignment_summary.json")
        for name in ("raw_summary.json", "alignment_summary.json", "alignment_original_analyze_cache.py"):
            path = args.legacy / name
            inputs[str(path)] = sha256(path)
        require(old_raw_summary["coco"]["segm"]["R75_TP_count"] == sum(raw.values()), "raw summary count mismatch")

        # Source 2's original official_gt_mask75.json was not present locally. Reconstruct
        # ONLY that matching table from both unchanged prediction partitions in original order.
        # gt_inventory.csv uses Box50 slots and cannot substitute for COCO GT matching.
        coco = COCO(str(annotations))
        predictions, observed_images = [], set()
        expected = [
            ("alignment_500_predictions.json", "0287d7b8e7da83b3b0139bb64ed5f0f85a196e1548691bce095e8e28ee28b88a"),
            ("alignment_4500_predictions.json", "dc61664daf96d9e543945eee6eb4ad1cc02437987c8aa9b301995438ddc61d7e"),
        ]
        for filename, digest in expected:
            path = args.legacy / filename
            require(sha256(path) == digest, "historical prediction changed: " + filename)
            inputs[str(path)] = digest
            rows = read_json(path)
            require(all("bbox" not in r for r in rows), "historical area policy changed")
            part_images = {r["image_id"] for r in rows}
            require(not observed_images.intersection(part_images), "historical prediction partitions overlap")
            observed_images.update(part_images)
            predictions.extend(rows)
        # Original analyze_cache.py explicitly evaluated all 5000 COCO IDs. Its
        # image_ids.json copies are absent locally, so use that documented full set
        # and verify the old published exact AP/count before accepting recovery.
        image_ids = sorted(coco.imgs)
        require(len(image_ids) == 5000 and observed_images <= set(image_ids), "legacy COCO set mismatch")
        require(len(predictions) == 561859, "legacy prediction count mismatch")
        with (run / "legacy_cocoeval.log").open("x", encoding="utf8") as log, contextlib.redirect_stdout(log):
            ev = COCOeval(coco, coco.loadRes(predictions), "segm")
            ev.params.imgIds = sorted(image_ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        require(abs(float(ev.stats[0]) - 0.43515306090223993) < 1e-10, "historical AP not recovered")
        require(abs(float(ev.stats[0]) - old_alignment_summary["segm_metrics"]["AP"]) < 1e-10, "historical summary differs")
        t75 = next(i for i, t in enumerate(ev.params.iouThrs) if abs(t - .75) < 1e-12)
        alignment = {}
        for item in ev.evalImgs:
            if item is None or list(item["aRng"]) != list(ev.params.areaRng[0]):
                continue
            for j, aid in enumerate(item["gtIds"]):
                if item["gtIgnore"][j]:
                    continue
                ann = coco.anns[int(aid)]
                key = (int(item["image_id"]), int(aid), ann["category_id"])
                require(key not in alignment, "duplicate recovered GT")
                alignment[key] = bool(item["gtMatches"][t75][j])
        require(set(alignment) == set(new) and sum(alignment.values()) == 22773, "legacy alignment mismatch")
        with (run / "RECOVERED_ALIGNMENT_GT.jsonl").open("x", encoding="utf8") as f:
            for (image, ann, cat), success in sorted(alignment.items()):
                f.write(json.dumps(dict(image_id=image, annotation_id=ann, category_id=cat, segm_coco75=success)) + "\n")

        results = {}
        for label, old, old_ap, source_run in [
            ("raw", raw, 100 * old_raw_summary["coco"]["segm"]["AP"], "RUN_a7c840bb695d45b4b66bc346715ca1aa"),
            ("alignment", alignment, 100 * float(ev.stats[0]), "RUN_1bea3cce89f441ec885b737feaa26989"),
        ]:
            counts = Counter()
            with (run / (label + "_OLD_NEW_GT_DELTA.csv")).open("x", encoding="utf8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_id", "annotation_id", "category_id", "old_mask75", "new_mask75"])
                for key in sorted(new):
                    before, after = old[key], new[key]
                    counts[("success" if before else "failure") + "_to_" + ("success" if after else "failure")] += 1
                    if before != after:
                        writer.writerow([*key, "success" if before else "failure", "success" if after else "failure"])
            results[label] = {"source_run_id": source_run, "ordinary_gt": len(new),
                              "old_success": sum(old.values()), "new_success": sum(new.values()),
                              "old_mask_ap_points": old_ap, "new_mask_ap_points": summary["metrics"]["segm"]["AP"],
                              "transitions": dict(counts), "gt_key_check": "passed"}
        results["raw_link_status"] = "not_comparable"
        results["raw_link_reason"] = "Historical batch=1 unfused/custom export and current fused rectangular batch=16 have different input geometry/decoding; numeric raw indices are not joined across forwards."
        results["alignment_recovery"] = {"status": "passed", "mode": "original saved mask-only predictions",
            "image_scope": "all 5000 GT image IDs, as asserted in original analyze_cache.py; old partition ID files unavailable",
            "original_area_policy": "mask_area", "new_area_policy": "bbox_area",
            "recovered_ap": float(ev.stats[0]), "recovered_mask75_success": sum(alignment.values()),
            "pycocotools": importlib.metadata.version("pycocotools"), "numpy": np.__version__}
        write_new(run / "INPUTS_SHA256.json", inputs)
        write_new(run / "SUMMARY.json", results)
        write_new(run / "DECISION.json", {"gate_id": "G4", "status": "passed", "claim_scope": "historical_per_GT_transition_audit",
            "conclusion": "Both historical GT tables are traceably reconciled; neither is substituted for the new native baseline.",
            "evidence_refs": {n: sha256(run / n) for n in ("SUMMARY.json", "INPUTS_SHA256.json", "NATIVE_LEDGER_DIAGNOSTICS.json")},
            "next_action": "close S0 report; no new inference or training"})
        meta.update(execution_status="completed", exit_code=0, scientific_status="historical_reconciliation_passed", artifact_status="complete")
    except BaseException as exc:
        meta.update(execution_status="failed", exit_code=1, error=str(exc))
        (run / "failure.log").write_text(traceback.format_exc(), encoding="utf8")
        print(traceback.format_exc(), flush=True)
    finally:
        meta["ended_at"] = now()
        meta["script_sha256"] = sha256(__file__)
        meta["protocol_sha256"] = sha256(run / "PROTOCOL.md")
        write_new(run / "run.json", meta)
        write_new(run / "ARTIFACT_MANIFEST.json", [{"relative_path": str(p.relative_to(run)), "sha256": sha256(p),
                  "size_bytes": p.stat().st_size} for p in run.iterdir() if p.is_file()])
        require(read_json(lock)["run_id"] == run.name, "G4 lock ownership changed")
        lock.unlink()
    print("FINAL_STATUS", meta["execution_status"], run.name, flush=True)
    return meta["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
