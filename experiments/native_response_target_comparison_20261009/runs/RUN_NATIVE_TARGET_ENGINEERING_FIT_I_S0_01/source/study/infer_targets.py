"""GT-free native inference of the two already frozen numeric HGB gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback

from target_common import ARMS, FROZEN_SOURCE, OFFICIAL_SHA, VAL_LIST_SHA, append, apply_gain, canonical_sha, dump, load_config, now, paths_from_list, peak_memory, runtime, sha, source_lock, verify_lock


def assemble(arm, ids):
    target = arm / "predictions.json"
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for iid in ids:
            rows = json.loads((arm / "images" / f"{iid:012d}.json").read_text())["detections"]
            for row in rows:
                if not first:
                    f.write(",\n")
                f.write(json.dumps({k: row[k] for k in ("image_id", "category_id", "bbox", "score", "segmentation")}, separators=(",", ":"), allow_nan=False))
                first = False
        f.write("\n]\n")
    temporary.replace(target)


def check_models(iroot, hroot):
    summaries = []
    for target, root in (("I", iroot), ("H", hroot)):
        complete = json.loads((root / "COMPLETE.json").read_text())
        s = json.loads((root / "SUMMARY.json").read_text())
        if not complete["passed"] or complete["summary_sha256"] != sha(root / "SUMMARY.json") or complete["model_sha256"] != sha(root / "model.json") or not s["passed"] or s["target"] != target:
            raise ValueError("Frozen estimator receipt incomplete")
        summaries.append(s)
    equality = ("extraction_run", "extraction_summary_sha256", "extraction_complete_sha256",
                "full_hgb_parameters", "sample_weight", "weight_bytes_sha256", "training_feature_bytes_sha256",
                "supervised_binding_sha256", "training_rows", "training_images", "engineering", "runtime")
    if any(summaries[0][k] != summaries[1][k] for k in equality):
        raise ValueError("Targets did not share the identical fit budget/rows/features/runtime")
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "protocol-sha256", "model-i", "model-h", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--engineering", action="store_true")
    args = p.parse_args()
    out = Path(args.output).resolve()
    if (out / "INPUTS.json").exists():
        raise FileExistsError("Inference retry requires a new Run")
    out.mkdir(parents=True, exist_ok=True)
    c = load_config(args.config, args.protocol_sha256)
    paths, ids = paths_from_list(c["val"]["images_list"], VAL_LIST_SHA, 5000)
    original_ids = ids
    if args.engineering:
        paths, ids = paths[:4], ids[:4]
    iroot, hroot = Path(args.model_i).resolve(), Path(args.model_h).resolve()
    models = check_models(iroot, hroot)
    if any(bool(m["engineering"]) != args.engineering for m in models):
        raise ValueError("Engineering/formal model scope differs")
    cache = Path(c["val"]["baseline_cache"]).resolve()
    prior_inputs = json.loads((cache / "EVALUATION_INPUTS.json").read_text())
    prior_summary = json.loads((cache / "SUMMARY.json").read_text())
    pc = prior_inputs["configuration"]
    if (not prior_summary["passed"] or prior_summary["image_count"] != 5000 or pc["source_image_ids"] != original_ids
            or pc["images_list_sha256"] != VAL_LIST_SHA or pc["weights_sha256"] != OFFICIAL_SHA
            or any(pc["source_sha256"][k] != v for k, v in FROZEN_SOURCE.items())):
        raise ValueError("Original native5k02 reference source/scope differs")
    parities = {v["image_id"]: v for v in (json.loads(line) for line in (cache / "BASELINE_PARITY_IMAGES.jsonl").read_text().splitlines())}
    if set(parities) != set(original_ids):
        raise ValueError("Original native5k input parity scope incomplete")
    input_paths = [c["val"]["images_list"], c["weights"], cache / "SUMMARY.json", cache / "EVALUATION_INPUTS.json", cache / "BASELINE_PARITY.json", cache / "BASELINE_PARITY_IMAGES.jsonl", cache / "NATIVE_DECISIONS.jsonl"]
    for root in (iroot, hroot):
        input_paths += [root / "SUMMARY.json", root / "COMPLETE.json", root / "model.json"]
    lock = source_lock(out, c, args.config, input_paths)
    began, extractor = time.perf_counter(), None
    try:
        import numpy as np
        import torch
        from pycocotools import mask as mask_utils
        from frozen_io import FrozenYOLO
        from native_mask_adapter import prepare_native, timed_native_forward, Timer
        from native_response import decode
        from portable_risk import PortableRisk
        extractor = FrozenYOLO(c["weights"], c["vendor"], "cuda")
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        env = runtime(True)
        vendor_files = dict(segment_validator="ultralytics/models/yolo/segment/val.py", ops="ultralytics/utils/ops.py",
                            nms="ultralytics/utils/nms.py", letterbox="ultralytics/data/augment.py")
        vendor_hashes = {k: sha(Path(c["vendor"]) / v) for k, v in vendor_files.items()}
        if vendor_hashes != pc["vendor_source_sha256"]:
            raise ValueError("Reference native vendor core differs")
        validator = SegmentationValidator(args=dict(conf=.001, max_det=300, save_json=True, plots=False,
                                                    imgsz=640, half=False, rect=False))
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        estimators = (PortableRisk(iroot / "model.json"), PortableRisk(hroot / "model.json"))
        inputs = dict(protocol_sha256=args.protocol_sha256, config_sha256=sha(args.config), runtime=env,
                      image_ids=ids, original_image_ids=original_ids, engineering=args.engineering,
                      model_runs=dict(target_I=str(iroot), target_H=str(hroot)),
                      model_sha256={"target_I": sha(iroot / "model.json"), "target_H": sha(hroot / "model.json")},
                      model_summary_sha256={"target_I": sha(iroot / "SUMMARY.json"), "target_H": sha(hroot / "SUMMARY.json")},
                      baseline_cache=str(cache), baseline_cache_summary_sha256=sha(cache / "SUMMARY.json"),
                      source_lock_sha256=sha(out / "SOURCE_LOCK.json"), gt_parsed=False, gt_used_in_inference=False,
                      arms=list(ARMS), gate="predicted_gain>0 AND first64/protoROI supported AND original trial nonempty")
        dump(out / "INPUTS.json", inputs)
        for arm in ARMS:
            (out / arm / "images").mkdir(parents=True, exist_ok=True)
        totals = dict(native_rows=0, selected_first64=0, supported_first64=0, unsupported_first64=0,
                      empty_baseline_masks=0, empty_trial_fallback=0, target_I_applied=0, target_H_applied=0,
                      different_decisions=0, different_binary_outputs=0)
        # Existing source evidence is read in original image order, not rematched.
        with (cache / "NATIVE_DECISIONS.jsonl").open(encoding="utf-8") as old_decisions, torch.inference_mode():
            for position, (path, iid) in enumerate(zip(paths, ids), 1):
                old = json.loads(next(old_decisions))
                if old["image_id"] != iid:
                    raise ValueError("Original native decision order differs")
                extracted, forward = timed_native_forward(extractor, path)
                with Timer("cuda") as timer:
                    prepared = prepare_native(extracted, validator, "cuda")
                    baseline, smooth, rows, binding = decode(extracted, prepared, iid, categories, mask_utils, "cuda")
                cachefile = cache / "baseline" / "images" / f"{iid:012d}.json"
                cached = json.loads(cachefile.read_text())
                prior = parities[iid]
                if cached["image_id"] != iid or cached["detections"] != baseline:
                    raise RuntimeError("All original native baseline rows/identity/losslessRLE differ: " + str(iid))
                for key in ("image_sha256", "input_sha256", "native_identity_sha256", "coefficients_sha256"):
                    if binding[key] != prior[key]:
                        raise RuntimeError("Original native input/row binding differs: " + key)
                for row in rows:
                    od = old["decisions"][row["detection_index"]]
                    if (row["features"].tolist() != od["response_features"] or row["smooth_tau"] != od["smooth_tau"]
                            or row["trial_nonempty"] != od["trial_nonempty"][0] or not od["first64_proto_supported"]):
                        raise RuntimeError("Canonical shared response feature/action bytes differ from native5k02")
                features = np.asarray([r["features"] for r in rows], dtype=np.float64).reshape(-1, 5)
                gains = [e.predict(features) for e in estimators] if len(rows) else [np.empty(0), np.empty(0)]
                if any(not np.isfinite(v).all() for v in gains):
                    raise ValueError("Nonfinite frozen gate prediction")
                outputs = {arm: list(baseline) for arm in ARMS}
                lookup = {r["detection_index"]: (j, r) for j, r in enumerate(rows)}
                decisions = []
                image_counts = {key: 0 for key in totals}
                image_counts.update(native_rows=len(baseline), selected_first64=min(64, len(baseline)),
                                    supported_first64=len(rows), unsupported_first64=min(64, len(baseline))-len(rows))
                for index, b in enumerate(baseline):
                    native_row = prepared.eligible_output_rows[index]
                    decision = dict(detection_index=index, native_output_row=native_row,
                                    raw_index=int(extracted["raw_indices"][native_row]), in_first64=index < 64,
                                    proto_roi_supported=bool(prepared.proto_supported[index]),
                                    action_supported=index in lookup, features=None, feature_sha256=None, smooth_tau=None,
                                    trial_nonempty=None, fallback=None, target_I_gain=None, target_H_gain=None,
                                    target_I_applied=False, target_H_applied=False)
                    image_counts["empty_baseline_masks"] += int(not bool(prepared.baseline_original[index].any()))
                    if index in lookup:
                        j, row = lookup[index]
                        decision.update(features=row["features"].tolist(), feature_sha256=row["feature_sha256"],
                                        smooth_tau=row["smooth_tau"], trial_nonempty=row["trial_nonempty"], fallback=row["fallback"],
                                        target_I_gain=float(gains[0][j]), target_H_gain=float(gains[1][j]))
                        image_counts["empty_trial_fallback"] += int(row["fallback"])
                        for k, arm in enumerate(ARMS[1:]):
                            use = apply_gain(gains[k][j], row["trial_nonempty"])
                            decision[arm + "_applied"] = use
                            if use:
                                outputs[arm][index] = smooth[index]
                                image_counts[arm + "_applied"] += 1
                    image_counts["different_decisions"] += int(decision["target_I_applied"] != decision["target_H_applied"])
                    image_counts["different_binary_outputs"] += int(outputs["target_I"][index]["segmentation"] != outputs["target_H"][index]["segmentation"])
                    decisions.append(decision)
                identity_keys = ("image_id", "category_id", "bbox", "score", "detection_index", "raw_input_box_xyxy",
                                 "raw_confidence", "box_xyxy", "model_class")
                for arm in ARMS:
                    if len(outputs[arm]) != len(baseline) or any(any(a[k] != b[k] for k in identity_keys)
                                                              for a, b in zip(baseline, outputs[arm])):
                        raise RuntimeError("Gate changed native all-normal candidate identity")
                    for decision, b, a in zip(decisions, baseline, outputs[arm]):
                        if not decision["action_supported"] or decision["fallback"]:
                            if a != b:
                                raise RuntimeError("Out-of-scope/empty fallback changed baseline")
                for key in totals:
                    totals[key] += image_counts[key]
                for arm in ARMS:
                    dump(out / arm / "images" / f"{iid:012d}.json", dict(image_id=iid, image_path=str(path),
                         original_shape=extracted["original_shape"], letterbox=extracted["letterbox"],
                         source_baseline_identity_sha256=binding["native_identity_sha256"], detections=outputs[arm]))
                parity = dict(image_id=iid, position=position, **binding, source_image_cache_sha256=sha(cachefile),
                              all_reference_baseline_rows_identity_rle_exact=True, canonical_feature_action_exact=True,
                              all_arm_native_identity_exact=True, counts=image_counts, gt_used=False)
                append(out / "BASELINE_PARITY_IMAGES.jsonl", parity)
                append(out / "NATIVE_DECISIONS.jsonl", dict(image_id=iid, native_identity_sha256=binding["native_identity_sha256"],
                                                           decisions=decisions, gt_used=False))
                append(out / "IMAGE_TIMINGS.jsonl", dict(image_id=iid, forward=forward, shared_native_decode_seconds=timer.seconds,
                                                         shared_work_not_isolated_arm_cost=True))
                if sha(path) != binding["image_sha256"] or sha(cachefile) != parity["source_image_cache_sha256"]:
                    raise ValueError("Image/reference changed during image processing")
                if position % 20 == 0 or args.engineering:
                    dump(out / "PROGRESS.json", dict(images=position, total_images=len(ids), totals=totals, elapsed_seconds=time.perf_counter()-began))
                    print(f"TARGET_INFER {position}/{len(ids)} rows={totals['native_rows']} decisions_different={totals['different_decisions']} elapsed_s={time.perf_counter()-began:.1f}", flush=True)
                del extracted, prepared, baseline, smooth, rows, outputs, cached
        frozen = extractor.verify_frozen()
        verify_lock(lock)
        dump(out / "BASELINE_PARITY.json", dict(passed=True, images=len(ids), totals=totals,
                                              all_zero_native_exact=True, all_native5k02_input_identity_rle_exact=True,
                                              all_canonical_features_actions_exact=True, all_arm_identity_exact=True,
                                              gt_used=False, frozen_integrity=frozen,
                                              image_receipts_sha256=sha(out / "BASELINE_PARITY_IMAGES.jsonl")))
        receipts = {}
        for arm in ARMS:
            assemble(out / arm, ids)
            r = dict(status="prediction_complete", image_count=len(ids), engineering=args.engineering,
                     image_ids=ids,
                     frozen_state_sha256=frozen["final_state_sha256"], frozen_integrity=frozen,
                     predictions_sha256=sha(out / arm / "predictions.json"), input_sha256=sha(out / "INPUTS.json"),
                     baseline_parity_sha256=sha(out / "BASELINE_PARITY.json"), empty_masks_preserved=True,
                     gt_used=False, model_sha256=inputs["model_sha256"].get(arm), model_arm=arm)
            dump(out / arm / "COMPLETE.json", r)
            receipts[arm] = dict(complete_sha256=sha(out / arm / "COMPLETE.json"), predictions_sha256=r["predictions_sha256"])
        summary = dict(passed=True, status="prediction_complete", engineering=args.engineering, image_count=len(ids),
                       image_ids=ids,
                       arm_names=list(ARMS), totals=totals, frozen_integrity=frozen, source_unchanged=True,
                       baseline_parity_sha256=sha(out / "BASELINE_PARITY.json"), source_lock_sha256=sha(out / "SOURCE_LOCK.json"),
                       inputs_sha256=sha(out / "INPUTS.json"), prediction_receipts=receipts, gt_parsed=False, gt_used_in_inference=False,
                       elapsed_seconds=time.perf_counter()-began, peak_memory=peak_memory(), completed_at=now(),
                       metrics=None, metrics_status="unknown; independent scoring required",
                       isolated_method_cost=None, cost_status="shared production timing is not isolated arm cost")
        dump(out / "SUMMARY.json", summary)
        dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"),
                                        baseline_parity_sha256=sha(out / "BASELINE_PARITY.json"), prediction_receipts=receipts))
        print(f"TARGET_INFERENCE_COMPLETE {out} images={len(ids)}", flush=True)
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_at=now()))
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
