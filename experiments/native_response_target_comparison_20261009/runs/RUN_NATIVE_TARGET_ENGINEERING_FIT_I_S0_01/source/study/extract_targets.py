"""One shared20k native extraction; original GT is used only for labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from target_common import (TRAIN_IDS_SHA, TRAIN_ANN_SHA, TRAIN_LIST_SHA, append, canonical_sha,
                           dump, integer_targets, load_config, now, paths_from_list,
                           peak_memory, runtime, sha, source_lock, synthetic_contract, verify_lock)


def part(out, records, number):
    import numpy as np
    path = out / "parts" / f"part_{number:04d}.npz"
    fields = ("image_id", "detection_index", "native_output_row", "raw_index", "annotation_id",
              "known", "baseline_tp", "baseline_union", "trial_tp", "trial_union", "gt_pixels",
              "baseline_area", "trial_area", "fallback", "threshold_ticks", "y_I", "y_H", "smooth_tau")
    arrays = {key: np.asarray([r[key] for r in records], dtype=np.float64 if key in ("y_I", "y_H", "smooth_tau") else np.int64)
              for key in fields}
    arrays["features"] = np.asarray([r["features"] for r in records], dtype=np.float64).reshape(-1, 5)
    np.savez_compressed(path, **arrays)
    return dict(path=str(path.relative_to(out)), sha256=sha(path), rows=len(records),
                supervised_rows=int(arrays["known"].sum()),
                features_bytes_sha256=hashlib.sha256(arrays["features"].tobytes()).hexdigest(),
                first_image_id=records[0]["image_id"] if records else None,
                last_image_id=records[-1]["image_id"] if records else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--engineering", action="store_true")
    args = p.parse_args()
    out = Path(args.output).resolve()
    if (out / "INPUTS.json").exists() or (out / "SUMMARY.json").exists():
        raise FileExistsError("Extraction retry requires a new Run")
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)
    c = load_config(args.config, args.protocol_sha256)
    t = c["train"]
    paths, ids = paths_from_list(t["images_list"], TRAIN_LIST_SHA, 20000)
    if sha(t["selected_ids"]) != TRAIN_IDS_SHA or json.loads(Path(t["selected_ids"]).read_text()) != ids:
        raise ValueError("Original20k ID selection/order differs")
    if sha(t["annotations"]) != TRAIN_ANN_SHA:
        raise ValueError("Original training annotation bytes differ")
    original_ids = ids
    if args.engineering:
        paths, ids = paths[:4], ids[:4]
    lock = source_lock(out, c, args.config, [t["images_list"], t["selected_ids"], t["annotations"], t["images_manifest"], c["weights"]])
    began, extractor = time.perf_counter(), None
    try:
        import numpy as np
        import torch
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_utils
        from frozen_io import FrozenYOLO
        from native_mask_adapter import prepare_native, timed_native_forward, Timer
        from native_response import decode
        extractor = FrozenYOLO(c["weights"], c["vendor"], "cuda")
        from ultralytics.data import converter
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        env = runtime(True)
        if sha(t["images_manifest"]) != "c1c9f6b7f4ad07e7c7e10e6fccd41243da346aa19c4d2497eff4262f0ef85546":
            raise ValueError("Original materialized JPEG manifest differs")
        # SHA records are source JPEG identities, independent of GT selection.
        expected_images = {}
        for line in Path(t["images_manifest"]).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                expected_images[int(Path(name.strip().lstrip("*")).stem)] = digest
        if set(expected_images) != set(original_ids):
            raise ValueError("Original JPEG manifest scope differs")
        coco = COCO(t["annotations"])
        validator = SegmentationValidator(args=dict(conf=.001, max_det=300, save_json=True,
                                                    plots=False, imgsz=640, half=False, rect=False))
        validator.nc, validator.end2end, validator.process = 80, True, ops.process_mask_native
        categories = converter.coco80_to_coco91_class()
        inputs = dict(config_sha256=sha(args.config), protocol_sha256=args.protocol_sha256,
                      source_lock_sha256=sha(out / "SOURCE_LOCK.json"), runtime=env, engineering=args.engineering,
                      original_image_ids=original_ids, image_ids=ids, gt_used="training labels only; never feature/action generation",
                      row_scope="first64 post-conf positions AND protoROI supported; matched valid sameclass ordinaryGT BoxIoU>=.5",
                      unknown_targets="NaN NPZ markers; excluded from both fits", sample_weight="unit per supervised candidate")
        dump(out / "INPUTS.json", inputs)
        totals = dict(native_rows=0, selected_first64=0, supported_rows=0, supervised_rows=0,
                      unknown_no_sameclass_gt=0, unknown_box_iou_below05=0, invalid_gt=0,
                      newly_empty_fallback_rows=0, supervised_empty_fallback_rows=0, repeated_gt_matches=0)
        pending, parts, unique_gt = [], [], set()
        feature_digest, supervised_digest = hashlib.sha256(), hashlib.sha256()
        with torch.inference_mode():
            for position, (path, iid) in enumerate(zip(paths, ids), 1):
                extracted, forward_time = timed_native_forward(extractor, path)
                if extracted["image_sha256"] != expected_images[iid]:
                    raise ValueError("Materialized source JPEG bytes differ: " + str(iid))
                with Timer("cuda") as timer:
                    prepared = prepare_native(extracted, validator, "cuda")
                    baseline, smooth, rows, binding = decode(extracted, prepared, iid, categories, mask_utils, "cuda")
                by_category, masks = {}, {}
                for ann in coco.imgToAnns.get(iid, []):
                    if not ann.get("iscrowd", 0):
                        by_category.setdefault(ann["category_id"], []).append(ann)
                image_counts = {key: 0 for key in totals}
                image_counts.update(native_rows=len(baseline), selected_first64=min(64, len(baseline)), supported_rows=len(rows))
                matched_ids = []
                for r in rows:
                    index = r["detection_index"]
                    d = baseline[index]
                    candidates = by_category.get(d["category_id"], [])
                    known, reason, gid, gt = False, None, -1, None
                    box_iou = None
                    if not candidates:
                        reason = "unknown_no_sameclass_gt"
                    else:
                        # Exact final8 float64 BoxIoU and original GT order tie.
                        box = np.asarray(d["box_xyxy"], dtype=np.float64)
                        boxes = np.asarray([[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
                                            for a in candidates], dtype=np.float64)
                        inter = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])-np.maximum(box[:2], boxes[:, :2])).prod(1)
                        union = np.maximum(0, box[2:]-box[:2]).prod() + np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(1)-inter
                        overlaps = inter / np.maximum(union, 1e-12)
                        j = int(np.argmax(overlaps))
                        box_iou = float(overlaps[j])
                        if box_iou < .5:
                            reason = "unknown_box_iou_below05"
                        else:
                            ann = candidates[j]
                            gid = int(ann["id"])
                            if gid not in masks:
                                masks[gid] = coco.annToMask(ann).astype(bool)
                            gt = masks[gid]
                            known = tuple(gt.shape) == tuple(extracted["original_shape"]) and bool(gt.any())
                            if not known:
                                reason = "invalid_gt"
                    rec = dict(image_id=iid, detection_index=index, native_output_row=r["native_output_row"], raw_index=r["raw_index"],
                               annotation_id=gid, known=int(known), baseline_tp=-1, baseline_union=-1, trial_tp=-1, trial_union=-1,
                               gt_pixels=-1, baseline_area=r["baseline_area"], trial_area=r["trial_area"], fallback=int(r["fallback"]),
                               threshold_ticks=-999, y_I=np.nan, y_H=np.nan, smooth_tau=r["smooth_tau"], features=r["features"])
                    if known:
                        base = prepared.baseline_original[index].numpy().astype(bool)
                        trial = r["trial_mask"].astype(bool)
                        btp = int(np.logical_and(base, gt).sum(dtype=np.int64))
                        ttp = int(np.logical_and(trial, gt).sum(dtype=np.int64))
                        gp = int(gt.sum(dtype=np.int64))
                        bu, tu = r["baseline_area"]+gp-btp, r["trial_area"]+gp-ttp
                        yi, yh, ticks = integer_targets(btp, bu, ttp, tu, r["fallback"])
                        if not (-1 <= yi <= 1 and -1 <= yh <= 1) or yi*yh < 0:
                            raise AssertionError("Signed target ledger failed")
                        rec.update(baseline_tp=btp, baseline_union=bu, trial_tp=ttp, trial_union=tu, gt_pixels=gp,
                                   y_I=yi, y_H=yh, threshold_ticks=ticks)
                        image_counts["supervised_rows"] += 1
                        image_counts["supervised_empty_fallback_rows"] += int(r["fallback"])
                        key = (iid, gid)
                        unique_gt.add(key)
                        matched_ids.append(gid)
                        supervised_digest.update(np.asarray([iid, index, r["native_output_row"], r["raw_index"], gid], dtype=np.int64).tobytes())
                    else:
                        image_counts[reason] += 1
                    image_counts["newly_empty_fallback_rows"] += int(r["fallback"])
                    feature_digest.update(r["features"].tobytes())
                    pending.append(rec)
                image_counts["repeated_gt_matches"] = len(matched_ids)-len(set(matched_ids))
                for key in totals:
                    totals[key] += image_counts[key]
                append(out / "IMAGES.jsonl", dict(image_id=iid, position=position, **binding,
                                                   counts=image_counts, forward=forward_time, native_decode_seconds=timer.seconds,
                                                   gt_association="original imgToAnns order; np.argmax first tie; duplicates allowed"))
                if sha(path) != extracted["image_sha256"]:
                    raise ValueError("JPEG changed during extraction: " + str(iid))
                if position % 100 == 0 or position == len(ids):
                    parts.append(part(out, pending, len(parts)))
                    pending = []
                    dump(out / "PROGRESS.json", dict(images=position, total_images=len(ids), totals=totals, elapsed_seconds=time.perf_counter()-began))
                if position % 20 == 0 or args.engineering:
                    print(f"TARGET_EXTRACT {position}/{len(ids)} supported={totals['supported_rows']} supervised={totals['supervised_rows']} elapsed_s={time.perf_counter()-began:.1f}", flush=True)
                del extracted, prepared, baseline, smooth, rows, masks
        frozen = extractor.verify_frozen()
        verify_lock(lock)
        dump(out / "ROWS_PARTS.json", dict(parts=parts, rows=totals["supported_rows"], supervised_rows=totals["supervised_rows"],
                                          feature_bytes_sha256=feature_digest.hexdigest(), supervised_binding_sha256=supervised_digest.hexdigest(),
                                          array_schema="features float64[N,5]; identities/counts int64; unknown targets NaN and known=0"))
        summary = dict(passed=True, status="extraction_complete", engineering=args.engineering, images=len(ids), totals=totals,
                       unique_supervised_gt=len(unique_gt), frozen_integrity=frozen, source_unchanged=True,
                       feature_bytes_sha256=feature_digest.hexdigest(), supervised_binding_sha256=supervised_digest.hexdigest(),
                       rows_parts_sha256=sha(out / "ROWS_PARTS.json"), source_lock_sha256=sha(out / "SOURCE_LOCK.json"),
                       inputs_sha256=sha(out / "INPUTS.json"), synthetic_contract=synthetic_contract(),
                       peak_memory=peak_memory(), elapsed_seconds=time.perf_counter()-began, completed_at=now(),
                       limitation="Normal native action-support candidates only, not complete raw; duplicates retained; unknown excluded")
        dump(out / "SUMMARY.json", summary)
        dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), rows_parts_sha256=sha(out / "ROWS_PARTS.json")))
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_at=now()))
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
