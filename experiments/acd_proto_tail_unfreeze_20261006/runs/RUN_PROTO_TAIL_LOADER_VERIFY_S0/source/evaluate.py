"""Offline native mask-head/prototype-tail evaluation (Ultralytics 8.4.100).

Both trained arms overlay the full native coefficient branches plus proto.cv2
and proto.cv3 onto the same original FP32 official checkpoint. All BN state,
including affine parameters, remains exactly official. The resulting detection
identities, boxes, classes and confidences must be bitwise identical across arms.
This evaluates final postprocessed detections, not the full pre-top-k raw head
candidate set; it does not provide the raw geometric five-state GT diagnosis.

Scope is exactly the supplied image list. A 196-image reused diagnostic split is
not a full-COCO or blind-test result; a list covering all official val2017 images
is reported explicitly as full val2017. Original COCO instance annotations,
including COCO crowd/area rules, are used for AP by pycocotools.

Mask decoding follows 8.4.100 SegmentationValidator's save_json=True route:
process_mask_native at the letterboxed input grid, then scale_preds to the
original image. In this source version scale_preds casts the bilinear-scaled
binary masks directly to byte (it does NOT threshold them at 0.5). We invoke
those source methods without changing that behavior. No retina_masks predictor
shortcut and no mask-empty detection filtering is used. Preprocessing is fixed
640x640 square LetterBox, scaleup=False, batch=1, FP32, no augmentation. This
explicit square preprocessing differs from a rectangular-batch stock validator
and must be retained when comparing these paired arms.

Instance damage is an explanatory readout, separate from COCO matching/AP.
For every baseline detection above conf=.001, the same-class non-crowd GT with
highest box IoU is selected if box IoU>=.5. Duplicate detections may select the
same GT and are counted as distinct paired detection identities. Baseline
mask-IoU>=.75 defines success; crossing below .75 defines damage; crossing from
below .75 to >=.75 defines repair. Comparisons use exactly the same detection
index only when frozen box/class parity holds. Bootstrap resamples images as
clusters, including empty images; it does not pretend detections are independent.

Example --models-json:
  {"baseline":{"base_weights":"D:/coco_wire/models/yolo26m-seg.pt",
                "coefficients":"D:/.../baseline/mask_final_ema.pt"},
   "official":"D:/coco_wire/models/yolo26m-seg.pt",
   "acd":{"base_weights":"D:/coco_wire/models/yolo26m-seg.pt",
           "coefficients":"D:/.../acd/mask_final_ema.pt"}}
All files must already exist. This script never installs or downloads packages.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

EVALUATOR_VERSION = "acd_proto_tail_eval_v1"
OVERLAY_PREFIXES = ("model.23.cv4.", "model.23.one2one_cv4.",
                    "model.23.proto.cv2.", "model.23.proto.cv3.")
OVERLAY_KIND = "mask_coefficient_proto_tail_ema_final"


def dump(path: Path, value):
    """Atomic metadata write; JSON has no NaN/Infinity unknown values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_models(value: str):
    if value.lstrip().startswith("{"):
        result = json.loads(value)
    else:
        result = json.loads(Path(value.removeprefix("@")).read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict) or not result:
        raise ValueError("--models-json must be a nonempty name:model mapping")
    for name, spec in result.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Unsafe model name: {name}")
        if not isinstance(spec, (str, dict)):
            raise ValueError(f"Model {name} must be a checkpoint path or base_weights/coefficients object")
        if isinstance(spec, dict) and set(spec) != {"base_weights", "coefficients"}:
            raise ValueError(f"Model {name}: object requires exactly base_weights and coefficients")
    return result


def weight_fingerprint(spec):
    paths = {"checkpoint": spec} if isinstance(spec, str) else spec
    result = {}
    for key, filename in paths.items():
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    return result


def overlay_state(model, torch):
    """Exact overlay keys, learnable non-BN parameters, and frozen BN state."""
    if len(model.model) != 24:
        raise ValueError("Expected the protocol's YOLO26m model.23 mask head")
    state = model.state_dict()
    expected = {key for key in state if key.startswith(OVERLAY_PREFIXES)}
    for prefix in OVERLAY_PREFIXES:
        if not any(key.startswith(prefix) for key in expected):
            raise ValueError(f"Checkpoint is missing required overlay scope: {prefix}")
    bn_keys = {module_name + "." + local_name
               for module_name, module in model.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for local_name in module.state_dict()}
    mutable = {key for key, _ in model.named_parameters()
               if key in expected and key not in bn_keys}
    if not mutable:
        raise ValueError("Overlay scope has no non-BN parameters")
    return expected, mutable, bn_keys


def frozen_digest(model, torch):
    _, mutable_keys, _ = overlay_state(model, torch)
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key in mutable_keys:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), sorted(mutable_keys)


def load_model(spec, YOLO, torch):
    source = spec if isinstance(spec, str) else spec["base_weights"]
    if not Path(source).is_file():
        raise FileNotFoundError(source)
    wrapper = YOLO(str(Path(source).resolve()))
    model = wrapper.model.cpu().float().eval()
    if not getattr(model, "end2end", False):
        raise ValueError("Expected native YOLO26 end-to-end one2one inference")
    expected, mutable_keys, bn_keys = overlay_state(model, torch)
    metadata = {"source_checkpoint": str(Path(source).resolve()),
                "reconstructed_coefficient_only": False,
                "reconstructed_mask_tail_overlay": False,
                "overlay_scope_prefixes": list(OVERLAY_PREFIXES),
                "overlay_state_keys": [],
                "overlay_non_bn_parameter_keys": sorted(mutable_keys),
                "overlay_bn_state_keys": sorted(expected & bn_keys),
                "frozen_digest_excludes": "only non-BN parameters in the declared overlay scope; all BN and all other state retained"}
    if isinstance(spec, dict):
        payload = torch.load(spec["coefficients"], map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Mask-tail overlay payload must be a dictionary")
        if payload.get("kind") != OVERLAY_KIND:
            raise ValueError(f"Overlay kind must be {OVERLAY_KIND}; failed/partial/legacy payloads are refused")
        if type(payload.get("epoch")) is not int or payload["epoch"] != 3:
            raise ValueError("Overlay must be the fixed final third-epoch EMA readout")
        if payload.get("audit_passed") is not True or payload.get("status") == "failed":
            raise ValueError("Overlay requires audit_passed=true; failed payloads are refused")
        if payload.get("base_weights_sha256") != sha(Path(source)):
            raise ValueError("Overlay initialization hash must match the supplied original checkpoint")
        coefficients = payload.get("state_dict")
        if not isinstance(coefficients, dict):
            raise ValueError("Mask-tail overlay payload requires state_dict")
        if set(coefficients) != expected:
            raise ValueError(f"Mask-tail overlay key mismatch: missing={sorted(expected-set(coefficients))}, "
                             f"extra={sorted(set(coefficients)-expected)}")
        state = model.state_dict()
        for key, value in coefficients.items():
            if not isinstance(value, torch.Tensor) or value.shape != state[key].shape:
                raise ValueError(f"Invalid mask-tail overlay tensor: {key}")
            if value.dtype != state[key].dtype:
                raise ValueError(f"Overlay dtype must exactly match original FP32 state: {key}")
            if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f"Overlay needs finite FP32 floating state: {key}")
            if key not in mutable_keys and not torch.equal(value, state[key]):
                raise ValueError(f"All BN affine/statistics and other frozen state must remain exactly official: {key}")
            state[key] = value.detach().cpu()
        model.load_state_dict(state, strict=True)
        metadata.update(reconstructed_mask_tail_overlay=True,
                        overlay_state_keys=sorted(expected), overlay_audit_passed=True,
                        overlay_kind=payload["kind"], overlay_epoch=payload["epoch"],
                        overlay_base_weights_sha256=payload["base_weights_sha256"],
                        all_overlay_bn_exact_official=True,
                        coefficient_file=str(Path(spec["coefficients"]).resolve()),
                        coefficient_kind=payload.get("kind"), coefficient_epoch=payload.get("epoch"))
    model.model[-1].max_det = 300
    model.model[-1].agnostic_nms = False
    return model, metadata


def gather_images(images_list: Path, coco):
    paths = []
    ids = []
    by_filename = {image["file_name"]: iid for iid, image in coco.imgs.items()}
    for line in images_list.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = images_list.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        iid = by_filename.get(path.name)
        if iid is None:
            raise ValueError(f"Image is absent from annotation file: {path}")
        if iid in ids:
            raise ValueError(f"Duplicate image identity: {iid}")
        ids.append(iid)
        paths.append(path)
    if not ids:
        raise ValueError("Image list is empty")
    return paths, ids


def predict_arm(name, spec, fingerprint, paths, ids, coco, out, modules, args):
    np, torch, cv2, YOLO, LetterBox, SegmentationValidator, ops, mask_utils, converter = modules
    arm = out / name
    arm.mkdir(parents=True, exist_ok=True)
    complete = arm / "COMPLETE.json"
    if complete.is_file():
        prior = json.loads(complete.read_text(encoding="utf-8"))
        files_ok = all((arm / "images" / f"{iid:012d}.json").is_file() for iid in ids)
        if prior.get("fingerprint") == fingerprint and files_ok and (arm / "predictions.json").is_file():
            print(f"CACHE_REUSE {name}: {len(ids)} images", flush=True)
            return prior
        raise RuntimeError(f"Existing completed cache does not match inputs: {arm}; use a new --out")
    partial_identity = arm / "INPUTS.json"
    if partial_identity.is_file():
        if json.loads(partial_identity.read_text(encoding="utf-8")) != fingerprint:
            raise RuntimeError(f"Partial cache does not match inputs: {arm}; use a new --out")
    else:
        dump(partial_identity, fingerprint)
    model, model_metadata = load_model(spec, YOLO, torch)
    frozen_hash, mutable_keys = frozen_digest(model, torch)
    model = model.to(args.device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    validator = SegmentationValidator(args={"conf": .001, "iou": .7, "max_det": 300,
                                            "save_json": True, "plots": False, "imgsz": 640,
                                            "half": False, "rect": False})
    # Do not call init_metrics: that routine would require faster-coco-eval and
    # may auto-install it. Its mask-selection and detection settings are fixed
    # below; AP is evaluated separately with the already-installed pycocotools.
    validator.nc = len(model.names)
    if validator.nc != 80:
        raise ValueError(f"Expected official COCO 80 classes, got {validator.nc}")
    validator.end2end = True
    validator.process = ops.process_mask_native
    categories = converter.coco80_to_coco91_class()
    letterbox = LetterBox((640, 640), auto=False, scaleup=False, stride=32)
    (arm / "images").mkdir(exist_ok=True)
    start = time.monotonic()
    reused = 0
    total_predictions = 0
    with torch.inference_mode():
        for position, (path, iid) in enumerate(zip(paths, ids), 1):
            cache_file = arm / "images" / f"{iid:012d}.json"
            if cache_file.is_file():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("image_id") != iid or cached.get("image_path") != str(path):
                    raise RuntimeError(f"Invalid partial image cache: {cache_file}")
                total_predictions += len(cached["detections"])
                reused += 1
                continue
            original = cv2.imread(str(path))
            if original is None:
                raise ValueError(f"Cannot read image: {path}")
            original_shape = original.shape[:2]
            gt_image = coco.imgs[iid]
            if original_shape != (gt_image["height"], gt_image["width"]):
                raise ValueError(f"Original image/GT dimensions differ: {path}")
            # Read the exact LetterBox parameters, including integer padding.
            params = letterbox.get_params({"img": original})
            resized = letterbox(image=original)
            tensor = torch.from_numpy(np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1)))
            tensor = tensor[None].to(args.device).float() / 255
            predictions = validator.postprocess(model(tensor))[0]
            raw_boxes = predictions["bboxes"].detach().cpu().numpy()
            raw_conf = predictions["conf"].detach().cpu().numpy()
            classes = predictions["cls"].detach().cpu().numpy().astype(int)
            pbatch = {"imgsz": tensor.shape[-2:], "ori_shape": original_shape,
                      "ratio_pad": (params["ratio"], (params["left"], params["top"])),
                      "im_file": str(path)}
            scaled = validator.scale_preds(predictions, pbatch)
            boxes = scaled["bboxes"].detach().cpu().numpy()
            masks = scaled["masks"].detach().cpu().numpy()
            detections = []
            for index, (box, raw_box, score, cls, mask) in enumerate(zip(boxes, raw_boxes, raw_conf, classes, masks)):
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                xywh = [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]
                detections.append({"image_id": int(iid), "category_id": int(categories[cls]),
                                   "bbox": [round(x, 3) for x in xywh], "score": round(float(score), 5),
                                   "segmentation": rle, "detection_index": index,
                                   "raw_input_box_xyxy": raw_box.tolist(), "raw_confidence": float(score),
                                   "box_xyxy": box.tolist(), "model_class": int(cls)})
            dump(cache_file, {"image_id": int(iid), "image_path": str(path),
                              "original_shape": list(original_shape), "letterbox": params | {"orig_shape": list(params["orig_shape"])},
                              "detections": detections})
            total_predictions += len(detections)
            if position == 1 or position % 25 == 0 or position == len(ids):
                print(f"PREDICT {name} {position}/{len(ids)} detections={total_predictions} "
                      f"elapsed_s={time.monotonic()-start:.1f}", flush=True)
    # Assemble standard COCO JSON without retaining all RLE objects in memory.
    temporary = arm / "predictions.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for iid in ids:
            cached = json.loads((arm / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))
            for detection in cached["detections"]:
                standard = {k: detection[k] for k in ("image_id", "category_id", "bbox", "score", "segmentation")}
                if not first:
                    handle.write(",\n")
                handle.write(json.dumps(standard, separators=(",", ":"), allow_nan=False))
                first = False
        handle.write("\n]\n")
    temporary.replace(arm / "predictions.json")
    del model, validator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    receipt = {"status": "prediction_complete", "fingerprint": fingerprint,
               "model": model_metadata, "frozen_state_sha256": frozen_hash,
               "overlay_non_bn_parameter_keys": mutable_keys,
               "coefficient_state_keys": [key for key in mutable_keys if key.startswith(OVERLAY_PREFIXES[:2])],
               "readout_view": "postprocessed native one2one detections after top-k/conf; not full raw candidates",
               "image_count": len(ids),
               "detections": total_predictions, "reused_partial_images": reused,
               "elapsed_seconds_this_invocation": time.monotonic()-start,
               "predictions_sha256": sha(arm / "predictions.json")}
    dump(complete, receipt)
    return receipt


def coco_ap(name, coco, ids, out, COCO, COCOeval):
    arm = out / name
    output = arm / "COCO_METRICS.json"
    prediction_sha = sha(arm / "predictions.json")
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("predictions_sha256") == prediction_sha and cached.get("image_ids") == sorted(ids):
            print(f"METRICS_CACHE_REUSE {name}", flush=True)
            return cached
    result = {"image_ids": sorted(ids), "predictions_sha256": prediction_sha,
              "backend": "pycocotools.COCOeval", "units": "fraction (multiply by 100 for AP points)"}
    names = ["AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge"]
    for iou_type in ("segm", "bbox"):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            predictions = json.loads((arm / "predictions.json").read_text(encoding="utf-8"))
            if iou_type == "segm":
                # COCO.loadRes prefers bbox when both keys are present. Drop
                # bbox for segmentation evaluation so detection area is the
                # mask area, as required for standard APsmall/medium/large.
                for prediction in predictions:
                    prediction.pop("bbox", None)
            else:
                for prediction in predictions:
                    prediction.pop("segmentation", None)
            if predictions:
                detected = coco.loadRes(predictions)
            else:
                detected = COCO()
                detected.dataset = {"images": list(coco.imgs.values()), "categories": list(coco.cats.values()), "annotations": []}
                detected.createIndex()
            evaluator = COCOeval(coco, detected, iou_type)
            evaluator.params.imgIds = sorted(ids)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        (arm / f"COCO_{iou_type}.txt").write_text(stream.getvalue(), encoding="utf-8")
        result[iou_type] = {key: float(value) if value >= 0 else None for key, value in zip(names, evaluator.stats)}
        # Save the standard accumulated tensors so AP is independently inspectable.
        import numpy as np
        np.savez_compressed(arm / f"COCO_{iou_type}_ACCUMULATED.npz",
                            precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                            scores=evaluator.eval["scores"], iou_thresholds=evaluator.params.iouThrs,
                            recall_thresholds=evaluator.params.recThrs, category_ids=evaluator.params.catIds,
                            max_detections=evaluator.params.maxDets, area_ranges=evaluator.params.areaRng)
        print(f"COCO_AP {name} {iou_type}: {json.dumps(result[iou_type])}", flush=True)
        del evaluator, detected, predictions
    dump(output, result)
    return result


def box_iou(a, b, np):
    if not len(b):
        return np.zeros(0)
    b = np.asarray(b, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(a[2:], b[:, 2:])-np.maximum(a[:2], b[:, :2])).prod(1)
    union = np.maximum(0, a[2:]-a[:2]).prod() + np.maximum(0, b[:, 2:]-b[:, :2]).prod(1) - intersection
    return intersection / np.maximum(union, 1e-12)


def bootstrap_summary(per_image, np, samples, seed):
    # Columns: matches, delta_sum, success_count, damage_count, failure_count,
    # repair_count, baseline_iou_sum, method_iou_sum, success_delta_sum.
    sums = per_image.sum(0)
    definitions = {"mean_mask_iou_delta": (1, 0), "damage_rate_of_baseline_success": (3, 2),
                   "repair_rate_of_baseline_failure": (5, 4), "baseline_mean_mask_iou": (6, 0),
                   "method_mean_mask_iou": (7, 0), "baseline_success_mean_iou_delta": (8, 2)}
    rng = np.random.default_rng(seed)
    draws = []
    # Bounded memory: full-val 5000x5000 draw arrays would consume 200 MB.
    for offset in range(0, samples, 100):
        batch = min(100, samples-offset)
        indices = rng.integers(0, len(per_image), size=(batch, len(per_image)))
        draws.append(per_image[indices].sum(1))
    draws = np.concatenate(draws, axis=0)
    result = {"matched_detection_count": int(sums[0]), "baseline_success_count": int(sums[2]),
              "damage_count": int(sums[3]), "baseline_failure_count": int(sums[4]),
              "repair_count": int(sums[5]), "images": len(per_image),
              "images_with_matched_detections": int((per_image[:, 0] > 0).sum()),
              "bootstrap_samples": samples, "bootstrap_seed": seed,
              "bootstrap_unit": "supplied images, paired across arms; ratio of clustered sums"}
    for key, (numerator, denominator) in definitions.items():
        valid = draws[:, denominator] > 0
        distribution = draws[valid, numerator] / draws[valid, denominator]
        result[key] = {"value": float(sums[numerator]/sums[denominator]) if sums[denominator] else None,
                       "ci95": np.quantile(distribution, [.025, .975]).tolist() if len(distribution) else None,
                       "valid_bootstrap_draws": int(valid.sum())}
    valid = per_image[:, 0] > 0
    result["image_macro_mask_iou_delta"] = float(np.mean(per_image[valid, 1]/per_image[valid, 0])) if valid.any() else None
    return result


def paired_metrics(baseline, names, coco, ids, out, receipts, np, mask_utils, args):
    output = {}
    baseline_hash = receipts[baseline]["frozen_state_sha256"]
    for name in names:
        if name == baseline:
            continue
        pairdir = out / f"paired_{name}_vs_{baseline}"
        pairdir.mkdir(parents=True, exist_ok=True)
        identity = {"baseline_predictions": receipts[baseline]["predictions_sha256"],
                    "method_predictions": receipts[name]["predictions_sha256"],
                    "evaluator_version": EVALUATOR_VERSION, "bootstrap": args.bootstrap,
                    "bootstrap_seed": args.bootstrap_seed}
        summary_file = pairdir / "SUMMARY.json"
        if summary_file.is_file():
            prior = json.loads(summary_file.read_text(encoding="utf-8"))
            if prior.get("input_identity") == identity:
                print(f"PAIRED_CACHE_REUSE {name}", flush=True)
                output[name] = prior
                continue
        parity = {"frozen_state_exact_equal": baseline_hash == receipts[name]["frozen_state_sha256"],
                  "image_count": len(ids), "detection_count_mismatch_images": 0,
                  "class_mismatch_detections": 0, "box_max_abs_error": 0., "confidence_max_abs_error": 0.,
                  "detection_index_equal": True, "all_images_box_class_exact_equal": True,
                  "pairing": "same native postprocess detection index; no mask-dependent filtering",
                  "readout_view": "output detections; raw candidate geometry/five-state diagnosis not computed here"}
        unique_gt = set()
        counts = np.zeros((len(ids), 9), dtype=np.float64)
        with (pairdir / "INSTANCES.jsonl").open("w", encoding="utf-8") as instances, \
             (pairdir / "IMAGES.jsonl").open("w", encoding="utf-8") as images:
            for image_pos, iid in enumerate(ids):
                base = json.loads((out / baseline / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))["detections"]
                method = json.loads((out / name / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))["detections"]
                image_equal = len(base) == len(method)
                if not image_equal:
                    parity["detection_count_mismatch_images"] += 1
                if image_equal and base:
                    base_boxes = np.asarray([d["raw_input_box_xyxy"] for d in base])
                    method_boxes = np.asarray([d["raw_input_box_xyxy"] for d in method])
                    base_conf = np.asarray([d["raw_confidence"] for d in base])
                    method_conf = np.asarray([d["raw_confidence"] for d in method])
                    mismatches = sum(a["category_id"] != b["category_id"] for a, b in zip(base, method))
                    box_error = float(np.max(np.abs(method_boxes-base_boxes)))
                    conf_error = float(np.max(np.abs(method_conf-base_conf)))
                    parity["class_mismatch_detections"] += mismatches
                    parity["box_max_abs_error"] = max(parity["box_max_abs_error"], box_error)
                    parity["confidence_max_abs_error"] = max(parity["confidence_max_abs_error"], conf_error)
                    index_equal = all(a["detection_index"] == b["detection_index"] for a, b in zip(base, method))
                    parity["detection_index_equal"] &= index_equal
                    image_equal &= mismatches == 0 and box_error == 0 and conf_error == 0 and index_equal
                parity["all_images_box_class_exact_equal"] &= image_equal
                if not image_equal:
                    images.write(json.dumps({"image_id": iid, "paired_valid": False,
                                             "reason": "box/class/confidence/detection-index parity failed"})+"\n")
                    continue
                anns_by_class = {}
                gt_rles = {}
                for ann in coco.imgToAnns.get(iid, []):
                    if not ann.get("iscrowd", 0):
                        anns_by_class.setdefault(ann["category_id"], []).append(ann)
                for a, b in zip(base, method):
                    candidates = anns_by_class.get(a["category_id"], [])
                    if not candidates:
                        continue
                    gt_boxes = [[x["bbox"][0], x["bbox"][1], x["bbox"][0]+x["bbox"][2], x["bbox"][1]+x["bbox"][3]] for x in candidates]
                    overlaps = box_iou(np.asarray(a["box_xyxy"]), gt_boxes, np)
                    gt_index = int(np.argmax(overlaps))
                    if overlaps[gt_index] < .5:
                        continue
                    gt = candidates[gt_index]
                    gid = int(gt["id"])
                    if gid not in gt_rles:
                        gt_rles[gid] = coco.annToRLE(gt)
                    def overlap(detection):
                        rle = detection["segmentation"] | {"counts": detection["segmentation"]["counts"].encode("ascii")}
                        return float(mask_utils.iou([rle], [gt_rles[gid]], [0])[0, 0])
                    baseline_iou, method_iou = overlap(a), overlap(b)
                    success = baseline_iou >= .75
                    damage = success and method_iou < .75
                    repair = not success and method_iou >= .75
                    delta = method_iou-baseline_iou
                    counts[image_pos] += [1, delta, int(success), int(damage), int(not success), int(repair),
                                          baseline_iou, method_iou, delta if success else 0]
                    unique_gt.add((iid, gid))
                    row = {"image_id": iid, "detection_index": a["detection_index"], "annotation_id": gid,
                           "category_id": a["category_id"], "baseline_score": a["raw_confidence"],
                           "baseline_box_iou": float(overlaps[gt_index]), "gt_area": float(gt["area"]),
                           "baseline_mask_iou": baseline_iou, "method_mask_iou": method_iou,
                           "delta_mask_iou": delta, "baseline_success": success, "damage": damage, "repair": repair}
                    instances.write(json.dumps(row, allow_nan=False)+"\n")
                c = counts[image_pos]
                images.write(json.dumps({"image_id": iid, "paired_valid": True,
                                         "matched_detections": int(c[0]), "mask_iou_delta_sum": float(c[1]),
                                         "success": int(c[2]), "damage": int(c[3]),
                                         "failure": int(c[4]), "repair": int(c[5])})+"\n")
        summary = {"input_identity": identity, "parity": parity,
                   "paired_readout_valid": parity["frozen_state_exact_equal"] and parity["all_images_box_class_exact_equal"],
                   "definition": "all baseline detections conf>.001, same-class highest-box-IoU noncrowd GT>=.5; "
                                 "duplicate detection-to-GT matches allowed; success/damage/repair at maskIoU=.75",
                   "unique_matched_gt_instances": len(unique_gt),
                   "statistics": bootstrap_summary(counts, np, args.bootstrap, args.bootstrap_seed)}
        if not summary["paired_readout_valid"]:
            summary["limitation"] = "Invalid frozen parity; paired rows only cover parity-passing images and cannot support the gate"
        dump(summary_file, summary)
        output[name] = summary
        print(f"PAIRED {name} valid={summary['paired_readout_valid']} "
              f"damage={summary['statistics']['damage_count']} repair={summary['statistics']['repair_count']}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", required=True, help="JSON object, JSON filename, or @JSON filename")
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--vendor", type=Path, default=Path("D:/coco_wire/vendor_8.4.100"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    if not (args.vendor / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"Expected already-present Ultralytics vendor: {args.vendor}")
    sys.path.insert(0, str(args.vendor.resolve()))
    import ultralytics
    if ultralytics.__version__ != "8.4.100":
        raise RuntimeError(f"Required Ultralytics 8.4.100, got {ultralytics.__version__}")
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO
    from ultralytics.data import converter
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.utils import ops
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    specs = parse_models(args.models_json)
    if args.baseline not in specs:
        raise ValueError(f"Baseline name {args.baseline!r} is absent from --models-json")
    coco = COCO(str(args.annotations.resolve()))
    paths, ids = gather_images(args.images_list.resolve(), coco)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source_files = {"segment_validator": Path(SegmentationValidator.__module__.replace(".", "/")+".py"),
                    "ops": Path("ultralytics/utils/ops.py"), "nms": Path("ultralytics/utils/nms.py"),
                    "letterbox": Path("ultralytics/data/augment.py")}
    # SegmentationValidator.__module__ begins ultralytics/models/... already.
    source_hashes = {name: sha(args.vendor / relative) for name, relative in source_files.items()}
    common = {"evaluator_version": EVALUATOR_VERSION, "evaluator_sha256": sha(Path(__file__)),
              "ultralytics": ultralytics.__version__, "torch": torch.__version__,
              "vendor_source_sha256": source_hashes,
              "annotations": {"path": str(args.annotations.resolve()), "sha256": sha(args.annotations)},
              "images_list": {"path": str(args.images_list.resolve()), "sha256": sha(args.images_list)},
              "image_ids": ids,
              "overlay_scope_prefixes": list(OVERLAY_PREFIXES),
              "config": {"imgsz": 640, "shape": [640, 640], "batch": 1, "half": False,
                         "conf": .001, "max_det": 300, "coco_maxDets": [1, 10, 100],
                         "rect": False, "scaleup": False, "augment": False, "native_one2one": True,
                         "decode": "SegmentationValidator.save_json=True: process_mask_native; scale_preds.byte()",
                         "device": args.device, "tf32": False}}
    modules = (np, torch, cv2, YOLO, LetterBox, SegmentationValidator, ops, mask_utils, converter)
    receipts = {}
    metrics = {}
    ordered_names = [args.baseline] + [name for name in specs if name != args.baseline]
    for name in ordered_names:
        fingerprint = common | {"weights": weight_fingerprint(specs[name])}
        receipts[name] = predict_arm(name, specs[name], fingerprint, paths, ids, coco, args.out, modules, args)
        metrics[name] = coco_ap(name, coco, ids, args.out, COCO, COCOeval)
    pairs = paired_metrics(args.baseline, ordered_names, coco, ids, args.out, receipts, np, mask_utils, args)
    deltas = {}
    for name in ordered_names:
        if name == args.baseline:
            continue
        deltas[name] = {}
        for iou_type in ("segm", "bbox"):
            deltas[name][iou_type] = {}
            for key in ("AP", "AP75", "APsmall"):
                a, b = metrics[args.baseline][iou_type][key], metrics[name][iou_type][key]
                deltas[name][iou_type][key+"_points"] = 100*(b-a) if a is not None and b is not None else None
    full_annotation_list = sorted(ids) == sorted(coco.imgs)
    summary = {"status": "complete", "baseline": args.baseline, "image_count": len(ids),
               "covers_all_annotation_images": full_annotation_list,
               "evaluation_scope": "all images in supplied original COCO annotations" if full_annotation_list else
                                   "supplied COCO image subset; do not interpret as full-val or blind test",
               "configuration": common, "prediction_receipts": receipts, "metrics": metrics,
               "delta_vs_baseline": deltas, "paired": pairs,
               "limitations": ["One seed / short training feasibility screen; inference evaluation does not establish statistical training-seed generalization",
                               "Instance-damage matching is a separate explanatory readout, not COCO one-to-one AP matching",
                               "This evaluator reads final postprocessed detections; full raw-candidate geometry and five-state GT classification require a separate diagnosis",
                               "Square batch-1 preprocessing is fixed explicitly; stock rectangular-batch validator may produce different absolute AP"]}
    dump(args.out / "SUMMARY.json", summary)
    print(f"EVALUATION_COMPLETE {args.out / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
