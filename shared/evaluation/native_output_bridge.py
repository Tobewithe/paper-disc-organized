"""Observe pinned native YOLO26 segmentation validation and replay its exact masks.

This is an inference/cache tool, not a trainer. Every trace writes a NEW directory.
GT is used by the native evaluator only, never by output/feature selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from posttrain_audit import compare_outputs, fingerprint_base_state, read_json, require, sha256, write_new

VERSION = "8.4.100"
MODEL_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
GT_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
SOURCE_MODULES = (
    "ultralytics.nn.modules.head", "ultralytics.models.yolo.segment.val",
    "ultralytics.models.yolo.detect.val", "ultralytics.utils.ops",
    "ultralytics.utils.nms", "ultralytics.engine.validator",
    "ultralytics.nn.autobackend", "ultralytics.nn.backends.pytorch",
)
COCO_METRICS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL")
RUNTIME_KEYS = ("matmul_precision", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
                "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms",
                "deterministic_warn_only", "cublas_workspace_config")


def apply_runtime_policy(policy):
    """Apply a declared numerical policy before the first CUDA operation, including replay."""
    require(set(policy) == set(RUNTIME_KEYS), "incomplete/unknown numerical policy")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = policy["cublas_workspace_config"]
    import torch
    torch.set_float32_matmul_precision(policy["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = policy["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = policy["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = policy["cudnn_deterministic"]
    torch.use_deterministic_algorithms(policy["deterministic_algorithms"],
                                       warn_only=policy["deterministic_warn_only"])


def coco_metric_receipt(evaluator):
    """Read the actual native evaluator, before its local object is discarded."""
    p = evaluator.params
    require(len(evaluator.stats) >= 12, "incomplete native COCO summary")
    require(list(p.maxDets) == [1, 10, 100] and p.useCats == 1, "COCO counting policy changed")
    return {"metrics": dict(zip(COCO_METRICS, [100 * float(v) if v >= 0 else None
                                             for v in evaluator.stats[:12]])),
            "params": {"iouType": p.iouType, "maxDets": list(p.maxDets), "useCats": int(p.useCats),
                       "iouThrs": [float(x) for x in p.iouThrs],
                       "areaRng": [[float(x) for x in a] for a in p.areaRng],
                       "areaRngLbl": list(p.areaRngLbl), "catIds": [int(x) for x in p.catIds],
                       "imgIds": [int(x) for x in p.imgIds]}}


def native_environment():
    import torch
    import ultralytics
    require(ultralytics.__version__ == VERSION, f"requires ultralytics=={VERSION}")
    pins = read_json(Path(__file__).with_name("native_source_pins.json"))
    hashes = {}
    for name in SOURCE_MODULES:
        source = Path(importlib.import_module(name).__file__).read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        require(digest == pins["sha256_normalized_utf8"][name], f"pinned source changed: {name}")
        hashes[name] = digest
    return {"ultralytics_version": VERSION, "torch_version": str(torch.__version__),
            "source_sha256": hashes, "python_version": sys.version,
            "cuda_version": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "execution_env": {k: os.environ.get(k) for k in
                              ("YOLO_OFFLINE", "YOLO_CONFIG_DIR", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torchvision", "numpy", "pycocotools", "opencv-python", "Pillow", "PyYAML")}}


def decode_native(cache, coefficients=None, device=None):
    """Same two-stage order as v8.4.100 save_json segmentation validation.

    process_mask_native thresholds logits, crops; scale_masks then resizes those
    BINARY masks to original shape and converts to byte. Do not threshold twice.
    """
    import torch
    from ultralytics.utils import ops
    device = device or cache["device"]
    p = cache["proto"].to(device)
    c = (cache["c0"] if coefficients is None else coefficients).to(device)
    boxes = cache["boxes_input"].to(device)
    require(c.shape == cache["c0"].shape and torch.isfinite(c).all().item(), "invalid coefficients")
    require(c.dtype == cache["c0"].dtype, "coefficient dtype changed")
    require(list(cache["native_decode_hw"]) == [4 * s for s in p.shape[1:]], "decoder shape changed")
    with torch.inference_mode():
        masks = ops.process_mask_native(p, c, boxes, shape=cache["native_decode_hw"])
        return ops.scale_masks(masks[None], cache["original_hw"],
                               ratio_pad=cache["ratio_pad"])[0].byte()


def encode_rle(mask):
    import numpy as np
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.detach().cpu().numpy().astype(np.uint8)))
    return {"size": [int(s) for s in rle["size"]], "counts": rle["counts"].decode("ascii")}


def raw_location(raw_id, shapes):
    """Actual concatenated head-feat ordering; repeated raw IDs are allowed."""
    require(type(raw_id) is int and raw_id >= 0, "invalid raw ID")
    offset = 0
    for level, shape in enumerate(shapes):
        channels, height, width = shape
        require(min(channels, height, width) > 0, "invalid feature shape")
        if raw_id < offset + height * width:
            local = raw_id - offset
            return {"level": level, "row": local // width, "column": local % width,
                    "level_offset": offset, "channels": channels}
        offset += height * width
    raise ValueError("raw ID outside actual head features")


def make_trace_validator(out, expected_ids, scope, cache_limit, annotations_sha,
                         requested_args=None, image_ids_sha=None, runtime_policy=None):
    """Factory keeps tracing state outside global monkey patches."""
    import torch
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.nn.modules.head import Segment26
    from ultralytics.utils import ops

    class TraceValidator(SegmentationValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace_head = None
            self.records, self.image_records = [], []
            self.cache_bytes = 0
            self.actual_ids = []
            self.trace_topk = None

        def __call__(self, trainer=None, model=None):
            require(trainer is None, "training invocation forbidden")
            try:
                result = super().__call__(trainer=None, model=model)
                require(sorted(self.actual_ids) == sorted(expected_ids), "image coverage mismatch")
                after = fingerprint_base_state(self.trace_model)
                write_new(out / "base_after.json", after)
                require(after == self.base_before, "base parameters/buffers/modes changed")
                predictions = self.save_dir / "predictions.json"
                if not predictions.exists() and not self.jdict:
                    write_new(predictions, [])
                source_sha = sha256(predictions)
                require(read_json(predictions) == self.jdict, "native JSON changed")
                for row in self.records:
                    row["output_detection_id"] = f"{source_sha}:{row['prediction_index']}"
                    row["source_prediction_sha"] = source_sha
                with (out / "OUTPUT_FEATURE_MAP.jsonl").open("x", encoding="utf-8") as f:
                    for row in self.records:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                write_new(out / "TRACE_MANIFEST.json", {
                    "schema_version": 1, "scope": scope, "environment": native_environment(),
                    "bridge_code_sha256": sha256(__file__),
                    "source_pins_sha256": sha256(Path(__file__).with_name("native_source_pins.json")),
                    "state_fingerprint_scope": "native fused base after backend init/warmup",
                    "model_sha256": MODEL_SHA, "annotations_sha256": annotations_sha,
                    "predictions_path": str(predictions.resolve()), "predictions_sha256": source_sha,
                    "image_ids": self.actual_ids, "images": self.image_records,
                    "native_args": json.loads(json.dumps(vars(self.args), default=str)), "cache_bytes": self.cache_bytes,
                    "all_output_rows_traced": True, "capture_zero_delta_rle_equal": True,
                    "base_state_unchanged": True,
                    "scientific_status": "not_adjudicated",
                })
                require(set(self.native_coco_receipts) == {"bbox", "segm"}, "native COCO backend did not complete")
                write_new(out / "native_eval_summary.json", {
                    "backend": "COCOeval_faster", "unit": "AP_points",
                    "backend_version": importlib.metadata.version("faster-coco-eval"),
                    "annotations_sha256": annotations_sha, "predictions_sha256": source_sha,
                    "image_ids_sha256": image_ids_sha,
                    **{task: receipt["metrics"] for task, receipt in self.native_coco_receipts.items()},
                    "evaluator_receipts": self.native_coco_receipts,
                    "capture": "observed native evaluator summarize; no separate native reevaluation",
                })
                write_new(out / "internal_metrics.json", self.metrics.results_dict)
                return result
            finally:
                if self.trace_head is not None:
                    if self.had_instance_topk:
                        self.trace_head.get_topk_index = self.old_instance_topk
                    else:
                        del self.trace_head.get_topk_index

        def eval_json(self, *args, **kwargs):
            # The pinned implementation catches backend errors and overwrites some B keys with M.
            # Observe each real evaluator directly; never promote the fallback internal stats to COCO AP.
            import faster_coco_eval
            original = faster_coco_eval.COCOeval_faster
            receipts = {}

            class ObservedEvaluator(original):
                def summarize(self, *a, **kw):
                    result = super().summarize(*a, **kw)
                    require(self.params.iouType not in receipts, "duplicate native evaluator")
                    receipts[self.params.iouType] = coco_metric_receipt(self)
                    return result

            try:
                faster_coco_eval.COCOeval_faster = ObservedEvaluator
                # Segment.eval_json calls super().coco_evaluate directly, bypassing a
                # subclass coco_evaluate override. Wrap the actual eval_json entry.
                result = super().eval_json(*args, **kwargs)
            finally:
                faster_coco_eval.COCOeval_faster = original
            require(set(receipts) == {"bbox", "segm"}, "native COCO failed; internal fallback forbidden")
            self.native_coco_receipts = receipts
            return result

        def init_metrics(self, model):
            super().init_metrics(model)
            require(self.is_coco and self.nc == 80, "native COCO category mapping required")
            from ultralytics.data.converter import coco80_to_coco91_class
            require(self.class_map == coco80_to_coco91_class(), "wrong COCO category mapping")
            for key, value in (requested_args or {}).items():
                if key != "data":
                    require(getattr(self.args, key) == value, f"effective argument drift: {key}")
            effective = native_environment()
            for key, value in (runtime_policy or {}).items():
                require(effective[key] == value, f"effective numerical policy drift: {key}")
            write_new(out / "NATIVE_ARGS.json", vars(self.args))
            write_new(out / "ENVIRONMENT.json", effective)
            require(self.end2end and self.args.save_json and self.process is ops.process_mask_native,
                    "native O2O save_json decoder required")
            require(not self.args.single_cls and not self.args.agnostic_nms and self.args.classes is None,
                    "class/candidate policy changed")
            require(not self.args.augment and not self.args.compile, "augmentation/compile not traced")
            require(not torch.distributed.is_initialized(), "only single-process tracing supported")
            require(sha256(self.data["path"] / "annotations" / "instances_val2017.json") == annotations_sha,
                    "native evaluator uses different annotations")
            dataset_ids = [int(Path(p).stem) for p in self.dataloader.dataset.im_files]
            require(len(set(dataset_ids)) == len(dataset_ids) and sorted(dataset_ids) == sorted(expected_ids),
                    "native dataloader must match predeclared image set exactly")
            require(getattr(model, "format", None) == "pt", "native PyTorch AutoBackend required")
            base = model.backend.model  # backend is NOT a registered nn.Module child in 8.4.100
            require(not any(p.requires_grad for p in base.parameters()), "base has trainable parameters")
            heads = [m for m in base.modules() if isinstance(m, Segment26)]
            require(len(heads) == 1, "exactly one Segment26 head required")
            head = heads[0]
            require(head.end2end and not head.export and not head.agnostic_nms, "head policy mismatch")
            require(not model.fp16 and next(base.parameters()).dtype == torch.float32, "FP32 required")
            require(base.is_fused() and head.max_det == self.args.max_det, "fusion/head budget changed")
            require(head.nc == 80 and head.nm == 32 and head.reg_max == 1, "unexpected head architecture")
            write_new(out / "MODEL_RECEIPT.json", {
                "end2end": head.end2end, "agnostic_nms": head.agnostic_nms, "max_det": head.max_det,
                "nc": head.nc, "nm": head.nm, "npr": head.npr, "reg_max": head.reg_max,
                "fused": base.is_fused(), "parameter_dtype": str(next(base.parameters()).dtype),
                "format": model.format, "device": str(model.device), "class_map": self.class_map,
                "is_coco": self.is_coco, "decoder": "process_mask_native_then_scale_masks_byte",
                "device_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "CPU"})
            self.trace_model = base
            self.feature_sources = {
                "head_module": next(name for name, m in base.named_modules() if m is head),
                "head_layer_index": head.i, "source_layer_indices": head.f,
                "feature_definition": "one2one.feats at actual head input; before prediction subheads",
            }
            self.base_before = fingerprint_base_state(base)
            require(not any(self.base_before["training_modes"].values()), "base not in eval mode")
            write_new(out / "base_before.json", self.base_before)
            self.had_instance_topk = "get_topk_index" in head.__dict__
            self.old_instance_topk = head.__dict__.get("get_topk_index")
            original_topk = head.get_topk_index

            def observe_topk(scores, max_det):
                result = original_topk(scores, max_det)
                require(self.trace_topk is None, "multiple/unconsumed head selection calls")
                self.trace_topk = tuple(t.detach().clone() for t in result)
                return result

            head.get_topk_index = observe_topk
            self.trace_head = head

        def postprocess(self, preds):
            require(isinstance(preds, (list, tuple)) and isinstance(preds[0], tuple)
                    and isinstance(preds[1], dict), "native PyTorch output structure changed")
            final, proto = preds[0]
            raw = preds[1]["one2one"]
            require(torch.is_tensor(proto) and "feats" in raw, "missing native P/F")
            require(self.trace_topk is not None, "actual head selection was not observed")
            scores, labels, indices = self.trace_topk
            self.trace_topk = None
            require(torch.equal(final[..., 4:5], scores) and torch.equal(final[..., 5:6], labels),
                    "top-k selection does not match inference output")
            require(torch.equal(raw["proto"], proto), "prototype source mismatch")
            gathered = raw["mask_coefficient"].transpose(1, 2).gather(
                1, indices.expand(-1, -1, self.trace_head.nm))
            require(torch.equal(gathered.to(final), final[..., 6:]), "raw coefficient identity mismatch")
            native = super().postprocess(preds)
            self.pending = []
            for i, pred in enumerate(native):
                # This is the pinned end2end branch's exact post-head filter, not a re-ranked Top-K.
                keep = (final[i, :, 4] > self.args.conf).nonzero(as_tuple=False).view(-1)[:self.args.max_det]
                selected = final[i, keep]
                require(torch.equal(selected[:, :4], pred["bboxes"])
                        and torch.equal(selected[:, 4], pred["conf"])
                        and torch.equal(selected[:, 5], pred["cls"]), "postprocess row identity mismatch")
                shapes = [list(f.shape[1:]) for f in raw["feats"]]
                raw_ids = indices[i, keep, 0].tolist()
                locations = [raw_location(r, shapes) for r in raw_ids]
                features = [raw["feats"][loc["level"]][i, :, loc["row"], loc["column"]].detach().cpu().clone()
                            for loc in locations]
                cache = {
                    "proto": proto[i].detach().cpu().clone(),
                    "c0": selected[:, 6:].detach().cpu().clone(),
                    "boxes_input": selected[:, :4].detach().cpu().clone(),
                    "scores": selected[:, 4].detach().cpu().clone(),
                    "classes": selected[:, 5].detach().cpu().clone(),
                    "features": features, "raw_ids": raw_ids, "raw_locations": locations,
                    "head_feature_shapes": shapes, "feature_sources": self.feature_sources,
                    "native_decode_hw": [4 * s for s in proto.shape[2:]],
                    "device": str(proto.device), "output_ranks": list(range(len(selected))),
                }
                # Check original decoder on the same device before any disk replay.
                again = ops.process_mask_native(proto[i], selected[:, 6:], selected[:, :4],
                                                shape=cache["native_decode_hw"])
                require(torch.equal(again, pred["masks"]), "input-space native replay mismatch")
                self.pending.append(cache)
            return native

        def update_metrics(self, preds, batch):
            start = len(self.jdict)
            super().update_metrics(preds, batch)
            cursor = start
            require(len(self.pending) == len(preds), "trace batch size mismatch")
            for i, (cache, pred) in enumerate(zip(self.pending, preds)):
                image_path = Path(batch["im_file"][i])
                image_id = int(image_path.stem)
                require(image_id not in self.actual_ids, "duplicate image")
                self.actual_ids.append(image_id)
                cache.update(image_id=image_id, input_hw=list(batch["img"].shape[2:]),
                             original_hw=[int(x) for x in batch["ori_shape"][i]],
                             ratio_pad=tuple(tuple(float(x) for x in pair) for pair in batch["ratio_pad"][i]),
                             batch_index=self.batch_i, batch_position=i,
                             image_sha256=sha256(image_path),
                             input_tensor_sha256=hashlib.sha256(
                                 batch["img"][i].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                require(cache["input_hw"] == cache["native_decode_hw"], "unexpected prototype/input scale")
                masks = decode_native(cache)
                n = len(cache["c0"])
                indices = list(range(cursor, cursor + n))
                for rank, index in enumerate(indices):
                    row = self.jdict[index]
                    require(row["image_id"] == image_id, "native serializer reordered images")
                    require(encode_rle(masks[rank]) == row["segmentation"], "original-size RLE mismatch")
                cache["prediction_indices"] = indices
                cache_path = out / "cache" / f"{image_id}.pt"
                # Serialize first to a bounded memory stream, then reject before writing over budget.
                import io
                stream = io.BytesIO()
                torch.save(cache, stream)
                size = stream.tell()
                require(self.cache_bytes + size <= cache_limit, "cache resource budget exceeded")
                with cache_path.open("xb") as f:
                    f.write(stream.getbuffer())
                self.cache_bytes += size
                cache_sha = sha256(cache_path)
                self.image_records.append({
                    "image_id": image_id, "cache": f"cache/{image_id}.pt", "cache_sha256": cache_sha,
                    "prediction_indices": indices, "input_hw": cache["input_hw"],
                    "original_hw": cache["original_hw"], "ratio_pad": cache["ratio_pad"],
                    "batch_index": self.batch_i, "batch_position": i,
                    "image_sha256": cache["image_sha256"],
                    "input_tensor_sha256": cache["input_tensor_sha256"],
                    "head_feature_shapes": cache["head_feature_shapes"],
                    "feature_sources": cache["feature_sources"],
                })
                for rank, index in enumerate(indices):
                    self.records.append({
                        "image_id": image_id, "prediction_index": index, "output_rank": rank,
                        "raw_id": cache["raw_ids"][rank], **cache["raw_locations"][rank],
                        "cache": f"cache/{image_id}.pt", "cache_sha256": cache_sha,
                    })
                cursor += n
            require(cursor == len(self.jdict), "untraced serialized predictions")
            self.pending = []

    return TraceValidator


def trace(a):
    # Explicit local files only. No automatic weight or dataset download.
    require(Path(a.weights).is_file() and sha256(a.weights) == MODEL_SHA, "wrong/missing local model")
    require(Path(a.annotations).is_file() and sha256(a.annotations) == GT_SHA, "wrong/missing COCO annotations")
    profile = read_json(a.profile) if getattr(a, "profile", None) else None
    if profile:
        apply_runtime_policy(profile["runtime_policy"])
        for package, version in profile["packages"].items():
            require(importlib.metadata.version(package) == version, f"dependency drift: {package}")
    native_environment()
    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import faster_coco_eval  # verify before native requirement checker
    from pycocotools import mask  # verify before inference
    args = read_json(a.args)
    if profile:
        for key, value in profile["native_args"].items():
            require(args.get(key) == value, f"request differs from frozen profile: {key}")
    required = {"data", "imgsz", "batch", "conf", "iou", "max_det", "rect",
                "device", "quantize", "workers", "end2end", "agnostic_nms",
                "single_cls", "augment", "compile", "classes"}
    require(required <= set(args), "native args must explicitly freeze all geometry/selection fields")
    require(not ({"model", "project", "name", "exist_ok", "validator"} & set(args)), "reserved native arguments")
    require(args["end2end"] is True and args["agnostic_nms"] is False
            and args["single_cls"] is False and args["classes"] is None
            and not args["augment"] and not args["compile"], "unsupported inference policy")
    require(args.get("split", "val") == "val" and args.get("save_json", True) is True,
            "native val/save_json required")
    require(Path(args["data"]).is_file(), "local dataset YAML required")
    check_det_dataset(args["data"], autodownload=False)
    ids = read_json(a.image_ids)
    require(ids and all(type(i) is int for i in ids) and len(set(ids)) == len(ids), "invalid image manifest")
    require(len(ids) == 5000 if a.scope == "full" else len(ids) <= 16, "wrong scope/image budget")
    annotation_ids = {r["id"] for r in read_json(a.annotations)["images"]}
    require(set(ids) <= annotation_ids, "image not in pinned COCO annotations")
    require(a.scope != "full" or set(ids) == annotation_ids, "full trace must cover every COCO val image")
    require(a.max_cache_bytes > 0, "explicit positive cache budget required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    validator = make_trace_validator(out, ids, a.scope, a.max_cache_bytes, GT_SHA,
                                     dict(args), sha256(a.image_ids), profile["runtime_policy"] if profile else None)
    args.update(split="val", save_json=True, project=str(out.resolve()), name="native", exist_ok=False)
    model = YOLO(str(Path(a.weights).resolve()))
    from ultralytics.utils import DEFAULT_CFG_DICT
    write_new(out / "CONFIG_SOURCES.json", {
        "framework_default": DEFAULT_CFG_DICT,
        "checkpoint_train_args": model.ckpt.get("train_args"),
        "checkpoint_version": model.ckpt.get("version"),
        "checkpoint_date": model.ckpt.get("date"),
        "model_overrides": model.overrides, "requested_native_args": args,
        "profile_sha256": sha256(a.profile) if profile else None})
    with torch.inference_mode():
        model.val(validator=validator, **args)


def replay(a):
    import torch
    manifest_path = Path(a.manifest)
    manifest = read_json(manifest_path)
    if all(k in manifest["environment"] for k in RUNTIME_KEYS):
        apply_runtime_policy({k: manifest["environment"][k] for k in RUNTIME_KEYS})
    environment = native_environment()
    # Same Torch, source and numerical settings; other devices are new checks, not silent substitutions.
    for k in ("ultralytics_version", "torch_version", "source_sha256", "cuda_version",
              "matmul_precision", "cuda_matmul_allow_tf32"):
        require(environment[k] == manifest["environment"][k], f"replay runtime changed: {k}")
    for k in RUNTIME_KEYS:
        if k in manifest["environment"]:
            require(environment[k] == manifest["environment"][k], f"replay runtime changed: {k}")
    require(sha256(a.predictions) == manifest["predictions_sha256"], "baseline predictions changed")
    require(sha256(__file__) == manifest["bridge_code_sha256"], "bridge code changed")
    require(sha256(Path(__file__).with_name("native_source_pins.json")) == manifest["source_pins_sha256"],
            "source pins changed")
    require(Path(a.out).resolve() != Path(a.report).resolve(), "output and report must differ")
    baseline = read_json(a.predictions)
    output = copy.deepcopy(baseline)
    seen = set()
    replacement = None
    if a.coefficients:
        require(a.coefficients_sha256 and sha256(a.coefficients) == a.coefficients_sha256,
                "replacement coefficient artifact SHA required")
        replacement = torch.load(a.coefficients, map_location="cpu", weights_only=True)
        require(set(replacement) == {str(r["image_id"]) for r in manifest["images"]},
                "replacement must explicitly cover every cached image")
    root = manifest_path.parent.resolve()
    with torch.inference_mode():
        for record in manifest["images"]:
            path = (root / record["cache"]).resolve()
            require(path.is_relative_to(root) and sha256(path) == record["cache_sha256"], "cache changed/escaped")
            cache = torch.load(path, map_location="cpu", weights_only=True)
            require(cache["image_id"] == record["image_id"]
                    and cache["prediction_indices"] == record["prediction_indices"], "cache identity changed")
            # Require actual baseline zero-delta replay before accepting replacement c'.
            zero_masks = decode_native(cache)
            for rank, index in enumerate(cache["prediction_indices"]):
                require(encode_rle(zero_masks[rank]) == baseline[index]["segmentation"], "zero-delta replay mismatch")
                require(baseline[index]["image_id"] == cache["image_id"], "prediction/cache image mismatch")
            masks = zero_masks if replacement is None else decode_native(cache, replacement[str(cache["image_id"])])
            for rank, index in enumerate(cache["prediction_indices"]):
                require(index not in seen, "duplicate output index")
                seen.add(index)
                output[index]["segmentation"] = encode_rle(masks[rank])
    require(sorted(seen) == list(range(len(baseline))), "incomplete output coverage")
    compare_outputs(baseline, output, zero_delta=replacement is None)
    require(not Path(a.out).exists() and not Path(a.report).exists(), "output/report already exists")
    write_new(a.out, output)
    write_new(a.report, {
        "schema_version": 1, "manifest_sha256": sha256(manifest_path),
        "baseline_sha256": sha256(a.predictions), "output_sha256": sha256(a.out),
        "native_zero_delta_passed": True, "output_identity_passed": True,
        "mode": "zero_delta" if replacement is None else "replacement_coefficients",
        "coefficients_sha256": a.coefficients_sha256 if replacement is not None else None,
        "outputs": len(output), "images": len(manifest["images"]),
        "scientific_status": "not_adjudicated",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("trace")
    for name in ("weights", "annotations", "args", "image-ids", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--scope", choices=("smoke", "full"), required=True)
    p.add_argument("--max-cache-bytes", type=int, required=True)
    p.add_argument("--profile", help="frozen research-line package/argument/numerical policy")
    p = sub.add_parser("replay")
    for name in ("manifest", "predictions", "out", "report"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--coefficients", help="torch dict: image_id string -> full [N,K] c-prime, same dtype")
    p.add_argument("--coefficients-sha256")
    a = parser.parse_args()
    if a.command == "trace":
        trace(a)
    else:
        replay(a)


if __name__ == "__main__":
    main()
