"""Independent, parity-gated deployment cost: RGB -> all original masks on CPU.

One fresh, serial child process per arm/repeat. No GT, training, AP or tuning.
The frozen seven-arm producer is a reference, never a shared execution path.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from frozen_io import FrozenYOLO, OFFICIAL_SHA256, dump_json, predicted_neighbors, sha256, state_digest
from native_mask_adapter import MULTI_SHA256, RESPONSE_SHA256, Timer, export_binary, native_proto_logits

VERSION = "frozen_individual_cost_v1"
ARMS = ("baseline", "RCMC_first64", "multi_local_first64", "global_minus025_first64", "TriFlow_final8")
LIST_SHA = "b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db"
PROTOCOL_SHA = "2211ae66a47560c81f658791e1a02680dbcccba5326c9288392c943be96277b0"
CORE_SHA = "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"
RUNTIME_SHA = "c24bc853ff17c7f4a0be3bc847a8b6733283b2aa06bc16388ab1b84fca3380b3"
HEAD_SHA = "5c47c39dd5912999af8d8a11e0052a9bcb291c55413eb63b3c4bdc3c054707c8"
HEAD_STATE_SHA = "9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8"
SOURCES = ("benchmark_individual_cost.py", "frozen_io.py", "native_mask_adapter.py", "portable_risk.py",
           "risk_calibration.py", "local_features.py", "mask_calibration.py")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def memory(torch):
    import psutil
    info = psutil.Process().memory_info()
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cpu_current_rss_bytes": info.rss,
            "cpu_process_cumulative_peak_bytes": getattr(info, "peak_wset", None)}


class IndividualExecutor:
    """No calibration gate/head from another arm is initialized or executed."""

    def __init__(self, args):
        import numpy as np
        import torch
        self.np, self.torch, self.arm, self.device = np, torch, args.arm, args.device
        self.base = FrozenYOLO(args.weights, args.vendor, args.device)
        # The three hidden-query hooks are needed only by TriFlow.
        self.base.close()
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        self.ops = ops
        self.validator = SegmentationValidator(args={"conf": .001, "max_det": 300, "save_json": True,
                                                   "plots": False, "imgsz": 640, "half": False, "rect": False})
        self.validator.nc, self.validator.end2end = 80, True
        self.validator.process = ops.process_mask_native
        self.loaded_assets, self.head, self.hidden, self.hooks = {}, None, {}, []
        if self.arm == "RCMC_first64":
            from portable_risk import PortableRisk
            self.gate = PortableRisk(args.response_model)
            if sha256(args.response_model) != RESPONSE_SHA256 or self.gate.features != 5:
                raise ValueError("Frozen response asset contract differs")
            self.loaded_assets["response"] = RESPONSE_SHA256
        elif self.arm == "multi_local_first64":
            from portable_risk import PortableRisk
            self.gate = PortableRisk(args.multi_model)
            if sha256(args.multi_model) != MULTI_SHA256 or self.gate.features != 18:
                raise ValueError("Frozen multi asset contract differs")
            self.loaded_assets["multi_local"] = MULTI_SHA256
        elif self.arm == "TriFlow_final8":
            self._load_triflow(args)
        elif self.arm not in ("baseline", "global_minus025_first64"):
            raise ValueError("Unsupported isolated arm")

    def _load_triflow(self, args):
        torch = self.torch
        origin = Path(args.triflow_reference)
        metadata = load_json(origin / "HEAD_PROVENANCE.json")
        core_path = origin / "source" / "triflow_model.py"
        runtime_path = origin / "diagnostics_provenance" / "diagnostics_runtime.py"
        head_path = origin / "head_epoch_08.pt"
        if (sha256(core_path), sha256(runtime_path), sha256(head_path)) != (CORE_SHA, RUNTIME_SHA, HEAD_SHA):
            raise ValueError("Actual final8/core/runtime source bytes differ")
        if metadata.get("evaluation_epoch") != 8 or metadata.get("loaded_state_sha256") != HEAD_STATE_SHA:
            raise ValueError("Actual final8 metadata identity differs")
        core = import_file("triflow_model", core_path)
        runtime = import_file("cost_triflow_runtime", runtime_path)
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        contract, state = payload["contract"], payload["state"]
        if (payload.get("kind") != "triflow_20k_training_snapshot" or payload.get("reason") != "epoch_boundary"
                or state.get("cursor") != {"epoch": 9, "image_position": 0, "chunk_start": 0}
                or len(state.get("epochs", [])) != 8 or contract.get("epochs") != 8
                or contract.get("effective_images") != 20000 or contract.get("smoke_only") is not False
                or contract.get("complete_declared_train_subset") is not True
                or contract.get("engineering_subset") is not False
                or payload.get("head_state_sha256") != HEAD_STATE_SHA
                or payload.get("frozen_integrity", {}).get("passed") is not True):
            raise ValueError("Actual locked final8 snapshot is not the completed formal20k boundary")
        configuration = contract["configuration"]
        kwargs = {field.name: configuration[field.name] for field in fields(core.TriFlowConfig)}
        self.head = runtime.install(core.TriFlowModel, "minimal")(
            configuration["feature_channels"], configuration["instance_hidden_channels"], core.TriFlowConfig(**kwargs)).float()
        self.head.load_state_dict(payload["head_state_dict"], strict=True)
        if state_digest(self.head) != HEAD_STATE_SHA:
            raise ValueError("Loaded actual final8 head differs")
        self.head = self.head.eval().requires_grad_(False).to(self.device)
        self.head_initial_sha = HEAD_STATE_SHA
        self.loaded_assets = {"head": HEAD_SHA, "head_state": HEAD_STATE_SHA,
                              "triflow_model.py": CORE_SHA, "diagnostics_runtime.py": RUNTIME_SHA,
                              "diagnostic_scope": "minimal"}
        for level, branch in enumerate(self.base.head.one2one_cv4):
            def capture(module, inputs, level=level):
                self.hidden[level] = inputs[0].detach()
            self.hooks.append(branch[-1].register_forward_pre_hook(capture))
        del payload, contract, state

    def _refine(self, native, proto, raw):
        """Same final8 executable-row grouping/einsum/scatter, no audit tolist."""
        torch = self.torch
        one = raw["one2one"]
        _, _, indices = self.base.head.get_topk_index(one["scores"].sigmoid().transpose(1, 2), 300)
        hidden = torch.cat([self.hidden[level].flatten(2) for level in range(self.base.head.nl)], 2).transpose(1, 2)
        h = hidden.gather(1, indices.expand(-1, -1, hidden.shape[-1]))[0].cpu()
        c0, boxes = native[0, :, 6:].cpu(), native[0, :, :4].cpu()
        scores, raw_indices = native[0, :, 4].cpu(), indices[0, :, 0].cpu()
        graph_rows, graph_valid = predicted_neighbors(boxes.numpy(), raw_indices.numpy(), scores.numpy(), k=2)
        eligible = (scores > .001).nonzero().flatten().tolist()
        height, width = proto.shape[-2:]
        boxes_proto = boxes * torch.tensor([width / 640, height / 640, width / 640, height / 640])
        executable = []
        for row in eligible[:64]:
            box = boxes_proto[row]
            if bool(((torch.arange(width) >= box[0]) & (torch.arange(width) < box[2])).any()
                    and ((torch.arange(height) >= box[1]) & (torch.arange(height) < box[3])).any()):
                executable.append(row)
        refined = c0.clone()
        p, feature = proto[0], one["feats"][0][0]
        for start in range(0, len(executable), 4):
            rows = executable[start:start + 4]
            targets = torch.tensor(rows, dtype=torch.long)
            neighbor_indices = torch.from_numpy(graph_rows[rows])
            valid = torch.from_numpy(graph_valid[rows]).to(self.device)
            neighbor_c = c0[neighbor_indices.clamp_min(0)].to(self.device)
            neighbor_logits = torch.einsum("chw,nkc->nkhw", p, neighbor_c).masked_fill(~valid[:, :, None, None], 0)
            result = self.head(p, c0[targets].to(self.device), feature, boxes_proto[targets].to(self.device),
                               neighbor_logits, valid, h[targets].to(self.device))
            if not torch.isfinite(result["coefficients_refined"]).all():
                raise FloatingPointError("Nonfinite true final8 coefficient")
            refined[targets] = result["coefficients_refined"].cpu()
        updated = torch.cat((native[0, :, :6], refined.to(self.device)), 1)[None]
        return updated, {"selected_first64": min(64, len(eligible)), "supported_first64": len(executable),
                         "method_target_rows": len(executable), "action_rows": 0,
                         "five_feature_rows": 0, "eighteen_feature_rows": 0, "gate_rows": 0,
                         "diagnostic_scope": "minimal", "coefficient_state_sha256": None}

    def execute(self, rgb):
        """Measured endpoint: prepared RGB bytes through final binary CPU export."""
        torch, np, ops = self.torch, self.np, self.ops
        params = self.base.letterbox.get_params({"img": rgb})
        resized = self.base.letterbox(image=rgb)
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))[None].to(self.device).float() / 255
        self.hidden.clear()
        (native, proto), raw = self.base.model(tensor)
        if isinstance(proto, tuple):
            proto = proto[0]
        method = {"selected_first64": 0, "supported_first64": 0, "method_target_rows": 0,
                  "action_rows": 0, "five_feature_rows": 0, "eighteen_feature_rows": 0, "gate_rows": 0}
        if self.arm == "TriFlow_final8":
            native, method = self._refine(native, proto, raw)
        pbatch = {"imgsz": (640, 640), "ori_shape": tuple(rgb.shape[:2]),
                  "ratio_pad": (params["ratio"], (params["left"], params["top"])), "im_file": "preloaded_rgb"}
        capture = {}

        def independent_decoder(prototypes, coefficients, boxes, shape):
            count = len(coefficients)
            height, width = prototypes.shape[-2:]
            # CPU FP32 multiplication matches the original first64 ROI predicate.
            proto_boxes = boxes.cpu() * torch.tensor([width / 640, height / 640, width / 640, height / 640])
            positions = []
            for row in range(min(count, 64)):
                box = proto_boxes[row]
                if bool(((torch.arange(width) >= box[0]) & (torch.arange(width) < box[2])).any()
                        and ((torch.arange(height) >= box[1]) & (torch.arange(height) < box[3])).any()):
                    positions.append(row)
            method.update(selected_first64=min(count, 64), supported_first64=len(positions), method_target_rows=len(positions))
            capture["positions"] = torch.tensor(positions, device=self.device, dtype=torch.long)
            capture["boxes"] = boxes[capture["positions"]]
            if count == 0:
                capture["logits"] = torch.empty((0, *shape), device=self.device)
                return torch.zeros((0, *shape), dtype=torch.uint8, device=self.device)
            # Full native M GEMM and native upsample chunks preserve FP32 batch rounding.
            all_logits = native_proto_logits(prototypes, coefficients)
            step = max(1, 32_000_000 // (shape[0] * shape[1]))
            masks = []
            for start in range(0, count, step):
                end = min(count, start + step)
                logits = ops.scale_masks(all_logits[start:end][None], shape)[0]
                local = capture["positions"][(capture["positions"] >= start) & (capture["positions"] < end)] - start
                if len(local):
                    capture["logits"] = logits[local].clone()
                masks.append(logits.gt_(0).byte())
            if "logits" not in capture:
                capture["logits"] = torch.empty((0, *shape), device=self.device)
            return ops.crop_mask(torch.cat(masks), boxes)

        calibration = self.arm in ("RCMC_first64", "multi_local_first64", "global_minus025_first64")
        self.validator.process = independent_decoder if calibration else ops.process_mask_native
        pred = self.validator.postprocess(((native, proto), None))[0]
        raw_boxes = pred["bboxes"].detach().cpu().numpy().copy()
        raw_scores = pred["conf"].detach().cpu().numpy().copy()
        classes = pred["cls"].long().detach().cpu().numpy().copy()
        scaled = self.validator.scale_preds(pred, pbatch)
        original_masks = scaled["masks"]
        if calibration and len(capture["positions"]):
            positions, boxes, logits = capture["positions"], capture["boxes"], capture["logits"]
            base640, base_original = pred["masks"][positions], original_masks[positions]
            area = base_original.sum((1, 2)).float()
            if self.arm == "global_minus025_first64":
                trial = ops.crop_mask((logits > .25).byte(), boxes)
                original_masks[positions] = export_binary(trial, pbatch)
                method["action_rows"] = len(positions)
            elif self.arm == "RCMC_first64":
                from risk_calibration import prediction_features
                tau = .75 / (1 + (area / 2304).square())
                trial = ops.crop_mask((logits > tau[:, None, None]).byte(), boxes)
                trial_original = export_binary(trial, pbatch)
                five = prediction_features(base640, base_original, trial_original, "response")
                gain = self.gate.predict(five)
                if not np.isfinite(gain).all():
                    raise FloatingPointError("Response nonfinite gain")
                use = (gain > 0) & trial_original.flatten(1).any(1).bool().cpu().numpy()
                original_masks[positions] = torch.where(torch.as_tensor(use, device=self.device)[:, None, None], trial_original, base_original)
                method.update(action_rows=len(positions), five_feature_rows=len(positions), gate_rows=len(positions))
            else:
                from risk_calibration import prediction_features
                from local_features import local_features
                features, exported, availability = [], [], []
                for index in range(5):
                    tau = .75 / (1 + (area / 2304).square()) if index == 0 else torch.full_like(area, (.25, .5, .75, 1)[index - 1])
                    trial = ops.crop_mask((logits > tau[:, None, None]).byte(), boxes)
                    trial_original = export_binary(trial, pbatch)
                    five = prediction_features(base640, base_original, trial_original, "response")
                    local = local_features(logits, base640, trial)
                    features.append(np.concatenate((five, tau.cpu().numpy().astype(np.float64)[:, None], local), 1))
                    exported.append(trial_original)
                    availability.append(trial_original.flatten(1).any(1).bool().cpu().numpy())
                eighteen = np.stack(features, 1)
                scores = self.gate.predict(eighteen.reshape(-1, 18)).reshape(len(positions), 5)
                if not np.isfinite(scores).all():
                    raise FloatingPointError("Multi nonfinite gain")
                masked = np.where(np.stack(availability, 1), scores, -np.inf)
                choices = np.where(masked.max(1) > 0, masked.argmax(1), -1)
                result = base_original.clone()
                for index in range(5):
                    use = torch.as_tensor(choices == index, device=self.device)
                    result[use] = exported[index][use]
                original_masks[positions] = result
                method.update(action_rows=5 * len(positions), five_feature_rows=5 * len(positions),
                              eighteen_feature_rows=5 * len(positions), gate_rows=5 * len(positions))
        result = {"masks": original_masks.detach().cpu().numpy(), "raw_boxes": raw_boxes,
                  "scores": raw_scores, "classes": classes, "boxes": scaled["bboxes"].detach().cpu().numpy(),
                  "method": method, "native_rows": len(raw_scores), "input_tensor_for_untimed_audit": tensor}
        self.hidden.clear()
        self.base.extracted_images += 1
        return result

    def integrity(self):
        frozen = self.base.verify_frozen()
        if self.head is not None and state_digest(self.head) != self.head_initial_sha:
            raise RuntimeError("True final8 head changed during cost panel")
        return frozen

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.base.close()


def records_and_verify(result, iid, reference, categories, mask_utils):
    import numpy as np
    began = time.perf_counter()
    rows = []
    for index, mask in enumerate(result["masks"]):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        box, score, cls = result["boxes"][index], result["scores"][index], result["classes"][index]
        rows.append({"image_id": iid, "category_id": int(categories[int(cls)]),
                     "bbox": [round(float(value), 3) for value in (box[0], box[1], box[2] - box[0], box[3] - box[1])],
                     "score": round(float(score), 5), "segmentation": rle, "detection_index": index,
                     "raw_input_box_xyxy": result["raw_boxes"][index].tolist(), "raw_confidence": float(score),
                     "box_xyxy": box.tolist(), "model_class": int(cls)})
    rle_seconds = time.perf_counter() - began
    began = time.perf_counter()
    if rows != reference.get("detections") or reference.get("image_id") != iid:
        raise RuntimeError(f"Isolated deployment output identity/RLE parity failed image {iid}")
    return rows, rle_seconds, time.perf_counter() - began


def check_isolation(arm, result, loaded_assets):
    n = result["method"]["method_target_rows"]
    method = result["method"]
    expected = {"baseline": (0, 0, 0, 0), "global_minus025_first64": (n, 0, 0, 0),
                "RCMC_first64": (n, n, 0, n), "multi_local_first64": (5 * n, 5 * n, 5 * n, 5 * n),
                "TriFlow_final8": (0, 0, 0, 0)}[arm]
    actual = tuple(method[key] for key in ("action_rows", "five_feature_rows", "eighteen_feature_rows", "gate_rows"))
    required = {"baseline": set(), "global_minus025_first64": set(), "RCMC_first64": {"response"},
                "multi_local_first64": {"multi_local"}, "TriFlow_final8": {"head", "head_state", "triflow_model.py", "diagnostics_runtime.py", "diagnostic_scope"}}[arm]
    if actual != expected or set(loaded_assets) != required or n > 64:
        raise ValueError("Isolated method work/asset contract differs")
    return True


def child(args):
    began = time.perf_counter()
    import cv2
    import numpy as np
    import torch
    dependency_seconds = time.perf_counter() - began
    if torch.__version__ != "2.5.1" or not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Cost protocol requires the registered laptop Torch2.5.1 CUDA environment")
    out = Path(args.child_output)
    if (out / "CHILD_COMPLETE.json").exists() or (out / "COST_SAMPLES.jsonl").exists():
        raise FileExistsError("Child history exists; use a new cost Run")
    out.mkdir(parents=True, exist_ok=True)
    started = now()
    inputs = load_json(Path(args.root) / "runs" / args.run_id / "COST_INPUTS.json")
    count = 4 if args.engineering else 32
    paths = [Path(value) for value in inputs["image_paths"][:count]]
    read_start = time.perf_counter()
    rgbs = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError("Cannot read registered panel image")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    read_seconds = time.perf_counter() - read_start
    raw_pixel_records = [{"image_id": int(path.stem), "rgb_shape": list(rgb.shape),
                          "rgb_bytes_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                          "channel_contract": "cv2 BGR decoded once outside endpoint, BGR2RGB; RGB LetterBox/transposed CHW inside endpoint"}
                         for path, rgb in zip(paths, rgbs)]
    dump_json(out / "RAW_RGB_INPUTS.json", {"images": raw_pixel_records})
    expected_inputs = {}
    wanted_ids = {int(path.stem) for path in paths}
    with (Path(args.reference_run) / "BASELINE_PARITY_IMAGES.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value["image_id"] in wanted_ids:
                expected_inputs[value["image_id"]] = value["input_sha256"]
            if len(expected_inputs) == len(paths):
                break
    if set(expected_inputs) != wanted_ids:
        raise ValueError("Actual source native input fingerprints are incomplete")
    initialize_start = time.perf_counter()
    executor = IndividualExecutor(args)
    from pycocotools import mask as mask_utils
    from ultralytics.data.converter import coco80_to_coco91_class
    categories = coco80_to_coco91_class()
    initialization_seconds = time.perf_counter() - initialize_start
    reference_root = Path(args.triflow_reference) / "triflow" if args.arm == "TriFlow_final8" else Path(args.reference_run) / args.arm
    samples, warmup_samples = [], []
    try:
        with torch.inference_mode():
            warmup_start = time.perf_counter()
            for path, rgb in zip(paths[:4], rgbs[:4]):
                with Timer(args.device) as timer:
                    result = executor.execute(rgb)
                check_isolation(args.arm, result, executor.loaded_assets)
                warmup_samples.append({"image_id": int(path.stem), "seconds": timer.seconds})
                del result
            warmup_seconds = time.perf_counter() - warmup_start
            postwarm = memory(torch)
            for path, rgb in zip(paths, rgbs):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = memory(torch)
                with Timer(args.device) as timer:
                    result = executor.execute(rgb)
                endpoint_seconds = timer.seconds
                after = memory(torch)
                iid = int(path.stem)
                check_isolation(args.arm, result, executor.loaded_assets)
                reference_path = reference_root / "images" / f"{iid:012d}.json"
                audit_start = time.perf_counter()
                reference = load_json(reference_path)
                read_reference_seconds = time.perf_counter() - audit_start
                rows, rle_seconds, parity_seconds = records_and_verify(result, iid, reference, categories, mask_utils)
                input_start = time.perf_counter()
                input_sha = hashlib.sha256(result["input_tensor_for_untimed_audit"].cpu().contiguous().numpy().tobytes()).hexdigest()
                if input_sha != expected_inputs[iid]:
                    raise RuntimeError("Measured RGB preprocess differs from actual original native tensor bytes")
                input_audit_seconds = time.perf_counter() - input_start
                sample = {"arm": args.arm, "repeat_index": args.repeat_index, "image_id": iid,
                          "deployment_seconds": endpoint_seconds, "native_rows": result["native_rows"],
                          "method_work": result["method"], "isolated_method_and_assets": True,
                          "all_rows_identity_and_rle_exact": True,
                          "native_input_tensor_sha256": input_sha, "native_input_tensor_exact": True,
                          "reference_file": str(reference_path), "reference_sha256": sha256(reference_path),
                          "output_mask_binary_sha256": hashlib.sha256(result["masks"].tobytes()).hexdigest(),
                          "reference_read_seconds": read_reference_seconds, "rle_records_encoding_seconds": rle_seconds,
                          "identity_rle_parity_seconds": parity_seconds, "input_tensor_audit_seconds": input_audit_seconds,
                          "memory_after_endpoint": after, "cpu_rss_before_endpoint_bytes": before["cpu_current_rss_bytes"],
                          "cpu_rss_after_endpoint_bytes": after["cpu_current_rss_bytes"],
                          "cuda_peak_scope": "reset immediately before this deployment endpoint; includes resident own arm model/assets",
                          "cpu_peak_scope": "entire fresh arm-repeat process, including initialization/warmup/prior audit; not reset or deployment-only"}
                write_start = time.perf_counter()
                dump_json(out / "outputs" / f"{iid:012d}.json", {"image_id": iid, "detections": rows})
                sample["output_write_seconds"] = time.perf_counter() - write_start
                with (out / "COST_SAMPLES.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n")
                samples.append(sample)
                del result, reference, rows
        frozen = executor.integrity()
        receipt = {"version": VERSION, "status": "completed", "passed": True, "arm": args.arm,
                   "repeat_index": args.repeat_index, "image_count": len(paths), "engineering": args.engineering,
                   "all_output_parity_passed": True, "all_isolation_passed": True, "loaded_assets": executor.loaded_assets,
                   "pid": os.getpid(), "interpreter": sys.executable, "started_at": started, "finished_at": now(),
                   "dependency_import_seconds": dependency_seconds, "image_read_and_rgb_prepare_seconds": read_seconds,
                   "weight_and_own_asset_initialization_seconds": initialization_seconds,
                   "warmup_seconds": warmup_seconds, "warmup_samples": warmup_samples,
                   "postwarmup_rss_bytes": postwarm["cpu_current_rss_bytes"],
                   "process_final_memory": memory(torch), "process_cpu_peak_scope": "fresh process entire lifetime including init/warmup/audit",
                   "frozen_integrity": frozen, "source_binding_sha256": inputs["source_binding_sha256"],
                   "samples_sha256": sha256(out / "COST_SAMPLES.jsonl"), "total_child_seconds": time.perf_counter() - began,
                   "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
                               "gpu": torch.cuda.get_device_name(), "tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_tf32": torch.backends.cudnn.allow_tf32, "threads": torch.get_num_threads()},
                   "head_unchanged": True if args.arm == "TriFlow_final8" else None,
                   "scientific_scale_claimed": False, "gt_used": False, "ap_measured": False}
        dump_json(out / "CHILD_COMPLETE.json", receipt)
        return 0
    except Exception as exc:
        dump_json(out / "CHILD_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now(), "passed": False})
        raise
    finally:
        executor.close()


def archive_inputs(args, run):
    if sha256(args.protocol) != PROTOCOL_SHA or sha256(args.images_list) != LIST_SHA:
        raise ValueError("Locked cost protocol/source list differs")
    all_paths = [Path(line.strip()).resolve() for line in Path(args.images_list).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(all_paths) != 5000 or len(set(all_paths)) != 5000 or not all(path.is_file() for path in all_paths[:32]):
        raise ValueError("Locked original5000 list/panel files incomplete")
    reference = Path(args.reference_run)
    if (load_json(reference / "SUMMARY.json").get("image_count") != 5000
            or load_json(reference / "SUMMARY.json").get("passed") is not True
            or load_json(reference / "BASELINE_PARITY.json").get("passed") is not True):
        raise ValueError("Frozen native reference must be the sealed complete5000 producer")
    rr = load_json(reference / "run.json")
    if rr.get("status") != "completed" or rr.get("return_code") != 0 or rr.get("artifact_completeness") != "complete":
        raise ValueError("Native reference actual runner is not complete")
    tf = Path(args.triflow_reference)
    metadata = load_json(tf / "HEAD_PROVENANCE.json")
    if metadata.get("snapshot_sha256") != HEAD_SHA or metadata.get("loaded_state_sha256") != HEAD_STATE_SHA:
        raise ValueError("Final8 reference head provenance differs")
    sources = {name: sha256(Path(__file__).with_name(name)) for name in SOURCES}
    bindings = {"source_sha256": sources, "protocol_sha256": PROTOCOL_SHA,
                "response_sha256": sha256(args.response_model), "multi_sha256": sha256(args.multi_model),
                "weights_sha256": sha256(args.weights), "images_list_sha256": LIST_SHA,
                "native_reference_summary_sha256": sha256(reference / "SUMMARY.json"),
                "native_reference_parity_sha256": sha256(reference / "BASELINE_PARITY.json"),
                "head_sha256": sha256(tf / "head_epoch_08.pt"),
                "head_provenance_sha256": sha256(tf / "HEAD_PROVENANCE.json"),
                "core_sha256": sha256(tf / "source" / "triflow_model.py"),
                "runtime_sha256": sha256(tf / "diagnostics_provenance" / "diagnostics_runtime.py")}
    if (bindings["weights_sha256"], bindings["response_sha256"], bindings["multi_sha256"], bindings["head_sha256"],
            bindings["core_sha256"], bindings["runtime_sha256"]) != (OFFICIAL_SHA256, RESPONSE_SHA256, MULTI_SHA256, HEAD_SHA, CORE_SHA, RUNTIME_SHA):
        raise ValueError("Locked own assets or final8 source bytes differ")
    vendor_files = ("ultralytics/utils/ops.py", "ultralytics/models/yolo/segment/val.py", "ultralytics/utils/nms.py", "ultralytics/data/augment.py")
    bindings["vendor_source_sha256"] = {name: sha256(Path(args.vendor) / name) for name in vendor_files}
    source = run / "source"
    source.mkdir(exist_ok=True)
    for name in SOURCES:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    shutil.copy2(args.protocol, source / "COST_PANEL_PROTOCOL.md")
    for name, path in (("triflow_model.py", tf / "source" / "triflow_model.py"),
                       ("diagnostics_runtime.py", tf / "diagnostics_provenance" / "diagnostics_runtime.py"),
                       ("HEAD_PROVENANCE.json", tf / "HEAD_PROVENANCE.json")):
        target = source / "triflow_reference" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    for name, path in (("response.json", args.response_model), ("multi_local.json", args.multi_model)):
        target = run / "assets" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    ids = [int(path.stem) for path in all_paths[:32]]
    for iid in ids:
        for arm in ARMS:
            path = tf / "triflow" / "images" / f"{iid:012d}.json" if arm == "TriFlow_final8" else reference / arm / "images" / f"{iid:012d}.json"
            if not path.is_file():
                raise FileNotFoundError("Actual frozen panel output is unavailable: " + str(path))
    inputs = {"version": VERSION, "source_binding": bindings, "source_binding_sha256": canonical_sha(bindings),
              "image_paths": [str(path) for path in all_paths[:32]], "image_ids": ids,
              "image_files": [{"image_id": int(path.stem), "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in all_paths[:32]],
              "arm_names": list(ARMS), "engineering": args.engineering, "measured_images": 4 if args.engineering else 32,
              "warmup_images": 4, "repeat_blocks": 1 if args.engineering else 3,
              "endpoint": "preloaded original RGB ndarray through preprocessing/native forward/method to all original binary masks on CPU",
              "reference_run": str(reference), "triflow_reference": str(tf), "interpreter": sys.executable,
              "deployment_excludes": ["disk read/RGB prep", "model/gate/head loading", "warmup", "RLE/identity audit", "JSON writing", "COCO/GT"],
              "sampling_rule": "original locked5000 list first32, engineering first4; no result-based selection"}
    dump_json(run / "COST_INPUTS.json", inputs)
    return inputs


def describe(values):
    import statistics
    ordered = sorted(values)
    def quantile(p):
        position = (len(ordered) - 1) * p
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "std_population": statistics.pstdev(values), "p10": quantile(.1), "p90": quantile(.9),
            "min": min(values), "max": max(values)}


def main(args):
    run = Path(args.root).resolve() / "runs" / args.run_id
    if (run / "COST_INPUTS.json").exists() or (run / "SUMMARY.json").exists():
        raise FileExistsError("Cost history exists; retry needs a new Run")
    run.mkdir(parents=True, exist_ok=True)
    began, started = time.perf_counter(), now()
    inputs = archive_inputs(args, run)
    if not args.engineering:
        if not args.engineering_receipt:
            raise ValueError("Formal panel requires actual passed engineering receipt")
        engineering = load_json(args.engineering_receipt)
        if (engineering.get("engineering_passed") is not True or engineering.get("source_binding_sha256") != inputs["source_binding_sha256"]
                or engineering.get("image_count_per_arm_repeat") != 4 or engineering.get("arm_names") != list(ARMS)):
            raise ValueError("Actual isolation/parity engineering gate did not pass the same locked inputs/source")
        dump_json(run / "ENGINEERING_GATE.json", {"path": str(Path(args.engineering_receipt).resolve()),
                                                  "sha256": sha256(args.engineering_receipt), "passed": True})
    children, all_samples = [], []
    try:
        repeats = 1 if args.engineering else 3
        for repeat in range(repeats):
            order = ARMS[repeat:] + ARMS[:repeat]
            for position, arm in enumerate(order):
                output = run / "measurements" / f"repeat{repeat}" / arm
                command = [sys.executable, str(Path(__file__).resolve())]
                for name in ("root", "run_id", "weights", "vendor", "images_list", "response_model", "multi_model", "reference_run", "triflow_reference", "protocol", "device"):
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                command += ["--child", "--arm", arm, "--repeat-index", str(repeat), "--child-output", str(output)]
                if args.engineering:
                    command.append("--engineering")
                output.mkdir(parents=True, exist_ok=True)
                child_started, child_start = now(), time.perf_counter()
                with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                    observed = {"arm": arm, "repeat_index": repeat, "order_position": position, "pid": process.pid,
                                "command": command, "started_at": child_started, "source_binding_sha256": inputs["source_binding_sha256"]}
                    dump_json(output / "PROCESS_OBSERVED.json", observed)
                    code = process.wait()
                observed.update(return_code=code, finished_at=now(), process_wall_seconds=time.perf_counter() - child_start)
                dump_json(output / "PROCESS_OBSERVED.json", observed)
                if code != 0:
                    raise RuntimeError(f"Real child {arm} repeat{repeat} failed code {code}; retain partial history")
                receipt = load_json(output / "CHILD_COMPLETE.json")
                if (receipt.get("passed") is not True or receipt.get("all_output_parity_passed") is not True
                        or receipt.get("all_isolation_passed") is not True or receipt.get("source_binding_sha256") != inputs["source_binding_sha256"]):
                    raise RuntimeError("Actual child isolation/parity/source receipt failed")
                samples = [json.loads(line) for line in (output / "COST_SAMPLES.jsonl").read_text().splitlines()]
                expected_ids = inputs["image_ids"][:4 if args.engineering else 32]
                if [item["image_id"] for item in samples] != expected_ids:
                    raise ValueError("Actual measured image order/coverage differs")
                if sha256(output / "COST_SAMPLES.jsonl") != receipt["samples_sha256"]:
                    raise ValueError("Actual child samples bytes differ")
                children.append({**observed, "child_complete_sha256": sha256(output / "CHILD_COMPLETE.json"),
                                 "initialization_seconds": receipt["weight_and_own_asset_initialization_seconds"],
                                 "dependency_import_seconds": receipt["dependency_import_seconds"],
                                 "image_read_seconds": receipt["image_read_and_rgb_prepare_seconds"],
                                 "warmup_seconds": receipt["warmup_seconds"],
                                 "cpu_process_peak_bytes": receipt["process_final_memory"]["cpu_process_cumulative_peak_bytes"]})
                all_samples.extend(samples)
                dump_json(run / "PROGRESS.json", {"status": "running", "completed_child_processes": len(children), "total_child_processes": 5 * repeats})
        if any(sha256(Path(__file__).with_name(name)) != expected for name, expected in inputs["source_binding"]["source_sha256"].items()):
            raise RuntimeError("Cost/runtime source changed while executing")
        binding = inputs["source_binding"]
        actual_after = {"protocol_sha256": sha256(args.protocol), "weights_sha256": sha256(args.weights),
                        "response_sha256": sha256(args.response_model), "multi_sha256": sha256(args.multi_model),
                        "head_sha256": sha256(Path(args.triflow_reference) / "head_epoch_08.pt"),
                        "core_sha256": sha256(Path(args.triflow_reference) / "source" / "triflow_model.py"),
                        "runtime_sha256": sha256(Path(args.triflow_reference) / "diagnostics_provenance" / "diagnostics_runtime.py")}
        if any(actual != binding[key] for key, actual in actual_after.items()) or any(
                sha256(Path(args.vendor) / name) != expected for name, expected in binding["vendor_source_sha256"].items()):
            raise RuntimeError("Cost protocol/assets/vendor changed during execution")
        dump_json(run / "COST_SAMPLES.json", {"samples": all_samples})
        lookup = {(item["repeat_index"], item["image_id"], item["arm"]): item for item in all_samples}
        statistics, increments = {}, {}
        for arm in ARMS:
            arm_samples = [item for item in all_samples if item["arm"] == arm]
            increments[arm] = [{"repeat_index": item["repeat_index"], "image_id": item["image_id"],
                                "signed_deployment_increment_seconds": item["deployment_seconds"] - lookup[(item["repeat_index"], item["image_id"], "baseline")]["deployment_seconds"]} for item in arm_samples]
            statistics[arm] = {"deployment_seconds": describe([item["deployment_seconds"] for item in arm_samples]),
                               "paired_signed_increment_seconds": describe([item["signed_deployment_increment_seconds"] for item in increments[arm]]),
                               "repeat_mean_seconds": [describe([item["deployment_seconds"] for item in arm_samples if item["repeat_index"] == repeat])["mean"] for repeat in range(repeats)],
                               "gpu_peak_allocated_bytes": max(item["memory_after_endpoint"]["cuda_peak_allocated_bytes"] for item in arm_samples),
                               "gpu_peak_reserved_bytes": max(item["memory_after_endpoint"]["cuda_peak_reserved_bytes"] for item in arm_samples),
                               "cpu_fresh_process_peak_bytes_per_repeat": [item["cpu_process_peak_bytes"] for item in children if item["arm"] == arm]}
        dump_json(run / "PAIRED_COST_INCREMENTS.json", increments)
        summary = {"version": VERSION, "status": "completed", "passed": True, "engineering": args.engineering,
                   "engineering_passed": True if args.engineering else None, "arm_names": list(ARMS),
                   "image_count_per_arm_repeat": 4 if args.engineering else 32, "repeat_blocks": repeats,
                   "child_processes": len(children), "sample_count": len(all_samples),
                   "all_output_parity_passed": True, "all_isolation_passed": True,
                   "source_binding_sha256": inputs["source_binding_sha256"], "source_unchanged": True,
                   "statistics": statistics, "children": children, "started_at": started, "completed_at": now(),
                   "total_elapsed_seconds": time.perf_counter() - began, "gt_used": False, "ap_measured": False,
                   "confidence_intervals": None,
                   "limitations": ["fixed first32 engineering cost panel, registered laptop environment and endpoint only",
                                   "three cyclic-left-shift blocks reduce drift, do not fully balance arm positions",
                                   "CPU OSpeak covers fresh process lifetime including initialization/warmup/audit; sampled endpoint RSS is not an exact peak",
                                   "RLE/read-reference/parity/write cost lies outside deployment endpoint; no AP or GT computed",
                                   "TriFlow diagnostic scope minimal; existing full final8 reference output must remain byte exact"]}
        dump_json(run / "SUMMARY.json", summary)
        dump_json(run / "COST_COMPLETE.json", {"status": "completed", "summary_sha256": sha256(run / "SUMMARY.json"),
                                              "all_output_parity_passed": True, "all_isolation_passed": True, "completed_at": now()})
        print(json.dumps({"completed": True, "engineering_passed": summary["engineering_passed"], "sample_count": len(all_samples)}), flush=True)
        return 0
    except Exception as exc:
        dump_json(run / "COST_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now(), "passed": False})
        raise


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "weights", "vendor", "images-list", "response-model", "multi-model", "reference-run", "triflow-reference", "protocol"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--engineering-receipt")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--child-output")
    args = parser.parse_args()
    if args.child and (args.arm is None or args.repeat_index is None or args.child_output is None):
        parser.error("Child mode needs a real arm/repeat/output")
    return args


if __name__ == "__main__":
    arguments = parse()
    raise SystemExit(child(arguments) if arguments.child else main(arguments))
