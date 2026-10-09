"""Frozen native mask actions, preserving every official post-conf row.

The portable gates are reused unchanged. Their prediction features now use the
declared native validator geometry with explicit ratio_pad. This is a new
unified readout comparison, not byte reproduction of the older4500 predictor.
All actions tighten the input640 logit threshold. No GT, fitting or scanning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

from frozen_io import sha256

ADAPTER_VERSION = "frozen_native_mask_actions_v1"
RESPONSE_SHA256 = "eb2fca8b9bbb0192d62723e3da086c303729752772aa1a357ea64f61f70888b9"
MULTI_SHA256 = "71874d3f23e6cbe2179ce0962c751b7afd04b9e29581a0c3227515ca48b76987"
ARMS = ("baseline", "RCMC_full", "global_minus025_full", "RCMC_first64", "global_minus025_first64", "multi_local_full", "multi_local_first64")
ACTIONS = ("smooth", "global_025", "global_05", "global_075", "global_10")
LOCAL_NAMES = ("removed_mean", "removed_std", "kept_mean", "kept_std", "boundary_mean", "boundary_std",
               "removed_boundary_fraction", "removed_near_kept_fraction", "boundary_foreground_fraction",
               "foreground_below025", "foreground_below075", "foreground_below10")


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        synchronize(self.device)
        self.seconds = time.perf_counter() - self.start


def timed_native_forward(extractor, image_path):
    """One real forward; separate synchronized extraction wall and model time."""
    import torch
    cuda = str(extractor.device).startswith("cuda")
    events, calls = {}, []
    if cuda:
        events = {key: torch.cuda.Event(enable_timing=True) for key in ("start", "end")}

    def before(module, args):
        calls.append(time.perf_counter())
        if cuda:
            events["start"].record()

    def after(module, args, result):
        if cuda:
            events["end"].record()
        events["cpu_end"] = time.perf_counter()

    hooks = [extractor.model.register_forward_pre_hook(before), extractor.model.register_forward_hook(after)]
    try:
        with Timer(extractor.device) as timer:
            extracted = extractor.extract(image_path)
    finally:
        for hook in hooks:
            hook.remove()
    if len(calls) != 1:
        raise RuntimeError("Expected exactly one real frozen model forward per image")
    model_seconds = events["start"].elapsed_time(events["end"]) / 1000 if cuda else events["cpu_end"] - calls[0]
    return extracted, {"frozen_extract_wall_seconds": timer.seconds, "native_model_seconds": model_seconds,
                       "model_timer": "CUDA events around actual model call" if cuda else "CPU wall around actual model call",
                       "extraction_wall_includes": "image I/O, preprocessing, model, head replay, GPU/CPU tensor transfers"}


@dataclass
class NativeRows:
    coefficients: object
    boxes: object
    baseline_input: object
    baseline_original: object
    raw_boxes: object
    scores: object
    classes: object
    original_boxes: object
    eligible_output_rows: list
    proto_supported: object
    supported_first64: object
    pbatch: dict


def prepare_native(extracted, validator, device):
    """Use the actual validator once; capture exactly its coefficient batch."""
    import torch
    from ultralytics.utils import ops
    capture = {}

    def process(proto, coefficients, boxes, shape):
        if capture:
            raise RuntimeError("Unexpected multiple native mask decoder calls")
        capture.update(coefficients=coefficients.detach().clone(), boxes=boxes.detach().clone(), shape=tuple(shape))
        masks = ops.process_mask_native(proto, coefficients, boxes, shape)
        capture["baseline"] = masks.clone()
        return masks

    native = torch.cat((extracted["boxes"], extracted["scores"][:, None], extracted["classes"].float()[:, None], extracted["c0"]), 1)[None].to(device)
    previous = validator.process
    validator.process = process
    try:
        prediction = validator.postprocess(((native, extracted["P"][None].to(device)), None))[0]
    finally:
        validator.process = previous
    eligible = (extracted["scores"] > .001).nonzero().flatten().tolist()
    if capture.get("shape") != (640, 640) or len(prediction["conf"]) != len(eligible):
        raise RuntimeError("Actual native post-conf candidate count or input shape differs")
    checks = (torch.equal(capture["coefficients"].cpu(), extracted["c0"][eligible]),
              torch.equal(prediction["bboxes"].cpu(), extracted["boxes"][eligible]),
              torch.equal(prediction["conf"].cpu(), extracted["scores"][eligible]),
              torch.equal(prediction["cls"].long().cpu(), extracted["classes"][eligible]))
    if not all(checks):
        raise RuntimeError("Native class/conf/box/coefficient output identity changed")
    pbatch = {"imgsz": (640, 640), "ori_shape": tuple(extracted["original_shape"]),
              "ratio_pad": (extracted["letterbox"]["ratio"], (extracted["letterbox"]["left"], extracted["letterbox"]["top"])),
              "im_file": extracted["image_path"]}
    scaled = validator.scale_preds(prediction, pbatch)
    h, w = extracted["P"].shape[-2:]
    support = []
    for position, row in enumerate(eligible):
        box = extracted["boxes_proto"][row]
        valid = bool(((torch.arange(w) >= box[0]) & (torch.arange(w) < box[2])).any()
                     and ((torch.arange(h) >= box[1]) & (torch.arange(h) < box[3])).any())
        support.append(valid)
    return NativeRows(capture["coefficients"], capture["boxes"], capture["baseline"], scaled["masks"].cpu(),
                      prediction["bboxes"].cpu().numpy().copy(), prediction["conf"].cpu().numpy().copy(),
                      prediction["cls"].long().cpu().numpy().copy(), scaled["bboxes"].cpu().numpy().copy(),
                      eligible, torch.tensor(support, device=device, dtype=torch.bool),
                      torch.tensor([position < 64 and valid for position, valid in enumerate(support)], device=device, dtype=torch.bool), pbatch)


def native_proto_logits(prototypes, coefficients):
    """Original32-channel, full-M GEMM; retain the actual post-conf row batch."""
    channels, height, width = prototypes.shape
    if channels != 32 or coefficients.ndim != 2 or coefficients.shape[1] != 32:
        raise ValueError("Native32-channel prototype/coefficients required")
    return (coefficients @ prototypes.float().view(channels, -1)).view(-1, height, width)


def export_binary(binary, pbatch):
    """Exactly the official scale_preds binary resize, explicit ratio_pad, byte."""
    from ultralytics.utils import ops
    return ops.scale_masks(binary[None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[0].byte()


def action_features(logits640, baseline640, baseline_original, trial640, trial_original, tau):
    """Reuse the frozen5 feature formula, followed by tau and original local12."""
    import numpy as np
    from risk_calibration import prediction_features
    from local_features import local_features
    five = prediction_features(baseline640, baseline_original, trial_original, "response")
    local = local_features(logits640, baseline640, trial640)
    thresholds = tau.detach().cpu().numpy().astype(np.float64)
    eighteen = np.concatenate((five, thresholds[:, None], local), axis=1)
    if five.shape[1] != 5 or local.shape[1] != 12 or eighteen.shape[1] != 18 or not np.isfinite(eighteen).all():
        raise FloatingPointError("Frozen action feature shape/finite contract failed")
    return five, eighteen


def _records(masks, prepared, image_id, offset, categories, mask_utils):
    import numpy as np
    masks = masks.detach().cpu().numpy()
    records = []
    for local, mask in enumerate(masks):
        index = offset + local
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        box = prepared.original_boxes[index]
        xywh = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
        records.append({"image_id": int(image_id), "category_id": int(categories[int(prepared.classes[index])]),
                        "bbox": [round(value, 3) for value in xywh], "score": round(float(prepared.scores[index]), 5),
                        "segmentation": rle, "detection_index": index, "raw_input_box_xyxy": prepared.raw_boxes[index].tolist(),
                        "raw_confidence": float(prepared.scores[index]), "box_xyxy": box.tolist(), "model_class": int(prepared.classes[index])})
    return records


class NativeMaskAdapter:
    def __init__(self, response_json, multi_json, device="cuda", expected_multi_sha256=MULTI_SHA256):
        from portable_risk import PortableRisk
        self.device = device
        if expected_multi_sha256 != MULTI_SHA256 or sha256(response_json) != RESPONSE_SHA256 or sha256(multi_json) != MULTI_SHA256:
            raise ValueError("Frozen response/multi model bytes differ from the registered assets")
        if Path(multi_json).stat().st_size != 196839:
            raise ValueError("Frozen multi asset size differs")
        self.response = PortableRisk(response_json)
        self.multi = PortableRisk(multi_json)
        if self.response.features != 5 or self.multi.features != 18:
            raise ValueError("Frozen regressor feature counts differ")
        self.assets = {"response": {"path": str(Path(response_json).resolve()), "sha256": RESPONSE_SHA256},
                       "multi_local": {"path": str(Path(multi_json).resolve()), "sha256": MULTI_SHA256}}

    def decode_arms(self, extracted, prepared, image_id, categories, mask_utils):
        import numpy as np
        import torch
        from ultralytics.utils import ops
        count = len(prepared.coefficients)
        outputs = {arm: [] for arm in ARMS}
        timing = {"proto_logits_seconds": 0.0, "native_upsample_seconds": 0.0, "base_export_seconds": 0.0,
                  "actions": {action: {"threshold_seconds": 0.0, "export_seconds": 0.0, "features_seconds": 0.0} for action in ACTIONS},
                  "response_gate_seconds": 0.0, "multi_gate_seconds": 0.0, "arm_export_rle_seconds": {arm: 0.0 for arm in ARMS}}
        decisions = []
        if count == 0:
            return outputs, {"native_rows": 0, "selected_first64": 0, "supported_first64": 0, "unsupported_first64": 0,
                             "decisions": [], "timing": timing, "tau0_native_exact": True,
                             "all_arm_native_identity_exact": True, "empty_and_first64_invariants": True, "gt_used": False}
        proto = extracted["P"].to(self.device)
        channels, height, width = proto.shape
        with Timer(self.device) as timer:
            # Preserve the official32 channels and full post-conf GEMM row batch.
            all_logits = native_proto_logits(proto, prepared.coefficients)
        timing["proto_logits_seconds"] += timer.seconds
        step = max(1, 32_000_000 // (640 * 640))
        for start in range(0, count, step):
            end = min(count, start + step)
            boxes = prepared.boxes[start:end]
            with Timer(self.device) as timer:
                logits = ops.scale_masks(all_logits[start:end][None], (640, 640))[0]
                baseline = ops.crop_mask((logits > 0).byte(), boxes)
            timing["native_upsample_seconds"] += timer.seconds
            if not torch.equal(baseline, prepared.baseline_input[start:end]):
                raise RuntimeError("Tau0 input binary differs from unmodified process_mask_native")
            with Timer(self.device) as timer:
                base_export = export_binary(baseline, prepared.pbatch)
            timing["base_export_seconds"] += timer.seconds
            if not torch.equal(base_export.cpu(), prepared.baseline_original[start:end]):
                raise RuntimeError("Tau0 original binary differs from unmodified native scale_preds")
            area = base_export.sum((1, 2)).float()
            trials, exported, feature5, feature18, thresholds, eligible = [], [], [], [], [], []
            for index, action in enumerate(ACTIONS):
                with Timer(self.device) as timer:
                    tau = .75 / (1.0 + (area / 2304.0).square()) if index == 0 else torch.full_like(area, (.25, .5, .75, 1.0)[index - 1])
                    trial = ops.crop_mask((logits > tau[:, None, None]).byte(), boxes)
                timing["actions"][action]["threshold_seconds"] += timer.seconds
                with Timer(self.device) as timer:
                    exp = export_binary(trial, prepared.pbatch)
                timing["actions"][action]["export_seconds"] += timer.seconds
                if bool((trial > baseline).any()) or bool((exp > base_export).any()):
                    raise RuntimeError("A positive-threshold action expanded baseline support")
                with Timer(self.device) as timer:
                    five, eighteen = action_features(logits, baseline, base_export, trial, exp, tau)
                timing["actions"][action]["features_seconds"] += timer.seconds
                trials.append(trial)
                exported.append(exp)
                feature5.append(five)
                feature18.append(eighteen)
                thresholds.append(tau)
                eligible.append(exp.flatten(1).any(1).bool().cpu().numpy())
            with Timer(self.device) as timer:
                gain = self.response.predict(feature5[0])
                if not np.isfinite(gain).all():
                    raise FloatingPointError("Response gate predicted a nonfinite gain")
                rcmc_chosen = (gain > 0) & eligible[0]
                rcmc = torch.where(torch.as_tensor(rcmc_chosen, device=self.device)[:, None, None], exported[0], base_export)
            timing["response_gate_seconds"] += timer.seconds
            with Timer(self.device) as timer:
                features = np.stack(feature18, 1)
                scores = self.multi.predict(features.reshape(-1, 18)).reshape(end - start, len(ACTIONS))
                if not np.isfinite(scores).all():
                    raise FloatingPointError("Multi gate predicted a nonfinite gain")
                availability = np.stack(eligible, 1)
                scores_masked = np.where(availability, scores, -np.inf)
                choices = np.where(scores_masked.max(1) > 0, scores_masked.argmax(1), -1)
                multi = base_export.clone()
                for action in range(len(ACTIONS)):
                    use = torch.as_tensor(choices == action, device=self.device)
                    multi[use] = exported[action][use]
            timing["multi_gate_seconds"] += timer.seconds
            first64 = prepared.supported_first64[start:end]
            arm_masks = {"baseline": base_export, "RCMC_full": rcmc, "global_minus025_full": exported[1],
                         "RCMC_first64": torch.where(first64[:, None, None], rcmc, base_export),
                         "global_minus025_first64": torch.where(first64[:, None, None], exported[1], base_export),
                         "multi_local_full": multi, "multi_local_first64": torch.where(first64[:, None, None], multi, base_export)}
            for arm, masks in arm_masks.items():
                with Timer(self.device) as timer:
                    outputs[arm].extend(_records(masks, prepared, image_id, start, categories, mask_utils))
                timing["arm_export_rle_seconds"][arm] += timer.seconds
            area_list = area.cpu().tolist()
            tau_list = thresholds[0].cpu().tolist()
            first64_list = first64.cpu().tolist()
            proto_supported_list = prepared.proto_supported[start:end].cpu().tolist()
            for local in range(end - start):
                position = start + local
                decisions.append({"detection_index": position, "native_output_row": prepared.eligible_output_rows[position],
                                  "raw_index": int(extracted["raw_indices"][prepared.eligible_output_rows[position]]),
                                  "in_first64": position < 64, "first64_proto_supported": bool(first64_list[local]),
                                  "proto_roi_supported": bool(proto_supported_list[local]),
                                  "baseline_area": float(area_list[local]), "smooth_tau": float(tau_list[local]),
                                  "response_features": feature5[0][local].tolist(), "response_gain": float(gain[local]),
                                  "response_chosen": bool(rcmc_chosen[local]), "trial_nonempty": availability[local].tolist(),
                                  "multi_gains": scores[local].tolist(), "multi_choice": int(choices[local]),
                                  "baseline_empty": area_list[local] == 0, "global025_empty": not bool(eligible[1][local])})
            del logits, baseline, base_export, trials, exported, feature5, feature18, arm_masks, rcmc, multi
        identity_keys = ("image_id", "category_id", "bbox", "score", "detection_index", "raw_input_box_xyxy", "raw_confidence", "box_xyxy", "model_class")
        for arm in ARMS:
            if len(outputs[arm]) != count or any(any(a[key] != b[key] for key in identity_keys) for a, b in zip(outputs["baseline"], outputs[arm])):
                raise RuntimeError("An arm changed native row/box/class/score/order: " + arm)
        for index, decision in enumerate(decisions):
            for arm in ("RCMC_first64", "global_minus025_first64", "multi_local_first64"):
                if not decision["first64_proto_supported"] and outputs[arm][index] != outputs["baseline"][index]:
                    raise RuntimeError("An unsupported/out-of-first64 row changed: " + arm)
            if decision["response_chosen"] and not decision["trial_nonempty"][0]:
                raise RuntimeError("RCMC selected an empty trial")
            if decision["multi_choice"] >= 0 and not decision["trial_nonempty"][decision["multi_choice"]]:
                raise RuntimeError("Multi selected an empty trial")
        return outputs, {"native_rows": count, "selected_first64": min(64, count), "supported_first64": int(prepared.supported_first64.sum()),
                         "unsupported_first64": min(64, count) - int(prepared.supported_first64.sum()),
                         "tau0_native_exact": True, "all_arm_native_identity_exact": True, "empty_and_first64_invariants": True, "gt_used": False,
                         "timing": timing, "timing_note": "synchronized stage wall times; RLE includes device/host transfer; shared work is not charged once per arm",
                         "decisions": decisions}
