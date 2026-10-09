"""Bound finite actions on the fresh, official native input640 mask grid.

No GT, RGB feature, fitted gate, proto-ROI exclusion, first64 restriction, or
action combination enters this module. Native post-conf rows remain distinct
even when their raw indices repeat. Dense actions are exported one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys

import numpy as np

from local_actions import BAND_RADIUS, MAX_ACTIONS_PER_SIGN, LocalAction, apply_action, generate_actions

# The source snapshot is self-contained; the working copy reuses reviewed code.
NATIVE_SCRIPTS = (Path(__file__).parent if Path(__file__).with_name("frozen_io.py").is_file()
                  else Path(__file__).resolve().parents[2] / "frozen_native_mask_comparison" / "scripts")
sys.path.insert(0, str(NATIVE_SCRIPTS))
from frozen_io import FrozenYOLO, OFFICIAL_SHA256, sha256  # noqa: E402
from native_mask_adapter import Timer, export_binary, native_proto_logits, prepare_native, timed_native_forward  # noqa: E402

ADAPTER_VERSION = "local_edit_native_opportunity_v1"
GLOBAL_ACTIONS = (("smooth", None), ("global_025", .25), ("global_05", .5),
                  ("global_075", .75), ("global_10", 1.0), ("global_expand_025", -.25))


def array_sha256(array):
    """Bind shape, dtype and C-order bytes, not merely foreground area."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((str(array.dtype), tuple(array.shape))).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateBinding:
    image_id: int
    detection_index: int
    native_output_row: int
    raw_index: int
    baseline_input_sha256: str
    hidden_query_sha256: str
    coefficient_sha256: str

    def metadata(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class BoundLocalAction:
    binding: CandidateBinding
    action: LocalAction

    def apply(self, baseline, binding):
        if binding != self.binding or array_sha256(baseline) != binding.baseline_input_sha256:
            raise ValueError("Local action cannot cross image/native row/raw/baseline identity")
        return apply_action(baseline, self.action)


def _band_morphology(mask, erode):
    result = mask.copy()
    height, width = mask.shape
    for _ in range(BAND_RADIUS):
        padded = np.pad(result, 1, constant_values=False)
        result = padded[:height, :width].copy()
        for row in range(3):
            for column in range(3):
                if row == 0 and column == 0:
                    continue
                neighbor = padded[row:row+height, column:column+width]
                if erode:
                    result &= neighbor
                else:
                    result |= neighbor
    return result


def component_inventory(mask):
    """Count all exact 8-components before top8, without expanding their pixels.

    Independent row-run union-find matches core's area/min-row/min-column/first
    flat-index ordering. Disjoint components cannot tie on first flat index,
    so this last key equals the full lexicographic sorted-pixel tie-break.
    """
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Component inventory requires 2-D bool input")
    width = mask.shape[1]
    parents, sizes, runs, previous = [], [], [], []

    def find(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parents[right] = left
        sizes[left] += sizes[right]

    for row, line in enumerate(mask):
        padded = np.empty(width+2, dtype=np.bool_)
        padded[0] = padded[-1] = False
        padded[1:-1] = line
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        current, cursor = [], 0
        for start, end in zip(transitions[::2], transitions[1::2]):
            start, end = int(start), int(end)
            label = len(parents)
            parents.append(label)
            sizes.append(1)
            runs.append((row, start, end, label))
            current.append((start, end, label))
            while cursor < len(previous) and previous[cursor][1] < start:
                cursor += 1
            neighbor = cursor
            while neighbor < len(previous) and previous[neighbor][0] <= end:
                union(label, previous[neighbor][2])
                neighbor += 1
        previous = current
    groups = {}
    for row, start, end, label in runs:
        root = find(label)
        if root not in groups:
            groups[root] = {"area": 0, "bbox": [row, start, row+1, end], "first_flat_index": row*width+start}
        item = groups[root]
        item["area"] += end-start
        item["bbox"][0] = min(item["bbox"][0], row)
        item["bbox"][1] = min(item["bbox"][1], start)
        item["bbox"][2] = max(item["bbox"][2], row+1)
        item["bbox"][3] = max(item["bbox"][3], end)
    ordered = sorted(groups.values(), key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1], item["first_flat_index"]))
    selected = ordered[:MAX_ACTIONS_PER_SIGN]
    return {"components_before_cap": len(ordered), "components_kept": len(selected),
            "components_truncated": max(0, len(ordered)-len(selected)),
            "pixels_before_cap": sum(item["area"] for item in ordered),
            "pixels_truncated": sum(item["area"] for item in ordered[MAX_ACTIONS_PER_SIGN:]),
            "selected": selected}


@dataclass
class NativeCandidate:
    binding: CandidateBinding
    z: object
    support_tensor: object
    baseline: np.ndarray
    support: np.ndarray
    baseline_original: np.ndarray
    pbatch: dict
    identity: dict
    local_actions: tuple[BoundLocalAction, ...]
    component_counts: dict
    generation_seconds: float
    smooth_tau: float


class FiniteNativeAdapter:
    def __init__(self, device="cuda", chunk_size=32_000_000 // (640*640), engineering=False):
        if chunk_size != 32_000_000 // (640*640):
            raise ValueError("Native decode must preserve reviewed step78")
        self.device, self.chunk_size, self.engineering = device, chunk_size, engineering
        self.timings = {name: 0.0 for name in ("proto_logits_seconds", "native_upsample_support_seconds",
                                              "baseline_export_seconds", "action_generation_seconds",
                                              "action_export_seconds", "action_input_validation_seconds",
                                              "engineering_empty_probe_export_seconds")}
        self.checks = {name: 0 for name in ("native_input_parity_candidates", "native_original_parity_candidates",
                                           "identity_bound_candidates", "zero_exact_actions", "support_protected_actions",
                                           "single_region_actions", "dtype_checked_actions", "empty_actions_retained",
                                           "cross_identity_rejection_candidates", "changed_baseline_rejection_candidates",
                                           "forced_empty_export_probe_candidates")}

    def candidates(self, extracted, prepared, image_id, categories):
        import torch
        from ultralytics.utils import ops
        count = len(prepared.coefficients)
        if count == 0:
            return
        with Timer(self.device) as timer:
            # Exactly one full post-conf M x 32 GEMM per image.
            all_logits = native_proto_logits(extracted["P"].to(self.device), prepared.coefficients)
        self.timings["proto_logits_seconds"] += timer.seconds
        for start in range(0, count, self.chunk_size):
            end = min(count, start+self.chunk_size)
            with Timer(self.device) as timer:
                zbatch = ops.scale_masks(all_logits[start:end][None], (640, 640))[0]
                # The support is independent of the value of z; negative tau
                # must never threshold a zero-filled cropped logit image.
                support_batch = ops.crop_mask(torch.ones_like(zbatch, dtype=torch.uint8), prepared.boxes[start:end]).bool()
                baseline_batch = ((zbatch > 0) & support_batch).byte()
            self.timings["native_upsample_support_seconds"] += timer.seconds
            if not torch.equal(baseline_batch, prepared.baseline_input[start:end]):
                raise RuntimeError("Fresh official input640 baseline parity failed")
            with Timer(self.device) as timer:
                base_original = export_binary(baseline_batch, prepared.pbatch)
            self.timings["baseline_export_seconds"] += timer.seconds
            if base_original.dtype != torch.uint8 or not torch.equal(base_original.cpu(), prepared.baseline_original[start:end]):
                raise RuntimeError("Fresh official original byte baseline parity failed")
            # The original G_old computes this FP32 vector from the exported
            # byte masks. Preserve its operation sequence and native chunk.
            with Timer(self.device) as timer:
                area = base_original.sum((1, 2)).float()
                smooth_taus = .75 / (1.0 + (area / 2304.0).square())
            self.timings["action_generation_seconds"] += timer.seconds
            for offset in range(end-start):
                position, row = start+offset, prepared.eligible_output_rows[start+offset]
                with Timer(self.device) as timer:
                    z, support_tensor = zbatch[offset], support_batch[offset]
                    baseline = baseline_batch[offset].cpu().numpy().astype(np.bool_)
                    support = support_tensor.cpu().numpy().copy()
                    positive = ((z > .25) & support_tensor).cpu().numpy()
                    negative = ((z > -.25) & support_tensor).cpu().numpy()
                    original = base_original[offset].cpu().numpy().astype(np.bool_)
                    binding = CandidateBinding(int(image_id), position, int(row), int(extracted["raw_indices"][row]),
                                               array_sha256(baseline), array_sha256(extracted["h"][row].numpy()),
                                               array_sha256(extracted["c0"][row].numpy()))
                    if int(extracted["output_rows"][row]) != row:
                        raise RuntimeError("Native output row identity differs")
                    if not torch.equal(prepared.coefficients[position].cpu(), extracted["c0"][row]):
                        raise RuntimeError("Native coefficients changed identity")
                    actions = generate_actions(baseline, positive, negative, support)
                    removal = baseline & ~positive & (baseline & ~_band_morphology(baseline, True)) & support
                    addition = ~baseline & negative & (~baseline & _band_morphology(baseline, False)) & support
                    inventories = {"remove": component_inventory(removal), "add": component_inventory(addition)}
                    for sign, name in ((-1, "remove"), (1, "add")):
                        actual = [action for action in actions if action.sign == sign]
                        expected = inventories[name]["selected"]
                        if len(actual) != len(expected) or any(action.area != item["area"] or list(action.bbox) != item["bbox"] or int(action.flat_indices[0]) != item["first_flat_index"] for action, item in zip(actual, expected)):
                            raise RuntimeError("Independent pre-cap inventory differs from core top8 ordering")
                    if self.engineering:
                        bound = BoundLocalAction(binding, actions[0])
                        for wrong in (replace(binding, image_id=binding.image_id+1),
                                      replace(binding, detection_index=binding.detection_index+1),
                                      replace(binding, native_output_row=binding.native_output_row+1),
                                      replace(binding, raw_index=binding.raw_index+1)):
                            try:
                                bound.apply(baseline, wrong)
                            except ValueError:
                                pass
                            else:
                                raise RuntimeError("Bound action accepted foreign candidate identity")
                        self.checks["cross_identity_rejection_candidates"] += 1
                        changed_baseline = baseline.copy()
                        changed_baseline[0, 0] = not changed_baseline[0, 0]
                        try:
                            bound.apply(changed_baseline, binding)
                        except ValueError:
                            pass
                        else:
                            raise RuntimeError("Bound action accepted different same-shape baseline")
                        self.checks["changed_baseline_rejection_candidates"] += 1
                    baseline.flags.writeable = support.flags.writeable = original.flags.writeable = False
                    identity = binding.metadata() | {
                        "model_class": int(prepared.classes[position]), "category_id": int(categories[int(prepared.classes[position])]),
                        "raw_confidence": float(prepared.scores[position]), "raw_input_box_xyxy": prepared.raw_boxes[position].tolist(),
                        "box_xyxy": prepared.original_boxes[position].tolist(), "input_shape": [640, 640],
                        "original_shape": list(original.shape), "ratio_pad": prepared.pbatch["ratio_pad"],
                        "baseline_original_sha256": array_sha256(original), "hidden_query_source": extracted["hidden_query_source"],
                        "input_sha256": extracted["input_sha256"], "source_image_sha256": extracted["image_sha256"]}
                self.timings["action_generation_seconds"] += timer.seconds
                for name in ("native_input_parity_candidates", "native_original_parity_candidates", "identity_bound_candidates"):
                    self.checks[name] += 1
                yield NativeCandidate(binding, z, support_tensor, baseline, support, original, prepared.pbatch,
                                      identity, tuple(BoundLocalAction(binding, action) for action in actions), inventories,
                                      timer.seconds, float(smooth_taus[offset].item()))
            del zbatch, support_batch, baseline_batch, base_original, area, smooth_taus
        del all_logits

    def export_actions(self, candidate):
        """Yield (metadata, original bool, sparse indices) for every finite action."""
        import torch
        baseline, support = candidate.baseline, candidate.support
        base_area = int(candidate.baseline_original.sum())
        smooth_tau = candidate.smooth_tau
        if self.engineering:
            with Timer(self.device) as timer:
                probe = export_binary(torch.zeros((1, 640, 640), dtype=torch.uint8, device=self.device), candidate.pbatch)
                if probe.dtype != torch.uint8 or bool(probe.any()) or tuple(probe.shape[1:]) != tuple(candidate.baseline_original.shape):
                    raise RuntimeError("Explicit engineering empty export probe failed")
                self.checks["forced_empty_export_probe_candidates"] += 1
            self.timings["engineering_empty_probe_export_seconds"] += timer.seconds
            del probe
        specifications = [("zero", 0.0, candidate.local_actions[0])]
        specifications += [(name, smooth_tau if tau is None else tau, None) for name, tau in GLOBAL_ACTIONS]
        specifications += [(bound.action.action_id, bound.action.tau, bound) for bound in candidate.local_actions[1:]]
        for action_id, tau, bound in specifications:
            with Timer(self.device) as generation_timer:
                if bound is not None:
                    binary_np = bound.apply(baseline, candidate.binding)
                    binary = torch.from_numpy(binary_np).to(self.device).byte()[None]
                else:
                    binary = ((candidate.z > tau) & candidate.support_tensor).byte()[None]
                    binary_np = binary[0].cpu().numpy().astype(np.bool_)
            self.timings["action_generation_seconds"] += generation_timer.seconds
            with Timer(self.device) as validation_timer:
                if binary_np.dtype != np.bool_ or binary.dtype != torch.uint8 or binary_np.shape != (640, 640):
                    raise RuntimeError("Action dtype/input shape contract failed")
                if np.any(binary_np & ~support):
                    raise RuntimeError("Action escaped true predicted-box crop support")
                input_changed = np.flatnonzero(binary_np.reshape(-1) != baseline.reshape(-1))
                if bound is not None:
                    if not np.array_equal(input_changed, bound.action.flat_indices):
                        raise RuntimeError("Local action changed pixels outside its one sparse region")
                    if bound.action.sign and self.engineering and component_inventory(binary_np != baseline)["components_before_cap"] != 1:
                        raise RuntimeError("Local action is not exactly one 8-connected region")
                    if bound.action.sign:
                        self.checks["single_region_actions"] += 1
                if action_id == "zero" and not np.array_equal(binary_np, baseline):
                    raise RuntimeError("Zero input action differs from baseline")
                if tau > 0 and bound is None and np.any(binary_np & ~baseline):
                    raise RuntimeError("Positive global action unexpectedly expanded")
                if tau < 0 and bound is None and np.any(baseline & ~binary_np):
                    raise RuntimeError("Negative global action unexpectedly contracted")
            self.timings["action_input_validation_seconds"] += validation_timer.seconds
            with Timer(self.device) as export_timer:
                exported = export_binary(binary, candidate.pbatch)
                if exported.dtype != torch.uint8:
                    raise RuntimeError("Native original export must have uint8 dtype")
                original = exported[0].cpu().numpy().astype(np.bool_)
            self.timings["action_export_seconds"] += export_timer.seconds
            if action_id == "zero":
                if not np.array_equal(original, candidate.baseline_original):
                    raise RuntimeError("Zero original action differs from fresh native byte baseline")
                self.checks["zero_exact_actions"] += 1
            changed = int(np.count_nonzero(original != candidate.baseline_original))
            area = int(original.sum())
            family = "zero" if action_id == "zero" else ("local" if bound is not None else "global")
            metadata = {"action_id": action_id, "family": family, "tau": float(tau),
                        "available_in": (["G_old", "G_eq", "G_eq_plus_L"] if family == "zero" or family == "global" and tau > 0
                                         else ["G_eq", "G_eq_plus_L"] if family == "global" else ["G_eq_plus_L"]),
                        "input_area": int(binary_np.sum()), "input_changed_area": int(input_changed.size),
                        "original_area": area, "original_changed_area": changed,
                        "original_changed_image_fraction": changed/original.size,
                        "original_changed_baseline_area_ratio": changed/base_area if base_area else None,
                        "input_empty": not bool(binary_np.any()), "original_empty": area == 0,
                        "timing": {"generation_seconds": generation_timer.seconds, "validation_seconds": validation_timer.seconds,
                                   "native_export_seconds": export_timer.seconds},
                        "sparse": None if bound is None else bound.action.metadata() | {
                            "binding": candidate.binding.metadata(), "indices_sha256": array_sha256(bound.action.flat_indices)}}
            self.checks["support_protected_actions"] += 1
            self.checks["dtype_checked_actions"] += 1
            if area == 0:
                self.checks["empty_actions_retained"] += 1
            yield metadata, original, None if bound is None else bound.action.flat_indices
            del binary, binary_np, exported, original
        if array_sha256(baseline) != candidate.binding.baseline_input_sha256:
            raise RuntimeError("Generating candidate baseline was mutated")
