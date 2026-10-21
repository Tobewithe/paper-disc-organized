"""Mask-only branches for the separately registered joint-adaptation study.

The native model/head is never registered as a child or mutated. Inputs identify
ALL final output rows, including repeated raw IDs. This module deliberately does
not apply the deployment area gate: the caller must retain native RLEs outside
the frozen gate. GT is accepted only by ``loss``, never by ``forward``.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn


REPO = Path(__file__).resolve().parents[3]
for _scripts in (
    REPO / "experiments/mask_readout_mechanism_v2_20261012/scripts",
    REPO / "experiments/mask_readout_soft_iou_pilot_20261014/scripts",
):
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
from adapter_model import CoefficientAdapter  # noqa: E402
from soft_iou_loss import LOSS_SPEC, soft_iou_grouped_loss  # noqa: E402


ARMS = ("G_budget", "C_native", "PC_native")
FEATURE_CHANNELS = (256, 512, 512)
PROTO_MASK_CHILDREN = ("cv1", "upsample", "cv2", "cv3", "feat_refine", "feat_fuse")


def detached_tensor(value: torch.Tensor, *, device=None, dtype=None) -> torch.Tensor:
    """Detach native tensors and materialize inference tensors outside that mode.

    Convolution backward must save F even when F itself is not trainable. An
    inference tensor cannot be saved for backward, hence the conditional clone.
    Ordinary input tensors need no copy unless device/dtype conversion needs it.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError("a tensor is required")
    if value.is_inference():
        with torch.inference_mode(False):
            value = value.detach().clone()
    else:
        value = value.detach()
    return value.to(device=device, dtype=dtype)


def snapshot_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """CPU byte-preserving snapshot for a frozen external model or buffers."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def assert_state_unchanged(module: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> dict:
    current = module.state_dict()
    if set(current) != set(snapshot):
        raise AssertionError("frozen state keys changed")
    changed = [name for name in current if current[name].dtype != snapshot[name].dtype
               or not torch.equal(current[name].detach().cpu(), snapshot[name])]
    if changed:
        raise AssertionError(f"frozen state changed: {changed}")
    return {"status": "passed", "state_tensors_checked": len(current)}


def _require_fused_mask_paths(head: nn.Module) -> None:
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError("provide the frozen native eval head; do not change it here")
    if not bool(getattr(head, "end2end", False)) or getattr(head, "nm", None) != 32:
        raise ValueError("the frozen YOLO26 O2O 32-coefficient head is required")
    if not hasattr(head, "one2one_cv4") or len(head.one2one_cv4) != 3:
        raise ValueError("three native O2O coefficient branches are required")
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm)
           for root in (head.one2one_cv4, head.proto) for m in root.modules()):
        raise ValueError("native mask paths must already be fused; BN found")
    if getattr(head.proto, "semseg", None) is not None:
        raise ValueError("native Proto must already have its semantic path removed")
    active_children = {name for name, child in head.proto.named_children() if child is not None}
    if active_children != set(PROTO_MASK_CHILDREN):
        raise ValueError(f"unexpected native Proto mask paths: {sorted(active_children)}")


def _raw_ids(cache: Mapping, shapes: list[list[int]], count: int, device) -> torch.Tensor:
    locations = cache["raw_locations"]
    if len(locations) != count:
        raise ValueError("raw-location count differs from output count")
    derived = []
    for location in locations:
        level = int(location["level"])
        if level not in range(len(shapes)):
            raise ValueError("invalid raw level")
        channels, height, width = shapes[level]
        row, column = int(location["row"]), int(location["column"])
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError("raw location outside feature grid")
        offset = sum(s[1] * s[2] for s in shapes[:level])
        if "level_offset" in location and int(location["level_offset"]) != offset:
            raise ValueError("raw-level offset changed")
        if "channels" in location and int(location["channels"]) != channels:
            raise ValueError("raw-level channel count changed")
        derived.append(offset + row * width + column)
    if "raw_ids" in cache:
        saved = torch.as_tensor(cache["raw_ids"], dtype=torch.long).cpu().tolist()
        if saved != derived:
            raise ValueError("raw ID and level/row/column identities disagree")
    if "output_ranks" in cache and list(cache["output_ranks"]) != list(range(count)):
        raise ValueError("final output rows were reordered")
    return torch.tensor(derived, dtype=torch.long, device=device)


def mask_loss_grids(cache: Mapping, proto: torch.Tensor, coefficients: torch.Tensor):
    """Same original ROI and P0*c0 baseline, without detaching current P."""
    p0 = detached_tensor(cache["proto"], device=coefficients.device, dtype=torch.float32)
    c0 = detached_tensor(cache["c0"], device=coefficients.device, dtype=torch.float32)
    if proto.ndim != 3 or proto.shape != p0.shape or coefficients.shape != c0.shape:
        raise ValueError("current/native P or coefficient geometry differs")
    if proto.shape[0] != 32 or coefficients.ndim != 2 or coefficients.shape[1] != 32:
        raise ValueError("the original 32-channel mask representation is required")
    height, width = proto.shape[-2:]
    logits = (coefficients @ proto.flatten(1)).reshape(len(coefficients), height, width)
    baseline = (c0 @ p0.flatten(1)).reshape_as(logits).detach()
    ih, iw = cache["input_hw"]
    if min(ih, iw) <= 0:
        raise ValueError("invalid native input geometry")
    boxes = detached_tensor(cache["boxes_input"], device=coefficients.device, dtype=torch.float32)
    if boxes.shape != (len(coefficients), 4):
        raise ValueError("box/output row count mismatch")
    boxes = boxes * coefficients.new_tensor([width / iw, height / ih, width / iw, height / ih])
    x = torch.arange(width, device=coefficients.device)[None, None, :]
    y = torch.arange(height, device=coefficients.device)[None, :, None]
    roi = ((x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None])
           & (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None]))
    return logits, baseline, roi


class JointMaskBranch(nn.Module):
    """One independently optimized arm, using native frozen final-row identity.

    ``head`` is the already fused/eval/frozen native Segment26. Native arms copy
    only O2O cv4; PC also copies Proto's mask path. ``adapter_initial`` is required
    for G and may be the original initial.pt path or its loaded dictionary.

    ``forward`` accepts a single-image cache, original full native-batch F, and
    its batch_index. ``forward_batch`` runs C/P on the complete original batch
    once, then selects the requested image rows. Changing the native convolution
    batch size can change numerical outputs and is not an allowed shortcut.
    Training uses all original outputs, not only outputs inside the area gate.
    """

    def __init__(self, head: nn.Module, arm: str, adapter_initial=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        _require_fused_mask_paths(head)
        self.arm = arm
        self.adapter = None
        self.coefficient_head = None
        self.proto = None
        if arm == "G_budget":
            if adapter_initial is None:
                raise ValueError("G_budget must start from the recorded original B initial.pt")
            saved = (torch.load(adapter_initial, map_location="cpu", weights_only=True)
                     if isinstance(adapter_initial, (str, Path)) else adapter_initial)
            if saved.get("arm") != "B" or saved.get("seed") != 0 or "state_dict" not in saved:
                raise ValueError("the original B seed-0 initialization is required")
            self.adapter = CoefficientAdapter("B")
            self.adapter.load_state_dict(saved["state_dict"], strict=True)
            if (torch.count_nonzero(self.adapter.readout[-1].weight).item()
                    or torch.count_nonzero(self.adapter.readout[-1].bias).item()):
                raise ValueError("G initialization is not the zero-residual original initial state")
        else:
            if adapter_initial is not None:
                raise ValueError("native branches must use the native head initialization")
            self.coefficient_head = copy.deepcopy(head.one2one_cv4)
            if arm == "PC_native":
                self.proto = copy.deepcopy(head.proto)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train(False)
        self._frozen_buffers = {name: value.detach().cpu().clone() for name, value in self.named_buffers()}

    def train(self, mode: bool = True):
        # eval controls behavior, not autograd. Native BN/semantic behavior must
        # never change just because the outer training loop calls model.train().
        super().train(False)
        return self

    def _pyramid(self, cache: Mapping, pyramid_features) -> list[torch.Tensor]:
        if pyramid_features is None or len(pyramid_features) != 3:
            raise ValueError("the actual three-scale native F pyramid is required")
        parameter = next(self.parameters())
        features = []
        for value in pyramid_features:
            value = detached_tensor(value, device=parameter.device, dtype=torch.float32)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or value.shape[0] < 1:
                raise ValueError("F must be the nonempty original native batch")
            features.append(value)
        if len({value.shape[0] for value in features}) != 1:
            raise ValueError("native F batch sizes differ between levels")
        shapes = [list(value.shape[1:]) for value in features]
        if tuple(s[0] for s in shapes) != FEATURE_CHANNELS:
            raise ValueError("native F channels changed")
        if [list(s) for s in cache["head_feature_shapes"]] != shapes:
            raise ValueError("native F geometry differs from cache")
        return features

    def forward(self, cache: Mapping, pyramid_features=None, phase: str = "train:0", batch_index: int = 0) -> dict:
        parameter = next(self.parameters())
        p0 = detached_tensor(cache["proto"], device=parameter.device, dtype=torch.float32)
        c0 = detached_tensor(cache["c0"], device=parameter.device, dtype=torch.float32)
        if c0.ndim != 2 or c0.shape[1] != 32:
            raise ValueError("invalid native c0")
        if self.arm == "G_budget":
            # Copy only the tensor references needed to turn inference-mode
            # caches into ordinary detached tensors. No cache is mutated.
            adapter_cache = dict(cache)
            for key in ("proto", "c0", "boxes_input", "scores", "classes"):
                adapter_cache[key] = detached_tensor(cache[key], device=parameter.device)
            adapter_cache["features"] = [detached_tensor(t, device=parameter.device) for t in cache["features"]]
            delta = self.adapter(adapter_cache, phase)
            current_coefficients, current_proto = c0 + delta, p0
            _raw_ids(cache, [list(s) for s in cache["head_feature_shapes"]], len(c0), parameter.device)
        else:
            return self.forward_batch([cache], pyramid_features, [batch_index], phase)[0]
        logits, baseline, roi = mask_loss_grids(cache, current_proto, current_coefficients)
        return {
            "current_proto": current_proto, "current_coefficients": current_coefficients,
            "native_proto": p0, "native_coefficients": c0,
            "logits": logits, "baseline_logits": baseline, "roi": roi,
        }

    def forward_batch(self, caches, pyramid_features=None, image_indices=None, phase: str = "train:0") -> list[dict]:
        """Compute full native-batch C/P once; return only the selected images.

        The caller must pass the ORIGINAL full source batch, even for a selected
        microgroup of four images. Sum/mean the four per-image losses before one
        backward call. Recompute this method after every optimizer update, while
        reusing the frozen original full-batch F. There is no GT in this API.
        """
        caches = list(caches)
        if not caches:
            raise ValueError("at least one image cache is required")
        indices = list(range(len(caches))) if image_indices is None else list(image_indices)
        if len(indices) != len(caches) or len(set(indices)) != len(indices):
            raise ValueError("selected image indices must be unique and match cache count")
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("invalid native image batch index")
        if self.arm == "G_budget":
            return [self(cache, phase=phase) for cache in caches]
        features = self._pyramid(caches[0], pyramid_features)
        batch_size = features[0].shape[0]
        if max(indices) >= batch_size:
            raise ValueError("image batch index is outside original native F batch")
        shapes = [list(feature.shape[1:]) for feature in features]
        for cache in caches:
            if [list(shape) for shape in cache["head_feature_shapes"]] != shapes:
                raise ValueError("native F geometry differs from an image cache")
        all_coefficients = torch.cat([
            branch(feature).view(batch_size, 32, -1)
            for branch, feature in zip(self.coefficient_head, features)
        ], dim=2)
        # Call the real Proto26 path once on the complete native batch, avoiding
        # Segment26's O2O detach and preserving the native convolution shapes.
        all_prototypes = self.proto(features, return_semantic=False) if self.proto is not None else None
        result = []
        for cache, image_index in zip(caches, indices):
            p0 = detached_tensor(cache["proto"], device=all_coefficients.device, dtype=torch.float32)
            c0 = detached_tensor(cache["c0"], device=all_coefficients.device, dtype=torch.float32)
            raw_ids = _raw_ids(cache, shapes, len(c0), all_coefficients.device)
            coefficients = all_coefficients[image_index].transpose(0, 1).index_select(0, raw_ids)
            proto = all_prototypes[image_index] if all_prototypes is not None else p0
            logits, baseline, roi = mask_loss_grids(cache, proto, coefficients)
            result.append({
                "current_proto": proto, "current_coefficients": coefficients,
                "native_proto": p0, "native_coefficients": c0,
                "logits": logits, "baseline_logits": baseline, "roi": roi,
            })
        return result

    @staticmethod
    def loss(output: Mapping, targets: Mapping[int, torch.Tensor]):
        """Original same-ROI soft-IoU + .1 preservation, one image's mean."""
        return soft_iou_grouped_loss(output["logits"], output["baseline_logits"], output["roi"], targets)

    def parameter_manifest(self) -> dict:
        parameters = {name: {"shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype),
                             "requires_grad": p.requires_grad} for name, p in self.named_parameters()}
        buffers = {name: {"shape": list(b.shape), "dtype": str(b.dtype)} for name, b in self.named_buffers()}
        return {"arm": self.arm, "trainable_whitelist": list(parameters),
                "trainable_parameter_count": sum(v["numel"] for v in parameters.values()),
                "trainable_tensor_count": len(parameters), "parameters": parameters,
                "mutable_buffer_whitelist": [], "frozen_buffers": buffers,
                "loss_spec": copy.deepcopy(LOSS_SPEC), "module_mode": "eval_with_autograd"}

    def verify_frozen_state(self, external_model=None, external_snapshot=None) -> dict:
        if any(module.training for module in self.modules()):
            raise AssertionError("mask module mode changed from eval")
        if not all(p.requires_grad for p in self.parameters()):
            raise AssertionError("trainable whitelist requires_grad changed")
        buffers = dict(self.named_buffers())
        if set(buffers) != set(self._frozen_buffers):
            raise AssertionError("buffer keys changed")
        if any(not torch.equal(buffers[name].detach().cpu(), value) for name, value in self._frozen_buffers.items()):
            raise AssertionError("a frozen buffer changed")
        result = {"status": "passed", "frozen_buffers_checked": len(buffers), "eval_mode": True}
        if external_model is not None:
            if external_snapshot is None:
                raise ValueError("external native-state snapshot is required")
            if external_model.training or any(p.requires_grad for p in external_model.parameters()):
                raise AssertionError("native model is not frozen/eval")
            result["external_native_state"] = assert_state_unchanged(external_model, external_snapshot)
        return result

    @torch.no_grad()
    def verify_initial_replay(self, cache: Mapping, pyramid_features=None, batch_index: int = 0) -> dict:
        """Require exact P, c, and proto-grid logits; RLE replay remains caller's job."""
        output = self(cache, pyramid_features, phase="eval:0", batch_index=batch_index)
        checks = {
            "proto_equal": torch.equal(output["current_proto"], output["native_proto"]),
            "coefficients_equal": torch.equal(output["current_coefficients"], output["native_coefficients"]),
            "logits_equal": torch.equal(output["logits"], output["baseline_logits"]),
        }
        if not all(checks.values()):
            diffs = {"proto_max_abs": float((output["current_proto"] - output["native_proto"]).abs().max()),
                     "coefficients_max_abs": float((output["current_coefficients"] - output["native_coefficients"]).abs().max())
                     if len(output["current_coefficients"]) else 0.0}
            raise AssertionError(f"initial native replay failed: {checks}; {diffs}")
        return {"status": "passed", **checks, "output_rows": len(output["current_coefficients"]),
                "rle_check": "caller_must_verify_original_decoder_and_original_geometry"}
