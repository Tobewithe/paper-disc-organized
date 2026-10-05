"""Read the existing frozen Proto26 semantic logits without a training forward.

One original full forward supplies raw one-to-one predictions and the actual
input of Proto26.feat_fuse. The existing semseg branch reads that same tensor
in eval mode. No GT, assignment, score filtering, new module or optimizer is
used. Semantic class information does not identify same-class instances.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

REQUIRED_FIELDS = frozenset({"input_uint8", "c0", "proto", "boxes", "raw_ids", "levels"})
OPTIONAL_FIELDS = frozenset({"predicted_classes", "predicted_scores"})
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
GRID_SIZES = (80, 40, 20)
OFFSETS = (0, 6400, 8000)
REPLAY_ATOL = REPLAY_RTOL = 3e-5


def _deadline(value):
    if value is not None and time.monotonic() >= value:
        raise TimeoutError("Frozen native semantic replay exceeded its fixed deadline")


def _baseline_logits(prototype, coefficient):
    low = (coefficient.float() @ prototype.float().flatten(1)).reshape(160, 160)
    return F.interpolate(low[None, None], (640, 640), mode="bilinear",
                         align_corners=False)[0, 0]


class SemanticSource:
    """Frozen native semantic response source; no computation on import."""

    def __init__(self, cfg, device="cuda", replay=None):
        self.cfg = dict(cfg)
        scripts = Path(cfg.get("source_script_dir", Path(cfg.get("original_screen_root",
                       cfg.get("screen_root", "/root/prototype_readout_fast_screen_20261003"))) / "scripts"))
        if str(scripts) not in sys.path:
            sys.path.append(str(scripts))
        from online_runtime import FrozenReplay
        self.replay = replay if replay is not None else FrozenReplay(cfg, device=device)
        self.source, self.head = self.replay.source, self.replay.head
        self.device = self.replay.device
        self.proto_module = self.head.proto
        self.semseg = getattr(self.proto_module, "semseg", None)
        if self.semseg is None or not hasattr(self.proto_module, "feat_fuse"):
            raise AssertionError("Require unfused original Proto26 with its existing semantic branch")
        if not self.head.end2end or self.head.export or self.source.training:
            raise AssertionError("Require original eval, nonexport, one-to-one Segment26")
        self.weights_sha256 = self.replay.weights_sha256
        self.last_audit = None
        self.import_info = dict(self.replay.import_info, detector_head_forward=True,
            input_transform="none; exact original cached 640 RGB canvas",
            semantic_source="existing unfused Proto26.semseg at captured feat_fuse input",
            semantic_train_mode_used=False, GT_used=False, TAL_recomputed=False,
            class_source="same frozen one-to-one raw candidate argmax classification logit",
            semantic_upsample="selected unique class channels bilinear to 640, align_corners=False",
            agnostic_reduction="maximum across 80 classes at native 80 grid, then bilinear to 640",
            topk_selection_used=False, confidence_filter_used=False)
        self.assert_unchanged()

    def assert_unchanged(self):
        self.replay.assert_unchanged()
        if not self.head.end2end or self.head.export or self.source.training:
            raise AssertionError("Frozen source mode or inference branch changed")
        if self.head.proto is not self.proto_module or self.proto_module.semseg is not self.semseg:
            raise AssertionError("Original prototype or semantic module identity changed")
        if any(module.training for module in self.semseg.modules()):
            raise AssertionError("Semantic response must be computed in original eval mode")
        if any(p.requires_grad or p.grad is not None for p in self.semseg.parameters()):
            raise AssertionError("Frozen semantic parameters received an update path")

    @torch.no_grad()
    def _forward(self, image):
        captured = []
        def capture(module, args):
            if len(args) != 1 or not torch.is_tensor(args[0]):
                raise AssertionError("Unexpected Proto26 feature-fusion input")
            captured.append(args[0].detach())
        hook = self.proto_module.feat_fuse.register_forward_pre_hook(capture)
        try:
            output = self.source(image.to(self.device).float()[None] / 255.)
        finally:
            hook.remove()
        if not (isinstance(output, tuple) and len(output) == 2
                and isinstance(output[0], tuple) and len(output[0]) == 2
                and isinstance(output[1], dict) and "one2one" in output[1]):
            raise AssertionError("Unexpected vendored Segment26 eval return structure")
        (_, prototype), branches = output
        raw = branches["one2one"]
        if not all(key in raw for key in ("boxes", "scores", "mask_coefficient", "proto", "feats")):
            raise AssertionError("Incomplete original one-to-one raw dictionary")
        if tuple(prototype.shape) != (1, 32, 160, 160) or not torch.equal(prototype, raw["proto"]):
            raise AssertionError("Returned and one-to-one complete prototypes differ")
        if [tuple(f.shape[-2:]) for f in raw["feats"]] != [(80, 80), (40, 40), (20, 20)]:
            raise AssertionError("Original raw concatenation geometry changed")
        if len(captured) != 1:
            raise AssertionError("Need exactly one original Proto26 fused-feature observation")
        feat = captured[0]
        if feat.ndim != 4 or feat.shape[0] != 1 or tuple(feat.shape[-2:]) != (80, 80):
            raise AssertionError("Expected original 80x80 Proto26 fusion input")
        # Capture is AFTER the native F3 + refined/upsampled F4/F5 sum and
        # BEFORE feat_fuse. This avoids a second backbone or fusion replay,
        # and exactly follows the source of the official semantic output.
        semantic = self.semseg(feat)
        if tuple(semantic.shape) != (1, 80, 80, 80):
            raise AssertionError("Expected existing 80-class, 80x80 native semantic logits")
        coefficients = raw["mask_coefficient"][0].T.contiguous()
        probabilities = raw["scores"][0].T.sigmoid()
        boxes = self.head._get_decode_boxes(raw)[0].T.contiguous()
        if coefficients.shape != (8400, 32) or probabilities.shape != (8400, 80) or boxes.shape != (8400, 4):
            raise AssertionError("Unexpected full one-to-one raw tensor shape")
        for value in (prototype, coefficients, probabilities, boxes, semantic, feat):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Nonfinite frozen original/semantic response")
        scores, classes = probabilities.max(1)
        return dict(proto=prototype[0], coefficients=coefficients, boxes=boxes,
                    classes=classes, scores=scores, semantic=semantic,
                    semantic_native_shape=list(semantic.shape), feature_shape=list(feat.shape))

    @torch.no_grad()
    def infer(self, payload, deadline_monotonic=None):
        fields = set(payload)
        if not REQUIRED_FIELDS.issubset(fields) or not fields.issubset(ALLOWED_FIELDS):
            raise ValueError(f"GT-free source requires {sorted(REQUIRED_FIELDS)}; optional {sorted(OPTIONAL_FIELDS)}")
        image = payload["input_uint8"]
        if image.dtype != torch.uint8 or tuple(image.shape) != (3, 640, 640):
            raise ValueError("Require exact original cached uint8 RGB canvas")
        for key in ("raw_ids", "levels"):
            if payload[key].dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{key} must retain its integer identity dtype")
        ids = payload["raw_ids"].to(self.device, dtype=torch.long)
        levels = payload["levels"].to(self.device, dtype=torch.long)
        c0 = payload["c0"].to(self.device, dtype=torch.float32)
        prototype = payload["proto"].to(self.device, dtype=torch.float32)
        boxes = payload["boxes"].to(self.device, dtype=torch.float32)
        n = len(ids)
        if ids.shape != (n,) or levels.shape != (n,) or c0.shape != (n, 32) or boxes.shape != (n, 4) or prototype.shape != (32, 160, 160):
            raise AssertionError("Compact same-raw frozen payload shape mismatch")
        if len(set(ids.tolist())) != n:
            raise AssertionError("Duplicate raw candidate identity in one image")
        for rid, level in zip(ids.tolist(), levels.tolist()):
            if level not in (0, 1, 2) or not OFFSETS[level] <= rid < OFFSETS[level] + GRID_SIZES[level] ** 2:
                raise AssertionError("Raw candidate address does not belong to the declared pyramid level")
        _deadline(deadline_monotonic)
        self.assert_unchanged()
        original = self._forward(image)
        errors = {}
        for name, actual, cached in (("c0", original["coefficients"][ids], c0),
                                     ("proto", original["proto"], prototype),
                                     ("boxes", original["boxes"][ids], boxes)):
            torch.testing.assert_close(actual, cached, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
            errors[name] = float((actual-cached).abs().max()) if actual.numel() else 0.
        if "predicted_classes" in payload:
            cached = payload["predicted_classes"].to(self.device)
            if cached.shape != (n,) or cached.dtype not in (torch.int32, torch.int64) or not torch.equal(cached.long(), original["classes"][ids]):
                raise AssertionError("Original raw predicted class does not reproduce its cached identity")
        if "predicted_scores" in payload:
            cached = payload["predicted_scores"].to(self.device, dtype=torch.float32)
            if cached.shape != (n,):
                raise AssertionError("Original raw predicted score cache shape changed")
            torch.testing.assert_close(original["scores"][ids], cached, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
            errors["predicted_scores"] = float((original["scores"][ids]-cached).abs().max()) if n else 0.
        _deadline(deadline_monotonic)
        selected_classes = sorted(set(original["classes"][ids].detach().cpu().tolist()))
        semantic_full = {}
        for cls in selected_classes:
            _deadline(deadline_monotonic)
            semantic_full[cls] = F.interpolate(original["semantic"][:, cls:cls+1], (640, 640),
                mode="bilinear", align_corners=False)[0, 0]
        # Frozen protocol: native-grid max THEN interpolation. Max and linear
        # interpolation do not commute; this order is not selected on results.
        agnostic_native = original["semantic"].max(dim=1, keepdim=True).values
        agnostic_full = F.interpolate(agnostic_native, (640, 640), mode="bilinear",
                                     align_corners=False)[0, 0]
        if any(not torch.isfinite(value).all() for value in semantic_full.values()) or not torch.isfinite(agnostic_full).all():
            raise FloatingPointError("Nonfinite full-resolution semantic response")
        self.assert_unchanged()
        self.last_audit = dict(passed=True, original_replay_max_abs=errors,
            replay_atol=REPLAY_ATOL, replay_rtol=REPLAY_RTOL, n_candidates=n, full_raw_count=8400,
            original_forward_count=1, semantic_forward_count=1, repeated_backbone_forward=False,
            semantic_native_shape=original["semantic_native_shape"], semantic_feature_shape=original["feature_shape"],
            semantic_full_shape=[len(selected_classes), 640, 640], semantic_train_mode_used=False,
            upsampled_class_ids=selected_classes, original_semantic_channel_count=80,
            semantic_feature_source="captured original Proto26.feat_fuse input",
            upsample=dict(mode="bilinear", align_corners=False, target=[640, 640]),
            agnostic_reduction="pixelwise max over all 80 native 80-grid class logits BEFORE bilinear upsampling",
            cached_predicted_classes_checked="predicted_classes" in payload,
            cached_predicted_scores_checked="predicted_scores" in payload,
            predicted_classes=original["classes"][ids].detach().cpu().tolist(),
            predicted_scores=original["scores"][ids].detach().cpu().tolist(),
            source_buffers_unchanged=True, semantic_parameters_frozen=True,
            GT_used=False, TAL_recomputed=False, topk_selection_used=False, confidence_filter_used=False)
        for k in range(n):
            _deadline(deadline_monotonic)
            rid = int(ids[k]); cls = int(original["classes"][rid])
            yield dict(candidate_index=k, raw_id=rid, level=int(levels[k]),
                predicted_class=cls, predicted_score=float(original["scores"][rid]),
                baseline_logits=_baseline_logits(prototype, c0[k]),
                semantic_logits=semantic_full[cls], agnostic_logits=agnostic_full)
        self.assert_unchanged()
        _deadline(deadline_monotonic)

