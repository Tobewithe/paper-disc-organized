"""GT-free full-canvas horizontal-flip replay. No work is done at import.

The mirror address denotes the reflected anchor grid location, not a proved
instance identity. Matching is prediction-only, within the target scale and
original predicted class, with no top-k, confidence threshold, GT, or TAL.
Every response uses the coefficient and full prototype of its own view.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


ALLOWED_FIELDS = frozenset({"input_uint8", "boxes", "c0", "proto", "raw_ids", "levels"})
GRID_SIZES = (80, 40, 20)
OFFSETS = (0, 6400, 8000)
REPLAY_ATOL = 3e-5
REPLAY_RTOL = 3e-5


def _deadline(value):
    if value is not None and time.monotonic() >= value:
        raise TimeoutError("Frozen full-canvas flip replay exceeded its fixed deadline")


def mirror_raw_id(raw_id, level):
    level, raw_id = int(level), int(raw_id)
    if level not in (0, 1, 2):
        raise ValueError("Cache levels must be zero-based P3/P4/P5 indices")
    width, offset = GRID_SIZES[level], OFFSETS[level]
    local = raw_id - offset
    if not 0 <= local < width * width:
        raise ValueError("Raw index is not in its declared pyramid level")
    row, col = divmod(local, width)
    return offset + row * width + (width - 1 - col)


def _inverse_boxes(boxes):
    result = boxes.clone()
    result[:, 0] = 640. - boxes[:, 2]
    result[:, 2] = 640. - boxes[:, 0]
    return result


def _box_ious(boxes, target):
    extent = (torch.minimum(boxes[:, 2:], target[None, 2:])
              - torch.maximum(boxes[:, :2], target[None, :2])).clamp_min(0)
    intersection = extent.prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    target_area = (target[2:] - target[:2]).clamp_min(0).prod()
    return intersection / (area + target_area - intersection).clamp_min(1e-12)


def _logits(proto, coefficient, inverse_flip=False):
    # This is the official continuous logit order, before any box crop.
    low = (coefficient.float() @ proto.float().flatten(1)).reshape(160, 160)
    high = F.interpolate(low[None, None], (640, 640), mode="bilinear",
                         align_corners=False)[0, 0]
    return high.flip(-1) if inverse_flip else high


class FlipSource:
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
        if not self.head.end2end or self.head.export or self.source.training:
            raise AssertionError("Require original eval, nonexport, one-to-one Segment26")
        self.weights_sha256 = self.replay.weights_sha256
        self.last_audit = None
        self.import_info = dict(self.replay.import_info, detector_head_forward=True,
                               source_pool="all raw at the same pyramid level and predicted class",
                               GT_used=False, TAL_recomputed=False, topk_used=False,
                               input_transform="horizontal flip of exact cached 640 canvas")

    def assert_unchanged(self):
        self.replay.assert_unchanged()
        if not self.head.end2end or self.head.export:
            raise AssertionError("Source inference branch changed")

    @torch.no_grad()
    def _forward(self, input_uint8):
        output = self.source(input_uint8.to(self.device).float()[None] / 255.)
        if not (isinstance(output, tuple) and len(output) == 2
                and isinstance(output[0], tuple) and len(output[0]) == 2
                and isinstance(output[1], dict) and "one2one" in output[1]):
            raise AssertionError("Unexpected vendored Segment26 eval return structure")
        (_, proto), branches = output
        raw = branches["one2one"]
        if not all(key in raw for key in ("boxes", "scores", "mask_coefficient", "proto", "feats")):
            raise AssertionError("Incomplete original one-to-one raw dictionary")
        if tuple(proto.shape) != (1, 32, 160, 160) or not torch.equal(proto, raw["proto"]):
            raise AssertionError("One-to-one and returned full prototype differ")
        if [tuple(x.shape[-2:]) for x in raw["feats"]] != [(80, 80), (40, 40), (20, 20)]:
            raise AssertionError("Original raw concatenation geometry changed")
        coefficients = raw["mask_coefficient"][0].T.contiguous()
        probabilities = raw["scores"][0].T.sigmoid()
        boxes = self.head._get_decode_boxes(raw)[0].T.contiguous()
        if (coefficients.shape != (8400, 32) or probabilities.shape != (8400, 80)
                or boxes.shape != (8400, 4)):
            raise AssertionError("Unexpected one-to-one raw tensor shape")
        for value in (coefficients, probabilities, boxes, proto):
            if not bool(torch.isfinite(value).all()):
                raise AssertionError("Nonfinite frozen source prediction")
        scores, classes = probabilities.max(1)
        return dict(coefficients=coefficients, boxes=boxes, classes=classes,
                    scores=scores, probabilities=probabilities, proto=proto[0])

    @staticmethod
    def _select(pool, aligned_boxes, target_box, target_class, level, fallback_id):
        lo = OFFSETS[level]
        hi = lo + GRID_SIZES[level] ** 2
        ids = torch.arange(lo, hi, device=aligned_boxes.device)
        same = pool["classes"][ids] == target_class
        choices = ids[same]
        fallback = len(choices) == 0
        if fallback:
            chosen = int(fallback_id)
        else:
            ious = _box_ious(aligned_boxes[choices], target_box)
            choices = choices[ious == ious.max()]
            # Primary max box IoU, secondary original predicted-class score,
            # final lower raw id. All values are frozen and contain no GT.
            scores = pool["probabilities"][choices, target_class]
            choices = choices[scores == scores.max()]
            chosen = int(choices.min())
        return chosen, fallback

    @staticmethod
    def _metadata(pool, aligned_boxes, source_id, target_box, target_class, level, fallback):
        cls = int(pool["classes"][source_id])
        return dict(source_raw_id=int(source_id), source_level=int(level), source_class=cls,
                    target_predicted_class=int(target_class), same_predicted_class=cls == int(target_class),
                    source_score=float(pool["probabilities"][source_id, target_class]),
                    source_argmax_score=float(pool["scores"][source_id]),
                    box_iou=float(_box_ious(aligned_boxes[source_id:source_id+1], target_box)[0]),
                    source_box=aligned_boxes[source_id].detach().cpu().tolist(), fallback=bool(fallback),
                    fallback_reason="no same-predicted-class candidate at target scale" if fallback else None,
                    source_coefficient=pool["coefficients"][source_id])

    @torch.no_grad()
    def infer(self, payload, deadline_monotonic=None):
        if set(payload) != ALLOWED_FIELDS:
            raise ValueError(f"GT-free payload fields must be exactly {sorted(ALLOWED_FIELDS)}")
        x = payload["input_uint8"]
        if x.dtype != torch.uint8 or tuple(x.shape) != (3, 640, 640):
            raise ValueError("Require exact original cached uint8 RGB canvas")
        ids = payload["raw_ids"].to(self.device, dtype=torch.long)
        levels = payload["levels"].to(self.device, dtype=torch.long)
        c0 = payload["c0"].to(self.device, dtype=torch.float32)
        proto = payload["proto"].to(self.device, dtype=torch.float32)
        boxes = payload["boxes"].to(self.device, dtype=torch.float32)
        n = len(ids)
        if (tuple(ids.shape) != (n,) or levels.shape != ids.shape or c0.shape != (n, 32)
                or boxes.shape != (n, 4) or proto.shape != (32, 160, 160)):
            raise AssertionError("GT-free frozen payload shape differs")
        for rid, level in zip(ids.tolist(), levels.tolist()):
            if mirror_raw_id(mirror_raw_id(rid, level), level) != rid:
                raise AssertionError("Mirror raw mapping is not involutive")
        _deadline(deadline_monotonic)
        self.assert_unchanged()
        flipped = x.flip(-1)
        pixel_exact = bool(torch.equal(flipped.flip(-1), x))
        if not pixel_exact:
            raise AssertionError("Horizontal flip is not pixel-exactly invertible")
        original = self._forward(x)
        errors = {}
        for name, actual, cached in (("c0", original["coefficients"][ids], c0),
                                     ("proto", original["proto"], proto),
                                     ("boxes", original["boxes"][ids], boxes)):
            torch.testing.assert_close(actual, cached, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
            errors[name] = float((actual - cached).abs().max()) if actual.numel() else 0.
        _deadline(deadline_monotonic)
        other = self._forward(flipped)
        aligned = _inverse_boxes(other["boxes"])
        self.last_audit = dict(passed=True, flip_twice_pixel_exact=pixel_exact,
                               original_replay_max_abs=errors, replay_atol=REPLAY_ATOL,
                               replay_rtol=REPLAY_RTOL, n_candidates=n, full_raw_count=8400,
                               original_forward_count=1, flipped_forward_count=1,
                               mirror_correspondence="anchor-grid address only, not guaranteed instance identity",
                               matches_within_scale=True, GT_used=False, topk_used=False)
        for k in range(n):
            _deadline(deadline_monotonic)
            rid, level = int(ids[k]), int(levels[k])
            cls = int(original["classes"][rid])
            mirror = mirror_raw_id(rid, level)
            match, fallback = self._select(other, aligned, boxes[k], cls, level, mirror)
            omatch, ofallback = self._select(original, original["boxes"], boxes[k], cls, level, rid)
            mmeta = self._metadata(other, aligned, mirror, boxes[k], cls, level, False)
            metadata = self._metadata(other, aligned, match, boxes[k], cls, level, fallback)
            ometa = self._metadata(original, original["boxes"], omatch, boxes[k], cls, level, ofallback)
            yield dict(candidate_index=k, raw_id=rid, level=level, c0=c0[k], proto=proto,
                       original_predicted_class=cls, original_predicted_score=float(original["scores"][rid]),
                       baseline_logits=_logits(proto, c0[k]),
                       mirror_logits=_logits(other["proto"], other["coefficients"][mirror], True),
                       match_logits=_logits(other["proto"], other["coefficients"][match], True),
                       original_match_logits=_logits(proto, original["coefficients"][omatch]),
                       original_match_coefficients=original["coefficients"][omatch],
                       mirror=mmeta, match=metadata, original_match=ometa)
        self.assert_unchanged()
        _deadline(deadline_monotonic)

