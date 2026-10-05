"""Frozen, GT-free consensus of existing raw coefficient responses.

No model execution, label access, candidate reassignment, or learned parameters.
Boxes are original predictions in the fixed 640 input coordinates. Scores are
already sigmoid probabilities. Raw tensor rows retain their original raw IDs.
Every arm uses the caller's original prototype and final target box.
"""
from __future__ import annotations

import torch


ARMS = ("A", "SAME", "TOP", "SCORE", "CONS", "CONS_MATCH", "ONE_MATCH")
MAX_DONORS = 8
MIN_BOX_IOU = 0.75
CONSENSUS_FLOOR = 0.05
EQUALITY_ATOL = 1e-5
EQUALITY_RTOL = 1e-5
_RAW_FIELDS = frozenset({"boxes", "scores", "coefficients"})


def _finite_float(name, tensor, shape, device):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: expected {tuple(shape)}, received {tuple(tensor.shape)}")
    if tensor.device != device or tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use the prototype device and float32")
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains nonfinite values")


def _validate_raw(name, raw, device):
    # Explicitly exclude annotations, owners, GT classes/boxes and label masks.
    if not isinstance(raw, dict) or set(raw) != _RAW_FIELDS:
        raise ValueError(f"{name} must contain exactly {sorted(_RAW_FIELDS)}; no GT fields")
    if not torch.is_tensor(raw["coefficients"]) or raw["coefficients"].ndim != 2:
        raise ValueError(f"{name}.coefficients must be [R,32]")
    count = int(raw["coefficients"].shape[0])
    _finite_float(name + ".coefficients", raw["coefficients"], (count, 32), device)
    _finite_float(name + ".boxes", raw["boxes"], (count, 4), device)
    _finite_float(name + ".scores", raw["scores"], (count, 80), device)
    if bool(((raw["scores"] < 0) | (raw["scores"] > 1)).any()):
        raise ValueError(f"{name}.scores must be probabilities, not raw logits")
    return count


def _box_iou(box, boxes):
    wh = (torch.minimum(box[2:], boxes[:, 2:]) - torch.maximum(box[:2], boxes[:, :2])).clamp_min(0)
    intersection = wh.prod(1)
    area = (box[2:] - box[:2]).clamp_min(0).prod()
    other_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    union = area + other_area - intersection
    # Safe denominator without perturbing any valid IoU or the .75 boundary.
    return torch.where(union > 0, intersection / torch.where(union > 0, union, torch.ones_like(union)),
                       torch.zeros_like(union))


def _pool(raw, box, predicted_class):
    iou = _box_iou(box, raw["boxes"])
    labels = raw["scores"].argmax(1)
    eligible = torch.where((labels == predicted_class) & (iou >= MIN_BOX_IOU))[0]
    # torch.where gives ascending original IDs. Stable sorting therefore makes
    # equal-score ties deterministic by ascending raw ID.
    order = torch.argsort(raw["scores"][eligible, predicted_class], descending=True, stable=True)
    ids = eligible[order[:MAX_DONORS]]
    return dict(ids=ids, coefficients=raw["coefficients"][ids],
                scores=raw["scores"][ids, predicted_class], box_iou=iou[ids],
                n_available=int(eligible.numel()))


def _pool_summary(pool):
    return dict(n_available=pool["n_available"], n_used=int(pool["ids"].numel()),
                donor_raw_ids=pool["ids"].tolist(), scores=pool["scores"].tolist(),
                box_iou=pool["box_iou"].tolist())


def _truncate(pool, count):
    return {key: (value[:count] if torch.is_tensor(value) else value) for key, value in pool.items()}


def _agreement(pool, support_prototype):
    count = int(pool["ids"].numel())
    if count == 0:
        return pool["scores"].new_empty(0)
    if count == 1:
        return pool["scores"].new_ones(1)
    # All donors have the SAME target support, not their individual boxes.
    binary = (pool["coefficients"] @ support_prototype > 0).to(torch.float32)
    intersection = binary @ binary.T
    area = binary.sum(1)
    union = area[:, None] + area[None, :] - intersection
    pair_iou = torch.where(union > 0,
        intersection / torch.where(union > 0, union, torch.ones_like(union)),
        torch.zeros_like(union))
    # Empty/empty IoU is zero; the diagonal is never part of reliability.
    pair_iou.fill_diagonal_(0)
    return pair_iou.sum(1) / (count - 1)


def _mix(pool, unnormalized, sample_prototype):
    if not bool(torch.isfinite(unnormalized).all()) or bool((unnormalized < 0).any()):
        raise FloatingPointError("Mixture weights must be finite and nonnegative")
    denominator = unnormalized.sum()
    if not bool(denominator > 0):
        return None, [], dict(passed=None, reason="zero_total_weight")
    weights = unnormalized / denominator
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise FloatingPointError("Normalized mixture weights are invalid")
    torch.testing.assert_close(weights.sum(), weights.new_tensor(1.), atol=1e-6, rtol=1e-6)
    coefficient = (weights[:, None] * pool["coefficients"]).sum(0)
    if not bool(torch.isfinite(coefficient).all()):
        raise FloatingPointError("Convex coefficient mixture is nonfinite")
    # This is an actual FP32 operation-order check, not a float64 identity
    # substituted for the proposed FP32 arm. Up to 64 deterministic support
    # locations are checked, with the protocol's fixed tolerances.
    left = coefficient @ sample_prototype
    right = weights @ (pool["coefficients"] @ sample_prototype)
    torch.testing.assert_close(left, right, atol=EQUALITY_ATOL, rtol=EQUALITY_RTOL)
    absolute = (left - right).abs()
    return coefficient, weights.tolist(), dict(passed=True, sample_pixels=int(left.numel()),
        maximum_absolute_error=float(absolute.max()) if left.numel() else 0.,
        atol=EQUALITY_ATOL, rtol=EQUALITY_RTOL)


@torch.no_grad()
def propose(proto, c0, target_boxes, target_classes, raw_ids, many, one):
    """Return ``({arm: [N,32] coefficients}, [JSON-ready diagnostic,...])``.

    ``target_classes`` is the original one-to-one argmax prediction, never a
    GT class. ``raw_ids`` indexes the original complete branch tensors. The
    caller is responsible for supplying the frozen official candidate list;
    this function never builds a new supervised list or reads labels.
    """
    if not torch.is_tensor(proto):
        raise TypeError("proto must be a tensor")
    device = proto.device
    _finite_float("proto", proto, (32, 160, 160), device)
    if not torch.is_tensor(c0) or c0.ndim != 2:
        raise ValueError("c0 must be [N,32]")
    count = int(c0.shape[0])
    _finite_float("c0", c0, (count, 32), device)
    _finite_float("target_boxes", target_boxes, (count, 4), device)
    for name, value in (("target_classes", target_classes), ("raw_ids", raw_ids)):
        if not torch.is_tensor(value) or value.shape != (count,) or value.dtype != torch.int64:
            raise ValueError(f"{name} must be an int64 [N] tensor")
        if value.device != device:
            raise ValueError(f"{name} must use the prototype device")
    nm = _validate_raw("many", many, device)
    no = _validate_raw("one", one, device)
    if nm != no or nm == 0:
        raise ValueError("Both complete raw branches must share the nonempty raw-ID grid")
    if bool(((raw_ids < 0) | (raw_ids >= nm)).any()):
        raise IndexError("Target raw ID is outside the original branch grid")
    if bool(((target_classes < 0) | (target_classes >= 80)).any()):
        raise ValueError("Predicted classes must be COCO channel IDs in [0,79]")
    # Source identity guards make accidental list-index/raw-ID substitution
    # fail before a proposal can be interpreted.
    torch.testing.assert_close(one["coefficients"][raw_ids], c0, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(one["boxes"][raw_ids], target_boxes, atol=3e-5, rtol=3e-5)
    if not torch.equal(one["scores"][raw_ids].argmax(1), target_classes):
        raise AssertionError("Target class differs from the same original one-to-one raw ID")
    result = {arm: c0.clone() for arm in ARMS}
    diagnostics = []
    flat_prototype = proto.flatten(1)
    yy, xx = torch.meshgrid(torch.arange(160, device=device), torch.arange(160, device=device), indexing="ij")
    for index in range(count):
        box = target_boxes[index]
        cls = int(target_classes[index])
        rid = int(raw_ids[index])
        mp, op = _pool(many, box, cls), _pool(one, box, cls)
        kmatch = min(MAX_DONORS, int(mp["ids"].numel()), int(op["ids"].numel()))
        box_p = box / 4
        support = ((xx >= box_p[0]) & (xx < box_p[2]) & (yy >= box_p[1]) & (yy < box_p[3])).flatten()
        pixels = torch.where(support)[0]
        diag = dict(candidate_index=index, raw_id=rid, predicted_class=cls,
            gt_free_input_fields=True, many=_pool_summary(mp), one=_pool_summary(op),
            Kmatched=kmatch,
            n_available={"many": mp["n_available"], "one": op["n_available"]},
            n_used={"many": int(mp["ids"].numel()), "one": int(op["ids"].numel())},
            support=dict(pixels=int(pixels.numel()), prototype_box=box_p.tolist(),
                rule="160-grid integer x>=x1/4,x<x2/4,y>=y1/4,y<y2/4; logit>0"),
            arms={"A": dict(source="original_one2one", donor_raw_ids=[rid], weights=[1.])},
            fallback={}, logit_equivalence={})
        diagnostics.append(diag)
        if pixels.numel() == 0:
            diag["fallback"] = {arm: "empty_target_support" for arm in ARMS if arm != "A"}
            continue
        support_p = flat_prototype[:, pixels]
        sample_ix = torch.linspace(0, pixels.numel()-1, min(64, pixels.numel()), device=device).long()
        sample_p = support_p[:, sample_ix]
        # SAME is a pre-existing cross-head, same-raw control. It is independent
        # of eligibility in the class/IoU-filtered donor pool.
        result["SAME"][index] = many["coefficients"][rid]
        diag["arms"]["SAME"] = dict(source="one2many_same_raw", donor_raw_ids=[rid], weights=[1.])

        def apply(arm, pool, consensus):
            record = _pool_summary(pool)
            diag["arms"][arm] = record
            if pool["ids"].numel() == 0:
                diag["fallback"][arm] = "no_donor" if "MATCH" not in arm else "no_common_donor_count"
                record["weights"] = []
                return
            reliability = _agreement(pool, support_p) if consensus else torch.ones_like(pool["scores"])
            raw_weight = pool["scores"] * (CONSENSUS_FLOOR + (1-CONSENSUS_FLOOR)*reliability) if consensus else pool["scores"]
            mixed, weights, equality = _mix(pool, raw_weight, sample_p)
            record.update(weights=weights, mean_other_mask_iou=reliability.tolist() if consensus else None,
                          weight_rule="score*(.05+.95*mean_other_mask_iou)" if consensus else "score")
            diag["logit_equivalence"][arm] = equality
            if mixed is None:
                diag["fallback"][arm] = "zero_total_weight"
                return
            result[arm][index] = mixed

        if mp["ids"].numel():
            result["TOP"][index] = mp["coefficients"][0]
            diag["arms"]["TOP"] = dict(source="one2many_highest_class_score",
                donor_raw_ids=mp["ids"][:1].tolist(), scores=mp["scores"][:1].tolist(),
                box_iou=mp["box_iou"][:1].tolist(), weights=[1.], n_used=1)
        else:
            diag["fallback"]["TOP"] = "no_donor"
            diag["arms"]["TOP"] = dict(donor_raw_ids=[], weights=[], n_used=0)
        apply("SCORE", mp, False)
        apply("CONS", mp, True)
        apply("CONS_MATCH", _truncate(mp, kmatch), True)
        apply("ONE_MATCH", _truncate(op, kmatch), True)
    for arm, tensor in result.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Nonfinite result in arm {arm}")
    return result, diagnostics
