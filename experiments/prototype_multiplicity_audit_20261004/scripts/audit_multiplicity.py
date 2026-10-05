"""Zero-training, same-image GT-assisted native-prototype sensitivity audit.

Every image starts at the identical official parameter point. M uses the
official one-to-many candidate mean; E changes only within-image GT weights;
O is the one-to-one mask objective differentiated through P for diagnosis.
The real official one-to-one path detaches P. O is NOT its training gradient.
No optimizer, accumulated update, method checkpoint, or held-out claim exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ARMS = ("A", "M", "E", "O")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
PAIRS = (("M", "A"), ("E", "A"), ("O", "A"), ("E", "M"), ("O", "M"))
KEYS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
ATOL = RTOL = 3e-5


def identity(row):
    return tuple(row[key] for key in KEYS)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def cosine(a, b):
    den = float(a.double().norm() * b.double().norm())
    return float((a.double() * b.double()).sum()) / den if den > 0 else None


def vector(parameters):
    import torch
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def gradient_vector(grads, parameters):
    import torch
    if any(g is None for g in grads):
        raise AssertionError("An active native prototype parameter is absent from the mask graph")
    if any(not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("Nonfinite native prototype parameter gradient")
    return torch.cat([g.detach().reshape(-1) for g, _ in zip(grads, parameters)])


def weights_for(owners, equal_gt):
    import torch
    n = len(owners)
    if not n:
        return torch.empty(0, device=owners.device, dtype=torch.float32)
    _, inverse, counts = torch.unique(owners, sorted=True, return_inverse=True, return_counts=True)
    w = (1. / (len(counts) * counts[inverse].float())) if equal_gt else torch.full(
        (n,), 1. / n, device=owners.device, dtype=torch.float32)
    torch.testing.assert_close(w.sum(), w.new_tensor(1.), atol=1e-6, rtol=0)
    if equal_gt:
        totals = w.new_zeros(len(counts)).index_add_(0, inverse, w)
        torch.testing.assert_close(totals, torch.full_like(totals, 1. / len(counts)), atol=1e-6, rtol=0)
    return w


def losses_and_output_gradients(image, support, prototype, criterion, equal_gt=False,
                               gradients=True, verify=False, individual_check=False,
                               check_time=lambda: None):
    """Exact 640-grid official per-candidate losses, with optional E reweight.

    The reference uses calculate_segmentation_loss on chunks, multiplied by
    chunk_n / total_n. This is its exact candidate-mean decomposition, not an
    approximation or a changed support. Gradient comparison is with dL/dP160.
    """
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import ops
    from ultralytics.utils.loss import v8SegmentationLoss
    dev = prototype.device
    c = support["c0"].detach().to(dev, dtype=torch.float32)
    owners = support["owners"].detach().to(dev).long()
    boxes = support["target_boxes"].detach().to(dev, dtype=torch.float32)
    masks = image["masks"].to(dev)
    n = len(c)
    if tuple(c.shape) != (n, 32) or boxes.shape != (n, 4) or len(owners) != n:
        raise AssertionError("Compact coefficient, owner and target-box identities differ")
    if tuple(masks.shape) != (640, 640) or tuple(prototype.shape) != (32, 160, 160):
        raise AssertionError("Require original full-resolution overlap labels and complete prototype")
    if not criterion.overlap:
        raise AssertionError("Frozen original dataset uses overlap masks")
    gain = float(image["segmentation_gain"])
    w_m = weights_for(owners, False)
    w_e = weights_for(owners, True) if equal_gt else w_m
    if not n:
        return dict(values=[], M=0., E=0., gradient_M=torch.zeros_like(prototype),
                    gradient_E=torch.zeros_like(prototype), weight_M=w_m, weight_E=w_e,
                    audit=dict(candidates=0, objective_defined=False))
    leaf = prototype.detach().float().requires_grad_(gradients)
    total_m = total_e = reference_total = 0.
    gm, ge, reference_gradient = (torch.zeros_like(prototype) for _ in range(3))
    values = []
    per_error = 0.
    # A fresh interpolation graph per chunk bounds memory and avoids retaining
    # all pixel graphs. Linear interpolation makes the sum of dP exact.
    for lo in range(0, n, 8):
        check_time()
        hi = min(n, lo + 8)
        expanded = F.interpolate(leaf[None], (640, 640), mode="bilinear", align_corners=False)[0]
        gt = (masks[None] == (owners[lo:hi] + 1)[:, None, None]).float()
        target_b = boxes[lo:hi]
        normalized_b = target_b / target_b.new_tensor([640., 640., 640., 640.])
        # Match official calculate_segmentation_loss's operations, including
        # the normalization round trip at integer pixel crop boundaries.
        b = normalized_b * target_b.new_tensor([640, 640, 640, 640])
        area = ops.xyxy2xywh(normalized_b)[..., 2:].prod(1)
        if bool((area <= 0).any()):
            raise AssertionError("Official positive has nonpositive target-box area")
        z = torch.einsum("in,nhw->ihw", c[lo:hi], expanded)
        vec = ops.crop_mask(F.binary_cross_entropy_with_logits(z, gt, reduction="none"), b).mean((1, 2)) / area * gain
        if not torch.isfinite(vec).all():
            raise FloatingPointError("Nonfinite official per-candidate mask loss")
        vm, ve = (vec * w_m[lo:hi]).sum(), (vec * w_e[lo:hi]).sum()
        total_m += float(vm.detach()); total_e += float(ve.detach())
        values.extend(vec.detach().cpu().double().tolist())
        if individual_check:
            with torch.no_grad():
                for k in range(hi - lo):
                    ref = v8SegmentationLoss.single_mask_loss(
                        gt[k:k+1], c[lo+k:lo+k+1], expanded.detach(), b[k:k+1], area[k:k+1]) * gain
                    torch.testing.assert_close(vec[k].detach(), ref, atol=ATOL, rtol=RTOL)
                    per_error = max(per_error, abs(float(vec[k].detach()) - float(ref)))
        if gradients:
            gm.add_(torch.autograd.grad(vm, leaf, retain_graph=(equal_gt or verify))[0])
            if equal_gt:
                ge.add_(torch.autograd.grad(ve, leaf, retain_graph=verify)[0])
            else:
                ge.copy_(gm)
        if verify:
            # All tensors are the SAME compact candidates in the same order.
            # calculate_segmentation_loss needs neither scores nor assignment.
            ref = criterion.calculate_segmentation_loss(
                torch.ones((1, hi-lo), device=dev, dtype=torch.bool), masks[None],
                owners[lo:hi][None], target_b[None],
                torch.zeros((len(image["all_annotation_ids"]), 1), device=dev),
                expanded[None], c[lo:hi][None], expanded.new_tensor([640., 640.])
            ) * gain * ((hi-lo) / n)
            reference_total += float(ref.detach())
            if gradients:
                reference_gradient.add_(torch.autograd.grad(ref, leaf)[0])
        del expanded, gt, z, vec, vm, ve
    audit = dict(candidates=n, objective_defined=True, scalar_reference_checked=verify,
                 prototype_gradient_reference_checked=bool(verify and gradients),
                 per_candidate_scalar_checked=individual_check,
                 per_candidate_scalar_max_absolute_error=per_error if individual_check else None,
                 atol=ATOL, rtol=RTOL, reference_chunk=8)
    if verify:
        torch.testing.assert_close(prototype.new_tensor(total_m), prototype.new_tensor(reference_total), atol=ATOL, rtol=RTOL)
        audit.update(total=total_m, reference_total=reference_total,
                     total_absolute_error=abs(total_m-reference_total))
        if gradients:
            torch.testing.assert_close(gm, reference_gradient, atol=ATOL, rtol=RTOL)
            audit["gradient_max_absolute_error"] = float((gm-reference_gradient).abs().max())
            audit["gradient_relative_l2_error"] = float((gm-reference_gradient).double().norm() /
                reference_gradient.double().norm().clamp_min(1e-30))
    if not torch.isfinite(gm).all() or not torch.isfinite(ge).all():
        raise FloatingPointError("Nonfinite gradient with respect to the full native prototype")
    return dict(values=values, M=total_m, E=total_e, gradient_M=gm.detach(), gradient_E=ge.detach(),
                weight_M=w_m, weight_E=w_e, audit=audit)


def bootstrap_mean(values, draws, seed=20261004):
    import numpy as np
    x = np.asarray([v for v in values if finite(v)], dtype=np.float64)
    if not len(x):
        return dict(mean=None, ci95=[None, None], images=0)
    rng = np.random.default_rng(seed)
    means = []
    for lo in range(0, draws, 128):
        ix = rng.integers(0, len(x), (min(128, draws-lo), len(x)))
        means.extend(x[ix].mean(1).tolist())
    return dict(mean=float(x.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), images=len(x))


def summarize(rows, image_rows, cfg, out, population, audit, em, dump):
    draws = int(cfg.get("bootstrap", 1000))
    em.ARMS, em.PAIRS, em.METRICS = ARMS, PAIRS, METRICS
    predicates = {"all": lambda r: True, "box_good_mask_bad": lambda r: r["box_good_mask_bad"],
                  "original_success": lambda r: bool(r["mask75_A"]),
                  "original_failure": lambda r: not bool(r["mask75_A"])}
    for level in range(3):
        predicates[f"P{level+3}"] = lambda r, level=level: int(r["pyramid_level"]) == level
    tables = {name: em.make_table([r for r in rows if pred(r)], 20261004, draws,
                                 "same-image GT-assisted local sensitivity") for name, pred in predicates.items()}
    geometry = {}
    for field in ("parameter_cos_M_E", "parameter_cos_M_O", "parameter_cos_E_O",
                  "parameter_alignment_E_minus_M", "output_cos_M_E", "output_cos_M_O",
                  "output_cos_E_O", "output_alignment_E_minus_M"):
        geometry[field] = bootstrap_mean([r.get(field) for r in image_rows], draws)
    align = geometry["parameter_alignment_E_minus_M"]
    all_delta = tables["all"]["comparisons"]["E_minus_M"]["iou"]["image_macro"]
    target_delta = tables["box_good_mask_bad"]["comparisons"]["E_minus_M"]["iou"]["image_macro"]
    directional = (finite(align["mean"]) and finite(align["ci95"][0]) and
                   align["mean"] >= float(cfg.get("direction_minimum", .05)) and align["ci95"][0] > 0)
    guard = finite(all_delta["delta"]) and all_delta["delta"] >= float(cfg.get("iou_guard_all", -.0005))
    utility = ((finite(all_delta["delta"]) and all_delta["delta"] >= float(cfg.get("iou_effect_all", .001))) or
               (finite(target_delta["delta"]) and target_delta["delta"] >= float(cfg.get("iou_effect_target", .003)) and guard))
    faithful_count = sum(bool(r["finite_difference_faithful"]) for r in image_rows)
    faithful_fraction = faithful_count / len(image_rows) if image_rows else 0.
    faithful = bool(image_rows) and faithful_fraction >= float(cfg["finite_difference_min_image_fraction"])
    multiplicity_varied = sum(len(set(r["many_gt_multiplicities"])) > 1 for r in image_rows)
    activated = bool(multiplicity_varied and align["images"])
    decision = dict(direction_signal=bool(directional), local_task_signal=bool(utility),
                    local_all_guard=bool(guard), finite_difference_interpretable=faithful,
                    finite_difference_joint_faithful_images=faithful_count,
                    finite_difference_joint_faithful_fraction=faithful_fraction,
                    finite_difference_joint_minimum_fraction=float(cfg["finite_difference_min_image_fraction"]),
                    mechanism_activated=activated,
                    multiplicity_varied_images=multiplicity_varied,
                    multiplicity_varied_fraction=multiplicity_varied / len(image_rows) if image_rows else 0.,
                    diagnostic_signal=bool(directional and utility and guard and faithful and activated),
                    outcome=("mechanism_not_activated" if not activated else
                             "inconclusive_local_linearization" if not faithful else
                             "local_diagnostic_signal" if directional and utility and guard else
                             "stop_current_multiplicity_hypothesis"),
                    automatic_training=False, automatic_followup=False,
                    interpretation="Only same-image GT-assisted local sensitivity; no learned or held-out method efficacy")
    result = dict(schema="prototype-multiplicity-local-audit-v1", arms=ARMS, population=population,
                  gradient_geometry=geometry, tables=tables, decision=decision, audit=audit,
                  bootstrap=dict(draws=draws, unit="paired image", seed=20261004),
                  multiplicity_histogram=dict(Counter(str(v) for r in image_rows for v in r["many_gt_multiplicities"])),
                  same_image_GT_assisted=True, held_out=False, no_training=True, no_COCO_AP=True)
    dump(out / "SUMMARY.json", em.clean(result))
    def pp(v):
        return "undefined" if v is None else f"{v * 100:+.4f}"
    lines = ["# Prototype multiplicity: local sensitivity audit", "",
             "This is a same-image GT-assisted diagnostic. Each image starts from the same original parameters; three equal-norm temporary negative-gradient displacements are evaluated and then restored. No model is trained or selected.", "",
             f"Planned images: {population['planned_images']}; effective one-to-one images: {population['effective_images']}; evaluated one-to-one candidates: {population['candidates']}.", "",
             "| Group | Comparison | Macro IoU delta [95% CI], pp | Repair/damage | Coverage delta, pp | AUC delta, pp | FPR delta, pp |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, table in tables.items():
        for comp, value in table["comparisons"].items():
            d = value["iou"]["image_macro"]; x = value["mask75_transition"]
            lines.append(f"| {name} | {comp} | {pp(d['delta'])} [{pp(d['ci95'][0])}, {pp(d['ci95'][1])}] | {x['repair']}/{x['damage']} | {pp(value['coverage']['image_macro']['delta'])} | {pp(value['auc']['image_macro']['delta'])} | {pp(value['fpr']['image_macro']['delta'])} |")
    lines += ["", "M: official one-to-many positive-candidate mean. E: the identical coefficients, owners and pixel losses, averaged within GT and then equally across represented GTs. O: frozen one-to-one coefficient loss artificially differentiated through P as a diagnostic direction (the original forward detaches P).", "",
              "Parameters, source features, coefficients, boxes, labels and assignments are restored/frozen. BN affine parameters and running buffers are both frozen; unused semseg is frozen. M is only the official many mask term, not the full original prototype-training gradient including semantic supervision and upstream paths. Normal-image evaluation always uses original one-to-one coefficients and boxes with the temporarily changed complete prototype. The variant helper's A metrics are discarded.", "",
              "Parameter- and prototype-output-space gradient cosines are reported separately. Neither negative cosine nor local improvement identifies the history of pretraining, a universal root cause, or generalization. M/E use all their own official many positives; different represented GTs from O are disclosed.", "",
              "Fixed diagnostic decision: " + json.dumps(decision, ensure_ascii=False), ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run(args):
    started = time.monotonic()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if os.name == "nt":
        raise RuntimeError("No model execution on the local desktop; Linux CUDA server only")
    sys.path.insert(0, cfg["source_python"])
    import torch
    from pycocotools.coco import COCO
    import evaluation_metrics as em
    from online_runtime import FrozenReplay, dump, load_asset, load_json, load_index, sha256, tensor_sha
    from native_proto_model import NativeProtoModel
    from assignment_replay import AssignmentReplay
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on the authorized server")
    if float(cfg["parameter_relative_step"]) != 1e-4:
        raise AssertionError("Only the predeclared relative parameter step 1e-4 is allowed")
    if tuple(cfg["arms"]) != ARMS or not cfg.get("no_training"):
        raise AssertionError("Frozen zero-training audit configuration changed")
    for key, expected in {"finite_difference_atol": 1e-6, "finite_difference_rtol": .10,
            "finite_difference_floor": 1e-7, "finite_difference_min_image_fraction": .90,
            "finite_difference_gate_objective": "O"}.items():
        if cfg.get(key) != expected:
            raise AssertionError(f"Predeclared finite-difference rule changed: {key}")
    deadline = started + float(cfg["smoke_max_seconds"] if args.smoke else cfg["max_evaluation_seconds"])
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed audit budget exceeded; do not analyze a truncated population")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("COMPLETE.json", "PER_CANDIDATE.jsonl", "INPUT_MANIFEST.json")):
        raise RuntimeError("Existing outputs are immutable; use an independent Run")
    if not args.smoke:
        receipt = load_json(Path(cfg["smoke_run"]) / "COMPLETE.json")
        if receipt.get("completed") is not True or receipt.get("passed") is not True:
            raise AssertionError("Official loss/replay smoke has not passed")
    planned = list(map(int, load_json(cfg["split"])["fit"]))
    if len(planned) != int(cfg["planned_images"]) or len(set(planned)) != len(planned):
        raise AssertionError("Fixed planned image list changed")
    index = {int(e["image_id"]): e for e in load_index(cfg)["fit"]}
    if not set(planned).issubset(index):
        raise AssertionError("Audit images must be from the original frozen fit assets")
    if args.smoke:
        selected = [iid for iid in planned if index[iid]["n"]][:2]
        if len(selected) != 2:
            raise AssertionError("Smoke needs the first two effective images")
    else:
        selected = planned
    replay = FrozenReplay(cfg)
    model = NativeProtoModel(replay, "P").to(replay.device).float().eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(False)
            module.eval()
    names = model.trainable_parameter_names()
    parameters = [p for p in model.parameters() if p.requires_grad]
    counts = model.parameter_counts()
    counts["all_frozen"] = counts["total"] - counts["trainable"]
    counts["semseg_frozen"] = sum(p.numel() for p in model.native_proto.semseg.parameters())
    counts["BN_affine_frozen_mask_path"] = sum(p.numel() for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not name.startswith("native_proto.semseg")
        for p in module.parameters(recurse=False))
    original = [p.detach().clone() for p in parameters]
    theta0 = vector(parameters)
    theta_norm = float(theta0.double().norm())
    eta = 1e-4 * theta_norm
    if eta <= 0 or not math.isfinite(eta):
        raise AssertionError("Invalid finite parameter displacement scale")
    initial_state = {name: tensor_sha(value) for name, value in model.state_dict().items()}
    manifest = dict(config_sha256=sha256(args.config), split_sha256=sha256(cfg["split"]),
                    original_weights_sha256=replay.weights_sha256,
                    cache_identity_sha256=sha256(Path(cfg["cache"]) / "CACHE_IDENTITY.json"),
                    cache_index_sha256=sha256(Path(cfg["cache"]) / "INDEX.json"),
                    code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    selected_images=selected, original_parameter_norm=theta_norm,
                    requested_displacement_norm=eta, trainable_parameter_names=names,
                    parameter_counts=counts, initial_state_sha256=initial_state)
    dump(out / "INPUT_MANIFEST.json", manifest); dump(out / "ENVIRONMENT.json", replay.import_info)
    loader = AssignmentReplay(replay, cfg, selected, out)
    check_time()
    coco = COCO(cfg["annotations_train"])
    em.ARMS = ARMS
    rows, image_rows, loss_audits, assignment_audits = [], [], [], []
    no_positive = []; no_positive_records = []
    seen = set()
    def restore():
        with torch.no_grad():
            for parameter, saved in zip(parameters, original):
                parameter.copy_(saved)
        if any(not torch.equal(p.detach(), saved) for p, saved in zip(parameters, original)):
            raise AssertionError("Original native prototype parameters did not restore exactly")
        model.assert_frozen_buffers()
    for position, iid in enumerate(selected):
        check_time(); restore()
        cached = load_asset(cfg, iid, verify=True)
        packet = loader.get(iid, cached)
        check_time()
        x, features, many = packet["image"], packet["features"], packet["many"]
        assignment_audits.append({"image_id": iid, **packet.get("audit", {})})
        if not len(x["raw_ids"]):
            no_positive.append(iid)
            no_positive_records.append(dict(image_id=iid, many_candidates=len(many["raw_ids"]),
                many_GT_count=len(set(map(int, many["owners"].tolist()))), many_replayed=True,
                local_steps_evaluated=False))
            continue
        expected = [identity(r) for r in x["rows"]]
        if len(set(expected)) != len(expected) or seen.intersection(expected):
            raise AssertionError("Original one-to-one candidate identity repeated")
        gpu = dict(x, proto=x["proto"].to(replay.device), c0=x["c0"].to(replay.device),
                   boxes=x["boxes"].to(replay.device))
        base_rows = em.evaluate_image(gpu, {"A": gpu["c0"]}, coco, chunk_size=4)
        if [identity(r) for r in base_rows] != expected:
            raise AssertionError("Normal baseline decoding changed candidate identities")
        result = model(features, [x])
        live_p = result["prototypes"][0]
        torch.testing.assert_close(live_p.detach(), gpu["proto"], atol=ATOL, rtol=RTOL)
        prototype_replay_error = float((live_p.detach()-gpu["proto"]).abs().max())
        torch.testing.assert_close(result["coefficients"][0], gpu["c0"], atol=0, rtol=0)
        # Require the native parameter copy to reproduce all baseline masks,
        # not just pass a broad tensor tolerance near the zero threshold.
        copied_rows = em.evaluate_image(dict(gpu, proto=live_p.detach()), {"A": gpu["c0"]}, coco, 4)
        for a, b in zip(base_rows, copied_rows):
            for metric in METRICS[:-1]:
                va, vb = a[metric + "_A"], b[metric + "_A"]
                if em.finite(va) != em.finite(vb) or (em.finite(va) and abs(va-vb) > 1e-12):
                    raise AssertionError("Original native prototype copy does not reproduce baseline metrics")
        before_many = losses_and_output_gradients(x, many, live_p, loader.criterion_many,
            equal_gt=True, verify=True, individual_check=args.smoke or not image_rows, check_time=check_time)
        before_one = losses_and_output_gradients(x, x, live_p, loader.criterion_one,
            verify=True, individual_check=args.smoke or not image_rows, check_time=check_time)
        output_gradients = {"M": before_many["gradient_M"], "E": before_many["gradient_E"],
                            "O": before_one["gradient_M"]}
        gradients = {}
        for k, arm in enumerate(("M", "E", "O")):
            gs = torch.autograd.grad(live_p, parameters, grad_outputs=output_gradients[arm], retain_graph=k < 2)
            gradients[arm] = gradient_vector(gs, parameters)
        del result, live_p
        norms = {a: float(g.double().norm()) for a, g in gradients.items()}
        multiplicities = Counter(map(int, many["owners"].tolist()))
        one_owners = set(map(int, x["owners"].tolist()))
        image_row = dict(split=x["split"], image_id=iid, one_candidates=len(x["raw_ids"]),
            many_candidates=len(many["raw_ids"]), many_gt_count=len(multiplicities),
            one_gt_count=len(one_owners), many_gt_multiplicities=list(multiplicities.values()),
            many_only_owners=sorted(set(multiplicities)-one_owners),
            one_only_owners=sorted(one_owners-set(multiplicities)),
            many_gt_counts={str(k): v for k, v in multiplicities.items()},
            parameter_norm=theta_norm, requested_step_norm=eta, gradient_norms=norms,
            original_objectives={"M": before_many["M"], "E": before_many["E"], "O": before_one["M"]},
            perturbations={}, prototype_replay_max_absolute_error=prototype_replay_error)
        for a, b in (("M", "E"), ("M", "O"), ("E", "O")):
            image_row[f"parameter_cos_{a}_{b}"] = cosine(gradients[a], gradients[b])
            image_row[f"output_cos_{a}_{b}"] = cosine(output_gradients[a], output_gradients[b])
        for kind in ("parameter", "output"):
            a, b = image_row[f"{kind}_cos_E_O"], image_row[f"{kind}_cos_M_O"]
            image_row[f"{kind}_alignment_E_minus_M"] = a-b if a is not None and b is not None else None
        bce_values = {"A": before_one["values"]}
        many_values = {"A": before_many["values"]}
        for arm in ("M", "E", "O"):
            check_time(); restore()
            direction = -gradients[arm] / norms[arm] if norms[arm] else torch.zeros_like(theta0)
            with torch.no_grad():
                offset = 0
                for parameter, saved in zip(parameters, original):
                    n = parameter.numel()
                    parameter.copy_(saved + eta * direction[offset:offset+n].reshape_as(parameter))
                    offset += n
                actual_delta = vector(parameters) - theta0
                changed = model(features, [x])
                p = changed["prototypes"][0]
                torch.testing.assert_close(changed["coefficients"][0], gpu["c0"], atol=0, rtol=0)
                cm = losses_and_output_gradients(x, many, p, loader.criterion_many, equal_gt=True,
                    gradients=False, check_time=check_time)
                co = losses_and_output_gradients(x, x, p, loader.criterion_one, gradients=False, check_time=check_time)
                candidate_rows = em.evaluate_image(dict(gpu, proto=p), {"A": gpu["c0"], arm: gpu["c0"]}, coco, 4)
            if [identity(r) for r in candidate_rows] != expected:
                raise AssertionError("Perturbed prototype changed the fixed evaluation population")
            objective_after = {"M": cm["M"], "E": cm["E"], "O": co["M"]}
            actual_norm = float(actual_delta.double().norm())
            if norms[arm] and abs(actual_norm-eta) > 3e-4 * eta:
                raise AssertionError("FP32 temporary displacement does not match the fixed equal-norm budget")
            prediction = {a: float((g.double() * actual_delta.double()).sum()) for a, g in gradients.items()}
            actual = {a: objective_after[a]-image_row["original_objectives"][a] for a in ("M", "E", "O")}
            image_row["perturbations"][arm] = dict(zero_gradient=not bool(norms[arm]), actual_step_norm=actual_norm,
                objectives_after=objective_after, predicted_loss_changes=prediction, actual_loss_changes=actual,
                linearization_absolute_error={a: abs(actual[a]-prediction[a]) for a in actual},
                linearization_relative_error={a: abs(actual[a]-prediction[a])/max(abs(prediction[a]), cfg["finite_difference_floor"]) for a in actual},
                finite_difference_faithful={a: abs(actual[a]-prediction[a]) <= cfg["finite_difference_atol"] or
                    abs(actual[a]-prediction[a])/max(abs(prediction[a]), cfg["finite_difference_floor"]) <= cfg["finite_difference_rtol"] for a in actual},
                prototype_change_l2=float((p-gpu["proto"]).double().norm()))
            bce_values[arm], many_values[arm] = co["values"], cm["values"]
            for row, candidate in zip(base_rows, candidate_rows):
                for metric in METRICS[:-1]:
                    row[f"{metric}_{arm}"] = candidate[f"{metric}_{arm}"]
                row[f"empty_mask_{arm}"] = candidate[f"empty_mask_{arm}"]
            del p, changed, cm, co, actual_delta, direction
            restore()
        # The predeclared interpretation check is the ONE objective response
        # to all three directions. Cross-objective M/E checks remain recorded
        # as diagnostics but do not silently redefine this gate.
        image_row["finite_difference_faithful"] = all(
            v["finite_difference_faithful"][cfg["finite_difference_gate_objective"]] for v in image_row["perturbations"].values())
        image_row["finite_difference_gate_objective"] = cfg["finite_difference_gate_objective"]
        supervision_rows = []
        for j, row in enumerate(base_rows):
            for arm in ARMS:
                row[f"bce_{arm}"] = bce_values[arm][j]
            row["same_image_GT_assisted"] = True
            row["original_parameter_point_restored"] = True
        image_row["quality"] = {metric: {arm: em.avg([r[f"{metric}_{arm}"] for r in base_rows])
            for arm in ARMS} for metric in METRICS}
        image_row["crossings_vs_A"] = {arm: dict(
            repair=sum(not r["mask75_A"] and r[f"mask75_{arm}"] for r in base_rows),
            damage=sum(r["mask75_A"] and not r[f"mask75_{arm}"] for r in base_rows)) for arm in ("M", "E", "O")}
        for support, by_arm, branch in ((many, many_values, "one2many"), (x, bce_values, "one2one")):
            owners = list(map(int, support["owners"].tolist()))
            counts = Counter(owners)
            for j, source in enumerate(support["rows"]):
                row = {k: source[k] for k in KEYS}
                if row["branch"] != branch or int(row["target_gt_idx"]) != owners[j]:
                    raise AssertionError("Supervision branch/owner metadata mismatch")
                row.update(official_GT_multiplicity=counts[owners[j]], weight_candidate=1./len(owners),
                           weight_GT_equal=1./(len(counts)*counts[owners[j]]))
                for arm in ARMS:
                    row[f"bce_{arm}"] = by_arm[arm][j]
                supervision_rows.append(row)
        em.append_rows(out / "SUPERVISION_CANDIDATE.jsonl", supervision_rows)
        em.append_rows(out / "PER_CANDIDATE.jsonl", base_rows)
        em.append_rows(out / "PER_IMAGE.jsonl", [image_row])
        for branch, result in (("one2many", before_many), ("one2one", before_one)):
            loss_audits.append(dict(image_id=iid, branch=branch, **result["audit"]))
        rows.extend(base_rows); image_rows.append(image_row); seen.update(expected)
        del gradients, output_gradients, before_many, before_one, features
        if position % 8 == 0 or position+1 == len(selected):
            progress = dict(processed_planned_images=position+1, planned_images=len(selected),
                            effective_images=len(image_rows), candidates=len(rows), elapsed_s=time.monotonic()-started)
            dump(out / "PROGRESS.json", progress); print(json.dumps(progress), flush=True)
    check_time(); restore(); replay.assert_unchanged()
    assignment_finish = loader.finish()
    final_state = {name: tensor_sha(value) for name, value in model.state_dict().items()}
    if final_state != initial_state:
        raise AssertionError("Native prototype state did not return bitwise to its original snapshot")
    if len(rows) != sum(int(index[iid]["n"]) for iid in selected):
        raise AssertionError("Do not summarize an incomplete fixed candidate population")
    population = dict(planned_images=len(selected), protocol_planned_images=len(planned),
                      effective_images=len(image_rows), candidates=len(rows), no_positive_images=no_positive,
                      no_positive_records=no_positive_records,
                      many_candidates=sum(r["many_candidates"] for r in image_rows) + sum(r["many_candidates"] for r in no_positive_records),
                      evaluated_many_candidates=sum(r["many_candidates"] for r in image_rows), smoke=bool(args.smoke))
    audit = dict(passed=True, same_image_GT_assisted=True, original_source_unchanged=True,
        original_prototype_final_state_exact=True, all_per_image_parameter_restorations_exact=True,
        BN_buffers_frozen=True, BN_affine_frozen=True, semseg_frozen=True, only_native_mask_prototype_parameters_differ=True,
        fixed_one_candidates=True, fixed_all_coefficients=True, full_prototype_decode=True,
        no_optimizer=True, no_checkpoint=True, no_cumulative_updates=True,
        one2one_gradient_is_diagnostic_undetach=True, assignment_audits=assignment_audits,
        assignment_finish=assignment_finish,
        parameter_names=names, parameter_counts=counts, population=population)
    dump(out / "LOSS_EQUIVALENCE.json", dict(passed=True, records=loss_audits, atol=ATOL, rtol=RTOL))
    dump(out / "ASSIGNMENT_AUDIT.json", dict(passed=True, records=assignment_audits, finish=assignment_finish))
    dump(out / "AUDIT.json", audit)
    summarize(rows, image_rows, cfg, out, population, audit, em, dump)
    check_time()
    dump(out / "COMPLETE.json", dict(completed=True, passed=True, smoke=bool(args.smoke),
        elapsed_s=time.monotonic()-started, population=population, no_training=True,
        scientific_scope=cfg["scope"], automatic_followup=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        payload = dict(completed=False, passed=False, incomplete=True, error=repr(exc), traceback=traceback.format_exc())
        output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
        for name in ("FAILURE.json", "COMPLETE.json"):
            path = output / name
            if name == "COMPLETE.json" and path.exists():
                continue
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
