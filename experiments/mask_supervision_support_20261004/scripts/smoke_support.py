"""Disposable remote-only gates for mask-support redistribution, never training output."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    if _cfg.get("source_python"):
        sys.path.insert(0, _cfg["source_python"])
    sys.path.append(str(Path(_cfg['screen_root']) / 'scripts'))

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss

from online_runtime import FrozenReplay, dump, load_asset, load_json, resolve_runtime_config, sha256, tensor_sha
from dense_runtime import (buffers, check_control, gpu_payload, new_model, optimizer_for,
                           setup, state_digest, verify_buffers, weighted_loss_and_grad)
from runtime_utils import loss_and_grad as old_loss_and_grad

LOSS_ATOL, LOSS_RTOL = 1e-6, 1e-5
REPLAY_ATOL, REPLAY_RTOL = 3e-5, 3e-5
RANDOM_DELTA_SEED, RANDOM_DELTA_STD = 6100407, 0.1


def tree_hash(value):
    if torch.is_tensor(value):
        return tensor_sha(value)
    if isinstance(value, dict):
        return {k: tree_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_hash(v) for v in value]
    return value


def comparison(actual, expected, atol=LOSS_ATOL, rtol=LOSS_RTOL):
    actual, expected = actual.detach(), expected.detach().to(actual.device)
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise FloatingPointError("Nonfinite equivalence check")
    diff = (actual - expected).abs()
    allowance = atol + rtol * expected.abs()
    stats = dict(max_absolute_error=float(diff.max()) if diff.numel() else 0.,
                 max_tolerance_ratio=float((diff / allowance).max()) if diff.numel() else 0.,
                 atol=atol, rtol=rtol, passed=bool((diff <= allowance).all()))
    return stats


def require(record, message):
    if not record["passed"]:
        raise AssertionError(f"{message}: {record}")


def official_reference(payload, coefficients, weights, denominator, single_mask_loss):
    """Independent official per-instance call, with the exact gain and GT averaging."""
    proto = F.interpolate(payload["proto"][None].float(), (640, 640),
                          mode="bilinear", align_corners=False)[0]
    leaf = coefficients.detach().requires_grad_(True)
    gradient = torch.zeros_like(leaf)
    # Summation in float64 only accumulates already-computed FP32 instance values;
    # the official loss, logits and their gradients retain the training precision.
    values = []
    for i in range(len(leaf)):
        truth = (payload["masks"][None] == (payload["owners"][i] + 1)).float()
        box = payload["target_boxes"][i:i+1]
        area = ((box[:, 2:] - box[:, :2]) / 640.).prod(1)
        value = single_mask_loss(truth, leaf[i:i+1], proto, box, area)
        value = value * float(payload["segmentation_gain"]) * weights[i].to(leaf)
        gradient += torch.autograd.grad(value / denominator, leaf)[0]
        values.append(value.detach().double())
    return torch.stack(values).sum(), gradient


def selections(supports, device, repeated=False):
    result = []
    for s in supports:
        ids = s["raw_ids"].clone()
        if repeated:
            original = {int(owner): int(rid) for owner, rid in
                        zip(s["owners"][:s["original_count"]], ids[:s["original_count"]])}
            ids = torch.tensor([original[int(owner)] for owner in s["owners"]], dtype=torch.long)
        result.append(dict(raw_ids=ids.to(device)))
    return result


def capture_parameter_gradients(model):
    grads = {}
    for name, parameter in model.named_parameters():
        if not name.startswith("native_cv4.") or not parameter.requires_grad:
            raise AssertionError(f"Unexpected trainable scope: {name}")
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"Missing/nonfinite native gradient: {name}")
        grads[name] = parameter.grad.detach().clone()
    if not any(bool(g.abs().max() > 0) for g in grads.values()):
        raise AssertionError("All native parameter gradients are zero")
    return grads


def smoke(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Execute only on the authorized Linux GPU server; desktop must not run models")
    started = time.monotonic()
    cfg = resolve_runtime_config(load_json(args.config))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists() or (out / "assignment").exists():
        raise RuntimeError("Preserve previous smoke outputs; use an independent Run")
    deadline = started + float(cfg["smoke_max_seconds"])
    report = dict(passed=False, kind="mask_support_disposable_smoke", formal_training_steps=0,
                  temporary_optimizer_steps=1, learned_checkpoint_saved=False,
                  tolerances=dict(loss_atol=LOSS_ATOL, loss_rtol=LOSS_RTOL,
                                  replay_atol=REPLAY_ATOL, replay_rtol=REPLAY_RTOL),
                  random_delta=dict(seed=RANDOM_DELTA_SEED, std=RANDOM_DELTA_STD),
                  loss_checks=[], replay=[], repeated_control={})

    def save():
        dump(out / "AUDIT.json", report)

    def budget():
        if time.monotonic() > deadline:
            raise TimeoutError("Fixed smoke budget exceeded; do not relax gates or resume")

    save()
    command = [sys.executable, "-u", str(Path(__file__).with_name("prepare_support.py")),
               "--config", str(Path(args.config).resolve()), "--out", str(out / "assignment"), "--smoke"]
    report["prepare_command"] = command
    with (out / "prepare.stdout.log").open("wb") as stdout, (out / "prepare.stderr.log").open("wb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False,
                                   timeout=max(1., deadline-time.monotonic()))
    report["prepare_exit_code"] = completed.returncode
    save()
    if completed.returncode:
        raise RuntimeError("Assignment smoke failed; inspect preserved prepare logs")
    root = out / "assignment" / "support"
    receipt, index = load_json(root / "COMPLETE.json"), load_json(root / "INDEX.json")
    if not receipt["passed"] or not index["complete"] or not index["smoke"] or len(index["images"]) != 2:
        raise AssertionError("Require exactly two completed smoke support images")
    report["assignment"] = dict(audit_sha256=sha256(out / "assignment" / "ASSIGNMENT_AUDIT.json"),
                                 index_sha256=sha256(root / "INDEX.json"), receipt=receipt)
    images, supports, expanded = [], [], []
    for entry in index["images"]:
        path = root / entry["filename"]
        if sha256(path) != entry["sha256"]:
            raise AssertionError("Support asset hash mismatch")
        s = torch.load(path, map_location="cpu", weights_only=False)
        x = load_asset(cfg, int(entry["image_id"]), verify=True)
        n = int(s["original_count"])
        if x["split"] != "fit" or n != len(x["raw_ids"]) or n <= 0:
            raise AssertionError("Smoke original population differs")
        if not torch.equal(s["raw_ids"][:n], x["raw_ids"]) or not torch.equal(s["owners"][:n], x["owners"]):
            raise AssertionError("Original candidate order/GT identity differs")
        if s["source_image_cache_sha256"] != x["_asset_integrity"]["compressed_sha256"]:
            raise AssertionError("Support belongs to another source asset")
        for owner in s["owners"].unique():
            if abs(float(s["weights"][s["owners"] == owner].sum()) - 1.) > 1e-12:
                raise AssertionError("Support does not conserve original GT weight")
        images.append(x); supports.append(s)
        expanded.append(dict(x, raw_ids=s["raw_ids"], owners=s["owners"], target_boxes=s["target_boxes"]))
    if not any(len(s["raw_ids"]) > s["original_count"] for s in supports):
        raise AssertionError("First two fit images have no real extra support; extension gate cannot pass vacuously")
    original_count = sum(len(x["raw_ids"]) for x in images)
    report.update(image_ids=[int(x["image_id"]) for x in images], original_count=original_count,
                  expanded_count=sum(len(s["raw_ids"]) for s in supports))
    input_hash = tree_hash(images); support_hash = tree_hash(supports)
    setup(int(cfg["seed"]))
    replay = FrozenReplay(cfg)
    source_hash = state_digest(replay.source)
    replay.source.args = SimpleNamespace(**replay.train_args)
    criterion = replay.source.init_criterion().one2one
    if criterion.assigner.topk != 7 or criterion.assigner.topk2 != 1:
        raise AssertionError("Official reference criterion was modified")
    if criterion.single_mask_loss is not v8SegmentationLoss.single_mask_loss:
        raise AssertionError("Unexpected official instance mask loss implementation")
    features = replay.replay(images)
    feature_hash = tree_hash(features)
    original_outputs = replay.native_outputs(features, images)
    extended_outputs = replay.native_outputs(features, expanded)
    model = new_model(replay, cfg).train()
    report["control"] = check_control(cfg, model)
    initial_state = state_digest(model)
    initial_buffers = buffers(model)
    sparse_selected = [dict(raw_ids=x["raw_ids"].to(replay.device)) for x in images]
    with torch.no_grad():
        coefficients, _, current = model.forward_details(features, sparse_selected)
    for x, reference, c, now in zip(images, original_outputs, coefficients, current):
        checks = dict(image_id=int(x["image_id"]),
            source_c_vs_cache=comparison(reference["c"], x["c0"], REPLAY_ATOL, REPLAY_RTOL),
            source_h_vs_cache=comparison(reference["h"], x["_operator"]["h0"], REPLAY_ATOL, REPLAY_RTOL),
            model_c_vs_source=comparison(c, reference["c"], REPLAY_ATOL, REPLAY_RTOL),
            model_h_vs_source=comparison(now["h_current"], reference["h"], REPLAY_ATOL, REPLAY_RTOL))
        report["replay"].append(checks); save()
        for name, item in checks.items():
            if name != "image_id": require(item, name)
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_DELTA_SEED)
    for population, rows, outputs in (("original", images, original_outputs), ("expanded", expanded, extended_outputs)):
        for k, (x, ref) in enumerate(zip(rows, outputs)):
            weight = torch.ones(len(ref["c"]), device=replay.device) if population == "original" else supports[k]["weights"].to(replay.device)
            delta = torch.randn(ref["c"].shape, generator=generator).to(replay.device) * RANDOM_DELTA_STD
            for point, coeff in (("c0", ref["c"]), ("c0_plus_fixed_delta", ref["c"] + delta)):
                budget()
                payload = gpu_payload(x, replay.device)
                accelerated_value, accelerated_gradient = weighted_loss_and_grad(payload, coeff, original_count, weight)
                official_value, official_gradient = official_reference(payload, coeff, weight, original_count, criterion.single_mask_loss)
                row = dict(population=population, image_id=int(x["image_id"]), point=point,
                    weighted_sum=comparison(official_value.new_tensor(accelerated_value), official_value),
                    normalized_loss=comparison(official_value.new_tensor(accelerated_value/original_count), official_value/original_count),
                    coefficient_gradient=comparison(accelerated_gradient, official_gradient))
                report["loss_checks"].append(row); save()
                for name in ("weighted_sum", "normalized_loss", "coefficient_gradient"): require(row[name], name)
    budget()
    # Sparse uses the unchanged historical full640 implementation. Repeat every
    # extra row's raw ID as its own original winner, preserving D's exact weights.
    model.zero_grad(set_to_none=True)
    coefficients = model(features, sparse_selected)
    sparse_value, gradients = 0., []
    for x, c in zip(images, coefficients):
        value, gradient = old_loss_and_grad(gpu_payload(x, replay.device), c, original_count)
        sparse_value += value; gradients.append(gradient)
    torch.autograd.backward(coefficients, gradients)
    sparse_grads = capture_parameter_gradients(model)
    model.zero_grad(set_to_none=True)
    coefficients = model(features, selections(supports, replay.device, repeated=True))
    repeated_value, gradients = 0., []
    for x, s, c in zip(expanded, supports, coefficients):
        value, gradient = weighted_loss_and_grad(gpu_payload(x, replay.device), c, original_count, s["weights"])
        repeated_value += value; gradients.append(gradient)
    torch.autograd.backward(coefficients, gradients)
    repeated_grads = capture_parameter_gradients(model)
    gradient_checks = {name: comparison(repeated_grads[name], old) for name, old in sparse_grads.items()}
    value_check = comparison(torch.tensor(repeated_value/original_count, dtype=torch.float64),
                             torch.tensor(sparse_value/original_count, dtype=torch.float64))
    report["repeated_control"] = dict(loss=value_check, parameter_gradients=gradient_checks,
        original_loss=sparse_value/original_count, repeated_loss=repeated_value/original_count,
        all_parameters_checked=len(gradient_checks), repeated_ids_keep_original_gt=True,
        normalization="original covered GT count; original+same-GT extras weight sums to 1")
    save(); require(value_check, "Repeated original loss")
    for name, check in gradient_checks.items(): require(check, "Repeated parameter gradient: " + name)
    if state_digest(model) != initial_state:
        raise AssertionError("Equivalence checks unexpectedly changed model state")
    del sparse_grads, repeated_grads, gradients, coefficients
    budget()
    optimizer = optimizer_for(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    coefficients = model(features, selections(supports, replay.device))
    gradients, true_value = [], 0.
    for x, s, c in zip(expanded, supports, coefficients):
        value, gradient = weighted_loss_and_grad(gpu_payload(x, replay.device), c, original_count, s["weights"])
        true_value += value; gradients.append(gradient)
    torch.autograd.backward(coefficients, gradients)
    true_grads = capture_parameter_gradients(model)
    before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
    optimizer.step()
    after = state_digest(model)
    changed = [name for name, value in after.items() if value != initial_state[name]]
    parameter_names = set(dict(model.named_parameters()))
    if not changed or any(name not in parameter_names or not name.startswith("native_cv4.") for name in changed):
        raise AssertionError("Disposable D update did not change only declared native parameters")
    verify_buffers(model, initial_buffers)
    replay.assert_unchanged()
    if state_digest(replay.source) != source_hash:
        raise AssertionError("Original source/prefix changed")
    if tree_hash(features) != feature_hash or tree_hash(images) != input_hash or tree_hash(supports) != support_hash:
        raise AssertionError("Frozen feature/label/support input changed")
    replay_after = replay.native_outputs(replay.replay(images), images)
    for before, after_ref in zip(original_outputs, replay_after):
        for name in ("h", "c"):
            torch.testing.assert_close(before[name], after_ref[name], atol=0, rtol=0)
    report["temporary_D_update"] = dict(loss=true_value/original_count,
        parameter_gradient_norms={k: float(g.norm()) for k, g in true_grads.items()},
        gradient_norm_before_clip=before_clip, changed_parameter_names=changed,
        native_parameters_updated=True, bn_buffers_unchanged=True, source_state_unchanged=True,
        cached_inputs_unchanged=True, original_h_c_replay_after_update_exact=True,
        optimizer="paired AdamW, one disposable step at branch_lr; no checkpoint written")
    report.update(passed=True, elapsed_s=time.monotonic()-started,
                  environment=replay.import_info, original_source_state_sha256=source_hash)
    budget(); save()
    dump(out / "COMPLETE.json", dict(passed=True, completed=True, complete=True,
        elapsed_s=report["elapsed_s"], audit_sha256=sha256(out / "AUDIT.json"),
        formal_training_steps=0, temporary_optimizer_steps=1, learned_checkpoint_saved=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    try:
        smoke(arguments)
    except BaseException as exc:
        output = Path(arguments.out)
        dump(output / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        dump(output / "COMPLETE.json", dict(passed=False, complete=False, completed=False,
                                            error=repr(exc), automatic_retry=False))
        raise
