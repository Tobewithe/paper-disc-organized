"""Disposable bridge/N equivalence and frozen-state gates; no learned artifact."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    sys.path.insert(0, _cfg["source_python"])
    for _key in ("screen_root", "dense_root"):
        if _cfg.get(_key):
            sys.path.append(str(Path(_cfg[_key]) / "scripts"))

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, sha256, tensor_sha
import dense_runtime as dr
from bridge_model import BridgeReadout
from prepare_features import load_feature_cache, selection_for
from runtime_utils import loss_and_grad as historical_loss_and_grad
from train_bridge import resolve_config, optimizer_for, initialization_control, feature_file_digests, check_gradients

LOSS_ATOL, LOSS_RTOL = 1e-6, 1e-5
REPLAY_ATOL, REPLAY_RTOL = 3e-5, 3e-5
GRAD_ATOL, GRAD_RTOL, GRAD_RELATIVE_L2 = 1e-5, 3e-4, 1e-4
RANDOM_DELTA_SEED, RANDOM_DELTA_STD = 6100408, .1


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
        raise FloatingPointError("Nonfinite equivalence values")
    diff = (actual - expected).abs()
    allowance = atol + rtol * expected.abs()
    return dict(max_absolute_error=float(diff.max()) if diff.numel() else 0.,
        max_tolerance_ratio=float((diff / allowance).max()) if diff.numel() else 0.,
        atol=atol, rtol=rtol, passed=bool((diff <= allowance).all()))


def require(result, label):
    if not result["passed"]:
        raise AssertionError(label + ": " + repr(result))


def official_reference(payload, coefficients, denominator, single_mask_loss):
    proto = F.interpolate(payload["proto"][None].float(), (640, 640),
                          mode="bilinear", align_corners=False)[0]
    leaf = coefficients.detach().requires_grad_(True)
    gradient, values = torch.zeros_like(leaf), []
    for i in range(len(leaf)):
        truth = (payload["masks"][None] == payload["owners"][i] + 1).float()
        box = payload["target_boxes"][i:i+1]
        area = ((box[:, 2:] - box[:, :2]) / 640).prod(1)
        value = single_mask_loss(truth, leaf[i:i+1], proto, box, area) * float(payload["segmentation_gain"])
        gradient += torch.autograd.grad(value / denominator, leaf)[0]
        values.append(value.detach().double())
    return torch.stack(values).sum(), gradient


def parameter_grads(model):
    result = {}
    for name, p in model.named_parameters():
        if name.startswith("native_cv4."):
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                raise AssertionError("Missing/nonfinite native gradient: " + name)
            result[name] = p.grad.detach().clone()
    return result


def backward(model, features, selections, images, denominator, old=False):
    model.zero_grad(set_to_none=True)
    coefficients = model(features, selections)
    value, gradients = 0., []
    for image, coeff in zip(images, coefficients):
        payload = dr.gpu_payload(image, coeff.device)
        if old:
            loss, gradient = historical_loss_and_grad(payload, coeff, denominator)
        else:
            loss, gradient = dr.weighted_loss_and_grad(payload, coeff, denominator,
                torch.ones(len(coeff), device=coeff.device, dtype=coeff.dtype))
        value += loss
        gradients.append(gradient)
    torch.autograd.backward(coefficients, gradients)
    return value / denominator, parameter_grads(model)


def smoke(args):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Execute only on the authorized Linux CUDA server")
    started = time.monotonic()
    cfg = resolve_config(load_json(args.config))
    if cfg["smoke_max_seconds"] != 300:
        raise ValueError("Fixed smoke budget is 300 seconds")
    deadline = started + cfg["smoke_max_seconds"]
    out = Path(args.out)
    if (out / "COMPLETE.json").exists() or (out / "AUDIT.json").exists():
        raise RuntimeError("Previous smoke must be preserved in its independent Run")
    report = dict(passed=False, kind="cross_task_bridge_disposable_smoke", formal_training_steps=0,
        checkpoint_saved=False, modes={}, loss_checks=[], source_replay=[],
        tolerances=dict(loss_atol=LOSS_ATOL, loss_rtol=LOSS_RTOL,
                        replay_atol=REPLAY_ATOL, replay_rtol=REPLAY_RTOL,
                        native_gradient_atol=GRAD_ATOL, native_gradient_rtol=GRAD_RTOL,
                        native_gradient_overall_relative_l2=GRAD_RELATIVE_L2),
        random_delta_seed=RANDOM_DELTA_SEED, random_delta_std=RANDOM_DELTA_STD)

    def save():
        dump(out / "AUDIT.json", report)

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed 300-second smoke budget exhausted")

    save()
    prepare_receipt = Path(cfg["prepare_run"]) / "COMPLETE.json"
    prepare = load_json(prepare_receipt)
    if not prepare.get("passed", prepare.get("completed", False)):
        raise AssertionError("Frozen source feature preparation not complete")
    bank = load_feature_cache(cfg)
    feature_identity = feature_file_digests(cfg)
    selected_items = [r for r in load_index(cfg)["fit"] if r["n"] > 0][:2]
    if len(selected_items) != 2:
        raise AssertionError("Exactly two effective original fit images required")
    images = [load_asset(cfg, r["image_id"], verify=True) for r in selected_items]
    denominator = sum(len(x["raw_ids"]) for x in images)
    input_hash = tree_hash(images)
    bank_hash = tree_hash(bank)
    dr.setup(cfg["seed"])
    replay = FrozenReplay(cfg)
    source_hash = dr.state_digest(replay.source)
    features = replay.replay(images)
    feature_hash = tree_hash(features)
    references = replay.native_outputs(features, images)
    replay.source.args = SimpleNamespace(**replay.train_args)
    criterion = replay.source.init_criterion().one2one
    if criterion.assigner.topk != 7 or criterion.assigner.topk2 != 1:
        raise AssertionError("Original official assignment configuration changed")
    if criterion.single_mask_loss is not v8SegmentationLoss.single_mask_loss:
        raise AssertionError("Unexpected official mask-loss implementation")
    report.update(image_ids=[int(x["image_id"]) for x in images], candidates=denominator,
                  prepare_receipt_sha256=sha256(prepare_receipt), feature_identity=feature_identity,
                  environment=replay.import_info,
                  loss_implementation=dict(accelerated_file=str(Path(dr.__file__).resolve()),
                      accelerated_sha256=sha256(dr.__file__),
                      official_file=sys.modules[v8SegmentationLoss.__module__].__file__,
                      official_sha256=sha256(sys.modules[v8SegmentationLoss.__module__].__file__)))
    with torch.no_grad():
        for family, branches in (("cls", replay.head.one2one_cv3), ("box", replay.head.one2one_cv2)):
            maps = [branch[:-1](feature).flatten(2) for branch, feature in zip(branches, features)]
            full = torch.cat(maps, 2).transpose(1, 2)
            for j, x in enumerate(images):
                cached = bank["images"][int(x["image_id"])][family]
                fresh = full[j, x["raw_ids"].to(replay.device)]
                check = comparison(fresh, cached, REPLAY_ATOL, REPLAY_RTOL)
                report["source_replay"].append(dict(image_id=int(x["image_id"]), source=family, check=check))
                save(); require(check, family + " hidden source replay")
    for x, ref in zip(images, references):
        for name, cached in (("c", x["c0"]), ("h", x["_operator"]["h0"])):
            check = comparison(ref[name], cached, REPLAY_ATOL, REPLAY_RTOL)
            report["source_replay"].append(dict(image_id=int(x["image_id"]), source="native_"+name, check=check))
            save(); require(check, "Frozen native " + name)
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_DELTA_SEED)
    for x, reference in zip(images, references):
        delta = torch.randn(reference["c"].shape, generator=generator).to(replay.device) * RANDOM_DELTA_STD
        for point, coeff in (("c0", reference["c"]), ("c0_plus_fixed_delta", reference["c"] + delta)):
            budget()
            payload = dr.gpu_payload(x, replay.device)
            value, gradient = dr.weighted_loss_and_grad(payload, coeff, denominator,
                torch.ones(len(coeff), device=coeff.device, dtype=coeff.dtype))
            expected_value, expected_gradient = official_reference(payload, coeff, denominator, criterion.single_mask_loss)
            checks = dict(loss=comparison(expected_value.new_tensor(value/denominator), expected_value/denominator),
                          coefficient_gradient=comparison(gradient, expected_gradient))
            report["loss_checks"].append(dict(image_id=int(x["image_id"]), point=point, **checks))
            save()
            for name, check in checks.items(): require(check, point + " official " + name)
    native = dr.new_model(replay, cfg).train()
    report["historical_N_control"] = dr.check_control(cfg, native)
    sparse = [dict(raw_ids=x["raw_ids"].to(replay.device)) for x in images]
    native_loss, native_gradients = backward(native, features, sparse, images, denominator, old=True)
    native_gradient_norm = math.sqrt(sum(float(g.double().square().sum()) for g in native_gradients.values()))
    if native_gradient_norm == 0:
        raise AssertionError("Historical N native gradient is zero; smoke would be vacuous")
    native_initial = {name: p.detach().clone() for name, p in native.named_parameters()}
    del native
    for mode in ("T", "M", "R"):
        budget()
        dr.setup(cfg["seed"])
        model = BridgeReadout(replay.native_cv4, replay.feature_channels, mode, cfg, bank["stats"]).to(replay.device).float().train()
        control = initialization_control(cfg, replay, model)
        selected = [selection_for(x, bank, mode, replay.device) for x in images]
        selection_hash = tree_hash(selected)
        initial_state = dr.state_digest(model)
        initial_buffers = dr.buffers(model)
        initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        if any(not torch.equal(initial_parameters[name], value) for name, value in native_initial.items()):
            raise AssertionError("Unpaired native initialization")
        bridge_parameters = sum(p.numel() for name, p in model.named_parameters() if name.startswith("bridges."))
        if bridge_parameters != (12480 if mode == "R" else 49344):
            raise AssertionError("Unexpected bridge shape/count")
        with torch.no_grad():
            zero_coefficients = model(features, selected)
        zero_checks = [comparison(c, ref["c"], REPLAY_ATOL, REPLAY_RTOL)
                       for c, ref in zip(zero_coefficients, references)]
        report["modes"][mode] = dict(control=control, counts=model.parameter_counts(),
            bridge_parameters=bridge_parameters, zero_bridge_replay=zero_checks)
        save()
        for check in zero_checks: require(check, mode + " zero bridge replay")
        value, gradients = backward(model, features, selected, images, denominator)
        checks = {name: comparison(gradients[name], old, GRAD_ATOL, GRAD_RTOL)
                  for name, old in native_gradients.items()}
        numerator = math.sqrt(sum(float((gradients[name]-old).double().square().sum())
                             for name, old in native_gradients.items()))
        relative_l2 = numerator / native_gradient_norm
        value_check = comparison(torch.tensor(value, dtype=torch.float64),
                                 torch.tensor(native_loss, dtype=torch.float64))
        family_norms = check_gradients(model)
        report["modes"][mode].update(loss_equivalence=value_check, native_parameter_gradient_checks=checks,
            native_gradient_relative_l2=relative_l2, family_gradient_norms=family_norms,
            all_native_parameters_compared=len(checks))
        save(); require(value_check, mode + " historical N loss")
        for name, check in checks.items(): require(check, mode + " historical N gradient " + name)
        if relative_l2 > GRAD_RELATIVE_L2:
            raise AssertionError(mode + " overall native gradient mismatch")
        if family_norms["bridge"] <= 0 or family_norms["native"] <= 0:
            raise AssertionError(mode + " has no bridge/native gradient")
        if dr.state_digest(model) != initial_state:
            raise AssertionError("Equivalence check changed model/buffers")
        optimizer = optimizer_for(model, cfg)
        before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
        optimizer.step()
        changed = {family: [name for name, p in model.named_parameters() if name.startswith(prefix)
                           and not torch.equal(p.detach(), initial_parameters[name])]
                   for family, prefix in (("native", "native_cv4."), ("bridge", "bridges."))}
        if not all(changed.values()):
            raise AssertionError(mode + " temporary update did not change both parameter families")
        dr.verify_buffers(model, initial_buffers)
        replay.assert_unchanged()
        if dr.state_digest(replay.source) != source_hash:
            raise AssertionError("Frozen source changed after temporary bridge update")
        if tree_hash(selected) != selection_hash or tree_hash(features) != feature_hash:
            raise AssertionError("Frozen selected source or prefix features changed")
        after_reference = replay.native_outputs(replay.replay(images), images)
        for old, new in zip(references, after_reference):
            for key in ("h", "c"):
                torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)
        with torch.no_grad():
            updated = model(features, selected)
        report["modes"][mode]["temporary_update"] = dict(steps=1, checkpoint_saved=False,
            changed_parameters=changed, gradient_norm_before_clip=before_clip,
            coefficient_change_mean_l2=float(torch.cat([(new-old).norm(dim=1)
                for new, old in zip(updated, zero_coefficients)]).mean()),
            bn_and_statistics_buffers_unchanged=True, original_source_unchanged=True,
            original_h_c_after_update_exact=True)
        save()
        del model, optimizer, gradients, selected, updated, zero_coefficients, initial_parameters
    if tree_hash(images) != input_hash or tree_hash(bank) != bank_hash:
        raise AssertionError("Original input/label/feature cache changed")
    if feature_file_digests(cfg) != feature_identity:
        raise AssertionError("Prepared asset files changed")
    budget()
    report.update(passed=True, elapsed_s=time.monotonic()-started, temporary_optimizer_steps=3,
                  source_state_unchanged=True, each_arm_temporary_step_discarded=True)
    save()
    dump(out / "COMPLETE.json", dict(passed=True, completed=True, complete=True,
        audit_sha256=sha256(out / "AUDIT.json"), elapsed_s=report["elapsed_s"],
        formal_training_steps=0, temporary_optimizer_steps=3, learned_checkpoint_saved=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if any((Path(args.out) / name).exists() for name in ("COMPLETE.json", "FAILURE.json", "AUDIT.json")):
        raise RuntimeError("Prior smoke artifacts exist; preserve this Run and register a separate attempt")
    try:
        smoke(args)
    except BaseException as exc:
        if not (Path(args.out) / "FAILURE.json").exists():
            dump(Path(args.out) / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        if not (Path(args.out) / "COMPLETE.json").exists():
            dump(Path(args.out) / "COMPLETE.json", dict(passed=False, completed=False, complete=False, error=repr(exc)))
        raise
