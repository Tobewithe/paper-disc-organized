"""Two native coadaptation arms; fixed epoch3, original fullmask objective.

Run only under the study runner on the authorized Linux CUDA host. The source
model supplies immutable full feature maps; no cached fixed-P operator is used
to differentiate a learned prototype. No dev selection or automatic resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

if "--config" in sys.argv:
    _early = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    _source_scripts = _early.get("source_script_dir") or (
        str(Path(_early["original_screen_root"]) / "scripts") if _early.get("original_screen_root") else None)
    if _source_scripts:
        sys.path.append(_source_scripts)
    if _early.get("source_python"):
        sys.path.insert(0, _early["source_python"])

import torch

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from runtime_utils import lr_factor, setup
from native_proto_model import (build_model, loss_and_grad,
                                assert_fixed_prototype_loss_equivalence)

FIXED = dict(seed=0, epochs=3, microbatch_images=2, effective_batch_images=16,
             branch_lr=1e-4, weight_decay=1e-4, warmup_epochs=1,
             eta_ratio=0.1, gradient_clip_norm=10)
MAX_SHARED_SECONDS = 40 * 60


def deadline_check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Shared P/PC 40-minute deadline exhausted; incomplete is not a negative result")


def digest(model):
    return {name: tensor_sha(value) for name, value in model.state_dict().items()}


def buffer_digest(model):
    return {name: tensor_sha(value) for name, value in model.named_buffers()}


def resolve_config(config):
    cfg = resolve_runtime_config(config)
    for key, expected in FIXED.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Protocol fixes {key}={expected}; received {cfg.get(key)}")
    return cfg


def optimizer_for(model, cfg):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith(("native_proto.", "native_cv4.")):
            raise AssertionError(f"Unexpected trainable parameter: {name}")
        if ".semseg." in name:
            raise AssertionError("Semantic auxiliary head must remain frozen")
        groups.setdefault(parameter.ndim > 1, []).append(parameter)
    if not groups:
        raise AssertionError("No native trainable parameters")
    return torch.optim.AdamW([
        dict(params=parameters, lr=cfg["branch_lr"], initial_lr=cfg["branch_lr"],
             weight_decay=cfg["weight_decay"] if decay else 0.,
             group_name="native_decay" if decay else "native_no_decay")
        for decay, parameters in groups.items()], betas=(0.9, 0.999), eps=1e-8)


def receipts(cfg, mode, smoke):
    origin = Path(cfg["original_screen_root"])
    original_runs = load_json(origin / "RUN_IDS.json")["runs"]
    preparation = origin / "runs" / original_runs["prepare"] / "COMPLETE.json"
    prepared = load_json(preparation)
    if not (prepared.get("completed") is True or prepared.get("complete") is True):
        raise RuntimeError("Frozen source asset preparation is incomplete")
    result = dict(prepare_receipt=str(preparation), prepare_sha256=sha256(preparation))
    if not smoke:
        root = Path(cfg["server_root"])
        runs = load_json(root / "RUN_IDS.json")["runs"]
        smoke_id = runs.get("smoke_" + mode, runs.get("smoke"))
        if smoke_id is None:
            raise ValueError(f"RUN_IDS lacks smoke_{mode} or smoke")
        path = root / "runs" / smoke_id / "COMPLETE.json"
        receipt = load_json(path)
        if receipt.get("passed") is not True:
            raise RuntimeError("Native prototype/gradient smoke has not passed")
        if receipt.get("mode") not in (None, mode) and mode not in receipt.get("modes", []):
            raise RuntimeError("Smoke belongs to a different trainable arm")
        result.update(smoke_receipt=str(path), smoke_sha256=sha256(path))
    return result


def verify_states(model, replay, initial_source, initial_buffers, initial_frozen):
    model.assert_frozen_buffers()
    if buffer_digest(model) != initial_buffers:
        raise AssertionError("Native model buffers changed")
    current = dict(model.named_parameters())
    if any(tensor_sha(current[name]) != expected for name, expected in initial_frozen.items()):
        raise AssertionError("A frozen native-model parameter changed")
    replay.assert_unchanged()
    if digest(replay.source) != initial_source:
        raise AssertionError("Original feature replay source changed")
    if any(p.grad is not None for p in replay.source.parameters()):
        raise AssertionError("Gradient reached original frozen feature source")


def backward_batch(model, replay, batch, denominator, mode):
    features = replay.replay(batch)
    result = model(features, batch)
    coefficients, prototypes = result["coefficients"], result["prototypes"]
    if len(coefficients) != len(batch) or len(prototypes) != len(batch):
        raise AssertionError("Model changed image membership")
    variables, gradients = [], []
    summed_loss, seen = 0.0, 0
    for image, coefficient, prototype in zip(batch, coefficients, prototypes):
        if coefficient.shape != image["c0"].shape or prototype.shape != image["proto"].shape:
            raise AssertionError("Model changed candidate or prototype shape")
        value, grad_c, grad_p = loss_and_grad(image, coefficient, prototype, denominator)
        if not math.isfinite(value) or not torch.isfinite(grad_c).all() or not torch.isfinite(grad_p).all():
            raise FloatingPointError("Nonfinite official loss or coefficient/prototype gradient")
        if mode == "P":
            if coefficient.requires_grad or not torch.equal(coefficient.detach().cpu(), image["c0"].float().cpu()):
                raise AssertionError("P arm must keep exact original c0")
        else:
            if not coefficient.requires_grad:
                raise AssertionError("PC coefficient branch is detached")
            variables.append(coefficient); gradients.append(grad_c)
        if not prototype.requires_grad:
            raise AssertionError("Learned prototype is detached from native Proto26")
        variables.append(prototype); gradients.append(grad_p)
        summed_loss += float(value)
        seen += len(image["raw_ids"])
    torch.autograd.backward(variables, gradients)
    return summed_loss, seen


def save_checkpoint(path, model, optimizer, epoch, mode, cfg, history, replay, identity):
    temporary = Path(str(path) + ".tmp")
    torch.save(dict(schema="native-proto-coadaptation-v1", mode=mode, epoch=epoch,
        state_dict=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
        history=history, feature_channels=replay.feature_channels, identity=identity,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        checkpoint_rule="fixed final epoch3; no dev selection"), temporary)
    temporary.replace(path)


def train(cfg, out, mode, deadline, smoke=False):
    for name in ("last.pt", "HISTORY.json", "SMOKE.json", "COMPLETE.json"):
        if (out / name).exists():
            raise RuntimeError("Existing artifacts; use a new Run, no automatic resume/overwrite")
    deadline_check(deadline)
    evidence = receipts(cfg, mode, smoke)
    index = load_index(cfg)
    planned = index["fit"]
    if len(planned) != 1024 or len(index.get("dev", [])) != 256 or index.get("val"):
        raise AssertionError("Frozen source cohort must remain fit1024/dev256/no val")
    if len({int(row["image_id"]) for row in planned}) != 1024:
        raise AssertionError("Repeated fit image identity")
    items = [row for row in planned if row["n"] > 0]
    setup(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    replay = FrozenReplay(cfg, device="cuda")
    original_state = digest(replay.source)
    model = build_model(mode, replay, cfg).to(replay.device).float().train()
    optimizer = optimizer_for(model, cfg)
    initial_state = digest(model)
    buffers = buffer_digest(model)
    frozen = {name: tensor_sha(parameter) for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if mode == "P" and any(name.startswith("native_cv4.") for name in trainable):
        raise AssertionError("P arm incorrectly trains coefficient branch")
    if mode == "PC" and not any(name.startswith("native_cv4.") for name in trainable):
        raise AssertionError("PC arm lacks trainable coefficient branch")
    if not any(name.startswith("native_proto.") for name in trainable):
        raise AssertionError("No trainable prototype parameters")
    identity = dict(original_weights_sha256=replay.weights_sha256, receipts=evidence,
        initialization_state_sha256=hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest(),
        frozen_fit_image_ids=[int(row["image_id"]) for row in planned])
    dump(out / "ENVIRONMENT.json", replay.import_info)
    dump(out / "RESOLVED_CONFIG.json", cfg)
    dump(out / "COHORT.json", dict(planned_images=1024, effective_images=len(items),
        candidates=sum(int(row["n"]) for row in items),
        no_positive_images=[int(row["image_id"]) for row in planned if not row["n"]],
        filtering="only original zero-positive images have no loss; no quality filtering", receipts=evidence))
    dump(out / "MODEL.json", dict(schema="native-proto-coadaptation-v1", mode=mode,
        trainable=trainable, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        feature_channels=replay.feature_channels, identity=identity,
        source_initial=original_state, bn_running_buffers="frozen original; BN affine native trainable except semseg",
        objective="unchanged official full640 BCE, original GT-box/area/gain and effective-batch candidate denominator",
        prototype_gradient="live full P, never cached A/K/G", deadline_unix=deadline))

    if smoke:
        subset = items[:2]
        batch = [load_asset(cfg, int(row["image_id"]), verify=True) for row in subset]
        denominator = sum(int(row["n"]) for row in subset)
        features = replay.replay(batch)
        output = model(features, batch)
        audits = []
        for image, coefficient, prototype in zip(batch, output["coefficients"], output["prototypes"]):
            torch.testing.assert_close(prototype.detach().cpu(), image["proto"].float().cpu(), atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(coefficient.detach().cpu(), image["c0"].float().cpu(), atol=3e-5, rtol=3e-5)
            audit = assert_fixed_prototype_loss_equivalence(image, coefficient, denominator)
            audits.append(dict(image_id=int(image["image_id"]), loss_equivalence=audit,
                initial_proto_max_abs=float((prototype.detach().cpu()-image["proto"].float().cpu()).abs().max()),
                initial_coefficient_max_abs=float((coefficient.detach().cpu()-image["c0"].float().cpu()).abs().max())))
        del output, features
        optimizer.zero_grad(set_to_none=True)
        value, seen = backward_batch(model, replay, batch, denominator, mode)
        gradient_by_branch = {}
        for prefix in ("native_proto.", "native_cv4."):
            gradient_by_branch[prefix] = sum(float(p.grad.detach().double().square().sum())
                for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)**0.5
        if gradient_by_branch["native_proto."] <= 0 or (mode == "PC" and gradient_by_branch["native_cv4."] <= 0):
            raise AssertionError("Declared native branch receives no gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
        optimizer.step()
        verify_states(model, replay, original_state, buffers, frozen)
        after = digest(model)
        changed = [name for name in trainable if initial_state[name] != after[name]]
        for prefix in (["native_proto."] if mode == "P" else ["native_proto.", "native_cv4."]):
            if not any(name.startswith(prefix) for name in changed):
                raise AssertionError(f"Declared trainable branch did not update: {prefix}")
        report = dict(passed=True, mode=mode, images=len(batch), candidates=seen,
            one_optimizer_update=True, original_initialization=True, audits=audits,
            summed_loss=value, gradient_norm_before_clip=norm, gradient_norm_by_branch=gradient_by_branch,
            changed_parameter_names=changed, frozen_source_parameters_buffers_unchanged=True,
            native_frozen_parameters_buffers_unchanged=True, fixed_P_operator_used=False)
        dump(out / "SMOKE.json", report)
        dump(out / "COMPLETE.json", dict(passed=True, completed=True, kind="smoke", mode=mode,
            report="SMOKE.json", no_training_checkpoint_reuse=True))
        return

    history, completed_epoch, current_epoch, current_step = [], 0, 0, 0
    started = time.monotonic()
    try:
        for epoch in range(cfg["epochs"]):
            deadline_check(deadline)
            current_epoch, current_step = epoch+1, 0
            order = list(items)
            random.Random(cfg["seed"]+epoch).shuffle(order)
            order_sha = hashlib.sha256(json.dumps([int(r["image_id"]) for r in order], separators=(",", ":")).encode()).hexdigest()
            epoch_started, total_loss, seen = time.monotonic(), 0., 0
            groups = math.ceil(len(order)/cfg["effective_batch_images"])
            for step, offset in enumerate(range(0, len(order), cfg["effective_batch_images"])):
                current_step = step+1
                deadline_check(deadline)
                group = order[offset:offset+cfg["effective_batch_images"]]
                denominator = sum(int(row["n"]) for row in group)
                factor = lr_factor(epoch+(step+1)/groups, cfg)
                for optim_group in optimizer.param_groups:
                    optim_group["lr"] = optim_group["initial_lr"]*factor
                optimizer.zero_grad(set_to_none=True)
                for inner in range(0, len(group), cfg["microbatch_images"]):
                    deadline_check(deadline)
                    subset = group[inner:inner+cfg["microbatch_images"]]
                    batch = [load_asset(cfg, int(row["image_id"]), verify=True) for row in subset]
                    if any(image["split"] != "fit" or len(image["raw_ids"]) != int(row["n"])
                           for image, row in zip(batch, subset)):
                        raise AssertionError("Fixed fit membership/count changed")
                    value, count = backward_batch(model, replay, batch, denominator, mode)
                    total_loss += value; seen += count
                    del batch
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
                optimizer.step()
                if step % 20 == 0 or step+1 == groups:
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-started
                    fraction = epoch+(step+1)/groups
                    state = dict(stage="training", mode=mode, epoch=epoch+1, epochs=3,
                        step=step+1, steps=groups, candidates_seen_this_epoch=seen,
                        trajectory_bce=total_loss/max(seen, 1), gradient_norm_before_clip=norm,
                        native_lr=cfg["branch_lr"]*factor, elapsed_s=elapsed,
                        remaining_budget_s=deadline-time.time(),
                        estimated_arm_remaining_s=elapsed*(3-fraction)/max(fraction, 1e-9),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    dump(out / "PROGRESS.json", state); print(json.dumps(state), flush=True)
            if seen != sum(int(row["n"]) for row in items):
                raise AssertionError("Epoch candidate population changed")
            verify_states(model, replay, original_state, buffers, frozen)
            history.append(dict(epoch=epoch+1, trajectory_bce=total_loss/seen,
                trajectory_not_fixed_checkpoint_objective=True, candidates=seen, images=len(items),
                image_order_sha256=order_sha, seconds=time.monotonic()-epoch_started,
                source_and_native_frozen_states_unchanged=True))
            completed_epoch = epoch+1
            save_checkpoint(out / "last.pt", model, optimizer, completed_epoch, mode, cfg, history, replay, identity)
            dump(out / "HISTORY.json", history)
        save_checkpoint(out / "final.pt", model, optimizer, 3, mode, cfg, history, replay, identity)
        dump(out / "COMPLETE.json", dict(completed=True, kind="training", mode=mode, epochs=3,
            checkpoint="final.pt", checkpoint_rule="fixed final epoch3", elapsed_training_s=time.monotonic()-started,
            ended_unix=time.time(), all_frozen_states_unchanged=True, automatic_retry=False, automatic_resume=False))
    except TimeoutError:
        dump(out / "BUDGET_STOP.json", dict(completed_epoch=completed_epoch, current_epoch=current_epoch,
            current_step=current_step, deadline_unix=deadline, scientific_status="incomplete; not a negative result",
            last_complete_epoch_not_final_epoch3=True, automatic_retry=False, automatic_resume=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("P", "PC"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt" or not torch.cuda.is_available():
            raise RuntimeError("Model work is only authorized on Linux CUDA server")
        cfg = resolve_config(load_json(args.config))
        if args.deadline > time.time()+MAX_SHARED_SECONDS+60:
            raise ValueError("Deadline exceeds shared 40-minute P/PC training budget")
        train(cfg, out, args.mode, args.deadline, args.smoke)
    except BaseException as error:
        if isinstance(error, TimeoutError) and not (out / "BUDGET_STOP.json").exists():
            dump(out / "BUDGET_STOP.json", dict(completed_epoch=0, mode=args.mode,
                scientific_status="incomplete; budget exhausted", automatic_retry=False))
        dump(out / "FAILURE.json", dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), mode=args.mode, smoke=args.smoke,
            automatic_retry=False, automatic_resume=False))
        raise


if __name__ == "__main__":
    main()
