"""Train native one-to-one cv4 jointly with the current box-evidence method.

N/S/D use identical immutable official candidates, batch order, training budget
and original mask BCE. Only native cv4 and (S/D) evidence are in the optimizer.
Upstream F/P, boxes, labels, assignment and prototype projection are cached and
fixed. No new oracle/loss/gate/threshold is introduced. GPU-host execution only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO

from joint_head import JointCoefficientReadout
from runtime_utils import dump, load, setup, lr_factor, bn_state, verify_bn
from train_evidence import (
    operator_fingerprint, read_training_image, feature_channels, feed,
    backwards, checked_loss, new_stats, add_stats, finish_stats,
    save_checkpoint, choose_smoke_images, direct_bce_check,
    independent_projection_check, expected_evidence_gradient,
)


FIXED_SETTINGS = {
    "seed": 0, "epochs": 12, "microbatch_images": 2,
    "effective_batch_images": 16, "branch_lr": 0.0001,
    "new_lr": 0.0003, "weight_decay": 0.0001,
    "warmup_epochs": 1, "eta_ratio": 0.1, "gradient_clip_norm": 10,
}
RECORDED_EPOCHS = (1, 4, 8, 12)


def resolve_config(config):
    cfg = dict(config)
    for key, expected in FIXED_SETTINGS.items():
        cfg.setdefault(key, expected)
        if cfg[key] != expected:
            raise ValueError(f"Joint protocol fixes {key}={expected}; got {cfg[key]}")
    cfg.setdefault("arms", ["N", "S", "D"])
    if set(cfg["arms"]) != {"N", "S", "D"} or len(cfg["arms"]) != 3:
        raise ValueError("N/S/D must each be registered exactly once")
    if not all(cfg.get(key) for key in ("cache", "operator_cache", "weights")):
        raise ValueError("Original cache, unchanged operator cache and original weights are required")
    return cfg


def make_model(channels, mode, cfg):
    setup(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = YOLO(cfg["weights"]).model.cpu().float().eval().requires_grad_(False)
    native = source.model[-1].one2one_cv4
    # Loading a checkpoint must not affect paired evidence initialization.
    setup(cfg["seed"])
    model = JointCoefficientReadout(native, channels, mode, cfg).cuda().float().train()
    del source
    return model


def optimizer_for(model, cfg):
    grouped = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = (name.startswith("native_cv4."), parameter.ndim > 1)
            grouped.setdefault(key, []).append(parameter)
    groups = []
    for (native, decay), parameters in grouped.items():
        learning_rate = cfg["branch_lr"] if native else cfg["new_lr"]
        groups.append(dict(params=parameters, lr=learning_rate, initial_lr=learning_rate,
                           weight_decay=cfg["weight_decay"] if decay else 0.,
                           group_name=("native" if native else "evidence") + ("_decay" if decay else "_no_decay")))
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def buffer_manifest(model):
    return {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                      sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            for name, value in model.named_buffers()}


def expected_residual_gradient(model, selection, gradient):
    if model.mode == "S":
        return expected_evidence_gradient(selection, gradient)
    boxes = selection["boxes"]
    clamped = boxes.clamp(0, 640)
    valid = selection["valid"] & torch.isfinite(boxes).all(1)
    valid &= ((clamped[:, 2:] - clamped[:, :2]) > 0).all(1)
    result = torch.zeros((len(gradient), 256), device=gradient.device, dtype=gradient.dtype)
    result[valid] = F.linear(gradient[valid], model.evidence.direct_projection.weight.T)
    return result.detach()


def smoke(cfg, out, index, deadline=0.):
    images, scanned = choose_smoke_images(cfg, index)
    channels = feature_channels(images[0])
    total = sum(len(x["raw_ids"]) for x in images)
    report = dict(ultralytics_import=ultralytics.__version__, source=str(ultralytics.__file__),
        image_ids=[x["image_id"] for x in images], fit_images_scanned=scanned,
        feature_channels=channels, candidates=total, coefficient_tolerance=dict(atol=3e-5, rtol=3e-5),
        arms={}, scope="GPU implementation verification; reset original weights for every formal arm")
    native_initial = evidence_initial = None
    for mode in cfg["arms"]:
        if deadline and time.time() > deadline:
            raise TimeoutError("Joint smoke deadline reached")
        model = make_model(channels, mode, cfg)
        optimizer = optimizer_for(model, cfg)
        original_bn = bn_state(model)
        original_buffers = buffer_manifest(model)
        before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        native_parameters = {name: value for name, value in before.items() if name.startswith("native_cv4.")}
        if native_initial is None:
            native_initial = native_parameters
        else:
            if native_parameters.keys() != native_initial.keys():
                raise AssertionError("Native branch parameter names differ across arms")
            for name, value in native_parameters.items():
                torch.testing.assert_close(value, native_initial[name], atol=0, rtol=0)
        if mode != "N":
            common = {name: value for name, value in before.items()
                      if name.startswith("evidence.") and not name.startswith("evidence.direct_projection.")}
            if evidence_initial is None:
                evidence_initial = common
            else:
                if common.keys() != evidence_initial.keys():
                    raise AssertionError("S/D evidence parameter names differ")
                for name, value in common.items():
                    torch.testing.assert_close(value, evidence_initial[name], atol=0, rtol=0)
        initial_c_error = initial_h_error = direct_error = projection_error = chain_error = 0.
        chain_checks = 0
        evidence_to_current = {"h_current": 0., "c_current": 0.}
        grads, step_grads, losses, stats = {}, [], [], new_stats()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for lo in range(0, len(images), cfg["microbatch_images"]):
                batch = images[lo:lo + cfg["microbatch_images"]]
                fs, selected = feed(batch)
                original_k = [row["K"].clone() for row in selected]
                coefficients, residuals, current = model.forward_details(fs, selected)
                gradients = []
                for x, c, row, live in zip(batch, coefficients, selected, current):
                    if step == 0:
                        torch.testing.assert_close(c.detach(), row["c0"], atol=3e-5, rtol=3e-5)
                        torch.testing.assert_close(live["h_current"].detach(), row["h0"], atol=3e-5, rtol=3e-5)
                        initial_c_error = max(initial_c_error, float((c.detach() - row["c0"]).abs().max()))
                        initial_h_error = max(initial_h_error, float((live["h_current"].detach() - row["h0"]).abs().max()))
                    value, gradient = checked_loss(x, c, total)
                    gradients.append(gradient)
                    total_loss += value
                    if step == 0 and lo == 0:
                        direct_error = max(direct_error, direct_bce_check(x, c, gradient, total))
                        if mode == "S":
                            projection_error = max(projection_error, independent_projection_check(row, gradient))
                expected = []
                if mode != "N":
                    expected = [expected_residual_gradient(model, row, g) for row, g in zip(selected, gradients)]
                    for residual in residuals:
                        residual.retain_grad()
                    if step == 1:
                        # Differentiate ONLY residual evidence, excluding the
                        # native coefficient passthrough. Nonzero derivatives
                        # establish the requested joint conditioning path.
                        live_inputs = [row[key] for row in current for key in ("h_current", "c_current")]
                        live_grads = torch.autograd.grad(residuals, live_inputs, grad_outputs=expected,
                                                       retain_graph=True, allow_unused=True)
                        for position, gradient in enumerate(live_grads):
                            if gradient is not None:
                                if not bool(torch.isfinite(gradient).all()):
                                    raise FloatingPointError("Nonfinite evidence-to-native derivative")
                                key = "h_current" if position % 2 == 0 else "c_current"
                                evidence_to_current[key] += float(gradient.norm())
                        # autograd.grad may populate retained nonleaf grads;
                        # clear these before checking the actual BCE backward.
                        for residual in residuals:
                            residual.grad = None
                backwards(coefficients, gradients)
                if mode != "N":
                    for residual, target in zip(residuals, expected):
                        if residual.grad is None:
                            raise AssertionError("Evidence residual missing from actual loss path")
                        torch.testing.assert_close(residual.grad, target, atol=3e-6, rtol=3e-5)
                        chain_error = max(chain_error, float((residual.grad - target).abs().max()))
                        chain_checks += 1
                for row, saved_k in zip(selected, original_k):
                    torch.testing.assert_close(row["K"], saved_k, atol=0, rtol=0)
                    if any(value.requires_grad for value in row.values() if torch.is_tensor(value)):
                        raise AssertionError("A cached detector/operator input acquired gradients")
                add_stats(stats, coefficients, selected)
            current_grads = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all()):
                        raise FloatingPointError(f"Nonfinite parameter gradient: {mode}/{name}")
                    current_grads[name] = float(p.grad.norm())
                    grads[name] = max(grads.get(name, 0.), current_grads[name])
            step_grads.append(current_grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            verify_bn(model, original_bn)
            if buffer_manifest(model) != original_buffers:
                raise AssertionError("Frozen module buffers changed during smoke")
            losses.append(total_loss / total)
        changed = {name: float((p.detach().cpu() - before[name]).norm()) for name, p in model.named_parameters()}
        for level, branch in enumerate(model.native_cv4):
            for layer in range(len(branch)):
                prefix = f"native_cv4.{level}.{layer}."
                if not any(value > 0 for name, value in grads.items() if name.startswith(prefix)):
                    raise AssertionError(f"{mode}: no gradient in native level/layer {level}/{layer}")
                if not any(value > 0 for name, value in changed.items() if name.startswith(prefix)):
                    raise AssertionError(f"{mode}: no update in native level/layer {level}/{layer}")
        if mode != "N":
            if not all(value > 0 for value in evidence_to_current.values()):
                raise AssertionError(f"{mode}: missing residual gradient to current h/c: {evidence_to_current}")
            for group, prefixes in {
                "encoder": ("evidence.encoder_adapters.", "evidence.encoder_fusion."),
                "query": ("evidence.query_mlp.", "evidence.query_film."),
                "decoder": ("evidence.evidence_decoder.",),
            }.items():
                if not any(value > 0 for name, value in step_grads[1].items() if name.startswith(prefixes)):
                    raise AssertionError(f"{mode}: evidence {group} missing second-step gradient")
                if not any(value > 0 for name, value in changed.items() if name.startswith(prefixes)):
                    raise AssertionError(f"{mode}: evidence {group} did not update")
        report["arms"][mode] = dict(initial_coefficient_max_error=initial_c_error,
            initial_h_max_error=initial_h_error, parameter_counts=model.parameter_counts(),
            losses=losses, gradient_norms=grads, gradient_norms_by_step=step_grads,
            parameter_changes=changed, official_bce_gradient_max_error=direct_error,
            independent_projection_gradient_max_error=projection_error if mode == "S" else None,
            actual_residual_gradient_max_error=chain_error if mode != "N" else None,
            actual_residual_chain_checks=chain_checks,
            evidence_only_gradient_to_current_features=evidence_to_current if mode != "N" else None,
            frozen_input_and_operator_tensors=True, all_bn_buffers_unchanged=True,
            all_module_buffers_unchanged=True, buffers=original_buffers, diagnostics=finish_stats(stats))
        dump(out / "SMOKE.json", report)
        del optimizer, model
        torch.cuda.empty_cache()
    report["native_initialization_identical"] = True
    report["shared_S_D_evidence_initialization_identical"] = True
    dump(out / "SMOKE.json", report)
    dump(out / "COMPLETE.json", dict(passed=True, kind="smoke", formal_training_steps=0,
                                     trained_smoke_weights_saved=False))


def train(cfg, out, index, mode, deadline):
    items = [row for row in index["fit"] if row["n"] > 0]
    if not items:
        raise ValueError("No official positive fit images")
    first = read_training_image(cfg, items[0]["image_id"])
    channels = feature_channels(first)
    del first
    model = make_model(channels, mode, cfg)
    optimizer = optimizer_for(model, cfg)
    frozen_bn = bn_state(model)
    frozen_buffers = buffer_manifest(model)
    dump(out / "MODEL.json", dict(mode=mode, counts=model.parameter_counts(), feature_channels=channels,
        trainable=model.trainable_parameter_names(), native_branch="head.one2one_cv4; all convolutions and BN affine",
        bn_statistics="original checkpoint buffers frozen in eval mode", buffers=frozen_buffers,
        objective="official full-640 GT-cropped area-normalized BCE only; gain 9.83241",
        cached_h0_c0_role="initialization checks and diagnostics only; live h/c recomputed from F",
        evidence_conditioning="current h and c; gradients retained to native branch",
        actual_ultralytics=ultralytics.__version__, source=str(ultralytics.__file__)))
    dump(out / "COHORT.json", dict(planned_images=len(index["fit"]), effective_images=len(items),
        candidates=sum(row["n"] for row in items),
        no_positive_images=[row["image_id"] for row in index["fit"] if not row["n"]],
        candidate_filtering="none; invalid operators receive zero residual but retain trainable native coefficient and official loss"))
    dump(out / "RESOLVED_CONFIG.json", cfg)
    start_epoch, history, last = 0, [], out / "last.pt"
    if last.exists():
        checkpoint = load(last)
        if checkpoint["mode"] != mode or checkpoint["config"] != cfg or checkpoint["feature_channels"] != channels:
            raise ValueError("Resume identity/configuration mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
        verify_bn(model, frozen_bn)
        if buffer_manifest(model) != frozen_buffers:
            raise AssertionError("Resume changed frozen buffers")
    started = time.monotonic()
    for epoch in range(start_epoch, cfg["epochs"]):
        order = list(items)
        random.Random(cfg["seed"] + epoch).shuffle(order)
        order_hash = hashlib.sha256(json.dumps([int(row["image_id"]) for row in order], separators=(",", ":")).encode()).hexdigest()
        epoch_start, total_loss, seen = time.monotonic(), 0., 0
        groups, stats = math.ceil(len(order) / cfg["effective_batch_images"]), new_stats()
        for step, lo in enumerate(range(0, len(order), cfg["effective_batch_images"])):
            if deadline and time.time() > deadline:
                dump(out / "BUDGET_STOP.json", dict(epoch_completed=epoch, mid_epoch_step=step,
                    note="No method conclusion. Resume last complete epoch; partial epoch is not epoch12."))
                raise TimeoutError("Joint training resource deadline reached")
            rows = order[lo:lo + cfg["effective_batch_images"]]
            denominator = sum(row["n"] for row in rows)
            factor = lr_factor(epoch + (step + 1) / groups, cfg)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(rows), cfg["microbatch_images"]):
                chosen = rows[offset:offset + cfg["microbatch_images"]]
                batch = [read_training_image(cfg, row["image_id"]) for row in chosen]
                if any(len(x["raw_ids"]) != row["n"] for x, row in zip(batch, chosen)):
                    raise ValueError("Frozen official candidate count changed")
                features, selected = feed(batch)
                coefficients = model(features, selected)
                gradients = []
                for x, coefficient in zip(batch, coefficients):
                    value, gradient = checked_loss(x, coefficient, denominator)
                    gradients.append(gradient)
                    total_loss += value
                    seen += len(x["raw_ids"])
                backwards(coefficients, gradients)
                add_stats(stats, coefficients, selected)
                del batch, features, coefficients, gradients, selected
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
            optimizer.step()
            if step % 25 == 0 or step + 1 == groups:
                state = dict(mode=mode, epoch=epoch + 1, epochs=cfg["epochs"], step=step + 1, steps=groups,
                    trajectory_bce=total_loss / max(1, seen), grad_norm_before_clip=grad,
                    branch_learning_rate=cfg["branch_lr"] * factor,
                    new_learning_rate=cfg["new_lr"] * factor if mode != "N" else None,
                    elapsed_s=time.monotonic() - started, epoch_elapsed_s=time.monotonic() - epoch_start,
                    diagnostics=finish_stats(stats))
                dump(out / "PROGRESS.json", state)
                print(json.dumps(state), flush=True)
        verify_bn(model, frozen_bn)
        if buffer_manifest(model) != frozen_buffers:
            raise AssertionError("A frozen module buffer changed")
        history.append(dict(epoch=epoch + 1, trajectory_bce=total_loss / seen, candidates=seen,
            seconds=time.monotonic() - epoch_start, image_order_sha256=order_hash,
            bn_unchanged=True, all_buffers_unchanged=True, diagnostics=finish_stats(stats)))
        save_checkpoint(last, model, optimizer, epoch + 1, mode, cfg, history, channels)
        if epoch + 1 in RECORDED_EPOCHS:
            save_checkpoint(out / f"epoch{epoch + 1:02d}.pt", model, optimizer, epoch + 1, mode, cfg, history, channels)
        dump(out / "HISTORY.json", history)
    save_checkpoint(out / "final.pt", model, optimizer, cfg["epochs"], mode, cfg, history, channels)
    dump(out / "COMPLETE.json", dict(completed=True, kind="training", mode=mode, epochs=cfg["epochs"],
        checkpoint="final.pt", checkpoint_rule="fixed epoch12; no dev/val selection",
        bn_unchanged=True, all_buffers_unchanged=True, elapsed_s=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index")
    parser.add_argument("--mode", choices=("N", "S", "D"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--deadline", type=float, default=0.)
    args = parser.parse_args()
    cfg = resolve_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.index or str(Path(cfg["cache"]) / "INDEX.json")).read_text(encoding="utf-8-sig"))
    if not torch.cuda.is_available():
        raise RuntimeError("Run only on the authorized GPU host; desktop/CPU training is forbidden")
    try:
        if args.smoke:
            smoke(cfg, out, index, args.deadline)
        else:
            if args.mode is None:
                parser.error("--mode is required for formal training")
            if not (Path(cfg["cache"]) / "COMPLETE.json").exists():
                raise RuntimeError("Original frozen cohort cache is incomplete")
            complete = json.loads((Path(cfg["operator_cache"]) / "COMPLETE.json").read_text(encoding="utf-8-sig"))
            if complete.get("smoke") or complete.get("fingerprint") != operator_fingerprint(cfg):
                raise RuntimeError("Frozen projection operator cache fingerprint mismatch")
            train(cfg, out, index, args.mode, args.deadline)
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc), mode=args.mode, smoke=args.smoke))
        raise


if __name__ == "__main__":
    main()
