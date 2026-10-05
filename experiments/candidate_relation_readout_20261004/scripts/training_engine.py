"""Train native one-to-one cv4 jointly with the current box-evidence method.

N/S/T/M use identical immutable official candidates, batch order, training budget
and original mask BCE. Only native cv4 and (S/T/M) evidence are in the optimizer.
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

from relation_head import RelationReadout
from runtime_utils import dump, load, setup, lr_factor, bn_state, verify_bn
from relation_io import read_training_image, feed
from train_evidence import (
    operator_fingerprint, feature_channels,
    backwards, checked_loss, new_stats, add_stats, finish_stats,
    save_checkpoint, choose_smoke_images, direct_bce_check,
    independent_projection_check, expected_evidence_gradient,
)


FIXED_SETTINGS = {
    "seed": 0, "epochs": 3, "microbatch_images": 2,
    "effective_batch_images": 16, "branch_lr": 0.0001,
    "new_lr": 0.0003, "weight_decay": 0.0001,
    "warmup_epochs": 1, "eta_ratio": 0.1, "gradient_clip_norm": 10,
}
RECORDED_EPOCHS = (1, 2, 3)


def resolve_config(config):
    cfg = dict(config)
    for key, expected in FIXED_SETTINGS.items():
        cfg.setdefault(key, expected)
        if cfg[key] != expected:
            raise ValueError(f"Candidate relation protocol fixes {key}={expected}; got {cfg[key]}")
    cfg.setdefault("arms", ["N", "S", "T", "M"])
    if set(cfg["arms"]) != {"N", "S", "T", "M"} or len(cfg["arms"]) != 4:
        raise ValueError("N/S/T/M must each be registered exactly once")
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
    model = RelationReadout(native, channels, mode, cfg).cuda().float().train()
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
        evidence_conditioning="single step; frozen GT-free neighbor response maps; live own c/h retained",
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
                    note="No method conclusion. Resume last complete epoch; partial epoch is not epoch3."))
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
            if step % 10 == 0 or step + 1 == groups:
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
        checkpoint="final.pt", checkpoint_rule="fixed epoch3; no dev/val selection",
        bn_unchanged=True, all_buffers_unchanged=True, elapsed_s=time.monotonic() - started))




