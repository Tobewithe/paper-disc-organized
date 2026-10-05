"""Three fixed-budget native coefficient-branch bridge arms; server CUDA only."""
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
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf-8-sig"))
    sys.path.insert(0, _cfg["source_python"])
    for _key in ("screen_root", "dense_root"):
        if _cfg.get(_key):
            sys.path.append(str(Path(_cfg[_key]) / "scripts"))

import torch

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256
import dense_runtime as dr
from bridge_model import BridgeReadout
from prepare_features import load_feature_cache, selection_for

FIXED = dict(seed=0, epochs=3, microbatch_images=2, effective_batch_images=16,
             branch_lr=1e-4, bridge_lr=1e-4, weight_decay=1e-4,
             warmup_epochs=1, eta_ratio=.1, gradient_clip_norm=10)


def resolve_config(config):
    cfg = resolve_runtime_config(config)
    for key, expected in FIXED.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Fixed paired setting {key}={expected}; got {cfg.get(key)}")
    limit = cfg.get("training_max_seconds", cfg.get("train_max_seconds", 1800))
    if limit != 1800:
        raise ValueError("Each arm has a fixed 1800-second total budget")
    cfg["training_max_seconds"] = limit
    return cfg


def optimizer_for(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or not name.startswith(("native_cv4.", "bridges.")):
            raise AssertionError("Unexpected trainable scope: " + name)
        family = "native" if name.startswith("native_cv4.") else "bridge"
        groups.setdefault((family, p.ndim > 1), []).append(p)
    return torch.optim.AdamW([
        dict(params=ps, lr=cfg["branch_lr"] if family == "native" else cfg["bridge_lr"],
             initial_lr=cfg["branch_lr"] if family == "native" else cfg["bridge_lr"],
             weight_decay=cfg["weight_decay"] if decay else 0.,
             group_name=family + ("_decay" if decay else "_no_decay"))
        for (family, decay), ps in groups.items()], betas=(.9, .999), eps=1e-8)


def initialization_control(cfg, replay, model):
    """Compare precisely the unchanged native part, not bridge/statistic buffers."""
    original = dr.new_model(replay, cfg)
    control = dr.check_control(cfg, original)
    before, now = original.native_cv4.state_dict(), model.native_cv4.state_dict()
    if set(before) != set(now):
        raise AssertionError("Native initialization keys differ from historical N")
    for name in before:
        torch.testing.assert_close(now[name], before[name], atol=0, rtol=0)
    for name, p in model.named_parameters():
        if name.startswith("bridges.") and bool(p.detach().count_nonzero()):
            raise AssertionError("Bridge initialization must be exactly zero: " + name)
    control["native_initialization_exact"] = True
    control["bridge_zero_initialization"] = True
    del original
    return control


def feature_file_digests(cfg):
    path = Path(cfg["feature_cache"])
    files = [path, Path(str(path) + ".meta.json")] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError("No prepared feature assets: " + str(path))
    return {str(p): sha256(p) for p in files}


def check_receipts(cfg):
    root = Path(cfg["server_root"])
    ids = load_json(root / "RUN_IDS.json")["runs"]
    receipts = {}
    for stage in ("prepare", "smoke"):
        path = Path(cfg[stage + "_run"]) / "COMPLETE.json" if cfg.get(stage + "_run") else root / "runs" / ids[stage] / "COMPLETE.json"
        record = load_json(path)
        if not record.get("passed", record.get("completed", record.get("complete", False))):
            raise AssertionError(stage + " is not complete/passed")
        receipts[stage] = dict(path=str(path), sha256=sha256(path))
    return receipts


def check_gradients(model):
    values = {"native": 0., "bridge": 0.}
    missing = []
    for name, p in model.named_parameters():
        if p.grad is None:
            # A scale with no selected candidate may have no bridge autograd path.
            # This is legitimate; overall bridge learning is checked separately.
            if not name.startswith("bridges."):
                raise AssertionError("Missing native parameter gradient: " + name)
            missing.append(name)
            continue
        if not bool(torch.isfinite(p.grad).all()):
            raise FloatingPointError("Nonfinite parameter gradient: " + name)
        key = "native" if name.startswith("native_cv4.") else "bridge"
        values[key] += float(p.grad.detach().double().square().sum())
    return dict(**{key: math.sqrt(value) for key, value in values.items()}, inactive_bridge_parameters=missing)


def save_checkpoint(path, model, optimizer, epoch, cfg, history, channels, identity):
    temporary = Path(str(path) + ".tmp")
    torch.save(dict(mode=model.mode, epoch=epoch, state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(), config=cfg, history=history,
                    feature_channels=channels, feature_identity=identity,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    checkpoint_rule="fixed final epoch3; no dev checkpoint selection"), temporary)
    temporary.replace(path)


def train(cfg, out, mode):
    started = time.monotonic()
    deadline = started + cfg["training_max_seconds"]

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed per-arm 1800-second budget exhausted; no automatic resume")

    if any((out / name).exists() for name in ("last.pt", "HISTORY.json", "UPDATES.jsonl", "COMPLETE.json")):
        raise RuntimeError("Preserve prior Run; no automatic overwrite or resume")
    receipts = check_receipts(cfg)
    index = load_index(cfg)
    planned = index["fit"]
    items = [r for r in planned if r["n"] > 0]
    if len(planned) != 1024 or len(index.get("dev", [])) != 256 or index.get("val"):
        raise AssertionError("Require frozen 1024/256/no-val cohort")
    if len({int(r["image_id"]) for r in planned}) != len(planned):
        raise AssertionError("Duplicate fit images")
    bank = load_feature_cache(cfg)
    feature_identity = feature_file_digests(cfg)
    dr.setup(cfg["seed"])
    replay = FrozenReplay(cfg, device="cuda")
    source_initial = dr.state_digest(replay.source)
    dr.setup(cfg["seed"])
    model = BridgeReadout(replay.native_cv4, replay.feature_channels, mode, cfg, bank["stats"]).to(replay.device).float().train()
    control = initialization_control(cfg, replay, model)
    optimizer = optimizer_for(model, cfg)
    initial_buffers = dr.buffers(model)
    initial_parameters = {k: p.detach().clone() for k, p in model.named_parameters()}
    initial_hash = hashlib.sha256(json.dumps(dr.state_digest(model), sort_keys=True).encode()).hexdigest()
    identity = dict(files=feature_identity, receipts=receipts, control=control)
    dump(out / "ENVIRONMENT.json", replay.import_info)
    dump(out / "RESOLVED_CONFIG.json", cfg)
    dump(out / "MODEL.json", dict(mode=mode, counts=model.parameter_counts(),
        trainable=model.trainable_parameter_names(), initialization_state_sha256=initial_hash,
        feature_channels=replay.feature_channels, source_initial=source_initial,
        bn_running_buffers="fixed original checkpoint; native BN affine trainable",
        objective="unchanged official sparse full640 GT-box BCE with original gain; candidate mean per effective batch",
        regularization="optimizer weight decay only as paired historical N; no output/teacher/new loss",
        source_features="T true cls hidden; R true box hidden; M other-image matched predicted-class/level cls hidden",
        feature_identity=identity, budget_seconds=cfg["training_max_seconds"]))
    dump(out / "COHORT.json", dict(planned_images=len(planned), effective_images=len(items),
        candidates=sum(int(r["n"]) for r in items),
        no_positive_images=[int(r["image_id"]) for r in planned if not r["n"]],
        domain="fixed original official one-to-one TAL fit; no reassignments/quality filtering"))
    control_history = load_json(Path(cfg["control_run"]) / "HISTORY.json")
    if len(control_history) != cfg["epochs"]:
        raise AssertionError("Historical N epoch count mismatch")
    history, updates = [], 0
    with (out / "UPDATES.jsonl").open("x", encoding="utf-8") as stream:
        for epoch in range(cfg["epochs"]):
            budget()
            order = list(items)
            random.Random(cfg["seed"] + epoch).shuffle(order)
            order_sha = hashlib.sha256(json.dumps([int(r["image_id"]) for r in order], separators=(",", ":")).encode()).hexdigest()
            if order_sha != control_history[epoch]["image_order_sha256"]:
                raise AssertionError("Epoch image order differs from historical N")
            epoch_started, total, seen = time.monotonic(), 0., 0
            residual_l2_sum = hidden_bridge_l2_sum = native_coefficient_l2_sum = 0.
            groups = math.ceil(len(order) / cfg["effective_batch_images"])
            for step, offset in enumerate(range(0, len(order), cfg["effective_batch_images"])):
                budget()
                group = order[offset:offset + cfg["effective_batch_images"]]
                denominator = sum(int(r["n"]) for r in group)
                factor = dr.lr_factor(epoch + (step + 1) / groups, cfg)
                for opt_group in optimizer.param_groups:
                    opt_group["lr"] = opt_group["initial_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                for inner in range(0, len(group), cfg["microbatch_images"]):
                    budget()
                    subset = group[inner:inner + cfg["microbatch_images"]]
                    images = [load_asset(cfg, r["image_id"], verify=True) for r in subset]
                    if any(len(x["raw_ids"]) != int(r["n"]) for x, r in zip(images, subset)):
                        raise AssertionError("Frozen official candidate count changed")
                    selected = [selection_for(x, bank, mode, replay.device) for x in images]
                    features = replay.replay(images)
                    coefficients, residuals, current = model.forward_details(features, selected)
                    gradients = []
                    for x, c, delta, row in zip(images, coefficients, residuals, current):
                        value, gradient = dr.weighted_loss_and_grad(dr.gpu_payload(x, replay.device), c,
                            denominator, torch.ones(len(c), device=c.device, dtype=c.dtype))
                        if not bool(torch.isfinite(gradient).all()):
                            raise FloatingPointError("Nonfinite coefficient gradient")
                        total += value
                        seen += len(c)
                        residual_l2_sum += float(delta.detach().norm(dim=1).sum())
                        hidden_bridge_l2_sum += float(row["h_bridge"].detach().norm(dim=1).sum())
                        native_coefficient_l2_sum += float(row["c_current"].detach().norm(dim=1).sum())
                        gradients.append(gradient)
                    torch.autograd.backward(coefficients, gradients)
                    del images, selected, features, coefficients, gradients, residuals, current
                norms = check_gradients(model)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
                optimizer.step()
                updates += 1
                record = dict(epoch=epoch+1, step=step+1, update=updates, denominator=denominator,
                              gradient_norm_before_clip=gradient_norm, family_gradient_norms=norms,
                              native_lr=cfg["branch_lr"]*factor, bridge_lr=cfg["bridge_lr"]*factor)
                stream.write(json.dumps(record) + "\n")
                if step % 25 == 0 or step + 1 == groups:
                    stream.flush()
                    torch.cuda.synchronize()
                    progress = dict(stage="training", mode=mode, epoch=epoch+1, epochs=3, step=step+1,
                        steps=groups, candidates_seen=seen, trajectory_bce=total/max(seen, 1),
                        gradient_norm_before_clip=gradient_norm, family_gradient_norms=norms,
                        trajectory_coefficient_bridge_l2_mean=residual_l2_sum/max(seen, 1),
                        trajectory_hidden_bridge_l2_mean=hidden_bridge_l2_sum/max(seen, 1),
                        elapsed_s=time.monotonic()-started, remaining_budget_s=deadline-time.monotonic(),
                        peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    dump(out / "PROGRESS.json", progress)
                    print(json.dumps(progress), flush=True)
            if seen != sum(int(r["n"]) for r in items):
                raise AssertionError("Epoch did not use exactly frozen fit candidates")
            dr.verify_buffers(model, initial_buffers)
            replay.assert_unchanged()
            if dr.state_digest(replay.source) != source_initial:
                raise AssertionError("Frozen source/prefix/heads changed")
            history.append(dict(epoch=epoch+1, trajectory_bce=total/seen,
                trajectory_not_fixed_checkpoint_objective=True, candidates=seen, images=len(items),
                trajectory_coefficient_bridge_l2_mean=residual_l2_sum/seen,
                trajectory_hidden_bridge_l2_mean=hidden_bridge_l2_sum/seen,
                trajectory_native_coefficient_l2_mean=native_coefficient_l2_sum/seen,
                seconds=time.monotonic()-epoch_started, image_order_sha256=order_sha,
                paired_N_image_order=True, bn_unchanged=True, all_buffers_unchanged=True,
                replay_source_unchanged=True))
            save_checkpoint(out / "last.pt", model, optimizer, epoch+1, cfg, history, replay.feature_channels, identity)
            dump(out / "HISTORY.json", history)
        budget()
        changed = {family: [k for k, p in model.named_parameters() if k.startswith(prefix)
                           and not torch.equal(p.detach(), initial_parameters[k])]
                   for family, prefix in (("native", "native_cv4."), ("bridge", "bridges."))}
        if not all(changed.values()):
            raise AssertionError("Both native and bridge parameters must update")
        displacement = {family: math.sqrt(sum(float((p.detach()-initial_parameters[k]).double().square().sum())
                            for k, p in model.named_parameters() if k.startswith(prefix)))
                        for family, prefix in (("native", "native_cv4."), ("bridge", "bridges."))}
        if feature_file_digests(cfg) != feature_identity:
            raise AssertionError("Frozen feature assets changed during training")
        if sha256(cfg["control_checkpoint"]) != control["checkpoint_sha256"]:
            raise AssertionError("Historical control checkpoint changed")
        save_checkpoint(out / "final.pt", model, optimizer, 3, cfg, history, replay.feature_channels, identity)
        dump(out / "UPDATE_AUDIT.json", dict(passed=True, updates=updates, changed_parameters=changed,
             parameter_displacement_l2=displacement, bridge_and_native_updated=True,
             source_unchanged=True, BN_and_feature_statistics_buffers_unchanged=True,
             note="Same image budget/optimizer does not imply equal gradients or parameter counts for R and T"))
        dump(out / "COMPLETE.json", dict(completed=True, passed=True, kind="training", mode=mode,
             epochs=3, checkpoint="final.pt", checkpoint_sha256=sha256(out / "final.pt"),
             checkpoint_rule="fixed epoch3; no dev selection", elapsed_training_s=time.monotonic()-started,
             all_frozen_states_unchanged=True, automatic_retry=False, automatic_resume=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("T", "R", "M"), required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # A refusal to overwrite must not itself overwrite an old failure/receipt.
    if any((out / name).exists() for name in ("COMPLETE.json", "FAILURE.json", "HISTORY.json", "last.pt", "UPDATES.jsonl")):
        raise RuntimeError("Prior training artifacts exist; preserve this Run and register a separate attempt")
    try:
        if os.name == "nt" or not torch.cuda.is_available():
            raise RuntimeError("Only the authorized Linux CUDA server may execute this experiment")
        train(resolve_config(load_json(args.config)), out, args.mode)
    except BaseException as exc:
        if not (out / "FAILURE.json").exists():
            dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                 mode=args.mode, automatic_retry=False, automatic_resume=False))
        if isinstance(exc, TimeoutError):
            dump(out / "BUDGET_STOP.json", dict(mode=args.mode, scientific_status="incomplete; not fixed epoch3", automatic_resume=False))
        if not (out / "COMPLETE.json").exists():
            dump(out / "COMPLETE.json", dict(completed=False, passed=False, mode=args.mode, error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
