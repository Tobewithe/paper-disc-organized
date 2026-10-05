"""Bounded six-arm, three-epoch screen; authorized Linux GPU execution only.

Original candidate identities and the full official loss remain fixed. All
selector preparation counts against one shared two-hour training budget.
No validation/model selection during training and no automatic resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import sys

if "--config" in sys.argv:
    _cfg=json.loads(Path(sys.argv[sys.argv.index("--config")+1]).read_text(encoding="utf-8-sig"))
    if _cfg.get("source_python"):
        sys.path.insert(0,_cfg["source_python"])

import torch

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from screen_models import build_model, build_operators
from runtime_utils import bn_state, loss_and_grad, lr_factor, setup, verify_bn

FIXED = dict(seed=0, epochs=3, microbatch_images=2, effective_batch_images=16,
             branch_lr=1e-4, new_lr=3e-4, weight_decay=1e-4,
             warmup_epochs=1, eta_ratio=0.1, gradient_clip_norm=10,
             roi_side=16, sampled_cells=64, roi_sampling_ratio=2,
             residual_bound=4.0, projection_lambda=0.1, solver_jitter=1e-6,
             uncertainty_floor=0.1, max_training_hours=2)
RECORDED_EPOCHS = (3,)
OPERATOR_FP64_TOL = 1e-8
OPERATOR_FP32_TOL = 1e-4
OPERATOR_RAM_LIMIT = 1024**3


def resolve_config(config):
    cfg = resolve_runtime_config(config)
    for key, expected in FIXED.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Protocol fixes {key}={expected}, got {cfg.get(key)}")
    if cfg.get("arms") != ["N", "U", "Q", "P", "L", "B"]:
        raise ValueError("The arm sequence must stay N/U/Q/P/L/B")
    return cfg


def deadline_check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Shared fixed training deadline reached; no automatic retry/resume")


def source_digest(model):
    return {name: tensor_sha(value) for name, value in model.state_dict().items()}


def buffer_digest(model):
    return {name: tensor_sha(value) for name, value in model.named_buffers()}


def optimizer_for(model, cfg):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups.setdefault((name.startswith("native_cv4."), parameter.ndim > 1), []).append(parameter)
    return torch.optim.AdamW([
        dict(params=parameters,
             lr=cfg["branch_lr"] if native else cfg["new_lr"],
             initial_lr=cfg["branch_lr"] if native else cfg["new_lr"],
             weight_decay=cfg["weight_decay"] if decay else 0.,
             group_name=("native" if native else "evidence") + ("_decay" if decay else "_no_decay"))
        for (native, decay), parameters in groups.items()], betas=(0.9, 0.999), eps=1e-8)


def official_gpu_payload(image, device):
    # Exactly the old fullmask loss fields; no selected-point labels or weights.
    return {key: (image[key].to(device) if torch.is_tensor(image[key]) else image[key])
            for key in ("proto", "masks", "owners", "target_boxes", "segmentation_gain")}


def permanent_keys(image):
    keys = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
    return [{key: row.get(key, image["split"] if key == "split" else None) for key in keys}
            for row in image["rows"]]


def check_receipts(cfg, smoke_run=None):
    root = Path(cfg["server_root"])
    runs = load_json(root / "RUN_IDS.json")["runs"]
    prepare_path = root / "runs" / runs["prepare"] / "COMPLETE.json"
    prepare = load_json(prepare_path)
    if prepare.get("completed") is not True and prepare.get("complete") is not True:
        raise RuntimeError("New frozen online-source preparation is incomplete")
    smoke_path = Path(smoke_run) if smoke_run else root / "runs" / runs["smoke"]
    smoke = load_json(smoke_path / "COMPLETE.json")
    if smoke.get("passed") is not True:
        raise RuntimeError("Screen gradient/frozen-state smoke did not pass")
    return dict(prepare_receipt=str(prepare_path), prepare_sha256=sha256(prepare_path),
                smoke_receipt=str(smoke_path / "COMPLETE.json"),
                smoke_sha256=sha256(smoke_path / "COMPLETE.json"))


def prepare_operators(cfg, out, items, mode, deadline, device):
    """Frozen original-c0 selection, once; no live-model or GT input to selector."""
    cache = {}
    ram_bytes = total = 0
    started = time.monotonic()
    partial = out / "SELECTION_MANIFEST.partial.jsonl"
    if partial.exists() or (out / "SELECTION_MANIFEST.jsonl").exists():
        raise RuntimeError("Existing operator artifacts; preserve them and use a new Run")
    with partial.open("x", encoding="utf-8") as stream:
        for position, item in enumerate(items):
            deadline_check(deadline)
            image = load_asset(cfg, int(item["image_id"]), verify=True)
            if image["split"] != "fit" or len(image["raw_ids"]) != int(item["n"]):
                raise AssertionError("Frozen screen cohort changed")
            built = build_operators(image, mode, cfg, device)
            diagnostic = built.pop("_operator_diagnostics", {})
            cached = {}
            for key, value in built.items():
                if not torch.is_tensor(value):
                    raise TypeError(f"Unexpected non-tensor selection field: {key}")
                if value.requires_grad:
                    raise AssertionError("Frozen selector output requires gradients")
                cached[key] = value.detach().cpu().contiguous()
            if not torch.equal(cached["raw_ids"].long(), image["raw_ids"].long()):
                raise AssertionError("Selector changed raw candidate identity")
            cache[int(item["image_id"])] = cached
            ram_bytes += sum(v.numel()*v.element_size() for v in cached.values())
            if ram_bytes > OPERATOR_RAM_LIMIT:
                raise MemoryError("Frozen operator RAM exceeds 1GiB; do not create a large F cache")
            total += len(image["raw_ids"])
            record = dict(image_id=int(item["image_id"]), candidates=len(image["raw_ids"]), mode=mode,
                candidate_keys=permanent_keys(image), tensors={k: tensor_sha(v) for k,v in cached.items()},
                indices=cached["indices"].tolist() if "indices" in cached else None,
                valid=cached["valid"].tolist() if "valid" in cached else None,
                diagnostics=diagnostic)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            if position % 50 == 0 or position+1 == len(items):
                stream.flush()
                state = dict(stage="precompute_frozen_operators",mode=mode,images=position+1,
                    total_images=len(items),candidates=total,operator_ram_bytes=ram_bytes,
                    elapsed_s=time.monotonic()-started,remaining_budget_s=deadline-time.time())
                dump(out/"PROGRESS.json",state);print(json.dumps(state),flush=True)
    manifest = out/"SELECTION_MANIFEST.jsonl";partial.replace(manifest)
    summary = dict(completed=True,mode=mode,images=len(cache),candidates=total,
        frozen_from_original_c0=True,parameter_dependent_reselection=False,
        operator_ram_bytes=ram_bytes,operator_disk_cache_created=False,
        manifest=manifest.name,manifest_sha256=sha256(manifest),elapsed_s=time.monotonic()-started)
    dump(out/"OPERATOR_SUMMARY.json",summary)
    return cache,summary


def selections_for(images, cache, device):
    result=[]
    for image in images:
        cached=cache[int(image["image_id"])]
        if not torch.equal(cached["raw_ids"].long(),image["raw_ids"].long()):
            raise AssertionError("Frozen raw identity changed")
        result.append({k:v.to(device) for k,v in cached.items()})
    return result


def save_checkpoint(path, model, optimizer, epoch, mode, cfg, history, feature_channels, identity):
    temporary = Path(str(path) + ".tmp")
    torch.save(dict(mode=mode, epoch=epoch, state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(), config=cfg, history=history,
                    feature_channels=feature_channels, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), operator_identity=identity,
                    checkpoint_rule="fixed final epoch3; no dev/val model selection"), temporary)
    temporary.replace(path)


def train(cfg, out, mode, deadline, smoke_run=None):
    if (out / "last.pt").exists() or (out / "HISTORY.json").exists():
        raise RuntimeError("Existing training artifacts; automatic resume/overwrite is forbidden")
    deadline_check(deadline)
    receipts = check_receipts(cfg, smoke_run)
    index = load_index(cfg)
    planned = index["fit"]
    items = [row for row in planned if row["n"] > 0]
    if len(planned) != 1024 or len(index.get("dev", [])) != 256 or index.get("val"):
        raise AssertionError("Official fit counts differ from the frozen cohort; inspect rather than resample")
    if len({int(row["image_id"]) for row in planned}) != len(planned):
        raise AssertionError("Duplicate fit image IDs")
    setup(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    replay = FrozenReplay(cfg, device="cuda")
    source_initial = source_digest(replay.source)
    dump(out / "ENVIRONMENT.json", replay.import_info)
    dump(out / "RESOLVED_CONFIG.json", cfg)
    dump(out / "COHORT.json", dict(planned_images=len(planned), effective_images=len(items),
        candidates=sum(row["n"] for row in items),
        no_positive_images=[int(row["image_id"]) for row in planned if not row["n"]],
        domain="official frozen one-to-one TAL fit; no geometric binding", receipts=receipts,
        filtering="no quality-based filtering; original invalid operators retain native coefficient and loss"))
    cache, operator_summary = prepare_operators(cfg, out, items, mode, deadline, replay.device)
    replay.assert_unchanged()
    if source_digest(replay.source) != source_initial:
        raise AssertionError("Original source state changed during operator preparation")
    # Pair all initialization, independent of selector work or operator order.
    setup(cfg["seed"])
    model = build_model(mode, replay, cfg).to(replay.device).float().train()
    optimizer = optimizer_for(model, cfg)
    frozen_bn, frozen_buffers = bn_state(model), buffer_digest(model)
    initial_state_sha = hashlib.sha256(json.dumps(source_digest(model), sort_keys=True).encode()).hexdigest()
    identity = dict(manifest_sha256=operator_summary["manifest_sha256"], mode=mode,
                    original_weights_sha256=replay.weights_sha256,
                    prepare_receipt_sha256=receipts["prepare_sha256"],
                    selection_frozen_from_original_c0=True)
    dump(out / "MODEL.json", dict(mode=mode, counts=model.parameter_counts(),
        trainable=model.trainable_parameter_names(), feature_channels=replay.feature_channels,
        initialization_state_sha256=initial_state_sha, source_initial=source_initial,
        bn_running_buffers="fixed original state; BN affine trainable",
        objective="original full640 GT-box BCE / original normalized area, original gain, candidate batch mean",
        loss_selection_weighting="none; selected uncertainty never weights loss",
        evidence="native-only N; same point architecture in U/Q/P/L/B; signed r=4tanh(raw/4)",
        operator_identity=identity, training_deadline_unix=deadline))
    history = []
    started = time.monotonic()
    run_wall_start = time.time()
    completed_epoch, current_epoch, current_step = 0, 0, 0
    try:
        for epoch in range(cfg["epochs"]):
            deadline_check(deadline)
            current_epoch, current_step = epoch + 1, 0
            order = list(items)
            random.Random(cfg["seed"] + epoch).shuffle(order)
            order_sha = hashlib.sha256(json.dumps([int(row["image_id"]) for row in order], separators=(",", ":")).encode()).hexdigest()
            epoch_started, total_loss, seen = time.monotonic(), 0.0, 0
            groups = math.ceil(len(order) / cfg["effective_batch_images"])
            spans = []
            replay_cuda_s = train_cuda_s = 0.0
            for step, offset in enumerate(range(0, len(order), cfg["effective_batch_images"])):
                current_step = step + 1
                deadline_check(deadline)
                group = order[offset:offset + cfg["effective_batch_images"]]
                denominator = sum(row["n"] for row in group)
                factor = lr_factor(epoch + (step + 1) / groups, cfg)
                for optim_group in optimizer.param_groups:
                    optim_group["lr"] = optim_group["initial_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                for inner in range(0, len(group), cfg["microbatch_images"]):
                    deadline_check(deadline)
                    subset = group[inner:inner + cfg["microbatch_images"]]
                    batch = [load_asset(cfg, row["image_id"], verify=True) for row in subset]
                    if any(len(image["raw_ids"]) != row["n"] for image, row in zip(batch, subset)):
                        raise AssertionError("Frozen candidate count changed after operator preparation")
                    selected = selections_for(batch, cache, replay.device)
                    begin, prefix_done, done = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    begin.record()
                    features = replay.replay(batch)
                    prefix_done.record()
                    coefficients = model(features, selected)
                    gradients = []
                    for image, coefficient in zip(batch, coefficients):
                        value, gradient = loss_and_grad(official_gpu_payload(image, replay.device), coefficient, denominator)
                        if not bool(torch.isfinite(gradient).all()):
                            raise FloatingPointError("Nonfinite original mask-loss coefficient gradient")
                        total_loss += value
                        seen += len(image["raw_ids"])
                        gradients.append(gradient)
                    torch.autograd.backward(coefficients, gradients)
                    done.record()
                    spans.append((begin, prefix_done, done))
                    del batch, features, coefficients, gradients, selected
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
                optimizer.step()
                if step % 25 == 0 or step + 1 == groups:
                    torch.cuda.synchronize()
                    replay_cuda_s += sum(begin.elapsed_time(prefix) for begin, prefix, end in spans) / 1000
                    train_cuda_s += sum(prefix.elapsed_time(end) for begin, prefix, end in spans) / 1000
                    spans.clear()
                    elapsed = time.monotonic() - started
                    finished_fraction = epoch + (step + 1) / groups
                    estimated_remaining = elapsed * (cfg["epochs"] - finished_fraction) / max(finished_fraction, 1e-9)
                    progress = dict(stage="training", mode=mode, epoch=epoch + 1, epochs=cfg["epochs"],
                        step=step + 1, steps=groups, candidates_seen_this_epoch=seen,
                        trajectory_bce=total_loss / max(seen, 1), gradient_norm_before_clip=gradient_norm,
                        native_lr=cfg["branch_lr"] * factor, evidence_lr=cfg["new_lr"] * factor,
                        elapsed_s=elapsed, epoch_elapsed_s=time.monotonic() - epoch_started,
                        remaining_budget_s=deadline - time.time(), estimated_arm_remaining_s=estimated_remaining,
                        epoch_cuda_prefix_span_s=replay_cuda_s, epoch_cuda_train_and_loss_span_s=train_cuda_s,
                        cuda_timing_note="event spans include scheduling gaps, not a utilization percentage",
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        operator_ram_bytes=operator_summary["operator_ram_bytes"])
                    dump(out / "PROGRESS.json", progress)
                    print(json.dumps(progress), flush=True)
            if seen != sum(int(row["n"]) for row in items):
                raise AssertionError("Epoch did not use exactly the frozen fit candidates")
            verify_bn(model, frozen_bn)
            if buffer_digest(model) != frozen_buffers:
                raise AssertionError("Trainable-model frozen buffers changed")
            replay.assert_unchanged()
            if source_digest(replay.source) != source_initial:
                raise AssertionError("Original replay model parameters/buffers changed")
            if any(parameter.grad is not None for parameter in replay.source.parameters()):
                raise AssertionError("Gradient reached immutable original replay model")
            history.append(dict(epoch=epoch + 1, trajectory_bce=total_loss / seen,
                trajectory_not_fixed_checkpoint_objective=True, candidates=seen, images=len(items),
                seconds=time.monotonic() - epoch_started, image_order_sha256=order_sha,
                cuda_prefix_span_s=replay_cuda_s, cuda_train_and_loss_span_s=train_cuda_s,
                bn_unchanged=True, all_buffers_unchanged=True, replay_source_unchanged=True))
            completed_epoch = epoch + 1
            save_checkpoint(out / "last.pt", model, optimizer, completed_epoch, mode, cfg, history,
                            replay.feature_channels, identity)
            if completed_epoch in RECORDED_EPOCHS:
                save_checkpoint(out / f"epoch{completed_epoch:02d}.pt", model, optimizer,
                                completed_epoch, mode, cfg, history, replay.feature_channels, identity)
            dump(out / "HISTORY.json", history)
        save_checkpoint(out / "final.pt", model, optimizer, cfg["epochs"], mode, cfg, history,
                        replay.feature_channels, identity)
        dump(out / "COMPLETE.json", dict(completed=True, kind="training", mode=mode,
            epochs=cfg["epochs"], checkpoint="final.pt", checkpoint_rule="fixed epoch3",
            selection_manifest_sha256=operator_summary["manifest_sha256"],
            elapsed_training_s=time.monotonic() - started, started_training_unix=run_wall_start,
            ended_training_unix=time.time(), operator_preparation_s=operator_summary["elapsed_s"],
            all_frozen_states_unchanged=True, automatic_retry=False, automatic_resume=False))
    except TimeoutError:
        dump(out / "BUDGET_STOP.json", dict(completed_epoch=completed_epoch, current_epoch=current_epoch,
            current_step=current_step, deadline_unix=deadline, automatic_retry=False, automatic_resume=False,
            scientific_status="incomplete; last complete epoch is not the fixed epoch3 result"))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True, choices=("N", "U", "Q", "P", "L", "B"))
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--smoke-run")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt" or not torch.cuda.is_available():
            raise RuntimeError("Run only on the authorized Linux CUDA server")
        cfg = resolve_config(load_json(args.config))
        if args.deadline > time.time() + cfg["max_training_hours"] * 3600 + 60:
            raise ValueError("Supplied shared deadline exceeds the shared fixed 2-hour budget")
        train(cfg, out, args.mode, args.deadline, args.smoke_run)
    except BaseException as error:
        if isinstance(error, TimeoutError) and not (out / "BUDGET_STOP.json").exists():
            dump(out / "BUDGET_STOP.json", dict(completed_epoch=0, mode=args.mode, scientific_status="incomplete; budget exhausted before a fixed epoch3 result", automatic_retry=False))
        dump(out / "FAILURE.json", dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), mode=args.mode, automatic_retry=False, automatic_resume=False))
        raise


if __name__ == "__main__":
    main()
