"""Full original-COCO online TriFlow training with durable exact-cursor resume.

Only the new module is optimized. Frozen native YOLO features and original GT
are regenerated one image at a time with the unchanged pilot method. Resume
starts at the snapshot's next unapplied chunk. Work after the last durable
snapshot is rolled back and may be recomputed; its former Run stays immutable.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from frozen_io import OFFICIAL_SHA256, dump_json, sha256, state_digest
from stream_data import OnlineCOCOProvider
from triflow_model import TriFlowConfig, TriFlowModel

TRAINER_VERSION = "triflow_fullscale_online_v1"
CHECKPOINT_KIND = "triflow_fullscale_training_snapshot"
FINAL_KIND = "triflow_fullscale_module_final"
SOURCE_FILES = ("train_fullscale.py", "stream_data.py", "frozen_io.py", "triflow_model.py")
GROUPS = ("token_projection", "ownership_embeddings", "ownership_projection", "cross_attention", "interaction_norm", "field_head")


class IntentionalEngineeringStop(Exception):
    """Explicit engineering interruption; never a completed training result."""


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def group_name(name):
    group = name.split(".")[0]
    if group not in GROUPS:
        raise ValueError(f"Unclassified learned parameter: {name}")
    return group


def to_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return copy.deepcopy(value)


def serial(value):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value


def tensors_finite(value):
    import torch
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensors_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_finite(item) for item in value)
    return True


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_state(shuffle_rng):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "shuffle": shuffle_rng.getstate()}


def restore_rng(value, shuffle_rng):
    import numpy as np
    import torch
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    cuda = value["torch_cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Resume CUDA RNG device count changed")
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
    shuffle_rng.setstate(value["shuffle"])


def prepare_image(payload, device):
    """Same tensors as pilot, with one P/F/c0/h/box device transfer per image."""
    if payload.get("hidden_query_source") != "one2one_cv4 final convolution input, exact native raw index":
        raise ValueError("Online input is missing the genuine hidden-query provenance")
    return {key: payload[key].to(device) for key in ("P", "F", "c0", "h", "boxes_proto")}


def collate_instances(prepared, chunk):
    import torch
    device = prepared["P"].device
    rows = torch.tensor([t["output_row"] for t in chunk], device=device, dtype=torch.long)
    neighbors = torch.stack([t["neighbor_rows"] for t in chunk]).to(device)
    valid = torch.stack([t["neighbor_valid"] for t in chunk]).to(device)
    neighbor_c = prepared["c0"][neighbors.clamp_min(0)]
    neighbor_logits = torch.einsum("chw,nkc->nkhw", prepared["P"], neighbor_c)
    neighbor_logits = neighbor_logits.masked_fill(~valid[:, :, None, None], 0)
    inputs = {"prototypes": prepared["P"], "coefficients": prepared["c0"][rows], "visual": prepared["F"],
              "boxes": prepared["boxes_proto"][rows], "neighbor_logits": neighbor_logits,
              "neighbor_valid": valid, "instance_hidden": prepared["h"][rows]}
    targets = {key: torch.stack([t["ownership_targets"][key] for t in chunk]).to(device)
               for key in chunk[0]["ownership_targets"]}
    if any(value.requires_grad for value in inputs.values()):
        raise ValueError("Frozen inputs unexpectedly require gradients")
    return inputs, targets


def task_probe(loss, parameters, outputs, epoch, iid, chunk):
    import torch
    gradients = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    groups = {name: 0. for name in GROUPS}
    phi_square = 0.
    for (name, _), gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Task-only compiler backward contains NaN/Inf")
        square = float(gradient.double().square().sum().cpu())
        groups[group_name(name)] += square
        if name in ("field_head.2.weight", "field_head.2.bias"):
            phi_square += float(gradient[:2].double().square().sum().cpu())
    groups = {key: value**.5 for key, value in groups.items()}
    phi = phi_square**.5
    attention = groups["cross_attention"]
    return {"epoch": epoch, "image_id": iid, "output_rows": [t["output_row"] for t in chunk],
            "all_task_gradients_finite": True, "group_task_gradient_norms": groups,
            "nonzero_task_gradient_reaches_learned_head": any(value > 0 for value in groups.values()),
            "phi_task_gradient_norm": phi, "cross_attention_task_gradient_norm": attention,
            "task_gradient_phi_attention_same_probe_nonzero": phi > 0 and attention > 0,
            "task_loss": float(loss.detach().cpu()), "diagnostics": serial(outputs["diagnostics"])}


def fresh_epoch_accumulator():
    return {"images_completed": 0, "images_without_usable_instances": 0, "instances": 0,
            "applied_steps": 0, "loss_sums": {}, "supervision_counts": {}, "compiler_counts": {},
            "frozen_input_checks": 0, "seconds": 0.}


def write_task_evidence(run, evidence):
    dump_json(run/"TASK_GRADIENT_EVIDENCE.json", {
        "evidence": evidence, "any_observed_nonzero": any(r["nonzero_task_gradient_reaches_learned_head"] for r in evidence),
        "any_observed_phi_attention_same_probe_nonzero": any(r["task_gradient_phi_attention_same_probe_nonzero"] for r in evidence),
        "no_hidden_warmup": True})


def atomic_torch_save(path, payload):
    """Flush before atomic replace; unfinished .tmp never becomes a snapshot."""
    import torch
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "weights", "vendor", "images-list", "annotations"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instance-chunk", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-images", type=int, help="Engineering limit only; always marks smoke_only=true")
    parser.add_argument("--checkpoint-seconds", type=float, default=300.)
    parser.add_argument("--trace-every", type=int, default=250)
    parser.add_argument("--engineering-stop-after-steps", type=int,
                        help="Engineering-only deliberate stop after N applied steps of this execution; exits2")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed != 0 or args.instance_chunk < 1 or args.checkpoint_seconds <= 0 or args.trace_every < 1:
        raise ValueError("Positive explicit budgets/cadence required; this protocol requires seed0")
    if args.instance_chunk != 4:
        raise ValueError("Unchanged optimizer cadence requires instance_chunk4")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("Engineering max-images must be positive")
    if args.engineering_stop_after_steps is not None and (args.max_images is None or args.engineering_stop_after_steps < 1):
        raise ValueError("Intentional step interruption is restricted to --max-images engineering mode")
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Run ID must be a plain directory name")
    root = Path(args.root).resolve()
    run = root/"runs"/args.run_id
    if any((run/name).exists() for name in ("TRAINING_INPUTS.json", "checkpoint_latest.pt", "head_final.pt", "TRAIN_TRACE.jsonl")):
        raise FileExistsError("Execution history exists; resume must use a distinct new Run ID")
    run.mkdir(parents=True, exist_ok=True)
    source = run/"source"
    source.mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        current = Path(__file__).with_name(name)
        target = source/name
        if target.exists() and sha256(target) != sha256(current):
            raise ValueError("Run source snapshot differs from executing source")
        if not target.exists():
            shutil.copy2(current, target)
    protocol = root/"PROTOCOL.md"
    if protocol.exists() and not (source/"PROTOCOL.md").exists():
        shutil.copy2(protocol, source/"PROTOCOL.md")
    started, began = now(), time.monotonic()
    provider = head = optimizer = state = contract = initial = shuffle_rng = None
    last_checkpoint_time = time.monotonic()
    checkpoint_sequence = 0
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    resume_payload = None

    def save_snapshot(reason):
        nonlocal last_checkpoint_time, checkpoint_sequence
        if not tensors_finite(head.state_dict()) or not tensors_finite(optimizer.state_dict()):
            raise FloatingPointError("Refuse resumable checkpoint with nonfinite module/optimizer state")
        freeze = provider.verify_frozen()
        if freeze.get("passed") is not True:
            raise RuntimeError("Frozen online YOLO integrity failed at checkpoint")
        snapshot = {"kind": CHECKPOINT_KIND, "trainer_version": TRAINER_VERSION, "run_id": args.run_id,
                    "created_at": now(), "reason": reason, "contract": contract, "contract_sha256": canonical_sha(contract),
                    "head_state_dict": to_cpu(head.state_dict()), "optimizer_state_dict": to_cpu(optimizer.state_dict()),
                    "initial_head_state_dict": initial, "state": copy.deepcopy(state), "rng": to_cpu(rng_state(shuffle_rng)),
                    "provider_state": provider.state_dict(), "frozen_integrity": freeze,
                    "head_state_sha256": state_digest(head), "resumable": True,
                    "resume_semantics": "next unapplied chunk in durable snapshot; later former-Run updates are rolled back, not committed twice"}
        latest = run/"checkpoint_latest.pt"
        previous = run/"checkpoint_previous.pt"
        if latest.exists():
            temp_previous = previous.with_name(previous.name+".tmp")
            shutil.copy2(latest, temp_previous)
            temp_previous.replace(previous)
        atomic_torch_save(latest, snapshot)
        checkpoint_sequence += 1
        record = {"sequence_this_run": checkpoint_sequence, "created_at": snapshot["created_at"], "reason": reason,
                  "path": str(latest), "sha256": sha256(latest), "bytes": latest.stat().st_size,
                  "optimizer_applied": state["applied"], "next_cursor": state["cursor"],
                  "head_state_sha256": snapshot["head_state_sha256"], "frozen_integrity": freeze,
                  "checkpoint_contract_sha256": snapshot["contract_sha256"],
                  "history_storage": "latest and previous payload retained; older payloads superseded, compact hashes retained"}
        with (run/"CHECKPOINTS.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")
        dump_json(run/"CHECKPOINT_LATEST.json", record)
        last_checkpoint_time = time.monotonic()
        print(f"CHECKPOINT applied={state['applied']} cursor={state['cursor']} reason={reason}", flush=True)
        return record

    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        provider = OnlineCOCOProvider(args.images_list, args.annotations, args.weights, args.vendor, args.device,
                                      run_dir=run, require_full_split=args.max_images is None)
        total_images = len(provider.ids) if args.max_images is None else min(args.max_images, len(provider.ids))
        active_indices = list(range(total_images))
        active_ids = [provider.ids[index] for index in active_indices]
        source_hashes = {name: sha256(source/name) for name in SOURCE_FILES}
        runtime = {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                   "python": sys.version, "device": args.device,
                   "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
        if resume_path:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if not isinstance(resume_payload, dict) or resume_payload.get("kind") != CHECKPOINT_KIND or resume_payload.get("resumable") is not True:
                raise ValueError("Resume requires a durable valid fullscale training snapshot")
            if resume_payload.get("run_id") == args.run_id:
                raise ValueError("Resume must belong to a distinct retry Run ID")
            provider.load_state_dict(resume_payload["provider_state"])
            dimensions = provider.dimensions
            if not dimensions:
                raise ValueError("Resume snapshot lacks actually observed frozen input dimensions")
        else:
            # A real dimension probe is extraction evidence, not an update.
            provider.get(0)
            dimensions = provider.dimensions
        # Re-seed AFTER frozen YOLO construction/probe, preserving pilot's new
        # module seed initialization irrespective of YOLO loader RNG use.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        head = TriFlowModel(dimensions["feature_channels"], dimensions["instance_hidden_channels"], TriFlowConfig()).float().to(args.device)
        head.train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
        config = head.configuration()
        contract = {"trainer_version": TRAINER_VERSION, "sources": source_hashes, "data_hashes": provider.data_hashes,
                    "base_weights_sha256": OFFICIAL_SHA256, "configuration": config, "configuration_sha256": canonical_sha(config),
                    "epochs": args.epochs, "seed": args.seed, "instance_chunk": args.instance_chunk,
                    "max_images": args.max_images, "effective_images": total_images,
                    "effective_image_identity_sha256": canonical_sha(active_ids), "smoke_only": args.max_images is not None,
                    "complete_original_train_split": provider.full_split and args.max_images is None,
                    "precision": "FP32, TF32 disabled, no AMP", "augmentation": False,
                    "optimizer": {"class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip_grad_norm": 10},
                    "runtime": runtime, "protocol_sha256": sha256(source/"PROTOCOL.md") if (source/"PROTOCOL.md").exists() else None,
                    "checkpoint_seconds": args.checkpoint_seconds, "trace_every": args.trace_every}
        shuffle_rng = random.Random(args.seed)
        parameters = list(head.named_parameters())
        if resume_payload:
            if resume_payload.get("trainer_version") != TRAINER_VERSION or resume_payload.get("contract") != contract:
                raise ValueError("Resume source/data/config/runtime/budget contract changed")
            if resume_payload.get("contract_sha256") != canonical_sha(contract):
                raise ValueError("Resume contract SHA differs from checkpoint bytes")
            saved_state = resume_payload["head_state_dict"]
            expected = head.state_dict()
            if set(saved_state) != set(expected) or any(saved_state[k].shape != expected[k].shape or saved_state[k].dtype != expected[k].dtype for k in expected):
                raise ValueError("Resume module state keys/shapes/dtypes differ")
            if not tensors_finite(saved_state) or not tensors_finite(resume_payload["optimizer_state_dict"]):
                raise ValueError("Resume module/optimizer tensors contain NaN/Inf")
            head.load_state_dict(saved_state, strict=True)
            if state_digest(head) != resume_payload["head_state_sha256"]:
                raise ValueError("Loaded resume head differs from saved state digest")
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            initial = resume_payload["initial_head_state_dict"]
            state = copy.deepcopy(resume_payload["state"])
            if sorted(state["order"]) != active_indices:
                raise ValueError("Resume shuffle is not the fixed input permutation")
            cursor = state["cursor"]
            if not (1 <= cursor["epoch"] <= args.epochs+1 and 0 <= cursor["image_position"] <= total_images
                    and cursor["chunk_start"] >= 0 and cursor["chunk_start"] % args.instance_chunk == 0):
                raise ValueError("Resume next-chunk cursor is invalid")
            restore_rng(resume_payload["rng"], shuffle_rng)
            parent = {"run_id": resume_payload["run_id"], "snapshot_path": str(resume_path), "snapshot_sha256": sha256(resume_path),
                      "durable_optimizer_applied": state["applied"], "next_cursor": cursor,
                      "rollback": "former Run work after this snapshot, if any, is discarded and may be replayed; original files retained"}
            dump_json(run/"RESUME_RECEIPT.json", parent)
        else:
            initial = to_cpu(head.state_dict())
            order = list(active_indices)
            shuffle_rng.shuffle(order)
            state = {"attempts": 0, "applied": 0, "instances": 0, "order": order,
                     "cursor": {"epoch": 1, "image_position": 0, "chunk_start": 0},
                     "epoch_accumulator": fresh_epoch_accumulator(), "epochs": [], "task_evidence": [],
                     "task_probed_epoch": None, "finite_groups": {group: True for group in GROUPS},
                     "gradient_steps": {group: 0 for group in GROUPS}, "frozen_input_checks": 0,
                     "initial_head_state_sha256": state_digest(head), "effective_training_seconds": 0.}
            parent = None
        inputs = {"trainer_version": TRAINER_VERSION, "started_at": started, "contract": contract,
                  "contract_sha256": canonical_sha(contract), "sources": source_hashes,
                  "configuration": config, "configuration_sha256": canonical_sha(config),
                  "base_weights_sha256": OFFICIAL_SHA256, "data_hashes": provider.data_hashes,
                  "epochs": args.epochs, "seed": args.seed, "instance_chunk": args.instance_chunk,
                  "images": total_images, "smoke_only": contract["smoke_only"], "parent": parent,
                  "frozen_yolo": "online actual extraction, all parameters and BN frozen; only new module optimized",
                  "selection": "fixed last epoch; no development or best selection",
                  "resume_semantics": "restore exact durable next chunk/order/all RNG; uncommitted former updates rolled back",
                  "execution_directory": str(run), "interpreter": sys.executable}
        inputs["engineering_stop_after_steps_this_execution"] = args.engineering_stop_after_steps
        starting_applied_this_run = state["applied"]
        dump_json(run/"TRAINING_INPUTS.json", inputs)
        write_task_evidence(run, state["task_evidence"])
        dump_json(run/"EPOCHS.json", {"epochs": state["epochs"]})
        save_snapshot("initial_or_resumed_state")
        while state["cursor"]["epoch"] <= args.epochs:
            cursor = state["cursor"]
            epoch = cursor["epoch"]
            image_position = cursor["image_position"]
            if image_position == total_images:
                accumulator = state["epoch_accumulator"]
                if accumulator["images_completed"] != total_images or accumulator["instances"] == 0:
                    raise RuntimeError("Epoch did not cover every declared image or has no usable training instances")
                result = {"epoch": epoch, "images": accumulator["images_completed"],
                          "images_without_usable_instances": accumulator["images_without_usable_instances"],
                          "instances": accumulator["instances"], "applied_steps": accumulator["applied_steps"],
                          "mean_losses_per_instance": {key: value/accumulator["instances"] for key, value in accumulator["loss_sums"].items()},
                          "supervision_counts": accumulator["supervision_counts"], "compiler_counts": accumulator["compiler_counts"],
                          "frozen_input_checks": accumulator["frozen_input_checks"], "seconds": accumulator["seconds"],
                          "shuffle_order_sha256": canonical_sha(state["order"])}
                state["epochs"].append(result)
                dump_json(run/"EPOCHS.json", {"epochs": state["epochs"]})
                print(f"EPOCH_COMPLETE {epoch}/{args.epochs} images={total_images} instances={result['instances']} steps={result['applied_steps']} mean_loss={result['mean_losses_per_instance']['loss']:.6f}", flush=True)
                state["cursor"] = {"epoch": epoch+1, "image_position": 0, "chunk_start": 0}
                state["epoch_accumulator"] = fresh_epoch_accumulator()
                if epoch < args.epochs:
                    shuffle_rng.shuffle(state["order"])
                save_snapshot("epoch_boundary")
                continue
            image_began = time.monotonic()
            index = state["order"][image_position]
            payload = provider.get(index)
            if payload["image_id"] != active_ids[index] or payload.get("base_weights_sha256") != OFFICIAL_SHA256:
                raise ValueError("Online image/checkpoint identity differs from fixed input")
            selected = payload["training_targets"]
            start_at = cursor["chunk_start"]
            if start_at > len(selected) or start_at and start_at == len(selected):
                raise ValueError("Resume cursor points past an already completed image")
            accumulator = state["epoch_accumulator"]
            if not selected:
                accumulator["images_without_usable_instances"] += 1
                accumulator["images_completed"] += 1
                state["cursor"] = {"epoch": epoch, "image_position": image_position+1, "chunk_start": 0}
            else:
                prepared = prepare_image(payload, args.device)
                for start in range(start_at, len(selected), args.instance_chunk):
                    chunk = selected[start:start+args.instance_chunk]
                    model_inputs, targets = collate_instances(prepared, chunk)
                    probe = state["task_probed_epoch"] != epoch
                    before = {key: tensor_sha(value) for key, value in model_inputs.items()} if probe else None
                    outputs = head(**model_inputs)
                    losses = head.loss(outputs, targets["self_masks"], targets["neighbor_masks"], model_inputs["neighbor_valid"], targets=targets)
                    if not tensors_finite(losses) or not tensors_finite(outputs["delta_coefficients"]) or not tensors_finite(outputs["logits_refined"]):
                        raise FloatingPointError("Online losses/compiler outputs contain NaN/Inf")
                    if probe:
                        evidence = task_probe(losses["loss_task"], parameters, outputs, epoch, payload["image_id"], chunk)
                        state["task_evidence"].append(evidence)
                        state["task_probed_epoch"] = epoch
                        write_task_evidence(run, state["task_evidence"])
                    optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    state["attempts"] += 1
                    observed = {group: False for group in GROUPS}
                    for name, parameter in parameters:
                        if parameter.grad is None:
                            continue
                        group = group_name(name)
                        if not torch.isfinite(parameter.grad).all():
                            state["finite_groups"][group] = False
                            raise FloatingPointError(f"Nonfinite gradient for {name}; update not applied")
                        observed[group] |= bool((parameter.grad != 0).any())
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True)
                    optimizer.step()
                    if not tensors_finite(head.state_dict()):
                        raise FloatingPointError("Module became nonfinite after update; current state not resumable")
                    state["applied"] += 1
                    state["instances"] += len(chunk)
                    accumulator["applied_steps"] += 1
                    accumulator["instances"] += len(chunk)
                    for group, seen in observed.items():
                        state["gradient_steps"][group] += int(seen)
                    next_start = start+len(chunk)
                    state["cursor"] = ({"epoch": epoch, "image_position": image_position+1, "chunk_start": 0}
                                       if next_start == len(selected) else {"epoch": epoch, "image_position": image_position, "chunk_start": next_start})
                    if next_start == len(selected):
                        accumulator["images_completed"] += 1
                    if probe:
                        after = {key: tensor_sha(value) for key, value in model_inputs.items()}
                        if before != after:
                            raise RuntimeError("Frozen feature tensors changed during module training")
                        state["frozen_input_checks"] += 1
                        accumulator["frozen_input_checks"] += 1
                    scalar_losses = serial(losses)
                    for key, value in scalar_losses.items():
                        target = accumulator["loss_sums"] if key.startswith("loss") else accumulator["supervision_counts"]
                        target[key] = target.get(key, 0.) + value*(len(chunk) if key.startswith("loss") else 1)
                    for key in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count"):
                        accumulator["compiler_counts"][key] = accumulator["compiler_counts"].get(key, 0)+int(outputs["diagnostics"][key].sum().detach().cpu())
                    if state["applied"] == 1 or state["applied"] % args.trace_every == 0 or probe:
                        row = {"epoch": epoch, "image_position": image_position, "image_id": payload["image_id"],
                               "optimizer_applied": state["applied"], "optimizer_attempts": state["attempts"],
                               "output_rows": [t["output_row"] for t in chunk], "annotation_ids": [t["annotation_id"] for t in chunk],
                               "instances": len(chunk), "losses": scalar_losses, "grad_norm_before_clip": float(norm.detach().cpu()),
                               "all_applied_gradients_finite": True, "all_parameters_finite": True,
                               "next_cursor": state["cursor"], "sampled_trace_every_steps": args.trace_every,
                               "compiler_counts": {key: int(outputs["diagnostics"][key].sum().detach().cpu()) for key in
                                                   ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count")}}
                        with (run/"TRAIN_TRACE.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                        print(f"TRAIN epoch={epoch}/{args.epochs} image={image_position+1}/{total_images} applied={state['applied']} loss={scalar_losses['loss']:.6f} elapsed_s={time.monotonic()-began:.1f}", flush=True)
                    if state["applied"] == 1 or time.monotonic()-last_checkpoint_time >= args.checkpoint_seconds:
                        save_snapshot("first_applied_or_timed")
                    if (args.engineering_stop_after_steps is not None and
                            state["applied"]-starting_applied_this_run >= args.engineering_stop_after_steps):
                        record = save_snapshot("intentional_engineering_stop")
                        dump_json(run/"INTENTIONAL_ENGINEERING_STOP.json", {
                            "status": "intentional_engineering_stop", "smoke_only": True, "exit_code": 2,
                            "requested_applied_steps_this_execution": args.engineering_stop_after_steps,
                            "actual_applied_steps_this_execution": state["applied"]-starting_applied_this_run,
                            "optimizer_applied": state["applied"], "checkpoint_sha256": record["sha256"],
                            "checkpoint_path": record["path"], "next_cursor": state["cursor"], "stopped_at": now(),
                            "final_head_written": False, "scientific_training_complete": False})
                        raise IntentionalEngineeringStop("Deliberate engineering checkpoint/resume diagnostic")
                del prepared
            seconds = time.monotonic()-image_began
            accumulator["seconds"] += seconds
            state["effective_training_seconds"] += seconds
            if time.monotonic()-last_checkpoint_time >= args.checkpoint_seconds:
                save_snapshot("timed_image_boundary")
            if image_position == 0 or (image_position+1) % 100 == 0:
                dump_json(run/"TRAINING_PROGRESS.json", {"observed_at": now(), "next_cursor": state["cursor"],
                          "optimizer_applied": state["applied"], "optimizer_attempts": state["attempts"],
                          "instances": state["instances"], "epoch_images_completed": accumulator["images_completed"],
                          "images_per_epoch": total_images, "smoke_only": contract["smoke_only"]})
        frozen = provider.verify_frozen()
        stream_receipt = provider.receipt(verify_frozen=True)
        dump_json(run/"STREAM_RECEIPT.json", stream_receipt)
        final = to_cpu(head.state_dict())
        changed = {group: {"changed_tensors": 0, "max_abs_delta": 0.} for group in GROUPS}
        for name, value in final.items():
            delta = (value-initial[name]).abs()
            group = group_name(name)
            if bool((delta != 0).any()):
                changed[group]["changed_tensors"] += 1
                changed[group]["max_abs_delta"] = max(changed[group]["max_abs_delta"], float(delta.max()))
        evidence = state["task_evidence"]
        joint_count = sum(row["task_gradient_phi_attention_same_probe_nonzero"] for row in evidence)
        audit = {"trainer_version": TRAINER_VERSION, "started_at": started, "completed_at": now(),
                 "epochs": args.epochs, "completed_epochs": len(state["epochs"]), "images_per_epoch": total_images,
                 "optimizer_attempts": state["attempts"], "optimizer_applied": state["applied"], "instances": state["instances"],
                 "gradient_finite_by_group": state["finite_groups"], "all_applied_gradients_finite": all(state["finite_groups"].values()),
                 "all_parameters_finite": tensors_finite(final), "all_optimizer_states_finite": tensors_finite(optimizer.state_dict()),
                 "nonzero_gradient_steps_by_group": state["gradient_steps"], "parameter_updates_by_group": changed,
                 "all_groups_really_changed": all(item["changed_tensors"] > 0 for item in changed.values()),
                 "initial_head_state_sha256": state["initial_head_state_sha256"], "final_head_state_sha256": state_digest(head),
                 "compiler_task_gradient_observed_nonzero": any(row["nonzero_task_gradient_reaches_learned_head"] for row in evidence),
                 "compiler_task_phi_and_attention_same_probe_observed_nonzero": joint_count > 0,
                 "compiler_task_phi_attention_same_probe_count": joint_count,
                 "compiler_task_phi_nonzero_probe_count": sum(row["phi_task_gradient_norm"] > 0 for row in evidence),
                 "compiler_task_attention_nonzero_probe_count": sum(row["cross_attention_task_gradient_norm"] > 0 for row in evidence),
                 "task_gradient_evidence_sha256": sha256(run/"TASK_GRADIENT_EVIDENCE.json"),
                 "no_hidden_warmup": True, "frozen_yolo_online": True, "frozen_integrity": frozen,
                 "frozen_input_mutation_checks": state["frozen_input_checks"], "frozen_input_mutation_checks_passed": True,
                 "base_weights_sha256": OFFICIAL_SHA256, "sources": source_hashes, "data_hashes": provider.data_hashes,
                 "configuration_sha256": canonical_sha(config), "contract_sha256": canonical_sha(contract),
                 "smoke_only": contract["smoke_only"], "complete_original_train_split": contract["complete_original_train_split"],
                 "every_epoch_declared_images_complete": all(e["images"] == total_images for e in state["epochs"]),
                 "parent": parent, "trainable_parameters": sum(p.numel() for _, p in parameters),
                 "module_bn_modules": sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in head.modules()),
                 "effective_training_seconds": state["effective_training_seconds"], "this_execution_seconds": time.monotonic()-began}
        audit["passed"] = (len(state["epochs"]) == args.epochs and state["applied"] == state["attempts"] and state["applied"] > 0
                           and audit["all_applied_gradients_finite"] and audit["all_parameters_finite"]
                           and audit["all_optimizer_states_finite"] and audit["all_groups_really_changed"]
                           and joint_count > 0 and frozen.get("passed") is True and audit["every_epoch_declared_images_complete"])
        dump_json(run/"TRAINING_AUDIT.json", audit)
        if not audit["passed"]:
            raise RuntimeError("Final fullscale module audit failed; refuse deployable final checkpoint")
        payload = {"kind": FINAL_KIND, "epoch": args.epochs, "audit_passed": True, "state_dict": final,
                   "configuration": config, "configuration_sha256": canonical_sha(config),
                   "base_weights_sha256": OFFICIAL_SHA256, "data_hashes": provider.data_hashes,
                   "sources": source_hashes, "training_audit_sha256": sha256(run/"TRAINING_AUDIT.json"),
                   "task_gradient_evidence_sha256": audit["task_gradient_evidence_sha256"],
                   "contract": contract, "contract_sha256": canonical_sha(contract), "seed": args.seed,
                   "fixed_final_epoch": True, "smoke_only": contract["smoke_only"],
                   "complete_original_train_split": contract["complete_original_train_split"], "parent": parent}
        atomic_torch_save(run/"head_final.pt", payload)
        save_snapshot("final_valid_state")
        dump_json(run/"TRAINING_COMPLETE.json", {"status": "completed", "epochs": args.epochs,
                  "images_per_epoch": total_images, "head_sha256": sha256(run/"head_final.pt"), "head_bytes": (run/"head_final.pt").stat().st_size,
                  "audit_passed": True, "smoke_only": contract["smoke_only"], "complete_original_train_split": contract["complete_original_train_split"],
                  "optimizer_applied": state["applied"], "completed_at": now(), "parent": parent})
        print(f"TRAINING_COMPLETE {run/'head_final.pt'} smoke_only={contract['smoke_only']}", flush=True)
    except IntentionalEngineeringStop as exc:
        print(f"INTENTIONAL_ENGINEERING_STOP {exc}", flush=True)
        return 2
    except BaseException as exc:
        failure = {"error": repr(exc), "traceback": traceback.format_exc(), "time": now(),
                   "next_cursor": state["cursor"] if state else None, "optimizer_applied": state["applied"] if state else None,
                   "last_durable_checkpoint": None, "partial_checkpoint_saved": False}
        if (run/"CHECKPOINT_LATEST.json").exists():
            failure["last_durable_checkpoint"] = json.loads((run/"CHECKPOINT_LATEST.json").read_text(encoding="utf-8"))
        if all(item is not None for item in (provider, head, optimizer, state, contract, initial, shuffle_rng)):
            try:
                record = save_snapshot("caught_exception_partial")
                failure["partial_checkpoint_saved"] = True
                failure["last_durable_checkpoint"] = record
            except BaseException as checkpoint_error:
                failure["partial_checkpoint_error"] = repr(checkpoint_error)
        dump_json(run/"TRAINING_FAILURE.json", failure)
        raise
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main() or 0)
