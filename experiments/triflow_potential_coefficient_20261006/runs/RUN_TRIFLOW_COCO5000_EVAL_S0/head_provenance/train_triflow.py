"""Train only the TriFlow module from authenticated frozen native features."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from frozen_io import OFFICIAL_SHA256, dump_json, sha256, state_digest
from triflow_model import TriFlowConfig, TriFlowModel

TRAINER_VERSION = "triflow_module_train_v1"
GROUPS = ("token_projection", "ownership_embeddings", "ownership_projection", "cross_attention", "interaction_norm", "field_head")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def group_name(name):
    group = name.split(".")[0]
    if group not in GROUPS:
        raise ValueError(f"Unclassified learned parameter: {name}")
    return group


def serial(value):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    return value


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def collate_instances(payload, selected, device):
    """No annotation/query/neighbor selection occurs here: cache identity is fixed."""
    import torch
    rows = torch.tensor([t["output_row"] for t in selected], dtype=torch.long)
    count = len(selected)
    p, f = payload["P"].to(device), payload["F"].to(device)
    neighbor_rows = torch.stack([t["neighbor_rows"] for t in selected])
    valid = torch.stack([t["neighbor_valid"] for t in selected]).to(device)
    neighbors_c = payload["c0"][neighbor_rows.clamp_min(0)].to(device)
    neighbor_logits = torch.einsum("chw,nkc->nkhw", p, neighbors_c)
    neighbor_logits = neighbor_logits.masked_fill(~valid[:, :, None, None], 0)
    targets = {key: torch.stack([t["ownership_targets"][key] for t in selected]).to(device)
               for key in selected[0]["ownership_targets"]}
    inputs = {"prototypes": p, "coefficients": payload["c0"][rows].to(device), "visual": f,
              "boxes": payload["boxes_proto"][rows].to(device), "neighbor_logits": neighbor_logits,
              "neighbor_valid": valid, "instance_hidden": payload["h"][rows].to(device)}
    if payload.get("hidden_query_source") != "one2one_cv4 final convolution input, exact native raw index":
        raise ValueError("Protocol requires authenticated genuine final-convolution hidden query")
    if any(t.requires_grad for t in inputs.values()):
        raise ValueError("Frozen feature inputs must not require gradients")
    return inputs, targets


def validate_cache(cache_run, receipt, current_source):
    if receipt.get("status") != "completed" or not (cache_run/"CACHE_COMPLETE.json").is_file():
        raise ValueError("Only a completed frozen cache may be used for training")
    integrity = receipt.get("frozen_integrity", {})
    if integrity.get("passed") is not True or receipt.get("all_native_replay_exact") is not True:
        raise ValueError("Cache lacks passed actual frozen extraction/replay audits")
    if receipt.get("base_weights_sha256") != OFFICIAL_SHA256:
        raise ValueError("Cache official checkpoint differs from protocol")
    if sha256(receipt["base_weights"]) != OFFICIAL_SHA256:
        raise ValueError("Official checkpoint bytes changed since cache extraction")
    if receipt["sources"].get("triflow_model.py") != sha256(current_source/"triflow_model.py"):
        raise ValueError("Ownership-target source changed since cache extraction")
    marker = json.loads((cache_run/"CACHE_COMPLETE.json").read_text(encoding="utf-8"))
    if marker.get("receipt_sha256") != sha256(cache_run/"CACHE_RECEIPT.json"):
        raise ValueError("Completed cache receipt changed")
    for item in receipt["cache_files"]:
        path = (cache_run / item["cache_file"]).resolve()
        if not path.is_relative_to(cache_run) or not path.is_file():
            raise ValueError(f"Missing or escaping cache payload: {path}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen cache payload differs from extracted bytes: {path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-run", required=True)
    parser.add_argument("--epochs", type=int, required=True, choices=(1, 8))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instance-chunk", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed != 0 or args.instance_chunk != 4:
        raise ValueError("Fixed screen requires seed0 and instance_chunk4")
    root = Path(args.root).resolve()
    run = root / "runs" / args.run_id
    cache_run = Path(args.cache_run)
    if not cache_run.is_absolute():
        cache_run = root / "runs" / args.cache_run
    cache_run = cache_run.resolve()
    if (run/"head_final.pt").exists() or (run/"TRAIN_TRACE.jsonl").exists():
        raise FileExistsError("Training history is immutable; retries need another Run ID")
    run.mkdir(parents=True, exist_ok=True)
    source = run / "source"
    source.mkdir(exist_ok=True)
    for name in ("train_triflow.py", "frozen_io.py", "triflow_model.py"):
        current = Path(__file__).with_name(name)
        target = source / name
        if target.exists() and sha256(target) != sha256(current):
            raise ValueError("Run source snapshot differs from executing source")
        if not target.exists():
            shutil.copy2(current, target)
    if (root/"PROTOCOL.md").is_file() and not (source/"PROTOCOL.md").exists():
        shutil.copy2(root/"PROTOCOL.md", source/"PROTOCOL.md")
    started, began = now(), time.monotonic()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        receipt = json.loads((cache_run/"CACHE_RECEIPT.json").read_text(encoding="utf-8"))
        validate_cache(cache_run, receipt, source)
        dimensions = receipt["dimensions"]
        configuration = TriFlowConfig()
        head = TriFlowModel(dimensions["feature_channels"], dimensions["instance_hidden_channels"], configuration).to(args.device).float()
        head.train()
        initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        initial_digest = state_digest(head)
        initial_input_hashes = {}
        parameters = list(head.named_parameters())
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
        source_hashes = {name: sha256(source/name) for name in ("train_triflow.py", "frozen_io.py", "triflow_model.py")}
        config_dict = head.configuration()
        config_sha = hashlib.sha256(json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        inputs_record = {"trainer_version": TRAINER_VERSION, "started_at": started, "cache_run": str(cache_run),
                         "cache_receipt_sha256": sha256(cache_run/"CACHE_RECEIPT.json"),
                         "base_weights_sha256": OFFICIAL_SHA256, "annotation_sha256": receipt["annotation_sha256"],
                         "images_list_sha256": receipt["images_list_sha256"], "sources": source_hashes,
                         "configuration": config_dict, "configuration_sha256": config_sha,
                         "epochs": args.epochs, "seed": args.seed, "instance_chunk": args.instance_chunk,
                         "optimizer": {"class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip_grad_norm": 10},
                         "precision": "FP32, TF32 disabled, no AMP", "augmentation": False,
                         "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                         "device": args.device,
                         "selection": "fixed last epoch; development set unused; no best checkpoint",
                         "frozen_yolo": "not instantiated by trainer; authenticated cached native inputs only",
                         "execution_directory": str(run), "interpreter": sys.executable}
        dump_json(run/"TRAINING_INPUTS.json", inputs_record)
        attempt, applied, total_instances, task_gradient_nonzero = 0, 0, 0, False
        phi_attention_task_gradient_nonzero = False
        finite_groups = {group: True for group in GROUPS}
        gradient_observed = {group: 0 for group in GROUPS}
        task_gradient_evidence = []
        epoch_results = []
        rng = random.Random(args.seed)
        ordered = list(receipt["cache_files"])
        frozen_input_checks = 0
        for epoch in range(1, args.epochs+1):
            rng.shuffle(ordered)
            losses_total, supervised_total, diagnostic_total = {}, {}, {}
            instances_epoch, steps_epoch, task_probed = 0, 0, False
            epoch_start = time.monotonic()
            for image_position, item in enumerate(ordered, 1):
                cache_path = cache_run/item["cache_file"]
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                if payload.get("base_weights_sha256") != OFFICIAL_SHA256 or payload.get("image_id") != item["image_id"]:
                    raise ValueError("Loaded cache payload identity differs from receipt")
                selected = payload["training_targets"]
                if not selected:
                    continue
                for start in range(0, len(selected), args.instance_chunk):
                    chunk = selected[start:start+args.instance_chunk]
                    model_inputs, targets = collate_instances(payload, chunk, args.device)
                    # Actual feature integrity is measured before/after forward,
                    # loss, backward and update for the first used chunk each epoch.
                    check_inputs = not task_probed
                    before_inputs = {k: tensor_digest(v) for k, v in model_inputs.items()} if check_inputs else None
                    outputs = head(**model_inputs)
                    losses = head.loss(outputs, targets["self_masks"], targets["neighbor_masks"], model_inputs["neighbor_valid"], targets=targets)
                    if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
                        raise FloatingPointError("TriFlow loss contains NaN/Inf")
                    if not torch.isfinite(outputs["delta_coefficients"]).all() or not torch.isfinite(outputs["logits_refined"]).all():
                        raise FloatingPointError("Compiler output contains NaN/Inf")
                    if not task_probed:
                        task_gradients = torch.autograd.grad(losses["loss_task"], [p for _, p in parameters], retain_graph=True, allow_unused=True)
                        group_norms = {group: 0.0 for group in GROUPS}
                        phi_squared_norm = 0.0
                        for (name, _), gradient in zip(parameters, task_gradients):
                            if gradient is not None:
                                if not torch.isfinite(gradient).all():
                                    raise FloatingPointError("Compiled mask task gradient contains NaN/Inf")
                                group_norms[group_name(name)] += float(gradient.double().square().sum().cpu())
                                if name in ("field_head.2.weight", "field_head.2.bias"):
                                    # Only the first two output rows predict phi.
                                    # The third stiffness row cannot certify the
                                    # potential -> compiler -> task gradient.
                                    phi_squared_norm += float(gradient[:2].double().square().sum().cpu())
                        group_norms = {key: value**.5 for key, value in group_norms.items()}
                        nonzero = any(value > 0 for value in group_norms.values())
                        task_gradient_nonzero |= nonzero
                        phi_task_norm = phi_squared_norm**.5
                        attention_task_norm = group_norms["cross_attention"]
                        same_probe_nonzero = phi_task_norm > 0 and attention_task_norm > 0
                        phi_attention_task_gradient_nonzero |= same_probe_nonzero
                        evidence = {"epoch": epoch, "image_id": item["image_id"], "output_rows": [t["output_row"] for t in chunk],
                                    "all_task_gradients_finite": True, "group_task_gradient_norms": group_norms,
                                    "nonzero_task_gradient_reaches_learned_head": nonzero,
                                    "phi_task_gradient_norm": phi_task_norm,
                                    "cross_attention_task_gradient_norm": attention_task_norm,
                                    "task_gradient_phi_attention_same_probe_nonzero": same_probe_nonzero,
                                    "task_loss": float(losses["loss_task"].detach().cpu()),
                                    "diagnostics": serial(outputs["diagnostics"])}
                        task_gradient_evidence.append(evidence)
                        dump_json(run/"TASK_GRADIENT_EVIDENCE.json", {"evidence": task_gradient_evidence,
                                                                   "any_observed_nonzero": task_gradient_nonzero,
                                                                   "any_observed_phi_attention_same_probe_nonzero": phi_attention_task_gradient_nonzero,
                                                                   "no_hidden_warmup": True})
                        task_probed = True
                    optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    attempt += 1
                    groups_this_step = {group: False for group in GROUPS}
                    for name, parameter in parameters:
                        if parameter.grad is not None:
                            group = group_name(name)
                            if not torch.isfinite(parameter.grad).all():
                                finite_groups[group] = False
                                raise FloatingPointError(f"Nonfinite applied gradient for {name}")
                            groups_this_step[group] |= bool((parameter.grad != 0).any())
                    grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True)
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for _, parameter in parameters):
                        raise FloatingPointError("Learned head parameter contains NaN/Inf after update")
                    for group, observed in groups_this_step.items():
                        gradient_observed[group] += int(observed)
                    if check_inputs:
                        after_inputs = {k: tensor_digest(v) for k, v in model_inputs.items()}
                        if before_inputs != after_inputs:
                            raise RuntimeError("Frozen input tensors were mutated during training")
                        frozen_input_checks += 1
                        initial_input_hashes[str(epoch)] = before_inputs
                    applied += 1
                    steps_epoch += 1
                    instances_epoch += len(chunk)
                    total_instances += len(chunk)
                    trace = {"epoch": epoch, "optimizer_attempt": attempt, "applied": True,
                             "image_position": image_position, "image_id": item["image_id"],
                             "output_rows": [t["output_row"] for t in chunk],
                             "annotation_ids": [t["annotation_id"] for t in chunk],
                             "instances": len(chunk), "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                             "all_applied_gradients_finite": True, "all_parameters_finite": True,
                             "losses": serial(losses), "diagnostics": serial(outputs["diagnostics"])}
                    with (run/"TRAIN_TRACE.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(trace, allow_nan=False) + "\n")
                    for key, value in losses.items():
                        scalar = float(value.detach().cpu())
                        if key.startswith("loss"):
                            losses_total[key] = losses_total.get(key, 0) + scalar*len(chunk)
                        else:
                            supervised_total[key] = supervised_total.get(key, 0) + scalar
                    for key in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count"):
                        diagnostic_total[key] = diagnostic_total.get(key, 0) + int(outputs["diagnostics"][key].sum().detach().cpu())
                    if applied == 1 or applied % 25 == 0:
                        print(f"TRAIN epoch={epoch}/{args.epochs} images={image_position}/{len(ordered)} applied={applied} instances={total_instances} loss={float(losses['loss'].detach().cpu()):.6f} elapsed_s={time.monotonic()-began:.1f}", flush=True)
            if not instances_epoch:
                raise ValueError("No valid matched training instances in authenticated cache")
            result = {"epoch": epoch, "image_order": [item["image_id"] for item in ordered], "instances": instances_epoch,
                      "applied_steps": steps_epoch, "mean_losses_per_instance": {k: v/instances_epoch for k, v in losses_total.items()},
                      "supervision_counts": supervised_total, "compiler_counts": diagnostic_total,
                      "seconds": time.monotonic()-epoch_start}
            epoch_results.append(result)
            dump_json(run/"EPOCHS.json", {"epochs": epoch_results})
            print(f"EPOCH_COMPLETE {epoch}/{args.epochs} instances={instances_epoch} steps={steps_epoch} mean_loss={result['mean_losses_per_instance']['loss']:.6f} seconds={result['seconds']:.1f}", flush=True)
        changed = {group: {"changed_tensors": 0, "max_abs_delta": 0.0} for group in GROUPS}
        final_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        for name, value in final_state.items():
            group = group_name(name)
            delta = (value-initial[name]).abs()
            if bool((delta != 0).any()):
                changed[group]["changed_tensors"] += 1
                changed[group]["max_abs_delta"] = max(changed[group]["max_abs_delta"], float(delta.max()))
        audit = {
            "trainer_version": TRAINER_VERSION, "started_at": started, "completed_at": now(), "epochs": args.epochs,
            "optimizer_attempts": attempt, "optimizer_applied": applied, "skipped_updates": 0,
            "all_applied_gradients_finite": all(finite_groups.values()), "gradient_finite_by_group": finite_groups,
            "nonzero_gradient_steps_by_group": gradient_observed, "parameter_updates_by_group": changed,
            "all_parameters_finite": all(bool(torch.isfinite(value).all()) for value in final_state.values()),
            "initial_head_state_sha256": initial_digest, "final_head_state_sha256": state_digest(head),
            "all_groups_really_changed": all(item["changed_tensors"] > 0 for item in changed.values()),
            "compiler_task_gradient_observed_nonzero": task_gradient_nonzero,
            "compiler_task_phi_and_attention_same_probe_observed_nonzero": phi_attention_task_gradient_nonzero,
            "compiler_task_phi_nonzero_probe_count": sum(e["phi_task_gradient_norm"] > 0 for e in task_gradient_evidence),
            "compiler_task_attention_nonzero_probe_count": sum(e["cross_attention_task_gradient_norm"] > 0 for e in task_gradient_evidence),
            "compiler_task_phi_attention_same_probe_count": sum(e["task_gradient_phi_attention_same_probe_nonzero"] for e in task_gradient_evidence),
            "task_gradient_evidence_file": "TASK_GRADIENT_EVIDENCE.json", "no_hidden_warmup": True,
            "frozen_yolo_instantiated": False, "frozen_cache_receipt_integrity_passed": True,
            "frozen_input_mutation_checks": frozen_input_checks, "frozen_input_mutation_checks_passed": True,
            "frozen_input_tensor_hashes": initial_input_hashes, "base_weights_sha256": OFFICIAL_SHA256,
            "cache_receipt_sha256": sha256(cache_run/"CACHE_RECEIPT.json"), "sources": source_hashes,
            "configuration_sha256": config_sha, "trainable_parameters": sum(p.numel() for _, p in parameters),
            "elapsed_seconds": time.monotonic()-began,
        }
        audit["passed"] = (applied == attempt and applied > 0 and audit["all_applied_gradients_finite"] and
                           audit["all_parameters_finite"] and audit["all_groups_really_changed"] and
                           task_gradient_nonzero and phi_attention_task_gradient_nonzero)
        dump_json(run/"TRAINING_AUDIT.json", audit)
        if not audit["passed"]:
            raise RuntimeError("Training audit failed; preserve partial trace and refuse usable final checkpoint")
        payload = {"kind": "triflow_module_final", "epoch": args.epochs, "audit_passed": True,
                   "state_dict": final_state, "configuration": config_dict, "configuration_sha256": config_sha,
                   "base_weights_sha256": OFFICIAL_SHA256, "cache_receipt_sha256": audit["cache_receipt_sha256"],
                   "data_hashes": {"annotation_sha256": receipt["annotation_sha256"], "images_list_sha256": receipt["images_list_sha256"]},
                   "sources": source_hashes, "training_audit_sha256": sha256(run/"TRAINING_AUDIT.json"),
                   "seed": args.seed, "fixed_final_epoch": True, "smoke_only": args.epochs == 1,
                   "trained_ownership_and_compiler_task": True}
        temporary = run/"head_final.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(run/"head_final.pt")
        dump_json(run/"TRAINING_COMPLETE.json", {"status": "completed", "epochs": args.epochs,
                                                "head_sha256": sha256(run/"head_final.pt"), "head_bytes": (run/"head_final.pt").stat().st_size,
                                                "audit_passed": True, "completed_at": now()})
    except Exception as exc:
        dump_json(run/"TRAINING_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
        raise


if __name__ == "__main__":
    main()
