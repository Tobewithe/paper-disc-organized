"""R3 declared diagnostics runtime and synchronous durable-epoch observation.

Original train_20k.py, source contract, optimizer cadence and PROTOCOL remain
untouched. Epoch evaluation pauses training after the actual atomic boundary
snapshot. Extra execution sources and the addendum are recorded separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import inspect
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

from frozen_io import dump_json, sha256, state_digest
from diagnostics_contract_r3 import archive_contract, verify_contract, import_runtime
import train_20k as trainer

OBSERVER_VERSION = "triflow_20k_epoch_observer_r3_v1"
TRAINING_SOURCE_FILES = ("train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py")
FORMAL_PREFIX = "RUN_TRIFLOW_20K_EPOCH"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def exact_tree_sha(value):
    """Typed byte digest, including every tensor/RNG array/float bit and key."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray\0" + str(item.dtype).encode() + repr(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0" + str(len(item)).encode() + b"\0")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"\0" + str(len(item)).encode() + b"\0")
            for element in item:
                add(element)
        elif isinstance(item, set):
            digest.update(b"set\0" + str(len(item)).encode() + b"\0")
            for element in sorted(item, key=lambda element: (type(element).__name__, repr(element))):
                add(element)
        elif isinstance(item, float):
            digest.update(b"float\0" + struct.pack("!d", item))
        elif isinstance(item, (bool, int, str, bytes)) or item is None:
            encoded = item if isinstance(item, bytes) else repr(item).encode("utf-8")
            digest.update(type(item).__name__.encode() + b"\0" + str(len(encoded)).encode() + b"\0" + encoded)
        else:
            raise TypeError(f"Unsupported scientific digest value: {type(item).__name__}")
    add(value)
    return digest.hexdigest()


def parent_digests(live):
    import torch
    head, optimizer, provider = live["head"], live["optimizer"], live["provider"]
    context = {
        "head_training": head.training, "frozen_model_training": provider.extractor.model.training,
        "head_requires_grad": {name: value.requires_grad for name, value in head.named_parameters()},
        "frozen_requires_grad": {name: value.requires_grad for name, value in provider.extractor.model.named_parameters()},
        "head_configuration": head.configuration(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "diagnostic_scope": getattr(head, "diagnostic_scope", None),
    }
    return {
        "head_state_sha256": state_digest(head),
        "optimizer_state_sha256": exact_tree_sha(optimizer.state_dict()),
        "rng_state_sha256": exact_tree_sha(trainer.rng_state(live["shuffle_rng"])),
        "training_state_sha256": exact_tree_sha(live["state"]),
        "provider_state_sha256": exact_tree_sha(provider.state_dict()),
        "frozen_model_state_sha256": state_digest(provider.extractor.model),
        "execution_context_sha256": exact_tree_sha(context),
    }


def verify_parent_equal(before, after):
    equality = {key: before[key] == after[key] for key in before}
    if set(before) != set(after) or not all(equality.values()):
        raise RuntimeError(f"Epoch callback changed parent scientific state: {equality}")
    return equality


def atomic_copy(source, target):
    if target.exists():
        raise FileExistsError(f"Epoch captures are immutable: {target}")
    temporary = target.with_name(target.name+".tmp")
    with Path(source).open("rb") as read, temporary.open("wb") as write:
        shutil.copyfileobj(read, write, length=4*1024*1024)
        write.flush()
        os.fsync(write.fileno())
    temporary.replace(target)


def normalize_trainer_args(args, tail):
    """Support wrapper root/run/resume flags or the same original passthrough."""
    tail = list(tail[1:] if tail[:1] == ["--"] else tail)
    query = argparse.ArgumentParser(add_help=False)
    for name in ("root", "run-id", "resume-from", "weights", "vendor", "images-list", "annotations", "device"):
        query.add_argument("--"+name)
    query.add_argument("--max-images", type=int)
    parsed, _ = query.parse_known_args(tail)
    for name in ("root", "run_id", "resume_from"):
        wrapper_value, original_value = getattr(args, name), getattr(parsed, name)
        if wrapper_value is not None:
            if original_value is not None:
                if name == "run_id":
                    same = wrapper_value == original_value
                else:
                    same = Path(wrapper_value).resolve() == Path(original_value).resolve()
                if not same:
                    raise ValueError("Wrapper and original CLI disagree: " + name)
            else:
                tail += ["--"+name.replace("_", "-"), wrapper_value]
    parsed, _ = query.parse_known_args(tail)
    if not parsed.root or not parsed.run_id:
        raise ValueError("Original training --root and --run-id are required")
    return tail, parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-from")
    parser.add_argument("--epoch-eval-script")
    parser.add_argument("--epoch-verifier-script")
    parser.add_argument("--engineering-callback-script")
    parser.add_argument("--addendum", required=True)
    parser.add_argument("--diagnostics-runtime", required=True)
    parser.add_argument("--diagnostics-parity-receipt", required=True)
    parser.add_argument("--diagnostics-addendum", required=True)
    parser.add_argument("--callback-images-list")
    parser.add_argument("--callback-annotations")
    parser.add_argument("--callback-baseline-cache")
    parser.add_argument("--callback-device")
    parser.add_argument("--epoch-run-prefix")
    parser.add_argument("--epoch-run-suffix", default="S0")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tail, training_args = normalize_trainer_args(args, args.trainer_args)
    root = Path(training_args.root).resolve()
    run_id = training_args.run_id
    run = root/"runs"/run_id
    run.mkdir(parents=True, exist_ok=True)
    if (run/"OBSERVER_INPUTS.json").exists():
        raise FileExistsError("Observer execution exists; use a distinct new training Run ID")
    engineering = training_args.max_images is not None
    if engineering:
        if not args.engineering_callback_script:
            raise ValueError("Engineering mode requires its explicit engineering callback sink")
    elif args.engineering_callback_script:
        raise ValueError("Engineering callback must never replace formal epoch evaluation")
    elif not all((args.epoch_eval_script, args.epoch_verifier_script, args.callback_images_list,
                  args.callback_annotations, args.callback_baseline_cache)):
        raise ValueError("Formal observer requires real evaluator/verifier and complete5000 callback inputs")
    scripts = {}
    for key, value in (("epoch_evaluator", args.epoch_eval_script), ("epoch_verifier", args.epoch_verifier_script),
                       ("engineering_callback", args.engineering_callback_script)):
        if value:
            path = Path(value).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            scripts[key] = {"path": str(path), "sha256": sha256(path)}
    addendum = Path(args.addendum).resolve()
    if not addendum.is_file():
        raise FileNotFoundError(addendum)
    source = run/"source"
    source.mkdir(exist_ok=True)
    actual_observer = Path(__file__).resolve()
    observer_sha = sha256(actual_observer)
    archived_observer = source/"train_epoch_observer_r3.py"
    if archived_observer.exists() and sha256(archived_observer) != observer_sha:
        raise ValueError("Observer archive differs from executed source")
    if not archived_observer.exists():
        shutil.copy2(actual_observer, archived_observer)
    addendum_target = source/"EPOCH_ADDENDUM.md"
    if addendum_target.exists() and sha256(addendum_target) != sha256(addendum):
        raise ValueError("Addendum archive differs from declared actual text")
    if not addendum_target.exists():
        shutil.copy2(addendum, addendum_target)
    source_hashes = {name: sha256(Path(trainer.__file__).with_name(name)) for name in TRAINING_SOURCE_FILES}
    diagnostics_execution = archive_contract(args.diagnostics_runtime, args.diagnostics_parity_receipt, args.diagnostics_addendum, run)
    diagnostics_runtime = import_runtime(run/diagnostics_execution["runtime_source_file"])
    original_model_class = trainer.TriFlowModel
    original_collate = trainer.collate_instances
    trainer.TriFlowModel = diagnostics_runtime.install(original_model_class, mode="minimal")
    mode_counts = {"minimal": 0, "full": 0}

    def collate_with_declared_diagnostics(prepared, chunk):
        caller = inspect.currentframe().f_back
        try:
            live_head, live_state, live_epoch = (caller.f_locals[key] for key in ("head", "state", "epoch"))
        finally:
            del caller
        mode = "full" if live_state["task_probed_epoch"] != live_epoch else "minimal"
        diagnostics_runtime.set_diagnostics(live_head, mode)
        mode_counts[mode] += 1
        return original_collate(prepared, chunk)

    trainer.collate_instances = collate_with_declared_diagnostics
    dump_json(run/"DIAGNOSTICS_RUNTIME.json", diagnostics_execution)
    prefix = args.epoch_run_prefix or (run_id+"_EPOCH" if engineering else FORMAL_PREFIX)
    suffix = args.epoch_run_suffix
    if any(Path(value).name != value or value in (".", "..") for value in (prefix, suffix)):
        raise ValueError("Epoch namespaces must be plain identifier components")
    if engineering and prefix == FORMAL_PREFIX:
        raise ValueError("Engineering callbacks cannot occupy formal epoch Run IDs")
    callback_device = args.callback_device or training_args.device or "cuda"
    observation = {
        "observer_version": OBSERVER_VERSION, "started_at": now(), "training_run_id": run_id,
        "observer_source_sha256": observer_sha, "observer_source_file": str(archived_observer),
        "original_training_source_sha256": source_hashes,
        "protocol_sha256": sha256(root/"PROTOCOL.md"),
        "addendum_source": str(addendum), "addendum_sha256": sha256(addendum_target),
        "addendum_archive": str(addendum_target), "callback_scripts": scripts,
        "diagnostics_execution": diagnostics_execution,
        "engineering_only": engineering, "complete5000_evaluation_required": not engineering,
        "epoch_run_prefix": prefix, "epoch_run_suffix": suffix, "original_trainer_args": tail,
        "routine_epochs_1_to_7": "complete5000 AP-only", "epoch8": "complete5000 AP and paired readout",
        "execution_change": "declared R3 diagnostics subclass plus synchronous epoch callbacks; original source4/contract preserved, actual runtime changed",
    }
    dump_json(run/"OBSERVER_INPUTS.json", observation)
    original_save = trainer.atomic_torch_save
    observed_epochs = set()

    def guard_sources(payload):
        if payload["contract"].get("sources") != source_hashes:
            raise ValueError("Boundary payload does not match the original four-source training contract")
        for name, digest in source_hashes.items():
            if sha256(Path(trainer.__file__).with_name(name)) != digest:
                raise ValueError("Original training source changed during observation: " + name)
        if sha256(actual_observer) != observer_sha or sha256(addendum_target) != observation["addendum_sha256"]:
            raise ValueError("Observer/addendum source changed during execution")
        verify_contract(run, diagnostics_execution)
        if sha256(Path(__file__).with_name("diagnostics_contract_r3.py")) != diagnostics_execution["adapter_sha256"]:
            raise ValueError("Declared diagnostics execution adapter changed")
        for current, key in ((args.diagnostics_runtime, "runtime"), (args.diagnostics_parity_receipt, "parity"), (args.diagnostics_addendum, "addendum")):
            if sha256(current) != diagnostics_execution[key+"_sha256"]:
                raise ValueError("Declared actual diagnostics execution input changed: " + key)
        if (sha256(root/"PROTOCOL.md") != observation["protocol_sha256"]
                or sha256(source/"PROTOCOL.md") != observation["protocol_sha256"]
                or payload["contract"].get("protocol_sha256") != observation["protocol_sha256"]):
            raise ValueError("Original locked protocol changed during epoch observation")
        for entry in scripts.values():
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Declared callback source changed during execution")

    def launch_stage(identifier, purpose, script_key, arguments):
        entry = scripts[script_key]
        script = Path(entry["path"])
        # run_stage archives actual sources from root/scripts. Its entry bytes
        # must be exactly the callback source declared by this observer.
        current = root/"scripts"/script.name
        if not current.is_file() or sha256(current) != entry["sha256"]:
            raise ValueError("Callback stage archive would execute different source: " + script.name)
        command = [sys.executable, "-X", "utf8", str(root/"scripts"/"run_stage.py"), "--root", str(root),
                   "--run-id", identifier, "--purpose", purpose, "--script", script.name, "--"]+arguments
        code = subprocess.run(command, cwd=root).returncode
        if code:
            raise RuntimeError(f"Epoch callback {identifier} failed with exit{code}; original boundary snapshot retained")

    def observed_save(path, payload):
        # Scientific tensors/state/contract retain the original trainer schema.
        # Top-level execution metadata explicitly binds the actual R3 runtime.
        if isinstance(payload, dict) and payload.get("kind") in (trainer.CHECKPOINT_KIND, trainer.FINAL_KIND):
            payload = dict(payload, execution_diagnostics_metadata=diagnostics_execution,
                           execution_diagnostics_mode_counts_this_execution=dict(mode_counts))
        # Persist before any epoch observation or child process.
        original_save(path, payload)
        if not isinstance(payload, dict) or payload.get("kind") != trainer.CHECKPOINT_KIND or payload.get("reason") != "epoch_boundary":
            return
        state = payload.get("state", {})
        epochs = state.get("epochs", [])
        epoch = len(epochs)
        if not 1 <= epoch <= 8:
            raise ValueError("Observed boundary must contain one through eight actual completed epochs")
        if state.get("cursor") != {"epoch": epoch+1, "image_position": 0, "chunk_start": 0} or [row.get("epoch") for row in epochs] != list(range(1, epoch+1)):
            raise ValueError("Observed snapshot is not an actual completed-epoch boundary")
        if epoch in observed_epochs:
            raise ValueError("An epoch boundary was observed twice in one training execution")
        if payload.get("run_id") != run_id or Path(path).resolve().parent != run:
            raise ValueError("Observed snapshot belongs to another training execution")
        if payload["contract"].get("smoke_only") is not engineering:
            raise ValueError("Callback scope differs from actual snapshot training scope")
        guard_sources(payload)
        caller = inspect.currentframe().f_back
        try:
            live = {name: caller.f_locals[name] for name in ("head", "optimizer", "provider", "shuffle_rng", "state")}
        finally:
            del caller
        before = parent_digests(live)
        if before["head_state_sha256"] != payload.get("head_state_sha256") or before["optimizer_state_sha256"] != exact_tree_sha(payload["optimizer_state_dict"]) or before["rng_state_sha256"] != exact_tree_sha(payload["rng"]):
            raise ValueError("Actual live parent does not match the durable boundary snapshot")
        eval_id = f"{prefix}{epoch:02d}_EVAL_{suffix}"
        verifier_id = f"{prefix}{epoch:02d}_READOUT_VERIFY_{suffix}"
        eval_run = root/"runs"/eval_id
        if (eval_run/"BOUNDARY_CAPTURE.json").exists() or (eval_run/"run.json").exists():
            raise FileExistsError("Epoch callback Run already exists; preserve it and use a retry namespace: " + eval_id)
        eval_run.mkdir(parents=True, exist_ok=True)
        captured = eval_run/f"head_epoch_{epoch:02d}.pt"
        atomic_copy(Path(path), captured)
        snapshot_sha = sha256(Path(path))
        if sha256(captured) != snapshot_sha:
            raise RuntimeError("Captured epoch file differs from the actually persisted original snapshot")
        provenance = eval_run/"epoch_provenance"
        provenance.mkdir(exist_ok=True)
        shutil.copy2(actual_observer, provenance/"train_epoch_observer_r3.py")
        shutil.copy2(addendum_target, provenance/"EPOCH_ADDENDUM.md")
        captured_diagnostics = archive_contract(args.diagnostics_runtime, args.diagnostics_parity_receipt, args.diagnostics_addendum, eval_run)
        if captured_diagnostics != diagnostics_execution:
            raise ValueError("Epoch diagnostics capture differs from actual training runtime contract")
        capture = {
            "capture_passed": True, "observer_version": OBSERVER_VERSION, "epoch": epoch,
            "training_run_id": run_id, "snapshot_run_id": payload["run_id"], "training_run": str(run),
            "reason": "epoch_boundary", "completed_epoch_count": epoch, "cursor": state["cursor"],
            "original_snapshot_path": str(Path(path).resolve()), "snapshot_sha256": snapshot_sha,
            "captured_snapshot_path": str(captured), "head_state_sha256": payload["head_state_sha256"],
            "optimizer_applied": state["applied"], "contract_sha256": payload["contract_sha256"],
            "data_hashes": payload["contract"]["data_hashes"], "training_source_sha256": source_hashes,
            "observer_source_sha256": observer_sha, "observer_source_file": "epoch_provenance/train_epoch_observer_r3.py",
            "addendum_sha256": observation["addendum_sha256"], "addendum_source_file": "epoch_provenance/EPOCH_ADDENDUM.md",
            "callback_script_sha256": {name: entry["sha256"] for name, entry in scripts.items()},
            "engineering_only": engineering, "final_head_kind_claimed": False,
            "parent_scientific_digests_before": before, "captured_at": now(),
            "diagnostics_execution": diagnostics_execution, "diagnostics_mode_at_boundary": getattr(live["head"], "diagnostic_scope", None),
            "diagnostics_mode_counts_this_execution": dict(mode_counts),
        }
        capture_path = eval_run/"BOUNDARY_CAPTURE.json"
        dump_json(capture_path, capture)
        import torch
        memory = {"gc_collected_objects": gc.collect(), "provider_data_modified": False,
                  "scope": "collect unreferenced temporaries; persistent COCO provider and live tensors retained"}
        if callback_device.startswith("cuda"):
            torch.cuda.synchronize()
            memory.update(cuda_allocated_before=torch.cuda.memory_allocated(), cuda_reserved_before=torch.cuda.memory_reserved())
            torch.cuda.empty_cache()
            memory.update(cuda_allocated_after=torch.cuda.memory_allocated(), cuda_reserved_after=torch.cuda.memory_reserved(),
                          cuda_synchronized=True, empty_cache_called=True)
        else:
            memory.update(cuda_synchronized=False, empty_cache_called=False)
        started = time.monotonic()
        record = {"epoch": epoch, "training_run_id": run_id, "evaluation_run_id": eval_id,
                  "verification_run_id": None if engineering else verifier_id, "snapshot_sha256": snapshot_sha,
                  "capture_receipt_sha256": sha256(capture_path), "engineering_only": engineering,
                  "memory_release": memory, "started_at": now(), "passed": False}
        try:
            arguments = ["--root", str(root), "--run-id", eval_id, "--snapshot", str(captured),
                         "--expected-epoch", str(epoch), "--training-run", str(run)]
            if engineering:
                launch_stage(eval_id, "engineering_epoch_callback_no_coco_result", "engineering_callback", arguments)
                sink = json.loads((eval_run/"ENGINEERING_CALLBACK.json").read_text(encoding="utf-8"))
                if sink.get("passed") is not True:
                    raise ValueError("Engineering callback did not provide its declared passed sink evidence")
            else:
                arguments += ["--weights", training_args.weights, "--vendor", training_args.vendor,
                              "--images-list", args.callback_images_list, "--annotations", args.callback_annotations,
                              "--baseline-cache", args.callback_baseline_cache, "--device", callback_device,
                              "--diagnostics-runtime", str(Path(args.diagnostics_runtime).resolve()),
                              "--diagnostics-parity-receipt", str(Path(args.diagnostics_parity_receipt).resolve()),
                              "--diagnostics-addendum", str(Path(args.diagnostics_addendum).resolve())]
                if epoch < 8:
                    arguments.append("--ap-only")
                launch_stage(eval_id, f"complete5000_native_epoch{epoch}_readout", "epoch_evaluator", arguments)
            after_eval = parent_digests(live)
            equality = verify_parent_equal(before, after_eval)
            parity = {"epoch": epoch, "training_run_id": run_id,
                      "scope": "after evaluator, before independent verifier", "before": before, "after": after_eval,
                      "equal_fields": equality, "passed": True, "capture_receipt_sha256": sha256(capture_path),
                      "observer_source_sha256": observer_sha, "created_at": now(), "engineering_only": engineering}
            dump_json(eval_run/"PARENT_STATE_PARITY.json", parity)
            if not engineering:
                verify_run = root/"runs"/verifier_id
                arguments = ["--run", str(eval_run), "--expected-epoch", str(epoch),
                             "--out", str(verify_run/"READOUT_VERIFICATION.json")]
                launch_stage(verifier_id, f"independent_complete5000_epoch{epoch}_aggregate_readout", "epoch_verifier", arguments)
                readout_path = verify_run/"READOUT_VERIFICATION.json"
                result = json.loads(readout_path.read_text(encoding="utf-8"))
                if result.get("passed") is not True:
                    raise ValueError("Independent epoch readout did not pass")
                record["readout_verification_sha256"] = sha256(readout_path)
            after_all = parent_digests(live)
            final_equality = verify_parent_equal(before, after_all)
            guard_sources(payload)
            final = {"epoch": epoch, "training_run_id": run_id, "scope": "after evaluator and independent verifier",
                     "before": before, "after": after_all, "equal_fields": final_equality, "passed": True,
                     "capture_receipt_sha256": sha256(capture_path), "post_eval_parity_sha256": sha256(eval_run/"PARENT_STATE_PARITY.json"),
                     "observer_source_sha256": observer_sha, "created_at": now(), "engineering_only": engineering}
            final_path = eval_run/"PARENT_STATE_FINAL_PARITY.json"
            dump_json(final_path, final)
            record.update(passed=True, final_parent_parity_sha256=sha256(final_path), completed_at=now(),
                          callback_seconds=time.monotonic()-started, source4_unchanged=True, protocol_unchanged=True,
                          same_parent_head_optimizer_all_rng_and_input_state=True)
            observed_epochs.add(epoch)
            dump_json(eval_run/"OBSERVER_EPOCH_COMPLETE.json", record)
            print(f"EPOCH_OBSERVATION_COMPLETE epoch={epoch} eval={eval_id} parent_state_exact=True", flush=True)
        except BaseException as error:
            record.update(error=repr(error), traceback=traceback.format_exc(), failed_at=now(), callback_seconds=time.monotonic()-started)
            try:
                record["parent_after_failure"] = parent_digests(live)
                record["parent_equal_after_failure"] = record["parent_after_failure"] == before
            except BaseException as state_error:
                record["parent_after_failure_error"] = repr(state_error)
            dump_json(eval_run/"OBSERVER_EPOCH_FAILURE.json", record)
            raise
        finally:
            with (run/"EPOCH_CALLBACKS.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")

    trainer.atomic_torch_save = observed_save
    original_argv = sys.argv
    sys.argv = [str(Path(trainer.__file__).resolve())]+tail
    try:
        code = trainer.main()
        dump_json(run/"OBSERVER_COMPLETE.json", {"observer_version": OBSERVER_VERSION, "training_exit_code": code or 0,
                  "observed_epochs_this_execution": sorted(observed_epochs), "engineering_only": engineering,
                  "observer_source_sha256": observer_sha, "addendum_sha256": observation["addendum_sha256"],
                  "diagnostics_execution": diagnostics_execution, "diagnostics_mode_counts_this_execution": dict(mode_counts),
                  "completed_at": now(), "scope": "callback execution completion; scientific outcome belongs to actual epoch readouts"})
        return code or 0
    except BaseException as error:
        dump_json(run/"OBSERVER_FAILURE.json", {"observer_version": OBSERVER_VERSION, "error": repr(error),
                  "observed_epochs_this_execution": sorted(observed_epochs), "failed_at": now(),
                  "scope": "Preserve original durable training snapshot and all partial callback Runs"})
        raise
    finally:
        trainer.atomic_torch_save = original_save
        trainer.TriFlowModel = original_model_class
        trainer.collate_instances = original_collate
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
