"""Real-input, smoke-only equivalence audit for explicit TriFlow diagnostics.

This runs the original full model and the minimal runtime on identical frozen
native inputs/targets and RNG. It checks exact learned outputs, loss terms,
every parameter gradient, one actual AdamW update, optimizer and RNG states.
It also profiles eigvalsh/Cholesky calls and times forward and training steps.
No synthetic completed training prefix or AP result is produced.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

VERIFIER_VERSION = "triflow_diagnostics_parity_v1"
SOURCE_FILES = ("verify_diagnostics_runtime.py", "diagnostics_runtime.py", "train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py")
CHECK_NAMES = (
    "coefficients_exact", "logits_exact", "loss_terms_exact",
    "all_parameter_gradients_exact", "adamw_head_exact", "adamw_optimizer_exact",
    "rng_state_exact", "full_scope_original_exact", "minimal_diagnostic_keys_exact",
    "minimal_expensive_diagnostics_absent", "checked_cholesky_preserved",
    "original_source_unmodified", "meaningful_real_inputs",
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item, torch) for item in value)
    return copy.deepcopy(value)


def capture_rng(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(value, torch, np):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"]:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item, torch) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


class ExactComparator:
    def __init__(self, torch, np):
        self.torch, self.np = torch, np
        self.mismatches = []
        self.max_abs_difference = 0.0
        self.tensor_comparisons = 0

    def fail(self, path, message):
        if len(self.mismatches) < 50:
            self.mismatches.append({"path": path, "reason": message})
        return False

    def compare(self, left, right, path="value"):
        torch, np = self.torch, self.np
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return self.fail(path, "tensor/non-tensor mismatch")
            self.tensor_comparisons += 1
            if left.dtype != right.dtype or left.shape != right.shape:
                return self.fail(path, "dtype or shape mismatch")
            left, right = left.detach().cpu(), right.detach().cpu()
            equal = bool(torch.equal(left, right))
            if not equal and left.numel() and left.dtype != torch.bool:
                difference = (left.double() - right.double()).abs()
                if bool(torch.isfinite(difference).all()):
                    self.max_abs_difference = max(self.max_abs_difference, float(difference.max()))
            return equal or self.fail(path, "not bit-exact torch.equal")
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            equal = (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                     and left.dtype == right.dtype and left.shape == right.shape and bool(np.array_equal(left, right)))
            return equal or self.fail(path, "numpy state differs")
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
                return self.fail(path, "dictionary keys differ")
            values = [self.compare(left[key], right[key], path + "." + str(key)) for key in left]
            return all(values)
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                return self.fail(path, "sequence type or length differs")
            values = [self.compare(a, b, path + "[" + str(index) + "]") for index, (a, b) in enumerate(zip(left, right))]
            return all(values)
        return type(left) is type(right) and left == right or self.fail(path, "scalar differs")

    def report(self):
        return {"exact": not self.mismatches, "tensor_comparisons": self.tensor_comparisons,
                "max_abs_difference": self.max_abs_difference, "mismatches": self.mismatches}


def synchronize(device, torch):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def run_step(model, state, rng, inputs, targets, torch, np, initial_optimizer=None):
    model.load_state_dict(state, strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    if initial_optimizer is not None:
        optimizer.load_state_dict(cpu_tree(initial_optimizer, torch))
    restore_rng(rng, torch, np)
    synchronize(inputs["prototypes"].device, torch)
    began = time.perf_counter()
    outputs = model(**inputs)
    terms = model.loss(outputs, targets["self_masks"], targets=targets)
    terms["loss"].backward()
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                 for name, parameter in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    if not bool(torch.isfinite(norm)) or not finite_tree(gradients, torch):
        raise FloatingPointError("Actual parity training step gradients are nonfinite")
    optimizer.step()
    synchronize(inputs["prototypes"].device, torch)
    elapsed = time.perf_counter() - began
    outputs_to_compare = {key: cpu_tree(value, torch) for key, value in outputs.items()
                          if not key.startswith("_") and key not in ("diagnostics", "diagnostic_scope", "diagnostics_unmeasured")}
    result = {"outputs": outputs_to_compare, "loss": cpu_tree(terms, torch), "gradients": gradients,
              "clip_grad_norm": norm.detach().cpu().clone(), "head": cpu_tree(model.state_dict(), torch),
              "optimizer": cpu_tree(optimizer.state_dict(), torch), "rng": capture_rng(torch, np),
              "diagnostic_keys": sorted(outputs["diagnostics"]), "diagnostics": cpu_tree(outputs["diagnostics"], torch),
              "diagnostic_scope": outputs.get("diagnostic_scope", "original_full"),
              "diagnostics_unmeasured": outputs.get("diagnostics_unmeasured", []), "elapsed_seconds": elapsed}
    if not finite_tree({key: result[key] for key in ("loss", "gradients", "head", "optimizer")}, torch):
        raise FloatingPointError("Actual parity update or losses became nonfinite")
    del outputs, terms, optimizer
    model.zero_grad(set_to_none=True)
    return result


def profile_forward(model, initial, rng, inputs, torch, np):
    model.load_state_dict(initial, strict=True)
    model.eval()
    restore_rng(rng, torch, np)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with torch.inference_mode():
            output = model(**inputs)
        synchronize(inputs["prototypes"].device, torch)
    events = {item.key: item.count for item in profiler.key_averages()}
    result = {"eigvalsh_calls": events.get("aten::linalg_eigvalsh", 0),
              "eigh_backend_calls": events.get("aten::_linalg_eigh", 0),
              "cholesky_ex_calls": events.get("aten::linalg_cholesky_ex", 0),
              "cholesky_solve_calls": events.get("aten::cholesky_solve", 0),
              "method": "actual torch.profiler CPU-dispatch events; not a torch monkeypatch"}
    del output
    return result


def timing(model, initial, rng, inputs, targets, repeats, torch, np, initial_optimizer=None):
    # Warm-up is timing-only, after parity gates; it does not supply training
    # evidence or alter a production checkpoint. Reset before every timed step.
    model.load_state_dict(initial, strict=True)
    model.eval()
    with torch.inference_mode():
        warmup = model(**inputs)
    del warmup
    synchronize(inputs["prototypes"].device, torch)
    forward, training = [], []
    for _ in range(repeats):
        model.load_state_dict(initial, strict=True)
        restore_rng(rng, torch, np)
        model.eval()
        synchronize(inputs["prototypes"].device, torch)
        began = time.perf_counter()
        with torch.inference_mode():
            output = model(**inputs)
        synchronize(inputs["prototypes"].device, torch)
        forward.append(time.perf_counter() - began)
        del output
        # A timed actual step avoids host-side tensor snapshots used by parity.
        model.load_state_dict(initial, strict=True)
        model.train()
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        if initial_optimizer is not None:
            optimizer.load_state_dict(cpu_tree(initial_optimizer, torch))
        restore_rng(rng, torch, np)
        synchronize(inputs["prototypes"].device, torch)
        began = time.perf_counter()
        outputs = model(**inputs)
        terms = model.loss(outputs, targets["self_masks"], targets=targets)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        synchronize(inputs["prototypes"].device, torch)
        training.append(time.perf_counter() - began)
        del optimizer, outputs, terms
    model.zero_grad(set_to_none=True)
    return {"repeats": repeats, "forward_seconds": forward, "forward_median_seconds": statistics.median(forward),
            "train_step_seconds": training, "train_step_median_seconds": statistics.median(training),
            "scope": "module-only actual chunk; excludes online YOLO/GT extraction and does not establish end-to-end speedup"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run-id", "images-list", "annotations", "weights", "vendor"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--instance-chunk", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--head", type=Path, help="Optional actual head/snapshot state; never modified")
    args = parser.parse_args()
    if args.max_images < 1 or args.instance_chunk != 4 or args.timing_repeats < 1:
        parser.error("Use positive engineering image/repeat counts and unchanged microchunk4")
    run = Path(args.root).resolve() / "runs" / args.run_id
    receipt_file = run / "DIAGNOSTICS_PARITY.json"
    if receipt_file.exists() or (run / "PARITY_FAILURE.json").exists():
        raise FileExistsError("Parity history exists; use a distinct Run ID")
    run.mkdir(parents=True, exist_ok=True)
    archive = run / "source"
    archive.mkdir(exist_ok=True)
    source_sha = {}
    for name in SOURCE_FILES:
        current = Path(__file__).with_name(name)
        target = archive / name
        if target.exists() and file_sha(target) != file_sha(current):
            raise ValueError("Archived parity source differs: " + name)
        if not target.exists():
            shutil.copy2(current, target)
        source_sha[name] = file_sha(current)
    receipt = {"verifier_version": VERIFIER_VERSION, "started_at": now(), "status": "running", "passed": False,
               "smoke_only": True, "scientific_scale_claimed": False, "chunk_count": 0,
               "original_model_source_sha256": source_sha["triflow_model.py"],
               "runtime_sha256": source_sha["diagnostics_runtime.py"], "verifier_sha256": source_sha["verify_diagnostics_runtime.py"],
               "checks": {name: None for name in CHECK_NAMES}, "cases": [], "timings": [],
               "executed_source_sha256": source_sha,
               "inputs": {"source_sha256": {name: source_sha[name] for name in ("train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py")}, "images_list": str(Path(args.images_list).resolve()),
                          "images_list_sha256": file_sha(args.images_list), "annotations": str(Path(args.annotations).resolve()),
                          "annotations_sha256": file_sha(args.annotations), "weights_sha256": file_sha(args.weights),
                          "device": args.device, "max_images": args.max_images, "instance_chunk": args.instance_chunk,
                          "head": None if args.head is None else {"path": str(args.head.resolve()), "sha256": file_sha(args.head)}},
               "limitations": ["Few real engineering images/chunks; no AP, method-quality or complete20k coverage claim",
                               "Exact gates apply only to observed tensors/steps/device; timing does not establish end-to-end speedup"]}
    provider = None
    try:
        import numpy as np
        import torch
        from diagnostics_runtime import (FULL_DIAGNOSTIC_KEYS, MINIMAL_KEYS, ORIGINAL_MODEL_SOURCE_SHA256,
                                         TriFlowRuntimeModel, diagnostics_scope, guard_original_source, install)
        from frozen_io import OFFICIAL_SHA256
        from stream_data import OnlineCOCOProvider
        from train_20k import prepare_image, collate_instances
        from triflow_model import TriFlowConfig, TriFlowModel
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        guard_original_source()
        receipt["environment"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                  "cuda": torch.version.cuda, "tf32": False,
                                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                  "cudnn_benchmark": torch.backends.cudnn.benchmark}
        if receipt["inputs"]["weights_sha256"] != OFFICIAL_SHA256:
            raise ValueError("Real parity inputs must use the locked official checkpoint")
        provider = OnlineCOCOProvider(args.images_list, args.annotations, args.weights, args.vendor, args.device,
                                      run_dir=run, max_instances=12, require_declared_subset=False)
        models = initial = None
        initial_optimizer = None
        gate_values = {name: [] for name in CHECK_NAMES}
        total_images = min(args.max_images, len(provider.ids))
        for image_index in range(total_images):
            payload = provider.get(image_index)
            prepared = prepare_image(payload, args.device)
            rows = payload["training_targets"]
            if not rows:
                receipt.setdefault("images_without_usable_instances", []).append(payload["image_id"])
            if rows and models is None:
                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                initialization_rng = capture_rng(torch, np)
                dimensions = provider.dimensions
                model_args = (dimensions["feature_channels"], dimensions["instance_hidden_channels"], TriFlowConfig())
                base = TriFlowModel(*model_args).float().to(args.device)
                base_rng = capture_rng(torch, np)
                restore_rng(initialization_rng, torch, np)
                minimal = install(TriFlowModel, "minimal")(*model_args).float().to(args.device)
                constructor_rng = ExactComparator(torch, np)
                constructor_exact = constructor_rng.compare(base_rng, capture_rng(torch, np), "constructor_rng")
                constructor_state = ExactComparator(torch, np)
                constructor_exact &= constructor_state.compare(base.state_dict(), minimal.state_dict(), "constructor_state")
                restore_rng(initialization_rng, torch, np)
                full = TriFlowRuntimeModel(*model_args, diagnostic_scope="full").float().to(args.device)
                constructor_exact &= constructor_state.compare(base.state_dict(), full.state_dict(), "full_constructor_state")
                initial = cpu_tree(base.state_dict(), torch)
                if args.head is not None:
                    actual = torch.load(args.head.resolve(), map_location="cpu", weights_only=False)
                    loaded = actual.get("head_state_dict") if isinstance(actual, dict) else None
                    if not isinstance(loaded, dict) or not finite_tree(loaded, torch) or any(value.dtype != torch.float32 for value in loaded.values()):
                        raise ValueError("Optional actual head lacks finite head_state_dict")
                    base.load_state_dict(loaded, strict=True)
                    initial = cpu_tree(base.state_dict(), torch)
                    initial_optimizer = actual.get("optimizer_state_dict")
                    if initial_optimizer is not None:
                        if not isinstance(initial_optimizer, dict) or not initial_optimizer.get("state") or not finite_tree(initial_optimizer, torch):
                            raise ValueError("Optional actual snapshot optimizer state is nonfinite or empty")
                        initial_optimizer = cpu_tree(initial_optimizer, torch)
                    receipt["optimizer_start_state_source"] = "actual supplied snapshot" if initial_optimizer is not None else "new AdamW (supplied module-only head has no optimizer)"
                    del actual, loaded
                else:
                    receipt["optimizer_start_state_source"] = "new AdamW, actual seed0 module initialization"
                models = (base, minimal, full)
                receipt["constructor_exact"] = bool(constructor_exact)
                receipt["configuration"] = base.configuration()
                receipt["configuration_unchanged"] = base.configuration() == minimal.configuration() == full.configuration()
                receipt["constructor_detail"] = {"state": constructor_state.report(), "rng": constructor_rng.report()}
            for start in range(0, len(rows), args.instance_chunk):
                chunk = rows[start:start + args.instance_chunk]
                inputs, targets = collate_instances(prepared, chunk)
                rng = capture_rng(torch, np)
                base, minimal, full = models
                reference = run_step(base, initial, rng, inputs, targets, torch, np, initial_optimizer)
                repeated_reference = run_step(base, initial, rng, inputs, targets, torch, np, initial_optimizer)
                candidate = run_step(minimal, initial, rng, inputs, targets, torch, np, initial_optimizer)
                complete = run_step(full, initial, rng, inputs, targets, torch, np, initial_optimizer)
                comparators = {}
                checks = {}
                repeatability = ExactComparator(torch, np)
                reference_reproducible = repeatability.compare(
                    {key: reference[key] for key in ("outputs", "loss", "gradients", "head", "optimizer", "rng")},
                    {key: repeated_reference[key] for key in ("outputs", "loss", "gradients", "head", "optimizer", "rng")}, "original_vs_original")
                for label, key in (("coefficients_exact", "coefficients_refined"), ("logits_exact", "logits_refined")):
                    comparator = ExactComparator(torch, np)
                    checks[label] = comparator.compare(reference["outputs"][key], candidate["outputs"][key], key)
                    comparators[label] = comparator.report()
                for label, key in (("loss_terms_exact", "loss"), ("all_parameter_gradients_exact", "gradients"),
                                   ("adamw_head_exact", "head"), ("adamw_optimizer_exact", "optimizer"), ("rng_state_exact", "rng")):
                    comparator = ExactComparator(torch, np)
                    checks[label] = comparator.compare(reference[key], candidate[key], key)
                    if label == "all_parameter_gradients_exact":
                        checks[label] &= comparator.compare(reference["clip_grad_norm"], candidate["clip_grad_norm"], "clip_grad_norm")
                    comparators[label] = comparator.report()
                full_comparator = ExactComparator(torch, np)
                checks["full_scope_original_exact"] = full_comparator.compare(
                    {key: reference[key] for key in ("outputs", "loss", "gradients", "head", "optimizer", "rng", "diagnostics")},
                    {key: complete[key] for key in ("outputs", "loss", "gradients", "head", "optimizer", "rng", "diagnostics")}, "full_scope")
                all_outputs = ExactComparator(torch, np)
                checks["coefficients_exact"] &= all_outputs.compare(reference["outputs"], candidate["outputs"], "all_learned_outputs")
                shared_diagnostics = ExactComparator(torch, np)
                checks["minimal_diagnostic_keys_exact"] = (set(candidate["diagnostic_keys"]) == set(MINIMAL_KEYS)
                    and candidate["diagnostic_scope"] == "minimal"
                    and set(candidate["diagnostics_unmeasured"]) == set(FULL_DIAGNOSTIC_KEYS) - set(MINIMAL_KEYS)
                    and shared_diagnostics.compare({key: reference["diagnostics"][key] for key in MINIMAL_KEYS}, candidate["diagnostics"], "required_diagnostics"))
                profiles = {"original_full": profile_forward(base, initial, rng, inputs, torch, np),
                            "minimal": profile_forward(minimal, initial, rng, inputs, torch, np)}
                checks["minimal_expensive_diagnostics_absent"] = (profiles["original_full"]["eigvalsh_calls"] > 0
                    and profiles["minimal"]["eigvalsh_calls"] == 0 and profiles["minimal"]["eigh_backend_calls"] == 0)
                expected_solves = 1 + TriFlowConfig().compiler_passes
                checks["checked_cholesky_preserved"] = all(profile["cholesky_ex_calls"] == expected_solves
                    and profile["cholesky_solve_calls"] == expected_solves for profile in profiles.values())
                checks["original_source_unmodified"] = file_sha(Path(__file__).with_name("triflow_model.py")) == ORIGINAL_MODEL_SOURCE_SHA256
                checks["meaningful_real_inputs"] = (payload["base_weights_sha256"] == OFFICIAL_SHA256
                    and bool(chunk) and any(value is not None and bool(value.abs().max() > 0) for value in reference["gradients"].values())
                    and any(not torch.equal(initial[key], reference["head"][key]) for key in initial))
                for name, value in checks.items():
                    gate_values[name].append(bool(value))
                case = {"image_id": payload["image_id"], "input_sha256": payload["input_sha256"], "image_sha256": payload["image_sha256"],
                        "output_rows": [row["output_row"] for row in chunk], "instances": len(chunk), "checks": checks,
                        "comparisons": comparators | {"full_scope": full_comparator.report(), "all_outputs": all_outputs.report(), "required_diagnostics": shared_diagnostics.report()},
                        "parameter_names": sorted(reference["gradients"]), "parameters_with_gradient": sorted(key for key, value in reference["gradients"].items() if value is not None),
                        "original_vs_original_exact": bool(reference_reproducible), "original_repeatability": repeatability.report(),
                        "profile": profiles, "status": "passed" if all(checks.values()) else "failed"}
                receipt["cases"].append(case)
                receipt["chunk_count"] += 1
                if receipt["chunk_count"] == 1:
                    # Timings are recorded separately and never gate parity.
                    measured = {"original_full": timing(base, initial, rng, inputs, targets, args.timing_repeats, torch, np, initial_optimizer),
                                "minimal": timing(minimal, initial, rng, inputs, targets, args.timing_repeats, torch, np, initial_optimizer)}
                    receipt["timings"].append(measured)
                print(f"PARITY real_image={payload['image_id']} chunk={start // 4} checks={all(checks.values())}", flush=True)
                del reference, repeated_reference, candidate, complete, inputs, targets
            del prepared, payload
        if receipt["chunk_count"] == 0:
            raise ValueError("No actual selected training instance in the requested real-image engineering scope")
        receipt["checks"] = {name: bool(values) and all(values) for name, values in gate_values.items()}
        receipt["original_source_after_sha256"] = {name: file_sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
        receipt["all_sources_unmodified"] = receipt["original_source_after_sha256"] == source_sha
        receipt["frozen_integrity"] = provider.verify_frozen()
        receipt["stream_receipt"] = provider.receipt()
        receipt["original_vs_original_exact"] = all(case["original_vs_original_exact"] for case in receipt["cases"])
        receipt["passed"] = (all(receipt["checks"].values()) and receipt["constructor_exact"] and receipt["configuration_unchanged"]
                             and receipt["all_sources_unmodified"] and receipt["original_vs_original_exact"] and receipt["frozen_integrity"].get("passed") is True)
        receipt["status"] = "passed" if receipt["passed"] else "failed"
        receipt["completed_at"] = now()
        write_json(receipt_file, receipt)
        print(f"DIAGNOSTICS_PARITY {receipt_file} passed={receipt['passed']}", flush=True)
        return 0 if receipt["passed"] else 1
    except Exception as exc:
        receipt.update(status="failed", passed=False, completed_at=now(), error=repr(exc), traceback=traceback.format_exc())
        write_json(receipt_file, receipt)
        write_json(run / "PARITY_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
        raise
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
