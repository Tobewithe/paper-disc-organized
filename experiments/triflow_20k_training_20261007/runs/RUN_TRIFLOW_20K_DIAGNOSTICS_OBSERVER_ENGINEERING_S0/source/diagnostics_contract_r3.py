"""Tensor-free provenance for the explicitly declared R3 diagnostics runtime."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

VERSION = "triflow_20k_diagnostics_execution_v1"
REQUIRED_DIAGNOSTICS = ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count", "trust_saturated")
ORIGINAL_MODEL_SHA256 = "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"
ORIGINAL_SOURCES = {
    "train_20k.py": "944dcb27d75f1d1d9ce8cc950c206c607b73ba6023a7a5f19517134bbd2a86bb",
    "stream_data.py": "6460a9277b47a8ca790e03213afbaa5d628592b9319608e0be70dd4799091fb6",
    "frozen_io.py": "cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293",
    "triflow_model.py": ORIGINAL_MODEL_SHA256,
}
PARITY_CHECKS = ("coefficients_exact", "logits_exact", "loss_terms_exact", "all_parameter_gradients_exact", "adamw_head_exact", "adamw_optimizer_exact", "rng_state_exact", "full_scope_original_exact", "minimal_diagnostic_keys_exact", "minimal_expensive_diagnostics_absent", "checked_cholesky_preserved", "original_source_unmodified", "meaningful_real_inputs")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_parity(receipt, runtime_sha):
    if receipt.get("passed") is not True or receipt.get("status") != "passed":
        raise ValueError("An actual passed diagnostics-only parity receipt is required")
    if receipt.get("runtime_sha256") != runtime_sha:
        raise ValueError("Diagnostics parity receipt does not bind the executed runtime bytes")
    if receipt.get("smoke_only") is not True:
        raise ValueError("Parity receipt must explicitly retain its engineering evidence scope")
    if receipt.get("scientific_scale_claimed") is not False or receipt.get("original_model_source_sha256") != ORIGINAL_MODEL_SHA256:
        raise ValueError("Parity receipt changed original method source or inflated its evidence scope")
    checks = receipt.get("checks", {})
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in PARITY_CHECKS):
        raise ValueError("Required actual diagnostics-only forward/loss/gradient/AdamW/RNG parity evidence is missing")
    if type(receipt.get("chunk_count")) is not int or receipt["chunk_count"] < 1:
        raise ValueError("Actual positive-count parity chunks are required")
    if not isinstance(receipt.get("verifier_sha256"), str) or len(receipt["verifier_sha256"]) != 64:
        raise ValueError("Parity verifier source fingerprint is required")
    if receipt.get("inputs", {}).get("source_sha256") != ORIGINAL_SOURCES:
        raise ValueError("Parity did not use exactly the original four scientific source versions")
    if any(receipt.get(key) is not True for key in ("constructor_exact", "configuration_unchanged", "all_sources_unmodified", "original_vs_original_exact")) or receipt.get("frozen_integrity", {}).get("passed") is not True:
        raise ValueError("Actual parity self-repeatability/constructor/configuration/source/frozen evidence is missing")
    return receipt


def archive_contract(runtime, parity, addendum, run):
    parity_path = Path(parity).resolve()
    paths = {"runtime": Path(runtime).resolve(), "parity": parity_path, "addendum": Path(addendum).resolve(), "adapter": Path(__file__).resolve(), "verifier": parity_path.parent/"source"/"verify_diagnostics_runtime.py"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if paths["runtime"].name != "diagnostics_runtime.py":
        raise ValueError("The declared diagnostics runtime must retain its actual source filename")
    digests = {key: file_sha(path) for key, path in paths.items()}
    if file_sha(parity_path.parent/"source"/"diagnostics_runtime.py") != digests["runtime"]:
        raise ValueError("Executed runtime does not match the actual parity Run runtime archive")
    actual_parity = validate_parity(json.loads(paths["parity"].read_text(encoding="utf-8")), digests["runtime"])
    if actual_parity["verifier_sha256"] != digests["verifier"]:
        raise ValueError("Actual parity verifier source archive differs from the receipt")
    archive = Path(run)/"diagnostics_provenance"
    archive.mkdir(parents=True, exist_ok=True)
    names = {"runtime": "diagnostics_runtime.py", "parity": "DIAGNOSTICS_PARITY.json", "addendum": "DIAGNOSTICS_EXECUTION_ADDENDUM.md", "adapter": "diagnostics_contract_r3.py", "verifier": "verify_diagnostics_runtime.py"}
    for key, name in names.items():
        target = archive/name
        if target.exists() and file_sha(target) != digests[key]:
            raise ValueError("R3 diagnostics archive differs from actual execution input: " + key)
        if not target.exists():
            shutil.copy2(paths[key], target)
    result = {"version": VERSION, "actual_runtime_changed": True, "original_scientific_contract_preserved": True,
              "parity_evidence_scope": "engineering diagnostics-only numerical parity; no COCO efficacy claim",
              "required_diagnostics": list(REQUIRED_DIAGNOSTICS), "training_default_scope": "minimal", "training_first_task_probe_scope": "full"}
    for key, name in names.items():
        result[key+"_source_file"] = "diagnostics_provenance/"+name
        result[key+"_sha256"] = digests[key]
    return result


def verify_contract(run, contract):
    if contract.get("version") != VERSION or contract.get("actual_runtime_changed") is not True or contract.get("original_scientific_contract_preserved") is not True:
        raise ValueError("Declared R3 execution/source distinction is missing")
    if contract.get("required_diagnostics") != list(REQUIRED_DIAGNOSTICS) or contract.get("training_default_scope") != "minimal" or contract.get("training_first_task_probe_scope") != "full":
        raise ValueError("R3 diagnostic mode contract differs")
    names = {"runtime": "diagnostics_runtime.py", "parity": "DIAGNOSTICS_PARITY.json", "addendum": "DIAGNOSTICS_EXECUTION_ADDENDUM.md", "adapter": "diagnostics_contract_r3.py", "verifier": "verify_diagnostics_runtime.py"}
    for key, name in names.items():
        relative = "diagnostics_provenance/"+name
        if contract.get(key+"_source_file") != relative or file_sha(Path(run)/relative) != contract.get(key+"_sha256"):
            raise ValueError("Actual R3 diagnostics archive/hash differs: " + key)
    receipt = validate_parity(json.loads((Path(run)/contract["parity_source_file"]).read_text(encoding="utf-8")), contract["runtime_sha256"])
    if receipt["verifier_sha256"] != contract["verifier_sha256"]:
        raise ValueError("Archived parity verifier/source fingerprint differs")
    return receipt


def import_runtime(path):
    path = Path(path).resolve()
    name = "triflow_declared_diagnostics_runtime"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError("Another diagnostics runtime is already loaded")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
