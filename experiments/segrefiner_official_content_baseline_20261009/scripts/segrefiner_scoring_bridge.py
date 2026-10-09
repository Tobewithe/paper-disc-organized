"""Three real source paths, with explicit validation and accepted CPU kernels.

No producer metadata is renamed into a fictitious seven-arm inference Run.
No torch/model import. Scope substitutions affect isolated loaded module objects.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ARMS = ("baseline", "RCMC_first64", "SegRefiner_LR_first64_batch8")
SEG_RUN_ID = "RUN_SEGREFINER_BATCH8_FIXED5K_INFERENCE_S0_02"
PROTOCOL_SHA = "f4703167b0fbb77fc324f89aa5d7d5b97ad6afe020371dc2ab3d2a2db11c1fde"
SUPPLEMENT_SHA = "4d4918a7a2844bd4bbb637915f9173dd9c163af01894a3a8267a41604ebf876a"
ADAPTATION_SHA = "59ce9e80d87947f63741f872d6421fe80ffc3884fa53267d0a6ecd2744fc7a1d"
ANN_SHA = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
BASE_PRED_SHA = "6c757a3a3ab302b807de7b77646cd7acc7d84afa3b2360e50b070e32d80d8f43"
CORES = {"comparison_io.py": "000ea7e883307ed1383bf005f543f099897ed2b523f8ad3960645049e594cd34",
         "score_and_verify_comparison.py": "d0c0a5e4b4f3ae5236a295cf5255dd93e221278dc391c04c11dadce6b2e5eb8c",
         "multiview_readout.py": "1076fc8da0de20b20d84b6aee2d871b17e69a39a7cebe41264e00f47f8242172",
         "verify_multiview_readout.py": "a82993ef7d17282b185b3d82b179afc124db78a979a3d85bef6fd55c20641184"}
PRODUCER_SOURCES = {"run_segrefiner_inference.py": "7f550762d2285453098e2d99b9b3211cdc0e2f9c6cb7e233307d85445d855445",
    "segrefiner_runtime.py": "5a296b083991adb6918db09a9f69204a0a2e4f65b582d5c9a2f2335d4183c107",
    "run_segrefiner_batch8.py": "c6a799f67b991a0aa6c263d52239093b3462187f7f10718157bc75a141416528",
    "segrefiner_batch8_runtime.py": "274b6b3b84813b81737cdb760a89aadad11b08469397205f4da5f7513b26f737",
    "wddm_memory_probe.py": "acb110856d89842d28e2b79ffdf20d5f769b4c3372e7bcc753b110e518a8c69f"}


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_cores(directory):
    directory = Path(directory).resolve()
    for name, expected in CORES.items():
        check(sha(directory / name) == expected, "Accepted metric core changed: " + name)
    sys.path.insert(0, str(directory))
    def load(name, alias):
        spec = importlib.util.spec_from_file_location(alias, directory / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    io = load("comparison_io.py", "seg_bridge_io")
    score = load("score_and_verify_comparison.py", "seg_bridge_score")
    multi = load("multiview_readout.py", "seg_bridge_multiview")
    score.ARMS, multi.ARMS = ARMS, ARMS
    return io, score, multi


def load_pixel_checker(directory):
    path = Path(directory).resolve() / "verify_multiview_readout.py"
    check(sha(path) == CORES[path.name], "Accepted pixel reconstruction core changed")
    spec = importlib.util.spec_from_file_location("seg_bridge_saved_pixel_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS = ARMS
    return module


def validate_runner_record(record):
    check(record.get("source_kind") == "runner_observed" and record.get("status") == "completed"
          and record.get("return_code") == 0 and record.get("artifact_completeness") == "complete"
          and record.get("finished_at") is not None, "Actual completed runner_observed process required")


def runner_gate(root, required, lock, fresh_inputs=False):
    root = Path(root).resolve()
    record = read(root / "run.json")
    lock.add(root / "run.json")
    validate_runner_record(record)
    original = Path(record["locations"][0]["path"])
    artifacts = {}
    for row in record["artifacts"]:
        try:
            relative = Path(row["path"]).relative_to(original).as_posix()
        except ValueError:
            continue
        artifacts[relative] = row
    for name in required:
        check(name in artifacts and artifacts[name]["exists"] is True
              and lock.add(root / name) == artifacts[name]["sha256"], "Runner final artifact SHA missing/different: " + name)
    if fresh_inputs:
        for row in record["inputs"]:
            check(row["exists"] is True and lock.add(row["path"]) == row["sha256"], "Producer actual input bytes changed")
        for row in record["snapshots"]:
            check(lock.add(root / row["snapshot"]) == row["revision"] and lock.add(row["path"]) == row["revision"], "Producer executed source snapshot changed")
    return record


def seal(root, names):
    root = Path(root).resolve()
    result = {}
    for name in names:
        p = (root / name).resolve(strict=True)
        p.relative_to(root)
        result[name] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    return result


def verify_seal(root, artifacts, lock):
    root = Path(root).resolve()
    for name, row in artifacts.items():
        p = (root / name).resolve(strict=True)
        p.relative_to(root)
        check(p.stat().st_size == row["size_bytes"] and lock.add(p) == row["sha256"], "Original output seal differs: " + name)


def score_gate(root, complete_name, lock):
    root = Path(root).resolve()
    runner_gate(root, ["SUMMARY.json", complete_name, "SOURCE_LOCK.json"], lock)
    summary, complete = read(root / "SUMMARY.json"), read(root / complete_name)
    check(complete["status"] == "completed" and complete["summary_sha256"] == sha(root / "SUMMARY.json")
          and summary["source_lock_sha256"] == sha(root / "SOURCE_LOCK.json"), "Scoring/readout terminal binding differs")
    verify_seal(root, complete["artifacts"], lock)
    for name, entry in read(root / "SOURCE_LOCK.json")["files"].items():
        check(entry["sha256_before"] == entry["sha256_after"], "Input changed during original scoring")
    return summary, complete


def validate_sources(plan, io, lock, engineering=False):
    study, native, seg = [Path(plan[key]).resolve() for key in ("study", "native_run", "seg_run")]
    check(engineering or seg.name == SEG_RUN_ID, "Finite formal continuation must bind literal approved inference02")
    runner_gate(seg, ["SUMMARY.json", "INFERENCE_COMPLETE.json"], lock, fresh_inputs=True)
    runner_gate(native, ["SUMMARY.json", "BASELINE_PARITY.json"], lock)
    for name, expected in (("PROTOCOL.md", PROTOCOL_SHA), ("EXECUTION_SUPPLEMENT.md", SUPPLEMENT_SHA), ("EXECUTION_BATCH8_ADAPTATION.md", ADAPTATION_SHA)):
        check(lock.add(study / name) == expected, "SegRefiner frozen protocol changed: " + name)
    inputs, source = read(seg / "INFERENCE_INPUTS.json"), read(seg / "SUMMARY.json")
    cfg = inputs["configuration"]
    check(source["passed"] is True and source["status"] == "prediction_complete" and source["engineering"] is engineering
          and source["gt_parsed"] is False and source["new_yolo_forward"] is False, "SegRefiner actual inference scope differs")
    check(source["hardware_batch_max"] == 8 and source["official_default_batch_max"] == 32
          and cfg["hardware_batch_max"] == 8 and cfg["base_seed"] == 20261009 and cfg["gt_used"] is False
          and cfg["new_yolo_forward"] is False and cfg["hardware_adaptation_sha256"] == ADAPTATION_SHA,
          "Fixed batch8/RNG/GT-free adaptation differs")
    check(source["source_sha256"] == PRODUCER_SOURCES and cfg["source_sha256"] == PRODUCER_SOURCES,
          "Locked actual SegRefiner producer source identity differs")
    for name, expected in PRODUCER_SOURCES.items():
        check(lock.add(seg / "source" / name) == expected and lock.add(study / "scripts" / name) == expected,
              "Actual producer source/archive differs: " + name)
    check(source["input_configuration_sha256"] == canonical(cfg)
          and read(seg / "INFERENCE_COMPLETE.json")["summary_sha256"] == sha(seg / "SUMMARY.json"), "Seg terminal/configuration binding differs")
    original = read(native / "baseline" / "COMPLETE.json")["fingerprint"]["image_ids"]
    ids = cfg["image_ids"]
    check(cfg["original_source_image_ids"] == original and ids == (original[:32] if engineering else original)
          and source["image_count"] == len(ids) and len(ids) == (32 if engineering else 5000), "Fixed original image order/scope differs")
    native_summary = read(native / "SUMMARY.json")
    check(native_summary["passed"] is True and native_summary["baseline_parity_passed"] is True
          and native_summary["image_count"] == 5000 and cfg["native_summary_sha256"] == lock.add(native / "SUMMARY.json")
          and cfg["native_decisions_sha256"] == lock.add(native / "NATIVE_DECISIONS.jsonl"), "Original native source binding differs")
    meta = read(plan["image_meta"])
    check(meta["gt_read"] is False and meta["image_count"] == 5000 and cfg["image_meta_sha256"] == lock.add(plan["image_meta"])
          and cfg["images_list_sha256"] == meta["bindings"]["local_images_list_sha256"], "Standalone RGB/source identity binding differs")
    check(source["image_diagnostics_sha256"] == lock.add(seg / "IMAGE_DIAGNOSTICS.jsonl"), "Per-image RNG/endpoint diagnostics not bound")
    arm_roots = {"baseline": native / "baseline", "RCMC_first64": native / "RCMC_first64", ARMS[2]: seg / ARMS[2]}
    for arm, path in arm_roots.items():
        receipt = read(path / "COMPLETE.json")
        lock.add(path / "COMPLETE.json")
        digest = lock.add(path / "predictions.json")
        check(receipt["status"] == "prediction_complete" and receipt["predictions_sha256"] == digest,
              "Actual complete lossless-RLE arm source differs: " + arm)
        if arm == "baseline":
            check(digest == BASE_PRED_SHA, "Accepted original full native baseline bytes changed")
        elif arm == ARMS[2]:
            check(receipt["fingerprint"] == cfg and source["predictions_sha256"] == digest
                  and receipt["image_count"] == len(ids) and receipt["gt_used"] is False
                  and receipt["all_class_box_score_order_exact"] is True and receipt["empty_ordinal_preserved"] is True,
                  "Refiner ordinal/configuration/GT-free receipt differs")
        else:
            check(native_summary["prediction_receipts"][arm]["predictions_sha256"] == digest, "Frozen RCMC source differs")
    return ids, arm_roots, {"sources": {arm: str(path) for arm, path in arm_roots.items()},
        "segrefiner_run": str(seg), "segrefiner_input_configuration_sha256": source["input_configuration_sha256"],
        "segrefiner_source_sha256": source["source_sha256"], "native_run": str(native), "source_scope_substitution": list(ARMS),
        "reused_metric_cores": CORES, "hardware_batch_max": 8, "official_default_batch_max": 32,
        "training_manifest_verified": False, "confidence_intervals": None,
        "reuse_is_independent_reimplementation": False, "GT_rematched": False}


class ScopeReader:
    """Keep original source receipt scope; read an explicitly declared prefix."""
    def __init__(self, io, root, lock, source_ids, selected_ids):
        self.reader = io.ArmReader(root, lock, source_ids)
        self.selected = selected_ids
        self.original = source_ids
        self.predictions_sha256 = self.reader.predictions_sha256
        self.receipt, self.mode = self.reader.receipt, self.reader.mode

    def read(self, iid, count):
        return self.reader.read(iid, count)

    def finish(self, image_count):
        check(self.reader.images == image_count == len(self.selected), "Selected scope image count differs")
        if self.original == self.selected:
            self.reader.finish(image_count)
        else:
            check(self.selected == self.original[:image_count], "Engineering must use exact original prefix")
            following = next(self.reader.stream, None)
            check(following is None or following["image_id"] in set(self.original[image_count:]), "Prefix reader did not stop at image boundary")
