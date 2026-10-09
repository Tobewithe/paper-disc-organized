"""B's fixed four-draw, per-arm AP wiring audit of sealed production outputs.

No fitting, GT decoding/matching, model execution, remote access, or new draws.
The accepted reaccumulate core is reused; each arm expands copies and sorts its
own scores. This independently checks the shared-sort wiring, not all 1000 APs.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PureWindowsPath
import platform
import shutil
import sys
import traceback

import numpy as np

STUDY_ID = "STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009"
INDICES = (0, 1, 499, 999)
TOLERANCE = 1e-14
FROZEN = {
    "bootstrap_target_ap.py": "05978268aae334298212bd27559e41e249d62bbac94dde9f982b4ff368cbd4d9",
    "target_scoring_common.py": "a4441164d589e7c929196ddfb3612cb9bf0c2c04d1c296514d1d8a75de8ac835",
    "evaluate_target_comparison.py": "54e278c9450aad1111a4864415255055b5522e8bfca3fc48af53ca5e933f316d",
    "verify_target_comparison.py": "a69927ce7db1e1b61b13fd30976cb861ffdcc7dc6d5ca8234c55d6b6c08f58a3",
}
# Frozen imports contain only NumPy/stdlib at module scope. Check before import.
for _name in ("target_scoring_common.py", "bootstrap_target_ap.py"):
    if hashlib.sha256(Path(__file__).with_name(_name).read_bytes()).hexdigest() != FROZEN[_name]:
        raise ValueError("Accepted accumulation dependency changed: " + _name)

from target_scoring_common import (ARMS, PROTOCOL_SHA, InputLock, check,
    completed_receipt, load_npz, read_json, seal_outputs, sha,
    verify_output_seal, write_json)
from bootstrap_target_ap import SEED, REPLICATES, cache_contract, paired_aps, reaccumulate


def within(root, relative):
    """Only files inside a producer/consumer Run; reject absolute/escaping paths."""
    check(not PureWindowsPath(relative).is_absolute(), "Absolute artifact path")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    check(path.is_file(), "Artifact is not a file: " + str(path))
    return path


def runner_receipt(root, required, script, lock):
    lock.add(root / "run.json")
    record = read_json(root / "run.json")
    check(record.get("source_kind") == "runner_observed"
          and record.get("study_id") == STUDY_ID and record.get("run_id") == root.name,
          "Run identity/source differs: " + root.name)
    check(record.get("status") == "completed" and record.get("return_code") == 0
          and record.get("artifact_completeness") == "complete" and record.get("finished_at"),
          "Actual runner completion is required: " + root.name)
    check(script in [PureWindowsPath(v).name for v in record["command"]], "Actual entry script differs")
    original = PureWindowsPath(record["locations"][0]["path"])
    artifacts = {}
    for entry in record.get("artifacts", []):
        try:
            relative = PureWindowsPath(entry["path"]).relative_to(original).as_posix()
        except ValueError:
            continue
        check(relative not in artifacts, "Duplicate runner artifact")
        artifacts[relative] = entry
        check(entry.get("exists") is True and lock.add(within(root, relative)) == entry["sha256"],
              "Runner terminal artifact bytes differ: " + relative)
    check(set(required).issubset(artifacts), "Runner did not seal required terminal files")
    snapshots = {}
    for entry in record.get("snapshots", []):
        name = PureWindowsPath(entry["path"]).name
        check(lock.add(within(root, entry["snapshot"])) == entry["revision"], "Runner source snapshot differs")
        if name in FROZEN:
            check(entry["revision"] == FROZEN[name], "Frozen producer source changed: " + name)
        snapshots[name] = entry["revision"]
    check(snapshots.get(script) == FROZEN[script], "Actual entry source snapshot missing/different")
    # For relocated Runs the full received manifest must be present. Local
    # engineering also has one; do not reinterpret its outbound transfer.json.
    manifest = root / "manifest.sha256"
    check(manifest.is_file() or original == PureWindowsPath(str(root)), "Relocated Run lacks manifest")
    if manifest.is_file():
        lock.add(manifest)
        seen = set()
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            digest, relative = line.split("  ", 1)
            check(relative not in seen, "Duplicate manifest member")
            seen.add(relative)
            check(lock.add(within(root, relative)) == digest, "Received manifest differs: " + relative)
        check({"run.json", *required}.issubset(seen), "Manifest omits terminal receipt")
    return record


def source_entries(source, root, lock, shared):
    """Check producer attestations; do not re-read original JPEG/GT/remote files."""
    check(isinstance(source.get("files"), dict) and source["files"], "Empty source lock")
    for original, entry in source["files"].items():
        digest = entry["sha256_before"]
        check(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
              and digest == entry["sha256_after"], "Producer input/source changed during Run")
        key = str(PureWindowsPath(original)).casefold()
        signature = (digest, entry["size_bytes"])
        check(key not in shared or shared[key] == signature, "Runs disagree on shared source bytes")
        shared[key] = signature
        if entry.get("snapshot"):
            path = within(root, entry["snapshot"])
            check(path.stat().st_size == entry["size_bytes"] and lock.add(path) == digest,
                  "Producer source snapshot differs")
    for name, digest in FROZEN.items():
        matches = [v for p, v in source["files"].items() if PureWindowsPath(p).name == name]
        for entry in matches:
            check(entry["sha256_after"] == digest, "Unaccepted scientific source in producer lock")


def bind_consumed(source, producer, original_root, relatives, lock):
    """Bind the exact saved files consumed upstream to this local sealed Run."""
    check(PureWindowsPath(original_root).name == producer.name, "Consumer command names another input Run")
    entries = {str(PureWindowsPath(p)).casefold(): v for p, v in source["files"].items()}
    for relative in relatives:
        key = str(PureWindowsPath(original_root) / relative).casefold()
        check(key in entries, "Consumer source lock omitted input: " + relative)
        path = within(producer, relative)
        entry = entries[key]
        check(lock.add(path) == entry["sha256_after"] and path.stat().st_size == entry["size_bytes"],
              "Consumer was bound to different saved bytes: " + relative)


def argument(record, flag):
    command = record["command"]
    check(command.count(flag) == 1, "Expected one actual command argument: " + flag)
    return command[command.index(flag) + 1]


def valid_cache(cache, ids):
    check(cache["image_ids"].tolist() == ids and len(ids) == len(set(ids)) == 5000,
          "Cache does not contain the same complete ordered 5000 images")
    cats, scores, offsets = cache["category_ids"], cache["dt_scores"], cache["offsets"]
    check(cats.shape == (80,) and len(set(cats.tolist())) == 80, "COCO category scope differs")
    check(offsets.dtype.kind in "iu" and offsets.shape == (80, 5001)
          and offsets[0, 0] == 0 and offsets[-1, -1] == len(scores)
          and np.array_equal(offsets[1:, 0], offsets[:-1, -1]), "Cache category offsets differ")
    lengths = np.diff(offsets, axis=1)
    check(np.all((lengths >= 0) & (lengths <= 100)), "Cache violates per-image/class max100")
    check(cache["gt_nonignore"].shape == (80, 5000) and cache["gt_nonignore"].dtype.kind in "iu"
          and np.all(cache["gt_nonignore"] >= 0), "Nonignore GT denominator differs")
    check(scores.ndim == 1 and np.isfinite(scores).all(), "Invalid detection scores")
    check(cache["dt_matches"].dtype.kind == cache["dt_ignore"].dtype.kind == "b"
          and cache["dt_matches"].shape == cache["dt_ignore"].shape == (10, len(scores)), "Invalid match/ignore cache")
    check(cache["iou_thresholds"].shape == (10,) and cache["recall_thresholds"].shape == (101,)
          and np.allclose(cache["iou_thresholds"], np.linspace(.5, .95, 10), rtol=0, atol=TOLERANCE)
          and np.allclose(cache["recall_thresholds"], np.linspace(0, 1, 101), rtol=0, atol=TOLERANCE),
          "Ordinary COCO AP grid differs")


def verify(args, out):
    lock, shared = InputLock(), {}
    score, boot, verify_run, engineering = [getattr(args, k).resolve(strict=True) for k in
        ("scoring_run", "bootstrap_run", "verification_run", "engineering_run")]
    protocol = args.protocol.resolve(strict=True)
    check(lock.add(protocol) == PROTOCOL_SHA, "Fixed Study protocol differs")
    source_dir = out / "source"
    source_dir.mkdir()
    snapshots = []
    for path in (Path(__file__).resolve(), protocol, *(Path(__file__).with_name(n) for n in FROZEN)):
        digest = lock.add(path)
        if path.name in FROZEN:
            check(digest == FROZEN[path.name], "Frozen local scientific source differs")
        target = source_dir / path.name
        shutil.copy2(path, target)
        relative = target.relative_to(out).as_posix()
        lock.files[str(path.resolve())]["snapshot"] = relative
        check(sha(target) == digest, "Own source snapshot differs")
        snapshots.append(relative)

    records, summaries, sources = {}, {}, {}
    for label, root, status, script in (
        ("score", score, "scoring_complete", "evaluate_target_comparison.py"),
        ("boot", boot, "bootstrap_complete", "bootstrap_target_ap.py"),
        ("engineering", engineering, "engineering_complete", "bootstrap_target_ap.py"),
    ):
        records[label] = runner_receipt(root, ("SUMMARY.json", "COMPLETE.json", "SOURCE_LOCK.json"), script, lock)
        summaries[label], _, sources[label] = completed_receipt(root, status, lock)
        source_entries(sources[label], root, lock, shared)
    records["verify"] = runner_receipt(verify_run, ("VERIFICATION.json", "COMPLETE.json", "SOURCE_LOCK.json"),
                                       "verify_target_comparison.py", lock)
    verification = read_json(verify_run / "VERIFICATION.json")
    completion = read_json(verify_run / "COMPLETE.json")
    sources["verify"] = read_json(verify_run / "SOURCE_LOCK.json")
    check(verification.get("status") == "passed" and verification.get("passed") is True
          and completion.get("status") == "verification_complete"
          and completion.get("verification_sha256") == lock.add(verify_run / "VERIFICATION.json")
          and verification.get("source_lock_sha256") == lock.add(verify_run / "SOURCE_LOCK.json"),
          "Actual production verification completion binding differs")
    source_entries(sources["verify"], verify_run, lock, shared)
    scored, bootstrap, engineer = [summaries[k] for k in ("score", "boot", "engineering")]
    check(scored["engineering"] is False and scored["image_count"] == 5000 and scored["arms"] == list(ARMS),
          "Formal three-arm 5000-image scoring required")
    check(bootstrap["seed"] == SEED == 20261009 and bootstrap["replicates"] == REPLICATES == 1000
          and bootstrap["draw_size"] == 5000 and bootstrap["ordered_independent_image_copies"] is True,
          "Fixed bootstrap scope differs")
    check(bootstrap["scoring_summary_sha256"] == lock.add(score / "SUMMARY.json")
          and verification["scoring_summary_sha256"] == lock.add(score / "SUMMARY.json")
          and verification["bootstrap_summary_sha256"] == lock.add(boot / "SUMMARY.json"), "Cross-Run summaries differ")
    check(engineer["checks"]["synthetic"]["passed"] is True and engineer["checks"]["real_fixed_first4"]["passed"] is True
          and engineer["provenance_negative_guards_passed"] is True
          and lock.add(engineering / "SUMMARY.json") == "1a94ebf5795cf65c6542031b3d3123db78b77219e9cb9361a16246cc6842c75f",
          "Previously accepted actual official-copy engineering required")
    for name in ("bootstrap_target_ap.py", "target_scoring_common.py"):
        check(sources["engineering"]["executed_sources"][name]["sha256"] == FROZEN[name], "Engineering core differs")
    terminal_files = ["run.json", "SUMMARY.json", "COMPLETE.json", "SOURCE_LOCK.json"]
    cache_files = [f"{a}/COCO_BOOTSTRAP_CACHE.npz" for a in ARMS]
    raw_boot_files = ["BOOTSTRAP_DRAWS.npz", "BOOTSTRAP_REPLICATES.npz"]
    score_files = terminal_files + cache_files
    boot_files = terminal_files + raw_boot_files
    bind_consumed(sources["boot"], score, argument(records["boot"], "--scoring-run"), score_files, lock)
    bind_consumed(sources["verify"], score, argument(records["verify"], "--scoring-run"), score_files, lock)
    bind_consumed(sources["verify"], boot, argument(records["verify"], "--bootstrap-run"), boot_files, lock)
    bind_consumed(sources["boot"], engineering, argument(records["boot"], "--engineering-run"),
                  terminal_files, lock)
    verify_output_seal(score, scored["artifacts"], cache_files, lock)
    verify_output_seal(boot, bootstrap["artifacts"], raw_boot_files, lock)
    check(lock.add(score / "source_provenance/PROTOCOL.md") == PROTOCOL_SHA, "Scored protocol snapshot differs")
    protocol_entries = [v for p, v in sources["score"]["files"].items() if PureWindowsPath(p).name == "PROTOCOL.md"]
    check(protocol_entries and all(v["sha256_after"] == PROTOCOL_SHA for v in protocol_entries), "Original scored protocol binding differs")
    for name, digest in FROZEN.items():
        check(lock.add(score / "source_provenance" / name) == digest, "Scored scientific source copy differs")

    caches = [load_npz(score / arm / "COCO_BOOTSTRAP_CACHE.npz") for arm in ARMS]
    for cache in caches:
        valid_cache(cache, scored["image_ids"])
    cache_contract(caches)
    check(lock.add(boot / "BOOTSTRAP_DRAWS.npz") == bootstrap["draws_sha256"]
          and lock.add(boot / "BOOTSTRAP_REPLICATES.npz") == bootstrap["replicates_sha256"], "Raw bootstrap seals differ")
    draw_data, reps = [load_npz(boot / name) for name in ("BOOTSTRAP_DRAWS.npz", "BOOTSTRAP_REPLICATES.npz")]
    draws = draw_data["draws"]
    check(draws.shape == (1000, 5000) and draws.dtype.kind in "iu" and np.all((draws >= 0) & (draws < 5000))
          and draw_data["image_ids"].tolist() == scored["image_ids"] and draw_data["seed"].shape == ()
          and int(draw_data["seed"]) == SEED, "Saved ordered draw identities/scope/seed differ")
    check(reps["AP"].shape == (1000, 3) and np.isfinite(reps["AP"]).all()
          and np.all((reps["AP"] >= 0) & (reps["AP"] <= 1))
          and reps["AP_H_minus_I"].shape == (1000,)
          and np.array_equal(reps["AP_H_minus_I"], reps["AP"][:, 2] - reps["AP"][:, 1]), "Saved replicate fields differ")
    generator = np.random.default_rng(SEED)
    for draw in draws:
        check(np.array_equal(draw, generator.integers(0, 5000, size=5000)), "Saved ordered RNG draws differ")
    findings = []
    for index in INDICES:
        draw = draws[index]
        independent = []
        for cache in caches:
            # No ordered_indices supplied: each arm expands every image copy
            # and performs its own stable score sort inside category_precision.
            ap, _, _ = reaccumulate(cache, draw)
            check(ap is not None and np.isfinite(ap), "Undefined independent AP")
            independent.append(ap)
        independent = np.asarray(independent, dtype=np.float64)
        shared_sort_replay = paired_aps(caches, draw)  # comparator, not independent answer
        saved = reps["AP"][index]
        delta = float(independent[2] - independent[1])
        saved_delta = float(reps["AP_H_minus_I"][index])
        errors = {"per_arm_vs_saved": np.abs(independent - saved).tolist(),
                  "per_arm_vs_shared_sort_replay": np.abs(independent - shared_sort_replay).tolist(),
                  "H_minus_I_vs_saved": abs(delta - saved_delta),
                  "H_minus_I_vs_shared_sort_replay": abs(delta - float(shared_sort_replay[2] - shared_sort_replay[1]))}
        maximum = max(*errors["per_arm_vs_saved"], *errors["per_arm_vs_shared_sort_replay"],
                      errors["H_minus_I_vs_saved"], errors["H_minus_I_vs_shared_sort_replay"])
        check(maximum <= TOLERANCE, "Independent AP wiring differs on fixed replicate " + str(index))
        findings.append({"replicate_index": index, "draw_size": len(draw), "unique_images": len(np.unique(draw)),
                         "draw_int32_bytes_sha256": hashlib.sha256(np.asarray(draw, dtype="<i4").tobytes()).hexdigest(),
                         "independent_per_arm_AP": independent.tolist(), "saved_per_arm_AP": saved.tolist(),
                         "production_shared_sort_replay_AP": shared_sort_replay.tolist(),
                         "independent_H_minus_I": delta, "saved_H_minus_I": saved_delta,
                         "absolute_errors": errors, "max_abs_error": maximum})
        print("independent three-arm AP check", index, "max_abs_error", maximum, flush=True)
    check(not any(k.split(".")[0] in {"torch", "joblib", "pycocotools"} for k in sys.modules),
          "Unexpected model or COCO package loaded")
    write_json(out / "SOURCE_LOCK.json", {"files": lock.finish(),
               "original_external_inputs": "Producer before/after attestations reconciled; no original JPEG/GT/remote reread"})
    source_hash = sha(out / "SOURCE_LOCK.json")
    result = {"status": "independent_ap4_complete", "passed": True, "replicate_indices": list(INDICES),
              "arms": list(ARMS), "absolute_tolerance": TOLERANCE, "AP_units": "fraction 0..1",
              "independent_AP_replicates": 4, "per_arm_reaccumulate_calls": 12,
              "ordered_rng_draws_checked": 1000, "all_1000_APs_independently_recomputed": False,
              "findings": findings, "max_abs_error": max(r["max_abs_error"] for r in findings),
              "independent_scope": "Per-arm image-copy expansion/stable sorting versus production shared-sort wiring; accepted accumulation core reused",
              "limitations": ["Only fixed replicates 0,1,499,999", "No new official GT matching", "No claim of independent full CI or 1000-AP reconstruction",
                              "Original external inputs checked as producer attestations, not re-read"],
              "GPU_or_model_execution": False, "GT_decoded_or_rematched": False,
              "source_lock_sha256": source_hash,
              "inputs": {label: {"run_id": root.name, "path": str(root), "run_json_sha256": sha(root / "run.json")}
                         for label, root in (("score", score), ("bootstrap", boot), ("verification", verify_run), ("engineering", engineering))},
              "interpreter": sys.executable, "python": platform.python_version(), "numpy": np.__version__}
    write_json(out / "VERIFICATION.json", result)
    artifacts = seal_outputs(out, ["VERIFICATION.json", "AP4_STARTED.json", *snapshots])
    write_json(out / "SUMMARY.json", {**result, "artifacts": artifacts})
    write_json(out / "COMPLETE.json", {"status": result["status"], "summary_sha256": sha(out / "SUMMARY.json"),
               "source_lock_sha256": source_hash, "verification_sha256": sha(out / "VERIFICATION.json"), "artifacts": artifacts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scoring-run", "bootstrap-run", "verification-run", "engineering-run", "protocol", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    runs = Path(__file__).resolve().parents[1] / "runs"
    out = args.out_dir.resolve()
    roots = [getattr(args, k).resolve() for k in ("scoring_run", "bootstrap_run", "verification_run", "engineering_run")]
    check(out.parent == runs and all(p.parent == runs for p in roots) and len(set(roots + [out])) == 5,
          "Require separate local Study Runs; output must not be a production Run")
    out.mkdir(parents=True, exist_ok=True)
    check(not any((out / name).exists() for name in ("AP4_STARTED.json", "VERIFICATION.json", "SUMMARY.json", "SOURCE_LOCK.json",
                                                   "COMPLETE.json", "VERIFICATION_FAILURE.json", "source")), "Independent Run already used")
    write_json(out / "AP4_STARTED.json", {"fixed_indices": list(INDICES), "tolerance": TOLERANCE,
               "formal_validation_passed": None, "GPU_or_model_execution": False})
    try:
        verify(args, out)
    except Exception as error:
        write_json(out / "VERIFICATION_FAILURE.json", {"status": "failed", "passed": False,
                   "error": repr(error), "traceback": traceback.format_exc(), "partial_outputs_preserved": True,
                   "GPU_or_model_execution": False, "formal_validation_passed": False})
        raise


if __name__ == "__main__":
    main()
