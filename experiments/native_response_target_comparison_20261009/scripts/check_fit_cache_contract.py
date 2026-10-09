"""Finite CPU-only NPZ access contract; no estimator fit, GT JSON or GPU.

All part arrays, supervised feature/target/identity bytes and integer targets
are compared. The original repeated per-row read is sampled deterministically
at part endpoints and real fallback rows. Timers describe this contract only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from target_common import dump, integer_targets, now, sha

ORIGINAL_SHA = "274143455b85fa6a25fa156aa69790dc58a0d0c79855c22ad8458aae678878d2"
CACHED_SHA = "a93a333f898e3069bd8d7a5f7e80d37a70d246bcdda5344e88a388a71101c386"
COMMON_SHA = "4de7929a7278038de66c0daacba17403217db71643d8b77ccd806658bb1f011f"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("extraction", "fit-reference", "fit-original", "fit-cached", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    source, fit, out = [Path(v).resolve() for v in (a.extraction, a.fit_reference, a.output)]
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "SUMMARY.json").exists(), "Cache contract retry requires a new Run")
    watched, started = {}, time.perf_counter()
    try:
        def watch(path, expected=None):
            digest = sha(path)
            require(expected is None or digest == expected, "Source bytes differ: " + str(path))
            watched[str(path)] = digest
            return digest
        watch(a.fit_original, ORIGINAL_SHA)
        watch(a.fit_cached, CACHED_SHA)
        watch(Path(__file__).with_name("target_common.py"), COMMON_SHA)
        watch(__file__)
        metadata = read(source / "run.json")
        require(metadata["source_kind"] == "runner_observed" and metadata["status"] == "completed" and
                metadata["return_code"] == 0 and metadata["artifact_completeness"] == "complete", "Actual full20k source gate differs")
        summary, complete, parts = [read(source / name) for name in ("SUMMARY.json", "COMPLETE.json", "ROWS_PARTS.json")]
        require(summary["passed"] is True and not summary["engineering"] and summary["images"] == 20000 and
                complete["passed"] is True and complete["summary_sha256"] == watch(source / "SUMMARY.json") and
                complete["rows_parts_sha256"] == watch(source / "ROWS_PARTS.json"), "Complete shared20k evidence differs")
        for name in ("run.json", "COMPLETE.json"):
            watch(source / name)
        fm, fs, fc = [read(fit / name) for name in ("run.json", "SUMMARY.json", "COMPLETE.json")]
        require(fm["source_kind"] == "runner_observed" and fm["status"] == "completed" and fm["return_code"] == 0 and
                fm["artifact_completeness"] == "complete" and fs["passed"] is True and fs["target"] == "I" and
                not fs["engineering"] and fc["passed"] is True and fc["summary_sha256"] == watch(fit / "SUMMARY.json"),
                "Original completed I fit reference differs")
        require(fs["extraction_summary_sha256"] == sha(source / "SUMMARY.json") and
                fs["extraction_complete_sha256"] == sha(source / "COMPLETE.json"), "Original fit extraction binding differs")
        for name in ("run.json", "COMPLETE.json"):
            watch(fit / name)
        names = ("raw_features", "features", "y_I", "y_H", "binding")
        cache_digests = {key: hashlib.sha256() for key in names}
        original_digests = {key: hashlib.sha256() for key in names}
        rows, known_rows, arrays_loaded, repeated_reads, sampled_rows, fallback_rows = 0, 0, 0, 0, 0, 0
        cache_load_seconds, reference_access_seconds, integer_replay_seconds = 0., 0., 0.
        records = []
        for info in parts["parts"]:
            path = (source / info["path"]).resolve()
            require(path.is_relative_to(source), "Part escapes extraction Run")
            watch(path, info["sha256"])
            t = time.perf_counter()
            with np.load(path, allow_pickle=False) as archive:
                cached = {key: archive[key] for key in archive.files}
            cache_load_seconds += time.perf_counter()-t
            arrays_loaded += len(cached)
            known = cached["known"].astype(bool)
            rows += len(known)
            known_rows += int(known.sum())
            fallback_rows += int(np.count_nonzero(known & cached["fallback"].astype(bool)))
            require(np.isnan(cached["y_I"][~known]).all() and np.isnan(cached["y_H"][~known]).all(), "Unknown targets changed")
            t = time.perf_counter()
            with np.load(path, allow_pickle=False) as original:
                original_keys = set(original.files)
                require(original_keys == set(cached), "Cached original NPZ key set differs")
                for key in original.files:
                    value = original[key]
                    repeated_reads += 1
                    require(value.shape == cached[key].shape and value.dtype == cached[key].dtype and
                            value.tobytes() == cached[key].tobytes(), "Cached key bytes differ: " + key)
                original_known = original["known"].astype(bool)
                repeated_reads += 1
                require(np.array_equal(known, original_known), "Known mask/order differs")
                for digests, values, mask in ((cache_digests, cached, known), (original_digests, original, original_known)):
                    digests["raw_features"].update(values["features"].tobytes())
                    digests["features"].update(values["features"][mask].tobytes())
                    for target in ("y_I", "y_H"):
                        digests[target].update(values[target][mask].tobytes())
                    binding = np.column_stack([values[key][mask] for key in
                                               ("image_id", "detection_index", "native_output_row", "raw_index", "annotation_id")])
                    digests["binding"].update(binding.tobytes())
                    if values is original:
                        repeated_reads += 9
                valid = np.flatnonzero(known)
                indices = set(valid[[0, -1]].tolist()) if len(valid) else set()
                empty = np.flatnonzero(known & cached["fallback"].astype(bool))
                if len(empty):
                    indices.add(int(empty[0]))
                for j in sorted(indices):
                    result = integer_targets(int(original["baseline_tp"][j]), int(original["baseline_union"][j]),
                                             int(original["trial_tp"][j]), int(original["trial_union"][j]), bool(original["fallback"][j]))
                    legacy_target = (float(original["y_I"][j]), float(original["y_H"][j]), int(original["threshold_ticks"][j]))
                    repeated_reads += 8
                    require(result == legacy_target == (float(cached["y_I"][j]), float(cached["y_H"][j]), int(cached["threshold_ticks"][j])),
                            "Original literal repeated row access label differs")
                sampled_rows += len(indices)
            reference_access_seconds += time.perf_counter()-t
            t = time.perf_counter()
            for j in np.flatnonzero(known):
                result = integer_targets(int(cached["baseline_tp"][j]), int(cached["baseline_union"][j]),
                                         int(cached["trial_tp"][j]), int(cached["trial_union"][j]), bool(cached["fallback"][j]))
                require(result == (float(cached["y_I"][j]), float(cached["y_H"][j]), int(cached["threshold_ticks"][j])),
                        "Cached full-row integer label replay differs")
            integer_replay_seconds += time.perf_counter()-t
            records.append(dict(part=info["path"], keys=len(cached), cache_reads_per_key=1,
                                rows=len(known), supervised_rows=int(known.sum()), original_repeated_row_samples=len(indices)))
        digests = {key: value.hexdigest() for key, value in cache_digests.items()}
        require(digests == {key: value.hexdigest() for key, value in original_digests.items()}, "All-part supervised byte/order digests differ")
        require(rows == parts["rows"] and known_rows == parts["supervised_rows"] == fs["training_rows"] and
                digests["raw_features"] == parts["feature_bytes_sha256"] and
                digests["features"] == fs["training_feature_bytes_sha256"] and digests["y_I"] == fs["target_bytes_sha256"] and
                digests["binding"] == parts["supervised_binding_sha256"] == fs["supervised_binding_sha256"],
                "Actual original training bytes/row denominator differ")
        unit_weight_sha = hashlib.sha256(np.ones(known_rows, dtype=np.float64).tobytes()).hexdigest()
        require(fs["weight_bytes_sha256"] == unit_weight_sha and fs["sample_weight"] == "unit default, no sample_weight argument",
                "Actual fixed unit weight contract differs")
        for path, digest in watched.items():
            require(sha(path) == digest, "Source changed during finite cache contract")
        result = dict(passed=True, status="cache_contract_complete", source_extraction=source.name, original_fit=fit.name,
                      original_fit_source_sha256=ORIGINAL_SHA, cached_fit_source_sha256=CACHED_SHA,
                      contract_source_sha256=sha(__file__), parts=len(records), rows=rows, supervised_rows=known_rows,
                      fallback_supervised_rows=fallback_rows, cached_array_loads=arrays_loaded,
                      original_reference_array_loads=repeated_reads, original_repeated_row_samples=sampled_rows,
                      all_array_bytes_and_order_equal=True, all_supervised_integer_targets_exact=True,
                      digests=digests, unit_weight_bytes_sha256=unit_weight_sha, part_receipts=records, cache_load_seconds=cache_load_seconds,
                      original_reference_access_seconds=reference_access_seconds, cached_integer_replay_seconds=integer_replay_seconds,
                      elapsed_seconds=time.perf_counter()-started, completed_at=now(), gt_json_parsed=False, gpu_used=False,
                      model_fitted=False, ap_measured=False, production_used_cached_fit=False,
                      limits="All arrays/ordered digests/full cached integer labels checked; old repeated per-row NPZ access sampled, not full old-fit wall-clock remeasurement or estimator retraining")
        dump(out / "SUMMARY.json", result)
        dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json")))
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
