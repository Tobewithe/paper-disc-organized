"""Verify the sealed CPU Score with explicit zero-row Counter normalization.

The archived scorer and helper execute unchanged. Its JSONL input adapter adds
two mathematically zero Counter keys only for independently declared zero-row
images whose seven arm dictionaries are all empty. No prediction or Score file
is rewritten, and no COCO matching, GT-mask calculation or model is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run, out):
    run, out = run.resolve(), out.resolve()
    assert run != out and run not in out.parents
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "VERIFICATION.json").exists()
    locked = json.loads((run / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    expected = locked["executed_source_sha256"]
    scorer = run / "source/score_and_verify_comparison.py"
    helper = run / "source/comparison_io.py"
    assert sha(scorer) == expected[scorer.name]
    assert sha(helper) == expected[helper.name]
    before = {str(path): sha(path) for path in (Path(__file__), scorer, helper)}
    archive = out / "source"
    archive.mkdir(exist_ok=True)
    for path in (Path(__file__), scorer, helper):
        target = archive / path.name
        assert not target.exists() or sha(target) == before[str(path)]
        if not target.exists():
            shutil.copy2(path, target)
    sys.path.insert(0, str(scorer.parent))
    spec = importlib.util.spec_from_file_location("sealed_native_cpu_score", scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == scorer
    assert sha(Path(sys.modules["comparison_io"].__file__)) == expected[helper.name]
    original_jsonl = module.jsonl
    identity_path = (run / "IDENTITY_IMAGES.jsonl").resolve()
    summary = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
    native = Path(summary["inference_run"])
    normalized_ids = []

    def explicit_counter_zeros(path):
        for row in original_jsonl(path):
            if Path(path).resolve() == identity_path and row.get("candidate_count") == 0:
                assert type(row["candidate_count"]) is int
                assert set(row["arm_counts"]) == set(module.ARMS)
                assert all(row[key] is True for key in (
                    "all_arms_identity_exact", "official_baseline_rle_exact",
                    "all_empty_ordinals_retained", "first64_domain_exact"))
                # The producer's empty Counter has no keys at all. Any supplied
                # nonzero or unrelated field is a contract error, not defaulted.
                assert all(type(row["arm_counts"][arm]) is dict and
                           row["arm_counts"][arm] == {} for arm in module.ARMS)
                for arm in module.ARMS:
                    image_file = (native / arm / "images" / f"{row['image_id']:012d}.json").resolve()
                    evidence = locked["files"][str(image_file)]
                    assert sha(image_file) == evidence["sha256_before"] == evidence["sha256_after"]
                    actual = json.loads(image_file.read_text(encoding="utf-8"))
                    assert actual["image_id"] == row["image_id"] and actual["detections"] == []
                row = dict(row)
                row["arm_counts"] = {arm: {"candidate_count": 0, "empty_mask_count": 0}
                                     for arm in module.ARMS}
                normalized_ids.append(row["image_id"])
            yield row

    module.jsonl = explicit_counter_zeros
    report = module.verify_scoring(run, out)
    assert all(sha(path) == digest for path, digest in before.items())
    assert "torch" not in sys.modules
    report["verification_source_sha256"] = before[str(Path(__file__))]
    report["archived_scorer_source_sha256"] = expected
    report["verification_source_before_after_unchanged"] = True
    report["consumer_normalization"] = {
        "scope": "only declared zero-candidate IDENTITY_IMAGES rows with all seven arm Counters exactly empty",
        "normalized_image_count": len(normalized_ids),
        "normalized_image_ids": normalized_ids,
        "keys_added_as_zero": ["candidate_count", "empty_mask_count"],
        "source_or_scoring_artifacts_rewritten": False,
        "nonzero_candidate_rows_normalized": False,
        "all_seven_actual_per_image_empty_lists_independently_checked": True,
        "actual_empty_image_sources_bound_to_score_before_after_hashes": True,
        "original_verifier_source_executed": str(scorer),
        "original_input_contract": "collections.Counter emits an empty dictionary before any candidate increment",
    }
    module.write_json(out / "VERIFICATION.json", report)
    print(json.dumps({"status": "passed", "normalized_zero_candidate_images": len(normalized_ids)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.run, args.out_dir)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "VERIFICATION_FAILURE.json").write_text(json.dumps({
            "status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
            "GPU_or_model_execution": False, "source_or_scoring_artifacts_rewritten": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
