"""Fit one predeclared HGB target on the identical shared supervised rows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from target_common import MODEL_PARAMS, dump, export_numeric, integer_targets, load_config, now, runtime, sha, source_lock, verify_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--target", choices=("I", "H"), required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out, source = Path(args.output).resolve(), Path(args.input).resolve()
    if (out / "INPUTS.json").exists():
        raise FileExistsError("Fit retry requires a distinct Run")
    out.mkdir(parents=True, exist_ok=True)
    c = load_config(args.config, args.protocol_sha256)
    receipt = json.loads((source / "COMPLETE.json").read_text())
    source_summary = json.loads((source / "SUMMARY.json").read_text())
    if not receipt["passed"] or receipt["summary_sha256"] != sha(source / "SUMMARY.json") or not source_summary["passed"]:
        raise ValueError("Shared extraction is incomplete")
    parts = json.loads((source / "ROWS_PARTS.json").read_text())
    if receipt["rows_parts_sha256"] != sha(source / "ROWS_PARTS.json"):
        raise ValueError("Shared row manifest changed")
    input_paths = [source / "COMPLETE.json", source / "SUMMARY.json", source / "ROWS_PARTS.json"]
    input_paths += [source / row["path"] for row in parts["parts"]]
    lock = source_lock(out, c, args.config, input_paths)
    began = time.perf_counter()
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        from portable_risk import PortableRisk
        env = runtime()
        xx, yy, row_ids = [], [], []
        all_feature_digest = hashlib.sha256()
        for info in parts["parts"]:
            path = source / info["path"]
            if sha(path) != info["sha256"]:
                raise ValueError("Shared rows changed")
            with np.load(path, allow_pickle=False) as arrays:
                known = arrays["known"].astype(bool)
                if arrays["features"].dtype != np.float64 or not np.isfinite(arrays["features"]).all():
                    raise ValueError("Shared feature dtype/finite contract changed")
                all_feature_digest.update(arrays["features"].tobytes())
                if not np.isnan(arrays["y_I"][~known]).all() or not np.isnan(arrays["y_H"][~known]).all():
                    raise ValueError("Unknown labels were filled")
                if not np.isfinite(arrays["y_I"][known]).all() or not np.isfinite(arrays["y_H"][known]).all():
                    raise ValueError("Known targets are invalid")
                if np.any(arrays["fallback"][known].astype(bool) & ((arrays["y_I"][known] != 0) | (arrays["y_H"][known] != 0))):
                    raise ValueError("Actual empty fallback target is nonzero")
                if (np.any(arrays["detection_index"] >= 64) or np.any(arrays["detection_index"] < 0)
                        or np.any(arrays["annotation_id"][known] < 0)):
                    raise ValueError("Supervised eligibility/binding changed")
                for j in np.flatnonzero(known):
                    yi, yh, ticks = integer_targets(int(arrays["baseline_tp"][j]), int(arrays["baseline_union"][j]),
                                                    int(arrays["trial_tp"][j]), int(arrays["trial_union"][j]), bool(arrays["fallback"][j]))
                    if yi != arrays["y_I"][j] or yh != arrays["y_H"][j] or ticks != arrays["threshold_ticks"][j]:
                        raise ValueError("Stored target differs from independent integer-pixel replay")
                xx.append(arrays["features"][known])
                yy.append(arrays["y_" + args.target][known])
                row_ids.append(np.column_stack([arrays[k][known] for k in ("image_id", "detection_index", "native_output_row", "raw_index", "annotation_id")]))
        x, y, binding = np.concatenate(xx), np.concatenate(yy), np.concatenate(row_ids)
        if all_feature_digest.hexdigest() != parts["feature_bytes_sha256"] or len(x) != parts["supervised_rows"]:
            raise ValueError("Shared row/feature digest mismatch")
        if hashlib.sha256(binding.tobytes()).hexdigest() != parts["supervised_binding_sha256"]:
            raise ValueError("Shared supervised identity differs")
        if not len(x):
            raise ValueError("No valid supervised rows; scientific fit cannot proceed")
        # No normalization, scaler, weighting or target-dependent row filtering.
        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        parameters = model.get_params(deep=True)
        inputs = dict(target=args.target, extraction_run=str(source), extraction_summary_sha256=sha(source / "SUMMARY.json"),
                      extraction_complete_sha256=sha(source / "COMPLETE.json"), protocol_sha256=args.protocol_sha256,
                      runtime=env, full_hgb_parameters=parameters, sample_weight="unit default, no sample_weight argument",
                      weight_bytes_sha256=hashlib.sha256(np.ones(len(x), dtype=np.float64).tobytes()).hexdigest(),
                      training_feature_bytes_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                      target_bytes_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                      supervised_binding_sha256=parts["supervised_binding_sha256"], training_rows=len(x),
                      training_images=len(set(binding[:, 0].tolist())), engineering=source_summary["engineering"])
        dump(out / "INPUTS.json", inputs)
        model.fit(x, y)
        if model.n_iter_ != 100 or model.get_params(deep=True) != parameters:
            raise ValueError("Frozen100-iteration capacity changed")
        joblib.dump(model, out / "model.joblib")
        export_numeric(model, out / "model.json")
        portable = PortableRisk(out / "model.json")
        maximum, mismatches, squared_error = 0., 0, 0.
        prediction_digest = hashlib.sha256()
        for start in range(0, len(x), 4096):
            reference = model.predict(x[start:start+4096])
            actual = portable.predict(x[start:start+4096])
            maximum = max(maximum, float(np.max(np.abs(reference-actual))))
            mismatches += int(np.count_nonzero((reference > 0) != (actual > 0)))
            squared_error += float(np.square(reference-y[start:start+4096]).sum())
            prediction_digest.update(actual.tobytes())
        if maximum >= 1e-12 or mismatches:
            raise ValueError("Portable estimator differs from fitted HGB")
        verify_lock(lock)
        summary = dict(passed=True, status="fit_complete", **inputs, iterations=int(model.n_iter_), features=5,
                       fit_mse=squared_error/len(y), portable_max_abs_error=maximum, portable_sign_mismatches=mismatches,
                       portable_training_prediction_sha256=prediction_digest.hexdigest(),
                       model_sha256=sha(out / "model.json"), joblib_sha256=sha(out / "model.joblib"),
                       source_lock_sha256=sha(out / "SOURCE_LOCK.json"), source_unchanged=True,
                       elapsed_seconds=time.perf_counter()-began, completed_at=now(),
                       gt_parsed=False, frozen_backbone_trained=False, model_selection=False)
        dump(out / "SUMMARY.json", summary)
        dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json"), model_sha256=sha(out / "model.json")))
        print(f"TARGET_FIT_COMPLETE {args.target} rows={len(x)} model={summary['model_sha256']}", flush=True)
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_at=now()))
        raise


if __name__ == "__main__":
    main()
