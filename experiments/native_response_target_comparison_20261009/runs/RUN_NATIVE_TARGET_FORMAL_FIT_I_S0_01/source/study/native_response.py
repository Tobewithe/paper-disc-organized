"""One shared native smooth decode. No GT parsing or learned policy here."""
from __future__ import annotations

import hashlib
import numpy as np

from target_common import canonical_sha


def decode(extracted, prepared, image_id, categories, mask_utils, device):
    import torch
    from ultralytics.utils import ops
    from native_mask_adapter import Timer, native_proto_logits, export_binary, _records
    from risk_calibration import prediction_features
    count = len(prepared.coefficients)
    base_records, smooth_records, rows = [], [], []
    timing = dict(proto_logits_seconds=0., input_scale_seconds=0., smooth_seconds=0.,
                  original_export_seconds=0., features_seconds=0., rle_seconds=0.)
    if count:
        with Timer(device) as timer:
            # Preserve full post-conf Mx32 GEMM; native step78 used thereafter.
            all_logits = native_proto_logits(extracted["P"].to(device), prepared.coefficients)
        timing["proto_logits_seconds"] = timer.seconds
        step = max(1, 32_000_000 // (640 * 640))
        for start in range(0, count, step):
            end = min(count, start + step)
            with Timer(device) as timer:
                logits = ops.scale_masks(all_logits[start:end][None], (640, 640))[0]
                baseline = ops.crop_mask((logits > 0).byte(), prepared.boxes[start:end])
            timing["input_scale_seconds"] += timer.seconds
            if not torch.equal(baseline, prepared.baseline_input[start:end]):
                raise RuntimeError("Fresh official zero input-grid byte parity failed")
            with Timer(device) as timer:
                base = export_binary(baseline, prepared.pbatch)
            timing["original_export_seconds"] += timer.seconds
            if not torch.equal(base.cpu(), prepared.baseline_original[start:end]):
                raise RuntimeError("Fresh official zero original-grid byte parity failed")
            with Timer(device) as timer:
                records = _records(base, prepared, image_id, start, categories, mask_utils)
            timing["rle_seconds"] += timer.seconds
            base_records.extend(records)
            smooth_records.extend(records)
            n = max(0, min(end, 64) - start)
            if n:
                with Timer(device) as timer:
                    area = base[:n].sum((1, 2)).float()
                    tau = .75 / (1.0 + (area / 2304.0).square())
                    trial640 = ops.crop_mask((logits[:n] > tau[:, None, None]).byte(), prepared.boxes[start:start+n])
                timing["smooth_seconds"] += timer.seconds
                with Timer(device) as timer:
                    trial = export_binary(trial640, prepared.pbatch)
                timing["original_export_seconds"] += timer.seconds
                if bool((trial640 > baseline[:n]).any()) or bool((trial > base[:n]).any()):
                    raise RuntimeError("Positive smooth action expanded baseline pixels")
                with Timer(device) as timer:
                    features = prediction_features(baseline[:n], base[:n], trial, "response")
                timing["features_seconds"] += timer.seconds
                if features.dtype != np.float64 or features.shape != (n, 5) or not np.isfinite(features).all():
                    raise RuntimeError("Canonical five-feature dtype/shape contract failed")
                nonempty = trial.flatten(1).any(1).bool().cpu().numpy()
                supported = prepared.supported_first64[start:start+n].cpu().numpy()
                with Timer(device) as timer:
                    trials = _records(trial, prepared, image_id, start, categories, mask_utils)
                timing["rle_seconds"] += timer.seconds
                trial_cpu = trial.cpu().numpy()
                for j in range(n):
                    index = start + j
                    if not bool(supported[j]):
                        continue
                    if nonempty[j]:
                        smooth_records[index] = trials[j]
                    feature = np.ascontiguousarray(features[j])
                    rows.append(dict(detection_index=index,
                                     native_output_row=int(prepared.eligible_output_rows[index]),
                                     raw_index=int(extracted["raw_indices"][prepared.eligible_output_rows[index]]),
                                     baseline_area=int(area[j].item()), smooth_tau=float(tau[j].item()),
                                     features=feature, feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest(),
                                     trial_nonempty=bool(nonempty[j]), fallback=not bool(nonempty[j]),
                                     trial_area=int(trial_cpu[j].sum(dtype=np.int64)), trial_mask=trial_cpu[j]))
            del logits, baseline, base
    if len(base_records) != count or len(smooth_records) != count:
        raise RuntimeError("Native rows missing")
    identity_keys = ("image_id", "category_id", "bbox", "score", "detection_index",
                     "raw_input_box_xyxy", "raw_confidence", "box_xyxy", "model_class")
    if any(any(a[k] != b[k] for k in identity_keys) for a, b in zip(base_records, smooth_records)):
        raise RuntimeError("Action altered native identity")
    coefficient_sha = hashlib.sha256(prepared.coefficients.cpu().contiguous().numpy().tobytes()).hexdigest()
    native_identity = dict(eligible_output_rows=prepared.eligible_output_rows,
                          raw_indices=extracted["raw_indices"][prepared.eligible_output_rows].tolist(),
                          coefficients_sha256=coefficient_sha, boxes=prepared.raw_boxes.tolist(),
                          scores=prepared.scores.tolist(), classes=prepared.classes.tolist())
    binding = dict(native_identity_sha256=canonical_sha(native_identity), coefficients_sha256=coefficient_sha,
                   native_output_rows=native_identity["eligible_output_rows"], raw_indices=native_identity["raw_indices"],
                   hidden_query_sha256=hashlib.sha256(extracted["h"][prepared.eligible_output_rows].contiguous().numpy().tobytes()).hexdigest(),
                   input_sha256=extracted["input_sha256"], image_sha256=extracted["image_sha256"],
                   native_rows=count, supported_first64=len(rows), selected_first64=min(64, count),
                   unsupported_first64=min(64, count)-len(rows), all_zero_native_bytes_exact=True,
                   baseline_original_binary_sha256=hashlib.sha256(prepared.baseline_original.contiguous().numpy().tobytes()).hexdigest(),
                   full_postconf_gemm=True, native_chunk_step=78, timing=timing)
    return base_records, smooth_records, rows, binding


def public_rows(rows):
    return [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items() if k != "trial_mask"} for r in rows]
