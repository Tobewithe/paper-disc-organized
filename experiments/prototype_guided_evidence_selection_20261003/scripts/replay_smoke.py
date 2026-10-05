"""Server-only replay audit of the migrated original YOLO input and neck maps.

This script performs no training, TAL recomputation, new prototype extraction,
candidate filtering or data-membership change. Partial witness smoke can aid
transport debugging but cannot authorize formal training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from asset_runtime import (FrozenReplay, dump, load_asset, load_json,
                           resolve_runtime_config, tensor_sha, sha256)


def difference(actual, expected, atol, rtol):
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(passed=False, shape_actual=list(actual.shape), shape_expected=list(expected.shape),
                    dtype_actual=str(actual.dtype), dtype_expected=str(expected.dtype),
                    error="shape or dtype mismatch")
    delta = (actual.double() - expected.double()).abs()
    allowed = float(atol) + float(rtol) * expected.double().abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return dict(passed=bool(finite and (delta <= allowed).all()), finite=finite,
                shape=list(actual.shape), dtype=str(actual.dtype),
                max_absolute_error=float(delta.max()) if delta.numel() else 0.,
                max_tolerance_fraction=float((delta / allowed).max()) if delta.numel() else 0.,
                out_of_tolerance_elements=int((delta > allowed).sum()),
                atol=float(atol), rtol=float(rtol))


@torch.no_grad()
def decode_difference(image, replayed_coefficients, coco):
    """Quantify zero-threshold flips without replacing the authoritative A."""
    import numpy as np
    from ultralytics.utils import ops
    device = replayed_coefficients.device
    proto = image["proto"].to(device)
    boxes = image["boxes"].to(device)
    original = image["c0"].to(device)
    shape, original_shape = tuple(image["input_shape"]), tuple(image["original_shape"])
    records = []
    for lo in range(0, len(original), 4):
        hi = min(lo + 4, len(original))
        pad_a = ops.process_mask(proto, original[lo:hi], boxes[lo:hi], shape, upsample=True)
        pad_r = ops.process_mask(proto, replayed_coefficients[lo:hi], boxes[lo:hi], shape, upsample=True)
        out_a = ops.scale_masks(pad_a[None], original_shape, ratio_pad=image["ratio_pad"])[0] > .5
        out_r = ops.scale_masks(pad_r[None], original_shape, ratio_pad=image["ratio_pad"])[0] > .5
        for j, row in enumerate(image["rows"][lo:hi]):
            ann = coco.anns[int(row["annotation_id"])]
            if int(ann["image_id"]) != int(image["image_id"]):
                raise AssertionError("COCO original mask does not match the fixed identity")
            truth = torch.as_tensor(coco.annToMask(ann).astype(np.bool_), device=device)
            if tuple(truth.shape) != original_shape:
                raise AssertionError("COCO original mask geometry changed")
            ia = int((out_a[j] & truth).sum()) / max(1, int((out_a[j] | truth).sum()))
            ir = int((out_r[j] & truth).sum()) / max(1, int((out_r[j] | truth).sum()))
            records.append(dict(annotation_id=int(row["annotation_id"]), raw_id=int(row["raw_id"]),
                padded_pixel_differences=int((pad_a[j].bool() != pad_r[j].bool()).sum()),
                original_pixel_differences=int((out_a[j] != out_r[j]).sum()),
                original_iou_A=ia, original_iou_replayed=ir, original_iou_delta=ir-ia,
                mask75_changed=bool((ia >= .75) != (ir >= .75))))
    return dict(candidates=len(records), padded_pixel_differences=sum(r["padded_pixel_differences"] for r in records),
                original_pixel_differences=sum(r["original_pixel_differences"] for r in records),
                max_original_iou_absolute_change=max((abs(r["original_iou_delta"]) for r in records), default=0.),
                mean_original_iou_delta=sum(r["original_iou_delta"] for r in records)/max(1, len(records)),
                mask75_changed_candidates=sum(r["mask75_changed"] for r in records),
                per_candidate=records,
                interpretation="Original P/c0 remain A. These differences quantify tolerated FP32 coefficient replay error near threshold; no new baseline or altered decoder is introduced.")


def main(args):
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise RuntimeError("Use an independent replay-audit Run; do not overwrite its receipt")
    report = dict(schema="prototype-evidence-server-replay-v1", passed=False,
                  formal_training_allowed=False, optimizer_steps=0, candidates_reassigned=0,
                  rows=[], source_prototype_and_masks_authoritative=True)
    try:
        cfg = resolve_runtime_config(load_json(args.config))
        assets = Path(cfg["assets"])
        migration = load_json(assets / "MIGRATION_IDENTITY.json")
        index = load_json(assets / "INDEX.json")
        witnesses = migration["witnesses"]
        if args.limit_per_split is not None and args.limit_per_split < 1:
            raise ValueError("Partial smoke count must be positive")
        expected = {s: [int(r["image_id"]) for r in index[s] if int(r["n"]) > 0][:8]
                    for s in ("fit", "dev", "val")}
        if witnesses != expected:
            raise AssertionError("F witness membership differs from first eight positive images per frozen split")
        selected = {s: ids if args.limit_per_split is None else ids[:args.limit_per_split]
                    for s, ids in witnesses.items()}
        report.update(tolerance=dict(atol=cfg["replay_atol"], rtol=cfg["replay_rtol"]),
                      planned_witnesses=witnesses, evaluated_witnesses=selected,
                      migration_fingerprint=migration["fingerprint"],
                      partial_witness_smoke=args.limit_per_split is not None,
                      config_sha256=sha256(args.config),
                      script_sha256=sha256(__file__))
        replay = FrozenReplay(cfg)
        report["environment"] = replay.import_info
        from pycocotools.coco import COCO
        root = Path(cfg["server_root"])
        annotations = {"fit": cfg.get("annotations_train", root / "data" / "instances_train2017.json"),
                       "val": cfg.get("annotations_val", root / "data" / "instances_val2017.json")}
        coco = {split: COCO(str(path)) for split, path in annotations.items()}
        coco["dev"] = coco["fit"]
        for split, ids in selected.items():
            for iid in ids:
                image = load_asset(cfg, iid)
                record = dict(split=split, image_id=iid, candidates=len(image["rows"]), passed=False,
                              asset_integrity=image["_asset_integrity"])
                report["rows"].append(record)
                if image["split"] != split or "F_reference" not in image:
                    raise AssertionError("Witness reference or frozen split is missing")
                mask_before, proto_before = tensor_sha(image["masks"]), tensor_sha(image["proto"])
                features = replay.replay([image])
                record["F"] = [difference(value[0], expected, cfg["replay_atol"], cfg["replay_rtol"])
                               for value, expected in zip(features, image["F_reference"])]
                native = replay.native_outputs(features, [image])[0]
                record["h"] = difference(native["h"], image["_operator"]["h0"], cfg["replay_atol"], cfg["replay_rtol"])
                record["c"] = difference(native["c"], image["c0"], cfg["replay_atol"], cfg["replay_rtol"])
                # This additional replay separates a prefix discrepancy from
                # an original-branch/library discrepancy if the audit fails.
                reference_features = [value.to(replay.device)[None] for value in image["F_reference"]]
                reference_native = replay.native_outputs(reference_features, [image])[0]
                record["native_on_reference_F_h"] = difference(reference_native["h"], image["_operator"]["h0"], cfg["replay_atol"], cfg["replay_rtol"])
                record["native_on_reference_F_c"] = difference(reference_native["c"], image["c0"], cfg["replay_atol"], cfg["replay_rtol"])
                record["normal_mask_replay"] = decode_difference(image, native["c"], coco[split])
                record["full_source_mask_unchanged"] = tensor_sha(image["masks"]) == mask_before
                record["full_source_prototype_unchanged"] = tensor_sha(image["proto"]) == proto_before
                checks = record["F"] + [record["h"], record["c"], record["native_on_reference_F_h"], record["native_on_reference_F_c"]]
                record["passed"] = all(check["passed"] for check in checks) and record["full_source_mask_unchanged"] and record["full_source_prototype_unchanged"]
                dump(out / "AUDIT.json", report)
                print(json.dumps(dict(stage="frozen_replay", split=split, image_id=iid, passed=record["passed"],
                                      images_completed=len(report["rows"]), elapsed_s=time.monotonic()-started)), flush=True)
                if not record["passed"]:
                    raise AssertionError(f"Frozen source replay failed at {split}/{iid}; do not relax tolerances")
                del image, features, native, reference_features, reference_native
        replay.assert_unchanged()
        migration_receipt_path = assets / "COMPLETE.json"
        migration_complete = migration_receipt_path.exists() and load_json(migration_receipt_path).get("complete") is True
        formal_scope = args.limit_per_split is None and len(report["rows"]) == sum(map(len, witnesses.values()))
        report.update(passed=True, frozen_source_buffers_unchanged=True, migration_complete=migration_complete,
                      formal_witness_scope_complete=formal_scope,
                      formal_training_allowed=bool(migration_complete and formal_scope),
                      elapsed_s=time.monotonic()-started,
                      note="This is only the prefix/data replay gate. Official-loss, selected-cell and trainable-branch smoke must also pass before training.")
        dump(out / "AUDIT.json", report)
        dump(out / "COMPLETE.json", dict(passed=True, complete=True,
             formal_training_allowed=report["formal_training_allowed"], images=len(report["rows"]),
             migration_complete=migration_complete, partial_witness_smoke=args.limit_per_split is not None,
             optimizer_steps=0, elapsed_s=report["elapsed_s"]))
    except Exception as exc:
        report.update(passed=False, formal_training_allowed=False,
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                      elapsed_s=time.monotonic()-started)
        dump(out / "AUDIT.json", report)
        dump(out / "COMPLETE.json", dict(passed=False, complete=False, formal_training_allowed=False,
             optimizer_steps=0, error_type=type(exc).__name__, error=str(exc), elapsed_s=report["elapsed_s"]))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-per-split", type=int)
    main(parser.parse_args())
