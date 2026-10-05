"""Fixed epoch-12 U/Q/P evaluation on unchanged official one-to-one candidates.

Run only on the authorized Linux GPU server. P/c0/labels are original archived
tensors. Frozen original-prefix F is replayed; no assignment is rerun. Final
mask quality uses original COCO masks and the already audited normal decoder.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops

from asset_runtime import FrozenReplay, dump, load_asset, load_json, resolve_runtime_config, sha256, tensor_sha
from point_head import JointPointCoefficientReadout
from point_selectors import build_selection_operators
from runtime_utils import gpu_image, setup
from evaluation_metrics import evaluate_image, append_rows, summarize, enable_historical_native

ARMS = ("U", "Q", "P")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
FINAL_EPOCH = 12
EVALUATION_LIMITS = {"fit": 1000, "dev": 1000, "val": 2000}
SMOKE_LIMITS = {"fit": 2, "dev": 2, "val": 2}


def identity(row):
    return (str(row["split"]), int(row["image_id"]), int(row["annotation_id"]), int(row["raw_id"]))


def selected_index(index, smoke):
    chosen = {}
    for split, limit in (SMOKE_LIMITS if smoke else EVALUATION_LIMITS).items():
        if not isinstance(index.get(split), list) or (not smoke and len(index[split]) < limit):
            raise ValueError(f"Missing fixed {split} evaluation population")
        chosen[split] = index[split][:limit]
        ids = [int(row["image_id"]) for row in chosen[split]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Repeated image identity in {split}")
    return chosen


def historical_rows(path, chosen, arm, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Same-ID historical {arm} rows required: {path}")
        return None
    wanted = {(split, int(item["image_id"])) for split, items in chosen.items() for item in items}
    result = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (str(row["split"]), int(row["image_id"])) not in wanted:
                continue
            key = identity(row)
            if key in result:
                raise ValueError(f"Historical {arm} duplicate identity: {key}")
            for field in ("branch", "pyramid_level", "target_gt_idx", f"iou_{arm}", f"mask75_{arm}"):
                if field not in row:
                    raise ValueError(f"Historical {arm} missing {field}: {key}")
            value = float(row[f"iou_{arm}"])
            if not math.isfinite(value) or not 0 <= value <= 1 or row["branch"] != "one2one":
                raise ValueError(f"Historical {arm} invalid row: {key}")
            if arm != "A":
                for metric in METRICS:
                    if f"{metric}_{arm}" not in row:
                        raise ValueError(f"Historical {arm} missing {metric}: {key}")
            result[key] = row
    expected = sum(int(row["n"]) for items in chosen.values() for row in items)
    if len(result) != expected:
        raise ValueError(f"Historical {arm} has {len(result)} rows for {expected} evaluated candidates")
    return result


@torch.no_grad()
def evaluation_selection(image, mode, device):
    """Frozen GT-free operators, same input and formula as training."""
    original = image["_operator"]
    a = original["A"].to(device)
    gram = original["G"].to(device)
    c0 = image["c0"].to(device)
    built = build_selection_operators(a, gram, c0, mode)
    source_valid = original["valid"].to(device).bool()
    if bool((source_valid & ~built["valid"]).any()):
        raise ArithmeticError(f"{mode}/{image['image_id']}: new numeric operator failure on previously valid candidates")
    valid = built["valid"] & source_valid
    diag = built["diagnostics"]
    for key, maximum in (("relative_residual_fp64", 1e-8), ("relative_residual_fp32", 1e-4)):
        values = diag[key][valid]
        if values.numel() and (not bool(torch.isfinite(values).all()) or float(values.max()) > maximum):
            raise ArithmeticError(f"{mode}/{image['image_id']}: {key} exceeds predeclared {maximum}")
    indices = built["indices"]
    ordered = indices.sort(1).values
    if indices.shape != (len(c0), 64) or bool(((indices < 0) | (indices >= 256)).any()) or bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise AssertionError("Every candidate must retain 64 unique frozen sampling cells")
    coefficient_operator = built["K"].float().detach()
    coefficient_operator[~valid] = 0
    selected = dict(raw_ids=image["raw_ids"].to(device).long().detach(),
        boxes=image["boxes"].to(device).float().detach(),
        h0=original["h0"].to(device).float().detach(), c0=c0.float().detach(),
        A_full=a.float().detach(), indices=indices.detach(), K=coefficient_operator, valid=valid.detach())
    if any(value.requires_grad for value in selected.values()):
        raise AssertionError("Frozen evaluation inputs unexpectedly require gradient")
    audit = dict(image_id=int(image["image_id"]), mode=mode, candidates=len(c0),
        invalid_candidates_retained=int((~valid).sum()),
        source_invalid_candidates=int((~source_valid).sum()),
        selector_invalid_candidates=int((~built["valid"]).sum()),
        indices_sha256=tensor_sha(indices), K_float32_sha256=tensor_sha(coefficient_operator),
        max_relative_residual_fp64=float(diag["relative_residual_fp64"][valid].max()) if bool(valid.any()) else None,
        max_relative_residual_fp32=float(diag["relative_residual_fp32"][valid].max()) if bool(valid.any()) else None)
    return selected, audit


@torch.no_grad()
def official_bce_values(image, coefficients, chunk_size=8):
    if tuple(image["input_shape"]) != (640, 640):
        raise ValueError("Fixed official label geometry must remain 640x640")
    proto = F.interpolate(image["proto"][None].float(), (640, 640), mode="bilinear", align_corners=False)[0]
    values = []
    for lo in range(0, len(coefficients), chunk_size):
        hi = min(lo + chunk_size, len(coefficients))
        target = (image["masks"][None] == (image["owners"][lo:hi] + 1)[:, None, None]).float()
        boxes = image["target_boxes"][lo:hi]
        area = ((boxes[:, 2:] - boxes[:, :2]) / 640).prod(1)
        if not bool((area > 0).all()):
            raise ValueError("Official target-box normalized area must be positive")
        logits = torch.einsum("in,nhw->ihw", coefficients[lo:hi].float(), proto)
        pixels = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        result = ops.crop_mask(pixels, boxes).mean((1, 2)) / area * float(image["segmentation_gain"])
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite official BCE")
        values.append(result.cpu())
    return torch.cat(values) if values else torch.empty(0)


def check_identity(actual, reference, key):
    for field in ("branch", "pyramid_level", "target_gt_idx"):
        if actual[field] != reference[field]:
            raise AssertionError(f"Permanent identity {field} differs for {key}")
    if "box_xyxy" in reference and actual["box_xyxy"] != reference["box_xyxy"]:
        raise AssertionError(f"Historical prediction-box support differs for {key}")


def run(args):
    cfg = resolve_runtime_config(load_json(args.config))
    if int(cfg["epochs"]) != FINAL_EPOCH or int(cfg["fit_eval_images"]) != 1000 or int(cfg["bootstrap"]) != 5000:
        raise ValueError("Fixed epoch12 / first1000 fit images / 5000 bootstrap must not change")
    if set(cfg["arms"]) != set(ARMS):
        raise ValueError("Protocol requires U/Q/P arms")
    if not args.smoke and not args.checkpoints:
        raise ValueError("Formal evaluation requires fixed final checkpoints")
    root, out = Path(cfg["server_root"]), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "PER_CANDIDATE.jsonl"
    if rows_path.exists() or (out / "COMPLETE.json").exists():
        raise RuntimeError("Use an independent evaluation Run; preserve partial outputs")
    if not args.smoke:
        receipt = load_json(Path(cfg["assets"]) / "COMPLETE.json")
        if receipt.get("complete") is not True:
            raise ValueError("Full immutable asset population is not complete")
    chosen = selected_index(load_json(Path(cfg["assets"]) / "INDEX.json"), args.smoke)
    dump(out / "EVALUATION_INDEX.json", chosen)
    baseline_path = Path(cfg.get("baseline_rows", root / "data" / "BASELINE_PER_CANDIDATE.jsonl"))
    baseline = historical_rows(baseline_path, chosen, "A")
    native_path = Path(cfg.get("historical_native_rows", root / "data" / "HISTORICAL_JOINT_PER_CANDIDATE.jsonl"))
    historical_native = historical_rows(native_path, chosen, "N", required=False)
    if historical_native is not None:
        if set(historical_native) != set(baseline):
            raise AssertionError("Historical native population does not match original baseline")
        enable_historical_native()
    replay = FrozenReplay(cfg)
    channels = replay.feature_channels
    checkpoints = load_json(args.checkpoints) if args.checkpoints else {}
    models, counts, checkpoint_hashes, fit_selections = {}, {}, {}, {}
    wanted_fit = {int(row["image_id"]) for row in chosen["fit"] if int(row["n"]) > 0}
    for mode in ARMS:
        setup(cfg["seed"])
        model = JointPointCoefficientReadout(replay.native_cv4, channels, mode, cfg).to(replay.device).float().eval()
        if not args.smoke:
            checkpoint = torch.load(checkpoints[mode], map_location="cpu", weights_only=False)
            if checkpoint["mode"] != mode or int(checkpoint["epoch"]) != FINAL_EPOCH:
                raise ValueError(f"Only the fixed final {mode} epoch12 is allowed")
            if list(checkpoint["feature_channels"]) != channels:
                raise ValueError("Checkpoint feature-channel identity changed")
            if resolve_runtime_config(checkpoint["config"]) != cfg:
                raise ValueError(f"Checkpoint configuration mismatch: {mode}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            checkpoint_hashes[mode] = sha256(checkpoints[mode])
            manifest_path = Path(checkpoints[mode]).parent / "SELECTION_MANIFEST.jsonl"
            if sha256(manifest_path) != checkpoint["operator_identity"]["manifest_sha256"]:
                raise AssertionError(f"{mode}: training selection-manifest hash differs")
            fit_selections[mode] = {}
            with manifest_path.open(encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    iid = int(item["image_id"])
                    if iid in wanted_fit:
                        if iid in fit_selections[mode]:
                            raise AssertionError("Duplicate image in fixed training operator manifest")
                        fit_selections[mode][iid] = item
            if set(fit_selections[mode]) != wanted_fit:
                raise AssertionError("Training operator manifest does not cover evaluated fit population")
        counts[mode] = model.parameter_counts()
        model.requires_grad_(False)
        models[mode] = model
    if any(counts[mode] != counts["U"] for mode in ARMS):
        raise AssertionError("U/Q/P trainable architecture capacities differ")
    coco = {"fit": COCO(str(cfg.get("annotations_train", root / "data" / "instances_train2017.json"))),
            "val": COCO(str(cfg.get("annotations_val", root / "data" / "instances_val2017.json")))}
    coco["dev"] = coco["fit"]
    expected, bce_summary, rows, visited = {}, {}, [], set()
    started = time.monotonic()
    maximum_baseline_error, maximum_initial_error = 0., 0.
    invalid_counts = {mode: 0 for mode in ARMS}
    for split, items in chosen.items():
        expected[split] = dict(planned_images=len(items), effective_images=sum(int(x["n"]) > 0 for x in items),
            candidates=sum(int(x["n"]) for x in items),
            no_positive_image_ids=[int(x["image_id"]) for x in items if not int(x["n"])])
        sums = {arm: 0. for arm in ("A", *ARMS, *(("N",) if historical_native is not None else ()))}
        seen = 0
        for position, entry in enumerate(items):
            if not int(entry["n"]):
                continue
            image = load_asset(cfg, int(entry["image_id"]))
            if image["split"] != split or len(image["rows"]) != int(entry["n"]):
                raise AssertionError("Frozen asset/INDEX candidate population differs")
            for row in image["rows"]:
                key = identity(row)
                if key in visited or key not in baseline:
                    raise AssertionError(f"Duplicate or missing baseline candidate: {key}")
                row["initial_iou"] = float(baseline[key]["iou_A"])
                visited.add(key)
            features = replay.replay([image])
            coefficients = {"A": image["c0"].to(replay.device)}
            operator_audits = []
            with torch.no_grad():
                for mode in ARMS:
                    selected, operator_audit = evaluation_selection(image, mode, replay.device)
                    if split == "fit" and not args.smoke:
                        previous = fit_selections[mode][int(image["image_id"])]
                        if (operator_audit["K_float32_sha256"] != previous["K_fp32_sha256"]
                                or selected["indices"].cpu().tolist() != previous["indices"]
                                or selected["valid"].cpu().tolist() != previous["valid"]
                                or image["_asset_integrity"]["compressed_sha256"] != previous["compressed_asset_sha256"]):
                            raise AssertionError(f"{mode}/{image['image_id']}: operator replay differs from training")
                        operator_audit["training_selection_exact"] = True
                    coefficients[mode] = models[mode](features, [selected])[0]
                    operator_audits.append(operator_audit)
                    invalid_counts[mode] += operator_audit["invalid_candidates_retained"]
                    if args.smoke:
                        error = float((coefficients[mode].cpu() - image["c0"]).abs().max())
                        maximum_initial_error = max(maximum_initial_error, error)
                        torch.testing.assert_close(coefficients[mode].cpu(), image["c0"], atol=3e-5, rtol=3e-5)
            gpu = gpu_image(image)
            bce = {arm: official_bce_values(gpu, value) for arm, value in coefficients.items()}
            decoded = evaluate_image(gpu, coefficients, coco[split])
            if len(decoded) != len(image["rows"]):
                raise AssertionError("Decoder changed the fixed candidate population")
            for j, row in enumerate(decoded):
                key = identity(row)
                original = baseline[key]
                check_identity(row, original, key)
                error = abs(row["iou_A"] - float(original["iou_A"]))
                maximum_baseline_error = max(maximum_baseline_error, error)
                if error > 1e-6 or row["mask75_A"] != original["mask75_A"]:
                    raise AssertionError(f"Original normal-decoding baseline differs: {key}, error={error}")
                for arm in coefficients:
                    row[f"bce_{arm}"] = float(bce[arm][j])
                if historical_native is not None:
                    old = historical_native[key]
                    check_identity(row, old, key)
                    if abs(row["iou_A"] - float(old["iou_A"])) > 1e-6:
                        raise AssertionError("Historical native baseline does not match this A")
                    for metric in METRICS:
                        row[f"{metric}_N"] = old[f"{metric}_N"]
                row["original_operator_valid"] = bool(image["_operator"]["valid"][j])
                row["original_tiny_box_roi_fallback"] = bool(image["_operator"]["tiny_box_roi_fallback"][j])
            append_rows(rows_path, decoded)
            append_rows(out / "OPERATOR_REPLAY_AUDIT.jsonl", operator_audits)
            rows.extend(decoded)
            for arm in sums:
                sums[arm] += sum(float(row[f"bce_{arm}"]) for row in decoded)
            seen += len(decoded)
            if position % 25 == 0 or position + 1 == len(items):
                progress = dict(stage="evaluation", split=split, planned_images_processed=position+1,
                    planned_images=len(items), candidates=seen, elapsed_s=time.monotonic()-started)
                dump(out / "PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
            del image, features, selected, coefficients, gpu, bce, decoded
        if seen != expected[split]["candidates"]:
            raise AssertionError(f"Evaluation population changed in {split}")
        bce_summary[split] = dict(candidates=seen, candidate_mean={a: v/max(1, seen) for a, v in sums.items()})
    if visited != set(baseline):
        raise AssertionError("Evaluation did not cover exactly its immutable baseline identities")
    replay.assert_unchanged()
    dump(out / "FINAL_BCE.json", dict(schema="fixed-checkpoint-official-bce-v1", splits=bce_summary,
        fit_scope="first1000 planned fit images; not full fit", full_fit_re_evaluated=False))
    info = dict(smoke=args.smoke, checkpoints=checkpoints, checkpoint_sha256=checkpoint_hashes,
        checkpoint_rule=cfg["checkpoint_rule"], feature_channels=channels, model_counts=counts,
        baseline_rows=str(baseline_path), baseline_rows_sha256=sha256(baseline_path),
        baseline_candidates_checked=len(visited),
        baseline_iou_max_error=maximum_baseline_error,
        initialization_coefficient_max_error=maximum_initial_error if args.smoke else None,
        historical_native_joined=historical_native is not None,
        historical_native_rows=str(native_path) if historical_native is not None else None,
        historical_native_rows_sha256=sha256(native_path) if historical_native is not None else None,
        historical_native_scope="same IDs; historical configuration; not a same-run additional training arm",
        invalid_operator_candidates_retained=invalid_counts,
        original_prefix_buffers_unchanged=True, source_model=replay.import_info,
        selection_uses_frozen_original_P_and_c0=True, prototype_and_labels_are_original_assets=True,
        fit_operator_replay_compared_to_training_manifest=not args.smoke,
        evaluation_limits=SMOKE_LIMITS if args.smoke else EVALUATION_LIMITS,
        full_fit_re_evaluated=False, no_new_blind_test=True, no_COCO_AP=True)
    dump(out / "EVALUATION_AUDIT.json", info)
    result = summarize(rows, out, expected_counts=expected, seed=20261003,
        bootstrap=100 if args.smoke else int(cfg["bootstrap"]), run_info=info)
    if result["audit"]["issues"]:
        raise AssertionError("Evaluation integrity audit failed; preserve partial artifacts and stop")
    dump(out / "COMPLETE.json", dict(completed=True, passed=bool(args.smoke),
        code_checks_passed=True, smoke=args.smoke,
        method_passed=result["assessment"]["method_passed"], decision=result["assessment"]["decision"],
        candidates=len(rows), elapsed_s=time.monotonic()-started, automatic_followup=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        out = Path(args.out)
        dump(out / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        if not (out / "COMPLETE.json").exists():
            dump(out / "COMPLETE.json", dict(completed=False, passed=False, error=str(exc), automatic_followup=False))
        raise


if __name__ == "__main__":
    main()
