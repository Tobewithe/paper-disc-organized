"""Fixed epoch12 native-plus-evidence joint readouts, unchanged official candidates and decode.

Run only on the authorized GPU host. This evaluator reconstructs native heads
from the original checkpoint but runs no detector forward and never trains. The smoke evaluates identity-initialized heads, not checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops

from runtime_utils import setup, load, gpu_image, dump
from train_joint import read_training_image, feed, feature_channels, resolve_config, make_model
from evaluate import evaluate_image, append_rows, summarize


ARMS = ("A", "N", "D", "S")
FINAL_EPOCH = 12
EVALUATION_LIMITS = {"fit": 1000, "dev": 1000, "val": 2000}
SMOKE_LIMITS = {"fit": 8, "dev": 3, "val": 3}
HISTORICAL_EVALUATION_RUN = "RUN_976ee94db0ed49d894567e0fc0151224"


def donor_boxes(x):
    """Old donor rule: class preference, geometry distance, raw-ID tie break.

    GT IDs are read only after a donor has been selected from predictions.
    Neither candidate membership nor final mask support changes.
    """
    boxes = x["boxes"]
    result = boxes.clone()
    classes = x["predicted_classes"].tolist()
    geometry = []
    for box in boxes.tolist():
        w, h = box[2]-box[0], box[3]-box[1]
        geometry.append((w*h, w/h) if w > 0 and h > 0 else None)
    for i, row in enumerate(x["rows"]):
        valid = [j for j in range(len(boxes)) if j != i and geometry[j] is not None]
        same = [j for j in valid if classes[j] == classes[i]]
        pool = same or valid
        row["wrong_box_available"] = bool(pool and geometry[i] is not None)
        if not row["wrong_box_available"]:
            continue
        area, ratio = geometry[i]
        j = min(pool, key=lambda j: (
            abs(math.log(geometry[j][0]/area))+abs(math.log(geometry[j][1]/ratio)),
            int(x["raw_ids"][j])))
        result[i] = boxes[j]
        row.update(wrong_box_donor_annotation_id=int(x["rows"][j]["annotation_id"]),
                   wrong_box_donor_raw_id=int(x["raw_ids"][j]),
                   wrong_box_donor_image_id=int(row["image_id"]))
    return result


def identity(row):
    return (str(row["split"]), int(row["image_id"]),
            int(row["annotation_id"]), int(row["raw_id"]))


def evaluation_index(index, smoke):
    """Slice by immutable index order BEFORE testing positive count/quality."""
    limits = SMOKE_LIMITS if smoke else EVALUATION_LIMITS
    result = {}
    for split, limit in limits.items():
        items = index.get(split)
        if not isinstance(items, list):
            raise ValueError(f"Missing frozen {split} INDEX list")
        if not smoke and len(items) < limit:
            raise ValueError(f"Formal {split} evaluation requires {limit} planned images")
        result[split] = items[:limit]
        ids = [int(item["image_id"]) for item in result[split]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate image IDs in frozen {split} evaluation list")
    return result


def load_historical_baseline(cfg, selected):
    default = Path(cfg["cache"]).parent / "runs" / HISTORICAL_EVALUATION_RUN / "PER_CANDIDATE.jsonl"
    source = Path(cfg.get("baseline_rows", default))
    if not source.exists():
        raise FileNotFoundError(f"Historical same-ID baseline is required: {source}")
    wanted = {(split, int(item["image_id"])) for split, items in selected.items() for item in items}
    rows = {}
    with source.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (str(row["split"]), int(row["image_id"])) not in wanted:
                continue
            key = identity(row)
            if key in rows:
                raise ValueError(f"Historical duplicate permanent identity: {key}")
            value = float(row["iou_A"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Historical nonfinite/invalid baseline: {key}")
            rows[key] = {"iou_A": value, "mask75_A": int(row["mask75_A"])}
    expected = sum(int(item["n"]) for items in selected.values() for item in items)
    if len(rows) != expected:
        raise ValueError(f"Historical baseline has {len(rows)} rows for {expected} frozen candidates")
    return source, rows


def load_frozen_projection(cfg, selected, baseline):
    """Join frozen-head S by full scientific identity, never rerun its model."""
    source = Path(cfg["frozen_projection_rows"])
    if not source.exists():
        raise FileNotFoundError(f"Historical frozen projection comparison required: {source}")
    wanted = {(split, int(item["image_id"])) for split, items in selected.items() for item in items}
    rows = {}
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (str(row["split"]), int(row["image_id"])) not in wanted:
            continue
        key = identity(row)
        if key in rows or key not in baseline:
            raise ValueError(f"Historical frozen-S identity mismatch: {key}")
        if row.get("branch") != "one2one":
            raise ValueError(f"Historical frozen-S non-one2one identity: {key}")
        if abs(float(row["iou_A"])-baseline[key]["iou_A"]) > 1e-6:
            raise ValueError(f"Historical frozen-S baseline IoU differs: {key}")
        for metric in ("iou", "coverage", "mask75", "auc", "fpr", "bce"):
            if f"{metric}_S" not in row:
                raise ValueError(f"Historical frozen-S missing {metric}: {key}")
        for metric in ("iou", "coverage", "mask75", "bce"):
            if not math.isfinite(float(row[f"{metric}_S"])):
                raise ValueError(f"Historical frozen-S invalid {metric}: {key}")
        rows[key] = row
    if set(rows) != set(baseline):
        raise ValueError("Frozen-S historical join is not exact for evaluated cohort")
    return source, rows


@torch.no_grad()
def official_bce_values(x, coefficients, chunk_size=8):
    """Unchanged official instance loss, retain each candidate for aggregation.

    Uses prepared polygon owner labels and GT target boxes exactly as training;
    mask-quality metrics separately use original COCO annToMask and pred boxes.
    """
    shape = tuple(map(int, x.get("input_shape", (640, 640))))
    if shape != (640, 640):
        raise ValueError("The frozen official loss protocol requires 640x640")
    proto = F.interpolate(x["proto"][None].float(), shape,
                          mode="bilinear", align_corners=False)[0]
    pieces = []
    for lo in range(0, len(coefficients), chunk_size):
        hi = min(lo+chunk_size, len(coefficients))
        gt = (x["masks"][None] == (x["owners"][lo:hi]+1)[:, None, None]).float()
        boxes = x["target_boxes"][lo:hi]
        area = ((boxes[:, 2:]-boxes[:, :2])/640).prod(1)
        if not bool((area > 0).all()):
            raise ValueError("Invalid frozen GT target-box loss area")
        logits = torch.einsum("in,nhw->ihw", coefficients[lo:hi].float(), proto)
        pixel_loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        values = ops.crop_mask(pixel_loss, boxes).mean((1, 2))/area*float(x["segmentation_gain"])
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite official candidate BCE")
        pieces.append(values.cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoints", help="JSON N/S/D -> fixed final epoch12 checkpoint paths")
    parser.add_argument("--index")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = resolve_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    if int(cfg["epochs"]) != FINAL_EPOCH or int(cfg.get("fit_eval_images", 1000)) != 1000:
        raise ValueError("Fixed epoch12 and fixed first1000 fit evaluation must not change")
    if not args.smoke and not args.checkpoints:
        parser.error("Formal evaluation requires --checkpoints")
    setup(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "PER_CANDIDATE.jsonl"
    if rows_path.exists() or (out / "COMPLETE.json").exists():
        raise RuntimeError("Use a fresh independent evaluation Run; partial candidate rows are preserved")
    index_path = Path(args.index) if args.index else Path(cfg["cache"]) / "INDEX.json"
    full_index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    selected_index = evaluation_index(full_index, args.smoke)
    dump(out / "EVALUATION_INDEX.json", selected_index)
    baseline_source, baseline = load_historical_baseline(cfg, selected_index)
    frozen_source, frozen_rows = load_frozen_projection(cfg, selected_index, baseline)
    first = next((item for items in selected_index.values() for item in items if int(item["n"]) > 0), None)
    if first is None:
        raise ValueError("No fixed candidates in evaluation selection")
    initial = read_training_image(cfg, int(first["image_id"]))
    channels = feature_channels(initial)
    del initial
    paths = json.loads(Path(args.checkpoints).read_text(encoding="utf-8-sig")) if args.checkpoints else {}
    models, model_counts = {}, {}
    for mode in ("N", "S", "D"):
        setup(cfg["seed"])
        model = make_model(channels, mode, cfg).cuda().float().eval()
        if not args.smoke:
            checkpoint = load(paths[mode])
            if checkpoint["mode"] != mode or int(checkpoint["epoch"]) != FINAL_EPOCH:
                raise ValueError(f"{mode}: only fixed final epoch12 may be evaluated")
            if list(checkpoint["feature_channels"]) != channels or checkpoint["config"] != cfg:
                raise ValueError(f"{mode}: checkpoint config or feature-channel mismatch")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        model_counts[mode] = model.parameter_counts()
        model.requires_grad_(False)
        models[mode] = model
    coco = {"fit": COCO(cfg["annotations_train"]), "val": COCO(cfg["annotations_val"])}
    coco["dev"] = coco["fit"]
    expected, bce_summary, all_rows, visited = {}, {}, [], set()
    started = time.monotonic()
    max_baseline_error = 0.0
    max_identity_error = 0.0
    initialization_decode = {arm: dict(padded_pixel_changes=0, original_pixel_changes=0,
                                       candidates_with_original_pixel_changes=0,
                                       max_original_iou_change=0.0, mask75_disagreements=0)
                             for arm in ("N", "S", "D", "SW")}
    invalid_operators = 0
    for split, items in selected_index.items():
        expected[split] = dict(planned_images=len(items),
                               effective_images=sum(int(item["n"]) > 0 for item in items),
                               candidates=sum(int(item["n"]) for item in items))
        loss_sums = {arm: 0.0 for arm in (*ARMS, "SF")}
        seen = 0
        for position, item in enumerate(items):
            if not int(item["n"]):
                continue
            iid = int(item["image_id"])
            x = read_training_image(cfg, iid)
            if x["split"] != split or len(x["rows"]) != int(item["n"]):
                raise ValueError(f"Frozen INDEX/cache identity or candidate count differs: {split}/{iid}")
            if feature_channels(x) != channels:
                raise ValueError(f"Frozen feature channels differ: {iid}")
            for row in x["rows"]:
                key = identity(row)
                if key in visited or key not in baseline:
                    raise ValueError(f"Duplicate or missing historical candidate identity: {key}")
                row["initial_iou"] = baseline[key]["iou_A"]
                visited.add(key)
            wrong_boxes = donor_boxes(x)
            features, selection = feed([x])
            own = selection[0]
            wrong_selection = [dict(own, sampling_boxes=wrong_boxes.cuda())]
            for key in ("h0", "raw_ids", "boxes", "c0", "A", "K", "G", "valid"):
                if wrong_selection[0][key] is not own[key]:
                    raise AssertionError(f"SW unexpectedly replaced its own {key}")
            with torch.no_grad():
                coefficients = {arm: model(features, selection)[0] for arm, model in models.items()}
                coefficients["A"] = x["c0"].cuda()
                coefficients["SW"] = models["S"](features, wrong_selection)[0]
            if args.smoke:
                for arm, value in coefficients.items():
                    error = float((value.cpu()-x["c0"]).abs().max())
                    max_identity_error = max(max_identity_error, error)
                    # Re-running native branches may differ in FP32 operation order.
                    # Cached A itself remains exact; report near-zero mask flips below.
                    torch.testing.assert_close(value.cpu(), x["c0"], atol=3e-5, rtol=3e-5)
            gx = gpu_image(x)
            bce = {arm: official_bce_values(gx, value) for arm, value in coefficients.items()}
            decoded = evaluate_image(gx, coefficients, coco[split])
            if len(decoded) != len(x["rows"]):
                raise AssertionError("Decoder changed frozen candidate count")
            for j, row in enumerate(decoded):
                key = identity(row)
                error = abs(row["iou_A"]-baseline[key]["iou_A"])
                max_baseline_error = max(max_baseline_error, error)
                if error > 1e-6 or row["mask75_A"] != baseline[key]["mask75_A"]:
                    raise AssertionError(f"Historical same-ID baseline differs at {key}: {error}")
                for arm in coefficients:
                    row[f"bce_{arm}"] = float(bce[arm][j])
                historical = frozen_rows[key]
                for field in ("branch", "pyramid_level", "target_gt_idx"):
                    if row[field] != historical[field]:
                        raise ValueError(f"Frozen-S permanent identity field {field} differs: {key}")
                if row["box_xyxy"] != historical["box_xyxy"]:
                    raise ValueError(f"Frozen-S prediction box differs: {key}")
                for metric in ("iou", "coverage", "mask75", "auc", "fpr", "bce"):
                    row[f"{metric}_SF"] = historical[f"{metric}_S"]
                if args.smoke:
                    for arm, summary in initialization_decode.items():
                        summary["padded_pixel_changes"] += int(row[f"padded_pixel_changes_{arm}_vs_A"])
                        diff = int(row[f"original_pixel_changes_{arm}_vs_A"])
                        summary["original_pixel_changes"] += diff
                        summary["candidates_with_original_pixel_changes"] += int(diff > 0)
                        summary["max_original_iou_change"] = max(summary["max_original_iou_change"], abs(row[f"iou_{arm}"]-row["iou_A"]))
                        summary["mask75_disagreements"] += int(row[f"mask75_{arm}"] != row["mask75_A"])
                row["operator_valid"] = bool(x["_operator"]["valid"][j])
                row["operator_tiny_box_roi_fallback"] = bool(x["_operator"]["tiny_box_roi_fallback"][j])
            invalid_operators += int((~x["_operator"]["valid"]).sum())
            append_rows(rows_path, decoded)
            all_rows.extend(decoded)
            seen += len(decoded)
            for arm in ARMS:
                loss_sums[arm] += float(bce[arm].double().sum())
            loss_sums["SF"] += sum(float(row["bce_SF"]) for row in decoded)
            if position % 25 == 0 or position+1 == len(items):
                state = dict(stage="evaluation", split=split, planned_images_processed=position+1,
                             planned_images=len(items), candidates=seen, elapsed_s=time.monotonic()-started)
                dump(out / "PROGRESS.json", state)
                print(json.dumps(state), flush=True)
            del x, gx, features, selection, own, wrong_selection, coefficients, decoded, bce
        if seen != expected[split]["candidates"]:
            raise AssertionError(f"Candidate coverage differs for {split}")
        bce_summary[split] = dict(candidates=seen, planned_images=len(items),
                                  candidate_mean={arm: value/max(1, seen) for arm, value in loss_sums.items()})
    if visited != set(baseline):
        raise AssertionError("Historical baseline join did not cover exactly the evaluated cohort")
    dump(out / "FINAL_BCE.json", dict(
        schema="fixed_checkpoint_evaluation_subset_bce_v1", splits=bce_summary,
        fit_scope="fixed first1000 planned fit images (smoke first8), not full fit",
        full_fit_re_evaluated=False,
        training_trajectory_is_not_final_checkpoint_population_loss=True))
    info = dict(smoke=args.smoke, checkpoints={} if args.smoke else paths,
                checkpoint_rule=cfg["checkpoint_rule"], feature_channels=channels, model_counts=model_counts,
                baseline_rows=str(baseline_source), baseline_candidates_checked=len(visited),
                frozen_projection_rows=str(frozen_source), frozen_projection_joined_candidates=len(frozen_rows),
                frozen_comparison_scope="same IDs, historical frozen-S versus new joint-S; no old-model replay; not same optimizer process",
                initialization_coefficient_atol=3e-5, initialization_coefficient_rtol=3e-5,
                initialization_decode_changes=initialization_decode if args.smoke else None,
                baseline_iou_max_error=max_baseline_error, identity_coefficient_max_error=max_identity_error if args.smoke else None,
                invalid_operator_candidates_retained=invalid_operators, full_fit_re_evaluated=False,
                evaluation_limits=SMOKE_LIMITS if args.smoke else EVALUATION_LIMITS,
                evaluation_scope="fixed index order; historically studied val; not a new blind test or full-output AP")
    dump(out / "EVALUATION_AUDIT.json", info)
    bootstrap = min(100, int(cfg["bootstrap"])) if args.smoke else int(cfg["bootstrap"])
    result = summarize(all_rows, out, expected_counts=expected, seed=20261003,
                       bootstrap=bootstrap, run_info=info)
    if result["audit"]["issues"]:
        raise AssertionError("Evaluation completeness audit failed; preserve partial outputs and stop")
    dump(out / "COMPLETE.json", dict(completed=True, smoke=args.smoke,
                                     method_passed=result["assessment"]["method_passed"],
                                     decision=result["assessment"]["decision"], candidates=len(all_rows),
                                     elapsed_s=time.monotonic()-started, automatic_followup=False))


if __name__ == "__main__":
    main()
