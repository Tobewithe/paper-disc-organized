"""Cache frozen native YOLO features and original COCO ownership supervision.

GT is used only to match sparse training instances and construct supervision.
The prediction-derived neighbor graph is built independently before reading GT.
Large image tensors stay on the execution host, with byte hashes in the receipt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from frozen_io import (FrozenYOLO, box_iou_xyxy, dump_json, predicted_neighbors,
                       sha256, transformed_box, transformed_mask)

CACHE_VERSION = "triflow_frozen_cache_v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def select_training_instances(extracted, annotations, max_instances=12, candidate_rows=64):
    """Class-free matching; one representative per GT before repeated matches."""
    import numpy as np
    annotations = sorted([a for a in annotations if not a.get("iscrowd", 0)], key=lambda a: int(a["id"]))
    if not annotations:
        return []
    gt_boxes = np.stack([transformed_box(a["bbox"], extracted["letterbox"]) for a in annotations])
    boxes = extracted["boxes"].numpy()
    scores = extracted["scores"].numpy()
    raw_ids = extracted["raw_indices"].numpy()
    overlaps = box_iou_xyxy(boxes, gt_boxes)
    seen_raw, groups = set(), {}
    for row in [i for i in range(len(boxes)) if scores[i] > .001][:candidate_rows]:
        if int(raw_ids[row]) in seen_raw:
            continue
        seen_raw.add(int(raw_ids[row]))
        # Sorted annotation ids and numpy's first argmax give explicit stable ties.
        index = int(overlaps[row].argmax())
        if overlaps[row, index] < .5:
            continue
        matched = {"output_row": row, "raw_index": int(raw_ids[row]),
                   "annotation_id": int(annotations[index]["id"]), "box_iou": float(overlaps[row, index]),
                   "predicted_class": int(extracted["classes"][row]),
                   "gt_category_id": int(annotations[index]["category_id"])}
        groups.setdefault(matched["annotation_id"], []).append(matched)
    # Round-robin in first-output-row order is fixed, and does not optimize losses.
    ordered = sorted(groups.values(), key=lambda group: group[0]["output_row"])
    result = []
    depth = 0
    while len(result) < max_instances and any(depth < len(group) for group in ordered):
        for group in ordered:
            if depth < len(group) and len(result) < max_instances:
                result.append(group[depth])
        depth += 1
    return result


def build_training_targets(extracted, selected, all_annotations, coco, neighbor_rows, neighbor_valid):
    import numpy as np
    import torch
    height, width = extracted["P"].shape[-2:]
    ordinary = sorted([a for a in all_annotations if not a.get("iscrowd", 0)], key=lambda a: int(a["id"]))
    masks = {int(a["id"]): transformed_mask(coco.annToMask(a), extracted["letterbox"], (height, width))
             for a in ordinary}
    union_all = np.zeros((height, width), dtype=bool)
    crowd = np.zeros_like(union_all)
    for mask in masks.values():
        union_all |= mask
    for annotation in all_annotations:
        if annotation.get("iscrowd", 0):
            crowd |= transformed_mask(coco.annToMask(annotation), extracted["letterbox"], (height, width))
    targets = []
    for match in selected:
        self_mask = masks[match["annotation_id"]]
        if not self_mask.any():
            # Distance to an empty quantized Self is undefined. Preserve this
            # exclusion in the image receipt instead of inventing zero distance.
            continue
        # The union must be recomputed without Self, then Self wins in overlaps.
        neighbor = np.zeros_like(union_all)
        for annotation_id, mask in masks.items():
            if annotation_id != match["annotation_id"]:
                neighbor |= mask
        neighbor &= ~self_mask
        ignored = crowd & ~self_mask
        background = ~(union_all | crowd)
        row = match["output_row"]
        targets.append(match | {
            "self_mask": torch.from_numpy(self_mask.copy()),
            "neighbor_mask": torch.from_numpy(neighbor.copy()),
            "background_mask": torch.from_numpy(background.copy()),
            "ignore_mask": torch.from_numpy(ignored.copy()),
            "valid_mask": torch.from_numpy((~ignored).copy()),
            "neighbor_rows": torch.from_numpy(neighbor_rows[row].copy()),
            "neighbor_valid": torch.from_numpy(neighbor_valid[row].copy()),
            "phi_neighbor_valid": bool(neighbor.any() and neighbor_valid[row].any()),
            "gt_neighbor_nonempty": bool(neighbor.any()),
            "self_priority": True,
        })
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images-list", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-instances", type=int, default=12)
    args = parser.parse_args()
    root, images_list, annotations = Path(args.root).resolve(), Path(args.images_list).resolve(), Path(args.annotations).resolve()
    run = root / "runs" / args.run_id
    if (run / "CACHE_COMPLETE.json").exists():
        raise FileExistsError("Completed cache is immutable; use another Run ID")
    cache_dir = run / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cache_dir.glob("*.pt"))
    if existing:
        raise FileExistsError("Partial cache retained; retries need a new Run ID")
    source = run / "source"
    source.mkdir(exist_ok=True)
    for name in ("frozen_io.py", "prepare_cache.py", "triflow_model.py"):
        current = Path(__file__).with_name(name)
        target = source / name
        if target.exists() and sha256(target) != sha256(current):
            raise ValueError("Run source snapshot differs from executing source")
        if not target.exists():
            shutil.copy2(current, target)
    if (root / "PROTOCOL.md").is_file() and not (source / "PROTOCOL.md").exists():
        shutil.copy2(root / "PROTOCOL.md", source / "PROTOCOL.md")
    started = now()
    began = time.monotonic()
    extractor = None
    try:
        import numpy as np
        import torch
        from pycocotools.coco import COCO
        from triflow_model import TriFlowConfig, build_ownership_targets
        coco = COCO(str(annotations))
        by_filename = {image["file_name"]: int(iid) for iid, image in coco.imgs.items()}
        paths = [Path(line.strip()).resolve() for line in images_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if args.limit:
            paths = paths[:args.limit]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("Training image list is empty or contains duplicate paths")
        extractor = FrozenYOLO(args.weights, args.vendor, args.device)
        index = []
        instance_count, ignored_pixels, phi_n_available, with_prediction_neighbor, empty_self_exclusions = 0, 0, 0, 0, 0
        unique_ids = set()
        dimensions = None
        for position, path in enumerate(paths, 1):
            if path.name not in by_filename:
                raise ValueError(f"Training image not in original COCO annotation file: {path}")
            iid = by_filename[path.name]
            if iid in unique_ids:
                raise ValueError("Duplicate COCO image identity")
            unique_ids.add(iid)
            extracted = extractor.extract(path)
            if extracted["original_shape"] != [coco.imgs[iid]["height"], coco.imgs[iid]["width"]]:
                raise ValueError(f"COCO/image shape mismatch: {iid}")
            extracted["image_id"] = iid
            graph_rows, graph_valid = predicted_neighbors(extracted["boxes"].numpy(), extracted["raw_indices"].numpy(),
                                                        extracted["scores"].numpy(), k=2)
            extracted["predicted_neighbor_rows"] = torch.from_numpy(graph_rows)
            extracted["predicted_neighbor_valid"] = torch.from_numpy(graph_valid)
            all_annotations = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))
            selected = select_training_instances(extracted, all_annotations, args.max_instances)
            targets = build_training_targets(extracted, selected, all_annotations, coco, graph_rows, graph_valid)
            retained_ids = {t["annotation_id"] for t in targets}
            extracted["empty_self_grid_exclusions"] = [t for t in selected if t["annotation_id"] not in retained_ids]
            empty_self_exclusions += len(extracted["empty_self_grid_exclusions"])
            if targets:
                ownership = build_ownership_targets(
                    torch.stack([t["self_mask"] for t in targets]),
                    torch.stack([t["neighbor_mask"] for t in targets]),
                    torch.stack([t["neighbor_valid"] for t in targets]),
                    background_masks=torch.stack([t["background_mask"] for t in targets]),
                    valid_pixels=torch.stack([t["valid_mask"] for t in targets]),
                    max_distance=TriFlowConfig().max_distance)
                for instance_index, target_record in enumerate(targets):
                    target_record["ownership_targets"] = {k: v[instance_index].cpu().clone() for k, v in ownership.items()}
            extracted["training_targets"] = targets
            extracted["cache_version"] = CACHE_VERSION
            extracted["base_weights_sha256"] = sha256(args.weights)
            extracted["coco_annotation_sha256"] = sha256(annotations) if position == 1 else index[0]["annotation_sha256"]
            target = cache_dir / f"{iid:012d}.pt"
            temp = target.with_suffix(".pt.tmp")
            torch.save(extracted, temp)
            temp.replace(target)
            dimensions = {"prototype_channels": int(extracted["P"].shape[0]),
                          "prototype_grid": list(extracted["P"].shape[-2:]),
                          "feature_channels": int(extracted["F"].shape[0]),
                          "feature_grid": list(extracted["F"].shape[-2:]),
                          "instance_hidden_channels": int(extracted["h"].shape[-1])}
            item = {"image_id": iid, "image_path": str(path), "image_sha256": extracted["image_sha256"],
                    "input_sha256": extracted["input_sha256"], "cache_file": str(target.relative_to(run)),
                    "sha256": sha256(target), "bytes": target.stat().st_size,
                    "instances": len(targets), "annotation_sha256": extracted["coco_annotation_sha256"],
                    "native_topk_rows": int(len(extracted["scores"])), "native_replay_exact": True}
            index.append(item)
            instance_count += len(targets)
            ignored_pixels += sum(int(t["ignore_mask"].sum()) for t in targets)
            phi_n_available += sum(int(t["phi_neighbor_valid"]) for t in targets)
            with_prediction_neighbor += sum(int(t["neighbor_valid"].any()) for t in targets)
            with (run / "CACHE_IMAGES.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, allow_nan=False) + "\n")
            print(f"CACHE {position}/{len(paths)} images instances={instance_count} bytes={sum(i['bytes'] for i in index)} elapsed_s={time.monotonic()-began:.1f}", flush=True)
        integrity = extractor.verify_frozen()
        receipt = {
            "cache_version": CACHE_VERSION, "status": "completed", "started_at": started, "completed_at": now(),
            "base_weights": str(Path(args.weights).resolve()), "base_weights_sha256": sha256(args.weights),
            "annotations": str(annotations), "annotation_sha256": sha256(annotations),
            "images_list": str(images_list), "images_list_sha256": sha256(images_list),
            "images": len(paths), "image_ids": [i["image_id"] for i in index], "instances": instance_count,
            "dimensions": dimensions, "cache_files": index, "cache_bytes": sum(i["bytes"] for i in index),
            "frozen_integrity": integrity, "all_native_replay_exact": True, "all_cache_float_tensors": "finite FP32",
            "selection": {"native_confidence": .001, "first_native_post_conf_rows": 64,
                          "max_instances_per_image": args.max_instances, "class_free_best_gt_box_iou": .5,
                          "tie": "annotation_id ascending", "deduplicate_raw": True,
                          "order": "round-robin matched GT groups ordered by first native output row"},
            "ownership": {"self_priority": True, "gt_neighbor": "union all other noncrowd GT minus Self",
                          "ignore": "union crowd minus Self", "background": "outside all ordinary GT and crowd",
                          "predicted_neighbor": "top2 positive box IoU, tie native output row; repeated raw id excluded; GT never used",
                          "phi_neighbor_supervised_instances": phi_n_available,
                          "predicted_neighbor_available_instances": with_prediction_neighbor,
                          "crowd_ignored_instance_pixels": ignored_pixels,
                          "empty_quantized_self_exclusions": empty_self_exclusions,
                          "neighbor_supervision_missing": "phi_N masked if GT Neighbor empty OR no eligible predicted neighbor"},
            "sources": {name: sha256(source/name) for name in ("prepare_cache.py", "frozen_io.py", "triflow_model.py")},
            "cache_artifact_return": "large cache tensors retained on execution host; compact receipt and hash list returned",
            "elapsed_seconds": time.monotonic()-began,
        }
        dump_json(run / "CACHE_RECEIPT.json", receipt)
        dump_json(run / "CACHE_COMPLETE.json", {"status": "completed", "receipt_sha256": sha256(run/"CACHE_RECEIPT.json"),
                                                "images": len(paths), "instances": instance_count})
    except Exception as exc:
        dump_json(run / "CACHE_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
        raise
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
