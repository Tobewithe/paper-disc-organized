"""Online declared 20,000-image COCO supervision with unchanged pilot semantics.

Only the current image's detached FP32 frozen features live in memory. No full
prototype or visual-feature dataset is materialized on disk. GT enters training
selection and targets only; the prediction-derived neighbor graph is made first.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from frozen_io import (OFFICIAL_SHA256, FrozenYOLO, box_iou_xyxy, dump_json,
                       predicted_neighbors, sha256, transformed_box,
                       transformed_mask)

STREAM_VERSION = "triflow_online_declared_train20k_v1"
# Original full train2017 JSON already authenticated in this research line.
COCO_TRAIN_ANNOTATION_SHA256 = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
PILOT_CACHE_VERSION = "triflow_frozen_cache_v1"
PILOT_SELECTION_SOURCE_SHA256 = "62579c9a38a9fefc44d469b4c8f283e64d9f72bde9891c8ecfc546f6d5ef8c5b"
DECLARED_TRAIN_IMAGES = 20000
SUBSET_SELECTOR_NAME = "sha256_triflow_train20k_seed0_v1"
SUBSET_SELECTOR_PREFIX = "triflow_train20k_seed0:"


def select_declared_train_ids(image_ids):
    """Selection depends only on all original IDs, never GT/image contents."""
    ids = [int(iid) for iid in image_ids]
    if len(ids) != 118287 or len(set(ids)) != 118287:
        raise ValueError("Declared subset selection requires the complete original 118287 unique image IDs")
    ranked = sorted(ids, key=lambda iid: (hashlib.sha256((SUBSET_SELECTOR_PREFIX+str(iid)).encode("utf-8")).digest(), iid))
    return sorted(ranked[:DECLARED_TRAIN_IMAGES])


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def select_training_instances(extracted, annotations, max_instances=12, candidate_rows=64):
    """Pilot's class-free matching and deterministic GT-balanced round robin."""
    import numpy as np
    annotations = sorted([a for a in annotations if not a.get("iscrowd", 0)], key=lambda a: int(a["id"]))
    boxes = extracted["boxes"].numpy()
    scores = extracted["scores"].numpy()
    raw_ids = extracted["raw_indices"].numpy()
    domain = [i for i in range(len(boxes)) if scores[i] > .001][:candidate_rows]
    diagnostics = {"domain_native_output_rows": domain,
                   "ordinary_gt_annotation_ids": [int(a["id"]) for a in annotations],
                   "duplicate_raw_excluded_rows": [], "candidate_gt_box_matches": [],
                   "below_match_threshold_rows": [], "matched_but_over_budget_rows": []}
    extracted["training_selection_diagnostics"] = diagnostics
    if not annotations:
        seen = set()
        unique = []
        for row in domain:
            if int(raw_ids[row]) in seen:
                diagnostics["duplicate_raw_excluded_rows"].append(row)
            else:
                unique.append(row)
                seen.add(int(raw_ids[row]))
        diagnostics.update(no_ordinary_gt=True, selected_rows=[], unique_raw_domain_rows=unique,
                           ordinary_gt_without_domain_box_match=[])
        return []
    gt_boxes = np.stack([transformed_box(a["bbox"], extracted["letterbox"]) for a in annotations])
    overlaps = box_iou_xyxy(boxes, gt_boxes)
    seen_raw, groups = set(), {}
    unique_rows = []
    for row in domain:
        if int(raw_ids[row]) in seen_raw:
            diagnostics["duplicate_raw_excluded_rows"].append(row)
            continue
        seen_raw.add(int(raw_ids[row]))
        unique_rows.append(row)
        index = int(overlaps[row].argmax())
        matched = {"output_row": row, "raw_index": int(raw_ids[row]),
                   "annotation_id": int(annotations[index]["id"]), "box_iou": float(overlaps[row, index]),
                   "predicted_class": int(extracted["classes"][row]),
                   "gt_category_id": int(annotations[index]["category_id"])}
        diagnostics["candidate_gt_box_matches"].append(matched | {"qualifies_iou_0_5": bool(overlaps[row, index] >= .5)})
        if overlaps[row, index] < .5:
            diagnostics["below_match_threshold_rows"].append(row)
            continue
        groups.setdefault(matched["annotation_id"], []).append(matched)
    ordered = sorted(groups.values(), key=lambda group: group[0]["output_row"])
    result = []
    depth = 0
    while len(result) < max_instances and any(depth < len(group) for group in ordered):
        for group in ordered:
            if depth < len(group) and len(result) < max_instances:
                result.append(group[depth])
        depth += 1
    selected_rows = {item["output_row"] for item in result}
    diagnostics.update(no_ordinary_gt=False, selected_rows=[item["output_row"] for item in result],
                       unique_raw_domain_rows=unique_rows,
                       matched_but_over_budget_rows=[item["output_row"] for group in ordered for item in group
                                                    if item["output_row"] not in selected_rows],
                       ordinary_gt_without_domain_box_match=[int(a["id"]) for a in annotations if int(a["id"]) not in groups])
    return result


def build_training_targets(extracted, selected, all_annotations, coco, neighbor_rows, neighbor_valid):
    """Pilot's exact integer LetterBox/nearest-grid GT ownership construction."""
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
    extracted["empty_self_grid_exclusions"] = []
    extracted["training_roi_exclusions"] = []
    for match in selected:
        row = match["output_row"]
        native_box = extracted["boxes_proto"][row].numpy()
        x_inside = (np.arange(width) >= native_box[0]) & (np.arange(width) < native_box[2])
        y_inside = (np.arange(height) >= native_box[1]) & (np.arange(height) < native_box[3])
        if not (x_inside.any() and y_inside.any()):
            extracted["training_roi_exclusions"].append(match | {"reason": "empty_native_prototype_roi"})
            continue
        self_mask = masks[match["annotation_id"]]
        if not self_mask.any():
            extracted["empty_self_grid_exclusions"].append(match | {"reason": "empty_gt_self_after_nearest_proto_grid_transform"})
            continue
        neighbor = np.zeros_like(union_all)
        for annotation_id, mask in masks.items():
            if annotation_id != match["annotation_id"]:
                neighbor |= mask
        neighbor &= ~self_mask
        ignored = crowd & ~self_mask
        background = ~(union_all | crowd)
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


class OnlineCOCOProvider:
    """Streaming input provider with strict original COCO image identity.

    ``get(index)`` and ``image_sample(path)`` produce the old cache payload in
    memory. ``require_declared_subset=False`` is reserved for declared engineering
    diagnostics; receipts explicitly mark the available training coverage.
    ``state_dict`` contains compact observed-input evidence for exact retries,
    but never tensors, model weights, or GT supervision arrays.
    """

    def __init__(self, images_list, annotations, weights, vendor, device="cuda",
                 run_dir=None, max_instances=12, require_declared_subset=True):
        if max_instances != 12:
            raise ValueError("Unchanged TriFlow selection requires max_instances=12")
        self.images_list = Path(images_list).resolve()
        self.annotations = Path(annotations).resolve()
        self.weights = Path(weights).resolve()
        self.vendor = Path(vendor).resolve()
        self.device = device
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        if self.run_dir is not None and (self.run_dir / "STREAM_INPUTS.json").exists():
            raise FileExistsError("Streaming run evidence already exists; use a new Run ID for retries")
        self.max_instances = max_instances
        self.require_declared_subset = bool(require_declared_subset)
        self.started_at = now()
        self.closed = False
        self.annotation_sha256 = sha256(self.annotations)
        self.images_list_sha256 = sha256(self.images_list)
        self.base_weights_sha256 = sha256(self.weights)
        if self.annotation_sha256 != COCO_TRAIN_ANNOTATION_SHA256:
            raise ValueError("Annotations are not the authenticated original full COCO train2017 JSON")
        if self.base_weights_sha256 != OFFICIAL_SHA256:
            raise ValueError("Official frozen weights differ from the locked research protocol")
        from pycocotools.coco import COCO
        self.coco = COCO(str(self.annotations))
        images = self.coco.dataset.get("images", [])
        annotation_records = self.coco.dataset.get("annotations", [])
        image_ids = [int(image["id"]) for image in images]
        filenames = [str(image["file_name"]) for image in images]
        annotation_ids = [int(annotation["id"]) for annotation in annotation_records]
        if len(image_ids) != len(set(image_ids)) or len(filenames) != len(set(filenames)):
            raise ValueError("COCO JSON has duplicate image identities or filenames")
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("COCO JSON has duplicate annotation identities")
        if any(int(annotation["image_id"]) not in self.coco.imgs for annotation in annotation_records):
            raise ValueError("COCO annotation references an unknown image identity")
        by_filename = dict(zip(filenames, image_ids))
        self.selected_image_ids = select_declared_train_ids(image_ids)
        self.original_image_ids = sorted(image_ids)
        self.declared_subset_identity_sha256 = hashlib.sha256(
            json.dumps(self.selected_image_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.selector = {
            "selector_name": SUBSET_SELECTOR_NAME, "selection_seed": 0,
            "rank_prefix": SUBSET_SELECTOR_PREFIX, "selection_size": DECLARED_TRAIN_IMAGES,
            "algorithm": "sort all original IDs by sha256('triflow_train20k_seed0:'+str(image_id)), take20000, then image_id ascending",
            "universe_images": len(image_ids), "selected_images": DECLARED_TRAIN_IMAGES,
            "universe_image_identity_sha256": hashlib.sha256(json.dumps(sorted(image_ids), separators=(",", ":")).encode("utf-8")).hexdigest(),
            "selected_image_identity_sha256": self.declared_subset_identity_sha256,
            "uses_gt_categories_masks_or_image_contents": False,
        }
        self.selector_sha256 = hashlib.sha256(json.dumps(self.selector, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.paths = [Path(line.strip()).resolve()
                      for line in self.images_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not self.paths or len(self.paths) != len(set(self.paths)):
            raise ValueError("Training image list is empty or contains duplicate paths")
        self.ids = []
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Listed declared-training image is unavailable: {path}")
            if path.name not in by_filename:
                raise ValueError(f"Image filename is not in the original COCO train2017 JSON: {path}")
            self.ids.append(by_filename[path.name])
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Training list repeats a COCO image identity under another path")
        self.full_split = set(self.ids) == set(image_ids)
        self.declared_subset_complete = self.ids == self.selected_image_ids
        if self.require_declared_subset and not self.declared_subset_complete:
            missing = sorted(set(self.selected_image_ids)-set(self.ids))
            extra = sorted(set(self.ids)-set(self.selected_image_ids))
            raise ValueError(f"Formal training requires the registered 20000-image subset in ascending image_id order: "
                             f"listed={len(self.ids)}, missing={len(missing)}, extra={len(extra)}, "
                             f"first_missing={missing[:12]}, first_extra={extra[:12]}")
        if self.require_declared_subset and self.full_split:
            raise ValueError("This study is the declared 20000-image stage, not a full118287 training stage")
        self.path_to_id = dict(zip(self.paths, self.ids))
        self.id_to_path = dict(zip(self.ids, self.paths))
        self.original_image_count = len(image_ids)
        self.original_annotation_count = len(annotation_ids)
        self.original_ordinary_gt_count = sum(not a.get("iscrowd", 0) for a in annotation_records)
        del annotation_ids, image_ids, filenames, by_filename, annotation_records
        self.image_identity_sha256 = hashlib.sha256(
            json.dumps(sorted(self.ids), separators=(",", ":")).encode("utf-8")).hexdigest()
        self.data_hashes = {"annotation_sha256": self.annotation_sha256,
                            "images_list_sha256": self.images_list_sha256,
                            "image_identity_sha256": self.image_identity_sha256,
                            "declared_subset_identity_sha256": self.declared_subset_identity_sha256,
                            "subset_selector_sha256": self.selector_sha256}
        self.source_hashes = {name: sha256(Path(__file__).with_name(name))
                              for name in ("stream_data.py", "frozen_io.py", "triflow_model.py")}
        self.extractor = FrozenYOLO(self.weights, self.vendor, self.device)
        self.dimensions = None
        self.observed_inputs = {}
        self.visits = self._empty_counts()
        self.unique_images = self._empty_counts()
        self.input_receipt = {
            "stream_version": STREAM_VERSION, "started_at": self.started_at,
            "annotations": str(self.annotations), "annotation_sha256": self.annotation_sha256,
            "images_list": str(self.images_list), "images_list_sha256": self.images_list_sha256,
            "image_identity_sha256": self.image_identity_sha256,
            "base_weights": str(self.weights), "base_weights_sha256": self.base_weights_sha256,
            "vendor": str(self.vendor), "device": self.device,
            "listed_images": len(self.ids), "original_split_images": self.original_image_count,
            "original_annotations": self.original_annotation_count,
            "original_ordinary_gt": self.original_ordinary_gt_count,
            "complete_original_train_split": self.full_split,
            "complete_declared_train_subset": self.declared_subset_complete,
            "declared_train_images": DECLARED_TRAIN_IMAGES,
            "declared_subset_identity_sha256": self.declared_subset_identity_sha256,
            "selected_image_ids": self.selected_image_ids,
            "original_image_ids": self.original_image_ids,
            "subset_selector": self.selector, "subset_selector_sha256": self.selector_sha256,
            "coverage_fraction": len(self.ids) / self.original_image_count,
            "engineering_subset": not self.require_declared_subset,
            "selection": {"native_confidence": .001, "first_native_post_conf_rows": 64,
                          "max_instances_per_image": 12, "class_free_best_gt_box_iou": .5,
                          "tie": "annotation_id ascending", "deduplicate_raw": True,
                          "order": "round-robin matched GT groups ordered by first native output row"},
            "ownership": {"self_priority": True, "gt_neighbor": "union all other noncrowd GT minus Self",
                          "ignore": "union crowd minus Self", "background": "outside all ordinary GT and crowd",
                          "predicted_neighbor": "top2 positive box IoU, tie native output row; same raw excluded; no GT",
                          "neighbor_supervision_missing": "phi_N masked if GT Neighbor empty OR no predicted neighbor"},
            "sources": self.source_hashes, "selection_origin_sha256": PILOT_SELECTION_SOURCE_SHA256,
            "storage": "one image in memory; no dense feature cache is created",
            "frozen_initial_state_sha256": self.extractor.initial_digest,
        }
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            inputs_path = self.run_dir / "STREAM_INPUTS.json"
            if inputs_path.exists():
                raise FileExistsError("Streaming run evidence already exists; use a new Run ID for retries")
            dump_json(inputs_path, self.input_receipt)

    @staticmethod
    def _empty_counts():
        return {"images": 0, "instances": 0, "images_without_usable_instances": 0,
                "native_domain_rows": 0, "unique_raw_domain_rows": 0,
                "duplicate_raw_excluded_rows": 0, "below_match_threshold_rows": 0,
                "matched_but_over_budget_rows": 0, "ordinary_gt_without_domain_box_match": 0,
                "ordinary_gt": 0, "images_without_ordinary_gt": 0,
                "unsupported_roi_exclusions": 0, "empty_self_grid_exclusions": 0,
                "phi_neighbor_supervised_instances": 0, "predicted_neighbor_available_instances": 0,
                "crowd_ignored_instance_pixels": 0}

    def __len__(self):
        return len(self.paths)

    def get(self, index):
        return self.image_sample(self.paths[index])

    def __getitem__(self, index):
        return self.get(index)

    def image_sample(self, image_path):
        if self.closed:
            raise RuntimeError("Cannot extract from a closed online provider")
        import torch
        from triflow_model import TriFlowConfig, build_ownership_targets
        path = Path(image_path).resolve()
        if path not in self.path_to_id:
            raise ValueError(f"Image is not in the authenticated input list: {path}")
        iid = self.path_to_id[path]
        extracted = self.extractor.extract(path)
        if extracted["original_shape"] != [self.coco.imgs[iid]["height"], self.coco.imgs[iid]["width"]]:
            raise ValueError(f"COCO/image shape mismatch: {iid}")
        extracted["image_id"] = iid
        # The graph is prediction-only and is completed before GT selection.
        graph_rows, graph_valid = predicted_neighbors(extracted["boxes"].numpy(), extracted["raw_indices"].numpy(),
                                                    extracted["scores"].numpy(), k=2)
        extracted["predicted_neighbor_rows"] = torch.from_numpy(graph_rows)
        extracted["predicted_neighbor_valid"] = torch.from_numpy(graph_valid)
        all_annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[iid]))
        selected = select_training_instances(extracted, all_annotations, self.max_instances)
        targets = build_training_targets(extracted, selected, all_annotations, self.coco, graph_rows, graph_valid)
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
        extracted["cache_version"] = PILOT_CACHE_VERSION
        extracted["stream_version"] = STREAM_VERSION
        extracted["base_weights_sha256"] = self.base_weights_sha256
        extracted["coco_annotation_sha256"] = self.annotation_sha256
        dimensions = {"prototype_channels": int(extracted["P"].shape[0]),
                      "prototype_grid": list(extracted["P"].shape[-2:]),
                      "feature_channels": int(extracted["F"].shape[0]),
                      "feature_grid": list(extracted["F"].shape[-2:]),
                      "instance_hidden_channels": int(extracted["h"].shape[-1])}
        if self.dimensions is None:
            self.dimensions = dimensions
        elif self.dimensions != dimensions:
            raise ValueError("Frozen input dimensions drifted during streaming")
        selection = extracted["training_selection_diagnostics"]
        counts = self._empty_counts()
        counts.update(images=1, instances=len(targets), images_without_usable_instances=int(not targets),
                      native_domain_rows=len(selection["domain_native_output_rows"]),
                      unique_raw_domain_rows=len(selection["unique_raw_domain_rows"]),
                      ordinary_gt=len(selection["ordinary_gt_annotation_ids"]),
                      images_without_ordinary_gt=int(selection["no_ordinary_gt"]),
                      unsupported_roi_exclusions=len(extracted["training_roi_exclusions"]),
                      empty_self_grid_exclusions=len(extracted["empty_self_grid_exclusions"]),
                      phi_neighbor_supervised_instances=sum(int(t["phi_neighbor_valid"]) for t in targets),
                      predicted_neighbor_available_instances=sum(int(t["neighbor_valid"].any()) for t in targets),
                      crowd_ignored_instance_pixels=sum(int(t["ignore_mask"].sum()) for t in targets))
        for key in ("duplicate_raw_excluded_rows", "below_match_threshold_rows", "matched_but_over_budget_rows",
                    "ordinary_gt_without_domain_box_match"):
            counts[key] = len(selection[key])
        identity = {"image_sha256": extracted["image_sha256"], "input_sha256": extracted["input_sha256"]}
        previous = self.observed_inputs.get(iid)
        if previous is not None and previous != identity:
            raise ValueError(f"Image or exact frozen input changed between visits: {iid}")
        for key, value in counts.items():
            self.visits[key] += value
        if previous is None:
            self.observed_inputs[iid] = identity
            for key, value in counts.items():
                self.unique_images[key] += value
            if self.run_dir is not None:
                item = {"image_id": iid, "image_path": str(path), **identity, "counts": counts,
                        "output_rows": [t["output_row"] for t in targets],
                        "annotation_ids": [t["annotation_id"] for t in targets],
                        "raw_indices": [t["raw_index"] for t in targets], "native_replay_exact": True}
                with (self.run_dir / "STREAM_IMAGES.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
        return extracted

    def verify_frozen(self):
        return self.extractor.verify_frozen()

    def receipt(self, verify_frozen=False):
        value = dict(self.input_receipt)
        value.update(observed_at=now(), observed_unique_images=len(self.observed_inputs),
                     all_listed_images_observed=len(self.observed_inputs) == len(self.ids),
                     dimensions=self.dimensions, visits=dict(self.visits), unique_images=dict(self.unique_images),
                     all_native_replay_exact=True if self.visits["images"] else None,
                     frozen_integrity=self.verify_frozen() if verify_frozen else None)
        value["observed_inputs_identity_sha256"] = hashlib.sha256(
            json.dumps(sorted(self.observed_inputs.items()), separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
        return value

    def state_dict(self):
        """Compact provenance counters for resuming into a distinct Run ID."""
        return {"stream_version": STREAM_VERSION, "data_hashes": self.data_hashes,
                "sources": self.source_hashes, "base_weights_sha256": self.base_weights_sha256,
                "dimensions": self.dimensions, "observed_inputs": self.observed_inputs,
                "visits": self.visits, "unique_images": self.unique_images,
                "frozen_initial_state_sha256": self.extractor.initial_digest}

    def load_state_dict(self, state):
        if self.observed_inputs or self.visits["images"]:
            raise ValueError("Streaming resume evidence must be loaded before extraction")
        for key, expected in (("stream_version", STREAM_VERSION), ("data_hashes", self.data_hashes),
                              ("sources", self.source_hashes), ("base_weights_sha256", self.base_weights_sha256),
                              ("frozen_initial_state_sha256", self.extractor.initial_digest)):
            if state.get(key) != expected:
                raise ValueError(f"Resume streaming input provenance differs: {key}")
        observed = {int(key): dict(value) for key, value in state["observed_inputs"].items()}
        if not set(observed).issubset(self.id_to_path):
            raise ValueError("Resumed evidence contains an image outside the authenticated input list")
        if set(state["visits"]) != set(self.visits) or set(state["unique_images"]) != set(self.unique_images):
            raise ValueError("Resume stream counter schema differs")
        if state["unique_images"]["images"] != len(observed):
            raise ValueError("Resume unique image count and observed identities disagree")
        for counts in (state["visits"], state["unique_images"]):
            if any(not isinstance(value, int) or value < 0 for value in counts.values()):
                raise ValueError("Resume stream counters are not nonnegative integers")
        self.observed_inputs = observed
        self.visits = dict(state["visits"])
        self.unique_images = dict(state["unique_images"])
        self.dimensions = state["dimensions"]

    def close(self):
        if not self.closed:
            self.extractor.close()
            self.closed = True

