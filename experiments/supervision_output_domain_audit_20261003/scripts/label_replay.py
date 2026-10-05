"""Replay frozen official labels, capturing independent masks before overlap.

No detector forward, TAL replay, conversion, training, or candidate selection.
Only the current sample's masks are retained. Dataset-derived cache handling is
left to the same official YOLODataset used by the completed screen preparation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps(dict(dtype=str(value.dtype), shape=list(value.shape)), sort_keys=True).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def _canonical(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


class LabelReplay:
    """Independent polygon and overlap labels on the identical frozen canvas.

    replay(x) returns candidate-ordered bool CPU tensors ``official`` and
    ``polygon``, plus a small JSON-serializable ``audit``. It never infers an
    annotation ID from a candidate-list offset or the original COCO ann order.
    """

    def __init__(self, cfg):
        if os.name == "nt":
            raise RuntimeError("Execute this audit on the authorized remote host, not the desktop")
        self.cfg = dict(cfg)
        vendor = Path(cfg["source_python"]).resolve()
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import ultralytics
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.converter import coco91_to_coco80_class
        import ultralytics.data.augment as augmentation
        from ultralytics.data.utils import polygons2masks
        from ultralytics.utils import ops
        if (ultralytics.__version__ != "8.4.100" or
                not Path(ultralytics.__file__).resolve().is_relative_to(vendor)):
            raise RuntimeError("Require the same vendored Ultralytics 8.4.100")
        self.augmentation, self.polygons2masks, self.ops = augmentation, polygons2masks, ops
        self.classmap = coco91_to_coco80_class()
        self.original_root = Path(cfg["original_screen_root"])
        self.dataset_root = self.original_root / "dataset"
        ready_path = self.dataset_root / "DATASET_READY.json"
        conversion_path = self.dataset_root / "CONVERSION.json"
        source_path = self.dataset_root / "SOURCE_IMAGES.json"
        ready = _json(ready_path)
        self.cache_identity = _json(Path(cfg["cache"]) / "CACHE_IDENTITY.json")
        if ready["fingerprint"] != self.cache_identity["fingerprint"]:
            raise AssertionError("Converted labels do not belong to the frozen cache")
        self.fingerprint = ready["fingerprint"]
        self.converted = (self.dataset_root / ready["directory"]).resolve()
        if not self.converted.is_relative_to(self.dataset_root.resolve()) or not self.converted.is_dir():
            raise AssertionError("Converted dataset directory is missing or escaped its original root")
        conversion = _json(conversion_path)
        self.source_meta = _json(source_path)
        self.conversion_ids = {int(k): [int(v) for v in vals]
                               for k, vals in conversion["source_annotation_ids"].items()}
        weights_path = Path(cfg["weights"])
        weights_sha = _sha(weights_path)
        if weights_sha != self.cache_identity["weights_sha256"]:
            raise AssertionError("Checkpoint metadata is not from the original frozen source")
        # Read only train_args/names. This model is never forwarded or moved to GPU.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        args = deepcopy(checkpoint["train_args"])
        metadata_model = checkpoint.get("ema")
        if metadata_model is None:
            metadata_model = checkpoint.get("model")
        if metadata_model is None or not hasattr(metadata_model, "names"):
            raise AssertionError("Original checkpoint has no class-name metadata")
        names = deepcopy(metadata_model.names)
        del metadata_model, checkpoint
        if args.get("mask_ratio") != 1 or args.get("overlap_mask") is not True:
            raise AssertionError("Expected original mask_ratio=1, overlap_mask=True")
        self.hyp = SimpleNamespace(**args)
        self.names = names
        self._YOLODataset = YOLODataset
        self.datasets, self.positions = {}, {}
        self.source_split = _json(self.original_root / "SPLIT.json")
        if len(self.source_split.get("dev", [])) != 256:
            raise AssertionError("The reused screen dev population changed")
        self.source_audit = dict(
            ultralytics_version=ultralytics.__version__, ultralytics_file=str(ultralytics.__file__),
            source_cache_fingerprint=self.fingerprint,
            ready_sha256=_sha(ready_path), conversion_sha256=_sha(conversion_path),
            source_images_sha256=_sha(source_path), checkpoint_sha256=weights_sha,
            mask_ratio=1, overlap_mask=True, train_args_from_original_checkpoint=True,
            model_forward=False, TAL_recomputed=False, conversion_recomputed=False,
            target_boxes_tolerance="exact FP32 equality; no tolerance relaxation")

    def _dataset(self, split):
        if split != "dev":
            raise ValueError("This audit is limited to the fixed 256-image dev cohort")
        if split not in self.datasets:
            listing = self.dataset_root / f"{split}_input.txt"
            if not listing.is_file():
                raise FileNotFoundError("Original frozen dataset input list is missing")
            expected = [int(v) for v in self.source_split[split]]
            listed = [int(Path(line.strip()).stem) for line in listing.read_text().splitlines() if line.strip()]
            if listed != expected:
                raise AssertionError("Original dataset image list differs from frozen split order")
            ds = self._YOLODataset(img_path=str(listing), imgsz=640, batch_size=1,
                augment=False, hyp=deepcopy(self.hyp), rect=False, cache=False, stride=32,
                data={"names": self.names, "nc": 80, "channels": 3}, task="segment")
            positions = {int(Path(p).stem): j for j, p in enumerate(ds.im_files)}
            if len(positions) != len(expected) or set(positions) != set(expected):
                raise AssertionError("Official dataset changed image membership")
            self.datasets[split], self.positions[split] = ds, positions
        return self.datasets[split], self.positions[split]

    def _pre_format_identity(self, ds, position, iid):
        meta = self.source_meta[str(iid)]
        domain = meta["source_split"]
        label = self.converted / "labels" / f"{domain}2017" / Path(meta["file_name"]).with_suffix(".txt")
        lines = label.read_text(encoding="utf-8").strip().splitlines() if label.exists() else []
        ids = self.conversion_ids[iid]
        if len(lines) != len(ids):
            raise AssertionError("Official converted TXT rows no longer match the annotation ledger")
        serialized = [(np.asarray(line.split(), dtype=np.float32), aid) for line, aid in zip(lines, ids)]
        raw = ds.labels[position]
        if len(raw["cls"]) != len(raw["segments"]):
            raise AssertionError("Official class/polygon lengths differ")
        pre_ids = []
        for cls, polygon in zip(raw["cls"].flatten(), raw["segments"]):
            key = np.r_[cls, polygon.flatten()].astype(np.float32)
            found = [aid for arr, aid in serialized if np.array_equal(key, arr)]
            if len(found) != 1:
                raise AssertionError(f"Missing or ambiguous official polygon identity for image {iid}")
            pre_ids.append(found[0])
        if len(set(pre_ids)) != len(pre_ids):
            raise AssertionError("Annotation identity repeated after official dataset verification")
        return pre_ids, np.asarray(raw["cls"]).copy(), label

    def replay(self, x):
        iid, split = int(x["image_id"]), str(x["split"])
        if x["fingerprint"] != self.fingerprint:
            raise AssertionError("Candidate payload belongs to another cache")
        ds, positions = self._dataset(split)
        if iid not in positions:
            raise AssertionError("Candidate image is outside the frozen dev population")
        position = positions[iid]
        pre_ids, pre_cls, label_path = self._pre_format_identity(ds, position, iid)
        captured = {}
        original = self.augmentation.polygons2masks_overlap

        def capture(imgsz, segments, downsample_ratio=1):
            if captured:
                raise AssertionError("Expected exactly one overlap rasterization per official sample")
            if tuple(imgsz) != (640, 640) or downsample_ratio != 1:
                raise AssertionError("Official rasterizer resolution changed")
            # Original overlap implementation is called unchanged. Capture the
            # independent version of precisely its transformed segment input.
            overlap, indices = original(imgsz, segments, downsample_ratio=downsample_ratio)
            independent = self.polygons2masks(imgsz, segments, color=1, downsample_ratio=downsample_ratio)
            indices = np.asarray(indices, dtype=np.int64)
            if sorted(indices.tolist()) != list(range(len(pre_ids))):
                raise AssertionError("Overlap sorting is not a permutation of official GT identities")
            captured.update(indices=indices.copy(), polygon=np.ascontiguousarray(independent[indices], dtype=bool))
            return overlap, indices

        self.augmentation.polygons2masks_overlap = capture
        try:
            sample = ds[position]
        finally:
            self.augmentation.polygons2masks_overlap = original
        if pre_ids:
            if not captured:
                raise AssertionError("Official overlap rasterizer was not observed")
            indices = captured["indices"]
            annotation_ids = np.asarray(pre_ids, dtype=np.int64)[indices].tolist()
            polygon_all = torch.from_numpy(captured["polygon"])
        else:
            if captured:
                raise AssertionError("Empty label sample unexpectedly rasterized polygons")
            indices = np.empty(0, dtype=np.int64)
            annotation_ids = []
            polygon_all = torch.empty((0, 640, 640), dtype=torch.bool)
        if annotation_ids != [int(v) for v in x["all_annotation_ids"]]:
            raise AssertionError("Official post-overlap annotation order differs from cache")
        image = sample["img"].detach().cpu()
        if image.dtype != torch.uint8 or not torch.equal(image, x["input_uint8"].detach().cpu()):
            raise AssertionError("Official input replay is not pixel-exact")
        overlap = sample["masks"][0].detach().cpu()
        if not torch.equal(overlap, x["masks"].detach().cpu()):
            raise AssertionError("Official overlap mask replay is not pixel-exact")
        if (_canonical(sample["ori_shape"]) != _canonical(x["original_shape"]) or
                _canonical(sample["ratio_pad"]) != _canonical(x["ratio_pad"]) or
                list(x["input_shape"]) != [640, 640]):
            raise AssertionError("Official dataset geometry differs from frozen cache")
        expected_cls = torch.from_numpy(np.ascontiguousarray(pre_cls[indices]))
        if not torch.equal(sample["cls"].cpu(), expected_cls):
            raise AssertionError("Official class reorder does not match polygon identity reorder")
        # Equivalent to the fixed batch1 criterion.preprocess: XYWH*640 then
        # xywh2xyxy, prior to TAL gathering. No model or assigner is invoked.
        gt_boxes = self.ops.xywh2xyxy(sample["bboxes"].detach().cpu().float()*640)
        owners = x["owners"].detach().cpu().long()
        n = len(owners)
        if owners.ndim != 1 or len(x["rows"]) != n or len(x["raw_ids"]) != n:
            raise AssertionError("Candidate count/owner vector changed")
        if n and (int(owners.min()) < 0 or int(owners.max()) >= len(annotation_ids)):
            raise AssertionError("Candidate owner is outside the official post-sort GT table")
        if not torch.equal(gt_boxes[owners], x["target_boxes"].detach().cpu().float()):
            raise AssertionError("GT owner boxes differ from cached official TAL target_boxes")
        for k, row in enumerate(x["rows"]):
            owner = int(owners[k]); aid = annotation_ids[owner]
            if (str(row.get("split", split)) != split or int(row["image_id"]) != iid or
                    int(row["annotation_id"]) != aid or row["branch"] != "one2one" or
                    int(row["raw_id"]) != int(x["raw_ids"][k]) or
                    int(row["target_gt_idx"]) != owner or
                    int(row["pyramid_level"]) != int(x["levels"][k])):
                raise AssertionError("Permanent candidate identity no longer matches owner/annotation ledger")
            annotation = self.source_meta[str(iid)]["annotations"][str(aid)]
            if int(annotation["image_id"]) != iid or int(sample["cls"][owner, 0]) != self.classmap[int(annotation["category_id"])-1]:
                raise AssertionError("Official class/annotation mapping changed")
        official = overlap[None] == (owners+1)[:, None, None]
        polygon = polygon_all[owners].clone()
        if official.shape != (n, 640, 640) or polygon.shape != official.shape:
            raise AssertionError("Candidate label tensor shape changed")
        if bool((official & ~polygon).any()):
            raise AssertionError("Overlap label is not a subset of its independent polygon mask")
        audit = dict(passed=True, image_id=iid, split=split, candidates=n,
            all_gt=len(annotation_ids), all_annotation_ids=annotation_ids,
            pre_format_annotation_ids=pre_ids, overlap_sort_indices=indices.tolist(),
            input_exact=True, overlap_exact=True, ratio_pad_exact=True,
            annotation_order_exact=True, class_order_exact=True, owner_boxes_exact=True,
            candidate_identity_exact=True, official_subset_polygon=True,
            input_sha256=_tensor_sha(image), overlap_sha256=_tensor_sha(overlap),
            independent_polygon_sha256=_tensor_sha(polygon),
            converted_label_path=str(label_path),
            converted_label_sha256=_sha(label_path) if label_path.exists() else None,
            source=self.source_audit)
        # No per-image capture is stored on self: all other GT masks can be
        # reclaimed immediately after this call, even for dense images.
        return dict(official=official.contiguous(), polygon=polygon.contiguous(), audit=audit)
