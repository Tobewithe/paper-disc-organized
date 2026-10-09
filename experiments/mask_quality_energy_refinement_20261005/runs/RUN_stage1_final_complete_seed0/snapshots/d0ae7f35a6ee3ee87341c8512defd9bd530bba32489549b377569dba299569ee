"""Batch-one FP32 official TAL data for FINAL evaluation and DEV witnesses.

Only selected source JPEGs, official converted labels, and small JSON receipts
are persisted. Prototypes, coefficients, hidden vectors, and full overlap masks
are returned one image at a time and are never written as a tensor cache.

The original frozen model receives only the normalized image. Ground truth is
used afterwards by the official one-to-one TAL criterion to identify evaluation
candidates, and the CPU raster labels are diagnostic supervision only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from online_runtime import FrozenReplay, dump, resolve_runtime_config, sha256
from prepare_qcr_cache import SCHEMA, ensure_space, prepare_images


class StreamingFinal:
    """Yield cache-compatible, transient CPU payloads from full official forwards.

    ``split='final'`` uses val2017; ``split='dev'`` uses train2017 and supports
    small, fixed-image witnesses against the prepared DEV cache. Image order is
    exactly the supplied order. Consume with a context manager or call ``close``
    after early exit so the immutable-model assertion is always performed.
    """

    def __init__(self, cfg, image_ids, out, split="final"):
        if split not in ("final", "dev"):
            raise ValueError("StreamingFinal supports only fixed FINAL or DEV evaluation")
        if os.name == "nt" or not torch.cuda.is_available():
            raise RuntimeError("Streaming evaluation may run only on the authorized Linux CUDA server")
        self.cfg = resolve_runtime_config(cfg)
        self.image_ids = [int(iid) for iid in image_ids]
        if not self.image_ids or len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("Image IDs must be a nonempty, duplicate-free fixed image list")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.domain = "val" if split == "final" else "train"
        self._started = time.monotonic()
        self._processed = 0
        self._candidates = 0
        self._closed = False
        self._started_iteration = False
        self._iterator = None
        self._handles = []
        self._augmentation = None
        self._original_rasterizer = None
        self._captured = {}
        self._order = {}

        # Validate source image membership before extracting anything or loading
        # the frozen checkpoint. No replacement IDs are drawn on failure.
        sys.path.insert(0, self.cfg["source_python"])
        from pycocotools.coco import COCO
        import legacy_prepare_cache as official

        self.coco = COCO(self.cfg[f"annotations_{self.domain}"])
        missing = set(self.image_ids) - set(self.coco.getImgIds())
        if missing:
            raise ValueError(f"Fixed {split} IDs absent in {self.domain} annotations: {sorted(missing)[:20]}")
        self.cat_to_idx = {int(cid): i for i, cid in enumerate(sorted(self.coco.getCatIds()))}
        ensure_space(self.cfg)
        receipts = prepare_images(self.cfg, {self.domain: self.image_ids})
        dump(self.out / "STREAMING_SOURCE_IMAGES.json", receipts)

        self.replay = FrozenReplay(self.cfg)
        model = self.replay.source
        model.args = SimpleNamespace(**self.replay.train_args)
        if model.args.mask_ratio != 1 or not model.args.overlap_mask:
            raise AssertionError("Official checkpoint must retain mask_ratio=1 and overlap_mask=True")
        self.criterion = model.init_criterion().one2one
        self.signature = dict(
            schema=SCHEMA,
            mode="streaming_full_forward",
            split=split,
            source_split=self.domain,
            image_ids=self.image_ids,
            weights_sha256=self.replay.weights_sha256,
            annotations={d: sha256(self.cfg[f"annotations_{d}"]) for d in ("train", "val")},
            dataset_code_sha256=sha256(official.__file__),
            official_mask_ratio=model.args.mask_ratio,
            official_overlap_mask=model.args.overlap_mask,
            segmentation_gain=float(model.args.box),
            forward="original FP32 eval batch1, no augmentation, no TF32",
        )
        # Conversion identity excludes this wrapper's hash, allowing code-only
        # repairs to reuse immutable labels/JPEGs without rewriting old data.
        self.fingerprint = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()
        dataset_root = Path(self.cfg["server_root"]) / "streaming_datasets" / f"{split}_{self.fingerprint[:16]}"
        fixed_split = {"fit": [], "dev": [], "val": []}
        fixed_split["val" if split == "final" else "dev"] = self.image_ids
        converted, self._identity, self._source_meta = official.prepare_dataset(
            self.cfg, fixed_split, dataset_root, self.fingerprint
        )
        self.dataset_root = dataset_root
        self.converted = converted

        from ultralytics.data.dataset import YOLODataset

        listing = dataset_root / f"{split}_input.txt"
        listing.write_text(
            "\n".join(str(converted / "images" / f"{self.domain}2017" / f"{iid:012d}.jpg")
                      for iid in self.image_ids) + "\n", encoding="utf-8"
        )
        self.dataset = YOLODataset(
            img_path=str(listing), imgsz=640, batch_size=1, augment=False,
            hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
            data={"names": model.names, "nc": 80, "channels": 3}, task="segment",
        )
        self._positions = {int(Path(p).stem): j for j, p in enumerate(self.dataset.im_files)}
        if len(self._positions) != len(self.image_ids) or set(self._positions) != set(self.image_ids):
            raise AssertionError("Official dataset changed the fixed image membership")
        self._planned_converted_annotations = sum(len(self._identity[iid]) for iid in self.image_ids)
        self._planned_noncrowd_annotations = sum(
            len(self.coco.getAnnIds(imgIds=[iid], iscrowd=False)) for iid in self.image_ids
        )
        dump(self.out / "STREAMING_DATA_ENVIRONMENT.json", {
            **self.replay.import_info,
            "detector_head_forward": True,
            "TAL_recomputed_during_training": False,
            "TAL_recomputed_for_evaluation_identity_only": True,
            "streaming_code_sha256": sha256(__file__),
            "prototype_dtype": "float32",
            "coefficient_dtype": "float32",
            "cpu_gt_raster_purpose": "diagnostic supervision only; never model evidence",
            "tensor_cache_written": False,
        })
        self._write_state("ready", immutable_verified=None)

    def _write_state(self, stage, immutable_verified):
        dump(self.out / "STREAMING_DATA.json", {
            "stage": stage,
            "split": self.split,
            "source_split": self.domain,
            "fingerprint": self.fingerprint,
            "signature": self.signature,
            "dataset_root": str(self.dataset_root),
            "planned_images": len(self.image_ids),
            "planned_converted_annotations": self._planned_converted_annotations,
            "planned_source_noncrowd_annotations": self._planned_noncrowd_annotations,
            "processed_images": self._processed,
            "tal_candidates": self._candidates,
            "completed_all_requested_images": self._processed == len(self.image_ids),
            "frozen_original_buffers_unchanged": immutable_verified,
            "tensor_cache_written": False,
            "elapsed_s": time.monotonic() - self._started,
        })

    def _install_capture(self):
        import ultralytics.data.augment as augmentation

        self._augmentation = augmentation
        self._original_rasterizer = augmentation.polygons2masks_overlap

        def record_order(*args, **kwargs):
            masks, indices = self._original_rasterizer(*args, **kwargs)
            self._order["indices"] = indices.copy()
            return masks, indices

        augmentation.polygons2masks_overlap = record_order
        head = self.replay.head
        self._handles = [head.register_forward_pre_hook(
            lambda _module, inputs: self._captured.__setitem__("F", list(inputs[0]))
        )]
        self._handles += [branch[-1].register_forward_pre_hook(
            lambda _module, inputs, level=level: self._captured.__setitem__(f"h{level}", inputs[0])
        ) for level, branch in enumerate(head.one2one_cv4)]

    @torch.no_grad()
    def _build_image(self, iid):
        """Forward and copy exactly one payload, releasing all live feature maps."""
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils import ops
        from ultralytics.utils.metrics import box_iou

        raw_label = self.dataset.labels[self._positions[iid]]
        pre_ids = []
        for cls, polygon in zip(raw_label["cls"].flatten(), raw_label["segments"]):
            key = np.r_[cls, polygon.flatten()].astype(np.float32)
            found = [aid for arr, aid in self._identity[iid] if np.array_equal(key, arr)]
            if len(found) != 1:
                raise AssertionError(f"Ambiguous official annotation identity: {iid}")
            pre_ids.append(found[0])
        self._order.clear()
        sample = self.dataset[self._positions[iid]]
        ann_ids = np.asarray(pre_ids)[self._order["indices"]].tolist() if pre_ids else []
        if len(ann_ids) != len(sample["cls"]):
            raise AssertionError(f"Official overlap sorting changed annotation length: {iid}")
        batch = YOLODataset.collate_fn([sample])
        batch = {key: value.to(self.replay.device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        if tuple(batch["img"].shape) != (1, 3, 640, 640) or tuple(batch["masks"].shape[-2:]) != (640, 640):
            raise AssertionError("Original input/full mask resolution differs")
        self._captured.clear()
        try:
            # No GT enters this call. The criterion below is used only after the
            # fixed prediction, to reproduce official evaluation identities.
            _, raw = self.replay.source(batch["img"].float() / 255)
            pred = raw["one2one"]
            assigned, _, _ = self.criterion.get_assigned_targets_and_loss(pred, batch)
            fg, owner, target_boxes = (value[0] for value in assigned[:3])
            raw_ids = torch.where(fg)[0]
            owners = owner[raw_ids]
            if len(set(owners.tolist())) != len(owners):
                raise AssertionError("Official one-to-one TAL repeated an owner")
            features = self._captured["F"]
            hs = [self._captured[f"h{level}"] for level in range(3)]
            levels = torch.cat([
                torch.full((h.shape[-2] * h.shape[-1],), level,
                           device=self.replay.device, dtype=torch.long)
                for level, h in enumerate(hs)
            ])[raw_ids]
            h0 = torch.cat([h[0].flatten(1) for h in hs], 1).T[raw_ids]
            raw_coefficients = pred["mask_coefficient"][0].T
            raw_boxes = self.replay.head._get_decode_boxes(pred)[0].T
            raw_scores, raw_classes = pred["scores"][0].sigmoid().max(0)
            coefficients = raw_coefficients[raw_ids]
            boxes = raw_boxes[raw_ids]
            scores, classes = raw_scores[raw_ids], raw_classes[raw_ids]
            boxes_original = ops.scale_boxes(
                (640, 640), boxes.clone(), sample["ori_shape"], ratio_pad=sample["ratio_pad"]
            )
            rows = []
            for k, (rid, gt_idx) in enumerate(zip(raw_ids.tolist(), owners.tolist())):
                ann = self._source_meta[str(iid)]["annotations"][str(ann_ids[gt_idx])]
                category_id = int(ann["category_id"])
                gt_class_id = self.cat_to_idx[category_id]
                if int(sample["cls"].flatten()[gt_idx]) != gt_class_id:
                    raise AssertionError("Official serialized class disagrees with original COCO class")
                gt_box = boxes.new_tensor(ann["bbox"])
                gt_box[2:] += gt_box[:2]
                predicted_class_id = int(classes[k])
                rows.append(dict(
                    split=self.split, image_id=iid, annotation_id=int(ann["id"]), branch="one2one",
                    raw_id=int(rid), pyramid_level=int(levels[k]), level=int(levels[k]),
                    target_gt_idx=int(gt_idx), gt_index=int(gt_idx), candidate_index=k, row_index=k,
                    area=float(ann["area"]),
                    box_iou=float(box_iou(gt_box[None], boxes_original[k:k + 1])[0, 0]),
                    predicted_class_id=predicted_class_id, predicted_score=float(scores[k]),
                    gt_class_id=gt_class_id, category_id=category_id,
                    class_correct=predicted_class_id == gt_class_id,
                ))
            proto = pred["proto"][0]
            if tuple(proto.shape) != (32, 160, 160):
                raise AssertionError("Original prototype resolution/channels changed")
            for tensor in (proto, raw_coefficients, raw_boxes, raw_scores, h0):
                if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise AssertionError("Nonfinite or non-FP32 original model evidence")
            cpu = lambda value: value.detach().cpu().contiguous()
            ids_cpu = cpu(raw_ids)
            x = dict(
                schema=SCHEMA, fingerprint=self.fingerprint, image_id=iid, split=self.split,
                source_split=self.domain, rows=rows, raw_ids=ids_cpu,
                c0=cpu(coefficients), boxes=cpu(boxes), levels=cpu(levels), owners=cpu(owners),
                target_boxes=cpu(target_boxes[raw_ids]), predicted_classes=cpu(classes),
                predicted_scores=cpu(scores),
                # All original one-to-one raw locations support the exhaustive
                # R_arg Box75/Mask75 existence diagnostic. TAL-selected rows
                # remain the shared deployment/effect population above.
                raw_coefficients=cpu(raw_coefficients), raw_boxes=cpu(raw_boxes),
                raw_predicted_classes=cpu(raw_classes), raw_predicted_scores=cpu(raw_scores),
                original_shape=sample["ori_shape"],
                ratio_pad=sample["ratio_pad"], all_annotation_ids=[int(aid) for aid in ann_ids],
                segmentation_gain=float(self.replay.source.args.box), input_shape=[640, 640],
                input_uint8=cpu(sample["img"]), proto=cpu(proto), masks=cpu(sample["masks"][0]),
                source_F_shapes=[list(feature.shape[1:]) for feature in features], no_positive=not bool(rows),
                source_annotation_count=len(self.coco.getAnnIds(imgIds=[iid])),
                source_noncrowd_annotation_count=len(self.coco.getAnnIds(imgIds=[iid], iscrowd=False)),
                converted_annotation_count=len(ann_ids), tal_candidate_count=len(rows),
                unmatched_annotation_ids=sorted(set(ann_ids) - {row["annotation_id"] for row in rows}),
                operator=dict(
                    schema="streaming_native_hidden_v1", source_fingerprint=self.fingerprint,
                    image_id=iid, raw_ids=ids_cpu, h0=cpu(h0),
                    feature_channels=list(self.replay.feature_channels),
                    operator_scope="evaluation native h0 only; unused training A/G not constructed",
                ),
                _asset_integrity=dict(original_server_forward=True, proto_storage_dtype="torch.float32",
                                      streaming=True, tensor_cache_written=False),
            )
            x["_operator"] = x["operator"]
            return x
        finally:
            # The caller receives CPU data. Full neck/head GPU tensors must not
            # remain held by hooks while the evaluator processes this image.
            self._captured.clear()

    def iter_images(self):
        if self._closed or self._started_iteration:
            raise RuntimeError("StreamingFinal is a single-pass iterator; create a new instance for a new Run")
        self._started_iteration = True
        self._iterator = self._iterate()
        return self._iterator

    def _iterate(self):
        try:
            self._install_capture()
            for iid in self.image_ids:
                x = self._build_image(iid)
                self._processed += 1
                self._candidates += len(x["rows"])
                if self._processed % 100 == 0:
                    self._write_state("streaming", immutable_verified=None)
                yield iid, x
                del x
        finally:
            self._cleanup()

    def _cleanup(self):
        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._augmentation is not None and self._original_rasterizer is not None:
            self._augmentation.polygons2masks_overlap = self._original_rasterizer
        self._captured.clear()
        self._closed = True
        try:
            self.replay.assert_unchanged()
        except BaseException:
            self._write_state("closed_model_integrity_failed", immutable_verified=False)
            raise
        self._write_state(
            "complete" if self._processed == len(self.image_ids) else "closed_partial",
            immutable_verified=True,
        )

    def close(self):
        """Release hooks and assert immutable original buffers on early exit."""
        if self._iterator is not None:
            self._iterator.close()
        self._cleanup()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("StreamingFinal is already closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
