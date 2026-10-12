"""Non-invasive sparse TAL recorder for Ultralytics 8.4.100 SegmentationModel.

Registers on the *actual* E2ELoss assigners. No alternate TAL calls, no dense
GT x raw tensors persisted, no parameter or assignment modification. The
original COCO annotation identity is UNKNOWN unless the actual augmented
data pipeline provides a rigorously aligned trace_gt_lineage sidecar.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch


REQUIRED_BRANCHES = ("one2many", "one2one")
SCHEMA_VERSION = 1


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


class SparseTALRecorder:
    """Record actual positive edges and EVERY augmented GT (including unassigned).

    Public lifecycle:
        attach(criterion)  # the criterion that will actually be used by model.loss
        begin_batch(batch, epoch, microstep, optimizer_attempts_before)
        capture_features(head_forward_inputs)
        [real model.loss -> actual branch assigners fire]
        end_batch()
        close()

    A batch with missing hooks fails closed. Trainer OOM recovery may call
    abort_batch() before a new batch; aborted records do NOT count as training.
    """

    def __init__(self, file: str | Path, run_id: str, *,
                 expected_branches=REQUIRED_BRANCHES, rank: int = 0) -> None:
        self.file = Path(file)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        require(not self.file.exists(), f"refusing to overwrite trace {self.file}")
        self.run_id = str(run_id)
        require(bool(self.run_id), "run_id is mandatory")
        self.rank = int(rank)
        self.expected = tuple(expected_branches)
        require(len(set(self.expected)) == len(self.expected), "duplicate branch names")
        self._stream = self.file.open("x", encoding="utf-8")
        self._handles = []
        self._criterion_id = None
        self._pending: dict[str, Any] | None = None
        self._observations: dict[str, Any] = {}
        self._feature_hw: list[list[int]] | None = None
        self._closed = False
        self.completed_batches = 0
        self.aborted_batches = 0

    def attach(self, criterion) -> None:
        """Attach once to the exact criterion used in model.loss."""
        require(not self._closed, "recorder closed")
        require(self._criterion_id is None, "criterion was already installed")
        branches = {}
        for name in self.expected:
            branch_loss = getattr(criterion, name, None)
            require(branch_loss is not None and hasattr(branch_loss, "assigner"),
                    f"actual criterion lacks {name}.assigner")
            branches[name] = branch_loss.assigner
        self._criterion_id = id(criterion)
        self._assigners = branches
        for name, assigner in branches.items():
            self._handles.append(assigner.register_forward_hook(
                lambda _module, inputs, result, name=name:
                    self._on_assigner(name, inputs, result)
            ))

    def begin_batch(self, batch: dict, *, epoch: int, microstep: int,
                    optimizer_attempts_before: int) -> None:
        require(not self._closed, "recorder closed")
        require(self._pending is None, "previous batch has not been committed or aborted")
        require(self._criterion_id is not None, "actual criterion hooks not installed")
        for key in ("img", "batch_idx", "cls", "bboxes"):
            require(key in batch, f"training batch misses {key}")
        image = batch["img"]
        require(isinstance(image, torch.Tensor) and image.ndim == 4, "img must be BCHW tensor")
        b, _, h, w = map(int, image.shape)
        batch_idx = batch["batch_idx"].detach().reshape(-1).cpu().long()
        classes = batch["cls"].detach().reshape(-1).cpu().long()
        bboxes = batch["bboxes"].detach().reshape(-1, 4).cpu().float()
        require(len(batch_idx) == len(classes) == len(bboxes), "batch target lengths mismatch")
        require(all(0 <= int(i) < b for i in batch_idx.tolist()), "invalid batch_idx")
        require(torch.isfinite(bboxes).all().item(), "nonfinite augmented boxes")
        source_paths = [str(x) for x in batch.get("im_file", [])]
        if source_paths:
            require(len(source_paths) == b, "primary im_file image count mismatch")
        lineage = batch.get("trace_gt_lineage")
        if lineage is not None:
            require(isinstance(lineage, (list, tuple)) and len(lineage) == len(classes),
                    "trace_gt_lineage must align 1-to-1 with collated target rows")
            for k, item in enumerate(lineage):
                require(isinstance(item, dict), f"lineage row {k} is not a dictionary")
                for field in ("source_image_id", "original_annotation_id", "occurrence_id"):
                    require(field in item, f"lineage row {k} missing {field}")
        self._pending = {
            "schema_version": SCHEMA_VERSION,
            "kind": "actual_train_assignment_event",
            "run_id": self.run_id, "rank": self.rank,
            "epoch": int(epoch), "microstep": int(microstep),
            "optimizer_attempts_before": int(optimizer_attempts_before),
            "input_hw": [h, w],
            "batch_images": b,
            "primary_im_file_is_augmentation_lineage": False,
            "primary_im_files": source_paths,
            "started_monotonic": time.monotonic(),
            "_batch_idx": batch_idx, "_classes": classes,
            "_bboxes": bboxes, "_lineage": lineage,
        }
        self._observations = {}
        self._feature_hw = None

    def capture_features(self, inputs) -> None:
        """PyTorch forward prehook on the *real* segmentation head."""
        if self._pending is None:
            return  # model warmup / AMP check is NOT a training event
        require(bool(inputs), "head prehook received no features")
        feats = inputs[0]
        require(isinstance(feats, (tuple, list)) and len(feats) >= 1,
                "head expected sequence of feature maps")
        hw = [[int(x.shape[-2]), int(x.shape[-1])] for x in feats]
        require(all(h > 0 and w > 0 for h, w in hw), "invalid feature grids")
        if self._feature_hw is not None:
            require(self._feature_hw == hw, "feature shapes changed within microbatch")
        self._feature_hw = hw

    def _on_assigner(self, branch: str, args, result) -> None:
        if self._pending is None:
            return  # e.g. preliminary AMP setup; never pretend it was a training step
        require(branch not in self._observations, f"duplicate {branch} TAL call in one batch")
        require(len(args) == 6 and len(result) == 5,
                "Ultralytics assigner signature/version changed")
        scores, boxes, anchors, labels, gboxes, valid = args
        target_labels, target_boxes, target_scores, fg, owner = result
        pending = self._pending
        b = int(pending["batch_images"])
        H, W = pending["input_hw"]
        require(self._feature_hw is not None, "head feature prehook missing")
        sizes = [p * q for p, q in self._feature_hw]
        nraw = sum(sizes)
        require(tuple(scores.shape[:2]) == (b, nraw), "TAL scores vs raw grid mismatch")
        require(tuple(boxes.shape) == (b, nraw, 4), "TAL boxes shape changed")
        require(tuple(fg.shape) == tuple(owner.shape) == (b, nraw),
                "TAL assigned-mask shape mismatch")
        require(tuple(target_scores.shape) == tuple(scores.shape),
                "TAL assigned score shape mismatch")
        require(tuple(labels.shape[:2]) == tuple(gboxes.shape[:2]) ==
                tuple(valid.shape[:2]), "TAL GT tensor shapes differ")
        require(tuple(anchors.shape) == (nraw, 2), "TAL anchor count changed")

        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        idx = pending["_batch_idx"]
        classes = pending["_classes"]
        bboxes = pending["_bboxes"]
        gt_by_image = [(idx == image).nonzero(as_tuple=True)[0].tolist()
                       for image in range(b)]
        pos = fg.detach().cpu().bool()
        targets = owner.detach().cpu().long()
        cls_tensor = labels.detach().cpu().reshape(b, -1).long()
        bbox_tensor = gboxes.detach().cpu().float()
        valid_tensor = valid.detach().cpu().reshape(b, -1).bool()
        # Transfer ONLY selected edges' scalar scores, not B x R x classes
        # dense logits/targets (which would destroy the sparse recorder budget).
        results = []
        for image in range(b):
            local_rows = gt_by_image[image]
            present_indices = valid_tensor[image].nonzero(as_tuple=True)[0].tolist()
            require(present_indices == list(range(len(local_rows))),
                    f"{branch} batch {image}: valid TAL labels not aligned to augmented GT")
            gt_records = []
            for gi, original_row in enumerate(local_rows):
                label = int(classes[original_row])
                require(int(cls_tensor[image, gi]) == label,
                        f"{branch} image {image} target class mismatch at GT {gi}")
                xywh = bboxes[original_row]
                cx, cy, ww, hh = map(float, xywh.tolist())
                expectation = torch.tensor([
                    (cx - ww / 2) * W, (cy - hh / 2) * H,
                    (cx + ww / 2) * W, (cy + hh / 2) * H,
                ])
                require(torch.allclose(expectation, bbox_tensor[image, gi],
                                       atol=0.03, rtol=0.0001),
                        f"{branch} image {image} target GT box/order mismatch at GT {gi}")
                gt_record = {
                    "aug_gt_idx": gi, "collated_target_idx": original_row,
                    "class_id": label, "aug_box_xyxy": bbox_tensor[image, gi].tolist(),
                    "lineage_status": "unknown", "source_image_id": None,
                    "original_annotation_id": None, "occurrence_id": None,
                    "n_positive": 0,
                }
                if pending["_lineage"] is not None:
                    source = pending["_lineage"][original_row]
                    gt_record.update({k: source[k] for k in (
                        "source_image_id", "original_annotation_id", "occurrence_id"
                    )})
                    gt_record["lineage_status"] = "provided_unverified"
                gt_records.append(gt_record)

            positive_raw = pos[image].nonzero(as_tuple=True)[0].tolist()
            positive_gpu = torch.tensor(positive_raw, dtype=torch.long, device=scores.device)
            source_class = torch.tensor(
                [gt_records[int(targets[image, ri])]["class_id"] for ri in positive_raw],
                dtype=torch.long, device=scores.device,
            )
            raw_scores = scores[image, positive_gpu, source_class].detach().float().cpu().tolist()
            assigned_scores = target_scores[
                image, positive_gpu, source_class
            ].detach().float().cpu().tolist()
            edge_rows = []
            for pi, ri in enumerate(positive_raw):
                gi = int(targets[image, ri])
                require(gi in present_indices,
                        f"{branch}: positive raw={ri} maps to invalid augmented GT {gi}")
                scale = next((level for level in range(len(sizes))
                              if offsets[level] <= ri < offsets[level + 1]), None)
                require(scale is not None, "raw index outside feature offsets")
                local_r = ri - offsets[scale]
                grid_w = self._feature_hw[scale][1]
                class_id = gt_records[gi]["class_id"]
                edge_rows.append({
                    "aug_gt_idx": gi, "raw_id_global": ri,
                    "pyramid_level": scale, "grid_y": local_r // grid_w,
                    "grid_x": local_r % grid_w,
                    "source_gt_class_score": float(raw_scores[pi]),
                    "assigned_target_class_score": float(assigned_scores[pi]),
                })
                gt_records[gi]["n_positive"] += 1
            require(sum(r["n_positive"] for r in gt_records) == len(positive_raw),
                    "GT positive edge count lost")
            results.append({
                "batch_image_idx": image, "n_raw": nraw,
                "n_gt": len(gt_records), "n_positive": len(positive_raw),
                "feature_hw": self._feature_hw,
                "offsets": offsets,
                "GT": gt_records, "assigned_edges": edge_rows,
            })
        self._observations[branch] = {
            "branch": branch,
            "assigner_topk": int(getattr(
                self._assigners[branch], "topk", -1)),
            "assigner_topk2": int(getattr(
                self._assigners[branch], "topk2", -1)),
            "alpha": float(getattr(self._assigners[branch], "alpha", float("nan"))),
            "beta": float(getattr(self._assigners[branch], "beta", float("nan"))),
            "images": results,
        }

    def end_batch(self) -> dict:
        pending = self._pending
        require(pending is not None, "no active training microbatch")
        require(set(self._observations) == set(self.expected),
                f"incomplete TAL hooks: seen={list(self._observations)} expected={self.expected}")
        item = {k: v for k, v in pending.items() if not k.startswith("_")}
        item.pop("started_monotonic", None)
        item["branches"] = self._observations
        item["microbatch_elapsed_wall_s"] = time.monotonic() - pending["started_monotonic"]
        self._stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        self.completed_batches += 1
        self._pending = None
        self._observations = {}
        self._feature_hw = None
        return item

    def abort_batch(self, reason: str) -> None:
        if self._pending is None:
            return
        item = {k: v for k, v in self._pending.items() if not k.startswith("_")}
        item.pop("started_monotonic", None)
        item.update({"kind": "aborted_tal_microbatch", "reason": str(reason),
                     "observed_branches": list(self._observations)})
        self._stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        self.aborted_batches += 1
        self._pending = None
        self._observations = {}
        self._feature_hw = None

    def close(self) -> dict:
        if self._closed:
            return {"completed_batches": self.completed_batches,
                    "aborted_batches": self.aborted_batches}
        if self._pending is not None:
            self.abort_batch("recorder_close_during_active_batch")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._stream.close()
        self._closed = True
        return {"completed_batches": self.completed_batches,
                "aborted_batches": self.aborted_batches,
                "trace_sha256": file_sha256(self.file)}

