"""Private-label replay of ORIGINAL many/one TAL for a bounded local audit.

No training, top-k change, confidence filtering, or GT filtering occurs here.
The population is the frozen first 128 planned fast-screen fit images (including
empty images), or the first two effective members for smoke. No replacement is
allowed. Only labels and the dataset's writable cache are copied to the current
Run; original image files, dataset caches and candidate assets are read-only.
The caller owns the Run deadline and must check it around construction/get.

Source: mask_supervision_support_20261004/scripts/prepare_support.py, with its
identity/geometry checks retained and densification removed. The original
criterion.one2many and criterion.one2one are used on their own predictions.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from online_runtime import dump, load_index, load_json, sha256

KEYS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")


def normalized(value):
    if isinstance(value, (tuple, list)):
        return [normalized(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def cache_snapshot(root):
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*.cache")) if path.is_file()}


class AssignmentReplay:
    def __init__(self, replay, cfg, planned, out):
        self.replay, self.cfg, self.out = replay, cfg, Path(out)
        self.planned = [int(i) for i in planned]
        index = load_index(cfg)["fit"]
        expected = [int(entry["image_id"]) for entry in index[:128]]
        smoke = [int(entry["image_id"]) for entry in index[:128] if int(entry["n"]) > 0][:2]
        split = load_json(cfg["split"])
        if split["fit"] != expected or self.planned not in (expected, smoke):
            raise AssertionError("Only frozen 128 planned fit images or first two effective smoke images")
        if len(set(self.planned)) != len(self.planned):
            raise AssertionError("Duplicate planned images")
        self.source_dataset = Path(cfg["screen_root"]) / "dataset"
        ready_path = self.source_dataset / "DATASET_READY.json"
        conversion_path = self.source_dataset / "CONVERSION.json"
        source_path = self.source_dataset / "SOURCE_IMAGES.json"
        ready = load_json(ready_path)
        conversion, source_meta = load_json(conversion_path), load_json(source_path)
        converted = self.source_dataset / ready["directory"]
        identity_path = Path(cfg["cache"]) / "CACHE_IDENTITY.json"
        if ready["fingerprint"] != load_json(identity_path)["fingerprint"]:
            raise AssertionError("Original dataset/cache fingerprint differs")
        self.original_caches = cache_snapshot(self.source_dataset)
        private = self.out / "dataset"
        if private.exists():
            raise RuntimeError("Private dataset already exists; retain prior Run and use a new Run")
        labels, images = private / "labels/train2017", private / "images/train2017"
        labels.mkdir(parents=True)
        images.parent.mkdir(parents=True)
        images.symlink_to((Path(cfg["screen_root"]) / "data/images/train2017").resolve(), target_is_directory=True)
        self.serialized, self.label_hashes = {}, {}
        for iid in self.planned:
            meta = source_meta[str(iid)]
            if meta["source_split"] != "train":
                raise AssertionError("Expected train2017 source")
            source = converted / "labels/train2017" / Path(meta["file_name"]).with_suffix(".txt").name
            annotation_ids = conversion["source_annotation_ids"][str(iid)]
            if source.exists():
                dest = labels / source.name
                shutil.copyfile(source, dest)
                digest = sha256(source)
                if sha256(dest) != digest:
                    raise AssertionError("Private label copy changed")
                lines = source.read_text(encoding="utf-8").strip().splitlines()
            else:
                if annotation_ids:
                    raise AssertionError(f"Missing nonempty original label: {iid}")
                digest, lines = None, []
            if len(lines) != len(annotation_ids):
                raise AssertionError("Original polygon/annotation row count differs")
            self.serialized[iid] = [(np.asarray(line.split(), dtype=np.float32), int(aid))
                                   for line, aid in zip(lines, annotation_ids)]
            self.label_hashes[str(iid)] = dict(source=str(source), sha256=digest)
        listing = private / "fit_input.txt"
        listing.write_text("\n".join(str(images / Path(source_meta[str(i)]["file_name"]).name)
                                    for i in self.planned) + "\n", encoding="utf-8")
        model = replay.source
        replay.assert_unchanged()
        model.args = SimpleNamespace(**replay.train_args)
        if model.args.mask_ratio != 1 or not model.args.overlap_mask:
            raise AssertionError("Expected original mask_ratio=1 and overlap_mask=True")
        criterion = model.init_criterion()
        self.criterion_many, self.criterion_one = criterion.one2many, criterion.one2one
        self.criteria = {"one2many": self.criterion_many, "one2one": self.criterion_one}
        self.assigners = {name: {key: normalized(getattr(crit.assigner, key))
                                for key in ("topk", "topk2", "alpha", "beta", "num_classes")}
                          for name, crit in self.criteria.items()}
        if (self.criterion_many.assigner.topk, self.criterion_many.assigner.topk2) != (10, 10):
            raise AssertionError("Original many assigner differs from 10/10")
        if (self.criterion_one.assigner.topk, self.criterion_one.assigner.topk2) != (7, 1):
            raise AssertionError("Original one assigner differs from 7/1")
        from ultralytics.data.dataset import YOLODataset
        import ultralytics.data.augment as augmentation
        self.Dataset, self.augmentation = YOLODataset, augmentation
        self.dataset = YOLODataset(img_path=str(listing), imgsz=640, batch_size=1,
            augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
            data={"names": model.names, "nc": 80, "channels": 3}, task="segment")
        if any(not Path(path).is_relative_to(labels) for path in self.dataset.label_files):
            raise AssertionError("Official label/cache lookup escaped Run-private directory")
        self.positions = {int(Path(path).stem): k for k, path in enumerate(self.dataset.im_files)}
        if set(self.positions) != set(self.planned) or len(self.dataset) != len(self.planned):
            raise AssertionError("Dataset dropped/duplicated planned images; do not replace them")
        self.processed, self.no_positive = [], []
        self.source_hashes = {str(path): sha256(path) for path in
            (ready_path, conversion_path, source_path, identity_path, Path(cfg["cache"])/"INDEX.json", Path(cfg["split"]))}
        dump(self.out / "ASSIGNMENT_SETUP.json", dict(planned_image_ids=self.planned,
            assigners=self.assigners, source_hashes=self.source_hashes, source_labels=self.label_hashes,
            original_dataset_caches=self.original_caches, no_assignment_changes=True,
            all_gt_kept_for_both_branches=True, tolerance=dict(atol=3e-5, rtol=3e-5)))

    @torch.no_grad()
    def get(self, iid, x):
        """Return CPU selected rows and GPU full official inputs, F and batch.

        Branch dictionaries contain full_fg/full_target_gt_idx/full_target_bboxes,
        pred_masks [1,R,32], full_masks, batch_idx and imgsz for direct invocation
        of criterion.calculate_segmentation_loss. Only the caller's temporary
        prototype copy may be differentiated; returned coefficients are detached.
        """
        iid = int(iid)
        if iid not in self.positions or iid in self.processed or int(x["image_id"]) != iid:
            raise AssertionError("Unexpected, duplicate or mismatched planned image")
        pos = self.positions[iid]
        raw_label = self.dataset.labels[pos]
        pre_ids = []
        for cls, polygon in zip(raw_label["cls"].flatten(), raw_label["segments"]):
            key = np.r_[cls, polygon.flatten()].astype(np.float32)
            found = [aid for arr, aid in self.serialized[iid] if np.array_equal(key, arr)]
            if len(found) != 1:
                raise AssertionError(f"Ambiguous official serialized polygon identity: {iid}")
            pre_ids.append(found[0])
        order = {}
        old_rasterizer = self.augmentation.polygons2masks_overlap
        def rasterizer(*args, **kwargs):
            masks, indices = old_rasterizer(*args, **kwargs)
            order["indices"] = indices.copy()
            return masks, indices
        self.augmentation.polygons2masks_overlap = rasterizer
        try:
            sample = self.dataset[pos]
        finally:
            self.augmentation.polygons2masks_overlap = old_rasterizer
        if pre_ids and "indices" not in order:
            raise AssertionError("Official overlap sorting was not captured")
        ann_ids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
        if ann_ids != list(x["all_annotation_ids"]) or len(ann_ids) != len(sample["cls"]):
            raise AssertionError("Official GT/annotation order changed")
        for key, old in (("ratio_pad", "ratio_pad"), ("ori_shape", "original_shape")):
            if normalized(sample[key]) != normalized(x[old]):
                raise AssertionError(f"Original geometry changed: {key}")
        for current, cached in ((sample["img"], x["input_uint8"]), (sample["masks"][0], x["masks"])):
            if current.dtype != cached.dtype or not torch.equal(current.cpu(), cached.cpu()):
                raise AssertionError("Original input/overlap-mask bytes changed")
        batch = self.Dataset.collate_fn([sample])
        batch = {key: value.to(self.replay.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        features = []
        def capture(module, args):
            features.extend(value.detach() for value in args[0])
        hook = self.replay.head.register_forward_pre_hook(capture)
        try:
            _, raw = self.replay.source(batch["img"].float()/255)
        finally:
            hook.remove()
        if len(features) != 3 or any(list(f.shape) != [1, *s] for f, s in zip(features, x["source_F_shapes"])):
            raise AssertionError("Frozen head input feature shapes changed")
        errors = {}
        def close(key, actual, cached):
            cached = cached.to(actual.device)
            torch.testing.assert_close(actual, cached, atol=3e-5, rtol=3e-5)
            errors[key] = float((actual-cached).abs().max()) if actual.numel() else 0.
        branches = {}
        for name, criterion in self.criteria.items():
            pred = raw[name]
            fg, target_gt_idx, target_bboxes = criterion.get_assigned_targets_and_loss(pred, batch)[0][:3]
            ids = torch.where(fg[0])[0]
            owners = target_gt_idx[0, ids]
            full_levels = torch.cat([torch.full((f.shape[-2]*f.shape[-1],), level,
                dtype=torch.long, device=self.replay.device) for level, f in enumerate(pred["feats"])])
            levels = full_levels[ids].cpu()
            coefficients = pred["mask_coefficient"][0].T[ids]
            boxes = target_bboxes[0, ids]
            rows = [dict(split=x["split"], image_id=iid, annotation_id=int(ann_ids[owner]),
                branch=name, raw_id=rid, pyramid_level=int(levels[k]), target_gt_idx=owner)
                for k, (rid, owner) in enumerate(zip(ids.tolist(), owners.tolist()))]
            close(name+"_proto", pred["proto"][0], x["proto"])
            if name == "one2one":
                if not torch.equal(ids.cpu(), x["raw_ids"]) or not torch.equal(owners.cpu(), x["owners"]):
                    raise AssertionError("Original one-to-one raw/owner identity changed")
                if not torch.equal(levels, x["levels"]) or len(rows) != len(x["rows"]):
                    raise AssertionError("Original one-to-one level/count changed")
                if any(any(row[key] != old[key] for key in KEYS) for row, old in zip(rows, x["rows"])):
                    raise AssertionError("Original seven-field candidate identity changed")
                close("one_target_boxes", boxes, x["target_boxes"])
                close("one_c0", coefficients, x["c0"])
                close("one_pred_boxes", self.replay.head._get_decode_boxes(pred)[0].T[ids], x["boxes"])
                scores, classes = pred["scores"][0].T[ids].sigmoid().max(1)
                close("one_scores", scores, x["predicted_scores"])
                if not torch.equal(classes.cpu(), x["predicted_classes"]):
                    raise AssertionError("Original class predictions changed")
            branches[name] = dict(c0=coefficients.cpu().contiguous(), raw_ids=ids.cpu(), owners=owners.cpu(),
                target_boxes=boxes.cpu().contiguous(), levels=levels, rows=rows, masks=x["masks"],
                segmentation_gain=float(self.replay.source.args.box), all_annotation_ids=ann_ids,
                image_id=iid, split=x["split"], proto=x["proto"],
                full_fg=fg.detach(), full_target_gt_idx=target_gt_idx.detach(),
                full_target_bboxes=target_bboxes.detach(), pred_masks=pred["mask_coefficient"].permute(0,2,1).detach(),
                full_masks=batch["masks"], batch_idx=batch["batch_idx"].view(-1,1),
                imgsz=torch.tensor([640,640], dtype=coefficients.dtype, device=self.replay.device))
        if float(x["segmentation_gain"]) != float(self.replay.source.args.box):
            raise AssertionError("Original segmentation gain changed")
        self.processed.append(iid)
        if not len(x["raw_ids"]):
            self.no_positive.append(iid)
        self.replay.assert_unchanged()
        audit = dict(image_id=iid, passed=True, original_input_and_masks_bitexact=True,
            annotation_order_exact=True, one_identity_exact=True, max_errors=errors,
            many_candidates=len(branches["one2many"]["rows"]), one_candidates=len(x["rows"]),
            many_unique_gt=len(set(branches["one2many"]["owners"].tolist())),
            all_gt_kept_before_assignment=True, no_filter_to_one_covered_gt=True)
        return dict(image=x, one=branches["one2one"], many=branches["one2many"],
                    features=features, batch=batch, audit=audit)

    def finish(self):
        self.replay.assert_unchanged()
        if self.processed != self.planned:
            raise AssertionError("Incomplete/out-of-order population; do not replace empty images")
        if cache_snapshot(self.source_dataset) != self.original_caches:
            raise AssertionError("Original dataset cache was modified")
        for path, digest in self.source_hashes.items():
            if sha256(path) != digest:
                raise AssertionError(f"Original assignment input changed: {path}")
        for item in self.label_hashes.values():
            source = Path(item["source"])
            if (source.exists() if item["sha256"] is None else sha256(source) != item["sha256"]):
                raise AssertionError("Original converted label changed")
        return dict(passed=True, planned=len(self.planned), processed=len(self.processed),
            no_positive_images=self.no_positive, source_dataset_caches_unchanged=True,
            source_files_unchanged=True, assigners=self.assigners)
