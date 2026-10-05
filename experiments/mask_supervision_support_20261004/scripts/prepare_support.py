"""Freeze mask-only support densification on the original one-to-one raw predictions.

Only the official assigner's final topk2 changes from 1 to its existing topk=7.
Full-image GT competition is preserved before filtering to original supervised GTs.
Run exclusively on the authorized Linux CUDA server. No training occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256

SCHEMA = "mask-only-o2o-topk2-support-v1"
KEYS = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")


def normalized(value):
    if isinstance(value, (list, tuple)):
        return [normalized(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def cache_snapshot(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.cache")) if p.is_file()}


def atomic_tensor(path, value):
    if path.exists():
        raise RuntimeError(f"Preserve existing support asset; use an independent Run: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def main(args):
    started = time.monotonic()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = resolve_runtime_config(load_json(args.config))
    assert cfg['dense_topk'] == cfg['dense_topk2'] == 7 and cfg['original_topk2'] == 1
    assert cfg['original_loss_weight'] == .5
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Support preparation is permitted only on the Linux CUDA server")
    limit = float(cfg.get("prepare_max_seconds", 900))
    if limit <= 0:
        raise ValueError("A positive fixed preparation budget is required")

    def budget():
        if time.monotonic() - started > limit:
            raise TimeoutError(f"Preparation exceeded fixed {limit} second budget; partial assets are not complete")

    screen = Path(cfg["screen_root"])
    support = out / "support" if args.smoke else Path(cfg["support_dir"])
    if (support / "COMPLETE.json").exists() or (support / "INDEX.json").exists():
        raise RuntimeError(f"Support already has a receipt/index; refusing duplicate preparation: {support}")
    support.mkdir(parents=True, exist_ok=True)
    old_index = load_index(cfg)
    original_fit = old_index["fit"]
    split = load_json(screen / "SPLIT.json")
    planned = [int(e["image_id"]) for e in original_fit]
    if planned != [int(i) for i in split["fit"]] or len(planned) != 1024:
        raise AssertionError("Require the complete previously frozen 1024-image fit list in original order")
    if args.smoke:
        planned = [int(e["image_id"]) for e in original_fit if int(e["n"]) > 0][:2]
        if len(planned) != 2:
            raise AssertionError("Smoke requires the first two effective fit images")

    source_dataset = screen / "dataset"
    ready_path = source_dataset / "DATASET_READY.json"
    ready = load_json(ready_path)
    converted = source_dataset / ready["directory"]
    if not converted.is_dir():
        raise FileNotFoundError(f"Existing converted dataset is missing: {converted}")
    conversion_path = source_dataset / "CONVERSION.json"
    source_meta_path = source_dataset / "SOURCE_IMAGES.json"
    conversion, source_meta = load_json(conversion_path), load_json(source_meta_path)
    cache_identity_path = Path(cfg["cache"]) / "CACHE_IDENTITY.json"
    if ready["fingerprint"] != load_json(cache_identity_path)["fingerprint"]:
        raise AssertionError("Original dataset/cache fingerprints differ")
    old_caches = cache_snapshot(source_dataset)

    # Image symlinks preserve private lexical paths; all writable labels/cache live in this Run.
    private = out / "dataset"
    labels = private / "labels/train2017"
    images = private / "images/train2017"
    if private.exists():
        raise RuntimeError(f"Private dataset exists; do not overwrite a previous attempt: {private}")
    labels.mkdir(parents=True)
    images.parent.mkdir(parents=True)
    images.symlink_to((screen / "data/images/train2017").resolve(), target_is_directory=True)
    listing = private / "fit_input.txt"
    label_hashes, serialized = {}, {}
    for iid in planned:
        budget()
        meta = source_meta[str(iid)]
        if meta["source_split"] != "train":
            raise AssertionError("Fit image is not from train2017")
        filename = Path(meta["file_name"]).name
        src = converted / "labels/train2017" / Path(filename).with_suffix(".txt")
        dst = labels / src.name
        annotation_ids = conversion["source_annotation_ids"][str(iid)]
        if src.exists():
            shutil.copyfile(src, dst)
            digest = sha256(src)
            if sha256(dst) != digest:
                raise AssertionError("Private copied label differs")
            lines = src.read_text(encoding="utf-8").strip().splitlines()
            label_hashes[str(iid)] = dict(source=str(src), sha256=digest, missing=False)
        else:
            if annotation_ids:
                raise AssertionError(f"Missing nonempty original label: {iid}")
            lines = []
            label_hashes[str(iid)] = dict(source=str(src), sha256=None, missing=True)
        if len(lines) != len(annotation_ids):
            raise AssertionError(f"Converted polygon/annotation identity count differs: {iid}")
        serialized[iid] = [(np.asarray(s.split(), dtype=np.float32), int(aid))
                           for s, aid in zip(lines, annotation_ids)]
    listing.write_text("\n".join(str(images / Path(source_meta[str(i)]["file_name"]).name)
                                  for i in planned) + "\n", encoding="utf-8")

    replay = FrozenReplay(cfg)
    model, head = replay.source, replay.head
    model.args = SimpleNamespace(**replay.train_args)
    if model.args.mask_ratio != 1 or not model.args.overlap_mask:
        raise AssertionError("Expected original checkpoint full640 overlap supervision")
    from ultralytics.data.dataset import YOLODataset
    import ultralytics.data.augment as augmentation
    import ultralytics.utils.loss as loss_module
    import ultralytics.utils.tal as tal_module

    sparse = model.init_criterion().one2one
    dense = deepcopy(sparse)
    if sparse.assigner.topk != 7 or sparse.assigner.topk2 != 1:
        raise AssertionError("Original one-to-one assigner configuration changed")
    dense.assigner.topk2 = dense.assigner.topk
    attributes = ("topk", "topk2", "num_classes", "alpha", "beta", "stride", "stride_val", "eps")
    sparse_settings = {k: normalized(getattr(sparse.assigner, k)) for k in attributes}
    dense_settings = {k: normalized(getattr(dense.assigner, k)) for k in attributes}
    if [k for k in attributes if sparse_settings[k] != dense_settings[k]] != ["topk2"]:
        raise AssertionError("Only topk2 may differ")
    dataset = YOLODataset(img_path=str(listing), imgsz=640, batch_size=1,
        augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
        data={"names": model.names, "nc": 80, "channels": 3}, task="segment")
    if any(not Path(p).is_relative_to(labels) for p in dataset.label_files):
        raise AssertionError("YOLODataset labels escaped the private directory")
    positions = {int(Path(p).stem): j for j, p in enumerate(dataset.im_files)}
    if set(positions) != set(planned) or len(dataset.im_files) != len(planned):
        raise AssertionError("Official dataset removed/duplicated planned images")
    budget()
    order = {}
    original_rasterizer = augmentation.polygons2masks_overlap

    def rasterizer(*a, **kw):
        masks, indices = original_rasterizer(*a, **kw)
        order["indices"] = indices.copy()
        return masks, indices

    augmentation.polygons2masks_overlap = rasterizer
    entries, no_positive = [], []
    extra_counts, cross_level = Counter(), Counter()
    max_errors = {k: 0. for k in ("target_boxes", "proto", "c0", "boxes", "scores")}
    n_original = n_total = total_unselected_dense = 0
    per_image = []
    source_hashes = {str(p): sha256(p) for p in (Path(args.config), Path(__file__),
        Path(inspect.getfile(FrozenReplay)), ready_path, conversion_path, source_meta_path,
        cache_identity_path, screen / "SPLIT.json", Path(cfg["cache"]) / "INDEX.json",
        Path(inspect.getfile(YOLODataset)), Path(augmentation.__file__),
        Path(loss_module.__file__), Path(tal_module.__file__))}

    def close(name, actual, expected):
        if actual.shape != expected.shape:
            raise AssertionError(f"{name} shape differs: {actual.shape} vs {expected.shape}")
        torch.testing.assert_close(actual, expected.to(actual.device), atol=3e-5, rtol=3e-5)
        if actual.numel():
            max_errors[name] = max(max_errors[name], float((actual - expected.to(actual.device)).abs().max()))

    try:
        for position, iid in enumerate(planned):
            budget()
            x = load_asset(cfg, iid)
            raw_label = dataset.labels[positions[iid]]
            pre_ids = []
            for cls, polygon in zip(raw_label["cls"].flatten(), raw_label["segments"]):
                key = np.r_[cls, polygon.flatten()].astype(np.float32)
                found = [aid for arr, aid in serialized[iid] if np.array_equal(key, arr)]
                if len(found) != 1:
                    raise AssertionError(f"Ambiguous official serialized polygon identity: {iid}")
                pre_ids.append(found[0])
            order.clear()
            sample = dataset[positions[iid]]
            if pre_ids and "indices" not in order:
                raise AssertionError("Official overlap ordering was not observed")
            ann_ids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
            if ann_ids != list(x["all_annotation_ids"]) or len(ann_ids) != len(sample["cls"]):
                raise AssertionError(f"Official GT/annotation order differs: {iid}")
            for key in ("ratio_pad", "ori_shape"):
                old_key = "original_shape" if key == "ori_shape" else key
                if normalized(sample[key]) != normalized(x[old_key]):
                    raise AssertionError(f"Original {key} changed: {iid}")
            for current, old in ((sample["img"], x["input_uint8"]), (sample["masks"][0], x["masks"])):
                if current.dtype != old.dtype or not torch.equal(current.cpu(), old.cpu()):
                    raise AssertionError(f"Original input/mask bytes changed: {iid}")
            batch = YOLODataset.collate_fn([sample])
            batch = {k: v.to(replay.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                _, raw = model(batch["img"].float() / 255)
                pred = raw["one2one"]
                s = sparse.get_assigned_targets_and_loss(pred, batch)[0]
                d = dense.get_assigned_targets_and_loss(pred, batch)[0]
                sid = torch.where(s[0][0])[0]
                so = s[1][0, sid]
                did = torch.where(d[0][0])[0]
                do = d[1][0, did]
                if not torch.equal(sid.cpu(), x["raw_ids"]) or not torch.equal(so.cpu(), x["owners"]):
                    raise AssertionError(f"Original one-to-one identity differs: {iid}")
                if len(set(so.tolist())) != len(so):
                    raise AssertionError("Original one-to-one assignment has repeated GT")
                close("target_boxes", s[2][0, sid], x["target_boxes"])
                close("proto", pred["proto"][0], x["proto"])
                close("c0", pred["mask_coefficient"][0].T[sid], x["c0"])
                close("boxes", head._get_decode_boxes(pred)[0].T[sid], x["boxes"])
                scores, classes = pred["scores"][0].T[sid].sigmoid().max(1)
                close("scores", scores, x["predicted_scores"])
                if not torch.equal(classes.cpu(), x["predicted_classes"]):
                    raise AssertionError("Original predicted classes differ")
                sparse_pairs = set(zip(sid.tolist(), so.tolist()))
                dense_pairs = set(zip(did.tolist(), do.tolist()))
                if not sparse_pairs <= dense_pairs:
                    raise AssertionError("Original (raw, owner) must remain in the dense result")
                all_levels = torch.cat([torch.full((f.shape[-2] * f.shape[-1],), l,
                    dtype=torch.long, device=replay.device) for l, f in enumerate(pred["feats"])])
                if not torch.equal(all_levels[sid].cpu(), x["levels"]):
                    raise AssertionError("Original pyramid level mapping differs")
                # Filter AFTER full-GT assignment and conflict resolution.
                originals = {int(owner): (k, int(rid)) for k, (rid, owner) in enumerate(zip(sid.tolist(), so.tolist()))}
                raw_ids, owners = sid.tolist(), so.tolist()
                weights = [1.] * len(raw_ids)
                flags = [True] * len(raw_ids)
                extras_image, unselected = 0, sum(int(owner) not in originals for owner in do.tolist())
                for owner in sorted(originals):
                    k, original_raw = originals[owner]
                    extra = sorted(int(rid) for rid, oo in dense_pairs if oo == owner and rid != original_raw)
                    extra_counts[len(extra)] += 1
                    if extra:
                        weights[k] = .5
                        raw_ids.extend(extra); owners.extend([owner] * len(extra))
                        weights.extend([.5 / len(extra)] * len(extra)); flags.extend([False] * len(extra))
                        for rid in extra:
                            cross_level[f"P{int(all_levels[original_raw]) + 3}->P{int(all_levels[rid]) + 3}"] += 1
                    extras_image += len(extra)
                ids_t = torch.tensor(raw_ids, dtype=torch.long, device=replay.device)
                owners_t = torch.tensor(owners, dtype=torch.long)
                weights_t = torch.tensor(weights, dtype=torch.float64)
                if len(set(raw_ids)) != len(raw_ids) or set(owners) != set(originals):
                    raise AssertionError("Duplicate raw identity or new supervised GT introduced")
                for owner in originals:
                    if abs(float(weights_t[owners_t == owner].sum()) - 1.) > 1e-12:
                        raise AssertionError("Per-GT supervision mass must remain exactly one")
                if abs(float(weights_t.sum()) - len(originals)) > 1e-10:
                    raise AssertionError("Image supervision mass differs from original GT count")
                levels = all_levels[ids_t].cpu()
                rows = [dict(split="fit", image_id=iid, annotation_id=int(ann_ids[owner]), branch="one2one",
                    raw_id=rid, pyramid_level=int(levels[k]), target_gt_idx=owner)
                    for k, (rid, owner) in enumerate(zip(raw_ids, owners))]
                for k in range(len(sid)):
                    if any(rows[k][key] != x["rows"][k][key] for key in KEYS):
                        raise AssertionError("Original permanent row identity changed")
                boxes_t = d[2][0, ids_t].detach().cpu().contiguous()
                close("target_boxes", boxes_t[:len(sid)], x["target_boxes"])
                payload = dict(schema=SCHEMA, image_id=iid, split="fit", raw_ids=ids_t.cpu(),
                    owners=owners_t, target_boxes=boxes_t, weights=weights_t,
                    is_original=torch.tensor(flags, dtype=torch.bool), original_count=len(sid), levels=levels,
                    rows=rows, source_image_cache_sha256=x["_asset_integrity"]["compressed_sha256"],
                    source_fingerprint=x["fingerprint"], source_label_sha256=label_hashes[str(iid)]["sha256"],
                    assignment="original one2one raw with topk2=topk=7; full GT before owner filtering")
            filename = f"{iid:012d}.pt"
            atomic_tensor(support / filename, payload)
            entry = dict(image_id=iid, n_original=len(sid), n_total=len(raw_ids), filename=filename,
                sha256=sha256(support / filename), n_extra=extras_image,
                source_image_cache_sha256=payload["source_image_cache_sha256"])
            entries.append(entry)
            per_image.append(dict(image_id=iid, dense_before_gt_filter=len(did),
                excluded_positions_for_other_gt=unselected, n_original=len(sid), n_extra=extras_image))
            if not len(sid):
                no_positive.append(iid)
            n_original += len(sid); n_total += len(raw_ids); total_unselected_dense += unselected
            index = dict(schema=SCHEMA, split="fit", smoke=bool(args.smoke), planned=len(planned),
                effective=len(entries) - len(no_positive), processed=len(entries), planned_image_ids=planned,
                no_positive_images=no_positive, images=entries, complete=False)
            dump(support / "INDEX.json", index)
            dump(out / "PROGRESS.json", dict(stage="freeze_mask_support", processed=len(entries), planned=len(planned),
                n_original=n_original, n_total=n_total, elapsed_s=time.monotonic() - started))
            if position % 25 == 0 or len(entries) == len(planned):
                print(json.dumps(dict(processed=len(entries), planned=len(planned), n_original=n_original,
                    n_total=n_total, elapsed_s=time.monotonic() - started)), flush=True)
            del x, batch, raw, pred, payload, s, d
        budget()
        replay.assert_unchanged()
        if cache_snapshot(source_dataset) != old_caches:
            raise AssertionError("Original dataset cache changed during private replay")
        for path, digest in source_hashes.items():
            if sha256(path) != digest:
                raise AssertionError(f"Input/code changed during preparation: {path}")
        for item in label_hashes.values():
            p = Path(item["source"])
            if item["missing"]:
                if p.exists():
                    raise AssertionError("Previously missing original label appeared")
            elif sha256(p) != item["sha256"]:
                raise AssertionError("Original converted label changed")
        if len(entries) != len(planned):
            raise AssertionError("Incomplete population cannot authorize training")
        budget()
        index["complete"] = True
        dump(support / "INDEX.json", index)
        audit = dict(passed=True, schema=SCHEMA, smoke=bool(args.smoke), source_hashes=source_hashes,
            source_labels=label_hashes, original_dataset_cache_hashes=old_caches,
            sparse_assigner=sparse_settings, dense_assigner=dense_settings,
            only_assigner_change="topk2:1->7", assignment_input_branch="one2one",
            full_gt_competition_preserved=True, filter_after_assignment=True,
            original_identity_subset=True, original_candidates_first=True, extra_order="owner index, raw ID",
            max_replay_errors=max_errors, tolerance=dict(atol=3e-5, rtol=3e-5),
            original_supervised_gt=n_original, positions_total=n_total, positions_extra=n_total - n_original,
            extra_count_per_gt_histogram={str(k): v for k, v in sorted(extra_counts.items())},
            extra_cross_level_counts=dict(sorted(cross_level.items())),
            excluded_dense_positions_for_other_gt=total_unselected_dense, per_image=per_image,
            weighting="per GT original 1 if no extras; else original .5 and each extra .5/m; average over original GT count",
            no_training=True, original_weights_sha256=replay.weights_sha256, environment=replay.import_info,
            elapsed_s=time.monotonic() - started, support_dir=str(support), index_sha256=sha256(support / "INDEX.json"))
        dump(out / "ASSIGNMENT_AUDIT.json", audit)
        dump(support / "ASSIGNMENT_AUDIT.json", audit)
        receipt = dict(passed=True, completed=True, complete=True, schema=SCHEMA, smoke=bool(args.smoke),
            planned=len(planned), effective=len(planned) - len(no_positive), no_positive_images=no_positive,
            n_original=n_original, n_total=n_total, support_dir=str(support),
            index_sha256=sha256(support / "INDEX.json"), audit_sha256=sha256(out / "ASSIGNMENT_AUDIT.json"),
            elapsed_s=time.monotonic() - started)
        dump(support / "COMPLETE.json", receipt)
        dump(out / "COMPLETE.json", receipt)
    finally:
        augmentation.polygons2masks_overlap = original_rasterizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        dump(Path(args.out) / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        dump(Path(args.out) / "COMPLETE.json", dict(complete=False, completed=False, passed=False, error=repr(exc)))
        raise
