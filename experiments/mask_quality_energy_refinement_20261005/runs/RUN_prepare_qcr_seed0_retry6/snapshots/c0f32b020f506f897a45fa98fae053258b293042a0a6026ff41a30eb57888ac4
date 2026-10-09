"""Generate the independent 1024/256 screen from server COCO, without training.

Original official YOLODataset, overlap ordering and one-to-one TAL are reused.
Full P/GT/uint8 input are stored losslessly; full neck F is never saved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from online_runtime import (FrozenReplay, dump, load_json, load_asset,
                            resolve_runtime_config, sha256, tensor_sha)

SCHEMA = "qcr-original-tal-v1"


def ensure_space(cfg, added_bytes=0):
    root = Path(cfg["server_root"])
    reserve = int(cfg.get("minimum_root_free_gib", 10) * 1024**3)
    if shutil.disk_usage(root).free < reserve + added_bytes:
        raise RuntimeError("Screen stopped before violating the 10 GiB system-disk reserve")


def prepare_images(cfg, groups):
    """Extract exact frozen JPG members only, never the complete COCO archive."""
    root = Path(cfg["images"])
    result = []
    for domain, ids in groups.items():
        directory = root / f"{domain}2017"
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(cfg["coco_root"]) / f"{domain}2017.zip") as z:
            for iid in ids:
                path = directory / f"{iid:012d}.jpg"
                if path.exists() and bool(cfg.get("reuse_extracted_images", False)):
                    # The initial preparation already verified these immutable JPEGs;
                    # do not read the ZIP member again during a retry.
                    result.append(dict(image_id=iid, source_split=domain, bytes=path.stat().st_size, sha256="reused-existing-verified"))
                    continue
                name = f"{domain}2017/{iid:012d}.jpg"
                data = z.read(name)
                digest = hashlib.sha256(data).hexdigest()
                if path.exists():
                    if sha256(path) != digest:
                        raise AssertionError(f"Existing screen JPG differs: {iid}")
                else:
                    ensure_space(cfg, len(data))
                    tmp = path.with_suffix(".jpg.tmp")
                    tmp.write_bytes(data); tmp.replace(path)
                result.append(dict(image_id=iid, source_split=domain, bytes=len(data), sha256=digest))
    return result


@torch.no_grad()
def build_operator(image, h_maps, cfg, device):
    """Original 16x16 prototype ROI and full predicted-box Gram; no obsolete K."""
    from torchvision.ops import roi_align
    n = len(image["raw_ids"])
    ids = image["raw_ids"].to(device)
    h0 = torch.cat([h[0].flatten(1) for h in h_maps], 1).T[ids]
    p = image["proto"].to(device)
    b = image["boxes"].to(device).clone().clamp(0, 640)
    valid = ((b[:, 2:] - b[:, :2]) > 0).all(1)
    rois = torch.cat((b.new_zeros((n, 1)), b), 1)
    a = roi_align(p[None], rois, (16, 16), spatial_scale=p.shape[-1] / 640,
                  sampling_ratio=2, aligned=True).flatten(2).transpose(1, 2)
    flat = p.flatten(1).T
    yy, xx = torch.meshgrid(torch.arange(p.shape[-2], device=device),
                           torch.arange(p.shape[-1], device=device), indexing="ij")
    xx, yy = xx.flatten(), yy.flatten()
    gram = torch.empty((n, 32, 32), device=device, dtype=torch.float64)
    counts, fallbacks = [], []
    for j, box in enumerate(b):
        scaled = box * box.new_tensor([p.shape[-1]/640, p.shape[-2]/640]*2)
        support = (xx >= scaled[0]) & (xx < scaled[2]) & (yy >= scaled[1]) & (yy < scaled[3])
        count = int(support.sum()); counts.append(count); fallbacks.append(not bool(count))
        response = flat[support].double() if count else a[j].double()
        gram[j] = response.T @ response / len(response)
    if not all(torch.isfinite(v).all() for v in (a, gram, h0)):
        raise FloatingPointError("Nonfinite original prototype operator")
    return dict(schema="box_evidence_full_response_v1", source_fingerprint=image["fingerprint"],
        image_id=image["image_id"], raw_ids=image["raw_ids"], h0=h0.cpu(), A=a.float().cpu(),
        G=gram.float().cpu(), valid=valid.cpu(), support_count=torch.tensor(counts, dtype=torch.long),
        tiny_box_roi_fallback=torch.tensor(fallbacks, dtype=torch.bool),
        feature_channels=[int(h) for h in cfg["feature_channels"]],
        operator_scope="same fixed 16x16 ROI/full support Gram; arm-specific K constructed separately")


def loss_audit(official, criterion, pred, batch, image, ids, owners, cfg):
    """Actual official loss versus shared training helper at c0 and fixed delta."""
    from runtime_utils import loss_and_grad
    reports = official.audit_loss(criterion, pred, batch, ids, owners,
                                 image["target_boxes"].to(ids.device), image["segmentation_gain"], image["image_id"])
    gen = torch.Generator(device=ids.device).manual_seed(image["image_id"])
    payload = {k: image[k].to(ids.device) if torch.is_tensor(image[k]) else image[k]
               for k in ("proto", "masks", "owners", "target_boxes", "segmentation_gain")}
    comparisons = []
    for perturb in (False, True):
        c = image["c0"].to(ids.device).clone()
        if perturb:
            c += torch.randn(c.shape, generator=gen, device=ids.device) * .03
        c.requires_grad_(True)
        modified = pred["mask_coefficient"].detach().clone()
        modified[0, :, ids] = c.T
        reference = criterion.loss({**pred, "mask_coefficient": modified}, batch)[0][1]
        expected_grad = torch.autograd.grad(reference, c)[0]
        total, actual_grad = loss_and_grad(payload, c, len(c))
        actual = c.new_tensor(total / len(c))
        torch.testing.assert_close(actual, reference.detach(), atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)
        comparisons.append(dict(perturbed=perturb, official=float(reference.detach()),
            runtime=float(actual), loss_absolute_error=float((actual-reference).detach().abs()),
            coefficient_gradient_max_absolute_error=float((actual_grad-expected_grad).abs().max())))
    return dict(passed=True, tolerance=dict(atol=3e-5, rtol=3e-5), legacy_reference=reports,
                training_runtime=comparisons)


def compare_witness(image, old, features):
    """Report old/new relation; a disagreement never silently changes targets."""
    def close(a, b):
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.shape != b.shape:
            return dict(passed=False, shape_new=list(a.shape), shape_old=list(b.shape))
        error = (a.double()-b.double()).abs()
        return dict(passed=bool(torch.isfinite(error).all() and (error <= 3e-5+3e-5*b.double().abs()).all()),
                    max_abs_error=float(error.max()) if error.numel() else 0.)
    key = lambda r: (r["image_id"], r["annotation_id"], r["branch"], r["raw_id"], r["pyramid_level"], r["target_gt_idx"])
    result = dict(image_id=image["image_id"], input_exact=torch.equal(image["input_uint8"], old["input_uint8"]),
        masks_exact=torch.equal(image["masks"], old["masks"]),
        annotation_order_exact=image["all_annotation_ids"] == old["all_annotation_ids"],
        identity_exact=[key(r) for r in image["rows"]] == [key(r) for r in old["rows"]],
        geometry_exact=image["original_shape"] == old["original_shape"] and image["ratio_pad"] == old["ratio_pad"],
        numeric={name: close(image[name], old[name]) for name in ("proto", "c0", "boxes", "target_boxes")})
    if "F_reference" in old:
        result["F"] = [close(x[0], y) for x, y in zip(features, old["F_reference"])]
    if result["identity_exact"]:
        from ultralytics.utils import ops
        device = features[0].device
        count = 0
        for lo in range(0, len(image["rows"]), 4):
            hi = lo + 4
            new = ops.process_mask(image["proto"].to(device), image["c0"][lo:hi].to(device),
                                   image["boxes"][lo:hi].to(device), (640, 640), upsample=True)
            prior = ops.process_mask(old["proto"].to(device), old["c0"][lo:hi].to(device),
                                     old["boxes"][lo:hi].to(device), (640, 640), upsample=True)
            count += int((new != prior).sum())
        result["normal_input_mask_pixel_differences"] = count
    result["equivalent_within_declared_tolerance"] = all(result[k] for k in
        ("input_exact", "masks_exact", "annotation_order_exact", "identity_exact", "geometry_exact"))
    result["equivalent_within_declared_tolerance"] &= all(v["passed"] for v in result["numeric"].values())
    return result


def main(args):
    started = time.monotonic()
    cfg = resolve_runtime_config(load_json(args.config))
    out, root, cache = Path(args.out), Path(cfg["server_root"]), Path(cfg["cache"])
    out.mkdir(parents=True, exist_ok=True); root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True); (cache / "images").mkdir(exist_ok=True)
    if (out / "COMPLETE.json").exists() or (cache / "COMPLETE.json").exists():
        raise RuntimeError("Prepared data already complete; reuse it rather than regenerate")
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Preparation may run only on the authorized Linux CUDA server")
    ensure_space(cfg)
    if int(cfg.get("max_cache_gib", 38)) < 38 or int(cfg.get("minimum_root_free_gib", 5)) < 5:
        raise ValueError("QCR cache/reserve budget is below the locked minimum")
    sys.path.insert(0, cfg["source_python"])
    import legacy_prepare_cache as official
    from ultralytics.data.dataset import YOLODataset
    import ultralytics.data.augment as augmentation
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    from pycocotools.coco import COCO
    raw_split = load_json(cfg["split"])
    split = {"fit": [int(x) for x in raw_split["fit_images"]], "dev": [int(x) for x in raw_split["dev_images"]]}
    expected = {"fit": list(split["fit"]), "dev": list(split["dev"])}
    if split != expected or len(set(split["fit"] + split["dev"])) != len(split["fit"])+len(split["dev"]):
        raise AssertionError("QCR split must be fixed image-disjoint FIT/DEV")
    # The six legacy files are optional implementation witnesses, never a
    # prerequisite or a source of screen training/selection observations.
    witness_files = {}
    witness_val = []
    image_receipts = prepare_images(cfg, dict(train=split["fit"] + split["dev"], val=witness_val))
    dump(out / "SOURCE_IMAGES_EXTRACTED.json", image_receipts)
    replay = FrozenReplay(cfg)
    cfg["feature_channels"] = replay.feature_channels
    model, head = replay.source, replay.head
    cat_to_idx = {int(cid): i for i, cid in enumerate(sorted(COCO(cfg["annotations_train"]).getCatIds()))}
    model.args = SimpleNamespace(**replay.train_args)
    if model.args.mask_ratio != 1 or not model.args.overlap_mask:
        raise AssertionError("Official checkpoint mask_ratio/overlap setting differs")
    criterion = model.init_criterion().one2one
    signature = dict(schema=SCHEMA, weights_sha256=replay.weights_sha256,
        split_sha256=sha256(cfg["split"]), annotations={d: sha256(cfg[f"annotations_{d}"]) for d in ("train", "val")},
        dataset_code_sha256=sha256(official.__file__), prepare_code_sha256=sha256(__file__),
        forward="original FP32 eval batch1, no augmentation, no TF32", official_mask_ratio=model.args.mask_ratio,
        official_overlap_mask=model.args.overlap_mask, segmentation_gain=float(model.args.box))
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    identity_path = cache / "CACHE_IDENTITY.json"
    if identity_path.exists():
        previous_identity = load_json(identity_path)
        # Retry code may change only cache bookkeeping/compaction. Reuse the
        # previously frozen model/split/annotation identity for existing assets.
        if previous_identity.get("weights_sha256") != signature.get("weights_sha256") or previous_identity.get("split_sha256") != signature.get("split_sha256") or previous_identity.get("annotations", {}) != signature.get("annotations", {}):
            raise AssertionError("Partial preparation has a different model/split/annotation identity; preserve it")
        fingerprint = previous_identity["fingerprint"]
    dump(identity_path, dict(fingerprint=fingerprint, **signature))
    dump(out / "ENVIRONMENT.json", replay.import_info)
    converted, identity, source_meta = official.prepare_dataset(cfg, {**split, "val": witness_val},
        root / "dataset", fingerprint)
    captured, order = {}, {}
    hooks = [head.register_forward_pre_hook(lambda module, inputs: captured.__setitem__("F", list(inputs[0])))]
    hooks += [b[-1].register_forward_pre_hook(lambda module, inputs, l=l: captured.__setitem__(f"h{l}", inputs[0]))
              for l, b in enumerate(head.one2one_cv4)]
    original_rasterizer = augmentation.polygons2masks_overlap
    def rasterizer(*a, **kw):
        masks, indices = original_rasterizer(*a, **kw)
        order["indices"] = indices.copy()
        return masks, indices
    augmentation.polygons2masks_overlap = rasterizer
    metas, witnesses, audits = {}, [], []
    groups = {**split, "witness_val": witness_val}
    try:
        for group, ids in groups.items():
            if not ids:
                continue
            domain = "val" if group == "witness_val" else "train"
            listing = root / "dataset" / f"{group}_input.txt"
            listing.write_text("\n".join(str(converted / "images" / f"{domain}2017" / f"{iid:012d}.jpg") for iid in ids) + "\n")
            dataset = YOLODataset(img_path=str(listing), imgsz=640, batch_size=1,
                augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
                data={"names": model.names, "nc": 80, "channels": 3}, task="segment")
            positions = {int(Path(p).stem): j for j, p in enumerate(dataset.im_files)}
            if set(positions) != set(ids):
                raise AssertionError("Official dataset changed the fixed image membership")
            audited = False
            for iid in ids:
                existing_path = cache / "images" / f"{iid:012d}.pt.gz"
                existing_meta = existing_path.with_name(f"{iid:012d}.meta.json")
                if existing_path.exists() and existing_meta.exists():
                    metas[iid] = load_json(existing_meta)
                    continue
                raw_label = dataset.labels[positions[iid]]
                pre_ids = []
                for cls, polygon in zip(raw_label["cls"].flatten(), raw_label["segments"]):
                    key = np.r_[cls, polygon.flatten()].astype(np.float32)
                    found = [aid for arr, aid in identity[iid] if np.array_equal(key, arr)]
                    if len(found) != 1:
                        raise AssertionError(f"Ambiguous official annotation identity: {iid}")
                    pre_ids.append(found[0])
                order.clear(); sample = dataset[positions[iid]]
                ann_ids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
                if len(ann_ids) != len(sample["cls"]):
                    raise AssertionError("Official overlap sorting changed annotation length")
                batch = YOLODataset.collate_fn([sample])
                batch = {k: v.to(replay.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                if tuple(batch["img"].shape) != (1, 3, 640, 640) or tuple(batch["masks"].shape[-2:]) != (640, 640):
                    raise AssertionError("Original input/full mask resolution differs")
                captured.clear()
                with torch.no_grad():
                    _, raw = model(batch["img"].float() / 255)
                    pred = raw["one2one"]
                    assigned, _, _ = criterion.get_assigned_targets_and_loss(pred, batch)
                    fg, owner, target_boxes = (v[0] for v in assigned[:3])
                    raw_ids = torch.where(fg)[0]; owners = owner[raw_ids]
                    if len(set(owners.tolist())) != len(owners):
                        raise AssertionError("Official one-to-one TAL repeated an owner")
                    features = captured["F"]
                    hs = [captured[f"h{l}"] for l in range(3)]
                    levels = torch.cat([torch.full((h.shape[-2]*h.shape[-1],), l, device=replay.device, dtype=torch.long)
                                        for l, h in enumerate(hs)])[raw_ids]
                    coefficients = pred["mask_coefficient"][0].T[raw_ids]
                    boxes = head._get_decode_boxes(pred)[0].T[raw_ids]
                    scores, classes = pred["scores"][0, :, raw_ids].sigmoid().max(0)
                    boxes_original = ops.scale_boxes((640, 640), boxes.clone(), sample["ori_shape"], ratio_pad=sample["ratio_pad"])
                rows = []
                for k, (rid, gt_idx) in enumerate(zip(raw_ids.tolist(), owners.tolist())):
                    ann = source_meta[str(iid)]["annotations"][str(ann_ids[gt_idx])]
                    gt_box = boxes.new_tensor(ann["bbox"]); gt_box[2:] += gt_box[:2]
                    rows.append(dict(split=group, image_id=iid, annotation_id=int(ann["id"]), branch="one2one",
                        raw_id=rid, pyramid_level=int(levels[k]), level=int(levels[k]), target_gt_idx=gt_idx,
                        gt_index=gt_idx, candidate_index=k, area=float(ann["area"]),
                        box_iou=float(box_iou(gt_box[None], boxes_original[k:k+1])[0, 0]),
                        predicted_class_id=int(classes[k]), predicted_score=float(scores[k])))
                x = dict(schema=SCHEMA, fingerprint=fingerprint, image_id=iid, split=group, source_split=domain,
                    rows=rows, raw_ids=raw_ids.cpu(), c0=coefficients.cpu().contiguous(), boxes=boxes.cpu().contiguous(),
                    levels=levels.cpu(), owners=owners.cpu(), target_boxes=target_boxes[raw_ids].cpu().contiguous(),
                    predicted_classes=classes.cpu(), predicted_scores=scores.cpu(), original_shape=sample["ori_shape"],
                    ratio_pad=sample["ratio_pad"], all_annotation_ids=ann_ids, segmentation_gain=float(model.args.box),
                    input_shape=[640, 640], input_uint8=sample["img"].cpu().contiguous(),
                    proto=pred["proto"][0].detach().cpu().contiguous(), masks=sample["masks"][0].cpu().contiguous(),
                    source_F_shapes=[list(f.shape[1:]) for f in features], no_positive=not bool(rows))
                x["operator"] = build_operator(x, hs, cfg, replay.device)
                if rows and not audited and group != "witness_val":
                    audits.append(dict(split=group, image_id=iid, **loss_audit(official, criterion, pred, batch, x, raw_ids, owners, cfg)))
                    dump(out / "LOSS_EQUIVALENCE.json", audits); audited = True
                if iid in witness_files:
                    with gzip.open(witness_files[iid], "rb") as stream:
                        old = torch.load(io.BytesIO(stream.read()), map_location="cpu", weights_only=False)
                    witnesses.append(compare_witness(x, old, features)); del old
                    dump(out / "LEGACY_WITNESS_AUDIT.json", dict(witnesses=witnesses,
                        scope="implementation comparison only; independent screen owns regenerated P/targets", not_a_migration_receipt=True))
                if group != "witness_val":
                    keep = group == "dev" or any(
                        int(row["predicted_class_id"]) == cat_to_idx.get(int(source_meta[str(iid)]["annotations"][str(row["annotation_id"])] ["category_id"]), -1)
                        and float(row["box_iou"]) >= 0.75 for row in rows)
                    if keep:
                        path = cache / "images" / f"{iid:012d}.pt.gz"
                        if path.exists():
                            raise RuntimeError(f"Existing partial asset retained: {iid}; use a new preparation target")
                        raw_buffer = io.BytesIO(); torch.save(x, raw_buffer)
                        estimate = raw_buffer.tell(); ensure_space(cfg, estimate)
                        tmp = path.with_name(path.name + ".tmp")
                        with gzip.open(tmp, "wb", compresslevel=1) as stream:
                            stream.write(raw_buffer.getbuffer())
                        size = tmp.stat().st_size
                        used = sum(m["compressed_bytes"] for m in metas.values())
                        if used + size > float(cfg.get("max_cache_gib", 38)) * 1024**3:
                            raise RuntimeError("QCR lossless cache exceeds its locked capacity")
                        ensure_space(cfg); tmp.replace(path)
                        meta = dict(image_id=iid, split=group, n=len(rows), fingerprint=fingerprint,
                            compressed_bytes=size, compressed_sha256=sha256(path),
                            input_uint8_sha256=tensor_sha(x["input_uint8"]), masks_sha256=tensor_sha(x["masks"]),
                            proto_sha256=tensor_sha(x["proto"]), source_image_sha256=sha256(Path(cfg["images"]) / f"{domain}2017/{iid:012d}.jpg"))
                        dump(path.with_name(f"{iid:012d}.meta.json"), meta); metas[iid] = meta
                        index = {s: [dict(image_id=j, n=metas[j]["n"]) for j in split[s] if j in metas] for s in split}
                        dump(cache / "INDEX.json", index)
                        if len(metas) % 25 == 0:
                            progress = dict(stage="prepare_original_official", images=len(metas), planned=len(split["fit"])+len(split["dev"]),
                                candidates=sum(m["n"] for m in metas.values()), cache_bytes=used+size,
                                elapsed_s=time.monotonic()-started)
                            dump(out / "PROGRESS.json", progress); print(json.dumps(progress), flush=True)
                del x, batch, raw, pred, features, hs, coefficients
        replay.assert_unchanged()
        if not any(i in metas for i in split["fit"]) or not all(i in metas for i in split["dev"]) or {r["split"] for r in audits} != {"fit", "dev"}:
            raise AssertionError("Preparation did not retain a valid FIT/DEV cache")
        receipt = dict(complete=True, completed=True, passed=True, schema=SCHEMA, fingerprint=fingerprint,
            planned={s: len(v) for s, v in split.items()},
            positive_images={s: sum(bool(metas[i]["n"]) for i in ids if i in metas) for s, ids in split.items()},
            candidates={s: sum(metas[i]["n"] for i in ids if i in metas) for s, ids in split.items()},
            no_positive_images={s: [i for i in ids if i not in metas or not metas[i]["n"]] for s, ids in split.items()},
            skipped_fit_images=[i for i in split["fit"] if i not in metas],
            bytes=sum(m["compressed_bytes"] for m in metas.values()), elapsed_s=time.monotonic()-started,
            old_migration_required=False, legacy_witness_count=len(witnesses),
            legacy_witness_all_match=all(w["equivalent_within_declared_tolerance"] for w in witnesses) if witnesses else None,
            reference_loss="official one2one mask BCE; full640 overlap GT; proto bilinear640 align_corners=False; GT-box crop/normalized area; gain=args.box",
            no_model_training=True, frozen_original_buffers_unchanged=True, full_features_saved=False)
        dump(cache / "COMPLETE.json", receipt); dump(out / "COMPLETE.json", receipt)
    finally:
        for hook in hooks:
            hook.remove()
        augmentation.polygons2masks_overlap = original_rasterizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        dump(Path(args.out) / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        dump(Path(args.out) / "COMPLETE.json", dict(completed=False, complete=False, passed=False, error=repr(exc)))
        raise
