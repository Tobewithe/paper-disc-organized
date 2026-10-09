"""Cache full frozen neck maps and official one-to-one TAL supervision.

No training, candidate filtering, FP16 compression, or geometric reassignment.
Adapted from coefficient_predictability_20260924/scripts/official_pipeline.py.
--limit N is a smoke view (first N fit, min(3,N) dev/val) of the frozen split.
The same per-image cache is resumed by the later invocation without --limit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch
import torch.nn.functional as functional
import ultralytics
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco, coco91_to_coco80_class
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as augment_module
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou


SCHEMA = "bgcr-native-full-neck-official-tal-v1"
GROUPS = ("fit", "dev", "val")
MAX_CACHE_BYTES = 250_000_000_000


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def make_directory_link(link, target):
    """Windows junctions need no symlink privilege; never move/delete data."""
    link, target = Path(link), Path(target).resolve()
    if link.exists():
        if not os.path.samefile(link, target):
            raise RuntimeError(f"existing image link has a different target: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # No cross-shell deletion/moving. Single PowerShell native junction creation.
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        command = ("$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "
                   + quote(link) + " -Target " + quote(target) + " | Out-Null")
        powershell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        subprocess.run([str(powershell), "-NoProfile", "-Command", command], check=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def prepare_dataset(cfg, split, root, fingerprint):
    """Official conversion, with exact ann-ID tracking before and after sorting."""
    marker = root / "DATASET_READY.json"
    if marker.exists():
        ready = read_json(marker)
        if ready["fingerprint"] != fingerprint:
            raise RuntimeError("converted dataset fingerprint mismatch")
        converted = root / ready["directory"]
        conversion = read_json(root / "CONVERSION.json")
        source_meta = read_json(root / "SOURCE_IMAGES.json")
        identities = {int(k): v for k, v in conversion["source_annotation_ids"].items()}
    else:
        inp = root / "conversion_input"
        inp.mkdir(parents=True, exist_ok=True)
        identities, source_meta, exclusions = {}, {}, Counter()
        classmap = coco91_to_coco80_class()
        for domain in ("train", "val"):
            data = read_json(cfg[f"annotations_{domain}"])
            selected = set(split["val"] if domain == "val" else split["fit"] + split["dev"])
            data["images"] = [r for r in data["images"] if int(r["id"]) in selected]
            data["annotations"] = [r for r in data["annotations"] if int(r["image_id"]) in selected]
            ims = {int(r["id"]): r for r in data["images"]}
            if set(ims) != selected:
                raise RuntimeError(f"{domain}: frozen image IDs absent in annotation file: {sorted(selected-set(ims))[:20]}")
            by_image = defaultdict(list)
            for ann in data["annotations"]:
                by_image[int(ann["image_id"])].append(ann)
            for iid in selected:
                kept, seen = [], []
                for ann in by_image[iid]:
                    if ann.get("iscrowd", False):
                        exclusions["crowd"] += 1
                        continue
                    box = np.array(ann["bbox"], dtype=np.float64)
                    box[:2] += box[2:] / 2
                    box[[0, 2]] /= ims[iid]["width"]
                    box[[1, 3]] /= ims[iid]["height"]
                    if box[2] <= 0 or box[3] <= 0:
                        exclusions["invalid_box"] += 1
                        continue
                    key = [classmap[ann["category_id"] - 1], *box.tolist()]
                    if key in seen:
                        exclusions["converter_duplicate_box"] += 1
                        continue
                    if not isinstance(ann.get("segmentation"), list) or not ann["segmentation"]:
                        raise RuntimeError(f"unsupported non-crowd segmentation: annotation {ann['id']}")
                    seen.append(key)
                    kept.append(int(ann["id"]))
                identities[iid] = kept
                source_meta[str(iid)] = {
                    "file_name": ims[iid]["file_name"], "source_split": domain,
                    "annotations": {str(a["id"]): {k: a[k] for k in ("id", "image_id", "bbox", "area", "category_id")}
                                    for a in by_image[iid]},
                }
            write_json(inp / f"instances_{domain}2017.json", data)
            del data, by_image
        # Unique uncommitted conversion directories leave failed attempts intact.
        converted = root / ("official_data_" + uuid.uuid4().hex[:12])
        convert_coco(str(inp), str(converted), use_segments=True)
        write_json(root / "CONVERSION.json", {
            "excluded": dict(exclusions), "source_annotation_ids": identities,
            "scope": "official convert_coco merged polygons; exact serialized polygon identity; original COCO GT for final evaluation",
        })
        write_json(root / "SOURCE_IMAGES.json", source_meta)
        for domain in ("train", "val"):
            make_directory_link(converted / "images" / f"{domain}2017", Path(cfg["images"]) / f"{domain}2017")
        write_json(marker, {"fingerprint": fingerprint, "directory": converted.name})
    mapping = {}
    for group, ids in split.items():
        domain = "val" if group == "val" else "train"
        for iid in ids:
            label = converted / "labels" / f"{domain}2017" / Path(source_meta[str(iid)]["file_name"]).with_suffix(".txt")
            lines = label.read_text(encoding="utf-8").strip().splitlines() if label.exists() else []
            if len(lines) != len(identities[iid]):
                raise RuntimeError(f"converter annotation identity mismatch: image {iid}, {len(lines)} vs {len(identities[iid])}")
            mapping[iid] = [(np.asarray(s.split(), dtype=np.float32), aid) for s, aid in zip(lines, identities[iid])]
    return converted, mapping, source_meta


def replay_loss(coeff, proto, masks, owners, boxes, gain):
    """Exact official per-image candidate-average mask BCE, after proto upsample."""
    up = functional.interpolate(proto[None], (640, 640), mode="bilinear", align_corners=False)[0]
    value = coeff.sum() * 0
    for k in range(len(coeff)):
        z = torch.einsum("c,chw->hw", coeff[k], up)
        y = (masks == owners[k] + 1).float()
        loss = functional.binary_cross_entropy_with_logits(z, y, reduction="none")
        loss = ops.crop_mask(loss[None], boxes[k:k+1])[0]
        area = ((boxes[k, 2:] - boxes[k, :2]) / 640).prod()
        value = value + loss.mean() / area
    return value / len(coeff) * gain


def audit_loss(criterion, pred, batch, ids, owners, target_boxes, gain, seed):
    records = []
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    for perturb in (False, True):
        coefficients = pred["mask_coefficient"][0].T[ids].detach().clone()
        if perturb:
            coefficients += torch.randn(coefficients.shape, generator=gen, device=ids.device) * .03
        coefficients.requires_grad_(True)
        modified = pred["mask_coefficient"].detach().clone()
        modified[0, :, ids] = coefficients.T
        with torch.enable_grad():
            reference = criterion.loss({**pred, "mask_coefficient": modified}, batch)[0][1]
            g_ref = torch.autograd.grad(reference, coefficients)[0]
            replay = replay_loss(coefficients, pred["proto"][0].detach(), batch["masks"][0],
                                 owners, target_boxes, gain)
            g_replay = torch.autograd.grad(replay, coefficients)[0]
        torch.testing.assert_close(reference, replay, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(g_ref, g_replay, atol=3e-5, rtol=3e-5)
        records.append({"perturbed": perturb, "official": float(reference.detach()),
                        "value_abs_error": float((reference - replay).detach().abs()),
                        "coefficient_gradient_max_abs_error": float((g_ref - g_replay).abs().max())})
    return records


def main(args):
    cfg = read_json(args.config)
    root, out = Path(cfg["cache"]), Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = root / "images"
    images_dir.mkdir(exist_ok=True)
    split_file = read_json(cfg["split"])
    split = {group: [int(x) for x in split_file[group]] for group in GROUPS}
    flat = [iid for group in GROUPS for iid in split[group]]
    if len(flat) != len(set(flat)):
        raise RuntimeError("duplicate image IDs within or across frozen splits")
    requested = {group: list(ids) for group, ids in split.items()}
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        requested = {g: ids[:args.limit if g == "fit" else min(3, args.limit)] for g, ids in split.items()}
    if ultralytics.__version__ != "8.4.100":
        raise RuntimeError(f"wrong Ultralytics version: {ultralytics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("full forward cache requires the authorized laptop GPU")
    torch.set_num_threads(int(cfg.get("cpu_threads", 6)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    signature = {
        "schema": SCHEMA, "ultralytics": ultralytics.__version__, "imgsz": 640,
        "forward": "FP32 eval, batch1, no TF32, no augment, frozen original checkpoint",
        "inputs": {k: {"path": str(cfg[k]), "sha256": sha256(cfg[k])}
                   for k in ("weights", "split", "annotations_train", "annotations_val")},
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if (root / "CACHE_IDENTITY.json").exists():
        previous = read_json(root / "CACHE_IDENTITY.json")
        if previous["fingerprint"] != fingerprint:
            raise RuntimeError("cache input identity changed; refusing to mix or overwrite")
    else:
        write_json(root / "CACHE_IDENTITY.json", {"fingerprint": fingerprint, **signature})
    write_json(out / "ENVIRONMENT.json", {
        **signature, "torch": torch.__version__, "python": sys.version,
        "ultralytics_file": ultralytics.__file__, "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda, "cache_fingerprint": fingerprint,
        "code_sha256": sha256(__file__), "limit": args.limit,
        "max_cache_bytes": MAX_CACHE_BYTES,
    })
    converted, identity, source_meta = prepare_dataset(cfg, split, root, fingerprint)
    wrapper = YOLO(str(cfg["weights"]))
    model = wrapper.model.cuda().float().eval()
    model.args = SimpleNamespace(**wrapper.ckpt["train_args"])
    if model.args.mask_ratio != 1 or not model.args.overlap_mask:
        raise RuntimeError("unexpected pretrained label configuration; no silent override")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    criterion = model.init_criterion().one2one
    head = model.model[-1]
    captured = {}

    def neck_hook(_module, inputs):
        captured["F"] = [v.detach().clone() for v in inputs[0]]

    handles = [head.register_forward_pre_hook(neck_hook)]
    handles += [branch[-1].register_forward_pre_hook(
        lambda _module, inputs, level=level: captured.__setitem__(f"h{level}", inputs[0].detach().clone()))
        for level, branch in enumerate(head.one2one_cv4)]
    original_rasterizer = augment_module.polygons2masks_overlap
    order = {}

    def record_order(*a, **kw):
        masks, indices = original_rasterizer(*a, **kw)
        order["indices"] = indices.copy()
        return masks, indices

    augment_module.polygons2masks_overlap = record_order
    metas = {}
    for path in images_dir.glob("*.meta.json"):
        meta = read_json(path)
        iid = int(meta["image_id"])
        artifact = images_dir / f"{iid:012d}.pt"
        if meta.get("fingerprint") != fingerprint or iid not in set(flat):
            raise RuntimeError(f"foreign cache image metadata: {path}")
        if not artifact.exists() or artifact.stat().st_size != meta["artifact_bytes"]:
            raise RuntimeError(f"incomplete/corrupt committed cache entry: {path}")
        metas[iid] = meta
    used_bytes = sum(p.stat().st_size for p in images_dir.iterdir() if p.is_file())
    start = time.monotonic()
    audits = read_json(root / "REPLAY_AUDIT.json") if (root / "REPLAY_AUDIT.json").exists() else []
    processed = 0

    def snapshot(stage, group=None):
        index = {g: [{"image_id": iid, "n": metas[iid]["n"]} for iid in split[g] if iid in metas] for g in GROUPS}
        run_index = {g: [{"image_id": iid, "n": metas[iid]["n"]} for iid in requested[g] if iid in metas] for g in GROUPS}
        write_json(root / "INDEX.json", index)
        write_json(out / "INDEX.json", run_index)
        state = {"stage": stage, "group": group, "elapsed_seconds": time.monotonic()-start,
                 "new_images_this_invocation": processed, "cache_bytes_images": used_bytes,
                 "planned": {g: len(split[g]) for g in GROUPS},
                 "requested": {g: len(requested[g]) for g in GROUPS},
                 "completed": {g: len(index[g]) for g in GROUPS},
                 "positive_candidates": {g: sum(x["n"] for x in index[g]) for g in GROUPS},
                 "no_positive_images": {g: [x["image_id"] for x in index[g] if x["n"] == 0] for g in GROUPS}}
        write_json(root / "PROGRESS.json", state)
        write_json(out / "PROGRESS.json", state)
        write_json(root / "REPLAY_AUDIT.json", audits)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return state

    try:
        snapshot("resume")
        for group in GROUPS:
            domain = "val" if group == "val" else "train"
            pending = [iid for iid in requested[group] if iid not in metas]
            if not pending:
                continue
            image_paths = [converted / "images" / f"{domain}2017" / source_meta[str(iid)]["file_name"] for iid in pending]
            missing = [str(p) for p in image_paths if not p.is_file()]
            if missing:
                write_json(out / "MISSING_IMAGES.json", missing)
                raise FileNotFoundError(f"{len(missing)} frozen images missing; no replacement sampling")
            dataset_list = converted / f"{group}_{'smoke' if args.limit is not None else 'full'}_pending.txt"
            dataset_list.write_text("\n".join(map(str, image_paths)) + "\n", encoding="utf-8")
            dataset = YOLODataset(img_path=str(dataset_list), imgsz=640, batch_size=1,
                augment=False, hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
                data={"names": wrapper.names, "nc": 80, "channels": 3}, task="segment")
            by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
            if set(by_id) != set(pending):
                raise RuntimeError(f"official loader changed frozen membership: {sorted(set(pending)-set(by_id))[:20]}")
            group_audited = any(a["split"] == group and "loss_equivalence" in a for a in audits)
            for iid in pending:
                raw_label = dataset.labels[by_id[iid]]
                pre_ids = []
                for cls, polygon in zip(raw_label["cls"].flatten(), raw_label["segments"]):
                    key = np.r_[cls, polygon.flatten()].astype(np.float32)
                    matches = [aid for arr, aid in identity[iid] if np.array_equal(key, arr)]
                    if len(matches) != 1:
                        raise RuntimeError(f"ambiguous serialized-label identity: image {iid}, matches={matches}")
                    pre_ids.append(matches[0])
                order.clear()
                sample = dataset[by_id[iid]]
                ann_ids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
                if len(ann_ids) != len(sample["cls"]):
                    raise RuntimeError(f"overlap-label ordering mismatch: image {iid}")
                batch = YOLODataset.collate_fn([sample])
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                tensor = batch["img"].float() / 255
                assert tensor.shape == (1, 3, 640, 640)
                assert batch["masks"].shape[-2:] == (640, 640)
                captured.clear()
                with torch.no_grad():
                    _, raw = model(tensor)
                    pred = raw["one2one"]
                    assigned, _, _ = criterion.get_assigned_targets_and_loss(pred, batch)
                    fg, owner, target_boxes = (x[0] for x in assigned[:3])
                    ids = torch.where(fg)[0]
                    owners = owner[ids]
                    if len(set(owners.tolist())) != len(owners):
                        raise RuntimeError("official one-to-one assigned repeated GT owner")
                    raw_coeff = pred["mask_coefficient"][0].T
                    captured_h = [captured[f"h{l}"] for l in range(3)]
                    feats = captured["F"]
                    replay_h = [head.one2one_cv4[l][:-1](feats[l]) for l in range(3)]
                    replay_c = torch.cat([head.one2one_cv4[l][-1](replay_h[l])[0].flatten(1).T for l in range(3)])
                    h_error = max(float((h - r).abs().max()) for h, r in zip(captured_h, replay_h))
                    c_error = float((raw_coeff - replay_c).abs().max())
                    torch.testing.assert_close(replay_c, raw_coeff, atol=1e-5, rtol=1e-5)
                    for expected_h, actual_h in zip(captured_h, replay_h):
                        torch.testing.assert_close(expected_h, actual_h, atol=1e-5, rtol=1e-5)
                    all_levels = torch.cat([torch.full((h.shape[-2]*h.shape[-1],), l, device=ids.device, dtype=torch.long)
                                            for l, h in enumerate(captured_h)])
                    boxes = head._get_decode_boxes(pred)[0].T[ids].detach()
                    score, cls = pred["scores"][0, :, ids].sigmoid().max(dim=0)
                    boxes_original = ops.scale_boxes((640, 640), boxes.clone(), sample["ori_shape"], ratio_pad=sample["ratio_pad"])
                rows = []
                for k, (rid, gtidx) in enumerate(zip(ids.tolist(), owners.tolist())):
                    annotation = source_meta[str(iid)]["annotations"][str(ann_ids[gtidx])]
                    gtbox = boxes.new_tensor(annotation["bbox"])
                    gtbox[2:] += gtbox[:2]
                    level = int(all_levels[rid])
                    rows.append({"split": group, "image_id": iid, "annotation_id": int(annotation["id"]),
                        "branch": "one2one", "raw_id": int(rid), "pyramid_level": level, "level": level,
                        "target_gt_idx": int(gtidx), "gt_index": int(gtidx), "candidate_index": k,
                        "area": float(annotation["area"]), "box_iou": float(box_iou(gtbox[None], boxes_original[k:k+1])[0, 0]),
                        "predicted_class_id": int(cls[k]), "predicted_score": float(score[k])})
                audit = {"image_id": iid, "split": group, "n": len(rows), "h_replay_max_abs": h_error,
                         "c_replay_max_abs": c_error, "F_shapes": [list(x.shape[1:]) for x in feats]}
                if rows and not group_audited:
                    audit["loss_equivalence"] = audit_loss(criterion, pred, batch, ids, owners, target_boxes[ids], float(model.args.box), iid)
                    group_audited = True
                source_image = Path(cfg["images"]) / f"{domain}2017" / source_meta[str(iid)]["file_name"]
                image = {"schema": SCHEMA, "fingerprint": fingerprint, "image_id": iid, "split": group,
                    "source_split": domain, "rows": rows, "raw_ids": ids.detach().cpu(),
                    "c0": raw_coeff[ids].detach().cpu().contiguous(), "boxes": boxes.cpu().contiguous(),
                    "levels": all_levels[ids].cpu(), "owners": owners.cpu(), "target_boxes": target_boxes[ids].detach().cpu(),
                    "predicted_classes": cls.cpu(), "predicted_scores": score.cpu(),
                    "original_shape": sample["ori_shape"], "ratio_pad": sample["ratio_pad"],
                    "all_annotation_ids": ann_ids, "segmentation_gain": float(model.args.box),
                    "source_image_sha256": sha256(source_image), "input_shape": [640, 640], "replay_audit": audit}
                if rows:
                    image.update(F=[x[0].detach().cpu().contiguous() for x in feats],
                        proto=pred["proto"][0].detach().cpu().contiguous(), masks=sample["masks"][0].cpu().contiguous())
                    if any(x.dtype != torch.float32 for x in image["F"] + [image["proto"], image["c0"]]):
                        raise RuntimeError("floating-point compression detected")
                else:
                    image["no_positive"] = True
                artifact = images_dir / f"{iid:012d}.pt"
                temporary = artifact.with_suffix(".pt.tmp")
                estimate = sum(x.numel()*x.element_size() for x in image.get("F", []))
                estimate += sum(v.numel()*v.element_size() for v in image.values() if isinstance(v, torch.Tensor)) + 2_000_000
                if used_bytes + estimate > MAX_CACHE_BYTES or shutil.disk_usage(root).free < estimate + 1_000_000_000:
                    snapshot("space_budget_exhausted", group)
                    raise RuntimeError("250 GB cache cap / free disk safeguard reached; old cache retained")
                torch.save(image, temporary)
                size = temporary.stat().st_size
                if used_bytes + size > MAX_CACHE_BYTES:
                    raise RuntimeError("cache cap exceeded by serialized artifact; temporary retained, not committed")
                os.replace(temporary, artifact)
                meta = {"image_id": iid, "split": group, "n": len(rows), "fingerprint": fingerprint,
                        "artifact_bytes": size, "source_image_sha256": image["source_image_sha256"],
                        "F_shapes": audit["F_shapes"], "no_positive": not bool(rows)}
                write_json(artifact.with_suffix(".meta.json"), meta)
                metas[iid] = meta
                used_bytes += size
                audits.append(audit)
                processed += 1
                if processed % 25 == 0:
                    snapshot("building", group)
                del image, batch, raw, pred, feats, captured_h, replay_h, replay_c
            del dataset
            snapshot("split_complete", group)
        state = snapshot("requested_complete")
        write_json(out / "COMPLETE.json", {**state, "smoke": args.limit is not None,
            "reference_loss": "official mask_ratio=1 overlap_mask, proto bilinear to640 align_corners=False, GT xyxy crop, mean/normalized-area, per-candidate mean, gain=args.box",
            "loss_tolerance": {"atol": 3e-5, "rtol": 3e-5}, "replay_tolerance": {"atol": 1e-5, "rtol": 1e-5}})
        if all(len(state["completed"]) and state["completed"][g] == len(split[g]) for g in GROUPS):
            write_json(root / "COMPLETE.json", read_json(out / "COMPLETE.json"))
    except Exception as exc:
        snapshot("failed")
        write_json(out / "FAILURE.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        augment_module.polygons2masks_overlap = original_rasterizer
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
