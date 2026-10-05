"""Lossless CPU migration of the frozen official cache, omitting only neck F.

Run on the authorized laptop, never forward a model here. The receiver must
replay F from input_uint8 using the original frozen checkpoint and pass the
predeclared F_reference audit before training. P, masks, candidates, boxes,
classes and identities remain the original tensors, not newly inferred ones.

Config: source_cache (or cache), source_operators (or source_operator_cache /
operator_cache), source_weights (or weights), split. Assets default to
<config directory>/assets. Per-image files
are gzip-compressed torch archives; no tensor is quantized. --limit-per-split
is a migration smoke only and never writes a full-completion receipt.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

# This program has no model forward or CUDA path. Set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
import ultralytics.data.augment as augment_module
from ultralytics.data.converter import coco91_to_coco80_class
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import ops


SCHEMA = "prototype-evidence-lossless-cpu-transfer-v1"
GROUPS = ("fit", "dev", "val")
CAP_BYTES = 47 * 1024**3
RESERVE_BYTES = 1024**3
WITNESSES_PER_SPLIT = 8


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_equal(actual, expected, name):
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise AssertionError(f"{name}: shape/value mismatch")
    elif jsonable(actual) != jsonable(expected):
        raise AssertionError(f"{name}: metadata mismatch: {actual!r} vs {expected!r}")


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class TransferDataset(YOLODataset):
    """Official reader/transforms, with new label caches isolated from sources."""

    def __init__(self, *args, transfer_label_cache, **kwargs):
        self.transfer_label_cache = Path(transfer_label_cache)
        self.transfer_label_cache.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        return super().cache_labels(self.transfer_label_cache)


def serialized_annotation_order(dataset, position, image_id, converted, conversion, source_meta):
    domain = source_meta[str(image_id)]["source_split"]
    filename = Path(source_meta[str(image_id)]["file_name"]).with_suffix(".txt")
    label_path = converted / "labels" / f"{domain}2017" / filename
    lines = label_path.read_text(encoding="utf-8").strip().splitlines() if label_path.exists() else []
    annotation_ids = conversion["source_annotation_ids"][str(image_id)]
    if len(lines) != len(annotation_ids):
        raise AssertionError(f"{image_id}: conversion label/annotation length mismatch")
    serialized = [(np.asarray(line.split(), dtype=np.float32), int(aid))
                  for line, aid in zip(lines, annotation_ids)]
    raw = dataset.labels[position]
    if len(raw["cls"]) != len(raw["segments"]):
        raise AssertionError(f"{image_id}: official class/segment length mismatch")
    result = []
    for cls, polygon in zip(raw["cls"].flatten(), raw["segments"]):
        key = np.r_[cls, polygon.flatten()].astype(np.float32)
        matched = [aid for row, aid in serialized if np.array_equal(key, row)]
        if len(matched) != 1:
            raise AssertionError(f"{image_id}: ambiguous exact polygon identity {matched}")
        result.append(matched[0])
    return result, label_path


def validate_sample(old, sample, ann_ids, source_meta):
    iid = int(old["image_id"])
    assert_equal(ann_ids, old["all_annotation_ids"], f"{iid} official sorted annotations")
    assert_equal(sample["ori_shape"], old["original_shape"], f"{iid} original shape")
    assert_equal(sample["ratio_pad"], old["ratio_pad"], f"{iid} ratio_pad")
    image = sample["img"].detach().cpu().contiguous()
    if image.dtype != torch.uint8 or tuple(image.shape) != (3, 640, 640):
        raise AssertionError(f"{iid}: official RGB CHW uint8 image expected, got {image.dtype}/{image.shape}")
    assert_equal(list(image.shape[-2:]), old["input_shape"], f"{iid} input shape")
    if tuple(sample["masks"].shape[-2:]) != (640, 640):
        raise AssertionError(f"{iid}: official mask resolution changed")
    n = len(old["raw_ids"])
    if n != len(old["rows"]) or n != len(old["owners"]):
        raise AssertionError(f"{iid}: frozen candidate lengths differ")
    if len(sample["cls"]) != len(ann_ids) or len(sample["bboxes"]) != len(ann_ids):
        raise AssertionError(f"{iid}: official GT arrays differ from annotation order")
    mapping = coco91_to_coco80_class()
    for owner, aid in enumerate(ann_ids):
        ann = source_meta[str(iid)]["annotations"][str(aid)]
        if int(sample["cls"][owner].item()) != mapping[int(ann["category_id"]) - 1]:
            raise AssertionError(f"{iid}: class does not match annotation {aid}")
    if n:
        assert_equal(sample["masks"][0], old["masks"], f"{iid} full overlap mask")
        owners = old["owners"].long()
        if bool(((owners < 0) | (owners >= len(ann_ids))).any()):
            raise AssertionError(f"{iid}: owner outside official GT list")
        expected_boxes = ops.xywh2xyxy(sample["bboxes"].float() * 640)[owners]
        torch.testing.assert_close(expected_boxes, old["target_boxes"].float(), atol=1e-4, rtol=1e-6)
        for k, row in enumerate(old["rows"]):
            owner = int(owners[k])
            if (int(row["annotation_id"]) != ann_ids[owner]
                    or int(row["target_gt_idx"]) != owner
                    or int(row["raw_id"]) != int(old["raw_ids"][k])
                    or int(row["pyramid_level"]) != int(old["levels"][k])
                    or row["branch"] != "one2one"):
                raise AssertionError(f"{iid}: candidate {k} identity mismatch")
        if old["proto"].dtype != torch.float32 or any(v.dtype != torch.float32 for v in old["F"]):
            raise AssertionError(f"{iid}: original P/F cache is not FP32")
    elif not old.get("no_positive", False):
        raise AssertionError(f"{iid}: empty candidates missing explicit no_positive state")
    return image, dict(annotation_order_exact=True, classes_exact=True,
                      target_boxes_checked=n, target_box_tolerance=dict(atol=1e-4, rtol=1e-6),
                      overlap_mask_exact=True if n else None,
                      overlap_mask_note=None if n else "original no-positive cache contains no mask tensor",
                      geometry_exact=True, input_uint8_sha256=tensor_sha(image))


def retained_operator(path, old):
    operator = torch_load(path)
    assert_equal(operator["image_id"], old["image_id"], "operator image")
    assert_equal(operator["raw_ids"], old["raw_ids"], "operator raw order")
    assert_equal(operator["source_fingerprint"], old["fingerprint"], "operator source")
    required = ("h0", "A", "G", "valid")
    if any(key not in operator for key in required):
        raise KeyError(f"required original operator fields missing: {path}")
    # Explicit omission prevents accidentally sending the obsolete 256-point K.
    return {key: value for key, value in operator.items() if key != "K"}


def write_archive(path, payload, used_bytes, assets):
    temp = path.with_name(path.name + ".tmp")
    # Reserve enough for the uncompressed tensor payload plus serialization overhead.
    estimate = tensor_bytes(payload) + 4 * 1024**2
    if used_bytes + estimate > CAP_BYTES:
        raise RuntimeError("47 GiB conservative migration budget would be exceeded; do not shrink dataset")
    if shutil.disk_usage(assets).free < estimate + RESERVE_BYTES:
        raise RuntimeError("migration free-space reserve reached; old cache remains untouched")
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
            torch.save(payload, compressed)
        raw.flush()
        os.fsync(raw.fileno())
    size = temp.stat().st_size
    if used_bytes + size > CAP_BYTES:
        raise RuntimeError("47 GiB serialized migration budget exceeded; temporary retained")
    digest = sha(temp)
    os.replace(temp, path)
    return size, digest


def write_manifest(path, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main(args):
    cfg = load_json(args.config)
    source = Path(cfg.get("source_cache", cfg.get("cache", ""))).resolve()
    operators = Path(cfg.get("source_operators", cfg.get("source_operator_cache", cfg.get("operator_cache", "")))).resolve()
    weights = cfg.get("source_weights", cfg.get("weights"))
    if not weights:
        raise KeyError("source_weights or weights is required")
    if not (source / "DATASET_READY.json").is_file() or not (operators / "images").is_dir():
        raise FileNotFoundError("existing official source cache and original operator cache are required")
    out = Path(args.out).resolve()
    assets = Path(args.assets or cfg.get("transfer_assets", Path(args.config).resolve().parent / "assets")).resolve()
    for target in (out, assets):
        if target == source or source in target.parents or target == operators or operators in target.parents:
            raise ValueError("migration outputs must not be inside either old cache")
    out.mkdir(parents=True, exist_ok=True)
    images = assets / "images"
    images.mkdir(parents=True, exist_ok=True)
    if args.limit_per_split is not None and args.limit_per_split < 1:
        raise ValueError("--limit-per-split must be positive")
    if ultralytics.__version__ != "8.4.100":
        raise RuntimeError(f"expected imported Ultralytics 8.4.100, got {ultralytics.__version__}")
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    split = {group: list(map(int, load_json(cfg["split"])[group])) for group in GROUPS}
    all_ids = [iid for group in GROUPS for iid in split[group]]
    if len(all_ids) != len(set(all_ids)) or [len(split[g]) for g in GROUPS] != [10000, 1000, 2000]:
        raise AssertionError("frozen 10000/1000/2000 unique image split changed")
    index = load_json(source / "INDEX.json")
    counts = {}
    for group in GROUPS:
        entries = index[group]
        if [int(row["image_id"]) for row in entries] != split[group]:
            raise AssertionError(f"{group}: source INDEX does not exactly match frozen split/order")
        counts.update({int(row["image_id"]): int(row["n"]) for row in entries})
    witnesses = {g: [iid for iid in split[g] if counts[iid] > 0][:WITNESSES_PER_SPLIT] for g in GROUPS}
    witness_ids = {iid for group in GROUPS for iid in witnesses[group]}
    ready = load_json(source / "DATASET_READY.json")
    converted = (source / ready["directory"]).resolve()
    conversion = load_json(source / "CONVERSION.json")
    source_meta = load_json(source / "SOURCE_IMAGES.json")
    identity = load_json(source / "CACHE_IDENTITY.json")
    if identity["fingerprint"] != ready["fingerprint"]:
        raise AssertionError("source DATASET_READY/CACHE_IDENTITY mismatch")
    inputs = {"source_identity": sha(source / "CACHE_IDENTITY.json"),
              "source_index": sha(source / "INDEX.json"), "source_conversion": sha(source / "CONVERSION.json"),
              "source_images": sha(source / "SOURCE_IMAGES.json"), "dataset_ready": sha(source / "DATASET_READY.json"),
              "weights": sha(weights), "split": sha(cfg["split"]), "script": sha(__file__)}
    if inputs["weights"] != identity["inputs"]["weights"]["sha256"]:
        raise AssertionError("original checkpoint differs from source cache")
    migration = dict(schema=SCHEMA, inputs=inputs, source_fingerprint=identity["fingerprint"],
                     split=split, witnesses=witnesses, compression="gzip level1 lossless torch.save",
                     max_asset_bytes=CAP_BYTES, omitted_fields=["F"], old_operator_omitted_fields=["K"],
                     replay_F_required=True, replay_F_tolerance=dict(atol=3e-5, rtol=3e-5),
                     source_unchanged=True)
    migration_fingerprint = signature(migration)
    identity_file = assets / "MIGRATION_IDENTITY.json"
    if identity_file.exists() and load_json(identity_file)["fingerprint"] != migration_fingerprint:
        raise RuntimeError("migration fingerprint differs; do not mix archives from changed code or inputs")
    dump(identity_file, dict(fingerprint=migration_fingerprint, **migration))
    dump(out / "MIGRATION_IDENTITY.json", dict(fingerprint=migration_fingerprint, **migration))
    dump(out / "ENVIRONMENT.json", dict(python=sys.version, torch=torch.__version__,
         ultralytics=ultralytics.__version__, ultralytics_file=ultralytics.__file__,
         device="CPU only; no model forward", limit_per_split=args.limit_per_split, inputs=inputs))
    # Constructing YOLO only deserializes the trusted original checkpoint on CPU.
    wrapper = YOLO(str(weights))
    model_args = SimpleNamespace(**wrapper.ckpt["train_args"])
    names = deepcopy(wrapper.names)
    if model_args.mask_ratio != 1 or not model_args.overlap_mask:
        raise AssertionError("official pretrained label configuration changed")
    if any(parameter.device.type != "cpu" for parameter in wrapper.model.parameters()):
        raise AssertionError("CPU migration unexpectedly allocated model parameters elsewhere")
    del wrapper
    requested = {g: ids if args.limit_per_split is None else ids[:args.limit_per_split] for g, ids in split.items()}
    committed = {}
    used_bytes = sum(p.stat().st_size for p in images.glob("*.pt.gz"))
    started = time.monotonic()
    completed_here = reused_here = 0
    original_rasterizer = augment_module.polygons2masks_overlap
    order = {}

    def record_order(*a, **kw):
        masks, indices = original_rasterizer(*a, **kw)
        order["indices"] = indices.copy()
        return masks, indices

    def snapshot(stage, group=None):
        rows = [committed[iid] for g in GROUPS for iid in split[g] if iid in committed]
        full_index = {g: [dict(image_id=iid, n=counts[iid], split=g,
                             no_positive=counts[iid] == 0, witness=iid in witness_ids,
                             status="complete" if iid in committed else "pending",
                             path=f"images/{iid:012d}.pt.gz",
                             compressed_sha256=committed[iid]["compressed_sha256"] if iid in committed else None)
                          for iid in split[g]] for g in GROUPS}
        state = dict(stage=stage, split=group, elapsed_s=time.monotonic()-started,
                     planned={g: len(split[g]) for g in GROUPS},
                     requested={g: len(requested[g]) for g in GROUPS},
                     completed={g: sum(iid in committed for iid in split[g]) for g in GROUPS},
                     positive_candidates={g: sum(counts[iid] for iid in split[g] if iid in committed) for g in GROUPS},
                     no_positive_images={g: [iid for iid in split[g] if counts[iid] == 0] for g in GROUPS},
                     archive_bytes=used_bytes, new_this_invocation=completed_here, reused_this_invocation=reused_here,
                     replay_F_status="required_on_destination_not_run_here")
        for base in (assets, out):
            dump(base / "INDEX.json", full_index)
            dump(base / "PROGRESS.json", state)
            write_manifest(base / "manifest.jsonl", rows)
        if stage in ("resume_verified", "packing", "split_requested_complete"):
            # A caller may poll progress between batches, but must not mistake
            # this record for the destination's still-required GPU audit.
            dump(out / "AUDIT_SCOPE.json", dict(schema=SCHEMA, source_tensors="retained unchanged",
                validation="exact official overlap masks, annotation order, source image hash and geometry",
                no_forward=True, replay_F_status="pending_destination", witnesses=witnesses))
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return state

    try:
        # Validate every committed archive before reuse, including source hashes.
        # Interrupted archives without metadata are preserved, never silently overwritten.
        for group in GROUPS:
            for iid in split[group]:
                target = images / f"{iid:012d}.pt.gz"
                meta_path = target.with_name(f"{iid:012d}.meta.json")
                if not target.exists() and not meta_path.exists():
                    continue
                if target.exists() and not meta_path.exists():
                    # A crash between the archive rename and metadata rename is
                    # recoverable without rewriting the committed tensor data.
                    with gzip.open(target, "rb") as stream:
                        recovered = torch.load(stream, map_location="cpu", weights_only=False)
                    provenance = recovered.get("transfer_provenance", {})
                    if (provenance.get("migration_fingerprint") != migration_fingerprint
                            or int(recovered.get("image_id", -1)) != iid
                            or recovered.get("split") != group
                            or len(recovered.get("raw_ids", [])) != counts[iid]
                            or tensor_sha(recovered["input_uint8"]) != provenance.get("input_uint8_sha256")):
                        raise AssertionError(f"{iid}: uncommitted archive could not be authenticated")
                    recovered_meta = dict(**provenance, image_id=iid, split=group, n=counts[iid],
                        no_positive=counts[iid] == 0, path=f"images/{iid:012d}.pt.gz",
                        compressed_bytes=target.stat().st_size, compressed_sha256=sha(target))
                    dump(meta_path, recovered_meta)
                    del recovered
                if not target.exists() or not meta_path.exists():
                    raise RuntimeError(f"uncommitted migration artifact needs explicit recovery: {target}")
                meta = load_json(meta_path)
                if (meta["migration_fingerprint"] != migration_fingerprint
                        or int(meta["image_id"]) != iid or meta["split"] != group
                        or int(meta["n"]) != counts[iid] or bool(meta["F_reference"]) != (iid in witness_ids)):
                    raise AssertionError(f"{iid}: committed metadata identity mismatch")
                if target.stat().st_size != meta["compressed_bytes"] or sha(target) != meta["compressed_sha256"]:
                    raise AssertionError(f"{iid}: committed compressed archive hash/size mismatch")
                source_path = source / "images" / f"{iid:012d}.pt"
                if sha(source_path) != meta["source_cache_sha256"]:
                    raise AssertionError(f"{iid}: original cache changed after migration")
                if counts[iid] and sha(operators / "images" / f"{iid:012d}.pt") != meta["operator_sha256"]:
                    raise AssertionError(f"{iid}: original operator changed after migration")
                source_image = converted / "images" / f"{source_meta[str(iid)]['source_split']}2017" / source_meta[str(iid)]["file_name"]
                if sha(source_image) != meta["source_image_sha256"]:
                    raise AssertionError(f"{iid}: source image changed after migration")
                committed[iid] = meta
                reused_here += 1
        snapshot("resume_verified")
        augment_module.polygons2masks_overlap = record_order
        for group in GROUPS:
            pending = [iid for iid in requested[group] if iid not in committed]
            if not pending:
                continue
            image_paths = [converted / "images" / f"{source_meta[str(iid)]['source_split']}2017" /
                           source_meta[str(iid)]["file_name"] for iid in pending]
            if any(not path.is_file() for path in image_paths):
                raise FileNotFoundError(f"{group}: missing original images; no membership replacement allowed")
            pending_list = out / f"{group}_pending.txt"
            pending_list.write_text("\n".join(map(str, image_paths)) + "\n", encoding="utf-8")
            dataset = TransferDataset(img_path=str(pending_list), imgsz=640, batch_size=1,
                augment=False, hyp=deepcopy(model_args), rect=False, cache=False, stride=32,
                data={"names": names, "nc": 80, "channels": 3}, task="segment",
                transfer_label_cache=out / "official_label_cache" / f"{group}_{signature(pending)[:12]}.cache")
            by_id = {int(Path(path).stem): j for j, path in enumerate(dataset.im_files)}
            if len(by_id) != len(dataset.im_files) or set(by_id) != set(pending):
                raise AssertionError(f"{group}: official loader changed frozen image membership")
            for iid in pending:
                source_path = source / "images" / f"{iid:012d}.pt"
                original_meta = load_json(source_path.with_suffix(".meta.json"))
                if (original_meta["fingerprint"] != identity["fingerprint"]
                        or int(original_meta["n"]) != counts[iid]
                        or int(original_meta["artifact_bytes"]) != source_path.stat().st_size):
                    raise AssertionError(f"{iid}: original cache metadata/size mismatch")
                source_hash = sha(source_path)
                old = torch_load(source_path)
                if (int(old["image_id"]) != iid or old["split"] != group
                        or old["fingerprint"] != identity["fingerprint"] or len(old["raw_ids"]) != counts[iid]):
                    raise AssertionError(f"{iid}: original cache identity/index mismatch")
                pre_ids, label_path = serialized_annotation_order(dataset, by_id[iid], iid, converted, conversion, source_meta)
                order.clear()
                sample = dataset[by_id[iid]]
                if pre_ids and "indices" not in order:
                    raise AssertionError(f"{iid}: official overlap ordering hook was not called")
                ann_ids = np.asarray(pre_ids)[order["indices"]].tolist() if pre_ids else []
                input_uint8, audit = validate_sample(old, sample, ann_ids, source_meta)
                source_image = Path(sample["im_file"])
                image_hash = sha(source_image)
                if image_hash != old["source_image_sha256"]:
                    raise AssertionError(f"{iid}: original RGB image content changed")
                operator_path = operators / "images" / f"{iid:012d}.pt"
                operator_hash = sha(operator_path) if counts[iid] else None
                payload = {key: value for key, value in old.items() if key != "F"}
                payload["input_uint8"] = input_uint8
                payload["official_sample_cls"] = sample["cls"].detach().cpu().contiguous()
                payload["official_sample_bboxes"] = sample["bboxes"].detach().cpu().contiguous()
                payload["transfer_operator"] = retained_operator(operator_path, old) if counts[iid] else None
                if iid in witness_ids:
                    payload["F_reference"] = old["F"]
                provenance = dict(schema=SCHEMA, migration_fingerprint=migration_fingerprint,
                                  source_cache_sha256=source_hash, source_image_sha256=image_hash,
                                  source_label_sha256=sha(label_path) if label_path.exists() else None,
                                  operator_sha256=operator_hash, checkpoint_sha256=inputs["weights"],
                                  input_uint8_sha256=audit["input_uint8_sha256"],
                                  source_F_shapes=[list(v.shape) for v in old.get("F", [])],
                                  source_F_dtypes=[str(v.dtype) for v in old.get("F", [])],
                                  F_reference=iid in witness_ids, audit=audit)
                payload["transfer_provenance"] = provenance
                target = images / f"{iid:012d}.pt.gz"
                if target.exists():
                    raise FileExistsError(f"refusing to overwrite uncommitted archive {target}")
                size, digest = write_archive(target, payload, used_bytes, assets)
                meta = dict(**provenance, image_id=iid, split=group, n=counts[iid],
                            no_positive=counts[iid] == 0, path=f"images/{iid:012d}.pt.gz",
                            compressed_bytes=size, compressed_sha256=digest)
                dump(target.with_name(f"{iid:012d}.meta.json"), meta)
                committed[iid] = meta
                used_bytes += size
                completed_here += 1
                if completed_here % 25 == 0:
                    snapshot("packing", group)
                del old, sample, payload, input_uint8
            del dataset
            snapshot("split_requested_complete", group)
        state = snapshot("requested_complete")
        all_complete = len(committed) == len(all_ids) and args.limit_per_split is None
        receipt = dict(**state, status="complete" if all_complete else "partial", complete=all_complete,
                       requested_complete=True, limit_per_split=args.limit_per_split,
                       fingerprint=migration_fingerprint, manifest_sha256=sha(out / "manifest.jsonl"),
                       witness_images=witnesses, no_model_forward_performed=True,
                       destination_F_audit_required=True, destination_F_tolerance=dict(atol=3e-5, rtol=3e-5))
        dump(out / "COMPLETE.json", receipt)
        if all_complete:
            dump(assets / "COMPLETE.json", receipt)
        else:
            dump(assets / "PARTIAL.json", receipt)
    except Exception as exc:
        snapshot("failed")
        dump(out / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        augment_module.polygons2masks_overlap = original_rasterizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--limit-per-split", type=int)
    main(parser.parse_args())
