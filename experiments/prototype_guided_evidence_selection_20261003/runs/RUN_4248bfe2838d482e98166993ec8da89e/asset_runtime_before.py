"""Lossless asset loading and frozen original-prefix replay on the GPU server.

The original cached prototype/labels/assignment always remain authoritative.
Only F is regenerated from the exact migrated uint8 input. No assignment,
confidence filtering, detector postprocessing, optimizer or model update occurs.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

TOLERANCE = 3e-5
VERSION = "8.4.100"
_VERIFIED_ARCHIVES = {}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def resolve_runtime_config(config):
    cfg = dict(config)
    root = Path(cfg.get("server_root", Path(__file__).resolve().parents[1]))
    cfg.setdefault("assets", str(root / "assets"))
    cfg.setdefault("weights", str(root / "models" / "yolo26m-seg.pt"))
    cfg.setdefault("source_python", str(root / "py"))
    for name in ("replay_atol", "replay_rtol"):
        cfg.setdefault(name, TOLERANCE)
        if cfg[name] != TOLERANCE:
            raise ValueError(f"Predeclared {name} must remain {TOLERANCE}")
    return cfg


def load_index(config):
    cfg = resolve_runtime_config(config)
    return load_json(Path(cfg["assets"]) / "INDEX.json")


def load_asset(config, image_id, verify=True):
    """Read one original CPU payload; no tensor quantization or recomputation.

    Compressed-file hashes are verified once per unchanged file per process.
    Input and optional individual original-tensor hashes are checked on loading.
    ``_operator`` aliases the retained source operator, whose obsolete K was
    intentionally omitted during migration. New arm-specific K is made elsewhere.
    """
    cfg = resolve_runtime_config(config)
    base = Path(cfg["assets"])
    image_id = int(image_id)
    path = base / "images" / f"{image_id:012d}.pt.gz"
    metadata_path = path.with_name(f"{image_id:012d}.meta.json")
    metadata = load_json(metadata_path)
    migration = load_json(base / "MIGRATION_IDENTITY.json")
    stat = path.stat()
    if stat.st_size != int(metadata["compressed_bytes"]):
        raise AssertionError(f"{image_id}: compressed size changed")
    token = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, metadata["compressed_sha256"])
    if verify and token not in _VERIFIED_ARCHIVES:
        if sha256(path) != metadata["compressed_sha256"]:
            raise AssertionError(f"{image_id}: compressed archive hash mismatch")
        _VERIFIED_ARCHIVES[token] = True
    with gzip.open(path, "rb") as stream:
        image = torch.load(stream, map_location="cpu", weights_only=False)
    provenance = image["transfer_provenance"]
    if (int(image["image_id"]) != image_id or int(metadata["image_id"]) != image_id
            or image["split"] != metadata["split"]
            or provenance["migration_fingerprint"] != migration["fingerprint"]
            or provenance["migration_fingerprint"] != metadata["migration_fingerprint"]
            or provenance["checkpoint_sha256"] != migration["inputs"]["weights"]
            or image["fingerprint"] != migration["source_fingerprint"]):
        raise AssertionError(f"{image_id}: source/migration identity mismatch")
    if "F" in image:
        raise AssertionError("Migration should omit F; F_reference is audit-only")
    inp = image["input_uint8"]
    if inp.dtype != torch.uint8 or tuple(inp.shape) != (3, 640, 640):
        raise AssertionError(f"{image_id}: expected exact [3,640,640] uint8 RGB input")
    if tensor_sha(inp) != provenance["input_uint8_sha256"]:
        raise AssertionError(f"{image_id}: exact input hash mismatch")
    n = len(image["raw_ids"])
    if n != int(metadata["n"]) or n != len(image["rows"]):
        raise AssertionError(f"{image_id}: frozen candidate count mismatch")
    hashes = provenance.get("retained_tensor_sha256", provenance.get("tensor_sha256", {}))
    verified_tensors = []
    for key, expected in hashes.items():
        if key not in image or not torch.is_tensor(image[key]) or tensor_sha(image[key]) != expected:
            raise AssertionError(f"{image_id}: retained tensor {key} hash mismatch")
        verified_tensors.append(key)
    if n:
        required = {"proto": (32, 160, 160), "c0": (n, 32), "boxes": (n, 4),
                    "masks": (640, 640), "target_boxes": (n, 4), "owners": (n,)}
        for key, shape in required.items():
            if tuple(image[key].shape) != shape:
                raise AssertionError(f"{image_id}: {key} shape differs: {image[key].shape}")
        if image["proto"].dtype != torch.float32 or image["c0"].dtype != torch.float32:
            raise AssertionError("Original P/c0 must remain FP32")
        if provenance["audit"].get("overlap_mask_exact") is not True:
            raise AssertionError("Source full overlap mask did not pass exact migration check")
        operator = image["transfer_operator"]
        if (int(operator["image_id"]) != image_id
                or operator["source_fingerprint"] != image["fingerprint"]
                or not torch.equal(operator["raw_ids"], image["raw_ids"])):
            raise AssertionError(f"{image_id}: frozen operator identity differs")
        for key, shape in {"h0": (n, 64), "A": (n, 256, 32), "G": (n, 32, 32), "valid": (n,)}.items():
            if tuple(operator[key].shape) != shape:
                raise AssertionError(f"{image_id}: source operator {key} shape differs")
        for i, row in enumerate(image["rows"]):
            owner = int(image["owners"][i])
            if (row["branch"] != "one2one" or int(row["image_id"]) != image_id
                    or int(row["raw_id"]) != int(image["raw_ids"][i])
                    or int(row["pyramid_level"]) != int(image["levels"][i])
                    or int(row["target_gt_idx"]) != owner
                    or int(row["annotation_id"]) != int(image["all_annotation_ids"][owner])):
                raise AssertionError(f"{image_id}: permanent candidate identity changed")
        image["_operator"] = operator
    elif not image.get("no_positive", False):
        raise AssertionError("No-positive asset must retain explicit state")
    image["_asset_integrity"] = dict(
        compressed_sha256=metadata["compressed_sha256"], compressed_hash_verified=bool(verify),
        input_uint8_sha256=provenance["input_uint8_sha256"],
        full_mask_tensor_sha256=tensor_sha(image["masks"]) if n else None,
        individual_source_mask_hash_verified="masks" in verified_tensors,
        verified_retained_tensor_hashes=verified_tensors,
        source_full_mask_exact=provenance["audit"].get("overlap_mask_exact"),
        source_prototype_authoritative=True)
    return image


class _CapturedHeadInput(Exception):
    pass


class FrozenReplay:
    """Original immutable YOLO prefix, stopped before the detection head.

    replay(images) returns three detached BCHW feature tensors. Each source
    image uses its own batch-1 forward, matching original cache construction.
    They are normal no-grad tensors, usable by a separately trainable cv4.
    """

    def __init__(self, config, device="cuda"):
        self.config = resolve_runtime_config(config)
        if os.name == "nt":
            raise RuntimeError("This replay is authorized on the Linux GPU server, not the desktop")
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError("An authorized CUDA server is required for model replay")
        self.device = torch.device(device)
        source_python = str(Path(self.config["source_python"]).resolve())
        if source_python not in sys.path:
            sys.path.insert(0, source_python)
        import ultralytics
        from ultralytics import YOLO
        if ultralytics.__version__ != VERSION:
            raise RuntimeError(f"Imported Ultralytics {ultralytics.__version__}, expected {VERSION}")
        if not Path(ultralytics.__file__).resolve().is_relative_to(Path(source_python)):
            raise RuntimeError("Ultralytics import is not the vendored frozen source")
        torch.set_num_threads(int(self.config.get("cpu_threads", 6)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(int(self.config.get("seed", 0)))
        np.random.seed(int(self.config.get("seed", 0)))
        torch.manual_seed(int(self.config.get("seed", 0)))
        migration = load_json(Path(self.config["assets"]) / "MIGRATION_IDENTITY.json")
        self.weights_sha256 = sha256(self.config["weights"])
        if self.weights_sha256 != migration["inputs"]["weights"]:
            raise AssertionError("Original checkpoint hash differs from migrated source")
        self.source = YOLO(self.config["weights"]).model.to(self.device).float().eval().requires_grad_(False)
        self.head = self.source.model[-1]
        if not hasattr(self.head, "one2one_cv4"):
            raise AssertionError("Original one-to-one cv4 is missing")
        self.native_cv4 = self.head.one2one_cv4
        self.feature_channels = [int(branch[0].conv.in_channels) for branch in self.native_cv4]
        if len(self.feature_channels) != 3:
            raise AssertionError("Expected three original coefficient scales")
        self._buffers = {key: value.detach().cpu().clone() for key, value in self.source.named_buffers()}
        self.import_info = dict(ultralytics=ultralytics.__version__, ultralytics_file=str(ultralytics.__file__),
                                torch=torch.__version__, cuda=torch.version.cuda,
                                cudnn=torch.backends.cudnn.version(), device=str(self.device),
                                device_name=torch.cuda.get_device_name(self.device),
                                weights_sha256=self.weights_sha256, dtype="float32", tf32=False,
                                replay_batch_size=1, detector_head_forward=False, TAL_recomputed=False)

    def read(self, image_id, verify=True):
        return load_asset(self.config, image_id, verify=verify)

    def assert_unchanged(self):
        if self.source.training or any(p.requires_grad for p in self.source.parameters()):
            raise AssertionError("Frozen original model state changed")
        for module in self.source.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
                raise AssertionError("Frozen source BatchNorm entered train mode")
        for key, value in self.source.named_buffers():
            torch.testing.assert_close(value.detach().cpu(), self._buffers[key], atol=0, rtol=0)

    @torch.no_grad()
    def replay(self, images):
        if not images:
            raise ValueError("At least one image is required")
        if self.source.training or any(p.requires_grad for p in self.source.parameters()):
            raise AssertionError("Original prefix is no longer frozen/eval")
        outputs = []
        for image in images:
            captured = []

            def capture(module, args):
                features = args[0]
                if len(features) != 3:
                    raise AssertionError("Expected full P3/P4/P5 input to original detection head")
                captured.extend(feature.detach() for feature in features)
                raise _CapturedHeadInput()

            hook = self.head.register_forward_pre_hook(capture)
            try:
                exact = image["input_uint8"].to(device=self.device, dtype=torch.float32)[None] / 255.0
                self.source(exact)
                raise AssertionError("Detection head capture did not stop the forward")
            except _CapturedHeadInput:
                pass
            finally:
                hook.remove()
            expected_shapes = image["transfer_provenance"]["source_F_shapes"]
            if not expected_shapes:
                expected_shapes = [[c, 80 // (2**l), 80 // (2**l)] for l, c in enumerate(self.feature_channels)]
            for level, feature in enumerate(captured):
                if list(feature.shape) != [1, *expected_shapes[level]] or feature.dtype != torch.float32:
                    raise AssertionError("Replayed feature shape/precision differs from source")
                if not bool(torch.isfinite(feature).all()):
                    raise FloatingPointError("Nonfinite frozen feature replay")
            outputs.append(captured)
        return [torch.cat([row[level] for row in outputs], dim=0).detach() for level in range(3)]

    features = replay

    @torch.no_grad()
    def native_outputs(self, features, images):
        h_maps, c_maps = [], []
        for branch, feature in zip(self.native_cv4, features):
            h = feature
            for layer in list(branch.children())[:-1]:
                h = layer(h)
            h_maps.append(h.flatten(2))
            c_maps.append(branch[-1](h).flatten(2))
        h_all = torch.cat(h_maps, dim=2).transpose(1, 2)
        c_all = torch.cat(c_maps, dim=2).transpose(1, 2)
        result = []
        for j, image in enumerate(images):
            raw = image["raw_ids"].to(device=self.device, dtype=torch.long)
            if bool(((raw < 0) | (raw >= h_all.shape[1])).any()):
                raise AssertionError("Frozen raw ID outside original feature grids")
            result.append(dict(h=h_all[j, raw], c=c_all[j, raw]))
        return result

    def native_copy(self):
        """Independent branch for training; never modify the replay source."""
        return copy.deepcopy(self.native_cv4)
