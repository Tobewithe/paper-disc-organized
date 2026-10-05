"""Small server-generated assets and immutable batch-one YOLO feature replay.

No tensors/models are created at import. This independent screen has no
dependency on the old 13,000-image migration receipt or its large tensor cache.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

VERSION = "8.4.100"
WEIGHTS_SHA256 = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
_VERIFIED = set()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def resolve_runtime_config(config):
    cfg = dict(config)
    root = Path(cfg.get("server_root", Path(__file__).resolve().parents[1]))
    source = Path(cfg.get("source_root", "/root/autodl-tmp/prototype_guided_evidence_selection_20261003"))
    cfg.setdefault("server_root", str(root))
    cfg.setdefault("source_root", str(source))
    cfg.setdefault("cache", cfg.get("assets", str(root / "cache")))
    cfg.setdefault("assets", cfg["cache"])
    if Path(cfg["assets"]) != Path(cfg["cache"]):
        raise ValueError("assets and cache must identify the same independent screen cache")
    cfg.setdefault("weights", str(source / "models/yolo26m-seg.pt"))
    cfg.setdefault("source_python", str(source / "py"))
    cfg.setdefault("coco_root", "/autodl-pub/data/COCO2017")
    cfg.setdefault("parent_split", str(source / "SPLIT.json"))
    cfg.setdefault("split", str(root / "SPLIT.json"))
    cfg.setdefault("images", str(root / "data/images"))
    cfg.setdefault("annotations_train", str(source / "data/instances_train2017.json"))
    cfg.setdefault("annotations_val", str(source / "data/instances_val2017.json"))
    cfg.setdefault("max_cache_gib", 8)
    cfg.setdefault("minimum_root_free_gib", 10)
    for key in ("replay_atol", "replay_rtol"):
        cfg.setdefault(key, 3e-5)
        if cfg[key] != 3e-5:
            raise ValueError(f"Fixed {key}=3e-5")
    return cfg


resolve_config = resolve_runtime_config


def load_index(config):
    cfg = resolve_runtime_config(config)
    return load_json(Path(cfg["cache"]) / "INDEX.json")


def load_asset(config, image_id, verify=True):
    cfg = resolve_runtime_config(config)
    root = Path(cfg["cache"])
    iid = int(image_id)
    path = root / "images" / f"{iid:012d}.pt.gz"
    meta = load_json(path.with_name(f"{iid:012d}.meta.json"))
    ident = load_json(root / "CACHE_IDENTITY.json")
    stat = path.stat()
    token = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, meta["compressed_sha256"])
    if stat.st_size != meta["compressed_bytes"]:
        raise AssertionError(f"{iid}: incomplete local cache")
    if verify and token not in _VERIFIED:
        if sha256(path) != meta["compressed_sha256"]:
            raise AssertionError(f"{iid}: cache archive changed")
        _VERIFIED.add(token)
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    x = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    n = len(x["raw_ids"])
    if (int(x["image_id"]) != iid or x["fingerprint"] != ident["fingerprint"]
            or meta["fingerprint"] != ident["fingerprint"] or n != meta["n"]
            or len(x["rows"]) != n or "F" in x):
        raise AssertionError(f"{iid}: local fixed population/cache identity changed")
    if x["input_uint8"].dtype != torch.uint8 or tuple(x["input_uint8"].shape) != (3, 640, 640):
        raise AssertionError("Exact original uint8 input is required")
    if tensor_sha(x["input_uint8"]) != meta["input_uint8_sha256"]:
        raise AssertionError(f"{iid}: original input bytes changed")
    if tuple(x["proto"].shape) != (32, 160, 160) or x["proto"].dtype != torch.float32:
        raise AssertionError("Full FP32 original prototype is required")
    if tuple(x["masks"].shape) != (640, 640) or tensor_sha(x["masks"]) != meta["masks_sha256"]:
        raise AssertionError("Full original overlap mask changed")
    op = x["operator"]
    if not torch.equal(op["raw_ids"], x["raw_ids"]) or op["source_fingerprint"] != x["fingerprint"]:
        raise AssertionError("Frozen operator/candidate identity changed")
    for k, row in enumerate(x["rows"]):
        owner = int(x["owners"][k])
        if (row["branch"] != "one2one" or row["raw_id"] != int(x["raw_ids"][k])
                or row["target_gt_idx"] != owner
                or row["annotation_id"] != int(x["all_annotation_ids"][owner])
                or row["pyramid_level"] != int(x["levels"][k])):
            raise AssertionError(f"{iid}: permanent candidate identity differs")
    x["_operator"] = op
    x["_asset_integrity"] = dict(compressed_sha256=meta["compressed_sha256"],
        input_uint8_sha256=meta["input_uint8_sha256"], full_mask_tensor_sha256=meta["masks_sha256"],
        original_server_forward=True, old_cache_reused=False)
    return x


class _CapturedHeadInput(Exception):
    pass


class FrozenReplay:
    """Immutable original weights; feature replay stops before detection head."""

    def __init__(self, config, device="cuda"):
        self.config = resolve_runtime_config(config)
        if os.name == "nt" or not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError("Model execution is permitted only on the Linux CUDA server")
        self.device = torch.device(device)
        source_python = Path(self.config["source_python"]).resolve()
        sys.path.insert(0, str(source_python)) if str(source_python) not in sys.path else None
        import ultralytics
        from ultralytics import YOLO
        if ultralytics.__version__ != VERSION or not Path(ultralytics.__file__).resolve().is_relative_to(source_python):
            raise RuntimeError("Require the original vendored Ultralytics 8.4.100")
        torch.set_num_threads(int(self.config.get("cpu_threads", 6)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed = int(self.config.get("seed", 0))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.weights_sha256 = sha256(self.config["weights"])
        if self.weights_sha256 != WEIGHTS_SHA256:
            raise AssertionError("Original official checkpoint differs")
        wrapper = YOLO(self.config["weights"])
        self.train_args = copy.deepcopy(wrapper.ckpt["train_args"])
        self.source = wrapper.model.to(self.device).float().eval().requires_grad_(False)
        self.head = self.source.model[-1]
        self.native_cv4 = self.head.one2one_cv4
        self.feature_channels = [int(b[0].conv.in_channels) for b in self.native_cv4]
        self._buffers = {k: v.detach().cpu().clone() for k, v in self.source.named_buffers()}
        self.import_info = dict(ultralytics=ultralytics.__version__, ultralytics_file=str(ultralytics.__file__),
            torch=torch.__version__, cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(self.device),
            weights_sha256=self.weights_sha256, dtype="float32", tf32=False, replay_batch_size=1,
            detector_head_forward=False, TAL_recomputed_during_training=False)

    def read(self, image_id, verify=True):
        return load_asset(self.config, image_id, verify)

    def assert_unchanged(self):
        if self.source.training or any(p.requires_grad or p.grad is not None for p in self.source.parameters()):
            raise AssertionError("Original model no longer immutable/eval")
        for module in self.source.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
                raise AssertionError("Original BatchNorm entered train mode")
        for name, value in self.source.named_buffers():
            torch.testing.assert_close(value.detach().cpu(), self._buffers[name], atol=0, rtol=0)

    @torch.no_grad()
    def replay(self, images):
        output = []
        for image in images:
            captured = []
            def capture(module, args):
                captured.extend(f.detach() for f in args[0])
                raise _CapturedHeadInput()
            hook = self.head.register_forward_pre_hook(capture)
            try:
                self.source(image["input_uint8"].to(self.device, dtype=torch.float32)[None] / 255)
                raise AssertionError("Head capture did not stop model forward")
            except _CapturedHeadInput:
                pass
            finally:
                hook.remove()
            if len(captured) != 3:
                raise AssertionError("Expected three complete feature maps")
            shapes = image["source_F_shapes"]
            for feature, shape in zip(captured, shapes):
                if list(feature.shape) != [1, *shape] or feature.dtype != torch.float32 or not torch.isfinite(feature).all():
                    raise AssertionError("Frozen feature shape/precision/finite check failed")
            output.append(captured)
        return [torch.cat([f[l] for f in output]).detach() for l in range(3)]

    features = replay

    @torch.no_grad()
    def native_outputs(self, features, images):
        hs, cs = [], []
        for branch, feature in zip(self.native_cv4, features):
            h = branch[:-1](feature)
            hs.append(h.flatten(2)); cs.append(branch[-1](h).flatten(2))
        h, c = torch.cat(hs, 2).transpose(1, 2), torch.cat(cs, 2).transpose(1, 2)
        return [dict(h=h[j, x["raw_ids"].to(self.device)], c=c[j, x["raw_ids"].to(self.device)])
                for j, x in enumerate(images)]

    def native_copy(self):
        return copy.deepcopy(self.native_cv4)
