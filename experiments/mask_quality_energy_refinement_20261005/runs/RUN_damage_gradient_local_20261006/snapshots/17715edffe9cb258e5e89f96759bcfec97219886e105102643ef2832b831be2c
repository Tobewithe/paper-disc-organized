"""Explicitly authorized Windows CUDA DEV replay for small damage diagnostics.

This adapter changes initialization and execution-environment receipts only.
The official image construction, TAL assignment, annotation identity, capture,
iteration and model-integrity cleanup come from ``StreamingFinal`` unchanged.
It neither trains a model nor writes a tensor cache. Historical server runs and
their Linux-only execution guard are left untouched.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from online_runtime import (
    FrozenReplay, VERSION, WEIGHTS_SHA256, dump, resolve_runtime_config, sha256,
)
from prepare_qcr_cache import SCHEMA, ensure_space, prepare_images
from qcr_streaming_data import StreamingFinal

_ANNOTATION_SHA256 = {
    "train": "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d",
    "val": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
}
_REQUIRED_PATHS = (
    "server_root", "source_python", "weights", "coco_root", "images",
    "annotations_train", "annotations_val",
)


def _local_config(config):
    if config.get("authorized_local_diagnostic") is not True:
        raise RuntimeError("Require explicit authorized_local_diagnostic=True")
    if os.name != "nt" or not torch.cuda.is_available():
        raise RuntimeError("This narrow diagnostic adapter requires local Windows CUDA")
    missing = [key for key in _REQUIRED_PATHS if key not in config]
    if missing:
        raise ValueError(f"Local diagnostic paths must be explicit: {missing}")
    cfg = dict(config)
    for key in _REQUIRED_PATHS:
        path = Path(cfg[key])
        if not path.is_absolute():
            raise ValueError(f"Local diagnostic {key} must be absolute")
        cfg[key] = str(path.resolve())
    # Avoid introducing an unused historical remote path into local receipts.
    cfg.setdefault("source_root", cfg["source_python"])
    return resolve_runtime_config(cfg)


class LocalFrozenReplay(FrozenReplay):
    """Local initializer; retain the audited immutable-model assertion methods."""

    def __init__(self, config, device="cuda"):
        self.config = _local_config(config)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("Local diagnostics require the explicitly selected CUDA device")
        source_python = Path(self.config["source_python"]).resolve()
        if str(source_python) not in sys.path:
            sys.path.insert(0, str(source_python))
        import ultralytics
        from ultralytics import YOLO

        if (ultralytics.__version__ != VERSION
                or not Path(ultralytics.__file__).resolve().is_relative_to(source_python)):
            raise RuntimeError("Require the pinned vendored Ultralytics 8.4.100")
        torch.set_num_threads(int(self.config.get("cpu_threads", 6)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed = int(self.config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.weights_sha256 = sha256(self.config["weights"])
        if self.weights_sha256 != WEIGHTS_SHA256:
            raise AssertionError("Original official checkpoint differs")
        wrapper = YOLO(self.config["weights"])
        self.train_args = deepcopy(wrapper.ckpt["train_args"])
        self.source = wrapper.model.to(self.device).float().eval().requires_grad_(False)
        self.head = self.source.model[-1]
        self.native_cv4 = self.head.one2one_cv4
        self.feature_channels = [int(branch[0].conv.in_channels) for branch in self.native_cv4]
        self._buffers = {name: value.detach().cpu().clone()
                         for name, value in self.source.named_buffers()}
        self.import_info = dict(
            mode="authorized_local_damage_diagnostic_only",
            execution_environment="local_windows_cuda",
            authorized_local_diagnostic=True,
            platform=platform.platform(), python=sys.version, executable=sys.executable,
            ultralytics=ultralytics.__version__, ultralytics_file=str(ultralytics.__file__),
            torch=torch.__version__, cuda=torch.version.cuda,
            device=str(self.device), device_name=torch.cuda.get_device_name(self.device),
            device_total_memory_bytes=torch.cuda.get_device_properties(self.device).total_memory,
            weights_path=self.config["weights"], weights_sha256=self.weights_sha256,
            dtype="float32", tf32=False, replay_batch_size=1,
            immutable_buffer_count=len(self._buffers),
            detector_head_forward=False, TAL_recomputed_during_training=False,
            local_adapter_code_sha256=sha256(__file__),
        )
        self.assert_unchanged()


class LocalStreamingDEV(StreamingFinal):
    """Replay at most 64 fixed DEV image IDs with the unchanged official methods.

    Required config: ``authorized_local_diagnostic=True`` and absolute paths for
    ``server_root`` (the Run directory), ``images``, ``source_python`` (the pinned
    vendor package parent), ``weights``, ``coco_root`` (ZIP parent), and both
    ``annotations_train`` and ``annotations_val``. Optional ``cpu_threads`` and
    ``seed`` retain the same meaning as the original replay implementation.
    Call ``LocalStreamingDEV(cfg, image_ids, out, split='dev')`` and consume
    ``iter_images()`` in a context manager, as with ``StreamingFinal``.
    """

    def __init__(self, cfg, image_ids, out, split="dev"):
        if split != "dev":
            raise ValueError("This local adapter supports fixed DEV diagnostics only")
        self.cfg = _local_config(cfg)
        self.image_ids = [int(iid) for iid in image_ids]
        if (not self.image_ids or len(self.image_ids) > 64
                or len(self.image_ids) != len(set(self.image_ids))):
            raise ValueError("Require 1 to 64 duplicate-free, fixed DEV image IDs")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        Path(self.cfg["server_root"]).mkdir(parents=True, exist_ok=True)
        self.split = split
        self.domain = "train"
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
        print(json.dumps({
            "mode": "authorized_local_damage_diagnostic_only",
            "execution_environment": "local_windows_cuda",
            "planned_fixed_dev_images": len(self.image_ids),
            "training": False, "tensor_cache_written": False,
        }), flush=True)

        source_python = Path(self.cfg["source_python"])
        if str(source_python) not in sys.path:
            sys.path.insert(0, str(source_python))
        from pycocotools.coco import COCO
        import legacy_prepare_cache as official

        annotation_hashes = {domain: sha256(self.cfg[f"annotations_{domain}"])
                             for domain in ("train", "val")}
        if annotation_hashes != _ANNOTATION_SHA256:
            raise AssertionError("Original COCO annotation hashes differ")
        self.coco = COCO(self.cfg["annotations_train"])
        missing = set(self.image_ids) - set(self.coco.getImgIds())
        if missing:
            raise ValueError(f"Fixed DEV IDs absent in train annotations: {sorted(missing)[:20]}")
        self.cat_to_idx = {int(cid): i for i, cid in enumerate(sorted(self.coco.getCatIds()))}
        ensure_space(self.cfg)
        receipts = prepare_images(self.cfg, {"train": self.image_ids})
        dump(self.out / "STREAMING_SOURCE_IMAGES.json", receipts)
        # The official converter creates links for both domains. Ensure its
        # unused validation target exists without extracting validation JPEGs.
        (Path(self.cfg["images"]) / "val2017").mkdir(parents=True, exist_ok=True)

        self.replay = LocalFrozenReplay(self.cfg)
        model = self.replay.source
        model.args = SimpleNamespace(**self.replay.train_args)
        if model.args.mask_ratio != 1 or not model.args.overlap_mask:
            raise AssertionError("Official checkpoint must retain mask_ratio=1 and overlap_mask=True")
        self.criterion = model.init_criterion().one2one
        self.signature = dict(
            schema=SCHEMA, mode="authorized_local_damage_diagnostic_streaming",
            execution_environment="local_windows_cuda", split=split, source_split=self.domain,
            image_ids=self.image_ids, weights_sha256=self.replay.weights_sha256,
            annotations=annotation_hashes, dataset_code_sha256=sha256(official.__file__),
            official_mask_ratio=model.args.mask_ratio, official_overlap_mask=model.args.overlap_mask,
            segmentation_gain=float(model.args.box),
            forward="original FP32 eval batch1, no augmentation, no TF32; local diagnostic only",
        )
        self.fingerprint = hashlib.sha256(
            json.dumps(self.signature, sort_keys=True).encode()
        ).hexdigest()
        dataset_root = (Path(self.cfg["server_root"]) / "streaming_datasets"
                        / f"dev_{self.fingerprint[:16]}")
        fixed_split = {"fit": [], "dev": self.image_ids, "val": []}
        converted, self._identity, self._source_meta = official.prepare_dataset(
            self.cfg, fixed_split, dataset_root, self.fingerprint
        )
        self.dataset_root = dataset_root
        self.converted = converted

        from ultralytics.data.dataset import YOLODataset

        listing = dataset_root / "dev_input.txt"
        listing.write_text(
            "\n".join(str(converted / "images" / "train2017" / f"{iid:012d}.jpg")
                      for iid in self.image_ids) + "\n", encoding="utf-8"
        )
        self.dataset = YOLODataset(
            img_path=str(listing), imgsz=640, batch_size=1, augment=False,
            hyp=deepcopy(model.args), rect=False, cache=False, stride=32,
            data={"names": model.names, "nc": 80, "channels": 3}, task="segment",
        )
        self._positions = {int(Path(path).stem): position
                           for position, path in enumerate(self.dataset.im_files)}
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
            "streaming_code_sha256": sha256(Path(__file__).with_name("qcr_streaming_data.py")),
            "local_adapter_code_sha256": sha256(__file__),
            "prototype_dtype": "float32", "coefficient_dtype": "float32",
            "cpu_gt_raster_purpose": "diagnostic supervision only; never model evidence",
            "tensor_cache_written": False,
        })
        self._write_state("ready", immutable_verified=None)

    def iter_images(self):
        # The model/data/TAL construction remains the inherited implementation.
        # Correct the inherited historical server-origin marker for these local
        # transient payloads before exposing them to the diagnostic consumer.
        original = super().iter_images()

        def local_payloads():
            for iid, payload in original:
                payload["_asset_integrity"]["original_server_forward"] = False
                payload["_asset_integrity"]["original_local_forward"] = True
                payload["_asset_integrity"]["execution_environment"] = "local_windows_cuda"
                yield iid, payload

        return local_payloads()
