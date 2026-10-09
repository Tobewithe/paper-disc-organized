"""Faithful author SegRefiner LR inference with bounded framework shims.

Author forward/sampling/crop/paste code is executed unchanged. Framework-only
imports are removed from an AST, with explicitly supplied construction symbols.
No framework installation, global module patch, GT loader or loss training.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import runpy
import types

import numpy as np
import torch
from torch import nn

WEIGHT_SHA = "d5806a41a4bff6f1e313d1a9dcf18d11dab28e3a8e1497871a156214cd8fe07a"
UPSTREAM_REVISION = "53419a2d38ea3da0b6e2be77e5b45e139195a0b3"
RUNTIME_VERSION = "segrefiner_author_lr_framework_shim_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Registry:
    def __init__(self):
        self.classes = {}
    def register_module(self):
        def registered(cls):
            if cls.__name__ in self.classes:
                raise ValueError("Duplicate author class registration")
            self.classes[cls.__name__] = cls
            return cls
        return registered
    def build(self, config):
        options = dict(config)
        name = options.pop("type")
        return self.classes[name](**options)


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg


class NoTrainingMaskLoss(nn.Module):
    """Author default CE has no state; training is outside this adapter."""
    def __init__(self, use_sigmoid=True, loss_weight=1.0):
        super().__init__()
        self.use_sigmoid, self.loss_weight = use_sigmoid, loss_weight
    def forward(self, *args, **kwargs):
        raise RuntimeError("SegRefiner adapter is inference-only; no training loss is implemented")


class CoarseBitmapMasks:
    def __init__(self, masks):
        self.masks = np.asarray(masks, dtype=np.uint8)
        if self.masks.ndim != 3 or not np.isin(self.masks, [0, 1]).all():
            raise ValueError("Original-grid binary masks required")
    @property
    def areas(self):
        return self.masks.sum(axis=(1, 2))


def execute_author(path, injected, allowed_framework_imports):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    removed = []
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module, node.level) in allowed_framework_imports:
            removed.append(ast.unparse(node))
        else:
            kept.append(node)
    if not removed:
        raise ValueError("Expected framework imports were absent; do not silently adapt another source")
    tree.body = kept
    namespace = {"__name__": "_segrefiner_author_" + Path(path).stem, "__file__": str(path), **injected}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace, removed


def build_author_classes(upstream):
    root = Path(upstream)
    heads, detectors, losses = Registry(), Registry(), Registry()
    head_path = root / "mmdet/models/dense_heads/diffusion_unet_head.py"
    base_path = root / "mmdet/models/detectors/segrefiner_base.py"
    instance_path = root / "mmdet/models/detectors/segrefiner_instance.py"
    texture_path = root / "mmdet/models/losses/textrue_l1_loss.py"
    # The genuine author training-only Sobel submodule supplies its two strict
    # checkpoint keys. No state key is ignored or loaded with strict=False.
    texture_ast = ast.parse(texture_path.read_text(encoding="utf-8"))
    texture_ast.body = [node for node in texture_ast.body if isinstance(node, ast.ClassDef) and node.name in ("TextureL1Loss", "SobelOperator")]
    if len(texture_ast.body) != 2:
        raise ValueError("Author training-only texture construction changed")
    def no_training_l1(*args, **kwargs):
        raise RuntimeError("Training texture loss is outside inference adapter")
    texture_scope = {"__name__": "_segrefiner_author_texture", "nn": nn, "torch": torch, "np": np,
                     "F": torch.nn.functional, "LOSSES": losses, "l1_loss": no_training_l1}
    exec(compile(ast.fix_missing_locations(texture_ast), str(texture_path), "exec"), texture_scope)
    def build_loss(config):
        values = dict(config)
        kind = values.pop("type")
        if kind == "CrossEntropyLoss":
            return NoTrainingMaskLoss(**values)
        return losses.classes[kind](**values)
    head, head_removed = execute_author(head_path, {"BaseModule": BaseModule, "HEADS": heads}, {("mmcv.runner", 0), ("builder", 2)})
    base, base_removed = execute_author(base_path, {"BaseModule": BaseModule, "DETECTORS": detectors,
        "build_head": heads.build, "build_loss": build_loss}, {("mmcv.runner", 0), ("builder", 2)})
    instance, instance_removed = execute_author(instance_path, {"SegRefiner": base["SegRefiner"], "DETECTORS": detectors},
                                                {("segrefiner_base", 1), ("builder", 2)})
    return instance, {"removed_framework_imports": {"head": head_removed, "base": base_removed, "instance": instance_removed},
                      "original_method_bodies_changed": False, "training_loss_forward_supported": False,
                      "state_keys_ignored": [], "author_source_sha256": {str(path.relative_to(root)): sha256(path) for path in (
                          head_path, base_path, instance_path, texture_path,
                          root / "configs/segrefiner/segrefiner_lr.py", root / "configs/segrefiner/segrefiner_coco.py")}}


def normalize_rgb(rgb):
    """MMCV imnormalize_ numeric operations, starting from genuine RGB bytes."""
    import cv2
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Unresized original RGB uint8 ndarray required")
    normalized = image.astype(np.float32).copy()
    # MMDet Normalize first stores config constants as float32; MMCV then
    # promotes these already-rounded values to float64 for OpenCV arithmetic.
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, -1).astype(np.float64)
    reciprocal_std = 1 / np.asarray([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, -1).astype(np.float64)
    cv2.subtract(normalized, mean, normalized)
    cv2.multiply(normalized, reciprocal_std, normalized)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


class OfficialSegRefiner:
    def __init__(self, upstream, checkpoint, device="cuda", base_seed=20261009):
        self.upstream, self.checkpoint, self.device = Path(upstream), Path(checkpoint), device
        self.base_seed = base_seed
        if sha256(checkpoint) != WEIGHT_SHA:
            raise ValueError("Official LR checkpoint bytes differ")
        author, adaptation = build_author_classes(upstream)
        lr = runpy.run_path(str(self.upstream / "configs/segrefiner/segrefiner_lr.py"))
        coco = runpy.run_path(str(self.upstream / "configs/segrefiner/segrefiner_coco.py"))
        configuration = {**lr["model"], **coco["model"]}
        kind = configuration.pop("type")
        if kind != "SegRefinerInstance" or configuration["step"] != 6 or configuration["test_cfg"] != {
                "pad_width": 20, "model_size": 256, "batch_max": 32, "area_thr": 512}:
            raise ValueError("Locked official LR/COCO configuration differs")
        self.author = author
        self.model = author["SegRefinerInstance"](**configuration).float().eval()
        state = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
        if len(state) != 455 or sum(name.startswith("denoise_model.") for name in state) != 453 or set(
                name for name in state if not name.startswith("denoise_model.")) != {
                    "loss_texture.sobel.conv_x.weight", "loss_texture.sobel.conv_y.weight"}:
            raise ValueError("Official complete455 key contract differs")
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Checkpoint tensors must all be finite FP32")
        loaded = self.model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError("Strict complete455 checkpoint loading failed")
        self.initial_state_digest = tensor_state_digest(state)
        if tensor_state_digest(self.model.state_dict()) != self.initial_state_digest:
            raise ValueError("Loaded tensors differ from all original checkpoint values")
        self.model.requires_grad_(False).to(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.forward_calls = 0
        def counted(module, inputs, output):
            self.forward_calls += 1
        self.forward_hook = self.model.denoise_model.register_forward_hook(counted)
        self.provenance = {"runtime_version": RUNTIME_VERSION, "checkpoint_sha256": WEIGHT_SHA,
                           "upstream_revision": UPSTREAM_REVISION, "configuration": configuration, "adaptation": adaptation,
                           "strict_loaded_state_keys": 455, "denoise_state_keys": 453, "sobel_state_keys": 2,
                           "all_loaded_tensor_bytes_exact": True, "loaded_state_sha256": self.initial_state_digest,
                           "base_seed": base_seed, "per_image_seed_formula": "20261009 + original COCO image_id",
                           "author_sampling": "i=5..0; original torch.rand fine_map in first5 steps, sigmoid final step",
                           "input_normalization": "original RGB255, config mean/std float32 then float64; OpenCV subtract then multiply(float64 reciprocal std) into float32",
                           "gt_loader_constructed": False, "training_loss_used": False}
        del state

    def seed(self, image_id):
        value = self.base_seed + int(image_id)
        torch.manual_seed(value)
        if str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(value)
        return value

    def author_inputs(self, rgb, masks, identities):
        masks = CoarseBitmapMasks(masks)
        image = torch.from_numpy(normalize_rgb(rgb))[None].to(self.device)
        h, w = rgb.shape[:2]
        metadata = [{"img_shape": (h, w, 3), "ori_shape": (h, w, 3)}]
        return image, [masks], [np.asarray(identities, dtype=np.float32)], metadata

    def restore_author_outputs(self, author_result, masks, identities):
        """Restore author's valid/tiny -> per-class groups to original ordinals."""
        bitmap = CoarseBitmapMasks(masks)
        valid = bitmap.areas >= 512
        order = np.concatenate((np.flatnonzero(valid), np.flatnonzero(~valid)))
        boxes, grouped_masks = author_result[0]
        cursors = np.zeros(80, dtype=np.int64)
        result = np.empty_like(bitmap.masks)
        identities = np.asarray(identities, dtype=np.float32)
        for ordinal in order:
            label = int(identities[ordinal, 5])
            slot = int(cursors[label])
            if not np.array_equal(boxes[label][slot], identities[ordinal, :5]):
                raise RuntimeError("Author formatter changed given bbox/score identity")
            result[ordinal] = grouped_masks[label][slot]
            cursors[label] += 1
        if sum(cursors) != len(masks) or any(cursors[label] != len(grouped_masks[label]) for label in range(80)):
            raise RuntimeError("Author class-group/ordinal coverage changed")
        if not np.array_equal(result[~valid], bitmap.masks[~valid]):
            raise RuntimeError("Author tiny/empty fallback changed")
        return result

    def refine(self, rgb, baseline_masks, eligible_indices, image_id, identities):
        cpu_state = torch.get_rng_state().clone()
        cuda_states = torch.cuda.get_rng_state_all() if str(self.device).startswith("cuda") else []
        devices = list(range(torch.cuda.device_count())) if cuda_states else []
        with torch.random.fork_rng(devices=devices):
            result, diagnostics = self._refine_seeded(rgb, baseline_masks, eligible_indices, image_id, identities)
        cpu_restored = torch.equal(cpu_state, torch.get_rng_state())
        cuda_restored = all(torch.equal(before, after) for before, after in zip(cuda_states, torch.cuda.get_rng_state_all())) if cuda_states else None
        if not cpu_restored or cuda_restored is False:
            raise RuntimeError("External CPU/CUDA random state was not restored")
        diagnostics.update(external_torch_cpu_rng_restored=cpu_restored, external_torch_cuda_rng_restored=cuda_restored)
        return result, diagnostics

    def _refine_seeded(self, rgb, baseline_masks, eligible_indices, image_id, identities):
        baseline = np.asarray(baseline_masks, dtype=np.uint8)
        if baseline.ndim != 3 or baseline.shape[1:] != rgb.shape[:2]:
            raise ValueError("Baseline original-grid mask/photo geometry differs")
        positions = np.asarray(eligible_indices, dtype=np.int64)
        if (len(positions) > 64 or len(set(positions.tolist())) != len(positions)
                or (len(positions) and (positions.min() < 0 or positions.max() >= len(baseline)))
                or positions.tolist() != sorted(positions.tolist())):
            raise ValueError("Sealed supported first64 ordinals required, no replacement")
        seed = self.seed(image_id)
        original_calls = self.forward_calls
        cpu_before_state = torch.get_rng_state().clone()
        cuda_before_state = torch.cuda.get_rng_state().clone() if str(self.device).startswith("cuda") else None
        before_rng = hashlib.sha256(cpu_before_state.numpy().tobytes()).hexdigest()
        cuda_before = hashlib.sha256(cuda_before_state.cpu().numpy().tobytes()).hexdigest() if cuda_before_state is not None else None
        result = baseline.copy()
        valid_count = 0
        if len(positions):
            masks = baseline[positions]
            valid_count = int((masks.sum((1, 2)) >= 512).sum())
            if valid_count:
                image, bitmap, boxes, metadata = self.author_inputs(rgb, masks, np.asarray(identities)[positions])
                with torch.inference_mode():
                    original = self.model.simple_test_instance(metadata, img=image, coarse_masks=bitmap, dt_bboxes=boxes)
                result[positions] = self.restore_author_outputs(original, masks, np.asarray(identities)[positions])
            # Alltiny includesempty: exact original fallback without a model call.
        expected_calls = 6 * ((valid_count + 31) // 32)
        if self.forward_calls - original_calls != expected_calls:
            raise RuntimeError("Actual author six-step/batch32 denoiser call count differs")
        unselected = np.ones(len(baseline), dtype=bool)
        unselected[positions] = False
        if not np.array_equal(result[unselected], baseline[unselected]):
            raise RuntimeError("An out-of-scope native ordinal changed")
        self.last_rng_states = {"cpu_before": cpu_before_state, "cpu_after": torch.get_rng_state().clone()}
        if cuda_before_state is not None:
            self.last_rng_states.update(cuda_before=cuda_before_state, cuda_after=torch.cuda.get_rng_state().clone())
        diagnostics = {"selected_supported_ordinals": positions.tolist(), "model_valid_area512": valid_count,
                       "tiny_or_empty_fallback": len(positions) - valid_count, "outside_scope_identity": int(unselected.sum()),
                       "actual_denoiser_forward_calls": expected_calls, "seed": seed,
                       "torch_cpu_rng_before_sha256": before_rng,
                       "torch_cpu_rng_after_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                       "torch_cuda_rng_before_sha256": cuda_before,
                       "torch_cuda_rng_after_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest() if str(self.device).startswith("cuda") else None,
                       "all_ordinal_count_preserved": len(result) == len(baseline), "gt_used": False,
                       "fallback_only_case": "no_supported_candidates" if len(positions) == 0 else "alltiny_or_empty" if valid_count == 0 else None}
        return result, diagnostics

    def verify_unchanged(self):
        final = tensor_state_digest(self.model.state_dict())
        if final != self.initial_state_digest or any(parameter.requires_grad or parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("Official model state/gradient integrity changed")
        return {"passed": True, "initial_state_sha256": self.initial_state_digest, "final_state_sha256": final,
                "all_parameter_values_exact": True, "all_requires_grad_false": True, "all_gradients_none": True}
