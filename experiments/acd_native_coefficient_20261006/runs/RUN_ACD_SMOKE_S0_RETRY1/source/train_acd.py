"""Fixed paired screen: native coefficient heads only, with auditable frozen state.

Ultralytics 8.4.100 is the base implementation. ACD adds a training-only
false-positive term to the segmentation component of each official branch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import traceback
import weakref
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("WANDB_MODE", "disabled")

# Select the offline vendor before importing Ultralytics. PYTHONPATH also works.
_vendor_parser = argparse.ArgumentParser(add_help=False)
_vendor_parser.add_argument("--vendor", type=Path)
_vendor_args, _ = _vendor_parser.parse_known_args()
if _vendor_args.vendor:
    _vendor = _vendor_args.vendor.resolve()
    if not (_vendor / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid Ultralytics vendor root: {_vendor}")
    sys.path.insert(0, str(_vendor))

import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask
from ultralytics.utils.torch_utils import strip_optimizer, unwrap_model
import ultralytics.engine.trainer as trainer_module
import ultralytics.utils.checks as checks_module

if ultralytics.__version__ != "8.4.100":
    raise RuntimeError(f"Expected Ultralytics 8.4.100, loaded {ultralytics.__version__} from {ultralytics.__file__}")

ORIGINAL_SEG_INIT = v8SegmentationLoss.__init__
ORIGINAL_SEG_LOSS = v8SegmentationLoss.loss
LOSS_MODELS = weakref.WeakKeyDictionary()
METHOD_ENABLED = False
RUN_ROOT: Path | None = None
BASE_WEIGHTS: Path | None = None
ACD_WEIGHT = 0.05
ACD_TAU = 0.50
ACD_MIN_GAIN = 0.01
ACD_MIN_COVERAGE = 0.80
ACD_MAX_PER_IMAGE = 12
COEFF_PREFIXES = ("model.23.cv4.", "model.23.one2one_cv4.")
STATS = {branch: {"loss_calls": 0, "candidate_count": 0, "action_positive": 0,
                  "pixel_terms": 0, "aux_sum": 0.0}
         for branch in ("one2many", "one2one")}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _is_coeff(name):
    return name.startswith(COEFF_PREFIXES)


def _tensor_hash(tensor: torch.Tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(model, names):
    state = model.state_dict()
    by_tensor = {name: _tensor_hash(state[name]) for name in sorted(names)}
    digest = hashlib.sha256()
    for name, value in by_tensor.items():
        digest.update(f"{name}:{value}\n".encode())
    return {"sha256": digest.hexdigest(), "tensor_count": len(by_tensor), "tensors": by_tensor}


def _lock_all_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters(recurse=False):
                param.requires_grad_(False)


def _freeze_for_coefficients(model):
    model = unwrap_model(model)
    for name, param in model.named_parameters():
        param.requires_grad_(_is_coeff(name))
    _lock_all_bn(model)
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    for prefix in COEFF_PREFIXES:
        if not any(name.startswith(prefix) for name in names):
            raise RuntimeError(f"Missing trainable native coefficient branch {prefix}")
    if any(not _is_coeff(name) for name in names):
        raise RuntimeError("Trainable parameters escaped native coefficient heads")
    return names


def _patched_seg_init(self, model, tal_topk=10, tal_topk2=None):
    ORIGINAL_SEG_INIT(self, model, tal_topk, tal_topk2)
    # Weak references stay out of serialized model/criterion state.
    LOSS_MODELS[self] = (weakref.ref(model), "one2many" if tal_topk == 10 else "one2one")


def _mask_iou(pred, gt):
    return (pred & gt).sum().float() / (pred | gt).sum().float().clamp_min(1)


def _get_gt_mask(self, masks, batch, image_index, gt_index):
    if self.overlap:
        # Preserve integer instance IDs until selecting this assigned instance.
        return masks[image_index] == gt_index + 1
    local_masks = masks[batch["batch_idx"].view(-1).to(masks.device) == image_index]
    return local_masks[gt_index] > 0.5


def _acd_aux(self, preds, batch, assigned, branch):
    pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
    proto = preds["proto"]
    if isinstance(proto, tuple):
        proto = proto[0]
    masks = batch["masks"].to(self.device).float()
    # Official 8.4.100 loss upsamples proto, never the GT instance map.
    if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
        proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
    fg_mask, target_gt_idx, target_bboxes, _, _ = assigned
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device,
                         dtype=pred_masks.dtype) * self.stride[0]
    h, w = masks.shape[-2:]
    boxes = target_bboxes / imgsz[[1, 0, 1, 0]]
    boxes = boxes * torch.tensor([w, h, w, h], device=self.device)
    stats = STATS[branch]
    terms = []
    for bi, fg in enumerate(fg_mask):
        # Deterministic first 12 assigned anchors per image, per branch.
        ids = fg.nonzero(as_tuple=False).view(-1)[:ACD_MAX_PER_IMAGE]
        for anchor in ids.tolist():
            gt_index = int(target_gt_idx[bi, anchor].item())
            gt = _get_gt_mask(self, masks, batch, bi, gt_index)
            with torch.no_grad():
                support = crop_mask(torch.ones_like(gt, dtype=torch.float32)[None],
                                    boxes[bi, anchor][None])[0].bool()
                gt = gt & support
            if not gt.any():
                continue
            logits = torch.einsum("c,chw->hw", pred_masks[bi, anchor], proto[bi])
            with torch.no_grad():
                base = (logits.detach() > 0) & support
                acted = (logits.detach() > ACD_TAU) & support
                base_iou, acted_iou = _mask_iou(base, gt), _mask_iou(acted, gt)
                coverage = (base & gt).sum().float() / gt.sum().float().clamp_min(1)
                useful = bool((coverage >= ACD_MIN_COVERAGE)
                              & (acted_iou >= base_iou + ACD_MIN_GAIN))
                fp = base & ~gt & support
            stats["candidate_count"] += 1
            if not useful or not fp.any():
                continue
            terms.append(F.softplus(logits[fp]).mean())
            stats["action_positive"] += 1
            stats["pixel_terms"] += int(fp.sum().item())
    if not terms:
        return pred_masks.sum() * 0.0
    aux = torch.stack(terms).mean()
    stats["aux_sum"] += float(aux.detach())
    return aux


def patched_loss(self, preds, batch):
    reference = LOSS_MODELS.get(self)
    model = reference[0]() if reference is not None else None
    # Validation uses its own EMA model in eval mode. No validation action loss.
    if not METHOD_ENABLED or model is None or not model.training or not torch.is_grad_enabled():
        return ORIGINAL_SEG_LOSS(self, preds, batch)
    assigned = []
    original_getter = self.get_assigned_targets_and_loss
    had_local = "get_assigned_targets_and_loss" in self.__dict__
    local_getter = self.__dict__.get("get_assigned_targets_and_loss")

    def capture(predictions, targets):
        result = original_getter(predictions, targets)
        assigned.append(result[0])
        return result

    self.get_assigned_targets_and_loss = capture
    try:
        base, items = ORIGINAL_SEG_LOSS(self, preds, batch)
    finally:
        if had_local:
            self.get_assigned_targets_and_loss = local_getter
        else:
            del self.get_assigned_targets_and_loss
    branch = reference[1]
    STATS[branch]["loss_calls"] += 1
    aux = _acd_aux(self, preds, batch, assigned[0], branch)
    weighted = ACD_WEIGHT * aux
    base = base.clone()
    items = items.clone()
    # Official five-component base is already multiplied by batch size.
    base[1] = base[1] + weighted * preds["mask_coefficient"].shape[0]
    items[1] = items[1] + weighted.detach()
    return base, items


v8SegmentationLoss.__init__ = _patched_seg_init
v8SegmentationLoss.loss = patched_loss


class _OptimizerView(torch.nn.Module):
    """Expose trainable leaves to the unchanged official optimizer builder."""

    def __init__(self, model):
        super().__init__()
        object.__setattr__(self, "source", model)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        for name, module in self.source.named_modules():
            parameters = {key: value for key, value in module._parameters.items()
                          if value is not None and value.requires_grad}
            if parameters:
                leaf = copy.copy(module)
                leaf._parameters = parameters
                yield name, leaf


class ACDTrainer(SegmentationTrainer):
    """Both arms share all freezing, optimizer, precision and data behavior."""

    def _build_train_pipeline(self):
        _freeze_for_coefficients(self.model)
        super()._build_train_pipeline()

    def build_optimizer(self, model, **kwargs):
        trainable_names = _freeze_for_coefficients(model)
        source = unwrap_model(model)
        optimizer = super().build_optimizer(_OptimizerView(source), **kwargs)
        expected = {id(param) for name, param in source.named_parameters() if name in trainable_names}
        actual = {id(param) for group in optimizer.param_groups for param in group["params"]}
        if actual != expected:
            raise RuntimeError("Optimizer differs from native coefficient training set")
        return optimizer

    def _model_train(self):
        super()._model_train()
        _lock_all_bn(unwrap_model(self.model))

    def _setup_train(self):
        super()._setup_train()
        model = unwrap_model(self.model)
        self._trainable_names = _freeze_for_coefficients(model)
        self._frozen_names = {name for name, _ in model.named_parameters()
                              if name not in self._trainable_names}
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                for name, _ in module.named_buffers(recurse=False):
                    self._frozen_names.add(f"{module_name}.{name}" if module_name else name)
        self._frozen_initial = _state_digest(model, self._frozen_names)
        self._initial_coeff = {name: param.detach().float().cpu().clone()
                               for name, param in model.named_parameters() if name in self._trainable_names}
        self._gradient_stats = {branch: {"optimizer_steps_checked": 0, "attempted_steps": 0,
                                         "applied_steps": 0, "skipped_overflow_steps": 0,
                                         "nonzero_steps": 0, "all_finite": True,
                                         "observed_all_gradients_finite": True,
                                         "observed_all_scaled_gradients_finite": True,
                                         "observed_nonfinite_gradient_steps": 0,
                                         "observed_nonfinite_scaled_gradient_steps": 0,
                                         "observed_nonfinite_skipped_overflow_steps": 0,
                                         "all_observed_nonfinite_steps_skipped": True,
                                         "all_applied_gradients_finite": True,
                                         "applied_zero_gradient_steps": 0,
                                         "maximum_absolute_gradient": 0.0}
                                for branch in ("one2many", "one2one")}
        self._optimizer_step_stats = {"attempted_steps": 0, "applied_steps": 0,
                                      "skipped_overflow_steps": 0,
                                      "observed_all_gradients_finite": True,
                                      "observed_nonfinite_gradient_steps": 0,
                                      "observed_nonfinite_skipped_overflow_steps": 0,
                                      "all_observed_nonfinite_steps_skipped": True,
                                      "all_applied_gradients_finite": True,
                                      "unsafe_step_prevented_count": 0,
                                      "unclassified_skipped_steps": 0}
        self._batch_counter = 0
        self._finite_loss = True
        self._input_stream_digest = hashlib.sha256()
        self._sync_frozen_ema()
        for label, network in [('live', model), ('ema', unwrap_model(self.ema.ema))]:
            if not all(torch.isfinite(value).all() for value in network.state_dict().values()
                       if value.is_floating_point()):
                raise RuntimeError(f'Non-finite {label} model state before checkpoint serialization')
        setup = {"status": "setup_complete", "timestamp_utc": _utc_now(),
                 "runtime_ultralytics": ultralytics.__version__,
                 "ultralytics_source": str(Path(ultralytics.__file__).resolve()),
                 "arm": "acd" if METHOD_ENABLED else "baseline", "base_weights": str(BASE_WEIGHTS),
                 "base_weights_sha256": _file_hash(BASE_WEIGHTS),
                 "data": str(self.args.data), "data_yaml_sha256": _file_hash(Path(self.args.data)),
                 "save_dir": str(self.save_dir), "device": str(self.device),
                 "epochs": self.epochs, "seed": self.args.seed, "batch": self.batch_size,
                 "train_images": len(self.train_loader.dataset), "validation_images": len(self.test_loader.dataset),
                 "optimizer": type(self.optimizer).__name__,
                 "optimizer_parameters": sorted(self._trainable_names),
                 "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                 "all_batchnorm_eval": True, "all_batchnorm_affine_frozen": True,
                 "amp": self.amp, "amp_capability_check": "offline bypass; smoke verifies finite loss",
                 "gradient_audit_policy": "raw unscaled gradients reported separately; all_finite means actual applied steps",
                 "optimizer_step_log": str(RUN_ROOT / "optimizer_step_audit.jsonl"),
                 "frozen_state_initial": self._frozen_initial,
                 "initial_coefficient_sha256": {name: _tensor_hash(value) for name, value in self._initial_coeff.items()},
                 "acd": {"weight": ACD_WEIGHT, "tau": ACD_TAU, "min_gain": ACD_MIN_GAIN,
                         "min_coverage": ACD_MIN_COVERAGE, "max_per_image_per_branch": ACD_MAX_PER_IMAGE},
                 "evaluation_policy": "fixed final epoch; best.pt never selected for final evaluation"}
        _write_json(RUN_ROOT / "TRAINING_SETUP.json", setup)
        print("TRAINING_SETUP", json.dumps(setup), flush=True)
        self.add_callback("on_train_batch_end", lambda trainer: self._check_loss())

    def _check_loss(self):
        self._finite_loss = self._finite_loss and bool(torch.isfinite(self.loss.detach()).all())
        if not self._finite_loss:
            raise RuntimeError("Non-finite training loss")

    @torch.no_grad()
    def _sync_frozen_ema(self):
        if not self.ema:
            return
        live = unwrap_model(self.model).state_dict()
        ema = unwrap_model(self.ema.ema).state_dict()
        # EMA averaging can round unchanged FP32 tensors. Preserve frozen state.
        for name in self._frozen_names:
            ema[name].copy_(live[name])

    def _gradient_snapshot(self):
        """Read gradients without changing them or consuming random numbers."""
        model = unwrap_model(self.model)
        snapshots = {}
        for branch, prefix in zip(("one2many", "one2one"), COEFF_PREFIXES):
            gradients = [(name, param.grad.detach()) for name, param in model.named_parameters()
                         if name.startswith(prefix) and param.requires_grad and param.grad is not None]
            nonfinite = [name for name, gradient in gradients if not bool(torch.isfinite(gradient).all())]
            maximum = float(torch.stack([gradient.abs().max() for _, gradient in gradients]).max()) \
                if gradients and not nonfinite else None
            snapshots[branch] = {"gradient_tensor_count": len(gradients),
                                 "all_gradients_finite": not nonfinite,
                                 "nonfinite_gradient_parameters": nonfinite,
                                 "maximum_absolute_gradient": maximum,
                                 "finite_nonzero_gradient": bool(maximum is not None and maximum > 0)}
        return snapshots

    @staticmethod
    def _snapshots_finite(snapshots):
        return all(details["all_gradients_finite"] for details in snapshots.values())

    def optimizer_step(self):
        """Observe the unchanged official step and distinguish scaler skips.

        The superclass retains unscale -> clip(10) -> scaler.step -> update ->
        zero_grad -> EMA, including the EMA update on a skipped optimizer step.
        Read-only hooks confirm whether optimizer.step actually ran; its usual
        None return value is never used to classify a skip.
        """
        summary = self._optimizer_step_stats
        summary["attempted_steps"] += 1
        scale_before = float(self.scaler.get_scale())
        ema_updates_before = int(self.ema.updates) if self.ema else None
        scaler_enabled = bool(self.scaler.is_enabled())
        observation = {"scaled": self._gradient_snapshot(), "unscaled": None,
                       "at_optimizer_call": None, "optimizer_invoked": False,
                       "optimizer_applied": False, "unsafe_step_prevented": False}
        original_unscale = self.scaler.unscale_
        had_local_unscale = "unscale_" in self.scaler.__dict__
        local_unscale = self.scaler.__dict__.get("unscale_")

        def observe_unscale(optimizer):
            result = original_unscale(optimizer)
            observation["unscaled"] = self._gradient_snapshot()
            return result

        def before_actual_step(optimizer, positional, keyword):
            observation["optimizer_invoked"] = True
            observation["at_optimizer_call"] = self._gradient_snapshot()
            if not self._snapshots_finite(observation["at_optimizer_call"]):
                observation["unsafe_step_prevented"] = True
                # GradScaler normally suppresses such calls. Stop before a
                # genuinely non-finite clipped gradient can be applied.
                raise RuntimeError("Non-finite gradient reached optimizer.step; unsafe application prevented")

        def after_actual_step(optimizer, positional, keyword):
            observation["optimizer_applied"] = True

        pre_handle = self.optimizer.register_step_pre_hook(before_actual_step)
        post_handle = self.optimizer.register_step_post_hook(after_actual_step)
        self.scaler.unscale_ = observe_unscale
        error = None
        try:
            super().optimizer_step()
            self._sync_frozen_ema()
        except BaseException as caught:
            error = caught
        finally:
            pre_handle.remove()
            post_handle.remove()
            if had_local_unscale:
                self.scaler.unscale_ = local_unscale
            else:
                del self.scaler.unscale_

        scale_after = float(self.scaler.get_scale())
        summary['ema_updates'] = int(self.ema.updates) if self.ema else 0
        observed = observation["unscaled"]
        applied = observation["optimizer_applied"]
        overflow_skip = bool(error is None and scaler_enabled and self.amp
                             and not observation["optimizer_invoked"] and not applied
                             and observed is not None and not self._snapshots_finite(observed)
                             and scale_after < scale_before)
        if applied:
            summary["applied_steps"] += 1
            summary["all_applied_gradients_finite"] &= self._snapshots_finite(observation["at_optimizer_call"])
        if observed is not None:
            observed_finite = self._snapshots_finite(observed)
            summary["observed_all_gradients_finite"] &= observed_finite
            summary["observed_nonfinite_gradient_steps"] += int(not observed_finite)
            summary["observed_nonfinite_skipped_overflow_steps"] += int(not observed_finite and overflow_skip)
            summary["all_observed_nonfinite_steps_skipped"] &= bool(observed_finite or overflow_skip)
        summary["skipped_overflow_steps"] += int(overflow_skip)
        summary["unsafe_step_prevented_count"] += int(observation["unsafe_step_prevented"])
        if error is None and not applied and not overflow_skip:
            summary["unclassified_skipped_steps"] += 1
            error = RuntimeError("Optimizer step was skipped without a confirmed AMP overflow")

        for branch in ("one2many", "one2one"):
            stats = self._gradient_stats[branch]
            stats["optimizer_steps_checked"] += 1
            stats["attempted_steps"] += 1
            stats["skipped_overflow_steps"] += int(overflow_skip)
            stats["observed_all_scaled_gradients_finite"] &= observation["scaled"][branch]["all_gradients_finite"]
            stats["observed_nonfinite_scaled_gradient_steps"] += int(
                not observation["scaled"][branch]["all_gradients_finite"])
            if observed is not None:
                observed_finite = observed[branch]["all_gradients_finite"]
                stats["observed_all_gradients_finite"] &= observed_finite
                stats["observed_nonfinite_gradient_steps"] += int(not observed_finite)
                stats["observed_nonfinite_skipped_overflow_steps"] += int(not observed_finite and overflow_skip)
                stats["all_observed_nonfinite_steps_skipped"] &= bool(observed_finite or overflow_skip)
            if applied:
                actual = observation["at_optimizer_call"][branch]
                stats["applied_steps"] += 1
                stats["all_applied_gradients_finite"] &= actual["all_gradients_finite"]
                stats["nonzero_steps"] += int(actual["finite_nonzero_gradient"])
                stats["applied_zero_gradient_steps"] += int(not actual["finite_nonzero_gradient"])
                if actual["maximum_absolute_gradient"] is not None:
                    stats["maximum_absolute_gradient"] = max(stats["maximum_absolute_gradient"],
                                                               actual["maximum_absolute_gradient"])
            # Backward-compatible name, now explicitly restricted to applied steps.
            stats["all_finite"] = stats["all_applied_gradients_finite"]
        record = {"epoch": self.epoch + 1, "attempt": summary["attempted_steps"],
                  "input_batches_seen": self._batch_counter, "amp": self.amp,
                  "learning_rates": [float(group['lr']) for group in self.optimizer.param_groups],
                  "accumulate": int(self.accumulate),
                  "ema_updates_before": ema_updates_before,
                  "ema_updates_after": summary['ema_updates'],
                  "scaler_enabled": scaler_enabled, "scale_before": scale_before,
                  "scale_after": scale_after, "skipped_overflow_step": overflow_skip,
                  **observation, "exception": str(error) if error is not None else None}
        with (RUN_ROOT / "optimizer_step_audit.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        if error is not None:
            raise error

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        model = unwrap_model(self.model)
        if any(module.training for module in model.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError("A BatchNorm module entered training mode")
        hashes = {key: _tensor_hash(batch[key]) for key in ("img", "masks", "batch_idx", "cls", "bboxes")
                  if key in batch and isinstance(batch[key], torch.Tensor)}
        sample = {"epoch": self.epoch, "batch_index": self._batch_counter,
                  "im_file": [str(value) for value in batch.get("im_file", [])], "tensor_sha256": hashes}
        line = json.dumps(sample, sort_keys=True)
        self._input_stream_digest.update((line + "\n").encode())
        with (RUN_ROOT / "input_batch_hashes.jsonl").open("a", encoding="utf-8") as output:
            output.write(line + "\n")
        self._batch_counter += 1
        return batch

    def _finish_audit(self, raise_on_failure=True, failure_reason=None):
        """Save final/partial FP32 artifacts before reporting any audit failure."""
        model = unwrap_model(self.model)
        final = _state_digest(model, self._frozen_names)
        unchanged = final["sha256"] == self._frozen_initial["sha256"]
        ema_model = unwrap_model(self.ema.ema)
        ema_frozen = _state_digest(ema_model, self._frozen_names)
        ema_unchanged = ema_frozen["sha256"] == self._frozen_initial["sha256"]
        nonfinite_state = {
            label: [name for name, value in network.state_dict().items()
                    if value.is_floating_point() and not bool(torch.isfinite(value).all())]
            for label, network in (("live", model), ("ema", ema_model))}
        branch_updates = {}
        failures = [failure_reason] if failure_reason else []
        if not unchanged:
            failures.append("Frozen live parameters or BN buffers changed")
        if not ema_unchanged:
            failures.append("Frozen EMA parameters or BN buffers changed")
        if not self._finite_loss:
            failures.append("Non-finite training loss was observed")
        for label, names in nonfinite_state.items():
            if names:
                failures.append(f"Non-finite {label} model state")
        optimizer_audit = self._optimizer_step_stats
        if not optimizer_audit["all_applied_gradients_finite"]:
            failures.append("Non-finite gradients were applied")
        if not optimizer_audit["all_observed_nonfinite_steps_skipped"]:
            failures.append("An observed non-finite gradient attempt was not confirmed as an AMP overflow skip")
        if optimizer_audit["unsafe_step_prevented_count"]:
            failures.append("Non-finite gradients reached an optimizer invocation and were prevented")
        if optimizer_audit["unclassified_skipped_steps"]:
            failures.append("An optimizer skip lacked confirmed AMP-overflow evidence")
        if optimizer_audit["attempted_steps"] != (optimizer_audit["applied_steps"]
                                                  + optimizer_audit["skipped_overflow_steps"]):
            failures.append("Optimizer attempts are not all accounted for as applied or confirmed AMP skips")
        for branch, prefix in zip(("one2many", "one2one"), COEFF_PREFIXES):
            names = sorted(name for name in self._trainable_names if name.startswith(prefix))
            changed, max_delta, nonfinite_coefficients = [], 0.0, []
            current = dict(model.named_parameters())
            for name in names:
                value = current[name].detach().float().cpu()
                if not bool(torch.isfinite(value).all()):
                    nonfinite_coefficients.append(name)
                elif bool(torch.isfinite(self._initial_coeff[name]).all()):
                    # FP64 diagnostic subtraction avoids overflowing the audit
                    # itself when comparing large but finite FP32 values.
                    delta = float((value.double() - self._initial_coeff[name].double()).abs().max())
                    max_delta = max(max_delta, delta)
                if not torch.equal(value, self._initial_coeff[name]):
                    changed.append(name)
            branch_updates[branch] = {"changed_parameters": changed, "changed_parameter_count": len(changed),
                                      "max_absolute_delta": max_delta if not nonfinite_coefficients else None,
                                      "nonfinite_coefficient_parameters": nonfinite_coefficients,
                                      "gradients": self._gradient_stats[branch]}
            gradients = self._gradient_stats[branch]
            if not changed or not gradients["nonzero_steps"]:
                failures.append(f"Native coefficient branch {branch} lacks a nonzero applied gradient and update")
            if not gradients["all_applied_gradients_finite"]:
                failures.append(f"Native coefficient branch {branch} had non-finite applied gradients")
        audit_passed = not failures
        audit = {"frozen_state_initial": self._frozen_initial, "frozen_state_final": final,
                 "frozen_state_unchanged": unchanged, "ema_frozen_state_final": ema_frozen,
                 "ema_frozen_state_unchanged": ema_unchanged,
                 "live_state_finite": not nonfinite_state["live"],
                 "ema_state_finite": not nonfinite_state["ema"],
                 "nonfinite_model_state": nonfinite_state, "branch_updates": branch_updates,
                 "optimizer_audit": optimizer_audit, "audit_passed": audit_passed,
                 "audit_failures": failures,
                 "finite_training_loss": self._finite_loss, "input_batches": self._batch_counter,
                 "input_batch_stream_sha256": self._input_stream_digest.hexdigest(),
                 "gradient_magnitude_units": "unscaled after official clip, at actual optimizer invocation",
                 "gradient_all_finite_semantics": "all successfully applied optimizer gradients are finite; raw observed gradients are separate",
                 "optimizer_step_log": str(RUN_ROOT / "optimizer_step_audit.jsonl")}
        _write_json(RUN_ROOT / "TRAINING_AUDIT.json", audit)
        live = {name: value.detach().float().cpu().clone() for name, value in model.state_dict().items()
                if _is_coeff(name)}
        ema = {name: value.detach().float().cpu().clone()
               for name, value in ema_model.state_dict().items() if _is_coeff(name)}
        metadata = {"base_weights": str(BASE_WEIGHTS), "base_weights_sha256": _file_hash(BASE_WEIGHTS),
                    "epoch": self.epoch + 1, "runtime_ultralytics": ultralytics.__version__,
                    "audit_passed": audit_passed, "frozen_state_unchanged": unchanged,
                    "ema_frozen_state_unchanged": ema_unchanged,
                    "source_last_pt": str(self.last), "audit": audit}
        torch.save({**metadata, "coefficients": live, "ema_coefficients": ema}, RUN_ROOT / "coeff_final.pt")
        torch.save({**metadata, "state_dict": ema,
                    "kind": "coefficient_only_ema_final" if audit_passed else "coefficient_only_ema_partial_failed"},
                   RUN_ROOT / "coeff_final_ema.pt")
        self._audit = audit
        if failures and raise_on_failure:
            raise RuntimeError("Training audit failed: " + "; ".join(failures))

    def final_eval(self):
        self._finish_audit()
        if not self.last.is_file():
            raise FileNotFoundError(f"No fixed final checkpoint: {self.last}")
        # Main experiment evaluation reconstructs original FP32 frozen weights
        # plus coeff_final_ema.pt; official full-checkpoint saver rounds to FP16.
        strip_optimizer(self.last)
        self.validator.args.plots = False
        self.validator.args.compile = False
        self.metrics = self.validator(model=self.last)
        self.metrics.pop("fitness", None)
        _write_json(RUN_ROOT / "trainer_final_epoch_metrics.json", {
            "evaluation_checkpoint": str(self.last), "epoch": self.epoch + 1,
            "frozen_weight_precision": "trainer FP16 checkpoint; coefficient overlay required for exact FP32 evaluation",
            "metrics": {key: float(value) for key, value in self.metrics.items()}})


def official_weight_config(weight):
    ckpt, _ = torch_safe_load(str(weight))
    source = ckpt["train_args"]
    keys = ["imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "warmup_momentum", "warmup_bias_lr", "nbs", "amp", "deterministic", "rect", "cos_lr",
            "close_mosaic", "multi_scale", "semseg_loss", "overlap_mask", "mask_ratio", "dropout",
            "box", "cls", "dfl", "o2m", "muon_w", "sgd_w", "cls_w", "stride_ratio", "topk",
            "hungarian", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
            "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste",
            "copy_paste_mode", "auto_augment", "erasing"]
    return {key: source[key] for key in keys if key in source and key in DEFAULT_CFG_DICT}


def main():
    global METHOD_ENABLED, RUN_ROOT, BASE_WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--arm", choices=["baseline", "acd"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Run root; trainer artifacts go in out/trainer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.epochs not in (1, 3):
        parser.error("Protocol permits 1-epoch smoke or 3-epoch paired feasibility")
    METHOD_ENABLED = args.arm == "acd"
    RUN_ROOT = args.out.resolve()
    BASE_WEIGHTS = args.weight.resolve()
    data = args.data.resolve()
    if not BASE_WEIGHTS.is_file() or not data.is_file():
        raise FileNotFoundError("Official checkpoint and dataset YAML must already exist")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT / "TRAINING_SETUP.json").exists() or (RUN_ROOT / "trainer" / "results.csv").exists():
        raise FileExistsError("Existing execution artifacts: use a new Run ID for a retry")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    settings.update({key: False for key in ("wandb", "comet", "mlflow", "clearml", "neptune") if key in settings})
    # Offline laptop cannot fetch the extra checkpoint used by check_amp.
    trainer_module.check_amp = lambda model: torch.cuda.is_available()
    checks_module.check_pip_update_available = lambda: None
    cfg = official_weight_config(BASE_WEIGHTS)
    cfg.update(data=str(data), epochs=args.epochs, batch=args.batch, workers=args.workers,
               device=args.device, project=str(RUN_ROOT), name="trainer", save_dir=str(RUN_ROOT / "trainer"),
               exist_ok=True, seed=args.seed, save=True, save_period=1, val=True, plots=False,
               cache=False, patience=0, verbose=True, compile=False, resume=False)
    yolo = YOLO(str(BASE_WEIGHTS))
    try:
        yolo.train(trainer=ACDTrainer, **cfg)
        trainer = yolo.trainer
        receipt = {"status": "complete", "timestamp_utc": _utc_now(), "arm": args.arm,
                   "seed": args.seed, "epochs": args.epochs, "batch": trainer.batch_size,
                   "workers": args.workers, "runtime_ultralytics": ultralytics.__version__,
                   "data": str(data), "weight": str(BASE_WEIGHTS), "stats": STATS,
                   "audit": trainer._audit, "save_dir": str(trainer.save_dir),
                   "fixed_final_checkpoint": str(trainer.last),
                   "exact_fp32_evaluation_coefficients": str(RUN_ROOT / "coeff_final_ema.pt"),
                   "trainer_final_metrics": {key: float(value) for key, value in trainer.metrics.items()}}
        _write_json(RUN_ROOT / "TRAINING_COMPLETE.json", receipt)
        print("TRAINING_COMPLETE", json.dumps(receipt), flush=True)
    except BaseException as error:
        partial_artifact_error = None
        trainer = getattr(yolo, "trainer", None)
        if trainer is not None and not hasattr(trainer, "_audit") and all(
                hasattr(trainer, attribute) for attribute in (
                    "_frozen_initial", "_initial_coeff", "_gradient_stats", "_optimizer_step_stats",
                    "_finite_loss", "_input_stream_digest")):
            try:
                trainer._finish_audit(raise_on_failure=False,
                                      failure_reason=f"Execution terminated: {type(error).__name__}: {error}")
            except BaseException as artifact_error:
                # Preserve the original failure even if saving partial artifacts
                # encounters a separate filesystem or device failure.
                partial_artifact_error = f"{type(artifact_error).__name__}: {artifact_error}"
        failed = {"status": "failed", "timestamp_utc": _utc_now(), "arm": args.arm,
                  "exception_type": type(error).__name__, "exception": str(error),
                  "traceback": traceback.format_exc(), "stats": STATS,
                  "audit": getattr(trainer, "_audit", None),
                  "partial_artifact_save_error": partial_artifact_error}
        _write_json(RUN_ROOT / "TRAINING_FAILED.json", failed)
        print("TRAINING_FAILED", json.dumps(failed), flush=True)
        raise


if __name__ == "__main__":
    main()
