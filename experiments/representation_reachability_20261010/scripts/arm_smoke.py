"""Explicit A/C/F/R engineering smoke runs; no AP or formal training."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from preflight import LocalCoefficientResidual, PrototypeResidual, digest_state, load_images, nested_tensors, sha256


def batch_forward(model, images, device, train_mode):
    finite = True
    differentiable = 0
    for offset in range(len(images)):
        batch = images[offset:offset + 1].to(device)
        raw = model(batch)
        tensors = [tensor for tensor in nested_tensors(raw) if tensor.is_floating_point()]
        finite = finite and bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        differentiable += sum(tensor.requires_grad for tensor in tensors)
        del raw, tensors, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return finite, differentiable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "C", "F", "R"], required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--fit-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-limit", type=int, default=32)
    parser.add_argument("--val-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.vendor.resolve()))
    import cv2
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    if ultralytics.__version__ != "8.4.100":
        raise RuntimeError(f"expected Ultralytics 8.4.100, got {ultralytics.__version__} from {ultralytics.__file__}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = YOLO(str(args.weight)).model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    coeff_prefixes = ("model.23.cv4.", "model.23.one2one_cv4.")
    coeff_names = [name for name, _ in model.named_parameters() if name.startswith(coeff_prefixes)]
    fit, fit_paths = load_images(args.fit_list.read_text(encoding="utf-8").splitlines(), 640, args.fit_limit, cv2, LetterBox)
    val, val_paths = load_images(args.val_list.read_text(encoding="utf-8").splitlines(), 640, args.val_limit, cv2, LetterBox)
    model.zero_grad(set_to_none=True)
    result = {"status": "G0_PASS", "arm": args.arm, "scope": "engineering smoke only; no AP or formal training", "fit_count": len(fit_paths), "val_count": len(val_paths), "weights_sha256": sha256(args.weight), "ultralytics": {"version": ultralytics.__version__, "file": str(Path(ultralytics.__file__).resolve())}}

    if args.arm == "A":
        model.eval()
        with torch.no_grad():
            fit_finite, _ = batch_forward(model, fit, device, False)
            val_finite, _ = batch_forward(model, val, device, False)
        result["forward"] = {"fit_finite": fit_finite, "val_finite": val_finite, "trainable_numel": 0}
        if not fit_finite or not val_finite:
            raise RuntimeError("A finite forward audit failed")
    elif args.arm == "C":
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(coeff_prefixes))
        frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
        frozen_before = digest_state(model, frozen_names)
        batch = fit[:1].to(device)
        raw = model(batch)
        tensors = [tensor for tensor in nested_tensors(raw) if tensor.is_floating_point() and tensor.requires_grad]
        if not tensors:
            raise RuntimeError("C returned no differentiable tensors")
        loss = sum((tensor.float().mean() for tensor in tensors), torch.zeros((), device=device))
        loss.backward()
        grads = [parameter.grad for name, parameter in model.named_parameters() if name in coeff_names and parameter.grad is not None]
        if not grads or not all(bool(torch.isfinite(grad).all()) for grad in grads) or max(float(grad.abs().max()) for grad in grads) <= 0:
            raise RuntimeError("C coefficient gradient audit failed")
        optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
        optimizer.step()
        frozen_after = digest_state(model, frozen_names)
        result["freeze_audit"] = {"trainable_numel": sum(parameter.numel() for name, parameter in model.named_parameters() if name in coeff_names), "frozen_unchanged": frozen_before == frozen_after, "gradient_max_abs": max(float(grad.abs().max()) for grad in grads)}
        if frozen_before != frozen_after:
            raise RuntimeError("C frozen state changed")
    else:
        model.eval()
        with torch.no_grad():
            fit_finite, _ = batch_forward(model, fit[:1], device, False)
        if not fit_finite:
            raise RuntimeError(f"{args.arm} native forward failed")
        feature = torch.nn.functional.avg_pool2d(fit[:2].to(device), kernel_size=8, stride=8)
        feature = feature.detach().requires_grad_(True)
        candidate = torch.arange(2 * 32, device=device, dtype=torch.float32).reshape(2, 32)
        if args.arm == "F":
            module = LocalCoefficientResidual(feature.shape[1]).to(device)
            delta = module(feature)
            logits = candidate + delta
            zero_init = bool(torch.allclose(delta, torch.zeros_like(delta)))
        else:
            module = PrototypeResidual(feature.shape[1]).to(device)
            q, delta_d = module(feature)
            logits = feature[:, :1] + (q * delta_d.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
            zero_init = bool(torch.allclose(delta_d, torch.zeros_like(delta_d)))
        loss = logits.square().mean()
        loss.backward()
        grads_ok = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())
        result["residual_audit"] = {"finite_gradients": grads_ok, "zero_init_reference": zero_init, "candidate_identity_equal": bool(torch.equal(candidate, candidate.clone())), "trainable_numel": sum(parameter.numel() for parameter in module.parameters())}
        if not grads_ok or not zero_init or not result["residual_audit"]["candidate_identity_equal"]:
            raise RuntimeError(f"{args.arm} residual audit failed")
    (args.out / "ARM_SMOKE.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
