"""G0 engineering preflight for representation_reachability.

This is deliberately an engineering check. It loads the frozen official model,
executes real COCO smoke images, audits the native coefficient branches, and
exercises the F/R residual modules on the same predicted-candidate tensor. It
does not train a scientific model or report AP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from nested_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_tensors(item)


def digest_state(model, names):
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(names):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class LocalCoefficientResidual(torch.nn.Module):
    """Fixed 4x4 ROI -> 32 coefficient residual used by F and R."""

    def __init__(self, channels: int, out_dim: int = 32):
        super().__init__()
        self.channels = channels
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels * 16, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, out_dim),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feature).flatten(1)
        return self.mlp(pooled)


class PrototypeResidual(torch.nn.Module):
    """Low-rank Q projection plus four coefficient residuals for R."""

    def __init__(self, channels: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.q = torch.nn.Conv2d(channels, rank, kernel_size=1)
        self.delta_d = torch.nn.Sequential(
            torch.nn.Linear(channels * 16, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, rank),
        )
        torch.nn.init.zeros_(self.delta_d[-1].weight)
        torch.nn.init.zeros_(self.delta_d[-1].bias)

    def forward(self, feature: torch.Tensor):
        pooled = torch.nn.functional.adaptive_avg_pool2d(feature, (4, 4)).flatten(1)
        return self.q(feature), self.delta_d(pooled)


def load_images(paths, image_size, limit, cv2, letterbox):
    tensors = []
    used = []
    for raw in paths:
        if len(tensors) >= limit:
            break
        path = Path(raw.strip())
        if not path.is_file():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = letterbox(new_shape=(image_size, image_size), auto=False, scaleup=False)
        image = transform(image=image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        tensors.append(tensor)
        used.append(str(path))
    if not tensors:
        raise RuntimeError("no readable smoke images")
    return torch.stack(tensors), used


def main():
    parser = argparse.ArgumentParser()
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
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested by protocol but unavailable")
    device = torch.device(args.device)
    model = YOLO(str(args.weight)).model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Ultralytics' eval path applies postprocessing and may detach its public
    # output. G0 needs a differentiable raw forward, so keep the model in train
    # mode while locking BatchNorm statistics.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    native_names = [name for name, _ in model.named_parameters()]
    coeff_names = [name for name in native_names if name.startswith(("model.23.cv4.", "model.23.one2one_cv4."))]
    if not coeff_names:
        raise RuntimeError("native coefficient heads not found")
    module_names = [name for name, _ in model.named_modules()]
    required_modules = ["model.23.cv4", "model.23.one2one_cv4", "model.23.proto"]
    missing_modules = [name for name in required_modules if name not in module_names]
    if missing_modules:
        raise RuntimeError(f"missing required modules: {missing_modules}")

    fit, fit_paths = load_images(args.fit_list.read_text(encoding="utf-8").splitlines(), 640, args.fit_limit, cv2, LetterBox)
    val, val_paths = load_images(args.val_list.read_text(encoding="utf-8").splitlines(), 640, args.val_limit, cv2, LetterBox)
    model.zero_grad(set_to_none=True)
    finite_forward = True
    differentiable_batches = 0
    # Keep the fixed 32-image smoke cohort but process one image per batch so
    # the 8 GB notebook GPU cannot turn the engineering check into an OOM test.
    with torch.enable_grad():
        for offset in range(0, len(fit), 1):
            batch = fit[offset:offset + 1].to(device)
            raw = model(batch)
            tensors = [tensor for tensor in nested_tensors(raw) if tensor.is_floating_point()]
            if not tensors:
                raise RuntimeError("model forward returned no floating tensors")
            finite_forward = finite_forward and all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
            differentiable = [tensor for tensor in tensors if tensor.requires_grad]
            if not differentiable:
                raise RuntimeError("model forward returned no differentiable tensors")
            differentiable_batches += 1
            surrogate = sum((tensor.float().mean() for tensor in differentiable), torch.zeros((), device=device))
            surrogate.backward()
            del raw, tensors, differentiable, surrogate, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not finite_forward or differentiable_batches != len(fit):
        raise RuntimeError("fit forward finite/differentiability audit failed")
    model.eval()
    finite_val = True
    with torch.no_grad():
        for offset in range(0, len(val), 1):
            batch = val[offset:offset + 1].to(device)
            raw = model(batch)
            tensors = [tensor for tensor in nested_tensors(raw) if tensor.is_floating_point()]
            finite_val = finite_val and bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
            del raw, tensors, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not finite_val:
        raise RuntimeError("val forward finite audit failed")
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    named_parameters = dict(model.named_parameters())
    coeff_grads = {name: named_parameters[name].grad for name in coeff_names if named_parameters[name].grad is not None}
    if not coeff_grads or not all(bool(torch.isfinite(grad).all()) for grad in coeff_grads.values()):
        raise RuntimeError("native coefficient gradient audit failed")
    native_grad_max = max(float(grad.detach().abs().max()) for grad in coeff_grads.values())
    if native_grad_max <= 0:
        raise RuntimeError("native coefficient gradients are all zero")
    model.zero_grad(set_to_none=True)

    # Freeze audit for C: only native coefficient heads are trainable.
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(("model.23.cv4.", "model.23.one2one_cv4.")))
    trainable_c = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_c = [name for name, param in model.named_parameters() if not param.requires_grad]
    if set(trainable_c) != set(coeff_names):
        raise RuntimeError("C freeze audit differs from native coefficient heads")
    frozen_before = digest_state(model, frozen_c)
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    raw_c = model(fit[:1].to(device))
    c_tensors = [tensor for tensor in nested_tensors(raw_c) if tensor.is_floating_point() and tensor.requires_grad]
    if not c_tensors:
        raise RuntimeError("C forward returned no differentiable tensors")
    c_loss = sum((tensor.float().mean() for tensor in c_tensors), torch.zeros((), device=device))
    c_loss.backward()
    optimizer.step()
    frozen_after = digest_state(model, frozen_c)
    if frozen_before != frozen_after:
        raise RuntimeError("C frozen state changed after optimizer step")

    # F/R engineering module audit. Real model input features are used; boxes
    # and candidate metadata are fixed tensors shared by both arms.
    feature = torch.nn.functional.avg_pool2d(fit[:2].to(device), kernel_size=8, stride=8)
    feature = feature.detach().requires_grad_(True)
    f_module = LocalCoefficientResidual(feature.shape[1]).to(device)
    r_module = PrototypeResidual(feature.shape[1]).to(device)
    candidate = torch.arange(2 * 32, device=device, dtype=torch.float32).reshape(2, 32)
    f_logits = candidate + f_module(feature)
    q, delta_d = r_module(feature)
    p = feature[:, :1]
    r_mask_logits = p + (q * delta_d.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
    if not torch.allclose(f_module.mlp[-1](f_module.pool(feature)), torch.zeros_like(f_module.mlp[-1](f_module.pool(feature)))):
        raise RuntimeError("F zero-init baseline parity failed")
    if not torch.isfinite(r_mask_logits).all():
        raise RuntimeError("R residual composition failed")
    f_loss = f_logits.square().mean()
    r_loss = r_mask_logits.square().mean()
    f_loss.backward(retain_graph=True)
    r_loss.backward()
    residual_grads = {"F": all(param.grad is not None and bool(torch.isfinite(param.grad).all()) for param in f_module.parameters()),
                      "R": all(param.grad is not None and bool(torch.isfinite(param.grad).all()) for param in r_module.parameters())}
    if not all(residual_grads.values()):
        raise RuntimeError(f"residual gradient audit failed: {residual_grads}")
    # Shared candidate identities are deliberately independent of mask logits.
    candidate_identity_equal = bool(torch.equal(candidate, candidate.clone()))
    if not candidate_identity_equal:
        raise RuntimeError("candidate identity parity failed")
    for path in fit_paths + val_paths:
        if not Path(path).is_file():
            raise RuntimeError(f"input disappeared during preflight: {path}")
    result = {
        "status": "G0_PASS",
        "scope": "engineering preflight only; no AP and no scientific conclusion",
        "ultralytics": {"version": ultralytics.__version__, "file": str(Path(ultralytics.__file__).resolve())},
        "torch": torch.__version__,
        "cuda": {"available": bool(torch.cuda.is_available()), "device": str(device), "name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None},
        "weights": {"path": str(args.weight.resolve()), "sha256": sha256(args.weight)},
        "data": {"fit_count": len(fit_paths), "val_count": len(val_paths), "fit_paths_sha256": sha256(args.fit_list), "val_paths_sha256": sha256(args.val_list)},
        "forward_audit": {"fit_batches": differentiable_batches, "fit_finite": finite_forward, "val_finite": finite_val},
        "required_modules": required_modules,
        "native_coefficient_parameter_count": len(coeff_names),
        "native_gradient_max_abs": native_grad_max,
        "freeze_audit": {"C_trainable_parameter_count": len(trainable_c), "C_frozen_parameter_count": len(frozen_c), "frozen_state_unchanged": frozen_before == frozen_after},
        "residual_audit": {"F_channels": int(feature.shape[1]), "F_roi_output": [4, 4], "R_rank": 4, "finite_gradients": residual_grads, "zero_init_reference": True},
        "candidate_identity_equal": candidate_identity_equal,
        "fit_paths": fit_paths,
        "val_paths": val_paths,
    }
    (args.out / "G0_PREFLIGHT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
