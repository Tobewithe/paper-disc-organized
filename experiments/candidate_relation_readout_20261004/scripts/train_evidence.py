"""Train only the new box-evidence readout on the immutable frozen cohort.

All detector/neck/prototype/candidate tensors come from the previous cache.
The additional cache contains only frozen h0 and projection operators. There
is no native model in the optimizer, no validation-based selection, no teacher,
and no auxiliary objective. Run this script only on the authorized GPU host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
import ultralytics
from ultralytics.utils.loss import v8SegmentationLoss

from evidence_head import EvidenceReadout
from runtime_utils import dump, load, setup, read_image, gpu_image, loss_and_grad, lr_factor


FIXED_SETTINGS = {
    "seed": 0, "epochs": 12, "microbatch_images": 2,
    "effective_batch_images": 16, "learning_rate": 0.0003,
    "weight_decay": 0.0001, "warmup_epochs": 1,
    "eta_ratio": 0.1, "gradient_clip_norm": 10,
}
OPERATOR_SHAPES = {"h0": (64,), "A": (256, 32), "K": (32, 256), "G": (32, 32)}
RECORDED_EPOCHS = (1, 4, 8, 12)


def operator_fingerprint(cfg):
    fields = {k: cfg[k] for k in ("cache", "weights", "projection_lambda", "solver_jitter", "roi_side")}
    fields["schema"] = "box_evidence_full_response_v1"
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def resolve_config(config):
    cfg = dict(config)
    for key, expected in FIXED_SETTINGS.items():
        cfg.setdefault(key, expected)
        if cfg[key] != expected:
            raise ValueError(f"Frozen training protocol requires {key}={expected}, got {cfg[key]}")
    cfg.setdefault("arms", ["S", "D"])
    if set(cfg["arms"]) != {"S", "D"} or len(cfg["arms"]) != 2:
        raise ValueError("Both S and D must be registered exactly once")
    if not cfg.get("cache") or not cfg.get("operator_cache"):
        raise ValueError("Both original cache and additional operator_cache are required")
    return cfg


def read_training_image(cfg, iid):
    """Join immutable per-image caches without copying the original F/proto/GT."""
    x = read_image(cfg, iid)
    op = load(Path(cfg["operator_cache"]) / "images" / f"{iid:012d}.pt")
    n = len(x["raw_ids"])
    if int(x["image_id"]) != int(iid) or int(op.get("image_id", iid)) != int(iid):
        raise ValueError(f"Image identity mismatch for {iid}")
    if "raw_ids" not in op or not torch.equal(op["raw_ids"], x["raw_ids"]):
        raise ValueError(f"Projection operator candidate order mismatch for {iid}")
    if op.get("source_fingerprint") != x["fingerprint"] or op.get("fingerprint") != operator_fingerprint(cfg):
        raise ValueError(f"Projection operator source/configuration identity mismatch for {iid}")
    if len(x["rows"]) != n or x["c0"].shape != (n, 32):
        raise ValueError(f"Original cached candidate shape mismatch for {iid}")
    if "c0" in op and not torch.equal(op["c0"], x["c0"]):
        raise ValueError(f"Projection operator baseline coefficient mismatch for {iid}")
    for key, tail in OPERATOR_SHAPES.items():
        tensor = op[key]
        if tensor.shape != (n, *tail) or tensor.dtype != torch.float32:
            raise ValueError(f"{iid}: {key} must be FP32 {(n, *tail)}, got {tensor.shape}/{tensor.dtype}")
        if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{iid}: cached {key} must be frozen and finite")
    if op["valid"].shape != (n,) or op["valid"].dtype != torch.bool:
        raise ValueError(f"{iid}: valid must be bool [N]")
    if abs(float(x["segmentation_gain"]) - 9.83241) > 1e-5:
        raise ValueError(f"{iid}: official segmentation gain changed")
    if n and (len(x["F"]) != 3 or any(v.dtype != torch.float32 for v in x["F"])):
        raise ValueError(f"{iid}: all three frozen neck maps must remain FP32")
    x["_operator"] = op
    return x


def feature_channels(image):
    return [int(f.shape[0]) for f in image["F"]]


def feed(images):
    features = [torch.stack([x["F"][level] for x in images]).cuda().detach() for level in range(3)]
    selected = []
    for x in images:
        row = {key: x[key].cuda().detach() for key in ("raw_ids", "boxes", "c0")}
        row.update({key: x["_operator"][key].cuda().detach()
                    for key in (*OPERATOR_SHAPES, "valid")})
        selected.append(row)
    return features, selected


def optimizer_for(model, cfg):
    # Both arms use identical grouping. Bias and normalization vectors follow
    # the previous training convention and are exempt from weight decay.
    groups = []
    for decay in (True, False):
        params = [p for p in model.parameters() if p.requires_grad and (p.ndim > 1) == decay]
        if params:
            groups.append(dict(params=params, lr=cfg["learning_rate"],
                               initial_lr=cfg["learning_rate"],
                               weight_decay=cfg["weight_decay"] if decay else 0.))
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def parameter_counts(model):
    if hasattr(model, "parameter_counts"):
        return model.parameter_counts()
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def make_model(channels, mode, cfg):
    setup(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = EvidenceReadout(channels, mode, cfg).cuda().float().train()
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("The new evidence model must not own native BatchNorm state")
    return model


def backwards(coefficients, gradients):
    """Only new, differentiable outputs may receive the exact external dL/dc."""
    pairs = [(c, g) for c, g in zip(coefficients, gradients) if c.requires_grad]
    if pairs:
        torch.autograd.backward([c for c, _ in pairs], [g for _, g in pairs])
    return len(pairs)


def checked_loss(x, coefficients, normalizer):
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("Non-finite predicted coefficients")
    value, gradient = loss_and_grad(gpu_image(x), coefficients, normalizer)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Non-finite official BCE coefficient gradient")
    return value, gradient


def new_stats():
    return dict(candidates=0, valid_candidates=0, coefficient_delta_l2_sum=0.,
                coefficient_delta_abs_max=0., roi_logit_delta_rms_sum=0.)


@torch.no_grad()
def add_stats(stats, coefficients, selected):
    for c, row in zip(coefficients, selected):
        dc = c.detach() - row["c0"]
        if not len(dc):
            continue
        dz = torch.bmm(row["A"], dc.unsqueeze(-1)).squeeze(-1)
        stats["candidates"] += len(dc)
        stats["valid_candidates"] += int(row["valid"].sum())
        stats["coefficient_delta_l2_sum"] += float(dc.norm(dim=1).sum())
        stats["coefficient_delta_abs_max"] = max(stats["coefficient_delta_abs_max"], float(dc.abs().max()))
        stats["roi_logit_delta_rms_sum"] += float(dz.square().mean(1).sqrt().sum())


def finish_stats(stats):
    count = max(1, stats["candidates"])
    return dict(candidates=stats["candidates"], valid_candidates=stats["valid_candidates"],
                invalid_candidates=stats["candidates"] - stats["valid_candidates"],
                coefficient_delta_l2_mean=stats["coefficient_delta_l2_sum"] / count,
                coefficient_delta_abs_max=stats["coefficient_delta_abs_max"],
                roi_logit_delta_rms_mean=stats["roi_logit_delta_rms_sum"] / count,
                logit_stat_grid="frozen A on 16x16 predicted-box grid; diagnostic only")


def save_checkpoint(path, model, optimizer, epoch, mode, cfg, history, channels):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(mode=mode, epoch=epoch, state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(), config=cfg, history=history,
                    feature_channels=channels, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    checkpoint_rule="fixed final epoch; intermediate epochs are records only"), temp)
    temp.replace(path)


def choose_smoke_images(cfg, index):
    """First eight positive fit images, plus earliest images covering missing levels."""
    chosen, covered, scanned = [], set(), 0
    for row in index["fit"]:
        if not row["n"]:
            continue
        x = read_training_image(cfg, row["image_id"])
        scanned += 1
        levels = set(int(v) for v in x["levels"][x["_operator"]["valid"]].tolist())
        if len(chosen) < 8 or levels - covered:
            chosen.append(x)
            covered.update(levels)
        if len(chosen) >= 8 and covered == {0, 1, 2}:
            break
    if len(chosen) < 8 or covered != {0, 1, 2}:
        raise RuntimeError(f"Smoke needs >=8 fit images with valid P3/P4/P5 coverage; got {len(chosen)}, {covered}")
    return chosen, scanned


def direct_bce_check(x, c, gradient, normalizer):
    gx = gpu_image(x)
    take = min(2, len(c))
    leaf = c[:take].detach().requires_grad_(True)
    proto = F.interpolate(gx["proto"][None].float(), (640, 640), mode="bilinear", align_corners=False)[0]
    gt = (gx["masks"][None] == (gx["owners"][:take] + 1)[:, None, None]).float()
    boxes = gx["target_boxes"][:take]
    area = ((boxes[:, 2:] - boxes[:, :2]) / 640).prod(1)
    direct = v8SegmentationLoss.single_mask_loss(gt, leaf, proto, boxes, area)
    reference = torch.autograd.grad(direct * float(gx["segmentation_gain"]) / normalizer, leaf)[0]
    torch.testing.assert_close(reference, gradient[:take], atol=3e-5, rtol=3e-5)
    return float((reference - gradient[:take]).abs().max())


def independent_projection_check(selection, gradient):
    # The 256-bin evidence gradient must be K^T dL/dc. This does not optimize
    # K, use GT as a feature, or substitute a different training objective.
    r = torch.zeros((len(gradient), 256), device=gradient.device, requires_grad=True)
    c = selection["c0"] + torch.bmm(selection["K"], r.unsqueeze(-1)).squeeze(-1)
    actual = torch.autograd.grad(c, r, grad_outputs=gradient)[0]
    expected = torch.bmm(selection["K"].transpose(1, 2), gradient.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    return float((actual - expected).abs().max())


def expected_evidence_gradient(selection, gradient):
    boxes = selection["boxes"]
    clamped = boxes.clamp(0, 640)
    valid = selection["valid"] & torch.isfinite(boxes).all(1)
    valid &= ((clamped[:, 2:] - clamped[:, :2]) > 0).all(1)
    expected = torch.zeros((len(gradient), 256), device=gradient.device, dtype=gradient.dtype)
    expected[valid] = torch.bmm(selection["K"][valid].transpose(1, 2), gradient[valid].unsqueeze(-1)).squeeze(-1)
    return expected


def smoke(cfg, out, index, deadline=0.):
    images, scanned = choose_smoke_images(cfg, index)
    channels = feature_channels(images[0])
    total = sum(len(x["raw_ids"]) for x in images)
    report = dict(ultralytics_import=ultralytics.__version__, source=str(ultralytics.__file__),
                  image_ids=[x["image_id"] for x in images], fit_images_scanned=scanned,
                  feature_channels=channels, candidates=total, arms={},
                  scope="GPU code verification only; no formal weights; models reset for every arm and formal run")
    shared_initial = None
    for mode in cfg["arms"]:
        if deadline and time.time() > deadline:
            raise TimeoutError("Smoke deadline reached")
        model = make_model(channels, mode, cfg)
        optimizer = optimizer_for(model, cfg)
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        common = {n: v for n, v in before.items() if not n.startswith("direct_projection.")}
        if shared_initial is None:
            shared_initial = common
        else:
            if common.keys() != shared_initial.keys():
                raise AssertionError("S/D shared encoder/decoder parameter names differ")
            for name, parameter in common.items():
                torch.testing.assert_close(parameter, shared_initial[name], atol=0, rtol=0)
        grads, step_grads, losses, stats = {}, [], [], new_stats()
        initial_error = direct_error = projection_error = actual_chain_error = 0.
        chain_checks = 0
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for lo in range(0, len(images), cfg["microbatch_images"]):
                batch = images[lo:lo + cfg["microbatch_images"]]
                fs, selected = feed(batch)
                frozen_k = [s["K"].clone() for s in selected]
                cs, residuals = model.forward_with_evidence(fs, selected)
                for r in residuals:
                    if mode == "S" and r.requires_grad:
                        r.retain_grad()
                gs = []
                for x, c, selection in zip(batch, cs, selected):
                    if step == 0:
                        error = float((c.detach() - selection["c0"]).abs().max())
                        initial_error = max(initial_error, error)
                        torch.testing.assert_close(c.detach(), selection["c0"], atol=0, rtol=0)
                    value, g = checked_loss(x, c, total)
                    gs.append(g)
                    total_loss += value
                    if step == 0 and lo == 0:
                        direct_error = max(direct_error, direct_bce_check(x, c, g, total))
                        if mode == "S":
                            projection_error = max(projection_error, independent_projection_check(selection, g))
                backwards(cs, gs)
                if mode == "S":
                    for selection, residual, gradient in zip(selected, residuals, gs):
                        if residual.grad is not None:
                            expected = expected_evidence_gradient(selection, gradient)
                            torch.testing.assert_close(residual.grad, expected, atol=3e-6, rtol=3e-5)
                            actual_chain_error = max(actual_chain_error, float((residual.grad - expected).abs().max()))
                            chain_checks += 1
                for selection, original_k in zip(selected, frozen_k):
                    torch.testing.assert_close(selection["K"], original_k, atol=0, rtol=0)
                    if any(v.requires_grad for v in selection.values() if torch.is_tensor(v)):
                        raise AssertionError("Frozen cached input acquired an autograd dependency")
                add_stats(stats, cs, selected)
            current = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f"{mode}: non-finite gradient for {name}")
                    current[name] = float(parameter.grad.norm())
                    grads[name] = max(grads.get(name, 0.), current[name])
            step_grads.append(current)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(total_loss / total)
        if mode == "S" and not chain_checks:
            raise AssertionError("Actual S model evidence tensors were not on the coefficient autograd path")
        for group, prefixes in {
            "encoder": ("encoder_adapters.", "encoder_fusion."),
            "query": ("query_mlp.", "query_film."),
        }.items():
            if not any(v > 0 for n, v in step_grads[1].items() if n.startswith(prefixes)):
                raise AssertionError(f"{mode}: {group} has no finite nonzero gradient on the second step")
        changed = {n: float((p.detach().cpu() - before[n]).norm()) for n, p in model.named_parameters()}
        report["arms"][mode] = dict(initial_coefficient_max_error=initial_error,
            parameter_counts=parameter_counts(model), losses=losses, gradient_norms=grads,
            gradient_norms_by_step=step_grads, parameter_changes=changed,
            official_bce_gradient_max_error=direct_error,
            independent_projection_gradient_max_error=projection_error if mode == "S" else None,
            actual_projection_gradient_max_error=actual_chain_error if mode == "S" else None,
            actual_projection_chain_checks=chain_checks, K_unchanged=True,
            frozen_inputs_detached=True, native_model_not_loaded=True, diagnostics=finish_stats(stats))
        dump(out / "SMOKE.json", report)
        del optimizer, model
        torch.cuda.empty_cache()
    report["shared_encoder_decoder_initialization_identical"] = True
    dump(out / "SMOKE.json", report)
    dump(out / "COMPLETE.json", dict(passed=True, kind="smoke", formal_training_steps=0,
                                     trained_smoke_weights_saved=False))


def train(cfg, out, index, mode, deadline):
    items = [r for r in index["fit"] if r["n"] > 0]
    if not items:
        raise ValueError("No positive training images")
    first = read_training_image(cfg, items[0]["image_id"])
    channels = feature_channels(first)
    del first
    model = make_model(channels, mode, cfg)
    optimizer = optimizer_for(model, cfg)
    dump(out / "MODEL.json", dict(mode=mode, counts=parameter_counts(model), feature_channels=channels,
        trainable=[n for n, p in model.named_parameters() if p.requires_grad],
        native_parameters_in_optimizer=0, original_detector="frozen cached tensors; not instantiated",
        objective="official full-640 GT-cropped area-normalized BCE only; gain 9.83241",
        diagnostic_regularization=False, actual_ultralytics=ultralytics.__version__, source=str(ultralytics.__file__)))
    dump(out / "COHORT.json", dict(planned_images=len(index["fit"]), effective_images=len(items),
        candidates=sum(r["n"] for r in items),
        no_positive_images=[r["image_id"] for r in index["fit"] if not r["n"]],
        candidate_filtering="none; invalid operators retain c0 and all original loss denominator entries"))
    dump(out / "RESOLVED_CONFIG.json", cfg)
    start_epoch, history, last = 0, [], out / "last.pt"
    if last.exists():
        ck = load(last)
        if ck["mode"] != mode or ck["config"] != cfg or ck["feature_channels"] != channels:
            raise ValueError("Resume checkpoint identity/configuration mismatch")
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        # load_state_dict casts moment tensors to their parameter device and
        # preserves the CPU step counter used by non-capturable AdamW.
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start_epoch, history = ck["epoch"], ck["history"]
    started = time.monotonic()
    for epoch in range(start_epoch, cfg["epochs"]):
        order = list(items)
        random.Random(cfg["seed"] + epoch).shuffle(order)
        order_hash = hashlib.sha256(json.dumps([int(r["image_id"]) for r in order], separators=(",", ":")).encode()).hexdigest()
        epoch_start, total_loss, seen = time.monotonic(), 0., 0
        groups, stats = math.ceil(len(order) / cfg["effective_batch_images"]), new_stats()
        for step, lo in enumerate(range(0, len(order), cfg["effective_batch_images"])):
            if deadline and time.time() > deadline:
                dump(out / "BUDGET_STOP.json", dict(epoch_completed=epoch, mid_epoch_step=step,
                    note="No method conclusion. Resume last completed epoch; partial epoch is not epoch12."))
                raise TimeoutError("Training resource deadline reached")
            rows = order[lo:lo + cfg["effective_batch_images"]]
            denominator = sum(r["n"] for r in rows)
            factor = lr_factor(epoch + (step + 1) / groups, cfg)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(rows), cfg["microbatch_images"]):
                chosen = rows[j:j + cfg["microbatch_images"]]
                batch = [read_training_image(cfg, r["image_id"]) for r in chosen]
                if any(len(x["raw_ids"]) != r["n"] for x, r in zip(batch, chosen)):
                    raise ValueError("Frozen fit index candidate count changed")
                fs, selection = feed(batch)
                cs = model(fs, selection)
                gs = []
                for x, c in zip(batch, cs):
                    value, g = checked_loss(x, c, denominator)
                    gs.append(g)
                    total_loss += value
                    seen += len(x["raw_ids"])
                backwards(cs, gs)
                add_stats(stats, cs, selection)
                del batch, fs, cs, gs, selection
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
            optimizer.step()
            if step % 25 == 0 or step + 1 == groups:
                state = dict(mode=mode, epoch=epoch + 1, epochs=cfg["epochs"], step=step + 1, steps=groups,
                    trajectory_bce=total_loss / max(1, seen), grad_norm_before_clip=grad,
                    learning_rate=cfg["learning_rate"] * factor,
                    elapsed_s=time.monotonic() - started, epoch_elapsed_s=time.monotonic() - epoch_start,
                    diagnostics=finish_stats(stats))
                dump(out / "PROGRESS.json", state)
                print(json.dumps(state), flush=True)
        history.append(dict(epoch=epoch + 1, trajectory_bce=total_loss / seen, candidates=seen,
                            seconds=time.monotonic() - epoch_start, image_order_sha256=order_hash,
                            diagnostics=finish_stats(stats)))
        save_checkpoint(last, model, optimizer, epoch + 1, mode, cfg, history, channels)
        if epoch + 1 in RECORDED_EPOCHS:
            save_checkpoint(out / f"epoch{epoch + 1:02d}.pt", model, optimizer, epoch + 1, mode, cfg, history, channels)
        dump(out / "HISTORY.json", history)
    save_checkpoint(out / "final.pt", model, optimizer, cfg["epochs"], mode, cfg, history, channels)
    dump(out / "COMPLETE.json", dict(completed=True, kind="training", mode=mode, epochs=cfg["epochs"],
                                     checkpoint="final.pt", checkpoint_rule="fixed epoch12; no dev/val selection",
                                     elapsed_s=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index")
    parser.add_argument("--mode", choices=("S", "D"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--deadline", type=float, default=0., help="Absolute Unix timestamp; 0 disables deadline")
    args = parser.parse_args()
    cfg = resolve_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.index or str(Path(cfg["cache"]) / "INDEX.json")).read_text(encoding="utf-8-sig"))
    if not torch.cuda.is_available():
        raise RuntimeError("Execute on the authorized GPU host; CPU model training is not permitted")
    try:
        if args.smoke:
            smoke(cfg, out, index, args.deadline)
        else:
            if args.mode is None:
                parser.error("--mode is required for formal training")
            if not (Path(cfg["cache"]) / "COMPLETE.json").exists():
                raise RuntimeError("Original frozen cohort cache must be complete before training")
            if not (Path(cfg["operator_cache"]) / "COMPLETE.json").exists():
                raise RuntimeError("Projection operator cache must be complete before training")
            complete = json.loads((Path(cfg["operator_cache"]) / "COMPLETE.json").read_text(encoding="utf-8-sig"))
            if complete.get("smoke") or complete.get("fingerprint") != operator_fingerprint(cfg):
                raise RuntimeError("Projection completion marker must match the full current operator cache")
            train(cfg, out, index, args.mode, args.deadline)
    except Exception as exc:
        dump(out / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc),
                                       mode=args.mode, smoke=args.smoke))
        raise


if __name__ == "__main__":
    main()

