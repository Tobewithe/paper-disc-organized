"""GPU-host-only implementation checks for the fixed U/Q/P protocol.

Uses 1--2 immutable transferred fit assets by default. Each arm takes exactly
two disposable optimizer steps to test gradient connectivity after zero-init.
No smoke weights are saved or reused for formal training. Full official mask
BCE remains the supervision; selected cells only define evidence readout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import ultralytics
from ultralytics.utils.loss import v8SegmentationLoss

from asset_runtime import FrozenReplay, load_asset, resolve_runtime_config
from point_head import JointPointCoefficientReadout, selected_cell_rois
from point_selectors import build_selection_operators, UNIFORM_INDICES
from runtime_utils import bn_state, dump, gpu_image, loss_and_grad, setup, verify_bn

ARMS = ("U", "Q", "P")
TOLERANCES = dict(replay_atol=3e-5, replay_rtol=3e-5,
                  roi_atol=3e-5, roi_rtol=3e-5,
                  loss_atol=3e-5, loss_rtol=3e-5,
                  chain_atol=3e-6, chain_rtol=3e-5,
                  operator_relative_fp64=1e-8, operator_relative_fp32=1e-4)


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def state_digest(module):
    return {name: tensor_digest(value) for name, value in module.state_dict().items()}


def tensor_tree_digest(value):
    if torch.is_tensor(value):
        return tensor_digest(value)
    if isinstance(value, dict):
        return {key: tensor_tree_digest(item) for key, item in value.items()
                if torch.is_tensor(item) or isinstance(item, (list, tuple, dict))}
    if isinstance(value, (list, tuple)):
        return [tensor_tree_digest(item) for item in value]
    return None


def finite_grad_norms(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            result[name] = float(parameter.grad.norm())
    return result


def optimizer_for(model, cfg):
    grouped = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped.setdefault((name.startswith("native_cv4."), parameter.ndim > 1), []).append(parameter)
    groups = [dict(params=parameters,
                   lr=float(cfg["branch_lr"] if native else cfg["new_lr"]),
                   weight_decay=float(cfg["weight_decay"]) if decay else 0.)
              for (native, decay), parameters in grouped.items()]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def selector_checks(device):
    # Identical rows make all uncertainty and all greedy scores exact ties.
    a = torch.zeros((1, 256, 32), device=device, dtype=torch.float64)
    a[:, :, 0] = 1.0
    g = torch.eye(32, device=device, dtype=torch.float64).unsqueeze(0)
    c0 = torch.zeros((1, 32), device=device, dtype=torch.float64)
    result = {}
    for mode in ARMS:
        row = build_selection_operators(a, g, c0, mode)
        expected = torch.tensor(UNIFORM_INDICES if mode == "U" else tuple(range(64)), device=device)
        torch.testing.assert_close(row["indices"][0], expected, atol=0, rtol=0)
        assert bool(row["valid"].all()), f"{mode} tie-control unexpectedly invalid"
        assert row["indices"][0].unique().numel() == 64
        residual = float(row["diagnostics"]["relative_residual_fp64"].max())
        assert residual <= TOLERANCES["operator_relative_fp64"]
        result[mode] = dict(unique_cells=64, tie_order_correct=True, relative_residual_fp64=residual)
    return result


def make_selections(images, mode, device):
    selections, summaries = [], []
    for image in images:
        op = image.get("_operator", image.get("transfer_operator"))
        if op is None:
            raise KeyError("Transferred asset has no original operator data")
        a = torch.as_tensor(op.get("A_full", op.get("A")), device=device).detach()
        g = torch.as_tensor(op["G"], device=device).detach()
        c0 = torch.as_tensor(image["c0"], device=device).detach()
        built = build_selection_operators(a, g, c0, mode)
        valid = built["valid"]
        if "valid" in op:
            valid = valid & torch.as_tensor(op["valid"], device=device).bool()
        if not bool(valid.all()):
            raise AssertionError(f"{mode}/{image['image_id']}: invalid operator in smoke; inspect instead of filtering")
        ids = built["indices"]
        assert bool((ids.sort(1).values[:, 1:] != ids.sort(1).values[:, :-1]).all())
        diag = built["diagnostics"]
        error64 = float(diag["relative_residual_fp64"].max())
        error32 = float(diag["relative_residual_fp32"].max())
        assert error64 <= TOLERANCES["operator_relative_fp64"], (mode, "FP64 K", error64)
        assert error32 <= TOLERANCES["operator_relative_fp32"], (mode, "FP32 K", error32)
        selection = dict(raw_ids=torch.as_tensor(image["raw_ids"], device=device).long().detach(),
                         boxes=torch.as_tensor(image["boxes"], device=device).float().detach(),
                         h0=torch.as_tensor(op["h0"], device=device).float().detach(),
                         c0=c0.float(), A_full=a.float(), indices=ids.detach(),
                         K=built["K"].float().detach(), valid=valid.detach())
        assert all(not item.requires_grad for item in selection.values())
        selections.append(selection)
        summaries.append(dict(image_id=int(image["image_id"]), candidates=len(c0), unique_cells=64,
                              relative_residual_fp64=error64, relative_residual_fp32=error32,
                              indices_sha256=tensor_digest(ids)["sha256"],
                              mean_selected_uncertainty=float(diag["mean_selected_uncertainty"].mean())))
    return selections, summaries


@torch.no_grad()
def roi_check(model, features, selections):
    encoded = model.evidence._encode(features, len(selections))
    scale = encoded.shape[-2] / 640.0
    maximum = 0.0
    for image_index, selected in enumerate(selections):
        boxes, indices = selected["boxes"].clamp(0, 640), selected["indices"]
        full_rois = torch.cat((boxes.new_zeros((len(boxes), 1)), boxes), dim=1)
        full = roi_align(encoded[image_index:image_index + 1], full_rois,
                         output_size=(16, 16), spatial_scale=scale, sampling_ratio=2, aligned=True)
        expected = full.flatten(2).transpose(1, 2).gather(1, indices[:, :, None].expand(-1, -1, 64))
        cells = selected_cell_rois(boxes, indices)
        actual = roi_align(encoded[image_index:image_index + 1], cells,
                           output_size=(1, 1), spatial_scale=scale, sampling_ratio=2, aligned=True)
        actual = actual.reshape(len(boxes), 64, 64)
        torch.testing.assert_close(actual, expected, atol=TOLERANCES["roi_atol"], rtol=TOLERANCES["roi_rtol"])
        maximum = max(maximum, float((actual - expected).abs().max()))
    return maximum


def direct_reference_check(image, coefficient, gradient, denominator, actual_value):
    x = gpu_image(image)
    leaf = coefficient.detach().requires_grad_(True)
    prototype = F.interpolate(x["proto"][None].float(), (640, 640), mode="bilinear", align_corners=False)[0]
    target = (x["masks"][None] == (x["owners"] + 1)[:, None, None]).float()
    boxes = x["target_boxes"]
    area = ((boxes[:, 2:] - boxes[:, :2]) / 640).prod(1)
    # This is the full original GT support, not the 64 selected ROI cells.
    direct = v8SegmentationLoss.single_mask_loss(target, leaf, prototype, boxes, area)
    direct = direct * float(x["segmentation_gain"])
    expected = torch.autograd.grad(direct / denominator, leaf)[0]
    torch.testing.assert_close(expected, gradient, atol=TOLERANCES["loss_atol"], rtol=TOLERANCES["loss_rtol"])
    torch.testing.assert_close(direct.detach(), direct.new_tensor(actual_value),
                               atol=TOLERANCES["loss_atol"], rtol=TOLERANCES["loss_rtol"])
    return dict(loss_absolute_error=abs(float(direct.detach()) - actual_value),
                coefficient_gradient_max_error=float((expected - gradient).abs().max()))


def isolated_projection_check(selections):
    maximum = 0.0
    for selected in selections:
        k = selected["K"].double()
        r = torch.linspace(-0.8, 0.8, 64, device=k.device, dtype=k.dtype).expand(len(k), -1).clone().requires_grad_(True)
        gradient = torch.linspace(-0.4, 0.7, 32, device=k.device, dtype=k.dtype).expand(len(k), -1)
        coefficient = torch.bmm(k, r.unsqueeze(2)).squeeze(2)
        actual = torch.autograd.grad(coefficient, r, gradient)[0]
        expected = torch.bmm(k.transpose(1, 2), gradient.unsqueeze(2)).squeeze(2)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        maximum = max(maximum, float((actual - expected).abs().max()))
    return maximum


def image_ids_for_smoke(cfg, explicit, count):
    if explicit:
        if not 1 <= len(explicit) <= 2 or len(set(explicit)) != len(explicit):
            raise ValueError("Smoke uses one or two distinct asset images")
        return explicit
    index = json.loads((Path(cfg["assets"]) / "INDEX.json").read_text(encoding="utf-8-sig"))
    available = [int(row["image_id"]) for row in index["fit"]
                 if row["n"] > 0 and (Path(cfg["assets"]) / "images" / f"{int(row['image_id']):012d}.pt.gz").is_file()]
    if len(available) < count:
        raise RuntimeError(f"Need {count} transferred positive fit assets; have {len(available)}")
    return available[:count]


def run_smoke(cfg, out, image_ids, deadline=0.0):
    setup(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = dict(kind="point_implementation_smoke", formal_training_steps=0,
                  disposable_steps_per_arm=2, smoke_weights_saved=False,
                  actual_ultralytics=ultralytics.__version__, ultralytics_source=str(ultralytics.__file__),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(), image_ids=image_ids, tolerances=TOLERANCES,
                  official_loss="full640 GT-box support, original owner/area/gain, candidate equal weighting",
                  selector_synthetic=selector_checks(torch.device("cuda")), arms={})
    dump(out / "SMOKE.json", report)
    images = [load_asset(cfg, iid, verify=True) for iid in image_ids]
    immutable_assets = tensor_tree_digest(images)
    replay = FrozenReplay(cfg, device="cuda")
    frozen_source = state_digest(replay.source)
    features = replay.replay(images)
    feature_hashes = tensor_tree_digest(features)
    channels = [int(feature.shape[1]) for feature in features]
    references = replay.native_outputs(features, images)
    report["feature_channels"] = channels
    report["candidates"] = sum(len(image["raw_ids"]) for image in images)
    report["replay"] = []
    for index, (image, reference) in enumerate(zip(images, references)):
        op = image.get("_operator", image["transfer_operator"])
        torch.testing.assert_close(reference["c"].cpu(), image["c0"].float(), atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(reference["h"].cpu(), op["h0"].float(), atol=3e-5, rtol=3e-5)
        errors = []
        if "F_reference" in image:
            for level, old in enumerate(image["F_reference"]):
                current = features[level][index].cpu()
                torch.testing.assert_close(current, old.float(), atol=3e-5, rtol=3e-5)
                errors.append(float((current - old.float()).abs().max()))
        report["replay"].append(dict(image_id=int(image["image_id"]),
                                    old_F_witness_available="F_reference" in image,
                                    F_max_errors=errors,
                                    h_max_error=float((reference["h"].cpu() - op["h0"].float()).abs().max()),
                                    c_max_error=float((reference["c"].cpu() - image["c0"].float()).abs().max())))
    if not report["candidates"]:
        raise ValueError("No fixed official candidates in smoke images")
    initial_state = initial_counts = None
    for mode in ARMS:
        if deadline and time.time() > deadline:
            raise TimeoutError("Point smoke fixed deadline reached")
        setup(int(cfg["seed"]))
        selected, operator_report = make_selections(images, mode, torch.device("cuda"))
        model = JointPointCoefficientReadout(replay.native_cv4, channels, mode, cfg).cuda().float().train()
        state = state_digest(model)
        counts = model.parameter_counts()
        if initial_state is None:
            initial_state, initial_counts = state, counts
        else:
            assert state == initial_state, "U/Q/P initial parameters or buffers differ"
            assert counts == initial_counts, "U/Q/P parameter budgets differ"
        before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        original_bn = bn_state(model)
        buffers = {name: tensor_digest(value) for name, value in model.named_buffers()}
        selection_hash = tensor_tree_digest(selected)
        optimizer = optimizer_for(model, cfg)
        roi_error = roi_check(model, features, selected)
        independent_projection_error = isolated_projection_check(selected)
        step_grads, losses, reference_checks = [], [], []
        conditional_grads = {"h_current": 0.0, "c_current": 0.0}
        initial_c_error = chain_error = 0.0
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            coefficients, residuals, current = model.forward_details(features, selected)
            gradients, total_value = [], 0.0
            for image, coefficient, live in zip(images, coefficients, current):
                if step == 0:
                    # Zero-init must be exactly equal to this live native path;
                    # the old-cache comparison separately permits replay error.
                    torch.testing.assert_close(coefficient, live["c_current"], atol=0, rtol=0)
                    torch.testing.assert_close(coefficient.detach().cpu(), image["c0"].float(), atol=3e-5, rtol=3e-5)
                    initial_c_error = max(initial_c_error, float((coefficient.detach().cpu() - image["c0"].float()).abs().max()))
                value, gradient = loss_and_grad(gpu_image(image), coefficient, report["candidates"])
                gradients.append(gradient)
                total_value += value
                if step == 0:
                    reference_checks.append(direct_reference_check(image, coefficient, gradient,
                                                                    report["candidates"], value))
            expected = [torch.bmm(row["K"].transpose(1, 2), gradient.unsqueeze(2)).squeeze(2)
                        for row, gradient in zip(selected, gradients)]
            for residual in residuals:
                residual.retain_grad()
            if step == 1:
                live_inputs = [row[key] for row in current for key in ("h_current", "c_current")]
                live_grads = torch.autograd.grad(residuals, live_inputs, grad_outputs=expected,
                                                retain_graph=True, allow_unused=True)
                for index, gradient in enumerate(live_grads):
                    if gradient is not None:
                        assert bool(torch.isfinite(gradient).all())
                        conditional_grads["h_current" if index % 2 == 0 else "c_current"] += float(gradient.norm())
                for residual in residuals:
                    residual.grad = None
            torch.autograd.backward(coefficients, gradients)
            for residual, expected_gradient in zip(residuals, expected):
                if residual.grad is None:
                    raise AssertionError("Selected evidence is not connected to the official loss")
                torch.testing.assert_close(residual.grad, expected_gradient,
                                           atol=TOLERANCES["chain_atol"], rtol=TOLERANCES["chain_rtol"])
                chain_error = max(chain_error, float((residual.grad - expected_gradient).abs().max()))
            step_grads.append(finite_grad_norms(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step()
            verify_bn(model, original_bn)
            assert buffers == {name: tensor_digest(value) for name, value in model.named_buffers()}
            assert selection_hash == tensor_tree_digest(selected), "Frozen indices/operator/box inputs mutated"
            losses.append(total_value / report["candidates"])
        changes = {name: float((parameter.detach().cpu() - before[name]).norm())
                   for name, parameter in model.named_parameters()}
        for label, prefixes in dict(native=("native_cv4.",),
                                    encoder=("evidence.encoder_adapters.", "evidence.encoder_fusion."),
                                    query=("evidence.query_mlp.", "evidence.query_film."),
                                    point_decoder=("evidence.evidence_decoder.",)).items():
            assert any(value > 0 for name, value in step_grads[1].items() if name.startswith(prefixes)), f"{mode}: no second-step {label} gradient"
            assert any(value > 0 for name, value in changes.items() if name.startswith(prefixes)), f"{mode}: no {label} update"
        assert all(value > 0 for value in conditional_grads.values()), "Evidence lost live h/c conditioning gradients"
        present_levels = sorted({int(row["pyramid_level"]) for image in images for row in image["rows"]
                                 if "pyramid_level" in row})
        report["arms"][mode] = dict(parameter_counts=counts, operators=operator_report,
            initial_coefficient_cache_max_error=initial_c_error, initial_exact_native_identity=True,
            selected_cell_vs_full_roi_max_error=roi_error, independent_projection_gradient_error=independent_projection_error,
            actual_residual_gradient_max_error=chain_error, official_reference_checks=reference_checks,
            loss_by_disposable_step=losses, gradient_norms_by_step=step_grads,
            conditional_gradients_to_native=conditional_grads, parameter_changes=changes,
            observed_pyramid_levels=present_levels, unobserved_scales_not_claimed_tested=True,
            all_module_buffers_unchanged=True, selection_tensors_unchanged=True)
        dump(out / "SMOKE.json", report)
        print(json.dumps(dict(mode=mode, passed=True, candidates=report["candidates"],
                              roi_max_error=roi_error, chain_max_error=chain_error)), flush=True)
        del coefficients, residuals, current, gradients, expected, optimizer, model
        torch.cuda.empty_cache()
    assert tensor_tree_digest(images) == immutable_assets, "Transferred original data changed"
    assert tensor_tree_digest(features) == feature_hashes, "Frozen feature maps changed"
    assert state_digest(replay.source) == frozen_source, "Original frozen source parameters/buffers changed"
    assert all(not p.requires_grad and p.grad is None for p in replay.source.parameters())
    report.update(passed=True, all_three_initial_states_identical=True,
                  frozen_source_unchanged=True, frozen_features_labels_prototypes_unchanged=True)
    dump(out / "SMOKE.json", report)
    dump(out / "COMPLETE.json", dict(passed=True, kind="smoke", formal_training_steps=0,
                                     trained_smoke_weights_saved=False, automatic_training_launch=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", type=int, choices=(1, 2), default=2)
    parser.add_argument("--image-ids", type=int, nargs="+")
    parser.add_argument("--deadline", type=float, default=0.0)
    args = parser.parse_args()
    cfg = resolve_runtime_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Only execute on the authorized remote GPU host; no CPU fallback")
        image_ids = image_ids_for_smoke(cfg, args.image_ids, args.images)
        run_smoke(cfg, out, image_ids, args.deadline)
    except Exception as error:
        dump(out / "FAILURE.json", dict(error_type=type(error).__name__, error=str(error),
                                        traceback=traceback.format_exc(), formal_training_steps=0))
        raise


if __name__ == "__main__":
    main()
