"""Remote-only disposable verification for N/U/Q/P/L/B; no saved learned weights.

Two steps per arm check the zero-initialized evidence branch after its final
layer moves. Supervision is the unchanged full official mask BCE. Synthetic
selection tests cover B's exact 32/32 quota, depleted pools and zero-logit ties.
N is the original native-only baseline, not a capacity-matched evidence model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Match the training CLI: standalone invocation must import the frozen source,
# even when a caller has not already set PYTHONPATH.
if "--config" in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index("--config")+1]).read_text(encoding="utf-8-sig"))
    if _cfg.get("source_python"):
        sys.path.insert(0, _cfg["source_python"])

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics.utils.loss import v8SegmentationLoss

from online_runtime import FrozenReplay, load_asset, load_index, resolve_runtime_config
from point_head import selected_cell_rois
from point_selectors import UNIFORM_INDICES, build_selection_operators
from runtime_utils import setup, loss_and_grad, bn_state, verify_bn
from screen_models import TRAIN_ARMS, POINT_ARMS, build_model, build_operators, select_operators

TOLERANCES = dict(replay_atol=3e-5, replay_rtol=3e-5, roi_atol=3e-5, roi_rtol=3e-5,
                  loss_atol=3e-5, loss_rtol=3e-5, chain_atol=3e-6, chain_rtol=3e-5,
                  operator_fp64=1e-8, operator_fp32=1e-4)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode()+str(list(value.shape)).encode()+value.numpy().tobytes()).hexdigest()


def tree_hash(value):
    if torch.is_tensor(value):
        return tensor_hash(value)
    if isinstance(value, dict):
        return {name: tree_hash(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_hash(item) for item in value]
    return value


def state_hash(module):
    return {name: tensor_hash(value) for name, value in module.state_dict().items()}


def payload(image, device):
    return {name: image[name].to(device) if torch.is_tensor(image[name]) else image[name]
            for name in ("proto", "masks", "owners", "target_boxes", "segmentation_gain")}


def optimizer_for(model, cfg):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups.setdefault((name.startswith("native_cv4."), parameter.ndim > 1), []).append(parameter)
    return torch.optim.AdamW([dict(params=params, lr=cfg["branch_lr"] if native else cfg["new_lr"],
                                  weight_decay=cfg["weight_decay"] if decay else 0.)
                             for (native, decay), params in groups.items()], betas=(0.9, 0.999), eps=1e-8)


def gradient_norms(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            result[name] = float(parameter.grad.norm())
    return result


@torch.no_grad()
def selector_checks(device):
    a = torch.zeros((1, 256, 32), device=device, dtype=torch.float64)
    a[:, :, 0] = 1.
    g = torch.eye(32, device=device, dtype=torch.float64)[None]
    c0 = torch.zeros((1, 32), device=device, dtype=torch.float64)
    checks = {}
    for mode in POINT_ARMS:
        built = select_operators(a, g, c0, mode)
        expected = torch.tensor(UNIFORM_INDICES if mode == "U" else range(64), device=device)
        torch.testing.assert_close(built["indices"][0], expected, atol=0, rtol=0)
        assert bool(built["valid"].all()) and built["indices"][0].unique().numel() == 64
        err64 = float(built["diagnostics"]["relative_residual_fp64"].max())
        err32 = float(built["diagnostics"]["relative_residual_fp32"].max())
        assert err64 <= TOLERANCES["operator_fp64"] and err32 <= TOLERANCES["operator_fp32"]
        if mode in ("U", "Q", "P"):
            old = build_selection_operators(a, g, c0, mode)
            for key in ("indices", "K", "valid"):
                torch.testing.assert_close(built[key], old[key], atol=0, rtol=0)
        checks[mode] = dict(unique_cells=64, grid_id_ties_correct=True, fp64_residual=err64,
                            fp32_residual=err32, old_implementation_exact=mode in ("U", "Q", "P"))
    # L's selection/K must ignore original coefficients, apart from validity.
    l0 = select_operators(a, g, c0, "L")
    l1 = select_operators(a, g, c0+2., "L")
    torch.testing.assert_close(l0["indices"], l1["indices"], atol=0, rtol=0)
    torch.testing.assert_close(l0["K"], l1["K"], atol=0, rtol=0)
    checks["L"]["coefficient_invariant_selection"] = True
    # Equal response magnitudes make P gains and uncertainty tie across signs.
    # Quota checks include empty pools, fewer than 32, exactly 32, and 32+.
    counts = [0, 1, 5, 31, 32, 33, 224, 225, 255, 256]
    ab = a.expand(len(counts), -1, -1).clone()
    for row, positives in enumerate(counts):
        ab[row, positives:, 0] = -1.
    gb = g.expand(len(counts), -1, -1).clone()
    cb = c0.expand(len(counts), -1).clone()
    cb[:, 0] = 1.
    actual = select_operators(ab, gb, cb, "B")
    assert bool(actual["valid"].all())
    quota_checks = []
    for row, positives in enumerate(counts):
        pools = [list(range(positives)), list(range(positives, 256))]
        expected = []
        for step in range(64):
            preferred = step % 2
            pool = preferred if pools[preferred] else 1-preferred
            expected.append(pools[pool].pop(0))
        torch.testing.assert_close(actual["indices"][row], torch.tensor(expected, device=device), atol=0, rtol=0)
        assert actual["indices"][row].unique().numel() == 64
        selected_positive = int((actual["indices"][row] < positives).sum())
        wanted = min(32, positives)+max(0, 32-(256-positives))
        assert selected_positive == wanted
        quota_checks.append(dict(nonnegative_cells=positives, selected_nonnegative=selected_positive,
                                 selected_negative=64-selected_positive, deterministic_order=True))
    checks["B"].update(quota_and_tiny_pool_checks=quota_checks, zero_is_nonnegative=True,
                       ordering="nonnegative first, alternate; exhausted-pool quota goes to other pool")
    return checks


@torch.no_grad()
def roi_check(model, features, selections):
    encoded = model.evidence._encode(features, len(selections))
    error = 0.
    for j, selected in enumerate(selections):
        keep = selected["valid"]
        if not bool(keep.any()):
            continue
        boxes, indices = selected["boxes"][keep].clamp(0, 640), selected["indices"][keep]
        rois = torch.cat((boxes.new_zeros((len(boxes), 1)), boxes), dim=1)
        full = roi_align(encoded[j:j+1], rois, output_size=(16, 16),
                         spatial_scale=encoded.shape[-2]/640., sampling_ratio=2, aligned=True)
        expected = full.flatten(2).transpose(1, 2).gather(1, indices[:, :, None].expand(-1, -1, 64))
        actual = roi_align(encoded[j:j+1], selected_cell_rois(boxes, indices), output_size=(1, 1),
                           spatial_scale=encoded.shape[-2]/640., sampling_ratio=2, aligned=True).reshape(len(boxes), 64, 64)
        torch.testing.assert_close(actual, expected, atol=TOLERANCES["roi_atol"], rtol=TOLERANCES["roi_rtol"])
        error = max(error, float((actual-expected).abs().max()))
    return error


def reference_loss_check(image, coefficient, gradient, denominator, value, device):
    x = payload(image, device)
    leaf = coefficient.detach().requires_grad_(True)
    proto = F.interpolate(x["proto"][None].float(), (640, 640), mode="bilinear", align_corners=False)[0]
    gt = (x["masks"][None] == (x["owners"]+1)[:, None, None]).float()
    boxes = x["target_boxes"]
    area = ((boxes[:, 2:]-boxes[:, :2])/640.).prod(1)
    direct = v8SegmentationLoss.single_mask_loss(gt, leaf, proto, boxes, area)*float(x["segmentation_gain"])
    expected = torch.autograd.grad(direct/denominator, leaf)[0]
    torch.testing.assert_close(expected, gradient, atol=TOLERANCES["loss_atol"], rtol=TOLERANCES["loss_rtol"])
    torch.testing.assert_close(direct.detach(), direct.new_tensor(value), atol=TOLERANCES["loss_atol"], rtol=TOLERANCES["loss_rtol"])
    return dict(loss_absolute_error=abs(float(direct.detach())-value), gradient_max_error=float((expected-gradient).abs().max()))


def smoke(cfg, out, image_ids, deadline=0.):
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError("Execute only on the authorized Linux GPU server")
    if not 1 <= len(image_ids) <= 2 or len(set(image_ids)) != len(image_ids):
        raise ValueError("Smoke uses one or two fixed distinct fit images")
    setup(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    images = [load_asset(cfg, iid, verify=True) for iid in image_ids]
    if any(image["split"] != "fit" for image in images):
        raise ValueError("Disposable gradient smoke must use fit images")
    count = sum(len(image["raw_ids"]) for image in images)
    if not count:
        raise ValueError("Smoke images contain no official positives")
    image_state = tree_hash(images)
    replay = FrozenReplay(cfg, device="cuda")
    source_state = state_hash(replay.source)
    features = replay.replay(images)
    feature_state = tree_hash(features)
    references = replay.native_outputs(features, images)
    report = dict(kind="fast_screen_implementation_smoke", formal_training_steps=0,
                  disposable_steps_per_arm=2, model_weights_saved=False, image_ids=image_ids,
                  candidates=count, feature_channels=replay.feature_channels, tolerances=TOLERANCES,
                  training_loss="Unchanged official full640 mask BCE; candidate-equal aggregation",
                  selector_checks=selector_checks(device), arms={}, replay=[])
    for image, ref in zip(images, references):
        op = image["_operator"]
        torch.testing.assert_close(ref["c"].cpu(), image["c0"].float(), atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(ref["h"].cpu(), op["h0"].float(), atol=3e-5, rtol=3e-5)
        report["replay"].append(dict(image_id=int(image["image_id"]),
                                   coefficient_max_error=float((ref["c"].cpu()-image["c0"]).abs().max()),
                                   h_max_error=float((ref["h"].cpu()-op["h0"]).abs().max())))
    dump(out/"SMOKE.json", report)
    point_initial = point_counts = native_initial = None
    for mode in TRAIN_ARMS:
        if deadline and time.time() >= deadline:
            raise TimeoutError("Smoke deadline reached; no retry")
        setup(int(cfg["seed"]))
        selected = [build_operators(image, mode, cfg, device) for image in images]
        selected_state = tree_hash(selected)
        model = build_model(mode, replay, cfg).to(device).float().train()
        state, counts = state_hash(model), model.parameter_counts()
        current_native = {key: value for key, value in state.items() if key.startswith("native_cv4.")}
        if native_initial is None:
            native_initial = current_native
        else:
            assert native_initial == current_native, "Native initial states differ between arms"
        if mode in POINT_ARMS:
            if point_initial is None:
                point_initial, point_counts = state, counts
            else:
                assert point_initial == state, "Five point models have different initial parameters"
                assert point_counts == counts, "Five point models have different parameter counts"
        before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        original_bn = bn_state(model)
        original_buffers = {name: tensor_hash(b) for name, b in model.named_buffers()}
        optimizer = optimizer_for(model, cfg)
        roi_error = roi_check(model, features, selected) if mode in POINT_ARMS else None
        steps, losses, reference_checks = [], [], []
        chain_error = cache_error = 0.
        conditional = dict(h_current=0., c_current=0.)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            coefficients, residuals, current = model.forward_details(features, selected)
            gradients, value = [], 0.
            for image, c, live in zip(images, coefficients, current):
                if step == 0:
                    torch.testing.assert_close(c, live["c_current"], atol=0, rtol=0)
                    torch.testing.assert_close(c.detach().cpu(), image["c0"].float(), atol=3e-5, rtol=3e-5)
                    cache_error = max(cache_error, float((c.detach().cpu()-image["c0"]).abs().max()))
                loss, gradient = loss_and_grad(payload(image, device), c, count)
                gradients.append(gradient)
                value += loss
                if step == 0:
                    reference_checks.append(reference_loss_check(image, c, gradient, count, loss, device))
            if mode in POINT_ARMS:
                expected = [torch.bmm(sel["K"].transpose(1, 2), grad.unsqueeze(2)).squeeze(2)
                            for sel, grad in zip(selected, gradients)]
                for residual in residuals:
                    residual.retain_grad()
                if step == 1:
                    inputs = [row[key] for row in current for key in ("h_current", "c_current")]
                    live_grads = torch.autograd.grad(residuals, inputs, grad_outputs=expected,
                                                    retain_graph=True, allow_unused=True)
                    for i, gradient in enumerate(live_grads):
                        if gradient is not None:
                            assert bool(torch.isfinite(gradient).all())
                            conditional["h_current" if i % 2 == 0 else "c_current"] += float(gradient.norm())
                    for residual in residuals:
                        residual.grad = None
            torch.autograd.backward(coefficients, gradients)
            if mode in POINT_ARMS:
                for residual, wanted in zip(residuals, expected):
                    if residual.grad is None:
                        raise AssertionError("Evidence has no official-loss gradient")
                    torch.testing.assert_close(residual.grad, wanted, atol=TOLERANCES["chain_atol"], rtol=TOLERANCES["chain_rtol"])
                    chain_error = max(chain_error, float((residual.grad-wanted).abs().max()))
            steps.append(gradient_norms(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step()
            verify_bn(model, original_bn)
            assert original_buffers == {name: tensor_hash(b) for name, b in model.named_buffers()}
            assert selected_state == tree_hash(selected), "Frozen selector/projection inputs changed"
            losses.append(value/count)
        changes = {name: float((p.detach().cpu()-before[name]).norm()) for name, p in model.named_parameters()}
        groups = dict(native=("native_cv4.",))
        if mode in POINT_ARMS:
            groups.update(encoder=("evidence.encoder_adapters.", "evidence.encoder_fusion."),
                          query=("evidence.query_mlp.", "evidence.query_film."),
                          decoder=("evidence.evidence_decoder.",))
            assert all(value > 0 for value in conditional.values()), "Lost live native h/c evidence gradient"
        for group, prefixes in groups.items():
            assert any(value > 0 for name, value in steps[1].items() if name.startswith(prefixes)), f"{mode}: {group} gradient missing"
            assert any(value > 0 for name, value in changes.items() if name.startswith(prefixes)), f"{mode}: {group} did not update"
        report["arms"][mode] = dict(parameter_counts=counts, original_coefficient_max_error=cache_error,
            initial_native_identity_exact=True, selected_cell_roi_max_error=roi_error,
            actual_K_chain_gradient_max_error=chain_error if mode in POINT_ARMS else None,
            reference_loss_checks=reference_checks, gradient_norms_by_step=steps,
            conditional_gradient_norms=conditional if mode in POINT_ARMS else None,
            parameter_changes=changes, disposable_step_bce=losses,
            fixed_buffers_unchanged=True, fixed_operators_unchanged=True,
            operators=[row["_operator_diagnostics"] for row in selected],
            observed_levels=sorted({int(v) for image in images for v in image["levels"].tolist()}),
            unobserved_scales_not_claimed_tested=True)
        dump(out/"SMOKE.json", report)
        print(json.dumps(dict(mode=mode, passed=True, parameters=counts["total"])), flush=True)
        del optimizer, model, coefficients, current, gradients, residuals, selected
        torch.cuda.empty_cache()
    assert image_state == tree_hash(images), "Original labels/prototypes/inputs changed"
    assert feature_state == tree_hash(features), "Frozen neck maps changed"
    assert source_state == state_hash(replay.source), "Original source weights/buffers changed"
    assert all(not p.requires_grad and p.grad is None for p in replay.source.parameters())
    replay.assert_unchanged()
    report.update(passed=True, all_five_point_initial_states_identical=True,
                  all_five_point_parameter_counts_identical=True, all_six_native_initial_states_identical=True,
                  native_only_not_capacity_matched=True, frozen_source_unchanged=True,
                  full_mask_supervision_preserved=True, no_GT_selection=True)
    dump(out/"SMOKE.json", report)
    dump(out/"COMPLETE.json", dict(passed=True, kind="fast_screen_smoke", formal_training_steps=0,
         disposable_steps_per_arm=2, trained_weights_saved=False, automatic_training_launch=False,
         all_five_point_initial_states_identical=True, frozen_source_unchanged=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-ids", type=int, nargs="+")
    parser.add_argument("--images", type=int, choices=(1, 2), default=2)
    parser.add_argument("--deadline", type=float, default=0.)
    args = parser.parse_args()
    cfg = resolve_runtime_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        ids = args.image_ids
        if ids is None:
            index = load_index(cfg)
            ids = [int(row["image_id"]) for row in index["fit"] if row["n"] > 0][:args.images]
            if len(ids) != args.images:
                raise RuntimeError("Not enough fixed fit assets for the smoke")
        smoke(cfg, out, ids, args.deadline)
    except BaseException as exc:
        dump(out/"FAILURE.json", dict(error=str(exc), traceback=traceback.format_exc(), formal_training_steps=0))
        raise


if __name__ == "__main__":
    main()
