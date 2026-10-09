"""Factual numerical and frozen-forward diagnostics for the TriFlow method.

Analytic ownership fields below are deliberately GT-oracle fixtures; passing
them certifies signs/mechanics, never deployable accuracy.  Learned-head task
gradient and actual frozen YOLO replay are separate mandatory checks.  The
caller owns Run creation, source snapshots and receipt collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch

from triflow_model import TriFlowConfig, TriFlowModel, build_ownership_targets, sample_grid

VERIFICATION_VERSION = "triflow_numerical_frozen_verification_v2"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(str((str(value.dtype), tuple(value.shape))).encode())
    result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def scalar(value):
    return float(value.detach().cpu())


def grid(size):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return xx.float(), yy.float()


def fixture(z, neighbor=False):
    height, width = z.shape
    xx, yy = grid(height)
    prototypes = torch.zeros(32, height, width)
    prototypes[0] = 1
    prototypes[1] = z
    coefficients = torch.zeros(1, 32)
    coefficients[0, 1] = 1
    visual = torch.stack((xx / width, yy / height, z / max(height, width), torch.ones_like(z)))
    boxes = torch.tensor([[0.0, 0.0, float(width), float(height)]])
    hidden = torch.arange(6).float()[None] / 6
    if neighbor:
        neighbor_logits = torch.full((1, 1, height, width), 2.0)
        neighbor_valid = torch.ones(1, 1, dtype=torch.bool)
    else:
        neighbor_logits = torch.empty(1, 0, height, width)
        neighbor_valid = torch.empty(1, 0, dtype=torch.bool)
    return prototypes, coefficients, visual, boxes, neighbor_logits, neighbor_valid, hidden


def diagnostic_config(**changes):
    # These relaxed limits isolate the compiler's sign, not production tuning.
    options = dict(hidden_dim=32, attention_heads=4, token_count=64, ray_samples=17,
                   ridge_lambda=1e-5, region_margin_max=0.0, max_logit_change=100.0,
                   max_relative_coefficient_change=100.0)
    options.update(changes)
    return TriFlowConfig(**options)


class OracleField(TriFlowModel):
    """Coordinate field replaces ONLY the learned predictor in oracle fixtures."""

    def __init__(self, field, config=None):
        super().__init__(4, 6, config or diagnostic_config())
        self.oracle_field = field

    def _field(self, inputs, xy, context, padding):
        phi = self.oracle_field(xy)
        stiffness = torch.full(xy.shape[:-1], 0.9, device=xy.device)
        attention = torch.full((*xy.shape[:-1], 3), 1 / 3, device=xy.device)
        return phi, stiffness, attention


def constant_field(neighbor_phi, background_phi):
    def field(xy):
        return torch.stack((torch.full_like(xy[..., 0], neighbor_phi),
                            torch.full_like(xy[..., 0], background_phi)), -1)
    return field


def compile_at(model, inputs_tuple, xy):
    inputs = model._prepare(*inputs_tuple)
    gram, context_xy, context_valid, _ = model._gram_and_tokens(inputs)
    context, padding = model._ownership_context(inputs, context_xy, context_valid)
    valid = torch.ones(xy.shape[:2], dtype=torch.bool)
    phi, stiffness, _ = model._field(inputs, xy, context, padding)
    result = model._compile(inputs, xy, valid, gram, context, padding, phi, stiffness)
    result["logits_initial"] = inputs["z"]
    result["stiffness"] = stiffness
    return result


def test_straight_edge(sign):
    size, current, displacement = 25, 12.4, sign * 0.75
    xx, _ = grid(size)
    inputs = fixture(current - xx)
    def field(xy):
        return torch.stack((torch.full_like(xy[..., 0], 16), current + displacement - xy[..., 0]), -1)
    model = OracleField(field)
    # Start away from the exact root: projection/residual must correct it.
    xy = torch.stack((torch.full((16,), current - 0.25), torch.linspace(3, 21, 16)), -1)[None]
    result = compile_at(model, inputs, xy)
    valid = result["root_valid"]
    require(valid.all(), "straight edge failed to find every known root")
    error = (result["displacement"][valid] - displacement).abs().max()
    require(error < 2e-5, "outward signed straight-edge displacement incorrect")
    surface_change = sample_grid(inputs[0][None], result["surface_xy"], (size, size)) @ result["delta_coefficients"][..., None]
    require((surface_change[..., 0] * sign > 0).all(), "compiler sign contradicts requested expansion/contraction")
    initial_area = int((result["logits_initial"] > 0).sum())
    refined_area = int((result["logits_refined"] > 0).sum())
    require((refined_area - initial_area) * sign > 0, "edge foreground area moved in wrong direction")
    require(finite(result), "nonfinite straight-edge diagnostic")
    return {"oracle_fixture": True, "requested_outward_pixels": displacement,
            "maximum_displacement_error": scalar(error), "initial_area": initial_area,
            "refined_area": refined_area, "root_count": int(valid.sum()),
            "minimum_signed_surface_logit_change": scalar((surface_change[..., 0] * sign).min()),
            "solve_residual": scalar(result["diagnostics"]["solve_residual"].max())}


def test_circle(sign):
    size, center, radius, displacement = 33, 16.0, 7.4, sign * 0.75
    xx, yy = grid(size)
    distance = ((xx - center).square() + (yy - center).square()).sqrt()
    inputs = fixture(radius - distance)
    def field(xy):
        rr = (xy - center).square().sum(-1).sqrt()
        return torch.stack((torch.full_like(rr, 16), radius + displacement - rr), -1)
    model = OracleField(field)
    angles = torch.arange(32).float() * (2 * math.pi / 32)
    xy = torch.stack((center + radius * angles.cos(), center + radius * angles.sin()), -1)[None]
    result = compile_at(model, inputs, xy)
    valid = result["root_valid"]
    require(valid.all(), "circle known ownership transitions not found")
    # Central-difference gradients and bilinear sampling are discrete approximations.
    error = (result["displacement"][valid] - displacement).abs().max()
    require(error < 0.05, "circle outward flow error exceeds finite-grid tolerance")
    radial_normal = (result["surface_xy"] - center)
    radial_normal = radial_normal / radial_normal.norm(dim=-1, keepdim=True)
    alignment = (radial_normal * result["outward_normal"]).sum(-1)
    require(alignment.min() > 0.99, "positive-mask circle normal does not point outward")
    initial_area = int((result["logits_initial"] > 0).sum())
    refined_area = int((result["logits_refined"] > 0).sum())
    require((refined_area - initial_area) * sign > 0, "circle compiler moved foreground area in wrong direction")
    require(finite(result), "nonfinite circle diagnostic")
    return {"oracle_fixture": True, "requested_outward_pixels": displacement,
            "maximum_displacement_error": scalar(error), "minimum_outward_alignment": scalar(alignment.min()),
            "initial_area": initial_area, "refined_area": refined_area,
            "root_count": int(valid.sum()), "solve_residual": scalar(result["diagnostics"]["solve_residual"].max())}


def test_neighbor_competition():
    size, current, shift = 25, 12.4, -0.75
    xx, _ = grid(size)
    inputs = fixture(current - xx, neighbor=True)
    def field(xy):
        return torch.stack((current + shift - xy[..., 0], torch.full_like(xy[..., 0], 16)), -1)
    model = OracleField(field)
    xy = torch.stack((torch.full((12,), current), torch.linspace(3, 21, 12)), -1)[None]
    result = compile_at(model, inputs, xy)
    require(result["root_valid"].all(), "neighbor ownership roots missing")
    require(result["transition_neighbor"].all(), "neighbor flow mislabeled as background")
    require((result["displacement"] < 0).all(), "neighbor encroachment must contract Self")
    require(int((result["logits_refined"] > 0).sum()) < int((result["logits_initial"] > 0).sum()), "neighbor conflict did not contract foreground")
    return {"oracle_fixture": True, "root_count": int(result["root_valid"].sum()),
            "neighbor_transition_count": int(result["transition_neighbor"].sum()),
            "mean_outward_displacement": scalar(result["displacement"].mean())}


def test_invalid_self_boundary():
    size, current = 25, 12.4
    xx, _ = grid(size)
    inputs = fixture(current - xx, neighbor=True)
    def field(xy):
        return torch.stack((current - xy[..., 0], torch.full_like(xy[..., 0], -3)), -1)
    model = OracleField(field)
    xy = torch.stack((torch.full((12,), current), torch.linspace(3, 21, 12)), -1)[None]
    result = compile_at(model, inputs, xy)
    require(not result["root_valid"].any(), "N zero inside already-BG area fabricated a Self boundary")
    require(torch.equal(result["displacement"], torch.zeros_like(result["displacement"])), "missing root fabricated signed flow")
    return {"oracle_fixture": True, "arbitrary_neighbor_zero_rejected": True,
            "root_count": 0, "displacement_exact_zero": True,
            "region_anchors_may_still_act": bool(result["anchor_active"].any())}


def test_absent_neighbor_and_no_root():
    size, current = 25, 12.4
    xx, _ = grid(size)
    inputs = fixture(current - xx, neighbor=False)
    # A nonexistent N zero must not enter the Self envelope.
    def field(xy):
        return torch.stack((current - xy[..., 0], torch.full_like(xy[..., 0], 3)), -1)
    model = OracleField(field)
    xy = torch.stack((torch.full((12,), current), torch.linspace(3, 21, 12)), -1)[None]
    result = compile_at(model, inputs, xy)
    require(not result["root_valid"].any(), "missing Neighbor influenced flow extraction")
    require(torch.equal(result["displacement"], torch.zeros_like(result["displacement"])), "no-root produced a made-up displacement")
    require(result["delta_coefficients"].abs().max() < 1e-6, "no-root satisfied zero-margin fixture should give no update")
    targets = build_ownership_targets((inputs[0][1] > 0)[None], neighbor_valid=torch.zeros(1, 2, dtype=torch.bool))
    require(not targets["potential_valid"][0, 0], "empty Neighbor GT received valid distance supervision")
    require(targets["potential_valid"][0, 1], "existing BG should remain supervised without Neighbor")
    return {"oracle_fixture": True, "root_count": 0, "displacement_exact_zero": True,
            "maximum_coefficient_update": scalar(result["delta_coefficients"].abs().max()),
            "neighbor_distance_invalid": True, "background_distance_valid": True}


def test_empty_and_overlap_regions():
    empty = torch.zeros(1, 17, 17, dtype=torch.bool)
    try:
        build_ownership_targets(empty)
    except ValueError as error:
        empty_message = str(error)
    else:
        raise AssertionError("empty Self must be explicitly rejected")
    full = torch.ones_like(empty)
    all_self = build_ownership_targets(full, full)
    require(not all_self["potential_valid"].any(), "absent competitor fields were supervised")
    require(not all_self["neighbor_masks"].any(), "Self precedence failed on overlapping GT")
    require(not all_self["background_masks"].any(), "full Self cannot contain valid BG")
    require(finite(all_self), "missing-distance placeholders must remain finite")
    inputs = fixture(torch.ones(17, 17))
    model = TriFlowModel(4, 6, diagnostic_config())
    output = model(*inputs)
    losses = model.loss(output, full, targets=all_self)
    require(int(losses["potential_supervised_count"]) == 0, "absent distances entered potential loss")
    require(scalar(losses["loss_potential"]) == 0, "invalid potentials produced nonzero supervised loss")
    require(finite(losses), "all-Self loss must remain finite")
    return {"empty_self_rejected": True, "rejection": empty_message,
            "empty_neighbor_and_background_explicitly_invalid": True,
            "overlap_self_precedence": True, "potential_supervised_count": 0,
            "finite_remaining_mask_task_loss": scalar(losses["loss_task"])}


def test_difference_before_clip():
    self_mask = torch.zeros(1, 41, 41, dtype=torch.bool)
    neighbor = torch.zeros_like(self_mask)
    self_mask[0, 20, 3] = True
    neighbor[0, 20, 7] = True
    target = build_ownership_targets(self_mask, neighbor, max_distance=5)
    actual = scalar(target["phi"][0, 0, 20, 40])
    require(abs(actual - (-4)) < 1e-6, "EDT components were clipped before their difference")
    return {"raw_self_distance": 37.0, "raw_neighbor_distance": 33.0,
            "distance_clip": 5.0, "actual_relative_potential": actual,
            "separately_clipped_wrong_value": 0.0}


def test_crowd_ignore():
    xx, yy = grid(25)
    self_mask = ((xx < 12) & (yy > 4) & (yy < 20))[None]
    neighbor = ((xx > 17) & (yy > 7) & (yy < 17))[None]
    valid = torch.ones_like(self_mask)
    valid[:, 8:13, 10:15] = False
    changed_self, changed_neighbor = self_mask.clone(), neighbor.clone()
    changed_self[~valid] = ~changed_self[~valid]
    changed_neighbor[~valid] = ~changed_neighbor[~valid]
    target1 = build_ownership_targets(self_mask, neighbor, valid_pixels=valid)
    target2 = build_ownership_targets(changed_self, changed_neighbor, valid_pixels=valid)
    require(torch.equal(target1["phi"], target2["phi"]), "ignored GT labels influenced ownership distance targets")
    require(not target1["background_masks"][~valid].any(), "crowd/ignore was assigned valid BG")
    model = TriFlowModel(4, 6, diagnostic_config())
    inputs = fixture(12.4 - xx, neighbor=True)
    output = model(*inputs)
    losses1 = model.loss(output, self_mask, targets=target1)
    losses2 = model.loss(output, changed_self, targets=target2)
    compare = ("loss_potential", "loss_direction", "loss_ordering", "loss_task", "loss")
    for key in compare:
        require(torch.equal(losses1[key], losses2[key]), f"ignored labels influenced {key}")
    return {"ignored_pixels": int((~valid).sum()), "distance_targets_exact_after_ignored_label_change": True,
            "potential_direction_order_and_task_losses_exact": True, "ignored_pixels_not_bg": True}


def test_region_anchors(kind, violation):
    size = 17
    z_value = ({"self": -0.5, "background": 0.5, "neighbor": 2.0}[kind]
               if violation else {"self": 2.0, "background": -2.0, "neighbor": 0.0}[kind])
    inputs = fixture(torch.full((size, size), z_value), neighbor=kind == "neighbor")
    values = {"self": (3.0, 3.0), "background": (3.0, -3.0), "neighbor": (-3.0, 3.0)}[kind]
    # Unlike curved/edge geometry, these P rows are exactly collinear.  S0
    # incorrectly inherited the relaxed geometric ridge 1e-5 * .001 = 1e-8;
    # that diagonal addition is lost at unit FP32 scale.  Use the production
    # ridge/epsilon here, retaining every original activity/direction assertion.
    model = OracleField(constant_field(*values), diagnostic_config(region_margin_max=3.0, ridge_lambda=0.1, gram_epsilon=1e-3))
    xy = torch.tensor([[[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]])
    result = compile_at(model, inputs, xy)
    expected_winner = {"self": 0, "neighbor": 1, "background": 2}[kind]
    require((result["region_winner"] == expected_winner).all(), "two-potential region decision incorrect")
    if violation:
        require(result["anchor_active"].all(), "violated inequality did not enter active set")
        change = result["logits_refined"] - result["logits_initial"]
        expected_sign = 1 if kind == "self" else -1
        require((change * expected_sign > 0).all(), "violated anchor pushed in wrong inequality direction")
    else:
        require(not result["anchor_active"].any(), "already satisfied inequality exerted force")
        require(torch.equal(result["delta_coefficients"], torch.zeros_like(result["delta_coefficients"])), "satisfied anchors generated an equality pull")
    require(finite(result), "region-anchor solve nonfinite")
    return {"oracle_fixture": True, "region": kind, "initial_violation": violation,
            "active_anchor_count": int(result["anchor_active"].sum()),
            "mean_logit_change": scalar((result["logits_refined"] - result["logits_initial"]).mean()),
            "prototype_rank": 1,
            "fixture_ridge_lambda": model.config.ridge_lambda,
            "fixture_gram_epsilon": model.config.gram_epsilon,
            "nominal_absolute_diagonal_ridge": model.config.ridge_lambda * model.config.gram_epsilon,
            "system_condition": scalar(result["diagnostics"]["system_condition"].max()),
            "active_set_oscillation_count": int(result["diagnostics"]["active_set_oscillation_count"].sum()),
            "verified_scope": "initial hinge activation and update direction; not a global inequality optimum"}


def test_rank_deficient_and_stiffness():
    xx, _ = grid(17)
    inputs = list(fixture(8.4 - xx))
    model = TriFlowModel(4, 6, diagnostic_config())
    inputs[0] = torch.zeros_like(inputs[0])
    with torch.no_grad():
        model.field_head[-1].bias[2] = -100
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(*inputs)
    require(finite(output), "rank-zero prototype solve produced nonfinite values")
    require(output["delta_coefficients"].dtype == torch.float32, "compiler arithmetic was demoted by autocast")
    require(torch.equal(output["delta_coefficients"], torch.zeros_like(output["delta_coefficients"])), "zero P should have zero observable coefficient correction")
    require(not output["boundary_valid"].any() and not output["root_valid"].any(), "zero gradient produced fabricated boundary flow")
    require((output["stiffness"] >= model.config.stiffness_floor).all(), "confidence collapsed below floor")
    require((output["diagnostics"]["gram_min_diagonal"] > 0).all(), "rank-deficient Gram lacks positive ridge")
    # A finite loss/backward is also required, without claiming nonzero gradients
    # in an intentionally unobservable P=0 fixture.
    losses = model.loss(output, (8.4 - xx > 0)[None])
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    require(gradients and all(finite(value) for value in gradients), "rank-zero supervised backward is not finite")
    # Exercise the declared scale-aware guard at a scale where an absolute
    # .1*.001 identity term alone can be rounded away.  This is a separate
    # adversarial fixture, not evidence of a failure on native frozen inputs.
    large_inputs = list(fixture(torch.full((17, 17), 0.5)))
    large_inputs[0] = large_inputs[0] * 1e4
    large_model = OracleField(constant_field(3.0, -3.0), diagnostic_config(
        region_margin_max=3.0, ridge_lambda=0.1, gram_epsilon=1e-3))
    large_xy = torch.tensor([[[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]]])
    large_output = compile_at(large_model, large_inputs, large_xy)
    large_diagnostics = large_output["diagnostics"]
    require(finite(large_output), "large-scale collinear prototype guard did not preserve finiteness")
    require((large_output["logits_refined"] - large_output["logits_initial"] < 0).all(), "large-scale violated BG anchor did not contract response")
    added_gram = large_diagnostics["added_gram_numeric_ridge"]
    added_system = large_diagnostics["added_system_numeric_ridge"]
    require((added_gram > 0).any() or (added_system > 0).any(), "large-scale fixture did not exercise explicit numerical ridge")
    require((large_diagnostics["gram_cholesky_info"] == 0).all(), "large-scale Gram SPD check not successful")
    require((large_diagnostics["system_cholesky_info"] == 0).all(), "large-scale per-pass system SPD check not successful")
    large_facts = {"prototype_rank": 1, "prototype_multiplier": 1e4,
                   "native_case_failure_claim": False,
                   "ridge_lambda": large_model.config.ridge_lambda,
                   "gram_epsilon": large_model.config.gram_epsilon,
                   "added_gram_numeric_ridge": added_gram.detach().cpu().tolist(),
                   "added_system_numeric_ridge": added_system.detach().cpu().tolist(),
                   "gram_identity_ridge": large_diagnostics["gram_identity_ridge"].detach().cpu().tolist(),
                   "effective_system_identity_ridge": large_diagnostics["effective_system_identity_ridge"].detach().cpu().tolist(),
                   "gram_cholesky_info": large_diagnostics["gram_cholesky_info"].detach().cpu().tolist(),
                   "system_cholesky_info": large_diagnostics["system_cholesky_info"].detach().cpu().tolist(),
                   "pass_system_condition": large_diagnostics["pass_system_condition"].detach().cpu().tolist(),
                   "mean_logit_change": scalar((large_output["logits_refined"] - large_output["logits_initial"]).mean()),
                   "solve_residual": scalar(large_diagnostics["solve_residual"].max()),
                   "trust_scale": scalar(large_diagnostics["trust_scale"].min()),
                   "constraint_satisfaction_required": False}
    return {"prototype_rank": 0, "compiler_dtype": str(output["delta_coefficients"].dtype),
            "zero_gradient_boundary_rejected": True, "minimum_stiffness": scalar(output["stiffness"].min()),
            "stiffness_floor": model.config.stiffness_floor,
            "minimum_gram_diagonal": scalar(output["diagnostics"]["gram_min_diagonal"].min()),
            "solve_residual": scalar(output["diagnostics"]["solve_residual"].max()),
            "finite_supervised_gradient_tensors": len(gradients),
            "large_scale_rank_one_guard_fixture": large_facts}


def task_gradient_measurement(model, inputs, mask):
    model.zero_grad(set_to_none=True)
    output = model(*inputs)
    losses = model.loss(output, mask)
    require(finite(losses), "nonfinite learned-head losses")
    losses["loss_task"].backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    require(all(finite(value) for value in gradients.values() if value is not None), "nonfinite task-only gradient")
    field = gradients["field_head.2.weight"]
    attention = gradients["cross_attention.in_proj_weight"]
    potential_norm = 0.0 if field is None else scalar(field[:2].norm())
    attention_norm = 0.0 if attention is None else scalar(attention.norm())
    return output, losses, {"task_loss": scalar(losses["loss_task"]),
                            "potential_output_gradient_norm": potential_norm,
                            "interaction_attention_gradient_norm": attention_norm,
                            "root_count": int(output["root_valid"].sum()),
                            "active_anchor_count": int(output["anchor_active"].sum()),
                            "maximum_coefficient_update": scalar(output["delta_coefficients"].abs().max())}


def test_task_only_gradient_and_update():
    torch.manual_seed(607)
    xx, _ = grid(25)
    inputs = fixture(12.4 - xx)
    mask = (13.3 - xx > 0)[None]
    config = TriFlowConfig(token_count=64)
    model = TriFlowModel(4, 6, config)
    output, losses, facts = task_gradient_measurement(model, inputs, mask)
    require(facts["potential_output_gradient_norm"] > 1e-10, "TASK_ONLY_GRADIENT_FAIL: no gradient reaches potential outputs at actual initialization")
    require(facts["interaction_attention_gradient_norm"] > 1e-10, "TASK_ONLY_GRADIENT_FAIL: no gradient reaches ownership interaction at actual initialization")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    optimizer.step()
    field_change = (model.field_head[-1].weight[:2] - before["field_head.2.weight"][:2]).abs().max()
    interaction_change = (model.cross_attention.in_proj_weight - before["cross_attention.in_proj_weight"]).abs().max()
    require(field_change > 0 and interaction_change > 0, "optimizer did not actually update potential and interaction parameters")
    require(all(finite(parameter) for parameter in model.parameters()), "optimizer produced nonfinite parameters")
    facts.update({"potential_parameter_maximum_update": scalar(field_change),
                  "interaction_parameter_maximum_update": scalar(interaction_change),
                  "optimizer": "AdamW(lr=0.001, weight_decay=0), one task-only fixture step",
                  "supervised_warmup_steps": 0, "fixture_hidden_dim": model.config.hidden_dim,
                  "production_accuracy_inference": False,
                  "frozen_input_gradients": [value.grad is not None for value in inputs if isinstance(value, torch.Tensor)]})
    require(not any(facts["frozen_input_gradients"]), "frozen input tensor received gradients")
    return facts


def test_zero_initialization_negative_control():
    torch.manual_seed(607)
    xx, _ = grid(25)
    inputs = fixture(12.4 - xx)
    model = TriFlowModel(4, 6, TriFlowConfig(token_count=64))
    with torch.no_grad():
        model.field_head[-1].weight.zero_()
        model.field_head[-1].bias.zero_()
    _, _, facts = task_gradient_measurement(model, inputs, (13.3 - xx > 0)[None])
    detected = facts["potential_output_gradient_norm"] <= 1e-10
    require(detected, "negative-control zero potential head did not trigger expected potential-gradient detector")
    facts.update({"deliberate_negative_control": True, "expected_failure_detected": True,
                  "would_reject_this_as_production_initialization": True,
                  "actual_production_initialization_changed": False,
                  "fixture_hidden_dim": model.config.hidden_dim})
    return facts


def test_frozen_yolo(args):
    from frozen_io import FrozenYOLO, predicted_neighbors
    require(args.weights and args.vendor, "mandatory actual frozen-forward check requires --weights and --vendor")
    import cv2
    import numpy as np
    if args.image:
        image_path = Path(args.image)
        image_kind = "caller-supplied image, diagnostic forward only"
    else:
        # User-owned Run fixture directory; does not create a dataset or cache.
        image_path = Path(args.out).resolve().parent / "fixtures" / "frozen_forward_fixture.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(607)
        image = rng.integers(0, 80, (360, 480, 3), dtype=np.uint8)
        cv2.circle(image, (170, 180), 92, (70, 180, 210), -1)
        cv2.rectangle(image, (280, 110), (410, 260), (190, 60, 40), -1)
        require(cv2.imwrite(str(image_path), image), "failed to save frozen-forward Run fixture")
        image_kind = "deterministic synthetic image, actual official-model forward"
    extractor = FrozenYOLO(args.weights, args.vendor, device=args.device, image_size=640)
    try:
        first = extractor.extract(image_path)
        second = extractor.extract(image_path)
        keys = ("P", "F", "h", "c0", "boxes", "boxes_proto", "scores", "classes", "raw_indices", "output_rows")
        require(first["native_replay_exact"] and second["native_replay_exact"], "official native row reconstruction not exact")
        require(first["input_sha256"] == second["input_sha256"], "repeated frozen input differs")
        for key in keys:
            require(torch.equal(first[key], second[key]), f"frozen repeated {key} differs")
        require(first["h"].shape[0] == first["c0"].shape[0] == 300, "native hidden/query row count wrong")
        require(first["raw_indices"].dtype == torch.int64 and first["classes"].dtype == torch.int64, "native index/class identity dtype wrong")
        frozen_before = extractor.verify_frozen()
        originals = {key: first[key].clone() for key in keys}
        rows, valid = predicted_neighbors(first["boxes"].numpy(), first["raw_indices"].numpy(), first["scores"].numpy(), k=2)
        # Check raw identity exclusion, including class-expanded duplicates.
        for row in range(len(rows)):
            for slot in range(2):
                if valid[row, slot]:
                    require(first["raw_indices"][rows[row, slot]] != first["raw_indices"][row], "duplicate raw index used as own Neighbor")
        selected = torch.arange(min(2, len(first["scores"])))
        neighbor_logits = torch.zeros(len(selected), 2, *first["P"].shape[-2:])
        neighbor_valid = torch.as_tensor(valid[selected.numpy()])
        for instance, row in enumerate(selected.tolist()):
            for slot in range(2):
                if neighbor_valid[instance, slot]:
                    neighbor_logits[instance, slot] = torch.einsum("chw,c->hw", first["P"], first["c0"][rows[row, slot]])
        head = TriFlowModel(first["F"].shape[0], first["h"].shape[1]).to(args.device)
        forward_inputs = (first["P"].to(args.device), first["c0"][selected].to(args.device),
                          first["F"].to(args.device), first["boxes_proto"][selected].to(args.device),
                          neighbor_logits.to(args.device), neighbor_valid.to(args.device), first["h"][selected].to(args.device))
        output = head(*forward_inputs)
        require(finite(output), "actual frozen-feature TriFlow forward not finite")
        # Actual integration graph check; this auxiliary diagnostic is separate
        # from the mandatory synthetic mask-task-only gradient test above.
        integration_loss = output["phi"].square().mean() + output["logits_refined"].square().mean()
        integration_loss.backward()
        require(all(finite(parameter.grad) for parameter in head.parameters() if parameter.grad is not None), "actual-feature TriFlow backward nonfinite")
        torch.optim.AdamW(head.parameters(), lr=1e-4).step()
        for key in keys:
            require(torch.equal(first[key], originals[key]), f"TriFlow mutated frozen input {key}")
        frozen_after = extractor.verify_frozen()
        third = extractor.extract(image_path)
        for key in keys:
            require(torch.equal(first[key], third[key]), f"native {key} changed after new-head update")
        final = extractor.verify_frozen()
        source = {}
        for relative in ("ultralytics/nn/modules/head.py", "ultralytics/nn/modules/block.py", "ultralytics/utils/ops.py", "ultralytics/data/augment.py"):
            path = Path(args.vendor) / relative
            require(path.is_file(), f"vendor source missing: {relative}")
            source[relative] = {"path": str(path.resolve()), "sha256": sha256(path)}
        return {"image_kind": image_kind, "image_path": str(image_path.resolve()),
                "image_sha256": first["image_sha256"], "input_sha256": first["input_sha256"],
                "three_actual_forwards": True, "all_native_rows_replayed_exact": True,
                "frozen_tensors_exact_before_and_after_head_optimizer_step": True,
                "tensor_shapes": {key: list(first[key].shape) for key in keys},
                "tensor_sha256": {key: tensor_sha256(first[key]) for key in keys},
                "hidden_query_source": first["hidden_query_source"],
                "prototype_channels": int(first["P"].shape[0]), "selected_rows_for_new_head": selected.tolist(),
                "integration_loss": scalar(integration_loss),
                "integration_loss_is_not_mask_task_gradient_certificate": True,
                "frozen_before": frozen_before, "frozen_after_new_head_step": frozen_after,
                "frozen_final": final, "vendor_source": source}
    finally:
        extractor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--vendor")
    parser.add_argument("--image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--numerical-only", action="store_true", help="Developer fixture check; never returns full-method PASS")
    args = parser.parse_args()
    path = Path(args.out).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing diagnostic receipt: {path}")
    torch.set_num_threads(4)
    torch.manual_seed(607)
    started = datetime.now(timezone.utc).isoformat()
    receipt = {"verification_version": VERIFICATION_VERSION, "started_at": started,
               "completed_at": None, "status": "running", "passed": False,
               "actual_frozen_forward_required": not args.numerical_only,
               "numerical_only": args.numerical_only, "fixture_seed": 607,
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "torch": torch.__version__, "requested_device": args.device},
               "source_files": {}, "checks": [],
               "verification_revision": {
                   "previous_run_id": "RUN_TRIFLOW_NUMERICAL_VERIFY_S0",
                   "previous_failed_checks": ["self_anchor_violated", "neighbor_anchor_violated", "background_anchor_violated"],
                   "anchor_fixture_correction": "Use production lambda=.1 and epsilon=.001 for constant rank-1 P, rather than relaxed geometry lambda=1e-5; no pass condition weakened.",
                   "failure_evidence": "Nominal 1e-8 ridge is below unit-scale FP32 ULP (about 1.19e-7); IEEE FP32 demo loses both leading diagonal additions and yields determinant zero at constant z=-.5,.5,2. This explains the observed singular solve; it is not a reconstruction of an archived exact lhs tensor.",
                   "learned_task_gradient_fixture_correction": "Use production hidden_dim=128 instead of prior compact hidden_dim=32; still no supervised warmup.",
                   "additional_adversarial_fixture": "Rank-1 P at multiplier 1e4 with production lambda/epsilon must explicitly exercise the declared numerical ridge and successful Cholesky checks; does not assert native inputs failed.",
                   "historical_receipt_mutated": False,
               },
               "scope_limitations": ["Oracle fields verify mechanics, not learned deployable accuracy.",
                                      "No AP, generalization, or novelty claim is established here.",
                                      "Anchor checks cover initial hinge activity and direction; not an exact global inequality optimum."]}
    for name in ("verify_triflow.py", "triflow_model.py", "frozen_io.py"):
        source = Path(__file__).resolve().with_name(name)
        if source.is_file():
            receipt["source_files"][name] = {"path": str(source), "sha256": sha256(source)}
    tests = [("straight_edge_expansion", lambda: test_straight_edge(1)),
             ("straight_edge_contraction", lambda: test_straight_edge(-1)),
             ("circle_expansion", lambda: test_circle(1)),
             ("circle_contraction", lambda: test_circle(-1)),
             ("neighbor_competition", test_neighbor_competition),
             ("zero_of_nonself_competitors_not_self_boundary", test_invalid_self_boundary),
             ("absent_neighbor_and_no_root", test_absent_neighbor_and_no_root),
             ("empty_self_background_and_overlap", test_empty_and_overlap_regions),
             ("potential_difference_before_clip", test_difference_before_clip),
             ("crowd_ignore_invariance", test_crowd_ignore)]
    for kind in ("self", "neighbor", "background"):
        for violation in (False, True):
            tests.append((f"{kind}_anchor_{'violated' if violation else 'satisfied'}", lambda kind=kind, violation=violation: test_region_anchors(kind, violation)))
    tests += [("rank_zero_spd_stiffness_floor_and_finite_backward", test_rank_deficient_and_stiffness),
              ("mask_task_only_gradient_and_actual_optimizer_update", test_task_only_gradient_and_update),
              ("zero_initialization_gradient_failure_negative_control", test_zero_initialization_negative_control)]
    if not args.numerical_only:
        tests.append(("actual_official_frozen_forward_feature_hidden_and_native_index_replay", lambda: test_frozen_yolo(args)))
    for name, operation in tests:
        check = {"name": name, "required": True, "status": "running"}
        try:
            check["facts"] = operation()
            check["passed"] = True
            check["status"] = "passed"
        except Exception as error:
            check.update({"status": "failed", "passed": False, "error_type": type(error).__name__,
                          "error": str(error), "traceback": traceback.format_exc()})
        receipt["checks"].append(check)
        print(json.dumps({"name": name, "passed": check["passed"], "error": check.get("error")}), flush=True)
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    receipt["all_executed_checks_passed"] = all(check["passed"] for check in receipt["checks"])
    receipt["passed"] = receipt["all_executed_checks_passed"] and not args.numerical_only
    receipt["status"] = "passed" if receipt["passed"] else ("numerical_only_passed_not_full_verified" if receipt["all_executed_checks_passed"] else "failed")
    receipt["check_count"] = len(receipt["checks"])
    receipt["failed_checks"] = [check["name"] for check in receipt["checks"] if not check["passed"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"], "check_count": len(tests), "receipt": str(path)}), flush=True)
    return 0 if receipt["all_executed_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
