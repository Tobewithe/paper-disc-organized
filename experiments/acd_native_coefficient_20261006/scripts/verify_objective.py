"""Offline synthetic checks for the coefficient-only ACD objective.

No detector, checkpoint, dataset download, or CUDA is needed. The original
Ultralytics segmentation loss is exercised with fixed assignments and real
mask BCE; only detection assignment and its unrelated losses are fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import weakref
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, help="Offline Ultralytics 8.4.100 package root")
    parser.add_argument("--output", type=Path, help="Optional verification JSON destination")
    args = parser.parse_args()
    if args.vendor:
        sys.path.insert(0, str(args.vendor.resolve()))

    import torch
    import train_acd as acd

    torch.set_num_threads(1)
    original_loss = acd.ORIGINAL_SEG_LOSS
    original_weight = acd.ACD_WEIGHT
    original_enabled = acd.METHOD_ENABLED
    results = []

    class FixedAssignmentCriterion:
        """Use official mask supervision with an auditable fixed assignment."""

        single_mask_loss = staticmethod(acd.v8SegmentationLoss.single_mask_loss)
        calculate_segmentation_loss = acd.v8SegmentationLoss.calculate_segmentation_loss

        def __init__(self, assigned):
            self.device = torch.device("cpu")
            self.stride = torch.tensor([1.0])
            self.hyp = SimpleNamespace(box=7.5)
            self.overlap = True
            self.assigned = assigned
            self.assignment_calls = 0

        def get_assigned_targets_and_loss(self, preds, batch):
            self.assignment_calls += 1
            # Only the independent detection losses are fixed. Segmentation
            # still goes through the official per-instance BCE/crop code.
            return self.assigned, torch.tensor([2.0, 3.0, 4.0]), None

    def fixture(batch_size=1):
        masks = torch.zeros((1, 6, 6))
        masks[0, 1:3, 1:3] = 1
        masks[0, 3:5, 3:5] = 2
        proto = torch.full((1, 3, 6, 6), -2.0)
        proto[0, 0, 1:3, 1:3] = 2.0
        proto[0, 0, 1:3, 3] = 0.25  # Removable FP inside instance-1 support.
        proto[0, 1, 3:5, 3:5] = 2.0
        proto[0, 1, 3:5, 2] = 0.25  # Removable FP inside instance-2 support.
        proto[0, 2] = 0.0
        proto[0, 2, 0, 0] = 1.0
        proto[0, 2, 0, 5] = 1.0  # This entire channel lies outside both boxes.
        coeff = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]]]).repeat(batch_size, 1, 1)
        coeff.requires_grad_()
        assigned = (
            torch.tensor([[True, True]]).repeat(batch_size, 1),
            torch.tensor([[0, 1]]).repeat(batch_size, 1),
            torch.tensor([[[1.0, 1.0, 4.0, 3.0], [2.0, 3.0, 5.0, 5.0]]]).repeat(batch_size, 1, 1),
            torch.zeros((2, 2)),
            torch.ones((2, 1)),
        )
        preds = {"mask_coefficient": coeff, "proto": proto.repeat(batch_size, 1, 1, 1),
                 "feats": [torch.zeros((batch_size, 1, 6, 6))]}
        batch = {"masks": masks.repeat(batch_size, 1, 1),
                 "batch_idx": torch.arange(batch_size).repeat_interleave(2)}
        return FixedAssignmentCriterion(assigned), preds, batch

    def exact_pair(actual, expected, label):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), label

    def reset_stats(branch):
        acd.STATS[branch] = {"loss_calls": 0, "candidate_count": 0,
                             "action_positive": 0, "pixel_terms": 0, "aux_sum": 0.0}

    try:
        for branch in ("one2many", "one2one"):
            criterion, preds, batch = fixture()
            model = torch.nn.Module().train()
            acd.LOSS_MODELS[criterion] = (weakref.ref(model), branch)
            reset_stats(branch)

            # Both nonzero instance IDs must survive selection individually.
            mask_one = acd._get_gt_mask(criterion, batch["masks"], batch, 0, 0)
            mask_two = acd._get_gt_mask(criterion, batch["masks"], batch, 0, 1)
            assert torch.equal(mask_one, batch["masks"][0] == 1)
            assert torch.equal(mask_two, batch["masks"][0] == 2)
            assert int(mask_one.sum()) == int(mask_two.sum()) == 4
            assert not bool((mask_one & mask_two).any())

            aux = acd._acd_aux(criterion, preds, batch, criterion.assigned, branch)
            expected_aux = torch.nn.functional.softplus(torch.tensor(0.25))
            assert torch.allclose(aux, expected_aux, atol=1e-7, rtol=0)
            assert acd.STATS[branch]["candidate_count"] == 2
            assert acd.STATS[branch]["action_positive"] == 2
            assert acd.STATS[branch]["pixel_terms"] == 4
            gradient = torch.autograd.grad(aux, preds["mask_coefficient"])[0]
            assert bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0
            assert float(gradient[0, 0, 0]) > 0 and float(gradient[0, 1, 1]) > 0
            assert torch.equal(gradient[:, 2], torch.zeros_like(gradient[:, 2]))

            # Arbitrarily large outside-box logits cannot influence selection
            # or the penalty, even while those pixels are positive foreground.
            outside_preds = dict(preds)
            outside_coeff = preds["mask_coefficient"].detach().clone()
            outside_coeff[:, 2] = 500.0
            outside_coeff.requires_grad_()
            outside_preds["mask_coefficient"] = outside_coeff
            outside_aux = acd._acd_aux(criterion, outside_preds, batch, criterion.assigned, branch)
            assert torch.equal(aux.detach(), outside_aux.detach())

            reference = original_loss(criterion, preds, batch)
            assert tuple(reference[0].shape) == (5,)
            acd.METHOD_ENABLED = True
            acd.ACD_WEIGHT = original_weight
            reset_stats(branch)
            calls_before = criterion.assignment_calls
            patched = acd.patched_loss(criterion, preds, batch)
            assert criterion.assignment_calls - calls_before == 1, "Assignment repeated inside patched loss"
            assert "get_assigned_targets_and_loss" not in criterion.__dict__, "Temporary getter leaked"
            other_indices = torch.tensor([0, 2, 3, 4])
            assert torch.equal(patched[0][other_indices], reference[0][other_indices])
            assert torch.equal(patched[1][other_indices], reference[1][other_indices])
            weighted = original_weight * expected_aux
            assert torch.allclose(patched[0][1] - reference[0][1], weighted, atol=1e-6, rtol=0)
            assert torch.allclose(patched[1][1] - reference[1][1], weighted, atol=1e-6, rtol=0)
            added_grad = torch.autograd.grad((patched[0] - reference[0]).sum(),
                                            preds["mask_coefficient"])[0]
            assert torch.allclose(added_grad, original_weight * gradient, atol=1e-7, rtol=0)

            acd.METHOD_ENABLED = False
            exact_pair(acd.patched_loss(criterion, preds, batch), reference, "Disabled method changed loss")
            acd.METHOD_ENABLED = True
            model.eval()
            exact_pair(acd.patched_loss(criterion, preds, batch), reference, "Eval mode changed loss")
            model.train()
            with torch.no_grad():
                exact_pair(acd.patched_loss(criterion, preds, batch), reference, "No-grad evaluation changed loss")
            acd.ACD_WEIGHT = 0.0
            exact_pair(acd.patched_loss(criterion, preds, batch), reference, "Zero weight changed loss")

            # The official optimization vector scales by batch size; displayed
            # per-image loss items do not. Check the factor with two images.
            acd.ACD_WEIGHT = original_weight
            pair_criterion, pair_preds, pair_batch = fixture(batch_size=2)
            acd.LOSS_MODELS[pair_criterion] = (weakref.ref(model), branch)
            pair_reference = original_loss(pair_criterion, pair_preds, pair_batch)
            pair_patched = acd.patched_loss(pair_criterion, pair_preds, pair_batch)
            assert torch.equal(pair_patched[0][other_indices], pair_reference[0][other_indices])
            assert torch.equal(pair_patched[1][other_indices], pair_reference[1][other_indices])
            assert torch.allclose(pair_patched[0][1] - pair_reference[0][1], 2 * weighted, atol=1e-6, rtol=0)
            assert torch.allclose(pair_patched[1][1] - pair_reference[1][1], weighted, atol=1e-6, rtol=0)

            # No covered foreground means no accepted action, with a valid
            # differentiable zero instead of a detached constant.
            empty_preds = dict(preds)
            empty_coeff = torch.zeros_like(preds["mask_coefficient"], requires_grad=True)
            empty_preds["mask_coefficient"] = empty_coeff
            empty_aux = acd._acd_aux(criterion, empty_preds, batch, criterion.assigned, branch)
            assert float(empty_aux.detach()) == 0.0
            empty_grad = torch.autograd.grad(empty_aux, empty_coeff)[0]
            assert torch.equal(empty_grad, torch.zeros_like(empty_grad))

            results.append({"branch": branch, "instance_ids": [1, 2],
                            "accepted_instances": 2, "penalized_pixels": 4,
                            "auxiliary_loss": float(aux.detach()),
                            "coefficient_gradient_l1": float(gradient.abs().sum()),
                            "outside_support_gradient_exact_zero": True,
                            "only_segmentation_component_changes": True,
                            "batch_two_optimization_and_display_scaling_correct": True,
                            "no_eligible_action_differentiable_zero": True,
                            "assignment_calls_per_patched_loss": 1,
                            "disabled_eval_no_grad_zero_weight_exact": True})
    finally:
        acd.METHOD_ENABLED = original_enabled
        acd.ACD_WEIGHT = original_weight

    training_script = Path(acd.__file__).resolve()
    receipt = {"status": "PASS", "verified_at": datetime.now(timezone.utc).isoformat(),
               "torch": torch.__version__, "ultralytics": acd.ultralytics.__version__,
               "ultralytics_path": str(Path(acd.ultralytics.__file__).resolve()),
               "training_script": str(training_script),
               "training_script_sha256": hashlib.sha256(training_script.read_bytes()).hexdigest(),
               "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "device": "cpu", "checks": results,
               "scope": "Synthetic objective checks with real official mask BCE and fixed assignments; no detector, dataloader, optimizer or AP claim."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print("OBJECTIVE_VERIFICATION", json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
