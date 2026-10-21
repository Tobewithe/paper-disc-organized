"""CPU behavioral checks; no data download, native model training, or GPU work."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from joint_model import (JointMaskBranch, CoefficientAdapter, mask_loss_grids,
                         snapshot_state, assert_state_unchanged)
from adapter_model import loss_grids
from soft_iou_loss import soft_iou_grouped_loss


class TinyProto(nn.Module):
    """Same mask-path interface/names with small tensors for CPU tests."""
    def __init__(self):
        super().__init__()
        self.feat_refine = nn.ModuleList([nn.Conv2d(512, 256, 1), nn.Conv2d(512, 256, 1)])
        self.feat_fuse = nn.Conv2d(256, 8, 1)
        self.cv1 = nn.Conv2d(8, 8, 1)
        self.upsample = nn.ConvTranspose2d(8, 8, 2, 2)
        self.cv2 = nn.Conv2d(8, 8, 1)
        self.cv3 = nn.Conv2d(8, 32, 1)
        self.semseg = None

    def forward(self, features, return_semantic=True):
        x = features[0]
        for branch, feature in zip(self.feat_refine, features[1:]):
            x = x + F.interpolate(branch(feature), size=x.shape[-2:], mode="nearest")
        return self.cv3(self.cv2(self.upsample(self.cv1(self.feat_fuse(x)))))


def fixture(empty=False, inference=False):
    torch.manual_seed(27)
    head = nn.Module()
    head.nm, head.end2end = 32, True
    head.one2one_cv4 = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in (256, 512, 512)])
    head.proto = TinyProto()
    head.eval().requires_grad_(False)
    with torch.inference_mode(inference), torch.no_grad():
        features = [torch.randn(1, c, h, w) for c, h, w in ((256, 4, 4), (512, 2, 2), (512, 1, 1))]
        raw_ids = [] if empty else [0, 16, 20, 0]
        locations = [] if empty else [
            {"level": 0, "row": 0, "column": 0, "level_offset": 0, "channels": 256},
            {"level": 1, "row": 0, "column": 0, "level_offset": 16, "channels": 512},
            {"level": 2, "row": 0, "column": 0, "level_offset": 20, "channels": 512},
            {"level": 0, "row": 0, "column": 0, "level_offset": 0, "channels": 256}]
        coefficients = torch.cat([b(f).view(1, 32, -1) for b, f in zip(head.one2one_cv4, features)], 2)
        cache = {"image_id": 5, "proto": head.proto(features)[0],
                 "c0": coefficients[0].T[raw_ids], "raw_ids": raw_ids, "raw_locations": locations,
                 "head_feature_shapes": [list(f.shape[1:]) for f in features], "input_hw": [32, 32],
                 "boxes_input": torch.tensor([[0., 0., 32., 32.]] * len(raw_ids)).reshape(-1, 4),
                 "scores": torch.full((len(raw_ids),), .5), "classes": torch.arange(len(raw_ids)),
                 "features": [features[x["level"]][0, :, x["row"], x["column"]] for x in locations],
                 "output_ranks": list(range(len(raw_ids)))}
    return head, features, cache


def original_initial():
    torch.manual_seed(0)
    return {"state_dict": CoefficientAdapter("B").state_dict(), "arm": "B", "seed": 0}


class JointModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_arm_initial_replay_and_duplicate_raw_identity(self):
        head, features, cache = fixture()
        before = snapshot_state(head)
        for arm in ("G_budget", "C_native", "PC_native"):
            model = JointMaskBranch(head, arm, original_initial() if arm == "G_budget" else None)
            self.assertEqual(model.verify_initial_replay(cache, features)["status"], "passed")
            output = model(cache, features)
            self.assertTrue(torch.equal(output["current_coefficients"][0], output["current_coefficients"][3]))
            self.assertEqual(len(output["logits"]), 4)
            model.verify_frozen_state(head, before)
        assert_state_unchanged(head, before)

    def test_pc_gradients_reach_proto_and_coefficients_not_inputs_or_base(self):
        head, features, cache = fixture()
        features = [f.detach().requires_grad_() for f in features]
        cache["proto"].requires_grad_()
        cache["c0"].requires_grad_()
        before = snapshot_state(head)
        model = JointMaskBranch(head, "PC_native")
        initial = snapshot_state(model)
        model.train()
        output = model(cache, features)
        loss, _ = model.loss(output, {0: torch.ones(8, 8), 1: torch.zeros(8, 8), 2: torch.ones(8, 8)})
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(all(f.grad is None for f in features))
        self.assertIsNone(cache["proto"].grad)
        self.assertIsNone(cache["c0"].grad)
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        self.assertTrue(any(not torch.equal(initial[n], p) for n, p in model.state_dict().items() if n.startswith("proto.")))
        self.assertTrue(any(not torch.equal(initial[n], p) for n, p in model.state_dict().items() if n.startswith("coefficient_head.")))
        model.verify_frozen_state(head, before)
        self.assertFalse(any(m.training for m in model.modules()))

    def test_c_native_never_registers_proto_or_detection_parameters(self):
        head, features, cache = fixture()
        model = JointMaskBranch(head, "C_native")
        manifest = model.parameter_manifest()
        self.assertTrue(all(n.startswith("coefficient_head.") for n in manifest["trainable_whitelist"]))
        self.assertEqual(manifest["mutable_buffer_whitelist"], [])
        output = model(cache, features)
        self.assertFalse(output["current_proto"].requires_grad)
        self.assertTrue(output["current_coefficients"].requires_grad)

    def test_grids_and_loss_exactly_match_original_at_frozen_p(self):
        head, features, cache = fixture()
        coefficients = cache["c0"].clone().requires_grad_()
        old = loss_grids(cache, coefficients)
        new = mask_loss_grids(cache, cache["proto"], coefficients)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, new)))
        targets = {1: torch.ones(8, 8), 2: torch.zeros(8, 8)}
        old_loss, old_stats = soft_iou_grouped_loss(*old, targets)
        new_loss, new_stats = JointMaskBranch.loss(dict(logits=new[0], baseline_logits=new[1], roi=new[2]), targets)
        self.assertTrue(torch.equal(old_loss, new_loss))
        self.assertEqual(old_stats, new_stats)

    def test_inference_tensors_are_safe_backward_inputs(self):
        head, features, cache = fixture(inference=True)
        for arm in ("G_budget", "C_native", "PC_native"):
            model = JointMaskBranch(head, arm, original_initial() if arm == "G_budget" else None)
            output = model(cache, features)
            loss, _ = model.loss(output, {0: torch.ones(8, 8)})
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_empty_output_retains_differentiable_zero_and_accounting(self):
        head, features, cache = fixture(empty=True)
        for arm in ("G_budget", "C_native", "PC_native"):
            model = JointMaskBranch(head, arm, original_initial() if arm == "G_budget" else None)
            output = model(cache, features)
            loss, counts = model.loss(output, {})
            self.assertEqual(loss.item(), 0)
            self.assertTrue(counts["no_output_image"])
            self.assertEqual(counts["preserved_instances"], 0)
            loss.backward()

    def test_raw_identity_and_geometry_changes_fail_closed(self):
        head, features, cache = fixture()
        model = JointMaskBranch(head, "C_native")
        bad = copy.deepcopy(cache)
        bad["raw_ids"][0] = 1
        with self.assertRaisesRegex(ValueError, "identities disagree"):
            model(bad, features)
        with self.assertRaisesRegex(ValueError, "outside original native F batch"):
            model(cache, features, batch_index=1)

    def test_full_batch_computed_once_with_selected_microgroup_and_gradients(self):
        head, features, cache = fixture()
        full_features = [torch.cat([feature + index * .1 for index in range(4)], 0) for feature in features]
        with torch.no_grad():
            full_p = head.proto(full_features)
            full_c = torch.cat([branch(feature).view(4, 32, -1)
                                for branch, feature in zip(head.one2one_cv4, full_features)], 2)
        caches = []
        for index in (3, 1):
            selected = copy.deepcopy(cache)
            selected["image_id"] = 5 + index
            selected["proto"] = full_p[index]
            selected["c0"] = full_c[index].T[cache["raw_ids"]]
            caches.append(selected)
        model = JointMaskBranch(head, "PC_native")
        seen_batches = []
        hook = model.proto.register_forward_pre_hook(lambda module, args: seen_batches.append(args[0][0].shape[0]))
        outputs = model.forward_batch(caches, full_features, [3, 1])
        hook.remove()
        self.assertEqual(seen_batches, [4])
        self.assertEqual(len(outputs), 2)
        for output, selected in zip(outputs, caches):
            self.assertTrue(torch.equal(output["current_proto"], selected["proto"]))
            self.assertTrue(torch.equal(output["current_coefficients"], selected["c0"]))
            self.assertTrue(torch.equal(output["logits"], output["baseline_logits"]))
        losses = [model.loss(output, {0: torch.ones(8, 8)})[0] for output in outputs]
        (torch.stack(losses).mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "unique"):
            model.forward_batch(caches, full_features, [1, 1])

    def test_refuses_unfrozen_or_wrong_initialization(self):
        head, _, _ = fixture()
        head.train()
        with self.assertRaisesRegex(ValueError, "frozen native eval"):
            JointMaskBranch(head, "PC_native")
        head.eval()
        initial = original_initial()
        initial["state_dict"]["readout.2.bias"] = torch.ones(32)
        with self.assertRaisesRegex(ValueError, "zero-residual"):
            JointMaskBranch(head, "G_budget", initial)


if __name__ == "__main__":
    unittest.main(verbosity=2)
