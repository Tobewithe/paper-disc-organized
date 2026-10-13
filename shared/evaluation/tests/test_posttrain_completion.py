"""Meaningful regression tests for damage acceptance, spatial ablation and replay."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/evaluation"))
sys.path.insert(0, str(ROOT / "experiments/mask_readout_mechanism_v2_20261012/scripts"))
from damage_acceptance import (damage_upper_bound, hypergeom_cdf_counts,
                               score_damage, select_damage_panel)
from native_output_bridge import decode_native, encode_rle, native_environment, raw_location
from adapter_primitives import grouped_mask_loss, sample_spatial_roi, shuffle_spatial, spatial_permutation

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NATIVE = HAS_TORCH and all(importlib.util.find_spec(p) for p in ("numpy", "ultralytics", "pycocotools"))


def baseline(n=200):
    return [dict(image_id=i // 4, annotation_id=i, is_ignored=False,
                 task_mask75="success", gt_category_id=1, source_prediction_sha="a"*64)
            for i in range(n)]


class ExactDamageChecks(unittest.TestCase):
    def test_integer_cdf_matches_enumeration(self):
        from itertools import combinations
        for N in range(1, 9):
            for n in range(1, N + 1):
                for K in range(N + 1):
                    counts = [sum(i < K for i in sample) for sample in combinations(range(N), n)]
                    for x in range(n + 1):
                        a, b = hypergeom_cdf_counts(N, K, n, x)
                        self.assertEqual(a, sum(c <= x for c in counts))
                        self.assertEqual(b, len(counts))

    def test_exact_coverage_for_every_small_finite_population(self):
        # Exhaust all population counts; arbitrary clustering does not enter SRSWOR design.
        for N in range(1, 21):
            for n in range(1, N + 1):
                bounds = [damage_upper_bound(N, n, x)["upper_damaged_count"] for x in range(n + 1)]
                self.assertEqual(bounds, sorted(bounds))
                for K in range(N + 1):
                    misses = sum(math.comb(K, x) * math.comb(N-K, n-x)
                                 for x in range(max(0, n-N+K), min(K, n)+1)
                                 if bounds[x] < K)
                    self.assertLessEqual(misses * 20, math.comb(N, n))

    def test_zero_events_census_and_invalid_counts(self):
        self.assertGreater(damage_upper_bound(10000, 1024, 0)["upper_rate"], 0)
        self.assertLess(damage_upper_bound(10000, 1024, 0)["upper_rate"], .01)
        self.assertGreater(damage_upper_bound(10000, 100, 0)["upper_rate"], .01)
        for x in (0, 1, 50, 100):
            self.assertEqual(damage_upper_bound(100, 100, x)["upper_damaged_count"], x)
        for values in ((0, 0, 0), (5, 6, 0), (5, 2, 3), (True, 1, 0)):
            with self.assertRaises(ValueError):
                damage_upper_bound(*values)

    def test_sample_integrity_full_image_evaluation_and_gate(self):
        rows = baseline()
        panel = select_damage_panel(rows, "b"*64, "c"*64, "d"*64, n=30, seed=17)
        method = [copy.deepcopy(r) for r in rows if r["image_id"] in panel["image_ids"]]
        result = score_damage(panel, rows, method)
        self.assertEqual(result["counted_GT"], 30)
        self.assertGreater(result["evaluated_GT"], 30)
        self.assertEqual(result["gate"], "blocked_insufficient_precision")
        with self.assertRaises(ValueError):
            score_damage(panel, rows, method[:-1])
        broken = copy.deepcopy(panel)
        broken["sample"].reverse()
        with self.assertRaises(ValueError):
            score_damage(broken, rows, method)
        full = select_damage_panel(rows, "b"*64, "c"*64, "d"*64, n=200, seed=17)
        self.assertEqual(score_damage(full, rows, rows)["gate"], "damage_guardrail_passed")
        damaged = copy.deepcopy(rows)
        for r in damaged[:3]:
            r["task_mask75"] = "failure"
        self.assertEqual(score_damage(full, rows, damaged)["gate"], "damage_guardrail_failed")
        damaged[0]["task_mask75"] = "ignored"
        with self.assertRaises(ValueError):
            score_damage(full, rows, damaged)


class IdentityAndAblationChecks(unittest.TestCase):
    def test_raw_id_crosses_real_level_boundaries(self):
        shapes = [[8, 2, 3], [16, 1, 2], [32, 1, 1]]
        self.assertEqual(raw_location(5, shapes), dict(level=0, row=1, column=2, level_offset=0, channels=8))
        self.assertEqual(raw_location(6, shapes)["channels"], 16)
        self.assertEqual(raw_location(8, shapes)["level"], 2)
        with self.assertRaises(ValueError):
            raw_location(9, shapes)

    def test_spatial_keys_are_reproducible_bijections(self):
        a = spatial_permutation(8, 8, 25, 2, "train:0")
        self.assertEqual(sorted(a), list(range(64)))
        self.assertEqual(a, spatial_permutation(8, 8, 25, 2, "train:0"))
        self.assertNotEqual(a, spatial_permutation(8, 8, 25, 2, "train:1"))
        self.assertNotEqual(a, spatial_permutation(8, 8, 25, 3, "train:0"))


@unittest.skipUnless(HAS_TORCH, "requires torch; no tensor checks claimed")
class TensorChecks(unittest.TestCase):
    def test_joint_spatial_shuffle_preserves_vectors_and_logit_relation(self):
        import torch
        p = torch.arange(128., dtype=torch.float32).reshape(2, 8, 8)
        z = p[0:1] * .5 - p[1:2] * 2
        x = torch.cat((p, z))
        y = shuffle_spatial(x, 10, 1, "eval:0")
        self.assertTrue(torch.equal(y[2], y[0] * .5 - y[1] * 2))
        self.assertTrue(torch.equal(x.flatten(1).sort().values, y.flatten(1).sort().values))
        self.assertFalse(torch.equal(x, y))

    def test_unmatched_outputs_do_not_dilute_GT_gradient(self):
        import torch
        gradients, losses = [], []
        for n in (1, 101):
            logits = torch.zeros(n, 2, 2, requires_grad=True)
            loss, counts = grouped_mask_loss(logits, logits.detach(), torch.ones_like(logits, dtype=torch.bool),
                                            {0: torch.ones(2, 2)})
            loss.backward()
            gradients.append(logits.grad[0].clone())
            losses.append(counts["gt_loss"])
            self.assertEqual(counts["matched_supervised_instances"], 1)
            self.assertEqual(counts["preserved_instances"], n)
        self.assertEqual(losses[0], losses[1])
        self.assertTrue(torch.equal(gradients[0], gradients[1]))

    def test_empty_rois_and_feature_sampling(self):
        import torch
        x = torch.zeros(2, 3, 3, requires_grad=True)
        loss, counts = grouped_mask_loss(x, x.detach(), torch.zeros_like(x, dtype=torch.bool), {})
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(counts["empty_roi_rows"], [0, 1])
        p = torch.ones(2, 4, 4)
        c = torch.tensor([[2., 3.]])
        features = sample_spatial_roi(p, c, torch.tensor([[4., 4., 12., 12.]]), (16, 16))
        self.assertTrue(torch.allclose(features[0, 2], torch.full((8, 8), 5.)))


@unittest.skipUnless(HAS_NATIVE, "requires pinned Ultralytics, torch, numpy, pycocotools")
class NativeDecoderChecks(unittest.TestCase):
    def test_nonzero_coefficients_against_real_native_serializer(self):
        import torch
        from ultralytics.models.yolo.segment.val import SegmentationValidator
        from ultralytics.utils import ops
        native_environment()
        v = object.__new__(SegmentationValidator)
        v.args = SimpleNamespace(conf=.001, iou=.7, task="segment", single_cls=False,
                                 agnostic_nms=False, max_det=300)
        v.nc, v.end2end, v.process = 80, True, ops.process_mask_native
        v.class_map, v.jdict = list(range(1, 81)), []
        p = torch.linspace(-1, 1, 24).reshape(1, 4, 6)
        c = torch.tensor([[1.], [-1.]])
        boxes = torch.tensor([[0., 0., 24., 16.], [3., 2., 20., 14.]])
        final = torch.cat((boxes, torch.tensor([[.9, 0.], [.8, 1.]]), c), 1)[None]
        pred = v.postprocess(((final, p[None]), {}))[0]
        geometry = dict(imgsz=(16, 24), ori_shape=(9, 15), ratio_pad=((1.6, 1.6), (0., 1.)), im_file="1.jpg")
        scaled = v.scale_preds(pred, geometry)
        v.pred_to_json(scaled, geometry)
        cache = dict(proto=p, c0=c, boxes_input=boxes, native_decode_hw=[16, 24],
                     original_hw=[9, 15], ratio_pad=geometry["ratio_pad"], device="cpu")
        replayed = decode_native(cache)
        self.assertEqual([encode_rle(m) for m in replayed], [r["segmentation"] for r in v.jdict])
        changed = decode_native(cache, -c)
        self.assertFalse(torch.equal(changed, replayed))


if __name__ == "__main__":
    if "--require-integration" in sys.argv:
        sys.argv.remove("--require-integration")
        if not HAS_TORCH or not HAS_NATIVE:
            raise SystemExit("BLOCKED: torch + numpy + pycocotools + pinned ultralytics required")
    unittest.main()
