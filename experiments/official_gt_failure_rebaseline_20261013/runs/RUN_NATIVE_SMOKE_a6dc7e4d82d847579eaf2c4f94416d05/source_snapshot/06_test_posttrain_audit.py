"""Synthetic regressions for identity, denominators, COCO matching and oracle safety."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared" / "evaluation"))
from posttrain_audit import check_run_identity, fingerprint_base_state, compare_outputs, make_panel, population, summarize_oracle, write_new


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ledger = module("experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py", "ledger")
solver = module("experiments/mask_readout_mechanism_v2_20261012/scripts/finite_coefficient_solver.py", "solver")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_COCO = HAS_NUMPY and importlib.util.find_spec("pycocotools") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
SOURCE = "a" * 64


def gt(aid, mask="failure", box=True, miou=.5, ignored=False):
    return dict(image_id=aid // 2 + 1, annotation_id=aid, task_mask75=mask,
        bbox75_match_output_id=f"output:{aid}" if box else None,
        box_iou_of_bbox_match=.8 if box else None,
        mask_iou_of_bbox_match=miou if box else None,
        is_ignored=ignored, gt_area=100., gt_category_id=1, source_prediction_sha=SOURCE)


def evidence(panel):
    controls = dict(schema_version=1, positive_control_passed=True, same_solver_config=True,
        native_zero_delta_passed=True, known_coefficient_mask_control_passed=True, solver_code_sha256="b"*64, oracle_config_sha256="c"*64)
    rows = [dict(image_id=t["image_id"], annotation_id=t["annotation_id"],
        output_detection_id=t["output_detection_id"], source_prediction_sha=SOURCE,
        baseline_mask_iou=t["baseline_mask_iou"], baseline_replay_equal=True,
        solver_code_sha256=controls["solver_code_sha256"], oracle_config_sha256=controls["oracle_config_sha256"],
        solver_status="converged", oracle_mask_iou=.8, delta_norm=1., gradient_inf_norm=1e-8)
        for t in panel["targets"]]
    return rows, controls


class CoreChecks(unittest.TestCase):
    def test_population_requires_task_failure_and_same_detection(self):
        self.assertEqual(population(gt(1, mask="success", miou=.2)), "S")
        self.assertEqual(population(gt(2, miou=.75)), "U")
        self.assertEqual(population(gt(3, box=False)), "F_without_box75")
        self.assertEqual(population(gt(4)), "T")
        self.assertEqual(population(gt(5, ignored=True)), "ignored")

    def test_no_duplicate_gt_or_output(self):
        with self.assertRaises(ValueError):
            make_panel([gt(1), gt(1)], SOURCE)
        rows = [gt(1), gt(2)]
        rows[1]["bbox75_match_output_id"] = rows[0]["bbox75_match_output_id"]
        with self.assertRaises(ValueError):
            make_panel(rows, SOURCE)

    def test_sampled_opportunities_not_extrapolated(self):
        panel = make_panel([gt(i) for i in range(10)] + [gt(20, box=False)], SOURCE, 2)
        rows, controls = evidence(panel)
        result = summarize_oracle(panel, rows, controls)
        self.assertEqual(result["found_fraction_of_selected_T"], 1.)
        self.assertAlmostEqual(result["observed_found_fraction_of_all_failures"], 2/11)
        self.assertEqual(result["gate_candidate_status"], "blocked_insufficient_sample")

    def test_solver_and_identity_fail_closed(self):
        panel = make_panel([gt(i) for i in range(200)], SOURCE)
        rows, controls = evidence(panel)
        self.assertEqual(summarize_oracle(panel, rows, controls)["gate_candidate_status"], "opportunity_threshold_met")
        with self.assertRaises(ValueError):
            summarize_oracle(panel, rows[:-1], controls)
        with self.assertRaises(ValueError):
            summarize_oracle(panel, rows, {**controls, "positive_control_passed": False})
        altered = copy.deepcopy(rows)
        altered[0]["output_detection_id"] = "another_prediction"
        with self.assertRaises(ValueError):
            summarize_oracle(panel, altered, controls)
        altered = copy.deepcopy(rows)
        altered[0]["oracle_mask_iou"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_oracle(panel, altered, controls)

    def test_budget_exhaustion_is_not_scientific_negative(self):
        panel = make_panel([gt(i) for i in range(200)], SOURCE)
        rows, controls = evidence(panel)
        for r in rows:
            r.update(oracle_mask_iou=.6, solver_status="budget_exhausted")
        self.assertEqual(summarize_oracle(panel, rows, controls)["gate_candidate_status"], "blocked_solver_budget")
        rows[0]["solver_status"] = "numerical_error"
        self.assertEqual(summarize_oracle(panel, rows, controls)["gate_candidate_status"], "blocked_numerical_error")

    def test_output_rows_not_deduplicated_by_geometry(self):
        a = dict(image_id=1, category_id=1, score=.5, bbox=[0, 0, 10, 10],
                 segmentation={"size": [10, 10], "counts": "mock-rle"})
        rows = [a, {**a, "category_id": 2}]
        self.assertEqual(compare_outputs(rows, copy.deepcopy(rows), True)["rows"], 2)
        with self.assertRaises(ValueError):
            compare_outputs(rows, rows[:1])
        with self.assertRaises(ValueError):
            compare_outputs(rows, list(reversed(rows)))

    def test_zero_delta_and_no_overwrite(self):
        a = dict(image_id=1, category_id=1, score=.5, bbox=[0, 0, 10, 10],
                 segmentation={"size": [10, 10], "counts": "a"})
        b = {**a, "segmentation": {"size": [10, 10], "counts": "b"}}
        compare_outputs([a], [b])
        with self.assertRaises(ValueError):
            compare_outputs([a], [b], True)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p = Path(temp)/"result.json"
            write_new(p, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_new(p, {"x": 2})

    def test_model_buffers_modes_whitelist_and_leakage(self):
        state = {"tensors": {"bn.running_mean": {"sha256": "a"}},
                 "training_modes": {"": False, "bn": False}}
        check_run_identity(state, copy.deepcopy(state), [1, 2], [3], ["adapter.weight"], ["adapter.weight"])
        with self.assertRaises(ValueError):
            check_run_identity(state, state, [1, 2], [2], ["adapter.weight"], ["adapter.weight"])
        changed = copy.deepcopy(state)
        changed["tensors"]["bn.running_mean"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            check_run_identity(state, changed, [1], [2], ["adapter.weight"], ["adapter.weight"])
        with self.assertRaises(ValueError):
            check_run_identity(state, state, [1], [2], ["base.weight"], ["adapter.weight"])

    def test_coco_eval_extraction_uses_ids_and_ignores(self):
        item = dict(image_id=1, aRng=[0, 1e10], maxDet=100,
                    gtIds=[10, 11], gtIgnore=[0, 1], dtIds=[8, 9],
                    gtMatches=[[8, 9], [8, 9]], dtMatches=[[10, 11], [10, 11]],
                    dtIgnore=[[0, 1], [0, 1]])
        ev = SimpleNamespace(params=SimpleNamespace(iouThrs=[.5, .75], areaRng=[[0, 1e10]]), evalImgs=[item])
        rows = ledger.extract_matches(ev)
        self.assertEqual(rows[(1, 10)]["d75"], 8)
        self.assertTrue(rows[(1, 11)]["ignore"])
        item["dtIgnore"][1][0] = 1
        with self.assertRaises(ValueError):
            ledger.extract_matches(ev)


@unittest.skipUnless(HAS_COCO, "requires local numpy + pycocotools; not claimed as executed")
class RealCOCOChecks(unittest.TestCase):
    def test_independent_pairs_budget_crowd_and_empty_predictions(self):
        import numpy as np
        from pycocotools import mask as mu
        def rle(full):
            m = np.zeros((10, 10), dtype=np.uint8)
            m[:, :10 if full else 2] = 1
            r = mu.encode(np.asfortranarray(m))
            r["counts"] = r["counts"].decode()
            return r
        images = [dict(id=i, height=10, width=10) for i in range(1, 6)]
        anns = [dict(id=i, image_id=i, category_id=1, bbox=[0,0,10,10],
                     area=100, segmentation=rle(True), iscrowd=int(i == 4)) for i in range(1, 6)]
        def pred(i, score, full, box=None):
            return dict(image_id=i, category_id=1, score=score, bbox=box or [0,0,10,10], segmentation=rle(full))
        preds = [pred(1,.9,False), pred(1,.8,True,[20,20,10,10]), pred(2,.9,False),
                 pred(4,.9,True)] + [pred(5,.99-j*.001,False) for j in range(100)] + [pred(5,.01,True)]
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp)/"gt.json"
            path.write_text(json.dumps(dict(info={}, images=images, annotations=anns,
                                           categories=[dict(id=1,name="object")])), encoding="utf-8")
            rows, _ = ledger.evaluate_ledger(path, preds, [1,2,3,4,5], SOURCE)
            r = {x["annotation_id"]: x for x in rows}
            self.assertNotEqual(r[1]["bbox75_match_output_id"], r[1]["segm75_match_output_id"])
            self.assertEqual(r[1]["population"], "S")
            self.assertEqual(r[2]["population"], "T")
            self.assertEqual(r[3]["population"], "F_without_box75")
            self.assertTrue(r[4]["is_ignored"])
            self.assertTrue(r[5]["correct_class_mask75_present_pre_coco_budget"])
            self.assertFalse(r[5]["correct_class_mask75_present_within_coco_maxdet100"])
            empty, _ = ledger.evaluate_ledger(path, [], [1,2,3,4,5], SOURCE)
            self.assertEqual(sum(x["task_mask75"] == "failure" for x in empty), 4)


@unittest.skipUnless(HAS_TORCH, "requires local torch; not claimed as executed")
class SolverChecks(unittest.TestCase):
    def test_known_nonzero_regularized_optimum(self):
        self.assertTrue(solver.positive_control()["passed"])

    @unittest.skipUnless(HAS_NUMPY, "fingerprint hook additionally requires numpy")
    def test_state_hook_captures_bn_buffers(self):
        import torch
        base = torch.nn.BatchNorm1d(2).eval()
        before = fingerprint_base_state(base)
        base.running_mean.add_(1)
        self.assertNotEqual(before, fingerprint_base_state(base))


if __name__ == "__main__":
    if "--require-integration" in sys.argv:
        sys.argv.remove("--require-integration")
        if not HAS_COCO or not HAS_TORCH:
            raise SystemExit("BLOCKED: --require-integration requires numpy, pycocotools and torch")
    unittest.main()
