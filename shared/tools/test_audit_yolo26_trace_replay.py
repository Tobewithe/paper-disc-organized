import importlib.util
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location("trace_replay", Path(__file__).with_name("audit_yolo26_trace_replay.py"))
assert SPEC and SPEC.loader
trace_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_replay)


class TraceReplayAuditTests(unittest.TestCase):
    def test_array_audit_is_dtype_and_value_exact(self) -> None:
        value = np.array([1.0], dtype=np.float32)
        self.assertTrue(trace_replay._array_audit(value, value.copy())["match"])
        self.assertFalse(trace_replay._array_audit(value, value.astype(np.float64))["match"])
        self.assertFalse(trace_replay._array_audit(value, value + 0.1)["match"])

    def test_mask_audit_reports_xor_and_threshold_effect(self) -> None:
        replayed = [np.array([[1, 0], [0, 0]], dtype=bool)]
        expected = [np.array([[1, 1], [0, 0]], dtype=bool)]
        audit = trace_replay._mask_replay_audit(replayed, expected, [7])
        self.assertFalse(audit["all_exact"])
        self.assertEqual(audit["xor_pixels_total"], 1)
        self.assertEqual(audit["iou_threshold_pass_count"]["0.50"], 1)
        self.assertEqual(audit["iou_threshold_pass_count"]["0.75"], 0)


if __name__ == "__main__":
    unittest.main()
