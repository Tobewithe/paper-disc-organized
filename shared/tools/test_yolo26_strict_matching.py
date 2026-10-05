import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_yolo26_strict_matching import match_image
from analyze_yolo26_strict_quality import QUALITY, STAGES
from run_yolo26_candidate_perturbation import load_legacy


class MatchingCheck(unittest.TestCase):
    def test_correct_neighbor_shares_only_candidate_and_missing_gt_is_separate(self):
        targets = [{"dataset": "test", "image_id": 1, "annotation_id": g, "baseline_class": label, "scene_label": "Touching", **{f"{q}_{s}_count": int(g < 3) for q in QUALITY for s in STAGES}} for g, label in ((1, "C"), (2, "X"), (3, "MISS"))]
        edges = [{"annotation_id": g, "source_candidate_id": 7, "source_score": .9, "global_rank": 1, "iou": 9 / 11, "coverage": .9, **{f"quality_{q}": 1 for q in QUALITY}, **{f"{s}_member": 1 for s in STAGES[1:]}} for g in (1, 2)]
        rows, groups = match_image(load_legacy(), targets, edges)
        for q in QUALITY:
            for s in STAGES:
                selected = [r for r in rows if r["quality"] == q and r["stage"] == s]
                self.assertEqual(sum(r["available"] for r in selected), 2)
                self.assertEqual(sum(r["matched"] for r in selected), 1)
                self.assertEqual(sum(r["assignment_conflict"] for r in selected), 1)
                self.assertEqual(next(r for r in selected if r["annotation_id"] == 3)["assignment_conflict"], 0)
                self.assertEqual(sum(r["gt_count"] for r in groups if r["quality"] == q and r["stage"] == s), 3)


if __name__ == "__main__":
    unittest.main()
