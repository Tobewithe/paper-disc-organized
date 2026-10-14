"""Decision-critical checks: task success census, immutable pairing, and diagnostic routing."""
import unittest
from run_fit_diagnostic import adjudicate, damage_census, fixed_effects


def row(aid, task, box=None, ignored=False):
    return dict(image_id=1, annotation_id=aid, is_ignored=ignored, task_mask75=task,
                bbox75_match_output_id=box, box_iou_of_bbox_match=.9 if box else None,
                mask_iou_of_bbox_match=.6 if box else None)


class DiagnosticTests(unittest.TestCase):
    def test_success_without_bbox_pair_is_in_damage_denominator(self):
        before = [row(1, 'success'), row(2, 'success', 'native:10'), row(3, 'ignored', ignored=True)]
        after = [row(1, 'failure'), row(2, 'success', 'B:0'), row(3, 'ignored', ignored=True)]
        result = damage_census(before, after)
        self.assertEqual((result['damaged_k'], result['population_N'], result['fraction']), (1, 2, .5))
        self.assertEqual(result['observed_n'], result['population_N'])

    def test_changed_detection_identity_is_rejected(self):
        before = [row(1, 'failure', 'native:10')]
        with self.assertRaises(Exception):
            fixed_effects(before, [row(1, 'failure', 'B:0')], [11], 'native')

    def test_loss_decrease_does_not_override_failed_repair(self):
        d = adjudicate({'mean': -.001, 'ci95': [-.002, 0]}, {'mean': -.003}, {'fraction': .07}, -.05)
        self.assertTrue(d['training_objective_lowered'])
        self.assertFalse(d['fit_repair_supported'])
        self.assertEqual(d['localization'], 'repair_requirements_already_unmet_on_fit')
        self.assertTrue(d['prior_S2_negative_decision_retained'])

    def test_fit_pass_routes_to_gap_without_reopening_S2(self):
        d = adjudicate({'mean': .006, 'ci95': [.002, .01]}, {'mean': 0}, {'fraction': .005}, -.01)
        self.assertTrue(d['fit_repair_supported'])
        self.assertEqual(d['localization'], 'fit_meets_requirements_dev_does_not')
        self.assertTrue(d['prior_S2_negative_decision_retained'])


if __name__ == '__main__':
    unittest.main()
