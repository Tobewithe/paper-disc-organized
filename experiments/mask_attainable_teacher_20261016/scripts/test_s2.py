import unittest
from s2_common import decide,validate_ledger


class S2Tests(unittest.TestCase):
    def primary(self,damage='damage_guardrail_passed'):
        return dict(target=dict(mean=.005,ci95=[.0001,.01]),all_bbox75=dict(mean=-.001),damage=dict(gate=damage))
    def test_absolute_pass_is_not_incremental_pass(self):
        self.assertEqual(decide(self.primary(),dict(ci95=[0.,.01]))[0],'failed')
        self.assertEqual(decide(self.primary(),dict(ci95=[.00001,.01]))[0],'passed')
    def test_precision_distinct_from_scientific_failure(self):
        self.assertEqual(decide(self.primary('blocked_insufficient_precision'),dict(ci95=[.001,.02]))[0],'blocked')
        self.assertEqual(decide(self.primary('blocked_insufficient_precision'),dict(ci95=[-.001,.02]))[0],'failed')
    def test_fixed_bbox_identity_and_full_GT_required(self):
        old=dict(image_id=2,annotation_id=3,is_ignored=False,gt_category_id=1,gt_area=5.,task_box75='success',bbox75_match_output_id='a:17')
        new={**old,'bbox75_match_output_id':'b:0'}
        validate_ledger([new],[old],[2],[17],'a')
        with self.assertRaises(ValueError):validate_ledger([new],[old],[2],[18],'a')
        with self.assertRaises(ValueError):validate_ledger([], [old],[2],[17],'a')


if __name__=='__main__':unittest.main()
