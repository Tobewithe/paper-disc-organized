"""Non-scientific fixtures for denominators, image clustering and routing."""
import unittest
from audit_stats import area_group,iou_group,effect,damage,standardized_gap,adjudicate

class Tests(unittest.TestCase):
    def test_integer_rle_accepts_native_bytes_and_json_strings(self):
        from verify import area,iou
        a=dict(size=[2,3],counts=b'0220')
        b=dict(size=[2,3],counts='123')
        self.assertEqual(area(a),4)
        self.assertEqual(iou(a,b),.2)
    def row(self,image,delta,cat=1):
        return dict(image_id=image,baseline_iou=.6,U_iou=.6+delta,G_iou=.6+delta,population='T',gate_enabled=True,category_id=cat,native_iou_group='50to75')
    def test_macro_not_pair_mean(self):
        r=effect([self.row(1,.1),self.row(1,.1),self.row(2,-.1)],'U_iou')
        self.assertAlmostEqual(r['mean'],0)
        self.assertEqual(r['images'],2)
    def test_area_and_iou_boundaries(self):
        self.assertEqual([area_group(x) for x in (1023,1024,9215,9216)],['S','M','M','L'])
        self.assertEqual([iou_group(x) for x in (.499,.5,.749,.75)],['lt50','50to75','50to75','ge75'])
    def test_task_damage_not_fixed_pair(self):
        d=damage([dict(baseline_success=True,U_success=False),dict(baseline_success=True,U_success=True),dict(baseline_success=False,U_success=True)],'U')
        self.assertEqual((d['damaged_k'],d['population_N'],d['recovered_k']),(1,2,1))
    def test_standardization_uses_dev_weights_and_clusters(self):
        a=[self.row(i,.1,1) for i in range(40)]+[self.row(100+i,.02,2) for i in range(20)]
        b=[self.row(200+i,.03,1) for i in range(20)]+[self.row(300+i,.01,2) for i in range(40)]
        g=standardized_gap(a,b)
        self.assertAlmostEqual(g['mean'],.03)
        self.assertTrue(g['sufficient_common_support'])
        self.assertEqual(g['valid_resamples'],2000)
    def test_fit_failure_has_priority(self):
        d=adjudicate({'G':{'fit_thresholds':{'a':False}}},dict(sufficient_common_support=True,mean=.02,ci95=[.01,.03]))
        self.assertEqual(d['finding'],'repair_requirements_already_unmet_on_fit')
    def test_sparse_support_not_positive(self):
        g=standardized_gap([self.row(i,.2) for i in range(9)],[self.row(100+i,0) for i in range(40)])
        self.assertFalse(g['sufficient_common_support'])
        self.assertIsNone(g['mean'])

if __name__=='__main__':
    unittest.main()
