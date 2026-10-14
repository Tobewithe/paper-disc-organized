"""Behavioral checks on identities, boundaries and clustered multiplicities."""
import unittest
import numpy as np
from stratified_stats import area_group, crowd_group, neighbors, aggregate, rate


class Checks(unittest.TestCase):
    def test_disjoint_boundaries(self):
        self.assertEqual([area_group(v) for v in (0,1023.9,1024,9215.9,9216)], ['S','S','M','M','L'])
        self.assertEqual([crowd_group(v) for v in (0,.099,.1,.499,.5,1)], ['Low','Low','Mid','Mid','High','High'])
        with self.assertRaises(ValueError):
            crowd_group(1.01)

    def test_same_raw_other_class_is_not_a_neighbor(self):
        got = neighbors([[0,0,10,10],[0,0,10,10],[0,0,5,10],[20,20,30,30]],
                        [0,1,1,0], [8,8,9,10], [0])[0]
        self.assertEqual(got['all_class_neighbor_iou'], .5)
        self.assertEqual(got['same_class_neighbor_iou'], 0.)
        self.assertEqual(got['different_raw_neighbors'], 2)
        alone = neighbors([[0,0,1,1]], [0], [8], [0])[0]
        self.assertEqual(alone['all_class_neighbor_iou'], 0.)

    def test_cluster_repeats_and_macro_differ_from_pair_average(self):
        rows = [dict(image_id=1), dict(image_id=1), dict(image_id=2)]
        frame = dict(ids=[1,2], index={1:0,2:1}, weights=np.tile([[2.,0.],[0.,2.],[1.,1.],[1.,1.]], (500,1)))
        point, draws = aggregate(rows, [0.,1.,1.], frame, True)
        self.assertEqual(point, .75)
        np.testing.assert_array_equal(draws[:4], [.5,1.,.75,.75])
        point, draws = aggregate(rows, [0.,1.,1.], frame, False)
        self.assertAlmostEqual(point, 2/3)
        np.testing.assert_allclose(draws[:4], [.5,1.,2/3,2/3])
        empty, samples = aggregate([], [], frame, False)
        self.assertIsNone(empty)
        self.assertTrue(np.isnan(samples).all())

    def test_zero_damage_is_not_zero_risk(self):
        rows = [dict(image_id=1)]*10
        frame = dict(ids=[1], index={1:0}, weights=np.ones((2000,1)))
        answer = rate(rows, [False]*10, frame)
        self.assertEqual(answer['rate'], 0.)
        self.assertTrue(answer['degenerate_cluster_rate'])
        self.assertGreater(answer['exact_binomial95_supplement'][1], .3)


if __name__ == '__main__':
    unittest.main()
