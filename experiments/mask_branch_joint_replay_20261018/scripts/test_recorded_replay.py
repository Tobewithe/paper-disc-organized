"""Regression checks for equal-aspect cross-batch swaps and aligned dataset state."""
import copy
from pathlib import Path
import unittest
from joint_data import ROOT, read, historical_val_plan, apply_recorded_plan, dataset_plan


class RecordedReplayTests(unittest.TestCase):
    def test_reorder_all_aligned_state_and_cross_batch_rectangles(self):
        class Dataset:
            augment = False
            buffer = []
            def __len__(self): return len(self.im_files)
        d = Dataset()
        scrambled = list(range(19)); scrambled[3], scrambled[17] = scrambled[17], scrambled[3]
        d.im_files = [f'/data/{i:012d}.jpg' for i in scrambled]
        d.labels = [{'im_file': p, 'identity': i} for i, p in zip(scrambled, d.im_files)]
        for key in ('ims', 'im_hw0', 'im_hw', 'npy_files'): setattr(d, key, list(scrambled))
        expected = [{'source_batch_index': 0, 'indices': list(range(16)), 'image_ids': list(range(16)), 'rect_shape': [352, 672]},
                    {'source_batch_index': 1, 'indices': [16, 17, 18], 'image_ids': [16, 17, 18], 'rect_shape': [384, 672]}]
        apply_recorded_plan(d, expected)
        self.assertEqual(dataset_plan(d), expected)
        self.assertEqual([x['identity'] for x in d.labels], list(range(19)))
        for key in ('ims', 'im_hw0', 'im_hw', 'npy_files'): self.assertEqual(getattr(d, key), list(range(19)))
        self.assertEqual(d.batch.tolist(), [0]*16+[1]*3)
        bad = copy.deepcopy(expected); bad[1]['image_ids'][-1] = 17
        with self.assertRaises(RuntimeError): apply_recorded_plan(d, bad)

    def test_real_5000_history_positions_geometry_and_parent_train_are_preserved(self):
        c = read(Path(__file__).resolve().parents[1] / 'CONFIG.json')
        relative = c['historical_native_trace'].split('/paper-disc-organized/', 1)[1]
        trace = read(ROOT / relative)
        plan = historical_val_plan(trace)
        images = {r['image_id']: r for r in trace['images']}
        self.assertEqual(len(plan), 313)
        self.assertEqual(len(plan[-1]['image_ids']), 8)
        for b in plan:
            for position, iid in enumerate(b['image_ids']):
                self.assertEqual((images[iid]['batch_index'], images[iid]['batch_position'], images[iid]['input_hw']),
                                 (b['source_batch_index'], position, b['rect_shape']))
        broken = copy.deepcopy(trace); broken['images'][0]['batch_position'] = 99
        with self.assertRaises(RuntimeError): historical_val_plan(broken)

    def test_scientific_configuration_unchanged(self):
        study = Path(__file__).resolve().parents[1]
        parent = read(ROOT / 'experiments/mask_branch_joint_adaptation_20261018/CONFIG.json')
        current = read(study/'CONFIG.json')
        routing = {'study_id','study_root','protocol','frozen_manifest'}
        for key, value in parent.items():
            if key not in routing: self.assertEqual(current[key], value, key)
        self.assertEqual(set(current)-set(parent), {'recovery_parent_contract','recovery_parent_orders'})
        self.assertEqual((study/'PROTOCOL.md').read_bytes(),
                         (ROOT/'experiments/mask_branch_joint_adaptation_20261018/PROTOCOL.md').read_bytes())


if __name__ == '__main__': unittest.main()
