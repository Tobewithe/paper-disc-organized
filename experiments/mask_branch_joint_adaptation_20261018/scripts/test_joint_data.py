"""CPU-only tests for frozen sampling, batch budgets, identity and fail-closed writes."""
import json
from pathlib import Path
import tempfile
import unittest
from joint_data import choose_training_ids, update_orders, dataset_plan, check_dataset_plan, write, read


class FrozenDataTests(unittest.TestCase):
    def test_fit_selection_preserves_original_and_is_stable(self):
        original = list(range(2000))
        chosen = choose_training_ids(range(118287), original)
        self.assertEqual(chosen, choose_training_ids(reversed(range(118287)), list(reversed(original))))
        self.assertEqual(len(chosen), 20000)
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertTrue(set(original) <= set(chosen))

    def test_reject_missing_or_duplicate_original(self):
        with self.assertRaises(RuntimeError): choose_training_ids(range(1999), list(range(2000)))
        with self.assertRaises(RuntimeError): choose_training_ids(range(118287), [0]*2000)

    def test_all_three_epoch_images_and_update_denominators(self):
        batches = [{'image_ids':list(range(i*16, (i+1)*16))} for i in range(1250)]
        orders = update_orders(batches)
        self.assertEqual(orders, update_orders(batches))
        self.assertNotEqual(orders[0], orders[1])
        total = 0
        for epoch in orders:
            self.assertEqual(sorted(x['source_batch_index'] for x in epoch), list(range(1250)))
            images = []
            for item in epoch:
                self.assertEqual([len(group) for group in item['microgroups']], [4]*4)
                positions = sum(item['microgroups'], [])
                self.assertEqual(sorted(positions), list(range(16)))
                images += [batches[item['source_batch_index']]['image_ids'][p] for p in positions]
                total += len(item['microgroups'])
            self.assertEqual(sorted(images), list(range(20000)))
        self.assertEqual(total, 15000)

    def test_partial_training_batch_rejected(self):
        with self.assertRaises(RuntimeError): update_orders([{'image_ids':[1]}])

    def test_dataset_plan_tracks_partial_val_batch_and_geometry(self):
        class Dataset:
            im_files = [f'/images/{i:012d}.jpg' for i in range(19)]
            batch_shapes = [[320,672],[672,320]]
            def __len__(self): return len(self.im_files)
        dataset = Dataset()
        plan = dataset_plan(dataset)
        self.assertEqual([len(x['image_ids']) for x in plan],[16,3])
        check_dataset_plan(dataset,plan)
        dataset.batch_shapes[1] = [640,320]
        with self.assertRaises(RuntimeError): check_dataset_plan(dataset,plan)

    def test_atomically_write_without_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'record.json'
            write(path,{'v':1})
            self.assertEqual(read(path),{'v':1})
            with self.assertRaises(RuntimeError): write(path,{'v':2})
            self.assertEqual(read(path),{'v':1})
            write(path,{'v':2},replace=True)
            self.assertEqual(read(path),{'v':2})

    def test_ledger_keeps_box_identity_separate_from_mask_repair(self):
        from joint_runner import compare_ledgers
        def row(aid, state, prefix='old', rank=7, ignored=False):
            return {'annotation_id':aid,'image_id':1,'is_ignored':ignored,'task_mask75':state,
                    'task_box75':'success','bbox75_match_output_id':f'{prefix}:{rank}'}
        native=[row(10,'success'),row(11,'failure'),row(12,'ignored',ignored=True)]
        current=[row(10,'failure','new'),row(11,'success','new'),row(12,'ignored','new',ignored=True)]
        self.assertEqual(compare_ledgers(native,current),([10],[11]))
        current[0]['bbox75_match_output_id']='new:8'
        with self.assertRaises(RuntimeError): compare_ledgers(native,current)


if __name__ == '__main__': unittest.main()
