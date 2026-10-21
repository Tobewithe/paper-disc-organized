"""Original-batch selection behavior; synthetic dataset, no images or base forward."""
import unittest
from types import SimpleNamespace
from rfc_native_replay import OriginalBatches


class Dataset:
    im_files=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg']
    def __getitem__(self,index): return dict(original_index=index,image_id=index+1)
    @staticmethod
    def collate_fn(items): return items


def manifest():
    return {'images':[dict(image_id=i+1,batch_index=i//2,batch_position=i%2) for i in range(5)]}


class BatchIntegrationTests(unittest.TestCase):
    def test_selected_image_keeps_entire_original_batch(self):
        chosen=OriginalBatches(SimpleNamespace(dataset=Dataset()),manifest(),{2,3})
        self.assertEqual(chosen.selected,[0,1])
        batches=list(chosen)
        self.assertEqual([[x['image_id'] for x in batch] for batch in batches],[[1,2],[3,4]])
        self.assertIs(chosen.dataset.im_files,Dataset.im_files)

    def test_original_last_partial_batch_is_not_repacked(self):
        chosen=OriginalBatches(SimpleNamespace(dataset=Dataset()),manifest(),{2,5})
        self.assertEqual(chosen.selected,[0,2])
        self.assertEqual([[x['original_index'] for x in b] for b in chosen],[[0,1],[4]])

    def test_reordered_dataset_cannot_reuse_old_input_fingerprints(self):
        dataset=Dataset(); dataset.im_files=['2.jpg','1.jpg','3.jpg','4.jpg','5.jpg']
        with self.assertRaisesRegex(ValueError,'regrouped/reordered'):
            OriginalBatches(SimpleNamespace(dataset=dataset),manifest(),{1})

    def test_invalid_original_batch_position_rejected(self):
        m=manifest(); m['images'][1]['batch_position']=0
        with self.assertRaisesRegex(ValueError,'batch positions'):
            OriginalBatches(SimpleNamespace(dataset=Dataset()),m,{1})


if __name__=='__main__': unittest.main(verbosity=2)
