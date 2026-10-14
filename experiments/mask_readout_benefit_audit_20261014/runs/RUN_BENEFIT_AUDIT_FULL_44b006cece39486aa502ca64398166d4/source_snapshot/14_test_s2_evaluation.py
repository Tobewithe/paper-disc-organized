"""Image-macro weighting and full-output task-damage regression checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from run_s2_evaluation import image_effects,bootstrap,ROOT
sys.path.insert(0,str(ROOT/'experiments/official_gt_failure_rebaseline_20261013/scripts'))
from build_native_ledger import evaluate_ledger
from damage_acceptance import select_damage_panel,score_damage


class EvaluationTests(unittest.TestCase):
    def test_image_macro_not_instance_macro_or_zero_images(self):
        rows=[{'image_id':1,'population':'T','delta_iou':.02}]*10
        rows += [{'image_id':2,'population':'T','delta_iou':-.01},
                 {'image_id':3,'population':'S','delta_iou':.05}]
        means=image_effects(rows)
        self.assertEqual(set(means),{1,2})
        self.assertAlmostEqual(bootstrap(means)['mean'],.005)
        self.assertEqual(bootstrap(means),bootstrap(means))
        self.assertAlmostEqual(bootstrap({i:0. for i in means})['ci95'][0],0.)

    def test_independent_box_mask_matching_and_damage(self):
        import numpy as np
        from pycocotools import mask
        def rle(m):
            encoded=mask.encode(np.asfortranarray(m.astype(np.uint8)))
            return {'size':encoded['size'],'counts':encoded['counts'].decode('ascii')}
        raster=np.zeros((16,16),dtype=np.uint8);raster[3:13,3:13]=1
        gt_rle=rle(raster)
        gt={'info':{},'licenses':[],'images':[{'id':i,'height':16,'width':16} for i in (1,2,3)],
            'categories':[{'id':1,'name':'object'}],
            'annotations':[{'id':i,'image_id':i,'category_id':1,'bbox':[3,3,10,10],
                            'area':100,'iscrowd':0,'segmentation':gt_rle} for i in (1,2,3)]}
        native=[{'image_id':i,'category_id':1,'bbox':[3,3,10,10],'score':.9,'segmentation':gt_rle} for i in (1,2,3)]
        changed=[dict(p) for p in native];changed[0]['segmentation']=rle(np.zeros_like(raster))
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            path=Path(directory)/'gt.json';path.write_text(json.dumps(gt))
            baseline,_=evaluate_ledger(path,native,[1,2,3],'a'*64)
            method,_=evaluate_ledger(path,changed,[1,2,3],'b'*64)
        self.assertTrue(all(row['task_box75']=='success' for row in method))
        self.assertEqual(sum(row['task_mask75']=='failure' for row in method),1)
        panel=select_damage_panel(baseline,'c'*64,'d'*64,'e'*64,n=3,seed=0)
        result=score_damage(panel,baseline,method)
        self.assertEqual(result['gate'],'damage_guardrail_failed')
        self.assertEqual(result['upper_damaged_count'],1)


if __name__=='__main__':
    unittest.main()
