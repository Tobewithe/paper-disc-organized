"""Independent CCL implementation (OpenCV), source masks and count checks."""
import sys
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1];ROOT=STUDY.parents[1]
sys.path.insert(0,str(ROOT/'_site'))
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as masks
from run_audit import read,lines,write,check,C,S1,GT,identity,RSM

def reference(gt,pred,neighbors,connectivity):
    ng,g=cv2.connectedComponents(gt.astype(np.uint8),connectivity=connectivity)
    npred,p=cv2.connectedComponents(pred.astype(np.uint8),connectivity=connectivity)
    matrix=np.bincount((g*npred+p).ravel(),minlength=ng*npred).reshape(ng,npred)
    coverage=matrix[1:,1:]
    split=int((np.count_nonzero(coverage,axis=1)>1).sum())
    missed=int((np.count_nonzero(coverage,axis=1)==0).sum())
    split_pixels=int((coverage.sum(axis=1)-coverage.max(axis=1,initial=0)).sum())
    other=np.zeros_like(gt)
    for _,v in neighbors:other|=v
    core=gt&~other;ids=np.unique(p[core]);ids=ids[ids>0]
    connected=np.isin(p,ids);foreign=np.zeros_like(gt);neighbor_ids=[]
    for i,v in neighbors:
        v=v&~gt;foreign|=v
        if (connected&v).any():neighbor_ids.append(i)
    return {'gt_components':ng-1,'prediction_components':npred-1,'split_gt_components':split,
            'missed_gt_components':missed,'split_pixels':split_pixels,
            'within_instance_false_joins':int((np.count_nonzero(coverage,axis=0)>1).sum()),
            'unmatched_prediction_islands':int((np.count_nonzero(coverage,axis=0)==0).sum()),
            'bridge_neighbor_ids':neighbor_ids,'bridge_pixels':int((connected&foreign).sum()),
            'bridge_eligible':bool(core.any())}

def verify(run):
    frozen=check();rows=lines(run/'PAIR_RESULTS.jsonl');p=lines(run/'PRESSURE_RESULTS.jsonl');summary=read(run/'SUMMARY.json')
    assert len(rows)==summary['pairs']==256 and len(p)==summary['pressure']['total']==6144
    originals={identity(r):r for r in lines(C/'PAIR_RESULTS.jsonl')}
    s1={identity(r):r for r in lines(S1/'ORACLE_RESULTS.jsonl')}
    rsm={identity(r):r for r in lines(RSM/'PAIR_RESULTS.jsonl')}
    coco=COCO(str(GT));last=None;neighbors=[];checks=0
    for row in rows:
        src=originals[identity(row)];truth=masks.decode(src['GT_rle']).astype(bool)
        if last!=row['image_id']:
            anns=coco.loadAnns(coco.getAnnIds(imgIds=[row['image_id']]))
            neighbors=[(a['id'],coco.annToMask(a).astype(bool)) for a in anns if not a.get('iscrowd',0) and not a.get('ignore',0)]
            last=row['image_id']
        others=[(i,m) for i,m in neighbors if i!=row['annotation_id']]
        arms={'N':src['arms']['N']['original_rle'],'C':src['arms']['C']['original_rle'],
              'G_old':src['arms']['R_G']['original_rle'],'S1':s1[identity(row)]['oracle_rle']}
        for name,rle in arms.items():
            mask=masks.decode(rle).astype(bool)
            for connectivity in [8,4]:
                observed=reference(truth,mask,others,connectivity)
                actual=row['endpoints'][name][str(connectivity)]
                assert all(actual[k]==v for k,v in observed.items()),(identity(row),name,connectivity)
                checks+=len(observed)
    valid=0
    for item in p:
        source=rsm[identity(item)]['pressure'][item['direction']]
        amp=rsm[identity(item)]['pressure_amplitude']
        expected=all(abs(source[k]/amp-1)<=5e-5 for k in ['C_actual_rms','C_decoded_difference_rms'])
        assert item['precision_valid']==expected;valid+=expected
    assert valid==summary['pressure']['valid']
    for name in ['N','S1','C','G_old']:
        for field in summary['endpoint'][name]['counts']:
            assert summary['endpoint'][name]['counts'][field]==sum(bool(r['endpoints'][name]['8'][field]) for r in rows)
    result={'status':'passed','independent_CCL':'OpenCV connectedComponents versus SciPy ndimage.label',
            'checked_endpoint_metric_fields':checks,'endpoint_masks':1024,'both_4_and_8_connectivity':True,
            'pressure_precision_flags_checked':6144,'all_inputs_sources_unchanged':True,
            'new_training_or_inference':False,'freeze_sha256':__import__('run_audit').sha(STUDY/'FREEZE_MANIFEST.json')}
    write(run/'INDEPENDENT_LAPTOP.json',result);return result

if __name__=='__main__':
    import json
    print(json.dumps(verify(STUDY/'runs'/sys.argv[1])))
