"""Post-hoc DEVelopment-only pixel transitions for existing nonzero arms.

No new parameter search, no validation retuning, no new method variants.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,write_json,write_csv,sha
from relative_ownership_decoder import decode
from relative_ownership_experiment import read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    config=json.loads((out/'protocol.json').read_text())['config'];ids=json.loads((out/'selection.json').read_text())['development']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_train2017.json'))
    meta={int(r['annotation_id']):r for r in read(ROOT/'census/train2017_instances.csv')}
    arms={'initial':dict(kind='initial',value=0),'relative_0.25':dict(kind='relative',value=.25),'relative_0.5':dict(kind='relative',value=.5)}
    rows=[];start=time.monotonic()
    for iid in ids:
        with np.load(out/'train_cache'/f'{iid}.npz') as q:item={k:q[k] for k in q.files}
        masks,changes,xor=decode({k:item[k] for k in ['proto','coeff','boxes','detections','input_shape']},arms,iid,config['eligibility'])
        assert xor==0
        bypred={int(j):int(aid) for aid,j in zip(item['mapping_gt'],item['mapping_pred'])}
        changemap={(r['arm'],r['prediction_index']):r for r in changes}
        shape=tuple(map(int,item['shape']));anns=gt.imgToAnns[iid]
        raster={x['id']:torch.tensor(gt.annToMask(x).astype(bool),device='cuda') for x in anns}
        ordinary=[x for x in anns if not x.get('iscrowd',0)]
        union=torch.stack([raster[x['id']] for x in ordinary]).any(0)
        crowd=torch.zeros(shape,dtype=torch.bool,device='cuda')
        for x in anns:
            if x.get('iscrowd',0):crowd|=raster[x['id']]
        valid=~crowd
        for j,aid in bypred.items():
            own=raster[aid]&valid;area=int(own.sum())
            if not area:continue
            same=torch.stack([raster[x['id']] for x in ordinary if x['category_id']==gt.anns[aid]['category_id']]).any(0)
            domains={'own':own,'same_neighbor':same&~own&valid,'other_category':union&~same&~own&valid,'background':~union&valid}
            base=ops.scale_masks(masks['initial'][j:j+1,None],shape)[0,0]>.5
            for arm in ['relative_0.25','relative_0.5']:
                changed=ops.scale_masks(masks[arm][j:j+1,None],shape)[0,0]>.5
                removed=base&~changed;added=changed&~base
                change=changemap.get((arm,j));neighbors=[int(k) for k in change['neighbor_indices'].split(',')] if change else []
                row=dict(image_id=iid,annotation_id=aid,ici=float(meta[aid]['ici_same']),arm=arm,eligible=change is not None,
                     input_pixels_changed=int(change['changed_pixels']) if change else 0,
                     selected_neighbors=len(neighbors),selected_neighbors_matched=sum(k in bypred for k in neighbors),
                     selected_neighbors_matched_sameclass=sum(k in bypred and gt.anns[bypred[k]]['category_id']==gt.anns[aid]['category_id'] for k in neighbors))
                for domain,mask in domains.items():
                    row['removed_'+domain]=int((removed&mask).sum())/area
                    row['added_'+domain]=int((added&mask).sum())/area
                rows.append(row)
    write_csv(out/'development_transfers.csv',rows)
    summary=[]
    for arm in ['relative_0.25','relative_0.5']:
        for group in ['all','high','low']:
            rr=[r for r in rows if r['arm']==arm and (group=='all' or (r['ici']>.5+1e-10)==(group=='high'))]
            summary.append(dict(arm=arm,group=group,targets=len(rr),eligible=sum(r['eligible'] for r in rr),changed=sum(r['input_pixels_changed']>0 for r in rr),
                 **{k:float(np.mean([r[k] for r in rr])*100) for k in rr[0] if k.startswith(('removed_','added_'))}))
    write_csv(out/'development_transfer_summary.csv',summary)
    write_json(out/'DEVELOPMENT_TRANSFERS_COMPLETE.json',dict(status='COMPLETE',scope='Post-hoc descriptive development only; explains existing arms; no re-selection or validation modification.',images=len(ids),seconds=time.monotonic()-start,source_sha256=sha(__file__),files={p.name:sha(p) for p in [out/'development_transfers.csv',out/'development_transfer_summary.csv']}))
    for r in summary:
        if r['group']=='high':print(json.dumps(r),flush=True)


if __name__=='__main__':main()
