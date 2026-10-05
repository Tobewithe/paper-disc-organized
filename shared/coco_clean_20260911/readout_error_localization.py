"""Original-mask error accounting and fixed-crop attainable-coverage bound.

No weights fitted and no intervention chosen; baseline on predeclared300 images.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops
from readout_input_probe import sha,write_json


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path);args=ap.parse_args()
    run=args.run;out=args.out or run/'error_localization';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((run/'cache/protocol.json').read_text())
    ann=Path(protocol['data_root'])/'annotations/instances_train2017.json'
    assert sha(ann)==protocol['annotation_sha256']
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ann))
    selection=json.loads((run/'cache/selection.json').read_text())['transfer']
    task={int(r['annotation_id']):r for r in read(run/'task/gt_recovery.csv') if r['arm']=='original_s-1_d0'}
    rows=[];start=time.monotonic()
    for iid in selection:
        with np.load(run/'cache/images'/f'{iid}.npz') as q:item={k:q[k] for k in q.files}
        tids=item['annotation_ids'];idx=item['prediction_indices']
        if not len(tids):continue
        shape=tuple(item['shape']);ishape=tuple(item['input_shape'])
        proto=torch.tensor(item['proto'],device='cuda');coeff=torch.tensor(item['coeff'][idx],device='cuda');boxes=torch.tensor(item['boxes'][idx],device='cuda')
        ordinary=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
        masks={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]}
        union=np.zeros(shape,bool);crowd=np.zeros(shape,bool)
        for a in gt.imgToAnns[iid]:
            if a.get('iscrowd',0):crowd|=masks[a['id']]
            else:union|=masks[a['id']]
        with torch.inference_mode():
            decoded=ops.process_mask(proto,coeff,boxes,ishape,upsample=True)
            support=ops.crop_mask(torch.ones((len(idx),*ishape),device='cuda'),boxes)
            restored=ops.scale_masks(decoded[:,None],shape)[:,0]>.5
            allowed=ops.scale_masks(support[:,None],shape)[:,0]>.5
        for k,aid0 in enumerate(tids):
            aid=int(aid0);a=gt.anns[aid];rawown=masks[aid]
            own=rawown&~crowd;pred=restored[k].cpu().numpy();supp=allowed[k].cpu().numpy()
            assert not np.any(pred&~supp),'Decoded mask escaped computed maximal crop support'
            same=np.zeros(shape,bool)
            for b in ordinary:
                if b['category_id']==a['category_id'] and b['id']!=aid:same|=masks[b['id']]
            missing=own&~pred;fp=pred&~rawown&~crowd;inside_missing=missing&supp;outside_missing=missing&~supp
            nn=fp&same;other=fp&union&~same;bg=fp&~union
            same_available=supp&same&~rawown&~crowd
            background_available=supp&~union&~crowd
            assert int(nn.sum()+other.sum()+bg.sum())==int(fp.sum())
            assert int(inside_missing.sum()+outside_missing.sum())==int(missing.sum())
            # Fixed, size-adaptive descriptive boundary width in original pixels.
            radius=max(2.,.02*np.sqrt(float(rawown.sum())))
            padded=np.pad(rawown.astype(np.uint8),1)
            dist_in=cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
            boundary=own&(dist_in<=radius);interior=own&(dist_in>radius)
            area=int(own.sum());rawarea=int(rawown.sum());tp=int((pred&rawown).sum());rawfp=int((pred&~rawown).sum())
            ceiling=float((rawown&supp).sum()/rawarea) if rawarea else 0.
            iou=tp/(rawarea+rawfp) if rawarea+rawfp else 0.
            removed_neighbor=int(nn.sum());removed_background=int(bg.sum())
            # GT-assisted pixel editing of THIS fixed bbox-assigned mask only.
            # It is neither a deployable method nor official all-GT recall.
            remove_neighbor_iou=tp/(rawarea+rawfp-removed_neighbor)
            remove_background_iou=tp/(rawarea+rawfp-removed_background)
            remove_both_iou=tp/(rawarea+rawfp-removed_neighbor-removed_background)
            no_fp_iou=tp/rawarea
            fill_inside_iou=(tp+int(inside_missing.sum()))/(rawarea+rawfp)
            t=task[aid];ici=float(t['ici'])
            row=dict(image_id=iid,annotation_id=aid,ici_same=ici,density='high' if ici>.5+1e-10 else 'other',
                official_hit75=t['hit75']=='True',fixed_iou=iou,crop_coverage_upper_bound=ceiling,
                fixed_hit75=iou>=.75,task_vs_fixed_disagreement=(t['hit75']=='True')!=(iou>=.75),
                crop_alone_precludes75=ceiling<.75,category_id=a['category_id'],area=float(a['area']),
                valid_area=area,boundary_width_px=radius,
                fn_inside_crop=int(inside_missing.sum()),fn_outside_crop=int(outside_missing.sum()),
                false_positive_same_neighbor=int(nn.sum()),false_positive_other_class=int(other.sum()),
                false_positive_background=int(bg.sum()),
                same_neighbor_available=int(same_available.sum()),background_available=int(background_available.sum()),
                neighbor_fp_rate=float(nn.sum()/same_available.sum()) if same_available.any() else None,
                background_fp_rate=float(bg.sum()/background_available.sum()) if background_available.any() else None,
                oracle_remove_neighbor_iou=remove_neighbor_iou,oracle_remove_background_iou=remove_background_iou,
                oracle_remove_both_iou=remove_both_iou,oracle_remove_all_fp_iou=no_fp_iou,
                oracle_fill_missing_inside_crop_iou=fill_inside_iou,
                missing_boundary=int((missing&boundary).sum()),missing_interior=int((missing&interior).sum()),
                boundary_area=int(boundary.sum()),interior_area=int(interior.sum()),
                own_coverage=float((pred&own).sum()/area) if area else None)
            rows.append(row)
    with (out/'instances.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for density in ['high','other']:
        for state in ['all','official_failed','official_recovered','fixed_failed','fixed_recovered']:
            rr=[r for r in rows if r['density']==density and (state=='all' or
                (r['fixed_hit75']==(state=='fixed_recovered') if state.startswith('fixed') else
                 r['official_hit75']==(state=='official_recovered')))]
            if not rr:continue
            valid=[r for r in rr if r['valid_area']>0]
            fields=['fn_inside_crop','fn_outside_crop','false_positive_same_neighbor','false_positive_other_class','false_positive_background','missing_boundary','missing_interior']
            groups.append(dict(density=density,state=state,targets=len(rr),images=len({r['image_id'] for r in rr}),
                crop_precludes75=sum(r['crop_alone_precludes75'] for r in rr),
                mean_crop_upper_bound=float(np.mean([r['crop_coverage_upper_bound'] for r in rr])),
                own_coverage=float(np.mean([r['own_coverage'] for r in valid])) if valid else None,
                task_vs_fixed_disagreement=sum(r['task_vs_fixed_disagreement'] for r in rr),
                spatial_rates={key:dict(n=sum(r[key] is not None for r in rr),
                    mean=float(np.mean([r[key] for r in rr if r[key] is not None])) if any(r[key] is not None for r in rr) else None)
                    for key in ['neighbor_fp_rate','background_fp_rate']},
                fixed_failed_count=sum(not r['fixed_hit75'] for r in rr),
                fixed_failed_rescued_by_gt_edit={key:sum(not r['fixed_hit75'] and r[key]>=.75 for r in rr)
                    for key in ['oracle_remove_neighbor_iou','oracle_remove_background_iou','oracle_remove_both_iou',
                                'oracle_remove_all_fp_iou','oracle_fill_missing_inside_crop_iou']},
                error_over_own_gt={key:float(np.mean([r[key]/r['valid_area'] for r in valid])) for key in fields},
                note='Each error/ownGT computed per target then averaged; errors are not mutually exclusive causal mechanisms.'))
    write_json(out/'SUMMARY.json',dict(groups=groups,seconds=time.monotonic()-start,
       scope='Baseline fixed bbox50-attribution errors; crop support bound is max possible originalGT recall under this exact fixed crop/scale path, '
       'hence IoU ceiling. Not an achieved oracle mask, not causal blame on coefficients. Boundary split descriptive, no tuning.',
       additional_scope='Oracle edits are exact GT-assisted editing of the fixed bbox-assigned binary mask. '
       'They measure conditional rescue opportunities, not all-GT recall or learnability. '
       'Task COCO matching and fixed bbox attribution differ; all groups explicitly separated. '
       'Conditional FPR uses available same-neighbor/background pixels inside the fixed crop, excludes crowd. '
       'High versus other comparisons are unadjusted for category, scale and other composition.',
       source_hashes={'task_gt':sha(run/'task/gt_recovery.csv'),'cache_complete':sha(run/'cache/COMPLETE.json'),
                      'script':sha(Path(__file__))}))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
