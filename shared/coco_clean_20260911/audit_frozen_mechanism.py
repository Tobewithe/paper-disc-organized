"""Independent replay audit and bounded GT-informed direction check (no training).

The additional direction is exploratory, chosen after observing the ridge oracle.
Official process_mask is used for replay; pixel counts use NumPy, not the original
torch metric helper. No checkpoint is loaded or updated.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
from collections import Counter
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from ultralytics.utils import ops

ROOT = Path(__file__).resolve().parent
METRICS = ['coverage', 'same_neighbor', 'neighbor', 'background', 'mask_iou']

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_json(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def save_csv(p, rows):
    with Path(p).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def measure(m, own, union, same, valid):
    m=m & valid; own=own & valid
    area=own.sum(); tp=(m & own).sum(); near=(m & union & ~own).sum()
    sn=(m & same & ~own).sum(); bg=(m & ~union).sum()
    assert tp+near+bg == m.sum()
    return dict(zip(METRICS, [tp/area,sn/area,near/area,bg/area,tp/(area+near+bg)]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);arg=ap.parse_args()
    run=arg.run;out=run/'audit';out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((run/'protocol.json').read_text())
    sources={'weight':ROOT/'weights/yolo26m-seg.pt', 'gt':ROOT/'data/annotations/instances_val2017.json',
             'metadata':ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv', 'script':ROOT/'frozen_mechanism_probe.py'}
    for name,path in sources.items():assert digest(path)==protocol[name+'_sha256'], name
    for name,h in json.loads((run/'capacity_check/COMPLETE.json').read_text())['file_hashes'].items():
        assert digest(run/'capacity_check'/name)==h
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(sources['gt']))
    meta={int(r['annotation_id']):r for r in read_csv(sources['metadata'])}
    selection=json.loads((run/'selection.json').read_text());statuses=json.loads((run/'statuses.json').read_text())
    assert len(selection)==300 and len({x['image_id'] for x in selection})==300
    assert Counter(x['pair_group'] for x in selection)=={'high':200,'low':100}
    assert {s['image_id'] for s in statuses}=={s['image_id'] for s in selection}
    # Independent scalar ICI computation from original annotation identities.
    def intersection(a,b):
        return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    for s in selection:
        a,b=(gt.anns[s[k]] for k in ['annotation_a','annotation_b'])
        assert a['image_id']==b['image_id']==s['image_id'] and a['category_id']==b['category_id']
        inter=intersection(a['bbox'],b['bbox']);iou=inter/(a['bbox'][2]*a['bbox'][3]+b['bbox'][2]*b['bbox'][3]-inter)
        assert iou>.05 and abs(iou-s['gt_box_iou'])<1e-10
        for own in [a,b]:
            ici=sum(intersection(own['bbox'],other['bbox']) for other in gt.imgToAnns[s['image_id']]
                    if other['id']!=own['id'] and not other.get('iscrowd',0) and other['category_id']==own['category_id'])/(own['bbox'][2]*own['bbox'][3])
            assert abs(ici-float(meta[own['id']]['ici_same']))<1e-9
    records=read_csv(run/'interventions.csv')
    key=lambda r:(int(r['image_id']),int(r['target_annotation']),r['domain'],r['mode'],float(r['degrees']))
    lookup={key(r):r for r in records};assert len(records)==len(lookup)==16768
    for r in records:
        assert abs(float(r['mask_iou'])-float(r['coverage'])/(1+float(r['neighbor'])+float(r['background'])))<1e-12
    eligible=[s for s in statuses if s['status']=='ok'];assert len(eligible)==262
    modes=[('zero',0.)]+[(m,d) for d in [2.,5.,10.] for m in ['push','pull','random_plus','random_minus','spatial_oracle']]
    for s in eligible:
        for aid in [s['annotation_a'],s['annotation_b']]:
            for domain in ['cropped','raw']:
                for mode,d in modes:assert (s['image_id'],aid,domain,mode,d) in lookup
    audit_ids=set(sorted([s['image_id'] for s in eligible], key=lambda iid:hashlib.sha256(f'audit:{iid}'.encode()).hexdigest())[:16])
    capacity={(int(r['image_id']),int(r['target_annotation'])):r for r in read_csv(run/'capacity_check/per_target.csv')}
    newrows=[];max_pixel_difference=0.;rle_checks=0;normal_max_error=0.;all_basis_cos=[]
    for k,s in enumerate(eligible,1):
        iid=s['image_id'];cache=np.load(run/'tensors'/f'{iid}.npz')
        p=torch.tensor(cache['proto'],device='cuda');c=torch.tensor(cache['coeff'],device='cuda');boxes=torch.tensor(cache['boxes'],device='cuda')
        allc=torch.tensor(cache['variants'],device='cuda');shape=tuple(int(x) for x in cache['shape']);ishape=tuple(int(x) for x in cache['input_shape'])
        aids=cache['annotation_ids'].tolist();anns=gt.imgToAnns[iid]
        masks={a['id']:gt.annToMask(a).astype(bool) for a in anns}
        union=np.logical_or.reduce([masks[a['id']] for a in anns if not a.get('iscrowd',0)])
        same=np.logical_or.reduce([masks[a['id']] for a in anns if not a.get('iscrowd',0) and a['category_id']==gt.anns[aids[0]]['category_id']])
        crowd=[masks[a['id']] for a in anns if a.get('iscrowd',0)]
        valid=~np.logical_or.reduce(crowd) if crowd else np.ones(shape,dtype=bool)
        # Algebraic geometry check in double precision: logit cosine equals Gram-metric cosine.
        cd=c.double();flat=p.flatten(1).double();z=cd@flat;gram=flat@flat.T
        gc=(cd[0]@gram@cd[1])/torch.sqrt((cd[0]@gram@cd[0])*(cd[1]@gram@cd[1]))
        zc=F.cosine_similarity(z[0:1],z[1:2],dim=1)[0]
        assert abs(float(gc-zc))<1e-10
        all_basis_cos.append(float(gc))
        for arm,(mode,degree) in enumerate(modes):
            if mode in ['push','pull','random_plus','random_minus']:
                cc=allc[arm*2:arm*2+2]
                err=float(((cc.norm(dim=1)-c.norm(dim=1))/c.norm(dim=1)).abs().max());normal_max_error=max(normal_max_error,err)
                assert err<2e-6
                row=lookup[(iid,aids[0],'cropped',mode,degree)]
                angle=torch.acos(F.cosine_similarity(cc.double(),cd).clamp(-1,1))*180/np.pi
                assert np.max(np.abs(angle.cpu().numpy()-float(row['effective_degrees'])))<1e-4
        # Official full decoder replay for all variants on a hash-selected audit subset.
        if iid in audit_ids:
            cm=ops.process_mask(p,allc,boxes.repeat(16,1),ishape,upsample=True)
            cm=(ops.scale_masks(cm[:,None],shape)[:,0]>.5).cpu().numpy()
            z=(allc@p.flatten(1)).reshape(-1,*p.shape[1:])
            rm=(F.interpolate(z[None],ishape,mode='bilinear',align_corners=False)[0]>0).to(torch.uint8)
            rm=(ops.scale_masks(rm[:,None],shape)[:,0]>.5).cpu().numpy()
            for domain,decoded in [('cropped',cm),('raw',rm)]:
                for arm,(mode,degree) in enumerate(modes):
                    for j,aid in enumerate(aids):
                        vals=measure(decoded[arm*2+j],masks[aid],union,same,valid)
                        saved=lookup[(iid,aid,domain,mode,degree)]
                        max_pixel_difference=max(max_pixel_difference,max(abs(vals[m]-float(saved[m])) for m in METRICS))
                        if domain=='cropped' and mode=='zero':
                            rles=[mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in [decoded[j]&valid,masks[aid]&valid]]
                            assert abs(float(mask_utils.iou(rles[:1],rles[1:],[0])[0,0])-vals['mask_iou'])<1e-12
                            rle_checks+=1
        # Independently recompute the closed-form fit, then move toward it by at most 5 degrees.
        feature=ops.scale_masks(F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0][:,None],shape)[:,0]
        support=ops.crop_mask(torch.ones((2,*ishape),dtype=torch.uint8,device='cuda'),boxes)
        support=(ops.scale_masks(support[:,None],shape)[:,0]>.5).cpu().numpy()
        fitted=[];bounded=[];angles=[]
        for j,aid in enumerate(aids):
            own=masks[aid]&valid;pos=own&support[j];neg=~own&valid&support[j]
            xp=feature[:,pos].double();xn=feature[:,neg].double()
            matrix=xp@xp.T/xp.shape[1]+xn@xn.T/xn.shape[1]
            rhs=xp.mean(1)-xn.mean(1);system=matrix+1e-4*matrix.trace()/32*torch.eye(32,device='cuda',dtype=torch.float64)
            fit=torch.linalg.solve(system,rhs).float();fitted.append(fit)
            u=F.normalize(c[j],dim=0);v=F.normalize(fit,dim=0);cos=(u*v).sum().clamp(-1,1)
            phi=torch.acos(cos);theta=torch.minimum(phi,torch.tensor(np.deg2rad(5),device='cuda'))
            tangent=v-cos*u;tangent/=tangent.norm().clamp_min(1e-12)
            direction=c[j].norm()*(torch.cos(theta)*u+torch.sin(theta)*tangent)
            bounded.append(direction);angles.append((float(phi*180/np.pi),float(theta*180/np.pi)))
        cc=torch.cat([c,torch.stack(fitted),torch.stack(bounded)])
        cm=ops.process_mask(p,cc,boxes.repeat(3,1),ishape,upsample=True)
        cm=(ops.scale_masks(cm[:,None],shape)[:,0]>.5).cpu().numpy()
        for j,aid in enumerate(aids):
            base=measure(cm[j],masks[aid],union,same,valid);full=measure(cm[j+2],masks[aid],union,same,valid);small=measure(cm[j+4],masks[aid],union,same,valid)
            saved=capacity[(iid,aid)];base_old=lookup[(iid,aid,'cropped','zero',0.)]
            for m in METRICS:
                assert abs(base[m]-float(base_old[m]))<1e-12
                assert abs(base[m]-float(saved['base_'+m]))<1e-12 and abs(full[m]-float(saved['oracle_'+m]))<1e-12
            row=dict(image_id=iid,target_annotation=aid,target_ici=float(meta[aid]['ici_same']),angle_to_fit=angles[j][0],bounded_angle=angles[j][1])
            for m in METRICS:
                random=(float(lookup[(iid,aid,'cropped','random_plus',5.)][m])+float(lookup[(iid,aid,'cropped','random_minus',5.)][m]))/2
                row.update({f'base_{m}':base[m],f'full_{m}':full[m],f'bounded_{m}':small[m],f'delta_{m}':small[m]-base[m],f'delta_random_{m}':small[m]-random})
            newrows.append(row)
        if k%50==0:print(json.dumps(dict(completed=k,total=len(eligible),seconds=round(time.monotonic()-start,1))),flush=True)
    assert max_pixel_difference<1e-12
    save_csv(out/'bounded_direction.csv',newrows)
    # Independent bootstrap loop; same saved draws, independently aggregate records by image.
    draws=np.load(run/'bootstrap_draws.npz');image_ids=draws['image_ids'];draw=draws['draws'];position={int(i):j for j,i in enumerate(image_ids)}
    def estimate(values):
        sums=np.zeros(len(image_ids));count=sums.copy()
        for iid,v in values:sums[position[iid]]+=v;count[position[iid]]+=1
        boot=sums[draw].sum(1)/count[draw].sum(1)
        return dict(mean=float(sums.sum()/count.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),targets=int(count.sum()),images=int((count>0).sum()))
    stats=[]
    for group in ['all','high','low']:
        part=[r for r in newrows if group=='all' or (r['target_ici']>.5)==(group=='high')]
        for control in ['zero','random_mean']:
            for m in METRICS:
                field=('delta_' if control=='zero' else 'delta_random_')+m
                stats.append(dict(group=group,contrast='bounded_ridge_minus_'+control,metric=m,**estimate([(r['image_id'],r[field]) for r in part])))
    save_csv(out/'bounded_summary.csv',stats)
    summary=read_csv(run/'summary.csv');checks=0
    for row in summary:
        if row['degrees']!='5.0' or row['contrast']!='push_minus_zero':continue
        part=[r for r in records if r['domain']==row['domain'] and r['mode']=='push' and r['degrees']=='5.0' and (row['group']=='all' or (float(r['target_ici'])>.5)==(row['group']=='high'))]
        calc=estimate([(int(r['image_id']),float(r[row['metric']])-float(lookup[(int(r['image_id']),int(r['target_annotation']),row['domain'],'zero',0.)][row['metric']])) for r in part])
        for m in ['mean','ci_low','ci_high']:assert abs(calc[m]-float(row[m]))<1e-12
        checks+=1
    # Representation contrasts are pair-level (one record/image), not duplicated target rows.
    rep=[]
    for mode in ['push','pull','random_plus','random_minus','spatial_oracle']:
        for metric in ['coefficient_cosine','logit_cosine','raw_pair_iou','positive_product']:
            vals=[]
            for s in eligible:
                iid=s['image_id'];aid=s['annotation_a'];a=lookup[(iid,aid,'cropped',mode,5.)];b=lookup[(iid,aid,'cropped','zero',0.)]
                vals.append((iid,float(a[metric])-float(b[metric])))
            rep.append(dict(mode=mode,metric=metric,**estimate(vals)))
    save_csv(out/'representation_summary.csv',rep)
    result=dict(status='PASS',training=False,source_hashes_match=True,selected_images=300,valid_pairs=262,unmatched_pairs=38,
                independently_recomputed_ici_instances=600,official_replay_audit_images=16,official_replay_metric_max_error=max_pixel_difference,
                pycocotools_mask_iou_checks=rle_checks,capacity_replay_targets=524,bootstrap_summary_checks=checks,
                rotation_max_relative_norm_error=normal_max_error,angle_to_fit_median=float(np.median([r['angle_to_fit'] for r in newrows])),
                bounded_less_than_5_degrees=sum(r['bounded_angle']<4.999 for r in newrows),elapsed_seconds=time.monotonic()-start,
                script_sha256=digest(__file__),limitations='Additional bounded ridge direction is a GT-informed post-hoc diagnostic, not training, AP or a strict optimum. All CIs pointwise; sample enriched and conditioned on bbox-matched pairs.')
    save_json(out/'AUDIT.json',result);print(json.dumps(result),flush=True)
    for r in stats:
        if r['group']=='high' and r['contrast'].endswith('zero'):print(json.dumps(r),flush=True)

if __name__=='__main__':main()
