"""S038 seed-averaged targets and image-cluster intervals; no pseudo seed counts."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    names=['support','errors','gradients','steps','witness']
    for name in names:
        if sha(run/f'{name}.csv')!=receipt['hashes'][f'{name}.csv']:raise RuntimeError('Changed results')
    data={name:read(run/f'{name}.csv') for name in names};summaries=[]
    metadata={(int(r['image_id']),int(r['annotation_id'])):r for r in data['support']}
    for split in ['fit','transfer']:
        images=cfg[split+'_images'];index={i:k for k,i in enumerate(images)}
        draws=np.random.default_rng(20260912).multinomial(len(images),np.full(len(images),1/len(images)),size=2000)
        def interval(rows,field):
            ok=[r for r in rows if r.get(field) is not None and np.isfinite(r[field])]
            if not ok:return dict(n=0,mean=None,ci95=None)
            vals=np.array([r[field] for r in ok]);ix=np.array([index[r['image_id']] for r in ok])
            count=np.bincount(ix,minlength=len(images)).astype(float);total=np.bincount(ix,weights=vals,minlength=len(images))
            den=np.einsum('bi,i->b',draws,count,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
            b=np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
            return dict(n=len(ok),images=len(set(ix)),mean=float(vals.mean()),ci95=np.nanquantile(b,[.025,.975]).tolist())
        for high in [True,False]:
            support=[r for r in data['support'] if r['split']==split and (r['high']=='True')==high]
            ss=[dict(image_id=int(r['image_id']),**{k:float(r[k]) for k in ['sample_own','sample_same','sample_other','sample_background','sample_ignored','unique_sample_pixels']},
                zero_same=float(int(r['sample_same'])==0),
                support_same_but_no_sample=float(int(r['support_same'])>0 and int(r['sample_same'])==0)) for r in support]
            summaries.append(dict(domain='support',split=split,high=high,targets=len(ss),metrics={k:interval(ss,k) for k in ss[0] if k!='image_id'}))
            by=defaultdict(list)
            for r in data['errors']:
                if r['split']==split and (r['high']=='True')==high:by[int(r['image_id']),int(r['annotation_id'])].append(r)
            ee=[]
            for (iid,aid),rr in by.items():
                if len(rr)!=3:raise RuntimeError('Wrong error seed count')
                get=lambda k:float(np.mean([float(r[k]) for r in rr]))
                fp=get('neighbor_fp');inside=get('neighbor_fp_inside_support')
                ee.append(dict(image_id=iid,neighbor_fp=fp,neighbor_inside=inside,
                    outside_fraction=(fp-inside)/fp if fp else None,
                    sampled_fraction=get('neighbor_fp_sample_unique')/fp if fp else None,
                    has_error=float(fp>0),error_but_zero_same_sample=float(fp>0 and int(rr[0]['sample_same'])==0)))
            summaries.append(dict(domain='error_support',split=split,high=high,targets=len(ee),
                pooled_outside_fraction=sum(r['neighbor_fp']-r['neighbor_inside'] for r in ee)/max(sum(r['neighbor_fp'] for r in ee),1),
                metrics={k:interval(ee,k) for k in ['outside_fraction','sampled_fraction','has_error','error_but_zero_same_sample']}))
            by=defaultdict(list)
            for r in data['gradients']:
                if r['split']==split and (r['high']=='True')==high:by[int(r['image_id']),int(r['annotation_id'])].append(r)
            gg=[]
            columns=['own_same_coefficient_cos','own_same_shared_cos','same_total_coefficient_cos','same_total_shared_cos',
                'own_background_coefficient_cos','own_background_shared_cos']
            for (iid,aid),rr in by.items():
                if len(rr)!=3:raise RuntimeError('Wrong gradient seed count')
                row=dict(image_id=iid)
                for k in columns:
                    v=[float(r[k]) for r in rr if r[k]];row[k]=float(np.mean(v)) if v else None
                for kind in ['coefficient','shared']:
                    v=[float(float(r[f'same_total_{kind}_cos'])<0) for r in rr if r[f'same_total_{kind}_cos']]
                    row[f'same_total_{kind}_opposed_fraction']=float(np.mean(v)) if v else None
                gg.append(row)
            summaries.append(dict(domain='gradient',split=split,high=high,targets=len(gg),metrics={k:interval(gg,k) for k in gg[0] if k!='image_id'}))
            for direction in ['total','same','background']:
                for magnitude in [.001,.01]:
                    by=defaultdict(list)
                    for r in data['steps']:
                        if r['split']==split and (r['high']=='True')==high and r['direction']==direction and float(r['requested_logit_rms'])==magnitude:
                            by[int(r['image_id']),int(r['annotation_id'])].append(r)
                    tt=[]
                    for (iid,aid),rr in by.items():
                        row=dict(image_id=iid,loss_delta=float(np.mean([float(r['loss_after'])-float(r['loss_before']) for r in rr])),
                            relative_step=float(np.mean([float(r['coefficient_relative_step']) for r in rr])))
                        for region in ['own','same','background']:
                            v=[float(r[f'{region}_bce_change']) for r in rr if r[f'{region}_bce_change']]
                            row[region+'_bce_change']=float(np.mean(v)) if v else None
                            row[region+'_bce_increase_fraction']=float(np.mean(np.asarray(v)>0)) if v else None
                            row[region+'_positive_change']=float(np.mean([int(r[f'{region}_positive_change']) for r in rr]))
                        tt.append(row)
                    summaries.append(dict(domain='step',split=split,high=high,direction=direction,magnitude=magnitude,targets=len(tt),
                        metrics={k:interval(tt,k) for k in tt[0] if k!='image_id'}))
    result=dict(summaries=summaries,receipt=dict(targets=receipt['targets'],gradient_targets=receipt['gradient_targets'],gradient_rows=receipt['gradient_rows'],seconds=receipt['seconds']),
        gradient_replay=dict(max_coefficient_error=max(float(r['coefficient_sum_error']) for r in data['gradients']),
            max_shared_error=max(float(r['shared_sum_error']) for r in data['gradients'])),
        max_relative_step=max(float(r['coefficient_relative_step']) for r in data['steps']),
        scope='160hashfit+300exploredtransferimages; actualfitYandnativeboxsupportreplayed,transfernottrained. '
            'Allseedmeanspertarget,2000imageclusterCI separatelybysplit. 32hashprobetargetsperdensity/split, '
            'notindependent96observations. Regionlogitderivativepartition ofexactBCE+Dice, notstandaloneregionalDiceobjectives. '
            'Own/negative opposition mayreflectgenericsharedresponse direction; no proofofdense-exclusiveconflict orhistoricaltrainingcause. '
            'Temporarycoefficientsteps notactualsharedAdamupdates, nooutside-sample/generalizationmeasure.',
        hashes={n:sha(run/f'{n}.csv') for n in names})
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(receipt=result['receipt'],gradient_replay=result['gradient_replay'],max_relative_step=result['max_relative_step'],
        primary=[r for r in summaries if r['high'] and (r['domain'] in ['support','error_support','gradient'] or (r['domain']=='step' and r['magnitude']==.01 and r['direction']=='same'))])),flush=True)


if __name__=='__main__':main()
