"""S037 conditional ranking curves, explicit exposure and composition controls."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json
from readout_composition_control import comparison


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    for name in ['ranking.csv','coverage.csv','all_gt.csv']:
        if sha(run/name)!=receipt['hashes'][name]:raise RuntimeError('Changed rows')
    ranking=read(run/'ranking.csv');coverage=read(run/'coverage.csv');allgt=read(run/'all_gt.csv')
    images=cfg['images'];iidx={i:k for k,i in enumerate(images)}
    draws=np.random.default_rng(20260912).multinomial(len(images),np.full(len(images),1/len(images)),size=2000)
    def interval(v,ix):
        v=np.asarray(v);ix=np.asarray(ix)
        good=np.isfinite(v);v=v[good];ix=ix[good]
        if not len(v):return dict(n=0,mean=None,ci95=None)
        total=np.bincount(ix,weights=v,minlength=len(images));count=np.bincount(ix,minlength=len(images)).astype(float)
        den=np.einsum('bi,i->b',draws,count,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
        b=np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den>0)
        return dict(n=len(v),mean=float(v.mean()),ci95=np.nanquantile(b,[.025,.975]).tolist())
    def selected(r,cohort,group):
        return (cohort=='all_matches' or r['coefficient_all_fail']=='True') and (group=='all' or (r['group']!='high' if group=='nonhigh' else r['group']==group))
    matched={int(r['annotation_id']) for r in ranking if r['arm']=='original'}
    accounting=[]
    for group in ['low','middle','high','nonhigh']:
        rr=[r for r in allgt if (float(r['ici'])>.5+1e-10 if group=='high' else float(r['ici'])<=.5+1e-10 if group=='nonhigh' else float(r['ici'])<=1e-10 if group=='low' else 1e-10<float(r['ici'])<=.5+1e-10)]
        remaining=[r for r in rr if r['coefficient_all_fail']=='True']
        accounting.append(dict(group=group,gt=len(rr),all_coefficient_fail=len(remaining),
            fail_with_fixed_bbox50=sum(r['matched']=='True' for r in remaining),
            fail_without_fixed_bbox50=sum(r['matched']!='True' for r in remaining)))
    values=defaultdict(list);metas={}
    fields=['threshold_best_iou','crop_coverage','neighbor_exposure','background_exposure','best_without_neighbor','best_without_background','fixed640_iou','fixed_original_iou']
    for r in ranking:
        aid=int(r['annotation_id']);mode='original' if r['arm']=='original' else 'coefficient';metas[aid]=r
        values[mode,aid].append({k:float(r[k]) for k in fields})
    rankingmeans=[];classification=[]
    for cohort in ['all_matches','residual_all3fail']:
        for group in ['all','low','middle','high','nonhigh']:
            ids=[t for t,r in metas.items() if selected(r,cohort,group)]
            if not ids:continue
            for mode in ['original','coefficient']:
                rankingmeans.append(dict(cohort=cohort,group=group,mode=mode,n=len(ids),**{
                    k:float(np.mean([np.mean([v[k] for v in values[mode,t]]) for t in ids])) for k in fields}))
                # Distinguish guarantees across the 3 saved coefficient runs.
                cases={'crop_below75':0,'threshold_reaches75_allseeds':0,'threshold_fails75_allseeds':0,
                    'threshold_mixed':0,'threshold_fails_crop_sufficient':0,'remove_neighbor_rescues_allseeds':0,'remove_background_rescues_allseeds':0}
                for t in ids:
                    vv=values[mode,t];best=[v['threshold_best_iou'] for v in vv];crop=vv[0]['crop_coverage']
                    cases['crop_below75']+=crop<.75
                    cases['threshold_reaches75_allseeds']+=min(best)>=.75
                    fail=max(best)<.75;cases['threshold_fails75_allseeds']+=fail;cases['threshold_mixed']+=min(best)<.75<=max(best)
                    cases['threshold_fails_crop_sufficient']+=fail and crop>=.75
                    cases['remove_neighbor_rescues_allseeds']+=fail and min(v['best_without_neighbor'] for v in vv)>=.75
                    cases['remove_background_rescues_allseeds']+=fail and min(v['best_without_background'] for v in vv)>=.75
                classification.append(dict(cohort=cohort,group=group,mode=mode,n=len(ids),**cases))
    qfields=['coverage','iou','neighbor','background','other','neighbor_fpr','background_fpr']
    qmeta={};qvalues=defaultdict(list);feasible=defaultdict(set);overshoot=[]
    for r in coverage:
        aid=int(r['annotation_id']);target=float(r['requested_coverage']);mode='original' if r['arm']=='original' else 'coefficient'
        if r['status']!='complete':continue
        qmeta[aid]=r;feasible[target,r['arm']].add(aid);overshoot.append(int(r['tie_overshoot']))
        qvalues[target,mode,aid].append([float(r[k]) if r[k] else np.nan for k in qfields])
    qmeans=[];contrasts=[];controlled=[]
    for target in [.8,.9,.95]:
        sets=[feasible[target,arm] for arm in cfg['arms']]
        if any(s!=sets[0] for s in sets):raise RuntimeError('Geometry feasibility depends on arm')
        tids=sorted(sets[0]);arr={m:np.array([np.mean(qvalues[target,m,t],axis=0) for t in tids]) for m in ['original','coefficient']}
        for cohort in ['all_matches','residual_all3fail']:
            for group in ['all','low','middle','high','nonhigh']:
                take=np.array([selected(qmeta[t],cohort,group) for t in tids]);ix=[iidx[int(qmeta[t]['image_id'])] for t in np.asarray(tids)[take]]
                if not take.any():continue
                for mode in ['original','coefficient']:
                    mm={}
                    for j,k in enumerate(qfields):
                        v=arr[mode][take,j];v=v[np.isfinite(v)];mm[k]=float(v.mean()) if len(v) else None;mm[k+'_n']=len(v)
                    qmeans.append(dict(target=target,cohort=cohort,group=group,mode=mode,n=int(take.sum()),**mm))
                for j,k in enumerate(qfields):
                    contrasts.append(dict(target=target,cohort=cohort,group=group,metric=k,comparison='coefficient-original',**interval((arr['coefficient']-arr['original'])[take,j],ix)))
            if target==.9:
                for mode in ['original','coefficient']:
                    for metric in ['iou','neighbor','background','neighbor_fpr','background_fpr']:
                        j=qfields.index(metric);rr=[]
                        for n,t in enumerate(tids):
                            r=qmeta[t]
                            if not selected(r,cohort,'all'):continue
                            value=arr[mode][n,j]
                            rr.append(dict(image_id=int(r['image_id']),category_id=int(r['category_id']),size=r['size'],box_bin=r['box_bin'],high=r['high']=='True',value=float(value) if np.isfinite(value) else None))
                        controlled.append(dict(cohort=cohort,mode=mode,metric=metric,**comparison(rr,('category_id','size','box_bin'),'value',images,draws)))
                for metric in ['iou','neighbor','background','neighbor_fpr']:
                    j=qfields.index(metric);rr=[]
                    for n,t in enumerate(tids):
                        r=qmeta[t]
                        if not selected(r,cohort,'all'):continue
                        value=(arr['coefficient']-arr['original'])[n,j]
                        rr.append(dict(image_id=int(r['image_id']),category_id=int(r['category_id']),size=r['size'],box_bin=r['box_bin'],high=r['high']=='True',value=float(value) if np.isfinite(value) else None))
                    controlled.append(dict(cohort=cohort,mode='coefficient-original',metric=metric,**comparison(rr,('category_id','size','box_bin'),'value',images,draws)))
    result=dict(accounting=accounting,ranking_means=rankingmeans,classification=classification,coverage_means=qmeans,
        contrasts=contrasts,composition=controlled,feasible={str(t):len(feasible[t,'original']) for t in [.8,.9,.95]},
        ties=dict(rows=len(overshoot),nonzero=sum(v>0 for v in overshoot),max_overshoot=max(overshoot)),
        replay=dict(count=receipt['fixed_iou_replays'],maxerr=receipt['max_replay_error']),
        scope='All300exploredtrainimages; fixedbbox50matches nottaskdenominator. GT-selected640threshold curves, notAP ororiginalresbest. '
            'RawCOCOnearestexact labels crowd/paddingexcluded. Three savedseed mean first. '
            'Paired2000imagebootstrap pointwise, no multipleadjustment. Commonstrata>=3GT/group, min-countweights recomputed. '
            'NeighborFPR excludes zeroexposure with separate denominator; residualsubset conditioned onallthreeofficialfailures, notcausal densityeffect.',
        hashes={n:sha(run/n) for n in ['ranking.csv','coverage.csv','all_gt.csv']})
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(accounting=accounting,ties=result['ties'],replay=result['replay'],
        primary=[r for r in contrasts if r['target']==.9 and r['group']=='high' and r['metric'] in ['iou','neighbor','background','neighbor_fpr']])),flush=True)


if __name__=='__main__':main()
