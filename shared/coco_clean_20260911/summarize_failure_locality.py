"""Conditional, image-cluster summaries; GT oracles never become task AP."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,json,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from frozen_mechanism_probe import sha,write_json
from three_region_probe import write_csv
from relative_ownership_experiment import read

ARMS=['original','threshold_oracle','proto_oracle','local_oracle','coordinate_oracle','shuffle_oracle']
METRICS=['sample_iou','coverage','same_neighbor','background','all_fp','auc','train_iou']


def grouping(rows):
    high=np.array([float(r['ici'])>.5+1e-10 for r in rows]);fail=np.array([float(r['iou'])<.75 for r in rows])
    adequate=np.array([float(r['crop_ceiling'])>=.75 for r in rows])
    area=np.array([float(r['coco_area']) for r in rows])
    return dict(all=np.ones(len(rows),bool),high=high,low=~high,high_fail75=high&fail,
        high_small=high&(area<32**2),high_medium=high&(area>=32**2)&(area<96**2),high_large=high&(area>=96**2),
        high_fail75_mediumlarge=high&fail&(area>=32**2),
        low_fail75=~high&fail,high_fail75_cropadequate=high&fail&adequate,high_fail75_cropblocked=high&fail&~adequate)


def paired_ci(delta,mask,ii,draw):
    count=np.bincount(ii[mask],minlength=draw.shape[1]);den=draw@count;valid=den>0
    sums=np.stack([np.bincount(ii[mask],weights=delta[mask,j],minlength=draw.shape[1]) for j in range(delta.shape[1])],1)
    boot=(draw@sums)[valid]/den[valid,None]*100
    return delta[mask].mean(0)*100,np.quantile(boot,[.025,.975],axis=0),int((count>0).sum())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out;started=time.monotonic()
    receipt=json.loads((out/'COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for name,digest in receipt['hashes'].items():assert sha(out/name)==digest,name
    protocol=json.loads((out/'protocol.json').read_text());exact=read(out/'exact_error_maps.csv');rr=read(out/'readouts.csv');statuses=read(out/'readout_statuses.csv')
    lookup={(r['split'],int(r['annotation_id'])):r for r in exact};bytarget=defaultdict(list)
    for r in rr:bytarget[(r['split'],int(r['annotation_id']))].append(r)
    expected={(arm,seed,fold) for arm in ARMS for seed in range(3) for fold in range(2)}
    complete={k:rows for k,rows in bytarget.items() if len(rows)==36 and {(r['readout'],int(r['seed']),int(r['fold'])) for r in rows}==expected}
    targetrows=[]
    for key,rows in complete.items():
        for arm in ARMS:
            selected=[r for r in rows if r['readout']==arm]
            targetrows.append(dict(split=key[0],image_id=int(rows[0]['image_id']),annotation_id=key[1],arm=arm,
                **{m:float(np.mean([float(r[m]) for r in selected])) for m in METRICS}))
    write_csv(out/'target_readout_means.csv',targetrows)
    locality=[];aggregate=[];contrasts=[];density=[];cohorts=[];statussummary={};stability=[]
    for split,ids in protocol['images'].items():
        ids=list(map(int,ids));idx={iid:j for j,iid in enumerate(ids)};draw=np.random.default_rng(20260911).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
        allgt=[r for r in exact if r['split']==split];matched=[r for r in allgt if r['status']=='matched']
        statussummary[split]=dict(gt=len(allgt),images=len(ids),gt_status=dict(Counter(r['status'] for r in allgt)),
            probe_status=dict(Counter(r['status'] for r in statuses if r['split']==split)),
            probe_status_unit='target for too_small_support; target x seed otherwise')
        for label,records in [('all_gt',allgt),('bbox_matched',matched),('all_seed_readout_eligible',[lookup[k] for k in complete if k[0]==split])]:
            for group in ['all','high','low']:
                group_rows=[r for r in records if group=='all' or (float(r['ici'])>.5+1e-10)==(group=='high')]
                cohorts.append(dict(split=split,cohort=label,group=group,n=len(group_rows),images=len({r['image_id'] for r in group_rows})))
        ii=np.array([idx[int(r['image_id'])] for r in matched]);areas=np.array([float(r['valid_gt_pixels']) for r in matched])
        countnames=['same_neighbor_pixels','other_category_pixels','background_pixels','fn_inside_pixels','fn_outside_pixels',
            'same_neighbor_near_pixels','same_neighbor_middle_pixels','same_neighbor_far_pixels',
            'background_near_pixels','background_middle_pixels','background_far_pixels','fn_inside_near_pixels','fn_inside_middle_pixels','fn_inside_far_pixels',
            'same_neighbor_core_pixels','same_neighbor_crop_rim_pixels','background_crop_rim_pixels']
        exact_fields=['iou','coverage','crop_ceiling']+countnames
        values=np.array([[float(r[m]) for m in exact_fields] for r in matched]);values[:,3:]/=areas[:,None]
        for group,mask in grouping(matched).items():
            if not mask.any():continue
            means,intervals,nimages=paired_ci(values,mask,ii,draw)
            totals={m:sum(int(float(matched[k][m])) for k in np.flatnonzero(mask)) for m in countnames}
            row=dict(split=split,group=group,n=int(mask.sum()),images=nimages,mean_percent=dict(zip(exact_fields,map(float,means))),
                ci95_percent={m:list(map(float,intervals[:,j])) for j,m in enumerate(exact_fields)},pixel_totals=totals)
            for region in ['same_neighbor','background','fn_inside']:
                total=totals[region+'_pixels']
                row[region+'_pooled_distance_fractions']={b:totals[region+'_'+b+'_pixels']/total if total else None for b in ['near','middle','far']}
                error_targets=[matched[k] for k in np.flatnonzero(mask) if int(matched[k][region+'_pixels'])>0]
                row[region+'_macro_distance_fractions_among_nonzero']={b:float(np.mean([int(r[region+'_'+b+'_pixels'])/int(r[region+'_pixels']) for r in error_targets])) if error_targets else None for b in ['near','middle','far']}
                row[region+'_nonzero_targets']=len(error_targets)
                counts=sorted([int(r[region+'_pixels']) for r in error_targets],reverse=True)
                row[region+'_top10_targets_pixel_share']=sum(counts[:10])/total if total else None
            total=totals['same_neighbor_pixels'];row['same_neighbor_pooled_core_fraction']=totals['same_neighbor_core_pixels']/total if total else None
            snrows=[matched[k] for k in np.flatnonzero(mask) if int(matched[k]['same_neighbor_pixels'])>0]
            row['same_neighbor_macro_core_fraction_among_nonzero']=float(np.mean([int(r['same_neighbor_core_pixels'])/int(r['same_neighbor_pixels']) for r in snrows])) if snrows else None
            row['n_targets_with_same_neighbor_fp']=sum(int(float(matched[k]['same_neighbor_pixels']))>0 for k in np.flatnonzero(mask))
            locality.append(row)
        keys=sorted(k for k in complete if k[0]==split);base=[lookup[k] for k in keys];ii=np.array([idx[int(r['image_id'])] for r in base])
        bykey={(r['split'],r['annotation_id'],r['arm']):r for r in targetrows}
        arrays={arm:np.array([[bykey[(split,k[1],arm)][m] for m in METRICS] for k in keys]) for arm in ARMS}
        for group,mask in grouping(base).items():
            if not mask.any():continue
            for arm,array in arrays.items():
                aggregate.append(dict(split=split,group=group,arm=arm,n=int(mask.sum()),images=len(set(ii[mask])),**dict(zip(METRICS,map(float,array[mask].mean(0))))))
            for treatment,control in [('threshold_oracle','original'),('proto_oracle','threshold_oracle'),('proto_oracle','original'),('local_oracle','proto_oracle'),('proto_oracle','coordinate_oracle'),('proto_oracle','shuffle_oracle')]:
                means,intervals,nimages=paired_ci(arrays[treatment]-arrays[control],mask,ii,draw)
                contrasts.extend(dict(split=split,group=group,treatment=treatment,control=control,metric=m,n=int(mask.sum()),images=nimages,
                    mean_pp=float(means[j]),ci_low_pp=float(intervals[0,j]),ci_high_pp=float(intervals[1,j])) for j,m in enumerate(METRICS))
        # Descriptive common category x COCO-size standardization, no interaction test.
        strata=defaultdict(lambda:dict(high=[],low=[]))
        for j,r in enumerate(base):
            area=float(r['coco_area']);size='small' if area<32**2 else 'medium' if area<96**2 else 'large'
            strata[(int(r['category_id']),size)]['high' if float(r['ici'])>.5+1e-10 else 'low'].append(j)
        shared={k:v for k,v in strata.items() if v['high'] and v['low']};n=sum(len(v['high'])+len(v['low']) for v in shared.values())
        for treatment,control in [('proto_oracle','threshold_oracle'),('local_oracle','proto_oracle')]:
            delta=(arrays[treatment]-arrays[control])[:,0]
            density.append(dict(split=split,treatment=treatment,control=control,common_strata=len(shared),common_targets=n,
                **{g+'_standardized_iou_gain_pp':float(sum((len(v['high'])+len(v['low']))/n*delta[v[g]].mean() for v in shared.values())*100) if n else None for g in ['high','low']}))
        for group in ['all','high','low']:
            selected=[r for r in statuses if r['split']==split and r['status']=='ok' and (group=='all' or (float(r['ici'])>.5+1e-10)==(group=='high'))]
            corr=np.array([float(r['fold_score_correlation']) for r in selected]);stability.append(dict(split=split,group=group,target_seed_rows=len(corr),median_correlation=float(np.median(corr)),p10_correlation=float(np.quantile(corr,.1)),fraction_negative=float((corr<0).mean())))
    write_csv(out/'readout_aggregate.csv',aggregate);write_csv(out/'cohorts.csv',cohorts)
    write_json(out/'LOCALITY_ANALYSIS.json',dict(scope='Original-resolution normal cropped masks, fixed original bbox50 matching; macro GT-normalized rates and separately labeled pooled pixel fractions; descriptive thresholds, reused data.',status=statussummary,groups=locality))
    write_json(out/'READOUT_ANALYSIS.json',dict(scope='GT-at-diagnostic spatial cross-fitting; all six arms and three sample seeds on the same eligible targets. Pointwise paired 2000 image-cluster bootstrap, fixed checkpoint, reused images; NOT AP, not deployable inference, no causal training claim.',
        contrasts=contrasts,density_standardization=density,fold_stability=stability,complete_targets=len(complete),partial_targets=len(bytarget)-len(complete)))
    write_json(out/'SUMMARY_COMPLETE.json',dict(status='COMPLETE',source_sha256=sha(__file__),seconds=time.monotonic()-started,
        hashes={p.name:sha(p) for p in [out/'target_readout_means.csv',out/'readout_aggregate.csv',out/'cohorts.csv',out/'LOCALITY_ANALYSIS.json',out/'READOUT_ANALYSIS.json']}))
    print(json.dumps(dict(status='COMPLETE',complete_targets=len(complete),partial_targets=len(bytarget)-len(complete),seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
