"""S016 exact original-COCO decoding and exploratory high/other comparisons."""
import argparse,contextlib,io,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha
from no_candidate_readout_probe import coefficient_logits

ARMS=['original','threshold_oracle','free_coefficient','free_coefficient_bias']

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    receipt=json.loads((out/'COMPLETE.json').read_text());need(receipt['status']=='COMPLETE','Not complete')
    for name,h in receipt['hashes'].items():need(sha(out/name)==h,name)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    ids=json.loads((out/'protocol.json').read_text())['images'];targets=read(out/'targets.csv');metrics=read(out/'metrics.csv')
    lookup={(int(r['annotation_id']),r['arm']):r for r in metrics};rows=[]
    for iid in ids:
        rr=[r for r in targets if int(r['image_id'])==iid and r['fit_status']=='ok']
        if not rr:continue
        with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as z:
            proto=torch.tensor(z['proto'],device='cuda');shape=tuple(map(int,z['shape']));inp=tuple(map(int,z['input_shape']))
        with np.load(out/'images'/f'{iid}.npz') as z:params={k:z[k] for k in z.files}
        for r in rr:
            aid=int(r['annotation_id']);box=torch.tensor(params[f'{aid}_box'],device='cuda');c0=torch.tensor(params[f'{aid}_original_coefficient'],device='cuda')
            for arm in ARMS:
                c=c0 if arm in ['original','threshold_oracle'] else torch.tensor(params[f'{aid}_{arm}_coefficient'],device='cuda')
                bias=float(lookup[(aid,arm)]['bias']);z=coefficient_logits(proto,c,inp,bias)
                binary=ops.crop_mask((z>0)[None].to(torch.uint8),box)[0]
                original=(ops.scale_masks(binary[None,None],shape)[0,0]>.5).cpu().numpy().astype(np.uint8)
                value=float(mu.iou([mu.encode(np.asfortranarray(original))],[gt.annToRLE(gt.anns[aid])],[0])[0,0])
                if arm=='original':need(abs(value-float(r['anchor_official_mask_iou']))<1e-12,'Original IoU replay')
                rows.append(dict(image_id=iid,annotation_id=aid,anchor=r['anchor'],arm=arm,original_coco_mask_iou=value,
                    valid_domain_iou=float(lookup[(aid,arm)]['mask_iou']),input_iou=float(lookup[(aid,arm)]['input_iou']),area=float(r['area'])))
    save_csv(out/'original_coco_mask_iou.csv',rows)
    highdir=ROOT/'diagnostics/no_candidate317_20260912';high=read(highdir/'targets.csv');highmetrics=read(highdir/'metrics.csv');highcoco=read(highdir/'original_coco_mask_iou.csv')
    alltargets={'high':high,'other':targets};allmetrics={'high':highmetrics,'other':metrics};allcoco={'high':highcoco,'other':rows}
    summary=[];stages={};spatial=[];paired=[];composed=[]
    imageids=sorted({int(r['image_id']) for rr in alltargets.values() for r in rr});ix={v:j for j,v in enumerate(imageids)}
    rng=np.random.default_rng(20260912);w=rng.multinomial(len(imageids),np.full(len(imageids),1/len(imageids)),size=2000)
    for density,tt in alltargets.items():
        stages[density]=dict(cohort=len(tt),stages=dict(Counter(r['stage'] for r in tt)),fit_status=dict(Counter(r['fit_status'] for r in tt)))
        tmap={int(r['annotation_id']):r for r in tt};cm={(int(r['annotation_id']),r['arm']):r for r in allcoco[density]};sm={(int(r['annotation_id']),r['arm']):r for r in allmetrics[density]}
        for cohort in ['all_eligible','original_final_match','small','medium_large']:
            eligible=[r for r in tt if r['fit_status']=='ok']
            if cohort=='original_final_match':eligible=[r for r in eligible if r['anchor']=='fixed_original_bbox50_match']
            if cohort=='small':eligible=[r for r in eligible if float(r['area'])<1024]
            if cohort=='medium_large':eligible=[r for r in eligible if float(r['area'])>=1024]
            aids=[int(r['annotation_id']) for r in eligible]
            for arm in ARMS:
                cr=[cm[(aid,arm)] for aid in aids];sr=[sm[(aid,arm)] for aid in aids]
                summary.append(dict(density=density,cohort=cohort,arm=arm,n=len(aids),coco_recovered75=sum(float(r['original_coco_mask_iou'])>=.75 for r in cr),
                    coco_iou=100*float(np.mean([float(r['original_coco_mask_iou']) for r in cr])),
                    **{k:100*float(np.mean([float(r[k]) for r in sr])) for k in ['mask_iou','coverage','same_neighbor','background']}))
            if cohort=='all_eligible':
                for aid in aids:
                    r=tmap[aid];base=float(cm[(aid,'original')]['original_coco_mask_iou']);delta=float(cm[(aid,'free_coefficient')]['original_coco_mask_iou'])-base
                    size=0 if float(r['area'])<1024 else 1 if float(r['area'])<9216 else 2
                    composed.append(dict(image_id=int(r['image_id']),annotation_id=aid,density=density,category_id=int(r['category_id']),area_bin=size,
                        baseline_bin=min(int(base/.15),4),baseline_iou=base,delta_iou=delta,
                        delta_over_threshold=float(cm[(aid,'free_coefficient')]['original_coco_mask_iou'])-float(cm[(aid,'threshold_oracle')]['original_coco_mask_iou']),
                        delta_same_neighbor=float(sm[(aid,'free_coefficient')]['same_neighbor'])-float(sm[(aid,'original')]['same_neighbor']),
                        delta_coverage=float(sm[(aid,'free_coefficient')]['coverage'])-float(sm[(aid,'original')]['coverage']),
                        delta_background=float(sm[(aid,'free_coefficient')]['background'])-float(sm[(aid,'original')]['background'])))
    for field in ['delta_iou','delta_over_threshold','delta_same_neighbor','delta_coverage','delta_background']:
        num=np.zeros((len(imageids),2));den=np.zeros_like(num)
        for r in composed:
            j=ix[r['image_id']];g=int(r['density']=='high');num[j,g]+=r[field];den[j,g]+=1
        bden=w@den;bnum=w@num;valid=(bden>0).all(1);vals=bnum[valid]/bden[valid]
        paired.append(dict(metric=field,high_minus_other_pp=100*float(num[:,1].sum()/den[:,1].sum()-num[:,0].sum()/den[:,0].sum()),
            ci95_pp=(100*np.quantile(vals[:,1]-vals[:,0],[.025,.975])).tolist(),valid_resamples=int(valid.sum()),
            interpretation='Unadjusted conditional failure-cohort interaction, not density causality.'))
    controls=[]
    for dimensions in [('category_id','area_bin'),('category_id','area_bin','baseline_bin')]:
        cells={}
        for r in composed:cells.setdefault(tuple(r[k] for k in dimensions),{'high':[],'other':[]})[r['density']].append(r)
        common={k:v for k,v in cells.items() if len(v['high'])>=3 and len(v['other'])>=3}
        weight=sum(len(v['high'])+len(v['other']) for v in common.values())
        controls.append(dict(strata=list(dimensions),minimum_per_group=3,common_strata=len(common),
            high_included=sum(len(v['high']) for v in common.values()),other_included=sum(len(v['other']) for v in common.values()),
            standardized={field:{g:100*sum((len(v['high'])+len(v['other']))*np.mean([r[field] for r in v[g]]) for v in common.values())/weight if weight else None for g in ['high','other']} for field in ['delta_iou','delta_over_threshold']},
            limitation='Descriptive pooled-size common-stratum weights, no adjusted CI; exclusions disclosed. Baseline bin width .15 fixed before this comparison; not sufficient confounder control.'))
    save_csv(out/'DENSITY_SUMMARY.csv',summary);save_csv(out/'DENSITY_COMPARISON_TARGETS.csv',composed)
    write_json(out/'DENSITY_ANALYSIS.json',dict(stages=stages,summary=summary,interactions=paired,common_stratum_controls=controls,
        scope='Same solver/no new hyperparameter tuning; all other-group matching failed candidates from same300exploredimages. GT aided whole-image fitting, no independent generalization.',
        script_sha256=sha(__file__),high_input_hashes={n:sha(highdir/n) for n in ['targets.csv','metrics.csv','original_coco_mask_iou.csv']},
        csv_hashes={n:sha(out/n) for n in ['original_coco_mask_iou.csv','DENSITY_SUMMARY.csv','DENSITY_COMPARISON_TARGETS.csv']}))
    print(json.dumps(dict(stages=stages,interactions=paired,controls=controls)),flush=True)

if __name__=='__main__':main()
