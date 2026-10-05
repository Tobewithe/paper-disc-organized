"""Post-hoc threshold/coverage curve on fixed evaluation images.

This is a descriptive diagnostic, not a deployable method evaluation: the
aggregate evaluation GT coverage locates an interpolated operating point.
Every bootstrap draw relocates that point. No selected threshold is claimed as
a held-out parameter, and no AP is computed for interpolated masks.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from threshold_coverage_control import measure,read_rows,require,cohort_summary,FIELDS,FAMILIES
from frozen_mechanism_probe import sha,write_json,write_csv


def interpolate(curve, target, grid):
    coverage=curve[:,0]
    require(bool(np.all(np.diff(coverage)<=1e-10)),'Nonmonotone coverage curve')
    require(coverage[-1]<=target<=coverage[0],'Target coverage outside curve')
    # Adjacent threshold points bracketing the target; choose the first crossing.
    k=int(np.flatnonzero(coverage<=target)[0])
    if k==0:return curve[0],float(grid[0]),dict(lower=0,upper=0,weight=0.)
    weight=float((coverage[k-1]-target)/(coverage[k-1]-coverage[k]))
    return (1-weight)*curve[k-1]+weight*curve[k],float((1-weight)*grid[k-1]+weight*grid[k]),dict(lower=k-1,upper=k,weight=weight)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.root;out=a.out;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=json.loads((a.source/'protocol.json').read_text());ids=source['evaluation_image_ids'];grid=source['threshold_grid']
    prior=root/'diagnostics/full_local_comparison_20260911';cache=root/'diagnostics/full_val_cache_20260911'
    ann_path=root/'data/annotations/instances_val2017.json';meta_path=root/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'
    write_json(out/'protocol.json',dict(status='DESCRIPTIVE_POST_HOC_NOT_DEPLOYABLE',training=False,evaluation_image_ids=ids,threshold_grid=grid,
        definition='Plot fixed baseline threshold sweep against mean GT coverage. Evaluate linearly interpolated metrics at each learned-family mean coverage using evaluation GT. The operating point is an oracle diagnostic, not a deployable selected parameter or new AP result. Bracketing raw points reported.',
        bootstrap='2,000 paired image draws; recompute baseline curve and learned mean coverage, then interpolate anew each draw. Three observed learned seeds averaged; no training-population interval. Pointwise, exploratory, no multiple-testing correction.',
        source_protocol_sha256=sha(a.source/'protocol.json'),source_summary_sha256=sha(a.source/'summary.csv'),prior_spatial_sha256=sha(prior/'spatial.csv'),script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('threshold_coverage_control.py'))))
    wanted=set(ids);old=[r for r in read_rows(prior/'spatial.csv') if int(r['image_id']) in wanted]
    replay={int(r['target_annotation']):r for r in old if r['arm']=='initial'}
    meta={int(r['annotation_id']):r for r in read_rows(meta_path)}
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ann_path))
    allrows=[];start=time.monotonic();xor=0;replay_error=0.
    for n,iid in enumerate(ids,1):
        with np.load(cache/'val'/f'{iid}.npz') as item:rows,x,e=measure(gt,item,iid,grid,meta,replay,'evaluation_curve')
        allrows.extend(rows);xor+=x;replay_error=max(replay_error,e)
        if n%50==0:
            state=dict(phase='curve',completed=n,total=len(ids),seconds=round(time.monotonic()-start,1));print(json.dumps(state),flush=True);write_json(out/'progress.json',state)
    write_csv(out/'spatial.csv',allrows);write_csv(out/'curve.csv',cohort_summary(allrows))
    base=sorted([r for r in allrows if r['threshold']==0.],key=lambda r:r['target_annotation']);aids=[r['target_annotation'] for r in base]
    image_lookup={iid:j for j,iid in enumerate(ids)};image_index=np.array([image_lookup[r['image_id']] for r in base]);high=np.array([r['target_ici']>.5+1e-10 for r in base])
    curves=[]
    for tau in grid:
        part=sorted([r for r in allrows if r['threshold']==tau],key=lambda r:r['target_annotation']);require([r['target_annotation'] for r in part]==aids,'Curve cohort mismatch')
        curves.append(np.array([[r[k] for k in FIELDS] for r in part]))
    curves=np.stack(curves,1)
    learned={}
    for family in FAMILIES:
        seeds=[]
        for seed in range(3):
            part=sorted([r for r in old if r['arm']==f'{family}_s{seed}'],key=lambda r:int(r['target_annotation']))
            require([int(r['target_annotation']) for r in part]==aids,'Learned cohort mismatch');seeds.append(np.array([[float(r[k]) for k in FIELDS] for r in part]))
        learned[family]=np.mean(seeds,0)
    rng=np.random.default_rng(20260911);draw=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000).astype(float);results=[]
    for group,mask in [('all',np.ones(len(base),bool)),('high',high),('low',~high)]:
        denominator=np.bincount(image_index[mask],minlength=len(ids));boot_den=draw@denominator;require(bool(np.all(boot_den>0)),'Empty group')
        image_sum=np.zeros((len(ids),len(grid),len(FIELDS)));np.add.at(image_sum,image_index[mask],curves[mask])
        curve=image_sum.sum(0)/denominator.sum();boot_curve=(draw@image_sum.reshape(len(ids),-1)).reshape(2000,len(grid),len(FIELDS))/boot_den[:,None,None]
        for family,values in learned.items():
            sums=np.zeros((len(ids),len(FIELDS)));np.add.at(sums,image_index[mask],values[mask]);point=sums.sum(0)/denominator.sum();boot_learned=(draw@sums)/boot_den[:,None]
            control,tau,bracket=interpolate(curve,point[0],grid);boot_delta=[]
            for b in range(2000):matched,_,_=interpolate(boot_curve[b],boot_learned[b,0],grid);boot_delta.append(boot_learned[b]-matched)
            delta=np.array(boot_delta)*100;ci=np.quantile(delta,[.025,.975],axis=0)
            results.append(dict(family=family,group=group,targets=int(mask.sum()),evaluation_GT_operating_point=True,interpolated_threshold=tau,bracket=bracket,
                bracket_thresholds=[grid[bracket['lower']],grid[bracket['upper']]],bracket_metrics=[dict(zip(FIELDS,(curve[k]*100).tolist())) for k in [bracket['lower'],bracket['upper']]],
                learned_metrics_pct=dict(zip(FIELDS,(point*100).tolist())),interpolated_control_metrics_pct=dict(zip(FIELDS,(control*100).tolist())),
                contrasts={k:dict(mean_pp=float((point[j]-control[j])*100),ci_low_pp=float(ci[0,j]),ci_high_pp=float(ci[1,j])) for j,k in enumerate(FIELDS)}))
    write_json(out/'DESCRIPTIVE_ANALYSIS.json',dict(scope='Post-hoc aggregate coverage-matched curve diagnostic, no deployable parameter or new AP estimate. Validation data reused.',results=results))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,images=len(ids),measured_replay_pixel_xor=xor,spatial_replay_max_error=replay_error,seconds=time.monotonic()-start,hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    for r in results:
        if r['group']=='high':print(json.dumps(r),flush=True)

if __name__=='__main__':main()
