"""Development-selected scalar thresholds versus frozen learned mask residuals.

No training. The primary diagnostic fixes original bbox ownership. Only the
development images select a scalar per learned family; evaluation IDs are
hash-selected before any threshold output and never choose a parameter.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, contextlib, csv, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import sha, write_json, write_csv

FIELDS=['coverage','same_neighbor','neighbor','background','mask_iou']
FAMILIES=['bce_global','dice_global']

def require(condition, message):
    if not condition: raise RuntimeError(message)

def read_rows(path):
    with path.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

@torch.inference_mode()
def measure(gt, item, iid, thresholds, metadata, replay, phase):
    device='cuda'
    p=torch.as_tensor(item['proto'],device=device)
    c=torch.as_tensor(item['coeff'],device=device)
    b=torch.as_tensor(item['boxes'],device=device)
    ishape=tuple(map(int,item['input_shape']));shape=tuple(map(int,item['shape']))
    mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
    require(len(set(mapping.values()))==len(mapping),'Nonunique fixed ownership')
    if not len(c): return [],0,0
    require(bool(torch.isfinite(p).all() and torch.isfinite(c).all()),'Nonfinite logits input')
    z=F.interpolate((c@p.float().flatten(1)).reshape(1,-1,*p.shape[-2:]),ishape,mode='bilinear',align_corners=False)[0]
    official=ops.process_mask(p,c,b,ishape,upsample=True)
    own_zero=ops.crop_mask((z>0).to(torch.uint8),b)
    xor=int(torch.count_nonzero(official!=own_zero))
    require(xor==0, f'Official decoder mismatch image {iid}: {xor}')
    # GT only defines diagnostic ownership/regions after all baseline logits exist.
    anns=gt.imgToAnns[iid];raster={a['id']:torch.as_tensor(gt.annToMask(a).astype(bool),device=device) for a in anns}
    ordinary=[a for a in anns if not a.get('iscrowd',0)]
    union=torch.stack([raster[a['id']] for a in ordinary]).any(0) if ordinary else torch.zeros(shape,dtype=torch.bool,device=device)
    same={cat:torch.stack([raster[a['id']] for a in ordinary if a['category_id']==cat]).any(0) for cat in {a['category_id'] for a in ordinary}}
    crowds=[raster[a['id']] for a in anns if a.get('iscrowd',0)]
    valid=~torch.stack(crowds).any(0) if crowds else torch.ones(shape,dtype=torch.bool,device=device)
    rows=[];replay_max=0.
    for aid,j in mapping.items():
        own=raster[aid]&valid;area=own.sum().double()
        if not bool(area): continue
        regions=[own, same[gt.anns[aid]['category_id']]&~own&valid, union&~own&valid, ~union&valid]
        weights=torch.stack(regions).flatten(1).float()
        values=[]
        for offset in range(0,len(thresholds),8):
            tau=torch.tensor(thresholds[offset:offset+8],device=device)
            binary=(z[j][None]>tau[:,None,None]).to(torch.uint8)
            cropped=ops.crop_mask(binary,b[j:j+1].expand(len(tau),-1))
            masks=ops.scale_masks(cropped[:,None],shape)[:,0]>.5
            counts=(masks.flatten(1).float()@weights.T).double()
            tp,sn,near,bg=counts.unbind(1)
            metrics=torch.stack([tp/area,sn/area,near/area,bg/area,tp/(area+near+bg)],1)
            require(bool(torch.isfinite(metrics).all()),'Nonfinite metrics')
            values.extend(metrics.cpu().tolist())
        # The original masks and zero-threshold spatial fields must replay the saved run.
        zero_index=thresholds.index(0.)
        old=replay[aid]
        err=max(abs(float(old[k])-v) for k,v in zip(FIELDS,values[zero_index]))
        require(err<1e-10, f'Spatial replay mismatch {iid}/{aid}: {err}')
        replay_max=max(replay_max,err)
        for tau,v in zip(thresholds,values):
            rows.append(dict(phase=phase,image_id=iid,target_annotation=aid,target_ici=float(metadata[aid]['ici_same']),threshold=float(tau),**dict(zip(FIELDS,v))))
    return rows,xor,replay_max

def cohort_summary(rows):
    output=[]
    for tau in sorted({r['threshold'] for r in rows}):
        for group in ['all','high','low']:
            selected=[r for r in rows if r['threshold']==tau and (group=='all' or (r['target_ici']>.5+1e-10)==(group=='high'))]
            if selected: output.append(dict(threshold=tau,group=group,targets=len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in FIELDS}))
    return output

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--evaluation-images',type=int,default=600)
    ap.add_argument('--dev-limit',type=int,default=0)
    a=ap.parse_args();root=a.root;out=a.out;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=root/'diagnostics/full_val_cache_20260911';prior=root/'diagnostics/full_local_comparison_20260911'
    dev_protocol=root/'diagnostics/local_coeff_eval_20260911/protocol.json'
    dev=sorted(json.loads(dev_protocol.read_text())['images'])
    excluded=set(dev)
    if a.dev_limit:dev=dev[:a.dev_limit]
    all_ids=sorted(json.loads((prior/'protocol.json').read_text())['images'])
    evaluate=sorted(sorted(set(all_ids)-excluded,key=lambda x:hashlib.sha256(f'coverage-control:20260911:{x}'.encode()).hexdigest())[:a.evaluation_images])
    require(not set(dev)&set(evaluate),'Overlapping images')
    thresholds=[-.5,0.]+[i/10 for i in range(1,21)]+[3.,4.]
    ann_path=root/'data/annotations/instances_val2017.json';meta_path=root/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'
    protocol=dict(training=False,dev_image_ids=dev,evaluation_image_ids=evaluate,threshold_grid=thresholds,families=FAMILIES,
        selection='One uniform scalar threshold per family; linearly interpolate development high-ICI mean coverage curve to the same family three-seed mean coverage. No evaluation GT selects or modifies threshold. Parameter bracketing is required.',
        criteria='Development coverage confirmation <=0.25 percentage points; evaluation coverage gap reported without retuning. Primary fixed-ownership neighbor/GT error and IoU, with target coverage and background guard. No task AP or precision-matched recall claim.',
        scope='Disjoint parameter-selection/evaluation images within previously explored val2017; not pristine final test. Fixed original bbox matching excludes unmatched GT. Original candidates, boxes and scores fixed. Three learned seeds averaged conditional on their observed values.',
        hashes={str(path.relative_to(root)):sha(path) for path in [ann_path,meta_path,dev_protocol,prior/'spatial.csv',prior/'protocol.json',cache/'protocol.json']},
        script_sha256=sha(__file__),ops_source_sha256=sha(ops.__file__),helper_sha256=sha(Path(__file__).with_name('frozen_mechanism_probe.py')),python_assertions_enabled=__debug__)
    write_json(out/'protocol.json',protocol)
    wanted=set(dev+evaluate);old_rows=[r for r in read_rows(prior/'spatial.csv') if int(r['image_id']) in wanted]
    replay={int(r['target_annotation']):r for r in old_rows if r['arm']=='initial'}
    metadata={int(r['annotation_id']):r for r in read_rows(meta_path)}
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ann_path))
    start=time.monotonic();parity=0;max_replay=0.
    def run(ids,grid,phase):
        nonlocal parity,max_replay
        rows=[]
        for n,iid in enumerate(ids,1):
            with np.load(cache/'val'/f'{iid}.npz') as item:
                rr,xor,error=measure(gt,item,iid,grid,metadata,replay,phase)
            rows.extend(rr);parity+=xor;max_replay=max(max_replay,error)
            if n%25==0 or n==len(ids):
                progress=dict(phase=phase,completed=n,total=len(ids),seconds=round(time.monotonic()-start,1))
                write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        write_csv(out/f'{phase}_spatial.csv',rows)
        return rows
    development=run(dev,thresholds,'development_grid');curve=cohort_summary(development);write_csv(out/'development_curve.csv',curve)
    selected={};dev_high=[r for r in curve if r['group']=='high']
    for family in FAMILIES:
        family_rows=[r for r in old_rows if r['arm'] in [f'{family}_s{s}' for s in range(3)] and int(r['image_id']) in set(dev) and float(r['target_ici'])>.5+1e-10]
        require(len(family_rows)>0,'No high-ICI development instances')
        target=float(np.mean([float(r['coverage']) for r in family_rows]))
        coverage=np.array([r['coverage'] for r in dev_high]);grid=np.array([r['threshold'] for r in dev_high])
        require(bool(np.all(np.diff(coverage)<=1e-12)),'Threshold coverage is not monotone')
        require(coverage[-1]<=target<=coverage[0],'Development target not bracketed')
        tau=float(np.interp(target,coverage[::-1],grid[::-1]))
        selected[family]=dict(threshold=tau,target_dev_coverage=target)
    write_json(out/'SELECTED_THRESHOLDS.json',dict(selection_phase='DEVELOPMENT_ONLY_LOCKED_BEFORE_EVALUATION',families=selected,development_source_sha256=sha(out/'development_grid_spatial.csv')))
    fixed_grid=sorted(set([0.]+[v['threshold'] for v in selected.values()]))
    confirmed=cohort_summary(run(dev,fixed_grid,'development_confirmation'))
    for family,item in selected.items():
        measured=next(r for r in confirmed if r['threshold']==item['threshold'] and r['group']=='high')['coverage']
        item['confirmed_dev_coverage']=measured;item['dev_coverage_gap_pp']=100*(measured-item['target_dev_coverage'])
        require(abs(item['dev_coverage_gap_pp'])<=.25,'Development coverage matching outside predeclared tolerance')
    write_json(out/'DEVELOPMENT_CONFIRMATION.json',selected)
    tested=run(evaluate,fixed_grid,'evaluation');write_csv(out/'evaluation_curve.csv',cohort_summary(tested))
    initial=[r for r in tested if r['threshold']==0.];ids=sorted(r['target_annotation'] for r in initial)
    ii_map={iid:j for j,iid in enumerate(evaluate)};initial=sorted(initial,key=lambda r:r['target_annotation'])
    image_index=np.array([ii_map[r['image_id']] for r in initial]);high=np.array([r['target_ici']>.5+1e-10 for r in initial])
    arrays={'initial':np.array([[r[k] for k in FIELDS] for r in initial])}
    for family,item in selected.items():
        control=sorted([r for r in tested if r['threshold']==item['threshold']],key=lambda r:r['target_annotation'])
        require([r['target_annotation'] for r in control]==ids,'Threshold cohort mismatch')
        arrays['threshold_'+family]=np.array([[r[k] for k in FIELDS] for r in control])
        seeds=[]
        for seed in range(3):
            records=sorted([r for r in old_rows if r['arm']==f'{family}_s{seed}' and int(r['image_id']) in set(evaluate)],key=lambda r:int(r['target_annotation']))
            require([int(r['target_annotation']) for r in records]==ids,'Learned cohort mismatch')
            seeds.append(np.array([[float(r[k]) for k in FIELDS] for r in records]))
        arrays[family]=np.mean(seeds,axis=0)
    summary=[];contrasts=[];rng=np.random.default_rng(20260911);draw=rng.multinomial(len(evaluate),np.full(len(evaluate),1/len(evaluate)),size=2000).astype(float)
    for group,mask in [('all',np.ones(len(initial),dtype=bool)),('high',high),('low',~high)]:
        counts=np.bincount(image_index[mask],minlength=len(evaluate));den=draw@counts
        require(bool(np.all(den>0)),'Empty bootstrap denominator')
        for family,values in arrays.items():summary.append(dict(arm=family,group=group,targets=int(mask.sum()),**{k:float(values[mask,j].mean()*100) for j,k in enumerate(FIELDS)}))
        for family in FAMILIES:
            for control in ['initial','threshold_'+family]:
                delta=arrays[family]-arrays[control]
                for j,k in enumerate(FIELDS):
                    total=np.bincount(image_index[mask],weights=delta[mask,j],minlength=len(evaluate));boot=draw@total/den*100;lo,hi=np.quantile(boot,[.025,.975])
                    contrasts.append(dict(treatment=family,control=control,group=group,metric=k,targets=int(mask.sum()),mean_pp=float(delta[mask,j].mean()*100),ci_low_pp=float(lo),ci_high_pp=float(hi)))
    write_csv(out/'summary.csv',summary);write_json(out/'PAIRED_ANALYSIS.json',dict(bootstrap='2,000 paired image-cluster draws, 3 observed seeds averaged; exploratory pointwise intervals; parameters fixed conditional on development data; no multiplicity adjustment.',contrasts=contrasts))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,dev_images=len(dev),evaluation_images=len(evaluate),measured_replay_pixel_xor=parity,spatial_replay_max_error=max_replay,seconds=time.monotonic()-start,
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    for r in summary:
        if r['group']=='high':print(json.dumps(r),flush=True)
    for r in contrasts:
        if r['group']=='high' and r['control'].startswith('threshold'):print(json.dumps(r),flush=True)

if __name__=='__main__':main()
