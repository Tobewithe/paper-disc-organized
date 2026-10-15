"""Independent post-run checks: scalar neighbors, explicit Gram, mask counts and statistics."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import hypergeom

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_subspace_repulsion_20261015'
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
OLD=ROOT/'experiments/mask_readout_large_mask_gate_20261015/runs/RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def jl(p):
    with p.open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def close(a,b,tolerance=1e-12):assert abs(a-b)<=tolerance*(1+abs(b)),(a,b)


def scalar_neighbors(meta):
    b=meta['boxes'];score=meta['scores'];cls=meta['classes'];raw=meta['raw_ids'];answer=[]
    for i,a in enumerate(b):
        choices=[]
        for j,c in enumerate(b):
            if cls[i]!=cls[j] or raw[i]==raw[j] or (-score[j],j)>=(-score[i],i):continue
            area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]);area_c=max(0,c[2]-c[0])*max(0,c[3]-c[1])
            intersection=max(0,min(a[2],c[2])-max(a[0],c[0]))*max(0,min(a[3],c[3])-max(a[1],c[1]))
            union=area_a+area_c-intersection;iou=intersection/union if union else 0
            if iou>.1:choices.append((-iou,-score[j],j))
        best=min(choices) if choices else None;answer.append((-1,0.) if best is None else (best[2],-best[0]))
    return answer


def explicit_gram(cache,meta,record):
    i=record['output_rank'];j=record['neighbor_rank'];b=np.asarray(meta['boxes'],dtype=np.float32)
    center=(b[:,:2]+b[:,2:])/np.float32(2);lo=np.maximum(b[i,:2],b[j,:2]);hi=np.minimum(b[i,2:],b[j,2:])
    core_lo=center[i]-(b[i,2:]-b[i,:2])/np.float32(4);core_hi=center[i]+(b[i,2:]-b[i,:2])/np.float32(4)
    h,w=meta['proto_hw'];ih,iw=meta['input_hw'];x=np.arange(w,dtype=np.float32)[None,:]*np.float32(iw/w);y=np.arange(h,dtype=np.float32)[:,None]*np.float32(ih/h)
    region=(x>=lo[0])&(x<hi[0])&(y>=lo[1])&(y<hi[1])
    core=(x>=core_lo[0])&(x<core_hi[0])&(y>=core_lo[1])&(y<core_hi[1])
    region &= ~core & ((x-center[j,0])**2+(y-center[j,1])**2 < (x-center[i,0])**2+(y-center[i,1])**2)
    assert int(region.sum())==record['omega_pixels']
    p=cache['proto'].numpy().astype(np.float64)[:,region]
    gram=np.einsum('kn,ln->kl',p,p,optimize=False)/p.shape[1]
    ci=cache['c0'][i].numpy().astype(np.float64);cj=cache['c0'][j].numpy().astype(np.float64)
    cross=float(np.einsum('k,kl,l->',ci,gram,cj,optimize=False));energy=float(np.einsum('k,kl,l->',cj,gram,cj,optimize=False))
    alpha=cross/(energy+1e-8)
    close(cross,record['numerator'],5e-5);close(energy,record['denominator'],5e-5);close(alpha,record['alpha'],5e-5)
    return abs(alpha-record['alpha'])/(1+abs(alpha))


def statistics(rows,target):
    group=defaultdict(list)
    for r in rows:
        if not target or r['population']=='T':group[r['image_id']].append(r['delta_iou'])
    v=[sum(group[i])/len(group[i]) for i in sorted(group)];rng=np.random.default_rng(1729);draws=[]
    for _ in range(2000):draws.append(sum(v[i] for i in rng.integers(0,len(v),size=len(v)))/len(v))
    return dict(mean=sum(v)/len(v),ci95=np.quantile(draws,[.025,.975]).tolist(),images=len(v))


def verify(run,cache_audit):
    state=read(run/'run.json');summary=read(run/'SUMMARY.json');contract=read(run/'EVAL_CONTRACT.json')
    assert state['execution_status']=='completed' and state['exit_code']==0 and read(run/'CONFIG_AUDIT.json')['status']=='passed'
    assert sha(run/'PROTOCOL.md')==state['protocol_sha256'] and sha(run/'EVAL_CONTRACT.json')==state['contract_sha256']
    artifacts=read(run/'ARTIFACT_MANIFEST.json')['artifacts']
    for a in artifacts:assert sha(run/a['relative_path'])==a['sha256'],a['relative_path']
    for rel,digest in read(run/'SOURCE_MANIFEST.json')['files'].items():
        assert sha(ROOT/rel)==sha(run/'source_snapshot'/rel)==digest,rel
    original=read(OLD/'NATIVE_SUBSET_predictions.json')
    pin=next(a['sha256'] for a in read(OLD/'ARTIFACT_MANIFEST.json')['artifacts'] if a['relative_path']=='NATIVE_SUBSET_predictions.json')
    assert sha(OLD/'NATIVE_SUBSET_predictions.json')==pin
    old_indices=read(OLD/'ROW_MAPPING.json')['original_global_indices'];native={index:r for index,r in zip(old_indices,original)}
    del original
    predictions=read(run/'REPULSION_predictions.json');indices=read(run/'ROW_MAPPING.json')['original_global_indices']
    assert len(indices)==len(predictions);position={idx:i for i,idx in enumerate(indices)}
    proof_stream=iter(jl(run/'RULE_ROWS.jsonl'));counts=Counter();actions={};gram_checks=0;max_gram_error=0.;max_corr=0.
    for number,meta in enumerate(jl(run/'IMAGE_METADATA.jsonl'),1):
        proofs=[next(proof_stream) for _ in meta['boxes']];neighbors=scalar_neighbors(meta)
        for i,(r,(j,iou)) in enumerate(zip(proofs,neighbors)):
            idx=meta['prediction_indices'][i];assert r['output_rank']==i and r['global_native_index']==idx and r['image_id']==meta['image_id']
            assert r['neighbor_rank']==j;close(r['neighbor_box_iou'],iou)
            applied=j>=0 and r['omega_pixels']>0 and r['denominator']>1e-8 and r['numerator']>0
            assert applied==r['applied']
            ci=np.asarray(r['c0'],dtype=np.float32)
            alpha=np.float32(r['numerator'])/np.float32(np.float32(r['denominator'])+np.float32(1e-8)) if applied else np.float32(0)
            close(float(alpha),r['alpha'],2e-6)
            expected=ci-alpha*np.asarray(proofs[max(j,0)]['c0'],dtype=np.float32)
            actual=np.asarray(r['c_new'],dtype=np.float32)
            assert np.all(np.abs(expected-actual)<=1e-5*(1+np.abs(expected)))
            a=native[idx];b=predictions[position[idx]]
            assert a.keys()==b.keys() and all(a[k]==b[k] for k in a if k!='segmentation')
            if not applied:assert a['segmentation']==b['segmentation'] and np.array_equal(ci,actual)
            assert r['mask_changed']==(a['segmentation']!=b['segmentation'])
            counts['outputs']+=1;counts['selected_coefficients']+=applied;counts['changed_masks']+=r['mask_changed']
            actions[idx]=applied
            if applied:max_corr=max(max_corr,abs(r['correlation_after'])/(1+abs(r['numerator'])))
        selected=next((r for r in proofs if r['applied']),None)
        if cache_audit and selected:
            import torch
            path=BASE/'trace'/meta['cache_relative_path'];assert sha(path)==meta['cache_sha256']
            cached=torch.load(path,map_location='cpu',weights_only=True)
            assert cached['prediction_indices']==meta['prediction_indices']
            assert cached['c0'].tolist()==[r['c0'] for r in proofs]
            for field,key in [('boxes_input','boxes'),('scores','scores'),('classes','classes')]:assert cached[field].tolist()==meta[key]
            assert cached['raw_ids']==meta['raw_ids']
            max_gram_error=max(max_gram_error,explicit_gram(cached,meta,selected));gram_checks+=1;del cached
        if number%100==0:print('INDEPENDENT',number,'images',flush=True)
    assert next(proof_stream,None) is None
    for k,v in counts.items():assert summary['coverage'][k]==v,(k,v)
    # Independently calculate RLE intersection/union instead of the production mask.iou call.
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    gt_path=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
    if gt_path.exists():raw=gt_path.read_bytes()
    else:
        with zipfile.ZipFile(STUDY/'RESULTS_BUNDLE.zip') as archive:raw=archive.read(gt_path.relative_to(ROOT).as_posix())
    assert hashlib.sha256(raw).hexdigest()==contract['input_hashes']['files'][gt_path.relative_to(ROOT).as_posix()]
    coco=COCO();coco.dataset=json.loads(raw);del raw;coco.createIndex()
    effects=list(jl(run/'FIXED_PAIR_EFFECTS.jsonl'))
    for r in effects:
        idx=int(r['baseline_output_detection_id'].rsplit(':',1)[1]);gt=coco.annToRLE(coco.anns[r['annotation_id']])
        for field,pred in [('baseline_iou',native[idx]['segmentation']),('method_iou',predictions[position[idx]]['segmentation'])]:
            intersection=int(masks.area(masks.merge([pred,gt],intersect=True)));union=int(masks.area(pred))+int(masks.area(gt))-intersection
            close(intersection/union if union else 0.,r[field])
        close(r['delta_iou'],r['method_iou']-r['baseline_iou']);assert r['applied']==actions[idx]
    result=dict(status='passed',run_id=state['run_id'],artifact_files=len(artifacts),source_files=len(read(run/'SOURCE_MANIFEST.json')['files']),
        output_rows_verified=counts['outputs'],all_coefficient_formulas_and_neighbors_verified=True,fixed_pair_RLE_IoU_verified=len(effects),
        explicit_float64_Gram_real_image_checks=gram_checks,max_normalized_Gram_alpha_error=max_gram_error,
        max_normalized_post_correlation=max_corr,cache_audit=cache_audit,run_json_sha256=sha(run/'run.json'),summary_sha256=sha(run/'SUMMARY.json'),
        config_audit_sha256=sha(run/'CONFIG_AUDIT.json'))
    if state['scope']=='full':
        assert len(effects)==2925 and len(predictions)==185621
        for name,target in [('target',True),('all_bbox75',False)]:
            stat=statistics(effects,target);reference=summary[name];assert stat['images']==reference['images']
            close(stat['mean'],reference['mean'])
            for a,b in zip(stat['ci95'],reference['ci95']):close(a,b)
            result[name]=stat
        panel=read(run/'DAMAGE_PANEL.json');ledger={(r['image_id'],r['annotation_id']):r for r in jl(run/'METHOD_GT_LEDGER.jsonl')}
        baseline={(r['image_id'],r['annotation_id']):r for r in jl(run/'BASELINE_SUBSET_GT_LEDGER.jsonl')}
        assert ledger.keys()==baseline.keys()
        sample=panel['sample'];x=sum(ledger[r['image_id'],r['annotation_id']]['task_mask75']=='failure' for r in sample)
        N=22957;n=len(sample);assert n==1024
        lo=x;hi=N-(n-x)
        while lo<hi:
            mid=(lo+hi+1)//2
            if hypergeom.cdf(x,N,mid,n)>=.05:lo=mid
            else:hi=mid-1
        assert hypergeom.cdf(x,N,lo,n)>=.05 and hypergeom.cdf(x,N,lo+1,n)<.05
        assert summary['damage']['damaged']==x and summary['damage']['upper_damaged_count']==lo
        close(lo/N,summary['damage']['upper_rate'])
        selected_S=0;damaged_unchanged=0
        for r in sample:
            key=r['image_id'],r['annotation_id'];idx=int(baseline[key]['segm75_match_output_id'].rsplit(':',1)[1])
            selected_S+=actions[idx]
            damaged_unchanged+=not actions[idx] and ledger[key]['task_mask75']=='failure'
        result.update(damaged=x,upper_damaged_count=lo,upper_rate=lo/N,success_panel_selected=selected_S,
            success_panel_damage_with_unchanged_original_match=damaged_unchanged,
            gates=dict(target_effect=summary['target']['mean']>=.005 and summary['target']['ci95'][0]>0,
                       all_bbox75_guardrail=summary['all_bbox75']['mean']>=-.001,success_damage=lo/N<=.01),
            applied_T_positive=sum(r['population']=='T' and r['applied'] and r['delta_iou']>0 for r in effects),
            applied_T_negative=sum(r['population']=='T' and r['applied'] and r['delta_iou']<0 for r in effects))
        assert result['gates']==summary['gates']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-audit',action='store_true')
    a=p.parse_args();result=verify(a.run,a.cache_audit)
    with a.out.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)
