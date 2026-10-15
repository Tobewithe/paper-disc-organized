"""Independent scalar/reference audit. No model training and no changes to completed Runs."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.stats import hypergeom

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_repulsion_revision_20261015'
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
OLD=ROOT/'experiments/mask_readout_large_mask_gate_20261015/runs/RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8'
sys.path.insert(0,str(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/scripts'))
from verify_repulsion import read,jl,sha,close,scalar_neighbors,statistics


def reference(cache,meta,r):
    # Explicit float64 equations, independent of production torch.linalg.solve and its geometry helper.
    i=r['output_rank'];j=r['neighbor_rank'];boxes=np.asarray(meta['boxes'],dtype=np.float32)
    ctr=(boxes[:,:2]+boxes[:,2:])/np.float32(2);lo=np.maximum(boxes[i,:2],boxes[j,:2]);hi=np.minimum(boxes[i,2:],boxes[j,2:])
    cl=ctr[i]-(boxes[i,2:]-boxes[i,:2])/np.float32(4);ch=ctr[i]+(boxes[i,2:]-boxes[i,:2])/np.float32(4)
    h,w=meta['proto_hw'];ih,iw=meta['input_hw'];x=np.arange(w,dtype=np.float32)[None,:]*np.float32(iw/w);y=np.arange(h,dtype=np.float32)[:,None]*np.float32(ih/h)
    omega=(x>=lo[0])&(x<hi[0])&(y>=lo[1])&(y<hi[1])
    core=(x>=cl[0])&(x<ch[0])&(y>=cl[1])&(y<ch[1])
    omega &= ~core & ((x-ctr[j,0])**2+(y-ctr[j,1])**2 < (x-ctr[i,0])**2+(y-ctr[i,1])**2)
    box=(x>=boxes[i,0])&(x<boxes[i,2])&(y>=boxes[i,1])&(y<boxes[i,3])
    p=cache['proto'].numpy().astype(np.float64);ci=cache['c0'][i].numpy().astype(np.float64);cj=cache['c0'][j].numpy().astype(np.float64)
    zi=np.einsum('k,khw->hw',ci,p,optimize=False);zj=np.einsum('k,khw->hw',cj,p,optimize=False)
    fg=omega&(zi>0)&(zj>zi)
    assert int(omega.sum())==r['omega_pixels'] and int(fg.sum())==r['conflict_pixels'] and int(box.sum())==r['box_pixels']
    num=float((zi[fg]*zj[fg]).mean());den=float((zj[fg]**2).mean());alpha=num/(den+1e-8)
    close(num,r['numerator'],5e-5);close(den,r['denominator'],5e-5);close(alpha,r['alpha'],5e-5)
    pp=p[:,box];m=pp.shape[1];g=np.einsum('kn,ln->kl',pp,pp,optimize=False)/m
    target=np.where(fg[box],-alpha*zj[box],0.);rhs=np.einsum('kn,n->k',pp,target,optimize=False)/m
    ridge=max(1e-8,.01*float(np.trace(g))/len(ci));matrix=g+ridge*np.eye(len(ci))
    # Gaussian elimination with partial pivoting, independent scalar arithmetic.
    a=matrix.tolist();b=rhs.tolist();k=len(b)
    for col in range(k):
        pivot=max(range(col,k),key=lambda row:abs(a[row][col]));a[col],a[pivot]=a[pivot],a[col];b[col],b[pivot]=b[pivot],b[col]
        for row in range(col+1,k):
            factor=a[row][col]/a[col][col]
            for q in range(col+1,k):a[row][q]-=factor*a[col][q]
            b[row]-=factor*b[col]
    delta=[0.]*k
    for row in range(k-1,-1,-1):delta[row]=(b[row]-sum(a[row][q]*delta[q] for q in range(row+1,k)))/a[row][row]
    actual=np.asarray(r['arms']['B']['c_new']);expected=ci+np.asarray(delta)
    error=float(np.max(np.abs(actual-expected)/(1+np.abs(expected))))
    assert error<=5e-4,(error,meta['image_id'],i)
    close(ridge,r['ridge'],5e-5)
    return error


def verify(run,cache_audit):
    state=read(run/'run.json');summary=read(run/'SUMMARY.json');contract=read(run/'EVAL_CONTRACT.json')
    assert state['execution_status']=='completed' and state['exit_code']==0 and read(run/'CONFIG_AUDIT.json')['status']=='passed'
    assert sha(run/'PROTOCOL.md')==state['protocol_sha256'] and sha(run/'EVAL_CONTRACT.json')==state['contract_sha256']
    artifacts=read(run/'ARTIFACT_MANIFEST.json')['artifacts']
    for v in artifacts:assert sha(run/v['relative_path'])==v['sha256'],v['relative_path']
    for rel,digest in read(run/'SOURCE_MANIFEST.json')['files'].items():assert sha(ROOT/rel)==sha(run/'source_snapshot'/rel)==digest,rel
    old_manifest=read(OLD/'ARTIFACT_MANIFEST.json')['artifacts'];pin=next(v['sha256'] for v in old_manifest if v['relative_path']=='NATIVE_SUBSET_predictions.json')
    assert sha(OLD/'NATIVE_SUBSET_predictions.json')==pin
    native=dict(zip(read(OLD/'ROW_MAPPING.json')['original_global_indices'],read(OLD/'NATIVE_SUBSET_predictions.json')))
    predictions={a:read(run/f'{a}_predictions.json') for a in ('A','B')};indices=read(run/'ROW_MAPPING.json')['original_global_indices'];position={idx:i for i,idx in enumerate(indices)}
    assert all(len(v)==len(indices) for v in predictions.values())
    proof_stream=iter(jl(run/'RULE_ROWS.jsonl'));counts={a:Counter() for a in predictions};actions={};sample_checks=0;max_error=0.;diagnostic=Counter()
    group_proof={};full=state['scope']=='full'
    effects={a:list(jl(run/f'{a}_FIXED_PAIR_EFFECTS.jsonl')) for a in predictions}
    target_ids={int(v['baseline_output_detection_id'].rsplit(':',1)[1]) for v in effects['A'] if v['population']=='T'}
    panel=read(run/'DAMAGE_PANEL.json');baseline={(v['image_id'],v['annotation_id']):v for v in jl(run/'BASELINE_SUBSET_GT_LEDGER.jsonl')}
    panel_ids={int(baseline[v['image_id'],v['annotation_id']]['segm75_match_output_id'].rsplit(':',1)[1]) for v in panel['sample'] if (v['image_id'],v['annotation_id']) in baseline}
    for number,meta in enumerate(jl(run/'IMAGE_METADATA.jsonl'),1):
        proofs=[next(proof_stream) for _ in meta['boxes']];neighbors=scalar_neighbors(meta)
        for i,(r,(j,iou)) in enumerate(zip(proofs,neighbors)):
            idx=meta['prediction_indices'][i];assert r['global_native_index']==idx and r['output_rank']==i and r['neighbor_rank']==j
            close(iou,r['neighbor_box_iou']);applied=j>=0 and r['conflict_pixels']>=4 and r['denominator']>1e-8 and r['numerator']>0
            assert r['applied']==applied;actions[idx]=applied;ci=np.asarray(r['c0'],dtype=np.float32)
            alpha=np.float32(r['numerator'])/(np.float32(r['denominator'])+np.float32(1e-8)) if applied else np.float32(0.)
            close(float(alpha),r['alpha'],2e-6);assert 0<=alpha<=1+1e-6
            ca=ci-alpha*np.asarray(proofs[max(j,0)]['c0'],dtype=np.float32)
            assert np.all(np.abs(ca-np.asarray(r['arms']['A']['c_new']))<=1e-5*(1+np.abs(ca)))
            close(r['old_positive_positive']+r['old_negative_negative']+r['old_opposite_sign'],r['old_numerator'],5e-5)
            assert r['normal_residual']<=5e-5
            for ref in ('objective_A','objective_identity'):assert r['objective_B']<=r[ref]+5e-5*(1+r[ref])
            for arm in predictions:
                a=native[idx];b=predictions[arm][position[idx]];info=r['arms'][arm]
                assert a.keys()==b.keys() and all(a[key]==b[key] for key in a if key!='segmentation')
                assert info['mask_changed']==(a['segmentation']!=b['segmentation'])
                if not applied:assert a['segmentation']==b['segmentation'] and ci.tolist()==info['c_new']
                counts[arm]['outputs']+=1;counts[arm]['selected_coefficients']+=applied;counts[arm]['changed_masks']+=info['mask_changed']
            if r['old_applied']:
                diagnostic['old_selected']+=1
                diagnostic['old_no_both_positive_pixels']+=int(r['old_positive_positive']==0)
                diagnostic['old_negative_contribution_exceeds_positive']+=int(r['old_negative_negative']>r['old_positive_positive'])
                diagnostic['old_lost_by_foreground_gate']+=int(not applied)
            if idx in target_ids or idx in panel_ids:group_proof[idx]=r
        selected=next((r for r in proofs if r['applied']),None)
        if cache_audit and selected:
            import torch
            path=BASE/'trace'/meta['cache_relative_path'];assert sha(path)==meta['cache_sha256']
            cache=torch.load(path,map_location='cpu',weights_only=True)
            assert cache['c0'].tolist()==[r['c0'] for r in proofs] and cache['prediction_indices']==meta['prediction_indices']
            max_error=max(max_error,reference(cache,meta,selected));sample_checks+=1;del cache
        if number%100==0:print('INDEPENDENT',number,'images',flush=True)
    assert next(proof_stream,None) is None
    for arm in counts:
        for key,value in counts[arm].items():assert value==summary['arms'][arm]['coverage'][key]
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    gt_path=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'
    if gt_path.exists():raw=gt_path.read_bytes()
    else:
        with zipfile.ZipFile(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/RESULTS_BUNDLE.zip') as archive:raw=archive.read(gt_path.relative_to(ROOT).as_posix())
    assert hashlib.sha256(raw).hexdigest()==contract['input_hashes']['files'][gt_path.relative_to(ROOT).as_posix()]
    coco=COCO();coco.dataset=json.loads(raw);del raw;coco.createIndex();results={}
    for arm,rows in effects.items():
        for r in rows:
            idx=int(r['baseline_output_detection_id'].rsplit(':',1)[1]);truth=coco.annToRLE(coco.anns[r['annotation_id']])
            for field,pred,name in [('baseline_iou',native[idx]['segmentation'],'native'),('method_iou',predictions[arm][position[idx]]['segmentation'],'method')]:
                inter=int(masks.area(masks.merge([pred,truth],intersect=True)));area=int(masks.area(pred));truth_area=int(masks.area(truth));union=area+truth_area-inter
                close(inter/union if union else 0.,r[field]);assert r['confusion'][name]==dict(tp=inter,fp=area-inter,fn=truth_area-inter)
            close(r['delta_iou'],r['method_iou']-r['baseline_iou']);assert r['applied']==actions[idx]
        result=dict(output_rows_verified=len(indices),fixed_pair_RLE_IoU_verified=len(rows))
        if full:
            assert len(rows)==2925 and len(indices)==185621
            for name,target in [('target',True),('all_bbox75',False)]:
                stat=statistics(rows,target);ref=summary['arms'][arm][name];assert stat['images']==ref['images'];close(stat['mean'],ref['mean'])
                for x,y in zip(stat['ci95'],ref['ci95']):close(x,y)
                result[name]=stat
            ledger={(v['image_id'],v['annotation_id']):v for v in jl(run/f'{arm}_GT_LEDGER.jsonl')};assert ledger.keys()==baseline.keys()
            damaged=sum(ledger[v['image_id'],v['annotation_id']]['task_mask75']=='failure' for v in panel['sample']);N=22957;n=1024;lo=damaged;hi=N-(n-damaged)
            while lo<hi:
                mid=(lo+hi+1)//2
                if hypergeom.cdf(damaged,N,mid,n)>=.05:lo=mid
                else:hi=mid-1
            assert hypergeom.cdf(damaged,N,lo,n)>=.05 and hypergeom.cdf(damaged,N,lo+1,n)<.05
            ref=summary['arms'][arm]['damage'];assert ref['damaged']==damaged and ref['upper_damaged_count']==lo;close(ref['upper_rate'],lo/N)
            gates=dict(target_effect=result['target']['mean']>=.005 and result['target']['ci95'][0]>0,all_bbox75_guardrail=result['all_bbox75']['mean']>=-.001,success_damage=lo/N<=.01)
            assert gates==summary['arms'][arm]['gates']
            result.update(damaged=damaged,upper_rate=lo/N,gates=gates,applied_T_positive=sum(r['population']=='T' and r['applied'] and r['delta_iou']>0 for r in rows),
                applied_T_negative=sum(r['population']=='T' and r['applied'] and r['delta_iou']<0 for r in rows))
        results[arm]=result
    groups={}
    for name,ids in [('T',target_ids),('success_original_match',panel_ids)]:
        selected=[group_proof[i] for i in ids if group_proof[i]['applied']];old=[group_proof[i] for i in ids if group_proof[i]['old_applied']]
        groups[name]=dict(total=len(ids),old_selected=len(old),old_negative_dominant=sum(v['old_negative_negative']>v['old_positive_positive'] for v in old),
            revised_selected=len(selected),mean_nonconflict_RMS={a:sum(v['arms'][a]['nonconflict_logit_rms'] for v in selected)/len(selected) if selected else 0 for a in predictions})
    return dict(status='passed',run_id=state['run_id'],run_json_sha256=sha(run/'run.json'),summary_sha256=sha(run/'SUMMARY.json'),
        config_audit_sha256=sha(run/'CONFIG_AUDIT.json'),artifact_files=len(artifacts),arms=results,old_trigger_decomposition=dict(diagnostic),groups=groups,
        all_neighbors_A_formulas_unselected_identity_verified=True,B_independent_reference_image_count=sample_checks,
        B_max_normalized_coefficient_error=max_error,cache_audit=cache_audit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-audit',action='store_true');a=p.parse_args()
    result=verify(a.run,a.cache_audit)
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
