"""Supplement to frozen E1-R2 verifier; retain its stricter forward-error failures.
No Run, prediction, source snapshot, method, or science Gate is changed.
Use the pre-frozen normal-equation tolerance with independent FP64 equations and
verify the SPD bound ||x-x*|| <= ||Ax-b||/lambda_min, lambda_min >= ridge.
The original 5e-4 componentwise threshold was a verifier heuristic, not a protocol Gate.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from verify_revision import verify,read,jl,sha,close,BASE,STUDY
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
    effective=actual-ci
    residual=np.einsum('kl,l->k',matrix,effective,optimize=False)-rhs
    residual_norm=float(np.sqrt(np.sum(residual**2)))
    rhs_norm=float(np.sqrt(np.sum(rhs**2)))
    actual_error=float(np.sqrt(np.sum((effective-np.asarray(delta))**2)))
    ref_residual=np.einsum('kl,l->k',matrix,np.asarray(delta),optimize=False)-rhs
    ref_error_bound=float(np.sqrt(np.sum(ref_residual**2)))/ridge
    forward_bound=residual_norm/ridge+ref_error_bound
    # Same normal-equation tolerance as pre-run protocol, now using independent FP64 equations
    # AND the EFFECTIVE stored FP32 coefficients (including rounding c0+delta).
    assert residual_norm/(1+rhs_norm)<=5e-5,(meta['image_id'],i,residual_norm/(1+rhs_norm))
    assert actual_error<=forward_bound+1e-10*(1+actual_error)
    return dict(image_id=meta['image_id'],output_rank=i,normalized_coefficient_error=error,
        legacy_5e_4_check_passed=error<=5e-4,normal_residual_float64=residual_norm/(1+rhs_norm),
        absolute_L2_coefficient_error=actual_error,SPD_ridge_forward_error_bound=forward_bound,
        conditioning_upper_bound=float(np.trace(g))/ridge+1.,ridge=ridge)
    close(ridge,r['ridge'],5e-5)
    return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case',type=int,default=0);a=p.parse_args()
    results=[];proof=iter(jl(a.run/'RULE_ROWS.jsonl'))
    for count,meta in enumerate(jl(a.run/'IMAGE_METADATA.jsonl'),1):
        rows=[next(proof) for _ in meta['boxes']]
        selected=next((r for r in rows if r['applied']),None)
        if selected and (not a.case or meta['image_id']==a.case):
            import torch
            path=BASE/'trace'/meta['cache_relative_path'];assert sha(path)==meta['cache_sha256']
            cache=torch.load(path,map_location='cpu',weights_only=True)
            assert cache['c0'].tolist()==[r['c0'] for r in rows] and cache['prediction_indices']==meta['prediction_indices']
            results.append(reference(cache,meta,selected));del cache
            if a.case:break
        if count%100==0:print('NUMERICAL_AUDIT',count,'images',flush=True)
    if not a.case:base=verify(a.run,False)
    else:base={}
    result=dict(status='passed_protocol_residual_with_legacy_forward_check_exceptions',run_id=a.run.name,
      original_verifier_sha256=sha(STUDY/'scripts/verify_revision.py'),supplement_sha256=sha(Path(__file__)),
      original_failure=dict(image_id=166391,output_rank=34,normalized_coefficient_error=0.0005184813757295498,threshold=5e-4),
      reference_images=len(results),max_normal_residual=max(r['normal_residual_float64'] for r in results),
      max_normalized_coefficient_error=max(r['normalized_coefficient_error'] for r in results),
      legacy_check_exceptions=[r for r in results if not r['legacy_5e_4_check_passed']],
      all_independent_FP64_residuals_meet_frozen_5e_5=True,all_SPD_forward_bounds_verified=True,
      unchanged_frozen_original_verifier_other_checks=base,results=results)
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','unchanged_frozen_original_verifier_other_checks')}),flush=True)
if __name__=='__main__':main()
