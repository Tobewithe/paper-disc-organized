"""Post-hoc bound at an EXISTING affine solution. NO optimization or mask evaluation.

Only official fit objects are read. Output-gradient projection uses the exact
implemented frozen design, with no numerical singular-value truncation. Identical
and zero columns may be removed with entrywise proofs; all other columns remain.
Error allowances below are conservative floating-point estimates, conditional on
ordinary correctly rounded FP64 arithmetic and the documented transcendental and
SVD backward-error allowances; this is not an interval-arithmetic proof.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

LAMBDA = 0.003
UNIT = np.finfo(np.float64).eps / 2

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def gamma(n):
    v = int(n) * UNIT
    return v / (1-v)

def modular_rank(array, prime):
    """A full-rank modular minor proves exact rank of the dyadic input matrix.
    A rank-deficient modular result is only a lower bound on real rank.
    """
    a = np.empty(array.shape, dtype=np.int64)
    denom_inv = {}
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            n, d = float(array[i,j]).as_integer_ratio()
            if d not in denom_inv:
                dm = d % prime
                if dm == 0:
                    raise ZeroDivisionError(f"float denominator divisible by prime {prime}")
                denom_inv[d] = pow(dm, -1, prime)
            a[i,j] = ((n % prime) * denom_inv[d]) % prime
    rank = 0
    for j in range(a.shape[1]):
        nz = np.flatnonzero(a[rank:,j])
        if len(nz) == 0: continue
        pivot = rank + int(nz[0]); a[[rank,pivot]] = a[[pivot,rank]]
        a[rank,j:] = a[rank,j:] * pow(int(a[rank,j]), -1, prime) % prime
        factors = a[rank+1:,j].copy()
        a[rank+1:,j:] = (a[rank+1:,j:] - factors[:,None]*a[rank,j:]) % prime
        rank += 1
        if rank == min(a.shape): break
    return rank

def basis_certificate(H):
    # These removals are exact in every frozen entry, not an SVD threshold.
    retained, removed = [], []
    for j in range(H.shape[1]):
        if np.all(H[:,j] == 0):
            removed.append({'column':j, 'reason':'entrywise_exact_zero'}); continue
        equal = next((k for k in retained if np.array_equal(H[:,j], H[:,k])), None)
        if equal is not None:
            removed.append({'column':j, 'reason':'entrywise_exact_duplicate', 'same_as':equal}); continue
        retained.append(j)
    X = H[:,retained]
    ranks, rank_prime_errors = {}, {}
    for p in (1000003,1000033,1000037,1000039):
        try:
            ranks[str(p)] = modular_rank(X,p)
        except ZeroDivisionError as exc:
            rank_prime_errors[str(p)] = str(exc)
    if not ranks:
        raise RuntimeError('All modular-rank primes were unusable for exact float rationals')
    full_rank_proven = max(ranks.values()) == X.shape[1]
    Q, s, VT = np.linalg.svd(X, full_matrices=False)
    # Pessimistic SVD/backward arithmetic allowance (explicitly conditional).
    operator_allowance = 1e-10 * max(1.,float(np.linalg.norm(X, 'fro')))
    s_lower = float(s[-1]) - operator_allowance
    reconstruction = float(np.linalg.norm(X - (Q*s)@VT, 'fro'))
    orthogonality = float(np.linalg.norm(Q.T@Q-np.eye(len(s)), 2))
    # QR/SVD basis need not be exactly orthonormal in floating point.
    angle_allowance = (reconstruction + operator_allowance) / s_lower if s_lower > 0 else None
    valid = full_rank_proven and s_lower > 0 and orthogonality < 1e-10
    data = {'shape':list(H.shape), 'retained_columns':retained, 'exact_removed_columns':removed,
            'modular_ranks':ranks, 'modular_rank_prime_errors':rank_prime_errors,
            'full_column_rank_proven_for_retained_dyadic_matrix':full_rank_proven,
            'singular_values':s.tolist(), 'no_nonzero_singular_values_discarded':True,
            'orthogonality_error_2':orthogonality, 'svd_reconstruction_error_fro':reconstruction,
            'backward_error_allowance':operator_allowance, 'smallest_singular_value_lower':s_lower,
            'subspace_angle_error_allowance':angle_allowance, 'valid_under_numerical_allowances':valid}
    return Q, data

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    for key in ('cache','shared','oracle','manifest','solver_source','out'):
        ap.add_argument('--'+key.replace('_','-'), required=True, type=Path)
    ap.add_argument('--seg-gain', type=float, default=9.83241)
    ap.add_argument('--chunk-pixels', type=int, default=32768)
    ap.add_argument('--max-seconds', type=float, default=1200)
    args = ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    def progress(stage, **kwargs):
        print(json.dumps({'stage':stage,'elapsed_s':time.monotonic()-start,**kwargs}),flush=True)
        if time.monotonic()-start > args.max_seconds: raise TimeoutError('Frozen certificate fixed time budget exceeded')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('Use the authorized remote GPU; no local fallback')
    device=torch.device('cuda')
    hashes={str(p):sha(p) for p in (args.shared,args.oracle,args.manifest,args.solver_source,args.cache/'INDEX.json')}
    saved=torch.load(args.shared,map_location='cpu',weights_only=False)
    A=[a.double() for a in saved['A']]; stats=saved['stats']
    before=[a.clone() for a in A]
    official=json.loads(args.manifest.read_text(encoding='utf-8-sig'))['identity_list']['fit']
    index=json.loads((args.cache/'INDEX.json').read_text())['fit']
    identities=[]; features=[]; records_by_image={}; cache_hashes={}
    for number,item in enumerate(index):
        iid=int(item['image_id']); path=args.cache/'images'/f'{iid:012d}.pt'
        cache_hashes[path.name]=sha(path)
        x=torch.load(path,map_location='cpu',weights_only=False)
        entries=[]
        for k,r in enumerate(x['rows']):
            level=int(r['level']); raw=int(r['raw_id']); mu,sd=stats[level]
            h=torch.cat(((x['h'][raw].float().double()-mu.double())/sd.double(),torch.ones(1,dtype=torch.float64)))
            if h.numel()!=A[level].shape[0] or A[level].shape[1]!=32: raise ValueError('Native affine shape mismatch')
            identity={'split':'fit','image_id':iid,'annotation_id':int(r['annotation_id']), 'branch':'one2one',
                      'raw_id':raw,'pyramid_level':level,'target_gt_idx':int(r['gt_index'])}
            identities.append(identity); features.append(h.numpy()); entries.append((len(identities)-1,k,level,raw))
        records_by_image[iid]=entries
        if number%100==0: progress('read_fit_design',images=number)
    if identities!=official: raise ValueError('Frozen official manifest/order mismatch')
    N=len(identities); H=np.stack(features); del features
    if N!=6058: raise ValueError('Official fixed fit denominator changed')
    levels=np.array([r['pyramid_level'] for r in identities])
    bases={}; rank_audit={}
    for level in range(3):
        bases[level],rank_audit[str(level)]=basis_certificate(H[levels==level])
    progress('basis_complete',N=N, retained=[len(rank_audit[str(l)]['retained_columns']) for l in range(3)])
    write(args.out/'DESIGN_AUDIT.json',rank_audit)
    np.savez_compressed(args.out/'FROZEN_DESIGN.npz',H=H,levels=levels)
    del bases # Recreated after collecting the gradients; no feature/parameter updates.
    G=np.zeros((N,32)); Gerror=np.zeros((N,32)); losses=[]; zeros=[]; bces=[]; regs=[]; errvalues=[]; softplus_values=[]
    coeff_errors=[]; pixel_count=0
    gpu_A=[a.to(device) for a in A]
    for number,item in enumerate(index):
        iid=int(item['image_id']); path=args.cache/'images'/f'{iid:012d}.pt'
        if sha(path)!=cache_hashes[path.name]: raise ValueError('Fit cache changed between reads')
        x=torch.load(path,map_location='cpu',weights_only=False)
        # Exact preprocessing used by the original Bank: float32 interpolation on CPU.
        up=F.interpolate(x['proto'].float()[None],(640,640),mode='bilinear',align_corners=False)[0]
        flat=up.reshape(32,-1); masks=x['masks'].float().reshape(-1)
        for i,k,level,raw in records_by_image[iid]:
            h=torch.from_numpy(H[i]).to(device); d=h@gpu_A[level]
            c0=x['coeff'][raw].float().double().to(device); c=c0+d
            d_error=gamma(h.numel()+2)*(h.abs()@gpu_A[level].abs())
            c_error=d_error+2*UNIT*(c0.abs()+d.abs())
            box=x['target_boxes'][k].float()
            support=ops.crop_mask(torch.ones((1,640,640),dtype=torch.float32),box[None])[0].bool().reshape(-1)
            ids=torch.nonzero(support).flatten(); owner=int(x['owners'][k])
            if len(ids)==0: raise ValueError('Empty support cannot be deleted')
            pixel_count+=len(ids)
            area=float(((box[2:]-box[:2])/640.).prod()*640.*640.)
            weight=args.seg_gain/area
            chunk_bce=[]; chunk_zero=[]; chunk_sp=[]; chunk_ev=[]
            grad=torch.zeros(32,dtype=torch.float64,device=device)
            grad_err=torch.zeros_like(grad); abs_contrib=torch.zeros_like(grad)
            for lo in range(0,len(ids),args.chunk_pixels):
                pp=flat[:,ids[lo:lo+args.chunk_pixels]].T.contiguous().double().to(device)
                yy=(masks[ids[lo:lo+args.chunk_pixels]]==owner+1).double().to(device)
                z=pp@c; z0=pp@c0; absp=pp.abs()
                ell=F.softplus(z,threshold=1000.)-yy*z
                ell0=F.softplus(z0,threshold=1000.)-yy*z0
                chunk_bce.append(weight*float(ell.sum())); chunk_zero.append(weight*float(ell0.sum()))
                chunk_sp.append(weight*float((F.softplus(z)-yy*z).sum()))
                zz_error=gamma(34)*(absp@c.abs())+absp@c_error
                prob=torch.sigmoid(z); ee=weight*(prob-yy)
                add=ee@pp; grad+=add
                absadd=ee.abs()@absp; abs_contrib+=absadd
                # Logistic is 1/4 Lipschitz; FP64 sigmoid absolute roundoff allowance 8u.
                grad_err+=(weight*(.25*zz_error+8*UNIT))@absp
                grad_err+=(gamma(len(yy)+10)+8*UNIT)*absadd
                # BCE is 1-Lipschitz; log/exp error allowance and reduction allowance.
                chunk_ev.append(weight*float((zz_error+16*UNIT*(1+z.abs())).sum()) +
                                weight*gamma(len(yy)+10)*float(ell.abs().sum()))
            chunks=math.ceil(len(ids)/args.chunk_pixels)
            grad+=LAMBDA*d
            grad_err+=gamma(chunks+5)*(abs_contrib+LAMBDA*d.abs())+LAMBDA*d_error
            G[i]=grad.cpu().numpy(); Gerror[i]=grad_err.cpu().numpy()
            reg=LAMBDA*float(d.square().sum())/2
            value_error=math.fsum(chunk_ev)+LAMBDA*float((d.abs()*d_error).sum())+gamma(70)*abs(reg)
            bce=math.fsum(chunk_bce)
            losses.append(bce+reg); zeros.append(math.fsum(chunk_zero)); bces.append(bce); regs.append(reg)
            errvalues.append(value_error); softplus_values.append(math.fsum(chunk_sp)+reg)
        if number%50==0: progress('frozen_gradient',images=number, candidates=len(losses),pixels=pixel_count)
    if len(losses)!=N: raise ValueError('Candidate denominator mismatch')
    Us=math.fsum(losses)/N; J0=math.fsum(zeros)/N
    objective_roundoff=math.fsum(errvalues)/N+1e-10 # protects aggregate/design error margin
    projection=[]; eps_raw_terms=[]; eps_safe_terms=[]; GA=[]
    for level in range(3):
        mask=levels==level; Hl=H[mask]; Gl=G[mask]; El=Gerror[mask]
        Q,info=basis_certificate(Hl); rank_audit[str(level)]=info
        Z=Q.T@Gl
        norm=float(np.linalg.norm(Z,'fro'))
        error=float(np.linalg.norm(El,'fro'))
        gram_allowance=gamma(len(Gl)+3)*float(np.linalg.norm(np.abs(Q).T@np.abs(Gl),'fro'))
        g_norm=float(np.linalg.norm(Gl,'fro'))
        if info['valid_under_numerical_allowances']:
            norm_upper=(norm+gram_allowance)/math.sqrt(1-info['orthogonality_error_2'])
            norm_upper+=info['subspace_angle_error_allowance']*(g_norm+error)+error
            safe=(norm_upper**2)/(2*LAMBDA*N)
            raw=norm**2/(2*LAMBDA*N)
        else:
            # Never discard a nonzero direction to claim a certificate. The whole
            # candidate-output space is a conservative relaxation, not a new model.
            norm_upper=g_norm+error; safe=norm_upper**2/(2*LAMBDA*N); raw=None
        ga=Hl.T@Gl/N; GA.append(ga)
        projection.append({'level':level,'n':len(Gl),'basis_columns':Q.shape[1],
            'output_gradient_global_mean_norm':norm/N,'projected_unaveraged_gradient_norm':norm,
            'coefficient_gradient_error_fro_allowance':error,'projection_roundoff_allowance':gram_allowance,
            'projected_gradient_norm_upper':norm_upper,'raw_epsilon_term':raw,
            'conservative_epsilon_term':safe,'parameter_gradient_fro':float(np.linalg.norm(ga)),
            'rank_resolved':info['valid_under_numerical_allowances']})
        eps_raw_terms.append(raw); eps_safe_terms.append(safe)
    eps_safe=math.fsum(eps_safe_terms)+2*objective_roundoff
    eps_raw=sum(eps_raw_terms) if all(x is not None for x in eps_raw_terms) else None
    oracle=torch.load(args.oracle,map_location='cpu',weights_only=False)
    oid=[(r['image_id'],r['annotation_id'],r['raw_id'],r['pyramid_level'],r['target_gt_idx']) for r in oracle['identities']]
    expected=[(r['image_id'],r['annotation_id'],r['raw_id'],r['pyramid_level'],r['target_gt_idx']) for r in identities]
    if oid!=expected: raise ValueError('Existing independent oracle identity mismatch')
    Ui=math.fsum(float(r['objective']) for r in oracle['identities'])/N
    epsI=math.fsum(float(r['gradient_norm'])**2/(2*LAMBDA) for r in oracle['identities'])/N
    independent_roundoff=1e-8
    delta=Us-Ui
    low=max(0.,delta-eps_safe-objective_roundoff-independent_roundoff)
    high=delta+epsI+objective_roundoff+independent_roundoff
    gap_bound=[low,high]
    params_unchanged=all(torch.equal(a,b) for a,b in zip(A,before)) and hashes[str(args.shared)]==sha(args.shared)
    if not params_unchanged: raise ValueError('Frozen shared parameters changed')
    np.savez_compressed(args.out/'GRADIENTS.npz',G=G,Gerror=Gerror,
                        gradient_A_P3=GA[0],gradient_A_P4=GA[1],gradient_A_P5=GA[2])
    write(args.out/'DESIGN_AUDIT.json',rank_audit)
    write(args.out/'CACHE_MANIFEST.json',cache_hashes)
    result={'scope':'existing official fit affine parameters; no optimizer; no model forward; no oracle solve; no mask evaluation',
      'lambda':LAMBDA,'segmentation_gain':args.seg_gain,'N':N,'fit_images':len(index),
      'fit_pixels':pixel_count,'U_S':Us,'J0':J0,'bce':math.fsum(bces)/N,'regularizer':math.fsum(regs)/N,
      'U_S_reference_softplus20':math.fsum(softplus_values)/N,
      'U_I':Ui,'epsilon_I_from_existing_gradients':epsI,'independent_gradient_converged_count':sum(r['gradient_norm']<=1e-4 for r in oracle['identities']),
      'independent_iteration_limit_count':sum(r['exit_reason']=='iteration_limit' for r in oracle['identities']),
      'epsilon_S_raw':eps_raw,'epsilon_S_with_allowances':eps_safe,'objective_error_allowance':objective_roundoff,
      'independent_record_roundoff_allowance':independent_roundoff,'shared_independent_current_gap':delta,
      'shared_independent_optimal_gap_interval':gap_bound,'projections':projection,
      'params_unchanged':params_unchanged,'input_sha256':hashes,'elapsed_s':time.monotonic()-start,
      'numerical_status':'conditional FP64 certificate; not directed-rounding/interval certification',
      'numerical_assumptions':['standard FP64 gamma_n rounding model; no overflow/underflow of relevant terms',
       'sigmoid absolute error <=8u; softplus/log-exp <=16u*(1+abs(z)); use threshold=1000',
       'SVD backward error <=1e-10*max(1,frobenius_design); all nonzero columns retained',
       'independent oracle stored gradients and objectives reflect the same frozen objective'],
      'decision':'positive lower bound supports fixed-fit mapping restriction under numerical allowances' if low>0 else 'bound not discriminating; no optimization authorized'}
    write(args.out/'CERTIFICATE.json',result); progress('complete',U_S=Us,epsilon_S=eps_safe,gap_interval=gap_bound)

if __name__=='__main__': main()
