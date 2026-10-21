"""Finite LP search; exact certificates are validated outside the optimizer."""
import time
import multiprocessing as mp
from fractions import Fraction as Q
import numpy as np
from scipy.optimize import linprog
from certificates import exact_row, rationalize_farkas, margin_upper, verify


def _exact_worker(queue, proto, pixels, labels):
    try:
        rows = [exact_row(proto, *p) for p in pixels]
        queue.put({'weights': rationalize_farkas(rows, labels)})
    except BaseException as e:
        queue.put({'error': repr(e)})


def certificate(proto, pixels, labels, kind, numeric_weights=None):
    if not len(pixels) or len(pixels) > 64:
        return None
    c = {'kind': kind, 'pixels_yx': np.asarray(pixels).astype(int).tolist(),
         'labels': np.asarray(labels).astype(bool).tolist()}
    if kind == 'farkas':
        ctx = mp.get_context('spawn'); queue = ctx.Queue()
        worker = ctx.Process(target=_exact_worker, args=(queue, proto, c['pixels_yx'], c['labels']))
        worker.start()
        try:
            result = queue.get(timeout=30)
        except Exception:
            result = {}
        finally:
            if worker.is_alive(): worker.terminate()
            worker.join(); queue.close()
        if not result.get('weights'):
            return None
        c['weights'] = result['weights']
    else:
        c['weights'] = [str(Q(float(max(0., w)))) for w in numeric_weights]
        rows = [exact_row(proto, *p) for p in c['pixels_yx']]
        c['upper_exact'] = str(margin_upper(rows, c['labels'], c['weights']))
        c['upper'] = float(Q(c['upper_exact']))
    verify(proto, c)
    c['exact_verification'] = 'passed'
    return c


def scan(B, labels, coefficients, basis_error):
    c = np.asarray(coefficients, dtype=np.float64)
    z = B@c
    # Standard gamma_n forward-error bound; first interpolation's 4 terms
    # have dyadic exact weights. basis_error conservatively bounds that stage.
    u = np.finfo(np.float64).eps/2
    error = (64*u/(1-64*u))*(np.abs(B)@np.abs(c)) + basis_error*np.abs(c).sum()
    signed = z*(2*labels.astype(float)-1)
    return {'strict_sign_lower': float((signed-error).min()),
            'strict_sign_verified': bool(np.all(signed-error > 0)),
            'floating_sign_mismatches': int(np.count_nonzero((z>0)!=labels)),
            'max_roundoff_bound': float(error.max())}


def audit(B, labels, pixels, proto, basis_error):
    start = time.monotonic(); labels = np.asarray(labels, bool)
    A = B*(2*labels.astype(float)-1)[:, None]
    s = np.abs(B).sum(1)
    active = np.unique(np.linspace(0,len(B)-1,min(len(B),512),dtype=int))
    history=[]; candidate=None; farkas=None; upper=None
    options={'presolve':True,'time_limit':30.,'dual_feasibility_tolerance':1e-9,
             'primal_feasibility_tolerance':1e-9}
    # Feasibility uses positive strictness only; no coefficient norm restriction.
    for iteration in range(12):
        if time.monotonic()-start >= 120: break
        r=linprog(np.zeros(32),A_ub=-A[active],b_ub=-labels[active].astype(float),
                  bounds=[(None,None)]*32,method='highs-ds',options=options)
        history.append({'stage':'feasibility','iteration':iteration,'rows':len(active),'status':int(r.status)})
        if r.status==2:
            d=linprog(np.zeros(len(active)),A_eq=np.vstack([A[active].T,labels[active].astype(float)]),
                      b_eq=np.r_[np.zeros(32),1.],bounds=(0,None),method='highs-ds',options=options)
            if d.success:
                support=np.flatnonzero(d.x>0)
                farkas=certificate(proto,pixels[active[support]],labels[active[support]],'farkas')
            break
        if not r.success: break
        candidate=r.x
        values=A@candidate-labels.astype(float)
        if values.min()>=-1e-8:break
        worst=np.argsort(values,kind='stable')[:512]
        expanded=np.unique(np.r_[active,worst])
        if len(expanded)==len(active):break
        active=expanded
    # Largest normalized symmetric margin. A subset dual is already a valid
    # upper bound for the full system; a lower bound always scans ALL rows.
    objective=np.r_[np.zeros(64),-1.]
    for iteration in range(12):
        if time.monotonic()-start >= 120:break
        aa=A[active];ss=s[active]
        constraint=np.vstack([np.c_[-aa,aa,ss],np.r_[np.ones(64),0.]])
        r=linprog(objective,A_ub=constraint,b_ub=np.r_[np.zeros(len(active)),1.],
                  bounds=[(0,None)]*64+[(None,None)],method='highs-ds',options=options)
        history.append({'stage':'margin','iteration':iteration,'rows':len(active),'status':int(r.status)})
        if not r.success:break
        candidate=r.x[:32]-r.x[32:64]
        weights=-r.ineqlin.marginals[:len(active)]
        support=np.flatnonzero(weights>0)
        if len(support) and len(support)<=64:
            try:
                proposal=certificate(proto,pixels[active[support]],labels[active[support]],'margin_upper',weights[support])
                if proposal and (upper is None or proposal['upper']<upper['upper']):upper=proposal
            except (ValueError,AssertionError):pass
        violations=A@candidate-r.x[-1]*s
        if violations.min()>=-1e-8:break
        expanded=np.unique(np.r_[active,np.argsort(violations,kind='stable')[:512]])
        if len(expanded)==len(active):break
        active=expanded
    result={'history':history,'elapsed_seconds':time.monotonic()-start,
            'farkas':farkas,'margin_certificate':upper,'coefficients_float64':None,
            'linear_status':'CERTIFIED_INFEASIBLE_REAL_LINEAR_GT_GRID' if farkas else 'NUMERICALLY_UNRESOLVED'}
    if candidate is not None and np.isfinite(candidate).all():
        result['coefficients_float64']=candidate.tolist()
        result['scan']=scan(B,labels,candidate,basis_error)
        norm=float(np.abs(candidate).sum())
        # s>=0, zero rows forbid strict sign verification, correctly unresolved.
        z=A@candidate
        u=np.finfo(float).eps/2
        err=64*u/(1-64*u)*(np.abs(B)@np.abs(candidate))+basis_error*norm
        usable=s>0
        s_upper=s+32*basis_error+64*u*np.abs(B).sum(1)
        lower=float(np.min((z[usable]-err[usable])/(s_upper[usable]*norm))) if norm and usable.any() else 0.
        result['margin_lower']=lower
        if result['scan']['strict_sign_verified']:
            if farkas:raise AssertionError('exact Farkas contradicts verified primal')
            result['linear_status']='VERIFIED_STRICTLY_SEPARABLE_REAL_LINEAR'
    result['small_margin_threshold']=1e-4
    result['margin_status']='CERTIFIED_SMALL' if upper and upper['upper']<=1e-4 and result.get('margin_lower',0)>0 else (
        'VERIFIED_LARGER_THAN_THRESHOLD' if result.get('margin_lower',0)>1e-4 else 'UNRESOLVED_OR_INFEASIBLE')
    return result
