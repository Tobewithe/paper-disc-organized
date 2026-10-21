"""Deterministic scientific-validity controls; no dataset or training."""
import numpy as np
from fractions import Fraction as Q
from scipy.optimize import linprog
from certificates import exact_row,rationalize_farkas,margin_upper,verify

def test():
    cases=[]
    rows=[[Q(1)],[Q(1)]];labels=[True,False]
    w=rationalize_farkas(rows,labels);assert w and w==['1','1'];cases.append('opposite_identical_features_exact_Farkas')
    assert rationalize_farkas([[Q(1)],[Q(-1)]],labels) is None;cases.append('separable_not_falsely_certified')
    # Background zero is valid under native >0 semantics.
    r=linprog([0.],A_ub=[[-1.],[0.]],b_ub=[-1.,0.],bounds=[(None,None)],method='highs-ds')
    assert r.success;assert rationalize_farkas([[Q(1)],[Q(0)]],labels) is None;cases.append('background_zero_allowed')
    eps=Q(1,2**20)
    upper=margin_upper([[Q(1),Q(0)],[Q(-1),eps]],[True,True],['1','1'])
    assert 0<upper<Q(1,10000);cases.append('true_small_normalized_margin_upper')
    proto=np.array([[[1.,2.],[3.,4.]]],dtype=np.float32)
    assert exact_row(proto,0,0)==[Q(1)] and exact_row(proto,7,7)==[Q(4)]
    assert exact_row(proto,3,3)==[Q(17,8)];cases.append('dyadic_bilinear_exact_geometry')
    invalid={'kind':'farkas','pixels_yx':[[0,0]],'labels':[True],'weights':['1']}
    try:verify(proto,invalid)
    except AssertionError:cases.append('false_certificate_rejected')
    else:raise AssertionError('false certificate accepted')
    return {'status':'passed','cases':cases,'training_steps':0}

if __name__=='__main__':
    import json
    print(json.dumps(test()))
