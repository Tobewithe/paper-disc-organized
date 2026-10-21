"""Meaningful model, maximin and denominator controls, before science freeze."""
import json
import numpy as np
import torch
from connectivity_loss import graph_tree,trace_path,select_pixels,make_spec,mask_loss
from scipy.sparse.csgraph import breadth_first_order
from lcf_model import make_model

def run(spec,loss):
    tests=[]
    # The stronger route avoids the weak direct middle row.
    p=np.full((3,3),.9);p[1,1]=.01
    tree=graph_tree(p,np.ones_like(p,bool));previous=breadth_first_order(tree,3,directed=False,return_predecessors=True)[1]
    path=trace_path(previous,3,5)
    assert 4 not in path;tests.append('maximin_avoids_weak_route')
    gt=np.zeros((7,9),bool);gt[1:6,1:4]=1;other=np.zeros_like(gt);other[1:6,6:8]=1
    target=make_spec(gt,[other],1,0,loss)
    z=torch.tensor(np.where(gt,1.,-.5),requires_grad=True)
    points,labels,account=select_pixels(z,target,loss,True)
    assert points and all(int(gt.ravel()[i])==v for i,v in zip(points,labels));tests.append('critical_labels_GT_consistent')
    control,values,receipt=select_pixels(z,target,loss,False)
    assert len(control)==len(points) and values==labels;tests.append('control_same_count_and_class')
    multipart=gt|other;sp=make_spec(multipart,[],1,0,loss)
    assert sp['GT_components']==2 and sp['negative'];tests.append('visible_multipart_separation')
    overlap=make_spec(gt,[gt.copy()],1,0,loss)
    assert not overlap['negative'];tests.append('legitimate_overlap_not_negative')
    torch.manual_seed(19);logits=torch.randn(3,8,8,requires_grad=True);baseline=torch.randn_like(logits)
    roi=torch.ones_like(logits,dtype=torch.bool);targets={0:(torch.rand(8,8)>.5).float(),2:(torch.rand(8,8)>.5).float()}
    whole,_=mask_loss(logits,baseline,roi,targets,{},'A',loss)
    denominator={'GT':2,'preservation':3,'connection':0};parts=[]
    for i in range(3):
        local={0:targets[i]} if i in targets else {}
        v,_=mask_loss(logits[i:i+1],baseline[i:i+1],roi[i:i+1],local,{},'A',loss,denominators=denominator);parts.append(v)
    assert torch.allclose(whole,sum(parts),atol=1e-6,rtol=1e-6);tests.append('chunk_denominators_exact')
    cache={'image_id':99,'batch_position':0,'proto':torch.randn(32,6,6),'c0':torch.randn(3,32),
           'boxes_input':torch.tensor([[0.,0.,24.,24.],[4.,4.,20.,20.],[0.,0.,0.,0.]]),
           'classes':torch.tensor([1,2,3]),'scores':torch.tensor([.7,.4,.2]),'input_hw':[24,24]}
    features=[torch.randn(1,c,h,h) for c,h in [(256,12),(512,6),(512,3)]]
    models={a:make_model(a,spec) for a in ['A','B','C','D']}
    states=[m.state_dict() for m in models.values()]
    assert all(all(torch.equal(states[0][k],v[k]) for k in states[0]) for v in states)
    counts={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()};assert len(set(counts.values()))==1
    tests.append('four_arms_exact_parameters_and_initial_state')
    active={}
    for a,m in models.items():
        before,base=m(cache,features);assert torch.equal(before,base)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3);seen=set()
        for step in range(3):
            opt.zero_grad();changed,base=m(cache,features);value=(changed-1).square().mean();value.backward()
            for n,q in m.named_parameters():
                if q.grad is not None:
                    assert torch.isfinite(q.grad).all()
                    if torch.count_nonzero(q.grad):seen.add(n)
            opt.step()
        assert seen=={n for n,_ in m.named_parameters()},(a,{n for n,_ in m.named_parameters()}-seen)
        assert not cache['proto'].requires_grad and not cache['c0'].requires_grad
        active[a]=sorted(seen)
    tests+=['all_trainable_tensors_receive_gradients','base_P_c0_detached','zero_delta_all_arms']
    return {'status':'passed','tests':tests,'parameters':counts,'active_parameters':active,'synthetic_updates_only':True}

if __name__=='__main__':
    from lcf_runtime import read,STUDY,write
    config=read(STUDY/'CONFIG.json');result=run(config['model'],config['loss']);write(STUDY/'preflight/COMPONENT_TESTS.json',result);print(json.dumps(result))
