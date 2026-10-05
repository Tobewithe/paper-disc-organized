"""Stock-loss equivalence, independent CCL oracle, and real GPU backward."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ccl_loss import CCLSegmentationLoss, install_ccl

ROOT=Path(__file__).resolve().parent

def reference(coeff,fg,assignment,batch):
    terms=[]
    for b in range(len(coeff)):
        boxes=batch['bboxes'][batch['batch_idx']==b]
        indices=assignment[b][fg[b]]
        selected=coeff[b][fg[b]].float()
        ids=indices.unique().tolist()
        for p,i in enumerate(ids):
            for j in ids[p+1:]:
                a=boxes[i].float();z=boxes[j].float()
                lo=torch.maximum(a[:2]-a[2:]/2,z[:2]-z[2:]/2)
                hi=torch.minimum(a[:2]+a[2:]/2,z[:2]+z[2:]/2)
                inter=(hi-lo).clamp(min=0).prod()
                iou=inter/(a[2:].prod()+z[2:].prod()-inter+1e-6)
                if iou>.05:
                    c1=selected[indices==i].mean(0);c2=selected[indices==j].mean(0)
                    terms.append(F.relu(F.cosine_similarity(c1,c2,dim=0)-.1))
    return torch.stack(terms) if terms else coeff.reshape(-1)[:0]

def synthetic_tests():
    crit=CCLSegmentationLoss.__new__(CCLSegmentationLoss)
    crit.ccl_iou=.05;crit.ccl_margin=.1
    gen=torch.Generator(device='cuda').manual_seed(41)
    coeff=(torch.randn(2,7,32,device='cuda',generator=gen)*.1+1).requires_grad_(True)
    fg=torch.tensor([[1,1,1,1,1,1,0],[1,1,1,1,1,0,0]],device='cuda',dtype=torch.bool)
    assigned=torch.tensor([[0,0,1,1,2,2,9],[0,0,1,1,2,9,9]],device='cuda')
    batch=dict(batch_idx=torch.tensor([0,0,0,1,1,1],device='cuda'),
        cls=torch.tensor([[0],[1],[1],[0],[0],[1]],device='cuda'),
        bboxes=torch.tensor([[.4,.4,.4,.4],[.5,.4,.4,.4],[.95,.95,.05,.05],[.4,.4,.3,.3],[.45,.4,.3,.3],[.95,.95,.05,.05]],device='cuda'))
    terms=crit.pair_penalties(coeff,fg,assigned,batch)
    ref=reference(coeff,fg,assigned,batch)
    torch.testing.assert_close(terms,ref,rtol=1e-6,atol=1e-7)
    assert len(terms)==2 # includes deliberately different-class pair in first image
    grad=torch.autograd.grad(terms.sum(),coeff,retain_graph=True)[0]
    grad_ref=torch.autograd.grad(ref.sum(),coeff)[0]
    torch.testing.assert_close(grad,grad_ref,rtol=1e-5,atol=1e-7)
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert torch.count_nonzero(grad[0,4:])==0 and torch.count_nonzero(grad[1,4:])==0
    no_fg=torch.zeros_like(fg)
    assert len(crit.pair_penalties(coeff,no_fg,assigned,batch))==0
    assert len(crit.pair_penalties(coeff,fg,torch.zeros_like(assigned),batch))==0
    # Same GT candidates cannot form a pair; nonoverlapping targets cannot either.
    far=deepcopy(batch);far['bboxes'][:,:2]=torch.tensor([[.1,.1],[.5,.5],[.9,.9]]*2,device='cuda');far['bboxes'][:,2:]=.01
    assert len(crit.pair_penalties(coeff,fg,assigned,far))==0
    half=coeff.detach().to(torch.bfloat16).requires_grad_(True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        values=crit.pair_penalties(half,fg,assigned,batch)
    assert values.dtype==torch.float32
    torch.testing.assert_close(values,reference(half,fg,assigned,batch),rtol=1e-6,atol=1e-7)
    return dict(eligible_pairs=2,loss_max_abs=float((terms-ref).abs().max()),gradient_max_abs=float((grad-grad_ref).abs().max()),
        empty_foreground=True,single_gt=True,nonoverlap=True,cross_class_included=True,bf16_reduced_fp32=True)

def main():
    torch.set_num_threads(4);torch.manual_seed(73);np.random.seed(73);random.seed(73)
    result=dict(synthetic=synthetic_tests())
    cfg=get_cfg(overrides=json.loads((ROOT/'train_config.json').read_text()))
    data=check_det_dataset(str(ROOT/'coco_clean.yaml'),autodownload=False)
    ds=build_yolo_dataset(cfg,str(ROOT/'stress_train.txt'),16,data,mode='val',rect=False)
    samples=[ds[i] for i in range(16)]
    batch=ds.collate_fn(samples[:2])
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    batch['img']=batch['img'].float()/255
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt')).model.cuda().train().requires_grad_(True)
    model.args=cfg
    # One forward supplies exactly the same outputs to all compared criteria.
    preds=model(batch['img'])
    stock=E2ELoss(model,v8SegmentationLoss)
    os.environ['CCL_WEIGHT']='0';zero=E2ELoss(model,CCLSegmentationLoss)
    loss0,items0=stock(preds,batch)
    lossz,itemsz=zero(preds,batch)
    torch.testing.assert_close(loss0,lossz,rtol=0,atol=0)
    assert items0.keys()==itemsz.keys()
    for key in items0:torch.testing.assert_close(items0[key],itemsz[key],rtol=0,atol=0)
    coeffs=[preds[k]['mask_coefficient'] for k in ['one2many','one2one']]
    g0=torch.autograd.grad(loss0.sum(),coeffs,retain_graph=True)
    gz=torch.autograd.grad(lossz.sum(),coeffs,retain_graph=True)
    for a,b in zip(g0,gz):torch.testing.assert_close(a,b,rtol=0,atol=0)
    result['lambda_zero']=dict(loss_max_abs=float((loss0-lossz).abs().max()),coefficient_gradient_max_abs=max(float((a-b).abs().max()) for a,b in zip(g0,gz)),logging_items_exact=True)
    os.environ['CCL_WEIGHT']='0.1';ccl=E2ELoss(model,CCLSegmentationLoss)
    lossc,itemsc=ccl(preds,batch)
    delta=lossc-loss0
    expected=.1*len(batch['img'])*(ccl.o2m*ccl.one2many.last_raw+ccl.o2o*ccl.one2one.last_raw)
    assert ccl.one2many.last_pairs>0 and ccl.one2one.last_pairs>0
    torch.testing.assert_close(delta[[0,2,3,4]],torch.zeros(4,device='cuda'),rtol=0,atol=0)
    assert abs(float(delta[1])-expected)<2e-5,(float(delta[1]),expected)
    direct=torch.autograd.grad(delta.sum(),coeffs,retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in direct)
    lossc.sum().backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    result['positive_lambda']=dict(o2m_pairs=ccl.one2many.last_pairs,o2o_pairs=ccl.one2one.last_pairs,weighted_added_loss=float(delta[1]),expected=expected,
        coefficient_gradient_l1=[float(g.abs().sum()) for g in direct],parameter_grads=len(grads))
    del preds,stock,zero,ccl,loss0,lossz,lossc,coeffs,g0,gz,direct,grads,batch
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    install_ccl();model.criterion=None
    model.criterion=model.init_criterion()
    assert isinstance(model.criterion.one2many,CCLSegmentationLoss) and isinstance(model.criterion.one2one,CCLSegmentationLoss)
    # BF16 real 640px batch, all 16 images chosen by GT count, no predictions.
    batch=ds.collate_fn(samples)
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    batch['img']=batch['img'].float()/255
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,items=model(batch)
    loss.sum().backward();torch.cuda.synchronize()
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result['stress_batch']=dict(images=16,instances=len(batch['cls']),input=list(batch['img'].shape),loss=[float(v) for v in loss.detach()],
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.perf_counter()-start,
        o2m_pairs=model.criterion.one2many.last_pairs,o2o_pairs=model.criterion.one2one.last_pairs)
    result['status']='PASS'
    (ROOT/'audits/CCL_GRADIENT_PASS.json').write_text(json.dumps(result,indent=2))
    print('CCL_GRADIENT_PASS',json.dumps(result),flush=True)

if __name__=='__main__':main()
