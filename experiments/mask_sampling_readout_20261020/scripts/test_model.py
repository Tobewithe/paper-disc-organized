"""Engineering only: nested inputs, matched parameters, gradients and padded ROI isolation."""
from ssr_runtime import *

def main():
    import torch
    from ssr_model import ARMS,make_model,predict,coarsen,permutation
    profile,env=environment();device='cuda:0'
    initial=torch.load(read(STUDY/'CONFIG.json')['adapter_initial'],map_location='cpu',weights_only=True)
    torch.manual_seed(901)
    pyramid=[torch.randn(1,c,h,w,device=device,requires_grad=True) for c,h,w in ((256,9,10),(512,5,5),(512,3,3))]
    p=torch.randn(32,18,20,device=device,requires_grad=True)
    c=torch.randn(3,32,device=device)
    cache={'image_id':123,'proto':p,'c0':c,'boxes_input':torch.tensor([[0,0,80,72],[2,4,60,68],[0,0,44,56]],device=device),
      'features':[f[0,:,0,0] for f in pyramid],'raw_locations':[{'level':i,'row':0,'column':0} for i in range(3)],
      'head_feature_shapes':[list(f.shape[1:]) for f in pyramid],'input_hw':[72,80],'batch_position':0,
      'scores':torch.tensor([.9,.8,.7],device=device),'classes':torch.tensor([0,1,2],device=device)}
    receipts={}
    constant=torch.ones(1,5,19,21,device=device)
    require(torch.equal(coarsen(constant),constant),'coarsening must preserve constants')
    idx=permutation(19*21,123,0,19,21,device)
    require(torch.equal(idx.sort().values,torch.arange(len(idx),device=device)),'invalid permutation')
    require(not torch.equal(idx,torch.arange(len(idx),device=device)),'identity intervention')
    for arm in ARMS:
        model=make_model(arm,initial).cuda().train()
        zero=predict(model,cache,pyramid)
        require(torch.equal(zero,torch.zeros_like(zero)),'initial delta is nonzero')
        with torch.no_grad():model.readout[-1].weight.normal_(0,.001)
        out=predict(model,cache,pyramid);out.square().sum().backward()
        grads={n:bool(t.grad is not None and torch.isfinite(t.grad).all() and torch.count_nonzero(t.grad)) for n,t in model.named_parameters()}
        require(all(grads.values()),'inactive trainable tensors: '+str([n for n,v in grads.items() if not v]))
        require(p.grad is None and all(f.grad is None for f in pyramid),'base inputs received gradients')
        receipts[arm]={'trainable_parameters':sum(t.numel() for t in model.parameters()),'zero_delta_exact':True,'all_parameter_gradients_finite_nonzero':True,'base_inputs_detached':True}
        del model,out,zero
    require(len({v['trainable_parameters'] for k,v in receipts.items() if k!='G20k'})==1,'five matched arm capacities differ')
    isolation={}
    for arm in ('A','D'):
        model=make_model(arm,initial).cuda().eval()
        with torch.no_grad():
            model.readout[-1].weight.normal_(0,.001)
            region=torch.randn(2,64,18,20,device=device)
            ctx=torch.randn(2,117,device=device)
            rects=((0,0,10,10),(1,2,16,17))
            together=model._chunk(p.detach(),region,c[:2],ctx,rects,123,(0,1),arm)
            separate=model._chunk(p.detach(),region[:1],c[:1],ctx[:1],rects[:1],123,(0,),arm)
            error=(together[:1]-separate).abs().max().item()
            require(torch.allclose(together[:1],separate,atol=1e-5,rtol=1e-5),'padding affected valid ROI')
            isolation[arm]={'maximum_absolute_error':error,'tolerance':'atol=1e-5,rtol=1e-5; engineering float32 comparison only'}
    write(STUDY/'preflight/MODEL_ENGINEERING.json',{'status':'passed','environment':env,'arms':receipts,'coarse_canvas_and_permutation_checked':True,
      'padding_isolation':isolation,'formal_optimizer_steps':0,'synthetic_gradient_backwards':6,'scientific_result':False,'source_sha256':sha(__file__)})
    print(json.dumps(receipts),flush=True)

if __name__=='__main__':main()
