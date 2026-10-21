"""Full native input-grid losses with exact image denominators and bounded memory."""
import torch
from torch.utils.checkpoint import checkpoint
from lcf_model import upsample,input_roi
from connectivity_loss import mask_loss,select_pixels

def loss_for(model,cache,features,prepared,config):
    targets,specs,join=prepared
    low,base=model(cache,features);roi=input_roi(cache)
    valid=roi.sum((1,2))>0
    denominators={'preservation':int(valid.sum()),'GT':sum(bool(valid[i]) for i in targets),'connection':len(specs)}
    total=low.sum()*0
    account={'critical_samples':0,'hardness_matched_noncritical':0,'hardness_fallback':0,'positive_samples':0,'negative_samples':0}
    for a in range(0,len(low),8):
        b=min(len(low),a+8);selected={i-a:v for i,v in targets.items() if a<=i<b}
        local_specs={i-a:v for i,v in specs.items() if a<=i<b};local_roi=roi[a:b]
        plans={}
        with torch.no_grad():
            logits=upsample(low[a:b].detach(),cache['input_hw'])
            for rank,(rectangle,spec) in local_specs.items():
                x0,y0,x1,y1=rectangle
                value=select_pixels(logits[rank,y0:y1,x0:x1],spec,config['loss'],model.ssr_arm in ('B','D'))
                plans[rank]=value
                for k,v in value[2].items():account[k]+=v
        def compute(changed,original,sel=selected,sp=local_specs,rr=local_roi,plans=plans):
            z=upsample(changed,cache['input_hw']);native=upsample(original,cache['input_hw'])
            return mask_loss(z,native,rr,sel,sp,model.ssr_arm,config['loss'],plans,denominators)[0]
        if model.training and torch.is_grad_enabled():
            total=total+checkpoint(compute,low[a:b],base[a:b],use_reentrant=False,preserve_rng_state=False)
        else:total=total+compute(low[a:b],base[a:b])
    account['matched_instances']=len(join)
    return total,account
