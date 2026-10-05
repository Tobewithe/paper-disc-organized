"""Implementation audit on 12 development instances, independent of model selection."""
import argparse,json
from common import *
from evaluate import decode_logits
from ultralytics.utils import ops

def main(a):
    setup()
    errors=[];residuals=[]
    for r in load(DEV_TARGETS)[:12]:
        image=load(OLD_BANK/"images"/f"{r['image_id']:012d}.pt")
        raw=r["raw_id"];p=image["proto"].cuda().float()
        c=image["coeff"][raw].cuda().float();box=image["boxes"][raw].cuda()
        z=(c@p.flatten(1)).reshape(160,160)
        ours=decode_logits(z[None].clone(),box,image)
        official=ops.process_mask(p,c[None],box[None],(640,640),upsample=True)
        official=ops.scale_masks(official.float()[None],image["original_shape"],ratio_pad=image["ratio_pad"])[0]>.5
        errors.append(float((ours!=official).float().mean()))
        support=ops.crop_mask(torch.ones(1,160,160,device="cuda"),box[None]/4)[0].bool()
        A=p[:,support].T.double();G=A.T@A
        ev,Q=torch.linalg.eigh(G);keep=ev>ev[-1]*1e-10
        inverse=(Q[:,keep]/ev[keep][None])@Q[:,keep].T
        d=r["delta"].cuda().double();truth=A@d;hat=A@(inverse@(A.T@truth))
        residuals.append(float((hat-truth).norm()/truth.norm().clamp_min(1e-12)))
    result=dict(decode_max_pixel_error=max(errors),oracle_full_projection_relative_error_max=max(residuals),instances=len(errors))
    write(a.out/"AUDIT.json",result)
    assert max(errors)==0 and max(residuals)<1e-3,result
    write(a.out/"COMPLETE.json",result);print(json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args())

