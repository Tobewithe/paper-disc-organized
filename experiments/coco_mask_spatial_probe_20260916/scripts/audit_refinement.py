"""Fixed-identity diagnosis: sampling, objective mismatch and correction strength."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pycocotools.coco import COCO
from learn_refinement import Refiner, features, roi_correct, save


def objective(z,y):
    p=z.sigmoid()
    return F.binary_cross_entropy_with_logits(z,y)+.5*(1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1))


def main():
    p=argparse.ArgumentParser()
    for k in ('data','annotations','models','bank','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--repair-models',action='store_true')
    p.add_argument('--skip-bank',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    coco=COCO(str(a.annotations));manifest=json.loads((a.data/'manifest.json').read_text())
    nets={}
    model_modes={'scalar_plain':'scalar','coeff_local4_plain':'coeff_local4','coeff_local4_safe':'coeff_local4'} if a.repair_models else {m:m for m in ('scalar','coeff_bias','local4','coeff_local4')}
    for label,mode in model_modes.items():
        net=Refiner(mode).cuda().eval();net.load_state_dict(torch.load(a.models/f'{label}_epoch8.pt',weights_only=False)['state_dict']);nets[label]=net
    start=time.monotonic();rows=[]
    with torch.no_grad():
        for row in manifest:
            with np.load(a.data/f"{row['image_id']:012d}.npz") as z:
                proto=torch.as_tensor(z['proto'],device='cuda');coef=torch.as_tensor(z['coefficients'],device='cuda')
                shape=tuple(z['input_shape']);orig=tuple(z['original_shape'])
                box=torch.as_tensor(z['boxes_input'][[row['raw_id']]],device='cuda')
            base=(coef.T@proto.flatten(1)).reshape(-1,*proto.shape[-2:])[row['raw_id']]
            base_up=F.interpolate(base[None,None],shape,mode='bilinear',align_corners=False)[0,0]
            gt=torch.tensor(coco.annToMask(coco.anns[row['annotation_id']]),device='cuda').float()
            ih,iw=shape;oh,ow=orig;gain=min(ih/oh,iw/ow);nh,nw=round(oh*gain),round(ow*gain)
            top,left=round((ih-nh)/2-.1),round((iw-nw)/2-.1)
            yinput=F.pad(F.interpolate(gt[None,None],(nh,nw),mode='bilinear',align_corners=False),(left,iw-nw-left,top,ih-nh-top))
            yy,xx=torch.meshgrid(torch.arange(ih,device='cuda'),torch.arange(iw,device='cuda'),indexing='ij')
            b=box[0];support=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3])
            scale=proto.square().mean((1,2)).sqrt().clamp(min=.1)
            x,roi,grid=features(proto,coef[:,[row['raw_id']]].T,box,shape)
            target=F.grid_sample(yinput,grid,align_corners=False)[:,0]
            sampled=F.grid_sample(base_up[None,None],grid,align_corners=False)[:,0]
            def metrics(logit):
                binary=((logit>0)&support).float()
                m=F.interpolate(binary[None,None,top:top+nh,left:left+nw],orig,mode='bilinear',align_corners=False)[0,0].byte().bool();g=gt.bool()
                tp=int((m&g).sum());fp=int((m&~g).sum());fn=int((~m&g).sum())
                return dict(iou=tp/max(tp+fp+fn,1),coverage=tp/max(tp+fn,1),purity=tp/max(tp+fp,1),fp=fp,fn=fn)
            baseline=metrics(base_up);assert abs(baseline['iou']-row['baseline_iou'])<2e-6
            out={**row,'baseline':baseline,'roi_sampling_mae':float((roi-sampled).abs().mean()),
                 'roi_sampling_sign_disagree':float(((roi>0)!=(sampled>0)).float().mean()),'arms':{}}
            for label,net in nets.items():
                mode=model_modes[label]
                c=net(x);d=torch.zeros_like(base_up)
                if mode in ('coeff_bias','coeff_local4'):
                    low=(2*c[:,:32]/scale[None])@proto.flatten(1)
                    d+=F.interpolate(low.reshape(1,1,*proto.shape[-2:]),shape,mode='bilinear',align_corners=False)[0,0]
                if mode in ('scalar','coeff_bias'):d+=4*c[0,-1]
                if mode in ('local4','coeff_local4'):
                    g=torch.stack((2*(xx-b[0])/(b[2]-b[0]).clamp(min=1)-1,2*(yy-b[1])/(b[3]-b[1]).clamp(min=1)-1),-1)[None]
                    d+=F.grid_sample(4*c[:,-16:].reshape(1,1,4,4),g,padding_mode='border',align_corners=True)[0,0]
                variants={}
                for alpha in (.25,.5,1.):
                    trainlogit=roi_correct(roi,x,c*alpha,mode)
                    v=metrics(base_up+alpha*d)
                    v.update(roi_loss_delta=float(objective(trainlogit,target)-objective(roi,target)),
                             full_loss_delta=float(objective((base_up+alpha*d)[support],yinput[0,0][support])-objective(base_up[support],yinput[0,0][support])))
                    variants[str(alpha)]=v
                out['arms'][label]={'variants':variants,'saturation_fraction':float((c.abs()>.95).float().mean()),
                    'mean_abs_logit_correction':float(d[support].abs().mean())}
            rows.append(out)
            with (a.out/'instances.jsonl').open('a') as f:f.write(json.dumps(out)+'\n')
    summary={}
    for group in ('failure','success'):
        rs=[r for r in rows if r['outcome']==group];summary[group]={}
        for mode in nets:
            summary[group][mode]={}
            for alpha in ('0.25','0.5','1.0'):
                vs=[r['arms'][mode]['variants'][alpha] for r in rs]
                summary[group][mode][alpha]=dict(n=len(rs),iou_delta=100*np.mean([v['iou']-r['baseline']['iou'] for v,r in zip(vs,rs)]),
                    improved=sum(v['iou']>r['baseline']['iou'] for v,r in zip(vs,rs)),
                    worsened=sum(v['iou']<r['baseline']['iou'] for v,r in zip(vs,rs)),
                    repaired75=sum(r['baseline']['iou']<.75<=v['iou'] for v,r in zip(vs,rs)),
                    damaged75=sum(v['iou']<.75<=r['baseline']['iou'] for v,r in zip(vs,rs)),
                    loss_better_iou_worse=sum(v['roi_loss_delta']<0 and v['iou']<r['baseline']['iou'] for v,r in zip(vs,rs)),
                    mean_roi_loss_delta=float(np.mean([v['roi_loss_delta'] for v in vs])),
                    mean_full_loss_delta=float(np.mean([v['full_loss_delta'] for v in vs])))
    summary['sampling']=dict(mean_logit_mae=float(np.mean([r['roi_sampling_mae'] for r in rows])),
        mean_sign_disagree=float(np.mean([r['roi_sampling_sign_disagree'] for r in rows])))
    if a.skip_bank:
        save(a.out/'SUMMARY.json',summary);save(a.out/'COMPLETE.json',dict(instances=len(rows),elapsed_s=time.monotonic()-start));print(json.dumps(summary),flush=True);return
    bank=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True)
    for label,net in nets.items():
        mode=model_modes[label]
        loss_before=[];loss_after=[];iou_before=[];iou_after=[]
        with torch.no_grad():
            for i in range(0,len(bank['x']),128):
                x=bank['x'][i:i+128].cuda().float();b=bank['base'][i:i+128].cuda().float();y=bank['target'][i:i+128].cuda().float();z=roi_correct(b,x,net(x),mode)
                for j in range(len(x)):
                    loss_before.append(float(objective(b[j],y[j])));loss_after.append(float(objective(z[j],y[j])))
                g=y>.5
                for zz,dest in ((b,iou_before),(z,iou_after)):
                    m=zz>0;dest.extend(((m&g).sum((1,2))/(m|g).sum((1,2)).clamp(min=1)).cpu().tolist())
        summary.setdefault('train_roi',{})[label]=dict(loss_before=float(np.mean(loss_before)),loss_after=float(np.mean(loss_after)),
            iou_before=float(np.mean(iou_before)),iou_after=float(np.mean(iou_after)),
            loss_better_iou_worse=sum(l1<l0 and i1<i0 for l0,l1,i0,i1 in zip(loss_before,loss_after,iou_before,iou_after)),
            samples=len(loss_before),note='Original bank includes143ambiguous identities; descriptive audit only, corrected training excluded them.')
    save(a.out/'SUMMARY.json',summary);save(a.out/'COMPLETE.json',dict(instances=len(rows),elapsed_s=time.monotonic()-start));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
