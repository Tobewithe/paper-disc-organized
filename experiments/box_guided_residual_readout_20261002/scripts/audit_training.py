"""Read-only numerical supervision audit; no optimizer or weight updates."""
from __future__ import annotations
import argparse, json, math, shutil, sys, time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,r'D:\coco_wire\py')
import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as am
from ultralytics.utils import ops
from torchvision.ops import roi_align
from bgcr_models import ResidualReadout,extract_box_features
from run_bgcr import loss_sum,normalized,load,onto,sha,dump,key


def cmp(a,b,atol,rtol=0.):
    a,b=a.detach(),b.detach().to(a.device)
    return dict(max_abs=float((a-b).abs().max()) if a.numel() else 0.,
                passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),atol=atol,rtol=rtol)


def analytic(x,c):
    p=F.interpolate(x['proto'].float()[None],(640,640),mode='bilinear',align_corners=False)[0].double()
    grad=torch.zeros_like(c,dtype=torch.float64);v=0.
    yy=torch.arange(640,device=c.device)[:,None];xx=torch.arange(640,device=c.device)[None,:]
    for i,b in enumerate(x['target_boxes']):
        support=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3])
        design=p[:,support].T;target=(x['masks'][support]==int(x['owners'][i])+1).double()
        z=design@c[i].double(); area=((b[2:]-b[:2]).double()).prod()
        gain=float(x['segmentation_gain'])/len(c)
        v=v+gain*(F.softplus(z)-target*z).sum()/area
        grad[i]=gain*((z.sigmoid()-target)@design)/area
    return v,grad


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--source-run',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));source=Path(args.source_run);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    t=time.monotonic();torch.set_num_threads(4);torch.manual_seed(7331)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert ultralytics.__version__=='8.4.100'
    cache=Path(cfg['cache']);prepared=Path(cfg['prepared']);index=json.loads((cache/'INDEX.json').read_text())
    selected={s:[int(q['image_id']) for q in index[s][:4 if s=='fit' else 2]] for s in ('fit','dev','val')}
    wrapper=YOLO(cfg['weights']);model=wrapper.model.cuda().float().eval().requires_grad_(False)
    model.args=SimpleNamespace(**wrapper.ckpt['train_args']); criterion=model.init_criterion().one2one
    captured={};hooks=[b[-1].register_forward_pre_hook(lambda _,x,level=l:captured.__setitem__(level,x[0].detach())) for l,b in enumerate(model.model[-1].one2one_cv4)]
    original_raster=am.polygons2masks_overlap;order={}
    def raster(*a,**kw):
        mask,ids=original_raster(*a,**kw);order['ids']=ids.copy();return mask,ids
    am.polygons2masks_overlap=raster
    saved=load(source/'ROI_FEATURES.pt'); frozen={int(x['image_id']):x for x in load(source/'FINAL_COEFFICIENTS.pt')}
    report={'no_optimizer_updates':True,'selection':selected,'tolerances':{'fresh_native':{'atol':2e-4,'rtol':2e-4},'loss_and_coefficient_gradient':{'atol':3e-5,'rtol':3e-5},'saved_coefficients':{'atol':1e-6,'rtol':0}},'images':[],'parameter_updates':{},'gradient_flow':{},'source_run':str(source)}
    heads={}
    for arm in ('B','C'):
        ck=load(source/f'{arm}_epoch10.pt');old=load(source/f'{arm}_epoch5.pt')
        assert ck['epoch']==10
        h=ResidualReadout().cuda();h.load_state_dict(ck['state_dict']);heads[arm]=h
        report['parameter_updates'][arm]={name:{'norm_from_zero':float(v.norm()),'nonzero':int(torch.count_nonzero(v)),'epoch5_to10_norm':float((v-old['state_dict'][name]).norm())} for name,v in ck['state_dict'].items()}
    conv=json.loads((cache/'CONVERSION.json').read_text())['source_annotation_ids']
    for split,iids in selected.items():
        domain='val2017' if split=='val' else 'train2017'; paths=[]
        for iid in iids:
            # Isolate dataset's .cache writes from original assets.
            for kind,ext in [('images','jpg'),('labels','txt')]:
                src=cache/'official_data'/kind/domain/f'{iid:012d}.{ext}';dst=out/'audit_data'/kind/domain/src.name
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                if kind=='images':paths.append(str(dst))
        listing=out/f'{split}_audit.txt';listing.write_text('\n'.join(paths)+'\n')
        ds=YOLODataset(img_path=str(listing),imgsz=640,batch_size=1,augment=False,hyp=deepcopy(model.args),rect=False,cache=False,stride=32,data={'names':wrapper.names,'nc':80,'channels':3},task='segment')
        dsids={int(Path(p).stem):j for j,p in enumerate(ds.im_files)}
        for iid in iids:
            raw=load(cache/'images'/f'{iid:012d}.pt');x=onto(load(prepared/f'{iid:012d}.pt'));j=dsids[iid]
            # Independently replay conversion/dedup/rasterizer annotation order.
            labelpath=out/'audit_data'/'labels'/domain/f'{iid:012d}.txt'
            serialized=[np.asarray(s.split(),dtype=np.float32) for s in labelpath.read_text().splitlines() if s.strip()]
            aidrows=conv[str(iid)]; before=[]
            for cls,poly in zip(ds.labels[j]['cls'].flatten(),ds.labels[j]['segments']):
                v=np.r_[cls,poly.flatten()].astype(np.float32)
                ids=[aid for a,aid in zip(serialized,aidrows) if np.array_equal(a,v)]
                assert len(ids)==1;before.append(ids[0])
            order.clear();sample=ds[j]; annids=np.asarray(before)[order['ids']].tolist()
            batch=onto(YOLODataset.collate_fn([sample])); inp=batch['img'].float()/255
            with torch.no_grad():
                _,preds=model(inp);pred=preds['one2one']
                assignment,_,_=criterion.get_assigned_targets_and_loss(pred,batch)
            ids=torch.where(assignment[0][0])[0]; cacheids=torch.tensor([r['raw_id'] for r in raw['rows']],device='cuda')
            rec=dict(split=split,image_id=iid,n=len(cacheids),annotation_order_equal=annids==raw['all_annotation_ids'],fresh_TAL_raw_ids_equal=bool(torch.equal(ids,cacheids)))
            rec['labels']=cmp(batch['masks'][0],x['masks'],0)
            rec['owners']=cmp(assignment[1][0][cacheids],x['owners'],0)
            rec['target_boxes']=cmp(assignment[2][0][cacheids],x['target_boxes'],1e-5)
            hfresh=torch.cat([captured[l][0].flatten(1).T for l in range(3)])
            rec['full_H']=cmp(hfresh,raw['h'].cuda(),2e-4,2e-4)
            rec['proto']=cmp(pred['proto'][0],raw['proto'].cuda(),2e-4,2e-4)
            rec['coefficients']=cmp(pred['mask_coefficient'][0].T,raw['coeff'].cuda(),2e-4,2e-4)
            rec['boxes']=cmp(model.model[-1]._get_decode_boxes(pred)[0].T,raw['boxes'].cuda(),2e-4,2e-4)
            # ROI independently on fresh native hooks vs previously saved descriptors.
            with torch.no_grad():
                f=extract_box_features({'h':hfresh,'levels':raw['levels'].cuda()},x)
            rec['fresh_roi_residual']=cmp(f['residual'],saved['features'][iid]['residual'].cuda(),3e-4,3e-4)
            rec['loss_checks']=[]
            for mode in ('c0','random_delta'):
                c=(x['c0']+(torch.randn_like(x['c0'])*.03 if mode=='random_delta' else 0)).detach().requires_grad_(True)
                own=loss_sum(x,c)/len(c);g=torch.autograd.grad(own,c)[0]
                modified=pred['mask_coefficient'].detach().clone();modified[0,:,cacheids]=c.T
                # Exact frozen P: fresh labels/official TAL, identical BCE input.
                p={**pred,'proto':raw['proto'].cuda()[None],'mask_coefficient':modified}
                official=criterion.loss(p,batch)[0][1]
                go=torch.autograd.grad(official,c)[0]
                av,ag=analytic(x,c.detach())
                gen=torch.Generator(device='cuda').manual_seed(iid+3)
                direction=torch.randn(c.shape,generator=gen,device='cuda',dtype=torch.float64);direction/=direction.norm()
                eps=1e-4
                plus,_=analytic(x,c.detach().double()+eps*direction);minus,_=analytic(x,c.detach().double()-eps*direction)
                numeric=(plus-minus)/(2*eps);directional=(ag*direction).sum()
                rec['loss_checks'].append(dict(mode=mode,custom=float(own),official=float(official),analytical=float(av),official_value=cmp(own,official,3e-5,3e-5),official_gradient=cmp(g,go,3e-5,3e-5),analytic_gradient=cmp(g.double(),ag,3e-5,3e-5),finite_difference=cmp(numeric,directional,1e-7,1e-5)))
            rec['saved_coefficients']={}
            assert frozen[iid]['keys']==[key(r) for r in x['rows']]
            for arm,inputname in [('B','h'),('C','residual')]:
                vec=x['h'] if arm=='B' else saved['features'][iid]['residual'].cuda()
                head=heads[arm];head.zero_grad(set_to_none=True)
                nf=normalized(vec,x['levels'],saved['norms'][arm]);c=x['c0']+head(nf,x['levels'])
                rec['saved_coefficients'][arm]=cmp(c,frozen[iid]['coeffs'][arm].cuda(),1e-6)
                v=loss_sum(x,c)/len(c);v.backward()
                for name,p in head.named_parameters():
                    k=arm+':'+name
                    report['gradient_flow'][k]=max(report['gradient_flow'].get(k,0.),float(p.grad.norm()) if p.grad is not None else 0.)
            rec['passed']=rec['annotation_order_equal'] and rec['fresh_TAL_raw_ids_equal'] and all(rec[k]['passed'] for k in ('labels','owners','target_boxes','full_H','proto','coefficients','boxes','fresh_roi_residual')) and all(d[k]['passed'] for d in rec['loss_checks'] for k in ('official_value','official_gradient','analytic_gradient','finite_difference')) and all(d['passed'] for d in rec['saved_coefficients'].values())
            report['images'].append(rec);dump(out/'AUDIT.json',report);print(json.dumps({'image_id':iid,'passed':rec['passed']}),flush=True)
    # Coordinate ramp: check spatial_scale/half-pixel conventions independently.
    ramp=[]
    for side in (80,40,20):
        yy,xx=torch.meshgrid(torch.arange(side,device='cuda'),torch.arange(side,device='cuda'),indexing='ij')
        hm=torch.stack((xx,yy)).float()[None];box=torch.tensor([[0.,128.,160.,384.,448.]],device='cuda')
        got=roi_align(hm,box,output_size=3,spatial_scale=side/640,sampling_ratio=2,aligned=True).mean((-2,-1))[0]
        exp=torch.tensor([(128+384)/2*side/640-.5,(160+448)/2*side/640-.5],device='cuda')
        ramp.append(dict(side=side,**cmp(got,exp,1e-6)))
    report['coordinate_ramps']=ramp
    report['all_head_parameters_changed']=all(d['nonzero']>0 for m in report['parameter_updates'].values() for d in m.values())
    report['all_head_parameters_receive_gradient']=all(g>0 for g in report['gradient_flow'].values()) and len(report['gradient_flow'])==12
    report['weight_sha256_unchanged']=sha(cfg['weights'])==cfg['weight_sha256']
    report['passed']=all(r['passed'] for r in report['images']) and all(r['passed'] for r in ramp) and report['all_head_parameters_changed'] and report['all_head_parameters_receive_gradient'] and report['weight_sha256_unchanged']
    report['elapsed_s']=time.monotonic()-t
    dump(out/'AUDIT.json',report);dump(out/'COMPLETE.json',dict(completed=True,scientific_check_passed=report['passed'],optimizer_steps=0,elapsed_s=report['elapsed_s']))
    for h in hooks:h.remove()
    am.polygons2masks_overlap=original_raster
    print(json.dumps({'completed':True,'passed':report['passed'],'seconds':report['elapsed_s']}),flush=True)


if __name__=='__main__':main()
