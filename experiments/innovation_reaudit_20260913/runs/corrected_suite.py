"""Frozen readout pilots; true same-position pairing and active basis regularization."""
from reexperiment import *
from torch import nn

class CorrectedHead(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.base=AdaCalibHead() if mode in ['ada','full','full_noorth'] else BifurcatedSpatialHead()
        if mode in ['adapter','orth','full','full_noorth']:self.basis=nn.Parameter(torch.eye(32))
        else:self.register_parameter('basis',None)
        if mode=='scalar':self.bias=nn.Parameter(torch.zeros(()))
        else:self.register_parameter('bias',None)
    def transform(self,p):return p if self.basis is None else p@self.basis.T
    def forward_sampled(self,x,c,p):
        z,info=self.base.forward_sampled(x,c,self.transform(p))
        return (z if self.bias is None else z+self.bias),info
    def forward_inference(self,x,c,p):
        pp=p if self.basis is None else torch.einsum('ab,bhw->ahw',self.basis,p)
        z,gate=self.base.forward_inference(x,c,pp)
        return (z if self.bias is None else z+self.bias),gate

def orth_loss(p):
    # Per-target sampled-channel correlations; not mixed across unrelated images.
    v=p-p.mean(1,keepdim=True)
    v=F.normalize(v,dim=1,eps=1e-6)
    corr=v.transpose(1,2)@v
    mask=~torch.eye(32,device=p.device,dtype=torch.bool)
    return corr[:,mask].square().mean()

def pair_loss(model,data,pairs,ix):
    a,b=pairs['a'][ix],pairs['b'][ix];p=pairs['p'][ix]
    za,_=model.forward_sampled(data['xn'][a],data['c'][a],p)
    zb,_=model.forward_sampled(data['xn'][b],data['c'][b],p)
    exclusive_a=pairs['ya'][ix]&~pairs['yb'][ix]
    exclusive_b=pairs['yb'][ix]&~pairs['ya'][ix]
    weights=(exclusive_a|exclusive_b).float()
    direction=exclusive_a.float()-exclusive_b.float()
    penalties=F.relu(1.5-direction*(za-zb))*weights
    return (penalties.sum(1)/weights.sum(1).clamp_min(1)).mean()

def load_data(sel,normalizer):
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(CACHE/'conversion_input/instances_probe.json'))
    rec=[];pairrec=[];offset=0;hashes={};audit_rows=[]
    for ni,iid in enumerate(sel['fit']):
        path=CACHE/'images'/f'{iid}.npz';q=read(path)
        idx=q['prediction_indices'];aids=q['annotation_ids'];labpath=LABELS/'labels'/f'{iid}.npz'
        if len(idx)==0:continue
        lab=read(labpath);assert np.array_equal(aids,lab['annotation_ids'])
        hashes[str(path)]=sha(path);hashes[str(labpath)]=sha(labpath)
        h,boxes=map(tensor,[q['h'],q['boxes']]);ishape=tuple(map(int,q['input_shape']))
        x=instance_features(h,torch.as_tensor(q['level'],device=DEVICE).long(),boxes,ishape)[idx]
        rec.append(dict(xn=norm(x,normalizer).cpu(),c=torch.tensor(q['coeff'][idx]).float(),
            p=torch.tensor(q['sample_p'][:,:512]).float(),factor=torch.tensor(q['loss_factor']).float(),
            y=torch.tensor(lab['raw_coco']).float()))
        raster={}
        def rasterize(aid):
            if aid not in raster:
                own=tensor(gt.annToMask(gt.anns[int(aid)]));oh,ow=own.shape
                gain=min(ishape[0]/oh,ishape[1]/ow);rh,rw=round(oh*gain),round(ow*gain)
                top,left=round((ishape[0]-rh)/2-.1),round((ishape[1]-rw)/2-.1)
                r=torch.zeros(ishape,device=DEVICE,dtype=torch.bool)
                r[top:top+rh,left:left+rw]=F.interpolate(own[None,None],(rh,rw),mode='nearest-exact')[0,0].bool()
                raster[aid]=r.flatten()
            return raster[aid]
        for a in range(len(aids)):
            aa=gt.anns[int(aids[a])]
            for b in range(a+1,len(aids)):
                ab=gt.anns[int(aids[b])]
                if aa['category_id']!=ab['category_id']:continue
                xa,ya,wa,ha=aa['bbox'];xb,yb,wb,hb=ab['bbox']
                inter=max(0,min(xa+wa,xb+wb)-max(xa,xb))*max(0,min(ya+ha,yb+hb)-max(ya,yb))
                if inter/max(wa*ha+wb*hb-inter,1e-9)<=.05:continue
                # 32 sampled coordinates from each instance, evaluated by BOTH heads.
                pos=np.concatenate([q['sample_positions'][a,:32],q['sample_positions'][b,:32]])
                pp=np.concatenate([q['sample_p'][a,:32],q['sample_p'][b,:32]])
                ta,tb=rasterize(aids[a])[pos],rasterize(aids[b])[pos]
                if not (ta^tb).any():continue
                pairrec.append(dict(a=offset+a,b=offset+b,p=pp,ya=ta.cpu().numpy(),yb=tb.cpu().numpy()))
                if len(audit_rows)<10:audit_rows.append(dict(image_id=int(iid),a=int(aids[a]),b=int(aids[b]),exclusive=int((ta^tb).sum()),coordinates=pos.tolist()))
        offset+=len(idx)
        if (ni+1)%100==0:state('prepare_corrected_pairs',images=ni+1,targets=offset,pairs=len(pairrec))
    data={k:torch.cat([r[k] for r in rec]).to(DEVICE) for k in rec[0]}
    pairs={k:torch.as_tensor(np.array([r[k] for r in pairrec]),device=DEVICE) for k in pairrec[0]}
    pairs['p']=pairs['p'].float();assert len(data['c'])==7811
    write(OUT/'corrected_data_receipt.json',dict(targets=offset,pairs=len(pairrec),coordinate_witness=audit_rows,hashes=hashes,
        selection='GT same class, boxIoU>.05, at least one exclusive sampled pixel; fit only',
        shared_coordinates=True,orth_scope='Per-target sampled transformed basis correlation; original YOLO frozen'))
    return data,pairs

def gradient_witness(data,pairs):
    torch.manual_seed(0);m=CorrectedHead('orth').to(DEVICE)
    loss=orth_loss(m.transform(data['p'][:8]));g=torch.autograd.grad(loss,m.basis)[0]
    assert g.abs().max()>0
    torch.manual_seed(0);m2=CorrectedHead('contrast').to(DEVICE)
    cl=pair_loss(m2,data,pairs,torch.arange(min(8,len(pairs['a'])),device=DEVICE))
    grads=torch.autograd.grad(cl,m2.parameters(),allow_unused=True)
    gn=sum(float(g.square().sum()) for g in grads if g is not None)**.5;assert gn>0
    write(OUT/'corrected_gradient_witness.json',dict(orth_basis_grad_norm=float(g.norm()),pair_head_grad_norm=gn,
        orth_loss=float(loss),pair_loss=float(cl),tests='Nonzero gradient to trained parameters; explicit same-coordinate pair construction.'))

def train(data,pairs):
    models={};modes=['bsr','ada','scalar','contrast','adapter','orth','full_noorth','full']
    dest=OUT/'corrected_training';dest.mkdir(exist_ok=True)
    for seed in range(3):
        for mode in modes:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=CorrectedHead(mode).to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            rng=np.random.default_rng(seed);prng=np.random.default_rng(1000+seed);hist=[]
            path=dest/f'{mode}_s{seed}';path.mkdir(exist_ok=True)
            for epoch in range(15):
                sums=np.zeros(3);n=0;order=rng.permutation(len(data['c']))
                for first in range(0,len(order),32):
                    ix=torch.as_tensor(order[first:first+32],device=DEVICE)
                    z,info=model.forward_sampled(data['xn'][ix],data['c'][ix],data['p'][ix])
                    loss,ld=bsr_loss(z,data['y'][ix],data['factor'][ix],info,lambda_spatial=.001)
                    cl=ol=loss.new_zeros(())
                    if mode in ['contrast','full_noorth','full']:
                        pi=torch.as_tensor(prng.integers(len(pairs['a']),size=16),device=DEVICE)
                        cl=pair_loss(model,data,pairs,pi);loss=loss+.05*cl
                    if mode in ['orth','full']:
                        ol=orth_loss(model.transform(data['p'][ix]));loss=loss+.01*ol
                    assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                    sums+=np.array([ld['task_loss'],float(cl.detach()),float(ol.detach())])*len(ix);n+=len(ix)
                hist.append(dict(epoch=epoch+1,task=sums[0]/n,contrast=sums[1]/n,orth=sums[2]/n,updates=(epoch+1)*245))
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,mode=mode,seed=seed,
                    rng=rng.bit_generator.state,pair_rng=prng.bit_generator.state),path/f'epoch{epoch+1:03d}.pt')
                write(path/'history.json',hist)
                if (epoch+1)%5==0:state('train_corrected',mode=mode,seed=seed,**hist[-1])
            models[f'new_{mode}_s{seed}']=model.eval()
    return models

def s080(models,normalizer):
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT.parent.parent/'datasets/coco/annotations/instances_val2017.json'))
    cases=pd.read_csv(ROOT/'diagnostics/fixed_prototype_split_20260913_v5/instances.csv');rows=[]
    for _,r in cases.iterrows():
        iid,aid=int(r.image_id),int(r.annotation_id)
        q=read(ROOT/'diagnostics/joint_failure_decoder_20260913_v2/tensor_cache'/f'{iid}.npz')
        ix=np.flatnonzero(q['annotation_ids']==aid);assert len(ix)==1;k=int(ix[0])
        p,c,box,h=map(tensor,[q['proto'],q['coeff'][k:k+1],q['boxes'][k:k+1],q['h'][k:k+1]])
        source=int(q['source_indices'][k]);level=torch.tensor([0 if source<6400 else 1 if source<8000 else 2],device=DEVICE)
        own=gt.annToMask(gt.anns[aid]).astype(bool);shape=own.shape
        x=norm(instance_features(h,level,box,(640,640)),normalizer)
        with torch.inference_mode():
            for key,m in models.items():
                if m is None:z=torch.einsum('bc,chw->bhw',c,p)
                elif key.startswith('s032'):z=torch.einsum('bc,chw->bhw',c+m(x),p)
                else:z=m.forward_inference(x,c,p)[0]
                center=key.startswith(('new_full_s','new_full_noorth_s'))
                if center:z=apply_centerness_prior(z,box/4,(160,160),margin_scale=.5)
                pred=decode(z,box,(640,640),shape)[0].cpu().numpy()
                iou=np.count_nonzero(pred&own)/max(1,np.count_nonzero(pred|own))
                rows.append(dict(arm=key,image_id=iid,annotation_id=aid,residual=r.residual,density=r.density,iou=iou,hit75=int(iou>=.75)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'s080_corrected_instances.csv',index=False)
    df.groupby(['arm','residual']).agg(n=('iou','size'),mean_iou=('iou','mean'),reaches75=('hit75','sum')).to_csv(OUT/'s080_corrected_summary.csv')

def main_corrected():
    # Serial queue: do not compete with the existing-weight evaluation.
    while not (OUT/'existing_aligned/COMPLETE.json').exists():
        time.sleep(10)
    sel=json.loads((CACHE/'selection.json').read_text())
    normalizer=torch.load(LABELS/'normalizer.pt',map_location=DEVICE,weights_only=False)
    write(OUT/'CORRECTED_PROTOCOL.json',dict(seeds=[0,1,2],epochs=15,updates_per_arm=3675,batch=32,lr=.0001,
        pair_batch=16,pair_margin=1.5,pair_weight=.05,orth_weight=.01,center_margin=.5,
        modes=['bsr','ada','scalar','contrast','adapter','orth','full_noorth','full'],
        fixed_data=True,decoder='same as reexperiment.py',no_transfer_selection=True,
        frozen='Original YOLO, boxes, scores, P; train readout and optional32x32 basis transform only.',
        caveat='Basis regularization is an explicit corrected alternative, not original YOLO prototype learning. All variants exploratory.'))
    data,pairs=load_data(sel,normalizer);gradient_witness(data,pairs)
    models=train(data,pairs);del data,pairs;torch.cuda.empty_cache()
    old=load_old();models['original']=None
    for s in range(3):models[f's032_s{s}']=old[f's032_s{s}']
    del old
    arms={key:(key,key.startswith(('new_full_s','new_full_noorth_s')),False) for key in models}
    for s in range(3):
        for m in ['bsr','ada','scalar','orth']:
            arms[f'new_{m}_center_s{s}']=(f'new_{m}_s{s}',True,False)
        for m in ['full','full_noorth']:
            arms[f'new_{m}_nocenter_s{s}']=(f'new_{m}_s{s}',False,False)
    reevaluate(models,arms,OUT/'corrected_aligned',sel['transfer'],normalizer)
    s080(models,normalizer)
    state('corrected_complete');write(OUT/'CORRECTED_COMPLETE.json',dict(status='COMPLETE',trained_heads=24,epochs=15,seeds=3))

if __name__=='__main__':main_corrected()
