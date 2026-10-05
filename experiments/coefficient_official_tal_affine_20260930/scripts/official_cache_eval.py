"""Frozen-identity utility replay from complete official cache, no new fitting."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops

def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=True),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def pixel_auc_fpr(z,gt,support):
    scores=z[support].detach().cpu().numpy(); labels=gt[support].detach().cpu().numpy().astype(bool)
    n1=int(labels.sum()); n0=int(len(labels)-n1)
    fpr=float(((scores>0)&~labels).sum()/n0) if n0 else float('nan')
    if not n1 or not n0: return float('nan'),fpr
    order=np.argsort(scores,kind='stable'); sorted_scores=scores[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_scores))+1]; ends=np.r_[starts[1:],len(scores)]
    ranks=np.repeat((starts+ends+1)*.5,ends-starts)
    auc=float((ranks[labels[order]].sum()-n1*(n1+1)*.5)/(n1*n0))
    return auc,fpr

def aggregate(rows,out):
    tables={}; image_rows=[]
    for split in ('fit','dev','val'):
        groups={'all':[r for r in rows if r['split']==split]}
        if split=='val':
            groups.update({f'P{l+3}':[r for r in rows if r['split']==split and r['pyramid_level']==l] for l in range(3)})
            groups.update({f'original_{name}':[r for r in rows if r['split']==split and bool(r['mask75_A'])==flag] for name,flag in [('success',True),('failure',False)]})
        for group,rr in groups.items():
            if not rr: continue
            ids=sorted({r['image_id'] for r in rr}); byimg=[[r for r in rr if r['image_id']==i] for i in ids]
            rng=np.random.default_rng(20260930); draw=rng.integers(0,len(ids),(5000,len(ids)))
            table={'n_candidates':len(rr),'n_images':len(ids),'candidate':{},'image_macro':{}}
            for metric in ('iou','coverage','auc','fpr','bce','regularizer','objective','mask75'):
                arms=('A','B','C') if f'{metric}_C' in rr[0] else ('A','B')
                sums={}; counts={}; per={}
                for arm in arms:
                    vals=[[r[f'{metric}_{arm}'] for r in x] for x in byimg]
                    sums[arm]=np.array([np.nansum(x) for x in vals]); counts[arm]=np.array([np.isfinite(x).sum() for x in vals])
                    per[arm]=np.divide(sums[arm],counts[arm],out=np.full(len(ids),np.nan),where=counts[arm]>0)
                candidate={arm:float(sums[arm].sum()/max(counts[arm].sum(),1)) for arm in arms}
                macro={arm:float(np.nanmean(per[arm])) for arm in arms}
                for arm in arms[1:]:
                    # Pairwise valid sets are identical in this protocol for AUC/FPR.
                    good=np.isfinite(per[arm])&np.isfinite(per['A'])
                    point=float(np.nanmean(per[arm]-per['A']))
                    boot=np.nanmean((per[arm]-per['A'])[draw],axis=1)
                    macro[arm+'_minus_A']={'delta':point,'ci95':np.quantile(boot,[.025,.975]).tolist(),'valid_images':int(good.sum())}
                    ratio_b=sums[arm][draw].sum(1)/np.maximum(counts[arm][draw].sum(1),1)
                    ratio_a=sums['A'][draw].sum(1)/np.maximum(counts['A'][draw].sum(1),1)
                    candidate[arm+'_minus_A']={'delta':candidate[arm]-candidate['A'],'ci95':np.quantile(ratio_b-ratio_a,[.025,.975]).tolist()}
                table['candidate'][metric]=candidate; table['image_macro'][metric]=macro
            table['mask75_counts']={'A':sum(r['mask75_A'] for r in rr),'B':sum(r['mask75_B'] for r in rr),'repair':sum(not r['mask75_A'] and r['mask75_B'] for r in rr),'damage':sum(r['mask75_A'] and not r['mask75_B'] for r in rr)}
            table['auc_undefined']=sum(not np.isfinite(r['auc_A']) for r in rr)
            tables[split+':'+group]=table
            if group=='all':
                for i,x in zip(ids,byimg):
                    one={'split':split,'image_id':i,'n_candidates':len(x)}
                    for key in x[0]:
                        if key.endswith(('_A','_B','_C')) and isinstance(x[0][key],(int,float)):
                            one[key]=float(np.nanmean([r[key] for r in x]))
                    image_rows.append(one)
    dump(out/'SUMMARY.json',{'tables':tables,'bootstrap':5000,'seed':20260930,'main':'val:all image_macro iou B_minus_A','statistical_scope':'official same-domain localization diagnostic, reused research images; not blind test; no AP'})
    with (out/'PER_IMAGE.jsonl').open('w') as f:
        for r in image_rows: f.write(json.dumps(r)+'\n')

def main():
    ap=argparse.ArgumentParser()
    for k in ('cache','shared','oracle','weights','data','out'): ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    w=YOLO(str(a.weights)); gain=float(w.ckpt['train_args']['box']); model=w.model.cpu().float().eval(); head=model.model[-1]
    s=torch.load(a.shared,map_location='cpu',weights_only=False)
    A=s['A']; stats=s['stats']; native=[]; merges=[]
    for l,branch in enumerate(head.one2one_cv4):
        conv=branch[-1]; W0=conv.weight.detach().clone(); b0=conv.bias.detach().clone()
        in_ch,out_ch=conv.in_channels,conv.out_channels
        assert tuple(A[l].shape)==(in_ch+1,out_ch)
        mu,sd=stats[l]; U,v=A[l][:-1],A[l][-1]
        dw=(U/sd[:,None]).T; db=v-(mu/sd)@U
        W1=W0+dw.float()[:,:,None,None]; b1=b0+db.float()
        native.append((W0.cuda(),b0.cuda(),W1.cuda(),b1.cuda()))
        conv.weight.data.copy_(W1); conv.bias.data.copy_(b1)
        merges.append({'level':l,'shape':list(W0.shape),'delta_W_max':float(dw.abs().max()),'delta_b_max':float(db.abs().max()),'formula':'U=A[:-1],v=A[-1]; delta_W=(U/std).T; delta_b=v-(mean/std)@U'})
    checkpoint={**w.ckpt,'model':model}; torch.save(checkpoint,a.out/'YOLO26m_seg_OFFICIAL_SHARED.pt')
    del w,model,head
    index=json.loads((a.cache/'INDEX.json').read_text()); results=[]; start=time.monotonic()
    checks={'coefficient_atol':1e-4,'coefficient_rtol':1e-5,'baseline_iou_atol':1e-6,'c0_head_maxerr':0.,'merge_coeff_maxerr':0.,'merge_logit_maxerr':0.,'baseline_cached_iou_maxerr':0.,'direct_merge_mask_pixel_differences':0,'direct_merge_changed_candidates':0,'source':'complete official cache; no geometric re-forward/preprocessing','unchanged':['h','prototype','boxes','class_logits','assignment'],'layers':merges}
    for split in ('fit','dev','val'):
        coco=COCO(str(a.data/'annotations'/('instances_val2017.json' if split=='val' else 'instances_train2017.json')))
        oracle=torch.load(a.oracle/f'ORACLE_{split}.pt',map_location='cpu',weights_only=False)
        oracle_map={(r['image_id'],r['annotation_id'],r['raw_id']):d for r,d in zip(oracle['identities'],oracle['delta'])}
        for pos,it in enumerate(index[split],1):
            iid=int(it['image_id']); x=torch.load(a.cache/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
            proto=x['proto'].cuda(); up=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
            ht=x['h'].cuda(); c0=x['coeff'].cuda(); boxes=x['boxes'].cuda(); label=x['masks'].cuda()
            c_native0=torch.empty_like(c0); c_native1=torch.empty_like(c0)
            for lev,(w0,b0,w1,b1) in enumerate(native):
                loc=torch.where(x['levels']==lev)[0].cuda(); hmap=ht[loc].T[None,:,None,:]
                c_native0[loc]=F.conv2d(hmap,w0,b0)[0,:,0,:].T
                c_native1[loc]=F.conv2d(hmap,w1,b1)[0,:,0,:].T
            for k,r in enumerate(x['rows']):
                rid=int(r['raw_id']); lev=int(r['level']); hh=ht[rid].double(); mu,sd=(q.cuda() for q in stats[lev])
                delta=torch.cat(((hh-mu)/sd,torch.ones(1,device='cuda',dtype=torch.float64)))@A[lev].cuda()
                cd=c0[rid].double()+delta; cb=c_native1[rid]
                torch.testing.assert_close(c_native0[rid],c0[rid],atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(cb,cd.float(),atol=1e-4,rtol=1e-5)
                checks['c0_head_maxerr']=max(checks['c0_head_maxerr'],float((c_native0[rid]-c0[rid]).abs().max()))
                checks['merge_coeff_maxerr']=max(checks['merge_coeff_maxerr'],float((cb-cd).abs().max()))
                checks['merge_logit_maxerr']=max(checks['merge_logit_maxerr'],float(((cb.double()-cd)@proto.double().flatten(1)).abs().max()))
                key=(iid,int(r['annotation_id']),rid); dC=oracle_map[key].cuda()
                coeffs={'A':c0[rid],'B':cb,'C':(c0[rid].double()+dC).float()}
                masks={}
                for arm,c in coeffs.items():
                    padded=ops.process_mask(proto,c[None],boxes[rid:rid+1],(640,640),upsample=True)
                    masks[arm]=ops.scale_masks(padded[None],x['original_shape'],ratio_pad=x['ratio_pad'])[0,0]>.5
                padded=ops.process_mask(proto,cd.float()[None],boxes[rid:rid+1],(640,640),upsample=True)
                md=ops.scale_masks(padded[None],x['original_shape'],ratio_pad=x['ratio_pad'])[0,0]>.5
                dif=int((md!=masks['B']).sum()); checks['direct_merge_mask_pixel_differences']+=dif; checks['direct_merge_changed_candidates']+=int(dif>0)
                orig=torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])]).astype(bool)).cuda()
                oh,ow=x['original_shape']; gain_im=min(640/oh,640/ow); nh,nw=round(oh*gain_im),round(ow*gain_im)
                left,top=(round(float(q)-.1) for q in x['ratio_pad'][1])
                gt_pad=F.pad(F.interpolate(orig.float()[None,None],(nh,nw),mode='nearest'),(left,640-nw-left,top,640-nh-top))[0,0].bool()
                pred_support=ops.crop_mask(torch.ones((1,640,640),device='cuda'),boxes[rid:rid+1])[0].bool()
                gtbox=x['target_boxes'][k].cuda(); support=ops.crop_mask(torch.ones((1,640,640),device='cuda'),gtbox[None])[0].bool()
                p=up[:,support].T.double(); y=(label[support]==int(x['owners'][k])+1).double()
                area=((gtbox[2:]-gtbox[:2])/640).prod().double()*640*640
                rr={'split':split,'image_id':iid,'annotation_id':int(r['annotation_id']),'branch':'one2one','raw_id':rid,'pyramid_level':lev,'target_gt_idx':int(r['gt_index']),'box_iou':float(r['box_iou']),'merge_mask_pixel_difference':dif}
                for arm,c in coeffs.items():
                    m=masks[arm]; inter=int((m&orig).sum()); union=int((m|orig).sum()); rr[f'iou_{arm}']=inter/max(union,1); rr[f'coverage_{arm}']=inter/max(int(orig.sum()),1); rr[f'mask75_{arm}']=int(rr[f'iou_{arm}']>=.75)
                    z=F.interpolate((c@proto.flatten(1)).reshape(1,1,*proto.shape[-2:]),(640,640),mode='bilinear',align_corners=False)[0,0]
                    rr[f'auc_{arm}'],rr[f'fpr_{arm}']=pixel_auc_fpr(z,gt_pad,pred_support)
                    exact=c0[rid].double() if arm=='A' else cd if arm=='B' else c0[rid].double()+dC
                    di=exact-c0[rid].double()
                    rr[f'bce_{arm}']=float(gain*F.binary_cross_entropy_with_logits(p@exact,y,reduction='sum')/area)
                    rr[f'regularizer_{arm}']=float(.003*.5*di.square().sum()); rr[f'objective_{arm}']=rr[f'bce_{arm}']+rr[f'regularizer_{arm}']
                if 'initial_iou' in r:
                    err=abs(rr['iou_A']-float(r['initial_iou'])); checks['baseline_cached_iou_maxerr']=max(checks['baseline_cached_iou_maxerr'],err)
                    assert err<=1e-6, (key,err)
                results.append(rr)
            if pos%50==0 or pos==len(index[split]): print(json.dumps({'split':split,'images':pos,'total':len(index[split]),'elapsed_s':time.monotonic()-start}),flush=True)
    checks.update(passed=True,merged_checkpoint_sha256=sha(a.out/'YOLO26m_seg_OFFICIAL_SHARED.pt'),original_checkpoint_sha256=sha(a.weights),baseline_reproduced_candidates=sum('initial_iou' in r for it in index['val'] for r in torch.load(a.cache/'images'/f"{int(it['image_id']):012d}.pt",map_location='cpu',weights_only=False)['rows']))
    with (a.out/'PER_CANDIDATE.jsonl').open('w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    dump(a.out/'MERGE_DECODE_AUDIT.json',checks); aggregate(results,a.out)
    print(json.dumps({'completed':True,'candidates':len(results),'merge_checks':checks}),flush=True)

if __name__=='__main__': main()
