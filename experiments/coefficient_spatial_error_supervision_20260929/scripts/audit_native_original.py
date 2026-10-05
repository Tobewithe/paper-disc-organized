import argparse, json, math, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from ultralytics.utils.ops import process_mask, process_mask_native, crop_mask, scale_masks
sys.path.insert(0, str(Path(__file__).parent))
from train_full import load
from train_capacity import BigGlobal

def auc_binary(y, s):
    y=np.asarray(y).astype(np.uint8).ravel(); s=np.asarray(s).ravel()
    pos=y==1; neg=y==0; n1=int(pos.sum()); n0=int(neg.sum())
    if n1==0 or n0==0: return float('nan')
    order=np.argsort(s, kind='mergesort'); ranks=np.empty(len(s), dtype=float); ranks[order]=np.arange(1,len(s)+1)
    return float((ranks[pos].sum()-n1*(n1+1)/2)/(n1*n0))

def letterbox_gt(coco, ann_id, raw):
    arr=coco.annToMask(coco.anns[int(ann_id)]).astype(np.float32)
    t=torch.from_numpy(arr)[None,None]
    H,W=map(int, raw['original_shape']); ih,iw=map(int, raw['input_shape'])
    gain=float(raw.get('gain', min(ih/H, iw/W)))
    nh,nw=round(H*gain), round(W*gain)
    rs=F.interpolate(t,(nh,nw),mode='nearest')[0,0]
    canvas=torch.zeros((ih,iw),dtype=torch.float32)
    left,top=int(raw.get('left', round((iw-nw)/2))),int(raw.get('top', round((ih-nh)/2)))
    canvas[top:top+nh,left:left+nw]=rs
    return canvas

def box_proto(box):
    x1=max(0,min(159,math.floor(float(box[0])*160/640))); y1=max(0,min(159,math.floor(float(box[1])*160/640)))
    x2=max(x1+1,min(160,math.ceil(float(box[2])*160/640))); y2=max(y1+1,min(160,math.ceil(float(box[3])*160/640)))
    return x1,y1,x2,y2

def stats(logit, gt, mask):
    y=gt.numpy().astype(np.uint8); s=logit.numpy(); pr=mask.numpy().astype(bool); g=y.astype(bool)
    inter=int((pr&g).sum()); union=int((pr|g).sum())
    return {'auc':auc_binary(y,s),'iou':float(inter/union) if union else 1.0,'coverage':float(inter/max(int(g.sum()),1)),'fpr':float((pr&~g).sum()/max(int((~g).sum()),1)),'mask75':float(inter/union>=.75) if union else 1.0}

def main(a):
    torch.set_num_threads(8)
    dev=load(a.dev)
    coco=__import__('pycocotools.coco',fromlist=['COCO']).COCO(str(a.coco))
    rawdir=Path(a.raw)
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev])
    ck=torch.load(a.weights,map_location='cpu',weights_only=True); norm=ck['norm']
    model=BigGlobal(h.shape[1]); model.load_state_dict(ck['state']); model.eval()
    hh=((h-norm['hm'])/norm['hs']); pp=((p-norm['pm'])/norm['ps'])
    with torch.no_grad(): pred=model(hh,pp).cpu().view(len(dev),16,16)
    names=['baseline','big_global']
    per={n:[] for n in names}; image_rows={n:{} for n in names}; cache={}
    for i,r in enumerate(dev):
        iid=int(r['image_id']); raw=cache.get(iid)
        if raw is None:
            raw=torch.load(rawdir/f'{iid:012d}.pt',map_location='cpu',weights_only=True); cache[iid]=raw
        rid=int(r['raw_id']); proto=raw['proto'].float(); c0=raw['coeff'][rid].float(); box=raw['boxes'][rid].float()
        zfull=(proto*c0[:,None,None]).sum(0)
        x1,y1,x2,y2=box_proto(box)
        dz=F.interpolate(pred[i][None,None],size=(y2-y1,x2-x1),mode='bilinear',align_corners=False)[0,0]
        zcorr=zfull.clone(); zcorr[y1:y2,x1:x2]+=dz
        gt640=letterbox_gt(coco,int(r['annotation_id']),raw)
        # official default path: threshold after upsampling to letterboxed input image.
        for name,z in [('baseline',zfull),('big_global',zcorr)]:
            z640=F.interpolate(z[None,None],(int(raw['input_shape'][0]),int(raw['input_shape'][1])),mode='bilinear',align_corners=False)[0,0]
            b640=box
            m640=crop_mask(z640[None], b640[None])[0].gt(0)
            row=stats(z640,gt640,m640); per[name].append(row)
            image_rows[name].setdefault(iid,[]).append(row)
    summary={n:{m:float(np.nanmean([r[m] for r in rows])) for m in ['auc','iou','coverage','fpr','mask75']} for n,rows in per.items()}
    image_macro={n:{m:float(np.nanmean([np.nanmean([r[m] for r in rows]) for rows in image_rows[n].values()])) for m in ['auc','iou','coverage','fpr','mask75']} for n in names}
    img_counts={str(k):len(v) for k,v in image_rows['baseline'].items()}
    diff={m:float(np.nanmean([per['big_global'][i][m]-per['baseline'][i][m] for i in range(len(dev))])) for m in ['auc','iou','coverage','fpr','mask75']}
    image_diff={m:float(np.nanmean([np.nanmean([r[m] for r in image_rows['big_global'][iid]])-np.nanmean([r[m] for r in image_rows['baseline'][iid]]) for iid in image_rows['baseline']])) for m in ['auc','iou','coverage','fpr','mask75']}
    rng=np.random.default_rng(0); candidate_boot={}; image_boot={}
    for m in ['auc','iou','coverage','fpr','mask75']:
        d=np.asarray([per['big_global'][i][m]-per['baseline'][i][m] for i in range(len(dev))],dtype=float)
        b=np.asarray([np.nanmean(d[rng.integers(0,len(d),len(d))]) for _ in range(5000)])
        candidate_boot[m]={'mean':float(np.nanmean(d)),'ci95':[float(np.nanquantile(b,.025)),float(np.nanquantile(b,.975))]}
        ids=list(image_rows['baseline'])
        di=np.asarray([np.nanmean([r[m] for r in image_rows['big_global'][iid]])-np.nanmean([r[m] for r in image_rows['baseline'][iid]]) for iid in ids],dtype=float)
        b2=np.asarray([np.nanmean(di[rng.integers(0,len(di),len(di))]) for _ in range(5000)])
        image_boot[m]={'mean':float(np.nanmean(di)),'ci95':[float(np.nanquantile(b2,.025)),float(np.nanquantile(b2,.975))]}
    out={'protocol':'official_default_process_mask_on_letterboxed_input','n_candidates':len(dev),'n_unique_images':len(img_counts),'candidate_per_image_min':min(img_counts.values()),'candidate_per_image_median':float(np.median(list(img_counts.values()))),'candidate_per_image_max':max(img_counts.values()),'summary_candidate_mean':summary,'summary_image_macro':image_macro,'difference_candidate_mean_big_global_minus_baseline':diff,'difference_image_macro_big_global_minus_baseline':image_diff,'candidate_paired_bootstrap':candidate_boot,'image_macro_paired_bootstrap':image_boot,'image_candidate_count_histogram':{str(k):list(img_counts.values()).count(k) for k in sorted(set(img_counts.values()))}}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--dev',required=True); q.add_argument('--raw',required=True); q.add_argument('--coco',required=True); q.add_argument('--weights',required=True); q.add_argument('--out',required=True); main(q.parse_args())





