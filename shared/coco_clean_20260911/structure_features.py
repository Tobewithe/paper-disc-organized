"""Within-image, spatial-block-held-out linear representation probes.

GT defines the pair and pixel regions. This is an oracle representation
diagnostic, never an instance-segmentation predictor or cross-image ID model.
All hyperparameters are fixed; no evaluation-based choice is performed.
"""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

def rank(value):
    return hashlib.sha256(f'structure-pair:20260911:{value}'.encode()).hexdigest()

def iou(a,b):
    wh=np.maximum(0,np.minimum(np.array(a[:2])+a[2:],np.array(b[:2])+b[2:])-np.maximum(a[:2],b[:2]));x=wh.prod()
    return float(x/(a[2]*a[3]+b[2]*b[3]-x+1e-12))

def choose_pairs(anns,iid):
    groups={'same_adjacent':[],'same_separate':[],'different_adjacent':[]}
    for j,a in enumerate(anns):
        for b in anns[j+1:]:
            v=iou(a['bbox'],b['bbox']);same=a['category_id']==b['category_id']
            key='same_adjacent' if same and v>.05 else 'same_separate' if same and v==0 else 'different_adjacent' if not same and v>.05 else None
            if key:groups[key].append((a,b,v))
    return [(key,*min(pairs,key=lambda q:rank(f'{iid}:{q[0]["id"]}:{q[1]["id"]}'))) for key,pairs in groups.items() if pairs]

def auc(y,score):
    p=score[y>0];n=score[y<0]
    return float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean())

def cosine(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm>1e-12 else None

def readout(x,train,y,seed):
    nchannel=x.shape[1]
    if nchannel>32:
        rng=np.random.default_rng(73000+seed+nchannel);projection=rng.normal(size=(nchannel,32))/np.sqrt(32)
        x=x@projection
    mu=x[train].mean(0);std=x[train].std(0).clip(.01);xt=(x-mu)/std
    tr=xt[train];te=xt[~train];yt=y[train]
    gram=tr.T@tr/len(tr)+.1*np.eye(tr.shape[1])
    rhs=tr.T@(yt-yt.mean())/len(tr);w=np.linalg.solve(gram,rhs)
    score=te@w+yt.mean()
    rng=np.random.default_rng(seed+91000);shuffled=rng.permutation(yt)
    null=np.linalg.solve(gram,tr.T@(shuffled-shuffled.mean())/len(tr));nullscore=te@null+shuffled.mean()
    return auc(y[~train],score),auc(y[~train],nullscore)


@torch.inference_mode()
def probe_image(gt,iid,maps,cap,mapping,meta,seed_offset=0):
    ordinary=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
    pairs=choose_pairs(ordinary,iid);rows=[];skips=[];shape=cap['shape'];ih,iw=cap['input_shape'];h,w=shape
    masks={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]}
    crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
    for a in gt.imgToAnns[iid]:
        if a.get('iscrowd',0):crowd|=masks[a['id']]
        else:union|=masks[a['id']]
    gain=min(ih/h,iw/w);rh,rw=round(h*gain),round(w*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
    maps={**maps,'proto':cap['proto'][None]}
    for kind,a,b,overlap in pairs:
        common=dict(image_id=iid,annotation_a=a['id'],annotation_b=b['id'],pair_type=kind,category_a=a['category_id'],category_b=b['category_id'],area_a=a['area'],area_b=b['area'],box_iou=overlap,ici_a=float(meta[a['id']]['ici_same']),ici_b=float(meta[b['id']]['ici_same']))
        ax,ay,aw,ah=a['bbox'];bx,by,bw,bh=b['bbox'];xmin=min(ax,bx);xmax=max(ax+aw,bx+bw);ymin=min(ay,by);ymax=max(ay+ah,by+bh)
        common['normalized_center_distance']=float(np.hypot(ax+aw/2-bx-bw/2,ay+ah/2-by-bh/2)/max(np.sqrt(aw*ah),1))
        ma=masks[a['id']]&~masks[b['id']]&~crowd;mb=masks[b['id']]&~masks[a['id']]&~crowd
        yy,xx=np.mgrid[:h,:w];support=(xx>=xmin)&(xx<xmax)&(yy>=ymin)&(yy<ymax)
        bg=support&~union&~crowd
        # Normalized 4x4 pair-box blocks, two parities. RF overlap remains a stated limitation.
        blocks=(np.floor((xx+.5-xmin)/max(xmax-xmin,1)*4).astype(int)+np.floor((yy+.5-ymin)/max(ymax-ymin,1)*4).astype(int))%2==0
        occupancy=torch.tensor(np.stack([ma,mb]).astype(np.float32),device='cuda')[:,None]
        occupancy=F.interpolate(occupancy,(rh,rw),mode='area')
        occupancy=F.pad(occupancy,(left,iw-rw-left,top,ih-rh-top))
        for task,neg in [('own_neighbor',mb),('own_background',bg)]:
            picks=[];labels=[];folds=[];rng=np.random.default_rng(20260911+iid+a['id']+b['id'])
            for sign,region in [(1,ma),(-1,neg)]:
                for istrain in [True,False]:
                    positions=np.flatnonzero(region&(blocks if istrain else ~blocks))
                    if len(positions)<8:break
                    chosen=rng.choice(positions,min(64,len(positions)),replace=False)
                    picks.extend(chosen.tolist());labels.extend([sign]*len(chosen));folds.extend([istrain]*len(chosen))
                else:continue
                break
            else:
                picks=np.array(picks);y=np.array(labels);train=np.array(folds);py,px=np.unravel_index(picks,shape)
                # Pixel centers use the actual rounded resize dimensions and letterbox padding.
                ix=(px+.5)*rw/w+left;iy=(py+.5)*rh/h+top
                grid=torch.tensor(np.stack([2*ix/iw-1,2*iy/ih-1],1),dtype=torch.float32,device='cuda').reshape(1,-1,1,2)
                coords=np.stack([(px+.5-xmin)/max(xmax-xmin,1),(py+.5-ymin)/max(ymax-ymin,1)],1)
                for layer,feature in maps.items():
                    sampled=F.grid_sample(feature.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,:,0].T.cpu().numpy().astype(float)
                    co=F.adaptive_avg_pool2d(occupancy,feature.shape[-2:])[:,0]
                    union_cells=(co.sum(0)>.1);mixed=(co[0]>.1)&(co[1]>.1)
                    pooled_cos=cosine(sampled[y>0].mean(0),sampled[y<0].mean(0))
                    for seed in [0,1,2]:
                        score,null=readout(sampled,train,y,seed);coordscore,_=readout(coords,train,y,seed)
                        row=dict(**common,task=task,layer=layer,channels=feature.shape[1],grid_h=feature.shape[-2],grid_w=feature.shape[-1],probe_seed=seed,train_pixels=int(train.sum()),eval_pixels=int((~train).sum()),region_mean_cosine=pooled_cos,auc=score,shuffled_train_auc=null,coordinate_auc=coordscore,mixed_cell_fraction=float(mixed.sum()/union_cells.sum().clamp_min(1)))
                        if layer=='proto' and a['id'] in mapping:
                            ca=cap['coeff'][mapping[a['id']]].cpu().numpy();sa=sampled@ca
                            row.update(actual_a_auc=auc(y[~train],sa[~train]),actual_a_mean_margin=float(sa[y>0].mean()-sa[y<0].mean()))
                            if task=='own_neighbor' and b['id'] in mapping:
                                cb=cap['coeff'][mapping[b['id']]].cpu().numpy();sd=sampled@(ca-cb)
                                row.update(actual_pair_difference_auc=auc(y[~train],sd[~train]),coefficient_cosine=cosine(ca,cb))
                        rows.append(row)
                continue
            skips.append(dict(**common,task=task,reason='fewer_than_8_pixels_in_a_class_fold'))
    return rows,skips
