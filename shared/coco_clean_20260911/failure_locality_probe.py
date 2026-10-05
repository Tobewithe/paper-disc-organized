"""Original-mask error distances and spatially cross-fitted GT readout opportunities."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,sha,write_json
from three_region_probe import write_csv
from relative_ownership_experiment import read
from crossimage_response_experiment import load,gt_input_regions
from crossimage_response_decoder import prepare


def distances(mask):
    padded=np.pad(mask,1,constant_values=False)
    inside=cv2.distanceTransform(padded.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    outside=cv2.distanceTransform((~padded).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    return inside,outside


def auc(pos,neg):
    # Mann-Whitney with tie half-credit, without a pairwise temporary matrix.
    ordered=np.sort(neg)
    return float((.5*(np.searchsorted(ordered,pos,'left')+np.searchsorted(ordered,pos,'right'))/len(neg)).mean())


def select_threshold(s,y):
    thresholds=np.unique(np.r_[np.quantile(s,np.linspace(0,1,257)),np.nextafter(s.min(),-np.inf),s.max(),0.])
    positives=np.sort(s[y]);negatives=np.sort(s[~y])
    tp=len(positives)-np.searchsorted(positives,thresholds,'right')
    fp=len(negatives)-np.searchsorted(negatives,thresholds,'right')
    iou=tp/(y.sum()+fp);best=np.flatnonzero(iou==iou.max())
    k=min(best,key=lambda j:(abs(thresholds[j]),thresholds[j]))
    return float(thresholds[k]),float(iou[k])


def ridge(xtrain,xtest,y):
    mean=xtrain.mean(0);std=xtrain.std(0).clip(.01);tr=(xtrain-mean)/std;te=(xtest-mean)/std
    labels=2*y.astype(float)-1
    beta=np.linalg.solve(tr.T@tr/len(tr)+.1*np.eye(tr.shape[1]),tr.T@(labels-labels.mean())/len(tr))
    w=beta/std;b=float(labels.mean()-mean@w)
    return xtrain@w+b,xtest@w+b,w,b


def heldout_metrics(s,y,same,bg,tau):
    pred=s>tau;positive=int(y.sum());tp=int((pred&y).sum());fp=int((pred&~y).sum())
    return dict(sample_iou=tp/(positive+fp),coverage=tp/positive,same_neighbor=float((pred&same).sum()/positive),
        background=float((pred&bg).sum()/positive),all_fp=fp/positive,auc=auc(s[y],s[~y]))


def exact_regions(gt,iid,item,prepared,meta,replay):
    shape=tuple(map(int,item['shape']));raster={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]}
    aa=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)];union=np.zeros(shape,bool);crowd=np.zeros(shape,bool)
    for a in gt.imgToAnns[iid]:
        if a.get('iscrowd',0):crowd|=raster[a['id']]
        else:union|=raster[a['id']]
    valid=~crowd;same={cat:np.logical_or.reduce([raster[a['id']] for a in aa if a['category_id']==cat]) for cat in {a['category_id'] for a in aa}}
    interior={a['id']:distances(raster[a['id']])[0] for a in aa}
    core={cat:np.logical_or.reduce([raster[a['id']]&(interior[a['id']]>.02*np.sqrt(max(1,(raster[a['id']]&valid).sum()))) for a in aa if a['category_id']==cat]) for cat in same}
    mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])));rows=[];errors=0.
    for a in aa:
        aid=a['id'];own=raster[aid]&valid;area=int(own.sum());j=mapping.get(aid);cat=a['category_id']
        base=dict(image_id=iid,annotation_id=aid,category_id=cat,coco_area=a['area'],ici=float(meta[aid]['ici_same']),valid_gt_pixels=area,
            status='no_valid_pixels' if not area else 'unmatched_box' if j is None else 'matched',prediction_index=-1 if j is None else j)
        if not area or j is None:rows.append(base);continue
        mask=(ops.scale_masks(prepared['initial'][j:j+1,None],shape)[0,0]>.5).cpu().numpy()&valid
        support=(ops.scale_masks(prepared['support'][j:j+1,None].to(torch.uint8),shape)[0,0]>.5).cpu().numpy()&valid
        tp=int((mask&own).sum());fn=own&~mask;sn=mask&same[cat]&~own;bg=mask&~union;other=mask&union&~own&~same[cat]
        fnin=fn&support;fnout=fn&~support;fp=int(sn.sum()+bg.sum()+other.sum());inside=int((own&support).sum())
        assert int(fn.sum())==int(fnin.sum()+fnout.sum()) and not np.any(mask&~support)
        din,dout=distances(raster[aid]);scale=np.sqrt(area)
        ih,iw=map(int,item['input_shape']);box=prepared['boxes'][j];yy,xx=torch.meshgrid(torch.arange(ih,device='cuda'),torch.arange(iw,device='cuda'),indexing='ij')
        rim=prepared['support'][j]&((xx-box[0]<2)|(box[2]-xx<=2)|(yy-box[1]<2)|(box[3]-yy<=2))
        rim=(ops.scale_masks(rim[None,None].to(torch.uint8),shape)[0,0]>.5).cpu().numpy()&valid
        base.update(iou=tp/(area+fp),coverage=tp/area,crop_ceiling=inside/area,true_positive=tp,false_positive=fp,false_negative=area-tp,
            same_neighbor_pixels=int(sn.sum()),other_category_pixels=int(other.sum()),background_pixels=int(bg.sum()),fn_inside_pixels=int(fnin.sum()),fn_outside_pixels=int(fnout.sum()),
            same_neighbor_core_pixels=int((sn&core[cat]).sum()),same_neighbor_crop_rim_pixels=int((sn&rim).sum()),background_crop_rim_pixels=int((bg&rim).sum()))
        for name,region,distance in [('same_neighbor',sn,dout),('background',bg,dout),('fn_inside',fnin,din)]:
            normalized=distance[region]/scale
            base.update({name+'_near_pixels':int((normalized<=.02).sum()),name+'_middle_pixels':int(((normalized>.02)&(normalized<=.05)).sum()),name+'_far_pixels':int((normalized>.05).sum())})
        expected=replay.get(aid)
        if expected:
            values={'coverage':base['coverage'],'same_neighbor':int(sn.sum())/area,'background':int(bg.sum())/area,'mask_iou':base['iou']}
            err=max(abs(float(expected[k])-v) for k,v in values.items());errors=max(errors,err);assert err<1e-10,(iid,aid,err)
        rows.append(base)
    return rows,errors


def probe_target(iid,aid,j,item,prepared,input_raster,input_union,input_valid,meta,seeds):
    ih,iw=map(int,item['input_shape']);ph,pw=item['proto'].shape[-2:];box=prepared['boxes'][j].cpu().numpy()
    support=prepared['support'][j].cpu().numpy()&input_valid.cpu().numpy()
    yy,xx=np.mgrid[:ih,:iw];px=(xx+.5)*pw/iw-.5;py=(yy+.5)*ph/ih-.5
    x0=np.floor(px).astype(int).clip(0,pw-1);x1=(np.floor(px)+1).astype(int).clip(0,pw-1)
    y0=np.floor(py).astype(int).clip(0,ph-1);y1=(np.floor(py)+1).astype(int).clip(0,ph-1)
    gy,gx=np.mgrid[:ph,:pw];cx=(gx+.5)*iw/pw;cy=(gy+.5)*ih/ph;wh=np.maximum(box[2:]-box[:2],1)
    cells=((np.floor((cx-box[0])/wh[0]*4).astype(int)+np.floor((cy-box[1])/wh[1]*4).astype(int))%2).astype(bool)
    stencil=np.stack([cells[y0,x0],cells[y0,x1],cells[y1,x0],cells[y1,x1]])
    foldmasks=[support&~stencil.any(0),support&stencil.all(0)];pools=[np.flatnonzero(m) for m in foldmasks]
    common=dict(image_id=iid,annotation_id=aid,ici=float(meta[aid]['ici_same']),prediction_index=j,support_pixels=int(support.sum()),fold0_pixels=len(pools[0]),fold1_pixels=len(pools[1]))
    if min(map(len,pools))<64:return [],[dict(**common,status='too_small_support')],[]
    own=input_raster[aid].flatten().cpu().numpy();bg=(~input_union&input_valid).flatten().cpu().numpy()
    # same category GT mask is supplied by the caller as a cached extra tensor.
    same=(prepared['same_input'][int(meta[aid]['category_id'])]&~input_raster[aid]&input_valid).flatten().cpu().numpy()
    p=torch.tensor(item['proto'],device='cuda').float();c=item['coeff'][j].astype(float);rows=[];statuses=[];directions=[]
    for seed in seeds:
        rng=np.random.default_rng(20260911+iid*31+aid*7+seed*7907);positions=[rng.choice(pool,min(1024,len(pool)),replace=False) for pool in pools]
        if any(min(int(own[pos].sum()),int((~own[pos]).sum()))<4 for pos in positions):
            statuses.append(dict(**common,seed=seed,status='too_few_sampled_classes'));continue
        pos=np.concatenate(positions);labels=own[pos];ss=same[pos];bb=bg[pos];folds=np.r_[np.zeros(len(positions[0]),int),np.ones(len(positions[1]),int)]
        ys,xs=np.unravel_index(pos,(ih,iw));st=np.stack([y0[ys,xs]*pw+x0[ys,xs],y0[ys,xs]*pw+x1[ys,xs],y1[ys,xs]*pw+x0[ys,xs],y1[ys,xs]*pw+x1[ys,xs]],1)
        assert not np.intersect1d(st[folds==0],st[folds==1]).size
        grid=torch.tensor(np.stack([2*(xs+.5)/iw-1,2*(ys+.5)/ih-1],1),device='cuda',dtype=torch.float32).reshape(1,-1,1,2)
        feature=F.grid_sample(p[None],grid,align_corners=False,padding_mode='border')[0,:,:,0].T.cpu().numpy().astype(float)
        raw=prepared['z'][j].flatten()[torch.tensor(pos,device='cuda')].cpu().numpy().astype(float)
        replay=float(np.max(np.abs(feature@c-raw)))
        assert replay<2e-4,(iid,aid,replay)
        coords=np.stack([(xs+.5-box[0])/wh[0],(ys+.5-box[1])/wh[1]],1).clip(0,1);u,v=coords.T
        gates=np.stack([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],1)
        local=np.concatenate([(gates[:,:,None]*feature[:,None,:]).reshape(len(pos),-1),gates],1)
        global_fits=[]
        for fold in [0,1]:
            tr=folds==fold;te=~tr;ytr=labels[tr];yte=labels[te]
            scores={'original':(raw[tr],raw[te],None,None),'threshold_oracle':(raw[tr],raw[te],None,None)}
            for name,x in [('proto_oracle',feature),('local_oracle',local),('coordinate_oracle',coords)]:
                scores[name]=ridge(x[tr],x[te],ytr)
            shuffled=np.random.default_rng(20260911+iid+aid+seed*7907+fold*3181).permutation(ytr)
            scores['shuffle_oracle']=ridge(feature[tr],feature[te],shuffled)
            global_fits.append((scores['proto_oracle'][2],scores['proto_oracle'][3]))
            for name,(strain,stest,w,b) in scores.items():
                if name=='original':tau=0.;trainmetric=heldout_metrics(strain,ytr,ss[tr],bb[tr],tau)['sample_iou']
                else:
                    tau,_=select_threshold(strain,shuffled if name=='shuffle_oracle' else ytr)
                    trainmetric=heldout_metrics(strain,ytr,ss[tr],bb[tr],tau)['sample_iou']
                metric=heldout_metrics(stest,yte,ss[te],bb[te],tau)
                row=dict(image_id=iid,annotation_id=aid,ici=common['ici'],seed=seed,fold=fold,readout=name,
                    train_pixels=int(tr.sum()),eval_pixels=int(te.sum()),train_positives=int(ytr.sum()),eval_positives=int(yte.sum()),
                    eval_same_pixels=int(ss[te].sum()),eval_bg_pixels=int(bb[te].sum()),threshold=tau,train_iou=trainmetric,raw_feature_replay_max_abs=replay,**metric)
                rows.append(row)
                if name=='proto_oracle':
                    cosine=float(np.dot(c,w)/(np.linalg.norm(c)*np.linalg.norm(w)+1e-15))
                    # Ridge and network scores have arbitrary units: compare signed
                    # margins standardized by each TRAIN fold score standard deviation.
                    delta=(stest-tau)/max(float(strain.std()),1e-8)-raw[te]/max(float(raw[tr].std()),1e-8)
                    shifts={f'standardized_margin_change_{label}':float(delta[region].mean()) if region.any() else None
                        for label,region in [('own',yte),('same_neighbor',ss[te]),('background',bb[te])]}
                    directions.append(dict(image_id=iid,annotation_id=aid,seed=seed,fold=fold,intercept=b,threshold=tau,original_coefficient_cosine=cosine,
                        **shifts,
                        **{f'w{k}':float(vv) for k,vv in enumerate(w)},**{f'c{k}':float(vv) for k,vv in enumerate(c)}))
        score0=feature@global_fits[0][0]+global_fits[0][1];score1=feature@global_fits[1][0]+global_fits[1][1]
        corr=float(np.corrcoef(score0,score1)[0,1]) if min(score0.std(),score1.std())>1e-10 else 0.
        statuses.append(dict(**common,seed=seed,status='ok',no_shared_proto_cells=True,fold_score_correlation=corr,raw_feature_replay_max_abs=replay))
    return rows,statuses,directions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'diagnostics/crossimage_response_20260911';chosen=json.loads((source/'selection.json').read_text());ids={'train':chosen['calibration'],'val':chosen['evaluation']}
    if a.smoke:ids={k:v[:4] for k,v in ids.items()}
    config=Path(__file__).with_name('failure_locality_protocol.json');write_json(out/'protocol.json',dict(config=json.loads(config.read_text()),images=ids,smoke=a.smoke,
        script_sha256=sha(__file__),source_selection_sha256=sha(source/'selection.json'),protocol_sha256=sha(config),
        source_cache_manifest_sha256={name:sha(source/name) for name in ['TRAIN_CACHE_HASHES.json','EVALUATION_CACHE_HASHES.json']},
        annotations={s:sha(ROOT/f'data/annotations/instances_{s}2017.json') for s in ids}))
    replay={int(r['target_annotation']):r for r in read(source/'evaluation_spatial.csv') if r['domain']=='normal' and r['arm']=='initial'}
    trainhash=json.loads((source/'TRAIN_CACHE_HASHES.json').read_text());valhash=json.loads((source/'EVALUATION_CACHE_HASHES.json').read_text())
    allmaps=[];allrows=[];statuses=[];directions=[];hashes={};start=time.monotonic();replayerror=0.
    for split,imageids in ids.items():
        with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/f'data/annotations/instances_{split}2017.json'))
        mp=ROOT/('census/train2017_instances.csv' if split=='train' else 'census/COCO_EVAL_INSTANCE_MANIFEST.csv');meta={int(r['annotation_id']):r for r in read(mp)}
        for iid in imageids:
            for annotation in gt.imgToAnns[iid]:
                if not annotation.get('iscrowd',0):meta[annotation['id']]['category_id']=annotation['category_id']
        for num,iid in enumerate(imageids,1):
            path=source/'calibration_cache'/f'{iid}.npz' if split=='train' else ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz'
            digest=sha(path);assert digest==(trainhash[f'calibration_cache/{iid}.npz'] if split=='train' else valhash[str(iid)]);hashes[f'{split}/{iid}']=digest
            item=load(path);prepared=prepare(item);exact,err=exact_regions(gt,iid,item,prepared,meta,replay if split=='val' else {})
            replayerror=max(replayerror,err);allmaps.extend(dict(split=split,**r) for r in exact)
            raster,union,valid=gt_input_regions(gt,iid,item)
            prepared['same_input']={cat:torch.stack([raster[aid] for aid in raster if gt.anns[aid]['category_id']==cat]).any(0) for cat in {gt.anns[aid]['category_id'] for aid in raster}}
            for r in exact:
                if r['status']!='matched':continue
                rr,ss,dd=probe_target(iid,r['annotation_id'],r['prediction_index'],item,prepared,raster,union,valid,meta,[0,1,2])
                allrows.extend(dict(split=split,**q) for q in rr);statuses.extend(dict(split=split,**q) for q in ss);directions.extend(dict(split=split,**q) for q in dd)
            if num%10==0 or num==len(imageids):
                progress=dict(split=split,images=num,total=len(imageids),readouts=len(allrows),seconds=round(time.monotonic()-start,1));print(json.dumps(progress),flush=True);write_json(out/'progress.json',progress)
    write_csv(out/'exact_error_maps.csv',allmaps);write_csv(out/'readouts.csv',allrows);write_csv(out/'readout_statuses.csv',statuses);write_csv(out/'oracle_directions.csv',directions);write_json(out/'CACHE_HASHES.json',hashes)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,GT_oracle=True,images={s:len(ii) for s,ii in ids.items()},exact_rows=len(allmaps),readouts=len(allrows),
        original_spatial_replay_max_abs=replayerror,seconds=time.monotonic()-start,hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':main()
