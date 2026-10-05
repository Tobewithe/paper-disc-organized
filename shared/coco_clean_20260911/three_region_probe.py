"""Frozen three-region, spatially cross-fitted representation diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, contextlib, csv, hashlib, io, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from frozen_mechanism_probe import ROOT, write_json, sha, box_iou
from relative_ownership_experiment import read


def rank(value):return hashlib.sha256(f'three-region:20260911:{value}'.encode()).hexdigest()


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def auc(positive,negative):
    return float(((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])).mean())


def ridge(train,test,y):
    mu=train.mean(0);std=train.std(0).clip(.01)
    tr=(train-mu)/std;te=(test-mu)/std
    weights=np.linalg.solve(tr.T@tr/len(tr)+.1*np.eye(tr.shape[1]),tr.T@(y-y.mean())/len(tr))
    return tr@weights+y.mean(),te@weights+y.mean()


def scores(features,coords,ca,cb,train,classes,seed):
    own=features@ca;neighbor=features@cb;difference=own-neighbor
    raw={'actual_own':own,'signed_difference':difference,'foreground_safe':np.minimum(own,difference)}
    out={k:(v[train],v[~train]) for k,v in raw.items()}
    freq=np.random.default_rng(81347).normal(0,2,size=(2,16))
    phase=2*np.pi*coords@freq
    geom=np.concatenate([np.sin(phase),np.cos(phase)],1)
    y=np.where(classes[train]==0,1.,-1.)
    for name,x in [('two_logits',np.stack([own,neighbor],1)),('proto32',features),('coordinate32',geom),('coordinate2',coords)]:
        out['oracle_'+name]=ridge(x[train],x[~train],y)
    for j in range(3):
        shuffled=np.random.default_rng(seed+j*1297).permutation(y)
        out[f'shuffle_proto_s{j}']=ridge(features[train],features[~train],shuffled)
    return out


@torch.inference_mode()
def analyze(gt,iid,item,metadata,split):
    annotations=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
    candidates=[]
    for j,a in enumerate(annotations):
        for b in annotations[j+1:]:
            if a['category_id']==b['category_id'] and box_iou(a['bbox'],b['bbox'])>.05:
                aa,bb=sorted([a['id'],b['id']]);candidates.append((aa,bb))
    common=dict(split=split,image_id=iid,gt=len(annotations),eligible_gt_pairs=len(candidates))
    if not candidates:return [],[dict(**common,status='no_gt_pair')]
    aid,bid=min(candidates,key=lambda pair:rank(f'{iid}:{pair[0]}:{pair[1]}'))
    mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
    common.update(annotation_a=aid,annotation_b=bid,matched_a=aid in mapping,matched_b=bid in mapping)
    if aid not in mapping or bid not in mapping:return [],[dict(**common,status='pair_not_both_bbox50_matched')]
    h,w=map(int,item['shape']);ih,iw=map(int,item['input_shape']);p=torch.tensor(item['proto'],device='cuda').float();ph,pw=p.shape[-2:]
    coeff=item['coeff'];boxes=item['boxes'];raster={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]}
    occupancy=sum((raster[a['id']].astype(np.int16) for a in annotations),np.zeros((h,w),np.int16))
    crowd=np.zeros((h,w),bool)
    for a in gt.imgToAnns[iid]:
        if a.get('iscrowd',0):crowd|=raster[a['id']]
    valid=~crowd;gain=min(ih/h,iw/w);rh,rw=round(h*gain),round(w*gain);top=round((ih-rh)/2-.1);left=round((iw-rw)/2-.1)
    yy,xx=np.mgrid[:h,:w];ix=(xx+.5)*rw/w+left;iy=(yy+.5)*rh/h+top
    px=ix/iw*pw-.5;py=iy/ih*ph-.5
    x0=np.floor(px).astype(int).clip(0,pw-1);x1=(np.floor(px).astype(int)+1).clip(0,pw-1)
    y0=np.floor(py).astype(int).clip(0,ph-1);y1=(np.floor(py).astype(int)+1).clip(0,ph-1)
    gy,gx=np.mgrid[:ph,:pw];cx=(gx+.5)*iw/pw;cy=(gy+.5)*ih/ph
    rows=[];statuses=[]
    for target,other in [(aid,bid),(bid,aid)]:
        a=gt.anns[target];b=gt.anns[other];j,k=mapping[target],mapping[other]
        for domain,expansion in [('predicted_crop',0.),('expand_each_side_20pct',.2)]:
            box=np.array(boxes[j],dtype=float);width,height=box[2:]-box[:2]
            box=box+np.array([-width,-height,width,height])*expansion
            inside=(ix>=box[0])&(ix<box[2])&(iy>=box[1])&(iy<box[3])
            cells=((np.floor((cx-box[0])/max(box[2]-box[0],1)*4).astype(int)+np.floor((cy-box[1])/max(box[3]-box[1],1)*4).astype(int))%2).astype(bool)
            stencil=np.stack([cells[y0,x0],cells[y0,x1],cells[y1,x0],cells[y1,x1]])
            fold0=~stencil.any(0);fold1=stencil.all(0)
            own=raster[target]&(occupancy==1)&valid&inside
            neighbor=raster[other]&(occupancy==1)&valid&inside
            background=(occupancy==0)&valid&inside
            status=dict(**common,target_annotation=target,other_annotation=other,domain=domain,
                target_ici=float(metadata[target]['ici_same']),category_id=a['category_id'],gt_area=a['area'],
                gt_pair_box_iou=box_iou(a['bbox'],b['bbox']),pred_pair_box_iou=box_iou([boxes[j,0],boxes[j,1],width,height],
                [boxes[k,0],boxes[k,1],boxes[k,2]-boxes[k,0],boxes[k,3]-boxes[k,1]]),
                support_pixels=int((inside&valid).sum()),gt_overlap_pixels=int((inside&valid&(occupancy>1)).sum()),
                own_pixels=int(own.sum()),neighbor_pixels=int(neighbor.sum()),background_pixels=int(background.sum()),
                third_instance_exclusive_pixels=int((inside&valid&(occupancy==1)&~raster[target]&~raster[other]).sum()))
            positions=[];labels=[];folds=[]
            seed=20260911+iid*31+target*7+int(expansion*100)
            rng=np.random.default_rng(seed)
            failed=False
            for fold,mask in [(0,fold0),(1,fold1)]:
                pools=[np.flatnonzero(region&mask) for region in [own,neighbor,background]]
                count=min(128,*map(len,pools));status[f'pixels_per_class_fold{fold}']=count
                if count<12:failed=True;break
                for label,pool in enumerate(pools):
                    positions.extend(rng.choice(pool,count,replace=False).tolist());labels.extend([label]*count);folds.extend([fold]*count)
            if failed:
                statuses.append(dict(**status,status='insufficient_three_regions_per_fold'));continue
            positions=np.array(positions);labels=np.array(labels);folds=np.array(folds);ys,xs=np.unravel_index(positions,(h,w))
            # Explicit stencil disjointness witness for the two sampled folds.
            stencils=np.stack([y0[ys,xs]*pw+x0[ys,xs],y0[ys,xs]*pw+x1[ys,xs],y1[ys,xs]*pw+x0[ys,xs],y1[ys,xs]*pw+x1[ys,xs]],1)
            assert not np.intersect1d(stencils[folds==0],stencils[folds==1]).size
            grid=torch.tensor(np.stack([2*ix[ys,xs]/iw-1,2*iy[ys,xs]/ih-1],1),device='cuda',dtype=torch.float32).reshape(1,-1,1,2)
            feature=F.grid_sample(p[None],grid,align_corners=False,padding_mode='border')[0,:,:,0].T.cpu().numpy().astype(float)
            coords=np.stack([(ix[ys,xs]-box[0])/max(box[2]-box[0],1),(iy[ys,xs]-box[1])/max(box[3]-box[1],1)],1)
            for fold in [0,1]:
                train=folds==fold;classes=labels[~train];trainclasses=labels[train]
                for name,(tr,te) in scores(feature,coords,coeff[j],coeff[k],train,labels,seed+fold*9817).items():
                    positive=te[classes==0];near=te[classes==1];bg=te[classes==2]
                    # Uniform threshold only from target-positive training pixels, not eval.
                    tau=float(np.quantile(tr[trainclasses==0],.1))
                    row=dict(split=split,image_id=iid,target_annotation=target,other_annotation=other,domain=domain,fold=fold,
                        target_ici=status['target_ici'],category_id=a['category_id'],gt_area=a['area'],readout=name,
                        eval_pixels_per_class=len(positive),auc_neighbor=auc(positive,near),auc_background=auc(positive,bg),
                        auc_mixed=auc(positive,np.concatenate([near,bg])),threshold_from_train=tau,
                        train_coverage=float((tr[trainclasses==0]>tau).mean()),coverage=float((positive>tau).mean()),
                        neighbor_fpr=float((near>tau).mean()),background_fpr=float((bg>tau).mean()),
                        sampled_precision=float((positive>tau).sum()/max((te>tau).sum(),1)),
                        zero_coverage=float((positive>0).mean()),zero_neighbor_fpr=float((near>0).mean()),zero_background_fpr=float((bg>0).mean()))
                    rows.append(row)
            statuses.append(dict(**status,status='ok',no_shared_proto_stencil=True))
    return rows,statuses


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'diagnostics/relative_ownership_20260911';chosen=json.loads((source/'selection.json').read_text())
    selection={'train':chosen['development'],'val':sorted(sorted(chosen['evaluation'],key=lambda x:rank(f'val:{x}'))[:200])}
    if a.smoke:selection={k:v[:8] for k,v in selection.items()}
    configpath=Path(__file__).with_name('three_region_protocol.json')
    write_json(out/'protocol.json',dict(config=json.loads(configpath.read_text()),images=selection,
        script_sha256=sha(__file__),source_selection_sha256=sha(source/'selection.json'),protocol_source_sha256=sha(configpath),
        annotations={s:sha(ROOT/f'data/annotations/instances_{s}2017.json') for s in selection}))
    allrows=[];allstatuses=[];hashes={};start=time.monotonic()
    for split,ids in selection.items():
        with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/f'data/annotations/instances_{split}2017.json'))
        mp=ROOT/('census/train2017_instances.csv' if split=='train' else 'census/COCO_EVAL_INSTANCE_MANIFEST.csv')
        meta={int(r['annotation_id']):r for r in read(mp)}
        for num,iid in enumerate(ids,1):
            path=source/'train_cache'/f'{iid}.npz' if split=='train' else ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz'
            hashes[f'{split}/{iid}']=sha(path)
            with np.load(path) as q:item={k:q[k] for k in ['proto','coeff','boxes','detections','shape','input_shape','mapping_gt','mapping_pred']}
            rows,statuses=analyze(gt,iid,item,meta,split);allrows.extend(rows);allstatuses.extend(statuses)
            if num%20==0 or num==len(ids):
                progress=dict(split=split,completed=num,total=len(ids),readout_rows=len(allrows),seconds=round(time.monotonic()-start,1))
                write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_csv(out/'readouts.csv',allrows);write_csv(out/'statuses.csv',allstatuses);write_json(out/'input_hashes.json',hashes)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,images={k:len(v) for k,v in selection.items()},
        readout_rows=len(allrows),seconds=time.monotonic()-start,hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':main()
