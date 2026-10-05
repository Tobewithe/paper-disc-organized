"""S047: fixed-target neighbor/background image interventions, no training.

Selection uses GT geometry before predictions. All failures and harmful changes
remain in the denominator. Saved input edits are deterministic non-generative
controls, not claims of natural counterfactual images or COCO method AP.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'4')
import argparse, contextlib, csv, hashlib, io, json, shutil, time, traceback, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops, nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from structure_candidate_trace import TraceCapture
from frozen_mechanism_probe import ownership, box_iou, sha
from eval_readout_input_pilot import ici

ROOT=Path(__file__).resolve().parent
ANNOTATION=ROOT.parent.parent/'datasets/coco/annotations/instances_train2017.json'
ARCHIVE=Path('C:/Dpan/document/model_datasets/datasets/coco/downloads/train2017.zip')
WEIGHT=ROOT/'weights/yolo26m-seg.pt'
GROUPS={'same_near':32,'different_near':16,'same_far':16}


def rank(*args):
    return hashlib.sha256(('S047:20260912:'+':'.join(map(str,args))).encode()).hexdigest()


def dump(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def csvsave(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def rng(*args):return np.random.default_rng(int(rank(*args)[:16],16))


def exclude_history():
    ids=set();sources={}
    # Conservative image-ID union: also excludes val IDs found in mixed records.
    # Only image-ID fields are read, never annotation IDs or generic numeric data.
    keys={'image_id','image_ids','images','train','fit','transfer','evaluation',
          'calibration','development','excluded_explored_pilot96'}
    def walk(obj,key=''):
        if isinstance(obj,dict):
            for k,v in obj.items():walk(v,k)
        elif isinstance(obj,list):
            if key in keys and all(isinstance(v,int) for v in obj):ids.update(obj)
            else:
                for v in obj:
                    if isinstance(v,dict):walk(v)
        elif key=='image_id' and isinstance(obj,int):ids.add(obj)
    for p in sorted((ROOT/'diagnostics').rglob('selection.json')):
        walk(json.loads(p.read_text(encoding='utf8')));sources[str(p.relative_to(ROOT))]=sha(p)
    for p in [ROOT/'relative_ownership_protocol.json',ROOT/'crossimage_response_protocol.json',
              ROOT/'failure_locality_protocol.json']:
        if p.exists():walk(json.loads(p.read_text(encoding='utf8')));sources[str(p.relative_to(ROOT))]=sha(p)
    # Existing materialized train images were all used in the local readout rounds.
    ids.update(int(p.stem) for p in (ROOT/'local_readout_runtime_20260912/data/images/train2017').glob('*.jpg'))
    return ids,sources


def gt_geometry(gt,iid,a,b):
    shape=(gt.imgs[iid]['height'],gt.imgs[iid]['width'])
    masks={q['id']:gt.annToMask(q).astype(bool) for q in gt.imgToAnns[iid]}
    own=masks[a['id']];other=np.zeros(shape,bool);union=np.zeros(shape,bool);crowd=np.zeros(shape,bool)
    for q in gt.imgToAnns[iid]:
        m=masks[q['id']]
        if q['id']!=b['id']:other|=m
        union|=m
        if q.get('iscrowd',0):crowd|=m
    protected=cv2.dilate(other.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    editable=masks[b['id']]&~protected
    background=~cv2.dilate(union.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    return own,masks[b['id']],editable,union,crowd,background,masks


def placements(editable,background,own,key):
    ys,xs=np.nonzero(editable)
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    template=editable[y0:y1,x0:x1].astype(np.float32);area=len(ys)
    score=cv2.matchTemplate(background.astype(np.float32),template,cv2.TM_CCORR)
    yy,xx=np.nonzero(score>=area-.25)
    if not len(yy):return dict(control=None,donors=[],candidates=0)
    if len(yy)>512:
        take=rng('placements',key).choice(len(yy),512,replace=False);yy,xx=yy[take],xx[take]
    offsets=np.c_[ys-y0,xs-x0];valid=[]
    ty,tx=np.nonzero(own);center=np.array([ty.mean(),tx.mean()])
    original_center=np.array([ys.mean(),xs.mean()]);original_distance=float(np.linalg.norm(original_center-center))
    for y,x in zip(yy,xx):
        coord=offsets+[y,x]
        if background[coord[:,0],coord[:,1]].all():
            radial=abs(float(np.linalg.norm(coord.mean(0)-center))-original_distance)
            valid.append((radial,int(y),int(x)))
    valid.sort()
    if not valid:return dict(control=None,donors=[],candidates=0)
    control=valid[0];pool=[v for v in valid if np.hypot(v[1]-control[1],v[2]-control[2])>=8]
    if not pool:return dict(control=None,donors=[],candidates=len(valid))
    chosen=rng('donors',key).choice(len(pool),3,replace=len(pool)<3)
    donors=[[pool[int(k)][1]-int(y0),pool[int(k)][2]-int(x0)] for k in chosen]
    return dict(control=[control[1]-int(y0),control[2]-int(x0)],donors=donors,
                candidates=len(valid),radial_distance_mismatch=control[0],
                original_distance=original_distance,shape_bbox=[int(y0),int(x0),int(y1),int(x1)])


def prepare(gt,out):
    excluded,sources=exclude_history();dump(out/'excluded_images.json',dict(image_ids=sorted(excluded),sources=sources))
    groups=defaultdict(list);examined=Counter();used=set();rejected=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        available={int(Path(n).stem) for n in archive.namelist() if n.lower().endswith('.jpg')}
        candidates=sorted(set(gt.imgs)&available-excluded,key=lambda i:rank('image',i))
        for iid in candidates:
            if all(len(groups[g])>=n for g,n in GROUPS.items()):break
            if iid in used:continue
            anns=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0) and q.get('area',0)>=128 and q['bbox'][2]>0 and q['bbox'][3]>0]
            pairs=defaultdict(list)
            for a in anns:
                for b in anns:
                    if a['id']==b['id']:continue
                    overlap=box_iou(a['bbox'],b['bbox']);same=a['category_id']==b['category_id']
                    ca=np.array(a['bbox'][:2])+np.array(a['bbox'][2:])/2;cb=np.array(b['bbox'][:2])+np.array(b['bbox'][2:])/2
                    distance=float(np.linalg.norm(ca-cb)/max(np.hypot(*a['bbox'][2:]),1))
                    group='same_near' if same and overlap>.05 else 'different_near' if not same and overlap>.05 else 'same_far' if same and overlap==0 and distance>=1.5 else None
                    if group and len(groups[group])<GROUPS[group]:pairs[group].append((rank('pair',iid,a['id'],b['id']),a,b,overlap,distance))
            for group in GROUPS:
                if not pairs[group]:continue
                # Exactly the first GT-hash pair per image/group; no alternative
                # target picked if this pair has poor editable/background support.
                _,a,b,overlap,distance=min(pairs[group],key=lambda v:v[0]);examined[group]+=1
                own,neighbor,edit,union,crowd,bg,masks=gt_geometry(gt,iid,a,b)
                if own.sum()<128 or edit.sum()<64 or edit.sum()/max(neighbor.sum(),1)<.2 or bg.sum()<64:
                    rejected.append(dict(image_id=iid,group=group,target=a['id'],neighbor=b['id'],reason='GT_edit_feasibility'));continue
                place=placements(edit,bg,own,(iid,a['id'],b['id']))
                # Keep missing-controls examples, but mark their paired analysis unavailable.
                ys,xs=np.nonzero(edit);nys,nxs=np.nonzero(neighbor)
                mindist=float(cv2.distanceTransform((~own).astype('uint8'),cv2.DIST_L2,5)[neighbor].min())
                item=dict(image_id=iid,target=a['id'],neighbor=b['id'],relation=group,ici=ici(a,[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)]),
                    category=a['category_id'],area=a['area'],gt_box_iou=overlap,normalized_center_distance=distance,
                    min_mask_distance=mindist,edit_pixels=int(edit.sum()),neighbor_pixels=int(neighbor.sum()),
                    edited_neighbor_fraction=float(edit.sum()/neighbor.sum()),placement=place)
                groups[group].append(item);used.add(iid)
                blob=archive.read('train2017/'+gt.imgs[iid]['file_name']);(out/'images'/gt.imgs[iid]['file_name']).write_bytes(blob)
                item['jpeg_sha256']=hashlib.sha256(blob).hexdigest();break
    chosen=sorted([q for v in groups.values() for q in v],key=lambda q:rank('image',q['image_id']))
    dump(out/'manifest.json',dict(status='FROZEN_BEFORE_INFERENCE',pairs=chosen,group_counts={g:len(groups[g]) for g in GROUPS},
        available_archive_images=len(available),excluded=len(excluded),examined=dict(examined),rejected=rejected,
        selection='GT geometry/hash before all predictions; one retained pair/image; GT editable>=64 and >=20%neighbor; no success/failure selection.',
        limits='Exclusion covers recorded selection files and materialized local train images, not proof of complete historical non-use. Official pretrained model saw train2017.'))
    if len(chosen)!=sum(GROUPS.values()):raise RuntimeError('Insufficient eligible image pool')
    return chosen


class InputCapture(TraceCapture):
    def preprocess(self,im):
        x=super().preprocess(im);self.input_tensor=x.detach().clone();return x


def encode(mask):
    r=mu.encode(np.asfortranarray(mask.astype('uint8')));r['counts']=r['counts'].decode('ascii');return r


def official_target(gt,iid,aid,bid,det,masks):
    cat=gt.anns[aid]['category_id'];categories=sorted(gt.cats)
    pp=[];indices=[]
    for j,q in enumerate(det.cpu().numpy()):
        if categories[int(q[5])]!=cat:continue
        pp.append(dict(image_id=iid,category_id=cat,score=float(q[4]),bbox=[float(q[0]),float(q[1]),float(q[2]-q[0]),float(q[3]-q[1])],segmentation=encode(masks[j])))
        indices.append(j)
    with contextlib.redirect_stdout(io.StringIO()):
        local=COCO();local.dataset=dict(info={},images=[gt.imgs[iid]],categories=list(gt.cats.values()),
            annotations=[q for q in gt.imgToAnns[iid] if q['category_id']==cat and q['id']!=bid]);local.createIndex()
        if pp:dt=local.loadRes(pp)
        else:
            dt=COCO();dt.dataset=dict(images=local.dataset['images'],categories=local.dataset['categories'],annotations=[]);dt.createIndex()
        result={}
        for task in ['bbox','segm']:
            ev=COCOeval(local,dt,task);ev.params.imgIds=[iid];ev.params.catIds=[cat];ev.params.iouThrs=np.array([.5,.75]);ev.params.areaRng=[[0,1e10]];ev.params.areaRngLbl=['all'];ev.params.maxDets=[100];ev.evaluate()
            hit50=hit75=False;match=-1
            for rec in ev.evalImgs:
                if rec is None or aid not in rec['gtIds']:continue
                k=rec['gtIds'].index(aid);hit50=bool(rec['gtMatches'][0,k]);hit75=bool(rec['gtMatches'][1,k]);match=indices[int(rec['gtMatches'][0,k])-1] if hit50 else -1
            result[task+'50']=hit50;result[task+'75']=hit75;result[task+'_pred50']=match
    return result


def scalar_iou(box,ann):
    b=np.asarray(box);return box_iou([b[0],b[1],b[2]-b[0],b[3]-b[1]],ann['bbox'])


def spatial(mask,own,neighbor,union,crowd):
    valid=~crowd;area=int((own&valid).sum());rawunion=int((mask|own).sum())
    return dict(mask_iou=float((mask&own).sum()/max(rawunion,1)),coverage=float((mask&own&valid).sum()/max(area,1)),
        old_neighbor_error=float((mask&neighbor&~own&valid).sum()/max(area,1)),
        background_error=float((mask&~union&valid).sum()/max(area,1)),pred_area=int(mask.sum()))


def decode(z,box,shape):
    m=ops.crop_mask((z>0).byte()[None],box[None]);return (ops.scale_masks(m[:,None],shape)[0,0]>.5).cpu().numpy()


def region_token(feature,mask):
    m=torch.as_tensor(mask,device='cuda',dtype=torch.float32)[None,None]
    m=F.interpolate(m,feature.shape[-2:],mode='nearest-exact')[0,0].bool()
    return feature[0,:,m].mean(1).cpu().numpy() if m.any() else np.zeros(feature.shape[1],np.float32)


def inferred(model,image,gt,iid,aid,bid,maps):
    maps.clear()
    with torch.inference_mode():
        model.predict(image,predictor=InputCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,half=False,retina_masks=False,device=0,verbose=False)
        pred=model.predictor;cap=pred.capture;raw=pred.dense;pre=pred.before_empty
        nn,ix=nms.non_max_suppression(raw.clone(),conf_thres=.001,iou_thres=.7,nc=80,max_det=8400,return_idxs=True,end2end=False)
        if not torch.equal(nn[0][:300],pre):raise RuntimeError('NMS source identity mismatch')
        mm=ops.process_mask(cap['proto'],pre[:,6:],pre[:,:4],(640,640),upsample=True)
        keep=ix[0].flatten()[:300][mm.flatten(1).any(1).bool()].long()
        if not torch.equal(raw[0,84:,keep].T,cap['coeff']):raise RuntimeError('Coefficient identity mismatch')
        masks=ops.process_mask(cap['proto'],cap['coeff'],cap['boxes'],(640,640),upsample=True) if len(keep) else raw.new_empty((0,640,640))
        restored=(ops.scale_masks(masks[:,None],cap['shape'])[:,0]>.5).cpu().numpy() if len(keep) else np.empty((0,*cap['shape']),bool)
        normal=official_target(gt,iid,aid,bid,cap['detections'],restored)
        normal['box_iou']=scalar_iou(cap['detections'][normal['bbox_pred50'],:4].cpu().numpy(),gt.anns[aid]) if normal['bbox_pred50']>=0 else 0.
        normal['mask_iou_at_box_match']=float((restored[normal['bbox_pred50']]&gt.annToMask(gt.anns[aid]).astype(bool)).sum()/max((restored[normal['bbox_pred50']]|gt.annToMask(gt.anns[aid]).astype(bool)).sum(),1)) if normal['bbox_pred50']>=0 else 0.
        return dict(raw=raw,cap=cap,keep=keep,input=pred.input_tensor,maps=dict(maps),normal=normal,
                    masks=restored,mapping=ownership(gt,iid,cap['detections']))


def run(out,gt,pairs):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(WEIGHT));model.model.eval().requires_grad_(False);head=model.model.model[-1]
    if head.end2end:raise RuntimeError('Unexpected end2end branch')
    maps={};hooks=[]
    for index in [4,6,10,16,19,22]:
        def hook(module,args,output,index=index):maps['layer'+str(index)]=output.detach().clone()
        hooks.append(model.model.model[index].register_forward_hook(hook))
    for level,branch in enumerate(head.cv4):
        def hook(module,args,output,level=level):maps['h'+str(level)]=args[0].detach().clone()
        hooks.append(branch[-1].register_forward_hook(hook))
    started=time.monotonic();rows=[];witness=[]
    for number,item in enumerate(pairs,1):
        iid,aid,bid=item['image_id'],item['target'],item['neighbor'];folder=out/'pairs'/str(iid);folder.mkdir(exist_ok=False)
        image=cv2.imread(str(out/'images'/gt.imgs[iid]['file_name']));shape=image.shape[:2]
        own,neighbor,edit,union,crowd,bg,masks=gt_geometry(gt,iid,gt.anns[aid],gt.anns[bid]);ys,xs=np.nonzero(edit)
        base=inferred(model,image,gt,iid,aid,bid,maps);j=base['mapping'].get(aid,-1)
        source=int(base['keep'][j]) if j>=0 else -1
        gain=min(640/shape[0],640/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top,left=round((640-rh)/2-.1),round((640-rw)/2-.1)
        own640=np.zeros((640,640),bool);own640[top:top+rh,left:left+rw]=cv2.resize(own.astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
        c0=base['raw'][0,84:,source] if source>=0 else None
        p0=base['cap']['proto'];b0=ops.xywh2xyxy(base['raw'][0,:4,source][None])[0] if source>=0 else None
        z0=F.interpolate((c0@p0.flatten(1)).reshape(1,1,*p0.shape[-2:]),(640,640),mode='bilinear',align_corners=False)[0,0] if source>=0 else None
        original_metrics=spatial(decode(z0,b0,shape),own,neighbor,union,crowd) if source>=0 else {}
        original_box=scalar_iou(base['cap']['detections'][j,:4].cpu().numpy(),gt.anns[aid]) if j>=0 else 0.
        status='no_original_match' if j<0 else 'good_box_bad_mask' if original_box>=.75 and original_metrics['mask_iou']<.75 else 'good_both' if original_box>=.75 else 'bad_box'
        tokens0={k:region_token(v,own640) for k,v in base['maps'].items()}
        h0=torch.cat([base['maps']['h'+str(k)][0].flatten(1).T for k in range(3)])[source] if source>=0 else None
        np.savez_compressed(folder/'original.npz',own=own,neighbor=neighbor,edit=edit,background=bg,source_index=source,
            proto=p0.cpu().numpy(),coeff=c0.cpu().numpy() if c0 is not None else np.empty(0),
            **{k:v for k,v in tokens0.items()})
        base_info=dict(image_id=iid,target=aid,neighbor=bid,relation=item['relation'],ici=item['ici'],status=status,
            original_source=source,original_box_iou=original_box,original_mask_iou=original_metrics.get('mask_iou'),
            original_box90=original_box>=.9,edit_fraction=item['edited_neighbor_fraction'],control_available=item['placement']['control'] is not None)
        variants=[('original','none',-1,image),('noop','none',-1,image.copy())]
        place=item['placement'];control=None
        if place['control'] is not None:
            shift=np.array(place['control']);control=np.c_[ys,xs]+shift
        # Both paired locations receive exactly the same fill values per seed/type.
        ring=cv2.dilate(edit.astype('uint8'),np.ones((61,61),np.uint8)).astype(bool)&bg
        colors=image[ring] if ring.sum()>=32 else image[bg]
        mean=np.median(colors,axis=0);std=np.std(colors.astype(float),axis=0)
        for seed in range(3):
            for fill in ['texture','local_color']:
                if fill=='texture':
                    if not place['donors']:continue
                    shift=np.array(place['donors'][seed]);values=image[ys+shift[0],xs+shift[1]]
                else:
                    noise=rng('color',iid,seed).normal(size=(*shape,3)).astype(np.float32)
                    noise=cv2.GaussianBlur(noise,(0,0),4)
                    values=np.clip(mean+noise[ys,xs]*std*.25,0,255).astype('uint8')
                changed=image.copy();changed[ys,xs]=values;variants.append(('neighbor',fill,seed,changed))
                if control is not None:
                    changed=image.copy();changed[control[:,0],control[:,1]]=values;variants.append(('background_control',fill,seed,changed))
        tensor_records={};pairrows=[]
        for mode,fill,seed,changed in variants:
            tag=f'{mode}_{fill}_{seed}'
            diff=np.any(changed!=image,axis=2)
            if (diff&own).any():raise RuntimeError('Own original pixels changed')
            if mode=='neighbor' and (diff&~edit).any():raise RuntimeError('Out of edit support')
            current=base if mode=='original' else inferred(model,changed,gt,iid,aid,bid,maps)
            own_difference=float((current['input']-base['input']).abs()[0,:,own640].max())
            if own_difference!=0:raise RuntimeError('Own model input pixels changed')
            raw=current['raw'];cap=current['cap'];rowbase=dict(**base_info,mode=mode,fill=fill,fill_seed=seed,
                modified_pixels=int(diff.sum()),own_input_max_difference=own_difference,
                **{'normal_'+k:v for k,v in current['normal'].items()})
            if mode=='noop':
                if not torch.equal(raw,base['raw']) or not torch.equal(cap['proto'],p0):raise RuntimeError('No-op inference did not replay')
            if mode not in ['original','noop']:
                cv2.imwrite(str(folder/(tag+'.png')),changed)
            if source>=0:
                c=raw[0,84:,source];p=cap['proto'];b=ops.xywh2xyxy(raw[0,:4,source][None])[0]
                zl=(c@p.flatten(1)).reshape(*p.shape[-2:]);z=F.interpolate(zl[None,None],(640,640),mode='bilinear',align_corners=False)[0,0]
                official=ops.process_mask(p,c[None],b[None],(640,640),upsample=True)
                if not torch.equal(ops.crop_mask((z>0).byte()[None],b[None]),official):raise RuntimeError('Factorial decoder mismatch')
                h=torch.cat([current['maps']['h'+str(k)][0].flatten(1).T for k in range(3)])[source]
                cls=int(raw[0,4:84,source].argmax());label=sorted(gt.cats).index(gt.anns[aid]['category_id'])
                transformed_box=ops.scale_boxes((640,640),b[None].clone(),shape)[0]
                rowbase.update(source_kept=bool((current['keep']==source).any()),source_correct_argmax=cls==label,
                    source_true_class_score=float(raw[0,4+label,source]),source_box_iou=scalar_iou(transformed_box.cpu().numpy(),gt.anns[aid]),
                    coefficient_relative_change=float((c-c0).norm()/c0.norm().clamp_min(1e-12)),
                    coefficient_cosine=float(F.cosine_similarity(c,c0,dim=0)),
                    head_input_relative_change=float((h-h0).norm()/h0.norm().clamp_min(1e-12)),
                    prototype_relative_change=float((p-p0).norm()/p0.norm().clamp_min(1e-12)))
                tensor_records[tag+'_logits160']=zl.cpu().numpy();tensor_records[tag+'_box640']=b.cpu().numpy();tensor_records[tag+'_coeff']=c.cpu().numpy();tensor_records[tag+'_head_input']=h.cpu().numpy()
                for k,v in current['maps'].items():
                    token=region_token(v,own640);tensor_records[tag+'_'+k]=token
                    rowbase[k+'_own_relative_change']=float(np.linalg.norm(token-tokens0[k])/max(np.linalg.norm(tokens0[k]),1e-12))
                for name,zz,bb in [('z0_b0',z0,b0),('z0_b1',z0,b),('z1_b0',z,b0),('z1_b1',z,b)]:
                    m=decode(zz,bb,shape);row=dict(**rowbase,combination=name,**spatial(m,own,neighbor,union,crowd))
                    pairrows.append(row)
            else:pairrows.append(dict(**rowbase,combination='no_source'))
        np.savez_compressed(folder/'responses.npz',**tensor_records)
        csvsave(folder/'metrics.csv',pairrows);rows.extend(pairrows)
        witness.append(dict(image_id=iid,predictions=len(base['keep']),source=source,status=status,
            own_original_unchanged=True,own_input_unchanged=True,noop_exact=True,decoder_exact=True,variants=len(variants)))
        dump(out/'progress.json',dict(stage='INFERENCE',completed=number,total=len(pairs),seconds=round(time.monotonic()-started,2),pid=os.getpid(),rows=len(rows)))
        print(json.dumps(dict(completed=number,total=len(pairs),relation=item['relation'],status=status,seconds=round(time.monotonic()-started,1))),flush=True)
        del base,current;torch.cuda.empty_cache()
    for hook in hooks:hook.remove()
    csvsave(out/'metrics.csv',rows);dump(out/'WITNESS.json',dict(images=witness,scope='Actual input invariance, NMS source identity, zero edit and decoder replay; no independent GPU repeat.'))
    dump(out/'COMPLETE.json',dict(status='COMPLETE',images=len(pairs),seconds=time.monotonic()-started,rows=len(rows),
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['COMPLETE.json']}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve()
    out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'pairs').mkdir()
    protocol=dict(experiment='S047_NEIGHBOR_BACKGROUND',training=False,groups=GROUPS,seeds=[0,1,2],seed_kind='fill randomness, not training',
        model=str(WEIGHT),weight_sha256=sha(WEIGHT),annotation=str(ANNOTATION),annotation_sha256=sha(ANNOTATION),archive=str(ARCHIVE),
        script_sha256=sha(__file__),torch=torch.__version__,ultralytics=ultralytics.__version__,vendor=ultralytics.__file__,
        inference='imgsz640 rectFalse FP32 TF32off conf.001 classNMS.7 max_det300; stock nonempty mask filtering; fixed original bbox50 ownership for source probes.',
        selection='Hash GT pairs from complete archive, exclude recorded historical image lists; same_near32,different_near16,same_far16, one pair/image; near bboxIoU>.05; far bboxIoU0 and center distance>=1.5 targetboxdiagonal. Minimum own128pixels,edit64pixels,edit>=.2neighbor. Preserve uneditable/missing-control accounting.',
        edit='Exclusive neighbor pixels outside dilated2px every other GT and crowd. Texture from exact-shape fully outside dilated all GT; local_color from surrounding background median plus seeded smooth weak color variation. Identical fill values at neighbor and translated same-shape background control. Own pixels must match exactly before AND after preprocessing.',
        controls='Background template translations match shape/area; choose minimal target radial-distance mismatch from <=512 deterministic candidates. Report mismatch and missing exact placements; no response-based reselection. Texture unavailable without valid donor/control placements; color intervention still reported.',
        matching='Official COCO bbox/segm one-to-one at.5/.75 for target category, all other GT retained except selected neighbor B excluded consistently for ALL images/arms. Report only preserved A. Original source ownership uses original complete GT bbox50. No edited-image full COCO AP.',
        factorial='z0/b0,z0/b1,z1/b0,z1/b1 at SAME raw source; stock decode. z1 changes P,c,upstream jointly, not coefficient-only intervention. Full normal predictions/matches reported separately.',
        regions='Original A/neighbor/background regions and denominator stay fixed even after neighbor removed. Coverage and errors exclude crowd; full IoU uses original full COCO A mask.',
        analysis='Primary fixed-original goodBox>=.75 badMask<.75 subgroup, alsoall original statuses; average fillseed deltas before imagecluster bootstrap2000. Neighbor-minus-background paired at same fill/seed; filltypes separate. No newGT method or causal natural-scene claim; geometric matched crossrelation analysis only on common support.',
        stop='One64image pilot; no effect-dependent image addition or fill tuning. Local feature/coeff changes descriptive only. Module patch requires separate declared follow-up if meaningful controlled response effect.',
        plan_sha256=sha(ROOT.parent.parent/'refine-logs/coco-structure/NEIGHBOR_BACKGROUND_INTERVENTION_PLAN_20260912.md'))
    dump(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANNOTATION))
    pairs=prepare(gt,out);print(json.dumps(dict(stage='SELECTED',groups=dict(Counter(p['relation'] for p in pairs)),controls=sum(p['placement']['control'] is not None for p in pairs))),flush=True)
    run(out,gt,pairs)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
