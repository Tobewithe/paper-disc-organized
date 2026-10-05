"""S022: local mask-head gradient interference on native640 GT; no update.

Last 1x1 layer only, frozen features/P/boxes/assignment. Same-image all-positive
mask-only first-order diagnostic, not historical minibatch/Adam/BN training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from copy import deepcopy
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from native_label_pipeline_probe import format_native,mask_of
from structure_candidate_trace import TraceCapture
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    a.out.mkdir(exist_ok=False);(a.out/'images').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prev=ROOT/'diagnostics/native_label887_20260912';assign=ROOT/'diagnostics/assignment300_20260912'
    targets=[r for r in read(prev/'metrics.csv') if r['arm']=='NATIVE_OVERLAP640_NATIVEBOX'];need(len(targets)==887,'Locked887 mismatch')
    old={int(r['annotation_id']):r for r in read(ROOT/'diagnostics/grid_support887_20260912/targets.csv') if r['arm']=='original'}
    ids=sorted({int(r['image_id']) for r in targets})
    if a.smoke:ids=sorted({min(int(r['image_id']) for r in targets if r['density']==d) for d in ['high','other']})
    targets=[r for r in targets if int(r['image_id']) in ids]
    cfg=get_cfg(overrides=json.loads((ROOT/'train_config.json').read_text()));data=check_det_dataset(str(ROOT/'coco_clean.yaml'),autodownload=False)
    ds=build_yolo_dataset(cfg,data['val'],16,data,mode='val',rect=False);dindex={int(Path(x['im_file']).stem):k for k,x in enumerate(ds.labels)}
    sidecar=json.loads((ROOT/'audits/actual_loader_ids_val2017.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);head=model.model.model[-1]
    need(not head.end2end,'Cache expects one-to-many path');features={};outputs={};linearweights={};handles=[]
    for level,branch in enumerate(head.cv4):
        need(isinstance(branch[-1],torch.nn.Conv2d) and branch[-1].kernel_size==(1,1),'Unexpected mask final layer')
        def hook(module,args,output,level=level):
            features[level]=args[0].detach().clone();outputs[level]=output.detach().clone()
            linearweights[level]=(module.weight.detach().clone(),module.bias.detach().clone())
        handles.append(branch[-1].register_forward_hook(hook))
    protocol=dict(experiment='S022',network_training=False,images=ids,target_ids=[int(r['annotation_id']) for r in targets],
        source_sha256=sha(prev/'metrics.csv'),script_sha256=sha(__file__),weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),
        head='Final cv4 1x1 Conv at strides8/16/32; frozen h, P, boxes, scores. c=W h+b verified. No parameter updates.',
        targets='Same native640 overlap labels/common input geometry as S020, same fixed target IDs. S017 all-positive assignment held fixed by annotationID, not native reassignment.',
        objective='Each assigned coefficient receives native640 GT-crop BCE/(640^2 * nativeGT area); gradients summed across same-image positive candidates at each scale, average normalization cancels in direction ratios.',
        interference='Target g_s=dL_s/dc_s, hbar=[h;1], G_u=g_u hbar_u^T. dL_s along -sumG proportional to -g_s dot (sumG hbar_s). Normalize by isolated own g norm^2 * hbar norm^2; negative total implies local increase along aggregate plain-gradient direction.',
        partitions='self anchor, other anchors of own GT, same-category GT with bbox intersection, same-category nonadjacent GT, other categories; all at same output layer.',
        limitations='Mask-only current-image frozen last-layer first-order diagnostic, not actual historical multi-image/multi-task Adam/MuSGD training, BN/backbone dynamics, AP or causal explanation. Positive assignment inherited from explicit diagnostic recipe.',
        native_helper_sha256=sha(Path(__file__).with_name('native_label_pipeline_probe.py')))
    write_json(a.out/'protocol.json',protocol);rows=[];witness=[];start=time.monotonic()
    for number,iid in enumerate(ids,1):
        cachepath=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz';rawpath=ROOT/'diagnostics/structure_main300_20260911/raw'/f'{iid}.npz';superpath=assign/'images'/f'{iid}.npz'
        with np.load(cachepath) as z:cache={k:z[k] for k in z.files}
        with np.load(rawpath) as z:raw={k:z[k] for k in z.files}
        with np.load(superpath) as z:sup={k:z[k] for k in z.files}
        prior=json.loads((prev/'images'/f'{iid}.json').read_text())
        need(sha(cachepath)==prior['input_hashes']['cache'] and sha(superpath)==prior['input_hashes']['supervision'],'Source changed')
        features.clear();outputs.clear()
        with torch.inference_mode():model.predict(str(ROOT/'data/images/val2017'/gt.imgs[iid]['file_name']),predictor=TraceCapture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
        cap=model.predictor.capture;errors={key:float(np.max(np.abs(cap[key].cpu().numpy()-cache[key]))) if cache[key].size else 0. for key in ['coeff','boxes','proto','detections']}
        need(max(errors.values())<2e-4,'Fresh feature forward cache replay failed')
        replay=torch.cat([outputs[k].flatten(2) for k in range(3)],2)[0].T.cpu().numpy()
        rawerr=float(np.max(np.abs(replay-raw['coefficients'])));need(rawerr<2e-4,'Raw head replay failed')
        hs=[];layererr=[]
        for level in range(3):
            weight,bias=linearweights[level];hmap=features[level][0].flatten(1).T
            local=F.linear(hmap,weight[:,:,0,0],bias);error=float((local-outputs[level][0].flatten(1).T).abs().max())
            need(error<2e-4,'Last linear head reconstruction failed');layererr.append(error)
            hs.append(torch.cat([hmap.double(),torch.ones((len(hmap),1),device='cuda',dtype=torch.float64)],1))
        k=dindex[iid];mapped=list(map(int,sidecar[str(iid)]['annotation_ids']));lab=deepcopy(ds.labels[k]);lab.pop('shape',None);lab['img']=cv2.imread(lab['im_file'])
        h,w=lab['img'].shape[:2];lab=ds.update_labels_info(lab);lab['ori_shape']=(h,w);lab['ratio_pad']=(1.,1.)
        common=LetterBox(new_shape=(640,640),auto=False,scaleup=True)(lab);native=format_native(common,mapped,1,True)
        proto=torch.tensor(cache['proto'],device='cuda');p640=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
        coeff=torch.tensor(raw['coefficients'],device='cuda');fg=sup['foreground'].astype(bool);tidx=sup['target_gt_index'].astype(int);annids=list(map(int,sup['annotation_ids']))
        gradients=torch.zeros_like(coeff,dtype=torch.float64);native_loss={};positive=np.flatnonzero(fg);need(len(positive)>0,'No positives')
        for g,aid in enumerate(annids):
            sources=np.flatnonzero(fg&(tidx==g))
            if not len(sources):continue
            norm=torch.tensor(native['boxes'][native['position'][aid]:native['position'][aid]+1],device='cuda');box=ops.xywh2xyxy(norm)*640;area=norm[0,2:].prod()
            support=ops.crop_mask(torch.ones((1,640,640),device='cuda'),box)[0].bool();x=p640[:,support].T.contiguous().double()
            y=torch.tensor(mask_of(native,aid),device='cuda').double()[support];factor=1./(640*640*area.double())
            for offset in range(0,len(sources),16):
                src=torch.tensor(sources[offset:offset+16],device='cuda');cc=coeff[src].double();logits=x@cc.T
                grad=(torch.sigmoid(logits)-y[:,None]).T@x*factor;gradients[src]=grad
                losses=F.binary_cross_entropy_with_logits(logits,y[:,None].expand_as(logits),reduction='none').sum(0)*factor
                native_loss.update({int(s):float(v) for s,v in zip(src.cpu().tolist(),losses.cpu().tolist())})
        offsets=[0,6400,8000,8400];image_rows=[];saved={}
        for r in [q for q in targets if int(q['image_id'])==iid]:
            aid=int(r['annotation_id']);source=int(old[aid]['source_index']);g=annids.index(aid);need(fg[source] and tidx[source]==g,'Target no longerown positive')
            level=0 if source<6400 else 1 if source<8000 else 2;local=source-offsets[level];ht=hs[level][local];gtarget=gradients[source]
            denom=float(gtarget.square().sum()*ht.square().sum());need(denom>0,'Zero target gradient')
            srcs=positive[(positive>=offsets[level])&(positive<offsets[level+1])];hc=hs[level][torch.tensor(srcs-offsets[level],device='cuda')]
            gc=gradients[torch.tensor(srcs,device='cuda')];pair=(gc@gtarget)*(hc@ht);ratios=pair/denom
            groups={name:[] for name in ['self','same_gt_other_anchor','same_class_adjacent','same_class_nonadjacent','other_class']}
            ann=gt.anns[aid];b=ann['bbox']
            for ix,src in enumerate(srcs):
                otherid=annids[tidx[src]];q=gt.anns[otherid]
                if src==source:name='self'
                elif otherid==aid:name='same_gt_other_anchor'
                elif q['category_id']!=ann['category_id']:name='other_class'
                else:
                    c=q['bbox'];inter=max(0,min(b[0]+b[2],c[0]+c[2])-max(b[0],c[0]))*max(0,min(b[1]+b[3],c[1]+c[3])-max(b[1],c[1]))
                    name='same_class_adjacent' if inter>0 else 'same_class_nonadjacent'
                groups[name].append(ix)
            shared=gc.T@hc;effective=shared@ht;total=float(gtarget@effective/denom)
            need(abs(total-float(ratios.sum()))<1e-8+1e-6*abs(total),'Kernel/weight-gradient mismatch')
            eps=1e-4;direction=-effective/effective.norm().clamp_min(1e-30)
            norm=torch.tensor(native['boxes'][native['position'][aid]:native['position'][aid]+1],device='cuda');box=ops.xywh2xyxy(norm)*640
            mask=ops.crop_mask(torch.ones((1,640,640),device='cuda'),box)[0].bool();x=p640[:,mask].T.double();y=torch.tensor(mask_of(native,aid),device='cuda').double()[mask]
            area=norm[0,2:].prod().double();c0=coeff[source].double()
            lossfn=lambda c:F.binary_cross_entropy_with_logits(x@c,y,reduction='sum')/(640*640*area)
            fd=float((lossfn(c0+eps*direction)-lossfn(c0-eps*direction))/(2*eps));analytic=float(gtarget@direction)
            need(abs(fd-analytic)<=1e-7+1e-4*abs(analytic),'Directional derivative witness failed')
            row=dict(image_id=iid,annotation_id=aid,density=r['density'],area=float(r['area']),source=source,stride=8*2**level,
                native_loss=native_loss[source],coefficient_gradient_norm=float(gtarget.norm()),feature_norm=float(ht.norm()),same_scale_positives=len(srcs),
                total_alignment_ratio=total,other_alignment_ratio=total-1.,aggregate_opposes_target=total<0,
                directional_derivative=analytic,finite_difference=fd,directional_error=abs(fd-analytic))
            for name,ix in groups.items():row[name+'_alignment_ratio']=float(ratios[ix].sum()) if ix else 0.;row[name+'_count']=len(ix)
            image_rows.append(row);saved[str(aid)]={'h':ht.cpu().numpy(),'g':gtarget.cpu().numpy(),'shared_effective':effective.cpu().numpy()}
        rows.extend(image_rows);wr=dict(image_id=iid,cache_max_abs_error=max(errors.values()),raw_max_abs_error=rawerr,last_linear_max_abs_error=max(layererr),positives=len(positive))
        witness.append(wr);write_json(a.out/'images'/f'{iid}.json',dict(rows=image_rows,witness=wr,input_hashes={'cache':sha(cachepath),'raw':sha(rawpath),'supervision':sha(superpath)}))
        np.savez_compressed(a.out/'images'/f'{iid}.npz',**{f'{aid}_{key}':v for aid,obj in saved.items() for key,v in obj.items()})
        if number%10==0 or number==len(ids):
            pr=dict(images=number,total=len(ids),targets=len(rows),seconds=time.monotonic()-start);write_json(a.out/'progress.json',pr);print(json.dumps(pr),flush=True)
    for handle in handles:handle.remove()
    save_csv(a.out/'targets.csv',rows);save_csv(a.out/'witness.csv',witness)
    summary=[]
    for density in ['high','other']:
        rr=[r for r in rows if r['density']==density]
        summary.append(dict(density=density,n=len(rr),aggregate_opposes=sum(r['aggregate_opposes_target'] for r in rr),
            adjacent_opposes=sum(r['same_class_adjacent_alignment_ratio']<0 for r in rr),has_adjacent=sum(r['same_class_adjacent_count']>0 for r in rr),
            median_total_alignment=float(np.median([r['total_alignment_ratio'] for r in rr])),median_adjacent_alignment=float(np.median([r['same_class_adjacent_alignment_ratio'] for r in rr]))))
    write_json(a.out/'ANALYSIS.json',dict(groups=summary,cache_error_max=max(r['cache_max_abs_error'] for r in witness),
        directional_derivative_error_max=max(r['directional_error'] for r in rows),scope=protocol['limitations']))
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,images=len(ids),targets=len(rows),seconds=time.monotonic()-start,
        hashes={str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()}))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
