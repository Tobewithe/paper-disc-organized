"""S020: native YOLODataset/Format label audit and controlled label substitutions.

Actual-loader targets are audited in their own geometry, never fitted to cached
features from another geometry. Controlled fitting uses native Format on the
same original-image LetterBox geometry as S019; original P/c/prediction boxes
and target identities remain fixed. Not a replay of historical training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset,polygons2masks,polygons2masks_overlap
from ultralytics.data.augment import LetterBox,Format
from ultralytics.utils import ops
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha
from crossimage_response_experiment import gt_input_regions
from grid_support_factorial_probe import solve

ARMS=['COCO_CV_LINEAR160','NATIVE_INDEPENDENT160','NATIVE_OVERLAP160','NATIVE_OVERLAP640',
      'NATIVE_OVERLAP160_NATIVEBOX','NATIVE_OVERLAP640_NATIVEBOX']

def format_native(raw,ids,ratio,overlap):
    # Call the actual native formatter; obtain only the documented order from
    # its same public overlap helper, then verify the formatter output exactly.
    inst=deepcopy(raw['instances']);h,w=raw['img'].shape[:2];inst.convert_bbox('xywh');inst.denormalize(w,h)
    if overlap:
        expected,order=polygons2masks_overlap((h,w),inst.segments,downsample_ratio=ratio)
    else:
        expected=polygons2masks((h,w),inst.segments,1,downsample_ratio=ratio);order=np.arange(len(ids))
    out=Format(bbox_format='xywh',normalize=True,return_mask=True,mask_ratio=ratio,mask_overlap=overlap,batch_idx=True,bgr=0.)(deepcopy(raw))
    actual=out['masks'].numpy()
    need(np.array_equal(actual[0] if overlap else actual,expected),'Native Format mask replay differs')
    need(np.array_equal(out['cls'].numpy(),raw['cls'][order]),'Native class/order mismatch')
    need(np.allclose(out['bboxes'].numpy(),inst.bboxes[order]/np.array([w,h,w,h]),rtol=1e-6,atol=1e-7),'Native bbox/order mismatch')
    position={int(ids[int(j)]):k for k,j in enumerate(order)}
    return dict(mask=actual,position=position,boxes=out['bboxes'].numpy(),overlap=overlap,order=np.asarray(order),output=out)

def mask_of(formatted,aid):
    k=formatted['position'][aid]
    return (formatted['mask'][0]==k+1).astype(np.uint8) if formatted['overlap'] else formatted['mask'][k]

def similarity(a,b):
    aa=a.astype(bool);bb=b.astype(bool);union=int((aa|bb).sum())
    return float((aa&bb).sum()/union) if union else 1.

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-iter',type=int,default=120);a=p.parse_args()
    a.out.mkdir(exist_ok=False);(a.out/'images').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    previous=ROOT/'diagnostics/grid_support887_20260912';assignment=ROOT/'diagnostics/assignment300_20260912'
    source=read(previous/'targets.csv');targets=[r for r in source if r['arm']=='original'];need(len(targets)==887,'Locked cohort mismatch')
    old={(int(r['annotation_id']),r['arm']):r for r in source}
    ids=sorted({int(r['image_id']) for r in targets})
    if a.smoke:
        ids=sorted({min(int(r['image_id']) for r in targets if r['density']==d) for d in ['high','other']}|
            {int(r['image_id']) for r in source if r.get('status')=='empty_support_unchanged'})
    targets=[r for r in targets if int(r['image_id']) in ids]
    cfg=get_cfg(overrides=json.loads((ROOT/'train_config.json').read_text()))
    data=check_det_dataset(str(ROOT/'coco_clean.yaml'),autodownload=False)
    ds=build_yolo_dataset(cfg,data['val'],16,data,mode='val',rect=False)
    dsindex={int(Path(x['im_file']).stem):i for i,x in enumerate(ds.labels)}
    sidecar=json.loads((ROOT/'audits/actual_loader_ids_val2017.json').read_text())
    native_source=json.loads((ROOT/'diagnostics/native_label_source_20260912/SOURCE_RECEIPT.json').read_text())
    protocol=dict(experiment='S020',network_training=False,oracle=True,images=ids,target_ids=[int(r['annotation_id']) for r in targets],
        label_arms=ARMS,source_sha256=sha(previous/'targets.csv'),script_sha256=sha(__file__),solver_sha256=sha(Path(__file__).with_name('grid_support_factorial_probe.py')),
        annotation_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),loader_idmap_sha256=sha(ROOT/'audits/actual_loader_ids_val2017.json'),
        native_source_receipt_sha256=sha(ROOT/'diagnostics/native_label_source_20260912/SOURCE_RECEIPT.json'),
        solver=dict(max_iter=a.max_iter,initialization='original coefficient',scaling='same S019 original-P RMS within raw-COCO GT160crop, all arms shared',
            loss='same S019 pure BCE, no bias/regularization/Dice, final state, no GT-IoU selection'),
        paths=dict(actual_loader='Official YOLODataset val pipeline load_image + transforms + Format, label audit only. Native geometry can differ from cached inference: never fit mismatched masks to cached P.',
            controlled='Official parsed/resampled polygons transformed by native LetterBox directly from ORIGINAL image to exact cached inference640 geometry; call native Format independent/overlap ratio4/ratio1. Not a historical training replay.',
            downsample_only='Same S019 COCO nearest-exact640 mask, replace torch nearest160 by OpenCV default uint8 INTER_LINEAR resize, no other change.'),
        fitted_support='First four arms keep S019 raw COCO GT box/area; final pair changes support+area to native segment-derived GT box. All final prediction boxes remain fixed. Assignment NOT rerun.',
        overlap='All native image instances participate, not just fitted targets. Native descending-area order mapped back to annotation IDs separately at each mask ratio.',
        restrictions='Current installed native components + clean-run receipts, not official asset original pretraining recipe. Fixed explored failures, GT same-image oracle, not method AP/causal history.',
        module_hashes=native_source['module_files'])
    write_json(a.out/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    labels=[];metrics=[];witness=[];start=time.monotonic()
    for num,iid in enumerate(ids,1):
        k=dsindex[iid];mapped=list(map(int,sidecar[str(iid)]['annotation_ids']));lab=ds.labels[k]
        need(sidecar[str(iid)]['index']==k,'Dataset index changed')
        need(len(mapped)==len(lab['cls'])==len(set(mapped)),'Native IDs/rows mismatch')
        textpath=ROOT/'official_yolo/labels/val2017'/f'{iid:012d}.txt'
        need(textpath.exists(),'Missing native label text')
        actualraw=ds.get_image_and_label(k)
        for transform in ds.transforms.transforms[:-1]:actualraw=transform(actualraw)
        nativeactual=format_native(actualraw,mapped,4,True)
        actualout=ds[k]
        need(torch.equal(actualout['img'],nativeactual['output']['img']) and torch.equal(actualout['masks'],nativeactual['output']['masks']), 'Actual ds[k] differs from inspected path')
        need(torch.equal(actualout['bboxes'],nativeactual['output']['bboxes']) and torch.equal(actualout['cls'],nativeactual['output']['cls']),'Actual ds[k] target ordering differs')
        # Reuse parsed/resampled labels but avoid the loader's pre-resize so the
        # controlled targets use the same geometry as the frozen inference cache.
        nativeinput=deepcopy(lab);nativeinput.pop('shape',None)
        nativeinput['img']=cv2.imread(lab['im_file']);need(nativeinput['img'] is not None,'Missing image')
        h,w=nativeinput['img'].shape[:2]
        nativeinput=ds.update_labels_info(nativeinput)
        nativeinput['ori_shape']=(h,w);nativeinput['ratio_pad']=(1.,1.)
        common=LetterBox(new_shape=(640,640),auto=False,scaleup=True)(nativeinput)
        f160=format_native(common,mapped,4,False);o160=format_native(common,mapped,4,True);o640=format_native(common,mapped,1,True)
        cachepath=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz';superpath=assignment/'images'/f'{iid}.npz'
        with np.load(cachepath) as z:item={key:z[key] for key in z.files}
        with np.load(superpath) as z:sup={key:z[key] for key in z.files}
        oldimage=json.loads((previous/'images'/f'{iid}.json').read_text())
        need(sha(cachepath)==oldimage['input_hashes']['cache'] and sha(superpath)==oldimage['input_hashes']['supervision'],'Input changed')
        need(set(map(int,sup['annotation_ids']))==set(mapped),'Native loader lost/added ordinary annotation')
        gindex={int(aid):g for g,aid in enumerate(sup['annotation_ids'])};sourceindex={int(v):j for j,v in enumerate(sup['original_final_sources'])}
        proto=torch.tensor(item['proto'],device='cuda');p640=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
        raster,_,_=gt_input_regions(gt,iid,item)
        gain=min(640/h,640/w);nh,nw=round(h*gain),round(w*gain);top,left=round((640-nh)/2-.1),round((640-nw)/2-.1)
        need(np.allclose(common['ratio_pad'][0],(gain,gain)) and tuple(common['ratio_pad'][1])==(left,top),'Common LetterBox differs from S019')
        actual_ratio=actualraw['ratio_pad'];image_delta=int(np.count_nonzero(nativeactual['output']['img'].numpy()!=f160['output']['img'].numpy()))
        witness.append(dict(image_id=iid,native_instances=len(mapped),actual_ds_replayed=True,geometry_pixel_channels_different=image_delta,
            actual_ratio=actual_ratio,common_ratio=common['ratio_pad'],native_label_sha256=sha(textpath),all_annotation_ids_retained=True))
        params={};image_labels=[];image_metrics=[]
        for r in [q for q in targets if int(q['image_id'])==iid]:
            aid=int(r['annotation_id']);ann=gt.anns[aid];g=gindex[aid];j=sourceindex[int(r['source_index'])]
            c0=torch.tensor(item['coeff'][j],device='cuda');predbox=torch.tensor(item['boxes'][j:j+1],device='cuda')
            norm=torch.tensor(sup['gt_boxes_normalized'][g:g+1],device='cuda');rawbox=ops.xywh2xyxy(norm)*640;area=norm[0,2:].prod()
            support=ops.crop_mask(torch.ones((1,160,160),device='cuda'),rawbox/4)[0].bool()
            scale=proto[:,support].square().mean(1).sqrt().clamp_min(.01) if bool(support.any()) else proto.square().mean((1,2)).sqrt().clamp_min(.01)
            y640=raster[aid].cpu().numpy().astype(np.uint8);old160=sup['gt_masks160'][g]
            need(np.array_equal(F.interpolate(torch.tensor(y640)[None,None].float(),(160,160),mode='nearest')[0,0].numpy(),old160),'Prior nearest target changed')
            linear=cv2.resize(y640,(160,160));indep=mask_of(f160,aid);overlap=mask_of(o160,aid);large=mask_of(o640,aid)
            actualmask=mask_of(nativeactual,aid);nativeboxnorm=torch.tensor(o160['boxes'][o160['position'][aid]:o160['position'][aid]+1],device='cuda')
            nb=ops.xywh2xyxy(nativeboxnorm)*640;na=nativeboxnorm[0,2:].prod();need(float(na)>0,'Native nonpositive box')
            # Overlap labels at each ratio have different area order; retrieve by
            # original annotation ID rather than assuming slot order is shared.
            nativebox640=o640['boxes'][o640['position'][aid]]
            need(np.allclose(nativeboxnorm[0].cpu().numpy(),nativebox640,rtol=1e-6,atol=1e-7),'Native box changed with mask ratio')
            ld=dict(image_id=iid,annotation_id=aid,density=r['density'],area=float(r['area']),category_id=ann['category_id'],parts=len(ann['segmentation']),
                old_nearest_pixels=int(old160.sum()),coco_linear_pixels=int(linear.sum()),native_independent_pixels=int(indep.sum()),native_overlap_pixels=int(overlap.sum()),actual_loader_pixels=int(actualmask.sum()),
                old_vs_linear_iou=similarity(old160,linear),old_vs_native_independent_iou=similarity(old160,indep),old_vs_native_overlap_iou=similarity(old160,overlap),
                native_independent_vs_overlap_iou=similarity(indep,overlap),native_overlap_loss_pixels=int(((indep>0)&(overlap==0)).sum()),
                nativebox_max_abs_input_px=float((nb-rawbox).abs().max()),actual_vs_common_label_xor=int((actualmask!=overlap).sum()),
                actual_image_diff=image_delta,gt640_pixels=int(y640.sum()),native640_pixels=int(large.sum()))
            image_labels.append(ld)
            specs={'COCO_CV_LINEAR160':(linear,rawbox,area),'NATIVE_INDEPENDENT160':(indep,rawbox,area),
                'NATIVE_OVERLAP160':(overlap,rawbox,area),'NATIVE_OVERLAP640':(large,rawbox,area),
                'NATIVE_OVERLAP160_NATIVEBOX':(overlap,nb,na),'NATIVE_OVERLAP640_NATIVEBOX':(large,nb,na)}
            truth=gt.annToRLE(ann);shape=tuple(map(int,item['shape']))
            for arm,(label,lossbox,lossarea) in specs.items():
                size=label.shape[-1];pp=proto if size==160 else p640;yy=torch.tensor(label,device='cuda').float()
                cc,details=solve(pp,yy,lossbox*(size/640),lossarea,c0,scale,a.max_iter)
                binary=ops.process_mask(proto,cc[None],predbox,(640,640),upsample=True)[0]
                original=(ops.scale_masks(binary[None,None],shape)[0,0]>.5).cpu().numpy().astype(np.uint8)
                value=float(mu.iou([mu.encode(np.asfortranarray(original))],[truth],[0])[0,0])
                image_metrics.append(dict(image_id=iid,annotation_id=aid,density=r['density'],area=float(r['area']),parts=ld['parts'],arm=arm,coco_iou=value,
                    original_iou=float(r['coco_iou']),s019_A_iou=float(old[(aid,'A160_GT')]['coco_iou']),s019_C_iou=float(old[(aid,'C640_GT')]['coco_iou']),
                    nativebox_max_abs_input_px=ld['nativebox_max_abs_input_px'],**details))
                params[f'{aid}_{arm}']=cc.cpu().numpy()
            params[f'{aid}_old160']=old160;params[f'{aid}_linear160']=linear;params[f'{aid}_native_independent160']=indep
            params[f'{aid}_native_overlap160']=overlap;params[f'{aid}_actual_native160']=actualmask
        labels.extend(image_labels);metrics.extend(image_metrics)
        np.savez_compressed(a.out/'images'/f'{iid}.npz',**params)
        write_json(a.out/'images'/f'{iid}.json',dict(image_id=iid,labels=image_labels,metrics=image_metrics,witness=witness[-1],
            input_hashes=dict(cache=sha(cachepath),supervision=sha(superpath),previous=sha(previous/'images'/f'{iid}.json'))))
        if num%5==0 or num==len(ids):
            progress=dict(images=num,total=len(ids),targets=len(labels),seconds=time.monotonic()-start);write_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    save_csv(a.out/'labels.csv',labels);save_csv(a.out/'metrics.csv',metrics);save_csv(a.out/'witness.csv',witness)
    summary=[]
    for density in ['high','other']:
        for arm in ARMS:
            rr=[r for r in metrics if r['density']==density and r['arm']==arm]
            if rr:summary.append(dict(density=density,arm=arm,n=len(rr),mean_coco_iou=100*float(np.mean([r['coco_iou'] for r in rr])),
                recovered75=sum(r['coco_iou']>=.75 for r in rr),worse_original=sum(r['coco_iou']<r['original_iou']-1e-6 for r in rr),
                iteration_limit=sum(r['iterations']>=a.max_iter for r in rr),empty_support=sum(r['status']=='empty_support_unchanged' for r in rr)))
    write_json(a.out/'ANALYSIS.json',dict(groups=summary,labels=len(labels),images=len(ids),geometry_different_images=sum(r['geometry_pixel_channels_different']>0 for r in witness),
        all_native_dataset_items_replayed=True,all_annotation_id_sets_match=True,scope=protocol['restrictions']))
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,images=len(ids),targets=len(labels),seconds=time.monotonic()-start,
        hashes={str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()}))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
