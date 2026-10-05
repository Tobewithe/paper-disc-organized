"""Real official YOLODataset audit; no custom data path used for training."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import segments2boxes

ROOT=Path(__file__).resolve().parent

def save(name, data):
    (ROOT/'audits'/name).write_text(json.dumps(data,indent=2))

def audit_dataset(split, cfg, data):
    mode='train' if split=='train2017' else 'val'
    ds=build_yolo_dataset(cfg,data[mode],16,data,mode=mode,rect=False)
    mapping={r['image_id']:r for r in map(json.loads,(ROOT/'audits'/f'official_label_source_ids_{split}.jsonl').read_text().splitlines())}
    assert len(ds)==len(mapping), (len(ds),len(mapping))
    total=0;dedup=[];loader_map={};hashes=hashlib.sha256()
    # Verify every actual cached row against official parse/dedup of source text,
    # preserving annotation correspondence even where the loader reorders rows.
    for k,label in enumerate(ds.labels):
        iid=int(Path(label['im_file']).stem)
        ids=mapping[iid]['annotation_ids_by_line']
        label_path=ROOT/'official_yolo/labels'/split/Path(label['im_file']).with_suffix('.txt').name
        content=label_path.read_bytes() if label_path.exists() else b''
        hashes.update(str(iid).encode()+b'\0'+content)
        raw=[line.split() for line in content.decode().splitlines() if line]
        assert len(raw)==len(ids)
        segments=[np.asarray(r[1:],dtype=np.float32).reshape(-1,2) for r in raw]
        cls=np.asarray([r[0] for r in raw],dtype=np.float32).reshape(-1,1)
        boxes=segments2boxes(segments) if raw else np.zeros((0,4),np.float32)
        parsed=np.concatenate([cls,boxes],1)
        _,keep=np.unique(parsed,axis=0,return_index=True)
        if len(keep)==len(raw):keep=np.arange(len(raw))
        np.testing.assert_array_equal(label['cls'],cls[keep])
        np.testing.assert_array_equal(label['bboxes'],boxes[keep])
        assert len(label['segments'])==len(keep)
        for a,b in zip(label['segments'],[segments[i] for i in keep]):np.testing.assert_array_equal(a,b)
        if len(keep)<len(raw):
            dropped=sorted(set(range(len(raw)))-set(keep))
            dedup.append(dict(image_id=iid,removed_annotation_ids=[ids[i] for i in dropped]))
        mapped=[ids[i] for i in keep]
        assert len(mapped)==len(set(mapped))
        loader_map[iid]=dict(index=k,annotation_ids=mapped)
        total+=len(keep)
    cached=np.load(ROOT/'official_yolo/labels'/f'{split}.cache',allow_pickle=True).item()
    nf,nm,ne,nc,n=cached['results']
    assert nc==0 and n==len(mapping),cached['results']
    summary=dict(split=split,images=len(ds),actual_instances=total,source_rows=sum(len(x['annotation_ids_by_line']) for x in mapping.values()),
        loader_additional_dedup=dedup,cache_results=dict(found=nf,missing=nm,empty=ne,corrupt=nc,images=n),
        source_label_content_sha256=hashes.hexdigest(),status='PASS')
    save(f'actual_loader_{split}.json',summary)
    save(f'actual_loader_ids_{split}.json',loader_map)
    print('ACTUAL_LOADER_PASS',json.dumps(summary),flush=True)
    return ds,loader_map

def check_format(raw, formatted, mask_ratio):
    inst=deepcopy(raw['instances']);h,w=raw['img'].shape[:2]
    inst.convert_bbox('xywh');inst.denormalize(w,h)
    masks=[]
    for segment in inst.segments:
        mask=np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[segment.reshape(-1,2).astype(np.int32)],1)
        masks.append(cv2.resize(mask,(w//mask_ratio,h//mask_ratio)))
    masks=np.stack(masks) if masks else np.empty((0,h//mask_ratio,w//mask_ratio),np.uint8)
    areas=masks.sum((1,2));order=np.argsort(-areas)
    merged=np.zeros(masks.shape[1:],np.int32)
    for k,m in enumerate(masks[order]):merged=np.clip(merged+m.astype(np.int32)*(k+1),0,k+1)
    np.testing.assert_array_equal(formatted['masks'][0].numpy(),merged)
    np.testing.assert_array_equal(formatted['cls'].numpy(),raw['cls'][order])
    np.testing.assert_allclose(formatted['bboxes'].numpy(),inst.bboxes[order]/np.array([w,h,w,h]),rtol=1e-6,atol=1e-7)
    return order,masks

def spatial_audit(ds,loader_map,cfg):
    coco=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    fragmented=sorted([iid for iid,meta in loader_map.items() if any(len(coco.anns[aid]['segmentation'])>1 for aid in meta['annotation_ids'])],
        key=lambda iid:hashlib.sha256(f'fragment:{iid}'.encode()).digest())[:128]
    regular=sorted([iid for iid in loader_map if iid not in set(fragmented)],key=lambda iid:hashlib.sha256(f'regular:{iid}'.encode()).digest())[:64]
    selected=fragmented+regular
    records=[];worst=[]
    for iid in selected:
        meta=loader_map[iid];k=meta['index'];label=ds.get_image_and_label(k)
        for transform in ds.transforms.transforms[:-1]:label=transform(label)
        raw=deepcopy(label);formatted=ds.transforms.transforms[-1](label)
        order,independent_masks=check_format(raw,formatted,cfg.mask_ratio)
        h,w=raw['img'].shape[:2];h0,w0=raw['ori_shape'];(gh,gw),(left,top)=raw['ratio_pad']
        rh,rw=round(h0*gh),round(w0*gw)
        id_to_pos={source:int(pos) for pos,source in enumerate(order)}
        for j,aid in enumerate(meta['annotation_ids']):
            ann=coco.anns[aid];mask=coco.annToMask(ann)
            mask=cv2.resize(mask,(rw,rh),interpolation=cv2.INTER_NEAREST)
            pad=cv2.copyMakeBorder(mask,int(top),h-rh-int(top),int(left),w-rw-int(left),cv2.BORDER_CONSTANT,value=0)
            gt=cv2.resize(pad,(w//cfg.mask_ratio,h//cfg.mask_ratio),interpolation=cv2.INTER_NEAREST).astype(bool)
            pred=independent_masks[j].astype(bool)
            final=(formatted['masks'][0].numpy()==id_to_pos[j]+1)
            union=(gt|pred).sum();iou=(gt&pred).sum()/union if union else 1.
            final_union=(gt|final).sum();final_iou=(gt&final).sum()/final_union if final_union else 1.
            rec=dict(image_id=iid,annotation_id=aid,components=len(ann['segmentation']),mask_area=ann['area'],
                coco_pixels=int(gt.sum()),polygon_pixels=int(pred.sum()),final_pixels=int(final.sum()),
                polygon_coco_iou=float(iou),overlap_target_coco_iou=float(final_iou),lost_by_overlap=int((pred&~final).sum()))
            records.append(rec)
            if len(ann['segmentation'])>1 and ann['area']>=1024 and gt.sum()>=64:worst.append(rec)
    assert records and worst
    with (ROOT/'audits/target_fidelity.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    summary=dict(images=len(selected),instances=len(records),fragmented_instances=sum(r['components']>1 for r in records),
        coco_zero_at_stride4=sum(r['coco_pixels']==0 for r in records),official_zero_at_stride4=sum(r['final_pixels']==0 for r in records),
        polygon_iou_quantiles={str(q):float(np.percentile([r['polygon_coco_iou'] for r in records],q)) for q in [0,10,50,90,100]},
        fragmented_large_worst=sorted(worst,key=lambda r:r['polygon_coco_iou'])[:10],
        note='Official polygon bridging, OpenCV rasterization, resampling and overlap encoding are audited approximations; original COCOeval is authoritative.')
    save('target_fidelity.json',summary)
    print('FORMAT_ALIGNMENT_PASS',len(selected),'images;',len(records),'instances',flush=True)

def augmented_audit(ds,cfg):
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    count=empty=0
    for k in np.linspace(0,len(ds)-1,64,dtype=int):
        raw=ds.get_image_and_label(int(k))
        for transform in ds.transforms.transforms[:-1]:raw=transform(raw)
        before=deepcopy(raw);out=ds.transforms.transforms[-1](raw)
        check_format(before,out,cfg.mask_ratio)
        assert len(out['cls'])==len(out['bboxes'])
        n=len(out['cls']);assert out['masks'].max()<=n
        assert torch.isfinite(out['bboxes']).all()
        empty+=sum(not torch.any(out['masks']==i+1) for i in range(n));count+=n
    # Check stock collation for actual four-sample augmented batches.
    batch=ds.collate_fn([ds[i] for i in range(4)])
    assert batch['img'].shape==(4,3,640,640) and batch['masks'].shape[0]==4
    assert len(batch['cls'])==len(batch['bboxes'])==len(batch['batch_idx'])
    result=dict(augmented_images=64,augmented_instances=count,empty_target_masks=empty,
        collated_image_shape=list(batch['img'].shape),status='PASS')
    save('augmentation_alignment.json',result);print('AUGMENTATION_PASS',json.dumps(result),flush=True)

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0)
    cfg=get_cfg(overrides=json.loads((ROOT/'train_config.json').read_text()))
    data=check_det_dataset(str(ROOT/'coco_clean.yaml'),autodownload=False)
    train,train_map=audit_dataset('train2017',cfg,data)
    val,val_map=audit_dataset('val2017',cfg,data)
    assert set(train_map).isdisjoint(val_map)
    # Store GT-only smoke subset and dense stress image IDs before any predictions.
    smoke_paths=[train.labels[i]['im_file'] for i in np.linspace(0,len(train)-1,64,dtype=int)]
    (ROOT/'smoke_train.txt').write_text('\n'.join(smoke_paths)+'\n')
    stress=sorted(train.labels,key=lambda x:len(x['cls']),reverse=True)[:16]
    (ROOT/'stress_train.txt').write_text('\n'.join(x['im_file'] for x in stress)+'\n')
    spatial_audit(val,val_map,cfg)
    augmented_audit(train,cfg)
    save('DATA_PIPELINE_PASS.json',dict(status='PASS',train_images=len(train),val_images=len(val),training_path='official YOLODataset',
        train_actual=sum(len(x['cls']) for x in train.labels),val_actual=sum(len(x['cls']) for x in val.labels)))
    print('DATA_PIPELINE_PASS',flush=True)

if __name__=='__main__':main()
