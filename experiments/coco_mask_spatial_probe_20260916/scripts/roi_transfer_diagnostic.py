"""Compare training ROI supervision to final masks on fixed prediction identities.

Use available original train2017 fit/selection images, or a fixed fresh cohort
with --fresh-unseen. Neither available-image subset represents full COCO.
"""
import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from pycocotools.coco import COCO
import component_seed_probe as probe
import learn_refinement as original
from repair_refinement import per_loss


def metrics(mask, gt):
    m, g = mask.bool().flatten(1), gt.bool().flatten(1)
    tp = (m & g).sum(1).float()
    size, area = m.sum(1).float(), g.sum(1).float()
    return torch.stack((tp/(size+area-tp).clamp(min=1), tp/area.clamp(min=1), tp/size.clamp(min=1)), 1)


def mask_gt(coco, anns, orig, shape, boxes):
    gt = torch.from_numpy(np.stack([coco.annToMask(v) for v in anns])).cuda().float()
    oh, ow = orig; ih, iw = shape; gain = min(ih/oh, iw/ow)
    nh, nw = round(oh*gain), round(ow*gain)
    top, left = round((ih-nh)/2-.1), round((iw-nw)/2-.1)
    dense = F.pad(F.interpolate(gt[:, None], (nh, nw), mode='bilinear', align_corners=False),
                  (left, iw-nw-left, top, ih-nh-top))[:, 0]
    grid = original.roi_grid(boxes, shape)
    roi = F.grid_sample(dense[:, None], grid, align_corners=False)[:, 0]
    return gt, dense, roi


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for k in ('root', 'data', 'reference', 'out'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--fresh-unseen',action='store_true')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    probe.setup(); start = time.monotonic()
    split = json.loads((a.reference/'SPLIT.json').read_text())
    selected = {}
    for subset, key in [('fit', 'fit_image_ids'), ('selection', 'selection_image_ids')]:
        selected[subset] = [i for i in split[key] if (a.data/'images/train2017'/f'{i:012d}.jpg').is_file()]
    if a.fresh_unseen:
        excluded=set(split['fit_image_ids'])|set(split['selection_image_ids'])|set(split['val_ids'])
        eligible=sorted({int(p.stem) for p in (a.data/'images/train2017').glob('*.jpg') if p.stem.isdigit()}-excluded)
        random.Random(20260917).shuffle(eligible)
        assert len(eligible)>=338
        selected={'fit':eligible[:268],'selection':eligible[268:338]}
    assert not set(selected['fit']) & set(selected['selection'])
    original.save(a.out/'SPLIT.json', dict(**selected, original_fit=800, original_selection=200,
        fresh_unseen=a.fresh_unseen,
        rule='Fresh train2017 images outside original refiner fit/selection/val sets, seed20260917, split268/70.' if a.fresh_unseen else
             'Use all existing original-split train2017 images; availability checked before inference. No val images.'))
    coco = COCO(str(a.data/'annotations/instances_train2017.json'))
    cat_ids = sorted(coco.cats); cat_to_cls = {v:i for i,v in enumerate(cat_ids)}
    model = YOLO(str(a.root/'yolo26m-seg.pt')).model.cuda().float().eval(); head = model.model[-1]
    assert head.end2end
    nets = {}
    for name, rid in [('plain','RUN_8217274af2db49ffa6fb8d6725f8496b'),
                      ('guard','RUN_95530b30e7c04daf8acbf175b8b1204a'),
                      ('safe','RUN_6c39da21dae64ec9893479d0d28c7d33')]:
        net = original.Refiner('local4').cuda().eval()
        net.load_state_dict(torch.load(a.reference.parent/rid/'epoch8.pt',weights_only=False)['state_dict'])
        nets[name] = net
    conditions = {'plain_half':('plain',.5), 'plain_full':('plain',1.),
                  'guard_half':('guard',.5), 'guard_full':('guard',1.),
                  'safe_half':('safe',.5), 'safe_full':('safe',1.)}
    transform = LetterBox((640,640), auto=True, stride=32)
    rows = []; dropped_ambiguous = 0; processed=0
    # Save the 32x32 input/target tensors for bounded training-only follow-up.
    cache = defaultdict(list); records=[]
    for subset, image_ids in selected.items():
        for image_id in image_ids:
            im = cv2.imread(str(a.data/'images/train2017'/coco.imgs[image_id]['file_name']))
            orig = im.shape[:2]; params=transform.get_params({'img':im})
            resized=transform.apply_image({'img':im},params)['img']
            inp=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            shape=tuple(inp.shape[2:]);_,raw=model(inp);out=raw['one2one']
            proto=out['proto'][0];allc=out['mask_coefficient'][0].T
            allboxes=head._get_decode_boxes(out)[0].T
            ts,tc,ti=head.get_topk_index(out['scores'].permute(0,2,1).sigmoid(),300)
            keep=ts[0,:,0]>.001;scores=ts[0,keep,0];classes=tc[0,keep,0].long();ids=ti[0,keep,0]
            boxes=allboxes[ids];coeff=allc[ids]
            anns=[v for v in coco.imgToAnns[image_id] if not v.get('iscrowd',0) and not v.get('ignore',0)]
            if anns and len(boxes):
                gb=torch.tensor([v['bbox'] for v in anns],device='cuda',dtype=torch.float32);gb[:,2:]+=gb[:,:2]
                ob=ops.scale_boxes(shape,boxes.clone(),orig)
                q=box_iou(ob,gb);gcls=torch.tensor([cat_to_cls[v['category_id']] for v in anns],device='cuda')
                q[classes[:,None]!=gcls[None]]=-1
                quality,owner=q.max(1);eligible=torch.where(quality>=.5)[0].cpu().tolist()
                owners=defaultdict(set)
                for k in eligible:owners[int(ids[k])].add(int(owner[k]))
                retained=[];seen=set()
                for k in eligible:
                    identity=int(ids[k])
                    if len(owners[identity])!=1:dropped_ambiguous+=1;continue
                    if identity not in seen:retained.append(k);seen.add(identity)
                for first in range(0,len(retained),8):
                    ix=retained[first:first+8];bb=boxes[ix];cc=coeff[ix]
                    owned=[anns[int(owner[k])] for k in ix]
                    x,b,_=original.features(proto,cc,bb,shape)
                    gt,dense_gt,y=mask_gt(coco,owned,orig,shape,bb)
                    low=(cc@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
                    base=F.interpolate(low[None],shape,mode='bilinear',align_corners=False)[0]
                    # Original official scaling converts interpolated binary mask to byte.
                    base_input=ops.crop_mask(base.clone(),bb)>0
                    base_full=ops.scale_masks(base_input.byte()[None],orig)[0].byte()
                    scale=proto.square().mean((1,2)).sqrt().clamp(min=.1)
                    values={'baseline':{'roi':metrics(b>0,y>=.5),'input':metrics(base_input,dense_gt>=.5),
                        'full':metrics(base_full,gt),'roi_loss':per_loss(b,y)}}
                    net_outputs={k:net(x) for k,net in nets.items()}
                    for name,(netname,alpha) in conditions.items():
                        c=net_outputs[netname]*alpha
                        rz=probe.roi_correct(b,x,c,'local4')
                        z=probe.image_correct(base,proto,scale,bb,shape,c,'local4','pixel_center')
                        inputmask=ops.crop_mask(z.clone(),bb)>0
                        full=ops.scale_masks(inputmask.byte()[None],orig)[0].byte()
                        values[name]={'roi':metrics(rz>0,y>=.5),'input':metrics(inputmask,dense_gt>=.5),
                                      'full':metrics(full,gt),'roi_loss':per_loss(rz,y)}
                    cpu={name:{k:v.cpu().tolist() for k,v in m.items()} for name,m in values.items()}
                    for j,k in enumerate(ix):
                        ann=owned[j]
                        row=dict(image_id=image_id,annotation_id=ann['id'],raw_id=int(ids[k]),split=subset,
                            size='small' if ann['area']<1024 else 'medium' if ann['area']<9216 else 'large',
                            box_iou=float(quality[k]),score=float(scores[k]),gt_area=ann['area'],
                            box_width_input=float(bb[j,2]-bb[j,0]),box_height_input=float(bb[j,3]-bb[j,1]),
                            values={name:{key:v[j] for key,v in m.items()} for name,m in cpu.items()})
                        rows.append(row);records.append({k:row[k] for k in ('image_id','annotation_id','raw_id','split')})
                    cache['x'].append(x.cpu().half());cache['base'].append(b.cpu().half());cache['target'].append(y.cpu().half())
            processed+=1
            if processed%10==0 or processed==sum(map(len,selected.values())):
                progress=dict(images=processed,total=sum(map(len,selected.values())),rows=len(rows),elapsed_s=time.monotonic()-start)
                original.save(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    with (a.out/'instances.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    bank={k:torch.cat(v) for k,v in cache.items()};bank['records']=records
    torch.save(bank,a.out/'diagnostic_bank.pt')
    summary={}
    for subset in selected:
        for size in ('all','small','medium','large'):
            rs=[r for r in rows if r['split']==subset and (size=='all' or r['size']==size)]
            data={}
            for name in conditions:
                full=np.array([r['values'][name]['full'][0]-r['values']['baseline']['full'][0] for r in rs])
                roi=np.array([r['values'][name]['roi'][0]-r['values']['baseline']['roi'][0] for r in rs])
                il=np.array([r['values'][name]['roi_loss']-r['values']['baseline']['roi_loss'] for r in rs])
                cov=np.array([r['values'][name]['full'][1]-r['values']['baseline']['full'][1] for r in rs])
                bad=(full < -1e-8)
                dmg=np.array([r['values']['baseline']['full'][0]>=.75 and r['values'][name]['full'][0]<.75 for r in rs])
                rep=np.array([r['values']['baseline']['full'][0]<.75 and r['values'][name]['full'][0]>=.75 for r in rs])
                data[name]=dict(n=len(rs),roi_delta_iou_pp=float(100*roi.mean()),full_delta_iou_pp=float(100*full.mean()),
                    full_delta_coverage_pp=float(100*cov.mean()),full_worse=int(bad.sum()),
                    full_worse_but_roi_better=int((bad&(roi>1e-8)).sum()),
                    full_worse_but_roi_loss_better=int((bad&(il < -1e-8)).sum()),
                    repaired75=int(rep.sum()),damaged75=int(dmg.sum()),
                    damaged75_roi_iou_better=int((dmg&(roi>1e-8)).sum()),
                    damaged75_roi_loss_better=int((dmg&(il < -1e-8)).sum()),
                    damaged75_roi_also_fails=int(sum(flag and r['values'][name]['roi'][0]<.75 for flag,r in zip(dmg,rs))))
            summary[subset+':'+size]=data
    original.save(a.out/'SUMMARY.json',dict(groups=summary,images={k:len(v) for k,v in selected.items()},rows=len(rows),
        ambiguous_rows_removed=dropped_ambiguous,limitations=['Train2017 available-image subset; inspect SPLIT.json for original vs fresh unseen selection.',
        'Unique raw predicted boxes matched to GT at same class BoxIoU>=.5 only for supervision diagnosis.',
        'Multiple raw predictions may share a GT; counts are prediction-GT pairs, not official COCO recall.',
        'No training and no validation-based tuning. ROI target follows existing bilinear mask training.',
        'Full-space mask uses current pixel-centre aligned local field and original mask crop/scale rule.']))
    original.save(a.out/'COMPLETE.json',dict(images=processed,rows=len(rows),elapsed_s=time.monotonic()-start))
    print(json.dumps({k:v for k,v in summary.items() if k.endswith(':all')}),flush=True)


if __name__=='__main__':main()
