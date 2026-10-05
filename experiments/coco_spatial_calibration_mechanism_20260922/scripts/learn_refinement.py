"""Frozen-YOLO learnability pilot: train2017 supervision, prediction-only refinement.

This trains separate small heads, not the pretrained YOLO network. Final-epoch
checkpoints are fixed in advance. GT is used only for training and COCO scoring.
"""
import argparse
import contextlib
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

MODES = {'scalar': 1, 'coeff_bias': 33, 'local4': 16, 'coeff_local4': 48}


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


class Refiner(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.features = nn.Sequential(nn.Conv2d(67, 32, 3, 2, 1), nn.SiLU(),
                                      nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(),
                                      nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                      nn.Linear(512, 128), nn.SiLU())
        self.output = nn.Linear(128, MODES[mode])
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.features(x)).tanh()


def roi_grid(boxes, shape, size=32):
    t = (torch.arange(size, device=boxes.device) + .5) / size
    yy, xx = torch.meshgrid(t, t, indexing='ij')
    gx = boxes[:, 0, None, None] + xx * (boxes[:, 2] - boxes[:, 0])[:, None, None]
    gy = boxes[:, 1, None, None] + yy * (boxes[:, 3] - boxes[:, 1])[:, None, None]
    return torch.stack((2 * gx / shape[1] - 1, 2 * gy / shape[0] - 1), -1)


def features(proto, coeff, boxes, shape):
    grid = roi_grid(boxes, shape)
    p = F.grid_sample(proto[None].expand(len(boxes), -1, -1, -1), grid, align_corners=False)
    scale = proto.square().mean((1, 2)).sqrt().clamp(min=.1)
    base = (p * coeff[:, :, None, None]).sum(1)
    t = torch.linspace(-1, 1, 32, device=proto.device)
    yy, xx = torch.meshgrid(t, t, indexing='ij')
    xy = torch.stack((xx, yy))[None].expand(len(boxes), -1, -1, -1)
    x = torch.cat((p / scale[None, :, None, None],
                   (p * coeff[:, :, None, None] / 8).clamp(-5, 5),
                   (base[:, None] / 8).clamp(-5, 5), xy), 1)
    return x, base, grid


def roi_correct(base, x, correction, mode):
    value = base
    if mode in ('coeff_bias', 'coeff_local4'):
        value = value + (2 * correction[:, :32, None, None] * x[:, :32]).sum(1)
    if mode in ('scalar', 'coeff_bias'):
        value = value + 4 * correction[:, -1, None, None]
    if mode in ('local4', 'coeff_local4'):
        local = 4 * correction[:, -16:].reshape(-1, 1, 4, 4)
        # Sampling at ROI cell centres matches the full-image decoder.
        t = (torch.arange(32, device=base.device) + .5) / 32 * 2 - 1
        yy, xx = torch.meshgrid(t, t, indexing='ij')
        grid = torch.stack((xx, yy), -1)[None].expand(len(base), -1, -1, -1)
        value = value + F.grid_sample(local, grid, padding_mode='border', align_corners=True)[:, 0]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-images', type=int, default=1000)
    ap.add_argument('--eval-images', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bank-source', type=Path)
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    assert ultralytics.__version__ == '8.4.100'
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    def progress(stage, **kw):
        v = dict(stage=stage, elapsed_s=round(time.monotonic()-start, 1), **kw)
        save(a.out / 'progress.json', v); print(json.dumps(v), flush=True)

    train = COCO(str(a.data / 'annotations/instances_train2017.json'))
    val = COCO(str(a.data / 'annotations/instances_val2017.json'))
    available = {int(p.stem) for p in (a.data / 'images/train2017').glob('*.jpg')}
    train_ids = sorted(available & set(train.imgs))
    random.Random(a.seed).shuffle(train_ids)
    train_ids = train_ids[:a.train_images]
    val_ids = sorted(val.imgs)[:a.eval_images]
    assert len(train_ids) == a.train_images and not (set(train_ids) & set(val_ids))
    save(a.out / 'SPLIT.json', dict(train_ids=train_ids, val_ids=val_ids,
         available_train_images=len(available), selected_before_inference=True,
         note='Base COCO checkpoint has seen train2017. val2017 has been used for earlier diagnostics.'))
    model = YOLO(str(a.weights)).model.cuda().float().eval()
    for p in model.parameters(): p.requires_grad_(False)
    head = model.model[-1]
    assert head.end2end
    cat_ids = sorted(train.cats)
    assert [model.names[i] for i in range(80)] == [train.cats[c]['name'] for c in cat_ids]
    cat_to_cls = {v: i for i, v in enumerate(cat_ids)}
    transform = LetterBox((640, 640), auto=True, stride=32)

    @torch.no_grad()
    def infer(coco, image_id, split):
        im = cv2.imread(str(a.data / 'images' / split / coco.imgs[image_id]['file_name']))
        if im is None: raise RuntimeError(f'Missing image {image_id}')
        orig = im.shape[:2]
        params = transform.get_params({'img': im})
        resized = transform.apply_image({'img': im}, params)['img']
        x = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
        _, raw = model(x)
        p = raw['one2one']; proto = p['proto'][0]
        boxes = head._get_decode_boxes(p)[0].T
        scores, classes, raw_ids = head.get_topk_index(p['scores'].permute(0, 2, 1).sigmoid(), 300)
        keep = scores[0, :, 0] > .001
        scores = scores[0, keep, 0]; classes = classes[0, keep, 0].long(); ids = raw_ids[0, keep, 0]
        return dict(orig=orig, shape=tuple(x.shape[2:]), params=params, proto=proto,
                    all_coeff=p['mask_coefficient'][0].T, boxes=boxes[ids], coeff=p['mask_coefficient'][0].T[ids],
                    scores=scores, classes=classes, raw_ids=ids)

    xs, bases, targets, records = [], [], [], []
    for ni, image_id in enumerate([] if a.bank_source else train_ids):
        q = infer(train, image_id, 'train2017')
        anns = [v for v in train.imgToAnns[image_id] if not v.get('iscrowd', 0) and not v.get('ignore', 0)]
        if anns and len(q['boxes']):
            gt_boxes = torch.tensor([v['bbox'] for v in anns], device='cuda', dtype=torch.float32)
            gt_boxes[:, 2:] += gt_boxes[:, :2]
            ob = ops.scale_boxes(q['shape'], q['boxes'].clone(), q['orig'])
            quality = box_iou(ob, gt_boxes)
            gt_cls = torch.tensor([cat_to_cls[v['category_id']] for v in anns], device='cuda')
            quality[q['classes'][:, None] != gt_cls[None]] = -1
            best, owner = quality.max(1)
            # Ordinary post-detection mask-head supervision. This does not replace
            # YOLO's assignment or claim that selected predictions were its positives.
            ix = torch.where(best >= .5)[0]
            if len(ix):
                x, base, grid = features(q['proto'], q['coeff'][ix], q['boxes'][ix], q['shape'])
                owners = owner[ix].cpu().tolist()
                originals = torch.tensor(np.stack([train.annToMask(anns[j]) for j in owners]), device='cuda', dtype=torch.float32)
                oh, ow = q['orig']; ih, iw = q['shape']; gain = min(ih/oh, iw/ow)
                nh, nw = round(oh*gain), round(ow*gain)
                top, left = round((ih-nh)/2-.1), round((iw-nw)/2-.1)
                input_gt = F.pad(F.interpolate(originals[:, None], (nh, nw), mode='bilinear', align_corners=False),
                                 (left, iw-nw-left, top, ih-nh-top))
                y = F.grid_sample(input_gt, grid, align_corners=False)[:, 0]
                xs.append(x.cpu().half()); bases.append(base.cpu().half()); targets.append(y.cpu().half())
                records.extend(dict(image_id=image_id, annotation_id=anns[j]['id'], raw_id=int(q['raw_ids'][k]))
                               for k, j in zip(ix.cpu().tolist(), owners))
        if ni % 25 == 0 or ni+1 == len(train_ids): progress('prepare_train', images=ni+1, total=len(train_ids), instances=len(records))
    if a.bank_source:
        old_split=json.loads((a.bank_source.parent/'SPLIT.json').read_text())
        assert old_split['train_ids']==train_ids and old_split['val_ids']==val_ids
        bank=torch.load(a.bank_source,map_location='cpu',weights_only=False,mmap=True)
    else:
        bank = dict(x=torch.cat(xs), base=torch.cat(bases), target=torch.cat(targets), records=records)
    del xs, bases, targets
    # Refiners are class-agnostic: one raw geometry must not receive incompatible
    # masks from different class entries of YOLO's top-k output.
    owners=defaultdict(set)
    for r in bank['records']: owners[(r['image_id'],r['raw_id'])].add(r['annotation_id'])
    seen=set();keep=[]
    for index,r in enumerate(bank['records']):
        key=(r['image_id'],r['raw_id'])
        if len(owners[key])==1 and key not in seen:keep.append(index);seen.add(key)
    save(a.out/'BANK_AUDIT.json',dict(source=str(a.bank_source) if a.bank_source else 'newly_extracted',
         original_rows=len(bank['records']),unique_geometry=len(owners),
         conflicting_geometry=sum(len(v)>1 for v in owners.values()),retained_rows=len(keep),
         policy='Drop every ambiguous raw identity and keep one copy of each remaining identity. Training only.'))
    ix=torch.tensor(keep,dtype=torch.long)
    bank={**{k:bank[k][ix] for k in ('x','base','target')},'records':[bank['records'][i] for i in keep]}
    assert len({(r['image_id'],r['raw_id']) for r in bank['records']})==len(bank['records'])
    if not a.bank_source:torch.save(bank, a.out / 'training_bank.pt')
    assert torch.isfinite(bank['x']).all() and torch.isfinite(bank['base']).all()
    if a.prepare_only:
        save(a.out / 'COMPLETE.json', dict(stage='prepare_only', instances=len(bank['records']),
             images=len(train_ids), bank='training_bank.pt'))
        return
    dataset = TensorDataset(bank['x'], bank['base'], bank['target'])
    refiners = {}; histories = {}
    for mode in MODES:
        torch.manual_seed(a.seed)
        net = Refiner(mode).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(a.seed)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator, num_workers=0)
        histories[mode] = []
        with torch.no_grad():
            x0=bank['x'][:2].cuda().float(); b0=bank['base'][:2].cuda().float()
            assert torch.equal(roi_correct(b0, x0, net(x0), mode), b0)
        for epoch in range(a.epochs):
            losses=[]
            for x, base, target in loader:
                x=x.cuda().float(); base=base.cuda().float(); target=target.cuda().float()
                logits=roi_correct(base, x, net(x), mode); prob=logits.sigmoid()
                loss=F.binary_cross_entropy_with_logits(logits,target)+.5*(1-(2*(prob*target).sum((1,2))+1)/(prob.sum((1,2))+target.sum((1,2))+1)).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),10); opt.step()
                losses.append(float(loss.detach()))
            histories[mode].append(float(np.mean(losses)))
            torch.save(dict(state_dict=net.state_dict(), mode=mode, epoch=epoch+1, seed=a.seed,
                            parameter_count=sum(p.numel() for p in net.parameters())), a.out/f'{mode}_epoch{epoch+1}.pt')
            progress('train', mode=mode, epoch=epoch+1, epochs=a.epochs, mean_loss=histories[mode][-1])
        refiners[mode]=net.eval()
    save(a.out/'TRAINING.json',dict(histories=histories,instances=len(dataset),epochs=a.epochs,
         checkpoint_selection='last epoch, no validation selection',
         parameters={m:sum(p.numel() for p in n.parameters()) for m,n in refiners.items()}))
    del bank, dataset, loader
    streams={m:(a.out/f'predictions_{m}.jsonl').open('w') for m in ['baseline',*MODES]}
    with torch.no_grad():
        for ni,image_id in enumerate(val_ids):
            q=infer(val,image_id,'val2017'); shape=q['shape']; orig=q['orig']; proto=q['proto']
            scale=proto.square().mean((1,2)).sqrt().clamp(min=.1)
            full_logits=(q['all_coeff']@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
            ob=ops.scale_boxes(shape,q['boxes'].clone(),orig)
            yy,xx=torch.meshgrid(torch.arange(shape[0],device='cuda'),torch.arange(shape[1],device='cuda'),indexing='ij')
            for first in range(0,len(q['boxes']),16):
                sl=slice(first,first+16); boxes=q['boxes'][sl]; coeff=q['coeff'][sl]; ids=q['raw_ids'][sl]
                x,_,_=features(proto,coeff,boxes,shape)
                base=F.interpolate(full_logits[ids][None],shape,mode='bilinear',align_corners=False)[0]
                variants={'baseline':base}
                for mode,net in refiners.items():
                    correction=net(x); logits=base
                    if mode in ('coeff_bias','coeff_local4'):
                        delta=(2*correction[:,:32]/scale[None])@proto.flatten(1)
                        logits=logits+F.interpolate(delta.reshape(-1,*proto.shape[-2:])[None],shape,mode='bilinear',align_corners=False)[0]
                    if mode in ('scalar','coeff_bias'): logits=logits+4*correction[:,-1,None,None]
                    if mode in ('local4','coeff_local4'):
                        gx=2*(xx[None]-boxes[:,0,None,None])/(boxes[:,2]-boxes[:,0]).clamp(min=1)[:,None,None]-1
                        gy=2*(yy[None]-boxes[:,1,None,None])/(boxes[:,3]-boxes[:,1]).clamp(min=1)[:,None,None]-1
                        grid=torch.stack((gx,gy),-1)
                        logits=logits+F.grid_sample(4*correction[:,-16:].reshape(-1,1,4,4),grid,padding_mode='border',align_corners=True)[:,0]
                    variants[mode]=logits
                for mode,logits in variants.items():
                    binary=ops.crop_mask(logits.clone(),boxes).gt(0).byte()
                    masks=ops.scale_masks(binary[None],orig)[0].byte().cpu().numpy()
                    for j,mask in enumerate(masks):
                        if not mask.any(): continue
                        k=first+j; rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                        bb=ob[k].clone();bb[2:]-=bb[:2]
                        streams[mode].write(json.dumps(dict(image_id=image_id,category_id=cat_ids[int(q['classes'][k])],
                            score=float(q['scores'][k]),bbox=bb.cpu().tolist(),segmentation=rle))+'\n')
            if ni%25==0 or ni+1==len(val_ids):
                for f in streams.values():f.flush()
                progress('evaluate_predictions',images=ni+1,total=len(val_ids))
    for f in streams.values():f.close()
    results={}
    for mode in ['baseline',*MODES]:
        predictions=[json.loads(line) for line in (a.out/f'predictions_{mode}.jsonl').read_text().splitlines()]
        # Segmentation area must come from RLE, not the simultaneously exported bbox.
        dets=[{k:p[k] for k in ('image_id','category_id','score','segmentation')} for p in predictions]
        dt=val.loadRes(dets)
        evaluator=COCOeval(val,dt,'segm');evaluator.params.imgIds=val_ids
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        results[mode]=dict(stats=evaluator.stats.tolist(),prediction_count=len(predictions))
        save(a.out/'METRICS.json',results)
        progress('score',mode=mode,mask_ap=100*evaluator.stats[0])
        del dt,evaluator,predictions
    save(a.out/'COMPLETE.json',dict(train_images=len(train_ids),eval_images=len(val_ids),results=results,
         note='Single-seed frozen-backbone pilot, not a final method or controlled full-network fine-tuning comparison.'))


if __name__=='__main__':main()
