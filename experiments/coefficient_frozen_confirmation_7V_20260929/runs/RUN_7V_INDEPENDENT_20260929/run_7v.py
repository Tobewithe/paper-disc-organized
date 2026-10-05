"""Experiment 7V: frozen correction confirmation on fixed official positive candidates.

The script runs the frozen YOLO26m-seg forward pass, reproduces the official
one-to-one assigner, and evaluates A/B/C/D on an image list.  It never trains
or changes the detector.  It is intentionally self-contained for the remote
Ultralytics 8.4.100 environment.
"""
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from pycocotools.coco import COCO


class Probe(nn.Module):
    def __init__(self, hdim, mode='h'):
        super().__init__(); self.mode = mode
        if mode != 'h':
            self.encoder = nn.Sequential(
                nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten())
            inp = hdim + 256
        else:
            inp = hdim
        self.head = nn.Sequential(nn.Linear(inp, 256), nn.SiLU(),
                                  nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, h, p):
        if self.mode == 'h':
            x = h
        else:
            x = torch.cat([h, self.encoder(p)], 1)
        return self.head(x)


class BigGlobal(nn.Module):
    def __init__(self, hdim, w=560):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(hdim + 256, w), nn.SiLU(), nn.Linear(w, w),
                                  nn.SiLU(), nn.Linear(w, 256))

    def forward(self, h, p):
        p = F.interpolate(F.adaptive_avg_pool2d(p, (4, 4)), size=(16, 16), mode='nearest')
        return self.head(torch.cat([h, self.enc(p)], 1))


def load_probe(path, mode='h'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = Probe(64, mode).cuda().eval(); m.load_state_dict(ck['state']); return m, ck


def load_big_global(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = BigGlobal(64).cuda().eval(); m.load_state_dict(ck['state']); return m, ck


def letterbox_gt(coco, ann_id, raw_shape, input_shape, gain, left, top):
    arr = coco.annToMask(coco.anns[int(ann_id)]).astype(np.float32)
    t = torch.from_numpy(arr)[None, None]
    H, W = map(int, raw_shape); ih, iw = map(int, input_shape)
    nh, nw = round(H * gain), round(W * gain)
    rs = F.interpolate(t, (nh, nw), mode='nearest')[0, 0]
    out = torch.zeros((ih, iw), dtype=torch.float32)
    out[int(top):int(top)+nh, int(left):int(left)+nw] = rs
    return out


def box_proto(box, h=160, w=160):
    x1 = max(0, min(w - 1, math.floor(float(box[0]) * w / 640)))
    y1 = max(0, min(h - 1, math.floor(float(box[1]) * h / 640)))
    x2 = max(x1 + 1, min(w, math.ceil(float(box[2]) * w / 640)))
    y2 = max(y1 + 1, min(h, math.ceil(float(box[3]) * h / 640)))
    return x1, y1, x2, y2


def auc_binary(y, s):
    y = np.asarray(y).astype(np.uint8).ravel(); s = np.asarray(s).ravel()
    p = y == 1; n = y == 0; np_, nn = int(p.sum()), int(n.sum())
    if np_ == 0 or nn == 0: return float('nan')
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s)+1)
    return float((ranks[p].sum() - np_*(np_+1)/2) / (np_*nn))


def metric(logit, gt, box, threshold=0.0):
    # Native input-space decoder: interpolate, crop fixed predicted box, threshold.
    m = ops.crop_mask(logit[None], box[None])[0].gt(threshold)
    y = gt.numpy().astype(bool); s = logit.numpy(); pr = m.numpy().astype(bool)
    inter = int((pr & y).sum()); union = int((pr | y).sum())
    return dict(iou=float(inter / union) if union else 1.0,
                mask75=float(inter / union >= .75) if union else 1.0,
                coverage=float(inter / max(int(y.sum()), 1)),
                fpr=float((pr & ~y).sum() / max(int((~y).sum()), 1)),
                auc=auc_binary(y, s))


def make_model(weights):
    model = YOLO(str(weights)).model.cuda().float().eval(); head = model.model[-1]
    assert head.end2end
    captured = {}; hooks = []
    for level, module in enumerate(head.one2one_cv4):
        hooks.append(module[-1].register_forward_pre_hook(
            lambda m, inputs, level=level: captured.__setitem__(level, inputs[0].detach())))
    return model, head, captured, hooks


def process_image(iid, coco, data_root, model, head, captured, corr_models, split='val2017', use_assign=True):
    name = coco.imgs[int(iid)]['file_name']; path = Path(data_root) / 'images' / split / name
    im = cv2.imread(str(path)); assert im is not None, path
    # The frozen raw bank and 7U audit use a square 640x640 letterbox.
    # Keeping auto=False is essential: auto=True would produce e.g. 480x640,
    # changing the prototype grid and invalidating the fixed correction heads.
    transform = LetterBox((640, 640), auto=False, stride=32)
    params = transform.get_params({'img': im}); resized = transform.apply_image({'img': im}, params)['img']
    x = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).cuda().float()[None] / 255
    shape = tuple(x.shape[2:]); _, raw = model(x); p = raw['one2one']; proto = p['proto'][0].float()
    c = p['mask_coefficient'][0].T.float(); boxes = head._get_decode_boxes(p)[0].T
    hidden = torch.cat([captured[l][0].flatten(1).T for l in range(3)])
    levels = torch.cat([torch.full((captured[l].shape[2]*captured[l].shape[3],), l, device='cuda', dtype=torch.long) for l in range(3)])
    # Reproduce the official one-to-one TAL assignment used by the existing bank.
    anns = sorted((v for v in coco.imgToAnns[int(iid)] if not v.get('iscrowd', 0) and not v.get('ignore', 0)), key=lambda v: v['id'])
    if not anns: return []
    # Ultralytics COCO class indices follow the sorted 80 COCO category IDs.
    cat_to_cls = {cid: j for j, cid in enumerate(sorted(coco.cats))}
    labels = torch.tensor([cat_to_cls[a['category_id']] for a in anns], device='cuda', dtype=torch.float32)[None, :, None]
    gt_boxes = torch.tensor([a['bbox'] for a in anns], device='cuda', dtype=torch.float32); gt_boxes[:, 2:] += gt_boxes[:, :2]
    H, W = im.shape[:2]; ih, iw = shape; gain = min(ih / H, iw / W); nh, nw = round(H*gain), round(W*gain)
    left, top = round((iw-nw)/2), round((ih-nh)/2)
    tboxes = gt_boxes * torch.tensor([gain, gain, gain, gain], device='cuda') + torch.tensor([left, top, left, top], device='cuda')
    criterion = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    anchors, strides = make_anchors(p['feats'], head.stride, .5)
    qboxes = criterion.bbox_decode(anchors, p['boxes'].permute(0, 2, 1)) * strides
    _, _, weights, fg, owner = criterion.assigner(p['scores'].permute(0, 2, 1).sigmoid(), qboxes, anchors * strides, labels, tboxes[None], torch.ones((1, len(anns), 1), device='cuda', dtype=torch.bool))
    fg = fg[0].cpu().numpy().astype(bool); owner = owner[0].cpu().numpy()
    if int(iid) == 1558:
        print('DEBUG_ASSIGN', int(iid), 'anns', len(anns), 'shape', shape, 'fg', int(fg.sum()),
              'owner_nonzero', int((owner >= 0).sum()), 'owner_unique', np.unique(owner)[:20], flush=True)
    raw_ids = np.flatnonzero(fg)
    if not len(raw_ids): return []
    # All correction models share frozen fit normalization from their checkpoints.
    rows=[]
    for rid in raw_ids.tolist():
        gi = int(owner[rid]); ann = anns[gi]; box = boxes[rid].float()
        x1,y1,x2,y2 = box_proto(box); roi = F.adaptive_avg_pool2d(proto[:, y1:y2, x1:x2], (16,16)).contiguous()
        z0 = (proto * c[rid, :, None, None]).sum(0)
        z640 = F.interpolate(z0[None,None], shape, mode='bilinear', align_corners=False)[0,0]
        gt640 = letterbox_gt(coco, ann['id'], im.shape[:2], shape, gain, left, top)
        row = {'image_id': int(iid), 'annotation_id': int(ann['id']), 'raw_id': int(rid), 'box': box.detach().cpu().tolist(), 'metrics': {}}
        # A and B are evaluated directly on the continuous baseline logit.
        row['_base_logit'] = z640.detach().cpu(); row['_gt'] = gt640; row['_box'] = box.detach().cpu()
        h = hidden[rid:rid+1]; pp = roi[None]
        for name, (net, ck) in corr_models.items():
            norm = ck['norm']; hh = (h - norm['hm'].cuda()) / norm['hs'].cuda()
            if name == 'h_only':
                pred = net(hh, pp).view(1, 16, 16)
            else:
                pm = norm['pm'].cuda(); ps = norm['ps'].cuda(); pred = net(hh, (pp-pm)/ps).view(1,16,16)
            # Use the actual clamped prototype slice size (some decoded boxes
            # touch the 160x160 border after integer conversion).
            target_hw = z0[y1:y2, x1:x2].shape
            dz = F.interpolate(pred[:,None], target_hw, mode='bilinear', align_corners=False)[0,0]
            z = z0.clone(); z[y1:y2, x1:x2] += dz
            row[name+'_logit'] = F.interpolate(z[None,None], shape, mode='bilinear', align_corners=False)[0,0].detach().cpu()
        rows.append(row)
    return rows


def choose_threshold(rows, grid):
    groups=defaultdict(list)
    for r in rows:
        groups[r['image_id']].append(metric(r['_base_logit'],r['_gt'],r['_box'],0.0)['iou'])
    best=None
    for b in grid:
        vals=[]
        for r in rows:
            vals.append((r['image_id'], metric(r['_base_logit'],r['_gt'],r['_box'],float(b))['iou']))
        by=defaultdict(list)
        for iid,v in vals: by[iid].append(v)
        macro=float(np.mean([np.mean(v) for v in by.values()]))
        if best is None or macro>best['macro_iou']: best={'b':float(b),'macro_iou':macro,'n_images':len(by)}
    return best


def summarize(rows, b):
    names=['A_official','B_scalar_threshold','C_h_only','D_big_global']; data={n:[] for n in names}; image={n:defaultdict(list) for n in names}
    for r in rows:
        mets={
          'A_official':metric(r['_base_logit'],r['_gt'],r['_box'],0.0),
          'B_scalar_threshold':metric(r['_base_logit'],r['_gt'],r['_box'],b),
          'C_h_only':metric(r['h_only_logit'],r['_gt'],r['_box'],0.0),
          'D_big_global':metric(r['big_global_logit'],r['_gt'],r['_box'],0.0)}
        for n,m in mets.items(): data[n].append(m); image[n][r['image_id']].append(m)
    out={'n_candidates':len(rows),'n_images':len(image['A_official']),'threshold_b':b,'methods':{},'paired_image_rows':{}}
    for n in names:
        out['methods'][n]={k:float(np.nanmean([m[k] for m in data[n]])) for k in ['iou','mask75','coverage','auc','fpr']}
        out['methods'][n]['image_macro']={k:float(np.nanmean([np.nanmean([m[k] for m in vals]) for vals in image[n].values()])) for k in ['iou','mask75','coverage','auc','fpr']}
    ids=sorted(image['A_official'])
    for n in names[1:]:
        out['paired_image_rows'][n]={k: [float(np.nanmean([m[k] for m in image[n][iid]])-np.nanmean([m[k] for m in image['A_official'][iid]])) for iid in ids] for k in ['iou','mask75','coverage','auc','fpr']}
    return out


def summarize_compact(rows, b, ids):
    names=['A_official','B_scalar_threshold','C_h_only','D_big_global']
    image={n:defaultdict(list) for n in names}
    for r in rows:
        for n,m in r['metrics'].items(): image[n][r['image_id']].append(m)
    out={'n_candidates':len(rows),'n_images':len(image['A_official']),'threshold_b':b,'methods':{},'paired_image_rows':{}}
    keys=['iou','mask75','coverage','auc','fpr']
    for n in names:
        out['methods'][n]={k:float(np.nanmean([m[k] for vals in image[n].values() for m in vals])) for k in keys}
        out['methods'][n]['image_macro']={k:float(np.nanmean([np.nanmean([m[k] for m in vals]) for vals in image[n].values()])) for k in keys}
    image_ids=sorted(image['A_official'])
    for n in names[1:]:
        out['paired_image_rows'][n]={k:[float(np.nanmean([m[k] for m in image[n][iid]])-np.nanmean([m[k] for m in image['A_official'][iid]])) for iid in image_ids] for k in keys}
    return out


def main(a):
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); coco=COCO(str(Path(a.data_root)/'annotations'/('instances_'+a.split+'.json')))
    ids=json.loads(Path(a.image_ids).read_text()); assert len(ids)==len(set(ids))
    model,head,captured,hooks=make_model(a.weights)
    hnet,hck=load_probe(a.h_only); dnet,dck=load_big_global(a.big_global)
    corr={'h_only':(hnet,hck),'big_global':(dnet,dck)}
    allrows=[]; selection=defaultdict(lambda: defaultdict(list)); t0=time.monotonic()
    try:
        for j,iid in enumerate(ids):
            rows=process_image(iid,coco,a.data_root,model,head,captured,corr,split=a.split)
            if a.mode=='select':
                for r in rows:
                    for b in [float(x) for x in a.grid.split(',')]:
                        selection[b][r['image_id']].append(metric(r['_base_logit'],r['_gt'],r['_box'],b)['iou'])
            else:
                for r in rows:
                    mets={'A_official':metric(r['_base_logit'],r['_gt'],r['_box'],0.0),
                          'B_scalar_threshold':metric(r['_base_logit'],r['_gt'],r['_box'],json.loads(Path(a.threshold).read_text())['b']),
                          'C_h_only':metric(r['h_only_logit'],r['_gt'],r['_box'],0.0),
                          'D_big_global':metric(r['big_global_logit'],r['_gt'],r['_box'],0.0)}
                    allrows.append({'image_id':r['image_id'],'annotation_id':r['annotation_id'],'metrics':mets})
            if (j+1)%25==0 or j+1==len(ids): print(json.dumps({'images':j+1,'total':len(ids),'candidates':len(allrows),'elapsed_s':round(time.monotonic()-t0,1)}),flush=True)
    finally:
        for h in hooks: h.remove()
    if a.mode=='select':
        candidates=[]
        for b,groups in selection.items():
            candidates.append({'b':float(b),'macro_iou':float(np.mean([np.mean(v) for v in groups.values()])),'n_images':len(groups)})
        res=dict(max(candidates,key=lambda x:(x['macro_iou'],-abs(x['b'])))); res['grid']=candidates
        (out/'THRESHOLD_SELECTION.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res)); return
    threshold=json.loads(Path(a.threshold).read_text())['b']
    res=summarize_compact(allrows,threshold,ids); res['image_ids']=ids; res['split']=a.split; res['model_weights']=str(a.weights); res['h_checkpoint']=str(a.h_only); res['big_global_checkpoint']=str(a.big_global); res['elapsed_s']=time.monotonic()-t0
    (out/'RESULTS.json').write_text(json.dumps(res,indent=2,allow_nan=False));
    # tensors are transient and deliberately not serialized; paired image values are sufficient.
    print(json.dumps({k:v for k,v in res.items() if k in ['n_candidates','n_images','threshold_b','methods']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['select','test'],required=True); p.add_argument('--out',required=True); p.add_argument('--data-root',required=True); p.add_argument('--image-ids',required=True); p.add_argument('--weights',required=True); p.add_argument('--h-only',required=True); p.add_argument('--big-global',required=True); p.add_argument('--threshold'); p.add_argument('--grid',default='-1,-0.5,-0.25,0,0.25,0.5,1'); p.add_argument('--split',default='val2017'); main(p.parse_args())
