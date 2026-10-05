"""Matched component ablations and paired seed replication of mask refinement.

Uses the existing frozen detector, bank, split and decoder. No GT in refinement
inference. Training GT is used for supervision and train-only alpha selection.
"""
import argparse
from collections import defaultdict
import gc
import gzip
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

import learn_refinement as original
from repair_refinement import per_loss, hard_iou, soft_iou

original.MODES['coeff'] = 32
MODES = ('scalar', 'coeff', 'local4', 'coeff_local4')
save = original.save


def roi_correct(base, x, correction, mode):
    if mode == 'coeff':
        return base + (2 * correction[:, :32, None, None] * x[:, :32]).sum(1)
    return original.roi_correct(base, x, correction, mode)


def image_correct(base, proto, scale, boxes, shape, correction, mode, coordinate_rule='pixel_center'):
    z = base
    if mode in ('coeff', 'coeff_local4'):
        d = (2 * correction[:, :32] / scale[None]) @ proto.flatten(1)
        z = z + F.interpolate(d.reshape(-1, *proto.shape[-2:])[None], shape,
                              mode='bilinear', align_corners=False)[0]
    if mode in ('local4', 'coeff_local4'):
        assert coordinate_rule in ('legacy_integer', 'pixel_center')
        offset = .5 if coordinate_rule == 'pixel_center' else 0.
        yy, xx = torch.meshgrid(torch.arange(shape[0], device=base.device),
                                torch.arange(shape[1], device=base.device), indexing='ij')
        yy, xx = yy + offset, xx + offset
        gx = 2 * (xx[None] - boxes[:, 0, None, None]) / (boxes[:, 2] - boxes[:, 0]).clamp(min=1)[:, None, None] - 1
        gy = 2 * (yy[None] - boxes[:, 1, None, None]) / (boxes[:, 3] - boxes[:, 1]).clamp(min=1)[:, None, None] - 1
        z = z + F.grid_sample(4 * correction[:, -16:].reshape(-1, 1, 4, 4),
                              torch.stack((gx, gy), -1), padding_mode='border', align_corners=True)[:, 0]
    if mode == 'scalar':
        z = z + 4 * correction[:, -1, None, None]
    return z


def setup():
    assert ultralytics.__version__ == '8.4.100'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def correctness_guard(z, baseline, target):
    """Protect correct positive/negative pixels with a capped signed margin.

    GT defines protected pixels during training only. Equal per-instance
    positive/negative normalization prevents numerous background pixels from
    overwhelming the target. Incorrect baseline pixels carry no penalty here.
    """
    fg = target >= .5
    sign = fg.to(z.dtype) * 2 - 1
    correct = ((baseline > 0) == fg).detach()
    margin = (sign * baseline).detach().clamp(min=0, max=1)
    penalty = F.relu(margin - sign*z).square()
    terms = []
    for region in (correct & fg, correct & ~fg):
        terms.append((penalty*region).sum((1,2)) / region.sum((1,2)).clamp(min=1))
    return .5*(terms[0]+terms[1])


def train(a):
    setup()
    start = time.monotonic()
    objective = getattr(a, 'objective', 'plain')
    bank = torch.load(a.bank, map_location='cpu', weights_only=False, mmap=True)
    split = json.loads((a.reference / 'SPLIT.json').read_text())
    fit_ids, dev_ids = set(split['fit_image_ids']), set(split['selection_image_ids'])
    assert len(fit_ids) == 800 and len(dev_ids) == 200 and not fit_ids & dev_ids
    owners = defaultdict(set)
    for row in bank['records']:
        owners[(row['image_id'], row['raw_id'])].add(row['annotation_id'])
    seen, fit, dev = set(), [], []
    for i, row in enumerate(bank['records']):
        key = (row['image_id'], row['raw_id'])
        if len(owners[key]) != 1 or key in seen:
            continue
        seen.add(key)
        assert row['image_id'] in fit_ids | dev_ids
        (fit if row['image_id'] in fit_ids else dev).append(i)
    assert (len(fit), len(dev)) == (14204, 3723)
    fitset = TensorDataset(*(bank[k][fit] for k in ('x', 'base', 'target')))
    devset = TensorDataset(*(bank[k][dev] for k in ('x', 'base', 'target')))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    net = original.Refiner(a.mode).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(fitset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=0)
    save(a.out / 'SPLIT.json', split)
    with torch.no_grad():
        x, b, _ = next(iter(DataLoader(fitset, batch_size=2)))
        x, b = x.cuda().float(), b.cuda().float()
        assert torch.equal(roi_correct(b, x, net(x), a.mode), b)
    history = []
    for epoch in range(1, 9):
        net.train(); losses = []
        for x, b, y in loader:
            x, b, y = x.cuda().float(), b.cuda().float(), y.cuda().float()
            z = roi_correct(b, x, net(x), a.mode)
            loss = per_loss(z, y).mean()
            if objective == 'guard':
                loss = loss + correctness_guard(z, b, y).mean()
            elif objective == 'safe':
                loss = loss + 2*F.relu(soft_iou(b,y).detach()-soft_iou(z,y)).mean() + .01*(z-b).square().mean()
            else:
                assert objective == 'plain'
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10, error_if_nonfinite=True)
            opt.step(); losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        torch.save(dict(state_dict=net.state_dict(), mode=a.mode, seed=a.seed, epoch=epoch,
                        objective=objective, parameter_count=sum(p.numel() for p in net.parameters())), a.out / f'epoch{epoch}.pt')
        progress = dict(stage='train', mode=a.mode, seed=a.seed, epoch=epoch,
                        loss=history[-1], grad_norm=float(norm), elapsed_s=time.monotonic()-start)
        save(a.out / 'progress.json', progress); print(json.dumps(progress), flush=True)
    net.eval(); alphas = (0., .25, .5, 1.)
    stats = {str(alpha): dict(total=0., n=0, repairs=0, damages=0, loss=0.) for alpha in alphas}
    with torch.no_grad():
        for x, b, y in DataLoader(devset, batch_size=128):
            x, b, y = x.cuda().float(), b.cuda().float(), y.cuda().float()
            c = net(x); biou = hard_iou(b, y)
            for alpha in alphas:
                z = roi_correct(b, x, c*alpha, a.mode); iou = hard_iou(z, y); s = stats[str(alpha)]
                s['total'] += float(iou.sum()); s['n'] += len(x)
                s['repairs'] += int(((iou >= .75) & (biou < .75)).sum())
                s['damages'] += int(((iou < .75) & (biou >= .75)).sum())
                s['loss'] += float(per_loss(z, y).sum())
    for s in stats.values():
        s['mean_iou'], s['mean_loss'] = s['total']/s['n'], s['loss']/s['n']
    alpha = max(alphas, key=lambda v: (stats[str(v)]['mean_iou'], -v))
    selection = dict(mode=a.mode, seed=a.seed, objective=objective, alpha=alpha, stats=stats, history=history,
                     checkpoint='epoch8.pt', parameters=sum(p.numel() for p in net.parameters()))
    save(a.out / 'SELECTION.json', selection)
    save(a.out / 'COMPLETE.json', dict(**selection, elapsed_s=time.monotonic()-start))
    print(json.dumps(selection), flush=True)


@torch.no_grad()
def evaluate(a, smoke=False):
    setup(); start = time.monotonic()
    ref_results = json.loads((a.reference / 'RESULTS.json').read_text())
    ref_matched = json.loads((a.reference / 'MATCHED_GT75.json').read_text())
    split = json.loads((a.reference / 'SPLIT.json').read_text())
    val_ids = split['val_ids'][:2] if smoke else split['val_ids']
    coco = COCO(str(a.data / 'annotations/instances_val2017.json'))
    cat_ids = sorted(coco.cats)
    nets, specs = {}, {}
    if smoke:
        specs = {'baseline': dict(mode=None, alpha=0),
                 'combo_s0': dict(mode='coeff_local4', alpha=.5, coordinate_rule='legacy_integer')}
        ck = torch.load(a.reference/'coeff_local4_plain_epoch8.pt', weights_only=False)
        net = original.Refiner('coeff_local4').cuda().eval(); net.load_state_dict(ck['state_dict']); nets['combo_s0'] = net
    else:
        manifest = json.loads(a.manifest.read_text())
        for entry in manifest['train']:
            if entry['mode'] is None:
                assert entry['name'] == 'baseline'
                specs['baseline'] = dict(mode=None, alpha=0, strength_rule='frozen detector')
                continue
            source = a.reference.parent / entry['run_id']
            source_record = json.loads((source/'run.json').read_text())
            assert source_record['status'] == 'completed'
            chosen = json.loads((source/'SELECTION.json').read_text())
            name = entry.get('name', f"{entry['mode']}_s{entry['seed']}")
            net = original.Refiner(entry['mode']).cuda().eval()
            net.load_state_dict(torch.load(source/'epoch8.pt', weights_only=False)['state_dict'])
            nets[name] = net
            alpha = entry.get('alpha_override', chosen['alpha'])
            assert 0 <= alpha <= 1
            specs[name] = dict(mode=entry['mode'], alpha=alpha, selection_alpha=chosen['alpha'],
                               coordinate_rule=entry.get('coordinate_rule', 'pixel_center'),
                               strength_rule='fixed diagnostic override' if 'alpha_override' in entry else 'train-only selection',
                               source_run=source_record['run_id'],
                               source_directory=entry['run_id'], seed=entry['seed'], objective=chosen.get('objective','plain'))
    model = YOLO(str(a.weights)).model.cuda().float().eval(); head = model.model[-1]
    assert head.end2end
    transform = LetterBox((640, 640), auto=True, stride=32)
    save(a.out/'VARIANTS.json', specs); save(a.out/'SPLIT.json', {**split, 'val_ids':val_ids})
    streams = {k:gzip.open(a.out/f'predictions_{k}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) for k in specs}
    for ni, image_id in enumerate(val_ids):
        im = cv2.imread(str(a.data/'images/val2017'/coco.imgs[image_id]['file_name'])); assert im is not None
        orig = im.shape[:2]; params = transform.get_params({'img': im})
        res = transform.apply_image({'img': im}, params)['img']
        inp = torch.from_numpy(np.ascontiguousarray(res[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
        shape = tuple(inp.shape[2:]); _, raw = model(inp); p = raw['one2one']
        proto, allc = p['proto'][0], p['mask_coefficient'][0].T
        allboxes = head._get_decode_boxes(p)[0].T
        ts, tc, ti = head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(), 300)
        keep = ts[0,:,0] > .001
        scores, classes, ids = ts[0,keep,0], tc[0,keep,0].long(), ti[0,keep,0]
        boxes, coeff = allboxes[ids], allc[ids]
        low = (allc@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
        scale = proto.square().mean((1,2)).sqrt().clamp(min=.1)
        for first in range(0,len(boxes),16):
            sl = slice(first,first+16); bb = boxes[sl]
            x, b, _ = original.features(proto,coeff[sl],bb,shape)
            base = F.interpolate(low[ids[sl]][None],shape,mode='bilinear',align_corners=False)[0]
            if smoke:
                # All component decoders must be exactly identity at zero correction.
                for mode in MODES:
                    c = torch.zeros((len(bb),original.MODES[mode]),device=base.device)
                    assert torch.equal(roi_correct(b,x,c,mode),b)
                    assert torch.equal(image_correct(base,proto,scale,bb,shape,c,mode),base)
            for name, spec in specs.items():
                z = base if not spec['mode'] else image_correct(base,proto,scale,bb,shape,nets[name](x)*spec['alpha'],spec['mode'],
                                                                spec.get('coordinate_rule','pixel_center'))
                masks = ops.scale_masks(ops.crop_mask(z.clone(),bb).gt(0).byte()[None],orig)[0].byte().cpu().numpy()
                for j,mask in enumerate(masks):
                    if not mask.any(): continue
                    k=first+j; rle=mu.encode(np.asfortranarray(mask)); rle['counts']=rle['counts'].decode('ascii')
                    streams[name].write(json.dumps(dict(image_id=image_id,category_id=cat_ids[int(classes[k])],
                        score=float(scores[k]),segmentation=rle,raw_id=int(ids[k])))+'\n')
        if ni%25==0 or ni+1==len(val_ids):
            for stream in streams.values(): stream.flush()
            progress=dict(stage='predict',images=ni+1,total=len(val_ids),elapsed_s=time.monotonic()-start)
            save(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for stream in streams.values(): stream.close()
    if smoke:
        def records(path):
            with gzip.open(path,'rt') as f:
                return {(r['image_id'],r['category_id'],r['raw_id']):r for line in f
                        if (r:=json.loads(line))['image_id'] in val_ids}
        checks={}
        for name,previous in [('baseline','baseline'),('combo_s0','combo_selected')]:
            new=records(a.out/f'predictions_{name}.jsonl.gz');old=records(a.reference/f'predictions_{previous}.jsonl.gz')
            assert new.keys()==old.keys()
            assert all(new[k]['segmentation']==old[k]['segmentation'] and new[k]['score']==old[k]['score'] for k in new)
            checks[name]=dict(records=len(new),identical=True)
        save(a.out/'COMPLETE.json',dict(smoke=checks,zero_identity_all_modes=True));return
    fresh = manifest.get('fresh_controls', False)
    if fresh:
        assert list(specs)[0] == 'baseline' and not manifest.get('reuse_evaluations')
        results, matched = {}, {}
    else:
        results={'baseline':ref_results['baseline'], 'scalar_s0':ref_results['scalar_selected'],
                 'coeff_local4_s0':ref_results['combo_selected']}
        matched={'baseline':ref_matched['baseline'],'scalar_s0':ref_matched['scalar_selected'],
                 'coeff_local4_s0':ref_matched['combo_selected']}
    # Previously completed paired controls are reused without retraining or reselecting.
    for reuse in manifest.get('reuse_evaluations', []):
        source = a.reference.parent/reuse
        old_result=json.loads((source/'RESULTS.json').read_text())
        old_matched=json.loads((source/'MATCHED_GT75.json').read_text())
        assert json.loads((source/'SPLIT.json').read_text())['val_ids'] == val_ids
        assert old_result['baseline']['metrics'] == results['baseline']['metrics']
        results.update(old_result);matched.update(old_matched)
    base_good=set(matched.get('baseline', []))
    ordinary={v['id']:v for image_id in val_ids for v in coco.imgToAnns[image_id] if not v.get('iscrowd',0) and not v.get('ignore',0)}
    def size(v): return 'small' if v['area']<1024 else 'medium' if v['area']<9216 else 'large'
    names=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL')
    for name in specs:
        with gzip.open(a.out/f'predictions_{name}.jsonl.gz','rt') as f: pred=[json.loads(line) for line in f]
        assert all('bbox' not in r for r in pred)
        dt=coco.loadRes(pred);ev=COCOeval(coco,dt,'segm');ev.params.imgIds=val_ids
        ev.evaluate();ev.accumulate();ev.summarize()
        ti=int(np.argmin(abs(ev.params.iouThrs-.75)));good=set()
        for entry in ev.evalImgs:
            if entry is not None and entry['aRng']==ev.params.areaRng[0]:
                good.update(int(g) for j,g in enumerate(entry['gtIds']) if not entry['gtIgnore'][j] and entry['gtMatches'][ti,j]>0)
        good &= ordinary.keys()
        if fresh and name == 'baseline':
            base_good = good.copy()
        results[name]=dict(metrics=dict(zip(names,map(float,ev.stats))),predictions=len(pred),matched75=len(good),
            repaired75=len(good-base_good),damaged75=len(base_good-good),size_recall75={g:dict(gt=sum(size(v)==g for v in ordinary.values()),
            matched=sum(size(ordinary[i])==g for i in good)) for g in ('small','medium','large')},**specs[name])
        matched[name]=sorted(good);save(a.out/'RESULTS.json',results);save(a.out/'MATCHED_GT75.json',matched)
        progress=dict(stage='score',mode=name,mask_ap=100*ev.stats[0],elapsed_s=time.monotonic()-start)
        save(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        del pred,dt,ev;gc.collect()
    save(a.out/'COMPLETE.json',dict(images=len(val_ids),evaluated=list(specs),elapsed_s=time.monotonic()-start,
        reference=str(a.reference), fresh_controls=fresh,
        note='Matched head evaluation; no new val-based strength selection.'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('train','evaluate','smoke'))
    for key in ('reference','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('bank','data','weights','manifest'):p.add_argument('--'+key,type=Path)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--mode',choices=MODES)
    p.add_argument('--objective',choices=('plain','guard','safe'),default='plain')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='train':train(a)
    else:evaluate(a,smoke=a.action=='smoke')


if __name__=='__main__':main()
