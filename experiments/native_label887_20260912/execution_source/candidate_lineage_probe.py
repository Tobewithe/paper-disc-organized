"""Frozen COCO diagnostic: exact NMS lineage, all post-conf masks, same-P swaps.

GT chooses alternatives only in explicitly labelled oracle diagnostics. No training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import contextlib
import csv
import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops, nms
from frozen_mechanism_probe import ROOT, sha, write_json, box_iou
from structure_candidate_trace import save_csv
from relative_ownership_experiment import spatial_and_predictions

ARMS = ['original', 'coefficient_only', 'box_only', 'both']
STAGES = ['score', 'nms', 'top300', 'nonempty', 'eval100']


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


@torch.inference_mode()
def replay_nms(boxes, scores, classes, source_ids, expected):
    """Actual first-selected suppressor using installed TorchNMS arithmetic/order."""
    xy = torch.as_tensor(boxes[source_ids], device='cuda')
    xy = xy + torch.as_tensor(classes[source_ids], device='cuda')[:, None] * 7680
    sc = torch.as_tensor(scores[source_ids], device='cuda')
    area = (xy[:, 2]-xy[:, 0])*(xy[:, 3]-xy[:, 1])
    order = sc.argsort(descending=True)
    parent = np.full(len(boxes), -2, dtype=np.int64)
    keeps = []
    while order.numel():
        i = order[0]
        raw_i = int(source_ids[int(i)])
        keeps.append(raw_i)
        parent[raw_i] = -1
        if order.numel() == 1:
            break
        rest = order[1:]
        wh = (torch.minimum(xy[i, 2:], xy[rest, 2:])-
              torch.maximum(xy[i, :2], xy[rest, :2])).clamp_(min=0)
        inter = wh[:, 0]*wh[:, 1]
        overlap = inter/(area[i]+area[rest]-inter)
        removed = overlap > .7
        parent[source_ids[rest[removed].cpu().numpy()]] = raw_i
        order = rest[~removed]
    need(np.array_equal(keeps, expected), 'NMS ordered raw IDs do not replay exactly')
    need(np.all(parent[source_ids] >= -1), 'Unaccounted post-conf candidate')
    return parent


def matching_size(edges):
    """Maximum cardinality, one unique candidate per ordinary GT; no SciPy."""
    owner = {}
    def augment(g, seen):
        for c in edges[g]:
            c = int(c)
            if c in seen:
                continue
            seen.add(c)
            if c not in owner or augment(owner[c], seen):
                owner[c] = g
                return True
        return False
    for g in sorted(range(len(edges)), key=lambda k:len(edges[k])):
        augment(g, set())
    return len(owner)


def pair_possible(a, b):
    return bool(len(a) and len(b) and (len(a)>1 or len(b)>1 or a[0]!=b[0]))


@torch.inference_mode()
def decode(proto, coeff, boxes, input_shape):
    return ops.process_mask(proto, torch.as_tensor(coeff, device='cuda'),
                            torch.as_tensor(boxes, device='cuda'), input_shape, upsample=True)


def process_image(gt, metadata, prior, iid, source, cache, out):
    tic = time.monotonic()
    rawfile = source/'raw'/f'{iid}.npz'
    cachefile = cache/f'{iid}.npz'
    with np.load(rawfile) as z:
        raw = {k:z[k] for k in z.files}
    with np.load(cachefile) as z:
        item = {k:z[k] for k in z.files}
    scores = raw['class_scores'].max(1)
    classes = raw['class_scores'].argmax(1)
    source_ids = np.flatnonzero(scores > .001)
    final = raw['nonempty_indices']
    shape = tuple(map(int, item['shape']))
    inp = tuple(map(int, item['input_shape']))
    for key, value in [('coeff',raw['coefficients'][final]), ('boxes',raw['boxes_input'][final]),
                       ('detections',np.column_stack([raw['boxes_original'][final],scores[final],classes[final]]))]:
        need(np.array_equal(item[key], value), f'Raw-to-final identity failed {iid} {key}')
    need(np.array_equal(raw['mapping_gt'],item['mapping_gt']) and
         np.array_equal(raw['mapping_pred'],item['mapping_pred']), 'Fixed bbox ownership mismatch')
    parent = replay_nms(raw['boxes_input'], scores, classes, source_ids, raw['nms_indices'])
    ordinary = [gt.anns[int(aid)] for aid in raw['annotation_ids']]
    need({a['id'] for a in ordinary} == {a['id'] for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)}, 'GT loss')
    categories = sorted(gt.cats)
    labels = np.array([categories.index(a['category_id']) for a in ordinary])
    gtrles = [gt.annToRLE(a) for a in ordinary]
    proto = torch.as_tensor(item['proto'],device='cuda')
    quality = np.zeros((len(ordinary),len(source_ids)),dtype=np.float64)
    raw_to_col = {int(r):i for i,r in enumerate(source_ids)}
    baseline = decode(proto,item['coeff'],item['boxes'],inp)
    replay_count = 0
    for first in range(0,len(source_ids),32):
        ids = source_ids[first:first+32]
        binary = decode(proto,raw['coefficients'][ids],raw['boxes_input'][ids],inp)
        for j,src in enumerate(ids):
            where = np.flatnonzero(final == src)
            if len(where):
                need(torch.equal(binary[j],baseline[int(where[0])]), 'Decoded final mask differs by batch')
                replay_count += 1
        orig = (ops.scale_masks(binary[:,None],shape)[:,0] > .5).cpu().numpy().astype(np.uint8)
        rles = [mu.encode(np.asfortranarray(m)) for m in orig]
        if ordinary:
            quality[:,first:first+len(ids)] = np.asarray(mu.iou(rles,gtrles,[0]*len(gtrles))).T
    need(replay_count == len(final), 'Not all final masks replayed')
    same_class = labels[:,None] == classes[source_ids][None,:]
    pools = {'score':source_ids, **{s:raw[s+'_indices'] for s in STAGES[1:]}}
    edges = {}
    image_counts = {'image_id':iid,'gt':len(ordinary),'postconf':len(source_ids),'final':len(final),
                    'nms_order_exact':True,'final_mask_replay_count':replay_count}
    for stage,pool in pools.items():
        columns = np.array([raw_to_col[int(j)] for j in pool],dtype=int)
        edges[stage] = [pool[(quality[g,columns]>=.75)&same_class[g,columns]] for g in range(len(ordinary))]
        image_counts[stage+'_max_unique75'] = matching_size(edges[stage])
        image_counts[stage+'_nonexclusive75'] = sum(bool(len(e)) for e in edges[stage])
    bbox_owners = [np.flatnonzero((raw['bbox_iou'][:,j]>=.5)&(labels==classes[j])) for j in range(len(classes))]
    def relation(j):
        p = int(parent[j])
        if p < 0:
            return 'not_suppressed'
        a,b = bbox_owners[j],bbox_owners[p]
        if len(a)!=1 or len(b)!=1:
            return 'ambiguous_or_unassigned'
        return 'same_gt' if a[0]==b[0] else 'different_gt'
    gtrows=[]
    for g,ann in enumerate(ordinary):
        col = np.flatnonzero(same_class[g])
        best = int(col[np.argmax(quality[g,col])]) if len(col) else None
        bestsrc = int(source_ids[best]) if best is not None else -1
        p = int(parent[bestsrc]) if bestsrc >= 0 else -2
        row = dict(image_id=iid,annotation_id=ann['id'],category_id=ann['category_id'],area=ann['area'],
                   ici=float(metadata[ann['id']]['ici_same']),official_mask75=prior[ann['id']],
                   best_mask_source=bestsrc,best_mask_iou=float(quality[g,best]) if best is not None else 0.,
                   best_mask_box_iou=float(raw['bbox_iou'][g,bestsrc]) if bestsrc>=0 else 0.,
                   best_mask_score=float(scores[bestsrc]) if bestsrc>=0 else 0.,
                   best_mask_suppressor=p,best_mask_suppression_relation=relation(bestsrc) if bestsrc>=0 else 'no_candidate',
                   suppressor_mask_iou=float(quality[g,raw_to_col[p]]) if p>=0 else None,
                   suppressor_box_iou=float(raw['bbox_iou'][g,p]) if p>=0 else None)
        for stage,pool in pools.items():
            selected = [raw_to_col[int(j)] for j in pool if classes[j] == labels[g]]
            row[stage+'_best_mask_iou'] = float(quality[g,selected].max()) if selected else 0.
            row[stage+'_mask75'] = bool(len(edges[stage][g]))
            row[stage+'_box75'] = bool(np.any((raw['bbox_iou'][g,pool]>=.75)&(classes[pool]==labels[g])))
        gtrows.append(row)
    pairrows=[]
    for g,a in enumerate(ordinary):
        for h in range(g+1,len(ordinary)):
            b=ordinary[h]
            if labels[g]!=labels[h] or box_iou(a['bbox'],b['bbox'])<=.05:
                continue
            pairrows.append(dict(image_id=iid,annotation_a=a['id'],annotation_b=b['id'],
                ici=max(float(metadata[q['id']]['ici_same']) for q in [a,b]),
                **{s+'_two_distinct75':pair_possible(edges[s][g],edges[s][h]) for s in STAGES}))
    # GT selects the actual candidate with best uncrowded-unmodified COCO mask IoU,
    # within correct argmax class, conf>.001, box IoU>=.5. Include original first for ties.
    mapping = dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
    gtindex = {a['id']:g for g,a in enumerate(ordinary)}
    alt = final.copy()
    choices=[]
    for aid,j in mapping.items():
        g = gtindex[aid]
        pool = source_ids[same_class[g] & (raw['bbox_iou'][g,source_ids]>=.5)]
        original = int(final[j])
        need(original in pool, 'Fixed match original excluded from replacement pool')
        pool = np.concatenate(([original],pool[pool!=original]))
        cols = [raw_to_col[int(v)] for v in pool]
        selected = int(pool[int(np.argmax(quality[g,cols]))])
        alt[j] = selected
        choices.append(dict(image_id=iid,annotation_id=aid,ici=float(metadata[aid]['ici_same']),
            prediction_slot=j,original_source=original,alternative_source=selected,eligible_candidates=len(pool),
            original_mask_iou=float(quality[g,raw_to_col[original]]),alternative_mask_iou=float(quality[g,raw_to_col[selected]]),
            original_box_iou=float(raw['bbox_iou'][g,original]),alternative_box_iou=float(raw['bbox_iou'][g,selected]),
            alternative_suppressor=int(parent[selected]),alternative_relation=relation(selected)))
    image_counts['reused_alternative_sources'] = len(alt)-len(set(map(int,alt)))
    # Preserve prediction slots, category and score in every arm. Shared alternative
    # IDs are allowed and logged; official rematching penalizes duplicate detections.
    masks = {'original':baseline,
             'coefficient_only':decode(proto,raw['coefficients'][alt],item['boxes'],inp),
             'box_only':decode(proto,item['coeff'],raw['boxes_input'][alt],inp),
             'both':decode(proto,raw['coefficients'][alt],raw['boxes_input'][alt],inp)}
    rows,predictions,skipped = spatial_and_predictions(gt,iid,item,masks,metadata,True)
    for arm,pp in predictions.items():
        with gzip.open(out/'predictions'/arm/f'{iid}.json.gz','wt',encoding='utf-8') as f:
            json.dump(pp,f,separators=(',',':'))
    np.savez_compressed(out/'images'/f'{iid}.npz',annotation_ids=raw['annotation_ids'],source_ids=source_ids,
        mask_iou=quality,first_suppressor=parent,alternative_sources=alt,
        original_sources=final,source_scores=scores,source_classes=classes)
    image_counts['seconds'] = time.monotonic()-tic
    write_json(out/'images'/f'{iid}.json',dict(counts=image_counts,gt=gtrows,pairs=pairrows,choices=choices,
        spatial=rows,skipped=skipped,input_hashes={'raw':sha(rawfile),'cache':sha(cachefile)}))
    return image_counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--images',type=int,default=300)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--budget-seconds',type=float,default=1800)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=ROOT/'diagnostics/structure_main300_20260911'
    cache=ROOT/'diagnostics/full_val_cache_20260911/val'
    ids=json.loads((source/'protocol.json').read_text())['images'][:a.images]
    protocol=dict(images=ids,training=False,evaluation_type='real_gt_oracle_diagnostic',arms=ARMS,
        script_sha256=sha(__file__),source_protocol_sha256=sha(source/'protocol.json'),
        helper_hashes={name:sha(Path(__file__).with_name(name)) for name in ['frozen_mechanism_probe.py','relative_ownership_experiment.py','structure_candidate_trace.py']},
        official_source_hashes={'ops':sha(ops.__file__),'nms':sha(nms.__file__)},
        annotation_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),
        metadata_sha256=sha(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'),
        scope='Reused exploratory dense-enriched 300 COCO val images. All ordinary GT and crowd retained in official task evaluation. No novel held-out confirmation.',
        candidate_pool='All raw points with argmax class score>.001; no GT prefilter for mask stage feasibility. One-to-one maximum cardinality pooled over all ordinary GT, no candidate budget at score/NMS stages. Not AP or achievable recall.',
        nms='Class offset 7680 in fp32, IoU>.7 suppression; exact ordered keep comparison mandatory. Parent is actual first selected suppressor.',
        alternative='Per fixed official bbox50 GT match, highest actual mask IoU candidate among same argmax category, score>.001, original-box IoU>=.5; original wins ties. Same P. Reused alternatives logged and allowed in original slots. No GT confidence modification. Not globally optimal assignment.',
        spatial='Fixed original bbox50 ownership; original pixels, crowd excluded, own GT priority. Selector uses ordinary COCO mask IoU without subtracting crowd; spatial IoU differs. Coverage and all wrong-region areas divided by own valid GT area.',
        caution='Oracle alternative may jointly favor box and coefficient. Four arms measure fixed selected-candidate swaps, not optimized per-factor ceilings. Retains scores and original slots; empty masks removed by official convention.')
    if a.resume:
        need(json.loads((a.out/'protocol.json').read_text())==protocol,'Resume protocol/hash mismatch')
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        (a.out/'images').mkdir()
        for arm in ARMS:
            (a.out/'predictions'/arm).mkdir(parents=True)
        write_json(a.out/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    metadata={int(r['annotation_id']):r for r in read(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv')}
    prior={int(r['annotation_id']):r['segm75']=='True' for r in read(source/'instances_same_iou.csv')}
    start=time.monotonic()
    for n,iid in enumerate(ids,1):
        if not (a.out/'images'/f'{iid}.json').exists():
            counts=process_image(gt,metadata,prior,iid,source,cache,a.out)
        else:
            counts=json.loads((a.out/'images'/f'{iid}.json').read_text())['counts']
        progress=dict(completed=n,total=len(ids),seconds=time.monotonic()-start,last_image=counts)
        write_json(a.out/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        if time.monotonic()-start>a.budget_seconds and n<len(ids):
            write_json(a.out/'PAUSED_BUDGET.json',progress)
            return
    agg={key:[] for key in ['counts','gt','pairs','choices','spatial','skipped']}
    for iid in ids:
        d=json.loads((a.out/'images'/f'{iid}.json').read_text())
        for key in agg:
            agg[key].extend([d[key]] if key=='counts' else d[key])
    for key,rows in agg.items():
        save_csv(a.out/f'{key}.csv',rows)
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),training=False,seconds=time.monotonic()-start,
        protocol_sha256=sha(a.out/'protocol.json'),hashes={p.name:sha(p) for p in a.out.glob('*.csv')}))


if __name__=='__main__':
    main()
