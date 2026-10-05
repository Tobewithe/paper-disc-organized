"""S036: frozen S032/S035 heads, fresh local forward, all COCO val2017.

No training or validation-based model/threshold selection. R@P90 is a descriptive
pooled operating-curve metric, not a threshold learned for deployment.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '4')
import argparse, contextlib, csv, gzip, io, json, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops, nms
from structure_candidate_trace import TraceCapture
from rich_pixel_readout import GlobalHead, instance_features
from train_readout_bias_controls import ConstantBias, InstanceBias
from readout_input_probe import sha
from eval_readout_input_pilot import ici

ROOT = Path(__file__).resolve().parent
ARMS = ['original', 'constant_fullbatch'] + [f'{m}_s{s}' for s in range(3)
        for m in ['constant', 'instance_bias', 'coefficient']]


def dump(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def save_gz(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=1) as f:
        json.dump(obj, f, separators=(',', ':'), allow_nan=False)
    tmp.replace(path)


def groups(value):
    return 'low' if value <= 1e-10 else 'middle' if value <= .5 + 1e-10 else 'high'


def evaluate(gt, meta, ids, predictions, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, 'segm'); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    t = int(np.flatnonzero(np.isclose(ev.params.iouThrs, .75))[0])
    records, scores, tp, fp = [], [], [], []
    for item in ev.evalImgs:
        if item is None or item['aRng'] != [0, 1e10] or item['maxDet'] != 100:
            continue
        score_map = dict(zip(item['dtIds'], item['dtScores']))
        valid = ~item['dtIgnore'][t].astype(bool)
        hit = item['dtMatches'][t] > 0
        scores.extend(np.asarray(item['dtScores'])[valid].tolist())
        tp.extend(hit[valid].tolist()); fp.extend((~hit[valid]).tolist())
        for j, aid in enumerate(item['gtIds']):
            if item['gtIgnore'][j]:
                continue
            ann = gt.anns[int(aid)]; matched = int(item['gtMatches'][t, j])
            v = meta[int(aid)]
            records.append(dict(arm=arm, image_id=item['image_id'], annotation_id=int(aid),
                category_id=ann['category_id'], area=ann['area'],
                area_bin='small' if ann['area'] < 1024 else 'medium' if ann['area'] < 9216 else 'large',
                ici=v, group=groups(v), hit75=bool(matched),
                matched_score75=float(score_map[matched]) if matched else None))
    if len(records) != len(meta) or len({r['annotation_id'] for r in records}) != len(meta):
        raise RuntimeError('Official all-GT denominator differs from raw ordinary annotations')
    scores = np.asarray(scores); order = np.argsort(-scores, kind='stable')
    scores = scores[order]
    ct = np.cumsum(np.asarray(tp, dtype=np.int64)[order]); cf = np.cumsum(np.asarray(fp, dtype=np.int64)[order])
    endpoints = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    precision = ct[endpoints] / (ct[endpoints] + cf[endpoints])
    qualifying = endpoints[(precision >= .90) & (ct[endpoints] > 0)]
    if len(qualifying):
        chosen = int(qualifying[-1]); threshold = float(scores[chosen])
        op = dict(attainable=True, threshold=threshold, precision=float(ct[chosen]/(ct[chosen]+cf[chosen])),
            recall=float(ct[chosen]/len(records)), tp=int(ct[chosen]), fp=int(cf[chosen]))
    else:
        threshold = None
        op = dict(attainable=False, threshold=None, precision=None, recall=0., tp=0, fp=0)
    for r in records:
        r['hit90'] = bool(threshold is not None and r['hit75'] and r['matched_score75'] >= threshold)
    if sum(r['hit90'] for r in records) != op['tp']:
        raise RuntimeError('P90 DT/GT matching count mismatch')
    row = dict(arm=arm, mask_ap=float(ev.stats[0]), mask_ap50=float(ev.stats[1]), mask_ap75=float(ev.stats[2]),
        predictions=len(predictions), gt=len(records), p90_threshold=threshold, p90_precision=op['precision'])
    for group in ['all', 'low', 'middle', 'high', 'nonhigh']:
        rr = [r for r in records if group == 'all' or (r['group'] != 'high' if group == 'nonhigh' else r['group'] == group)]
        row[f'n_{group}'] = len(rr)
        row[f'r75_{group}'] = float(np.mean([r['hit75'] for r in rr])) if rr else None
        row[f'r90_{group}'] = float(np.mean([r['hit90'] for r in rr])) if rr else None
    row['gap'] = row['r75_nonhigh'] - row['r75_high']
    row['gap_low_high'] = row['r75_low'] - row['r75_high']
    row['gap90'] = row['r90_nonhigh'] - row['r90_high']
    return row, records, op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--annotation', type=Path, required=True)
    ap.add_argument('--initialize-only', action='store_true')
    ap.add_argument('--reuse-from', type=Path)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    def progress(stage, **kw):
        row = dict(stage=stage, pid=os.getpid(), seconds=round(time.monotonic()-start, 2), **kw)
        dump(out/'progress.json', row); print(json.dumps(row), flush=True)
    source = ROOT/'diagnostics/shared_label_controls_20260912'
    bias_source = ROOT/'diagnostics/readout_bias_controls_20260912'
    weight = ROOT/'weights/yolo26m-seg.pt'
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(a.annotation))
    ids = sorted(gt.imgs)
    if len(ids) != 5000:
        raise RuntimeError('Expected all 5000 val2017 images')
    checkpoints = {f'{m}_s{s}': (source/f'raw_coco_s{s}' if m == 'coefficient' else bias_source/f'{m}_s{s}')/'checkpoints/epoch015.pt'
        for s in range(3) for m in ['constant', 'instance_bias', 'coefficient']}
    for path in checkpoints.values():
        if sha(path) != json.loads((path.parent.parent/'COMPLETE.json').read_text())['final_sha256']:
            raise RuntimeError('Saved model checkpoint changed')
    source_files = ['eval_frozen_readouts_val_v2.py', 'structure_candidate_trace.py', 'frozen_mechanism_probe.py',
        'rich_pixel_readout.py', 'train_readout_bias_controls.py', 'readout_input_probe.py', 'eval_readout_input_pilot.py']
    fingerprint = {str(p.resolve()): sha(p) for p in [weight, a.annotation, source/'normalizer.pt',
        bias_source/'constant_fullbatch.json', *checkpoints.values(), *[ROOT/n for n in source_files], Path(ops.__file__), Path(nms.__file__)]}
    protocol = dict(experiment='S036_FROZEN_FULL_VAL', image_ids=ids, image_root=str(a.images.resolve()),
        annotation=str(a.annotation.resolve()), arms=ARMS, source=str(source), bias_source=str(bias_source),
        fingerprint=fingerprint, training=False, checkpoint_rule='All saved final epoch015, seeds0/1/2; no retraining or selection.',
        inference=dict(imgsz=640, rect=False, conf=.001, iou=.7, max_det=300, half=False, retina_masks=False,
            device=0, end2end=False, tf32=False, torch=torch.__version__, ultralytics=ultralytics.__version__,
            ultralytics_path=ultralytics.__file__),
        candidate_scope='Original post-NMS top300 nonempty slots fixed; all arms use original scores/classes/boxes. '
            'Modified empty masks are dropped; cannot resurrect an originally empty candidate. Matches S032/S035.',
        decode='Stock process_mask coefficient+P -> bilinear640 -> >0 -> original box crop; '
            'bias added to original upsampled logit. Binary scale_masks to original dimensions then>.5.',
        evaluation='All5000images all ordinary annotations; official COCO segm AP atmaxDets100 perimage/category. '
            'R75 from all-GT official matching then ICIstratification. No GT used for inference.',
        density='Same-class sum of other ordinary GT bbox intersection area / own bbox area. '
            'low<=1e-10,middle(1e-10,.5+1e-10],high>.5+1e-10;nonhigh=low+middle. Not image screening.',
        p90='Descriptive pooled micro R@P>=.90 at maskIoU.75 over official sameclass matches/ignore. '
            'Sweep confidence including all tied scores, choose largest recall attainable with precision>=.90. '
            'One global threshold per arm shared by every density; no within-bin threshold fitting. '
            'Validation PR operating point, NOT a train-calibrated deployment threshold. '
            'If unattainable report recall0 and threshold/precision null. PrimaryCIs are R75, not P90 threshold uncertainty.',
        limitations='Project previously explored val2017; not pristine blind testing. '
            'Heads fitted on1200train2017images and explored on300othertrainimages. '
            'Frozen calibrations are ordinary controls, no novelty claim. No model/threshold chosen for future use from these results.',
        resume='Atomic complete perimage shards contain all11arms; verify fingerprint and shard hash receipt before reuse.')
    if a.reuse_from:
        parent = a.reuse_from.resolve()
        old = json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
        old_code_path = ROOT/'eval_frozen_readouts_val.py'
        if old['fingerprint'][str(old_code_path.resolve())] != sha(old_code_path):
            raise RuntimeError('Original executed source changed')
        previous_fp = {k:v for k,v in old['fingerprint'].items() if k!=str(old_code_path.resolve())}
        current_fp = {k:v for k,v in fingerprint.items() if k!=str(Path(__file__).resolve())}
        if previous_fp!=current_fp or {k:v for k,v in old.items() if k!='fingerprint'} != {k:v for k,v in protocol.items() if k!='fingerprint'}:
            raise RuntimeError('Only empty-index fix may differ from original protocol')
        protocol['continuation'] = dict(parent=str(parent),parent_protocol_sha256=sha(parent/'protocol.json'),
            parent_receipts_sha256=sha(parent/'image_receipts.json'),
            fix='Only zero-element NMS source index is cast to int64. Nonempty source indices untouched; no candidate or GT filtering change.',
            source_preserved=str(old_code_path),reason='Stock NMS returns float empty index for image560371; original stopped after4809savedimages.')
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise RuntimeError('Frozen protocol changed; use a separate run')
    else:
        dump(out/'protocol.json', protocol); (out/'source').mkdir()
        for name in source_files:
            shutil.copy2(ROOT/name, out/'source'/name)
    if a.initialize_only:
        progress('INITIALIZED', images=5000, arms=len(ARMS)); return
    if (out/'COMPLETE.json').exists():
        print('Already complete', flush=True); return
    missing = [gt.imgs[i]['file_name'] for i in ids if not (a.images/gt.imgs[i]['file_name']).is_file()]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} images; first={missing[0]}')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    normalizer = torch.load(source/'normalizer.pt', map_location='cuda', weights_only=True)
    heads = {}
    for name, path in checkpoints.items():
        cls = GlobalHead if name.startswith('coefficient') else InstanceBias if name.startswith('instance_bias') else ConstantBias
        model = cls().cuda(); model.load_state_dict(torch.load(path, map_location='cuda', weights_only=False)['model'])
        heads[name] = model.eval().requires_grad_(False)
    refbias = json.loads((bias_source/'constant_fullbatch.json').read_text())['bias']
    model = YOLO(str(weight)); model.model.eval().requires_grad_(False)
    head = model.model.model[-1]
    if head.end2end:
        raise RuntimeError('Unexpected runtime branch')
    maps, hooks = {}, []
    for level, branch in enumerate(head.cv4):
        def hook(module, args, output, level=level):
            maps[level] = args[0].detach().clone()
        hooks.append(branch[-1].register_forward_hook(hook))
    shard = out/'images'; shard.mkdir(exist_ok=True)
    categories = sorted(gt.cats)
    receipt_path = out/'image_receipts.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if a.reuse_from and not receipts:
        parent_receipts=json.loads((a.reuse_from/'image_receipts.json').read_text())
        for iid,record in parent_receipts.items():
            original_path=a.reuse_from/'images'/f'{iid}.json.gz'
            if sha(original_path)!=record['sha256']:raise RuntimeError('Parent shard changed')
            target=shard/original_path.name
            if target.exists():
                if sha(target)!=record['sha256']:raise RuntimeError('Existing reuse shard differs')
            else:os.link(original_path,target)
            receipts[iid]=record
        dump(receipt_path,receipts)
        dump(out/'CONTINUATION_RECEIPT.json',dict(reused=len(receipts),verified_sha256=True,
            parent=str(a.reuse_from.resolve()),source_index_fix_only=True))
        progress('RESUMED',images=len(receipts),total=5000)
    for number, iid in enumerate(ids, 1):
        path = shard/f'{iid}.json.gz'
        if str(iid) in receipts:
            if sha(path) != receipts[str(iid)]['sha256']:
                raise RuntimeError('Completed image shard changed')
            continue
        maps.clear(); image_path = a.images/gt.imgs[iid]['file_name']
        with torch.inference_mode():
            model.predict(str(image_path), predictor=TraceCapture, imgsz=640, rect=False, conf=.001,
                iou=.7, max_det=300, half=False, retina_masks=False, device=0, verbose=False)
            cap, raw, pre = model.predictor.capture, model.predictor.dense, model.predictor.before_empty
            allnms, idx = nms.non_max_suppression(raw.clone(), conf_thres=.001, iou_thres=.7,
                nc=80, max_det=8400, return_idxs=True, end2end=False)
            if not torch.equal(allnms[0][:300], pre):
                raise RuntimeError('NMS identity mismatch')
            pre_mask = ops.process_mask(cap['proto'], pre[:,6:], pre[:,:4], (640,640), upsample=True)
            keep = idx[0].flatten()[:300][pre_mask.flatten(1).any(1).bool()]
            if keep.numel()==0:
                keep=keep.to(dtype=torch.long)
            c, p, boxes, det = cap['coeff'], cap['proto'].float(), cap['boxes'], cap['detections']
            if not torch.equal(raw[0,84:,keep].T, c):
                raise RuntimeError('Source coefficient mismatch')
            h = torch.cat([maps[k][0].flatten(1).T for k in range(3)])[keep]
            lv = torch.cat([torch.full((maps[k].shape[-2]*maps[k].shape[-1],), k, device='cuda', dtype=torch.long)
                for k in range(3)])[keep]
            x = (instance_features(h, lv, boxes, (640,640))-normalizer['mean'])/normalizer['std']
            original = ops.process_mask(p,c,boxes,(640,640),upsample=True)
            z = F.interpolate((c@p.flatten(1)).reshape(1,len(c),*p.shape[-2:]), (640,640), mode='bilinear', align_corners=False)[0] if len(c) else c.new_empty((0,640,640))
            if not torch.equal(ops.crop_mask((z>0).byte(),boxes), original):
                raise RuntimeError('Baseline decode mismatch')
            payload, count = {}, {}
            detcpu = det.cpu().numpy()
            for arm in ARMS:
                if arm == 'original':
                    binary = original
                elif arm.startswith('coefficient'):
                    binary = ops.process_mask(p,c+heads[arm](x),boxes,(640,640),upsample=True)
                else:
                    b = c.new_full((len(c),),refbias) if arm=='constant_fullbatch' else heads[arm](x)
                    binary = ops.crop_mask((z+b[:,None,None]>0).byte(),boxes)
                nonempty = binary.flatten(1).any(1).bool().cpu().numpy()
                restored = (ops.scale_masks(binary[:,None],cap['shape'])[:,0]>.5).cpu().numpy() if len(c) else np.empty((0,*cap['shape']),bool)
                rows = []
                for j in np.flatnonzero(nonempty):
                    rle = mu.encode(np.asfortranarray(restored[j].astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    rows.append(dict(image_id=iid,category_id=categories[int(detcpu[j,5])],score=float(detcpu[j,4]),segmentation=rle))
                payload[arm]=rows; count[arm]=len(rows)
            payload['original_boxes'] = [dict(image_id=iid,category_id=categories[int(q[5])],score=float(q[4]),
                bbox=[float(q[0]),float(q[1]),float(q[2]-q[0]),float(q[3]-q[1])]) for q in detcpu]
            payload['witness'] = dict(image_id=iid,source_indices=keep.cpu().tolist(),slots=len(c),counts=count,
                jpeg_sha256=sha(image_path),source_replay=True,original_decoder_xor=0)
        save_gz(path,payload)
        receipts[str(iid)] = dict(sha256=sha(path),jpeg_sha256=payload['witness']['jpeg_sha256'],slots=len(c),counts=count)
        dump(receipt_path,receipts)
        if number%50==0 or number==5000:
            progress('INFERENCE',images=number,total=5000)
    for hook in hooks: hook.remove()
    del heads,model; torch.cuda.empty_cache()
    dump(out/'INFERENCE_COMPLETE.json',dict(images=5000,arms=ARMS,image_receipts_sha256=sha(receipt_path)))
    progress('EVALUATION',arms_completed=0,total=len(ARMS))
    meta={ann['id']:ici(ann,[q for q in gt.imgToAnns[ann['image_id']] if not q.get('iscrowd',0)])
        for ann in gt.anns.values() if not ann.get('iscrowd',0)}
    evaluation=out/'evaluation'; evaluation.mkdir(exist_ok=True)
    task=[]
    for arm in ARMS:
        marker=evaluation/f'{arm}.json'; records_path=evaluation/f'{arm}_gt.csv'
        if marker.exists():
            saved=json.loads(marker.read_text()); row=saved['summary']
            if sha(records_path)!=saved['records_sha256']:raise RuntimeError('Evaluation records changed')
        else:
            pp=[]
            for iid in ids:pp.extend(load_gz(shard/f'{iid}.json.gz')[arm])
            row,records,op=evaluate(gt,meta,ids,pp,arm)
            save_csv(records_path,records)
            dump(marker,dict(summary=row,p90=op,records_sha256=sha(records_path)))
            del pp,records
        task.append(row);save_csv(out/'task_summary.csv',task)
        progress('EVALUATION',arms_completed=len(task),total=len(ARMS),result=row)
    if not (out/'original_box_ap.json').exists():
        pp=[]
        for iid in ids:pp.extend(load_gz(shard/f'{iid}.json.gz')['original_boxes'])
        with contextlib.redirect_stdout(io.StringIO()):
            ev=COCOeval(gt,gt.loadRes(pp),'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        dump(out/'original_box_ap.json',dict(box_ap=float(ev.stats[0]),box_ap50=float(ev.stats[1]),box_ap75=float(ev.stats[2]),
            scope='Original fixed candidate boxes. All arms retain these box values; only modified empty-mask filtering may remove slots.'))
    dump(out/'COMPLETE.json',dict(status='COMPLETE',images=5000,ordinary_gt=len(meta),arms=ARMS,
        seconds=time.monotonic()-start,hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
        if p.is_file() and p.parent!=shard and p.name not in ['COMPLETE.json','progress.json','FAILED.json']}))
    progress('COMPLETE',images=5000,ordinary_gt=len(meta))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            if out.is_dir():dump(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
