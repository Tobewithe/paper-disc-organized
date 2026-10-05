"""Frozen epoch-3 native prototype screen; each variant uses its own full P."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

if '--config' in sys.argv:
    _cfg = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text(encoding='utf-8-sig'))
    sys.path.insert(0, _cfg['source_python'])

import torch
from pycocotools.coco import COCO
from online_runtime import FrozenReplay, dump, load_asset, load_index, load_json, resolve_runtime_config, sha256, tensor_sha
from native_proto_model import NativeProtoModel
import evaluation_metrics as em

ARMS = ('A', 'N', 'P', 'PC')
METRICS = ('iou', 'mask75', 'coverage', 'auc', 'fpr')
PAIRS = (('P','A'), ('PC','A'), ('P','N'), ('PC','N'), ('PC','P'))
em.ARMS = ARMS


def identity(row):
    return tuple(row[k] for k in ('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx'))


def buffer_hashes(model):
    return {n:tensor_sha(t) for n,t in model.named_buffers()}


def history_rows(path, items):
    images = {int(x['image_id']) for x in items}
    rows = {}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] != 'dev' or int(row['image_id']) not in images:
                continue
            key = identity(row)
            if key in rows:
                raise AssertionError('Duplicate historical reference identity')
            for arm in ('A','N'):
                for metric in METRICS:
                    if f'{metric}_{arm}' not in row:
                        raise AssertionError(f'Historical {metric}_{arm} is missing')
            rows[key] = row
    if len(rows) != sum(int(x['n']) for x in items):
        raise AssertionError('Historical A/N reference does not cover all fixed candidates')
    return rows


def checkpoints(cfg):
    if 'checkpoints' in cfg:
        return {m:Path(cfg['checkpoints'][m]) for m in ('P','PC')}
    runs = load_json(cfg.get('runs_file',str(Path(cfg['server_root'])/'RUN_IDS.json')))['runs']
    return {m:Path(cfg['server_root'])/'runs'/runs[m]/'final.pt' for m in ('P','PC')}


def summarize(rows, out, population, audit):
    predicates = {'all':lambda r:True, 'box_good_mask_bad':lambda r:r['box_good_mask_bad'],
        'original_success':lambda r:bool(r['mask75_A']), 'original_failure':lambda r:not bool(r['mask75_A'])}
    tables = {}; per_image = []
    for name,predicate in predicates.items():
        subset = [r for r in rows if predicate(r)]; groups = em.image_groups(subset)
        table = dict(images=len(groups), candidates=len(subset), candidate={},image_macro={},undefined={},comparisons={})
        for metric in METRICS:
            table['candidate'][metric] = {a:em.avg([r.get(f'{metric}_{a}') for r in subset]) for a in ARMS}
            table['image_macro'][metric] = {a:em.avg([em.avg([r.get(f'{metric}_{a}') for r in g]) for g in groups]) for a in ARMS}
            table['undefined'][metric] = {a:sum(not em.finite(r.get(f'{metric}_{a}')) for r in subset) for a in ARMS}
        for arm,ref in PAIRS:
            comparison = {m:em.paired(groups,arm,ref,m,20261003,1000) for m in METRICS}
            repair = sum(r[f'mask75_{ref}']==0 and r[f'mask75_{arm}']==1 for r in subset)
            damage = sum(r[f'mask75_{ref}']==1 and r[f'mask75_{arm}']==0 for r in subset)
            comparison['crossings'] = dict(repair=repair,damage=damage,net=repair-damage)
            table['comparisons'][f'{arm}_minus_{ref}'] = comparison
        tables[name] = table
        for g in groups:
            item = dict(split='dev',image_id=g[0]['image_id'],group=name,candidates=len(g))
            for metric in METRICS:
                for arm in ARMS:
                    item[f'{metric}_{arm}'] = em.avg([r.get(f'{metric}_{arm}') for r in g])
                    item[f'{metric}_{arm}_defined'] = sum(em.finite(r.get(f'{metric}_{arm}')) for r in g)
                for arm,ref in PAIRS:
                    item[f'delta_{metric}_{arm}_minus_{ref}'] = em.avg([r[f'{metric}_{arm}']-r[f'{metric}_{ref}']
                        for r in g if em.finite(r.get(f'{metric}_{arm}')) and em.finite(r.get(f'{metric}_{ref}'))])
            per_image.append(item)
    em.append_rows(out/'PER_IMAGE.jsonl',per_image)
    result = dict(schema='native-prototype-coadaptation-screen-v1',arms=ARMS,tables=tables,population=population,
        evaluation=audit,bootstrap=dict(draws=1000,unit='paired image',seed=20261003),
        no_new_blind_test=True,no_COCO_AP=True,fixed_prototype=False,automatic_followup=False)
    dump(out/'SUMMARY.json',em.clean(result))
    def pp(value):return 'undefined' if value is None else f'{value*100:+.4f}'
    lines = ['# Native prototype/coefficient coadaptation screen','',
        'P and PC are evaluated with their own trained full prototypes. A uses original P and c0; N reuses the same-candidate epoch-3 native-coefficient control from the prior screen.', '',
        f"{population['planned_images']} planned / {population['effective_images']} effective dev images; {len(rows)} fixed official candidates. These development images were previously viewed; no blind confirmation or COCO AP claim.", '',
        'All differences are percentage points. The 1000 paired-image bootstrap intervals are descriptive; candidate-mean intervals also resample whole images.', '',
        '| Group | Comparison | Macro IoU Δ [95% CI] | Candidate IoU Δ | Net Mask75 | Repair/damage | Coverage Δ | AUC Δ | FPR Δ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,table in tables.items():
        for name,c in table['comparisons'].items():
            d=c['iou']['image_macro']; lo,hi=d['ci95']; x=c['crossings']
            lines.append(f"| {group} | {name} | {pp(d['delta'])} [{pp(lo)}, {pp(hi)}] | {pp(c['iou']['candidate']['delta'])} | {x['net']} | {x['repair']}/{x['damage']} | {pp(c['coverage']['image_macro']['delta'])} | {pp(c['auc']['image_macro']['delta'])} | {pp(c['fpr']['image_macro']['delta'])} |")
    lines += ['', 'Evaluation scope:', '',
        '- A is always the unchanged original model. Helper evaluations with variant P also compute an internal reference; all those reference metrics and classifications are discarded.',
        '- P trains the native prototype path with original c0; PC jointly trains the native prototype and coefficient paths. The inference prefix, original boxes, assignment and candidate identities stay fixed.',
        '- N is joined from the historical screen by all seven identity fields; it is not rerun or retrained. Same checkpoint budget is epoch 3, with no dev checkpoint selection.',
        '- Normal masks use process_mask(upsample=True), original predicted boxes, original zero threshold and true letterbox inverse. IoU/coverage use original COCO masks. AUC/FPR use continuous input-grid logits on fixed predicted-box support.',
        '- Original success/failure and box-good/mask-bad groups are defined exclusively by original A. No candidate is removed because it worsens or produces an empty mask; undefined AUC does not remove IoU.',
        '- Prototypes change in this study, so these results do not establish fixed-P coefficient reachability. Three-epoch negative outcomes do not disprove the mechanism. Consider useful individual metric improvements together with their costs; not all metrics must improve.',
        '- No threshold search, gate, new GT-based selection, full AP evaluation or automatic next experiment.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def run(args):
    started=time.monotonic(); cfg=resolve_runtime_config(load_json(args.config))
    deadline=started+float(cfg.get('max_evaluation_seconds',1200))
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('Fixed evaluation budget reached; retain incomplete Run')
    if os.name=='nt' or not torch.cuda.is_available():
        raise RuntimeError('Run only on the authorized Linux GPU server')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'PER_CANDIDATE.jsonl').exists() or (out/'COMPLETE.json').exists():
        raise RuntimeError('Use an independent Run; existing results are immutable')
    items=load_index(cfg)['dev']; planned=list(map(int,load_json(cfg['split'])['dev']))
    if len(items)!=256 or [int(i['image_id']) for i in items]!=planned:
        raise AssertionError('Fixed dev identity/order changed')
    population=dict(planned_images=len(items),effective_images=sum(bool(i['n']) for i in items),
        candidates=sum(int(i['n']) for i in items),no_positive_images=[i['image_id'] for i in items if not i['n']])
    historical=history_rows(cfg['original_screen_results'],items)
    replay=FrozenReplay(cfg); models={};paths=checkpoints(cfg);hashes={};buffers={}
    for mode,path in paths.items():
        receipt=load_json(path.parent/'COMPLETE.json')
        if receipt.get('completed') is not True or path.name!='final.pt':
            raise AssertionError(f'{mode}: unfinished arm or nonfinal checkpoint')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('mode')!=mode or ck.get('epoch')!=3:
            raise AssertionError('Only fixed final epoch 3 is eligible')
        if resolve_runtime_config(ck['config'])!=cfg or list(ck['feature_channels'])!=list(replay.feature_channels):
            raise AssertionError('Checkpoint configuration or feature channels differ')
        model=NativeProtoModel(replay,mode).to(replay.device).float().eval()
        original_buffers=buffer_hashes(model)
        model.load_state_dict(ck['state_dict'],strict=True); model.requires_grad_(False);model.eval()
        model.assert_frozen_buffers()
        buffers[mode]=buffer_hashes(model)
        for name,value in original_buffers.items():
            if any(s in name for s in ('running_mean','running_var','num_batches_tracked')) and buffers[mode][name]!=value:
                raise AssertionError(f'{mode}: BN running state changed from original checkpoint: {name}')
        models[mode]=model;hashes[mode]=sha256(path)
    inputs=dict(config_sha256=sha256(args.config),split_sha256=sha256(cfg['split']),
        original_weights_sha256=replay.weights_sha256,historical_rows_sha256=sha256(cfg['original_screen_results']),
        cache_index_sha256=sha256(Path(cfg['cache'])/'INDEX.json'),
        cache_identity_sha256=sha256(Path(cfg['cache'])/'CACHE_IDENTITY.json'),checkpoint_sha256=hashes,
        historical_N_checkpoint_sha256=sha256(cfg['original_N_checkpoint']))
    dump(out/'INPUT_MANIFEST.json',inputs);dump(out/'EVALUATION_INDEX.json',dict(dev=items,population=population))
    dump(out/'ENVIRONMENT.json',replay.import_info)
    coco=COCO(str(cfg['annotations_train']));rows=[];seen=set();ref_error=0.
    for position,entry in enumerate(items):
        check_time()
        if not entry['n']:continue
        x=load_asset(cfg,int(entry['image_id']),verify=True)
        if x['split']!='dev' or len(x['rows'])!=int(entry['n']):raise AssertionError('Population changed')
        gpu=dict(x,proto=x['proto'].to(replay.device),c0=x['c0'].to(replay.device),boxes=x['boxes'].to(replay.device))
        expected=[identity(r) for r in x['rows']]
        if len(set(expected))!=len(expected) or seen.intersection(expected):raise AssertionError('Duplicate identity')
        original=em.evaluate_image(gpu,{'A':gpu['c0']},coco,chunk_size=4)
        if [identity(r) for r in original]!=expected:raise AssertionError('Baseline decoder changed identities')
        for r in original:
            old=historical[identity(r)]
            for metric in METRICS:
                a,b=r[f'{metric}_A'],old[f'{metric}_A']
                if em.finite(a)!=em.finite(b):raise AssertionError('A definedness changed')
                if em.finite(a):
                    ref_error=max(ref_error,abs(a-b))
                    if abs(a-b)>1e-12:raise AssertionError(f'A {metric} no longer reproduces reference')
                r[f'{metric}_N']=old[f'{metric}_N']
            r['historical_N_same_identity']=True
        features=replay.replay([x])
        model_input={k:x[k] for k in ('image_id','raw_ids','c0')}
        for mode,model in models.items():
            with torch.no_grad():prediction=model(features,[model_input])
            c=prediction['coefficients'][0];p=prediction['prototypes'][0]
            if c.shape!=gpu['c0'].shape or tuple(p.shape)!=(32,160,160) or not torch.isfinite(c).all() or not torch.isfinite(p).all():
                raise AssertionError('Variant returned invalid full prototype or coefficient')
            if mode=='P':torch.testing.assert_close(c,gpu['c0'],atol=0,rtol=0)
            # em requires A=c0 under its supplied P. That internal reference is
            # deliberately thrown away: it is NOT the original model baseline.
            variant=em.evaluate_image(dict(gpu,proto=p),{'A':gpu['c0'],mode:c},coco,chunk_size=4)
            if [identity(r) for r in variant]!=expected:raise AssertionError('Variant decoder changed identities')
            prototype_hash=tensor_sha(p)
            for r,v in zip(original,variant):
                for metric in METRICS:r[f'{metric}_{mode}']=v[f'{metric}_{mode}']
                r[f'empty_mask_{mode}']=v[f'empty_mask_{mode}']
                r[f'prototype_sha256_{mode}']=prototype_hash
                r[f'coefficient_source_{mode}']='original c0' if mode=='P' else 'trained native one2one_cv4'
            if buffer_hashes(model)!=buffers[mode]:raise AssertionError('Model buffers changed during evaluation')
        em.append_rows(out/'PER_CANDIDATE.jsonl',original);rows.extend(original);seen.update(expected)
        if position%20==0 or position+1==len(items):
            progress=dict(images_processed=position+1,planned_images=256,candidates=len(rows),elapsed_s=time.monotonic()-started)
            dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    if len(rows)!=population['candidates'] or seen!=set(historical):raise AssertionError('Fixed evaluation is incomplete')
    check_time();replay.assert_unchanged()
    audit=dict(population=population,checkpoints={k:str(v) for k,v in paths.items()},inputs=inputs,
        original_A_max_reference_error=ref_error,original_source_unchanged=True,BN_original_and_buffers_unchanged=True,
        model_input_GT_free_allowlist=['image_id','raw_ids','c0'],historical_N_reused=True,
        variant_prototypes_used=True,variant_helper_A_metrics_discarded=True,fixed_epoch=3,
        original_A_defines_strata=True,no_new_blind_test=True,no_COCO_AP=True)
    dump(out/'EVALUATION_AUDIT.json',audit);summarize(rows,out,population,audit);check_time()
    dump(out/'COMPLETE.json',dict(completed=True,passed=True,candidates=len(rows),arms=ARMS,
        elapsed_s=time.monotonic()-started,automatic_followup=False,scientific_scope='developmental screen'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        payload=dict(completed=False,passed=False,error=repr(exc),traceback=traceback.format_exc(),incomplete=True)
        dump(Path(args.out)/'FAILURE.json',payload);dump(Path(args.out)/'COMPLETE.json',payload);raise

if __name__=='__main__':main()
