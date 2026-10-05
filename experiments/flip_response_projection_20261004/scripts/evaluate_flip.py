"""Server-only whole-canvas reflection and full-response coefficient projection."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from pathlib import Path
if '--config' in sys.argv:
    _config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text(encoding='utf-8-sig'))
    sys.path.insert(0,_config['source_python'])
import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops
from online_runtime import dump,load_json,load_asset,load_index,resolve_runtime_config,sha256,tensor_sha
from flip_source import FlipSource
from full_projection import prepare_prototype,solve_full_projection
import evaluation_metrics as em

ARMS=('A','ORIG_MATCH','MIRROR','MATCH','MIX','PROJ')
PAIRS=(('MIX','A'),('PROJ','A'),('PROJ','MIX'),('MATCH','A'),('MATCH','ORIG_MATCH'),
       ('MATCH','MIRROR'),('MIRROR','A'),('ORIG_MATCH','A'),('MIX','MATCH'))
METRICS=('iou','mask75','coverage','auc','fpr')
KEYS=('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx')
em.ARMS=ARMS

def key(row):return tuple(row[k] for k in KEYS)
def ensure_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Fixed resource limit; retain incomplete Run, never shrink cohort')

@torch.no_grad()
def direct_metrics(logits,image,j,coco,annotation_id):
    if logits.shape!=(640,640) or not torch.isfinite(logits).all():raise AssertionError('Invalid continuous source')
    support=ops.crop_mask(torch.ones((1,640,640),device=logits.device),image['boxes'][j:j+1].to(logits.device))[0].bool()
    binary=(logits.gt(0)&support).byte()[None]
    manual=ops.crop_mask(logits[None].clone(),image['boxes'][j:j+1].to(logits.device)).gt(0).byte()
    if not torch.equal(binary,manual):raise AssertionError('Crop/threshold baseline mismatch')
    mask=em._scale_binary(binary,tuple(image['original_shape']),image['ratio_pad'])[0]
    ann=coco.anns[int(annotation_id)]
    if int(ann['image_id'])!=int(image['image_id']):raise AssertionError('Annotation identity changed')
    gt=torch.as_tensor(coco.annToMask(ann).astype(bool),device=logits.device)
    intersection=int((mask&gt).sum());iou=intersection/max(1,int((mask|gt).sum()))
    padded=em._padded_gt(gt,image['ratio_pad'],(640,640))
    auc,fpr=em._pixel_auc_fpr(logits,padded,support)
    return dict(iou=iou,mask75=int(iou>=.75),coverage=intersection/max(1,int(gt.sum())),
                auc=auc,fpr=fpr,empty_mask=not bool(mask.any()))

def summarize(rows,out,population):
    groupspec={'all':lambda r:True,'box_good_mask_bad':lambda r:bool(r['box_good_mask_bad']),
        'original_success':lambda r:bool(r['mask75_A']),'original_failure':lambda r:not r['mask75_A']}
    for level in range(3):groupspec['P'+str(3+level)]=lambda r,l=level:r['pyramid_level']==l
    for size in ('small','medium','large'):groupspec[size]=lambda r,s=size:r['size_group']==s
    tables={};per_image=[]
    for name,pred in groupspec.items():
        selected=[r for r in rows if pred(r)];groups=em.image_groups(selected)
        t=dict(images=len(groups),candidates=len(selected),candidate={},image_macro={},undefined={},comparisons={})
        for metric in METRICS:
            t['candidate'][metric]={a:em.avg([r.get(f'{metric}_{a}') for r in selected]) for a in ARMS}
            t['image_macro'][metric]={a:em.avg([em.avg([r.get(f'{metric}_{a}') for r in g]) for g in groups]) for a in ARMS}
            t['undefined'][metric]={a:sum(not em.finite(r.get(f'{metric}_{a}')) for r in selected) for a in ARMS}
        for arm,ref in PAIRS:
            pair={m:em.paired(groups,arm,ref,m,20261004,1000) for m in METRICS}
            repair=sum(r[f'mask75_{arm}']==1 and r[f'mask75_{ref}']==0 for r in selected)
            damage=sum(r[f'mask75_{arm}']==0 and r[f'mask75_{ref}']==1 for r in selected)
            pair['crossings']=dict(repair=repair,damage=damage,net=repair-damage)
            t['comparisons'][arm+'_minus_'+ref]=pair
        for g in groups:
            item=dict(split='dev',image_id=g[0]['image_id'],group=name,candidates=len(g))
            for m in METRICS:
                for arm in ARMS:item[m+'_'+arm]=em.avg([r.get(m+'_'+arm) for r in g])
                for arm,ref in PAIRS:item[f'delta_{m}_{arm}_minus_{ref}']=em.avg([r[m+'_'+arm]-r[m+'_'+ref] for r in g if em.finite(r.get(m+'_'+arm)) and em.finite(r.get(m+'_'+ref))])
            per_image.append(item)
        tables[name]=t
    em.append_rows(out/'PER_IMAGE.jsonl',per_image)
    decisions={}
    for arm in ('MIX','PROJ'):
        all_=tables['all']['comparisons'][arm+'_minus_A'];target=tables['box_good_mask_bad']['comparisons'][arm+'_minus_A']
        d=all_['iou']['image_macro'];dt=target['iou']['image_macro']
        overall=em.finite(d['delta']) and d['delta']>=.002 and d['ci95'][0]>0 and all_['crossings']['net']>0
        subgroup=em.finite(dt['delta']) and dt['delta']>=.005 and dt['ci95'][0]>0 and d['delta']>=-.001
        decisions[arm]=dict(prespecified_overall_signal=bool(overall),prespecified_target_signal=bool(subgroup),
                            other_metrics_require_effect_tradeoff_review=True)
    result=dict(schema='whole-canvas-flip-full-response-projection-v1',population=population,arms=ARMS,tables=tables,
        bootstrap=dict(draws=1000,seed=20261004,unit='paired image'),decisions=decisions,
        no_training=True,scope='Historical dev / fixed GT-conditioned official candidates; not blind test or COCO AP',
        automatic_followup=False)
    dump(out/'SUMMARY.json',em.clean(result))
    def pp(v):return 'undefined' if v is None else f'{v*100:+.4f}'
    lines=['# Whole-canvas reflection and original-basis projection','',
        'Frozen original YOLO26m-seg, no training. All effects are percentage points, normal full-mask decoding.',
        f"{population['planned_images']} planned dev images, {population['effective_images']} effective images, {len(rows)} candidates.",'',
        'MIRROR maps anchor locations, not guaranteed instance identity. MATCH uses same level/predicted class and inverse-box overlap. MIX is ordinary equal logit averaging. PROJ solves only a coefficient increment in original full P.','',
        '| Group | Pair | Image macro IoU Δ [95% CI] | Candidate IoU Δ | Repair / damage | Coverage Δ | AUC Δ | FPR Δ |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,t in tables.items():
        for pair,record in t['comparisons'].items():
            m=record['iou']['image_macro'];cross=record['crossings']
            lines.append(f"| {name} | {pair} | {pp(m['delta'])} [{pp(m['ci95'][0])}, {pp(m['ci95'][1])}] | {pp(record['iou']['candidate']['delta'])} | {cross['repair']} / {cross['damage']} | {pp(record['coverage']['image_macro']['delta'])} | {pp(record['auc']['image_macro']['delta'])} | {pp(record['fpr']['image_macro']['delta'])} |")
    lines+=['','The paired intervals are exploratory across multiple arms/metrics. Original failure-group damage is structurally zero versus A; it does not imply no continuous degradation.',
        'No candidate is dropped after observing a mask. Undefined AUC/FPR remains in IoU. The projection target, image, class and box contain no GT; evaluation membership itself is GT-conditioned.',
        'Only reliable useful effects warrant a separately registered confirmation. A projection advantage requires PROJ versus MIX, not merely PROJ versus A.',
        'Use AUDIT.json and SOURCE_ROWS.jsonl to assess replay, source correspondence, normal equations and projection error.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

@torch.no_grad()
def run(args):
    if os.name=='nt' or not torch.cuda.is_available():raise RuntimeError('Authorized Linux CUDA server only')
    started=time.monotonic();cfg=resolve_runtime_config(load_json(args.config))
    limit=180 if args.smoke else 1800;deadline=started+limit
    if cfg['alpha']!=.5 or cfg['projection_lambda']!=.003 or cfg['bootstrap']!=1000:raise AssertionError('Frozen configuration changed')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'PER_CANDIDATE.jsonl').exists():raise RuntimeError('Existing results; use an independent Run')
    index=load_index(cfg);split=load_json(cfg['split'])
    if len(index['dev'])!=256 or [int(e['image_id']) for e in index['dev']]!=list(map(int,split['dev'])):raise AssertionError('Changed dev membership/order')
    items=[e for e in index['dev'] if e['n']][:2] if args.smoke else index['dev']
    pop=dict(planned_images=len(items),effective_images=sum(bool(e['n']) for e in items),candidates=sum(int(e['n']) for e in items),
             no_positive_images=[int(e['image_id']) for e in items if not e['n']],smoke=args.smoke)
    wanted={int(e['image_id']) for e in items};previous={}
    for line in Path(cfg['prior_results']).read_text().splitlines():
        r=json.loads(line)
        if r['split']=='dev' and int(r['image_id']) in wanted:
            if key(r) in previous:raise AssertionError('Duplicate historical identity')
            previous[key(r)]=r
    if len(previous)!=pop['candidates']:raise AssertionError('Historical population differs')
    source=FlipSource(cfg);coco=COCO(cfg['annotations_train'])
    dump(out/'ENVIRONMENT.json',source.import_info)
    inputs={str(p):sha256(p) for p in (Path(args.config),Path(cfg['split']),Path(cfg['weights']),Path(cfg['prior_results']),Path(cfg['cache'])/'CACHE_IDENTITY.json',Path(cfg['cache'])/'INDEX.json')}
    dump(out/'INPUT_MANIFEST.json',inputs);dump(out/'EVALUATION_INDEX.json',pop)
    rows=[];seen=set();image_audits=[];solver_audits=[];source_rows=[];reference_error=0.
    for pos,e in enumerate(items):
        ensure_time(deadline)
        if not e['n']:continue
        x=load_asset(cfg,int(e['image_id']),True)
        if x['split']!='dev' or len(x['rows'])!=int(e['n']):raise AssertionError('Changed image cohort')
        for field in ('proto','boxes','c0'):x[field]=x[field].to('cuda').float()
        payload={k:x[k] for k in ('input_uint8','boxes','c0','proto','raw_ids','levels')}
        pup=prepare_prototype(x['proto'])
        cs=[];orig=[];direct=[]
        for record in source.infer(payload,deadline_monotonic=deadline):
            ensure_time(deadline);j=int(record['candidate_index'])
            if j!=len(cs):raise AssertionError('Candidate order changed')
            base=record['baseline_logits'];matched=record['match_logits']
            if j==0:
                zero=solve_full_projection(x['proto'],x['c0'][j],x['boxes'][j],torch.zeros_like(base),prototype_up=pup,deadline_monotonic=deadline)
                if not torch.equal(zero['coefficients'],x['c0'][j]) or bool(zero['delta'].ne(0).any()):raise AssertionError('Self projection is not identity')
            solution=solve_full_projection(x['proto'],x['c0'][j],x['boxes'][j],.5*(matched-base),prototype_up=pup,deadline_monotonic=deadline)
            cs.append(solution['coefficients_fp32']);orig.append(record['original_match']['source_coefficient'])
            ann=x['rows'][j]['annotation_id']
            ds={a:direct_metrics(z,x,j,coco,ann) for a,z in [('MIRROR',record['mirror_logits']),('MATCH',matched),('MIX',.5*(base+matched))]}
            direct.append(ds)
            sr=dict(image_id=int(e['image_id']),candidate_index=j,raw_id=int(x['raw_ids'][j]),projection=solution['diagnostics'])
            for field in ('mirror','match','original_match'):
                sr[field]={k:v for k,v in record[field].items() if k!='source_coefficient'}
            source_rows.append(sr);solver_audits.append(solution['diagnostics'])
        if len(cs)!=len(x['rows']):raise AssertionError('Source omitted candidate')
        current=em.evaluate_image(x,{'A':x['c0'],'ORIG_MATCH':torch.stack(orig),'PROJ':torch.stack(cs)},coco)
        for j,r in enumerate(current):
            for arm,metric in direct[j].items():
                for name,value in metric.items():r[name+'_'+arm]=value
            k=key(r)
            if k in seen or k not in previous:raise AssertionError('Candidate identity changed or repeated')
            seen.add(k)
            for metric in METRICS:
                value=r.get(metric+'_A');old=previous[k].get(metric+'_A')
                if em.finite(value)!=em.finite(old):raise AssertionError('Baseline definedness changed')
                if em.finite(value):reference_error=max(reference_error,abs(value-old))
            if reference_error>1e-12:raise AssertionError('Historical A replay changed')
            r['source_match']=source_rows[-len(current)+j]['match']
        rows.extend(current);em.append_rows(out/'PER_CANDIDATE.jsonl',current)
        image_audits.append(dict(image_id=int(e['image_id']),**source.last_audit))
        dump(out/'PROGRESS.json',dict(stage='evaluating',images_done=pos+1,images_planned=len(items),candidates_done=len(rows),seconds=time.monotonic()-started))
        del pup
    if len(rows)!=pop['candidates'] or seen!=set(previous):raise AssertionError('Population not complete')
    ensure_time(deadline);source.replay.assert_unchanged()
    em.append_rows(out/'SOURCE_ROWS.jsonl',em.clean(source_rows))
    audit=dict(passed=True,population=pop,reference_metric_max_error=reference_error,identity_unique=True,
        no_training=True,original_model_unchanged=True,source_image_audits=image_audits,
        solver_diagnostics=solver_audits,GT_free_payload_fields=list(payload),
        source_rows='SOURCE_ROWS.jsonl',self_zero_target_exact=True,limit_seconds=limit)
    dump(out/'AUDIT.json',em.clean(audit));summary=summarize(rows,out,pop)
    ensure_time(deadline)
    dump(out/'COMPLETE.json',dict(passed=True,completed=True,smoke=args.smoke,seconds=time.monotonic()-started,
        population=pop,decisions=summary['decisions'],no_training=True,no_automatic_followup=True))
    print(json.dumps(dict(completed=True,seconds=time.monotonic()-started,population=pop,decisions=summary['decisions'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        dump(Path(args.out)/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=False));raise
if __name__=='__main__':main()
