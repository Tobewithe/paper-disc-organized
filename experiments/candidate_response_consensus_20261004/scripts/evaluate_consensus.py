"""Zero-training, same-target comparison of frozen candidate responses."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from collections import Counter
from pathlib import Path
if '--config' in sys.argv:
    _cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text(encoding='utf-8-sig'))
    sys.path.insert(0,_cfg['source_python'])
import torch
import numpy as np
from pycocotools.coco import COCO
from online_runtime import FrozenReplay,load_asset,load_index,load_json,dump,sha256
import evaluation_metrics as em
from consensus import propose

ARMS=('A','SAME','TOP','SCORE','CONS','CONS_MATCH','ONE_MATCH')
METRICS=('iou','mask75','coverage','auc','fpr')
PAIRS=tuple((a,'A') for a in ARMS[1:])+(('CONS','TOP'),('CONS','SCORE'),('CONS_MATCH','ONE_MATCH'))
em.ARMS=ARMS

def identity(row):return tuple(row[k] for k in ('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx'))

def donor_summary(donors,rows):
    ds=[x['diagnostics'] for x in donors]
    def dist(values):
        a=np.asarray(values,dtype=float)
        return dict(n=len(a),mean=float(a.mean()),q0_25_50_75_100=np.quantile(a,[0,.25,.5,.75,1]).tolist()) if len(a) else dict(n=0)
    out=dict(count=len(ds),matched_K=dict(Counter(str(x['Kmatched']) for x in ds)),branches={},arms={})
    for branch in ('many','one'):
        out['branches'][branch]=dict(available=dist([x[branch]['n_available'] for x in ds]),
            used=dist([x[branch]['n_used'] for x in ds]),box_iou=dist([v for x in ds for v in x[branch]['box_iou']]))
    for arm in ARMS[1:]:
        rs=[x['arms'].get(arm,{}) for x in ds]
        out['arms'][arm]=dict(fallback_reasons=dict(Counter(x['fallback'][arm] for x in ds if arm in x['fallback'])),
            effective_weight_count=dist([1/sum(w*w for w in r['weights']) for r in rs if r.get('weights')]),
            mean_other_mask_iou=dist([v for r in rs for v in (r.get('mean_other_mask_iou') or [])]),
            original_pixel_changes_total=sum(r['original_pixel_changes_'+arm+'_vs_A'] for r in rows),
            changed_candidates=sum(r['original_pixel_changes_'+arm+'_vs_A']>0 for r in rows),
            auc_undefined=sum(not em.finite(r.get('auc_'+arm)) for r in rows))
    out['logit_equivalence_max_abs']=max((v.get('maximum_absolute_error',0.) for x in ds for v in x['logit_equivalence'].values()),default=0.)
    return out

def summarize(rows,donors,out,audit):
    tables={};per_image=[]
    predicates={'all':lambda r:True,'box_good_mask_bad':lambda r:r['box_good_mask_bad'],
        'original_failure':lambda r:not r['mask75_A'],'original_success':lambda r:bool(r['mask75_A'])}
    for name,pred in predicates.items():
        rs=[r for r in rows if pred(r)]
        if not rs:continue
        groups=em.image_groups(rs);table=dict(images=len(groups),candidates=len(rs),comparisons={},candidate={},image_macro={})
        for m in METRICS:
            table['candidate'][m]={a:em.avg([r.get(m+'_'+a) for r in rs]) for a in ARMS}
            table['image_macro'][m]={a:em.avg([em.avg([r.get(m+'_'+a) for r in g]) for g in groups]) for a in ARMS}
        for a,b in PAIRS:
            v={m:em.paired(groups,a,b,m,20261004,1000) for m in METRICS}
            v['repair']=sum(not r['mask75_'+b] and r['mask75_'+a] for r in rs)
            v['damage']=sum(r['mask75_'+b] and not r['mask75_'+a] for r in rs)
            v['net']=v['repair']-v['damage'];table['comparisons'][a+'_minus_'+b]=v
        tables[name]=table
        for g in groups:
            q=dict(split='dev',image_id=g[0]['image_id'],group=name,candidates=len(g))
            for a in ARMS:
                for m in METRICS:q[m+'_'+a]=em.avg([r.get(m+'_'+a) for r in g])
            per_image.append(q)
    em.append_rows(out/'PER_IMAGE.jsonl',per_image)
    all_effect=tables['all']['comparisons']['CONS_minus_A']['iou']['image_macro']
    target=tables.get('box_good_mask_bad',{}).get('comparisons',{}).get('CONS_minus_A',{}).get('iou',{}).get('image_macro',{})
    task_positive=(all_effect['delta']>=.002 and all_effect['ci95'][0]>0) or (
        target.get('delta',-1)>=.005 and target.get('ci95',[-1])[0]>0 and all_effect['delta']>=-.001)
    weight_positive=all(tables['all']['comparisons'][k]['iou']['image_macro']['ci95'][0]>0 for k in ('CONS_minus_TOP','CONS_minus_SCORE'))
    decision='candidate_for_independent_confirmation' if task_positive and weight_positive else 'stop_current_consensus_configuration'
    summary=dict(tables=tables,audit=audit,donors=donor_summary(donors,rows),decision=decision,task_positive=task_positive,
        consensus_weight_supported=weight_positive,scope='previously viewed dev, no AP, no new training',
        diagnostics_file='DONORS.jsonl',automatic_followup=False)
    dump(out/'SUMMARY.json',em.clean(summary))
    lines=['# Frozen candidate-response consensus screen','',f'Decision: {decision}.',
        'All predictions use the original P, target box, class score and zero mask threshold. Donor pools and consensus weights never use GT.','',
        '| Group | Comparison | Macro IoU Δ pp [95% CI] | Candidate Δ pp | Repair/damage | Coverage Δ pp | AUC Δ pp | FPR Δ pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def pp(v):return 'undefined' if v is None else f'{100*v:+.4f}'
    for group,tab in tables.items():
        for pair,c in tab['comparisons'].items():
            m=c['iou']['image_macro'];lo,hi=m['ci95']
            lines.append(f"| {group} | {pair} | {pp(m['delta'])} [{pp(lo)},{pp(hi)}] | {pp(c['iou']['candidate']['delta'])} | {c['repair']}/{c['damage']} | {pp(c['coverage']['image_macro']['delta'])} | {pp(c['auc']['image_macro']['delta'])} | {pp(c['fpr']['image_macro']['delta'])} |")
    lines+=['','Interpretation limits:','',
        '- Mask voting has FCIS precedent; weighted coefficient/logit equivalence is algebra, not novelty.',
        '- CONS−TOP tests whether combining donors adds value; CONS−SCORE tests the extra response-consensus weighting. A gain over SAME alone is insufficient.',
        '- CONS_MATCH and ONE_MATCH use the same per-target donor count; positions and prediction quality still differ, so this is not a proof of a training-assignment cause.',
        '- All donor fallback/empty-support cases remain in the population. Donors are prediction-associated candidates, not GT-authorized positive labels.',
        '- Primary intervals resample whole images. These are reused development images; no blind confirmation, original full inference AP, or trained method claim.',
        '- Stop this fixed configuration if no useful signal. Do not automatically sweep pool size, IoU threshold, weights or start a learned selector.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')

@torch.no_grad()
def run(args):
    started=time.monotonic();cfg=load_json(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if os.name=='nt' or not torch.cuda.is_available():raise RuntimeError('Authorized remote CUDA only')
    if (out/'PER_CANDIDATE.jsonl').exists():raise RuntimeError('Run already has output')
    deadline=started+(180 if args.smoke else 900)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('Fixed screen budget reached')
    planned=load_index(cfg)['dev'];assert len(planned)==256
    if [e['image_id'] for e in planned]!=load_json(cfg['split'])['dev']:raise AssertionError('Frozen split mismatch')
    items=[e for e in planned if e['n']][:2] if args.smoke else planned
    old={}
    for line in Path(cfg['prior_results']).read_text(encoding='utf8').splitlines():
        row=json.loads(line);key=identity(row)
        if key in old:raise AssertionError('Duplicate old identity')
        old[key]=row
    replay=FrozenReplay(cfg);coco=COCO(cfg['annotations_train']);rows=[];donors=[];seen=set()
    dump(out/'ENVIRONMENT.json',dict(replay.import_info,detector_head_forward=True,both_unfused_branches=True))
    max_errors={'c0':0.,'box':0.,'P':0.,'scores':0.,'metric':0.};forward_s=0.;operator_s=0.;metric_s=0.
    for position,entry in enumerate(items):
        budget()
        if not entry['n']:continue
        x=load_asset(cfg,entry['image_id']);keys=[identity(r) for r in x['rows']]
        if len(keys)!=len(set(keys)) or seen.intersection(keys):raise AssertionError('Duplicate candidate')
        torch.cuda.synchronize();t=time.monotonic()
        features=replay.replay([x]);_,raw=replay.head(features)
        torch.cuda.synchronize();forward_s+=time.monotonic()-t
        if set(raw)!= {'one2many','one2one'}:raise AssertionError('Need both unfused raw branches')
        p=raw['one2one']['proto'][0]
        torch.testing.assert_close(raw['one2many']['proto'],raw['one2one']['proto'],atol=0,rtol=0)
        pools={}
        for branch,pred in raw.items():
            pools[branch]=dict(boxes=replay.head._get_decode_boxes(pred)[0].T,
                scores=pred['scores'][0].T.sigmoid(),coefficients=pred['mask_coefficient'][0].T)
            assert pools[branch]['scores'].shape==(8400,80) and pools[branch]['coefficients'].shape==(8400,32)
            for value in pools[branch].values():
                if not torch.isfinite(value).all():raise AssertionError('Nonfinite frozen raw')
        ids=x['raw_ids'].to(replay.device);one=pools['one2one'];cs=one['coefficients'][ids];bb=one['boxes'][ids]
        comparisons={'c0':(cs,x['c0'].to(replay.device)),'box':(bb,x['boxes'].to(replay.device)),
            'P':(p,x['proto'].to(replay.device)),'scores':(one['scores'][ids].max(1).values,x['predicted_scores'].to(replay.device))}
        for name,(actual,expected) in comparisons.items():
            torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
            max_errors[name]=max(max_errors[name],float((actual-expected).abs().max()))
        if not torch.equal(one['scores'][ids].argmax(1),x['predicted_classes'].to(replay.device)):raise AssertionError('Original predicted classes changed')
        # Use exact original cache for A and original decoding, after full replay witness.
        p=x['proto'].to(replay.device);c0=x['c0'].to(replay.device);bb=x['boxes'].to(replay.device)
        torch.cuda.synchronize();t=time.monotonic()
        coeffs,diags=propose(p,c0,bb,x['predicted_classes'].to(replay.device),ids,pools['one2many'],one)
        torch.cuda.synchronize();operator_s+=time.monotonic()-t
        if tuple(coeffs)!=ARMS:raise AssertionError('Output arms/order differ')
        if len(diags)!=len(keys):raise AssertionError('Missing donor diagnostics')
        for k,d in enumerate(diags):
            donors.append(dict(**{name:x['rows'][k][name] for name in ('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx')},diagnostics=d))
        t=time.monotonic()
        results=em.evaluate_image(dict(x,proto=p,c0=c0,boxes=bb),coeffs,coco,chunk_size=4)
        if [identity(r) for r in results]!=keys:raise AssertionError('Decoder changed identities')
        for r in results:
            prior=old[identity(r)]
            for metric in METRICS:
                a,b=r[metric+'_A'],prior[metric+'_A']
                if em.finite(a)!=em.finite(b):raise AssertionError('A metric definedness changed')
                if em.finite(a):
                    e=abs(a-b);max_errors['metric']=max(max_errors['metric'],e)
                    if e>1e-12:raise AssertionError('A no longer reproduces exact baseline')
        metric_s+=time.monotonic()-t
        em.append_rows(out/'PER_CANDIDATE.jsonl',results);em.append_rows(out/'DONORS.jsonl',donors[-len(keys):])
        rows.extend(results);seen.update(keys)
        if position%20==0 or position+1==len(items):
            pr=dict(images=position+1,candidates=len(rows),elapsed_s=time.monotonic()-started)
            dump(out/'PROGRESS.json',pr);print(json.dumps(pr),flush=True)
    if len(rows)!=sum(e['n'] for e in items):raise AssertionError('Population incomplete')
    if not args.smoke and seen!=set(old):raise AssertionError('Frozen candidate set differs')
    replay.assert_unchanged();budget()
    audit=dict(max_reference_errors=max_errors,no_training=True,no_TAL_reassignment=True,shared_P_exact=True,
        donor_GT_free=True,weights_sha256=replay.weights_sha256,config_sha256=sha256(args.config),
        prior_rows_sha256=sha256(cfg['prior_results']),cache_identity_sha256=sha256(Path(cfg['cache'])/'CACHE_IDENTITY.json'),
        planned_images=len(items),effective_images=sum(bool(e['n']) for e in items),candidates=len(rows),
        no_positive_images=[e['image_id'] for e in items if not e['n']],
        forward_seconds=forward_s,operator_seconds=operator_s,metric_seconds=metric_s,
        no_COCO_AP=True,no_new_blind_test=True,smoke=args.smoke)
    dump(out/'AUDIT.json',audit);summarize(rows,donors,out,audit);budget()
    dump(out/'COMPLETE.json',dict(completed=True,passed=True,elapsed_s=time.monotonic()-started,candidates=len(rows),automatic_followup=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        dump(Path(args.out)/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        dump(Path(args.out)/'COMPLETE.json',dict(completed=False,passed=False));raise

if __name__=='__main__':main()
