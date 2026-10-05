"""Fixed-identity oracle headroom, with nested action sets and matched budgets."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from common import setup, atomic


def best(actions, budget=None):
    feasible = [a for a in actions if budget is None or a['cost'] <= budget + 1e-12]
    return max(feasible, key=lambda a: (a['iou'], -a['cost'], a['name'] == 'baseline'))


def stats(rows, names):
    result = {}
    for group in ['all', 'target', 'other_failure', 'success', 'success_high_coverage', 'success_low_coverage']:
        rr = [r for r in rows if group == 'all' or
              (group == 'target' and r['target']) or
              (group == 'other_failure' and not r['target'] and r['base'] < .75) or
              (group == 'success' and r['base'] >= .75) or
              (group == 'success_high_coverage' and r['base'] >= .75 and r['coverage'] >= .95) or
              (group == 'success_low_coverage' and r['base'] >= .75 and r['coverage'] < .95)]
        if not rr:
            continue
        result[group] = {}
        for name in names:
            repair = sum(r['base'] < .75 <= r['actions'][name]['iou'] for r in rr)
            damage = sum(r['actions'][name]['iou'] < .75 <= r['base'] for r in rr)
            result[group][name] = dict(n=len(rr), repairs=repair, damages=damage,
                net_repairs=repair-damage,
                iou_gain_pp=float(np.mean([100*(r['actions'][name]['iou']-r['base']) for r in rr])),
                coverage_loss_pp=float(np.mean([100*r['actions'][name]['cost'] for r in rr])),
                removed_fp_per_gt_pp=float(np.mean([100*r['actions'][name]['fp_gain'] for r in rr])),
                fp_removed_fraction=float(np.mean([r['actions'][name]['fp_gain']/r['fp_per_gt']
                    for r in rr if r['fp_per_gt'] > 0])) if any(r['fp_per_gt'] > 0 for r in rr) else None)
    return result


def comparison(rows, names, pairs, ids, repeats, seed):
    positions = {iid:i for i,iid in enumerate(ids)}
    totals = np.zeros((len(ids),len(names),3))
    den = np.zeros((len(ids),2))
    for r in rows:
        i=positions[r['image_id']]
        den[i] += [r['target'], r['base'] >= .75]
        for j,name in enumerate(names):
            a=r['actions'][name]
            repaired = r['base'] < .75 <= a['iou']; damaged = a['iou'] < .75 <= r['base']
            totals[i,j] += [r['target'] and repaired, damaged, int(repaired)-int(damaged)]
    rng=np.random.default_rng(seed); boot=np.empty((repeats,len(names),3))
    for k in range(repeats):
        sample=rng.integers(0,len(ids),len(ids)); ds=den[sample].sum(0)
        boot[k]=totals[sample].sum(0)
        boot[k,:,:2]=100*boot[k,:,:2]/np.maximum(ds,1)
    estimates=totals.sum(0);estimates[:,:2]=100*estimates[:,:2]/np.maximum(den.sum(0),1)
    result={}
    for lhs,rhs in pairs:
        a=names.index(lhs);b=names.index(rhs)
        result[lhs+'_vs_'+rhs]={key:dict(difference=float(estimates[a,k]-estimates[b,k]),
            ci95=np.quantile(boot[:,a,k]-boot[:,b,k],[.025,.975]).tolist())
            for k,key in enumerate(['target_repair_pp','success_damage_pp','net_repair_count'])}
        gain=lost=0
        for r in rows:
            if r['target']:
                ra=r['actions'][lhs]['iou']>=.75;rb=r['actions'][rhs]['iou']>=.75
                gain+=ra and not rb;lost+=rb and not ra
        result[lhs+'_vs_'+rhs]['target_newly_repaired']=int(gain)
        result[lhs+'_vs_'+rhs]['target_repairs_lost']=int(lost)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();p,out=setup(args.protocol,args.output)
    source=list(csv.DictReader(Path(p['decisions']).open(newline='',encoding='utf-8')))
    rows=[];lookup={}
    fullnames=['baseline','smooth_all','frozen_rcmc','direct','oracle_single','oracle_single_cov01','oracle_single_cov02']
    for s in source:
        j=float(s['baseline_iou']);g=float(s['gt_area']);c=float(s['baseline_coverage'])
        fp=c/j-1 if j>0 else 0
        base=dict(name='baseline',iou=j,cost=0.,fp_gain=0.)
        smooth=dict(name='smooth',iou=float(s['trial_iou']),cost=float(s['coverage_cost']),fp_gain=float(s['fp_benefit']))
        # Original bank already implements restoring baseline for empty trials.
        assert smooth['cost']>=-1e-10 and smooth['fp_gain']>=-1e-10
        assert abs(smooth['iou']-(c-smooth['cost'])/(1+fp-smooth['fp_gain'])) < 1e-8
        r=dict(image_id=int(s['image_id']),annotation_id=int(s['annotation_id']),candidate_index=int(s['candidate_index']),
            base=j,coverage=c,fp_per_gt=fp,gt_area=g,target=s['target']=='True',actions={
            'baseline':base,'smooth_all':smooth,'frozen_rcmc':smooth if s['use_frozen_rcmc']=='1' else base,
            'direct':smooth if s['use_direct']=='1' else base,'oracle_single':best([base,smooth]),
            'oracle_single_cov01':best([base,smooth],.01),'oracle_single_cov02':best([base,smooth],.02)})
        key=(r['image_id'],r['annotation_id']);assert key not in lookup
        lookup[key]=r;rows.append(r)
    assert len(rows)==30426 and sum(r['target'] for r in rows)==1701
    fullstats=stats(rows,fullnames)
    assert fullstats['target']['frozen_rcmc']['repairs']==457 and fullstats['success']['frozen_rcmc']['damages']==445
    for n in fullnames[4:]:assert fullstats['success'][n]['damages']==0
    fullids=sorted(json.loads(Path(p['image_ids']).read_text()))
    fullpairs=[('oracle_single','frozen_rcmc'),('oracle_single','direct'),('oracle_single_cov02','frozen_rcmc')]
    fullcomparison=comparison(rows,fullnames,fullpairs,fullids,p['bootstrap_repeats'],p['seed'])

    panel=defaultdict(list)
    for s in csv.DictReader(Path(p['threshold_pixels']).open(newline='',encoding='utf-8')):
        panel[(int(s['image_id']),int(s['annotation_id']))].append(s)
    prows=[]
    pnames=['baseline','frozen_rcmc','smooth_all','oracle_single','oracle_single_cov01','oracle_single_cov02',
            'oracle_grid','oracle_union','oracle_union_cov01','oracle_union_cov02']
    for key,rr in panel.items():
        r=lookup[key];r={k:v for k,v in r.items() if k!='actions'};actions={}
        allacts=[];grid=[]
        for s in rr:
            variant=s['variant'];j=float(s['iou']);c=float(s['coverage']);fp=float(s['fp'])/float(s['gt_pixels'])
            if variant=='threshold' and float(s['threshold'])==0:
                assert abs(j-r['base'])<1e-8 and abs(c-r['coverage'])<1e-8
            a=dict(name=('baseline' if variant=='threshold' and float(s['threshold'])==0 else variant+'_'+s['threshold']),
                iou=j,cost=r['coverage']-c,fp_gain=r['fp_per_gt']-fp)
            if s['empty']=='True':
                a=dict(name='baseline',iou=r['base'],cost=0.,fp_gain=0.)
            if variant=='threshold':grid.append(a)
            elif variant=='smooth':actions['smooth_all']=a
            elif variant=='response':actions['frozen_rcmc']=a
        actions['baseline']=dict(name='baseline',iou=r['base'],cost=0.,fp_gain=0.)
        assert abs(actions['smooth_all']['iou']-lookup[key]['actions']['smooth_all']['iou'])<1e-8
        single=[actions['baseline'],actions['smooth_all']];union=single+grid
        actions.update(oracle_single=best(single),oracle_single_cov01=best(single,.01),oracle_single_cov02=best(single,.02),
            oracle_grid=best(grid),oracle_union=best(union),oracle_union_cov01=best(union,.01),oracle_union_cov02=best(union,.02))
        for b in ['', '_cov01','_cov02']:
            assert actions['oracle_union'+b]['iou'] >= actions['oracle_single'+b]['iou']-1e-12
        r['actions']=actions;prows.append(r)
    assert len(prows)==473 and sum(r['target'] for r in prows)==256
    pstats=stats(prows,pnames)
    ppairs=[('oracle_single','frozen_rcmc'),('oracle_union','oracle_single'),
            ('oracle_union_cov01','oracle_single_cov01'),('oracle_union_cov02','oracle_single_cov02')]
    pcomparison=comparison(prows,pnames,ppairs,sorted({r['image_id'] for r in prows}),p['bootstrap_repeats'],p['seed'])
    flat=[]
    for cohort,rr in [('full',rows),('panel',prows)]:
        for r in rr:
            for n,a in r['actions'].items():
                flat.append(dict(cohort=cohort,image_id=r['image_id'],annotation_id=r['annotation_id'],
                    candidate_index=r['candidate_index'],target=r['target'],baseline_iou=r['base'],
                    baseline_coverage=r['coverage'],method=n,**a))
    with (out/'oracle_decisions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    result=dict(full=dict(images=len(fullids),groups=fullstats,comparisons=fullcomparison),
        panel=dict(images=len({r['image_id'] for r in prows}),groups=pstats,comparisons=pcomparison),
        protocol=p,limitations=[
            'GT oracles are diagnostic action-set ceilings, not deployed gains or standard COCO AP.',
            'Baseline included in every oracle makes zero IoU75 damage true by construction.',
            'Panel successes are high-coverage controls, not representative of all successes.',
            'Nested union includes original smooth, preventing a non-nested action-set comparison.',
            'No network forward, changed matching, new threshold selection, or fitting in this run.',
            'Fixed associated instances only: no bound on missed detections or unmatched predictions.',
            'Historical validation data; exploratory paired image intervals, not a new blind test.'])
    atomic(out/'SUMMARY.json',result)
    print(json.dumps({'completed':True,'full':fullstats['target'],'panel':pstats['target']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
