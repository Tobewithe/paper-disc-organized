"""Paired image statistics; Windows workers load one method cache at a time."""
from __future__ import annotations
import contextlib,json,multiprocessing as mp,time
from collections import defaultdict
from ssr_runtime import *

_CACHE=None
_DRAWS=None
def _init(cache,draws):
    global _CACHE,_DRAWS
    import numpy as np
    from coco_bootstrap import load_cache
    _CACHE=load_cache(cache);_DRAWS=np.load(draws,mmap_mode='r')
def _replicate(i):
    from coco_bootstrap import resampled_evaluator,ap
    return i,ap(resampled_evaluator(_CACHE,_DRAWS[i]))

def paired(rows,a,b,target=True):
    import numpy as np
    groups=defaultdict(list)
    for r in rows:
        if target is None or r['target']==target:groups[r['image_id']].append(r['ious'][a]-r['ious'][b])
    arr=np.array([sum(v)/len(v) for k,v in sorted(groups.items())],dtype=np.float64)
    require(len(arr)>1,'empty paired population')
    rng=np.random.default_rng(1729)
    values=arr[rng.integers(0,len(arr),size=(2000,len(arr)))].mean(1)
    return {'mean':float(arr.mean()),'ci95':np.quantile(values,[.025,.975],method='linear').tolist(),
            'images':len(arr),'instances':sum(len(v) for v in groups.values()),'unit':'IoU_fraction'}

def statistics_stage():
    import numpy as np
    from coco_bootstrap import evaluate,compact,save_cache,ap,verify_replication
    from ssr_evaluation import NORMAL
    evaluation=pointer('evaluate');summary=read(evaluation/'SUMMARY.json')
    with Run('statistics') as run:
        write(run.path/'ENVIRONMENT.json',{'python':sys.version,'numpy':np.__version__,'workers':2,'scope':'CPU-only statistics'})
        ids=read(STUDY/'inputs/DATA_CONTRACT.json')['val_ids']
        draws=np.random.default_rng(1729).integers(0,len(ids),size=(2000,len(ids)),dtype=np.int32)
        draws_path=run.path/'PAIRED_IMAGE_DRAWS.npy';np.save(draws_path,draws,allow_pickle=False)
        values=np.full((2000,len(NORMAL)),np.nan,dtype=np.float64)
        dataset=read(run.config['val_annotations'])
        crowd=next(a['image_id'] for a in dataset['annotations'] if a.get('iscrowd'))
        check_draws=[0,0,ids.index(crowd),123,999,4999]
        for j,name in enumerate(NORMAL):
            run.check();pred=read(evaluation/(name+'_predictions.json'))
            with (run.path/(name+'_cache.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                ev=evaluate(run.config['val_annotations'],pred,ids,ap_only=True)
                require(abs(ap(ev)-summary['methods'][name]['metrics']['segm']['AP'])<1e-10,'compact AP mismatch')
                cache=compact(ev);verification=verify_replication(cache,dataset,pred,check_draws)
            cache_path=run.path/(name+'_COCO_CACHE.pkl');save_cache(cache_path,cache)
            write(run.path/(name+'_DUPLICATE_VERIFICATION.json'),verification)
            del pred,ev,cache
            with (run.path/(name+'_BOOTSTRAP_PROGRESS.jsonl')).open('x',encoding='utf-8') as progress:
                with mp.get_context('spawn').Pool(2,initializer=_init,initargs=(str(cache_path),str(draws_path))) as pool:
                    for count,(index,value) in enumerate(pool.imap_unordered(_replicate,range(2000),chunksize=2),1):
                        values[index,j]=value;progress.write(json.dumps({'replicate':index,'AP_points':value})+'\n')
                        if count%50==0:
                            progress.flush();run.check();run.update(phase='paired_AP_bootstrap',method=name,completed=count,total=2000)
                            print('BOOTSTRAP',name,count,2000,flush=True)
            np.save(run.path/'BOOTSTRAP_AP_POINTS.npy',values,allow_pickle=False)
        require(np.isfinite(values).all(),'incomplete AP bootstrap')
        rows=[json.loads(s) for s in (evaluation/'FIXED_PAIRS.jsonl').read_text(encoding='utf-8').splitlines()]
        fixed={}
        for a,b in [('D','N'),('D','G_old'),('D','G20k'),('D','A'),('D','B'),('D','C'),('D','M'),('D','J'),('Deq','Jeq'),
                    ('B','A'),('C','A'),('A','N'),('B','N'),('C','N'),('M','N')]:fixed[a+'-'+b]=paired(rows,a,b)
        overall=paired(rows,'D','N',None)
        # Directly bootstrap the per-image factorial interaction, not CI subtraction.
        interaction=[]
        for r in rows:
            if r['target']:
                v=dict(r);v['ious']={'I':(r['ious']['D']-r['ious']['B'])-(r['ious']['C']-r['ious']['A']),'zero':0};interaction.append(v)
        fixed['interaction']=paired(interaction,'I','zero')
        point={n:summary['methods'][n]['metrics']['segm']['AP'] for n in NORMAL};apdiff={}
        for reference in ('N','G_old','G20k','A','B','C','M'):
            d=values[:,NORMAL.index('D')]-values[:,NORMAL.index(reference)]
            apdiff['D-'+reference]={'mean':point['D']-point[reference],'ci95':np.quantile(d,[.025,.975],method='linear').tolist(),'unit':'AP_points'}
        def useful(x):return x['mean']>=.005 and x['ci95'][0]>0
        mechanism=all(useful(fixed['D-'+r]) for r in ('B','C','M'))
        interventions=fixed['D-J']['ci95'][0]>0 and fixed['Deq-Jeq']['ci95'][0]>0 and summary['amplitude_invalid_rows']==0
        damage=summary['damage']['D']
        utility=useful(fixed['D-N']) and useful(fixed['D-G20k']) and overall['mean']>=-.001 and damage['damage_fraction']<=.01
        ap_pass=apdiff['D-N']['mean']>=.30 and apdiff['D-N']['ci95'][0]>0 and apdiff['D-G20k']['ci95'][0]>0
        result={'fixed_pair_comparisons':fixed,'overall_D_minus_N':overall,'AP_comparisons':apdiff,
                'mechanism_pass':mechanism,'interventions_pass':interventions,'utility_pass':utility,'AP_pass':ap_pass,
                'all_required_pass':mechanism and interventions and utility and ap_pass,'damage':damage,
                'claim_scope':'single seed; repeatedly used COCO development domain; no independent confirmation'}
        write(run.path/'STATISTICS.json',result)
        science='supported_at_scope' if result['all_required_pass'] else 'requirements_not_jointly_met'
        write(run.path/'DECISION.json',{'scientific_status':science,'continue_training':False,
              'next_action':'propose independent confirmation only' if result['all_required_pass'] else 'close this implementation; interpret predeclared single-factor contrasts without promoting a control'})
        run.finish(result,science)
        s=read(STUDY/'study.json');s.update(plan_status='closed',scientific_status=science,current_phase='completed_pending_report',active_run_id=None)
        write(STUDY/'study.json',s)
