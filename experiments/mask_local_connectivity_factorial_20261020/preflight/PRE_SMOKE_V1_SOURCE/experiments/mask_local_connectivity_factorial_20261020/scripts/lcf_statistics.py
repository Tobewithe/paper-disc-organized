"""Paired full-image topology/IoU/AP inference and preregistered decisions."""
from lcf_runtime import *
from collections import defaultdict
import contextlib,multiprocessing as mp

_CACHE=None;_DRAWS=None
def _init(cache,draws):
    global _CACHE,_DRAWS
    import numpy as np
    from coco_bootstrap import load_cache
    _CACHE=load_cache(cache);_DRAWS=np.load(draws,mmap_mode='r')
def _replicate(i):
    from coco_bootstrap import resampled_evaluator,ap
    return i,ap(resampled_evaluator(_CACHE,_DRAWS[i]))

def paired(rows,function,eligible=lambda r:True):
    import numpy as np
    groups=defaultdict(list)
    for row in rows:
        if eligible(row):groups[row['image_id']].append(float(function(row)))
    require(len(groups)>1,'insufficient fixed paired images')
    values=np.array([sum(v)/len(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(1729)
    boot=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975],method='linear').tolist(),
            'images':len(values),'instances':sum(map(len,groups.values())),'unit':'fraction'}

def event(row,name,kind='union',connectivity='8'):
    m=row['topology'][name][connectivity]
    if kind=='union':return m['material_split'] or m['material_bridge']
    return m['material_'+kind]

def statistics_stage():
    import numpy as np
    from coco_bootstrap import evaluate,compact,save_cache,ap,verify_replication
    from lcf_evaluation import NORMAL,ALL
    evaluation=pointer('evaluate');summary=read(evaluation/'SUMMARY.json')
    with Run('statistics') as run:
        write(run.path/'ENVIRONMENT.json',{'python':sys.version,'numpy':np.__version__,'workers':2,'CPU_only':True})
        rows=[json.loads(s) for s in (evaluation/'FIXED_PAIRS.jsonl').read_text(encoding='utf-8').splitlines()]
        eligible=lambda r:r['topology']['N']['8']['bridge_eligible']
        endpoints={name:paired(rows,lambda r:event(r,name),eligible) for name in ALL}
        topology={};iou={}
        for ref in ['N','G_old','A','B','C','Dproj']:
            topology['D_vs_'+ref]=paired(rows,lambda r:event(r,ref)-event(r,'D'),eligible)
            baseline=endpoints[ref]['mean'];topology['D_vs_'+ref]['relative_reduction']=topology['D_vs_'+ref]['mean']/baseline if baseline else None
            for kind in ['split','bridge']:
                topology['D_vs_'+ref+'_'+kind]=paired(rows,lambda r:event(r,ref,kind)-event(r,'D',kind),eligible)
            iou['D-'+ref]=paired(rows,lambda r:r['ious']['D']-r['ious'][ref])
        topology['Deq_vs_DprojEq']=paired(rows,lambda r:event(r,'DprojEq')-event(r,'Deq'),eligible)
        sensitivity={name:paired(rows,lambda r:event(r,name,connectivity='4'),eligible) for name in NORMAL}
        counts={name:{'pairs':len(rows),'bridge_eligible':sum(eligible(r) for r in rows),
            'material_split':sum(event(r,name,'split') for r in rows),'material_bridge':sum(event(r,name,'bridge') for r in rows),
            'GT_part_missed':sum(r['topology'][name]['8']['missed_gt_components']>0 for r in rows)} for name in ALL}
        totals={a:defaultdict(int) for a in ['A','B','C','D']}
        for line in (pointer('train')/'training.jsonl').open(encoding='utf-8'):
            row=json.loads(line)
            for key,v in row['supervision'].items():totals[row['arm']][key]+=v
        hardness={a:{**totals[a],'match_fraction':totals[a]['hardness_matched_noncritical']/max(1,totals[a]['critical_samples'])} for a in ['A','C']}
        write(run.path/'TOPOLOGY_AND_IOU.json',{'endpoints':endpoints,'reductions':topology,'IoU_comparisons':iou,
            'counts':counts,'four_connectivity_sensitivity':sensitivity,'hardness':hardness})
        ids=read(STUDY/'inputs/DATA_CONTRACT.json')['val_ids'];dataset=read(run.config['val_annotations'])
        draws=np.random.default_rng(1729).integers(0,len(ids),size=(2000,len(ids)),dtype=np.int32)
        draws_path=run.path/'PAIRED_IMAGE_DRAWS.npy';np.save(draws_path,draws,allow_pickle=False)
        values=np.full((2000,len(NORMAL)),np.nan);crowd=next(a['image_id'] for a in dataset['annotations'] if a.get('iscrowd'))
        for j,name in enumerate(NORMAL):
            run.check();pred=read(evaluation/(name+'_predictions.json'))
            with (run.path/(name+'_cache.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                ev=evaluate(run.config['val_annotations'],pred,ids,ap_only=True)
                require(abs(ap(ev)-summary['methods'][name]['metrics']['segm']['AP'])<1e-10,'compact AP mismatch')
                cache=compact(ev);verification=verify_replication(cache,dataset,pred,[0,0,ids.index(crowd),123,999,4999])
            cache_path=run.path/(name+'_COCO_CACHE.pkl');save_cache(cache_path,cache);write(run.path/(name+'_DUPLICATE_VERIFICATION.json'),verification)
            del pred,ev,cache
            with (run.path/(name+'_BOOTSTRAP_PROGRESS.jsonl')).open('x',encoding='utf-8') as progress:
                with mp.get_context('spawn').Pool(2,initializer=_init,initargs=(str(cache_path),str(draws_path))) as pool:
                    for count,(index,value) in enumerate(pool.imap_unordered(_replicate,range(2000),chunksize=2),1):
                        values[index,j]=value;progress.write(json.dumps({'replicate':index,'AP_points':value})+'\n')
                        if count%50==0:
                            progress.flush();run.check();run.update(phase='paired_AP_bootstrap',method=name,completed=count,total=2000)
                            print('BOOTSTRAP',name,count,flush=True)
            np.save(run.path/'BOOTSTRAP_AP_POINTS.npy',values,allow_pickle=False)
        require(np.isfinite(values).all(),'AP bootstrap incomplete');apdiff={}
        for ref in ['N','G_old','A','B','C']:
            point=summary['methods']['D']['metrics']['segm']['AP']-summary['methods'][ref]['metrics']['segm']['AP']
            v=values[:,NORMAL.index('D')]-values[:,NORMAL.index(ref)]
            apdiff['D-'+ref]={'mean':point,'ci95':np.quantile(v,[.025,.975],method='linear').tolist(),'unit':'AP_points'}
        n=topology['D_vs_N'];c=topology['D_vs_C']
        topo_use=n['relative_reduction'] is not None and n['relative_reduction']>=.2 and n['ci95'][0]>0 and topology['D_vs_G_old']['ci95'][0]>0
        no_type_worse=all(topology['D_vs_N_'+k]['ci95'][1]>=0 for k in ['split','bridge'])
        supervision=c['relative_reduction'] is not None and c['relative_reduction']>=.1 and c['ci95'][0]>0 and all(v['match_fraction']>=.9 for v in hardness.values())
        local=topology['D_vs_B']['ci95'][0]>0 and topology['D_vs_Dproj']['ci95'][0]>0 and topology['Deq_vs_DprojEq']['ci95'][0]>0 and summary['amplitude_invalid_rows']==0
        guard=iou['D-N']['mean']>=-.001 and summary['damage']['D']['damage_fraction']<=.01
        ap_no_loss=apdiff['D-N']['ci95'][0]>=0 and apdiff['D-G_old']['ci95'][0]>=0
        result={'topology_utility_pass':topo_use,'neither_error_type_significantly_worse':no_type_worse,
                'supervision_mechanism_pass':supervision,'local_intervention_pass':local,'pixel_damage_guard_pass':guard,
                'AP_no_loss_evidence':ap_no_loss,'all_required_pass':topo_use and no_type_worse and supervision and local and guard and ap_no_loss,
                'AP_comparisons':apdiff,'damage':summary['damage'],'topology':topology,'hardness':hardness,
                'claim_scope':'single-seed COCO development evidence; no globally optimal fixed-P proof; no independent confirmation'}
        write(run.path/'STATISTICS.json',result)
        science='supported_at_scope' if result['all_required_pass'] else 'requirements_not_jointly_met'
        write(run.path/'DECISION.json',{'scientific_status':science,'continue_training':False,
           'next_action':'review frozen contrasts and propose one evidence-supported successor; do not train automatically'})
        run.finish(result,science)
        s=read(STUDY/'study.json');s.update(plan_status='closed',scientific_status=science,current_phase='completed_pending_report',active_run_id=None);write(STUDY/'study.json',s)
