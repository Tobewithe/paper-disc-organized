"""Image-cluster summaries, with fixed draw order and no selected subgroups."""
import numpy as np


def summarize(rows,config,indices=None):
    ids=sorted({r['image_id'] for r in rows})
    arms=['C','G',*config['pressure_arms']]
    points=[]
    for r in rows:
        means={a:float(np.mean([t['iou'] for t in r['trials'] if t['arm']==a])) for a in config['pressure_arms']}
        means.update(C=r['C'],G=r['G'],N=r['N'])
        row={f'{a}-N':means[a]-means['N'] for a in arms}
        row.update({f'{a}-G':means[a]-means['G'] for a in arms if a!='G'})
        row.update({f'D_{a}':means['C']-means[a] for a in config['pressure_arms']})
        row['D_S-D_R']=means['R']-means['S'];row['D_S-D_M']=means['M']-means['S']
        points.append(row)
    keys=list(points[0])
    values=np.array([[np.mean([points[j][key] for j,r in enumerate(rows) if r['image_id']==iid]) for key in keys] for iid in ids],dtype=np.float64)
    if indices is None:indices=np.random.Generator(np.random.PCG64(config['seed'])).integers(len(ids),size=(config['bootstrap_resamples'],len(ids)))
    if indices.shape!=(config['bootstrap_resamples'],len(ids)): raise ValueError('bootstrap dimensions')
    samples=values[indices].mean(axis=1)
    stats={key:dict(mean=float(values[:,k].mean()),ci95=np.quantile(samples[:,k],[.025,.975]).tolist()) for k,key in enumerate(keys)}
    tests=['D_S-D_R','D_S-D_M']
    positive=stats['D_S']['ci95'][0]>0 and all(stats[k]['mean']>=config['minimum_effect'] and stats[k]['ci95'][0]>0 for k in tests)
    negative=stats['D_S']['ci95'][1]<=0 or any(stats[k]['ci95'][1]<config['minimum_effect'] for k in tests)
    decision='TRANSFERABLE_DIRECTIONAL_RISK' if positive else ('NO_MATERIAL_TRANSFER_AT_SCOPE' if negative else 'INCONCLUSIVE_NOT_READY')
    successful=[r for r in rows if r['C']>=.75]
    damage={a:dict(teacher_success_pairs=len(successful),trials=sum(t['arm']==a for r in successful for t in r['trials']),
        damaged_trials=sum(t['arm']==a and t['iou']<.75 for r in successful for t in r['trials'])) for a in config['pressure_arms']}
    for x in damage.values():x['fraction']=x['damaged_trials']/x['trials'] if x['trials'] else None
    return dict(pairs=len(rows),images=len(ids),unit='IoU_0_to_1',reduction='mean_repeats_then_pairs_within_image_then_images',
        effects=stats,teacher_success_damage=damage,decision=decision,prototype_tolerance_training_design_warranted=positive,
        new_training_authorized=False,new_training_steps=0,new_method_AP=None,scope='GT_assisted_frozen_internal_holdout'),indices,ids
