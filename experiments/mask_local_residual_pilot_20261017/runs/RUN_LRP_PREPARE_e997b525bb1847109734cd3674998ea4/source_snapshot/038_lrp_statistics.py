from collections import defaultdict
import numpy as np

def summary(pairs,tasks):
    target=[r for r in pairs if r['population']=='T']
    if (len(pairs),len(target),sum(r['N'] for r in tasks))!=(2246,490,1808): raise ValueError('evaluation denominator drift')
    results={}; samples={}
    for label,rows in (('target',target),('overall',pairs)):
        group=defaultdict(list)
        for r in rows: group[r['image_id']].append(r)
        ids=sorted(group)
        if len(ids)!=(203 if label=='target' else 394): raise ValueError('image denominator drift')
        draws=np.random.default_rng(1729).integers(0,len(ids),(2000,len(ids)))
        samples[label]=draws
        results[label]={}
        for left,right in [('L','N'),('L','G'),('L','U'),('U','N'),('U','G')]:
            v=np.array([np.mean([r[left]-r[right] for r in group[i]]) for i in ids])
            results[label][left+'-'+right]=dict(mean=float(v.mean()),ci95=np.quantile(v[draws].mean(1),[.025,.975],method='linear').tolist(),images=len(ids),pairs=len(rows))
    census={k:dict(damaged=sum(r['N'] and not r[k] for r in tasks),recovered=sum(not r['N'] and r[k] for r in tasks),native_successes=1808) for k in ('G','L','U')}
    t=results['target']; a=results['overall']
    mechanism=t['L-U']['mean']>=.005 and t['L-U']['ci95'][0]>0
    utility=t['L-N']['mean']>=.005 and t['L-N']['ci95'][0]>0 and t['L-G']['ci95'][0]>0 and a['L-N']['mean']>=-.001 and census['L']['damaged']<=18
    negative=t['L-U']['ci95'][1]<.005 or t['L-N']['ci95'][1]<.005 or t['L-G']['ci95'][1]<=0 or a['L-N']['mean']<-.001 or census['L']['damaged']>18
    results.update(census=census,mechanism_pass=mechanism,utility_pass=utility,
                   decision='POSITIVE_INTERNAL_SCREEN' if mechanism and utility else 'NEGATIVE_AT_BUDGET' if negative else 'INCONCLUSIVE_AT_BUDGET',unit='IoU_0_to_1',reduction='image_macro',new_AP=False)
    return results,samples
