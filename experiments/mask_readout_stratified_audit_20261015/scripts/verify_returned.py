"""Independent result checks: scalar box arithmetic and explicit resampled image indices."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def jl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a,b):
    if a is None or b is None:
        assert a is b,(a,b)
    else:
        assert abs(a-b)<1e-12,(a,b)


def verify(run,root):
    summary,contract=read(run/'SUMMARY.json'),read(run/'EVAL_CONTRACT.json')
    for item in read(run/'ARTIFACT_MANIFEST.json')['artifacts']:
        p=run/item['relative_path']
        assert p.stat().st_size==item['size_bytes'] and digest(p)==item['sha256'],str(p)
    parent=root/'experiments/mask_readout_benefit_audit_20261014/runs/RUN_BENEFIT_AUDIT_FULL_44b006cece39486aa502ca64398166d4'
    old={(r['role'],r['image_id'],r['annotation_id']):r for r in jl(parent/'PAIR_MEASUREMENTS.jsonl')}
    rows=jl(run/'STRATIFIED_PAIRS.jsonl')
    metadata={r['image_id']:r for r in jl(run/'BOX_METADATA.jsonl')}
    max_error=0.
    for row in rows:
        previous=old[row['role'],row['image_id'],row['annotation_id']]
        assert all(row[k]==v for k,v in previous.items()),'source measurement mutated'
        entry=metadata[row['image_id']]
        rank=row['output_rank']; box=entry['boxes_input'][rank]
        all_max=same_max=0.
        for j,other in enumerate(entry['boxes_input']):
            if entry['raw_ids'][j]==entry['raw_ids'][rank]:continue
            width=max(0.,min(box[2],other[2])-max(box[0],other[0]))
            height=max(0.,min(box[3],other[3])-max(box[1],other[1]))
            inter=width*height
            union=max(0.,box[2]-box[0])*max(0.,box[3]-box[1])+max(0.,other[2]-other[0])*max(0.,other[3]-other[1])-inter
            iou=inter/union if union else 0.
            all_max=max(all_max,iou)
            if entry['classes'][j]==entry['classes'][rank]: same_max=max(same_max,iou)
        max_error=max(max_error,abs(all_max-row['all_class_neighbor_iou']),abs(same_max-row['same_class_neighbor_iou']))
        for field,value in [('gt_area_group',row['gt_area']),('native_area_group',row['native_foreground_area'])]:
            assert row[field]==('S' if value<1024 else 'M' if value<9216 else 'L')
        for field,value in [('all_crowd_group',all_max),('same_crowd_group',same_max)]:
            assert row[field]==('Low' if value<.1 else 'Mid' if value<.5 else 'High')
    assert max_error<1e-12
    cohorts={'dev_all':[r for r in rows if r['role']=='dev_bbox75'],
             'target_T':[r for r in rows if r['role']=='dev_bbox75' and r['population']=='T'],
             'success_panel':[r for r in rows if r['role']=='success_segm75']}
    for table in summary['tables']:
        selected=[r for r in cohorts[table['cohort']] if (table['area']=='All' or r[table['area_definition']]==table['area'])
                  and (table['crowd']=='All' or r[table['crowd_definition']]==table['crowd'])]
        assert len(selected)==table['n']
        assert len({r['image_id'] for r in selected})==table['images']
        for stage,record in table['stages'].items():
            samples=defaultdict(list)
            for r in selected:
                v=r['stages'][stage]['delta']
                if v is not None:samples[r['image_id']].append(v)
            values=[v for vs in samples.values() for v in vs]
            close(float(np.mean(values)) if values else None,record['pair_mean'])
            close(float(np.mean([np.mean(v) for v in samples.values()])) if samples else None,record['image_macro_mean'])
        for key,field in [('task_damage','task_damaged'),('task_repair','task_repaired')]:
            if key in table:
                assert table[key]['numerator']==sum(r[field] for r in selected)
                assert table[key]['denominator']==len(selected)
        if 'damaged_stage_counts' in table:
            counts=Counter(r['damage_stage_signature'] for r in selected if r['task_damaged'])
            assert all(v==counts[k] for k,v in table['damaged_stage_counts'].items())
    # Unlike the production matrix-multiplicity implementation, explicitly repeat image indices.
    checked=[]
    for c in summary['contrasts']:
        success=c['endpoint']=='damage'
        cohort=cohorts['success_panel' if success else 'target_T']
        ids=sorted(contract['success_image_ids' if success else 'dev_ids']); index={v:i for i,v in enumerate(ids)}
        ordered=sorted(ids,key=lambda i:hashlib.sha256(f'stratified-audit-v1:{i}'.encode()).hexdigest())
        halves=[set(ordered[:len(ids)//2]),set(ordered[len(ids)//2:])]
        rng=np.random.default_rng(1729)
        resampled=rng.integers(0,len(ids),(2000,len(ids)))
        draws=np.zeros(2000);point=0.;half_points=[0.,0.]
        adequate_full=adequate_half=True
        adjustment=c['area_definition'] if c['factor']=='crowd' else 'all_crowd_group'
        levels=['S','M','L'] if c['factor']=='crowd' else ['Low','Mid','High']
        for level in levels:
            subset=[r for r in cohort if r[adjustment]==level]
            weight=len(subset)/len(cohort)
            if c['factor']=='crowd':
                sides=[[r for r in subset if r['all_crowd_group']==g] for g in ['Low','High']]
            else:
                sides=[[r for r in subset if r[c['area_definition']]!='S'],[r for r in subset if r[c['area_definition']]=='S']]
            values=[]
            for sign,group in zip([-1,1],sides):
                image_values=defaultdict(list)
                for r in group:image_values[r['image_id']].append(float(r['task_damaged']) if success else r['stages']['binary_original']['delta'])
                adequate_full &= len(group)>=10 and len(image_values)>=5
                for h,keep in enumerate(halves):
                    selected=[v for i,v in image_values.items() if i in keep]
                    adequate_half &= sum(map(len,selected))>=5 and len(selected)>=3
                    half_value=(np.mean([v for vs in selected for v in vs]) if success else np.mean([np.mean(v) for v in selected])) if selected else float('nan')
                    half_points[h]+=weight*sign*half_value
                value=(np.mean([v for vs in image_values.values() for v in vs]) if success else np.mean([np.mean(v) for v in image_values.values()])) if image_values else float('nan')
                point+=weight*sign*value
                sums=np.zeros(len(ids));den=np.zeros(len(ids))
                for i,vs in image_values.items():
                    sums[index[i]]=sum(vs) if success else np.mean(vs)
                    den[index[i]]=len(vs) if success else 1
                total=sums[resampled].sum(axis=1); divisor=den[resampled].sum(axis=1)
                val=np.divide(total,divisor,out=np.full(2000,np.nan),where=divisor>0)
                draws+=weight*sign*val
        valid=draws[np.isfinite(draws)]
        ci=np.quantile(valid,[.05/16,1-.05/16],method='linear').tolist() if len(valid) else None
        close(point if np.isfinite(point) else None,c['estimate'])
        for x,y in zip(half_points,c['half_estimates']):close(x if np.isfinite(x) else None,y)
        assert len(valid)==c['familywise95']['valid_draws']
        if ci:
            for a,b in zip(ci,c['familywise95']['interval']):close(a,b)
        assert adequate_full==c['adequate_full_counts'] and adequate_half==c['adequate_half_counts']
        passed=bool(adequate_full and adequate_half and np.isfinite(point) and len(valid)==2000 and ci and
                    (ci[0]>0 or ci[1]<0) and all(x*point>0 for x in half_points) and abs(point)>=(.02 if success else .005))
        assert passed==c['stable_exploratory_heterogeneity']
        checked.append(dict(area=c['area_definition'],factor=c['factor'],endpoint=c['endpoint'],equal=True))
    decision=read(run/'ROUTE_DECISION.json')
    supported=[]
    for endpoint in ('damage','target_delta'):
        for factor in ('crowd','area'):
            cs=[c for c in summary['contrasts'] if c['endpoint']==endpoint and c['factor']==factor]
            if len(cs)==2 and all(c['stable_exploratory_heterogeneity'] for c in cs) and cs[0]['estimate']*cs[1]['estimate']>0:
                supported.append(dict(factor=factor,endpoint=endpoint,direction_sign=int(np.sign(cs[0]['estimate']))))
    assert decision.get('supported',[])==supported
    result=dict(status='passed',run_id=run.name,pairs_checked=len(rows),tables_checked=len(summary['tables']),
                scalar_neighbor_max_error=max_error,source_stage_values_exactly_preserved=True,
                all_frozen_contrasts_independently_recomputed=checked,route_selection_equal=True,
                verification_script_sha256=digest(Path(__file__)),summary_sha256=digest(run/'SUMMARY.json'),
                method='Scalar box IoU, fresh table grouping, explicit repeated image indices for all contrast bootstrap draws')
    output=run/'LOCAL_INDEPENDENT_RECHECK.json'
    with output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    verify(Path(args.run).resolve(),Path(__file__).resolve().parents[3])
