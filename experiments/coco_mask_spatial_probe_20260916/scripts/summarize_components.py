"""Report paired seeds, trained components, and fixed baseline failure strata.

GT strata are descriptive analysis only. No subgroup is used by inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for key in ('source','raw_instances','annotations','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference-only',action='store_true')
    p.add_argument('--preservation',action='store_true')
    p.add_argument('--strength-control',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    results=json.loads((a.source/'RESULTS.json').read_text())
    matched={k:set(v) for k,v in json.loads((a.source/'MATCHED_GT75.json').read_text()).items()}
    raw={r['annotation_id']:r for line in a.raw_instances.read_text().splitlines() if (r:=json.loads(line))}
    annotations=json.loads(a.annotations.read_text());anns={r['id']:r for r in annotations['annotations'] if r['id'] in raw}
    categories={r['id']:r['name'] for r in annotations['categories']}
    assert len(raw)==len(anns)==36335
    ids=json.loads((a.source/'SPLIT.json').read_text())['val_ids'];img_index={v:i for i,v in enumerate(ids)}
    nimage=len(ids);ordered=sorted(raw);row_index={v:i for i,v in enumerate(ordered)}
    image=np.array([img_index[anns[i]['image_id']] for i in ordered])
    rng=np.random.default_rng(20260916)
    draws=rng.multinomial(nimage,[1/nimage]*nimage,size=2000).astype(np.float32)
    groups={'all':ordered}
    def size(i):return 'small' if anns[i]['area']<1024 else 'medium' if anns[i]['area']<9216 else 'large'
    for s in ('small','medium','large'):groups['size:'+s]=[i for i in ordered if size(i)==s]
    for state in sorted({r['geometry_state'] for r in raw.values()}):
        groups['baseline_raw:'+state]=[i for i in ordered if raw[i]['geometry_state']==state]
    def fill(i):
        box=anns[i]['bbox'];return raw[i]['mask_area']/max(box[2]*box[3],1e-9)
    bands=[('lt25',0,.25),('25to50',.25,.5),('50to75',.5,.75),('ge75',.75,float('inf'))]
    for label,lo,hi in bands:
        group=[i for i in ordered if lo<=fill(i)<hi]
        groups['fill:'+label]=group
        for s in ('small','medium','large'):
            groups[f'size_fill:{s}:{label}']=[i for i in group if size(i)==s]
    groups['raw_box75_mask_unavailable_small']=[i for i in ordered if size(i)=='small' and raw[i]['geometry_state']=='box_good_mask_unavailable']
    base=matched['baseline']
    strata={}
    for label,group in groups.items():
        b=base&set(group);strata[label]={'gt':len(group),'baseline_matched':len(b),'variants':{}}
        for mode,m in matched.items():
            if mode=='baseline':continue
            good=m&set(group)
            strata[label]['variants'][mode]={'matched':len(good),'repaired':len(good-b),'damaged':len(b-good),
                'delta_recall_points':100*(len(good)-len(b))/len(group) if group else None}
    category=[]
    for cat,name in categories.items():
        group={i for i in ordered if anns[i]['category_id']==cat};b=base&group
        category.append(dict(category_id=cat,name=name,gt=len(group),baseline_matched=len(b),
            changes={k:dict(net_gt=len(m&group)-len(b),repaired=len((m&group)-b),damaged=len(b-(m&group))) for k,m in matched.items() if k!='baseline'}))
    if a.reference_only:
        pairs=[('combo_selected','baseline'),('combo_selected','scalar_selected')]
    else:
        pairs=[('coeff_local4_s0','coeff_s0'),('coeff_local4_s0','local4_s0'),('coeff_s0','scalar_s0'),('local4_s0','scalar_s0')]
        pairs += [(f'coeff_local4_s{s}',f'scalar_s{s}') for s in range(3)]
        pairs += [(f'coeff_local4_s{s}','baseline') for s in range(3)]
        if a.preservation:
            pairs += [(f'local4_s{s}',f'scalar_s{s}') for s in range(3)]
            pairs += [(f'local4_s{s}',f'coeff_local4_s{s}') for s in range(3)]
            pairs += [('local4_guard_s0','local4_s0'),('local4_safe_s0','local4_s0'),('local4_guard_s0','local4_safe_s0')]
            if a.strength_control:
                for objective in ('guard','safe'):
                    pairs += [(f'local4_{objective}_half_s0','local4_s0'),
                              (f'local4_{objective}_half_s0',f'local4_{objective}_s0')]
    contrasts={}
    for left,right in pairs:
        result={}
        for label in ('all','size:small','size:medium','size:large','baseline_raw:box_good_mask_unavailable'):
            group=groups[label];ix=np.array([row_index[i] for i in group]);im=image[ix]
            delta=np.array([int(i in matched[left])-int(i in matched[right]) for i in group],dtype=np.float32)
            d=np.bincount(im,weights=delta,minlength=nimage).astype(np.float32)
            counts=np.bincount(im,minlength=nimage).astype(np.float32)
            samples=draws@d/(draws@counts)*100
            result[label]=dict(gt=len(group),net_gt=int(delta.sum()),delta_recall_points=float(delta.sum()/len(group)*100),
                              ci95=np.quantile(samples,[.025,.975]).tolist())
        contrasts[left+'__minus__'+right]=result
    out={'strata':strata,'category':category,'contrasts':contrasts,
         'limitations':['Baseline raw geometry states are frozen, score-independent and determined on original predictions.',
                        'Outcomes are normal confidence-ranked COCO matches, not remeasured all-raw capabilities.',
                        'GT bbox fill is diagnosis-only. Not deployed as a gate or interpreted as a unique causal shape factor.',
                        'Image-clustered 2000x CI for fixed M75 matches; not AP CI or seed CI. Multiple strata are exploratory.']}
    if not a.reference_only:
        ss={}
        for mode in (('scalar','coeff_local4','local4') if a.preservation else ('scalar','coeff_local4')):
            vals=np.array([100*results[f'{mode}_s{s}']['metrics']['AP'] for s in range(3)])
            ss[mode]={'AP_by_seed':vals.tolist(),'mean_AP':float(vals.mean()),'sample_sd_AP':float(vals.std(ddof=1))}
        dif=np.array(ss['coeff_local4']['AP_by_seed'])-np.array(ss['scalar']['AP_by_seed'])
        ss['paired_delta_AP']={'by_seed':dif.tolist(),'mean':float(dif.mean()),'sample_sd':float(dif.std(ddof=1))}
        out['seed_summary']=ss
        if a.preservation:
            for other in ('scalar','coeff_local4'):
                delta=np.array(ss['local4']['AP_by_seed'])-np.array(ss[other]['AP_by_seed'])
                ss['local4_minus_'+other]={'by_seed':delta.tolist(),'mean':float(delta.mean()),'sample_sd':float(delta.std(ddof=1))}
            reference_good=matched['local4_s0'];reference_repair=reference_good-base;reference_damage=base-reference_good
            transitions={}
            transition_modes = ['local4_guard_s0','local4_safe_s0']
            if a.strength_control:
                transition_modes += ['local4_guard_half_s0','local4_safe_half_s0']
            for mode in transition_modes:
                m=matched[mode];rep=m-base;dam=base-m
                transitions[mode]=dict(retained_original_repairs=len(rep&reference_repair),lost_original_repairs=len(reference_repair-rep),
                    added_repairs=len(rep-reference_repair),rescued_original_damage=len(reference_damage-dam),new_damage=len(dam-reference_damage),
                    original_repair_count=len(reference_repair),original_damage_count=len(reference_damage))
            out['preservation_transitions']=transitions
    (a.out/'ANALYSIS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# 组件与种子结果','','| 方法 | Mask AP | ΔAP | AP75 | AP小 | AP中 | AP大 | 修复 | 误伤 | 净增 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode,v in results.items():
        m=v['metrics'];lines.append(f"| {mode} | {100*m['AP']:.3f} | {100*(m['AP']-results['baseline']['metrics']['AP']):+.3f} | {100*m['AP75']:.3f} | {100*m['APS']:.3f} | {100*m['APM']:.3f} | {100*m['APL']:.3f} | {v['repaired75']} | {v['damaged75']} | {v['repaired75']-v['damaged75']} |")
    if not a.reference_only:lines+=['','## 三种子配对','',json.dumps(out['seed_summary'],ensure_ascii=False,indent=2)]
    lines+=['','## 保持原失败分类的效果','','| 原始状态 | GT数 | 基线Mask75 | 各方法净恢复数 |','|---|---:|---:|---|']
    for label,v in strata.items():
        if label.startswith('baseline_raw:'):
            lines.append(f"| {label} | {v['gt']} | {v['baseline_matched']} | "+', '.join(k+':'+str(m['repaired']-m['damaged']) for k,m in v['variants'].items())+' |')
    lines+=['','统计区间仅对应固定COCO匹配结果的图片重采样，不是AP区间。不同种子仍共用同一训练子集。GT占比与失败分类用于解释，不参与预测。','']
    (a.out/'TABLE.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'seed_summary':out.get('seed_summary'),'all_contrasts':{k:v['all'] for k,v in contrasts.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
