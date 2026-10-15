"""Descriptive decomposition prespecified in PROTOCOL; no selection or tuning."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(p):
    with p.open(encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def analyze(run):
    result={}
    for arm in ('A','B'):
        effects=list(rows(run/f'{arm}_FIXED_PAIR_EFFECTS.jsonl'));groups={}
        for label,group in [('T',[r for r in effects if r['population']=='T']),('all_bbox75',effects)]:
            summary={}
            for selected in (False,True):
                rr=[r for r in group if r['applied']] if selected else group
                counts=Counter();old=Counter();means=Counter()
                for r in rr:
                    counts.update(r['changed_pixels']['method']);old.update(r['changed_pixels']['old_E1'])
                    for key in ('tp','fp','fn'):
                        means['method_minus_native_'+key]+=r['confusion']['method'][key]-r['confusion']['native'][key]
                        means['old_minus_native_'+key]+=r['confusion']['old_E1'][key]-r['confusion']['native'][key]
                summary['applied' if selected else 'full_denominator']=dict(pairs=len(rr),positive=sum(r['delta_iou']>0 for r in rr),negative=sum(r['delta_iou']<0 for r in rr),
                    unchanged=sum(r['delta_iou']==0 for r in rr),changed_pixel_totals=dict(counts),old_E1_changed_pixel_totals=dict(old),confusion_total_differences=dict(means))
            groups[label]=summary
        result[arm]=groups
    result['scope']='descriptive pixel totals; not IoU Gates; objects may overlap, totals are per fixed pair'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    result=analyze(a.run)
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
