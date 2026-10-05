"""Freeze candidate reconciliation and reduced original COCO annotation inputs."""
import argparse,json,csv
from pathlib import Path
from collections import Counter

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--coco',type=Path,required=True)
a=p.parse_args();cur=json.loads((a.run/'current_candidate_identities.json').read_text());old=json.loads((a.run/'official_candidate_identities.json').read_text())
summary={};diff=[]
for group in ['fit','dev','val']:
    by_old={(x['image_id'],x['annotation_id']):x for x in old[group]}
    by_new={(x['image_id'],x['annotation_id']):x for x in cur[group]}
    counts=Counter()
    for key in sorted(set(by_old)|set(by_new)):
        o,n=by_old.get(key),by_new.get(key)
        if o and n:
            same=all(o[k]==n[k] for k in ['raw_id','pyramid_level','branch'])
            reason='identical_candidate' if same else 'same_GT_different_assigned_raw_candidate'
        elif o:reason='official_positive_GT_absent_from_Box50_geometric_cache'
        else:reason='Box50_mapped_GT_without_official_positive'
        counts[reason]+=1
        diff.append(dict(group=group,image_id=key[0],annotation_id=key[1],reason=reason,old=o,current=n))
    summary[group]=dict(official_candidates=len(old[group]),current_candidates=len(cur[group]),official_effective_images=len({x['image_id'] for x in old[group]}),current_effective_images=len({x['image_id'] for x in cur[group]}),counts=dict(counts))
(a.run/'candidate_differences.json').write_text(json.dumps(diff,indent=2),encoding='utf-8')
(a.run/'candidate_reconciliation.json').write_text(json.dumps(dict(summary=summary,explanation='Official YOLODataset+TAL assignment versus native_bank.py score/class-independent max-cardinality BoxIoU>=0.5 geometry. Neither is a confidence gate; different preprocessing/label formats may also affect correspondence. Frozen Phase A solutions are defined on current geometric cache only; no candidate re-selection.'),indent=2),encoding='utf-8')
for domain,groups in [('train',['fit','dev']),('val',['val'])]:
    wanted={x['annotation_id'] for g in groups for x in cur[g]+old[g]}
    source=json.loads((a.coco/f'instances_{domain}2017.json').read_text(encoding='utf-8'))
    subset={int(x['id']):x for x in source['annotations'] if x['id'] in wanted}
    assert len(subset)==len(wanted),(domain,len(subset),len(wanted))
    (a.run/f'{domain}_annotations_subset.json').write_text(json.dumps(subset),encoding='utf-8')
    del source
print(json.dumps(summary,indent=2))
