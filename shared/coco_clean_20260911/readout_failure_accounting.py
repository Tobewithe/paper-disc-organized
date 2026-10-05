"""Descriptive failure partition on all GT, using original fixed box matching.

This distinguishes retained-box vs mask recovery states. It does not separate
missing raw candidates from NMS rejection and is not a causal decomposition.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from readout_input_probe import write_json,sha


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    selected=json.loads((args.run/'cache/selection.json').read_text())['transfer']
    status={int(r['annotation_id']):r for r in read(args.run/'cache/gt_status.csv') if r['split']=='transfer'}
    gtrows=[r for r in read(args.run/'task/gt_recovery.csv') if r['arm']=='original_s-1_d0']
    assert set(status)=={int(r['annotation_id']) for r in gtrows}
    # Explicitly use raw fixed bbox matching, not whether a native label survived.
    mapping={}
    for iid in selected:
        # gt_status covers native retention then bbox status; converted-away rare
        # cases cannot be treated as definitely unmatched and are unknown below.
        for r in status.values():
            if int(r['image_id'])==iid:
                mapping[int(r['annotation_id'])]=(True if r['status'] in ['matched','native_empty_support']
                           else False if r['status']=='bbox50_unmatched' else None)
    groups=[]
    for label in ['low','middle','high','nonhigh','all']:
        rr=[r for r in gtrows if label=='all' or
            (float(r['ici'])<=1e-10 if label=='low' else
             1e-10<float(r['ici'])<=.5+1e-10 if label=='middle' else
             float(r['ici'])>.5+1e-10 if label=='high' else float(r['ici'])<=.5+1e-10)]
        categories={k:0 for k in ['recovered75','not_recovered_with_bbox50','not_recovered_no_bbox50','not_recovered_bbox_unknown']}
        for r in rr:
            if r['hit75']=='True':key='recovered75'
            elif mapping[int(r['annotation_id'])] is True:key='not_recovered_with_bbox50'
            elif mapping[int(r['annotation_id'])] is False:key='not_recovered_no_bbox50'
            else:key='not_recovered_bbox_unknown'
            categories[key]+=1
        assert sum(categories.values())==len(rr)
        groups.append(dict(group=label,gt=len(rr),counts=categories,
                       fractions={k:v/len(rr) if rr else None for k,v in categories.items()}))
    high=next(r for r in groups if r['group']=='high');other=next(r for r in groups if r['group']=='nonhigh')
    gap_terms={key:high['fractions'][key]-other['fractions'][key] for key in high['counts'] if key!='recovered75'}
    write_json(args.run/'task/FAILURE_ACCOUNTING.json',dict(groups=groups,
        high_excess_failure_fraction=gap_terms,
        gap=sum(gap_terms.values()),
        source_hashes={p.name:sha(p) for p in [args.run/'cache/gt_status.csv',args.run/'task/gt_recovery.csv']},
        scope='Descriptive mutually exclusive states on all ordinaryGT. Box match uses native-retained fixed class-awarebbox50; '
        'converted-away IDs unknown. No causal attribution or rawcandidate-vsNMS claim. Whole task Mask75 matching is separate.'))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
