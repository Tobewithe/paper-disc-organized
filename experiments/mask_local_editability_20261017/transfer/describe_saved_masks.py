"""Closeout arithmetic on saved masks only; does not alter the frozen Gate."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from pycocotools import mask as masks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('out')
    a=p.parse_args()
    run=Path(a.run)
    rows=[json.loads(s) for s in (run/'PAIRS.jsonl').read_text(encoding='utf8').splitlines()]
    details=[]
    for r in rows:
        n,c,y,b,roi=[masks.decode(r[k]).astype(bool) for k in
                     ('native_rle','teacher_rle','GT_rle','B_rle','ROI_rle')]
        high=roi & ~b
        outside=~roi
        details.append(dict(image_id=r['image_id'],annotation_id=r['annotation_id'],
            C_success=r['arms']['C']['iou']>=.75,O_B_success=r['arms']['O_B']['iou']>=.75,
            C_B_success=r['arms']['C_B']['iou']>=.75,
            gt_pixels=int(y.sum()),high_conf_locked_FN=int((~n&y&high).sum()),
            high_conf_locked_FP=int((n&~y&high).sum()),outside_bbox_FN=int((~n&y&outside).sum()),
            outside_bbox_FP=int((n&~y&outside).sum()),
            teacher_corrects_high_conf_errors=int(((n!=y)&(c==y)&high).sum()),
            teacher_damages_high_conf_correct=int(((n==y)&(c!=y)&high).sum()),
            teacher_corrects_outside_bbox=int(((n!=y)&(c==y)&outside).sum())))
    lost=[r for r in details if r['C_success'] and not r['O_B_success']]
    result=dict(description='Post-closeout descriptive arithmetic, no additional intervention or Gate change',
                source_pairs_sha256=hashlib.sha256((run/'PAIRS.jsonl').read_bytes()).hexdigest(),
                lost_C_witnesses=len(lost),
                lost_C_witness_ids=[r['annotation_id'] for r in lost],
                lost_with_teacher_corrections_in_locked_high_conf=sum(r['teacher_corrects_high_conf_errors']>0 for r in lost),
                lost_with_teacher_corrections_outside_bbox=sum(r['teacher_corrects_outside_bbox']>0 for r in lost),
                lost_witness_pixel_totals={k:sum(r[k] for r in lost) for k in details[0]
                                          if k not in ('image_id','annotation_id','C_success','O_B_success','C_B_success')},
                details=details,new_model_calls=0,new_training_steps=0)
    with Path(a.out).open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))


if __name__=='__main__':
    main()
