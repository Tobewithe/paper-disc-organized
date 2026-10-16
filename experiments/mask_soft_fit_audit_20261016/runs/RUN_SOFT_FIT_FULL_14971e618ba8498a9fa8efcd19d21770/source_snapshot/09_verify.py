"""Independent identities, ledger counts, raw RLE geometry and frozen statistical reproduction."""
from common import *
import argparse
from collections import Counter,defaultdict
import math
from audit_stats import build_summary

def close(a,b,label='value'):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),'keys differ: '+label)
        for k,v in a.items(): close(v,b[k],label+'.'+k)
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),'list differs: '+label)
        for i,(v,w) in enumerate(zip(a,b)): close(v,w,label+str(i))
    elif isinstance(a,float):
        require(isinstance(b,(int,float)) and abs(a-b)<=1e-12,'number differs: '+label)
    else:
        require(a==b,'value differs: '+label)

def intervals(rle):
    # Independent integer decoder, same COCO RLE format used in the prior opportunity verifier.
    counts=[]; cursor=0; code=rle['counts']
    if isinstance(code,bytes): code=code.decode('ascii')
    while cursor<len(code):
        value=shift=0
        while True:
            token=ord(code[cursor])-48; cursor+=1
            require(0<=token<64,'bad RLE token')
            value|=(token&31)<<(5*shift); shift+=1
            if not token&32:
                if token&16: value|=-1<<(5*shift)
                break
        if len(counts)>2: value+=counts[-2]
        require(value>=0,'negative RLE count')
        counts.append(value)
    require(sum(counts)==rle['size'][0]*rle['size'][1],'bad RLE shape')
    pos=0; out=[]
    for i,n in enumerate(counts):
        if i%2 and n: out.append((pos,pos+n))
        pos+=n
    return out

def area(rle):
    return sum(b-a for a,b in intervals(rle))

def iou(a,b):
    require(a['size']==b['size'],'mask shape mismatch')
    x,y=intervals(a),intervals(b)
    i=j=inter=0
    while i<len(x) and j<len(y):
        inter+=max(0,min(x[i][1],y[j][1])-max(x[i][0],y[j][0]))
        if x[i][1]<=y[j][1]: i+=1
        else: j+=1
    union=sum(v-u for u,v in x)+sum(v-u for u,v in y)-inter
    return inter/union if union else 0.

def ledger_checks(run,prefix,rows,tasks):
    paths = ['BASELINE_GT_LEDGER.jsonl','U_GT_LEDGER.jsonl','G_GT_LEDGER.jsonl'] if prefix=='FIT' else ['DEV_BASELINE_GT_LEDGER.jsonl','DEV_U_GT_LEDGER.jsonl','DEV_G_GT_LEDGER.jsonl']
    ledgers=[{(r['image_id'],r['annotation_id']):r for r in read_jsonl(run/p)} for p in paths]
    require(set(ledgers[0])==set(ledgers[1])==set(ledgers[2]),'independent GT domain')
    for r in tasks:
        key=(r['image_id'],r['annotation_id'])
        entries=[d[key] for d in ledgers]
        require(all(not a['is_ignored'] for a in entries),'ignored task included')
        require([r[k] for k in ('baseline_success','U_success','G_success')]==[a['task_mask75']=='success' for a in entries],'task success flags differ')
    require(len(tasks)==sum(not r['is_ignored'] for r in ledgers[0].values()),'task denominator mismatch')
    for r in rows:
        key=(r['image_id'],r['annotation_id'])
        entries=[d[key] for d in ledgers]
        require(entries[0]['bbox75_match_output_id']==r['baseline_output_detection_id'],'pair identity differs')
        for field,a in zip(('baseline_iou','U_iou','G_iou'),entries):
            require(abs(r[field]-a['mask_iou_of_bbox_match'])<1e-12,'fixed-pair IoU differs from ledger')
        require(r['G_iou']==(r['U_iou'] if r['gate_enabled'] else r['baseline_iou']),'gated fixed pair formula')
    return len(tasks),len(rows)

def raw_fit_checks(run,rows,coverage):
    from s1_geometry import iter_prediction_array
    from pycocotools.coco import COCO
    mapping=read_json(run/'ROW_MAPPING.json')
    indices=mapping['original_global_indices']; wanted=set(indices)
    predpath=Path(read_json(CACHE/'trace/TRACE_MANIFEST.json')['predictions_path'])
    native={i:r for i,r in enumerate(iter_prediction_array(predpath)) if i in wanted}
    u=read_json(run/'U_predictions.json'); g=read_json(run/'G_predictions.json')
    require(len(u)==len(g)==len(indices),'raw prediction length differs')
    corrected={i:(a,b) for i,a,b in zip(indices,u,g)}
    count=Counter(); gated=Counter()
    for index,a,b in zip(indices,u,g):
        n=native[index]
        require(set(n)==set(a)==set(b),'raw field set changed')
        require(all(n[k]==a[k]==b[k] for k in n if k!='segmentation'),'raw nonmask drift')
        ar=area(n['segmentation'])
        require(b['segmentation']==(a['segmentation'] if ar>=9216 else n['segmentation']),'raw mixed RLE wrong')
        count[n['image_id']]+=1; gated[n['image_id']]+=int(ar>=9216)
    for r in coverage:
        require((r['outputs'],r['gate_outputs'])==(count[r['image_id']],gated[r['image_id']]),'gate coverage mismatch')
    coco=COCO(str(GT))
    maximum=0.
    for r in rows:
        index=int(r['baseline_output_detection_id'].rsplit(':',1)[1])
        n=native[index]; a,b=corrected[index]
        ann=coco.anns[r['annotation_id']]
        require(n['image_id']==ann['image_id']==r['image_id'] and n['category_id']==ann['category_id']==r['category_id'],'raw pair image/class mismatch')
        gt=coco.annToRLE(ann)
        require(area(n['segmentation'])==r['native_area'],'native area differs')
        for value,pred in zip(('baseline_iou','U_iou','G_iou'),(n,a,b)):
            error=abs(iou(pred['segmentation'],gt)-r[value]); maximum=max(maximum,error)
            require(error<1e-12,'independent integer RLE IoU mismatch')
    return dict(all_output_rows=len(indices),fixed_pairs=len(rows),maximum_integer_RLE_IoU_error=maximum,gate_mask_selection_exact=True,nonmask_fields_exact=True)

def verify(run,smoke,laptop):
    omitted=[]
    for path in (run,smoke):
        require(read_json(path/'run.json')['execution_status']=='completed','Run incomplete')
        require(read_json(path/'CONFIG_AUDIT.json')['status']=='passed','configuration audit failed')
        for entry in read_json(path/'ARTIFACT_MANIFEST.json')['artifacts']:
            p=path/entry['relative_path']
            if not p.exists() and not laptop and p.name in ('U_predictions.json','G_predictions.json'):
                omitted.append(dict(run=path.name,**entry)); continue
            require(sha256(p)==entry['sha256'],'artifact hash mismatch: '+str(p))
        manifest=read_json(path/'SOURCE_MANIFEST.json')
        require(manifest['files']==read_json(STUDY/'FREEZE_MANIFEST.json')['files'],'freeze mismatch')
        require(manifest['frozen_at']<read_json(path/'run.json')['started_at'],'freeze after start')
        for rel,p in read_json(path/'SOURCE_SNAPSHOT_MAP.json').items():
            require(sha256(path/p)==manifest['files'][rel],'source snapshot mismatch')
    train=read_jsonl(run/'FIT_FIXED_PAIRS.jsonl'); dev=read_jsonl(run/'DEV_FIXED_PAIRS.jsonl')
    train_tasks=read_jsonl(run/'FIT_TASKS.jsonl'); dev_tasks=read_jsonl(run/'DEV_TASKS.jsonl')
    coverage=read_jsonl(run/'COVERAGE_BY_IMAGE.jsonl'); losses=read_jsonl(run/'LOSS_BY_IMAGE.jsonl')
    contract=read_json(run/'EVAL_CONTRACT.json')
    require(len(coverage)==len(losses)==2000,'fit image denominator')
    require({r['image_id'] for r in coverage}=={r['image_id'] for r in losses}==set(contract['fit_ids']),'fit image identity')
    require(len(contract['dev_ids'])==512 and not set(contract['dev_ids'])&set(contract['fit_ids']),'split overlap')
    identities={}
    for prefix,rows,tasks in (('FIT',train,train_tasks),('DEV',dev,dev_tasks)):
        require(len({(r['image_id'],r['annotation_id']) for r in rows})==len(rows),'duplicate fixed GT')
        identities[prefix]=ledger_checks(run,prefix,rows,tasks)
    supervision=read_jsonl(run/'TRAINING_PAIRS.jsonl')
    require(len(supervision)==14276 and sum(r['valid_supervision'] for r in supervision)==14268,'supervision denominator')
    require(sum(r['gate_enabled'] and r['valid_supervision'] for r in supervision)==sum(r['gate_valid_supervised'] for r in coverage),'gate supervision count')
    result=build_summary(train,dev,train_tasks,dev_tasks,coverage,losses,'full')
    summary=read_json(run/'SUMMARY.json')
    for k,v in result.items(): close(v,summary[k],k)
    state=read_json(run/'FROZEN_MODEL_STATE.json')
    require(state['before']==state['after'] and state['new_optimizer_steps']==0,'state drift')
    raw=None
    if laptop:
        inspect_inputs()
        raw=raw_fit_checks(run,train,coverage)
    return dict(status='passed',at=now(),interpreter=sys.executable,python=sys.version,frozen_verifier_sha256=sha256(Path(__file__)),fit_images=2000,dev_images=512,identities=identities,statistics_tolerance=1e-12,all_frozen_statistics_recomputed=True,original_supervision_counts_verified=True,raw_fit_verification=raw,large_predictions_retained_on_laptop=omitted,decision=summary['decision'])

def main():
    ap=argparse.ArgumentParser()
    for name in ('run','smoke-run','out'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--laptop-input-audit',action='store_true')
    a=ap.parse_args()
    result=verify(a.run,a.smoke_run,a.laptop_input_audit)
    write_new(a.out,result)
    print(json.dumps(dict(status=result['status'],fit_images=2000,decision=result['decision'])),flush=True)

if __name__=='__main__':
    main()
