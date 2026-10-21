"""Independent arithmetic, fixed-pair RLE, complete-row, panel, and matching verification.

No models or optimizer. Does not import the evaluation runner or its summarizer.
The laptop additionally verifies every reported pair against GT RLE. A local
receipt can independently check the transferred statistics without pycocotools.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170'
GROUPS=('N','G','A','Q','J','AeqQ','QeqA','AeqJ','JeqA')

def load(p): return json.loads(p.read_text(encoding='utf8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def check(x,msg):
    if not x:raise ValueError(msg)
def key(r):return r['image_id'],r['annotation_id']


def exact_upper(N,n,x):
    # Direct binomial-coefficient sum, independent of production recurrence.
    denominator=math.comb(N,n)
    def accepted(K):
        total=sum(math.comb(K,j)*math.comb(N-K,n-j)
                  for j in range(max(0,n-(N-K)),min(x,n,K)+1))
        return 20*total>=denominator
    lo,hi=x,N-n+x
    while lo<hi:
        mid=(lo+hi+1)//2
        if accepted(mid):lo=mid
        else:hi=mid-1
    return lo


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--verify-rle',action='store_true')
    parser.add_argument('--annotations',type=Path,default=ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json')
    args=parser.parse_args(); run=args.run
    summary=load(run/'SUMMARY.json'); effects=lines(run/'FIXED_PAIR_EFFECTS.jsonl')
    check(load(run/'run.json')['execution_status']=='completed','run not complete')
    check(load(run/'CONFIG_AUDIT.json')['status']=='passed','config audit failed')
    checked=0
    for item in load(run/'ARTIFACT_MANIFEST.json')['artifacts']:
        check(digest(run/item['relative_path'])==item['sha256'],'artifact SHA: '+item['relative_path']); checked+=1
    check(len(effects)==2925 and len(set(map(key,effects)))==2925,'all pair identity')
    baseline=lines(BASE/'ledger/PER_GT_FAILURE.jsonl'); base_by={key(r):r for r in baseline}
    split=load(ROOT/'experiments/mask_readout_mechanism_v2_20261012/runs/RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec/SPLIT_CONTRACT.json')
    dev=set(split['dev_ids']); expected={key(r) for r in baseline if r['image_id'] in dev and not r['is_ignored'] and r['bbox75_match_output_id'] is not None}
    check(set(map(key,effects))==expected,'fixed original pairs changed')
    for r in effects:
        b=base_by[key(r)]
        check(r['baseline_output_detection_id']==b['bbox75_match_output_id'],'pair detector changed')
        check(abs(r['IoU']['N']-b['mask_iou_of_bbox_match'])<1e-12,'native pair value')
        for g in GROUPS:check(r[g+'_delta']==r['IoU'][g]-r['IoU']['N'],'pair arithmetic')
    aggregate={}
    exact_stats=0
    for scope,target,n in [('target',True,257),('all',False,508)]:
        buckets=defaultdict(list)
        for r in effects:
            if not target or r['population']=='T':buckets[r['image_id']].append(r)
        order=sorted(buckets);check(len(order)==n,'effective image denominator')
        if target:check(sum(map(len,buckets.values()))==655,'target denominator')
        resample=np.random.default_rng(1729).integers(0,n,size=(2000,n))
        means={g:np.array([sum(r[g+'_delta'] for r in buckets[i])/len(buckets[i]) for i in order],dtype=np.float64) for g in GROUPS}
        for contrast,record in summary['contrasts'].items():
            a,b=contrast.split('-'); delta=means[a]-means[b]
            observed=[float(delta.mean()),*map(float,np.quantile(delta[resample].mean(axis=1),[.025,.975],method='linear'))]
            recorded=[record[scope]['mean'],*record[scope]['ci95']]
            check(observed==recorded,'exact paired statistic mismatch '+contrast+' '+scope)
            exact_stats+=3;aggregate[contrast+'_'+scope]=observed
    panel=load(run/'DAMAGE_PANEL.json')
    success=sorted(key(r) for r in baseline if not r['is_ignored'] and r['task_mask75']=='success')
    check(len(success)==22957,'baseline success population')
    selected=sorted(random.Random(panel['seed']).sample(success,1024))
    check(selected==list(map(key,panel['sample'])),'sample changed')
    check(digest(BASE/'ledger/PER_GT_FAILURE.jsonl')==panel['baseline_ledger_sha256'],'panel ledger SHA')
    native=load(run/'N_predictions.json');nmap=load(run/'N_ROW_MAPPING.json')['original_global_indices']
    native_by=dict(zip(nmap,native));native_ledger={key(r):r for r in lines(run/'N_GT_LEDGER.jsonl')}
    pair_by_output=defaultdict(list)
    for r in effects:pair_by_output[int(r['baseline_output_detection_id'].rsplit(':',1)[1])].append(r)
    coco=None; pair_checks=0
    if args.verify_rle:
        from pycocotools.coco import COCO
        from pycocotools import mask as masks
        check(digest(args.annotations)==load(ROOT/'paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json')['annotations_sha256'],'GT source SHA')
        coco=COCO(str(args.annotations))
    identity_counts={}; damage_counts={}
    for g in GROUPS:
        predictions=load(run/(g+'_predictions.json'));mapping=load(run/(g+'_ROW_MAPPING.json'))
        indices=mapping['original_global_indices']
        check(len(indices)==len(predictions) and indices==sorted(set(indices)),'prediction row mapping')
        check(mapping['source_prediction_sha256']==panel['source_prediction_sha'],'native prediction source')
        selected_images=set(load(run/'EVAL_CONTRACT.json')['evaluation_ids']) if g in ('N','G','A','Q') else dev
        check(indices==[i for i,r in native_by.items() if r['image_id'] in selected_images],'full rows missing/reordered')
        for index,row in zip(indices,predictions):
            original=native_by[index]
            check({k:v for k,v in row.items() if k!='segmentation'}=={k:v for k,v in original.items() if k!='segmentation'},'nonmask changed')
            if coco is not None:
                if int(masks.area(original['segmentation']))<9216:check(row['segmentation']==original['segmentation'],'gate-off RLE changed')
                for effect in pair_by_output.get(index,[]):
                    target=coco.annToRLE(coco.anns[effect['annotation_id']])
                    actual=float(masks.iou([row['segmentation']],[target],[0])[0,0])
                    check(abs(actual-effect['IoU'][g])<1e-12,'fixed-pair GT RLE IoU '+g);pair_checks+=1
        ledger=lines(run/(g+'_GT_LEDGER.jsonl'));actual={key(r):r for r in ledger}
        check(set(actual)=={k for k,r in base_by.items() if r['image_id'] in selected_images},'full GT coverage')
        for k,r in actual.items():
            check(r['is_ignored']==base_by[k]['is_ignored'] and r['gt_category_id']==base_by[k]['gt_category_id'],'GT semantic identity')
            oid=r['bbox75_match_output_id']
            converted=None if oid is None else panel['source_prediction_sha']+':'+str(indices[int(oid.rsplit(':',1)[1])])
            check(converted==base_by[k]['bbox75_match_output_id'],'COCO bbox match identity')
        if g in ('N','G','A','Q'):
            damaged=sum(actual[k]['task_mask75']=='failure' for k in selected)
            upper=exact_upper(22957,1024,damaged);r=summary['damage'][g]
            check(damaged==r['damaged'] and upper==r['upper_damaged_count'],'exact damage inversion')
            gate='damage_guardrail_passed' if 100*upper<=22957 else 'damage_guardrail_failed' if 100*damaged>1024 else 'blocked_insufficient_precision'
            check(r['gate']==gate,'damage classification');damage_counts[g]={'damaged':damaged,'upper_count':upper,'gate':gate}
        identity_counts[g]=len(predictions)
        del predictions,actual,ledger
    s=summary['contrasts']; damage=summary['damage']['A']
    gates={'extra_utility_A_G':s['A-G']['target']['mean']>=.005 and s['A-G']['target']['ci95'][0]>0,
           'S2_repair_A_N':s['A-N']['target']['mean']>=.005 and s['A-N']['target']['ci95'][0]>0,
           'S2_all_pairs':s['A-N']['all']['mean']>=-.001,
           'S2_damage':damage['gate']=='damage_guardrail_passed','identity_and_configuration':True}
    check(gates==summary['gates'],'gate decision changed')
    mechanism={k:s[k]['target']['ci95'][0]>0 for k in ('A-Q','A-J','AeqQ-QeqA','AeqJ-JeqA')}
    check(mechanism==summary['mechanism_gates'] and all(mechanism.values())==summary['mechanism_supported'],'mechanism classification')
    status='passed' if all(gates.values()) else 'blocked' if all(v for k,v in gates.items() if k!='S2_damage') and damage['gate']=='blocked_insufficient_precision' else 'failed'
    check(status==summary['utility_status']==load(run/'DECISION.json')['status'],'final decision')
    amplitude_rows=0; reasons=defaultdict(int)
    amplitude_records=lines(run/'AMPLITUDE_CONTROLS.jsonl')
    check(len(amplitude_records)==1024,'512 images times two matched comparisons required')
    check({(r['image_id'],r['pair']) for r in amplitude_records}=={(i,p) for i in dev for p in ('A-Q','A-J')},'matched-control image coverage')
    for record in amplitude_records:
        check(record['uses_GT'] is False and record['all_rows_retained'],'GT or row exclusion in amplitude control')
        columns=['q_first','q_second','q_matched_first','q_matched_second','scale_first','scale_second','roi_pixels','reason']
        check(len({len(record[k]) for k in columns})==1,'amplitude receipt row lengths')
        for a,b,after_a,after_b,sa,sb,pixels,reason in zip(*(record[k] for k in columns)):
            amplitude_rows+=1;reasons[reason]+=1
            check(all(math.isfinite(v) for v in (a,b,after_a,after_b,sa,sb)),'nonfinite amplitude')
            check(0<=sa<=1 and 0<=sb<=1,'amplified control')
            if reason=='matched':
                check(pixels>0 and min(a,b)>1e-12,'invalid active control')
                common=min(a,b)
                check(abs(sa-common/a)<=2*math.ulp(common/a) and abs(sb-common/b)<=2*math.ulp(common/b),'wrong symmetric downward scale')
                check(abs(after_a-common)<=1e-7+1e-5*common and abs(after_b-common)<=1e-7+1e-5*common,'functional norm changed')
            else:
                check(reason in ('gate_off','empty_roi','near_zero_direction'),'unregistered control exclusion')
                check(sa==sb==after_a==after_b==0,'zero control not native')
            check(abs(after_a-after_b)<=1e-7+1e-5*abs(after_b),'matched norm tolerance')
    result={'status':'passed','run_id':run.name,'verifier_sha256':digest(Path(__file__)),
            'summary_sha256':digest(run/'SUMMARY.json'),'python':sys.version,'numpy':np.__version__,
            'artifact_hashes_checked':checked,'statistics_exact_scalar_checks':exact_stats,
            'full_prediction_identity_rows':identity_counts,'damage_independent_integer_inversion':damage_counts,
            'GT_RLE_pair_checks':pair_checks,'GT_RLE_check_enabled':args.verify_rle,
            'matched_amplitude_rows_checked':amplitude_rows,'amplitude_row_reasons':dict(reasons),
            'gates':gates,'scientific_status':status,'new_model_forwards':0,'new_training_steps':0}
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
