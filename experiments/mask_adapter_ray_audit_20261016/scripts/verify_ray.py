"""Independent RLE/event/FP32/paired-statistic verification; no torch or search imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def require(ok,message):
    if not ok:raise AssertionError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(rle):
    lengths=[];code=rle['counts'];cursor=0
    while cursor<len(code):
        value=shift=0
        while True:
            token=ord(code[cursor])-48;cursor+=1
            require(0<=token<64,'invalid RLE code');value|=(token&31)<<(5*shift);shift+=1
            if not token&32:
                if token&16:value|=-1<<(5*shift)
                break
        if len(lengths)>2:value+=lengths[-2]
        require(value>=0,'negative RLE length');lengths.append(value)
    require(sum(lengths)==np.prod(rle['size']),'RLE area mismatch')
    pos=0;foreground=[]
    for i,n in enumerate(lengths):
        if i%2 and n:foreground.append((pos,pos+n))
        pos+=n
    return foreground


def iou(a,b):
    require(a['size']==b['size'],'RLE shape differs')
    x,y=intervals(a),intervals(b);i=j=intersection=0
    while i<len(x) and j<len(y):
        intersection+=max(0,min(x[i][1],y[j][1])-max(x[i][0],y[j][0]))
        if x[i][1]<=y[j][1]:i+=1
        else:j+=1
    union=sum(v-u for u,v in x)+sum(v-u for u,v in y)-intersection
    return intersection/union if union else 0.


def close(a,b,message):
    require(np.allclose(a,b,rtol=0,atol=1e-12),message)


def verify(run):
    run=Path(run);cfg=read(run/'CONFIG.json');contract=read(run/'EVAL_CONTRACT.json');stats=read(run/'STATISTICS.json')
    rows=[json.loads(line) for line in (run/'PAIR_RESULTS.jsonl').read_text().splitlines() if line.strip()]
    targets=read(run/'TARGETS.json');targetmap={t['output_detection_id']:t for t in targets}
    require(len(rows)==len(targetmap)==len(targets),'target denominator')
    require({r['output_detection_id'] for r in rows}==set(targetmap),'panel identity')
    frozen=read(run/'SOURCE_MANIFEST.json');mapping=read(run/'SOURCE_SNAPSHOT_MAP.json')
    for rel,path in mapping.items():require(sha(run/path)==frozen['files'][rel],'source snapshot hash')
    require(sha(run/'PROTOCOL.md')==frozen['files']['experiments/mask_adapter_ray_audit_20261016/PROTOCOL.md'],'protocol hash')
    root=Path(__file__).resolve().parents[3]
    source=root/'experiments/mask_repair_preservation_20261016/runs'/('RUN_REPAIR_KEEP_FULL_68b1d8211b30456abb172db046963e01' if contract['scope']=='full' else 'RUN_REPAIR_KEEP_SMOKE_1473cf76889b4163b962d2e7086995b7')
    controls={r['output_detection_id']:r for r in map(json.loads,(source/'PAIR_RESULTS.jsonl').read_text().splitlines())}
    require(sha(source/'PAIR_RESULTS.jsonl')==contract['input_files'][(source/'PAIR_RESULTS.jsonl').relative_to(root).as_posix()],'reference C hash')
    history={}
    if contract['scope']=='full':
        hpath=root/'experiments/mask_opportunity_audit_20261015/runs/RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd/PAIRED_MASKS.jsonl'
        require(sha(hpath)==contract['input_files'][hpath.relative_to(root).as_posix()],'historical reference hash')
        history={r['output_detection_id']:r for r in map(json.loads,hpath.read_text().splitlines())}
    decoded=events_checked=actual_candidates=0
    for row in rows:
        identity=row['output_detection_id'];old=controls[identity]
        for key,value in targetmap[identity].items():require(row[key]==value,'target field changed: '+key)
        require(row['GT_rle']==old['GT_rle'] and row['native_prediction']==old['native_prediction'] and row['c0']==old['c0'],'source identity')
        require(row['arms']['N']['original_rle']==old['arms']['N']['original_rle'],'native source RLE')
        require(row['arms']['C']['original_rle']==old['arms']['C']['original_rle'],'C source RLE')
        if history:
            require(row['arms']['U']['original_rle']==history[identity]['ungated_rle'],'U source RLE')
            require(row['arms']['G']['original_rle']==history[identity]['gated_rle'],'G source RLE')
        for arm,record in row['arms'].items():
            close(record['original_iou'],iou(record['original_rle'],row['GT_rle']),'arm IoU');decoded+=1
        event_path=run/row['sweep_file'];require(sha(event_path)==row['sweep_sha256'],'sweep receipt hash')
        with np.load(event_path,allow_pickle=False) as data:
            times=data['times'];maximum=float(data['alpha_max']);require(maximum==cfg['ray']['alpha_max'],'alpha bound')
            require(np.all(np.diff(times)>0) and np.all((times>0)&(times<maximum)),'event order')
            require(len(times)==len(data['delta_area'])==len(data['delta_intersection']),'event lengths')
            boundaries=np.r_[0.,times,maximum];midpoints=(boundaries[1:]+boundaries[:-1])*.5
            area=np.cumsum(np.r_[int(data['initial_area']),data['delta_area']])
            intersection=np.cumsum(np.r_[int(data['initial_intersection']),data['delta_intersection']])
            gt_area=sum(v-u for u,v in intervals(row['GT_rle']))
            require(int(data['gt_area'])==gt_area,'event GT area')
            require(np.all((intersection>=0)&(intersection<=gt_area)&(intersection<=area)),'event counts')
            valid=(midpoints>boundaries[:-1])&(midpoints<boundaries[1:]);midpoints=midpoints[valid]
            scores=intersection[valid]/np.maximum(1,gt_area+area[valid]-intersection[valid])
            chosen=sorted(range(len(midpoints)),key=lambda i:(-scores[i],abs(midpoints[i]-1),midpoints[i]))[:cfg['ray']['top_candidates']]
            candidates=[] if bool(data['zero_direction']) else [float(midpoints[i]) for i in chosen]
            candidates=list(dict.fromkeys(candidates+cfg['ray']['fixed_candidates']))
            require(candidates==row['candidate_alphas'],'candidate selection differs')
            require(len(midpoints)==row['sweep_interval_count'],'sweep interval count');close(scores.max(),row['proxy_interval_best_iou'],'sweep optimum')
            events_checked+=len(times)
        require(len(row['candidates'])==len(candidates)<=cfg['ray']['max_actual_candidates'],'candidate count')
        numerical=True
        for c,a in zip(row['candidates'],candidates,strict=True):
            require(c['alpha']==a,'candidate alpha')
            expected=(np.asarray(row['c0'],dtype=np.float64)+a*np.asarray(row['adapter_delta'],dtype=np.float64)).astype(np.float32)
            require(np.array_equal(expected,np.asarray(c['coefficients_fp32'],dtype=np.float32)),'coefficient ray changed')
            actual=iou(c['original_rle'],row['GT_rle']);proxy=iou(c['proxy_original_rle'],row['GT_rle']);decoded+=2
            close(c['original_iou'],actual,'candidate actual IoU');close(c['proxy_original_iou'],proxy,'candidate proxy IoU')
            close(c['proxy_original_iou_error'],abs(actual-proxy),'candidate numeric error')
            valid=c['proxy_logit_exceeded_pixels']==0 and abs(actual-proxy)<=cfg['numerics']['original_iou_tolerance']
            require(c['numerical_valid']==valid,'candidate validity');numerical &= valid;actual_candidates+=1
        require(row['numerical_valid']==numerical,'row validity')
        chosen=min(range(len(candidates)),key=lambda j:(-row['candidates'][j]['original_iou'],abs(candidates[j]-1),candidates[j]))
        require(chosen==row['selected_candidate_index'] and row['arms']['R']==row['candidates'][chosen],'actual selection differs')
        for alpha,arm in ((0.,'N'),(1.,'U')):
            endpoint=row['candidates'][candidates.index(alpha)]
            require(endpoint['original_rle']==row['arms'][arm]['original_rle'],'endpoint replay')
            require(row['arms']['R']['original_iou']>=endpoint['original_iou'],'selected point below endpoint')
    ids=sorted({r['image_id'] for r in rows});groups=[[r for r in rows if r['image_id']==i] for i in ids]
    draws=np.random.default_rng(cfg['statistics']['seed']).integers(len(ids),size=(cfg['statistics']['bootstrap_replicates'],len(ids)))
    arrays={a:np.array([sum(r['arms'][a]['original_iou'] for r in group)/len(group) for group in groups]) for a in stats['metrics']}
    def check_receipt(v,receipt):
        close(v.mean(),receipt['mean'],'macro mean');close(np.quantile(v[draws].mean(axis=1),[.025,.975]),receipt['ci95'],'paired CI')
    for arm,v in arrays.items():
        check_receipt(v,stats['metrics'][arm]['original_iou']);check_receipt(v-arrays['N'],stats['metrics'][arm]['delta_vs_N'])
        require(sum(r['arms'][arm]['original_iou']>=.75 for r in rows)==stats['metrics'][arm]['repaired'],'Mask75 count')
        if arm!='R':check_receipt(arrays['R']-v,stats['comparisons']['R_minus_'+arm])
    gain=arrays['R']-arrays['U'];gap=arrays['C']-arrays['U'];gap_draw=gap[draws].mean(axis=1)
    ccounts=np.array([sum(r['arms']['C']['original_iou']>=.75 for r in group) for group in groups])
    retained=np.array([sum(r['arms']['C']['original_iou']>=.75 and r['arms']['R']['original_iou']>=.75 for r in group) for group in groups])
    reference=bool(gap.mean()>0 and np.all(gap_draw>0) and ccounts.sum()>0 and np.all(ccounts[draws].sum(axis=1)>0))
    if reference:
        q=float(gain.mean()/gap.mean());qci=np.quantile(gain[draws].mean(axis=1)/gap_draw,[.025,.975])
        capture=float(retained.sum()/ccounts.sum());cci=np.quantile(retained[draws].sum(axis=1)/ccounts[draws].sum(axis=1),[.025,.975])
        close(q,stats['opportunity_capture']['value'],'capture q');close(qci,stats['opportunity_capture']['ci95'],'q CI')
        close(capture,stats['C_witness_capture']['value'],'witness capture');close(cci,stats['C_witness_capture']['ci95'],'witness CI')
    require(stats['reference_valid']==reference,'reference validity')
    numerical=all(r['numerical_valid'] for r in rows);require(stats['numerical_valid']==numerical,'numeric status')
    if contract['scope']=='smoke':decision='ENGINEERING_ONLY'
    elif not numerical:decision='INCONCLUSIVE_NUMERICAL'
    elif not reference:decision='INSUFFICIENT_REFERENCE'
    elif q>=cfg['gates']['amplitude_capture'] and capture>=cfg['gates']['amplitude_capture'] and gain.mean()>=cfg['gates']['min_gain_over_U'] and np.quantile(gain[draws].mean(axis=1),.025)>0:decision='AMPLITUDE_PRIORITY'
    elif qci[1]<cfg['gates']['direction_upper_capture'] and cci[1]<cfg['gates']['direction_upper_capture']:decision='DIRECTION_PRIORITY'
    else:decision='MIXED_DIRECTION_AMPLITUDE'
    require(stats['decision']==decision,'decision mismatch')
    if contract['scope']=='full':require(len(rows)==256 and len(ids)==243 and int(ccounts.sum())==159,'full frozen denominators')
    if (run/'ARTIFACT_MANIFEST.json').exists():
        for record in read(run/'ARTIFACT_MANIFEST.json')['artifacts']:require(sha(run/record['relative_path'])==record['sha256'],'original artifact hash')
    return dict(status='passed',pairs=len(rows),images=len(ids),original_RLE_checks=decoded,actual_candidate_checks=actual_candidates,event_transitions_checked=events_checked,
        source_snapshots=len(mapping),decision=decision,independent_no_torch_or_search_imports=True,local_GPU_replay_claim=False,statistics_sha256=sha(run/'STATISTICS.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=verify(args.run);target=Path(args.run)/args.output
    with target.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
