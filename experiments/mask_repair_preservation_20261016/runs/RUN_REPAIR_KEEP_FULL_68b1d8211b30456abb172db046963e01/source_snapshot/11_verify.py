"""Independent RLE/count/statistic verification. No objective or solver imports.

Uses an independent compressed-COCO-RLE decoder and NumPy. Can run locally
without Torch/COCO packages. It verifies recorded pixels/statistics, not GPU
re-execution of P; the executor's exact native replay is a separate receipt.
"""
from pathlib import Path
import argparse
import hashlib
import json
from collections import defaultdict
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def decode(rle):
    data=rle['counts'];runs=[];offset=0
    if not isinstance(data,str):raise ValueError('expected compressed string RLE')
    while offset<len(data):
        value=0;shift=0
        while True:
            b=ord(data[offset])-48;offset+=1
            if not 0<=b<=63:raise ValueError('invalid compressed RLE character')
            value|=(b&31)<<(5*shift);shift+=1
            if not b&32:
                if b&16:value|=-1<<(5*shift)
                break
        if len(runs)>2:value+=runs[-2]
        if value<0:raise ValueError('negative RLE run')
        runs.append(value)
    size=int(np.prod(rle['size']))
    if sum(runs)!=size:raise ValueError('RLE total does not match shape')
    flat=np.empty(size,dtype=bool);start=0
    for i,length in enumerate(runs):
        flat[start:start+length]=bool(i%2);start+=length
    return flat.reshape(rle['size'],order='F')


def close(a,b,label):
    if not np.allclose(a,b,atol=1e-12,rtol=0,equal_nan=False):
        raise ValueError(label+': '+str(a)+' != '+str(b))


def verify(run_path):
    path=Path(run_path)
    cfg=read(path/'CONFIG.json');contract=read(path/'EVAL_CONTRACT.json')
    expected=read(path/'STATISTICS.json');panel=read(path/'TARGETS.json')
    rows=[json.loads(line) for line in (path/'PAIR_RESULTS.jsonl').read_text(encoding='utf8').splitlines() if line]
    def identity(r):return (r['image_id'],r['annotation_id'],r['output_detection_id'])
    if len(rows)!=len(panel) or {identity(r) for r in rows}!={identity(r) for r in panel}:
        raise ValueError('panel identity or denominator drift')
    if len({identity(r) for r in rows})!=len(rows):raise ValueError('duplicate rows')
    grouped=defaultdict(list);pixel_checks=0;mask_checks=0;numerical=True
    for row in rows:
        grouped[row['image_id']].append(row)
        truth=decode(row['GT_rle']);target=decode(row['target_grid_rle'])
        native=decode(row['arms']['N']['input_rle']);h,w=row['input_hw'];box=np.asarray(row['boxes_input'],dtype=np.float32)
        x=np.arange(w,dtype=np.float32)[None,:];y=np.arange(h,dtype=np.float32)[:,None]
        roi=(x>=box[0])&(x<box[2])&(y>=box[1])&(y<box[3])
        if target.shape!=(h,w) or native.shape!=(h,w) or int(roi.sum())!=row['roi_pixels']:
            raise ValueError('ROI geometry drift')
        tp=native&target&roi;tn=~native&~target&roi;fp=native&~target&roi;fn=~native&target&roi
        if int(tp.sum())!=row['native_tp']:raise ValueError('native TP count drift')
        for arm,result in row['arms'].items():
            binary=decode(result['original_rle'])
            if binary.shape!=truth.shape:raise ValueError('original mask shape drift')
            union=np.count_nonzero(binary|truth)
            value=np.count_nonzero(binary&truth)/union if union else 0.
            close(value,result['original_iou'],'full-image IoU '+arm);mask_checks+=1
            if arm in ('R_G','R_S1'):continue
            changed=decode(result['input_rle'])
            if changed.shape!=native.shape or (changed&~roi).any():raise ValueError('input crop drift')
            counts=dict(native_tp=int(tp.sum()),native_tn=int(tn.sum()),native_fp=int(fp.sum()),native_fn=int(fn.sum()),
                tp_to_fn=int((tp&~changed).sum()),tn_to_fp=int((tn&changed).sum()),fp_to_tn=int((fp&~changed).sum()),fn_to_tp=int((fn&changed).sum()))
            for key,val in counts.items():
                if val!=result[key]:raise ValueError('transition count drift: '+key)
            pixel_checks+=1
            if arm=='N':continue
            c=(np.asarray(row['c0'],dtype=np.float64)+np.asarray(result['delta_float64'],dtype=np.float64)).astype(np.float32)
            if not np.array_equal(c,np.asarray(result['coefficients_fp32'],dtype=np.float32)):
                raise ValueError('coefficient casting changed')
            if result['proxy_exceeded_pixels']!=0:numerical=False
            if arm in ('C','E','EK'):
                solver=result['solver']
                if solver['roi_pixels']!=int(roi.sum()) or solver['error_pixels']!=int((fp|fn).sum()) or solver['correct_pixels']!=int((tp|tn).sum()):
                    raise ValueError('fixed group denominator drift')
                converged=solver['gradient_inf_norm']<=cfg['solver']['gradient_acceptance']
                if (solver['solver_status']=='converged')!=converged:raise ValueError('convergence label drift')
                if solver['function_evals']>cfg['solver']['max_eval']:raise ValueError('function-evaluation budget exceeded')
                numerical &= converged
        equal=row['arms']['E_equal'];ek=row['arms']['EK'];e=row['arms']['E']
        er,kr=e['actual_delta_logit_rms'],ek['actual_delta_logit_rms']
        scale=kr/er if er else 0.
        close(equal['scale'],scale,'equal scalar')
        close(equal['delta_float64'],np.asarray(e['delta_float64'])*scale,'equal direction')
        error=abs(equal['actual_delta_logit_rms']-kr)
        close(error,equal['rms_match_error'],'equal RMS receipt')
        per_numeric=all(row['arms'][a]['proxy_exceeded_pixels']==0 for a in ('C','E','EK','E_equal'))
        per_numeric &= not(er==0 and kr!=0) and error<=cfg['numerics']['equal_rms_atol']+cfg['numerics']['equal_rms_rtol']*kr
        if bool(per_numeric)!=bool(row['numerical_valid']):raise ValueError('numerical validity receipt differs')
        numerical &= per_numeric

    ids=sorted(grouped);n=len(ids)
    draws=np.random.default_rng(cfg['statistics']['seed']).integers(0,n,size=(cfg['statistics']['bootstrap_replicates'],n))
    def vector(fn):return np.array([sum(fn(r) for r in grouped[i])/len(grouped[i]) for i in ids],dtype=np.float64)
    def receipt(values):
        low,high=np.percentile(np.mean(values[draws],axis=1),[2.5,97.5],method='linear')
        return dict(mean=float(np.mean(values)),ci95=[float(low),float(high)])
    def compare_receipt(actual,wanted,label):
        close(actual['mean'],wanted['mean'],label+' mean');close(actual['ci95'],wanted['ci95'],label+' CI')
    vectors={a:vector(lambda r:r['arms'][a]['original_iou']) for a in expected['metrics']}
    for a,v in vectors.items():
        compare_receipt(receipt(v),expected['metrics'][a]['original_iou'],a)
        compare_receipt(receipt(v-vectors['N']),expected['metrics'][a]['delta_vs_native'],a+' native delta')
        if sum(r['arms'][a]['original_iou']>=.75 for r in rows)!=expected['metrics'][a]['repaired_instances']:
            raise ValueError('repaired-instance count differs')
    for name,record in expected['comparisons'].items():
        a,b=name.split('_minus_');compare_receipt(receipt(vectors[a]-vectors[b]),record,name)
    damage={a:vector(lambda r:r['arms'][a]['tp_to_fn']/max(1,r['native_tp'])) for a in ('C','E','EK','E_equal')}
    for a,v in damage.items():compare_receipt(receipt(v),expected['tp_loss'][a],a+' damage')
    decrease=receipt(damage['E_equal']-damage['EK'])
    compare_receipt(decrease,expected['protection_decrease'],'protection decrease')
    denom=float(damage['E_equal'].mean());relative=decrease['mean']/denom if denom>0 else None
    if relative is None:
        if expected['relative_tp_loss_reduction'] is not None:raise ValueError('empty relative reduction')
    else:close(relative,expected['relative_tp_loss_reduction'],'relative reduction')
    cs={r['output_detection_id'] for r in rows if r['arms']['C']['original_iou']>=.75}
    ks={r['output_detection_id'] for r in rows if r['arms']['EK']['original_iou']>=.75}
    if sorted(ks-cs)!=expected['added_vs_C'] or sorted(cs-ks)!=expected['lost_vs_C']:raise ValueError('witness identity drift')
    if len(cs)!=expected['control_successes'] or len(cs&ks)!=expected['retained_C_successes']:raise ValueError('witness count drift')
    utility=receipt(vectors['EK']-vectors['C']);ei=receipt(vectors['EK']-vectors['E_equal']);g=cfg['gates']
    points=dict(utility=utility['mean']>=g['min_increment_over_C'],protection=relative is not None and relative>=g['min_relative_tp_loss_reduction'],
        equal_iou=ei['mean']>=-g['max_iou_loss_to_equal'],retention=bool(cs) and len(cs&ks)/len(cs)>=g['min_C_witness_retention'])
    intervals=dict(utility=utility['ci95'][0]>0,protection=decrease['ci95'][0]>0)
    complete=len(rows)==cfg['panel']['pairs'] and n==cfg['panel']['images']
    decision=('ENGINEERING_ONLY' if contract['scope']=='smoke' else 'INCONCLUSIVE_NUMERICAL' if not complete or not numerical else
        'FINITE_MECHANISM_SUPPORTED' if all(points.values()) and all(intervals.values()) else
        'CONFIGURATION_REJECTED' if not all(points.values()) else 'INSUFFICIENT_PRECISION')
    if points!=expected['point_gates'] or intervals!=expected['interval_gates'] or decision!=expected['decision']:
        raise ValueError('independent Gate decision differs')
    if bool(numerical)!=expected['numerical_valid'] or complete!=expected['denominator_complete']:raise ValueError('validity state differs')
    return dict(status='passed',verification_scope='independent compressed RLE pixels, transition counts, coefficient casting, paired image statistics and Gates; not GPU re-execution',
        pairs=len(rows),images=n,original_masks_checked=mask_checks,input_masks_checked=pixel_checks,decision=decision,
        pair_results_sha256=hashlib.sha256((path/'PAIR_RESULTS.jsonl').read_bytes()).hexdigest(),
        statistics_sha256=hashlib.sha256((path/'STATISTICS.json').read_bytes()).hexdigest(),numpy_version=np.__version__)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=verify(args.run)
    with Path(args.output).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
