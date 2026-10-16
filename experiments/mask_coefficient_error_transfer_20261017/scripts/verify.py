"""Separate-process RLE/fit/statistical audit; --local performs no CUDA work."""
import argparse,math,time
from collections import defaultdict,Counter
import numpy as np
from common import *


def spans(rle):
    counts=[];s=rle['counts'];p=0
    while p<len(s):
        value=0;shift=0
        while True:
            c=ord(s[p])-48;p+=1;value|=(c&31)<<shift;shift+=5
            if not c&32:
                if c&16:value|=(-1)<<shift
                break
        if len(counts)>2:value+=counts[-2]
        require(value>=0,'negative decoded RLE count');counts.append(value)
    require(sum(counts)==math.prod(rle['size']),'RLE area domain')
    result=[];position=0
    for i,count in enumerate(counts):
        if i%2 and count:result.append((position,position+count))
        position+=count
    return result


def intersect(a,b):
    i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if hi>lo:out.append((lo,hi))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def area(a):return sum(y-x for x,y in a)


def independent_values(pred,gt,native):
    p=spans(pred);pg=intersect(p,gt);common=intersect(p,native)
    intersection=area(pg);union=area(p)+area(gt)-intersection
    native_tp=area(intersect(native,gt));common_tp=area(intersect(common,gt))
    return dict(iou=intersection/union if union else 0.,intersection=intersection,union=union,
        native_TP_to_FN=native_tp-common_tp,native_FP_to_TN=area(native)-native_tp-(area(common)-common_tp))


def statistical_verify(path,rows,config):
    summary=read_json(path/'SUMMARY.json');ids=read_json(path/'STATISTICAL_IMAGE_ORDER.json')
    require(ids==sorted({r['image_id'] for r in rows}),'image grouping')
    draws=np.load(path/'BOOTSTRAP_INDICES.npz')['indices']
    require(np.array_equal(draws,np.random.Generator(np.random.PCG64(config['seed'])).integers(len(ids),size=(config['bootstrap_resamples'],len(ids)))),'bootstrap identity')
    by_image=defaultdict(list)
    for r in rows:by_image[r['image_id']].append(r)
    estimates={}
    for key,want in summary['effects'].items():
        values=[]
        for iid in ids:
            pair=[]
            for r in by_image[iid]:
                v={a:math.fsum(t['iou'] for t in r['trials'] if t['arm']==a)/sum(t['arm']==a for t in r['trials']) for a in config['pressure_arms']}
                v.update(N=r['N'],C=r['C'],G=r['G'])
                if key=='D_S-D_R':x=v['R']-v['S']
                elif key=='D_S-D_M':x=v['M']-v['S']
                elif key.startswith('D_'):x=v['C']-v[key[2:]]
                else:
                    a,b=key.rsplit('-',1);x=v[a]-v[b]
                pair.append(x)
            values.append(math.fsum(pair)/len(pair))
        samples=sorted(math.fsum(values[int(j)] for j in sample)/len(ids) for sample in draws)
        def percentile(q):
            k=q*(len(samples)-1);lo=math.floor(k);hi=math.ceil(k)
            return samples[lo]+(k-lo)*(samples[hi]-samples[lo])
        actual=[math.fsum(values)/len(values),percentile(.025),percentile(.975)]
        expected=[want['mean'],*want['ci95']]
        require(max(abs(a-b) for a,b in zip(actual,expected))<=1e-12,'independent statistic differs: '+key)
        estimates[key]=actual
    positive=estimates['D_S'][1]>0 and all(estimates[k][0]>=config['minimum_effect'] and estimates[k][1]>0 for k in ('D_S-D_R','D_S-D_M'))
    negative=estimates['D_S'][2]<=0 or any(estimates[k][2]<config['minimum_effect'] for k in ('D_S-D_R','D_S-D_M'))
    decision='TRANSFERABLE_DIRECTIONAL_RISK' if positive else 'NO_MATERIAL_TRANSFER_AT_SCOPE' if negative else 'INCONCLUSIVE_NOT_READY'
    if summary['scope']=='smoke':decision='ENGINEERING_SMOKE_ONLY'
    require(summary['decision']==decision,'independent decision')
    for arm,want in summary['teacher_success_damage'].items():
        cases=[t for r in rows if r['C']>=.75 for t in r['trials'] if t['arm']==arm]
        require(want['teacher_success_pairs']==sum(r['C']>=.75 for r in rows) and want['trials']==len(cases) and want['damaged_trials']==sum(t['iou']<.75 for t in cases),'crossing denominator')
    return dict(status='passed',comparisons=len(estimates),resamples=len(draws),decision=decision,tolerance=1e-12)


def verify_fit(path,local):
    library=np.load(path/'FIT_ERRORS.npz');identities=read_json(path/'FIT_IDENTITIES.json');model=completed_fit(path)
    errors=library['c0']-library['teacher'];require(np.array_equal(errors,library['errors']),'fit errors')
    counts=Counter(r['image_id'] for r in identities)
    weights=np.array([1/len(counts)/counts[r['image_id']] for r in identities])
    require(np.array_equal(weights,library['weights']),'image weights')
    mu=np.array([math.fsum(float(e[k]*w) for e,w in zip(errors,weights))/math.fsum(weights) for k in range(32)])
    require(np.max(np.abs(mu-library['mean']))<1e-12,'fit mean')
    centered=errors-mu
    covariance=np.einsum('ni,nj,n->ij',centered,centered,weights)/math.fsum(weights)
    require(np.max(np.abs(covariance-library['covariance']))<1e-10,'fit covariance')
    ordered=sorted(range(len(errors)),key=lambda i:library['native_input_rms'][i]);total=0.
    for index in ordered:
        total+=weights[index]
        if total>=math.fsum(weights)/2:
            median=float(library['native_input_rms'][index]);break
    require(median==model['amplitude'],'weighted median')
    split=read_json(STUDY/'SPLIT.json')
    require(set(counts).issubset(split['fit_image_ids']) and not set(counts)&set(split['heldout_image_ids']),'fit leakage')
    if not local:
        inspect_inputs()
        from adapter_data import FrozenCache
        from math_ops import functional_rms
        source=FrozenCache(CACHE/'trace',sorted(counts));teacher={(r['image_id'],r['rank']):r for r in jsonlines(STUDY/'FIT_TEACHERS.jsonl')}
        for iid in sorted(counts):
            cache,unused=source.load(iid)
            for index,r in enumerate(identities):
                if r['image_id']!=iid:continue
                c0=cache['c0'][r['rank']].numpy().astype(np.float64);c=np.array(teacher[(iid,r['rank'])]['coefficients_fp32'],dtype=np.float32).astype(np.float64)
                require(np.array_equal(c0,library['c0'][index]) and np.array_equal(c,library['teacher'][index]),'fit tensor provenance')
                value=float(functional_rms(cache,r['rank'],(c0-c)[None])[0])
                require(abs(value-library['native_input_rms'][index])<1e-12,'fit functional amplitude provenance')
        source.verify()
    return dict(status='passed',kind='fit_error_statistics',fit_pairs=len(errors),fit_images=len(counts),heldout_rows_used=0,amplitude=median,remote_tensor_replay=not local)


def verify_eval(path,local):
    config=read_json(STUDY/'CONFIG.json');rows=jsonlines(path/'PAIRS.jsonl');contract=read_json(path/'EVAL_CONTRACT.json')
    fit_path=STUDY/'runs'/contract['fit_run'];model=completed_fit(fit_path)
    require(contract['fit_model_sha256']==sha256(fit_path/'FIT_MODEL.json'),'fit freeze')
    library=np.load(fit_path/'FIT_ERRORS.npz');coefficients=np.load(path/'PRESSURE_COEFFICIENTS.npz')['coefficients']
    summary=read_json(path/'SUMMARY.json');require(len(rows)==summary['pairs'],'pair count')
    if summary['scope']=='full':require(len(rows)==581 and len({r['image_id'] for r in rows})==208,'full denominator')
    teacher={(r['image_id'],r['rank']):r for r in jsonlines(STUDY/'HELDOUT_TEACHERS.jsonl')}
    indices=[];decoded_count=0
    for r in rows:
        t=teacher[(r['image_id'],r['rank'])]
        require(t['output_detection_id']==r['output_detection_id'] and t['annotation_id']==r['annotation_id'] and t['oracle_rle']==r['teacher_rle'],'teacher identity')
        gt=spans(r['GT_rle']);native=spans(r['native_rle'])
        for arm,key in [('N','native_rle'),('C','teacher_rle'),('G','G_rle')]:
            require(abs(independent_values(r[key],gt,native)['iou']-r[arm])<=1e-12,'anchor independent RLE')
        for trial in r['trials']:
            actual=independent_values(trial['rle'],gt,native)
            require(all(abs(actual[k]-trial[k])<=1e-12 for k in actual),'independent RLE measurement')
            indices.append(trial['coefficient_index'])
            if trial['donor_image_id'] is not None:require(trial['donor_image_id'] in set(library['image_ids']),'foreign donor')
        require(Counter(t['arm'] for t in r['trials'])=={'S':16,'R':16,'M':16,'E':8,'MU+':1,'MU-':1},'pressure arm counts')
    require(indices==list(range(len(coefficients))),'coefficient coverage')
    stats=statistical_verify(path,rows,config)
    if not local:
        import torch
        from adapter_data import FrozenCache
        from native_output_bridge import decode_native,encode_rle
        from math_ops import draw_directions,normalize,decode_selected
        inspect_inputs();source=FrozenCache(CACHE/'trace',sorted({r['image_id'] for r in rows}))
        from s1_geometry import iter_prediction_array
        mapping=read_json(OLD/'ROW_MAPPING.json')['original_global_indices']
        wanted={r['global_index'] for r in rows}
        old_g={mapping[j]:r for j,r in enumerate(iter_prediction_array(OLD/'G_predictions.json')) if mapping[j] in wanted}
        require(set(old_g)==wanted,'independent G coverage')
        from pycocotools.coco import COCO
        coco=COCO(str(GT));groups=defaultdict(list)
        for r in rows:groups[r['image_id']].append(r)
        with torch.inference_mode():
            for pos,iid in enumerate(sorted(groups),1):
                cache,original=source.load(iid)
                for r in groups[iid]:
                    rank=r['rank'];t=teacher[(iid,rank)];expected_gt=coco.annToRLE(coco.anns[r['annotation_id']])
                    require(expected_gt['counts'].decode('ascii')==r['GT_rle']['counts'],'GT provenance')
                    require(original[rank]['segmentation']==r['native_rle'],'native provenance')
                    require(old_g[r['global_index']]['segmentation']==r['G_rle'],'G provenance')
                    require({k:v for k,v in old_g[r['global_index']].items() if k!='segmentation'}=={k:v for k,v in original[rank].items() if k!='segmentation'},'independent G nonmask identity')
                    vectors,labels=draw_directions(library['errors'],library['mean'],library['image_ids'],iid,rank,config)
                    c,actual,before=normalize(cache,rank,vectors,np.array(t['coefficients_fp32'],dtype=np.float32),model['amplitude'],config)
                    selected=np.array([q['coefficient_index'] for q in r['trials']])
                    require(np.array_equal(c,coefficients[selected]),'pressure tensor regeneration')
                    decoded=decode_selected(cache,rank,c)
                    for j,(label,pred) in enumerate(zip(labels,decoded)):
                        trial=r['trials'][j]
                        require(all(label[k]==trial[k] for k in label),'donor/permutation regeneration')
                        require(pred==trial['rle'],'independent native decode')
                        require(abs(actual[j]-trial['functional_rms'])<1e-12 and abs(before[j]-trial['raw_direction_rms'])<1e-12,'functional norm regeneration')
                    decoded_count+=len(decoded)
                if pos%20==0 or pos==len(groups):print('VERIFY',pos,len(groups),flush=True)
        source.verify();inspect_inputs()
    return dict(status='passed',kind='heldout_pressure_audit',pairs=len(rows),rle_checks=len(indices)+3*len(rows),
        pressure_decoder_replays=decoded_count,fit_only_structure=True,statistics=stats,new_training_steps=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--local',action='store_true');args=parser.parse_args()
    start=time.monotonic();path=args.run.resolve();science_check()
    require(read_json(path/'run.json')['execution_status']=='completed','Run incomplete')
    result=verify_fit(path,args.local) if (path/'FIT_MODEL.json').exists() else verify_eval(path,args.local)
    require(time.monotonic()-start<=read_json(STUDY/'CONFIG.json')['verification_budget_seconds'],'verification budget')
    result.update(verified_at=now(),interpreter=sys.executable,elapsed_seconds=time.monotonic()-start,verifier_sha256=sha256(__file__))
    write_new(path/('INDEPENDENT_LOCAL.json' if args.local else 'INDEPENDENT_LAPTOP.json'),result)
    print(json.dumps(result,ensure_ascii=False))
