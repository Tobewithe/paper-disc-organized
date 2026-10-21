"""Independent replay, RLE endpoints, census and arithmetic for future CLD evaluation."""
from collections import defaultdict
import math
import numpy as np
import torch
from cld_common import *
from adapter_model import CoefficientAdapter
from adapter_data import FrozenCache
from native_output_bridge import decode_native, encode_rle
from pycocotools import mask as masks
from pycocotools.coco import COCO
from s1_geometry import predicted_roi


def iou(a,b):
    intersection = int(masks.area(masks.merge([a,b], intersect=True)))
    union = int(masks.area(masks.merge([a,b], intersect=False)))
    return intersection / union if union else 0.


def statistical_check(path,direction,pairs,tasks):
    summary = read_json(path/'SUMMARY.json')
    groups = dict(direction=[r for r in direction if r['split']=='heldout'],fit_direction=[r for r in direction if r['split']=='fit'],
                  target=[r for r in pairs if r['population']=='T'],overall=pairs)
    require([len(groups[k]) for k in groups]==[581,2510,490,2246], 'independent denominators differ')
    saved = np.load(path/'BOOTSTRAP_INDICES.npz',allow_pickle=False)
    ids_saved = read_json(path/'STATISTICAL_IMAGE_ORDER.json')
    recomputed = {}
    for label, rows in groups.items():
        by_image = defaultdict(list)
        for row in rows:
            by_image[row['image_id']].append(row)
        ids = sorted(by_image)
        require(ids==ids_saved[label], 'bootstrap image ordering differs')
        draws = np.random.default_rng(1729).integers(0,len(ids),size=(2000,len(ids)))
        require(np.array_equal(draws,saved[label]), 'resampling matrix differs')
        comparisons = [('R','P'),('R','J'),('R','N'),('R','C')] if 'direction' in label else [('M','N'),('M','G'),('M','S')]
        recomputed[label] = {}
        for left,right in comparisons:
            values = [math.fsum(r[left]-r[right] for r in by_image[i])/len(by_image[i]) for i in ids]
            estimates = sorted(math.fsum(values[int(j)] for j in row)/len(ids) for row in draws)
            def percentile(q):
                at=q*(len(estimates)-1)
                lo=int(math.floor(at)); hi=int(math.ceil(at))
                return estimates[lo]+(at-lo)*(estimates[hi]-estimates[lo])
            point=math.fsum(values)/len(values)
            bounds=[percentile(.025),percentile(.975)]
            key=left+'-'+right
            q=summary[label][key]
            require(math.isclose(point,q['mean'],abs_tol=1e-12,rel_tol=0) and all(math.isclose(a,b,abs_tol=1e-12,rel_tol=0) for a,b in zip(bounds,q['ci95'])), 'independent macro inference differs')
            recomputed[label][key]=(point,bounds)
    n_success=sum(r['N'] for r in tasks)
    require(n_success==1808, 'success census denominator differs')
    for arm in ('M','S','G'):
        q=summary['census'][arm]
        require(q['damaged']==sum(r['N'] and not r[arm] for r in tasks) and q['recovered']==sum(not r['N'] and r[arm] for r in tasks), 'independent census differs')
    d,t,a=recomputed['direction'],recomputed['target'],recomputed['overall']
    kd=all(d[k][0]>=.005 and d[k][1][0]>0 for k in ('R-P','R-J','R-N'))
    damaged=sum(r['N'] and not r['M'] for r in tasks)
    ku=t['M-N'][0]>=.005 and t['M-N'][1][0]>0 and a['M-N'][0]>=-.001 and t['M-G'][1][0]>0 and damaged<=18
    neg_d=any(d[k][1][1]<.005 for k in ('R-P','R-J','R-N'))
    neg_u=t['M-N'][1][1]<.005 or t['M-G'][1][1]<=0 or damaged>18 or a['M-N'][0]<-.001
    expected='POSITIVE_INTERNAL_SCREEN' if kd and ku else 'NEGATIVE_AT_BUDGET' if neg_d or neg_u else 'INCONCLUSIVE_AT_BUDGET'
    require((kd,ku,neg_d,neg_u,expected)==tuple(summary[k] for k in ('mechanism_pass','utility_pass','mechanism_negative','utility_negative','decision')), 'independent Gate decision differs')
    return dict(mechanism_pass=kd,utility_pass=ku,decision=expected,success_GT=1808,paired_resamples=2000)


def verify(path,data,budget):
    contract=read_json(path/'EVAL_CONTRACT.json')
    models={}
    for arm,rid in contract['checkpoint_runs'].items():
        run=STUDY/'runs'/rid
        check_completed(run)
        require(sha256(run/'last.pt')==contract['checkpoints'][arm], 'checkpoint bytes differ')
        model=CoefficientAdapter('B').cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(run/'last.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        models[arm]=model
    direction=read_jsonl(path/'DIRECTION_ROWS.jsonl')
    pairs=read_jsonl(path/'FIXED_PAIRS.jsonl')
    tasks=read_jsonl(path/'TASKS.jsonl')
    by_teacher=defaultdict(list)
    for row in direction: by_teacher[row['image_id']].append(row)
    by_pair=defaultdict(list)
    for row in pairs: by_pair[row['image_id']].append(row)
    held=set(data['split']['heldout_image_ids'])
    ids=sorted(held | set(by_teacher))
    teacher={(r['image_id'],r['annotation_id']):r for values in teacher_rows(ids).values() for r in values}
    source=FrozenCache(CACHE/'trace',ids)
    mapping=read_json(path/'ROW_MAPPING.json')
    outputs={k:dict(zip(mapping['global_indices'],read_json(path/f'{k}_predictions.json'))) for k in ('N','M','S')}
    coco=COCO(str(GT))
    forwards=0
    with torch.inference_mode():
        for pos,iid in enumerate(ids,1):
            budget()
            cache,original=source.load(iid)
            saved=torch.load(path/'directions'/f'{iid}.pt',map_location='cpu',weights_only=True)
            require(saved['global_indices']==cache['prediction_indices'] and saved['donor_ranks']==data['maps'][iid], 'saved direction identity differs')
            raw={}
            for arm,model in models.items():
                raw[arm]=model(cache,'independent:evaluation')
                forwards+=1
                torch.testing.assert_close(raw[arm].cpu(),saved['raw'][arm],atol=0,rtol=0)
            index=torch.tensor(data['maps'][iid],device='cuda')
            ds=dict(R=raw['REAL'],P=raw['PERM'][index],J=raw['REAL'][index])
            p=cache['proto'].cuda().flatten(1)
            c0=cache['c0'].cuda()
            normal={}
            for arm,d in ds.items():
                response=(d@p).reshape(len(d),*cache['proto'].shape[-2:])
                probe=c0.clone(); coeff=c0.clone()
                for rank in range(len(d)):
                    roi=predicted_roi(cache,rank).cuda()
                    rms=float(response[rank][roi].square().mean().sqrt()) if roi.any() else 0.
                    if roi.any(): coeff[rank]+=contract['amplitude']*d[rank]/max(1.,rms)
                    for record in by_teacher[iid]:
                        if record['rank']==rank and rms>1e-6:
                            probe[rank]+=record['scale']*d[rank]/rms
                # Only direction arithmetic comparisons have a predeclared FP32 tolerance;
                # endpoint RLEs are verified from the producer expression below, shared decoder.
                from cld_math import context, deploy_coefficients, probe_coefficients
                ctx=context(cache,d.device)
                producer_probe=probe_coefficients(d,ctx,{r['rank']:r['scale'] for r in by_teacher[iid]})
                torch.testing.assert_close(probe,producer_probe,atol=3e-6,rtol=2e-5)
                decoded=decode_native(cache,producer_probe)
                for record in by_teacher[iid]:
                    t=teacher[(iid,record['annotation_id'])]
                    gt=coco.annToRLE(coco.anns[record['annotation_id']])
                    rle=encode_rle(decoded[record['rank']])
                    require(rle==record['probe_RLE'][arm], 'direction RLE replay differs')
                    for name,mask in ((arm,rle),('N',original[record['rank']]['segmentation']),('C',t['oracle_rle'])):
                        require(abs(iou(mask,gt)-record[name])<1e-12, 'independent direction IoU differs')
                del decoded
                if iid in held and arm in ('R','P'):
                    producer_coeff=deploy_coefficients(d,ctx,contract['amplitude'])
                    torch.testing.assert_close(coeff,producer_coeff,atol=3e-6,rtol=2e-5)
                    decoded=decode_native(cache,producer_coeff)
                    key='M' if arm=='R' else 'S'
                    normal[key]=[]
                    for rank,row in enumerate(original):
                        mixed={**row,'segmentation':encode_rle(decoded[rank]) if int(masks.area(row['segmentation']))>=9216 else row['segmentation']}
                        require(outputs[key][cache['prediction_indices'][rank]]==mixed, 'mixed prediction replay differs')
                        normal[key].append(mixed)
                    del decoded
                del response,probe,coeff,ctx,producer_probe
            if iid in held:
                rank_of={g:j for j,g in enumerate(cache['prediction_indices'])}
                for rank,row in enumerate(original): require(outputs['N'][cache['prediction_indices'][rank]]==row, 'native output differs')
                for record in by_pair[iid]:
                    rank=rank_of[int(record['output_detection_id'].rsplit(':',1)[1])]
                    gt=coco.annToRLE(coco.anns[record['annotation_id']])
                    for key in ('M','S'):
                        require(abs(iou(normal[key][rank]['segmentation'],gt)-record[key])<1e-12, 'independent fixed IoU differs')
            if pos%100==0: print('VERIFY_EVALUATION',pos,'/',len(ids),flush=True)
            del cache,original,saved,raw,ds,p,c0,normal
    # Census is independently regenerated with the same pinned official COCO matching engine.
    from evaluate_cld import ledger
    historic={(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT/'FIT_TASKS.jsonl') if r['image_id'] in held}
    task_by_id={(r['image_id'],r['annotation_id']):r for r in tasks}
    base_lookup=None
    for arm in ('N','M','S'):
        budget()
        rows=ledger(path,[outputs[arm][i] for i in mapping['global_indices']],sorted(held),mapping['global_indices'],mapping['source_prediction_sha256'],'independent_coco_matching.log')
        if arm=='N': base_lookup={(r['image_id'],r['annotation_id']):r for r in rows}
        for r in rows:
            key=(r['image_id'],r['annotation_id'])
            require(r['bbox75_match_output_id']==base_lookup[key]['bbox75_match_output_id'], 'independent bbox matching differs')
            if not r['is_ignored']:
                require((r['task_mask75']=='success')==task_by_id[key][arm], 'independent task status differs')
                require(task_by_id[key]['G']==historic[key]['G_success'], 'historical task reference differs')
    result=statistical_check(path,direction,pairs,tasks)
    source.verify()
    return dict(kind='full_evaluation',adapter_forward_calls=forwards,additional_optimizer_steps=0,
                independent_arithmetic_and_RLE_endpoints=True,shared_official_decoder_and_COCO_engine=True,**result)
