"""Fixed error sampling and the pinned pre-threshold functional metric."""
import hashlib
import numpy as np


def weighted_median(values, weights):
    order=np.argsort(values,kind='stable')
    return float(values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum()/2, side='left')]])


def draw_directions(errors, mu, image_ids, image_id, rank, config):
    seed=int.from_bytes(hashlib.sha256(f"CET-v1|{config['seed']}|{image_id}|{rank}".encode('ascii')).digest()[:8],'little')
    rng=np.random.Generator(np.random.PCG64(seed))
    groups={int(i):np.flatnonzero(image_ids==i) for i in np.unique(image_ids)}
    ids=sorted(groups)
    require=image_id not in groups
    if not require: raise ValueError('recipient is in fit error library')
    vectors=[]; labels=[]
    for draw in range(config['donors_per_pair']):
        iid=ids[int(rng.integers(len(ids)))]; index=int(rng.choice(groups[iid]))
        e=errors[index]; centered=e-mu
        gaussian=rng.normal(size=32)
        while True:
            permutation=rng.permutation(32)
            if np.all(permutation != np.arange(32)): break
        for arm, vector in [('S',centered),('R',gaussian),('M',centered[permutation])]:
            for sign in (1,-1):
                vectors.append(sign*vector)
                labels.append(dict(arm=arm,draw=draw,sign=sign,donor_index=index,donor_image_id=iid,permutation=permutation.tolist() if arm=='M' else None))
        vectors.append(e)
        labels.append(dict(arm='E',draw=draw,sign=1,donor_index=index,donor_image_id=iid,permutation=None))
    for arm,sign in [('MU+',1),('MU-',-1)]:
        vectors.append(sign*mu);labels.append(dict(arm=arm,draw=0,sign=sign,donor_index=None,donor_image_id=None,permutation=None))
    return np.stack(vectors),labels


def functional_rms(cache, rank, directions):
    import torch
    from ultralytics.utils import ops
    p=cache['proto'].to('cuda').float()
    d=torch.as_tensor(directions,dtype=torch.float32,device='cuda')
    hw=cache['native_decode_hw']
    box=cache['boxes_input'][rank].to('cuda')
    yy=torch.arange(hw[0],device='cuda')[:,None]
    xx=torch.arange(hw[1],device='cuda')[None,:]
    support=(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3])
    if not support.any(): raise ValueError('empty native input crop')
    result=[]
    for part in d.split(16):
        low=(part@p.flatten(1)).reshape(-1,*p.shape[-2:])
        logits=ops.scale_masks(low[None],hw)[0]
        result.extend(logits[:,support].double().square().mean(1).sqrt().cpu().tolist())
    return np.array(result,dtype=np.float64)


def normalize(cache,rank,vectors,center,amplitude,config):
    rms=functional_rms(cache,rank,vectors)
    if not np.isfinite(rms).all() or np.any(rms<=config['zero_rms_epsilon']): raise ValueError('zero/nonfinite direction')
    coefficients=(np.asarray(center,dtype=np.float64)[None]+vectors*(amplitude/rms)[:,None]).astype(np.float32)
    actual=functional_rms(cache,rank,coefficients.astype(np.float64)-np.asarray(center,dtype=np.float64))
    relative=np.abs(actual/amplitude-1)
    if not np.all(relative<=config['functional_rms_relative_tolerance']): raise ValueError('functional RMS mismatch: '+str(relative.max()))
    return coefficients,actual,rms


def decode_selected(cache,rank,coefficients):
    import torch
    from native_output_bridge import decode_native,encode_rle
    result=[]
    a=np.asarray(coefficients,dtype=np.float32).reshape(-1,32)
    for start in range(0,len(a),16):
        c=torch.from_numpy(a[start:start+16].copy())
        selected=dict(cache,c0=c,boxes_input=cache['boxes_input'][rank:rank+1].repeat(len(c),1))
        decoded=decode_native(selected,c)
        result.extend(encode_rle(mask) for mask in decoded)
    return result


def mask_measure(pred,gt,native):
    from pycocotools import mask as m
    def area(r):return int(m.area(r))
    p=area(pred);g=area(gt);intersection=area(m.merge([pred,gt],intersect=True));union=p+g-intersection
    native_tp=area(m.merge([native,gt],intersect=True))
    common=area(m.merge([native,pred],intersect=True))
    common_tp=area(m.merge([native,pred,gt],intersect=True))
    return dict(iou=intersection/union if union else 0.,intersection=intersection,union=union,
                native_TP_to_FN=native_tp-common_tp,native_FP_to_TN=area(native)-native_tp-(common-common_tp))
