"""GT-consistent maximin-path critical pixels and hardness-matched controls.

No auxiliary prediction can satisfy this term: selected pixels belong to the
actual prethreshold mask logits. Discrete graph selection is detached.
"""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi,sparse
from scipy.sparse.csgraph import minimum_spanning_tree,breadth_first_order

def center(mask):
    if not mask.any():return None
    distance=ndi.distance_transform_edt(mask)
    return int(np.argmax(distance))

def graph_tree(probability,allowed):
    h,w=probability.shape;index=np.arange(h*w).reshape(h,w)
    aa=[];bb=[];weights=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        ya=slice(0,h-dy);yb=slice(dy,h)
        xa=slice(max(0,-dx),min(w,w-dx));xb=slice(max(0,dx),min(w,w+dx))
        use=allowed[ya,xa]&allowed[yb,xb]
        a=index[ya,xa][use];b=index[yb,xb][use]
        aa.append(a);bb.append(b)
        weights.append(2-np.minimum(probability.ravel()[a],probability.ravel()[b]))
    a=np.concatenate(aa);b=np.concatenate(bb);weight=np.concatenate(weights)
    matrix=sparse.csr_matrix((weight,(a,b)),shape=(h*w,h*w))
    return minimum_spanning_tree(matrix).tocsr()

def trace_path(predecessor,start,end):
    path=[int(end)]
    for _ in range(len(predecessor)):
        if path[-1]==start:return path
        parent=int(predecessor[path[-1]])
        if parent<0:return []
        path.append(parent)
    raise RuntimeError('tree path did not terminate')

def make_spec(gt,others,image_id,rank,config):
    gt=np.asarray(gt,bool);labels,n=ndi.label(gt,ndi.generate_binary_structure(2,2))
    seed=int.from_bytes(hashlib.sha256(f'LCF|{image_id}|{rank}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);positive=[];negative=[]
    parts=sorted([labels==i for i in range(1,n+1)],key=lambda m:-int(m.sum()))
    for part in parts:
        pixels=np.flatnonzero(part);root=center(part)
        if len(pixels)<2:continue
        other=pixels[pixels!=root]
        remaining=config['max_positive_pairs']-len(positive)
        if remaining<=0:break
        chosen=rng.choice(other,size=min(remaining,len(other)),replace=False)
        positive.extend((root,int(v)) for v in chosen)
    union=np.zeros_like(gt)
    for other in others:union|=other
    core=gt&~union;root=center(core)
    if root is not None:
        for other in others:
            foreign=other&~gt;target=center(foreign)
            if target is not None:negative.append((root,target))
        # Separate genuine visible GT parts, rather than forcing one component.
        root_part=int(labels.ravel()[root])
        for part in parts:
            target=center(part)
            if target is not None and labels.ravel()[target]!=root_part:negative.append((root,target))
    return {'gt':gt,'positive':positive,'negative':negative[:config['max_negative_pairs']],
            'seed':seed,'GT_components':n,'target_has_exclusive_core':root is not None}

def select_pixels(logits,spec,config,topological):
    z=logits.detach().float().cpu().numpy();prob=1/(1+np.exp(-np.clip(z,-80,80)));g=spec['gt']
    selected=[];truth=[];matched=0;fallback=0
    for pairs,allowed,label in [(spec['positive'],g,1),(spec['negative'],np.ones_like(g),0)]:
        if not pairs:continue
        tree=graph_tree(prob,allowed);previous={}
        for start,end in pairs:
            if start not in previous:previous[start]=breadth_first_order(tree,start,directed=False,return_predecessors=True)[1]
            path=trace_path(previous[start],start,end)
            candidates=[v for v in path if int(g.ravel()[v])==label]
            if not candidates:continue
            # Positive: weakest foreground along the strongest GT-valid path.
            # Negative: weakest GT-background point on a strongest false path.
            # Selecting true foreground for suppression is prohibited.
            point=min(candidates,key=lambda v:(float(z.ravel()[v]),v))
            selected.append(point);truth.append(label)
    if not topological and selected:
        rng=np.random.default_rng(spec['seed']);critical=set(selected);bins=np.asarray(config['difficulty_bins'])
        replacements=[];flat=z.ravel();gt=g.ravel();bin_index=np.digitize(flat,bins)
        noncritical=np.ones(len(flat),bool);noncritical[list(critical)]=False
        by_class={v:np.flatnonzero((gt==v)&noncritical) for v in [0,1]}
        for point,label in zip(selected,truth):
            allowed=by_class[label]
            same=allowed[bin_index[allowed]==bin_index[point]]
            if len(same):
                # Nearest logit value controls supervision hardness; deterministic ties.
                error=np.abs(flat[same]-flat[point]);best=same[error<=error.min()+1e-7]
                replacement=int(rng.choice(best));matched+=1
            elif len(allowed):
                replacement=int(allowed[np.argmin(np.abs(flat[allowed]-flat[point]))]);fallback+=1
            else:
                # A tiny target can have no noncritical same-class point.
                replacement=point;fallback+=1
            replacements.append(replacement)
        selected=replacements
    return selected,truth,{'critical_samples':len(truth),'hardness_matched_noncritical':matched,
                            'hardness_fallback':fallback,'positive_samples':sum(truth),'negative_samples':len(truth)-sum(truth)}

def mask_loss(logits,baseline,roi,targets,specs,arm,config,plans=None,denominators=None):
    valid=roi.flatten(1).sum(1)>0;zero=logits.sum()*0
    preserve=F.binary_cross_entropy_with_logits(logits,baseline.detach().sigmoid(),reduction='none')
    count=roi.sum((1,2)).clamp_min(1)
    denominator=denominators or {'preservation':int(valid.sum()),'GT':len(targets),'connection':len(specs)}
    preservation=((preserve*roi).sum((1,2))/count)[valid].sum()/max(1,denominator['preservation']) if valid.any() else zero
    terms=[];topology=[];account={'critical_samples':0,'hardness_matched_noncritical':0,'hardness_fallback':0,'positive_samples':0,'negative_samples':0}
    for rank,target in targets.items():
        if not bool(valid[rank]):continue
        pred=logits[rank];r=roi[rank];y=target.to(pred)
        probability=pred.sigmoid();intersection=(probability*y*r).sum();union=((probability+y-probability*y)*r).sum()
        soft=1-(intersection+config['epsilon'])/(union+config['epsilon'])
        bce=(F.binary_cross_entropy_with_logits(pred,y,reduction='none')*r).sum()/count[rank]
        terms.append(soft+config['GT_BCE']*bce)
        if rank in specs:
            rectangle,spec=specs[rank];x0,y0,x1,y1=rectangle;local=pred[y0:y1,x0:x1]
            points,labels,counts=plans[rank] if plans is not None else select_pixels(local,spec,config,arm in ('B','D'))
            for k,v in counts.items():account[k]+=v
            if points:
                chosen=local.flatten()[torch.tensor(points,device=pred.device)]
                yy=torch.tensor(labels,device=pred.device,dtype=pred.dtype)
                parts=[F.binary_cross_entropy_with_logits(chosen[yy==c],yy[yy==c]) for c in [0,1] if (yy==c).any()]
                topology.append(torch.stack(parts).mean())
    gt=torch.stack(terms).sum()/max(1,denominator['GT']) if terms else zero
    connect=torch.stack(topology).sum()/max(1,denominator['connection']) if topology else zero
    total=gt+config['preservation_BCE']*preservation+config['connection_pixel_BCE']*connect
    account.update(GT_loss=float(gt.detach()),preservation_loss=float(preservation.detach()),connection_loss=float(connect.detach()),matched_instances=len(terms))
    return total,account
