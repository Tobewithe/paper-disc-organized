"""GT-aware connectivity audit. No morphology, inference or optimization."""
import numpy as np
from scipy import ndimage as ndi


class Context:
    def __init__(self, gt, neighbors=(), connectivity=8):
        self.gt=np.asarray(gt,bool);self.area=int(self.gt.sum())
        self.structure=ndi.generate_binary_structure(2,2 if connectivity==8 else 1)
        self.labels,self.components=ndi.label(self.gt,self.structure)
        self.parts=[self.labels==i for i in range(1,self.components+1)]
        self.neighbors=[(int(i),np.asarray(mask,bool)&~self.gt) for i,mask in neighbors]
        other=np.zeros_like(self.gt)
        for _,mask in neighbors:other|=mask
        self.core=self.gt&~other
        self.foreign=np.zeros_like(self.gt)
        for _,mask in self.neighbors:self.foreign|=mask

    def measure(self,prediction):
        p=np.asarray(prediction,bool)
        if p.shape!=self.gt.shape:raise ValueError('prediction/GT shape mismatch')
        labels,count=ndi.label(p,self.structure)
        split=missed=split_pixels=0
        gt_touch={}
        for gi,part in enumerate(self.parts,1):
            counts=np.bincount(labels[part],minlength=count+1);counts[0]=0
            ids=np.flatnonzero(counts)
            if not len(ids):missed+=1
            if len(ids)>1:split+=1
            split_pixels+=int(counts.sum()-counts.max(initial=0))
            for pi in ids:gt_touch.setdefault(int(pi),[]).append(gi)
        core_ids=np.unique(labels[self.core]);core_ids=core_ids[core_ids>0]
        connected=np.isin(labels,core_ids) if len(core_ids) else np.zeros_like(p)
        neighbor_ids=[i for i,mask in self.neighbors if np.any(connected&mask)]
        foreign_pixels=int((connected&self.foreign).sum())
        intersection=int((p&self.gt).sum());union=int((p|self.gt).sum())
        unmatched=count-len(gt_touch)
        return {'iou':intersection/union if union else 0.,'gt_components':self.components,
            'prediction_components':int(count),'gt_pixels':self.area,'prediction_pixels':int(p.sum()),
            'coverage':intersection/self.area if self.area else 0.,'split_gt_components':split,
            'missed_gt_components':missed,'split_pixels':split_pixels,
            'split_mass':split_pixels/self.area if self.area else 0.,
            'bridge_eligible':bool(self.core.any()),'bridge_neighbor_ids':neighbor_ids,
            'bridge_pixels':foreign_pixels,'bridge_mass':foreign_pixels/self.area if self.area else 0.,
            'within_instance_false_joins':sum(len(v)>1 for v in gt_touch.values()),
            'unmatched_prediction_islands':unmatched,
            'split_event':bool(split),'bridge_event':bool(neighbor_ids),
            'material_split':bool(split_pixels>=.01*self.area) if self.area else False,
            'material_bridge':bool(foreign_pixels>=.01*self.area) if self.area else False}


def signature(m):
    return (m['split_gt_components'],tuple(m['bridge_neighbor_ids']),m['within_instance_false_joins'],
            m['unmatched_prediction_islands'],m['missed_gt_components'])


def tests():
    cases=[];gt=np.zeros((7,11),bool);gt[2:5,1:5]=1
    p=gt.copy();p[:,3]=0
    assert Context(gt).measure(p)['split_event'];cases.append('true_split')
    neighbor=np.zeros_like(gt);neighbor[2:5,7:10]=1
    p=gt|neighbor;p[3,4:8]=1
    m=Context(gt,[(22,neighbor)]).measure(p)
    assert m['bridge_neighbor_ids']==[22] and m['bridge_pixels']==9;cases.append('foreign_bridge')
    m=Context(gt,[(22,gt.copy())]).measure(gt)
    assert not m['bridge_eligible'] and not m['bridge_event'];cases.append('GT_overlap_not_bridge')
    multi=gt|neighbor;m=Context(multi).measure(multi)
    assert m['gt_components']==2 and not m['split_event'] and not m['within_instance_false_joins'];cases.append('GT_multipart_legitimate')
    m=Context(multi).measure(gt)
    assert m['missed_gt_components']==1 and not m['split_event'];cases.append('missing_part_not_split')
    assert Context(multi).measure(p)['within_instance_false_joins']==1;cases.append('false_within_instance_join')
    diagonal=np.eye(3,dtype=bool)
    assert Context(diagonal,connectivity=8).components==1 and Context(diagonal,connectivity=4).components==3;cases.append('4_8_connectivity')
    m=Context(gt).measure(np.zeros_like(gt))
    assert m['missed_gt_components']==1 and not m['split_event'];cases.append('empty_not_split')
    return {'status':'passed','cases':cases,'training_steps':0}
