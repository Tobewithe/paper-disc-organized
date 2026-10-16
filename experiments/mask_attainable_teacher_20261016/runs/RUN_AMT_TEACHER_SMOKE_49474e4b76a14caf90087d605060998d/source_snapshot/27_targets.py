"""Only training targets change; model forward and reduction remain original."""
import hashlib
from common import read_json, sha256, require

def mask_hash(mask):
    return hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def accept(baseline_iou, oracle_iou, solver_status, foreground):
    return solver_status=='converged' and oracle_iou-baseline_iou>=.01 and foreground>0

class TeacherTargets:
    def __init__(self,teacher,original_projected):
        self.root=teacher
        self.manifest=read_json(teacher/'TEACHER_MANIFEST.json')
        self.records={int(i):r for i,r in self.manifest['images'].items()}
        self.original=original_projected
        self.pending=None
        self.visits=[]
    def __call__(self,coco,cache,join):
        import torch
        image_id=cache['image_id']
        require(image_id in self.records,'image absent from teacher')
        record=self.records[image_id]
        path=self.root/record['path']
        require(sha256(path)==record['sha256'],'teacher package drift')
        package=torch.load(path,map_location='cpu',weights_only=True)
        require(package['image_id']==image_id and package['prediction_indices']==cache['prediction_indices'],'teacher output identity drift')
        require(package['join']==join,'teacher/training join differs')
        targets=self.original(coco,cache,join)
        require(set(targets)==set(join),'GT target population changed')
        for rank,q in package['targets'].items():
            require(rank in targets and q.dtype==torch.uint8 and tuple(q.shape)==tuple(cache['proto'].shape[-2:]),'invalid teacher target')
            require(mask_hash(q)==package['target_hashes'][rank] and bool(((q==0)|(q==1)).all()),'teacher target bytes changed')
            targets[rank]=q.float().detach()
        self.pending=dict(image_id=image_id,teacher_replaced=len(package['targets']),teacher_package_sha256=record['sha256'])
        self.visits.append(self.pending.copy())
        return targets
    def loss(self,logits,baseline,roi,targets):
        from soft_iou_loss import soft_iou_grouped_loss
        require(self.pending is not None,'loss did not consume a registered target package')
        value,stats=soft_iou_grouped_loss(logits,baseline,roi,targets)
        require(self.pending['teacher_replaced']<=stats['matched_supervised_instances'],'teacher target disappeared from denominator')
        stats.update(self.pending)
        self.pending=None
        return value,stats
