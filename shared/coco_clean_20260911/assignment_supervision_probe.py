"""S017 frozen checkpoint: official assigner/loss on exact COCO GT, no update.

Inference BN statistics and original unaugmented images intentionally held fixed.
This is a current-checkpoint supervision-path diagnostic, not training-history
reconstruction or a claim about the original pretrained annotation recipe.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha
from crossimage_response_experiment import gt_input_regions


class TracedLoss(v8SegmentationLoss):
    def get_assigned_targets_and_loss(self,preds,batch):
        result=super().get_assigned_targets_and_loss(preds,batch)
        self.assignment=result[0]
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--images',type=int,default=300);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'diagnostics/candidate_lineage300_20260912';ids=json.loads((source/'protocol.json').read_text())['images'][:a.images]
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    prior={int(r['annotation_id']):r for r in read(source/'gt.csv')};s015={int(r['annotation_id']):r for r in read(ROOT/'diagnostics/no_candidate317_20260912/targets.csv')}
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt')).model.cuda().eval()
    model.fuse(verbose=False);model.requires_grad_(False)
    model.args=get_cfg(overrides={'task':'segment','overlap_mask':False,'mask_ratio':4})
    criterion=TracedLoss(model)
    need(not model.end2end,'Expected cached one-to-many inference branch')
    write_json(a.out/'protocol.json',dict(images=ids,network_training=False,optimizer_steps=0,
        operation='Exact frozen cached raw boxes/classes/coefficients/prototypes; official TaskAlignedAssigner.forward and v8SegmentationLoss.calculate_segmentation_loss; differentiate actual segmentation loss w.r.t. coefficient outputs only. Model supplies stride/hyperparameters, no fresh forward or BN updates.',
        GT='Original COCO annotation, multipart union annToMask, normalized bbox transformed with actual LetterBox ratio. All ordinary GT included; GT masks independent channels, nearest-exact to640 then nearest to160. No polygon conversion or GT-derived predictions.',
        mask_recipe='Explicit nonoverlap diagnostic labels, official criterion overlap_mask=False mask_ratio4. Padding/background outside ordinary masks follows normal loss; crowd pixels are not masked out of official BCE. Not reconstruction of original pretraining targets or augmentation.',
        restriction='Unaugmented explored COCO val. Current-checkpoint assignment is not historical training assignment; no change of BN statistics/weights. Assigned source includes points before score/NMS filtering.',
        helper_sha256=sha(Path(__file__).with_name('crossimage_response_experiment.py')),script_sha256=sha(__file__),
        annotation_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),
        loss_source_sha256=sha(__import__('ultralytics.utils.loss',fromlist=['']).__file__),
        tal_source_sha256=sha(__import__('ultralytics.utils.tal',fromlist=['']).__file__)))
    rows=[];witness=[];start=time.monotonic()
    for num,iid in enumerate(ids,1):
        cachepath=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz';rawpath=ROOT/'diagnostics/structure_main300_20260911/raw'/f'{iid}.npz'
        expected=json.loads((source/'images'/f'{iid}.json').read_text())['input_hashes'];need(expected=={'raw':sha(rawpath),'cache':sha(cachepath)},'Changed cached source')
        with np.load(cachepath) as z:item={k:z[k] for k in z.files}
        with np.load(rawpath) as z:raw={k:z[k] for k in z.files}
        proto=torch.tensor(item['proto'][None],device='cuda')
        coeff=torch.tensor(raw['coefficients'][None],device='cuda',requires_grad=True)
        scores=torch.tensor(raw['class_scores'][None],device='cuda');physical=torch.tensor(raw['boxes_input'][None],device='cuda')
        # Exact same anchor/source layout as the frozen 640-input cached head.
        feats=[torch.empty((1,1,640//int(s),640//int(s)),device='cuda') for s in criterion.stride]
        anchors,strides=make_anchors(feats,criterion.stride,.5)
        need(anchors.shape[0]==coeff.shape[1],'Anchor/source layout mismatch')
        need(np.array_equal(raw['coefficients'][raw['nonempty_indices']],item['coeff']),'Final coefficient source mismatch')
        anns=[gt.anns[int(aid)] for aid in raw['annotation_ids']];cats=sorted(gt.cats);n=len(anns)
        h,w=map(int,item['shape']);gain=min(640/h,640/w);rh,rw=round(h*gain),round(w*gain);top,left=round((640-rh)/2-.1),round((640-rw)/2-.1)
        boxes=np.array([q['bbox'] for q in anns],dtype=np.float32).reshape(-1,4);boxes[:,:2]+=boxes[:,2:]/2;boxes*=gain;boxes[:,0]+=left;boxes[:,1]+=top;boxes/=640
        raster,_,valid=gt_input_regions(gt,iid,item)
        full=torch.stack([raster[q['id']] for q in anns]).float() if n else torch.zeros((0,640,640),device='cuda')
        masks=F.interpolate(full[:,None],(160,160),mode='nearest')[:,0]
        batch={'batch_idx':torch.zeros(n,device='cuda'),'cls':torch.tensor([cats.index(q['category_id']) for q in anns],device='cuda').float()[:,None],
            'bboxes':torch.tensor(boxes,device='cuda'),'masks':masks}
        packed=torch.cat([batch['batch_idx'][:,None],batch['cls'],batch['bboxes']],1)
        targets=criterion.preprocess(packed,1,scale_tensor=torch.tensor([640,640,640,640],device='cuda'))
        labels,gtboxes=targets.split((1,4),2);maskgt=gtboxes.sum(2,keepdim=True).gt(0.)
        _,tboxes,tscores,fg,tidx=criterion.assigner(scores,physical,anchors*strides,labels,gtboxes,maskgt)
        if fg.any():
            seg=criterion.calculate_segmentation_loss(fg,masks,tidx,tboxes,batch['batch_idx'][:,None],proto,coeff,torch.tensor([640,640],device='cuda'))*criterion.hyp.box
        else:seg=coeff.sum()*0.
        gradient=torch.autograd.grad(seg,coeff)[0][0].detach()
        fg=fg[0];tidx=tidx[0];tboxes=tboxes[0]
        need(torch.isfinite(gradient).all().item(),'Nonfinite mask gradient')
        need(not gradient[~fg].any().item(),'Nonpositive source receives coefficient mask gradient')
        gradient_norm=gradient.norm(dim=1).cpu().numpy();fgnp=fg.cpu().numpy();tidxnp=tidx.cpu().numpy()
        mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
        for g,ann in enumerate(anns):
            aid=ann['id'];assigned=np.flatnonzero(fgnp&(tidxnp==g));j=mapping.get(aid);src=int(raw['nonempty_indices'][j]) if j is not None else -1
            if src<0:srcstate='unmatched_final'
            elif not fgnp[src]:srcstate='not_positive'
            elif tidxnp[src]==g:srcstate='own_gt_positive'
            else:srcstate='different_gt_positive'
            gtbox=torch.tensor([ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]],device='cuda')*gain
            gtbox+=torch.tensor([left,top,left,top],device='cuda');gtcrop=ops.crop_mask(torch.ones((1,160,160),device='cuda'),gtbox[None]/4)[0].bool()
            gtmask=masks[g].bool();maskcells=int(gtmask.sum());supervisedcells=int((gtmask&gtcrop).sum())
            row=dict(image_id=iid,annotation_id=aid,category_id=ann['category_id'],area=ann['area'],ici=float(prior[aid]['ici']),
                official_mask75=prior[aid]['official_mask75'],no_good_postconf_mask75=prior[aid]['score_mask75']=='False',
                positive_count=len(assigned),positive_s8=int((assigned<6400).sum()),positive_s16=int(((assigned>=6400)&(assigned<8000)).sum()),positive_s32=int((assigned>=8000).sum()),
                positive_nonzero_gradient=int((gradient_norm[assigned]>0).sum()),final_source=src,final_source_assignment=srcstate,
                final_source_mask_gradient=float(gradient_norm[src]) if src>=0 else None,
                gt_mask_cells160=maskcells,gt_mask_cells_in_gt_crop160=supervisedcells,gt_crop_cells160=int(gtcrop.sum()),
                seg_weight_share=len(assigned)/max(int(fg.sum()),1),
                positive_gradient_mean=float(gradient_norm[assigned].mean()) if len(assigned) else None)
            if aid in s015:row['s015_fit_status']=s015[aid]['fit_status']
            rows.append(row)
        np.savez_compressed(a.out/'images'/f'{iid}.npz',annotation_ids=raw['annotation_ids'],foreground=fgnp,target_gt_index=tidxnp,
            coefficient_gradient=gradient.cpu().numpy(),gt_masks160=masks.cpu().numpy().astype(np.uint8),gt_boxes_normalized=boxes,
            original_final_sources=raw['nonempty_indices'])
        witness.append(dict(image_id=iid,gt=n,positive_sources=int(fg.sum()),raw_cached_exact=True,
            seg_loss=float(seg.detach()),positive_gradient_nonzero=int((gradient_norm[fgnp]>0).sum()),nonpositive_gradient_nonzero=int((gradient_norm[~fgnp]>0).sum())))
        if num%10==0 or num==len(ids):
            progress=dict(images=num,total=len(ids),seconds=time.monotonic()-start);write_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    save_csv(a.out/'targets.csv',rows);save_csv(a.out/'witness.csv',witness)
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),gt=len(rows),network_training=False,optimizer_steps=0,
        seconds=time.monotonic()-start,hashes={str(q.relative_to(a.out)):sha(q) for q in a.out.rglob('*') if q.is_file()}))

if __name__=='__main__':main()
