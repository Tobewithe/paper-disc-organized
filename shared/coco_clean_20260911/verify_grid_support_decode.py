"""Re-decode persisted S019 coefficients; quantify interpolation-order roundoff."""
import contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from crossimage_response_experiment import gt_input_regions
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=ROOT/'diagnostics/grid_support887_20260912';receipt=json.loads((out/'COMPLETE.json').read_text());need(receipt['status']=='COMPLETE','Not complete')
targets=read(out/'targets.csv');protocol=json.loads((out/'protocol.json').read_text());rows=[];start=time.monotonic()
with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
for iid in protocol['images']:
    with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as z:item={k:z[k] for k in z.files}
    with np.load(ROOT/'diagnostics/assignment300_20260912/images'/f'{iid}.npz') as z:
        supervised={k:z[k] for k in z.files};sources={int(src):j for j,src in enumerate(z['original_final_sources'])}
    with np.load(out/'images'/f'{iid}.npz') as z:params={k:z[k] for k in z.files}
    proto=torch.tensor(item['proto'],device='cuda');expanded=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
    rasters,_,_=gt_input_regions(gt,iid,item);gindex={int(aid):g for g,aid in enumerate(supervised['annotation_ids'])}
    for r in targets:
        if int(r['image_id'])!=iid or r['arm']=='original':continue
        aid=int(r['annotation_id']);c=torch.tensor(params[f"{aid}_{r['arm']}"],device='cuda');j=sources[int(r['source_index'])]
        box=torch.tensor(item['boxes'][j:j+1],device='cuda');shape=tuple(map(int,item['shape']))
        exact=ops.process_mask(proto,c[None],box,(640,640),upsample=True)[0]
        logits=F.interpolate((c@proto.flatten(1)).reshape(1,1,160,160),(640,640),mode='bilinear',align_corners=False)[0,0]
        alternate=(c@expanded.flatten(1)).reshape(640,640)
        altmask=ops.crop_mask((alternate>0)[None].to(torch.uint8),box)[0]
        xor=int((exact!=altmask).sum());truth=gt.annToRLE(gt.anns[aid])
        def iou(mask):
            pred=(ops.scale_masks(mask[None,None],shape)[0,0]>.5).cpu().numpy().astype(np.uint8)
            return float(mu.iou([mu.encode(np.asfortranarray(pred))],[truth],[0])[0,0])
        value=iou(exact);need(abs(value-float(r['coco_iou']))<1e-7,'Saved-coefficient decode differs from result')
        altvalue=iou(altmask) if xor else value
        size,domain=protocol['arms'][r['arm']];g=gindex[aid]
        pp=proto if size==160 else expanded
        truth_tensor=torch.tensor(supervised['gt_masks160'][g],device='cuda').float() if size==160 else rasters[aid].float()
        normalized=torch.tensor(supervised['gt_boxes_normalized'][g:g+1],device='cuda');area=normalized[:,2:].prod(1)
        lossbox=(ops.xywh2xyxy(normalized)*640 if domain=='gt' else box)*(size/640)
        support=ops.crop_mask(torch.ones((1,size,size),device='cuda'),lossbox)[0].bool()
        initial=torch.tensor(item['coeff'][j],device='cuda',requires_grad=True)
        official=v8SegmentationLoss.single_mask_loss(truth_tensor[None],initial[None],pp,lossbox,area)
        official_gradient=torch.autograd.grad(official,initial)[0]
        x=pp[:,support].T;y=truth_tensor[support]
        direct=F.binary_cross_entropy_with_logits(x@initial,y,reduction='sum')/(size*size*area[0])
        direct_gradient=torch.autograd.grad(direct,initial)[0]
        gradient_error=float((direct_gradient-official_gradient).abs().max())
        tolerance=1e-6+1e-4*float(official_gradient.abs().max())
        need(gradient_error<=tolerance,'Full coefficient-gradient amplitude witness failed')
        if not bool(support.any()):need(float(official.detach())==0. and not bool(official_gradient.any()),'Empty crop not zero loss/gradient')
        rows.append(dict(image_id=iid,annotation_id=aid,density=r['density'],arm=r['arm'],mask_xor_pixels=xor,
            max_abs_logit_roundoff=float((alternate-logits).abs().max()),logit_max_abs=float(logits.abs().max()),
            official_coco_iou=value,alternate_coco_iou=altvalue,coefficient_norm=float(c.norm()),
            initial_gradient_max_error=gradient_error,gradient_error_tolerance=tolerance,empty_support=not bool(support.any()),
            initial_loss_abs_error=abs(float(direct.detach())-float(official.detach()))))
save_csv(out/'decode_roundoff_witness.csv',rows)
summary=[]
for density in ['high','other']:
    for arm in protocol['arms']:
        q=[r for r in rows if r['density']==density and r['arm']==arm]
        summary.append(dict(density=density,arm=arm,n=len(q),any_mask_roundoff=sum(r['mask_xor_pixels']>0 for r in q),
            mask_xor_pixels_max=max(r['mask_xor_pixels'] for r in q),max_abs_logit_roundoff=max(r['max_abs_logit_roundoff'] for r in q),
            alternative_minus_official_mean_pp=100*float(np.mean([r['alternate_coco_iou']-r['official_coco_iou'] for r in q])),
            recovery75_disagreements=sum((r['alternate_coco_iou']>=.75)!=(r['official_coco_iou']>=.75) for r in q)))
write_json(out/'DECODE_WITNESS.json',dict(groups=summary,all_saved_coefficient_iou_replayed=True,targets=len(rows),seconds=time.monotonic()-start,
    script_sha256=sha(__file__),csv_sha256=sha(out/'decode_roundoff_witness.csv'),
    coefficient_gradient_amplitude_all_pass=True,initial_gradient_max_error=max(r['initial_gradient_max_error'] for r in rows),
    empty_support_official_witness_count=sum(r['empty_support'] for r in rows),
    scope='Official same-script re-decoding and float32 interpolation order witness; not independent evaluator or generalization.'))
print(json.dumps(summary),flush=True)
