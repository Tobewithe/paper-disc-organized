"""S020 persisted coefficient/official decode replay and native-target loss witness."""
import contextlib, io, json, time
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from candidate_lineage_probe import ROOT, read, need, write_json, save_csv, sha

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
out=ROOT/'diagnostics/native_label887_20260912'
receipt=json.loads((out/'COMPLETE.json').read_text());need(receipt['status']=='COMPLETE','Not complete')
for name,h in receipt['hashes'].items():need(sha(out/name)==h,'Receipt mismatch: '+name)
rows=read(out/'metrics.csv')
old={int(r['annotation_id']):r for r in read(ROOT/'diagnostics/grid_support887_20260912/targets.csv') if r['arm']=='original'}
protocol=json.loads((out/'protocol.json').read_text());witness=[];start=time.monotonic()
with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
for iid in protocol['images']:
    with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as z:item={k:z[k] for k in z.files}
    with np.load(ROOT/'diagnostics/assignment300_20260912/images'/f'{iid}.npz') as z:sup={k:z[k] for k in z.files}
    with np.load(out/'images'/f'{iid}.npz') as z:params={k:z[k] for k in z.files}
    source={int(s):j for j,s in enumerate(sup['original_final_sources'])}
    gindex={int(a):g for g,a in enumerate(sup['annotation_ids'])}
    proto=torch.tensor(item['proto'],device='cuda')
    for row in [r for r in rows if int(r['image_id'])==iid]:
        aid=int(row['annotation_id']);arm=row['arm'];j=source[int(old[aid]['source_index'])]
        cc=torch.tensor(params[f'{aid}_{arm}'],device='cuda');box=torch.tensor(item['boxes'][j:j+1],device='cuda')
        pred=ops.process_mask(proto,cc[None],box,(640,640),upsample=True)[0]
        original=(ops.scale_masks(pred[None,None],tuple(map(int,item['shape'])))[0,0]>.5).cpu().numpy().astype(np.uint8)
        value=float(mu.iou([mu.encode(np.asfortranarray(original))],[gt.annToRLE(gt.anns[aid])],[0])[0,0])
        need(abs(value-float(row['coco_iou']))<1e-7,'Saved coefficient decode differs')
        record=dict(image_id=iid,annotation_id=aid,arm=arm,coco_iou=value,loss_before_error=None,loss_after_error=None,gradient_max_error=None)
        if arm=='NATIVE_OVERLAP160':
            y=torch.tensor(params[f'{aid}_native_overlap160'],device='cuda').float()
            norm=torch.tensor(sup['gt_boxes_normalized'][gindex[aid]:gindex[aid]+1],device='cuda')
            lossbox=ops.xywh2xyxy(norm)*160;area=norm[:,2:].prod(1)
            c0=torch.tensor(item['coeff'][j],device='cuda',requires_grad=True)
            off=v8SegmentationLoss.single_mask_loss(y[None],c0[None],proto,lossbox,area)
            grad=torch.autograd.grad(off,c0)[0]
            support=ops.crop_mask(torch.ones((1,160,160),device='cuda'),lossbox)[0].bool()
            direct=F.binary_cross_entropy_with_logits(proto[:,support].T@c0,y[support],reduction='sum')/(160*160*area[0])
            direct_grad=torch.autograd.grad(direct,c0)[0]
            err=float((grad-direct_grad).abs().max());need(err<=1e-6+1e-4*float(grad.abs().max()),'Native-target gradient amplitude mismatch')
            after=v8SegmentationLoss.single_mask_loss(y[None],cc[None],proto,lossbox,area)
            before_error=abs(float(off.detach())-float(row['loss_before']))
            after_error=abs(float(after.detach())-float(row['loss_after']))
            need(before_error<=1e-5+1e-4*abs(float(off.detach())),'Initial native loss mismatch')
            need(after_error<=1e-5+1e-4*abs(float(after.detach())),'Final native loss mismatch')
            record.update(loss_before_error=before_error,loss_after_error=after_error,gradient_max_error=err)
        witness.append(record)
save_csv(out/'decode_witness.csv',witness)
native=[r for r in witness if r['gradient_max_error'] is not None]
write_json(out/'DECODE_WITNESS.json',dict(all_saved_coefficients_replayed=True,decoded=len(witness),
    native_target_loss_witness=len(native),gradient_max_error=max(r['gradient_max_error'] for r in native),
    before_loss_max_error=max(r['loss_before_error'] for r in native),after_loss_max_error=max(r['loss_after_error'] for r in native),
    receipt_files_verified=len(receipt['hashes']),seconds=time.monotonic()-start,script_sha256=sha(__file__),
    csv_sha256=sha(out/'decode_witness.csv'),scope='Same installed official decode re-read saved coefficients; native-target single-mask loss witness. Not independent refit or official model historical training.'))
print((out/'DECODE_WITNESS.json').read_text())
