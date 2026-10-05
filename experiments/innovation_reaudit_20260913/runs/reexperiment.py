"""Isolated, fixed-protocol reassessment. No original artifacts overwritten."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['YOLO_CONFIG_DIR']=str(__import__('pathlib').Path(__file__).parent/'settings')
import sys, json, gzip, hashlib, contextlib, io, time, argparse, shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'local_readout_runtime_20260912/vendor'))
sys.path.insert(0,str(OUT/'source'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
import ultralytics
from bifurcated_spatial_readout import BifurcatedSpatialHead,bsr_loss
from ada_calib_readout import AdaCalibHead
from contrastive_spatial_loss import apply_centerness_prior,prototype_orthogonality_loss
from rich_pixel_readout import GlobalHead,instance_features
from summarize_relative_ownership import evaluate
from eval_readout_input_pilot import ici

CACHE=ROOT/'diagnostics/readout_input_scale1200_20260912/cache'
LABELS=ROOT/'diagnostics/shared_label_controls_20260912'
OLD=ROOT/'runs/innovations_suite_aligned_s3_e15'
DEVICE='cuda'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.38)

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(a):return torch.as_tensor(a,device=DEVICE).float()
def read(p):
    with np.load(p) as f:return {k:f[k] for k in f.files}
def norm(x,n):return (x-n['mean'])/n['std']
def state(stage,**kw):
    row=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    write(OUT/'STATUS.json',row);print(json.dumps(row),flush=True)

def decode(z,boxes,ishape,shape,late=False):
    up=F.interpolate(z[:,None],ishape,mode='bilinear',align_corners=False)[:,0]
    cropped=ops.crop_mask(up,boxes)
    if late:return ops.scale_masks(cropped[:,None],shape)[:,0]>0
    binary=(cropped>0).float()
    return ops.scale_masks(binary[:,None],shape)[:,0]>.5

def load_old():
    models={'original':None};hashes={}
    for seed in range(3):
        paths={'s032':LABELS/f'raw_coco_s{seed}/checkpoints/epoch015.pt',
               'bsr':ROOT/f'runs/bsr_head_aligned_s3_e15/bsr_head_s{seed}/checkpoints/epoch015.pt'}
        paths.update({m:OLD/f'{m}_s{seed}/checkpoints/epoch015.pt' for m in ['ada_calib','contrastive','ortho_center','full_synergy']})
        for mode,path in paths.items():
            m=GlobalHead() if mode=='s032' else AdaCalibHead() if mode in ['ada_calib','full_synergy'] else BifurcatedSpatialHead()
            m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            models[f'{mode}_s{seed}']=m.to(DEVICE).eval();hashes[str(path)]=sha(path)
    write(OUT/'old_checkpoint_hashes.json',hashes)
    return models

def get_gt(ids):
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(CACHE/'conversion_input/instances_probe.json'))
    sub=COCO();sub.dataset=dict(info=gt.dataset.get('info',{}),categories=list(gt.cats.values()),
        images=[gt.imgs[i] for i in ids],annotations=[a for i in ids for a in gt.imgToAnns[i]])
    with contextlib.redirect_stdout(io.StringIO()):sub.createIndex()
    meta={a['id']:{'ici_same':ici(a,[b for b in sub.imgToAnns[a['image_id']] if not b.get('iscrowd',0)])}
          for a in sub.anns.values() if not a.get('iscrowd',0)}
    return sub,meta

def reevaluate(models,arms,dest,ids,normalizer):
    dest.mkdir(parents=True,exist_ok=True)
    gt,meta=get_gt(ids);cats=sorted(gt.cats)
    # Each arm is (model key, center on/off, late threshold on/off).
    pred={name:[] for name in arms};witness=[]
    for ni,iid in enumerate(ids):
        q=read(CACHE/'images'/f'{iid}.npz')
        c,p,boxes,det,h=map(tensor,[q['coeff'],q['proto'],q['boxes'],q['detections'],q['h']])
        ishape=tuple(map(int,q['input_shape']));shape=tuple(map(int,q['shape']))
        level=torch.as_tensor(q['level'],device=DEVICE).long()
        with torch.inference_mode():
            x=norm(instance_features(h,level,boxes,ishape),normalizer)
            for first in range(0,len(c),8):
                sl=slice(first,first+8);zs={}
                for name,(key,center,late) in arms.items():
                    if key not in zs:
                        m=models[key]
                        if m is None:zs[key]=torch.einsum('bc,chw->bhw',c[sl],p)
                        elif key.startswith('s032'):zs[key]=torch.einsum('bc,chw->bhw',c[sl]+m(x[sl]),p)
                        else:zs[key]=m.forward_inference(x[sl],c[sl],p)[0]
                    z=zs[key]
                    if center:
                        ph,pw=z.shape[-2:];pb=boxes[sl]*boxes.new_tensor([pw/ishape[1],ph/ishape[0],pw/ishape[1],ph/ishape[0]])
                        z=apply_centerness_prior(z,pb,(ph,pw),margin_scale=.5)
                    pm=decode(z,boxes[sl],ishape,shape,late)
                    if ni<3 and key=='original' and not center and not late:
                        official=ops.process_mask(p,c[sl],boxes[sl],ishape,upsample=True)
                        official=ops.scale_masks(official[:,None],shape)[:,0]>.5
                        err=int((pm!=official).sum());assert err==0,err
                        witness.append(dict(image_id=iid,first=first,decoder_xor=err))
                    masks=pm.cpu().numpy().astype(np.uint8)
                    for k,mask in enumerate(masks):
                        # Keep all frozen candidates, including empty masks, consistently.
                        rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
                        j=first+k
                        pred[name].append(dict(image_id=int(iid),category_id=cats[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
        if (ni+1)%10==0:state('decode_'+dest.name,images=ni+1,total=len(ids),arms=len(arms))
    write(dest/'decoder_witness.json',witness)
    rows=[];gtrows=[];pairrows=[]
    for name,pp in pred.items():
        with gzip.open(dest/f'{name}.json.gz','wt',encoding='utf-8') as f:json.dump(pp,f)
        row,rr,pr=evaluate(gt,meta,ids,pp,name);row['gap']=row['r75_low']-row['r75_high']
        rows.append(row);gtrows+=rr;pairrows+=pr
        pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False)
        state('score_'+dest.name,arm=name,ap=row['mask_ap'],high=row['r75_high'])
    pd.DataFrame(gtrows).to_csv(dest/'gt_recovery.csv',index=False)
    pd.DataFrame(pairrows).to_csv(dest/'pair_recovery.csv',index=False)
    write(dest/'COMPLETE.json',dict(images=len(ids),arms=len(arms),gt=len(meta),scope='explored train2017 transfer; no claim of pristine test',hashes={p.name:sha(p) for p in dest.glob('*.csv')}))
    return pd.DataFrame(rows)

def audit(models):
    a=models['bsr_s0'];b=models['ortho_center_s0']
    diffs={}
    for s in range(3):
        sa=models[f'bsr_s{s}'].state_dict();sb=models[f'ortho_center_s{s}'].state_dict()
        diffs[str(s)]=max(float((sa[k]-sb[k]).abs().max()) for k in sa)
    q=read(CACHE/'images'/f'{json.loads((CACHE/"selection.json").read_text())["fit"][0]}.npz')
    p=tensor(q['sample_p'][:,:512]);loss=prototype_orthogonality_loss(p)
    write(OUT/'implementation_audit.json',dict(orth_loss_requires_grad=loss.requires_grad,
        old_orth_vs_bsr_max_parameter_difference=diffs,orth_value=float(loss),
        note='Frozen prototype loss has no parameter gradient. Differences in checkpoints do not establish orthogonalization.'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='reeval');args=parser.parse_args()
    sel=json.loads((CACHE/'selection.json').read_text());assert not set(sel['fit'])&set(sel['transfer'])
    normalizer=torch.load(LABELS/'normalizer.pt',map_location=DEVICE,weights_only=False)
    write(OUT/'PROTOCOL.json',dict(phase=args.phase,torch=torch.__version__,ultralytics=ultralytics.__version__,
        runtime_source=ultralytics.__file__,fit_images=len(sel['fit']),transfer_images=len(sel['transfer']),
        seeds=[0,1,2],center_margin_scale=.5,decoder='bilinear to input, predicted-box crop, threshold0, resize binary to original, threshold.5',
        fixed_candidates=True,old_results_untouched=True,selection_sha=sha(CACHE/'selection.json'),
        limitation='Repeatedly explored COCO train2017 transfer; preliminary mechanism selection only. No AP significance claim from seed spread.'))
    models=load_old();audit(models)
    arms={key:(key,key.startswith(('ortho_center','full_synergy')),False) for key in models}
    for s in range(3):
        for m in ['s032','bsr','ada_calib']:arms[f'{m}_center_s{s}']=(f'{m}_s{s}',True,False)
        for m in ['ortho_center','full_synergy']:arms[f'{m}_nocenter_s{s}']=(f'{m}_s{s}',False,False)
        arms[f's032_late_s{s}']=(f's032_s{s}',False,True)
    arms['original_late']=('original',False,True)
    reevaluate(models,arms,OUT/'existing_aligned',sel['transfer'],normalizer)
    state('existing_reevaluation_complete')

if __name__=='__main__':main()
