"""S054: candidate ownership responses on fixed same-neighbor failures.

No input edit, training, GT coefficient fitting, or candidate reselection.
GT selects the already-frozen 15 S053 pairs and defines diagnostic pixels only.
"""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,io,json,shutil,time
import cv2,numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics import YOLO
from ultralytics.utils import ops,LOGGER
import ultralytics
from neighbor_background_probe import inferred,sha,dump,csvsave,gt_geometry
from frozen_mechanism_probe import ownership
from crowded_pixel_flow_probe import decode,encode
import crowded_failure_branch_probe as prior

BASE=Path(__file__).resolve().parent;SRC=BASE/'diagnostics/feature_cell_partition_20260912';PAIR=BASE/'diagnostics/subtype_neighbor_20260912'

def logits(c,p):
    z=(c@p.flatten(1)).reshape(1,1,*p.shape[-2:])
    return F.interpolate(z,(640,640),mode='bilinear',align_corners=False)[0,0]
def toorig(z,shape):
    return ops.scale_masks(z[None,None],shape)[0,0]
def rates(m,own,neighbor,union,crowd):
    v=~crowd;area=max(int((own&v).sum()),1);fp=m&~own&v
    return dict(iou=float((m&own).sum()/max((m|own).sum(),1)),coverage=float((m&own&v).sum()/area),neighbor_error=float((m&neighbor&~own&v).sum()/area),background_error=float((m&~union&v).sum()/area))
def stat(x):
    x=np.asarray(x,float);n=len(x);rng=np.random.default_rng(20260913);b=x[rng.integers(n,size=(2000,n))].mean(1)
    return dict(n=n,mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),positive=int((x>0).sum()),negative=int((x<0).sum())) if n else dict(n=0,mean=None,ci95=None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False);(out/'pairs').mkdir()
    manifest=json.loads((SRC/'manifest.json').read_text())['pairs'];pairs=[p for p in manifest if p['error_type']=='same_neighbor'];
    protocol=dict(experiment='S054_CANDIDATE_OWNERSHIP_RESPONSE',training=False,question='On pixels where target candidate leaks into its same-class neighbor, is the true neighbor candidate response already stronger?',
        source_manifest_sha256=sha(SRC/'manifest.json'),weight_sha256=sha(prior.WEIGHT),selection='All15same_neighbor pairs frozen in S053; no outcome re-selection.',
        response='Target and neighbor candidate coefficient vectors from original NMS source indices, multiplied by shared original prototype. Logits are bilinear upsampled to640 then scale-mapped to original image. Analyze target-owned, leaked-neighbor, chosen-neighbor, and background regions.',
        candidate='Neighbor source is COCO bbox50 ownership mapping in original detections. If no matched neighbor candidate, retain witness and exclude only pair-level neighbor comparison; do not replace.',
        competition='Diagnostic only: inside the original target candidate crop, assign each pixel to target if z_target >= z_neighbor and to neighbor otherwise, then threshold target response at zero. This uses no GT at inference; GT is used only to score fixed target candidate. Also report an oracle-like reassignment using max(target,neighbor) only as response ranking, not a submitted method.',
        metrics='Response margin z_target-z_neighbor; sign accuracy on GT target versus chosen neighbor; target leak pixels where target binary mask intersects neighbor GT; paired candidate competition IoU/error changes relative to original target mask.',
        limits='A matched neighbor box/candidate may not exist; candidate score/NMS already fixed. Logit comparison is not calibrated probability. Two candidates are only one local control and do not establish full COCO AP or causal mechanism. Pointwise image bootstrap 2000, exploratory.',
        stopping='Single fixed15-pair original-image pass; no training or automation changes.')
    dump(out/'protocol.json',protocol)
    for n in [Path(__file__).name,'neighbor_background_probe.py','crowded_failure_branch_probe.py','frozen_mechanism_probe.py','crowded_pixel_flow_probe.py']: 
        p=BASE/n
        if p.exists():shutil.copy2(p,out/n)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(prior.ANNOTATION))
    model=YOLO(str(prior.WEIGHT));model.model.eval().requires_grad_(False);rows=[];witness=[];start=time.monotonic();LOGGER.setLevel(40)
    with torch.inference_mode():
        for ix,item in enumerate(pairs,1):
            iid,aid,bid=item['image_id'],item['target'],item['neighbor'];folder=out/'pairs'/str(iid);folder.mkdir();im=cv2.imread(str(prior.IMAGES/f'{iid:012}.jpg'));shape=im.shape[:2]
            own,neighbor,_,union,crowd,_,_=gt_geometry(gt,iid,gt.anns[aid],gt.anns[bid]);base=inferred(model,im,gt,iid,aid,bid,{});proto=model.predictor.model.model.model[-1].proto;mapping=base['mapping'];tsource=item['source_index'];nslot=mapping.get(bid,-1);nsource=int(base['keep'][nslot]) if nslot>=0 else -1
            if int(base['keep'][item['prediction_slot']])!=tsource:raise RuntimeError('target source identity')
            c_t=base['raw'][0,84:,tsource];p=base['cap']['proto'];b_t=ops.xywh2xyxy(base['raw'][0,:4,tsource][None])[0];
            zt=logits(c_t,p);zt_o=toorig(zt,shape);mt=decode(c_t,p,b_t,shape);basevals=rates(mt,own,neighbor,union,crowd)
            rowbase=dict(image_id=iid,target=aid,neighbor=bid,original_iou=basevals['iou'],original_box_iou=item['original_box_iou'],neighbor_source=nsource,neighbor_matched=nsource>=0)
            if nsource<0:
                witness.append(dict(**rowbase,status='NO_NEIGHBOR_CANDIDATE'));dump(folder/'responses.json',rowbase);continue
            c_n=base['raw'][0,84:,nsource];b_n=ops.xywh2xyxy(base['raw'][0,:4,nsource][None])[0];zn=logits(c_n,p);zn_o=toorig(zn,shape);mn=decode(c_n,p,b_n,shape)
            valid=~crowd;targetpix=own&valid;neighpix=neighbor&~own&valid;leakpix=mt&neighpix;backpix=~union&valid
            margin=(zt_o-zn_o).cpu().numpy();ztv=zt_o.cpu().numpy();znv=zn_o.cpu().numpy();
            def region(x):
                q=margin[x];return dict(n=int(x.sum()),target_mean=float(ztv[x].mean()) if x.any() else None,neighbor_mean=float(znv[x].mean()) if x.any() else None,margin=stat(q),target_win=float((q>=0).mean()) if x.any() else None)
            # response-only competition; output target mask uses target-v-neighbor winner and target's threshold.
            compmask=(zt_o>=zn_o)&(zt_o>0)&(ops.scale_masks((ops.crop_mask((zt>0).byte()[None],b_t[None]))[:,None],shape)[0,0]>0)
            # crop-mask is the original target support; competition only removes pixels where neighbor logit wins.
            compvals=rates(compmask.cpu().numpy(),own,neighbor,union,crowd)
            # A second diagnostic preserves target positive support but uses max-response ownership.
            target_support=decode(c_t,p,b_t,shape);oracle_keep=(zt_o>=zn_o);rankmask=target_support&oracle_keep.cpu().numpy();rankvals=rates(rankmask,own,neighbor,union,crowd)
            for name,x in [('target_gt',targetpix),('neighbor_gt',neighpix),('leak',leakpix),('background',backpix),('target_boundary',cv2.dilate(own.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)&valid)]:
                q=region(x);rows.append(dict(**rowbase,region=name,**q,competition_iou=compvals['iou'],competition_coverage=compvals['coverage'],competition_neighbor_error=compvals['neighbor_error'],competition_background_error=compvals['background_error'],rank_iou=rankvals['iou'],rank_coverage=rankvals['coverage'],rank_neighbor_error=rankvals['neighbor_error'],rank_background_error=rankvals['background_error'],target_mask_iou=basevals['iou'],target_mask_coverage=basevals['coverage'],target_mask_neighbor_error=basevals['neighbor_error'],target_mask_background_error=basevals['background_error']))
            # Save exact response fields and masks for review; no image selection after results.
            np.savez_compressed(folder/'responses.npz',target_logit=ztv,neighbor_logit=znv,margin=margin,target_mask=mt,neighbor_mask=mn,own=own,neighbor_gt=neighbor,competition=compmask.cpu().numpy(),target_support=target_support)
            dump(folder/'responses.json',dict(**rowbase,regions={name:region(x) for name,x in [('target_gt',targetpix),('neighbor_gt',neighpix),('leak',leakpix),('background',backpix)]},
                competition=compvals,rank_mask=rankvals,leak_pixels=int(leakpix.sum()),target_mask_pixels=int(mt.sum()),neighbor_mask_pixels=int(mn.sum())))
            witness.append(dict(**rowbase,status='COMPLETE',target_source_exact=True,neighbor_candidate_exact=True,response_saved=True));dump(out/'progress.json',dict(completed=ix,total=len(pairs),seconds=round(time.monotonic()-start,2)))
            print(json.dumps(dict(completed=ix,total=len(pairs),image_id=iid,neighbor_matched=True,leak=int(leakpix.sum()),seconds=round(time.monotonic()-start,1))),flush=True)
            del base;torch.cuda.empty_cache()
    csvsave(out/'metrics.csv',rows);dump(out/'WITNESS.json',witness);dump(out/'RUNTIME.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(),weight=str(prior.WEIGHT),logit_resolution=[640,640]))
    # Image-level aggregate: one row per pair/region, and paired score changes on all matched pairs.
    frame=[]
    for name,q in __import__('pandas').DataFrame(rows).groupby('region'):
        frame.append(dict(region=name,n=len(q),margin=stat(q.margin_mean if 'margin_mean' in q else q['margin'].map(lambda x:np.nan))))
    dump(out/'COMPLETE.json',dict(status='COMPLETE',images=len(pairs),matched=sum(x['status']=='COMPLETE' for x in witness),rows=len(rows),seconds=round(time.monotonic()-start,3),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}));print('COMPLETE',len(pairs),len(rows),flush=True)
if __name__=='__main__':main()
