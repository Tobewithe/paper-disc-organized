"""Original-COCO mask IoUs and bounded convergence witness for remaining failures."""
import argparse,contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops
from no_candidate_readout_probe import coefficient_logits,fit_readout
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha
from crossimage_response_experiment import gt_input_regions


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    metrics=read(out/'metrics.csv');targets=read(out/'targets.csv');ids=json.loads((out/'protocol.json').read_text())['images']
    rows=[];convergence=[];start=time.monotonic();parameter_files={}
    for iid in ids:
        rr=[r for r in targets if int(r['image_id'])==iid and r['fit_status']=='ok']
        if not rr:continue
        with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as z:item={k:z[k] for k in z.files}
        with np.load(out/'images'/f'{iid}.npz') as z:params={k:z[k] for k in z.files}
        proto=torch.tensor(item['proto'],device='cuda');shape=tuple(map(int,item['shape']));inp=tuple(map(int,item['input_shape']))
        raster,_,valid=gt_input_regions(gt,iid,item)
        expanded=torch.nn.functional.interpolate(proto[None],inp,mode='bilinear',align_corners=False)[0].flatten(1).T
        for r in rr:
            aid=int(r['annotation_id']);box=torch.tensor(params[f'{aid}_box'],device='cuda');c0=torch.tensor(params[f'{aid}_original_coefficient'],device='cuda')
            current={r['arm']:r for r in metrics if int(r['annotation_id'])==aid};gtrle=gt.annToRLE(gt.anns[aid])
            def measure(c,b):
                z=coefficient_logits(proto,c,inp,b)
                binary=ops.crop_mask((z>0)[None].to(torch.uint8),box)[0]
                orig=(ops.scale_masks(binary[None,None],shape)[0,0]>.5).cpu().numpy().astype(np.uint8)
                iou=float(mu.iou([mu.encode(np.asfortranarray(orig))],[gtrle],[0])[0,0])
                vv=binary.bool()&valid;tp=int((vv&raster[aid]).sum());fp=int((vv&~raster[aid]).sum());area=int((raster[aid]&valid).sum())
                return iou,tp/max(area+fp,1)
            for arm in ['original','threshold_oracle','free_coefficient','free_coefficient_bias']:
                c=c0 if arm in ['original','threshold_oracle'] else torch.tensor(params[f'{aid}_{arm}_coefficient'],device='cuda')
                bias=float(current[arm]['bias']);iou,inputiou=measure(c,bias)
                need(abs(inputiou-float(current[arm]['input_iou']))<1e-12,'Input decode replay changed')
                if arm=='original':need(abs(iou-float(r['anchor_official_mask_iou']))<1e-12,'Original RLE IoU mismatch')
                rows.append(dict(image_id=iid,annotation_id=aid,anchor=r['anchor'],arm=arm,
                    original_coco_mask_iou=iou,valid_domain_iou=float(current[arm]['mask_iou']),input_iou=inputiou,area=float(r['area'])))
            if float(current['free_coefficient']['mask_iou'])>=.75:continue
            # Post-hoc execution witness only for unresolved valid-domain targets.
            # Start from previously selected coefficient; no input/target changes.
            c=torch.tensor(params[f'{aid}_free_coefficient_coefficient'],device='cuda')
            support=ops.crop_mask(torch.ones((1,*inp),device='cuda',dtype=torch.uint8),box)[0].bool()
            pos=(support&valid).flatten().nonzero().flatten();x=expanded[pos];y=raster[aid].flatten()[pos].float();total=int((raster[aid]&valid).sum())
            states=fit_readout(x,y,c,False,total,240);candidates=[]
            for state in states:
                cc=torch.tensor(state['coefficient'],device='cuda');iou,inputiou=measure(cc,0.)
                candidates.append((inputiou,state,iou))
                convergence.append(dict(image_id=iid,annotation_id=aid,phase=state['phase'],
                    input_iou=inputiou,original_coco_mask_iou=iou,iterations=state['iterations'],gradient_max=state['gradient_max'],
                    crop_coverage=float(r['crop_coverage']),reference_input_iou=float(current['free_coefficient']['input_iou']),
                    reference_valid_iou=float(current['free_coefficient']['mask_iou'])))
            chosen=max(candidates,key=lambda q:q[0]);parameter_files[str(aid)]=chosen[1]['coefficient']
    save_csv(out/'original_coco_mask_iou.csv',rows);save_csv(out/'convergence_witness.csv',convergence)
    np.savez_compressed(out/'convergence_coefficients.npz',**parameter_files)
    groups=[]
    for group in ['all_eligible','original_final_match','unretained_raw_anchor','small','medium_large']:
        for arm in ['original','threshold_oracle','free_coefficient','free_coefficient_bias']:
            rr=[r for r in rows if r['arm']==arm]
            if group=='original_final_match':rr=[r for r in rr if r['anchor']=='fixed_original_bbox50_match']
            if group=='unretained_raw_anchor':rr=[r for r in rr if r['anchor']=='GT_best_geometry_postconf_candidate']
            if group=='small':rr=[r for r in rr if r['area']<1024]
            if group=='medium_large':rr=[r for r in rr if r['area']>=1024]
            groups.append(dict(group=group,arm=arm,n=len(rr),coco_iou75=sum(r['original_coco_mask_iou']>=.75 for r in rr),
                coco_iou90=sum(r['original_coco_mask_iou']>=.9 for r in rr),
                mean_coco_iou=100*float(np.mean([r['original_coco_mask_iou'] for r in rr])) if rr else None))
    write_json(out/'ORIGINAL_DOMAIN_ANALYSIS.json',dict(groups=groups,convergence_targets=len(parameter_files),
        scope='Exact pycocotools mask IoU against original uncropped ordinary GT annotation, crowd regions not removed. Fixed target comparison, not official rematched AP/R75.',
        convergence='Post-hoc 240+240 LBFGS steps on 26 previous valid-domain unresolved targets, starting selected coefficient. Select by same fitting input GT, not original-resolution score.',
        seconds=time.monotonic()-start,script_sha256=sha(__file__),hashes={n:sha(out/n) for n in ['original_coco_mask_iou.csv','convergence_witness.csv','convergence_coefficients.npz']}))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
