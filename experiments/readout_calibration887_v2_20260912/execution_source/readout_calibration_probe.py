"""S021: calibration versus spatial ranking, native640 labels, same887 failures.

Temporary per-instance oracle only. Freeze network/prototypes/boxes. Reuse S020
native640 coefficients; fit positive scalar/affine calibration with exactly the
same native640 GT-crop BCE. Rank diagnostics use disjoint COCO spatial regions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from copy import deepcopy
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from native_label_pipeline_probe import format_native,mask_of
from no_candidate_readout_probe import exact_threshold
from crossimage_response_experiment import gt_input_regions
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha

def calibrate(z,y,factor,bias):
    """Positive alpha=exp(rho), optional beta; same native640 BCE support/area."""
    zz=z.detach().double();yy=y.detach().double();v=torch.nn.Parameter(torch.zeros(2 if bias else 1,device=z.device,dtype=torch.float64))
    def objective():return F.binary_cross_entropy_with_logits(v[0].exp()*zz+(v[1] if bias else 0.),yy,reduction='sum')*factor
    before=float(objective().detach());opt=torch.optim.LBFGS([v],lr=1.,max_iter=120,max_eval=240,
        tolerance_grad=1e-7,tolerance_change=1e-12,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=objective();need(bool(torch.isfinite(loss)),'Nonfinite calibration objective');loss.backward();return loss
    opt.step(closure);loss=objective();gradient=torch.autograd.grad(loss,v)[0];state=opt.state[v]
    alpha=float(v[0].detach().exp());beta=float(v[1].detach()) if bias else 0.;after=float(loss.detach())
    need(np.isfinite([alpha,beta,after]).all() and alpha>0,'Nonfinite calibration')
    need(after<=before+1e-6,'Calibration increased loss')
    return dict(alpha=alpha,beta=beta,loss_before=before,loss_after=after,
        success=int(state['n_iter'])<120,status='iteration_limit' if int(state['n_iter'])>=120 else 'stopped_before_limit',
        iterations=int(state['n_iter']),evaluations=int(state['func_evals']),gradient_max=float(gradient.abs().max()),alpha_near_zero=alpha<1e-6)

def auc(pos,neg):
    if not len(pos) or not len(neg):return None
    sorted_neg=np.sort(neg.astype(np.float64))
    below=np.searchsorted(sorted_neg,pos,side='left');leq=np.searchsorted(sorted_neg,pos,side='right')
    return float((below+.5*(leq-below)).mean()/len(neg))

def original_measure(binary,shape,truth,own,neighbor,bg,other,valid):
    pred=(ops.scale_masks(binary[None,None],shape)[0,0]>.5).cpu().numpy().astype(bool)
    value=float(mu.iou([mu.encode(np.asfortranarray(pred.astype(np.uint8)))],[truth],[0])[0,0])
    mask=pred&valid;area=int(own.sum());tp=int((mask&own).sum());sn=int((mask&neighbor).sum());back=int((mask&bg).sum());oth=int((mask&other).sum())
    return dict(coco_iou=value,valid_pixels=area,coverage=tp/area if area else None,same_neighbor=sn/area if area else None,
        background=back/area if area else None,other_error=oth/area if area else None,valid_iou=tp/(area+sn+back+oth) if area else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    a.out.mkdir(exist_ok=False);(a.out/'images').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    previous=ROOT/'diagnostics/native_label887_20260912';s019=ROOT/'diagnostics/grid_support887_20260912';assignment=ROOT/'diagnostics/assignment300_20260912'
    selected=[r for r in read(previous/'metrics.csv') if r['arm']=='NATIVE_OVERLAP640_NATIVEBOX'];need(len(selected)==887,'Fixed887 mismatch')
    old={int(r['annotation_id']):r for r in read(s019/'targets.csv') if r['arm']=='original'}
    ids=sorted({int(r['image_id']) for r in selected})
    if a.smoke:ids=sorted({min(int(r['image_id']) for r in selected if r['density']==d) for d in ['high','other']})
    selected=[r for r in selected if int(r['image_id']) in ids]
    cfg=get_cfg(overrides=json.loads((ROOT/'train_config.json').read_text()));data=check_det_dataset(str(ROOT/'coco_clean.yaml'),autodownload=False)
    ds=build_yolo_dataset(cfg,data['val'],16,data,mode='val',rect=False);dsindex={int(Path(x['im_file']).stem):i for i,x in enumerate(ds.labels)}
    sidecar=json.loads((ROOT/'audits/actual_loader_ids_val2017.json').read_text())
    protocol=dict(experiment='S021',network_training=False,oracle=True,images=ids,target_ids=[int(r['annotation_id']) for r in selected],
        arms=['original','positive_scale_BCE','positive_affine_BCE','native640_free32_BCE','native640_threshold_oracle','free32_norm_restored','free32_double_decode'],
        common='Same S020 common cached geometry, native640 overlap masks, nativeGT crop and area; original P/c/final boxes. Native labels reconstructed with official Format and independently ID-reordered at ratio1.',
        calibration='alpha=exp(rho)>0; optional beta; pure native640 GT-crop BCE / (640^2 * native GT area); torch float64 LBFGS 120 iterations/240 evaluation budget; final state, no IoU selection, no convergence claim.',
        free32='Reuse S020 same-target achieved120-iteration float32 LBFGS coefficient; no refit. Different parameter count/solver implementation are disclosed; no convergence claim.',
        threshold='Auxiliary GT oracle chooses threshold maximizing native640 input-grid IoU inside fixed prediction box; native GT pixels outside box retained in denominator; explicit zero control; different selection objective from BCE arms, not original-resolution IoU optimum.',
        spatial='Original COCO RLE final IoU. Separate original-resolution COCO coverage/neighbor/background excludes crowd. Ranking AUC uses input640 nearest-exact COCO GT, valid image and crowd exclusion, inside fixed prediction box; own/neighbor/background disjoint.',
        scale='Positive scaling alone preserves threshold0 signs. Decode this arm with exact original binary to avoid float underflow artifacts; affine uses threshold tau=-beta/alpha on original logits.',
        numerical='Full coefficient norm restoration and float64 low-grid combination+bilinear interpolation test float32 cancellation, no optimization. Saved objective replay uses exact S020 scaled support-matrix order and shared RMS; also disclose unscaled full-map rounding difference.',
        restrictions='Fixed explored failure cohort; same-image GT oracle, no shared-head learning, no new-image validation, no AP, no historical training causality or innovation claim.',
        script_sha256=sha(__file__),source_sha256=sha(previous/'metrics.csv'),annotation_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),
        helper_hashes={name:sha(Path(__file__).with_name(name)) for name in ['native_label_pipeline_probe.py','no_candidate_readout_probe.py','crossimage_response_experiment.py']})
    write_json(a.out/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    metrics=[];ranking=[];calibration=[];started=time.monotonic()
    for number,iid in enumerate(ids,1):
        k=dsindex[iid];mapped=list(map(int,sidecar[str(iid)]['annotation_ids']));lab=deepcopy(ds.labels[k]);lab.pop('shape',None)
        lab['img']=cv2.imread(lab['im_file']);h,w=lab['img'].shape[:2];lab=ds.update_labels_info(lab);lab['ori_shape']=(h,w);lab['ratio_pad']=(1.,1.)
        common=LetterBox(new_shape=(640,640),auto=False,scaleup=True)(lab);native=format_native(common,mapped,1,True)
        paths=dict(cache=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz',supervision=assignment/'images'/f'{iid}.npz',native=previous/'images'/f'{iid}.npz')
        prior=json.loads((previous/'images'/f'{iid}.json').read_text())
        for key in ['cache','supervision']:need(sha(paths[key])==prior['input_hashes'][key],'Input changed')
        with np.load(paths['cache']) as z:item={key:z[key] for key in z.files}
        with np.load(paths['supervision']) as z:sup={key:z[key] for key in z.files}
        with np.load(paths['native']) as z:params={key:z[key] for key in z.files}
        sources={int(v):j for j,v in enumerate(sup['original_final_sources'])};proto=torch.tensor(item['proto'],device='cuda')
        p640=F.interpolate(proto[None],(640,640),mode='bilinear',align_corners=False)[0]
        rasters,union640,valid640=gt_input_regions(gt,iid,item);shape=tuple(map(int,item['shape']))
        orig={ann['id']:gt.annToMask(ann).astype(bool) for ann in gt.imgToAnns[iid]};valid=np.ones(shape,bool);union=np.zeros(shape,bool)
        for ann in gt.imgToAnns[iid]:
            if ann.get('iscrowd',0):valid&=~orig[ann['id']]
            else:union|=orig[ann['id']]
        image_metrics=[];image_ranking=[];image_cal=[];saved={}
        for row in [r for r in selected if int(r['image_id'])==iid]:
            aid=int(row['annotation_id']);ann=gt.anns[aid];j=sources[int(old[aid]['source_index'])]
            c0=torch.tensor(item['coeff'][j],device='cuda');cf=torch.tensor(params[f'{aid}_NATIVE_OVERLAP640_NATIVEBOX'],device='cuda')
            box=torch.tensor(item['boxes'][j:j+1],device='cuda');norm=torch.tensor(native['boxes'][native['position'][aid]:native['position'][aid]+1],device='cuda')
            lossbox=ops.xywh2xyxy(norm)*640;area=norm[0,2:].prod();support=ops.crop_mask(torch.ones((1,640,640),device='cuda'),lossbox)[0].bool()
            predsupport=ops.crop_mask(torch.ones((1,640,640),device='cuda'),box)[0].bool()
            y=torch.tensor(mask_of(native,aid),device='cuda').float()
            zfit=(c0@p640.flatten(1)).reshape(640,640);zf_fit=(cf@p640.flatten(1)).reshape(640,640)
            z=F.interpolate((c0@proto.flatten(1)).reshape(1,1,160,160),(640,640),mode='bilinear',align_corners=False)[0,0]
            zf=F.interpolate((cf@proto.flatten(1)).reshape(1,1,160,160),(640,640),mode='bilinear',align_corners=False)[0,0]
            factor=1./(640*640*float(area));before=float(F.binary_cross_entropy_with_logits(zfit[support],y[support],reduction='sum')*factor)
            after=float(F.binary_cross_entropy_with_logits(zf_fit[support],y[support],reduction='sum')*factor)
            need(abs(before-float(row['loss_before']))<=1e-5+1e-4*abs(before),'Native initial target/loss replay failed')
            g=list(map(int,sup['annotation_ids'])).index(aid)
            raw_norm=torch.tensor(sup['gt_boxes_normalized'][g:g+1],device='cuda');rawbox=ops.xywh2xyxy(raw_norm)*640
            scale_support=ops.crop_mask(torch.ones((1,160,160),device='cuda'),rawbox/4)[0].bool()
            scale=proto[:,scale_support].square().mean(1).sqrt().clamp_min(.01) if bool(scale_support.any()) else proto.square().mean((1,2)).sqrt().clamp_min(.01)
            xs=p640[:,support].T.contiguous()/scale
            scaled_before=float(F.binary_cross_entropy_with_logits(xs@(c0*scale),y[support],reduction='sum')/(640*640*area))
            scaled_after=float(F.binary_cross_entropy_with_logits(xs@(cf*scale),y[support],reduction='sum')/(640*640*area))
            scaled_error=abs(scaled_after-float(row['loss_after']))
            # Saved c=w/scale may itself lose a final rounding bit; retain the
            # discrepancy explicitly rather than treating this as exact w replay.
            need(abs(scaled_before-float(row['loss_before']))<=1e-5+1e-4*abs(scaled_before),'Scaled initial objective replay failed')
            need(int(y[support].sum())==int(row['positive_pixels']) and int(support.sum())==int(row['pixels']),'Native support/positive count changed')
            original=ops.process_mask(proto,c0[None],box,(640,640),upsample=True)[0];free=ops.process_mask(proto,cf[None],box,(640,640),upsample=True)[0]
            need(torch.equal(original,ops.crop_mask((z>0)[None].to(torch.uint8),box)[0]),'Original logit decode differs')
            need(torch.equal(free,ops.crop_mask((zf>0)[None].to(torch.uint8),box)[0]),'Free logit decode differs')
            fits={arm:calibrate(zfit[support],y[support],factor,bias) for arm,bias in [('positive_scale_BCE',False),('positive_affine_BCE',True)]}
            tau,_=exact_threshold(z[predsupport],y[predsupport],int(y.sum()))
            normalized_cf=cf*(c0.norm()/cf.norm().clamp_min(1e-30))
            normalized_binary=ops.process_mask(proto,normalized_cf[None],box,(640,640),upsample=True)[0]
            zd=F.interpolate((cf.double()@proto.double().flatten(1)).reshape(1,1,160,160),(640,640),mode='bilinear',align_corners=False)[0,0]
            double_binary=ops.crop_mask((zd>0)[None].to(torch.uint8),box)[0]
            binaries={'original':original,'positive_scale_BCE':original,'native640_free32_BCE':free,
                'positive_affine_BCE':ops.crop_mask((z.double()>-fits['positive_affine_BCE']['beta']/fits['positive_affine_BCE']['alpha'])[None].to(torch.uint8),box)[0],
                'native640_threshold_oracle':ops.crop_mask((z>tau)[None].to(torch.uint8),box)[0],
                'free32_norm_restored':normalized_binary,'free32_double_decode':double_binary}
            own=orig[aid]&valid;same=np.zeros(shape,bool);same640=torch.zeros((640,640),device='cuda',dtype=torch.bool)
            for q in gt.imgToAnns[iid]:
                if not q.get('iscrowd',0) and q['category_id']==ann['category_id']:same|=orig[q['id']];same640|=rasters[q['id']]
            neighbor=same&~own&valid;bg=~union&valid;other=union&~same&~own&valid
            base=dict(image_id=iid,annotation_id=aid,density=row['density'],area=float(row['area']),parts=int(row['parts']),source_index=int(old[aid]['source_index']))
            for arm,binary in binaries.items():
                measurement=original_measure(binary,shape,gt.annToRLE(ann),own,neighbor,bg,other,valid)
                if arm=='original':need(abs(measurement['coco_iou']-float(row['original_iou']))<1e-7,'Original IoU replay failed')
                if arm=='native640_free32_BCE':need(abs(measurement['coco_iou']-float(row['coco_iou']))<1e-7,'Free IoU replay failed')
                input_tp=int((binary.bool()&y.bool()).sum());input_fp=int((binary.bool()&~y.bool()).sum())
                image_metrics.append(dict(**base,arm=arm,**measurement,native_input_iou=input_tp/max(int(y.sum())+input_fp,1)))
            for arm,fit in fits.items():image_cal.append(dict(**base,arm=arm,**fit,free32_loss=after,threshold_native_oracle=tau))
            regions={'own':rasters[aid]&predsupport&valid640,'neighbor':same640&~rasters[aid]&predsupport&valid640,
                'background':~union640&predsupport&valid640,'other':union640&~same640&~rasters[aid]&predsupport&valid640}
            scores0={key:z[mask].detach().cpu().numpy() for key,mask in regions.items()}
            scoresf={key:zf[mask].detach().cpu().numpy() for key,mask in regions.items()}
            rankrow=dict(**base,**{key+'_pixels':len(values) for key,values in scores0.items()},
                native_label_pixels=int(y.sum()),fit_pixels=int(support.sum()),native_initial_loss=before,native_free_loss=after,
                logit_interpolation_max_error=float((z-zfit).abs().max()),free_interpolation_max_error=float((zf-zf_fit).abs().max()),
                original_coefficient_norm=float(c0.norm()),free_coefficient_norm=float(cf.norm()),coefficient_norm_ratio=float(cf.norm()/c0.norm()),
                saved_scaled_loss=float(row['loss_after']),scaled_reconstructed_loss=scaled_after,scaled_loss_replay_error=scaled_error,
                unscaled_loss_replay_error=abs(after-float(row['loss_after'])),
                double_decode_xor=int((double_binary!=free).sum()),norm_restored_xor=int((normalized_binary!=free).sum()))
            for domain in ['neighbor','background','other']:
                rankrow['original_auc_'+domain]=auc(scores0['own'],scores0[domain]);rankrow['free_auc_'+domain]=auc(scoresf['own'],scoresf[domain])
            # Best least-squares affine explanation of the changed response,
            # measured in response space; diagnostic only, never used to decode.
            x=z[predsupport&valid640].double().cpu().numpy();v=zf[predsupport&valid640].double().cpu().numpy()
            if len(x)>1:
                xc=x-x.mean();vc=v-v.mean();slope=float(xc@vc/max(xc@xc,1e-30));residual=vc-slope*xc
                rankrow.update(response_affine_slope=slope,response_affine_r2=1-float(residual@residual/max(vc@vc,1e-30)))
            else:rankrow.update(response_affine_slope=None,response_affine_r2=None)
            image_ranking.append(rankrow);saved[str(aid)]=dict(fits=fits,threshold=tau,source_index=int(old[aid]['source_index']))
        metrics.extend(image_metrics);ranking.extend(image_ranking);calibration.extend(image_cal)
        write_json(a.out/'images'/f'{iid}.json',dict(image_id=iid,metrics=image_metrics,ranking=image_ranking,calibration=image_cal,parameters=saved,
            input_hashes={key:sha(p) for key,p in paths.items()}))
        if number%10==0 or number==len(ids):
            progress=dict(images=number,total=len(ids),targets=len(ranking),seconds=time.monotonic()-started);write_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    save_csv(a.out/'metrics.csv',metrics);save_csv(a.out/'ranking.csv',ranking);save_csv(a.out/'calibration.csv',calibration)
    summary=[]
    for density in ['high','other']:
        for arm in protocol['arms']:
            rr=[r for r in metrics if r['density']==density and r['arm']==arm]
            if rr:summary.append(dict(density=density,arm=arm,n=len(rr),mean_coco_iou=100*float(np.mean([r['coco_iou'] for r in rr])),recovered75=sum(r['coco_iou']>=.75 for r in rr)))
    write_json(a.out/'ANALYSIS.json',dict(groups=summary,calibrations=len(calibration),optimizer_unsuccessful=sum(not r['success'] for r in calibration),
        initial_and_free_iou_all_replayed=True,native640_initial_loss_and_target_counts_all_replayed=True,
        free_scaled_loss_max_error=max(r['scaled_loss_replay_error'] for r in ranking),
        free_unscaled_loss_max_error=max(r['unscaled_loss_replay_error'] for r in ranking),
        double_decode_changed=sum(r['double_decode_xor']>0 for r in ranking),norm_restored_changed=sum(r['norm_restored_xor']>0 for r in ranking),scope=protocol['restrictions']))
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,targets=len(ranking),images=len(ids),seconds=time.monotonic()-started,
        hashes={str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()}))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
