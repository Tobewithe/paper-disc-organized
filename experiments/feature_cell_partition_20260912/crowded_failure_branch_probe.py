"""S049: explicitly selected crowded mask failures, c/P output factorial.

All selection depends on original GT geometry/failure, never treatment outcomes.
Same-budget background edits have stricter target-distance support than S047.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,gzip,io,json,shutil,time,hashlib
from pathlib import Path
from collections import Counter
import cv2,numpy as np,pandas as pd,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops
import ultralytics
from neighbor_background_probe import gt_geometry,inferred,decode,spatial,sha,dump,csvsave,rng

ROOT=Path(__file__).resolve().parent
CENSUS=ROOT/'diagnostics/mask_geometry_failure_census_20260912'
PRIOR=ROOT/'diagnostics/frozen_readouts_fullval_20260912_v2'
IMAGES=ROOT/'local_readout_runtime_20260912/data/images/val2017'
ANNOTATION=ROOT.parent.parent/'datasets/coco/annotations/instances_val2017.json'
WEIGHT=ROOT/'weights/yolo26m-seg.pt'
GROUPS={'same_failure':32,'same_success':16,'different_failure':16}


def placement(edit,bg,own,key,gain):
    ys,xs=np.nonzero(edit);y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    template=edit[y0:y1,x0:x1].astype('float32');score=cv2.matchTemplate(bg.astype('float32'),template,cv2.TM_CCORR)
    yy,xx=np.nonzero(score>=len(ys)-.25)
    if not len(yy):return None,'no_exact_shape_background'
    offsets=np.c_[ys-y0,xs-x0];oy,ox=np.nonzero(own);center=np.array([oy.mean(),ox.mean()]);ec=np.array([ys.mean(),xs.mean()])
    radial0=np.linalg.norm(ec-center);candidate_centers=np.c_[yy,xx]+offsets.mean(0)
    radial_diff=np.abs(np.linalg.norm(candidate_centers-center,axis=1)-radial0)*gain
    # Unlike S047, rank ALL possible placements before truncating computation.
    choice=np.argsort(radial_diff,kind='stable')[:256]
    distance=cv2.distanceTransform((~own).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*gain
    baseq=np.quantile(distance[ys,xs],[0,.25,.5,.75,1]);valid=[]
    for i in choice:
        coord=offsets+[yy[i],xx[i]]
        if not bg[coord[:,0],coord[:,1]].all():continue
        qq=np.quantile(distance[coord[:,0],coord[:,1]],[0,.25,.5,.75,1]);err=np.abs(qq-baseq)
        if radial_diff[i]<=8 and err[0]<=4 and max(err[1:4])<=8:
            valid.append((float(max(err[1:4])+radial_diff[i]),int(yy[i]),int(xx[i]),qq.tolist(),float(radial_diff[i])))
    if not valid:return None,'distance_matched_background_unavailable'
    valid.sort(key=lambda q:q[0]);best=valid[0];control=[best[1]-int(y0),best[2]-int(x0)]
    donororder=rng('S049_donor',key).permutation(len(yy));donors=[]
    for idx in donororder[:2048]:
        coord=offsets+[yy[idx],xx[idx]]
        if np.linalg.norm(np.array([yy[idx]-y0,xx[idx]-x0])-control)<8:continue
        if bg[coord[:,0],coord[:,1]].all():donors.append([int(yy[idx]-y0),int(xx[idx]-x0)])
        if len(donors)==3:break
    if not donors:return None,'no_separate_texture_donor'
    while len(donors)<3:donors.append(donors[0])
    return dict(control=control,donors=donors,all_template_positions=len(yy),distance_positions_checked=len(choice),
        target_distance_quantiles_input=baseq.tolist(),control_distance_quantiles_input=best[3],radial_mismatch_input=best[4]),None


def choose(gt,out):
    pools={'same_failure':'CROWDED_SUPPORTED_MASK_FAILURE_POOL.csv','same_success':'CROWDED_SUCCESS_POOL.csv','different_failure':'DIFFERENT_CROWDED_MASK_FAILURE_POOL.csv'}
    used=set();selected=[];rejected=[];counts={}
    for group,n in GROUPS.items():
        frame=pd.read_csv(CENSUS/pools[group]).sort_values('hash_rank');count=0
        for row in frame.to_dict('records'):
            if count>=n:break
            iid=int(row['image_id']);aid=int(row['annotation_id']);bid=int(row['different_neighbor' if group=='different_failure' else 'same_neighbor'])
            if iid in used:continue
            shape=(gt.imgs[iid]['height'],gt.imgs[iid]['width']);gain=min(640/shape[0],640/shape[1])
            own,neighbor,edit,union,crowd,bg,masks=gt_geometry(gt,iid,gt.anns[aid],gt.anns[bid])
            if edit.sum()<64 or edit.sum()/max(neighbor.sum(),1)<.2:
                rejected.append(dict(group=group,image_id=iid,target=aid,neighbor=bid,reason='edit_support'));continue
            pp,reason=placement(edit,bg,own,(iid,aid,bid),gain)
            if reason:
                rejected.append(dict(group=group,image_id=iid,target=aid,neighbor=bid,reason=reason));continue
            selected.append(dict(group=group,image_id=iid,target=aid,neighbor=bid,source_index=int(row['source_index']),
                prediction_slot=int(row['prediction_slot']),original_box_iou=row['box_iou'],original_mask_iou=row['mask_iou'],
                box_coverage_proxy=row['box_gt_pixel_coverage'],boundary_exposure=row['different_boundary_exposure4' if group=='different_failure' else 'same_boundary_exposure4'],
                same_exposure=row['same_boundary_exposure4'],ici=row['ici'],dominant_error=row['dominant_error'],
                category=int(row['category_id']),area=row['area'],edit_pixels=int(edit.sum()),neighbor_pixels=int(neighbor.sum()),
                edit_fraction=float(edit.sum()/neighbor.sum()),placement=pp,hash_rank=row['hash_rank']))
            used.add(iid);count+=1
        counts[group]=count
        print(json.dumps(dict(stage='SELECTING',group=group,selected=count,target=n,examined=count+sum(q['group']==group for q in rejected))),flush=True)
    dump(out/'manifest.json',dict(status='FROZEN_BEFORE_EDITED_INFERENCE',pairs=selected,counts=counts,rejected=rejected,
        rule='First GT/edit/background-feasible original-failure-hash instances,one/image; no treatment outcome selection; retain fewer if exhausted.'))
    return selected


def run(gt,out,pairs):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(WEIGHT));model.model.eval().requires_grad_(False);maps={};rows=[];witness=[];start=time.monotonic()
    for num,item in enumerate(pairs,1):
        iid,aid,bid=item['image_id'],item['target'],item['neighbor'];folder=out/'pairs'/str(iid);folder.mkdir()
        image=cv2.imread(str(IMAGES/gt.imgs[iid]['file_name']));shape=image.shape[:2]
        own,neighbor,edit,union,crowd,bg,masks=gt_geometry(gt,iid,gt.anns[aid],gt.anns[bid]);ys,xs=np.nonzero(edit)
        baseline=inferred(model,image,gt,iid,aid,bid,maps);source=item['source_index'];cap=baseline['cap'];raw=baseline['raw']
        if baseline['mapping'].get(aid,-1)<0 or int(baseline['keep'][baseline['mapping'][aid]])!=source:raise RuntimeError('Original target source mapping changed')
        c0=raw[0,84:,source];p0=cap['proto'];b0=ops.xywh2xyxy(raw[0,:4,source][None])[0]
        z0=F.interpolate((c0@p0.flatten(1)).reshape(1,1,*p0.shape[-2:]),(640,640),mode='bilinear',align_corners=False)[0,0]
        m0=decode(z0,b0,shape);old=spatial(m0,own,neighbor,union,crowd)
        if abs(old['mask_iou']-item['original_mask_iou'])>1e-10:raise RuntimeError('Original maskIoU replay changed')
        oldshard=json.loads(gzip.decompress((PRIOR/'images'/f'{iid}.json.gz').read_bytes()));j=item['prediction_slot']
        # S036 mask is RLE in original shape; verify actual same mask, not only IoU.
        from pycocotools import mask as mu
        if not np.array_equal(m0,mu.decode(oldshard['original'][j]['segmentation']).astype(bool)):raise RuntimeError('Fullmask replay changed')
        support=decode(torch.ones((640,640),device='cuda'),b0,shape);support_coverage=float((support&own).sum()/max(own.sum(),1))
        gain=min(640/shape[0],640/shape[1]);rh,rw=round(shape[0]*gain),round(shape[1]*gain);top,left=round((640-rh)/2-.1),round((640-rw)/2-.1)
        own640=np.zeros((640,640),bool);own640[top:top+rh,left:left+rw]=cv2.resize(own.astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
        if support_coverage<.95:
            witness.append(dict(image_id=iid,target=aid,status='EXACT_CROP_INELIGIBLE',coverage=support_coverage,proxy=item['box_coverage_proxy']));continue
        control=np.c_[ys,xs]+np.array(item['placement']['control']);ring=cv2.dilate(edit.astype('uint8'),np.ones((61,61),np.uint8)).astype(bool)&bg
        colors=image[ring] if ring.sum()>=32 else image[bg];mean=np.median(colors,axis=0);std=np.std(colors.astype(float),axis=0)
        variants=[('original','none',-1,image),('noop','none',-1,image.copy())]
        for seed in range(3):
            for fill in ['texture','local_color']:
                if fill=='texture':
                    shift=np.array(item['placement']['donors'][seed]);values=image[ys+shift[0],xs+shift[1]]
                else:
                    noise=rng('S049_color',iid,seed).normal(size=(*shape,3)).astype(np.float32);noise=cv2.GaussianBlur(noise,(0,0),4)
                    values=np.clip(mean+noise[ys,xs]*std*.25,0,255).astype('uint8')
                for mode,positions in [('neighbor',np.c_[ys,xs]),('background_control',control)]:
                    changed=image.copy();changed[positions[:,0],positions[:,1]]=values;variants.append((mode,fill,seed,changed))
        tensor={};pairrows=[];maxalgebra=0.
        np.savez_compressed(folder/'geometry.npz',own=own,neighbor=neighbor,edit=edit,background=bg,original_crop=support)
        for mode,fill,seed,changed in variants:
            if np.any(changed[own]!=image[own]):raise RuntimeError('Own original pixels changed')
            current=baseline if mode=='original' else inferred(model,changed,gt,iid,aid,bid,maps)
            inpdelta=float((current['input']-baseline['input']).abs()[0,:,own640].max())
            if inpdelta!=0:raise RuntimeError('Own transformed pixels changed')
            if mode=='noop' and (not torch.equal(current['raw'],raw) or not torch.equal(current['cap']['proto'],p0)):raise RuntimeError('Noop failed')
            c1=current['raw'][0,84:,source];p1=current['cap']['proto'];b1=ops.xywh2xyxy(current['raw'][0,:4,source][None])[0]
            tag=f'{mode}_{fill}_{seed}';tensor[tag+'_c']=c1.cpu().numpy();tensor[tag+'_P']=p1.cpu().numpy();tensor[tag+'_box']=b1.cpu().numpy()
            responses={'c0p0':c0@p0.flatten(1),'c1p0':c1@p0.flatten(1),'c0p1':c0@p1.flatten(1),'c1p1':c1@p1.flatten(1)}
            algebra=(c1-c0)@p0.flatten(1)+c0@(p1-p0).flatten(1)+(c1-c0)@(p1-p0).flatten(1)
            err=float((responses['c1p1']-responses['c0p0']-algebra).abs().max());maxalgebra=max(maxalgebra,err)
            # FP64 factorization witness, without assuming FP32 arithmetic is exact.
            dc=(c1-c0).double();dp=(p1-p0).double()
            exact=(c1.double()@p1.double().flatten(1)-c0.double()@p0.double().flatten(1))-((c1.double()-c0.double())@p0.double().flatten(1)+c0.double()@(p1.double()-p0.double()).flatten(1)+(c1.double()-c0.double())@(p1.double()-p0.double()).flatten(1))
            if float(exact.abs().max())>1e-9:raise RuntimeError('Response algebra inconsistent')
            fields=dict(group=item['group'],image_id=iid,target=aid,neighbor=bid,mode=mode,fill=fill,fill_seed=seed,
                source=source,original_box_iou=item['original_box_iou'],original_mask_iou=old['mask_iou'],exact_crop_coverage=support_coverage,
                exposure=item['boundary_exposure'],ici=item['ici'],dominant_error=item['dominant_error'],own_input_difference=inpdelta,
                source_kept=bool((current['keep']==source).any()),**{'normal_'+k:v for k,v in current['normal'].items()})
            for name,value in responses.items():
                z=F.interpolate(value.reshape(1,1,*p1.shape[-2:]),(640,640),mode='bilinear',align_corners=False)[0,0]
                m=decode(z,b0,shape);pairrows.append(dict(**fields,combination=name,**spatial(m,own,neighbor,union,crowd)))
                if name=='c1p1':
                    pairrows.append(dict(**fields,combination='c1p1_newbox',**spatial(decode(z,b1,shape),own,neighbor,union,crowd)))
            pairrows.append(dict(**fields,combination='c0p0_newbox',**spatial(decode(z0,b1,shape),own,neighbor,union,crowd)))
            if mode not in ['original','noop']:cv2.imwrite(str(folder/(tag+'.png')),changed)
        np.savez_compressed(folder/'branch_tensors.npz',**tensor);csvsave(folder/'metrics.csv',pairrows);rows.extend(pairrows)
        witness.append(dict(image_id=iid,target=aid,status='COMPLETE',coverage=support_coverage,originalIoUexact=True,originalmaskexact=True,sourceexact=True,
            own_input_difference=0,noop=True,fp64_algebra=True,max_fp32_algebra_error=maxalgebra))
        dump(out/'progress.json',dict(stage='INFERENCE',completed=num,total=len(pairs),seconds=round(time.monotonic()-start,2),pid=os.getpid()))
        print(json.dumps(dict(completed=num,total=len(pairs),group=item['group'],seconds=round(time.monotonic()-start,1))),flush=True)
        del baseline,current,tensor;torch.cuda.empty_cache()
    csvsave(out/'metrics.csv',rows);dump(out/'WITNESS.json',witness)
    dump(out/'COMPLETE.json',dict(status='COMPLETE',selected=len(pairs),evaluated=sum(q['status']=='COMPLETE' for q in witness),seconds=time.monotonic()-start,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False);(out/'pairs').mkdir()
    protocol=dict(experiment='S049_CROWDED_FAILURE_C_P_FACTORIAL',training=False,groups=GROUPS,seed_kind='3 fillseeds, one fixed model',
        census=str(CENSUS),census_sha256=sha(CENSUS/'instances.csv'),weight_sha256=sha(WEIGHT),script_sha256=sha(__file__),
        pools={p.name:sha(p) for p in CENSUS.glob('*POOL.csv')},helper_sha256=sha(ROOT/'neighbor_background_probe.py'),
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,vendor=ultralytics.__file__),
        selection='First hashed originalfailure/success targets inS048pools passingGTediting andshape/distancecontrol,one/image,32/16/16max, no treatmentbased expansion. All source/mask replay exact before editedinput; exactcropcoverage>=.95, failing cases retainedwitness and not replaced.',
        background='All templatepositions ranked byradialdistance BEFORE computing first256 candidates. Same shape/area; max radial mismatch8inputpixels; min targetdistance mismatch<=4; q25/q50/q75distance mismatches<=8. Record rejectioncounts. Upperbound search256 means feasibleposition maybemissed; notnaturalcounterfactual.',
        factorial='Fixedoriginalb0: c0P0,c1P0,c0P1,c1P1; additionallyc0P0/b1 andc1P1/b1. Identicalnetworkbasis, no coefficientoracle. Crossinput hybrids are branchoutput interventions and may be offmanifold, not universalcausality. Keep nonlinear metricinteraction and exactlogitidentity.',
        analysis='Perimage seedaverage neighbor-minus-background, twofilltypes separate. Samefailureprimary; success/differentfailure rangecontrols, category/geometrycompositionnotguaranteed. GTselectionconditioned explorationonval, notmethodAP. Need responseimprovement pluscontrols beforeclaiminganypathway.',
        predictions='SameofficialFP32one-to-many+NMS asS047; preserve rawsource; fullnormal targetmatches separately, manipulated neighbor excluded from allarms GT matching consistently.',
        stopping='Single declaredpilot; reportfewer controls if exhausted; no adaptivedoses or training. Nextmodulepatch not included.')
    dump(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANNOTATION))
    pairs=choose(gt,out);run(gt,out,pairs)


if __name__=='__main__':main()
