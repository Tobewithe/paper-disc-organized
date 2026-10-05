"""S051: error/size stratified local-neighbor intervention; frozen before outcomes."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,gzip,io,json,shutil,time
from collections import Counter
import cv2,numpy as np,pandas as pd,torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
import ultralytics
import crowded_failure_branch_probe as old
from neighbor_background_probe import gt_geometry,sha,dump,csvsave

BASE=Path(__file__).resolve().parent
POOL=BASE/'diagnostics/crowded_pixel_flow_20260912/UNUSED_FAILURE_SUBTYPE_CANDIDATES.csv'
SIZES={'small':6,'medium':6,'large':3}
TYPES=['same_neighbor','target_fn','background']

def geometry(gt,iid,a,b):
    own,n,edit,union,crowd,bg,masks=gt_geometry(gt,iid,a,b)
    if not edit.any():return own,n,edit,union,crowd,bg,masks
    gain=min(640/own.shape[0],640/own.shape[1]);dist=cv2.distanceTransform((~own).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Original geometry only: closest editable neighbor pixel, row-major tie break.
    values=np.where(edit,dist,np.inf);cy,cx=np.unravel_index(values.argmin(),values.shape)
    yy,xx=np.ogrid[:own.shape[0],:own.shape[1]];edit=edit&(((yy-cy)**2+(xx-cx)**2)<=(16/gain)**2)
    return own,n,edit,union,crowd,bg,masks

def choose(gt,out):
    frame=pd.read_csv(POOL).sort_values('hash_rank');used=set();selected=[];rejected=[];counts={}
    for kind in TYPES:
        for size,quota in SIZES.items():
            group=kind+'__'+size;pool=frame[(frame.dominant_error==kind)&(frame.area_bin==size)];count=0
            for row in pool.to_dict('records'):
                if count>=quota:break
                iid,aid=int(row['image_id']),int(row['annotation_id']);reason=None
                if iid in used:
                    rejected.append(dict(group=group,image_id=iid,target=aid,reason='image_used_by_previous_stratum'));continue
                a=gt.anns[aid];own=gt.annToMask(a).astype(bool);gain=min(640/own.shape[0],640/own.shape[1]);bid=int(row['same_neighbor'])
                share=None;leak=0
                if kind=='same_neighbor':
                    shard=json.loads(gzip.decompress((old.PRIOR/'images'/f'{iid}.json.gz').read_bytes()));pred=mu.decode(shard['original'][int(row['prediction_slot'])]['segmentation']).astype(bool)
                    boundary=own&~cv2.erode(own.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                    crowd=np.zeros(own.shape,bool);same=np.zeros(own.shape,bool);candidates=[]
                    for b in gt.imgToAnns[iid]:
                        m=gt.annToMask(b).astype(bool)
                        if b.get('iscrowd',0):crowd|=m
                        elif b['id']!=aid and b['category_id']==a['category_id']:same|=m
                    denom=int((pred&same&~own&~crowd).sum())
                    for b in gt.imgToAnns[iid]:
                        if b.get('iscrowd',0) or b['id']==aid or b['category_id']!=a['category_id']:continue
                        m=gt.annToMask(b).astype(bool);overlap=(m&own).sum()/max(own.sum(),1)
                        dist=cv2.distanceTransform((~m).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*gain
                        exposure=(boundary&~crowd&(dist<=4)).sum()/max((boundary&~crowd).sum(),1);fp=int((pred&m&~own&~crowd).sum())
                        if overlap<=.01 and exposure>=.1:candidates.append((fp,exposure,-b['id']))
                    if not candidates:reason='no_close_low_overlap_leak_receiver'
                    else:
                        leak,_,negid=max(candidates);bid=-negid;share=leak/max(denom,1)
                        if leak<16 or share<.5:reason='selected_neighbor_not_major_leak_receiver'
                if reason:
                    rejected.append(dict(group=group,image_id=iid,target=aid,reason=reason));continue
                own,n,edit,union,crowd,bg,masks=geometry(gt,iid,a,gt.anns[bid])
                if edit.sum()<64 or edit.sum()*gain*gain<16:
                    rejected.append(dict(group=group,image_id=iid,target=aid,reason='local_edit_support'));continue
                pp,reason=old.placement(edit,bg,own,('S051',iid,aid,bid),gain)
                if reason:
                    rejected.append(dict(group=group,image_id=iid,target=aid,neighbor=bid,reason=reason));continue
                selected.append(dict(group=group,error_type=kind,area_bin=size,image_id=iid,target=aid,neighbor=bid,
                    source_index=int(row['source_index']),prediction_slot=int(row['prediction_slot']),original_box_iou=row['box_iou'],original_mask_iou=row['mask_iou'],
                    box_coverage_proxy=row['box_gt_pixel_coverage'],boundary_exposure=row['same_boundary_exposure4'],same_exposure=row['same_boundary_exposure4'],ici=row['ici'],dominant_error=kind,
                    category=int(row['category_id']),area=row['area'],edit_pixels=int(edit.sum()),neighbor_pixels=int(n.sum()),edit_fraction=float(edit.sum()/n.sum()),placement=pp,hash_rank=row['hash_rank'],
                    chosen_neighbor_leak_pixels=leak,chosen_neighbor_leak_share=share))
                used.add(iid);count+=1
            counts[group]=count
            print(json.dumps(dict(stage='SELECTION',group=group,selected=count,quota=quota,pool=len(pool),reasons=dict(Counter(r['reason'] for r in rejected if r['group']==group)))),flush=True)
    dump(out/'manifest.json',dict(status='FROZEN_BEFORE_EDITED_INFERENCE',pairs=selected,rejected=rejected,counts=counts,
        rule='No quota replacement across size or failure type; first hash feasible, one/image globally, predeclared 16-input-pixel-radius local edit. Leak receiver chosen from original errors, not treatment result.'))
    return selected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False);(out/'pairs').mkdir()
    protocol=dict(experiment='S051_ERROR_SIZE_STRATIFIED_LOCAL_NEIGHBOR',training=False,weight_sha256=sha(old.WEIGHT),pool_sha256=sha(POOL),pool=str(POOL),
        script_sha256=sha(__file__),runner_sha256=sha(Path(old.__file__)),helper_sha256=sha(BASE/'neighbor_background_probe.py'),
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,vendor=ultralytics.__file__),
        selection=dict(types=TYPES,quota_per_type=SIZES,one_image_once=True,exclude='All S049 selected image IDs excluded by S050 inventory; original val already explored. No treatment outcomes consulted for current selection.',
            leakage='For same_neighbor type choose max original false-positive receiver among same-class neighbors with boundary exposure>=.1 and overlap<=1%; >=16 erroneous original pixels and >=50% same-class union FP. Others use greatest original geometry neighbor.'),
        edit='Local neighbor exclusive area, protected original otherGT/crowd dilated2pixels. Disk radius16inputpixels centered at closest editable pixel to ownmask, row-major ties. Min64 original and16 input-equivalent pixels; no required whole-neighbor fraction. This is local removal, not whole-neighbor ablation.',
        controls='S049 exactshape/area background, radial<=8inputpixels, min distance mismatch<=4, q25/50/75<=8. Same texture/color values at paired sites,3seeds. No quota backfill or loosening after results.',
        primary='Per error type x size, neighbor-original AND neighbor-background MaskIoU/coverage/allneighbor+background errors, originalbox fixed c1p0,c0p1,c1p1. Normalprocess separately. Primary repair requires actual original improvement and paired advantage, same expected region direction in BOTH fills; pointwise bootstrap exploratory, no route declaration on CI alone.',
        stopping='One fixed selection/inference round max45targets. Fewer/empty strata reported. Exact actual crop>=95% and originalsource replay before edits; failures retained no replacement. No training or outcome-driven expansion.')
    dump(out/'protocol.json',protocol)
    for p in [Path(__file__),Path(old.__file__),BASE/'neighbor_background_probe.py']:shutil.copy2(p,out/p.name)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(old.ANNOTATION))
    pairs=choose(gt,out)
    if pairs:
        old.gt_geometry=geometry
        old.run(gt,out,pairs)
    else:dump(out/'COMPLETE.json',dict(status='COMPLETE_NO_ELIGIBLE',selected=0,evaluated=0,hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
if __name__=='__main__':main()
