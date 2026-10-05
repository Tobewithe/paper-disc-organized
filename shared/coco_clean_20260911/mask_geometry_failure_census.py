"""S048: GT mask boundary geometry and SAME prediction box/mask failure census.

Reuses S036 final predictions without model inference. Never pairs an unrelated
box winner with a different mask winner. Not a newly trained method evaluation.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,csv,gzip,hashlib,io,json,shutil,time
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'diagnostics/frozen_readouts_fullval_20260912_v2'
ANNOTATION=ROOT.parent.parent/'datasets/coco/annotations/instances_val2017.json'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');t.replace(p)


def save_csv(p,rows):
    if not rows:return
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def box_overlap(a,b):
    x,y,w,h=a;u,v,ww,hh=b;inter=max(0,min(x+w,u+ww)-max(x,u))*max(0,min(y+h,v+hh)-max(y,v))
    return inter/max(w*h+ww*hh-inter,1e-12),inter/max(w*h,1e-12)


def match_boxes(gt,iid,pred):
    with contextlib.redirect_stdout(io.StringIO()):
        local=COCO();local.dataset=dict(info={},images=[gt.imgs[iid]],categories=list(gt.cats.values()),annotations=gt.imgToAnns[iid]);local.createIndex()
        if pred:dt=local.loadRes(pred)
        else:
            dt=COCO();dt.dataset=dict(images=local.dataset['images'],categories=local.dataset['categories'],annotations=[]);dt.createIndex()
        ev=COCOeval(local,dt,'bbox');ev.params.imgIds=[iid];ev.params.iouThrs=np.array([.5,.75,.9]);ev.params.areaRng=[[0,1e10]];ev.params.areaRngLbl=['all'];ev.params.maxDets=[100];ev.evaluate()
    mapping={};hits={}
    for item in ev.evalImgs:
        if item is None:continue
        for k,aid in enumerate(item['gtIds']):
            if item['gtIgnore'][k]:continue
            matches=item['gtMatches'][:,k];mapping[int(aid)]=int(matches[0])-1 if matches[0] else -1
            hits[int(aid)]=(bool(matches[1]),bool(matches[2]))
    return mapping,hits


def geometry(ann,anns,masks,boundaries,counts,classcounts,crowd,gain):
    aid=ann['id'];own=masks[aid];bd=boundaries[aid]&~crowd;ys,xs=np.nonzero(own)
    if not len(ys):return {'geometry_valid':False},None
    h,w=own.shape;rmax=8/gain;margin=int(np.ceil(rmax))+2
    y0=max(0,int(ys.min())-margin);y1=min(h,int(ys.max())+margin+1);x0=max(0,int(xs.min())-margin);x1=min(w,int(xs.max())+margin+1)
    sl=np.s_[y0:y1,x0:x1];ob=own[sl];valid=~crowd[sl];boundary=bd[sl];den=int(boundary.sum())
    same=(classcounts[ann['category_id']][sl]-ob.astype('int16'))>0
    different=(counts[sl]-classcounts[ann['category_id']][sl])>0
    result=dict(geometry_valid=den>0,boundary_pixels=den,gt_pixels=int(own.sum()),valid_gt_pixels=int((own&~crowd).sum()))
    def distance(m):
        return cv2.distanceTransform((~m).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if m.any() else np.full(m.shape,1e6,np.float32)
    for kind,other in [('same',same),('different',different),('any',same|different)]:
        other=other&valid;d=distance(other);exclusive=other&~ob;de=distance(exclusive)
        result[kind+'_overlap_fraction']=float((other&ob).sum()/max((ob&valid).sum(),1))
        for r in [2,4,8]:
            result[f'{kind}_boundary_exposure{r}']=float((d[boundary]<=r/gain).mean()) if den else 0.
        result[kind+'_exclusive_exposure4']=float((de[boundary]<=4/gain).mean()) if den else 0.
        # Relative object-size scale is diagnostic sensitivity, not selected by outcomes.
        rr=min(rmax,max(1/gain,.02*np.sqrt(own.sum())))
        result[kind+'_exposure_size_scale']=float((d[boundary]<=rr).mean()) if den else 0.
    neighbors=[]
    for other in anns:
        if other['id']==aid:continue
        ox,oy,ow,oh=other['bbox'];ax,ay,aw,ah=ann['bbox']
        if ox>x1 or oy>y1 or ox+ow<x0 or oy+oh<y0:continue
        om=masks[other['id']][sl]&valid
        if not om.any():continue
        d=distance(om);frac=float((d[boundary]<=4/gain).mean()) if den else 0.;frac8=float((d[boundary]<=8/gain).mean()) if den else 0.
        if frac8==0:continue
        neighbors.append(dict(neighbor=other['id'],same_class=other['category_id']==ann['category_id'],
            pair_exposure4=frac,pair_exposure8=frac8,pair_min_distance_input=float(d[ob&valid].min()*gain) if (ob&valid).any() else None,
            pair_overlap_fraction=float((ob&om).sum()/max((ob&valid).sum(),1))))
    selected={}
    for kind in ['same','different']:
        choices=[q for q in neighbors if q['same_class']==(kind=='same')]
        winner=min(choices,key=lambda q:(-q['pair_exposure4'],-q['pair_exposure8'],q['neighbor'])) if choices else None
        selected[kind]=winner
        for name in ['neighbor','pair_exposure4','pair_exposure8','pair_min_distance_input','pair_overlap_fraction']:
            result[kind+'_'+name]=winner[name] if winner else None
    return result,selected


def classify(box_iou,mask_iou):
    return ('box_good' if box_iou>=.75 else 'box_bad')+'__'+('mask_good' if mask_iou>=.75 else 'mask_bad')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out.resolve()
    out.mkdir(exist_ok=False);(out/'images').mkdir();start=time.monotonic();cv2.setNumThreads(4)
    protocol=dict(experiment='S048_MASK_GEOMETRY_FAILURE_CENSUS',training=False,model_inference=False,source=str(SOURCE),
        source_receipt_sha256=sha(SOURCE/'image_receipts.json'),annotation=str(ANNOTATION),annotation_sha256=sha(ANNOTATION),script_sha256=sha(__file__),
        population='All5000val images,36335 ordinaryGT; previously explored val, descriptive census/diagnostic selection, not pristine confirmation.',
        geometry='GT annToMask one annotation=one instance; same/different/all-other visible mask unions; one-pixel 8-neighbor erosion boundary; exclude crowd boundary pixels. Boundary exposure is fraction of target boundary within Euclidean r=2/4/8 MODEL-input-equivalent pixels of otherGT masks. Primaryr4. Record direct mask overlap separately plus exclusive-neighbor exposure and size-normalized sensitivity.',
        bins='Provisional operational primary same-class exposure4: low0 (<=1e-10), medium(0,.2), high>=.2. Not an established standard/new metric novelty claim. Preserve continuous values and2/8sensitivity, no effect-based threshold selection.',
        attribution='Original saved final slot boxes and masks paired by identical slot/class/score/source index. Official allGT sameclass bbox50 one-to-one assignment, percategory100. A matched SAME slot supplies bothboxIoU andmaskIoU. Separate officialMask75 and any-kept-goodMask availability prevent conflating independent winners and duplicate/suppressed candidates.',
        failure='Matched2x2 BoxIoU.75 x MaskIoU.75, plus no_bbox50_match; best retained sameclass geometry availability and samepred diagnostic clearly separate. No rawpreNMS claim. Mainmechanismtarget=boxIoU>=.75 & maskIoU<.75 & pixelGTcoveredbyfloatingoriginalbox>=.95 & officialMask75false & no sameclasskeptMask75; strictbox.90 flagged. Original-coordinate box raster is a geometricproxy, not exact decodedcropupperbound; exact GPU crop support mustbevalidated before mechanismselection.',
        errors='Fixed attribution full maskGT IoU, targetFN, exclusive same/otherGT falsepositive, background falsepositive, prediction fractional area. Error component ratios denominatorownvalidarea; crowd excluded from componentcounts but rawIoU uses fullCOCOmask.',
        selection='After census freeze GT-geometry/failure-class manifests BEFORE new interventions. Ranking uses hashID within predefined states, never postintervention benefit. No new cP mixing/GT solving in census.',
        scope='This is final-output failure identification. No final bbox50 match cannot distinguish absentrawcandidate/classification/NMS. Intermediate tensors require source replay. No evidence of absent correctcandidate priorNMS inferred.')
    dump(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    receipts=json.loads((SOURCE/'image_receipts.json').read_text());official={}
    with (SOURCE/'evaluation/original_gt.csv').open() as f:
        for r in csv.DictReader(f):official[int(r['annotation_id'])]=r
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANNOTATION))
    rows=[];witness=[]
    for number,iid in enumerate(sorted(gt.imgs),1):
        blob=(SOURCE/'images'/f'{iid}.json.gz').read_bytes()
        if hashlib.sha256(blob).hexdigest()!=receipts[str(iid)]['sha256']:raise RuntimeError('Cached image changed')
        shard=json.loads(gzip.decompress(blob));pp=shard['original'];boxes=shard['original_boxes'];src=shard['witness']['source_indices']
        if not len(pp)==len(boxes)==len(src):raise RuntimeError('Missing original slot mapping')
        for a,b in zip(pp,boxes):
            if a['category_id']!=b['category_id'] or a['score']!=b['score']:raise RuntimeError('Box/mask slot identity mismatch')
        anns=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)];shape=(gt.imgs[iid]['height'],gt.imgs[iid]['width']);gain=min(640/shape[0],640/shape[1])
        masks={q['id']:gt.annToMask(q).astype(bool) for q in gt.imgToAnns[iid]};crowd=np.zeros(shape,bool);counts=np.zeros(shape,np.int16);classcounts={};boundaries={}
        for q in gt.imgToAnns[iid]:
            m=masks[q['id']]
            if q.get('iscrowd',0):crowd|=m;continue
            counts+=m
            if q['category_id'] not in classcounts:classcounts[q['category_id']]=np.zeros(shape,np.int16)
            classcounts[q['category_id']]+=m
            er=cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            boundaries[q['id']]=m&~er
        mapping,hits=match_boxes(gt,iid,boxes)
        if set(mapping)!=set(q['id'] for q in anns):raise RuntimeError('Official bbox mapping denominator differs')
        gtrles=[gt.annToRLE(q) for q in anns]
        overlap=mu.iou([q['segmentation'] for q in pp],gtrles,[0]*len(anns)) if pp and anns else np.zeros((len(pp),len(anns)))
        decoded={};image_rows=[]
        for k,ann in enumerate(anns):
            aid=ann['id'];own=masks[aid];area=int(own.sum());valid=~crowd;va=int((own&valid).sum());j=mapping[aid]
            geo,selected=geometry(ann,anns,masks,boundaries,counts,classcounts,crowd,gain)
            samepred=[u for u,q in enumerate(pp) if q['category_id']==ann['category_id']]
            bi=[box_overlap(ann['bbox'],boxes[u]['bbox'])[0] for u in samepred]
            bestbox=max(bi,default=0.);bestmask=max((float(overlap[u,k]) for u in samepred),default=0.)
            bestboxslot=samepred[int(np.argmax(bi))] if samepred else -1
            ic=sum(box_overlap(ann['bbox'],q['bbox'])[1] for q in anns if q['id']!=aid and q['category_id']==ann['category_id'])
            old=official[aid]
            if abs(ic-float(old['ici']))>1e-8:raise RuntimeError('ICI replay mismatch')
            row=dict(image_id=iid,annotation_id=aid,category_id=ann['category_id'],area=ann['area'],area_bin=old['area_bin'],ici=ic,ici_high=ic>.5+1e-10,
                official_mask75=old['hit75']=='True',official_box75=hits[aid][0],official_box90=hits[aid][1],
                matched=j>=0,prediction_slot=j,source_index=src[j] if j>=0 else -1,
                best_kept_box_iou=bestbox,best_kept_mask_iou=bestmask,any_kept_mask75=bestmask>=.75,
                bestbox_slot=bestboxslot,bestbox_slot_mask_iou=float(overlap[bestboxslot,k]) if bestboxslot>=0 else 0.,
                **geo)
            if j>=0:
                if j not in decoded:decoded[j]=mu.decode(pp[j]['segmentation']).astype(bool)
                pred=decoded[j];box=boxes[j]['bbox'];biou=box_overlap(ann['bbox'],box)[0];miou=float(overlap[j,k]);px,py=np.nonzero(own)[1],np.nonzero(own)[0]
                support=((px>=box[0])&(px<box[0]+box[2])&(py>=box[1])&(py<box[1]+box[3]))
                cover=float(support.mean()) if area else 0.;pv=pred&valid;ov=own&valid;fp=pv&~own
                same=(classcounts[ann['category_id']]-own.astype('int16'))>0;diff=(counts-classcounts[ann['category_id']])>0
                both=same&diff;near=int((fp&same).sum());otheronly=int((fp&diff&~same).sum());bg=int((fp&(counts==0)).sum());tp=int((pv&ov).sum());fn=int((~pv&ov).sum())
                if near+otheronly+bg!=int(fp.sum()):raise RuntimeError('FP partition does not add up')
                if abs(miou-int((pred&own).sum())/max(int((pred|own).sum()),1))>1e-10:raise RuntimeError('RLE/fullmaskIoU mismatch')
                state=classify(biou,miou)
                row.update(box_iou=biou,mask_iou=miou,box_gt_pixel_coverage=cover,state=state,score=boxes[j]['score'],
                    coverage=tp/max(va,1),same_neighbor_error=near/max(va,1),other_only_error=otheronly/max(va,1),background_error=bg/max(va,1),
                    target_fn=fn/max(va,1),pred_area=int(pred.sum()),same_other_gt_overlap_fp=int((fp&both).sum()),
                    support_sufficient=cover>=.95,strict_box90=biou>=.9,
                    unrescued_mask_failure=biou>=.75 and miou<.75 and cover>=.95 and not row['official_mask75'] and bestmask<.75,
                    dominant_error=max([('target_fn',fn),('same_neighbor',near),('other_neighbor',otheronly),('background',bg)],key=lambda q:q[1])[0])
            else:
                row.update(box_iou=None,mask_iou=None,box_gt_pixel_coverage=None,state='no_bbox50_match',score=None,
                    coverage=None,same_neighbor_error=None,other_only_error=None,background_error=None,target_fn=None,pred_area=None,same_other_gt_overlap_fp=None,
                    support_sufficient=False,strict_box90=False,unrescued_mask_failure=False,dominant_error=None)
            exposure=row.get('same_boundary_exposure4',0.)
            row['mask_density']='high' if exposure>=.2 else 'middle' if exposure>1e-10 else 'low'
            image_rows.append(row);rows.append(row)
        dump(out/'images'/f'{iid}.json',image_rows)
        witness.append(dict(image_id=iid,gt=len(anns),predictions=len(pp),shard_hash=True,slot_identity=True,rle_replay=True))
        if number%100==0 or number==5000:
            progress=dict(stage='CENSUS',completed=number,total=5000,gt=len(rows),seconds=round(time.monotonic()-start,2),pid=os.getpid());dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    if len(rows)!=36335 or {q['annotation_id'] for q in rows}!=set(official):raise RuntimeError('AllGT denominator mismatch')
    save_csv(out/'instances.csv',rows);dump(out/'WITNESS.json',witness)
    dump(out/'COMPLETE.json',dict(status='COMPLETE',images=5000,gt=len(rows),seconds=time.monotonic()-start,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}))


if __name__=='__main__':main()
