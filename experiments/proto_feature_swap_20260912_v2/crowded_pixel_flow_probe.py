"""S050: saved S049 tensors -> exact spatial error flow; no model forward/train."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,csv,hashlib,shutil,time,contextlib,io
import cv2,numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics.utils import ops

BASE=Path(__file__).resolve().parent;SRC=BASE/'diagnostics/crowded_failure_branch_20260912'
ANN=BASE.parent.parent/'datasets/coco/annotations/instances_val2017.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def save(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def decode(c,p,b,shape):
    value=c@p.flatten(1);z=F.interpolate(value.reshape(1,1,*p.shape[-2:]),(640,640),mode='bilinear',align_corners=False)
    return (ops.scale_masks(ops.crop_mask((z[0]>0).byte(),b[None])[:,None],shape)[0,0]>.5).cpu().numpy()
def encode(m):
    r=mu.encode(np.asfortranarray(m.astype('uint8')));r['counts']=r['counts'].decode('ascii');return r
def distance_to(m):return cv2.distanceTransform((~m).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(exist_ok=False);(out/'masks').mkdir()
    protocol=dict(experiment='S050_SAVED_OUTPUT_PIXEL_FLOW',training=False,model_forward=False,source=str(SRC),source_receipt=sha(SRC/'COMPLETE.json'),source_analysis=sha(SRC/'ANALYSIS.json'),
        analysis_status='Exploratory followup to observed S049 aggregate; region rules frozen before S050 pixel readout. Not a new independent sample.',
        primary='All32same_failure; all63evaluated used without outcome exclusion. Original dominanterror,COCOarea small<1024/middle<9216/large,Box90 are subgroup descriptors, not new confirmatory endpoints.',
        regions='Valid~crowd: own interior beyond4inputpixels fromown8neighborboundary; own boundary band near chosenneighbor<=4inputpixels vsremaining own boundary; chosenneighbor exclusive of own; otherGT exclusive own/chosen; background within4inputpixels ofown vsfar. These partition valid pixels.',
        metrics='Corrected ownFN, harmed ownTP, removedFP, addedFP ineach region; normalizebyownvalidGTarea. Exact validIoU change decomposed additively with endpointdenominators; not causal module shares. Retain rawCOCOIoU separately.',
        contrasts='Threefillseedaverage perimage, neighbor-minus-distance-matched-background; texture/color separately, c1p0/c0p1/c1p1 originalboxfixed. Paired2000imagebootstrap, pointwise exploratory CI.',
        consistency='For each branch: pairedMaskIoU >1pp inBOTHfilltypes robustgain, <-1pp inBOTH robustharm, else mixed/small; reportallcounts. 1pp rule frozen now, not tuned. Cases forinspection may be response-selected and labelled.',
        stopping='One replay of storedtensors, no tuning/new training or model inference; reusable RLE/flows archived. No claim that pixel decomposition proves upstream origin.')
    dump(out/'protocol.json',protocol);shutil.copy2(__file__,out/Path(__file__).name)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANN))
    manifest=json.loads((SRC/'manifest.json').read_text());witness=json.loads((SRC/'WITNESS.json').read_text());validids={r['image_id'] for r in witness if r['status']=='COMPLETE'}
    receipt=json.loads((SRC/'COMPLETE.json').read_text())['hashes'];pairs=[r for r in manifest['pairs'] if r['image_id'] in validids]
    rows=[];checks=[];start=time.monotonic()
    for i,item in enumerate(pairs):
        iid,aid,bid=item['image_id'],item['target'],item['neighbor'];folder=SRC/'pairs'/str(iid);tp=folder/'branch_tensors.npz'
        assert sha(tp)==receipt[str(tp.relative_to(SRC))]
        masks={a['id']:gt.annToMask(a).astype(bool) for a in gt.imgToAnns[iid]};own=masks[aid];chosen=masks[bid];shape=own.shape;union=np.zeros(shape,bool);crowd=np.zeros(shape,bool)
        for a in gt.imgToAnns[iid]:
            union|=masks[a['id']]
            if a.get('iscrowd',0):crowd|=masks[a['id']]
        valid=~crowd;area=int((own&valid).sum());r=4/min(640/shape[0],640/shape[1]);boundary=own&~cv2.erode(own.astype('uint8'),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        band=distance_to(boundary)<=r;near=distance_to(chosen)<=r
        regions={'own_interior':own&~band&valid,'own_boundary_neighbor':own&band&near&valid,'own_boundary_other':own&band&~near&valid,
            'chosen_neighbor':chosen&~own&valid,'other_gt':union&~own&~chosen&valid,
            'background_near':~union&(distance_to(own)<=r)&valid,'background_far':~union&(distance_to(own)>r)&valid}
        assert np.array_equal(sum(x.astype('uint8') for x in regions.values()),valid.astype('uint8'))
        with np.load(tp) as tensors:
            c0=torch.from_numpy(tensors['original_none_-1_c']).cuda();p0=torch.from_numpy(tensors['original_none_-1_P']).cuda();b0=torch.from_numpy(tensors['original_none_-1_box']).cuda()
            base=decode(c0,p0,b0,shape);tp0=int((base&own&valid).sum());fp0=int((base&~own&valid).sum());iou0=tp0/(area+fp0);raw0=(base&own).sum()/(base|own).sum()
            old=list(csv.DictReader((folder/'metrics.csv').open(encoding='utf8')));ref={(v['mode'],v['fill'],int(v['fill_seed']),v['combination']):v for v in old}
            assert abs(raw0-item['original_mask_iou'])<1e-10
            pack=dict(image_id=iid,target=aid,neighbor=bid,regions={k:encode(v) for k,v in regions.items()},original=encode(base),variants=[])
            for mode in ['neighbor','background_control']:
                for fill in ['texture','local_color']:
                    for seed in range(3):
                        tag=f'{mode}_{fill}_{seed}';c1=torch.from_numpy(tensors[tag+'_c']).cuda();p1=torch.from_numpy(tensors[tag+'_P']).cuda()
                        for combination,c,p in [('c1p0',c1,p0),('c0p1',c0,p1),('c1p1',c1,p1)]:
                            m=decode(c,p,b0,shape);raw=(m&own).sum()/(m|own).sum();prior=ref[(mode,fill,seed,combination)]
                            assert abs(raw-float(prior['mask_iou']))<1e-10,('Decode changed',iid,tag,combination,raw,prior['mask_iou'])
                            tp1=int((m&own&valid).sum());fp1=int((m&~own&valid).sum());iou1=tp1/(area+fp1);added=m&~base;removed=base&~m
                            row=dict(group=item['group'],image_id=iid,target=aid,neighbor=bid,dominant_error=item['dominant_error'],area=item['area'],area_bin='small' if item['area']<1024 else ('middle' if item['area']<9216 else 'large'),box90=item['original_box_iou']>=.9,
                                mode=mode,fill=fill,fill_seed=seed,combination=combination,raw_iou_delta=float(raw-raw0),valid_iou_delta=iou1-iou0,raw_iou=float(raw),valid_iou=iou1,original_raw_iou=float(raw0),original_valid_iou=iou0,valid_gt_area=area)
                            terms=[]
                            for name,region in regions.items():
                                a=int((added&region).sum());d=int((removed&region).sum());regionarea=int(region.sum());isown=name.startswith('own_')
                                correct=a if isown else d;harm=d if isown else a
                                term=(a-d)/(area+fp1) if isown else -tp0*(a-d)/((area+fp0)*(area+fp1))
                                row.update({name+'_area':regionarea,name+'_corrected':correct/area,name+'_harmed':harm/area,name+'_net_corrected':(correct-harm)/area,name+'_iou_term':term})
                                terms.append(term)
                            assert abs(sum(terms)-(iou1-iou0))<1e-12
                            row['all_corrected']=sum(row[k+'_corrected'] for k in regions);row['all_harmed']=sum(row[k+'_harmed'] for k in regions)
                            rows.append(row);pack['variants'].append(dict(mode=mode,fill=fill,seed=seed,combination=combination,mask=encode(m)))
            dump(out/'masks'/f'{iid}.json',pack)
        checks.append(dict(image_id=iid,source_hash=True,all36decoder_replay=True,pixel_partition=True,iou_identity=True))
        if (i+1)%8==0:print(json.dumps(dict(completed=i+1,total=len(pairs),seconds=round(time.monotonic()-start,1))),flush=True)
    save(out/'pixel_flows.csv',rows);dump(out/'WITNESS.json',checks)
    dump(out/'COMPLETE.json',dict(status='COMPLETE',images=len(checks),rows=len(rows),seconds=round(time.monotonic()-start,3),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print('COMPLETE',len(checks),len(rows),round(time.monotonic()-start,2),flush=True)
if __name__=='__main__':main()
