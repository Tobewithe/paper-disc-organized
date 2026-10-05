"""S033: where S032 label differences and saved-model pixel changes occur.

No fitting, new heads, thresholds, GT mask edits or AP recomputation. All 300
previous transfer images retained. Raw originalCOCO masks define spatial errors.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,shutil,time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from native_label_pipeline_probe import format_native,mask_of
from readout_input_probe import sha,write_json
from rich_pixel_readout import GlobalHead,instance_features
from run_rich_pixel_readout import read_np,csv_save,cuda,norm
from eval_readout_input_pilot import ici


def distances(mask):
    padded=np.pad(mask,1,constant_values=False)
    di=cv2.distanceTransform(padded.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    do=cv2.distanceTransform((~padded).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    return np.where(mask,di,do)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    src=a.source;out=a.out;out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_protocol=json.loads((src/'protocol.json').read_text());cache=Path(source_protocol['cache'])
    cr={k.replace('\\','/'):v for k,v in json.loads((cache/'COMPLETE.json').read_text())['hashes'].items()}
    sr={k.replace('\\','/'):v for k,v in json.loads((src/'COMPLETE.json').read_text())['hashes'].items()}
    images=json.loads((cache/'selection.json').read_text())['transfer']
    write_json(out/'protocol.json',dict(experiment='S033_BOUNDARY_AND_ERROR_FLOW',network_training=False,images=images,
        source=str(src.resolve()),cache=str(cache.resolve()),modes=['original','native_independent','raw_coco'],seeds=[0,1,2],
        labels='ALL native-retained ordinaryGT in same300transfer images: nativeoverlap, nativeindependent, rawCOCO nearest-exact640. '
               'Label differences measured at640 before crop. Decode savedheads only forfixedbbox50matched targets.',
        primary_boundary='Euclidean distance to nearest opposite-label pixel center <=2 INPUT640 pixels. '
               'Originalresolution flow uses originalmask distance multiplied by exactLetterBoxgain. Width1,4 and .02sqrt(GTarea) sensitivity.',
        label_partition='Differences split boundary<=2 first, then farther pixels covered by another ordinaryGT, then elsewhere. '
               'GT-sharing counts separate; observed labeloverlap is not necessarily physicalocclusion.',
        flow='Added/removed pixels partition ownGT, exclusive sameclassneighbor, otherclassGT, background; crowdmasked forflow. '
             'No GT selects prediction, threshold,parameter or subset. Pixel flow is descriptive,not causal attribution of AP.',
        comparisons=['raw_coco-original','raw_coco-native_independent'],
        statistics='All3savedseeds; image-cluster2000pointwiseCI; category+sizecommon-support density comparisons. '
            'NoAP gain claim from fixedmatches and no equalized-density causal inference.',
        sources=dict(source_receipt=sha(src/'COMPLETE.json'),cache_receipt=sha(cache/'COMPLETE.json'),script=sha(__file__))))
    shutil.copy2(__file__,out/Path(__file__).name)
    subset=cache/'conversion_input/instances_probe.json'
    if sha(subset)!=cr['conversion_input/instances_probe.json']:raise RuntimeError('Changed annotations')
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(subset))
    cfg=get_cfg(overrides=dict(task='segment',imgsz=640,mask_ratio=1,overlap_mask=True,rect=False,cache=False,workers=0,fraction=1.0))
    data=dict(names={k:gt.cats[c]['name'] for k,c in enumerate(sorted(gt.cats))},nc=80,channels=3)
    with contextlib.redirect_stdout(io.StringIO()):ds=build_yolo_dataset(cfg,str(cache/'converted/images/probe'),1,data,mode='val',rect=False)
    dsindex={int(Path(l['im_file']).stem):k for k,l in enumerate(ds.labels)}
    normalization=torch.load(src/'normalizer.pt',map_location='cuda',weights_only=True)
    heads={}
    for mode in ['native_independent','raw_coco']:
        for seed in [0,1,2]:
            path=src/f'{mode}_s{seed}/checkpoints/epoch015.pt';cp=json.loads((path.parent.parent/'COMPLETE.json').read_text())
            if sha(path)!=cp['final_sha256']:raise RuntimeError('Checkpoint changed')
            model=GlobalHead().cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);heads[mode,seed]=model.eval()
    with (src/'spatial.csv').open(encoding='utf-8') as f:
        previous={(r['arm'],int(r['annotation_id'])):r for r in csv.DictReader(f)}
    labelrows=[];flowrows=[];replayrows=[];empty=[];maximum_error=0.;zero_label_images=[]
    for ni,iid in enumerate(images,1):
        path=cache/'images'/f'{iid}.npz'
        if sha(path)!=cr[f'images/{iid}.npz']:raise RuntimeError('Changed cache')
        item=read_np(path);witness=json.loads((cache/'images'/f'{iid}.json').read_text())
        labelpath=cache/'converted/labels/probe'/f'{iid:012d}.txt'
        actual_label_hash=sha(labelpath) if labelpath.exists() else None
        if actual_label_hash!=witness['label_sha256']:raise RuntimeError('Changed TXT')
        if not witness['native_ids']:
            if len(item['annotation_ids']) or any(not ann.get('iscrowd',0) for ann in gt.imgToAnns[iid]):
                raise RuntimeError('Empty native labels but ordinary targets exist')
            zero_label_images.append(iid)
            continue
        lab=ds.labels[dsindex[iid]];raw=deepcopy(lab);raw.pop('shape',None);raw['img']=cv2.imread(lab['im_file'])
        shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']));h,w=shape
        if raw['img'].shape[:2]!=shape:raise RuntimeError('Geometry mismatch')
        raw=ds.update_labels_info(raw);raw['ori_shape']=shape;raw['ratio_pad']=(1.,1.)
        raw=LetterBox(new_shape=ishape,auto=False,scaleup=True)(raw)
        over=format_native(raw,witness['native_ids'],1,True);ind=format_native(raw,witness['native_ids'],1,False)
        gain=min(ishape[0]/h,ishape[1]/w);rh,rw=round(h*gain),round(w*gain)
        top=round((ishape[0]-rh)/2-.1);left=round((ishape[1]-rw)/2-.1)
        annos=gt.imgToAnns[iid];ordinary=[r for r in annos if not r.get('iscrowd',0)]
        masks={r['id']:gt.annToMask(r).astype(bool) for r in annos};crowd=np.zeros(shape,bool);union=np.zeros(shape,bool)
        count=np.zeros(ishape,np.int16);raster={}
        for ann in annos:
            if ann.get('iscrowd',0):crowd|=masks[ann['id']];continue
            union|=masks[ann['id']]
            big=torch.zeros(ishape,device='cuda',dtype=torch.bool)
            big[top:top+rh,left:left+rw]=F.interpolate(cuda(masks[ann['id']]).float()[None,None],(rh,rw),mode='nearest-exact')[0,0].bool()
            raster[ann['id']]=big.cpu().numpy();count+=raster[ann['id']]
        mapping={int(aid):(k,int(j)) for k,(aid,j) in enumerate(zip(item['annotation_ids'],item['prediction_indices']))}
        with torch.inference_mode():
            c=cuda(item['coeff']).float();p=cuda(item['proto']).float();boxes=cuda(item['boxes']).float()
            x=norm(instance_features(cuda(item['h']).float(),cuda(item['level']).long(),boxes,ishape),normalization)
            binary={('original',-1):ops.process_mask(p,c,boxes,ishape,upsample=True)}
            for key,head in heads.items():binary[key]=ops.process_mask(p,c+head(x),boxes,ishape,upsample=True)
        for ann in ordinary:
            aid=ann['id'];density=ici(ann,ordinary);group='low' if density<=1e-10 else 'middle' if density<=.5+1e-10 else 'high'
            base=dict(image_id=iid,annotation_id=aid,category_id=ann['category_id'],coco_area=ann['area'],
                size='small' if ann['area']<1024 else 'medium' if ann['area']<9216 else 'large',ici=density,density=group,
                matched=aid in mapping)
            if aid not in ind['position']:
                empty.append(dict(**base,status='native_label_removed'));continue
            yc=raster[aid];yi=mask_of(ind,aid).astype(bool);yo=mask_of(over,aid).astype(bool)
            dist=distances(yc);other=count-yc.astype(np.int16)>0;area640=int(yc.sum())
            differences={'ind_extra':yi&~yc,'ind_missing':yc&~yi,'overlap_deleted':yi&~yo}
            rr=dict(**base,raw_area640=area640,ind_area640=int(yi.sum()),overlap_area640=int(yo.sum()),raw_shared_pixels=int((yc&other).sum()))
            for name,region in differences.items():
                rr[name]=int(region.sum());rr[name+'_near1']=int((region&(dist<=1)).sum())
                rr[name+'_near2']=int((region&(dist<=2)).sum());rr[name+'_near4']=int((region&(dist<=4)).sum())
                rr[name+'_far2_othergt']=int((region&(dist>2)&other).sum());rr[name+'_far2_elsewhere']=int((region&(dist>2)&~other).sum())
                rr[name+'_sharedgt']=int((region&yc&other).sum())
                if rr[name] != rr[name+'_near2']+rr[name+'_far2_othergt']+rr[name+'_far2_elsewhere']:raise RuntimeError('Label partition incomplete')
            labelrows.append(rr)
            if aid not in mapping:continue
            k,j=mapping[aid];pos=item['sample_positions'][k]
            if not np.array_equal(yo.flatten()[pos],item['sample_y'][k]):raise RuntimeError('Cached labels differ')
            valid=~crowd;own=masks[aid]&valid;area=int(own.sum())
            if not area:
                empty.append(dict(**base,status='no_valid_pixels'));continue
            same=np.zeros(shape,bool);others=np.zeros(shape,bool)
            for aa in ordinary:
                if aa['id']!=aid:
                    others|=masks[aa['id']]
                    if aa['category_id']==ann['category_id']:same|=masks[aa['id']]
            regions={'own':own,'same':same&~own&valid,'other':union&~own&~same&valid,'bg':~union&valid}
            dd=distances(masks[aid]);distance_input=dd*gain;near2=distance_input<=2;near4=distance_input<=4
            normalized_near=dd<=.02*np.sqrt(area)
            keys=list(binary)
            with torch.inference_mode():pm=ops.scale_masks(torch.stack([binary[t][j] for t in keys])[:,None],shape)[:,0]>.5
            decoded={key:pm[z].cpu().numpy() for z,key in enumerate(keys)}
            input_masks={key:binary[key][j].cpu().numpy().astype(bool) for key in keys}
            for mode,seed in keys:
                pred=decoded[mode,seed];vp=pred&valid;un=int((pred|masks[aid]).sum())
                actual=dict(iou=int((pred&masks[aid]).sum())/un if un else 1.,coverage=int((vp&own).sum())/area,
                    neighbor=int((vp&regions['same']).sum())/area,background=int((vp&regions['bg']).sum())/area)
                old=previous[f'{mode}_s{seed}_d0',aid]
                err=max(abs(actual[m]-float(old[m])) for m in actual);maximum_error=max(maximum_error,err)
                if err>1e-12:raise RuntimeError('Prior spatial replay differs')
            for before in ['original','native_independent']:
                for seed in [0,1,2]:
                    bkey=(before,-1 if before=='original' else seed);nkey=('raw_coco',seed)
                    bp=decoded[bkey]&valid;npred=decoded[nkey]&valid
                    remove=bp&~npred;add=npred&~bp
                    row=dict(**base,comparison='raw_coco-'+before,seed=seed,valid_area=area,
                        before_area=int(bp.sum()),after_area=int(npred.sum()),removed=int(remove.sum()),added=int(add.sum()))
                    for direction,change in [('removed',remove),('added',add)]:
                        for name,region in regions.items():
                            q=change&region;prefix=direction+'_'+name
                            row[prefix]=int(q.sum());row[prefix+'_near2']=int((q&near2).sum());row[prefix+'_near4']=int((q&near4).sum())
                            row[prefix+'_near_normalized']=int((q&normalized_near).sum());row[prefix+'_far4']=int((q&~near4).sum())
                        if sum(row[direction+'_'+name] for name in regions)!=row[direction]:raise RuntimeError('Flow partition incomplete')
                    row['removed_own_shared']=int((remove&own&others).sum());row['added_own_shared']=int((add&own&others).sum())
                    bi=input_masks[bkey];nn=input_masks[nkey];rr640=bi&~nn;aa640=nn&~bi
                    row.update(removed640=int(rr640.sum()),added640=int(aa640.sum()),
                        removed640_ind_extra=int((rr640&differences['ind_extra']).sum()),
                        added640_ind_missing=int((aa640&differences['ind_missing']).sum()),
                        removed640_near2=int((rr640&(dist<=2)).sum()),added640_near2=int((aa640&(dist<=2)).sum()))
                    flowrows.append(row)
            replayrows.append(dict(**base,spatial_modes_replayed=len(keys),cached_label_xor=0))
        if ni%20==0:
            progress=dict(stage='flow',images=ni,total=len(images),targets=len(replayrows),seconds=time.monotonic()-start)
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for name,rows in [('labels',labelrows),('flows',flowrows),('replay',replayrows),('excluded',empty)]:csv_save(out/f'{name}.csv',rows)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(images),label_targets=len(labelrows),flow_targets=len(replayrows),
        seconds=time.monotonic()-start,maximum_spatial_replay_error=maximum_error,zero_ordinary_gt_images=zero_label_images,
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(dict(stage='COMPLETE',seconds=time.monotonic()-start,targets=len(replayrows))),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys,traceback
        if '--out' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--out')+1])
            if p.is_dir():write_json(p/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
