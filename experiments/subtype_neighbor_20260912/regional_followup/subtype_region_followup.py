"""S051 saved-output spatial followup: edited vs untouched neighbor and target."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,csv,contextlib,io,time,shutil
import numpy as np,cv2,torch
from pycocotools.coco import COCO
from crowded_pixel_flow_probe import decode,encode,sha,dump,save,distance_to,ANN

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();src=a.source.resolve();out=src/'regional_followup';out.mkdir(exist_ok=False);(out/'masks').mkdir()
    dump(out/'protocol.json',dict(experiment='S051_REGIONAL_FOLLOWUP',model_forward=False,training=False,source_receipt=sha(src/'COMPLETE.json'),
        scope='Exploratory spatial decomposition after aggregateS051 seen. Same44images, no new confirmation. Keepallinstances/fills/paths, no effect-based exclusion.',
        regions='own pixels <=4inputpixels fromchosenneighbor vs other own; chosenneighbor exclusive own edited vs unedited; allotherGT; background. Exclude crowd, partition valid. Original error receiver and original GT definitions maintained after editing.',
        metrics='Net correct pixel changes normalized by ownvalidGTarea; original FN recovered vs originalTP lost, originalFP removed vs added. allareas and originalerrorcounts stored. Originalmask and36variants RLEsaved; rawIoU replaysS051. Compare both neighbor-original and neighbor-background; notcausalmediation proof.'))
    for p in [Path(__file__),Path(__file__).parent/'crowded_pixel_flow_probe.py']:shutil.copy2(p,out/p.name)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANN))
    meta=json.loads((src/'manifest.json').read_text())['pairs'];validids={x['image_id'] for x in json.loads((src/'WITNESS.json').read_text()) if x['status']=='COMPLETE'}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rows=[];checks=[];start=time.monotonic()
    for item in meta:
        iid,aid,bid=item['image_id'],item['target'],item['neighbor']
        if iid not in validids:continue
        folder=src/'pairs'/str(iid);geo=np.load(folder/'geometry.npz');own=geo['own'];neighbor=geo['neighbor'];edit=geo['edit'];union=np.zeros(own.shape,bool);crowd=union.copy()
        for q in gt.imgToAnns[iid]:
            m=gt.annToMask(q).astype(bool);union|=m
            if q.get('iscrowd',0):crowd|=m
        valid=~crowd;area=int((own&valid).sum());near=distance_to(neighbor)*min(640/own.shape[0],640/own.shape[1])<=4
        regions=dict(own_near=own&near&valid,own_other=own&~near&valid,neighbor_edited=neighbor&~own&edit&valid,neighbor_untouched=neighbor&~own&~edit&valid,other_gt=union&~own&~neighbor&valid,background=~union&valid)
        assert np.array_equal(sum(v.astype('uint8') for v in regions.values()),valid.astype('uint8'))
        ref={(r['mode'],r['fill'],int(r['fill_seed']),r['combination']):r for r in csv.DictReader((folder/'metrics.csv').open(encoding='utf8'))}
        with np.load(folder/'branch_tensors.npz') as t:
            c0=torch.from_numpy(t['original_none_-1_c']).cuda();p0=torch.from_numpy(t['original_none_-1_P']).cuda();b0=torch.from_numpy(t['original_none_-1_box']).cuda();m0=decode(c0,p0,b0,own.shape)
            pack=dict(image_id=iid,target=aid,neighbor=bid,regions={k:encode(v) for k,v in regions.items()},original=encode(m0),variants=[])
            for mode in ['neighbor','background_control']:
                for fill in ['texture','local_color']:
                    for seed in range(3):
                        tag=f'{mode}_{fill}_{seed}';c1=torch.from_numpy(t[tag+'_c']).cuda();p1=torch.from_numpy(t[tag+'_P']).cuda()
                        for combo,c,p in [('c1p0',c1,p0),('c0p1',c0,p1),('c1p1',c1,p1)]:
                            m=decode(c,p,b0,own.shape);iou=(m&own).sum()/(m|own).sum();assert abs(iou-float(ref[(mode,fill,seed,combo)]['mask_iou']))<1e-10
                            add=m&~m0;remove=m0&~m;row=dict(image_id=iid,target=aid,neighbor=bid,error_type=item['error_type'],area_bin=item['area_bin'],mode=mode,fill=fill,fill_seed=seed,combination=combo,mask_iou=float(iou),original_iou=item['original_mask_iou'],delta_iou=float(iou-item['original_mask_iou']),own_area=area)
                            for name,reg in regions.items():
                                isown=name.startswith('own_');correct=int(((add if isown else remove)&reg).sum());harm=int(((remove if isown else add)&reg).sum());orig=int(((~m0 if isown else m0)&reg).sum())
                                row.update({name+'_area':int(reg.sum()),name+'_original_errors':orig,name+'_corrected':correct/area,name+'_harmed':harm/area,name+'_net':(correct-harm)/area})
                            rows.append(row);pack['variants'].append(dict(mode=mode,fill=fill,seed=seed,combination=combo,mask=encode(m)))
            dump(out/'masks'/f'{iid}.json',pack)
        checks.append(dict(image_id=iid,all36_iou_replayed=True,valid_partition=True))
    save(out/'regions.csv',rows);dump(out/'WITNESS.json',checks);dump(out/'COMPLETE.json',dict(status='COMPLETE',images=len(checks),rows=len(rows),seconds=round(time.monotonic()-start,3),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print('REGIONAL_COMPLETE',len(checks),len(rows),round(time.monotonic()-start,2))
if __name__=='__main__':main()
