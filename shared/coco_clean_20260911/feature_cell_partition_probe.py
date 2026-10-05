"""S053: fixed P3 mixed-cell versus target-disjoint edit-cell interventions."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,io,json,shutil,time
import cv2,numpy as np,torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from ultralytics import YOLO
from ultralytics.utils import ops,LOGGER
import ultralytics
from neighbor_background_probe import inferred,sha,dump,csvsave,gt_geometry
from crowded_pixel_flow_probe import decode,encode
from feature_branch_probe import metrics,tensor_hash,input_mask
import crowded_failure_branch_probe as prior

BASE=Path(__file__).resolve().parent;SRC=BASE/'diagnostics/subtype_neighbor_20260912';OLD=BASE/'diagnostics/proto_feature_swap_20260912_v2'

def subset(mask,k,rng):
    flat=np.flatnonzero(mask);chosen=rng.permutation(flat)[:k];result=np.zeros(mask.shape,bool);result.flat[chosen]=True;return result

def prepare(out):
    pairs=[r for r in json.loads((SRC/'manifest.json').read_text())['pairs'] if r['error_type']=='same_neighbor'];frozen=[];all_masks={}
    for item in pairs:
        iid=item['image_id'];old=OLD/'pairs'/str(iid)/'feature_patch_masks.json';z=json.loads(old.read_text())['stages']['input0'];t=mu.decode(z['target']).astype(bool);n=mu.decode(z['neighbor']).astype(bool);b=mu.decode(z['background_control']).astype(bool)
        mixed=n&t;exclusive=n&~t;back=b&~t;k=min(int(mixed.sum()),int(exclusive.sum()),int(back.sum()))
        assert np.array_equal(mixed|exclusive,n) and not np.any(mixed&exclusive);assert k>0
        masks={('neighbor','all_edit',-1):n,('neighbor','mixed_all',-1):mixed,('neighbor','exclusive_all',-1):exclusive,('background_control','all_edit',-1):b}
        for seed in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([20260912,iid,seed]))
            masks[('neighbor','equal_mixed',seed)]=subset(mixed,k,rng);masks[('neighbor','equal_exclusive',seed)]=subset(exclusive,k,rng);masks[('background_control','equal_background',seed)]=subset(back,k,rng)
        folder=out/'pairs'/str(iid);folder.mkdir(parents=True)
        dump(folder/'PATCHES.json',dict(image_id=iid,source_sha256=sha(old),target=encode(t),k=k,patches=[dict(mode=m,region=r,selection_seed=s,cells=int(v.sum()),mask=encode(v)) for (m,r,s),v in masks.items()]))
        frozen.append(dict(**item,k=k,mixed_cells=int(mixed.sum()),exclusive_cells=int(exclusive.sum()),background_cells=int(back.sum()),source_grid_sha256=sha(old)));all_masks[iid]=masks
    dump(out/'manifest.json',dict(status='FROZEN_BEFORE_NEW_INFERENCE',pairs=frozen,selection='All15S052same_neighbor targets; no effect-based exclusions; k based on geometry only.'))
    return frozen,all_masks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
    protocol=dict(experiment='S053_P3_CELL_PARTITION',training=False,source_manifest_sha256=sha(SRC/'manifest.json'),source_S052_sha256=sha(OLD/'COMPLETE.json'),weight_sha256=sha(prior.WEIGHT),
        question='Does the reversible P3 edit-location response reside in target/edit mixed cells or edit cells disjoint from the target grid?',
        inputs='Same15same_neighbor targets; saved texture/local_color seed0; frozen original c0/b0. Each target has same5inputstates asS052.',
        geometry='Use exact saved S052P3masks: M=target intersect edit; E=edit minus target; B=background_control minus target. Disjoint grid cells do not imply disjoint receptivefields.',
        full_partition='AllM,allE,andMunionE; exact union must replayS052. Backgroundall replaysS052. Report interaction delta_union-delta_M-delta_E, not additive attribution.',
        count_control='k=min(countM,countE,countB) perimage; uniform without-replacement subsets, selection seeds0,1,2, frozen before inference. Same subsets for bothfills and directions. Average subset seeds withinimage before image bootstrap. Repeated masks where k=count are not independent replicates.',
        contrasts='Primary equal_mixed minus equal_exclusive, and each minus equal_background; insertion and restoration separately. Also actual changes versus original/edited. Same c0/b0; noGTcoefficient fitting or new candidate selection.',
        caveats='GT regions diagnostic; already explored15targets; only one fillseed pertype; pointwise2000imagebootstrap without multiple-comparison correction; equal cellcount not equal feature-change magnitude or spatial arrangement. Artificial hybridstates localize a carrier not origin.',
        stopping='One fixed15target run, no layer sweep or outcome-adaptive mask selection. Save allmasks plus localP3values. No training or automation changes.')
    dump(out/'protocol.json',protocol)
    for name in [Path(__file__).name,'feature_branch_probe.py','neighbor_background_probe.py','crowded_pixel_flow_probe.py','crowded_failure_branch_probe.py']:shutil.copy2(BASE/name,out/name)
    pairs,masks=prepare(out);source_hashes=json.loads((SRC/'COMPLETE.json').read_text())['hashes']
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(prior.ANNOTATION))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(prior.WEIGHT));model.model.eval().requires_grad_(False);enabled={'capture':True};capture={}
    def hook(module,args):
        if enabled['capture']:capture['inputs']=tuple(x.detach().clone() for x in args[0])
    handle=model.model.model[-1].proto.register_forward_pre_hook(hook)
    def natural(im,iid,aid,bid):
        capture.clear();enabled['capture']=True;state=inferred(model,im,gt,iid,aid,bid,{});inputs=capture['inputs'];enabled['capture']=False;return state,inputs
    def call(proto,inputs,donor,mask):
        return proto([torch.where(mask,donor,inputs[0]),inputs[1],inputs[2]])[0]
    rows=[];witness=[];start=time.monotonic();LOGGER.setLevel(40)
    with torch.inference_mode():
        for index,item in enumerate(pairs,1):
            iid,aid,bid=item['image_id'],item['target'],item['neighbor'];folder=out/'pairs'/str(iid);im=cv2.imread(str(prior.IMAGES/f'{iid:012}.jpg'));shape=im.shape[:2]
            own,n,_,union,crowd,_,_=gt_geometry(gt,iid,gt.anns[aid],gt.anns[bid]);geo=np.load(SRC/'pairs'/str(iid)/'geometry.npz');edit=geo['edit'];gain=min(640/shape[0],640/shape[1]);near=cv2.distanceTransform((~n).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*gain<=4
            regions=dict(own_near=own&near&~crowd,neighbor_untouched=n&~own&~edit&~crowd);base,x0=natural(im,iid,aid,bid);proto=model.predictor.model.model.model[-1].proto
            source=item['source_index'];assert int(base['keep'][item['prediction_slot']])==source;c0=base['raw'][0,84:,source];b0=ops.xywh2xyxy(base['raw'][0,:4,source][None])[0];p0=base['cap']['proto'];saved=np.load(SRC/'pairs'/str(iid)/'branch_tensors.npz')
            assert np.array_equal(c0.cpu().numpy(),saved['original_none_-1_c']);assert np.array_equal(p0.cpu().numpy(),saved['original_none_-1_P']);assert torch.equal(proto(list(x0))[0],p0)
            m0=decode(c0,p0,b0,shape);v0=metrics(m0,own,n,union,crowd,regions);assert abs(v0['mask_iou']-item['original_mask_iou'])<1e-10
            old_pack=json.loads((OLD/'pairs'/str(iid)/'masks.json').read_text());old={ (v['mode'],v['fill'],v['direction']):mu.decode(v['mask']).astype(bool) for v in old_pack['variants'] if v['stage']=='input0' and v['region']=='edit_location'}
            tgrid=mu.decode(json.loads((folder/'PATCHES.json').read_text())['target']).astype(bool);total=np.zeros((80,80),bool)
            for v in masks[iid].values():total|=v
            ys,xs=np.nonzero(total);tensor_pack=dict(coords_y=ys,coords_x=xs,original_p3=x0[0][0,:,ys,xs].cpu().numpy());outmasks=dict(image_id=iid,target=aid,original=encode(m0),variants=[]);patchstats=[];replays=0;noops=0
            for mode in ['neighbor','background_control']:
                for fill in ['texture','local_color']:
                    tag=f'{mode}_{fill}_0';png=SRC/'pairs'/str(iid)/(tag+'.png');assert sha(png)==source_hashes[str(png.relative_to(SRC))];changed=cv2.imread(str(png));assert np.array_equal(changed[own],im[own]);state,x1=natural(changed,iid,aid,bid);p1=state['cap']['proto'];assert np.array_equal(p1.cpu().numpy(),saved[tag+'_P']);assert torch.equal(proto(list(x1))[0],p1)
                    own_input=input_mask(own)[0,0].bool();assert float((state['input']-base['input']).abs()[0,:,own_input].max())==0
                    m1=decode(c0,p1,b0,shape);v1=metrics(m1,own,n,union,crowd,regions);tensor_pack[tag+'_p3']=x1[0][0,:,ys,xs].cpu().numpy();outmasks['variants'].append(dict(mode=mode,fill=fill,region='endpoint',selection_seed=-1,direction='edited',mask=encode(m1)))
                    for (m,region,seed),mask_cpu in masks[iid].items():
                        if m!=mode:continue
                        mask=torch.from_numpy(mask_cpu).cuda()[None,None];donor_change=x1[0]-x0[0];norm=float(donor_change[:,:,mask_cpu].norm());count=int(mask_cpu.sum())
                        patchstats.append(dict(mode=mode,fill=fill,region=region,selection_seed=seed,cells=count,target_overlap=int((mask_cpu&tgrid).sum()),delta_norm=norm,delta_rms=norm/max(count*256,1)**.5,original_p3_hash=tensor_hash(x0[0]),edited_p3_hash=tensor_hash(x1[0])))
                        assert torch.equal(call(proto,x0,x0[0],mask),p0);noops+=1
                        for direction,inputs,donor,ref in [('insert',x0,x1[0],v0),('restore',x1,x0[0],v1)]:
                            p=call(proto,inputs,donor,mask);mask_out=decode(c0,p,b0,shape);vals=metrics(mask_out,own,n,union,crowd,regions)
                            if region=='all_edit':assert np.array_equal(mask_out,old[(mode,fill,direction)]);replays+=1
                            rows.append(dict(image_id=iid,target=aid,error_type=item['error_type'],area_bin=item['area_bin'],mode=mode,fill=fill,fill_seed=0,region=region,selection_seed=seed,direction=direction,patch_cells=count,delta_norm=norm,original_iou=v0['mask_iou'],edited_iou=v1['mask_iou'],**vals,**{'delta_'+k:vals[k]-ref[k] for k in vals}))
                            outmasks['variants'].append(dict(mode=mode,fill=fill,region=region,selection_seed=seed,direction=direction,mask=encode(mask_out)))
                    del state,x1
            np.savez_compressed(folder/'p3_patch_values.npz',**tensor_pack);dump(folder/'masks.json',outmasks);dump(folder/'PATCH_STATS.json',patchstats)
            witness.append(dict(image_id=iid,k=item['k'],original_c_P_replay=True,edited_P_replays=4,own_input_difference=0,S052_local_mask_replays=replays,selfswaps=noops,partition_disjoint_and_complete=True))
            dump(out/'progress.json',dict(completed=index,total=len(pairs),pid=os.getpid(),seconds=round(time.monotonic()-start,1)));print(json.dumps(dict(completed=index,total=len(pairs),seconds=round(time.monotonic()-start,1))),flush=True)
            del base,x0,tensor_pack;saved.close();torch.cuda.empty_cache()
    handle.remove();csvsave(out/'metrics.csv',rows);dump(out/'WITNESS.json',witness);dump(out/'RUNTIME.json',dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(),stage='Proto26input0',shape=[1,256,80,80]))
    dump(out/'COMPLETE.json',dict(status='COMPLETE',images=len(pairs),rows=len(rows),seconds=round(time.monotonic()-start,3),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}));print('COMPLETE',len(pairs),len(rows),flush=True)
if __name__=='__main__':main()
