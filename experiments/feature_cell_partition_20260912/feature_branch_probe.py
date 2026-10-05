"""S052: reversible feature swaps inside Proto26, all fixed S051 targets."""
import os,sys
from pathlib import Path
if os.name=='nt':os.environ['PATH']=str(Path(sys.prefix)/'Library/bin')+os.pathsep+os.environ.get('PATH','')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,contextlib,io,json,shutil,time,hashlib
import cv2,numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops,LOGGER
import ultralytics
from neighbor_background_probe import inferred,sha,dump,csvsave,gt_geometry
from crowded_pixel_flow_probe import decode,encode
import crowded_failure_branch_probe as prior

BASE=Path(__file__).resolve().parent;SRC=BASE/'diagnostics/subtype_neighbor_20260912';STAGES=['input0','input1','input2','fused','decoder']
ANCHORS={572408,82812,561465}

def metrics(m,own,n,union,crowd,regions):
    valid=~crowd;area=max(int((own&valid).sum()),1)
    return dict(mask_iou=float((m&own).sum()/max((m|own).sum(),1)),coverage=int((m&own&valid).sum())/area,
        neighbor_error=int((m&n&~own&valid).sum())/area,background_error=int((m&~union&valid).sum())/area,
        own_near_tp=int((m&regions['own_near']).sum())/area,neighbor_untouched_fp=int((m&regions['neighbor_untouched']).sum())/area)
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
def input_mask(mask):
    h,w=mask.shape;gain=min(640/h,640/w);rh,rw=round(h*gain),round(w*gain);top,left=round((640-rh)/2-.1),round((640-rw)/2-.1)
    result=np.zeros((640,640),np.uint8);result[top:top+rh,left:left+rw]=cv2.resize(mask.astype('uint8'),(rw,rh),interpolation=cv2.INTER_NEAREST_EXACT)
    return torch.from_numpy(result).cuda()[None,None].float()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False);(out/'pairs').mkdir()
    comp=json.loads((SRC/'COMPLETE.json').read_text());manifest=json.loads((SRC/'manifest.json').read_text());pairs=manifest['pairs']
    protocol=dict(experiment='S052_PROTO_FEATURE_REVERSIBLE_SWAP',training=False,source_manifest=sha(SRC/'manifest.json'),source_receipt=sha(SRC/'COMPLETE.json'),weight_sha256=sha(prior.WEIGHT),
        selection='All44S051pairs, no effect-based exclusion; existing44have now been explored. S051 predeclared strata kept. Anchors only for feature dumps/visuals, all44forstatistics.',
        input_variants='Original plus S051 saved neighbor/background_control, texture/local_color seed0 only. No seed average: one fixed fill seed in each type for bounded layer localization.',
        stages=dict(input0='Proto26 x[0],P3',input1='Proto26 x[1],P4',input2='Proto26 x[2],P5',fused='feat_fuse output, after multiscale sum and conv',decoder='cv2 output, after upsample and decoderconv before lastcv3'),
        patches='Target-own, actual edited-location, full spatialmap; masks from actual640letterbox nearest-exact then adaptive maxpool to layer resolution. Allchannels replaced. Pool cells may mix target/neighbor; save masks+overlap, do not claim disjointRF.',
        directions='insert: original prototype inputs plus edited donor stage; restore: edited inputs plus original donor stage. Coefficient c0 and originalboxb0 fixed forALLarms. Whole3inputswap must equaleditedP; noops exact.',
        endpoints='Both baseP and editedP decodedc0b0, same as S051c0p1, not c1p1. No normaldetector reranking for feature arms. Insert-minus-original and restore-minus-edited, plusmatched backgroundcontrast.',
        integrity='Original source/raw/P/mask and bothfillseed0 saved S051 c0p1 replay exact; own modelinput zero change; moduleplain replay equality; full-list swap endpoints and perstage selfswap exact.',
        interpretation='A stage carrying a reversible change does not identify where error originated. Sharedmultiscaleinputs/offmanifoldhybrids remain. Pointwise2000imagebootstrap exploratory,no multiplicitycorrection; wholeinputswap identity is acontrol not discovery.',
        stopping='Single44target round; no additional training, dose/layer selection after results, or automation changes.')
    dump(out/'protocol.json',protocol)
    for p in [Path(__file__),BASE/'neighbor_background_probe.py',BASE/'crowded_pixel_flow_probe.py']:shutil.copy2(p,out/p.name)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(prior.ANNOTATION))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=YOLO(str(prior.WEIGHT));model.model.eval().requires_grad_(False);proto=model.model.model[-1].proto
    if type(proto).__name__!='Proto26':raise RuntimeError('Expected actual Proto26')
    capture={};enabled={'capture':True};handles=[]
    def prehook(module,args):
        if enabled['capture']:capture['inputs']=tuple(x.detach().clone() for x in args[0])
    handles.append(proto.register_forward_pre_hook(prehook))
    for name,module in [('fused',proto.feat_fuse),('decoder',proto.cv2)]:
        def h(module,args,output,name=name):
            if enabled['capture']:capture[name]=output.detach().clone()
        handles.append(module.register_forward_hook(h))
    def natural(im,iid,aid,bid):
        capture.clear();enabled['capture']=True
        state=inferred(model,im,gt,iid,aid,bid,{})
        feats={f'input{i}':v for i,v in enumerate(capture['inputs'])};feats.update({k:capture[k] for k in ['fused','decoder']});enabled['capture']=False
        return state,feats
    def call(inputs,stage=None,donor=None,mask=None):
        x=list(inputs)
        if stage and stage.startswith('input'):
            index=int(stage[-1]);x[index]=torch.where(mask,donor,x[index]);return proto(x)[0]
        h=None
        if stage:
            module=proto.feat_fuse if stage=='fused' else proto.cv2
            h=module.register_forward_hook(lambda mod,args,output:torch.where(mask,donor,output))
        try:return proto(x)[0]
        finally:
            if h is not None:h.remove()
    LOGGER.setLevel(40)
    rows=[];checks=[];start=time.monotonic();runtime_written=False
    with torch.inference_mode():
        for num,item in enumerate(pairs,1):
            iid,aid,bid=item['image_id'],item['target'],item['neighbor'];folder=out/'pairs'/str(iid);folder.mkdir()
            image=cv2.imread(str(prior.IMAGES/f'{iid:012}.jpg'));shape=image.shape[:2]
            own,n,_,union,crowd,_,_=gt_geometry(gt,iid,gt.anns[aid],gt.anns[bid]);geo=np.load(SRC/'pairs'/str(iid)/'geometry.npz');edit=geo['edit'];ys,xs=np.nonzero(edit);shift=np.array(item['placement']['control']);control=np.zeros(shape,bool);control[ys+shift[0],xs+shift[1]]=True
            gain=min(640/shape[0],640/shape[1]);near=cv2.distanceTransform((~n).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*gain<=4
            regions=dict(own_near=own&near&~crowd,neighbor_untouched=n&~own&~edit&~crowd);spatial={'target':input_mask(own),'neighbor':input_mask(edit),'background_control':input_mask(control)}
            base,f0=natural(image,iid,aid,bid)
            # AutoBackend deep-copies/fuses the loaded model; direct submodule calls
            # must use that same active GPU inference instance.
            proto=model.predictor.model.model.model[-1].proto
            source=item['source_index'];c0=base['raw'][0,84:,source];b0=ops.xywh2xyxy(base['raw'][0,:4,source][None])[0];p0=base['cap']['proto'];x0=tuple(f0[f'input{i}'] for i in range(3))
            if int(base['keep'][item['prediction_slot']])!=source:raise RuntimeError('Original source changed')
            saved=np.load(SRC/'pairs'/str(iid)/'branch_tensors.npz');assert np.array_equal(c0.cpu().numpy(),saved['original_none_-1_c']);assert np.array_equal(p0.cpu().numpy(),saved['original_none_-1_P'])
            plain0=call(x0);assert torch.equal(plain0,p0);m0=decode(c0,p0,b0,shape);v0=metrics(m0,own,n,union,crowd,regions);assert abs(v0['mask_iou']-item['original_mask_iou'])<1e-10
            if not runtime_written:
                dump(out/'RUNTIME.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(),proto_type=type(proto).__name__,stages={k:list(v.shape) for k,v in f0.items()},source_model=str(prior.WEIGHT)));runtime_written=True
            bundle=dict(image_id=iid,target=aid,neighbor=bid,original=encode(m0),variants=[]);stored={};patchstats=[];noops=0
            if iid in ANCHORS:stored.update({'original_'+k:v.cpu().numpy() for k,v in f0.items()})
            for stage in STAGES:
                full=torch.ones((1,1,*f0[stage].shape[-2:]),device='cuda',dtype=torch.bool);assert torch.equal(call(x0,stage,f0[stage],full),p0);noops+=1
            for mode in ['neighbor','background_control']:
                for fill in ['texture','local_color']:
                    tag=f'{mode}_{fill}_0';png=SRC/'pairs'/str(iid)/(tag+'.png');assert sha(png)==comp['hashes'][str(png.relative_to(SRC))]
                    changed=cv2.imread(str(png));assert np.array_equal(changed[own],image[own]);state,f1=natural(changed,iid,aid,bid);p1=state['cap']['proto'];assert np.array_equal(p1.cpu().numpy(),saved[tag+'_P'])
                    diff=(state['input']-base['input']).abs();assert float(diff[0,:,spatial['target'][0,0].bool()].max())==0
                    x1=tuple(f1[f'input{i}'] for i in range(3));plain1=call(x1);assert torch.equal(plain1,p1)
                    m1=decode(c0,p1,b0,shape);v1=metrics(m1,own,n,union,crowd,regions)
                    if iid in ANCHORS:stored.update({tag+'_'+k:v.cpu().numpy() for k,v in f1.items()})
                    common=dict(image_id=iid,target=aid,error_type=item['error_type'],area_bin=item['area_bin'],mode=mode,fill=fill,seed=0,original_iou=v0['mask_iou'],edited_iou=v1['mask_iou'])
                    rows.append(dict(**common,stage='all_inputs',region='full',direction='insert',**v1,**{'delta_'+k:v1[k]-v0[k] for k in v0}))
                    rows.append(dict(**common,stage='all_inputs',region='full',direction='restore',**v0,**{'delta_'+k:v0[k]-v1[k] for k in v0}))
                    bundle['variants'].append(dict(mode=mode,fill=fill,stage='endpoint',region='full',direction='edited',mask=encode(m1)))
                    for stage in STAGES:
                        h,w=f0[stage].shape[-2:];target=F.adaptive_max_pool2d(spatial['target'],(h,w)).bool();location=F.adaptive_max_pool2d(spatial[mode],(h,w)).bool()
                        patchstats.append(dict(mode=mode,fill=fill,stage=stage,target_cells=int(target.sum()),edit_cells=int(location.sum()),overlap_cells=int((target&location).sum()),total_cells=h*w,donor_delta_norm=float((f1[stage]-f0[stage]).norm()),original_hash=tensor_hash(f0[stage]),edited_hash=tensor_hash(f1[stage])))
                        for region,mask in [('target',target),('edit_location',location),('full',torch.ones_like(target))]:
                            for direction,inputs,donor,reference in [('insert',x0,f1[stage],v0),('restore',x1,f0[stage],v1)]:
                                p=call(inputs,stage,donor,mask);m=decode(c0,p,b0,shape);v=metrics(m,own,n,union,crowd,regions)
                                rows.append(dict(**common,stage=stage,region=region,direction=direction,patch_cells=int(mask.sum()),**v,**{'delta_'+k:v[k]-reference[k] for k in v}))
                                bundle['variants'].append(dict(mode=mode,fill=fill,stage=stage,region=region,direction=direction,mask=encode(m)))
                    del state,f1
            dump(folder/'masks.json',bundle);dump(folder/'patch_geometry.json',patchstats)
            if stored:np.savez_compressed(folder/'anchor_features.npz',**stored)
            checks.append(dict(image_id=iid,original_and4variant_P_replay=True,own_input_difference=0,plain_module_replay=True,stage_selfswaps_exact=noops,full_input_endpoints_exact=True))
            dump(out/'progress.json',dict(completed=num,total=len(pairs),seconds=round(time.monotonic()-start,1),pid=os.getpid()));print(json.dumps(dict(completed=num,total=len(pairs),seconds=round(time.monotonic()-start,1))),flush=True)
            del base,f0,stored;saved.close();torch.cuda.empty_cache()
    for h in handles:h.remove()
    csvsave(out/'metrics.csv',rows);dump(out/'WITNESS.json',checks);dump(out/'COMPLETE.json',dict(status='COMPLETE',images=len(checks),rows=len(rows),seconds=round(time.monotonic()-start,3),hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print('COMPLETE',len(checks),len(rows),flush=True)
if __name__=='__main__':main()
