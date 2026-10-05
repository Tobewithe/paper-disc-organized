"""B0: exhaustive, score-independent native-mask quality and parallel TAL/output replay.

No model training, oracle optimization, coefficient/box modification or AP.
All raw masks are decoded at normal resolution; no geometric or score pruning.
"""
import argparse, contextlib, hashlib, inspect, io, json, math, os, resource, time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.stats import chi2
from pycocotools.coco import COCO
import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as augment_module
from ultralytics.utils import ops, nms
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.tal import TaskAlignedAssigner
import official_pipeline

K, CONF, TAU = 300, .001, .75
WEIGHT_HASH = '16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576), b''): h.update(x)
    return h.hexdigest()

def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def serial(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x

def match(edges, subset=None):
    # Matrix is GT x physical raw, one raw has capacity one regardless of classes.
    e=edges if subset is None else edges[:,np.asarray(subset,dtype=int)]
    if not e.size: return 0,[]
    pairs=maximum_bipartite_matching(csr_matrix(e),perm_type='column')
    witness=[(i,int(j if subset is None else subset[j])) for i,j in enumerate(pairs) if j>=0]
    return min(K,len(witness)),witness

def matching_gate():
    tests={}
    for name,e,expected in [('one_raw_two_gt',[[1],[1]],1),('empty',[[0],[0]],0),('augmenting_path',[[1,1],[1,0]],2)]:
        got,_=match(np.array(e,bool)); assert got==expected,(name,got)
        tests[name]={'actual':got,'expected':expected}
    e=np.array([[1,1],[1,0]],bool)
    assert match(e,[0])[0]<=match(e)[0]
    return {'passed':True,'tests':tests,'tie_rule':'scipy CSR augmenting matching, sorted annotation_id/raw_id; witnesses are nonunique examples'}

def geometry_hash(sample, shape):
    g={'image_id':int(Path(sample['im_file']).stem),'original_shape':list(sample['ori_shape']),
       'input_shape':list(shape),'ratio_pad':sample['ratio_pad'],'preprocess':'native YOLODataset augment=False rect=False RGB FP32/255'}
    return hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest(),g

def summary(per,out):
    valid=[x for x in per if x['ordinary_gt']>0]
    n=len(valid); gt=sum(x['ordinary_gt'] for x in valid)
    keys=['raw_existence','R0','R1','R2','R3_conf','R3','R3_class','TAL','selection_gap','first_topk_gap','class_topk_gap','conf_gap','empty_mask_gap','class_gap','assignment_gap','raw_good_tal_bad']
    rng=np.random.default_rng(20260930); draws=rng.integers(0,n,(5000,n)) if n else None
    table={}
    for key in keys:
        vals=np.array([x[key]/x['ordinary_gt'] for x in valid]); counts=np.array([x[key] for x in valid]); den=np.array([x['ordinary_gt'] for x in valid])
        table[key]={'count':int(counts.sum()),'GT_weighted_ratio':float(counts.sum()/gt) if gt else None,
                    'image_macro_ratio':float(vals.mean()) if n else None,
                    'image_macro_ci95':np.quantile(vals[draws].mean(1),[.025,.975]).tolist() if n else None,
                    'GT_weighted_ci95':np.quantile(counts[draws].sum(1)/den[draws].sum(1),[.025,.975]).tolist() if n else None}
    gap=table['selection_gap']; ci=gap['image_macro_ci95']
    finding=('MATERIAL_GAP' if ci and ci[0]>=.02 else 'SMALL_GAP' if ci and ci[1]<.02 else 'INCONCLUSIVE')
    dump(out/'SUMMARY.json',{'validity':'VALID','finding':finding,'images':len(per),'images_with_ordinary_gt':n,'ordinary_gt':gt,
        'no_ordinary_gt_images':[x['image_id'] for x in per if not x['ordinary_gt']], 'metrics':table,
        'bootstrap':5000,'seed':20260930,'main':'image_macro selection coverage gap R0-R3',
        'precision_half_width':(ci[1]-ci[0])/2 if ci else None,'target_half_width':.01,'material_gap':.02,
        'scope':'exploratory reused COCO validation images; GT-assisted fixed-mask coverage, not COCO AR/AP',
        'tal_gts':sum(x['tal_gt_count'] for x in per),'no_tal_gts':sum(x['ordinary_gt']-x['tal_gt_count'] for x in per),
        'raw_good_tal_bad_with_final_alternative':sum(x['raw_good_tal_bad_with_final_alternative'] for x in per)})

def run(a):
    start=time.monotonic(); a.out.mkdir(parents=True,exist_ok=True); (a.out/'images').mkdir(exist_ok=True)
    assert ultralytics.__version__=='8.4.100'; assert sha(a.weights)==WEIGHT_HASH
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.manual_seed(20260930); np.random.seed(20260930)
    torch.cuda.set_per_process_memory_fraction(.8)
    dump(a.out/'MATCHING_GATE.json',matching_gate())
    manifest=json.loads(a.manifest.read_text(encoding='utf-8-sig')); ids=manifest['image_ids']
    with contextlib.redirect_stdout(io.StringIO()): coco=COCO(str(a.data/'annotations/instances_val2017.json'))
    w=YOLO(str(a.weights)); model=w.model.cuda().float().eval(); model.args=SimpleNamespace(**w.ckpt['train_args'])
    for p in model.parameters(): p.requires_grad_(False)
    head=model.model[-1]; assert head.end2end and not head.agnostic_nms
    head.max_det=K; cr=model.init_criterion(); assigner=cr.one2one.assigner
    # Reuse the official metadata-preserving converter, never private label assignment.
    convargs=SimpleNamespace(data=a.data,out=a.out)
    with contextlib.redirect_stdout(io.StringIO()): converted,sources,identity=official_pipeline.prepare_dataset(convargs,{'fit':[],'dev':[],'val':ids},w.names)
    ds=YOLODataset(img_path=str(converted/'val.txt'),imgsz=640,batch_size=1,augment=False,hyp=deepcopy(model.args),
         rect=False,cache=False,stride=32,data={'names':w.names,'nc':80,'channels':3},task='segment')
    assert {int(Path(p).stem) for p in ds.im_files}==set(ids)
    by_id={int(Path(p).stem):j for j,p in enumerate(ds.im_files)}
    category_by_class={i:c for i,c in enumerate(sorted(coco.cats))}; class_by_cat={v:k for k,v in category_by_class.items()}
    codefiles=[Path(inspect.getsourcefile(q)) for q in [type(head),TaskAlignedAssigner,ops.process_mask,nms.non_max_suppression,YOLODataset]]
    identity_meta={'checkpoint_path':str(a.weights),'checkpoint_sha256':sha(a.weights),'ultralytics':ultralytics.__version__,
        'python':os.sys.version,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'model_mode':'eval','dtype':'FP32','batch':1,
        'input':'640 square native YOLODataset letterbox','augmentation':False,'branch':'one2one','K':K,'conf':CONF,
        'retina_masks':False,'NMS':'absent on one2one end2end path','code':[{'path':str(p),'sha256':sha(p)} for p in dict.fromkeys(codefiles)],
        'B0_script_sha256':sha(__file__),'metadata_converter_sha256':sha(official_pipeline.__file__)}
    dump(a.out/'MODEL_AND_CODE_IDENTITY.json',identity_meta); dump(a.out/'IMAGE_MANIFEST.json',manifest)
    dump(a.out/'RAW_SCHEMA.json',{'RAW_CANDIDATES':'images/<iid>_raw.npz: complete raw_id/level/grid/coeff/boxes/class_logits/proto, geometry JSON contains hashes',
        'TAL_ASSIGNMENTS':'images/<iid>_tal.npz: fg_mask,target_gt_idx,target_scores, eligibility,pre_topk_positive, alignment_metric, overlap, annotation_ids',
        'OUTPUT_TRACE':'OUTPUT_TRACE.jsonl: each original output row, physical raw_id, class_id, score, stage membership; duplicated raw rows retained',
        'QUALITY':'images/<iid>_quality.npz: complete exact original-grid MaskIoU and BoxIoU matrices GT x raw; no score/class/box prefilter',
        'mask_threshold':TAU,'decoder':'native process_mask upsample=True; scale_masks real ratio_pad, bilinear, >0.5 on original grid',
        'capacity':'physical raw capacity one; K cap applied uniformly; all matching witnesses are illustrative'} )
    dump(a.out/'BRANCH_AND_ASSIGNER_AUDIT.json',{'branch':'one2one','native_criterion':type(cr.one2one).__name__,
        'assigner':type(assigner).__name__,'parameters':{k:getattr(assigner,k,None) for k in ['topk','topk2','alpha','beta','eps','num_classes']},
        'two_parallel_paths':['same frozen forward -> official TAL','same frozen forward -> native topk -> fixed conf -> native empty-mask filter'],
        'assignment_scope':'current frozen unaugmented snapshot; not pretraining history'})
    order={}; orig_raster=augment_module.polygons2masks_overlap
    def raster(*args,**kw):
        mask,idx=orig_raster(*args,**kw); order['idx']=idx.copy(); return mask,idx
    augment_module.polygons2masks_overlap=raster
    capture={}; orig_forward=assigner.forward; orig_pos=assigner.get_pos_mask
    def pos(*args,**kw):
        ret=orig_pos(*args,**kw); capture['pre_pos']=[x.clone() for x in ret]
        capture['eligible']=assigner.select_candidates_in_gts(args[4],args[3],args[5]).clone(); return ret
    def forward(*args,**kw):
        capture['inputs']=[x.clone() for x in args]; ret=orig_forward(*args,**kw); capture['ret']=[x.clone() for x in ret]; return ret
    assigner.get_pos_mask=pos; assigner.forward=forward
    all_image=[]; audits=[]; pergt=(a.out/'PER_GT_TRACE.jsonl').open('w'); perimg=(a.out/'PER_IMAGE_COVERAGE.jsonl').open('w')
    witness=(a.out/'MATCHING_WITNESSES.jsonl').open('w'); outtrace=(a.out/'OUTPUT_TRACE.jsonl').open('w')
    for num,iid in enumerate(ids,1):
      image_start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
      with torch.inference_mode():
        j=by_id[iid]; raw_label=ds.labels[j]; preids=[]
        for cls,poly in zip(raw_label['cls'].flatten(),raw_label['segments']):
            key=np.r_[cls,poly.flatten()].astype(np.float32); found=[aid for arr,aid in identity[iid] if np.array_equal(key,arr)]
            assert len(found)==1,(iid,found); preids.append(found[0])
        order.clear(); sample=ds[j]; annids=np.asarray(preids)[order['idx']].tolist() if preids else []
        assert len(annids)==len(sample['cls'])
        batch=YOLODataset.collate_fn([sample]); batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        x=batch['img'].float()/255; assert tuple(x.shape)==(1,3,640,640)
        geomhash,geom=geometry_hash(sample,x.shape[-2:]); (a.out/'images'/f'{iid:012d}_geometry.json').write_text(json.dumps(geom))
        inference,raw=model(x); p=raw['one2one']; proto=p['proto'][0]; coeff=p['mask_coefficient'][0].T
        boxes=head._get_decode_boxes(p)[0].T; scores=p['scores'].permute(0,2,1).sigmoid(); nraw=len(boxes)
        shapes=[list(f.shape[-2:]) for f in p['feats']]; levels=torch.cat([torch.full((h*ww,),l,device='cuda',dtype=torch.long) for l,(h,ww) in enumerate(shapes)])
        assert len(levels)==nraw and nraw==sum(h*ww for h,ww in shapes)
        grid=np.concatenate([np.stack(np.meshgrid(np.arange(ww),np.arange(h)),axis=-1).reshape(-1,2) for h,ww in shapes])
        cached_path=a.cache/'images'/f'{iid:012d}.pt'; cached=None; cacheerr={}
        if cached_path.exists():
            cached=torch.load(cached_path,map_location='cpu',weights_only=False)
            for key,val in [('proto',proto),('coeff',coeff),('boxes',boxes),('class_logits',p['scores'][0])]:
                cacheerr[key]=float((val.cpu()-cached[key]).abs().max()); torch.testing.assert_close(val.cpu(),cached[key],atol=1e-5,rtol=1e-5)
            assert tuple(sample['ori_shape'])==tuple(cached['original_shape']) and sample['ratio_pad']==cached['ratio_pad']
        capture.clear(); cr.one2one.get_assigned_targets_and_loss(p,batch)
        tl,tb,ts,fg,owner=capture['ret']; fg=fg[0].bool(); owner=owner[0].long(); assigned=torch.where(fg)[0].tolist()
        native_rows=[(rid,int(annids[int(owner[rid])])) for rid in assigned]
        if cached is not None: assert set(native_rows)=={(int(r['raw_id']),int(r['annotation_id'])) for r in cached['rows']}
        orig=sample['ori_shape']; ordinary=sorted([a for a in coco.imgToAnns[iid] if not a.get('iscrowd',0) and not a.get('ignore',0)],key=lambda a:a['id'])
        gm=np.stack([coco.annToMask(a) for a in ordinary]).astype(bool) if ordinary else np.zeros((0,*orig),bool)
        g=len(ordinary); areas=gm.sum((1,2)); assert not g or (areas>0).all(), 'empty ordinary mask'
        gtgpu=torch.from_numpy(gm.reshape(g,-1)).cuda().float(); gtboxes=torch.tensor([a['bbox'] for a in ordinary],device='cuda',dtype=torch.float32).reshape(-1,4)
        gtboxes[:,2:]+=gtboxes[:,:2]; ob=ops.scale_boxes((640,640),boxes.clone(),orig,ratio_pad=sample['ratio_pad'])
        b=box_iou(gtboxes,ob).cpu().numpy(); quality=np.empty((g,nraw),np.float32)
        first=scores.max(-1)[0].topk(min(K,nraw))[1][0].cpu().numpy(); tops,classes,rawids=head.get_topk_index(scores,K)
        rid=rawids[0,:,0].long(); cls=classes[0,:,0].long(); sc=tops[0,:,0]
        # This is exactly the model's native end2end head output, before conf filtering.
        formal=inference[0] if isinstance(inference,tuple) else inference
        assert tuple(formal.shape)==(1,len(rid),6+coeff.shape[1]),formal.shape
        expected=torch.cat([boxes[rid],sc[:,None],cls.float()[:,None],coeff[rid]],1)
        torch.testing.assert_close(formal[0],expected,atol=1e-5,rtol=1e-5)
        kept,indices=nms.non_max_suppression(formal,conf_thres=CONF,max_det=K,end2end=True,return_idxs=True)
        outids=indices[0]; nr=kept[0]
        native_mask=ops.process_mask(proto,nr[:,6:],nr[:,:4],(640,640),upsample=True) if len(nr) else torch.empty((0,640,640),device='cuda',dtype=torch.uint8)
        nonempty=native_mask.amax((-2,-1))>0 if len(nr) else torch.zeros(0,device='cuda',dtype=torch.bool)
        mask_orig=ops.scale_masks(native_mask[None],orig,ratio_pad=sample['ratio_pad'])[0]>.5
        formal_by_raw={}
        for k,r in enumerate(rid[outids].tolist()):
            if r in formal_by_raw: assert torch.equal(mask_orig[k],formal_by_raw[r]),'duplicate raw mask differed'
            formal_by_raw[r]=mask_orig[k]
        R1=sorted(set(first.tolist())); R2=sorted(set(rid.tolist())); Rconf=sorted(set(rid[outids].tolist())); R3=sorted(set(rid[outids[nonempty]].tolist()))
        assert set(R3)<=set(Rconf)<=set(R2)<=set(R1)<=set(range(nraw))
        mask_errors=0; max_pixel_diff=0; empty_raw=np.zeros(nraw,bool)
        for lo in range(0,nraw,a.chunk):
            ix=torch.arange(lo,min(lo+a.chunk,nraw),device='cuda'); decoded=ops.process_mask(proto,coeff[ix],boxes[ix],(640,640),upsample=True)
            empty_raw[lo:lo+len(ix)]=serial(decoded.amax((-2,-1))==0)
            m=ops.scale_masks(decoded[None],orig,ratio_pad=sample['ratio_pad'])[0]>.5
            flat=m.flatten(1).float(); inter=gtgpu@flat.T
            union=torch.as_tensor(areas,device='cuda')[:,None]+flat.sum(1)[None]-inter
            quality[:,lo:lo+len(ix)]=(inter/union.clamp_min(1)).cpu().numpy()
            for k,r in enumerate(ix.tolist()):
                if r in formal_by_raw:
                    diff=int((m[k]!=formal_by_raw[r]).sum()); mask_errors+=int(diff>0); max_pixel_diff=max(max_pixel_diff,diff)
        # Numerical identity is required before interpreting selection gaps.
        audit={'image_id':iid,'n_raw':nraw,'proto_shape':list(proto.shape),'feature_shapes':shapes,'cached_raw_maxerrors':cacheerr,
               'output_box_score_class_coeff_replay':True,'physical_output_rows':len(nr),'duplicate_output_rows':len(nr)-len(Rconf),
               'raw_to_formal_mask_changed_candidates':mask_errors,'max_changed_pixels':max_pixel_diff,'passed':mask_errors==0}
        audits.append(audit); dump(a.out/'OUTPUT_REPLAY_AUDIT.json',{'passed':all(v['passed'] for v in audits),'images':audits})
        if mask_errors: raise RuntimeError(f'output replay mask mismatch: {iid}, {mask_errors} candidates, {max_pixel_diff} pixels; metrics not interpreted')
        edges=quality>=TAU; counts={}; matches={}
        for name,subset in [('R0',None),('R1',R1),('R2',R2),('R3_conf',Rconf),('R3',R3)]: counts[name],matches[name]=match(edges,subset)
        assert counts['R0']>=counts['R1']>=counts['R2']>=counts['R3_conf']>=counts['R3']
        final_classes=defaultdict(set)
        for row,k in enumerate(outids.tolist()):
            rr=int(rid[k]); cc=int(cls[k]); passed_empty=bool(nonempty[row]);
            outtrace.write(json.dumps({'image_id':iid,'branch':'one2one','output_row_id':int(k),'final_output_row_id':row if passed_empty else None,
                'raw_id':rr,'class_id':cc,'category_id':category_by_class[cc],'score':float(sc[k]),'conf_pass':True,'empty_mask_pass':passed_empty})+'\n')
            if passed_empty: final_classes[rr].add(category_by_class[cc])
        # Preserve the R2 rows rejected by conf as well, without silently deduplicating.
        rejected=set(range(len(rid)))-set(outids.tolist())
        for k in sorted(rejected):
            outtrace.write(json.dumps({'image_id':iid,'branch':'one2one','output_row_id':k,'final_output_row_id':None,'raw_id':int(rid[k]),
                'class_id':int(cls[k]),'category_id':category_by_class[int(cls[k])],'score':float(sc[k]),'conf_pass':False,'empty_mask_pass':None})+'\n')
        classedges=edges.copy()
        for gi,ann in enumerate(ordinary): classedges[gi]&=np.array([ann['category_id'] in final_classes[r] for r in range(nraw)])
        counts['R3_class'],matches['R3_class']=match(classedges,R3)
        talmap=defaultdict(list)
        for rr,aid in native_rows: talmap[aid].append(rr)
        talgood=[]; rawgoodtalbad=0; finalalternative=0
        for gi,ann in enumerate(ordinary):
            aid=ann['id']; tr=talmap[aid]; exists=bool(edges[gi].any()); tq=max((float(quality[gi,r]) for r in tr),default=None)
            good=bool(tq is not None and tq>=TAU); talgood.append(good)
            rawgoodtalbad+=int(exists and not good)
            alt=any(edges[gi,r] and ann['category_id'] in final_classes[r] for r in R3); finalalternative+=int(exists and not good and alt)
            best=int(quality[gi].argmax()); pergt.write(json.dumps({'image_id':iid,'annotation_id':aid,'category_id':ann['category_id'],'iscrowd':0,
                'raw_mask75_exists':exists,'best_raw_id':best,'best_mask_iou':float(quality[gi,best]),'best_box_iou':float(b[gi,best]),
                'TAL_raw_ids':tr,'TAL_mask_iou':tq,'TAL_mask75':good,'has_TAL_positive':bool(tr),
                'GT_in_official_labels':aid in annids,'final_geometric_exists':bool(any(edges[gi,r] for r in R3)),
                'final_class_correct_exists':alt,'raw_good_TAL_bad':exists and not good,'best_raw_foreground':bool(fg[best]),
                'best_raw_owner_annotation_id':int(annids[int(owner[best])]) if fg[best] else None})+'\n')
        counts['TAL']=min(K,sum(talgood)); counts['raw_existence']=int(edges.any(1).sum())
        counts.update(selection_gap=counts['R0']-counts['R3'],first_topk_gap=counts['R0']-counts['R1'],class_topk_gap=counts['R1']-counts['R2'],
            conf_gap=counts['R2']-counts['R3_conf'],empty_mask_gap=counts['R3_conf']-counts['R3'],class_gap=counts['R3']-counts['R3_class'],
            assignment_gap=counts['R0']-counts['TAL'],raw_good_tal_bad=rawgoodtalbad)
        assert counts['assignment_gap']>=0
        for stage,mm in matches.items():
            witness.write(json.dumps({'image_id':iid,'stage':stage,'maximum_count_before_K':len(mm),'K':K,
                'witness_not_unique':True,'pairs':[{'annotation_id':ordinary[gi]['id'],'raw_id':rr,'mask_iou':float(quality[gi,rr])} for gi,rr in mm[:K]]})+'\n')
        entry={'image_id':iid,'ordinary_gt':g,'tal_gt_count':sum(bool(talmap[a['id']]) for a in ordinary),
              'raw_good_tal_bad_with_final_alternative':finalalternative,'n_raw':nraw,'output_rows':len(outids[nonempty]),'duplicate_output_rows':len(outids[nonempty])-len(R3),
              'input_geometry_hash':geomhash,**counts,'seconds':time.monotonic()-image_start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
        all_image.append(entry); perimg.write(json.dumps(entry)+'\n'); perimg.flush(); pergt.flush(); witness.flush(); outtrace.flush()
        # Keep a compact raw-candidate identity table for the full run.  The
        # complete coefficient/prototype tensors are not needed after this
        # image's exact quality matrix has been written and would exceed the
        # remote disk budget at 5,000 images.
        np.savez_compressed(a.out/'images'/f'{iid:012d}_raw_meta.npz',raw_id=np.arange(nraw),
            pyramid_level=serial(levels),grid_position=grid,boxes=serial(boxes),
            score_max=serial(scores.max(-1)[0]),predicted_class=serial(scores.argmax(-1)),
            checkpoint_hash=WEIGHT_HASH,input_geometry_hash=geomhash)
        np.savez_compressed(a.out/'images'/f'{iid:012d}_quality.npz',mask_iou=quality,box_iou=b,annotation_ids=[a['id'] for a in ordinary],
            first_topk_raw=np.array(R1),second_topk_raw=np.array(R2),conf_raw=np.array(Rconf),final_raw=np.array(R3),empty_mask=empty_raw)
        talarr={'fg_mask':serial(fg),'target_gt_idx':serial(owner),'target_scores':serial(ts[0]),'annotation_ids':annids}
        if annids:
            talarr.update(eligibility=serial(capture['eligible'][0]),pre_topk_positive=serial(capture['pre_pos'][0][0]),
                alignment_metric=serial(capture['pre_pos'][1][0]),overlap=serial(capture['pre_pos'][2][0]))
        np.savez_compressed(a.out/'images'/f'{iid:012d}_tal.npz',**talarr)
        elapsed=time.monotonic()-start; disk=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
        print(json.dumps({'images':num,'total':len(ids),'image_id':iid,'GT':g,'R0':counts['R0'],'R3':counts['R3'],'TAL':counts['TAL'],
                          'seconds':entry['seconds'],'elapsed_s':elapsed,'disk_bytes':disk}),flush=True)
        dump(a.out/'PROGRESS.json',{'images':num,'total':len(ids),'elapsed_s':elapsed,'disk_bytes':disk})
        if elapsed>a.hours*3600 or disk>50*1024**3: raise RuntimeError('budget exceeded; partial results retained')
      if cached is not None: del cached
    augment_module.polygons2masks_overlap=orig_raster; assigner.get_pos_mask=orig_pos; assigner.forward=orig_forward
    for f in [pergt,perimg,witness,outtrace]: f.close()
    summary(all_image,a.out)
    elapsed=time.monotonic()-start; disk=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
    pref={'completed_images':len(all_image),'seconds_total':elapsed,'per_image_seconds':[x['seconds'] for x in all_image],
          'peak_gpu_allocated_bytes':max(x['peak_gpu_allocated_bytes'] for x in all_image),'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory,
          'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'disk_bytes':disk,
          'exhaustive':True,'chunk':a.chunk,'precision':'FP32','no_pruning':True,'validity':'VALID'}
    if a.preflight:
        v=np.array([x['selection_gap']/x['ordinary_gt'] for x in all_image if x['ordinary_gt']])
        sd=float(v.std(ddof=1)); upper_sd=sd*math.sqrt((len(v)-1)/chi2.ppf(.05,len(v)-1))
        required=math.ceil((1.96*upper_sd/.01)**2)
        pref.update(point_sd=sd,upper_95_sd=upper_sd,required_images_precision=required,
                    planning='95% upper pilot SD; nominal normal-approximation CI half-width <=.01; not a significance stopping rule')
    dump(a.out/'RESOURCE_PREFLIGHT.json',pref); dump(a.out/'COMPLETE.json',{'validity':'VALID','images':len(all_image),'seconds':elapsed})

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['weights','data','cache','manifest','out']: ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--chunk',type=int,default=16); ap.add_argument('--hours',type=float,default=6); ap.add_argument('--preflight',action='store_true')
    run(ap.parse_args())
