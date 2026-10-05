"""Frozen official TAL Stage I. The only optimized tensors belong to adapters."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, r'D:\coco_wire\py')
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from pycocotools.coco import COCO
from pcdcr_models import build_adapter, parameter_count

def dump(p, x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def log(**x): print(json.dumps(x,ensure_ascii=False),flush=True)
def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)
def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def onto(x):
    return {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in x.items()}
def loss_image(x, delta):
    proto=F.interpolate(x['proto'][None],x['masks'].shape[-2:],mode='bilinear',align_corners=False)[0]
    c=x['c0']+delta
    target=(x['masks'][None]==(x['owners']+1)[:,None,None]).float()
    area=((x['target_boxes'][:,2:]-x['target_boxes'][:,:2])/640).prod(1)
    # Official sum over instances, then candidate mean outside each batch.
    return v8SegmentationLoss.single_mask_loss(target,c,proto,x['target_boxes'],area)*x['segmentation_gain']
def condition(x):
    # Formal ROI: input 640 grid, same bilinear proto interpolation and predicted
    # box support as native process_mask. GT quantities are deliberately unused.
    proto=F.interpolate(x['proto'][None],(640,640),mode='bilinear',align_corners=False)[0]
    values=[]; mins=[]; empty=0
    for c,box,lev in zip(x['c0'],x['boxes'],x['levels']):
        support=ops.crop_mask(torch.ones((1,640,640),device=c.device),box[None])[0].bool()
        p=proto[:,support].T
        if len(p):
            prob=(p@c).sigmoid(); w=torch.stack((prob,1-prob,4*prob*(1-prob)))
            sums=w.sum(1); means=(w@p)/(sums[:,None]+1e-6)
            response=(p.T@prob)/len(p); mins.append(float(sums.min()))
        else:
            means=torch.zeros((3,32),device=c.device); response=torch.zeros(32,device=c.device); mins.append(0.); empty+=1
        wh=(box[2:]-box[:2]).clamp_min(0)/640
        geo=torch.cat((wh,wh.prod().reshape(1),F.one_hot(lev,3).float()))
        values.append(torch.cat((means.flatten(),response,c,geo)))
    e=torch.stack(values); assert e.shape==(len(x['rows']),166) and torch.isfinite(e).all()
    return e,{'minimum_raw_weight_sum':min(mins),'minimum_denominator':min(mins)+1e-6,'empty_rois':empty}
def baseline_masks(x,coeff):
    padded=ops.process_mask(x['proto'],coeff,x['boxes'],(640,640),upsample=True)
    return ops.scale_masks(padded[:,None].float(),x['original_shape'],ratio_pad=x['ratio_pad'])[:,0]>.5
def original_gt(x,coco):
    return torch.stack([torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])]).astype(bool)) for r in x['rows']])
@torch.no_grad()
def quick_iou(images, adapter, coco, cache_gt):
    per=[]
    for raw in images:
        x=onto(raw); iid=int(x['rows'][0]['image_id'])
        if iid not in cache_gt: cache_gt[iid]=original_gt(raw,coco)
        gt=cache_gt[iid].cuda()
        delta=adapter(x['h'],x['condition'],x['levels']) if adapter is not None else torch.zeros_like(x['c0'])
        m=baseline_masks(x,x['c0']+delta)
        iou=(m&gt).flatten(1).sum(1)/(m|gt).flatten(1).sum(1).clamp_min(1)
        per.append(float(iou.mean()))
    return float(np.mean(per))

def prepare(cfg,out):
    cache=Path(cfg['cache']); prep=Path(cfg['prepared']); prep.mkdir(parents=True,exist_ok=True)
    assert ultralytics.__version__=='8.4.100',ultralytics.__version__
    assert sha(cfg['weights'])==cfg['weight_sha256']
    index=json.loads((cache/'INDEX.json').read_text(encoding='utf-8-sig'))
    assert sha(cache/'INDEX.json')==cfg['index_sha256']
    counts={s:[len(index[s]),sum(int(z['n']) for z in index[s])] for s in ('fit','dev','val')}
    assert counts=={'fit':[796,6058],'dev':[197,1402],'val':[196,1346]},counts
    sets={s:{int(z['image_id']) for z in index[s]} for s in counts}
    assert not any(sets[a]&sets[b] for a,b in [('fit','dev'),('fit','val'),('dev','val')])
    coco={s:COCO(cfg['annotations_val'] if s=='val' else cfg['annotations_train']) for s in counts}
    model=YOLO(cfg['weights']); model.model.eval().requires_grad_(False)
    head=model.model.model[-1]
    audit={'passed':True,'counts':counts,'source':str(cache),'ultralytics':ultralytics.__version__,
           'ultralytics_import_path':ultralytics.__file__,'weights_sha256':sha(cfg['weights']),
           'index_sha256':sha(cache/'INDEX.json'),'proto_head':str(type(head.proto)),
           'tower_shapes':[list(t[-1].weight.shape) for t in head.one2one_cv4],
           'source_file_hashes':{str(p):sha(p) for p in (Path(ops.__file__),Path(ultralytics.__file__).parent/'utils'/'loss.py',Path(ultralytics.__file__).parent/'nn'/'modules'/'head.py')},
           'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'checkpoint_train_args':model.ckpt['train_args'],
           'raw_gt_annotation_hashes':{s:sha(cfg['annotations_val'] if s=='val' else cfg['annotations_train']) for s in counts},
           'baseline_coeff_max_error':0.,'empty_predicted_rois':0,'parameter_counts':{a:parameter_count(build_adapter(a)) for a in ('static','concat','pcdcr')}}
    allrows=[]; t=time.monotonic(); n=0; condition_times=[]
    for split in counts:
        for item in index[split]:
            iid=int(item['image_id']); source=cache/'images'/f'{iid:012d}.pt'; raw=load(source)
            ids=torch.tensor([r['raw_id'] for r in raw['rows']],dtype=torch.long)
            for k,r in enumerate(raw['rows']):
                assert int(raw['owners'][k])==int(r['gt_index'])
                assert int(raw['all_annotation_ids'][int(r['gt_index'])])==int(r['annotation_id'])
                assert int(raw['levels'][int(r['raw_id'])])==int(r['level'])
                assert int(r['annotation_id']) in coco[split].anns
            x={k:raw[k] for k in ('proto','masks','owners','target_boxes','rows','original_shape','ratio_pad','segmentation_gain')}
            x.update(h=raw['h'][ids],c0=raw['coeff'][ids],boxes=raw['boxes'][ids],levels=raw['levels'][ids],split=split)
            for lev in range(3):
                take=x['levels']==lev
                if take.any():
                    layer=head.one2one_cv4[lev][-1]
                    c=F.linear(x['h'][take],layer.weight[:,:,0,0],layer.bias)
                    err=float((c-x['c0'][take]).abs().max()); audit['baseline_coeff_max_error']=max(audit['baseline_coeff_max_error'],err)
                    torch.testing.assert_close(c,x['c0'][take],atol=1e-5,rtol=1e-5)
            with torch.no_grad():
                xgpu=onto(x);torch.cuda.synchronize();tc=time.perf_counter();e,denoms=condition(xgpu);torch.cuda.synchronize();dt=time.perf_counter()-tc
            condition_times.append({'image_id':iid,'split':split,'candidates':len(x['rows']),'seconds':dt})
            x['condition']=e.cpu(); x['condition_denominator_min']=denoms['minimum_denominator']
            audit['empty_predicted_rois']+=denoms['empty_rois']
            for r in x['rows']:
                ann=coco[split].anns[int(r['annotation_id'])]
                r.update(class_id=int(ann['category_id']),coco_area=float(ann['area']),predicted_class_id=int(raw['class_logits'][:,int(r['raw_id'])].argmax()))
                allrows.append({'split':split,**r,'branch':'one2one'})
            torch.save(x,prep/f'{iid:012d}.pt'); n+=1
            if n%50==0: log(stage='prepare',images=n,elapsed_s=time.monotonic()-t)
    expected=json.loads(Path(cfg['official_manifest']).read_text(encoding='utf-8-sig'))['identity_list']
    key=lambda r:(r['split'],int(r['image_id']),int(r['annotation_id']),int(r['raw_id']),int(r.get('level',r.get('pyramid_level'))),int(r.get('gt_index',r.get('target_gt_idx'))))
    assert {key(r) for r in allrows}=={key(r) for rr in expected.values() for r in rr},'official manifest identity mismatch'
    audit['historical_official_identity_match']=True
    dump(out/'ASSET_MANIFEST.json',audit); dump(out/'OFFICIAL_MANIFEST.json',allrows)
    dump(prep/'INDEX.json',index)
    # Wrong-condition selection is deterministic and frozen before training.
    mismatch=[]; rng=np.random.default_rng(cfg['seed'])
    for split in counts:
        rr=[r for r in allrows if r['split']==split]
        for i,r in enumerate(rr):
            eligible=[j for j,s in enumerate(rr) if s['image_id']!=r['image_id'] and s['level']==r['level']]
            same=[j for j in eligible if rr[j]['predicted_class_id']==r['predicted_class_id']]
            pool=same if same else eligible
            # Geometry used for matching is the predicted box, never GT area.
            # Load once by split below to obtain the exact inference area.
            r['_eligible']=pool
        areas={}
        for it in index[split]:
            x=load(prep/f"{int(it['image_id']):012d}.pt")
            for r,b in zip(x['rows'],x['boxes']): areas[(r['image_id'],r['raw_id'])]=float(((b[2:]-b[:2]).clamp_min(0)).prod())
        for r in rr:
            pool=r.pop('_eligible'); a=max(areas[(r['image_id'],r['raw_id'])],1e-6)
            pool=sorted(pool,key=lambda j:abs(np.log(max(areas[(rr[j]['image_id'],rr[j]['raw_id'])],1e-6)/a)))
            # Fixed closest 5 geometry matches; seed chooses exactly once.
            j=int(rng.choice(pool[:5])); donor=rr[j]
            mismatch.append({'split':split,'image_id':r['image_id'],'raw_id':r['raw_id'],'donor_image_id':donor['image_id'],'donor_raw_id':donor['raw_id'],'same_predicted_category':r['predicted_class_id']==donor['predicted_class_id']})
    dump(out/'MISMATCH_MANIFEST.json',mismatch)
    dump(out/'CONDITION_LATENCY.json',{'unit':'seconds, synchronized GPU; includes proto upsample and full predicted-ROI pooling; excludes disk/H2D','per_image':condition_times,'total_seconds':sum(v['seconds'] for v in condition_times),'per_candidate_mean_ms':1000*sum(v['seconds'] for v in condition_times)/len(allrows)})
    dump(out/'PREPARE_COMPLETE.json',{'images':n,'candidates':len(allrows),'elapsed_s':time.monotonic()-t})
    del model
    return index,coco

def smoke(cfg,out,index,coco):
    started=time.monotonic(); prep=Path(cfg['prepared']); seed(cfg['seed'])
    historical={}
    for line in Path(cfg['historical_baseline']).read_text(encoding='utf-8-sig').splitlines():
        r=json.loads(line); historical[(r['split'],r['image_id'],r['annotation_id'],r['raw_id'])]=r['iou_A']
    weights_sha_before=sha(cfg['weights'])
    # Greedy coverage in a fixed image-hash order; uses only frozen baseline
    # status and metadata, never any trained adapter's result.
    ordered=sorted(index['fit'],key=lambda z:hashlib.sha256(str(z['image_id']).encode()).hexdigest())
    required={('level',0),('level',1),('level',2),('status',False),('status',True),('size','small'),('size','medium'),('size','large')}
    covered=set(); images=[]; chosen=set(); selection=[]
    for item in ordered:
        raw=load(prep/f"{int(item['image_id']):012d}.pt"); features=set()
        for r,lev in zip(raw['rows'],raw['levels']):
            key=('fit',int(r['image_id']),int(r['annotation_id']),int(r['raw_id']))
            size='small' if r['coco_area']<1024 else 'medium' if r['coco_area']<9216 else 'large'
            features.update((('level',int(lev)),('status',historical[key]>=.75),('size',size)))
        gained=(features & required)-covered
        if gained:
            images.append(raw);chosen.add(int(item['image_id']));covered.update(gained)
            selection.append({'image_id':int(item['image_id']),'new_coverage':sorted(f'{k}:{v}' for k,v in gained)})
        if covered==required: break
    assert covered==required and len(images)<=8,('smoke coverage unavailable',sorted(map(str,required-covered)),len(images))
    for item in ordered:
        if len(images)==8: break
        if int(item['image_id']) not in chosen:
            images.append(load(prep/f"{int(item['image_id']):012d}.pt"));chosen.add(int(item['image_id']))
            selection.append({'image_id':int(item['image_id']),'new_coverage':[]})
    assert len(images)==8
    report={'image_ids':[int(x['rows'][0]['image_id']) for x in images],'checks':{},'gradients':{},
            'selection_rule':'fixed image SHA256 order; greedy baseline level/status/size coverage, then fill to eight in same order',
            'selection':selection}
    input_audit={'tensor_checks':0,'keys':set()}
    def check_cached_inputs(x):
        for key,value in x.items():
            if torch.is_tensor(value):
                assert not value.requires_grad and value.grad_fn is None and value.grad is None,('cached input entered autograd',key)
                input_audit['tensor_checks']+=1;input_audit['keys'].add(key)
    model=YOLO(cfg['weights']).model.eval().requires_grad_(False)
    for p in model.parameters(): p.grad=None
    maxima={'coefficient_error':0.,'logit_error':0.,'mask_bit_difference':0,'historical_iou_max_error':0.}; details=[]; seen=set(); statuses=set(); sizes=set()
    for kind in ('static','concat','pcdcr'):
        seed(cfg['seed']); adapter=build_adapter(kind).cuda(); opt=torch.optim.AdamW(adapter.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        for raw in images:
            check_cached_inputs(raw);x=onto(raw);check_cached_inputs(x)
            seen.update(int(v) for v in x['levels']); d=adapter(x['h'],x['condition'],x['levels'])
            assert torch.count_nonzero(d)==0
            before=baseline_masks(x,x['c0']); after=baseline_masks(x,x['c0']+d)
            diff=int((before!=after).sum()); assert diff==0
            maxima['mask_bit_difference']+=diff
            delta_z=d@x['proto'].flatten(1); maxima['logit_error']=max(maxima['logit_error'],float(delta_z.abs().max()))
            gt=original_gt(raw,coco['fit']).cuda(); q=(before&gt).flatten(1).sum(1)/(before|gt).flatten(1).sum(1).clamp_min(1)
            for r,qi in zip(raw['rows'],q.tolist()):
                er=abs(qi-historical[('fit',int(r['image_id']),int(r['annotation_id']),int(r['raw_id']))]);maxima['historical_iou_max_error']=max(maxima['historical_iou_max_error'],er);assert er<1e-6,('baseline replay mismatch',r,er)
                statuses.add(qi>=.75); sizes.add('small' if r['coco_area']<1024 else 'medium' if r['coco_area']<9216 else 'large')
                if kind=='pcdcr': details.append({**r,'baseline_iou':qi,'zero_init_iou':qi,'mask_bit_difference':0})
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            for raw in images:
                x=onto(raw);check_cached_inputs(x);d=adapter(x['h'],x['condition'],x['levels'])
                (loss_image(x,d)/sum(len(a['rows']) for a in images)+cfg['lambda_delta']*d.square().sum()/sum(len(a['rows']) for a in images)).backward()
                check_cached_inputs(x)
            norms={n:float(p.grad.norm()) if p.grad is not None else None for n,p in adapter.named_parameters()}
            assert all(v is not None and np.isfinite(v) for v in norms.values())
            report['gradients'][kind+':'+str(step)]=norms
            if kind=='pcdcr' and step==2:
                for l in seen:
                    for part in ('.U.', '.V.', '.G.'):
                        assert sum(v for n,v in norms.items() if n.startswith('levels.'+str(l)+'.') and part in n)>0,(l,part,norms)
            opt.step()
        assert all(p.grad is None for p in model.parameters())
    weights_sha_after=sha(cfg['weights']);assert weights_sha_before==weights_sha_after,'original checkpoint changed during smoke'
    report['checks']={**maxima,'condition_finite':True,'denominator_epsilon':1e-6,'frozen_parameters_have_no_grad':True,
                      'cached_inputs_outside_autograd':{'passed':True,'tensor_checks':input_audit['tensor_checks'],'keys':sorted(input_audit['keys'])},
                      'original_model_role':'frozen baseline parameter audit; adapter-only training reads detached cached inputs',
                      'original_checkpoint_sha256_before':weights_sha_before,'original_checkpoint_sha256_after':weights_sha_after,
                      'original_checkpoint_unchanged':weights_sha_before==weights_sha_after,
                      'levels':sorted(seen),'baseline_states':sorted(statuses),'size_groups':sorted(sizes),'all_passed':True}
    assert seen=={0,1,2} and statuses=={True,False} and sizes=={'small','medium','large'},report
    elapsed=time.monotonic()-started
    # Conservative scaling of three arms x three updates x eight images;
    # includes zero-init decode, GT checks and model-loading overhead.
    estimate=elapsed/(3*3*8)*(3*cfg['epochs']*(796+197))*2
    report['resources']={'smoke_wall_s':elapsed,'formal_upper_estimate_hours':estimate/3600,'formal_limit_hours':cfg['max_training_hours'],'peak_vram_bytes':torch.cuda.max_memory_allocated()}
    assert estimate<cfg['max_training_hours']*3600,'estimated formal resource budget exceeded'
    dump(out/'SMOKE.json',report); dump(out/'SMOKE_PER_CANDIDATE.json',details)
    log(stage='smoke',passed=True)

def train(cfg,out,index,coco):
    prep=Path(cfg['prepared']); t=time.monotonic()
    fit=[load(prep/f"{int(i['image_id']):012d}.pt") for i in index['fit']]
    dev=[load(prep/f"{int(i['image_id']):012d}.pt") for i in index['dev']]
    gt={}; baseline=quick_iou(dev,None,coco['dev'],gt); log(stage='dev_baseline',macro_iou=baseline)
    for kind,arm in [('static','B'),('concat','C'),('pcdcr','D')]:
        seed(cfg['seed']); adapter=build_adapter(kind).cuda(); opt=torch.optim.AdamW(adapter.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr']*.1)
        run=out/arm;run.mkdir(exist_ok=True); best=baseline; best_epoch=0
        torch.save({'kind':kind,'epoch':0,'state_dict':{k:v.detach().cpu() for k,v in adapter.state_dict().items()},'dev_macro_iou':best},run/'best.pt')
        history=[]; torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,cfg['epochs']+1):
            order=np.random.default_rng(cfg['seed']+epoch).permutation(len(fit)); agg=0.; nr=0;te=time.monotonic()
            adapter.train()
            for start in range(0,len(order),cfg['batch_images']):
                if time.monotonic()-t>cfg['max_training_hours']*3600: raise RuntimeError('predefined formal resource limit reached')
                batch=[fit[int(j)] for j in order[start:start+cfg['batch_images']]]; nc=sum(len(z['rows']) for z in batch)
                opt.zero_grad(set_to_none=True)
                for raw in batch:
                    x=onto(raw); delta=adapter(x['h'],x['condition'],x['levels'])
                    loss=(loss_image(x,delta)+cfg['lambda_delta']*delta.square().sum())/nc
                    assert torch.isfinite(loss);loss.backward();agg+=float(loss.detach())*nc;nr+=len(x['rows'])
                opt.step()
            adapter.eval(); score=quick_iou(dev,adapter,coco['dev'],gt)
            if score>best:
                best=score;best_epoch=epoch;torch.save({'kind':kind,'epoch':epoch,'state_dict':{k:v.cpu() for k,v in adapter.state_dict().items()},'dev_macro_iou':best},run/'best.pt')
            record={'epoch':epoch,'train_objective_candidate_mean':agg/nr,'dev_macro_iou':score,'best_epoch':best_epoch,'best_dev':best,'lr':opt.param_groups[0]['lr'],'elapsed_s':time.monotonic()-te,'peak_vram_bytes':torch.cuda.max_memory_allocated()}
            history.append(record);dump(run/'TRAIN_HISTORY.json',history);scheduler.step();log(stage='train',arm=arm,**record)
        torch.save({'kind':kind,'epoch':cfg['epochs'],'state_dict':{k:v.cpu() for k,v in adapter.state_dict().items()}},run/'last.pt')
        dump(run/'SELECTION.json',{'best_epoch':best_epoch,'dev_macro_iou':best,'baseline_dev':baseline,'rule':'maximum normal original-image dev macro IoU; include zero-initialized epoch0, earliest tie','parameter_count':parameter_count(adapter)})
    dump(out/'TRAIN_COMPLETE.json',{'elapsed_s':time.monotonic()-t,'arms':['B','C','D']})

def evaluate(cfg,out,index,coco):
    from pcdcr_evaluation import evaluate_image,summarize
    prep=Path(cfg['prepared']); models={}
    for arm in ('B','C','D'):
        ck=load(out/arm/'best.pt'); m=build_adapter(ck['kind']).cuda();m.load_state_dict(ck['state_dict']);models[arm]=m.eval()
    maps=json.loads((out/'MISMATCH_MANIFEST.json').read_text(encoding='utf-8')); donor={(r['split'],r['image_id'],r['raw_id']):(r['donor_image_id'],r['donor_raw_id']) for r in maps}
    # Scalar logit bias is selected only on dev, using the identical finite support.
    sums={float(b):[] for b in cfg['scalar_bias_grid']}
    t=time.monotonic()
    with torch.no_grad():
        for it in index['dev']:
            raw=load(prep/f"{int(it['image_id']):012d}.pt");x=onto(raw);gt=original_gt(raw,coco['dev']).cuda()
            z=F.interpolate((x['c0']@x['proto'].flatten(1)).reshape(1,-1,*x['proto'].shape[-2:]),(640,640),mode='bilinear',align_corners=False)[0]
            official=baseline_masks(x,x['c0'])
            for b in sums:
                pad=ops.crop_mask((z+b).clone(),x['boxes'])>0
                pred=ops.scale_masks(pad[:,None].float(),x['original_shape'],ratio_pad=x['ratio_pad'])[:,0]>.5
                if b==0: assert torch.equal(pred,official),'zero bias baseline mismatch'
                vals=(pred&gt).flatten(1).sum(1)/(pred|gt).flatten(1).sum(1).clamp_min(1)
                sums[b].append(float(vals.mean()))
        scores={b:float(np.mean(v)) for b,v in sums.items()}; chosen=sorted(scores,key=lambda b:(-scores[b],abs(b),b))[0]
        dump(out/'SCALAR_BIAS_SELECTION.json',{'bias':chosen,'dev_scores':scores,'rule':'max dev image-macro original-image IoU; tie smallest abs bias then signed value','selected_before_val':True})
        rows=[]; latency={}
        for split in ('dev','val'):
            bank={}
            for it in index[split]:
                raw=load(prep/f"{int(it['image_id']):012d}.pt")
                for r,e in zip(raw['rows'],raw['condition']): bank[(r['image_id'],r['raw_id'])]=e
            for pos,it in enumerate(index[split]):
                raw=load(prep/f"{int(it['image_id']):012d}.pt");x=onto(raw)
                coeffs={'A':x['c0']}; gates={}; deltas={}
                for arm,m in models.items():
                    torch.cuda.synchronize();st=time.perf_counter();d,g=m(x['h'],x['condition'],x['levels'],return_gate=True);torch.cuda.synchronize()
                    latency.setdefault(arm,[]).append((time.perf_counter()-st)/len(x['rows']))
                    coeffs[arm]=x['c0']+d;deltas[arm]=d;gates[arm]=g
                wrong=torch.stack([bank[donor[(split,r['image_id'],r['raw_id'])]] for r in raw['rows']]).cuda()
                dm,gm=models['D'](x['h'],wrong,x['levels'],return_gate=True);coeffs['M']=x['c0']+dm;deltas['M']=dm;gates['M']=gm
                measured=evaluate_image(x,coeffs,coco[split],scalar_bias=chosen)
                for k,r in enumerate(measured):
                    r['c0']=x['c0'][k].tolist();r['condition_statistics']={'mean':float(x['condition'][k].mean()),'std':float(x['condition'][k].std()),'prototype_fg_norm':float(x['condition'][k,:32].norm()),'prototype_bg_norm':float(x['condition'][k,32:64].norm())}
                    for arm in deltas:r['delta_c_norm_'+arm]=float(deltas[arm][k].norm())
                    r['gate_D']=gates['D'][k].tolist();r['gate_M']=gm[k].tolist();r['gate_difference_norm']=float((gates['D'][k]-gm[k]).norm())
                    rows.append(r)
                if pos%25==0: log(stage='evaluate',split=split,images=pos+1,elapsed_s=time.monotonic()-t)
        summarize(rows,out,seed=cfg['seed'],bootstrap=5000)
        dump(out/'ADAPTER_LATENCY.json',{a:{'median_ms_per_candidate':1000*float(np.median(v)),'mean_ms_per_candidate':1000*float(np.mean(v)),'scope':'adapter only, excludes condition computation and backbone'} for a,v in latency.items()})
    dump(out/'COMPLETE.json',{'stage':'Stage I','status':'completed','rows':len(rows),'elapsed_eval_s':time.monotonic()-t,'stage_II_not_started':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','smoke','train','evaluate','finish','all'],default='all');a=ap.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;seed(cfg['seed'])
    assert torch.cuda.is_available(),'requires authorized laptop GPU'
    if a.stage in ('prepare','all'): index,coco=prepare(cfg,a.out)
    else:
        index=json.loads((Path(cfg['prepared'])/'INDEX.json').read_text(encoding='utf-8-sig'))
        coco={s:COCO(cfg['annotations_val'] if s=='val' else cfg['annotations_train']) for s in ('fit','dev','val')}
    if a.stage in ('smoke','finish','all'):smoke(cfg,a.out,index,coco)
    if a.stage in ('train','finish','all'):
        assert json.loads((a.out/'SMOKE.json').read_text(encoding='utf-8'))['checks']['all_passed']
        train(cfg,a.out,index,coco)
    if a.stage in ('evaluate','finish','all'):evaluate(cfg,a.out,index,coco)

if __name__=='__main__':main()
