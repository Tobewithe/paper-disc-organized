"""Full original predictions, masks, fixed-pair topology and exact success census."""
from lcf_runtime import *
from collections import defaultdict
import contextlib
NORMAL=('N','G_old','A','B','C','D')
ALL=(*NORMAL,'Dproj','Deq','DprojEq')

class Writer:
    def __init__(self,path):self.f=path.open('x',encoding='utf-8');self.f.write('[');self.n=0
    def append(self,rows):
        for row in rows:
            if self.n:self.f.write(',\n')
            self.f.write(json.dumps(row,separators=(',',':'),allow_nan=False));self.n+=1
        self.f.flush()
    def close(self):self.f.write(']\n');self.f.close()

def predictions(cache,rows,low):
    from lcf_model import decode_low
    from native_output_bridge import encode_rle
    if not rows:return []
    masks=decode_low(cache,low)
    changed=[dict(r,segmentation=encode_rle(masks[i])) for i,r in enumerate(rows)]
    require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(rows,changed)),'non-mask identity drift')
    return changed

def rms(cache,changed,baseline):
    from lcf_model import upsample,input_roi
    roi=input_roi(cache)
    field=upsample(changed,cache['input_hw'])-upsample(baseline,cache['input_hw'])
    return ((field.double().square()*roi).sum((1,2))/roi.sum((1,2)).clamp_min(1)).sqrt()

def equalize(cache,local,projected,baseline,delta_c):
    import torch
    a,b=rms(cache,local,baseline),rms(cache,projected,baseline)
    target=torch.minimum(a,b);usable=(a>1e-12)&(b>1e-12)
    sa=torch.where(usable,target/a.clamp_min(1e-12),0).float()
    sb=torch.where(usable,target/b.clamp_min(1e-12),0).float()
    p=cache['proto'].flatten(1);shape=baseline.shape
    for _ in range(8):
        left=baseline+sa[:,None,None]*(local-baseline)
        right=((cache['c0']+sb[:,None]*delta_c)@p).reshape(shape)
        qa,qb=rms(cache,left,baseline),rms(cache,right,baseline)
        ok=(qa-qb).abs()<=1e-7+1e-4*torch.maximum(qa,qb)
        if bool(ok.all()):break
        # Only shrink; do not enlarge an arm beyond the original response.
        new_target=torch.minimum(qa,qb)
        sa=sa*torch.where(qa>1e-12,new_target/qa.clamp_min(1e-12),0).float()
        sb=sb*torch.where(qb>1e-12,new_target/qb.clamp_min(1e-12),0).float()
    return left,right,{'image_id':cache['image_id'],'local_original_rms':a.tolist(),'projected_original_rms':b.tolist(),
          'local_actual_rms':qa.tolist(),'projected_actual_rms':qb.tolist(),'invalid_rows':(~ok).nonzero().flatten().tolist(),
          'zeroed_rows':(~usable).nonzero().flatten().tolist()}

def evaluate_stage():
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    from lcf_runner import native,new_models
    from adapter_model import CoefficientAdapter
    from native_output_bridge import decode_native,encode_rle
    from s1_geometry import iter_prediction_array,full_rle_iou
    from build_native_ledger import evaluate_ledger
    sys.path.append(str(ROOT/'experiments/mask_split_merge_topology_audit_20261020/scripts'))
    from topology import Context
    train=pointer('train');require(all(v==15000 for v in read(train/'SUMMARY.json')['steps_by_arm'].values()),'training incomplete')
    with Run('evaluate') as run:
        profile,env=environment();write(run.path/'ENVIRONMENT.json',env)
        stream=native(run.config,'val',run);base_hash=state_hash(stream.model);models=new_models(run.config,False)
        for arm,model in models.items():
            checkpoint=torch.load(train/arm/'last.pt',map_location='cpu',weights_only=True)
            require(checkpoint['arm']==arm and checkpoint['optimizer_steps']==15000,'checkpoint identity')
            model.load_state_dict(checkpoint['state_dict'],strict=True)
        old=CoefficientAdapter('B').cuda().eval();old.load_state_dict(torch.load(run.config['existing_G'],map_location='cpu',weights_only=True)['state_dict'])
        trace=read(run.config['historical_native_trace']);history={r['image_id']:r for r in trace['images']}
        niter=iter(iter_prediction_array(trace['predictions_path']));giter=iter(iter_prediction_array(run.config['existing_G_predictions']))
        ledger=[json.loads(s) for s in Path(run.config['historical_native_ledger']).read_text(encoding='utf-8').splitlines() if s.strip()]
        paired=defaultdict(list)
        for row in ledger:
            if not row['is_ignored'] and row['bbox75_match_output_id'] is not None:paired[row['image_id']].append(row)
        coco=COCO(run.config['val_annotations']);writers={n:Writer(run.path/(n+'_predictions.json')) for n in ALL}
        global_rank=0;invalid=0;examples=0
        with (run.path/'FIXED_PAIRS.jsonl').open('x',encoding='utf-8',buffering=1) as output,(run.path/'AMPLITUDE_CONTROLS.jsonl').open('x',encoding='utf-8',buffering=1) as controls,torch.no_grad():
            for bi in range(len(stream.plan)):
                run.check();caches,rows,features=stream.batch(bi,masks=True)
                for cache,native_rows in zip(caches,rows):
                    h=history[cache['image_id']]
                    for key in ('image_sha256','input_tensor_sha256','input_hw','original_hw','batch_index','batch_position'):
                        require(json.loads(json.dumps(cache[key]))==json.loads(json.dumps(h[key])),'historical input mismatch')
                    for row in native_rows:require(next(niter)==row,'native full RLE mismatch')
                    predictions_by={'N':native_rows}
                    gold_masks=decode_native(cache,cache['c0']+old(cache,'eval:0')) if native_rows else []
                    predictions_by['G_old']=[dict(r,segmentation=encode_rle(gold_masks[i])) if int(masks.area(r['segmentation']))>=9216 else dict(r) for i,r in enumerate(native_rows)]
                    for row in predictions_by['G_old']:require(next(giter)==row,'old G exact replay failed')
                    low_by={}
                    for arm,model in models.items():
                        low,baseline=model(cache,features);low_by[arm]=low;predictions_by[arm]=predictions(cache,native_rows,low)
                    projected,_=models['D'](cache,features,True);dc=models['D'].last_delta_c if native_rows else cache['c0']
                    predictions_by['Dproj']=predictions(cache,native_rows,projected)
                    if native_rows:
                        deq,peq,receipt=equalize(cache,low_by['D'],projected,baseline,dc)
                    else:deq=peq=baseline;receipt={'image_id':cache['image_id'],'invalid_rows':[],'zeroed_rows':[]}
                    predictions_by['Deq']=predictions(cache,native_rows,deq);predictions_by['DprojEq']=predictions(cache,native_rows,peq)
                    controls.write(json.dumps(receipt,allow_nan=False)+'\n');invalid+=len(receipt['invalid_rows'])
                    ordinary=[a for a in coco.imgToAnns[cache['image_id']] if not a.get('iscrowd',0) and not a.get('ignore',0)]
                    gt_masks={a['id']:coco.annToMask(a).astype(bool) for a in ordinary}
                    for row in paired[cache['image_id']]:
                        rank=int(row['bbox75_match_output_id'].rsplit(':',1)[1])-global_rank
                        require(0<=rank<len(native_rows),'fixed detection mapping')
                        gt=coco.annToRLE(coco.anns[row['annotation_id']]);others=[(i,m) for i,m in gt_masks.items() if i!=row['annotation_id']]
                        contexts={str(k):Context(gt_masks[row['annotation_id']],others,k) for k in [8,4]}
                        values={name:full_rle_iou(p[rank]['segmentation'],gt) for name,p in predictions_by.items()}
                        require(abs(values['N']-row['mask_iou_of_bbox_match'])<1e-12,'source pair IoU')
                        metrics={name:{k:context.measure(masks.decode(p[rank]['segmentation']).astype(bool)) for k,context in contexts.items()} for name,p in predictions_by.items()}
                        output.write(json.dumps({'image_id':cache['image_id'],'annotation_id':row['annotation_id'],
                           'native_output_index':global_rank+rank,'target':row['task_mask75']=='failure' and values['N']<.75,
                           'native_success':row['task_mask75']=='success','ious':values,'topology':metrics},allow_nan=False)+'\n');examples+=1
                    for name,writer in writers.items():writer.append(predictions_by[name])
                    global_rank+=len(native_rows)
                run.update(phase='native_and_method_export',completed_images=sum(len(v['image_ids']) for v in stream.plan[:bi+1]),outputs=global_rank)
                print('EXPORT',bi+1,len(stream.plan),global_rank,flush=True)
                del caches,rows,features,predictions_by,low_by
            require(next(niter,None) is None and next(giter,None) is None,'historical stream incomplete')
        for writer in writers.values():writer.close()
        require(state_hash(stream.model)==base_hash,'base model changed')
        del models,old,stream,coco;torch.cuda.empty_cache()
        methods={};damage={};nledger=None
        ids=read(STUDY/'inputs/DATA_CONTRACT.json')['val_ids']
        for name in NORMAL:
            run.check();path=run.path/(name+'_predictions.json');pred=read(path)
            with (run.path/(name+'_coco.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                current,metrics=evaluate_ledger(run.config['val_annotations'],pred,ids,sha(path))
            with (run.path/(name+'_LEDGER.jsonl')).open('x',encoding='utf-8') as file:
                for row in current:file.write(json.dumps(row)+'\n')
            if name=='N':
                nledger={r['annotation_id']:r for r in current};require(set(nledger)=={r['annotation_id'] for r in ledger},'native ledger IDs')
                for prior in ledger:
                    new=nledger[prior['annotation_id']]
                    for key in ['image_id','is_ignored','task_mask75','task_box75','mask_iou_of_bbox_match']:require(new[key]==prior[key],'native ledger differs')
            damaged=[];repaired=[]
            for row in current:
                before=nledger[row['annotation_id']]
                def rank(v):return v.rsplit(':',1)[1] if v else None
                require(before['task_box75']==row['task_box75'] and rank(before['bbox75_match_output_id'])==rank(row['bbox75_match_output_id']),'Box identity guard')
                if not row['is_ignored']:
                    if before['task_mask75']=='success' and row['task_mask75']!='success':damaged.append(row['annotation_id'])
                    if before['task_mask75']=='failure' and row['task_mask75']=='success':repaired.append(row['annotation_id'])
            total=sum(not r['is_ignored'] and r['task_mask75']=='success' for r in nledger.values())
            damage[name]={'damaged':len(damaged),'repaired':len(repaired),'native_success_denominator':total,
                          'damage_fraction':len(damaged)/total,'damaged_annotation_ids':damaged,'repaired_annotation_ids':repaired}
            methods[name]={'metrics':metrics,'predictions_sha256':sha(path),'outputs':len(pred)}
            write(run.path/(name+'_METRICS.json'),methods[name]);del pred,current
        write(run.path/'DAMAGE_CENSUS.json',damage)
        run.finish({'status':'completed','images':5000,'methods':methods,'fixed_pairs':examples,'damage':damage,
                    'identity_rows_checked':global_rank,'full_native_and_oldG_RLE_exact':True,
                    'amplitude_invalid_rows':invalid,'checkpoints_frozen':True},'evaluated_pending_statistics')
