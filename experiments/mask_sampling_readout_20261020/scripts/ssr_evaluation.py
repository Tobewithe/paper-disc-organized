"""Complete normal predictions, fixed N pairs, same-D interventions, full GT census."""
from __future__ import annotations
import contextlib,json,time
from collections import defaultdict
from pathlib import Path
from ssr_runtime import *

NORMAL=('N','G_old','G20k','A','B','C','D','M')
ALL=(*NORMAL,'J','Deq','Jeq')

class Writer:
    def __init__(self,path):self.f=path.open('x',encoding='utf-8');self.f.write('[');self.count=0
    def append(self,rows):
        for r in rows:
            if self.count:self.f.write(',\n')
            self.f.write(json.dumps(r,separators=(',',':'),allow_nan=False));self.count+=1
        self.f.flush()
    def close(self):self.f.write(']\n');self.f.close()

def decoded(cache,rows,delta,gate):
    from native_output_bridge import decode_native,encode_rle
    if not rows:
        require(len(delta)==0,'empty native output acquired rows')
        return []
    masks=decode_native(cache,cache['c0']+delta)
    result=[dict(r,segmentation=encode_rle(masks[i])) if gate[i] else dict(r) for i,r in enumerate(rows)]
    require(all({k:v for k,v in a.items() if k!='segmentation'}=={k:v for k,v in b.items() if k!='segmentation'} for a,b in zip(rows,result)),
            'non-Mask detection identity changed')
    return result

def response_rms(cache,delta,gate):
    import torch
    from ultralytics.utils import ops
    n=len(delta);result=delta.new_zeros(n,dtype=torch.float64)
    ids=[i for i,g in enumerate(gate) if g]
    if not ids:return result
    ih,iw=cache['input_hw'];ph,pw=cache['proto'].shape[-2:]
    p=cache['proto'].flatten(1)
    # Preserve the normal decoder's COMPLETE row-matrix shape before selecting
    # gated rows; changing GEMM batch geometry can alter FP32 rounding.
    base=cache['c0']@p
    changed=(cache['c0']+delta)@p
    up_base=ops.scale_masks(base.reshape(1,n,ph,pw),(ih,iw))[0]
    up_changed=ops.scale_masks(changed.reshape(1,n,ph,pw),(ih,iw))[0]
    field=(up_changed-up_base)[ids]
    b=cache['boxes_input'][ids]
    x=torch.arange(iw,device=delta.device)[None,None,:];y=torch.arange(ih,device=delta.device)[None,:,None]
    roi=(x>=b[:,0,None,None])&(x<b[:,2,None,None])&(y>=b[:,1,None,None])&(y<b[:,3,None,None])
    count=roi.sum((1,2)).clamp_min(1)
    value=((field.double().square()*roi).sum((1,2))/count).sqrt()
    return result.index_copy(0,torch.tensor(ids,device=delta.device),value)

def equalize(cache,d,j,gate):
    import torch
    qd,qj=response_rms(cache,d,gate),response_rms(cache,j,gate)
    target=torch.minimum(qd,qj)
    valid=(qd>1e-12)&(qj>1e-12)
    dd=d*(torch.where(valid,target/qd.clamp_min(1e-12),0).float()[:,None])
    jj=j*(torch.where(valid,target/qj.clamp_min(1e-12),0).float()[:,None])
    actual_d,actual_j=response_rms(cache,dd,gate),response_rms(cache,jj,gate)
    ok=(actual_d-actual_j).abs()<=1e-7+1e-4*torch.maximum(actual_d,actual_j)
    ok=ok|~torch.tensor(gate,device=ok.device,dtype=torch.bool)
    return dd,jj,{'image_id':cache['image_id'],'qd':qd.cpu().tolist(),'qj':qj.cpu().tolist(),
                 'actual_Deq':actual_d.cpu().tolist(),'actual_Jeq':actual_j.cpu().tolist(),
                 'zeroed_rows':(~valid).nonzero().flatten().tolist(),'invalid_rows':(~ok).nonzero().flatten().tolist()}

def evaluate_stage():
    import torch
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    from ssr_runner import native,new_models
    from ssr_model import predict
    from adapter_model import CoefficientAdapter
    from s1_geometry import iter_prediction_array,full_rle_iou
    from build_native_ledger import evaluate_ledger
    train=pointer('train');require(read(train/'SUMMARY.json')['steps_by_arm']['D']==15000,'final training incomplete')
    with Run('evaluate') as run:
        profile,env=environment();write(run.path/'ENVIRONMENT.json',env)
        stream=native(run.config,'val',run);base_before=state_hash(stream.model)
        models=new_models(run.config,False)
        for arm,model in models.items():
            cp=train/arm/'last.pt';saved=torch.load(cp,map_location='cpu',weights_only=True)
            require(saved['optimizer_steps']==15000 and saved['completed_epochs']==3 and saved['arm']==arm,'checkpoint identity')
            model.load_state_dict(saved['state_dict'],strict=True);del saved
        old=CoefficientAdapter('B').cuda().eval()
        old.load_state_dict(torch.load(run.config['existing_G'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        hist_trace=read(run.config['historical_native_trace']);hist={r['image_id']:r for r in hist_trace['images']}
        native_iter=iter(iter_prediction_array(hist_trace['predictions_path']))
        old_iter=iter(iter_prediction_array(run.config['existing_G_predictions']))
        native_ledger=[json.loads(s) for s in Path(run.config['historical_native_ledger']).read_text(encoding='utf-8').splitlines() if s.strip()]
        paired=defaultdict(list)
        for row in native_ledger:
            if not row['is_ignored'] and row['bbox75_match_output_id'] is not None:paired[row['image_id']].append(row)
        coco=COCO(run.config['val_annotations']);writers={n:Writer(run.path/(n+'_predictions.json')) for n in ALL}
        global_rank=0;identity_rows=0;control_invalid=0
        with (run.path/'FIXED_PAIRS.jsonl').open('x',encoding='utf-8',buffering=1) as effects,(run.path/'AMPLITUDE_CONTROLS.jsonl').open('x',encoding='utf-8',buffering=1) as controls,torch.no_grad():
            for bi in range(len(stream.plan)):
                run.check();caches,rows,features=stream.batch(bi,masks=True)
                for c,native_rows in zip(caches,rows):
                    h=hist[c['image_id']]
                    for key in ('image_sha256','input_tensor_sha256','input_hw','original_hw','batch_index','batch_position'):
                        require(json.loads(json.dumps(c[key]))==json.loads(json.dumps(h[key])),'native historical identity drift')
                    for row in native_rows:require(next(native_iter)==row,'full historical native RLE/identity mismatch')
                    gate=[int(masks.area(r['segmentation']))>=9216 for r in native_rows]
                    predictions={'N':native_rows}
                    predictions['G_old']=decoded(c,native_rows,old(c,'eval:0'),gate)
                    for row in predictions['G_old']:require(next(old_iter)==row,'full historical G replay mismatch')
                    deltas={a:predict(m,c,features,'eval:0') for a,m in models.items()}
                    for a,d in deltas.items():predictions[a]=decoded(c,native_rows,d,gate)
                    j=predict(models['D'],c,features,'eval:0','J')
                    predictions['J']=decoded(c,native_rows,j,gate)
                    deq,jeq,receipt=equalize(c,deltas['D'],j,gate)
                    predictions['Deq']=decoded(c,native_rows,deq,gate);predictions['Jeq']=decoded(c,native_rows,jeq,gate)
                    controls.write(json.dumps(receipt,allow_nan=False)+'\n');control_invalid+=len(receipt['invalid_rows'])
                    for row in paired[c['image_id']]:
                        rank=int(row['bbox75_match_output_id'].rsplit(':',1)[1])-global_rank
                        require(0<=rank<len(native_rows),'fixed detection mapping changed')
                        gt=coco.annToRLE(coco.anns[row['annotation_id']])
                        values={name:full_rle_iou(pred[rank]['segmentation'],gt) for name,pred in predictions.items()}
                        require(abs(values['N']-row['mask_iou_of_bbox_match'])<1e-12,'native fixed pair IoU drift')
                        effects.write(json.dumps({'image_id':c['image_id'],'annotation_id':row['annotation_id'],
                            'native_output_index':global_rank+rank,'target':row['task_mask75']=='failure' and values['N']<.75,
                            'native_success':row['task_mask75']=='success','ious':values},allow_nan=False)+'\n')
                    for name in ALL:writers[name].append(predictions[name])
                    global_rank+=len(native_rows);identity_rows+=len(native_rows)
                run.update(phase='full_native_and_method_export',completed_images=sum(len(b['image_ids']) for b in stream.plan[:bi+1]),
                           native_rows=global_rank,amplitude_invalid_rows=control_invalid)
                print('EXPORT',bi+1,len(stream.plan),global_rank,flush=True)
                del caches,rows,features,predictions,deltas,j,deq,jeq
            require(next(native_iter,None) is None and next(old_iter,None) is None,'historical stream coverage incomplete')
        for w in writers.values():w.close()
        require(state_hash(stream.model)==base_before,'base model changed in evaluation')
        del models,old,stream,coco,native_ledger,paired;torch.cuda.empty_cache()
        methods={};nledger=None;damage={}
        ids=read(STUDY/'inputs/DATA_CONTRACT.json')['val_ids']
        for name in NORMAL:
            run.check();path=run.path/(name+'_predictions.json');predictions=read(path)
            with (run.path/(name+'_coco.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                ledger,metrics=evaluate_ledger(run.config['val_annotations'],predictions,ids,sha(path))
            with (run.path/(name+'_LEDGER.jsonl')).open('x',encoding='utf-8') as f:
                for row in ledger:f.write(json.dumps(row)+'\n')
            if name=='N':
                nledger={r['annotation_id']:r for r in ledger}
                historical=[json.loads(s) for s in Path(run.config['historical_native_ledger']).read_text(encoding='utf-8').splitlines() if s.strip()]
                require(set(nledger)=={r['annotation_id'] for r in historical},'historical GT identity differs')
                for prior in historical:
                    current=nledger[prior['annotation_id']]
                    for key in ('image_id','is_ignored','task_mask75','task_box75','mask_iou_of_bbox_match'):
                        require(current[key]==prior[key],'historical native ledger differs: '+key)
                require(sum(not r['is_ignored'] for r in ledger)==36335,'ordinary GT denominator changed')
            damaged=[];repaired=[]
            for row in ledger:
                prev=nledger[row['annotation_id']]
                def rank(v):return None if v is None else v.rsplit(':',1)[1]
                require(prev['task_box75']==row['task_box75'] and rank(prev['bbox75_match_output_id'])==rank(row['bbox75_match_output_id']),'bbox guard failed')
                if not row['is_ignored']:
                    if prev['task_mask75']=='success' and row['task_mask75']!='success':damaged.append(row['annotation_id'])
                    if prev['task_mask75']=='failure' and row['task_mask75']=='success':repaired.append(row['annotation_id'])
            success=sum(r['task_mask75']=='success' for r in nledger.values())
            damage[name]={'damaged':len(damaged),'repaired':len(repaired),'native_success_denominator':success,
                          'damage_fraction':len(damaged)/success,'damaged_annotation_ids':damaged,'repaired_annotation_ids':repaired}
            methods[name]={'metrics':metrics,'predictions_sha256':sha(path),'outputs':len(predictions)}
            write(run.path/(name+'_METRICS.json'),methods[name]);del predictions,ledger
            run.update(phase='complete_COCO_and_damage_census',method=name)
        write(run.path/'DAMAGE_CENSUS.json',damage)
        run.finish({'status':'completed','images':5000,'methods':methods,'identity_rows_checked':identity_rows,
                    'full_native_and_oldG_RLE_exact':True,'amplitude_invalid_rows':control_invalid,
                    'damage':damage,'all_final_checkpoints_frozen':True},'evaluated_pending_statistics')
