"""One fixed 16-image historical batch, exact native replay; no optimizer/GT fitting."""
from ssr_runtime import *

def main():
    import torch
    from native_stream import NativeBatches,historical_val_plan
    from ssr_runner import new_models
    from ssr_model import predict
    from s1_geometry import iter_prediction_array
    from ssr_evaluation import decoded
    from pycocotools import mask as masks
    from adapter_model import CoefficientAdapter
    config=read(STUDY/'CONFIG.json')
    run=STUDY/'runs'/('RUN_SSR_NATIVE_PREFLIGHT_'+uuid.uuid4().hex);run.mkdir(parents=True)
    started=time.perf_counter()
    meta={'run_id':run.name,'study_id':config['study_id'],'stage':'native_input_preflight','pid':os.getpid(),
          'started_at':now(),'execution_status':'running','optimizer_steps':0,'protocol_sha256':sha(STUDY/'PROTOCOL.md'),
          'code_commit':'bc0507c5400b61935fe93f981a4ff5026964c271'}
    write(run/'run.json',meta)
    try:
        profile,env=environment();write(run/'ENVIRONMENT.json',env)
        parent=read(STUDY/'inputs/PARENT_DATA_CONTRACT.json');hist=read(config['historical_native_trace'])
        contract=copy=__import__('copy').deepcopy(parent)
        contract['val_batches']=historical_val_plan(hist)
        keys=('weights','profile','adapter_initial','existing_G','train_annotations','val_annotations','historical_native_trace')
        contract['inputs']={k:{'path':config[k],'sha256':sha(config[k])} for k in keys}
        prepared=run/'inputs';prepared.mkdir();write(prepared/'DATA_CONTRACT.json',contract)
        (prepared/'val.txt').write_text(''.join(f'D:/coco_wire/data/images/val2017/{i:012d}.jpg\n' for i in parent['val_ids']),encoding='utf-8')
        stream=NativeBatches(config,prepared,'val',run/'native_config');before=state_hash(stream.model)
        caches,rows,features=stream.batch(0,masks=True)
        previous={r['image_id']:r for r in hist['images']}
        for c in caches:
            h=previous[c['image_id']]
            for k in ('image_sha256','input_tensor_sha256','input_hw','original_hw','batch_index','batch_position'):
                require(json.loads(json.dumps(c[k]))==json.loads(json.dumps(h[k])),'historical metadata drift: '+k)
            original=torch.load(Path(config['historical_native_trace']).parent/h['cache'],map_location='cpu',weights_only=True)
            for k in ('proto','c0','boxes_input','scores','classes'):
                require(torch.equal(c[k].detach().cpu(),original[k]),'historical tensor drift: '+k)
        native_original=iter(iter_prediction_array(hist['predictions_path']))
        old_original=iter(iter_prediction_array(config['existing_G_predictions']))
        old=CoefficientAdapter('B').cuda().eval()
        old.load_state_dict(torch.load(config['existing_G'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        with torch.no_grad():
            for c,r in zip(caches,rows):
                require(all(next(native_original)==v for v in r),'historical native serialized RLE/row mismatch')
                gate=[int(masks.area(v['segmentation']))>=9216 for v in r]
                mixed=decoded(c,r,old(c,'eval:0'),gate)
                require(all(next(old_original)==v for v in mixed),'historical G serialized RLE/row mismatch')
        models=new_models(config,False);timings={}
        with torch.no_grad():
            # A real, predeclared image tests full-size inputs. All heads remain
            # exactly zero; no segmentation score is read or used for selection.
            for arm,m in models.items():
                torch.cuda.synchronize();t=time.perf_counter();d=predict(m,caches[0],features,'eval:0');torch.cuda.synchronize()
                require(torch.count_nonzero(d).item()==0,'preflight residual nonzero');timings[arm]=time.perf_counter()-t
        require(state_hash(stream.model)==before,'base changed')
        summary={'status':'passed','images':16,'historical_native_tensors_exact':True,'native_zero_RLE_exact':True,
                 'historical_native_serialized_rows_exact':True,'historical_existing_G_serialized_rows_exact':True,
                 'method_forward_image_id':caches[0]['image_id'],'zero_delta_all_arms':True,'forward_seconds':timings,
                 'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved(),
                 'optimizer_steps':0,'elapsed_seconds':time.perf_counter()-started}
        write(run/'SUMMARY.json',summary)
        meta.update(execution_status='completed',finished_at=now(),exit_code=0);write(run/'run.json',meta)
        write(STUDY/'preflight/NATIVE_PREFLIGHT.json',{'run_id':run.name,**summary})
        print(json.dumps(summary),flush=True)
    except BaseException as e:
        (run/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        meta.update(execution_status='failed',error=str(e),exit_code=1);write(run/'run.json',meta);raise

if __name__=='__main__':main()
