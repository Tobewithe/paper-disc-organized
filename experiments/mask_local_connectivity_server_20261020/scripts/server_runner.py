"""One GPU per independent arm. Slurm owns all GPU execution."""
import json, os, sys, time, math
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
stage=sys.argv[1]
arm=os.environ.get('LCF_ARM') or ('ABCD'[int(os.environ['SLURM_PROCID'])] if 'SLURM_PROCID' in os.environ else None)
if stage in ('smoke','train'):
    assert arm in 'ABCD' and len(arm)==1
    os.environ['LCF_ARM']=arm
from lcf_runtime import *

def install_initialization_reference():
    """Exact zero-step tensors, fresh optimizer; platform RNG differences are excluded."""
    import torch,lcf_model
    reference=BASE/'LAPTOP_INITIAL_REFERENCE.pt'
    require(sha(reference)=='a561c529b2dbf168da9dd6378de44ea69eab388198070fe272b7fc7c9a3c2b4a','initial reference SHA')
    checkpoint=torch.load(reference,map_location='cpu',weights_only=True)
    require(checkpoint['optimizer_steps']==0 and checkpoint['completed_epochs']==0,'reference is not zero-step initialization')
    require(checkpoint['config_sha256']=='46c15235154f726f4ac60767154feb04dc7018786f54d505b1fef37d1088f939','wrong parent initialization configuration')
    original=lcf_model.make_model
    def exact_initialization(arm,spec):
        model=original(arm,spec)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        require(all(torch.equal(v,checkpoint['state_dict'][k]) for k,v in model.state_dict().items()),'initialization tensor drift')
        return model
    lcf_model.make_model=exact_initialization

def smoke():
    import torch
    from pycocotools.coco import COCO
    from adapter_model import CoefficientAdapter
    from lcf_model import make_model,decode_low
    from lcf_runner import native,loss_for
    from lcf_targets import prepare_targets
    from native_stream import batch_identity,tensor_sha
    from native_output_bridge import encode_rle,decode_native
    from s1_geometry import iter_prediction_array
    from test_components import run as component_tests
    with Run('smoke') as run:
        profile,env=environment()
        import scipy
        require(scipy.__version__=='1.15.2','SciPy version drift')
        env.update(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),scipy_version=scipy.__version__,device_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')))
        write(run.path/'ENVIRONMENT.json',env)
        torch.set_num_threads(1)
        write(run.path/'COMPONENT_TESTS.json',component_tests(run.config['model'],run.config['loss']))
        stream=native(run.config,'train',run);base_hash=state_hash(stream.model)
        bi=max(range(len(stream.plan)),key=lambda i:(math.prod(stream.plan[i]['rect_shape']),-i))
        start=time.perf_counter();caches,rows,features=stream.batch(bi,masks=True);native_seconds=time.perf_counter()-start
        identity=batch_identity(caches,rows,features)
        coco=COCO(run.config['train_annotations']);targets=[prepare_targets(coco,c,r,run.config) for c,r in zip(caches,rows)]
        model=make_model(arm,run.config['model']).cuda().train();initial=state_hash(model);opt=optimizer(model,run.config);grads=set();times=[]
        require(sum(p.numel() for p in model.parameters())==71361,'parameter count drift')
        with torch.no_grad():
            for c,r in zip(caches,rows):
                low,base=model(c,features);require(torch.equal(low,base),'zero residual not exact')
                if r:
                    decoded=decode_low(c,low)
                    require(all(encode_rle(decoded[i])==x['segmentation'] for i,x in enumerate(r)),'zero delta native RLE mismatch')
        for group in range(4):
            run.check();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            for i in range(group*4,group*4+4):
                loss,account=loss_for(model,caches[i],rows[i],features,targets[i],'train:0');require(torch.isfinite(loss).item(),'nonfinite loss');(loss/4).backward()
            for n,p in model.named_parameters():
                if p.grad is not None:
                    require(torch.isfinite(p.grad).all().item(),'nonfinite gradient')
                    if torch.count_nonzero(p.grad).item():grads.add(n)
            opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-start);run.update(optimizer_steps=group+1,phase='smoke_updates')
        require(grads=={n for n,_ in model.named_parameters()},'inactive trainable tensors')
        require(state_hash(stream.model)==base_hash,'base changed')
        forecast=(max(times)+native_seconds/4)*15000/.75
        require(forecast<=96*3600,'forecast exceeds fixed 96h per-arm cap')
        del model,opt,caches,features,targets,stream,coco,loss,base,low,decoded
        torch.cuda.empty_cache()
        val=native(run.config,'val',run);caches,rows,features=val.batch(0,masks=True)
        history={x['image_id']:x for x in read(run.config['historical_native_trace'])['images']}
        ids={c['image_id'] for c in caches}
        references={n:{} for n in ['N','G']}
        for name,path in [('N',read(run.config['historical_native_trace'])['predictions_path']),('G',run.config['existing_G_predictions'])]:
            for x in iter_prediction_array(path):
                if x['image_id'] in ids:references[name].setdefault(x['image_id'],[]).append(x)
        old=CoefficientAdapter('B').cuda().eval();old.load_state_dict(torch.load(run.config['existing_G'],map_location='cpu',weights_only=True)['state_dict'])
        count=0
        with torch.no_grad():
            for c,r in zip(caches,rows):
                h=history[c['image_id']]
                for key in ['image_sha256','input_tensor_sha256','input_hw','original_hw','batch_index','batch_position']:
                    require(json.loads(json.dumps(c[key]))==json.loads(json.dumps(h[key])),'historical input geometry '+key)
                expected=references['N'].get(c['image_id'],[])
                require(r==expected,'same-server native complete RLE/row mismatch')
                masks=decode_native(c,c['c0']+old(c,'eval:0')) if r else []
                mixed=[]
                for i,x in enumerate(r):
                    area=__import__('pycocotools.mask',fromlist=['area']).area(x['segmentation'])
                    mixed.append(dict(x,segmentation=encode_rle(masks[i])) if area>=9216 else x)
                require(mixed==references['G'].get(c['image_id'],[]),'same-server old G RLE mismatch')
                count+=len(r)
        receipt={'status':'passed','arm':arm,'parameters':71361,'initial_tensor_sha256':initial,'optimizer_steps':4,'four_update_seconds':times,'forecast_hours':forecast/3600,'train_source_batch':bi,'train_identity':identity,'val_identity':batch_identity(caches,rows,features),'native_G_rows_exact':count,'zero_delta_RLE_exact':True,'all_trainable_gradient_names':sorted(grads),'native_base_unchanged':True,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()}
        write(run.path/'SMOKE_RECEIPT.json',receipt);run.finish(receipt)
        print(json.dumps({'arm':arm,'run_id':run.path.name,'status':'passed','forecast_hours':forecast/3600,'rows':count}),flush=True)

def train():
    gate=read(BASE/'SERVER_SMOKE_GATE.json');require(gate['status']=='passed','four-arm smoke gate not passed')
    require(gate['freeze_sha256']==sha(BASE/'SERVER_FREEZE_MANIFEST.json'),'gate/freeze drift')
    import lcf_model
    lcf_model.ARMS=(arm,)
    from lcf_runner import train as actual
    actual()

if __name__=='__main__':
    install_initialization_reference()
    if stage=='smoke':smoke()
    elif stage=='train':train()
    else:raise ValueError(stage)
