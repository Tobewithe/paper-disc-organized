"""Prepare/reuse assets, GPU preflight, then fixed-budget training. No desktop computation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()


def prepare(cfg,out,root):
    from runtime_utils import dump
    from train_evidence import operator_fingerprint
    original=Path(cfg['cache'])/'INDEX.json'
    index=json.loads(original.read_text(encoding='utf-8-sig'))
    selected={s:index[s][:n] for s,n in cfg['planned_images'].items()}
    ids=[]; counts={}
    for split,items in selected.items():
        if len(items)!=cfg['planned_images'][split]: raise ValueError('Insufficient existing images')
        ids.extend(int(r['image_id']) for r in items)
        counts[split]=dict(planned=len(items),positive=sum(r['n']>0 for r in items),candidates=sum(r['n'] for r in items),
                           no_positive=[r['image_id'] for r in items if not r['n']])
        for row in items:
            iid=int(row['image_id'])
            if not (Path(cfg['cache'])/'images'/f'{iid:012d}.pt').is_file(): raise FileNotFoundError(iid)
            if row['n'] and not (Path(cfg['operator_cache'])/'images'/f'{iid:012d}.pt').is_file(): raise FileNotFoundError(iid)
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate or overlapping fit/dev IDs')
    for base in ('cache','operator_cache'):
        if not (Path(cfg[base])/'COMPLETE.json').exists(): raise RuntimeError('Source cache incomplete')
    op=json.loads((Path(cfg['operator_cache'])/'COMPLETE.json').read_text(encoding='utf-8-sig'))
    if op['fingerprint']!=operator_fingerprint(cfg): raise ValueError('Operator identity mismatch')
    dump(root/'INDEX.json',selected)
    dump(root/'SPLIT.json',{s:[int(r['image_id']) for r in items] for s,items in selected.items()})
    sources=[original,Path(cfg['cache'])/'COMPLETE.json',Path(cfg['operator_cache'])/'COMPLETE.json',Path(cfg['weights']),Path(cfg['baseline_rows'])]
    dump(out/'ASSET_REUSE.json',dict(counts=counts,sources={str(p):sha(p) for p in sources},
         no_new_assignment=True,operator_fingerprint=op['fingerprint'],selection='first planned INDEX rows, then retain zero-positive records'))
    import torch,ultralytics
    dump(root/'ENVIRONMENT.json',dict(python=sys.version,host=platform.node(),torch=torch.__version__,cuda=torch.version.cuda,
         device=torch.cuda.get_device_name(0),ultralytics=ultralytics.__version__,ultralytics_source=ultralytics.__file__))
    dump(out/'COMPLETE.json',dict(completed=True,counts=counts))


def smoke(cfg,out,index):
    import torch
    from training_engine import make_model,optimizer_for,buffer_manifest
    from runtime_utils import dump
    from train_evidence import choose_smoke_images,feature_channels,feed,checked_loss,backwards,direct_bce_check
    from evaluate_feedback import decode_smoke
    images,scanned=choose_smoke_images(cfg,index)
    channels=feature_channels(images[0]); denom=sum(len(x['raw_ids']) for x in images)
    report=dict(images=[x['image_id'] for x in images],scanned=scanned,arms={})
    paired=None; native_initial=None
    for mode in cfg['arms']:
        model=make_model(channels,mode,cfg); opt=optimizer_for(model,cfg)
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        buffers=buffer_manifest(model)
        native={n:p for n,p in before.items() if n.startswith('native_cv4.')}
        if native_initial is None: native_initial=native
        else:
            for n,p in native.items(): torch.testing.assert_close(p,native_initial[n],rtol=0,atol=0)
        if mode!='N':
            if paired is None: paired=before
            else:
                assert paired.keys()==before.keys()
                for n,p in before.items(): torch.testing.assert_close(p,paired[n],rtol=0,atol=0)
        maximum=0.; loss_check=0.; grads={}; losses=[]
        decode=None
        for step in range(2):
            opt.zero_grad(set_to_none=True); total=0.
            for lo in range(0,len(images),cfg['microbatch_images']):
                batch=images[lo:lo+cfg['microbatch_images']]; features,selected=feed(batch)
                versions=[(v,v._version) for v in features]+[(v,v._version) for r in selected for v in r.values() if torch.is_tensor(v)]
                cs,first,trace,current=model.forward_trace(features,selected)
                gs=[]
                for x,c,row,live in zip(batch,cs,selected,current):
                    if step==0:
                        torch.testing.assert_close(c,row['c0'],atol=3e-5,rtol=3e-5)
                        torch.testing.assert_close(live['h_current'],row['h0'],atol=3e-5,rtol=3e-5)
                        maximum=max(maximum,float((c-row['c0']).abs().max()))
                    value,g=checked_loss(x,c,denom);gs.append(g);total+=value
                    if step==0 and lo==0:
                        loss_check=max(loss_check,direct_bce_check(x,c,g,denom))
                        # Same reference and gradient agreement away from original c0.
                        shifted=c.detach()+torch.linspace(-.05,.05,32,device=c.device)
                        _,sg=checked_loss(x,shifted,denom)
                        loss_check=max(loss_check,direct_bce_check(x,shifted,sg,denom))
                if step==0 and lo==0: decode=decode_smoke(cfg,batch[0],cs[0],mode)
                backwards(cs,gs)
                assert all(v._version==version and not v.requires_grad for v,version in versions)
            for n,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError(n)
                    grads[n]=max(grads.get(n,0.),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
            opt.step();losses.append(total/denom)
            assert buffers==buffer_manifest(model),'Frozen buffers changed'
        changes={n:float((p.detach().cpu()-before[n]).norm()) for n,p in model.named_parameters()}
        for level in range(3):
            for layer in range(3):
                prefix=f'native_cv4.{level}.{layer}.'
                assert any(v>0 for n,v in changes.items() if n.startswith(prefix)),prefix
        feedback={}
        if mode!='N':
            for prefix in ['evidence.encoder','evidence.query_mlp','evidence.query_film','evidence.evidence_decoder']:
                assert any(v>0 for n,v in grads.items() if n.startswith(prefix)),prefix
            fs,sel=feed(images[:2])
            with torch.no_grad():
                final,first,traces,live=model.forward_trace(fs,sel)
                hold,first_hold,hold_traces,_=model.forward_trace(fs,sel,hold=True)
                for a,b in zip(first,first_hold): torch.testing.assert_close(a,b,rtol=0,atol=0)
                feedback=dict(step2_context_change=max(t['context_change_step2'] for t in traces),
                    held_context_change=max(t['context_change_step2'] for t in hold_traces),
                    final_hold_max_difference=max(float((a-b).abs().max()) for a,b in zip(final,hold)),first_step_identical=True)
                assert feedback['held_context_change']==0
                if mode=='F': assert feedback['final_hold_max_difference']==0
                else:
                    assert feedback['step2_context_change']>0,'Feedback did not alter spatial aggregation'
                    assert feedback['final_hold_max_difference']>0,'Updated aggregation does not affect output'
        report['arms'][mode]=dict(counts=model.parameter_counts(),initial_max_error=maximum,official_gradient_max_error=loss_check,
            losses=losses,gradients=grads,changes=changes,buffers_unchanged=True,decode=decode,feedback=feedback)
        dump(out/'SMOKE.json',report)
        del model,opt;torch.cuda.empty_cache()
    assert report['arms']['F']['counts']==report['arms']['R']['counts']
    report['paired_parameter_initialization_identical']=True
    dump(out/'SMOKE.json',report)
    dump(out/'COMPLETE.json',dict(passed=True,smoke_weights_discarded=True,formal_training_steps=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--stage',choices=['prepare','smoke','train'],required=True)
    parser.add_argument('--mode',choices=['N','F','R']);parser.add_argument('--deadline',type=float,default=0)
    args=parser.parse_args();root=Path(args.config).parent;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    from training_engine import resolve_config,train
    from runtime_utils import dump
    import torch
    cfg=resolve_config(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
    if not torch.cuda.is_available(): raise RuntimeError('Authorized laptop GPU required')
    try:
        if args.stage=='prepare': prepare(cfg,out,root)
        else:
            index=json.loads((root/'INDEX.json').read_text(encoding='utf-8-sig'))
            if args.stage=='smoke': smoke(cfg,out,index)
            else: train(cfg,out,index,args.mode,args.deadline)
    except Exception as e:
        dump(out/'FAILURE.json',dict(error=repr(e),stage=args.stage,mode=args.mode));raise


if __name__=='__main__': main()
