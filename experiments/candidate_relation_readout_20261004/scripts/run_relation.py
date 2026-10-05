"""Bounded GPU-host preparation, smoke and training; no automatic retries."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import torch
import ultralytics
from ultralytics import YOLO
from runtime_utils import dump, setup
from train_evidence import read_training_image as original_read, feature_channels, direct_bce_check, backwards, checked_loss
from relation_io import read_training_image, feed


@torch.no_grad()
def prepare(cfg,out,root):
    from relation_maps import build_relations
    setup(cfg['seed'])
    original=json.loads((Path(cfg['cache'])/'INDEX.json').read_text(encoding='utf-8-sig'))
    index={s:original[s][:n] for s,n in cfg['planned_images'].items()}
    allids=[int(r['image_id']) for rows in index.values() for r in rows]
    assert len(allids)==len(set(allids))
    assert all(len(index[s])==n for s,n in cfg['planned_images'].items())
    dump(root/'INDEX.json',index)
    dump(root/'SPLIT.json',{s:[r['image_id'] for r in rows] for s,rows in index.items()})
    dest=Path(cfg['relation_cache']);(dest/'images').mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists(): raise RuntimeError('Preserve completed cache; no overwrite')
    source=YOLO(cfg['weights']).model.float().cuda().eval().requires_grad_(False)
    head=source.model[-1]
    assert head.end2end and ultralytics.__version__=='8.4.100'
    stats={};start=time.monotonic()
    for split,items in index.items():
        stat=dict(planned=len(items),images=0,candidates=0,with_neighbors=0,with_same_class=0,changed=0,max_box_replay_error=0.,max_coefficient_replay_error=0.)
        for position,item in enumerate(items):
            if not item['n']: continue
            iid=int(item['image_id']);x=original_read(cfg,iid)
            fs=[v[None].cuda() for v in x['F']]
            raw=head.forward_head(fs,**head.one2one)
            boxes=head._get_decode_boxes(raw)[0].T
            coefs=raw['mask_coefficient'][0].T
            scores,classes=raw['scores'][0].sigmoid().max(0)
            own=x['raw_ids'].cuda()
            boxerr=float((boxes[own]-x['boxes'].cuda()).abs().max())
            coeferr=float((coefs[own]-x['c0'].cuda()).abs().max())
            torch.testing.assert_close(boxes[own],x['boxes'].cuda(),atol=5e-3,rtol=5e-5)
            torch.testing.assert_close(coefs[own],x['c0'].cuda(),atol=3e-5,rtol=3e-5)
            top=torch.argsort(scores,descending=True,stable=True)[:cfg['pool_topk']]
            top=top[scores[top]>=cfg['pool_min_score']]
            valid=torch.isfinite(boxes[top]).all(1)&((boxes[top,2:]-boxes[top,:2])>0).all(1)
            top=top[valid]
            result=build_relations(x['_operator']['A'].cuda(),x['c0'].cuda(),x['boxes'].cuda(),own,
                boxes[top],coefs[top],scores[top],top,classes[top],classes[own])
            result={k:v.cpu() if torch.is_tensor(v) else v for k,v in result.items()}
            result.update(image_id=iid,raw_ids=x['raw_ids'],source_fingerprint=x['fingerprint'],
                pool_raw_ids=top.cpu(),pool_scores=scores[top].cpu(),pool_classes=classes[top].cpu())
            torch.save(result,dest/'images'/f'{iid:012d}.pt')
            stat['images']+=1;stat['candidates']+=int(item['n'])
            stat['with_neighbors']+=int((result['neighbor_count']>0).sum())
            stat['with_same_class']+=int((result['same_class_count']>0).sum())
            stat['changed']+=int((result['changed_fraction']>0).sum())
            stat['max_box_replay_error']=max(stat['max_box_replay_error'],boxerr)
            stat['max_coefficient_replay_error']=max(stat['max_coefficient_replay_error'],coeferr)
            if position%25==0 or position+1==len(items):
                progress=dict(stage='prepare',split=split,position=position+1,total=len(items),stats=stat,elapsed_s=time.monotonic()-start)
                dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
        stats[split]=stat
    dump(root/'ENVIRONMENT.json',dict(python=sys.version,host=platform.node(),torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),ultralytics=ultralytics.__version__,ultralytics_source=ultralytics.__file__))
    dump(dest/'COMPLETE.json',dict(completed=True,stats=stats,gt_free_pool=True,selection='top300 raw anchors by max class probability, score>=0.05; at most4 score*IoU overlapping neighbors; exclude own raw ID',elapsed_s=time.monotonic()-start))
    dump(out/'COMPLETE.json',dict(completed=True,stats=stats,elapsed_s=time.monotonic()-start))


def smoke(cfg,out,index):
    from training_engine import make_model, optimizer_for, buffer_manifest
    from evaluate_relation import decode_smoke
    items=[r for r in index['fit'] if r['n']]
    # First fit images covering all scales, as in the inherited smoke; no metric selection.
    images=[];covered=set()
    for item in items:
        x=read_training_image(cfg,int(item['image_id']))
        levels=set(x['levels'][x['_operator']['valid']].tolist())
        if len(images)<8 or levels-covered: images.append(x);covered|=levels
        if len(images)>=8 and covered=={0,1,2}: break
    assert covered=={0,1,2}
    channels=feature_channels(images[0]);denom=sum(len(x['raw_ids']) for x in images)
    report=dict(images=[x['image_id'] for x in images],arms={},pooled_candidates_are_gt_free=True)
    paired=None;native_ref=None
    for mode in cfg['arms']:
        model=make_model(channels,mode,cfg);opt=optimizer_for(model,cfg)
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters()};buffers=buffer_manifest(model)
        native={n:p for n,p in before.items() if n.startswith('native_cv4.')}
        if native_ref is None: native_ref=native
        else:
            for n,p in native.items(): torch.testing.assert_close(p,native_ref[n],atol=0,rtol=0)
        if mode!='N':
            if paired is None: paired=before
            else:
                assert before.keys()==paired.keys()
                for n,p in before.items(): torch.testing.assert_close(p,paired[n],rtol=0,atol=0)
        initial_error=0.;gradient_error=0.;grads={};losses=[];decode=None
        for step in range(2):
            opt.zero_grad(set_to_none=True);total=0.
            for lo in range(0,len(images),cfg['microbatch_images']):
                batch=images[lo:lo+cfg['microbatch_images']];fs,sel=feed(batch)
                versions=[(t,t._version) for t in fs]+[(v,v._version) for row in sel for v in row.values() if torch.is_tensor(v)]
                cs,residuals,live=model.forward_details(fs,sel);gs=[]
                for x,c,row,cur in zip(batch,cs,sel,live):
                    if step==0:
                        torch.testing.assert_close(c,row['c0'],atol=3e-5,rtol=3e-5)
                        torch.testing.assert_close(cur['h_current'],row['h0'],atol=3e-5,rtol=3e-5)
                        initial_error=max(initial_error,float((c-row['c0']).abs().max()))
                    value,g=checked_loss(x,c,denom);gs.append(g);total+=value
                    if step==0 and lo==0: gradient_error=max(gradient_error,direct_bce_check(x,c,g,denom))
                if step==0 and lo==0: decode=decode_smoke(cfg,batch[0],cs[0],mode)
                backwards(cs,gs)
                assert all(t._version==v and not t.requires_grad for t,v in versions)
            for n,p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),n
                    grads[n]=max(grads.get(n,0.),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
            opt.step();losses.append(total/denom)
            assert buffers==buffer_manifest(model)
        changes={n:float((p.detach().cpu()-before[n]).norm()) for n,p in model.named_parameters()}
        for level in range(3):
            assert any(v>0 for n,v in changes.items() if n.startswith(f'native_cv4.{level}.'))
        if mode!='N':
            for prefix in ('evidence.encoder','evidence.query','evidence.evidence_decoder'):
                assert any(v>0 for n,v in grads.items() if n.startswith(prefix)),prefix
        report['arms'][mode]=dict(counts=model.parameter_counts(),initial_max_error=initial_error,official_gradient_error=gradient_error,
            gradients=grads,changes=changes,decode=decode,losses=losses,buffers_unchanged=True)
        dump(out/'SMOKE.json',report);del model,opt;torch.cuda.empty_cache()
    assert report['arms']['S']['counts']==report['arms']['T']['counts']==report['arms']['M']['counts']
    dump(out/'COMPLETE.json',dict(passed=True,smoke_weights_discarded=True,formal_training_steps=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--stage',choices=['prepare','smoke','train'],required=True);p.add_argument('--mode',choices=['N','S','T','M']);p.add_argument('--deadline',type=float,default=0)
    args=p.parse_args();root=Path(args.config).parent;out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    from training_engine import resolve_config,train
    cfg=resolve_config(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
    if not torch.cuda.is_available(): raise RuntimeError('Laptop GPU required')
    try:
        if args.stage=='prepare': prepare(cfg,out,root)
        else:
            index=json.loads((root/'INDEX.json').read_text(encoding='utf-8-sig'))
            if args.stage=='smoke': smoke(cfg,out,index)
            else: train(cfg,out,index,args.mode,args.deadline)
    except Exception as e:
        dump(out/'FAILURE.json',dict(error=repr(e),stage=args.stage,mode=args.mode));raise

if __name__=='__main__': main()

