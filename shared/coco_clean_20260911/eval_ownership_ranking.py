"""GT-free frozen-head decoding then separate original-COCO task/spatial evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,contextlib,gzip,io,json,time
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils import ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from local_coefficient_head import LocalCoefficientHead
from ownership_ranking import input_features,coefficients
from frozen_mechanism_probe import ROOT,sha,write_json
from three_region_probe import write_csv
from relative_ownership_experiment import read,spatial_and_predictions
from summarize_relative_ownership import evaluate

@torch.inference_mode()
def decode(item,models,norm):
    # Only this allowlist enters inference. GT mappings/annotations cannot be features.
    assert set(item)=={'features','level','coeff','proto','boxes','input_shape'}
    x=input_features(item);c=torch.as_tensor(item['coeff'],device='cuda').float();p=torch.as_tensor(item['proto'],device='cuda').float();boxes=torch.as_tensor(item['boxes'],device='cuda').float();shape=tuple(map(int,item['input_shape']))
    masks={};changed={}
    for arm,model in [('initial',None),*models.items()]:
        coeff=c if model is None else coefficients(x,c,model,norm)
        chunks=[ops.process_mask(p,coeff[k:k+32],boxes[k:k+32],shape,upsample=True) for k in range(0,len(c),32)]
        masks[arm]=torch.cat(chunks) if chunks else torch.empty((0,*shape),device='cuda',dtype=torch.uint8)
        if arm!='initial':changed[arm]=int((masks[arm]!=masks['initial']).sum())
    return masks,changed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train=a.training;receipt=json.loads((train/'COMPLETE.json').read_text())
    for name,digest in receipt['hashes'].items():
        if sha(train/name)!=digest:raise RuntimeError(('training receipt',name))
    models={};checkpoints={};norm=torch.load(train/'normalizer.pt',map_location='cuda',weights_only=True)
    for r in receipt['runs']:
        path=train/r['checkpoint'];arm=f'{r["arm"]}_s{r["seed"]}'
        if sha(path)!=r['sha256']:raise RuntimeError(('checkpoint hash',arm))
        state=torch.load(path,map_location='cpu',weights_only=False);model=LocalCoefficientHead().cuda().eval();model.load_state_dict(state['model']);models[arm]=model;checkpoints[arm]=sha(path)
    selection=json.loads((a.cache/'selection.json').read_text());ids=selection['evaluation'];write_json(out/'selection.json',selection)
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),dependencies={n:sha(Path(__file__).with_name(n)) for n in ['ownership_ranking.py','local_coefficient_head.py','relative_ownership_experiment.py','summarize_relative_ownership.py']},
        training_complete_sha256=sha(train/'COMPLETE.json'),training_protocol_sha256=sha(train/'protocol.json'),checkpoint_sha256=checkpoints,normalizer_sha256=sha(train/'normalizer.pt'),
        annotation_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),census_sha256=sha(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'),images=ids,
        inference_keys=['features','level','coeff','proto','boxes','input_shape'],GT_inference=False,official_evaluator='COCOeval segm all default thresholds/maxDets',
        scope='500 hash-selected reused val images (8 in smoke), all GT in task denominators, fixed matched spatial metrics; original nonempty candidates only. No end-to-end training or pristine test claim.'))
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    meta={int(r['annotation_id']):r for r in read(ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv')};allpred={arm:[] for arm in ['initial',*models]};spatial=[];skipped=[];corrections=[];cachehash={};boxpred=[];start=time.monotonic()
    for n,iid in enumerate(ids,1):
        path=ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz';cachehash[str(iid)]=sha(path)
        with np.load(path) as q:item={k:q[k] for k in q.files}
        masks,changes=decode({k:item[k] for k in ['features','level','coeff','proto','boxes','input_shape']},models,norm)
        # One zero-head runtime witness using this exact decoder path.
        if n==1:
            zero=LocalCoefficientHead().cuda().eval();zero_masks,_=decode({k:item[k] for k in ['features','level','coeff','proto','boxes','input_shape']},{'zero':zero},norm)
            xor=int((zero_masks['zero']!=masks['initial']).sum())
            if xor:raise RuntimeError('Zero head official decoder mismatch')
            write_json(out/'DECODER_WITNESS.json',dict(status='PASS',image_id=iid,zero_head_pixel_xor=xor,GT_allowlist=True))
        rr,pp,ss=spatial_and_predictions(gt,iid,item,masks,meta,True);spatial.extend(rr);skipped.extend(ss)
        for arm in allpred:allpred[arm].extend(pp[arm])
        corrections.extend(dict(arm=arm,image_id=iid,changed_pixels=value) for arm,value in changes.items())
        cats=sorted(gt.cats)
        for d in item['detections']:boxpred.append(dict(image_id=iid,category_id=cats[int(d[5])],score=float(d[4]),bbox=[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])]))
        if n%50==0 or n==len(ids):
            progress=dict(phase='decode',images=n,total=len(ids),seconds=time.monotonic()-start);print(json.dumps(progress),flush=True);write_json(out/'progress.json',progress)
    write_csv(out/'spatial.csv',spatial);write_csv(out/'corrections.csv',corrections);write_json(out/'skipped.json',skipped);write_json(out/'CACHE_HASHES.json',cachehash)
    stats=[];gtrows=[];pairs=[];predhash={}
    for arm,pred in allpred.items():
        path=out/'predictions'/f'{arm}.json.gz'
        with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(pred,f,separators=(',',':'))
        predhash[arm]=sha(path);row,gg,pr=evaluate(gt,meta,ids,pred,arm);stats.append(row);gtrows.extend(gg);pairs.extend(pr)
        if len(gg)!=sum(sum(not ann.get('iscrowd',0) for ann in gt.imgToAnns[iid]) for iid in ids):raise RuntimeError('Missing GT denominator')
        print(json.dumps(row),flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(boxpred);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    write_csv(out/'task_summary.csv',stats);write_csv(out/'gt_recovery.csv',gtrows);write_csv(out/'pair_recovery.csv',pairs);write_json(out/'PREDICTION_HASHES.json',predhash)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),arms=list(allpred),seconds=time.monotonic()-start,box_ap_common=float(ev.stats[0]),hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
