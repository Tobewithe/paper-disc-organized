"""Post-hoc decomposition of frozen responses; replays the three-region samples.

No network updates, new data selection, threshold tuning or task-AP claim.
"""
import argparse, contextlib, csv, io, json, time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
import three_region_probe as probe
from frozen_mechanism_probe import ROOT, sha, write_json


def read(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'diagnostics/three_region_20260911'
    protocol=json.loads((parent/'protocol.json').read_text())
    receipt=json.loads((parent/'COMPLETE.json').read_text())
    for name,digest in receipt['hashes'].items():assert sha(parent/name)==digest
    write_json(out/'PROTOCOL.json',dict(
        purpose='Post-hoc explanatory common/difference decomposition on EXACT prior samples. No independent confirmation or deployable method.',
        parent_complete_sha256=sha(parent/'COMPLETE.json'),parent_script_sha256=sha(probe.__file__),script_sha256=sha(__file__),
        images=protocol['images'],network_training=False,
        scores='m=(z_own+z_neighbor)/2; d=(z_own-z_neighbor)/2; z_own=m+d. No fit or free parameter.',
        metrics='Held-out balanced sampled own-v-neighbor AUC, own-v-background, neighbor-v-background, and equal own/neighbor mixture-v-background AUC. Shared response is algebraic, not automatically a semantic objectness latent.',
        inheritance='Identical parent pair matching, regions, support, exclusions, seeds and interpolation-stencil disjoint folds. All 4 scores on same pixels.',
        limits='Post-hoc, same previously explored data; matched-pair conditioning and small high cohort. Instance relative orientation, predicted-crop supports can differ by direction. No task AP, no causal CCL conclusion.'))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_scores=probe.scores;calls=[]
    def capture(features,coords,ca,cb,train,classes,seed):
        own=features@ca;neighbor=features@cb
        values={'own':own,'neighbor':neighbor,'common_mean':(own+neighbor)/2,'half_difference':(own-neighbor)/2}
        cc=classes[~train];records=[]
        for name,score in values.items():
            s=score[~train];a=s[cc==0];b=s[cc==1];bg=s[cc==2]
            records.append(dict(readout=name,identity_auc=probe.auc(a,b),own_background_auc=probe.auc(a,bg),
                neighbor_background_auc=probe.auc(b,bg),union_background_auc=probe.auc(np.concatenate([a,b]),bg),
                own_positive=float((a>0).mean()),neighbor_positive=float((b>0).mean()),background_positive=float((bg>0).mean()),
                algebra_max_abs=float(np.max(np.abs((own+neighbor)/2+(own-neighbor)/2-own)))))
        calls.append(records)
        return original_scores(features,coords,ca,cb,train,classes,seed)
    probe.scores=capture
    old={tuple(r[k] for k in ['split','image_id','target_annotation','domain','fold','readout']):r for r in read(parent/'readouts.csv')}
    checkmetrics=['auc_neighbor','auc_background','auc_mixed','coverage','neighbor_fpr','background_fpr','zero_coverage','zero_neighbor_fpr','zero_background_fpr']
    result=[];maximum=0.;checked=0;started=time.monotonic();hashes={}
    for split,ids in protocol['images'].items():
        with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/f'data/annotations/instances_{split}2017.json'))
        mp=ROOT/('census/train2017_instances.csv' if split=='train' else 'census/COCO_EVAL_INSTANCE_MANIFEST.csv')
        meta={int(r['annotation_id']):r for r in read(mp)}
        for num,iid in enumerate(ids,1):
            path=ROOT/'diagnostics/relative_ownership_20260911/train_cache'/f'{iid}.npz' if split=='train' else ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz'
            hashes[f'{split}/{iid}']=sha(path)
            with np.load(path) as q:item={k:q[k] for k in ['proto','coeff','boxes','detections','shape','input_shape','mapping_gt','mapping_pred']}
            calls.clear();rows,_=probe.analyze(gt,iid,item,meta,split)
            ownrows=[r for r in rows if r['readout']=='actual_own'];assert len(calls)==len(ownrows)
            for origin,extras in zip(ownrows,calls):
                metadata={k:origin[k] for k in ['split','image_id','target_annotation','other_annotation','domain','fold','target_ici','category_id','gt_area','eval_pixels_per_class']}
                result.extend(dict(**metadata,**extra) for extra in extras)
            for r in rows:
                key=tuple(str(r[k]) for k in ['split','image_id','target_annotation','domain','fold','readout'])
                reference=old[key]
                maximum=max(maximum,max(abs(float(r[m])-float(reference[m])) for m in checkmetrics));checked+=1
            if num%40==0 or num==len(ids):print(json.dumps(dict(split=split,images=num,total=len(ids),seconds=round(time.monotonic()-started,1))),flush=True)
    assert checked==len(old) and maximum<1e-12,(checked,len(old),maximum)
    assert hashes==json.loads((parent/'input_hashes.json').read_text())
    probe.write_csv(out/'readouts.csv',result)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,seconds=time.monotonic()-started,
        replay_rows_checked=checked,replay_max_abs=maximum,input_hashes_match=True,rows=len(result),
        hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':main()
