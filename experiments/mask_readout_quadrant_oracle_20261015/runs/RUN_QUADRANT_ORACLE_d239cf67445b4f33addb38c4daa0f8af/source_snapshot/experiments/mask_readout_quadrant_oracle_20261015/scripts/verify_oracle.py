"""Independent whole-mask intersection/union and fixed-index spatial replay."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from common import ROOT,STUDY,S1_RUN,BASE,GT,read_json,sha256,read_jsonl


def verify(run,cache_audit):
    state=read_json(run/'run.json');summary=read_json(run/'SUMMARY.json');contract=read_json(run/'EVAL_CONTRACT.json')
    assert state['execution_status']=='completed' and read_json(run/'CONFIG_AUDIT.json')['status']=='passed'
    assert sha256(run/'PROTOCOL.md')==state['protocol_sha256'] and sha256(run/'EVAL_CONTRACT.json')==state['contract_sha256']
    for v in read_json(run/'ARTIFACT_MANIFEST.json')['artifacts']:assert sha256(run/v['relative_path'])==v['sha256']
    for p,h in read_json(run/'SOURCE_MANIFEST.json')['files'].items():assert sha256(ROOT/p)==sha256(run/'source_snapshot'/p)==h
    if GT.exists():raw=GT.read_bytes()
    else:
        with zipfile.ZipFile(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/RESULTS_BUNDLE.zip') as z:raw=z.read(GT.relative_to(ROOT).as_posix())
    assert hashlib.sha256(raw).hexdigest()==contract['input_hashes']['files'][GT.relative_to(ROOT).as_posix()]
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    coco=COCO();coco.dataset=json.loads(raw);coco.createIndex()
    source_panel=read_json(S1_RUN/'TARGET_PANEL.json')['targets'];selected=source_panel if state['scope']=='full' else source_panel[:4]
    original={(r['image_id'],r['annotation_id']):r for r in read_jsonl(S1_RUN/'ORACLE_RESULTS.jsonl')}
    rows=read_jsonl(run/'ORACLE_RESULTS.jsonl');assert len(rows)==len(selected)
    expected={(r['image_id'],r['annotation_id']):r for r in selected};assert {(r['image_id'],r['annotation_id']) for r in rows}==expected.keys()
    replay_ids={v['output_detection_id'] for v in selected[:8]};replayed=0
    def iou(a,b):
        inter=int(masks.area(masks.merge([a,b],intersect=True)));union=int(masks.area(a))+int(masks.area(b))-inter
        return inter/union if union else 0
    for r in rows:
        key=r['image_id'],r['annotation_id'];target=expected[key];truth=coco.annToRLE(coco.anns[r['annotation_id']])
        assert all(r[k]==v for k,v in target.items());assert abs(iou(r['native_rle'],truth)-r['baseline_mask_iou'])<1e-12
        assert original[key]['output_detection_id']==r['output_detection_id']
        old_iou=iou(original[key]['oracle_rle'],truth);assert abs(old_iou-r['old_global_iou'])<1e-12
        for arm,a in r['arms'].items():
            value=iou(a['oracle_rle'],truth);assert abs(value-a['oracle_mask_iou'])<1e-12 and a['recovered']==(value>=.75)
            for q in set(range(4))-set(a['active_quadrants']):assert all(v==0 for v in a['delta_float64'][q])
        if cache_audit and r['output_detection_id'] in replay_ids:
            import torch
            from torch.nn import functional as F
            from common import configure_native
            from ultralytics.utils import ops
            configure_native();path=BASE/'trace'/r['cache_relative_path'];assert sha256(path)==r['cache_sha256']
            cache=torch.load(path,map_location='cpu',weights_only=True);rank=r['rank'];idx=int(r['output_detection_id'].rsplit(':',1)[1]);assert cache['prediction_indices'][rank]==idx
            ph,pw=cache['proto'].shape[-2:];ih,iw=cache['input_hw'];box=cache['boxes_input'][rank];cx=float((box[0]+box[2])/2);cy=float((box[1]+box[3])/2)
            # Scalar labels followed by the explicitly prescribed 3x3 replicate average.
            labels=[[int(x*(iw/pw)>=cx)+2*int(y*(ih/ph)>=cy) for x in range(pw)] for y in range(ph)]
            hard=torch.nn.functional.one_hot(torch.tensor(labels,device='cuda'),4).permute(2,0,1).float()
            weights=F.avg_pool2d(F.pad(hard[None],(1,1,1,1),mode='replicate'),3,stride=1)[0]
            proto=cache['proto'].cuda().float();c=cache['c0'].cuda().float();native=(c@proto.flatten(1)).reshape(len(c),ph,pw)
            for name,a in r['arms'].items():
                d=torch.tensor(a['delta_float64'],device='cuda',dtype=torch.float32);z=native.clone()
                if torch.count_nonzero(d):z[rank]+=((d@proto.flatten(1)).reshape(4,ph,pw)*weights).sum(0)
                # Native interpolation uses [1,N,H,W]; channel slicing only after the same full-matrix multiply.
                up=ops.scale_masks(z[rank:rank+1][None],cache['native_decode_hw'])[0].gt(0).byte()
                cropped=ops.crop_mask(up,cache['boxes_input'][rank:rank+1].cuda())
                final=ops.scale_masks(cropped[None],cache['original_hw'],ratio_pad=cache['ratio_pad'])[0].byte()[0].cpu().numpy()
                assert (final==masks.decode(a['oracle_rle'])).all()
            replayed+=1;del cache,proto,c,native;torch.cuda.empty_cache()
    arms={}
    for arm in ('Q-all','Q-local'):
        found=sum(r['arms'][arm]['recovered'] for r in rows);old=sum(r['old_global_iou']>=.75 for r in rows)
        unc=sum(r['arms'][arm]['solver_status']=='budget_exhausted' for r in rows);passing=found/len(rows)>=.2 and found-old>=13
        status=('passed' if passing else 'blocked_precision' if unc else 'failed') if state['scope']=='full' else 'engineering_only'
        arms[arm]=dict(found=found,total=len(rows),net_new=found-old,unconverged=unc,status=status)
        for k,v in arms[arm].items():assert summary['arms'][arm][k]==v
    return dict(status='passed',run_id=state['run_id'],run_json_sha256=sha256(run/'run.json'),summary_sha256=sha256(run/'SUMMARY.json'),
        same_output_target_pairs=len(rows),all_native_global_and_quadrant_RLE_IoUs_recomputed=True,spatial_replay_targets=replayed,cache_audit=cache_audit,arms=arms)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--cache-audit',action='store_true');a=p.parse_args()
    result=verify(a.run,a.cache_audit)
    with a.out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
