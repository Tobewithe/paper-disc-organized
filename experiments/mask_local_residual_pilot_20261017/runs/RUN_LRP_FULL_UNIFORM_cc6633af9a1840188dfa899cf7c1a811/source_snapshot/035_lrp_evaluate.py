from lrp_common import *
from collections import defaultdict
import numpy as np,torch
from adapter_data import FrozenCache
from lrp_math import PixelResidual,image_context,points,deploy
from lea_ops import encode
from pycocotools.coco import COCO
from pycocotools import mask as masks
from s1_geometry import full_rle_iou
from evaluate_cld import ledger
from lrp_statistics import summary
from posttrain_audit import compare_outputs

def main():
    data=inspect_inputs(); cfg=read_json(STUDY/'CONFIG.json')
    parents={arm:STUDY/'runs'/read_json(STUDY/f'FULL_{arm}_POINTER.json')['run_id'] for arm in ('LOCAL','UNIFORM')}
    models={}; checkpoint_sha={}
    for arm,path in parents.items():
        completed(path)
        saved=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        require(saved['steps']==1200 and saved['epochs']==3 and saved['arm']==arm,'incomplete training')
        require(all(float(v['step'])==1200 for v in saved['optimizer_state_dict']['state'].values()),'optimizer checkpoint steps drift')
        checkpoint_sha[arm]=sha256(path/'last.pt')
        model=PixelResidual().cuda().eval().requires_grad_(False); model.load_state_dict(saved['state_dict'])
        require(tensor_state_hash(model)==read_json(path/'SUMMARY.json')['final_state_sha256'],'checkpoint state drift')
        models[arm]=model
    with Run('EVALUATION',parents['LOCAL']) as run:
        run.freeze(dict(stage='final_400_evaluation',training_runs={k:v.name for k,v in parents.items()},checkpoints=checkpoint_sha,environment=data['environment'],optimizer_steps=0))
        ids=sorted(data['split']['heldout_image_ids']); held=set(ids)
        source=FrozenCache(CACHE/'trace',ids); coco=COCO(str(GT))
        historic=defaultdict(list)
        for r in read_jsonl(FIT/'FIT_FIXED_PAIRS.jsonl'):
            if r['image_id'] in held: historic[r['image_id']].append(r)
        all_outputs={k:[] for k in ('N','L','U')}; mapping=[]; pairs=[]; receipts=[]
        with torch.inference_mode():
            for pos,iid in enumerate(ids,1):
                run.check_budget(); cache,rows=source.load(iid)
                ranks,selected,basis=image_context(cache,rows)
                outputs={k:[dict(r) for r in rows] for k in ('N','L','U')}
                for rank in ranks:
                    rec=points(cache,rows[rank],rank,selected[rank],basis)
                    changed={}
                    for arm,key in (('LOCAL','L'),('UNIFORM','U')):
                        mask=deploy(models[arm],rec,arm)
                        outputs[key][rank]['segmentation']=encode(mask)
                        changed[key]=int(np.count_nonzero(mask!=rec['native']))
                    receipts.append(dict(image_id=iid,rank=rank,global_index=cache['prediction_indices'][rank],band_rle=encode(rec['band']),editable_pixels=int(rec['band'].sum()),changed=changed))
                    del rec
                for key in ('L','U'): compare_outputs(rows,outputs[key])
                rank_of={g:j for j,g in enumerate(cache['prediction_indices'])}
                for old in historic[iid]:
                    rank=rank_of[int(old['baseline_output_detection_id'].rsplit(':',1)[1])]
                    gt=coco.annToRLE(coco.anns[old['annotation_id']])
                    require(abs(full_rle_iou(rows[rank]['segmentation'],gt)-old['baseline_iou'])<1e-12,'fixed native pair drift')
                    pairs.append(dict(image_id=iid,annotation_id=old['annotation_id'],output_detection_id=old['baseline_output_detection_id'],population=old['population'],N=old['baseline_iou'],G=old['G_iou'],GT_rle=gt,
                                      L=full_rle_iou(outputs['L'][rank]['segmentation'],gt),U=full_rle_iou(outputs['U'][rank]['segmentation'],gt)))
                for key in all_outputs: all_outputs[key].extend(outputs[key])
                mapping.extend(cache['prediction_indices'])
                if pos%25==0 or pos==len(ids): run.update(completed_images=pos); print('EVALUATE',pos,'/400',flush=True)
                del cache,rows,ranks,selected,basis,outputs
        require(len(mapping)==42167 and len(set(mapping))==42167,'output denominator drift')
        order=sorted(range(len(mapping)),key=mapping.__getitem__); mapping=[mapping[j] for j in order]
        ledgers={}; native_sha=source.manifest['predictions_sha256']
        for key in all_outputs:
            all_outputs[key]=[all_outputs[key][j] for j in order]
            write_new(run.path/f'{key}_predictions.json',all_outputs[key])
            ledgers[key]=ledger(run.path,all_outputs[key],ids,mapping,native_sha)
            jsonl(run.path/f'{key}_GT_LEDGER.jsonl',ledgers[key]); run.check_budget()
        previous={(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT/'BASELINE_GT_LEDGER.jsonl') if r['image_id'] in held}
        old_tasks={(r['image_id'],r['annotation_id']):r for r in read_jsonl(FIT/'FIT_TASKS.jsonl') if r['image_id'] in held}
        lookup={k:{(r['image_id'],r['annotation_id']):r for r in v} for k,v in ledgers.items()}; tasks=[]
        for identity,n in lookup['N'].items():
            h=previous[identity]
            for key in ('is_ignored','task_mask75','task_box75','bbox75_match_output_id','segm75_match_output_id'):
                require(n[key]==h[key],'native task identity drift: '+key)
            for key in ('L','U'):
                q=lookup[key][identity]
                for field in ('is_ignored','task_box75','bbox75_match_output_id'):
                    require(q[field]==n[field],'box/ignore identity changed')
            if not n['is_ignored']:
                tasks.append(dict(image_id=identity[0],annotation_id=identity[1],G=old_tasks[identity]['G_success'],**{k:lookup[k][identity]['task_mask75']=='success' for k in lookup}))
        # COCO may return bytes for annToRLE; serialize without changing counts.
        for r in pairs:
            if isinstance(r['GT_rle']['counts'],bytes): r['GT_rle']['counts']=r['GT_rle']['counts'].decode('ascii')
        jsonl(run.path/'FIXED_PAIRS.jsonl',pairs); jsonl(run.path/'TASKS.jsonl',tasks); jsonl(run.path/'SUPPORT_RECEIPTS.jsonl',receipts)
        write_new(run.path/'ROW_MAPPING.json',dict(global_indices=mapping,source_prediction_sha256=native_sha))
        write_new(run.path/'ANNOTATIONS_SUBSET.json',dict(images=[coco.imgs[i] for i in ids],annotations=[a for a in coco.anns.values() if a['image_id'] in held],categories=coco.dataset['categories'],info={}))
        results,draws=summary(pairs,tasks); np.savez(run.path/'BOOTSTRAP.npz',**draws)
        source.verify()
        require(all(sha256(parents[a]/'last.pt')==checkpoint_sha[a] for a in parents),'checkpoint changed')
        config_audit(run,new_optimizer_steps=0,images=400,predictions=42167,fixed_pairs=2246,target_pairs=490,native_successes=1808,nonmask_and_bbox_guard=True,native_ledger_reproduced=True,changed_pixels_outside_support=0,checkpoint_sha256=checkpoint_sha)
        run.finish(results,status=results['decision'],scientific=results['decision'])
    save(STUDY/'EVALUATION_POINTER.json',dict(run_id=run.run_id))

if __name__=='__main__': main()
